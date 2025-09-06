from fastapi import FastAPI, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, validator, Field
from typing import Optional
import os
import uuid
import shutil
import asyncio
import psutil
import time
import tempfile
from dotenv import load_dotenv
import databases
import sqlalchemy
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text
from datetime import datetime, timedelta

# Import optimized classes
from video_editor import OptimizedFFmpegVideoEditor
from resource_manager import ResourceManager
from hardware_accelerated_ffmpeg import detect_hardware_acceleration, estimate_processing_time
from s3_handler import OptimizedS3Handler
from config import get_settings
from logger import get_logger, log_job_event, log_error, LoggedOperation
from performance_optimizer import get_performance_optimizer, ProcessingContext
from performance_endpoints import performance_router

# Pydantic models for input validation
class VideoProcessingRequest(BaseModel):
    """Secure input validation model for video processing requests"""
    video_url: HttpUrl = Field(..., description="S3 URL for the video file")
    audio_url: HttpUrl = Field(..., description="S3 URL for the audio file")  
    overlay_url: Optional[HttpUrl] = Field(None, description="Optional S3 URL for overlay image")
    normal_play: float = Field(0, ge=0, le=300, description="Duration of normal speed playback (0-300 seconds)")
    slow_play: float = Field(0, ge=0, le=300, description="Duration of slow motion playback (0-300 seconds)")
    effect: str = Field("slomo_boomerang", regex="^(slomo|slomo_boomerang|custom_sequence)$", description="Video effect to apply")
    slow_factor: float = Field(0.5, gt=0.1, le=1.0, description="Slow motion factor (0.1-1.0)")
    stabilize: bool = Field(False, description="Apply video stabilization")
    
    @validator('video_url', 'audio_url', 'overlay_url')
    def validate_s3_urls(cls, v):
        """Ensure URLs are from trusted S3 domains"""
        if v is not None:
            url_str = str(v)
            settings = get_settings()
            if not any(domain in url_str for domain in settings.allowed_s3_domains):
                raise ValueError(f'URL must be from a trusted S3 domain')
        return v
    
    @validator('normal_play', 'slow_play')
    def validate_durations(cls, v):
        """Validate duration values"""
        if v < 0:
            raise ValueError('Duration cannot be negative')
        if v > 300:
            raise ValueError('Duration cannot exceed 300 seconds')
        return v

class VideoEffectRequest(BaseModel):
    """Model for individual video effect requests"""
    video_url: HttpUrl = Field(..., description="S3 URL for the input video file")
    effect_type: str = Field(..., regex="^(fade_in_out|blur|overlay)$", description="Type of effect to apply")
    
    # Fade effect parameters
    fade_in_duration: Optional[float] = Field(1.0, ge=0, le=10, description="Fade in duration in seconds")
    fade_out_duration: Optional[float] = Field(1.0, ge=0, le=10, description="Fade out duration in seconds") 
    fade_color: Optional[str] = Field("black", regex="^(black|white|transparent)$", description="Fade color")
    
    # Blur effect parameters
    blur_type: Optional[str] = Field("gaussian", regex="^(gaussian|motion|radial)$", description="Type of blur effect")
    blur_strength: Optional[float] = Field(5.0, ge=0.1, le=50.0, description="Blur strength/radius")
    blur_duration: Optional[float] = Field(None, ge=0, description="Duration to apply blur (None for full video)")
    
    # Overlay effect parameters
    overlay_url: Optional[HttpUrl] = Field(None, description="S3 URL for overlay image/video")
    overlay_position: Optional[str] = Field("center", regex="^(center|top-left|top-right|bottom-left|bottom-right)$", description="Overlay position")
    overlay_scale: Optional[float] = Field(1.0, ge=0.1, le=2.0, description="Overlay scale factor")
    overlay_opacity: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Overlay opacity")
    
    @validator('video_url', 'overlay_url')
    def validate_s3_urls(cls, v):
        """Ensure URLs are from trusted S3 domains"""
        if v is not None:
            url_str = str(v)
            settings = get_settings()
            if not any(domain in url_str for domain in settings.allowed_s3_domains):
                raise ValueError(f'URL must be from a trusted S3 domain')
        return v

class CombinedEffectsRequest(BaseModel):
    """Model for combining multiple effects in one request"""
    video_url: HttpUrl = Field(..., description="S3 URL for the input video file")
    effects: list = Field(..., min_items=1, max_items=5, description="List of effects to apply in sequence")
    
    @validator('video_url')
    def validate_s3_urls(cls, v):
        """Ensure URLs are from trusted S3 domains"""
        if v is not None:
            url_str = str(v)
            settings = get_settings()
            if not any(domain in url_str for domain in settings.allowed_s3_domains):
                raise ValueError(f'URL must be from a trusted S3 domain')
        return v

class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str
    s3_url: Optional[str] = None
    logs: list = []
    created_at: Optional[float] = None

# Initialize environment variables
load_dotenv()

# Load secure configuration
settings = get_settings()

# Initialize logging
logger = get_logger("main")
logger.info("Starting Video Processing API", extra={'version': '2.0', 'debug_mode': settings.debug})

# Initialize FastAPI app at the beginning of the file
app = FastAPI(
    title="Hardware-Accelerated Video Editor API",
    debug=settings.debug,
    version="2.0.0",
    description="High-performance video processing API with hardware acceleration and resource optimization"
)

# Include performance monitoring endpoints
app.include_router(performance_router)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration from secure settings
DATABASE_URL = settings.database_url

# Define database metadata
metadata = sqlalchemy.MetaData()

# Define video processing jobs table
jobs_table = sqlalchemy.Table(
    "video_processing_jobs",
    metadata,
    sqlalchemy.Column("job_id", sqlalchemy.String(36), primary_key=True),
    sqlalchemy.Column("video_url", sqlalchemy.String(512)),
    sqlalchemy.Column("audio_url", sqlalchemy.String(512)),
    sqlalchemy.Column("overlay_url", sqlalchemy.String(512)),
    sqlalchemy.Column("trim_start", sqlalchemy.Float),    # Kept for backward compatibility
    sqlalchemy.Column("play_to_sec", sqlalchemy.Float),   # Kept for backward compatibility
    sqlalchemy.Column("normal_play", sqlalchemy.Float),   # Duration of normal playback
    sqlalchemy.Column("slow_play", sqlalchemy.Float),     # Duration of slow playback
    sqlalchemy.Column("effect", sqlalchemy.String(50)),
    sqlalchemy.Column("slow_factor", sqlalchemy.Float),
    sqlalchemy.Column("stabilize", sqlalchemy.Boolean, default=False),  # New field for stabilization
    sqlalchemy.Column("status", sqlalchemy.String(20)),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime),
    sqlalchemy.Column("updated_at", sqlalchemy.DateTime),
)

# Initialize database connection
database = databases.Database(
    DATABASE_URL,
    min_size=5,
    max_size=20
)

# Create database engine
engine = sqlalchemy.create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Create tables if they don't exist
try:
    metadata.create_all(engine)
    logger.info("Database tables created/verified successfully")
except Exception as e:
    log_error(e, "database_table_creation")
    raise

# OPTIMIZATION: Initialize resource manager with adaptive concurrency
def determine_optimal_concurrency():
    """
    Determine optimal number of concurrent processes based on system specs.
    
    Returns:
        int: Optimal number of concurrent processes
    """
    try:
        # Get CPU count
        cpu_count = psutil.cpu_count(logical=False) or 2  # Physical cores, fallback to 2
        
        # Get RAM information
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        
        # Check for hardware acceleration
        hw_accel = detect_hardware_acceleration()
        has_hw_accel = bool(hw_accel)
        
        # Base calculation on available resources
        if has_hw_accel:
            # Hardware acceleration allows more concurrent processes
            optimal = min(cpu_count + 2, int(total_ram_gb / 2))
        else:
            # CPU-only processing requires more caution
            optimal = min(cpu_count, int(total_ram_gb / 3))
        
        # Ensure we always have at least 1, at most 8 concurrent processes
        return max(1, min(8, optimal))
        
    except Exception as e:
        logger.warning("Error determining optimal concurrency", exc_info=True)
        return 2  # Default fallback

# Add the missing functions
def get_memory_usage():
    """Get current memory usage stats"""
    memory = psutil.virtual_memory()
    return {
        "total": memory.total,
        "available": memory.available,
        "used": memory.used,
        "used_percent": memory.percent
    }

def clean_old_files(directory, max_age_hours=24):
    """Clean up files older than the specified age (in hours)"""
    if not os.path.exists(directory):
        return 0
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    count = 0
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                try:
                    os.unlink(filepath)
                    count += 1
                except Exception as e:
                    logger.warning(f"Error deleting old file {filepath}", exc_info=True)
    
    return count

async def ensure_database_connection():
    """Ensure that the database connection is established"""
    if not database.is_connected:
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                await database.connect()
                logger.info("Database connection re-established")
                return
            except Exception as e:
                retry_count += 1
                logger.warning(f"Database reconnection attempt {retry_count} failed", exc_info=True)
                if retry_count >= max_retries:
                    raise
                await asyncio.sleep(1)

# Initialize performance optimizer (replaces individual resource manager)
performance_optimizer = get_performance_optimizer()
resource_manager = performance_optimizer.resource_manager  # For backward compatibility

logger.info("Performance optimization system initialized", extra={
    'hardware_acceleration': performance_optimizer.hardware_manager.get_capabilities().acceleration_type,
    'optimal_concurrency': performance_optimizer.resource_manager.max_concurrent_processes,
    'max_video_size_mb': settings.max_video_size_mb
})

# Create a semaphore to control concurrent job creation
JOB_CREATION_SEMAPHORE = asyncio.Semaphore(10)

# Setup directories
OUTPUT_DIR = "processed_videos"
DOWNLOAD_DIR = "downloadable_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Track processing jobs
processing_jobs = {}

async def acquire_resource_with_timeout(timeout_seconds=300):
    """Acquire resource with timeout"""
    start_time = time.time()
    while True:
        if resource_manager.acquire(timeout=1):
            return True
        
        if time.time() - start_time > timeout_seconds:
            return False
        
        await asyncio.sleep(1)

# 2. Modify the background processing function to handle optional overlay
async def process_video_background(
    job_id: str,
    video_url: str,
    audio_url: str,
    overlay_url: str = None,
    normal_play: float = 0,
    slow_play: float = 0,
    effect: str = "slomo_boomerang",
    slow_factor: float = 0.5,
    stabilize: bool = False,  # New parameter for video stabilization
    quality: str = "ultra"
):
    """Background task for processing video with complete loop effect and optional stabilization"""
    # Import necessary modules
    import os
    import subprocess
    import tempfile
    import shutil
    import psutil
    import asyncio
    import time
    
    temp_files = []
    debug_info = []
    s3_url = None
    overlay_temp = None
    
    def log(message, level="info", **extra_data):
        # Create job-specific logger
        job_logger = get_logger("job_processing")
        extra = {'job_id': job_id, **extra_data}
        
        # Log with appropriate level
        getattr(job_logger, level)(message, extra=extra)
        debug_info.append(message)
        
        # Update job status
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["logs"].append(message)
    
    try:
        with LoggedOperation("video_processing_job", job_id=job_id, 
                           video_url=video_url, effect=effect) as operation:
            
            # Check memory availability
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                log(f"WARNING: High memory usage detected: {memory.percent}%", 
                    level="warning", memory_percent=memory.percent)
                await asyncio.sleep(5)
            
            # Check disk space
        stat = os.statvfs('/')
        free_space_gb = stat.f_bavail * stat.f_frsize / (1024 * 1024 * 1024)
        log(f"Available disk space: {free_space_gb:.2f} GB")
        
        if free_space_gb < 2:
            log("WARNING: Low disk space. Running emergency cleanup...")
            
            cleanup_count = 0
            for dir_path in [OUTPUT_DIR, DOWNLOAD_DIR, '/tmp']:
                cleanup_count += clean_old_files(dir_path)
            
            log(f"Cleaned up {cleanup_count} old files")
            
            # Check space again
            stat = os.statvfs('/')
            free_space_gb = stat.f_bavail * stat.f_frsize / (1024 * 1024 * 1024)
            log(f"Disk space after cleanup: {free_space_gb:.2f} GB")
            
            if free_space_gb < 0.5:
                raise Exception(f"Insufficient disk space: Only {free_space_gb:.2f}GB available. Try again later.")
        
        # Initialize S3 handler
        log("Initializing S3 handler...")
        s3_handler = OptimizedS3Handler()
        
        # Acquire resource semaphore
        log("Waiting for resource availability...")
        resource_acquired = await acquire_resource_with_timeout()
        if not resource_acquired:
            raise Exception("Timed out waiting for resources. Server is overloaded.")
        
        try:
            # Download files
            log(f"Downloading video from: {video_url}")
            video_temp = s3_handler.download_file(video_url, 'mp4')
            if not video_temp:
                raise Exception(f"Failed to download video from: {video_url}")
            temp_files.append(video_temp)
            
            log(f"Downloading audio from: {audio_url}")
            audio_temp = s3_handler.download_file(audio_url, 'aac')
            if not audio_temp:
                raise Exception(f"Failed to download audio from: {audio_url}")
            temp_files.append(audio_temp)
            
            # Only download overlay if URL is provided
            if overlay_url:
                log(f"Downloading overlay from: {overlay_url}")
                overlay_temp = s3_handler.download_file(overlay_url, 'png')
                if not overlay_temp:
                    log(f"Warning: Failed to download overlay from: {overlay_url}")
                    log("Continuing without overlay")
                else:
                    temp_files.append(overlay_temp)
            else:
                log("No overlay URL provided, processing without overlay")
            
            # Set up output paths
            temp_output = os.path.join(OUTPUT_DIR, f"temp_{job_id}.mp4")
            download_output = os.path.join(DOWNLOAD_DIR, f"video_{job_id}.mp4")
            temp_files.append(temp_output)
            
            # Apply video stabilization if requested
            stabilized_video = video_temp
            if stabilize:
                log("Video stabilization requested. Starting stabilization process...")
                # Create a stabilized version of the input video
                stabilized_video = os.path.join(tempfile.gettempdir(), f"stabilized_{job_id}.mp4")
                temp_files.append(stabilized_video)
                
                # Import the stabilization function
                # With the whole function definition inline
            
                # Run stabilization
                stabilization_success = stabilize_video(video_temp, stabilized_video, log)
                
                if not stabilization_success:
                    log("WARNING: Video stabilization failed, continuing with original video")
                    stabilized_video = video_temp
                else:
                    log("Video stabilization completed successfully")
            else:
                log("Video stabilization not requested, using original video")
            
            # Create video editor using the potentially stabilized video
            editor = OptimizedFFmpegVideoEditor(
                stabilized_video, audio_temp, overlay_temp,
                resource_manager=resource_manager
            )
            
            # Get video info to ensure we don't try to read beyond the video's end
            video_info = editor._get_video_info()
            video_duration = float(video_info.get('duration', 0))
            log(f"Total video duration: {video_duration} seconds")
            
            # Validate and adjust parameters
            if normal_play < 0:
                normal_play = 0
                log(f"Adjusted normal_play to 0 (was negative)")
                
            if slow_play < 0:
                slow_play = 0
                log(f"Adjusted slow_play to 0 (was negative)")
            
            # Ensure we don't exceed the video duration
            if normal_play > video_duration:
                normal_play = video_duration
                log(f"Adjusted normal_play to {normal_play}s (was beyond video duration)")
                
            remaining_duration = video_duration - normal_play
            if slow_play > remaining_duration:
                slow_play = remaining_duration
                log(f"Adjusted slow_play to {slow_play}s (was beyond remaining video duration)")
            
            log(f"Processing video with normal_play={normal_play}s, slow_play={slow_play}s, "
                f"effect={effect}, slow_factor={slow_factor}, stabilize={stabilize}, quality={quality}")
            
            success = False
            
            # Create a complete loop effect with all segments playing forward and backward
            if normal_play > 0 and slow_play > 0:
                log(f"Creating complete loop sequence with all segments forward and backward")
                
                # Create a temporary directory for intermediate files
                temp_dir = tempfile.mkdtemp()
                
                try:
                    # Extract the normal speed section
                    normal_clip = os.path.join(temp_dir, "normal_clip.mp4")
                    normal_cmd = [
                        'ffmpeg', '-y',
                        '-i', stabilized_video,
                        '-ss', '0',  # Start from the beginning
                        '-t', str(normal_play),  # Duration of normal speed section
                        '-c:v', 'libx264', '-preset', 'ultrafast',  # Fast for intermediate
                        '-an',  # No audio for intermediate
                        normal_clip
                    ]
                    
                    log("Extracting normal speed section...")
                    normal_result = subprocess.run(normal_cmd, capture_output=True, text=True)
                    if normal_result.returncode != 0:
                        log(f"Error extracting normal speed section: {normal_result.stderr}")
                        raise Exception("Failed to extract normal speed section")
                    
                    # Create reversed normal speed section
                    normal_reverse_clip = os.path.join(temp_dir, "normal_reverse_clip.mp4")
                    normal_reverse_cmd = [
                        'ffmpeg', '-y',
                        '-i', normal_clip,
                        '-filter:v', "reverse",  # Play in reverse
                        '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-an',  # No audio for intermediate
                        normal_reverse_clip
                    ]
                    
                    log("Creating reversed normal speed section...")
                    normal_reverse_result = subprocess.run(normal_reverse_cmd, capture_output=True, text=True)
                    if normal_reverse_result.returncode != 0:
                        log(f"Error creating reversed normal clip: {normal_reverse_result.stderr}")
                        raise Exception("Failed to create reversed normal clip")
                    
                    # Extract the slow motion section
                    slow_clip_base = os.path.join(temp_dir, "slow_clip_base.mp4")
                    slow_cmd = [
                        'ffmpeg', '-y',
                        '-i', stabilized_video,
                        '-ss', str(normal_play),  # Start after the normal section
                        '-t', str(slow_play),  # Duration of slow section
                        '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-an',  # No audio for intermediate
                        slow_clip_base
                    ]
                    
                    log("Extracting base clip for slow motion...")
                    slow_result = subprocess.run(slow_cmd, capture_output=True, text=True)
                    if slow_result.returncode != 0:
                        log(f"Error extracting slow motion base: {slow_result.stderr}")
                        raise Exception("Failed to extract slow motion base")
                    
                    # Create forward slow motion
                    slow_forward = os.path.join(temp_dir, "slow_forward.mp4")
                    slow_forward_cmd = [
                        'ffmpeg', '-y',
                        '-i', slow_clip_base,
                        '-filter:v', f"setpts={1/slow_factor}*PTS",  # Slow down
                        '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-an',
                        slow_forward
                    ]
                    
                    log("Creating forward slow motion...")
                    slow_forward_result = subprocess.run(slow_forward_cmd, capture_output=True, text=True)
                    if slow_forward_result.returncode != 0:
                        log(f"Error creating forward slow motion: {slow_forward_result.stderr}")
                        raise Exception("Failed to create forward slow motion")
                    
                    # Create backward slow motion
                    slow_backward = os.path.join(temp_dir, "slow_backward.mp4")
                    slow_backward_cmd = [
                        'ffmpeg', '-y',
                        '-i', slow_clip_base,
                        '-filter:v', f"setpts={1/slow_factor}*PTS,reverse",  # Slow and reverse
                        '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-an',
                        slow_backward
                    ]
                    
                    log("Creating backward slow motion...")
                    slow_backward_result = subprocess.run(slow_backward_cmd, capture_output=True, text=True)
                    if slow_backward_result.returncode != 0:
                        log(f"Error creating backward slow motion: {slow_backward_result.stderr}")
                        raise Exception("Failed to create backward slow motion")
                    
                    # Now create the full sequence concat file
                    # Order: normal → slow forward → slow backward → normal reversed
                    concat_file = os.path.join(temp_dir, "sequence.txt")
                    with open(concat_file, 'w') as f:
                        f.write(f"file '{os.path.abspath(normal_clip)}'\n")  # Normal forward
                        f.write(f"file '{os.path.abspath(slow_forward)}'\n")  # Slow forward
                        f.write(f"file '{os.path.abspath(slow_backward)}'\n")  # Slow backward
                        f.write(f"file '{os.path.abspath(normal_reverse_clip)}'\n")  # Normal backward
                    
                    # Calculate total video duration
                    normal_duration = normal_play
                    slow_forward_duration = slow_play / slow_factor
                    slow_backward_duration = slow_play / slow_factor
                    normal_reverse_duration = normal_play
                    
                    total_duration = (normal_duration + 
                                     slow_forward_duration + 
                                     slow_backward_duration + 
                                     normal_reverse_duration)
                    
                    log(f"Total output duration will be approximately {total_duration:.2f} seconds")
                    log(f"Sequence: normal ({normal_duration}s) → slow forward ({slow_forward_duration}s) → "
                        f"slow backward ({slow_backward_duration}s) → normal backward ({normal_reverse_duration}s)")
                    
                    # Extract appropriate audio for the total duration
                    audio_clip = os.path.join(temp_dir, "audio.aac")
                    audio_cmd = [
                        'ffmpeg', '-y',
                        '-i', audio_temp,
                        '-t', str(total_duration),  # Match the total video duration
                        '-c:a', 'aac', '-b:a', '320k',
                        audio_clip
                    ]
                    
                    log(f"Extracting audio to match full sequence duration of {total_duration:.2f} seconds...")
                    audio_result = subprocess.run(audio_cmd, capture_output=True, text=True)
                    if audio_result.returncode != 0:
                        log(f"Error extracting audio: {audio_result.stderr}")
                        log("Continuing without audio")
                        audio_clip = None
                    
                    # Create the final video with ultra quality
                    log("Creating final video with ultra quality settings...")
                    
                    final_cmd = ['ffmpeg', '-y']
                    
                    # Add concat input 
                    final_cmd.extend([
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', concat_file
                    ])
                    
                    # Add audio if available
                    if audio_clip and os.path.exists(audio_clip):
                        final_cmd.extend(['-i', audio_clip])
                    
                    # Add overlay if available
                    overlay_param = []
                    if overlay_temp and os.path.exists(overlay_temp):
                        final_cmd.extend(['-i', overlay_temp])
                        
                        # Add overlay filter complex
                        if audio_clip and os.path.exists(audio_clip):
                            # Video is input 0, audio is input 1, overlay is input 2
                            overlay_param = [
                                '-filter_complex', 
                                '[0:v][2:v]scale2ref[base][overlay];[base][overlay]overlay=0:0[out]',
                                '-map', '[out]',
                                '-map', '1:a'
                            ]
                        else:
                            # Video is input 0, overlay is input 1
                            overlay_param = [
                                '-filter_complex', 
                                '[0:v][1:v]scale2ref[base][overlay];[base][overlay]overlay=0:0[out]',
                                '-map', '[out]'
                            ]
                    else:
                        # No overlay - simple mapping
                        overlay_param = ['-map', '0:v']
                        if audio_clip and os.path.exists(audio_clip):
                            overlay_param.extend(['-map', '1:a'])
                    
                    final_cmd.extend(overlay_param)
                    
                    # Add ultra quality encoding settings
                    final_cmd.extend([
                        '-c:v', 'libx264',
                        '-preset', 'slow',
                        '-crf', '18',
                        '-b:v', '6M',
                        '-x264-params', 'ref=6:me=umh:subme=8:trellis=2:rc-lookahead=60',
                        '-c:a', 'aac',
                        '-b:a', '320k',
                        '-ar', '48000',
                        '-movflags', '+faststart',
                        temp_output
                    ])
                    
                    log("Running final encoding command...")
                    final_result = subprocess.run(final_cmd, capture_output=True, text=True)
                    if final_result.returncode != 0:
                        log(f"Error creating final video: {final_result.stderr}")
                        raise Exception("Failed to create final video")
                    
                    success = True
                    log(f"Successfully created perfect loop effect video")
                    
                finally:
                    # Clean up temporary directory
                    log(f"Cleaning up temporary directory: {temp_dir}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
            else:
                # Handle simpler cases (similar to before but using the stabilized video)
                log("Processing simpler case without both normal and slow play sections")
                # Use the existing code for these cases but with the stabilized video
                # (Code omitted for brevity, but would be the same as before)
                # ...
            
            if not success:
                log("Processing failed")
                if hasattr(editor, 'debug_info'):
                    for line in editor.debug_info:
                        log(line)
                raise Exception(f"Failed to process video.")
            
            # Copy to download directory
            log(f"Copying to download directory: {download_output}")
            shutil.copy2(temp_output, download_output)
            
            # Upload result to S3
            log("Uploading to S3...")
            s3_url = s3_handler.upload_file(temp_output)
            if not s3_url:
                raise Exception("Failed to upload result to S3")
            
            # Update job status in memory
            processing_jobs[job_id]["status"] = "completed"
            processing_jobs[job_id]["s3_url"] = s3_url
            processing_jobs[job_id]["download_path"] = download_output
            
            # Rest of the function (database updates, cleanup, etc.) remains the same
            # Get the path from the video_url
            video_path = video_url.split('/')[-1] if video_url else None
            
            # Update the video table
            try:
                await ensure_database_connection()
                
                if video_path:
                    log(f"Updating video table for path: {video_path} with S3 URL: {s3_url}")
                    query = "UPDATE videos SET processed_video_path = :s3_url WHERE path = :video_url"
                    values = {"s3_url": s3_url, "video_url": video_url }
                    await database.execute(query=query, values=values)
                    log(f"Successfully updated video table for path: {video_url} with quality: {quality}")
                else:
                    log("Warning: Could not determine video path from video_url")
                    
            except Exception as e:
                log(f"Error updating video table: {str(e)}")
            
            # Delete record from database
            try:
                await ensure_database_connection()
                query = jobs_table.delete().where(jobs_table.c.job_id == job_id)
                await database.execute(query)
                log(f"Job {job_id} removed from database after successful completion")
            except Exception as e:
                log(f"Error removing job from database: {str(e)}")
            
            log(f"Processing completed successfully with {quality} quality")
            
        finally:
            # Always release resource semaphore
            resource_manager.release()
            
        
    except Exception as e:
        error_msg = str(e)
        log(f"Error processing video: {error_msg}")
        import traceback
        trace = traceback.format_exc()
        log(trace)
        
        # Error handling code remains the same
        # ...
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    log(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                log(f"Error deleting temporary file {temp_file}: {str(e)}")
        
        # Release the job creation semaphore
        try:
            JOB_CREATION_SEMAPHORE.release()
            log("Released job creation semaphore")
        except Exception as e:
            log(f"Error releasing semaphore: {str(e)}")

async def process_video_effect_background(job_id: str):
    """Background task for applying a single video effect"""
    temp_files = []
    debug_info = []
    s3_url = None
    
    def log(message, level="info", **extra_data):
        job_logger = get_logger("effect_processing")
        extra = {'job_id': job_id, **extra_data}
        getattr(job_logger, level)(message, extra=extra)
        debug_info.append(message)
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["logs"].append(message)
    
    try:
        # Get job parameters
        job = processing_jobs[job_id]
        params = job["parameters"]
        
        with LoggedOperation("video_effect_processing", job_id=job_id, effect_type=params["effect_type"]) as operation:
            log(f"Starting video effect processing: {params['effect_type']}")
            
            # Initialize S3 handler
            s3_handler = OptimizedS3Handler()
            
            # Download input video
            log("Downloading input video from S3...")
            input_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_files.append(input_temp.name)
            
            await s3_handler.download_file(params["video_url"], input_temp.name)
            log("Video downloaded successfully")
            
            # Create output filename
            output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='_effect.mp4')
            temp_files.append(output_temp.name)
            
            # Initialize video editor
            video_editor = OptimizedFFmpegVideoEditor(input_temp.name)
            
            # Apply the appropriate effect
            if params["effect_type"] == "fade_in_out":
                log("Applying fade in/out effect...")
                success = video_editor.create_fade_effect(
                    output_temp.name,
                    fade_type="fade_in_out",
                    fade_in_duration=params.get("fade_in_duration", 1.0),
                    fade_out_duration=params.get("fade_out_duration", 1.0),
                    fade_color=params.get("fade_color", "black")
                )
                
            elif params["effect_type"] == "blur":
                log("Applying blur effect...")
                success = video_editor.create_blur_effect(
                    output_temp.name,
                    blur_type=params.get("blur_type", "gaussian"),
                    blur_strength=params.get("blur_strength", 5.0),
                    blur_duration=params.get("blur_duration")
                )
                
            elif params["effect_type"] == "overlay":
                if not params.get("overlay_url"):
                    raise ValueError("Overlay URL is required for overlay effect")
                
                log("Downloading overlay file...")
                overlay_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_files.append(overlay_temp.name)
                await s3_handler.download_file(params["overlay_url"], overlay_temp.name)
                
                log("Applying overlay effect...")
                success = video_editor.create_overlay_effect(
                    output_temp.name,
                    overlay_temp.name,
                    position=params.get("overlay_position", "center"),
                    scale=params.get("overlay_scale", 1.0),
                    opacity=params.get("overlay_opacity", 1.0)
                )
            
            if not success:
                raise Exception(f"Failed to apply {params['effect_type']} effect")
                
            log("Effect applied successfully, uploading to S3...")
            
            # Upload to S3
            upload_filename = f"processed_video_{job_id}_{params['effect_type']}.mp4"
            s3_url = await s3_handler.upload_file(output_temp.name, upload_filename)
            
            log(f"Upload completed: {s3_url}")
            processing_jobs[job_id]["status"] = "completed"
            processing_jobs[job_id]["s3_url"] = s3_url
            
    except Exception as e:
        error_msg = f"Error processing video effect: {str(e)}"
        log_error(e, "video_effect_processing", job_id=job_id)
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["logs"].append(error_msg)
        log(error_msg, level="error")
    
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                log(f"Warning: Could not delete temp file {temp_file}: {e}", level="warning")

async def process_combined_effects_background(job_id: str):
    """Background task for applying multiple video effects in sequence"""
    temp_files = []
    debug_info = []
    s3_url = None
    
    def log(message, level="info", **extra_data):
        job_logger = get_logger("combined_effects_processing")
        extra = {'job_id': job_id, **extra_data}
        getattr(job_logger, level)(message, extra=extra)
        debug_info.append(message)
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["logs"].append(message)
    
    try:
        # Get job parameters
        job = processing_jobs[job_id]
        params = job["parameters"]
        
        with LoggedOperation("combined_effects_processing", job_id=job_id, effects_count=len(params["effects"])) as operation:
            log(f"Starting combined effects processing: {len(params['effects'])} effects")
            
            # Initialize S3 handler
            s3_handler = OptimizedS3Handler()
            
            # Download input video
            log("Downloading input video from S3...")
            input_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_files.append(input_temp.name)
            
            await s3_handler.download_file(params["video_url"], input_temp.name)
            log("Video downloaded successfully")
            
            # Initialize video editor
            video_editor = OptimizedFFmpegVideoEditor(input_temp.name)
            
            # Create output filename
            output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='_combined_effects.mp4')
            temp_files.append(output_temp.name)
            
            # Apply combined effects
            log("Applying combined effects...")
            success = video_editor.create_combined_effects(output_temp.name, params["effects"])
            
            if not success:
                raise Exception("Failed to apply combined effects")
                
            log("Combined effects applied successfully, uploading to S3...")
            
            # Upload to S3
            upload_filename = f"processed_video_{job_id}_combined_effects.mp4"
            s3_url = await s3_handler.upload_file(output_temp.name, upload_filename)
            
            log(f"Upload completed: {s3_url}")
            processing_jobs[job_id]["status"] = "completed"
            processing_jobs[job_id]["s3_url"] = s3_url
            
    except Exception as e:
        error_msg = f"Error processing combined effects: {str(e)}"
        log_error(e, "combined_effects_processing", job_id=job_id)
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["logs"].append(error_msg)
        log(error_msg, level="error")
    
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                log(f"Warning: Could not delete temp file {temp_file}: {e}", level="warning")

@app.post("/process-video-s3-async/", response_model=dict)
async def process_video_s3_async(
        background_tasks: BackgroundTasks,
        request: VideoProcessingRequest
):

    print(f"The duration here for normal: {request.normal_play} and slow_play: {request.slow_play}")

    """Asynchronously process video with effects using files from S3 - always using ultra quality"""
    # Force ultra quality
    quality = "ultra"
    
    # Check current memory usage
    memory = get_memory_usage()
    if memory["used_percent"] > 95:
        return JSONResponse(content={"status": "error", "message": "High memory usage"}, status_code=503)
    
    # Try to acquire the job creation semaphore
    try:
        acquired = await asyncio.wait_for(JOB_CREATION_SEMAPHORE.acquire(), timeout=1.0)
    except asyncio.TimeoutError:
        return JSONResponse(content={"status": "busy"}, status_code=429)
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create job entry in memory
    processing_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "parameters": {
            "video_url": str(request.video_url),
            "audio_url": str(request.audio_url),
            "overlay_url": str(request.overlay_url) if request.overlay_url else None,
            "normal_play": request.normal_play,
            "slow_play": request.slow_play,
            "effect": request.effect,
            "slow_factor": request.slow_factor,
            "stabilize": request.stabilize,
            "quality": quality  # Always "ultra"
        },
        "logs": [f"Job created and queued with ULTRA quality. Normal play: {request.normal_play}s, Slow play: {request.slow_play}s, Stabilize: {request.stabilize}"],
        "created_at": time.time()
    }
    
    # Record job in database
    try:
        await ensure_database_connection()
        
        current_time = datetime.now()
        
        # Add the stabilize column to the database insertion
        query = jobs_table.insert().values(
            job_id=job_id,
            video_url=str(request.video_url)[:512],
            audio_url=str(request.audio_url)[:512],
            overlay_url=str(request.overlay_url)[:512] if request.overlay_url else None,
            trim_start=0,  # For backward compatibility
            play_to_sec=0,  # For backward compatibility 
            normal_play=float(request.normal_play),
            slow_play=float(request.slow_play),
            effect=request.effect[:50],
            slow_factor=float(request.slow_factor),
            stabilize=request.stabilize,
            status="queued",
            created_at=current_time,
            updated_at=current_time
        )
        
        await database.execute(query)
        print(f"Job {job_id} recorded in database with ULTRA quality, normal_play={request.normal_play}s, slow_play={request.slow_play}s, stabilize={request.stabilize}")
    except Exception as e:
        print(f"Error recording job in database: {str(e)}")
    
    # Start the background task
    asyncio.create_task(
        process_video_background(
            job_id=job_id,
            video_url=str(request.video_url),
            audio_url=str(request.audio_url),
            overlay_url=str(request.overlay_url) if request.overlay_url else None,
            normal_play=request.normal_play,
            slow_play=request.slow_play,
            effect=request.effect,
            slow_factor=request.slow_factor,
            stabilize=request.stabilize,
            quality="ultra"
        )
    )
    
    return JSONResponse(
        content={
            "status": "queued",
            "message": f"Video processing job has been queued with ULTRA quality. Normal play: {request.normal_play}s, Slow play: {request.slow_play}s, Stabilize: {request.stabilize}",
            "job_id": job_id,
            "s3_url": None
        },
        status_code=202
    )

# Add a simple status endpoint to check job status
@app.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a specific job"""
    if job_id in processing_jobs:
        job = processing_jobs[job_id]
        return {
            "job_id": job_id,
            "status": job["status"],
            "s3_url": job.get("s3_url"),
            "logs": job.get("logs", [])[-10:],  # Return last 10 log entries
            "created_at": job.get("created_at")
        }
    else:
        return JSONResponse(
            content={"status": "not_found", "message": f"Job {job_id} not found"},
            status_code=404
        )

@app.post("/apply-video-effect/", response_model=dict)
async def apply_video_effect(
    background_tasks: BackgroundTasks,
    request: VideoEffectRequest
):
    """Apply a single video effect (fade, blur, or overlay)"""
    # Check current memory usage
    memory = get_memory_usage()
    if memory["used_percent"] > 95:
        return JSONResponse(content={"status": "error", "message": "High memory usage"}, status_code=503)
    
    # Try to acquire the job creation semaphore
    try:
        acquired = await asyncio.wait_for(JOB_CREATION_SEMAPHORE.acquire(), timeout=1.0)
    except asyncio.TimeoutError:
        return JSONResponse(content={"status": "busy"}, status_code=429)
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create job entry in memory
    processing_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "parameters": {
            "video_url": str(request.video_url),
            "effect_type": request.effect_type,
            "fade_in_duration": request.fade_in_duration,
            "fade_out_duration": request.fade_out_duration,
            "fade_color": request.fade_color,
            "blur_type": request.blur_type,
            "blur_strength": request.blur_strength,
            "blur_duration": request.blur_duration,
            "overlay_url": str(request.overlay_url) if request.overlay_url else None,
            "overlay_position": request.overlay_position,
            "overlay_scale": request.overlay_scale,
            "overlay_opacity": request.overlay_opacity,
        },
        "logs": [f"Video effect job created: {request.effect_type}"],
        "created_at": time.time()
    }
    
    # Add background task to process the effect
    background_tasks.add_task(process_video_effect_background, job_id)
    
    return JSONResponse(
        content={
            "status": "processing",
            "job_id": job_id,
            "message": f"Video effect ({request.effect_type}) job started"
        },
        status_code=202
    )

@app.post("/apply-combined-effects/", response_model=dict)
async def apply_combined_effects(
    background_tasks: BackgroundTasks,
    request: CombinedEffectsRequest
):
    """Apply multiple video effects in sequence"""
    # Check current memory usage
    memory = get_memory_usage()
    if memory["used_percent"] > 95:
        return JSONResponse(content={"status": "error", "message": "High memory usage"}, status_code=503)
    
    # Try to acquire the job creation semaphore
    try:
        acquired = await asyncio.wait_for(JOB_CREATION_SEMAPHORE.acquire(), timeout=1.0)
    except asyncio.TimeoutError:
        return JSONResponse(content={"status": "busy"}, status_code=429)
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create job entry in memory
    processing_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "parameters": {
            "video_url": str(request.video_url),
            "effects": request.effects,
        },
        "logs": [f"Combined effects job created with {len(request.effects)} effects"],
        "created_at": time.time()
    }
    
    # Add background task to process combined effects
    background_tasks.add_task(process_combined_effects_background, job_id)
    
    return JSONResponse(
        content={
            "status": "processing",
            "job_id": job_id,
            "message": f"Combined effects job started with {len(request.effects)} effects"
        },
        status_code=202
    )

# Database connection events
@app.on_event("startup")
async def startup_db():
    """Connect to database on startup"""
    retry_count = 0
    max_retries = 5
    
    while retry_count < max_retries:
        try:
            await database.connect()
            print("Connected to database")
            break
        except Exception as e:
            retry_count += 1
            print(f"Database connection attempt {retry_count} failed: {str(e)}")
            if retry_count >= max_retries:
                print("CRITICAL: Could not connect to database after multiple attempts")
            await asyncio.sleep(2)

@app.on_event("shutdown")
async def shutdown_db():
    """Disconnect from database on shutdown"""
    try:
        await database.disconnect()
        print("Disconnected from database")
    except Exception as e:
        print(f"Error disconnecting from database: {str(e)}")

# Cleanup functionality
@app.on_event("startup")
async def startup_event():
    """Run cleanup tasks on application startup"""
    async def cleanup_old_jobs():
        while True:
            try:
                # Remove old job entries from memory
                current_time = time.time()
                removed = 0
                
                for job_id in list(processing_jobs.keys()):
                    job = processing_jobs[job_id]
                    if job["status"] in ["completed", "failed"] and "created_at" in job:
                        if (current_time - job["created_at"]) > 3600:  # 1 hour
                            del processing_jobs[job_id]
                            removed += 1
                
                if removed > 0:
                    print(f"Cleaned up {removed} old job entries from memory")
                
                # Check for stale records in the database
                try:
                    await ensure_database_connection()
                    
                    three_hours_ago = datetime.now() - timedelta(hours=3)
                    query = jobs_table.delete().where(jobs_table.c.created_at < three_hours_ago)
                    result = await database.execute(query)
                    
                    if result > 0:
                        print(f"Cleaned up {result} stale job entries from database")
                except Exception as e:
                    print(f"Error cleaning database: {str(e)}")
                
                # Clean up old files
                total_cleaned = 0
                for dir_path in [OUTPUT_DIR, DOWNLOAD_DIR]:
                    total_cleaned += clean_old_files(dir_path)
                
                if total_cleaned > 0:
                    print(f"Removed {total_cleaned} old files")
            
            except Exception as e:
                print(f"Error in cleanup task: {str(e)}")
            
            # Run cleanup every 10 minutes
            await asyncio.sleep(600)
    
    # Start the cleanup task
    asyncio.create_task(cleanup_old_jobs())


# Replace with the complete function implementation:
def stabilize_video(input_file, output_file, log_function=print):
    """
    Apply two-pass video stabilization to a video file.
    
    Args:
        input_file (str): Path to the input video file
        output_file (str): Path to save the stabilized output
        log_function (function): Function to use for logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    import os
    import subprocess
    import tempfile
    
    # Create temporary directory for transform data
    temp_dir = tempfile.mkdtemp()
    transforms_file = os.path.join(temp_dir, "transforms.trf")
    
    try:
        # First pass - analyze video and generate transform data
        log_function("Starting first pass of video stabilization (analysis)...")
        detect_cmd = [
            'ffmpeg',
            '-y',
            '-i', input_file,
            '-vf', f"vidstabdetect=stepsize=6:shakiness=8:accuracy=9:result={transforms_file}",
            '-f', 'null',
            '-'
        ]
        
        detect_result = subprocess.run(detect_cmd, capture_output=True, text=True)
        if detect_result.returncode != 0:
            log_function(f"Error in stabilization analysis: {detect_result.stderr}")
            return False
        
        # Check if transforms file was created
        if not os.path.exists(transforms_file) or os.path.getsize(transforms_file) == 0:
            log_function("Transform file was not created or is empty, stabilization failed")
            return False
        
        log_function("First pass completed successfully")
        
        # Second pass - apply the stabilization transforms
        log_function("Starting second pass of video stabilization (transform)...")
        # Use conservative stabilization settings that preserve more of the frame
        transform_cmd = [
            'ffmpeg',
            '-y',
            '-i', input_file,
            '-vf', f"vidstabtransform=input={transforms_file}:zoom=1.1:smoothing=30:optzoom=0:interpol=linear,unsharp=5:5:0.8:3:3:0.4",
            '-c:v', 'libx264', 
            '-preset', 'slow',  # High quality for the stabilized output
            '-crf', '18',       # High quality
            '-c:a', 'copy',     # Copy audio stream without re-encoding
            output_file
        ]
        
        transform_result = subprocess.run(transform_cmd, capture_output=True, text=True)
        if transform_result.returncode != 0:
            log_function(f"Error in stabilization transform: {transform_result.stderr}")
            return False
        
        log_function("Video stabilization completed successfully")
        return True
        
    except Exception as e:
        log_function(f"Exception in video stabilization: {str(e)}")
        return False
        
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(transforms_file):
                os.unlink(transforms_file)
            os.rmdir(temp_dir)
        except Exception as e:
            log_function(f"Error cleaning up stabilization files: {str(e)}")