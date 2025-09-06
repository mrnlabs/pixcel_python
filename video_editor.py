"""
Optimized FFmpeg Video Editor

This module provides memory-optimized and hardware-accelerated video processing capabilities
using FFmpeg. It intelligently manages system resources to prevent crashes when processing
large videos and leverages hardware acceleration when available.
"""

import os
import tempfile
import shutil
import subprocess
import json
import platform
import time
import gc
import threading
import psutil
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple, Any
from logger import get_logger, log_error, LoggedOperation
from hardware_manager import get_hardware_manager
from video_effects import get_effects_processor, EffectType


def detect_hardware_acceleration():
    """
    Legacy compatibility function - delegates to centralized hardware manager
    
    Returns:
        dict: Hardware acceleration information with 'type' key
    """
    from hardware_manager import detect_hardware_acceleration as hw_detect
    return hw_detect()


def estimate_processing_time(width, height, duration, effect, slow_factor=0.5, quality="high"):
    """
    Estimate the video processing time based on resolution, duration and effect.
    
    Args:
        width (int): Video width in pixels
        height (int): Video height in pixels
        duration (float): Video duration in seconds
        effect (str): Processing effect
        slow_factor (float): Slow motion factor
        quality (str): Quality setting
        
    Returns:
        float: Estimated processing time in seconds
    """
    # Default to safe values if dimensions are not available
    if width is None or width == 0:
        width = 1280
    if height is None or height == 0:
        height = 720
    
    pixel_count = width * height
    
    # Base processing factors (seconds per megapixel-second)
    base_factors = {
        "ultra": 0.3,
        "high": 0.15,
        "medium": 0.08,
        "low": 0.04
    }
    
    # Effect complexity multipliers
    effect_multipliers = {
        "none": 1.0,
        "slomo": 2.5,
        "slomo_boomerang": 3.0,
        "custom_sequence": 3.5
    }
    
    base_factor = base_factors.get(quality.lower(), 0.15)
    effect_multiplier = effect_multipliers.get(effect.lower(), 1.0)
    
    # Calculate effective duration for slow motion
    if effect in ["slomo", "slomo_boomerang", "custom_sequence"]:
        effective_duration = duration / slow_factor
    else:
        effective_duration = duration
    
    # More pixels = more processing time
    pixel_factor = pixel_count / (1920 * 1080)  # Normalized to 1080p
    
    # Calculate estimated time
    estimated_seconds = base_factor * pixel_factor * effective_duration * effect_multiplier
    
    # Check if hardware acceleration is detected
    hw_accel = detect_hardware_acceleration()
    if hw_accel:
        # Hardware acceleration typically provides 2-4x speedup
        estimated_seconds /= 3.0
    
    return max(1.0, estimated_seconds)  # Ensure minimum of 1 second


def supports_10bit():
    """Check if the system supports 10-bit encoding"""
    try:
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',
            '-f', 'lavfi',
            '-i', 'nullsrc=s=640x480:d=1',
            '-pix_fmt', 'yuv420p10le',
            '-f', 'null',
            '-'
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    except Exception:
        return False


class ResourceManager:
    """
    Manages system resources to prevent memory issues.
    Implements a semaphore system to limit concurrent processing.
    """
    
    def __init__(self, max_concurrent_processes=2):
        """
        Initialize the resource manager.
        
        Args:
            max_concurrent_processes (int): Maximum number of concurrent video processes
        """
        self.semaphore = threading.Semaphore(max_concurrent_processes)
        self.active_processes = {}
        self._lock = threading.Lock()
        self.monitor_thread = None
        self._stop_monitor = False
        
        # Track system resources
        self.system_info = {
            'cpu_cores': psutil.cpu_count(logical=False) or 1,
            'total_memory': psutil.virtual_memory().total,
            'total_memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        print(f"ResourceManager initialized with {max_concurrent_processes} max concurrent processes")
        print(f"System has {self.system_info['cpu_cores']} physical CPU cores and {self.system_info['total_memory_gb']:.1f}GB RAM")
    
    def start_monitoring(self):
        """Start the background resource monitoring thread"""
        if self.monitor_thread is None:
            self._stop_monitor = False
            self.monitor_thread = threading.Thread(target=self._monitor_resources)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop the background resource monitoring thread"""
        if self.monitor_thread is not None:
            self._stop_monitor = True
            self.monitor_thread.join(timeout=3)
            self.monitor_thread = None
            print("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Monitor system resources and terminate processes if memory is low"""
        while not self._stop_monitor:
            try:
                # Get system memory information
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Log resource usage periodically
                if memory.percent > 70 or cpu_percent > 70:
                    print(f"Resource usage: Memory: {memory.percent}%, CPU: {cpu_percent}%")
                
                # MEMORY MANAGEMENT: Take action at different thresholds
                if memory.percent > 85:
                    print(f"WARNING: High memory usage detected: {memory.percent}%")
                    
                    # First action: Clear disk cache if on Linux
                    if os.path.exists('/proc/sys/vm/drop_caches') and memory.percent > 90:
                        try:
                            # Need sudo for this, might not work in all environments
                            os.system('sync && echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null')
                            print("Dropped system caches to free memory")
                        except Exception:
                            pass
                    
                    # Second action: Terminate the newest process if memory is critical
                    if memory.percent > 95 and self.active_processes:
                        with self._lock:
                            if self.active_processes:
                                # Find the newest process (highest start_time)
                                newest_pid = max(self.active_processes.items(), 
                                               key=lambda x: x[1]['start_time'])[0]
                                print(f"CRITICAL: Terminating process {newest_pid} due to memory pressure")
                                
                                try:
                                    process = psutil.Process(newest_pid)
                                    # Get process info before terminating
                                    process_info = {
                                        'cmd': process.cmdline(),
                                        'memory': process.memory_info().rss / (1024*1024),  # MB
                                        'cpu': process.cpu_percent()
                                    }
                                    print(f"Process using {process_info['memory']:.1f}MB of memory")
                                    
                                    # Terminate the process
                                    process.terminate()
                                    
                                    # Wait briefly and kill if not terminated
                                    try:
                                        process.wait(timeout=3)
                                    except psutil.TimeoutExpired:
                                        print(f"Process did not terminate gracefully, killing forcibly")
                                        process.kill()
                                    
                                    # Remove from our tracking
                                    if newest_pid in self.active_processes:
                                        del self.active_processes[newest_pid]
                                        
                                    # Release the semaphore to allow another process to start
                                    try:
                                        self.semaphore.release()
                                        print(f"Released semaphore after terminating process {newest_pid}")
                                    except Exception:
                                        pass
                                except Exception as e:
                                    print(f"Error terminating process: {str(e)}")
            
            except Exception as e:
                print(f"Error in resource monitor: {str(e)}")
            
            # Sleep before checking again (shorter sleep when resources are constrained)
            if memory.percent > 90 or cpu_percent > 90:
                time.sleep(1)  # Check more frequently when resources are constrained
            else:
                time.sleep(3)  # Standard check interval
    
    def acquire(self, timeout=300):
        """
        Acquire permission to start a new video process.
        
        Args:
            timeout (int): Maximum time in seconds to wait
            
        Returns:
            bool: True if acquired, False if timed out
        """
        # Check resource availability first
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            print(f"Cannot acquire resource: memory usage at {memory.percent}%")
            return False
            
        # Try to acquire the semaphore
        result = self.semaphore.acquire(timeout=timeout)
        if result:
            # Log current concurrency
            with self._lock:
                concurrency = len(self.active_processes)
            print(f"Resource acquired. Active processes: {concurrency}")
        return result
    
    def release(self):
        """Release permission after a video process completes"""
        try:
            self.semaphore.release()
            print("Resource released")
        except ValueError:
            # This happens if release is called more times than acquire
            print("Warning: Attempted to release an unacquired resource")
    
    def register_process(self, pid, metadata=None):
        """Register a new FFmpeg process for tracking"""
        with self._lock:
            # Get process info
            try:
                process = psutil.Process(pid)
                mem_info = process.memory_info()
                cpu_percent = process.cpu_percent(interval=0.1)
                
                process_info = {
                    'start_time': time.time(),
                    'memory_mb': mem_info.rss / (1024 * 1024),
                    'cpu_percent': cpu_percent,
                    'metadata': metadata or {}
                }
                
                self.active_processes[pid] = process_info
                print(f"Registered process {pid} (type: {metadata.get('type', 'unknown') if metadata else 'unknown'})")
            except Exception as e:
                # Process might no longer exist
                print(f"Error registering process {pid}: {str(e)}")
    
    def unregister_process(self, pid):
        """Unregister an FFmpeg process after completion"""
        with self._lock:
            if pid in self.active_processes:
                process_info = self.active_processes[pid]
                duration = time.time() - process_info['start_time']
                print(f"Unregistered process {pid} (ran for {duration:.1f}s)")
                del self.active_processes[pid]


class OptimizedFFmpegVideoEditor:
    """
    Memory-optimized video editor that leverages hardware acceleration
    and implements efficient processing techniques for FFmpeg.
    """
    
    def __init__(self, video_path, audio_path, overlay_path=None, resource_manager=None):
        """
        Initialize the video editor with source files.
        
        Args:
            video_path (str): Path to video file
            audio_path (str): Path to audio file
            overlay_path (str, optional): Path to overlay image
            resource_manager (ResourceManager, optional): Resource manager instance
        """
        self.logger = get_logger("video_editor")
        self.video_path = video_path
        self.audio_path = audio_path
        self.overlay_path = overlay_path  # Can be None
        self.debug_info = []
        self.resource_manager = resource_manager
        self.running_processes = []
        
        try:
            # Use centralized hardware manager
            self.hardware_manager = get_hardware_manager()
            self.hw_capabilities = self.hardware_manager.get_capabilities()
            self.hw_accel = detect_hardware_acceleration()  # For legacy compatibility
            self.effects_processor = get_effects_processor()
            
            self.logger.info("Hardware capabilities detected", extra={
                'acceleration_type': self.hw_capabilities.acceleration_type,
                'max_concurrent_streams': self.hw_capabilities.max_concurrent_streams,
                'supports_hevc': self.hw_capabilities.supports_hevc
            })
            
            # Get video info - this needs to be robust for corrupted videos
            self.video_info = self._get_video_info()
            
            # Get video dimensions for optimizations
            self.width = self.video_info.get('width', 1280)  # Default to HD if unknown
            self.height = self.video_info.get('height', 720)  # Default to HD if unknown
            
            # Log initialization info
            self.logger.info("Video editor initialized", extra={
                'video_path': video_path,
                'audio_path': audio_path,
                'overlay_path': overlay_path,
                'dimensions': f"{self.width}x{self.height}",
                'hw_accel': self.hw_accel['type'] if self.hw_accel else 'cpu'
            })
            
        except Exception as e:
            log_error(e, "video_editor_initialization", video_path=video_path)
            raise
        
    def log(self, message, level="info", **extra):
        """Add a message to the debug log with proper logging"""
        getattr(self.logger, level)(message, extra=extra)
        self.debug_info.append(message)
        
    def _get_video_info(self):
        """
        Get detailed information about a video file using alternative methods that are more robust.
        Using the more reliable format for modern FFmpeg versions.
        
        Returns:
            dict: Video information (width, height, duration, etc.)
        """
        try:
            # First try the more modern and robust ffprobe command format
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-show_entries', 'stream=width,height,r_frame_rate',
                '-of', 'json',
                self.video_path
            ]
            
            self.logger.debug("Running ffprobe", extra={'command': ' '.join(cmd)})
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            info = {'width': None, 'height': None, 'duration': None, 'fps': None}
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Extract streams information first
                if 'streams' in data:
                    for stream in data['streams']:
                        if 'width' in stream and not info['width']:
                            info['width'] = int(stream['width'])
                        if 'height' in stream and not info['height']:
                            info['height'] = int(stream['height'])
                        if 'r_frame_rate' in stream and not info['fps']:
                            fps_str = stream['r_frame_rate']
                            if '/' in fps_str:
                                num, den = map(int, fps_str.split('/'))
                                info['fps'] = num / den if den != 0 else 30
                            else:
                                info['fps'] = float(fps_str)
                
                # Extract format information
                if 'format' in data and 'duration' in data['format']:
                    info['duration'] = float(data['format']['duration'])
                
                # If we still don't have duration, try to get it from streams
                if not info['duration'] and 'streams' in data:
                    for stream in data['streams']:
                        if 'duration' in stream and not info['duration']:
                            info['duration'] = float(stream['duration'])
                
                # Log found data
                duration_str = f"{info['duration']}s" if info['duration'] else "unknown"
                dims_str = f"{info['width']}x{info['height']}px" if info['width'] and info['height'] else "unknown"
                print(f"Video info: duration={duration_str}, width={dims_str}")
                
                return info
            else:
                print(f"Error in ffprobe: {result.stderr}")
                
                # Fallback method to get just duration if dimensions failed
                try:
                    duration_cmd = [
                        'ffprobe',
                        '-v', 'error',
                        '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1',
                        self.video_path
                    ]
                    duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                    if duration_result.returncode == 0:
                        duration = float(duration_result.stdout.strip())
                        return {'width': 1280, 'height': 720, 'duration': duration, 'fps': 30.0}
                except Exception:
                    pass
                
                # Return safe defaults if everything fails
                return {'width': 1280, 'height': 720, 'duration': 10.0, 'fps': 30.0}
        except Exception as e:
            print(f"Exception in _get_video_info: {str(e)}")
            # Always return safe defaults
            return {'width': 1280, 'height': 720, 'duration': 10.0, 'fps': 30.0}
        
    def get_video_duration(self):
        """Get the duration of the video"""
        return float(self.video_info.get('duration', 10.0))
    
    def _run_process(self, cmd, capture_output=True, text=True):
        """
        Run a subprocess with resource tracking and memory optimization.
        
        Args:
            cmd (list): Command line arguments
            capture_output (bool): Whether to capture stdout/stderr
            text (bool): Whether to return text output (vs bytes)
            
        Returns:
            subprocess.CompletedProcess: Process result
        """
        # OPTIMIZATION: Enable multithreading globally with a reasonable limit
        if cmd[0] == 'ffmpeg' and all(x != '-threads' for x in cmd):
            # Add -threads 4 by default if not specified
            # Insert right after ffmpeg command
            cmd.insert(1, '-threads')
            cmd.insert(2, '4')
        
        # Launch process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=text
        )
        
        # Register process with resource manager if available
        if self.resource_manager:
            self.resource_manager.register_process(
                process.pid, 
                {'command': cmd[0], 'type': 'ffmpeg' if cmd[0] == 'ffmpeg' else 'other'}
            )
        
        self.running_processes.append(process.pid)
        
        # For brief processes, just wait for completion
        stdout, stderr = process.communicate()
        
        # Clean up process tracking
        if self.resource_manager:
            self.resource_manager.unregister_process(process.pid)
        
        if process.pid in self.running_processes:
            self.running_processes.remove(process.pid)
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr
        )
    
    def get_debug_info(self):
        """Return debug info as a string for error reporting."""
        return "\n".join(self.debug_info)
    
    def cleanup_processes(self):
        """Terminate any running processes"""
        import psutil
        
        for pid in self.running_processes[:]:
            try:
                process = psutil.Process(pid)
                process.terminate()
                
                # Wait for graceful termination
                try:
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    self.logger.warning(f"Process {pid} did not terminate gracefully, killing forcibly")
                    process.kill()
                
                self.logger.info("Process terminated", extra={'pid': pid})
                
                # Unregister from resource manager
                if self.resource_manager:
                    self.resource_manager.unregister_process(pid)
                
                self.running_processes.remove(pid)
            except Exception as e:
                log_error(e, f"process_termination", pid=pid)
                
    def cleanup_temp_files(self, temp_files: list):
        """Clean up temporary files with proper error handling"""
        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    self.logger.debug("Temporary file cleaned up", extra={'file': temp_file})
                except Exception as e:
                    log_error(e, "temp_file_cleanup", file_path=temp_file)
    
    def _generate_ffmpeg_command(self, output_path, trim_start, clip_duration, effect, slow_factor, with_overlay=True):
        """
        Generate an FFmpeg command with ultra quality settings and audio sync
        that ensures audio duration exactly matches the final video length
        
        Args:
            output_path: Path where the output video will be saved
            trim_start: Start time in seconds to trim from
            clip_duration: Duration of the clip in seconds
            effect: Effect to apply ("slomo", "slomo_boomerang", etc.)
            slow_factor: Slow-motion factor (0.5 = 2x slower)
            with_overlay: Whether to include the overlay
        
        Returns:
            list: FFmpeg command as a list of arguments
        """
        # Always use ultra quality settings
        preset = "slow"       # Use the slow preset for better quality
        crf = "18"            # Lower CRF means higher quality (18 is very high quality)
        video_bitrate = "6M"  # Higher bitrate for ultra quality
        audio_bitrate = "320k" # High quality audio

        # Start building the FFmpeg command
        cmd = ['ffmpeg', '-y']
        
        # Add hardware acceleration if available
        if self.hw_accel:
            hwaccel_type = self.hw_accel.get('type')
            if hwaccel_type == 'nvenc':
                cmd.extend(['-hwaccel', 'cuda'])
                codec = 'h264_nvenc'
                self.log("Using NVIDIA GPU hardware acceleration with ULTRA quality")
            elif hwaccel_type == 'qsv':
                cmd.extend(['-hwaccel', 'qsv'])
                codec = 'h264_qsv'
                self.log("Using Intel QuickSync hardware acceleration with ULTRA quality")
            elif hwaccel_type == 'vaapi':
                cmd.extend(['-hwaccel', 'vaapi', '-vaapi_device', '/dev/dri/renderD128'])
                codec = 'h264_vaapi'
                self.log("Using VAAPI hardware acceleration with ULTRA quality")
            elif hwaccel_type == 'videotoolbox':
                cmd.extend(['-hwaccel', 'videotoolbox'])
                codec = 'h264_videotoolbox'
                self.log("Using VideoToolbox hardware acceleration with ULTRA quality")
            else:
                codec = 'libx264'
                self.log("No suitable hardware acceleration found, using software encoding with ULTRA quality")
        else:
            codec = 'libx264'
            self.log("Hardware acceleration not detected, using software encoding with ULTRA quality")
        
        # Input files
        cmd.extend(['-i', self.video_path, '-i', self.audio_path])
        
        # Add overlay if provided and requested
        if with_overlay and self.overlay_path and os.path.exists(self.overlay_path):
            cmd.extend(['-i', self.overlay_path])
            has_overlay = True
        else:
            has_overlay = False
        
        # Base filter for trimming
        trim_filter = f"trim=start={trim_start}:duration={clip_duration},setpts=PTS-STARTPTS"
        
        # Build filter complex based on effect
        if effect == "slomo":
            # Calculate speed factor
            if slow_factor <= 0 or slow_factor >= 1:
                self.log(f"Invalid slow_factor: {slow_factor}. Using 0.5 (2x slower)")
                slow_factor = 0.5
            
            # Calculate final video duration after slowdown
            final_duration = clip_duration / slow_factor
            self.log(f"Original clip duration: {clip_duration}s, slowed duration: {final_duration}s")
            
            # Slow motion filter
            video_filter = f"[0:v]{trim_filter},setpts={1/slow_factor}*PTS[v]"
            
            # Important: For audio, we want to EXACTLY match the final video duration
            # Instead of using atempo (which can be inaccurate), use aresample+atrim+asetpts
            audio_filter = (
                f"[1:a]aresample=async=1:first_pts=0,atrim=0:{final_duration},"
                f"asetpts=PTS-STARTPTS[a]"
            )
            
            # Combine video and audio filters
            filter_complex = f"{video_filter};{audio_filter}"
            map_options = ["-map", "[v]", "-map", "[a]"]
            
        elif effect == "slomo_boomerang":
            # Boomerang effect (forward then reverse)
            video_filter = (
                f"[0:v]{trim_filter},setpts={1/slow_factor}*PTS[slomo];"
                f"[0:v]{trim_filter},setpts={1/slow_factor}*PTS,reverse[rev];"
                f"[slomo][rev]concat=n=2:v=1:a=0[v]"
            )
            
            # First calculate final video duration after effect
            final_duration = (clip_duration / slow_factor) * 2  # Double for forward+reverse
            self.log(f"Boomerang final duration: {final_duration}s")
            
            # For audio, create a precise length audio track by using atrim
            audio_filter = (
                f"[1:a]aresample=async=1:first_pts=0,atrim=0:{final_duration},"
                f"asetpts=PTS-STARTPTS[a]"
            )
            
            # Combine video and audio filters
            filter_complex = f"{video_filter};{audio_filter}"
            map_options = ["-map", "[v]", "-map", "[a]"]
            
        elif effect == "custom_sequence":
            # Use the custom sequence implementation with precise audio matching
            # This will be handled in the create_20_second_sequence method
            # Here we just provide a basic implementation
            video_filter = f"[0:v]{trim_filter}[v]"
            audio_filter = f"[1:a]atrim=start={trim_start}:duration={clip_duration},asetpts=PTS-STARTPTS[a]"
            filter_complex = f"{video_filter};{audio_filter}"
            map_options = ["-map", "[v]", "-map", "[a]"]
            
        else:  # default or simple processing
            # Simple processing with no special effects
            video_filter = f"[0:v]{trim_filter}[v]"
            
            # Just trim audio to match
            audio_filter = f"[1:a]atrim=start={trim_start}:duration={clip_duration},asetpts=PTS-STARTPTS[a]"
            
            # Combine video and audio filters
            filter_complex = f"{video_filter};{audio_filter}"
            map_options = ["-map", "[v]", "-map", "[a]"]
        
        # Add overlay if it's available
        if has_overlay:
            # Insert overlay into the filter complex
            filter_complex = filter_complex.replace("[v]", "[vbase]")
            filter_complex += f";[vbase][2:v]scale2ref[base][overlay];[base][overlay]overlay=0:0[v]"
            # Update map options to use the new output
            map_options[0] = "-map"
            map_options[1] = "[v]"
        
        # Add filter complex to command
        cmd.extend(['-filter_complex', filter_complex])
        
        # Add mapping options
        cmd.extend(map_options)
        
        # Add codec-specific options
        if 'nvenc' in codec:
            # NVIDIA-specific high quality settings
            cmd.extend([
                '-c:v', codec,
                '-preset', 'p7',  # Highest quality preset
                '-rc', 'vbr_hq',  # High quality variable bitrate
                '-b:v', video_bitrate,
                '-maxrate', '10M',
                '-bufsize', '20M'
            ])
        elif 'qsv' in codec:
            # Intel QuickSync specific high quality settings
            cmd.extend([
                '-c:v', codec,
                '-preset', 'veryslow',  # Highest quality preset
                '-b:v', video_bitrate,
                '-maxrate', '10M'
            ])
        elif 'vaapi' in codec:
            # VAAPI specific high quality settings
            cmd.extend([
                '-c:v', codec,
                '-b:v', video_bitrate,
                '-maxrate', '10M'
            ])
        elif 'videotoolbox' in codec:
            # VideoToolbox specific settings
            cmd.extend([
                '-c:v', codec,
                '-b:v', video_bitrate,
                '-profile:v', 'high',
                '-allow_sw', '1'
            ])
        else:
            # x264 software encoding with high quality settings
            cmd.extend([
                '-c:v', codec,
                '-preset', preset,
                '-crf', crf,
                '-b:v', video_bitrate,
                '-x264-params', 'ref=6:me=umh:subme=8:trellis=2:rc-lookahead=60'
            ])
        
        # Audio settings - always high quality
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', audio_bitrate,
            '-ar', '48000',  # High sample rate
            '-shortest',     # Important: Ensure output duration equals the shortest input stream
            '-movflags', '+faststart',  # Better streaming
            output_path
        ])
        
        return cmd
                
    def create_slomo(self, output_path: str, trim_start: float = 0, clip_duration: float = 1.0, 
                     slow_factor: float = 0.5, with_boomerang: bool = False, quality: str = "high") -> bool:
        """
        Create a slow motion effect video with memory optimization and hardware acceleration.
        
        Args:
            output_path: Path to save the final video
            trim_start: Time to start the clip from (in seconds)
            clip_duration: Duration of the clip to slow down (in seconds)
            slow_factor: Speed factor (0.5 = half speed, 0.25 = quarter speed)
            with_boomerang: Whether to apply boomerang effect after slowing down
            quality: Video quality ("ultra", "high", "medium", "low")
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Always use ultra quality
            quality = "ultra"
            
            # Estimate processing time
            self.log(f"Estimating processing time...")
            estimated_time = estimate_processing_time(
                self.width, self.height, clip_duration, 
                "slomo_boomerang" if with_boomerang else "slomo", 
                slow_factor, quality
            )
            self.log(f"Estimated processing time: {estimated_time:.1f} seconds ({estimated_time/60:.1f} minutes)")
            
            # Log file information
            for path, name in [(self.video_path, "Video"), (self.audio_path, "Audio")]:
                if not os.path.exists(path):
                    self.log(f"{name} file not found: {path}")
                    return False
                self.log(f"{name} file exists: {path}, size: {os.path.getsize(path)} bytes")
            
            # Check overlay if it exists
            if self.overlay_path:
                if not os.path.exists(self.overlay_path):
                    self.log(f"Overlay file not found: {self.overlay_path}")
                    self.log("Will continue without overlay")
                    self.overlay_path = None
                else:
                    self.log(f"Overlay file exists: {self.overlay_path}, size: {os.path.getsize(self.overlay_path)} bytes")
            else:
                self.log("No overlay path provided, will process without overlay")
            
            # Get video duration from our video info
            input_duration = self.video_info.get('duration', 0)
            self.log(f"Video properties: Duration={input_duration}s, Dimensions={self.width}x{self.height}")
            
            # Validate parameters
            if trim_start >= input_duration:
                self.log(f"Error: trim_start ({trim_start}s) exceeds video duration ({input_duration}s)")
                return False
            
            # Calculate actual duration for the clip
            max_clip_duration = input_duration - trim_start
            if clip_duration > max_clip_duration:
                clip_duration = max_clip_duration
                self.log(f"Adjusted clip_duration to {clip_duration}s to fit within video duration")
            
            # MEMORY OPTIMIZATION: Limit clip duration for very long videos
            if clip_duration > 20:
                old_duration = clip_duration
                clip_duration = min(clip_duration, 20)
                self.log(f"MEMORY OPTIMIZATION: Reduced clip duration from {old_duration}s to {clip_duration}s")
            
            # Use the optimized FFmpeg command generator
            effect = "slomo_boomerang" if with_boomerang else "slomo"
            cmd = self._generate_ffmpeg_command(
                output_path=output_path,
                trim_start=trim_start,
                clip_duration=clip_duration,
                effect=effect,
                slow_factor=slow_factor,
                with_overlay=(self.overlay_path is not None)
            )
            
            # Log the command (for debugging)
            self.log(f"Running FFmpeg command: {' '.join(cmd)}")
            
            # Run the command
            result = self._run_process(cmd)
            
            if result.returncode != 0:
                self.log(f"Error processing video: {result.stderr}")
                return False
            
            # Verify output
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                self.log(f"Output file problem: {output_path}")
                return False
            
            self.log(f"Successfully created video: {output_path} ({os.path.getsize(output_path)} bytes) with ULTRA quality")
            return True
            
        except Exception as e:
            log_error(e, "video_processing", 
                     output_path=output_path, 
                     trim_start=trim_start, 
                     clip_duration=clip_duration)
            return False
            
        finally:
            # Always clean up resources regardless of success/failure
            try:
                # Clean up any running processes
                self.cleanup_processes()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                self.logger.info("Resource cleanup completed", extra={
                    'output_path': output_path,
                    'processing_completed': os.path.exists(output_path) if output_path else False
                })
                
            except Exception as cleanup_error:
                log_error(cleanup_error, "resource_cleanup_during_slomo")
    
    def create_20_second_sequence(self, output_path: str, trim_start: float = 0, 
                                slow_factor: float = 0.5, quality: str = "high") -> bool:
        """
        Create a 20-second sequence with various effects and matching audio length
        
        Args:
            output_path: Path where the output video will be saved
            trim_start: Start time in seconds to trim from
            slow_factor: Slow-motion factor
            quality: Output quality level (ignored, always uses "ultra")
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Always use ultra quality settings
            quality = "ultra"
            preset = "slow"
            crf = "18"
            video_bitrate = "6M"
            audio_bitrate = "320k"
            self.log("Using ULTRA quality settings for sequence")
            
            # Determine if we have enough video duration
            video_duration = self.video_info.get('duration', 0)
            if video_duration < trim_start + 3:
                self.log(f"Video too short for sequence: {video_duration}s")
                return False
            
            # Create filter complex for the 20-second sequence
            # This creates multiple segments with different effects
            
            # Base segment - normal speed for 3 seconds
            segment1_duration = min(3.0, (video_duration - trim_start) / 3)
            segment1 = f"[0:v]trim=start={trim_start}:duration={segment1_duration},setpts=PTS-STARTPTS[seg1];"
            
            # Slow-mo segment - slowed down for 6 seconds
            segment2_start = trim_start + segment1_duration
            segment2_duration = min(2.0, (video_duration - segment2_start) / 3)
            segment2_output_duration = segment2_duration / slow_factor
            segment2 = f"[0:v]trim=start={segment2_start}:duration={segment2_duration},setpts=PTS-STARTPTS,setpts={1/slow_factor}*PTS[seg2];"
            
            # Reverse segment - played in reverse for 3 seconds
            segment3_start = segment2_start + segment2_duration
            segment3_duration = min(1.5, (video_duration - segment3_start) / 3)
            segment3 = f"[0:v]trim=start={segment3_start}:duration={segment3_duration},setpts=PTS-STARTPTS,reverse[seg3];"
            
            # Boomerang segment - played forward then backward
            segment4_start = segment3_start + segment3_duration
            segment4_duration = min(1.0, (video_duration - segment4_start) / 3)
            segment4_output_duration = segment4_duration * 2
            segment4 = (
                f"[0:v]trim=start={segment4_start}:duration={segment4_duration},setpts=PTS-STARTPTS[seg4a];"
                f"[0:v]trim=start={segment4_start}:duration={segment4_duration},setpts=PTS-STARTPTS,reverse[seg4b];"
            )
            
            # Final segment - normal speed
            segment5_start = segment4_start + segment4_duration
            segment5_duration = min(3.0, video_duration - segment5_start)
            segment5 = f"[0:v]trim=start={segment5_start}:duration={segment5_duration},setpts=PTS-STARTPTS[seg5];"
            
            # Calculate total video duration for audio matching
            total_video_duration = (
                segment1_duration + 
                segment2_output_duration + 
                segment3_duration + 
                segment4_output_duration + 
                segment5_duration
            )
            
            self.log(f"Total sequence output duration: {total_video_duration}s")
            
            # Concatenate all video segments
            concat = "[seg1][seg2][seg3][seg4a][seg4b][seg5]concat=n=6:v=1:a=0[v]"
            
            # Create precise audio that exactly matches the video duration
            audio_filter = (
                f"[1:a]aresample=async=1:first_pts=0,"
                f"atrim=0:{total_video_duration},"
                f"asetpts=PTS-STARTPTS[a]"
            )
            
            # Complete filter complex
            filter_complex = segment1 + segment2 + segment3 + segment4 + segment5 + concat + ";" + audio_filter
            
            # Prepare command with hardware acceleration if available
            cmd = ['ffmpeg', '-y']
            
            # Add hardware acceleration if available
            codec = 'libx264'  # Default to software encoding
            
            if self.hw_accel:
                hwaccel_type = self.hw_accel.get('type')
                if hwaccel_type == 'nvenc':
                    cmd.extend(['-hwaccel', 'cuda'])
                    codec = 'h264_nvenc'
                    self.log("Using NVIDIA GPU hardware acceleration for sequence with ULTRA quality")
                elif hwaccel_type == 'qsv':
                    cmd.extend(['-hwaccel', 'qsv'])
                    codec = 'h264_qsv'
                    self.log("Using Intel QuickSync hardware acceleration for sequence with ULTRA quality")
                elif hwaccel_type == 'vaapi':
                    cmd.extend(['-hwaccel', 'vaapi', '-vaapi_device', '/dev/dri/renderD128'])
                    codec = 'h264_vaapi'
                    self.log("Using VAAPI hardware acceleration for sequence with ULTRA quality")
                elif hwaccel_type == 'videotoolbox':
                    cmd.extend(['-hwaccel', 'videotoolbox'])
                    codec = 'h264_videotoolbox'
                    self.log("Using VideoToolbox hardware acceleration for sequence with ULTRA quality")
                else:
                    self.log("No suitable hardware acceleration found, using software encoding for sequence with ULTRA quality")
            else:
                self.log("Hardware acceleration not detected, using software encoding for sequence with ULTRA quality")
                
            # Input files
            cmd.extend(['-i', self.video_path, '-i', self.audio_path])
            
            # Add overlay if provided
            if self.overlay_path and os.path.exists(self.overlay_path):
                cmd.extend(['-i', self.overlay_path])
                
                # Add overlay to the filter complex
                filter_complex = filter_complex.replace("[v]", "[vbase]")
                filter_complex += ";[vbase][2:v]scale2ref[v][overlay];[v][overlay]overlay=0:0[final]"
                map_options = ["-map", "[final]", "-map", "[a]"]
            else:
                map_options = ["-map", "[v]", "-map", "[a]"]
                
            # Complete the command
            cmd.extend([
                '-filter_complex', filter_complex
            ])
            
            # Add mapping options
            cmd.extend(map_options)
            
            # Add codec-specific options
            if 'nvenc' in codec:
                cmd.extend([
                    '-c:v', codec,
                    '-preset', 'p7',
                    '-rc', 'vbr_hq',
                    '-b:v', video_bitrate,
                    '-maxrate', '10M',
                    '-bufsize', '20M'
                ])
            elif 'qsv' in codec:
                cmd.extend([
                    '-c:v', codec,
                    '-preset', 'veryslow',
                    '-b:v', video_bitrate,
                    '-maxrate', '10M'
                ])
            elif 'vaapi' in codec:
                cmd.extend([
                    '-c:v', codec,
                    '-b:v', video_bitrate,
                    '-maxrate', '10M'
                ])
            else:
                cmd.extend([
                    '-c:v', codec,
                    '-preset', preset,
                    '-crf', crf,
                    '-b:v', video_bitrate,
                    '-x264-params', 'ref=6:me=umh:subme=8:trellis=2:rc-lookahead=60'
                ])
                
            # Audio and output file
            cmd.extend([
                '-c:a', 'aac',
                '-b:a', audio_bitrate,
                '-ar', '48000',
                '-shortest',  # Ensure final duration matches the shortest input stream 
                '-movflags', '+faststart',
                output_path
            ])
            
            # Log the command
            self.log(f"Running FFmpeg sequence command: {' '.join(cmd)}")
            
            # Estimate processing time
            est_time = estimate_processing_time(
                width=self.width,
                height=self.height,
                duration=sum([segment1_duration, segment2_duration, segment3_duration, 
                            segment4_duration*2, segment5_duration]),
                effect="custom_sequence",
                slow_factor=slow_factor,
                quality=quality
            )
            
            self.log(f"Estimated processing time: {est_time:.2f} seconds")
            
            # Run the command
            start_time = time.time()
            result = self._run_process(cmd)
            elapsed_time = time.time() - start_time
            
            if result.returncode != 0:
                self.log(f"Error processing sequence: {result.stderr}")
                return False
                
            self.log(f"Successfully created sequence at {output_path} in {elapsed_time:.2f} seconds with ULTRA quality")
            return True
            
        except Exception as e:
            log_error(e, "create_20_second_sequence", 
                     output_path=output_path, 
                     trim_start=trim_start, 
                     slow_factor=slow_factor)
            return False
            
        finally:
            # Always clean up resources
            try:
                self.cleanup_processes()
                import gc
                gc.collect()
                
                self.logger.info("20-second sequence resource cleanup completed", extra={
                    'output_path': output_path,
                    'success': os.path.exists(output_path) if output_path else False
                })
            except Exception as cleanup_error:
                log_error(cleanup_error, "resource_cleanup_during_sequence_creation")
    
    def create_fade_effect(self, output_path: str, fade_type: str = "fade_in_out",
                         fade_in_duration: float = 1.0, fade_out_duration: float = 1.0,
                         fade_color: str = "black") -> bool:
        """
        Create fade effects on the video
        
        Args:
            output_path: Output video file path
            fade_type: Type of fade ("fade_in", "fade_out", "fade_in_out")
            fade_in_duration: Duration of fade-in effect in seconds
            fade_out_duration: Duration of fade-out effect in seconds
            fade_color: Color to fade to/from ("black", "white", "transparent")
            
        Returns:
            bool: True if successful
        """
        try:
            with LoggedOperation("fade_effect", fade_type=fade_type) as op:
                
                if fade_type == "fade_in":
                    success = self.effects_processor.create_fade_in_effect(
                        self.video_path, output_path, fade_in_duration, fade_color
                    )
                elif fade_type == "fade_out":
                    success = self.effects_processor.create_fade_out_effect(
                        self.video_path, output_path, fade_out_duration, fade_color
                    )
                else:  # fade_in_out
                    success = self.effects_processor.create_fade_in_out_effect(
                        self.video_path, output_path, fade_in_duration, fade_out_duration, fade_color
                    )
                
                if success:
                    self.logger.info("Fade effect created successfully", extra={
                        'fade_type': fade_type,
                        'output_path': output_path
                    })
                else:
                    self.logger.error("Fade effect creation failed")
                
                return success
                
        except Exception as e:
            log_error(e, "create_fade_effect", fade_type=fade_type, output_path=output_path)
            return False
        
        finally:
            try:
                self.cleanup_processes()
                import gc
                gc.collect()
            except Exception as cleanup_error:
                log_error(cleanup_error, "fade_effect_cleanup")
    
    def create_blur_effect(self, output_path: str, blur_type: str = "gaussian",
                         intensity: float = 5.0, start_time: float = 0.0, 
                         duration: float = 0.0) -> bool:
        """
        Create blur effect on the video
        
        Args:
            output_path: Output video file path
            blur_type: Type of blur ("gaussian", "box", "motion")
            intensity: Blur intensity (1.0 to 20.0)
            start_time: When to start blur effect (0 = beginning)
            duration: Duration of blur effect (0 = entire video)
            
        Returns:
            bool: True if successful
        """
        try:
            with LoggedOperation("blur_effect", blur_type=blur_type, intensity=intensity) as op:
                
                success = self.effects_processor.create_blur_effect(
                    self.video_path, output_path, blur_type, intensity, start_time, duration
                )
                
                if success:
                    self.logger.info("Blur effect created successfully", extra={
                        'blur_type': blur_type,
                        'intensity': intensity,
                        'output_path': output_path
                    })
                else:
                    self.logger.error("Blur effect creation failed")
                
                return success
                
        except Exception as e:
            log_error(e, "create_blur_effect", blur_type=blur_type, output_path=output_path)
            return False
        
        finally:
            try:
                self.cleanup_processes()
                import gc
                gc.collect()
            except Exception as cleanup_error:
                log_error(cleanup_error, "blur_effect_cleanup")
    
    def create_overlay_effect(self, output_path: str, overlay_path: str,
                            overlay_type: str = "image", position_x: str = "center",
                            position_y: str = "center", scale: float = 1.0,
                            opacity: float = 1.0, start_time: float = 0.0,
                            duration: float = 0.0) -> bool:
        """
        Create overlay effect on the video
        
        Args:
            output_path: Output video file path
            overlay_path: Path to overlay file (image or video)
            overlay_type: Type of overlay ("image" or "video")
            position_x: X position ("left", "center", "right" or pixel value)
            position_y: Y position ("top", "center", "bottom" or pixel value)
            scale: Scale factor for overlay
            opacity: Opacity (0.0 to 1.0)
            start_time: When to show overlay
            duration: Duration of overlay (0 = entire video, ignored for video overlay)
            
        Returns:
            bool: True if successful
        """
        try:
            with LoggedOperation("overlay_effect", overlay_type=overlay_type) as op:
                
                if not os.path.exists(overlay_path):
                    raise FileNotFoundError(f"Overlay file not found: {overlay_path}")
                
                if overlay_type == "video":
                    success = self.effects_processor.create_video_overlay(
                        self.video_path, overlay_path, output_path,
                        position_x, position_y, scale, opacity, start_time
                    )
                else:  # image
                    success = self.effects_processor.create_image_overlay(
                        self.video_path, overlay_path, output_path,
                        position_x, position_y, scale, opacity, start_time, duration
                    )
                
                if success:
                    self.logger.info("Overlay effect created successfully", extra={
                        'overlay_type': overlay_type,
                        'overlay_path': overlay_path,
                        'output_path': output_path
                    })
                else:
                    self.logger.error("Overlay effect creation failed")
                
                return success
                
        except Exception as e:
            log_error(e, "create_overlay_effect", overlay_type=overlay_type, output_path=output_path)
            return False
        
        finally:
            try:
                self.cleanup_processes()
                import gc
                gc.collect()
            except Exception as cleanup_error:
                log_error(cleanup_error, "overlay_effect_cleanup")
    
    def create_combined_effects(self, output_path: str, effects: List[Dict[str, Any]]) -> bool:
        """
        Apply multiple effects in a single pass for optimal performance
        
        Args:
            output_path: Output video file path
            effects: List of effect configurations, each containing:
                - type: Effect type ("fade_in", "fade_out", "blur", "overlay")
                - Additional parameters specific to each effect type
            
        Returns:
            bool: True if successful
            
        Example:
            effects = [
                {"type": "fade_in", "duration": 2.0, "color": "black"},
                {"type": "blur", "intensity": 3.0, "start_time": 10.0, "duration": 5.0},
                {"type": "overlay", "overlay_path": "logo.png", "position_x": "right", "position_y": "top"}
            ]
        """
        try:
            with LoggedOperation("combined_effects", effect_count=len(effects)) as op:
                
                success = self.effects_processor.create_combined_effects(
                    self.video_path, output_path, effects
                )
                
                if success:
                    self.logger.info("Combined effects created successfully", extra={
                        'effect_count': len(effects),
                        'output_path': output_path
                    })
                else:
                    self.logger.error("Combined effects creation failed")
                
                return success
                
        except Exception as e:
            log_error(e, "create_combined_effects", output_path=output_path, effect_count=len(effects))
            return False
        
        finally:
            try:
                self.cleanup_processes()
                self.effects_processor.cleanup_temp_files()
                import gc
                gc.collect()
            except Exception as cleanup_error:
                log_error(cleanup_error, "combined_effects_cleanup")