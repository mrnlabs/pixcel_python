"""
Video Processing Limits Manager
Enforces video size, duration, and processing time limits to prevent resource exhaustion.
"""
import os
import time
import asyncio
import aiohttp
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from logger import get_logger, log_error
from config import get_settings

@dataclass
class VideoLimits:
    """Video processing limits configuration"""
    max_file_size_mb: int = 500
    max_duration_seconds: int = 300  # 5 minutes
    max_resolution_pixels: int = 1920 * 1080  # 1080p
    max_processing_time_seconds: int = 600  # 10 minutes
    max_bitrate_mbps: float = 50.0
    allowed_formats: list = None
    
    def __post_init__(self):
        if self.allowed_formats is None:
            self.allowed_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm']

class ProcessingTimeoutError(Exception):
    """Raised when processing exceeds time limits"""
    pass

class VideoSizeLimitError(Exception):
    """Raised when video exceeds size limits"""
    pass

class LimitsManager:
    """
    Manages and enforces video processing limits to prevent resource exhaustion
    """
    
    def __init__(self):
        self.logger = get_logger("limits_manager")
        self.settings = get_settings()
        
        # Initialize limits from configuration
        self.limits = VideoLimits(
            max_file_size_mb=self.settings.max_video_size_mb,
            max_processing_time_seconds=self.settings.max_processing_duration
        )
        
        # Track active processing jobs for timeout management
        self.active_jobs = {}
        
        self.logger.info("Limits manager initialized", extra={
            'max_file_size_mb': self.limits.max_file_size_mb,
            'max_duration_seconds': self.limits.max_duration_seconds,
            'max_processing_time_seconds': self.limits.max_processing_time_seconds
        })
    
    async def validate_video_url(self, video_url: str) -> Dict[str, Any]:
        """
        Validate video URL and check file size before downloading
        
        Args:
            video_url: URL of the video to validate
            
        Returns:
            dict: Validation results with file info
            
        Raises:
            VideoSizeLimitError: If file exceeds size limits
        """
        try:
            # Get file size without downloading
            async with aiohttp.ClientSession() as session:
                async with session.head(video_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: Could not access video URL")
                    
                    content_length = response.headers.get('content-length')
                    content_type = response.headers.get('content-type', '')
                    
                    if content_length:
                        file_size_mb = int(content_length) / (1024 * 1024)
                        
                        if file_size_mb > self.limits.max_file_size_mb:
                            raise VideoSizeLimitError(
                                f"Video file size ({file_size_mb:.1f}MB) exceeds limit of "
                                f"{self.limits.max_file_size_mb}MB"
                            )
                    
                    # Validate content type
                    if content_type and not any(fmt in content_type.lower() 
                                              for fmt in ['video/', 'application/octet-stream']):
                        raise Exception(f"Invalid content type: {content_type}")
                    
                    validation_result = {
                        'file_size_mb': float(content_length) / (1024 * 1024) if content_length else None,
                        'content_type': content_type,
                        'url_valid': True,
                        'within_size_limit': True
                    }
                    
                    self.logger.info("Video URL validated", extra={
                        'url': video_url[:100] + '...' if len(video_url) > 100 else video_url,
                        'file_size_mb': validation_result['file_size_mb'],
                        'content_type': content_type
                    })
                    
                    return validation_result
                    
        except VideoSizeLimitError:
            raise
        except Exception as e:
            log_error(e, "video_url_validation", url=video_url)
            raise
    
    def validate_video_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate local video file against limits
        
        Args:
            file_path: Path to the video file
            
        Returns:
            dict: Validation results
            
        Raises:
            VideoSizeLimitError: If file exceeds limits
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Video file not found: {file_path}")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size_mb > self.limits.max_file_size_mb:
                raise VideoSizeLimitError(
                    f"Video file size ({file_size_mb:.1f}MB) exceeds limit of "
                    f"{self.limits.max_file_size_mb}MB"
                )
            
            # Check file extension
            file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            if file_ext not in self.limits.allowed_formats:
                raise Exception(f"Unsupported video format: {file_ext}")
            
            validation_result = {
                'file_size_mb': file_size_mb,
                'file_extension': file_ext,
                'file_exists': True,
                'within_size_limit': True,
                'format_supported': True
            }
            
            self.logger.info("Video file validated", extra={
                'file_path': file_path,
                'file_size_mb': file_size_mb,
                'file_extension': file_ext
            })
            
            return validation_result
            
        except VideoSizeLimitError:
            raise
        except Exception as e:
            log_error(e, "video_file_validation", file_path=file_path)
            raise
    
    def validate_video_parameters(self, width: int, height: int, duration: float, 
                                 bitrate: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate video parameters against limits
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            duration: Video duration in seconds
            bitrate: Video bitrate in Mbps (optional)
            
        Returns:
            dict: Validation results
            
        Raises:
            VideoSizeLimitError: If parameters exceed limits
        """
        issues = []
        
        # Check resolution
        total_pixels = width * height
        if total_pixels > self.limits.max_resolution_pixels:
            issues.append(f"Resolution {width}x{height} ({total_pixels} pixels) exceeds limit of {self.limits.max_resolution_pixels} pixels")
        
        # Check duration
        if duration > self.limits.max_duration_seconds:
            issues.append(f"Duration {duration}s exceeds limit of {self.limits.max_duration_seconds}s")
        
        # Check bitrate if provided
        if bitrate and bitrate > self.limits.max_bitrate_mbps:
            issues.append(f"Bitrate {bitrate}Mbps exceeds limit of {self.limits.max_bitrate_mbps}Mbps")
        
        if issues:
            raise VideoSizeLimitError("; ".join(issues))
        
        validation_result = {
            'width': width,
            'height': height,
            'duration': duration,
            'bitrate': bitrate,
            'total_pixels': total_pixels,
            'within_resolution_limit': True,
            'within_duration_limit': True,
            'within_bitrate_limit': bitrate is None or bitrate <= self.limits.max_bitrate_mbps
        }
        
        self.logger.info("Video parameters validated", extra=validation_result)
        
        return validation_result
    
    def start_processing_timeout(self, job_id: str) -> None:
        """
        Start processing timeout tracking for a job
        
        Args:
            job_id: Unique job identifier
        """
        self.active_jobs[job_id] = {
            'start_time': time.time(),
            'max_duration': self.limits.max_processing_time_seconds
        }
        
        self.logger.info("Processing timeout started", extra={
            'job_id': job_id,
            'max_duration': self.limits.max_processing_time_seconds
        })
    
    def check_processing_timeout(self, job_id: str) -> None:
        """
        Check if processing has exceeded timeout limits
        
        Args:
            job_id: Job identifier to check
            
        Raises:
            ProcessingTimeoutError: If processing has exceeded time limits
        """
        if job_id not in self.active_jobs:
            return
        
        job_info = self.active_jobs[job_id]
        elapsed_time = time.time() - job_info['start_time']
        
        if elapsed_time > job_info['max_duration']:
            self.stop_processing_timeout(job_id)
            raise ProcessingTimeoutError(
                f"Processing time ({elapsed_time:.1f}s) exceeded limit of "
                f"{job_info['max_duration']}s for job {job_id}"
            )
    
    def stop_processing_timeout(self, job_id: str) -> Optional[float]:
        """
        Stop processing timeout tracking and return elapsed time
        
        Args:
            job_id: Job identifier
            
        Returns:
            float: Elapsed processing time in seconds, or None if job not found
        """
        if job_id not in self.active_jobs:
            return None
        
        job_info = self.active_jobs.pop(job_id)
        elapsed_time = time.time() - job_info['start_time']
        
        self.logger.info("Processing timeout stopped", extra={
            'job_id': job_id,
            'elapsed_time': elapsed_time,
            'within_limit': elapsed_time <= job_info['max_duration']
        })
        
        return elapsed_time
    
    def get_processing_stats(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get processing statistics for a job
        
        Args:
            job_id: Job identifier
            
        Returns:
            dict: Processing statistics or None if job not found
        """
        if job_id not in self.active_jobs:
            return None
        
        job_info = self.active_jobs[job_id]
        elapsed_time = time.time() - job_info['start_time']
        remaining_time = max(0, job_info['max_duration'] - elapsed_time)
        
        return {
            'job_id': job_id,
            'elapsed_time': elapsed_time,
            'remaining_time': remaining_time,
            'max_duration': job_info['max_duration'],
            'progress_percent': min(100, (elapsed_time / job_info['max_duration']) * 100)
        }
    
    def cleanup_expired_jobs(self) -> int:
        """
        Clean up expired job tracking entries
        
        Returns:
            int: Number of expired jobs cleaned up
        """
        current_time = time.time()
        expired_jobs = []
        
        for job_id, job_info in self.active_jobs.items():
            if current_time - job_info['start_time'] > job_info['max_duration'] * 2:  # Double timeout
                expired_jobs.append(job_id)
        
        for job_id in expired_jobs:
            del self.active_jobs[job_id]
        
        if expired_jobs:
            self.logger.info(f"Cleaned up {len(expired_jobs)} expired job tracking entries")
        
        return len(expired_jobs)
    
    def get_current_limits(self) -> Dict[str, Any]:
        """
        Get current processing limits
        
        Returns:
            dict: Current limits configuration
        """
        return {
            'max_file_size_mb': self.limits.max_file_size_mb,
            'max_duration_seconds': self.limits.max_duration_seconds,
            'max_resolution_pixels': self.limits.max_resolution_pixels,
            'max_processing_time_seconds': self.limits.max_processing_time_seconds,
            'max_bitrate_mbps': self.limits.max_bitrate_mbps,
            'allowed_formats': self.limits.allowed_formats,
            'active_jobs_count': len(self.active_jobs)
        }

# Global limits manager instance
limits_manager = LimitsManager()

def get_limits_manager() -> LimitsManager:
    """Get the global limits manager instance"""
    return limits_manager