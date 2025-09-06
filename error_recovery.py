"""
Error Recovery Mechanisms
Provides retry logic, fallback strategies, and recovery mechanisms for video processing.
"""
import time
import functools
from typing import Callable, Any, Optional, Dict, List
from logger import get_logger, log_error
from config import get_settings

class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_multiplier: float = 2.0,
                 retriable_exceptions: tuple = (Exception,)):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.retriable_exceptions = retriable_exceptions

def exponential_backoff_retry(config: RetryConfig):
    """Decorator for exponential backoff retry logic"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_logger("retry")
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(f"Operation succeeded on attempt {attempt}", extra={
                            'function': func.__name__,
                            'attempt': attempt,
                            'total_attempts': config.max_attempts
                        })
                    return result
                    
                except config.retriable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error(f"Operation failed after {config.max_attempts} attempts", extra={
                            'function': func.__name__,
                            'final_error': str(e),
                            'total_attempts': config.max_attempts
                        })
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.backoff_multiplier ** (attempt - 1)),
                        config.max_delay
                    )
                    
                    logger.warning(f"Attempt {attempt} failed, retrying in {delay:.2f}s", extra={
                        'function': func.__name__,
                        'attempt': attempt,
                        'error': str(e),
                        'retry_delay': delay
                    })
                    
                    time.sleep(delay)
            
            # This should never be reached due to the raise above, but just in case
            raise last_exception
            
        return wrapper
    return decorator

class VideoProcessingRecovery:
    """Recovery mechanisms for video processing operations"""
    
    def __init__(self):
        self.logger = get_logger("recovery")
        self.settings = get_settings()
        
    def recover_from_memory_error(self, operation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement recovery strategies for memory-related errors
        
        Returns:
            dict: Recovery actions taken and new parameters
        """
        recovery_actions = []
        new_params = operation_context.copy()
        
        # Reduce video quality
        if new_params.get('quality', 'high') != 'low':
            old_quality = new_params.get('quality', 'high')
            new_params['quality'] = 'low'
            recovery_actions.append(f"Reduced quality from {old_quality} to low")
            
        # Reduce concurrent processes
        current_processes = new_params.get('max_concurrent', 4)
        if current_processes > 1:
            new_params['max_concurrent'] = max(1, current_processes // 2)
            recovery_actions.append(f"Reduced concurrent processes from {current_processes} to {new_params['max_concurrent']}")
        
        # Limit clip duration
        if new_params.get('clip_duration', 0) > 60:
            old_duration = new_params['clip_duration']
            new_params['clip_duration'] = 60
            recovery_actions.append(f"Limited clip duration from {old_duration}s to 60s")
        
        self.logger.info("Memory error recovery actions applied", extra={
            'actions': recovery_actions,
            'new_params': new_params
        })
        
        return {
            'actions': recovery_actions,
            'new_params': new_params,
            'recovery_applied': len(recovery_actions) > 0
        }
    
    def recover_from_ffmpeg_error(self, error_message: str, operation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement recovery strategies for FFmpeg-related errors
        
        Returns:
            dict: Recovery actions and fallback parameters
        """
        recovery_actions = []
        new_params = operation_context.copy()
        
        # Check for codec issues
        if 'codec' in error_message.lower() or 'encoder' in error_message.lower():
            # Fall back to software encoding
            if new_params.get('hw_accel_enabled', True):
                new_params['hw_accel_enabled'] = False
                recovery_actions.append("Disabled hardware acceleration, falling back to software encoding")
            
            # Try different codec
            if new_params.get('video_codec') != 'libx264':
                new_params['video_codec'] = 'libx264'
                recovery_actions.append("Switched to libx264 codec for better compatibility")
        
        # Check for resolution issues
        if 'resolution' in error_message.lower() or 'dimensions' in error_message.lower():
            # Reduce resolution
            width = new_params.get('width', 1920)
            height = new_params.get('height', 1080)
            
            if width > 1280 or height > 720:
                new_params['width'] = 1280
                new_params['height'] = 720
                recovery_actions.append("Reduced resolution to 1280x720 for compatibility")
        
        # Check for filter issues
        if 'filter' in error_message.lower():
            # Simplify filters
            if new_params.get('use_complex_filters', True):
                new_params['use_complex_filters'] = False
                recovery_actions.append("Simplified video filters to avoid filter chain issues")
        
        self.logger.info("FFmpeg error recovery actions applied", extra={
            'error_message': error_message,
            'actions': recovery_actions,
            'new_params': new_params
        })
        
        return {
            'actions': recovery_actions,
            'new_params': new_params,
            'recovery_applied': len(recovery_actions) > 0
        }
    
    def recover_from_network_error(self, operation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement recovery strategies for network-related errors
        
        Returns:
            dict: Recovery actions and retry parameters
        """
        recovery_actions = []
        new_params = operation_context.copy()
        
        # Increase timeout
        current_timeout = new_params.get('timeout', 30)
        if current_timeout < 120:
            new_params['timeout'] = min(current_timeout * 2, 120)
            recovery_actions.append(f"Increased timeout from {current_timeout}s to {new_params['timeout']}s")
        
        # Enable retry with longer delays
        new_params['retry_delays'] = [5, 15, 30]  # Progressive delays
        recovery_actions.append("Enabled progressive retry delays for network operations")
        
        # Use chunked downloads for large files
        new_params['use_chunked_download'] = True
        recovery_actions.append("Enabled chunked downloads for better network reliability")
        
        self.logger.info("Network error recovery actions applied", extra={
            'actions': recovery_actions,
            'new_params': new_params
        })
        
        return {
            'actions': recovery_actions,
            'new_params': new_params,
            'recovery_applied': len(recovery_actions) > 0
        }

class CircuitBreaker:
    """Circuit breaker pattern for preventing cascade failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self.logger = get_logger("circuit_breaker")
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == 'open':
                if time.time() - self.last_failure_time < self.timeout:
                    raise Exception("Circuit breaker is OPEN - operation temporarily unavailable")
                else:
                    self.state = 'half-open'
                    self.logger.info("Circuit breaker transitioning to HALF-OPEN")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == 'half-open':
                    self.state = 'closed'
                    self.failure_count = 0
                    self.logger.info("Circuit breaker CLOSED - service recovered")
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                    self.logger.error(f"Circuit breaker OPEN - {self.failure_count} failures", extra={
                        'function': func.__name__,
                        'error': str(e),
                        'failure_count': self.failure_count
                    })
                
                raise
        
        return wrapper

# Pre-configured retry strategies for common operations
FFMPEG_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=30.0,
    retriable_exceptions=(subprocess.SubprocessError, OSError, IOError)
)

NETWORK_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0,
    retriable_exceptions=(ConnectionError, TimeoutError, OSError)
)

DATABASE_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    retriable_exceptions=(Exception,)  # Database-specific exceptions would go here
)

# Global recovery instance
recovery_manager = VideoProcessingRecovery()