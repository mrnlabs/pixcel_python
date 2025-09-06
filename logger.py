"""
Structured Logging System
Provides centralized logging with file output, structured data, and proper error handling.
"""
import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from config import get_settings

class StructuredFormatter(logging.Formatter):
    """Custom formatter that creates structured JSON logs"""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add job_id if present
        if hasattr(record, 'job_id'):
            log_entry['job_id'] = record.job_id
            
        # Add process info if present
        if hasattr(record, 'process_id'):
            log_entry['process_id'] = record.process_id
            
        # Add error info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and record.stack_info:
            log_entry['stack_trace'] = record.stack_info
            
        return json.dumps(log_entry)

class VideoProcessingLogger:
    """Centralized logging system for video processing application"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        self.logger = logging.getLogger("video_processor")
        self.logger.setLevel(logging.DEBUG if self.settings.debug else logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add handlers
        self._setup_file_handler()
        self._setup_console_handler()
        self._setup_error_handler()
        
    def _setup_file_handler(self):
        """Setup rotating file handler for general logs"""
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.logs_dir / "video_processor.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
        
    def _setup_console_handler(self):
        """Setup console handler for immediate feedback"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Simple format for console
        console_format = logging.Formatter(
            '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
    def _setup_error_handler(self):
        """Setup dedicated handler for errors"""
        error_handler = logging.handlers.RotatingFileHandler(
            filename=self.logs_dir / "errors.log",
            maxBytes=25 * 1024 * 1024,  # 25MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(error_handler)
        
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific component"""
        return logging.getLogger(f"video_processor.{name}")

# Global logger instance
_logger_instance = None

def get_logger(name: str = "main") -> logging.Logger:
    """Get application logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = VideoProcessingLogger()
    return _logger_instance.get_logger(name)

def log_job_event(job_id: str, event: str, **kwargs):
    """Log job-specific events with structured data"""
    logger = get_logger("jobs")
    extra = {'job_id': job_id, **kwargs}
    logger.info(event, extra=extra)

def log_process_event(process_id: int, event: str, **kwargs):
    """Log process-specific events"""
    logger = get_logger("processes")
    extra = {'process_id': process_id, **kwargs}
    logger.info(event, extra=extra)

def log_error(error: Exception, context: str = None, **kwargs):
    """Log errors with full context and stack trace"""
    logger = get_logger("errors")
    
    error_data = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        **kwargs
    }
    
    logger.error(f"Error in {context}: {error}", extra=error_data, exc_info=True)

def log_system_metrics(memory_percent: float, cpu_percent: float, active_processes: int):
    """Log system resource metrics"""
    logger = get_logger("system")
    metrics = {
        'memory_percent': memory_percent,
        'cpu_percent': cpu_percent,
        'active_processes': active_processes
    }
    logger.info("System metrics", extra=metrics)

# Context manager for operation logging
class LoggedOperation:
    """Context manager for logging operation start/end with timing"""
    
    def __init__(self, operation_name: str, job_id: str = None, **kwargs):
        self.operation_name = operation_name
        self.job_id = job_id
        self.kwargs = kwargs
        self.logger = get_logger("operations")
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.utcnow()
        extra = {'job_id': self.job_id} if self.job_id else {}
        extra.update(self.kwargs)
        
        self.logger.info(f"Starting {self.operation_name}", extra=extra)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        extra = {'job_id': self.job_id, 'duration_seconds': duration} if self.job_id else {'duration_seconds': duration}
        extra.update(self.kwargs)
        
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name}", extra=extra)
        else:
            self.logger.error(f"Failed {self.operation_name}: {exc_val}", extra=extra, exc_info=True)

# Initialize logging system on import
try:
    get_logger("startup").info("Logging system initialized")
except Exception as e:
    # Fallback to basic logging if setup fails
    logging.basicConfig(level=logging.INFO)
    logging.error(f"Failed to initialize structured logging: {e}")