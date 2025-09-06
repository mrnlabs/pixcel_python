#!/usr/bin/env python3
"""
Test script to demonstrate the new logging system
"""
import os
from logger import get_logger, log_job_event, log_error, LoggedOperation

def test_logging_system():
    """Test the structured logging system"""
    
    # Get loggers for different components
    main_logger = get_logger("test_main")
    job_logger = get_logger("test_jobs")
    
    # Test basic logging
    main_logger.info("Testing logging system startup", extra={'version': '2.0'})
    
    # Test job event logging
    log_job_event("test-job-123", "Job processing started", 
                  video_duration=120.5, effect="slomo")
    
    # Test error logging
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        log_error(e, "test_error_handling", test_param="test_value")
    
    # Test logged operation context manager
    with LoggedOperation("test_video_processing", job_id="test-job-456", 
                        video_file="test.mp4") as operation:
        main_logger.info("Processing video file", extra={'file_size_mb': 150})
        # Simulate some work
        import time
        time.sleep(1)
        main_logger.info("Video processing completed successfully")
    
    # Test different log levels
    main_logger.debug("Debug message - only shown if debug enabled")
    main_logger.info("Info message - general information")
    main_logger.warning("Warning message - something to be aware of")
    main_logger.error("Error message - something went wrong")
    
    print("\n=== Logging Test Complete ===")
    print("Check the following log files:")
    print("- logs/video_processor.log (general logs)")
    print("- logs/errors.log (error logs only)")
    print("\nThe logs are in structured JSON format for easy parsing!")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    print("=== Testing New Logging System ===")
    test_logging_system()