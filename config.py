"""
Secure Configuration Management
Centralizes all configuration settings with proper validation and security.
"""
from pydantic import Field, field_validator
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Secure application settings with validation"""
    
    # Database Configuration
    database_url: str = Field(..., env="DATABASE_URL", description="Database connection URL")
    
    # AWS Configuration  
    aws_access_key: str = Field(..., env="AWS_ACCESS_KEY", description="AWS Access Key")
    aws_secret_key: str = Field(..., env="AWS_SECRET_KEY", description="AWS Secret Key") 
    aws_region: str = Field("us-east-1", env="AWS_REGION", description="AWS Region")
    s3_bucket: str = Field("pixcelcapetown", env="S3_BUCKET", description="S3 bucket name")
    
    # Application Configuration
    max_concurrent_processes: int = Field(4, ge=1, le=16, env="MAX_CONCURRENT_PROCESSES", description="Maximum concurrent video processing tasks")
    max_video_size_mb: int = Field(500, ge=10, le=2000, env="MAX_VIDEO_SIZE_MB", description="Maximum video file size in MB")
    max_processing_duration: int = Field(300, ge=30, le=1800, env="MAX_PROCESSING_DURATION", description="Maximum processing time in seconds")
    
    # Security Settings
    allowed_s3_domains: list = Field(
        default=[
            "amazonaws.com",
            "s3.amazonaws.com", 
            "s3-",
            "pixcelcapetown.s3."
        ],
        description="Allowed S3 domains for URL validation"
    )
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST", description="API host")
    api_port: int = Field(8000, ge=1000, le=65535, env="API_PORT", description="API port")
    debug: bool = Field(False, env="DEBUG", description="Debug mode")
    
    # Resource Management
    memory_warning_threshold: int = Field(85, ge=50, le=95, env="MEMORY_WARNING_THRESHOLD", description="Memory usage warning threshold (%)")
    memory_critical_threshold: int = Field(95, ge=90, le=99, env="MEMORY_CRITICAL_THRESHOLD", description="Memory usage critical threshold (%)")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", description="Logging level")
    log_to_file: bool = Field(True, env="LOG_TO_FILE", description="Enable logging to files")
    log_max_size_mb: int = Field(50, ge=10, le=500, env="LOG_MAX_SIZE_MB", description="Maximum log file size in MB")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }
        
    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format"""
        if not v.startswith(('mysql://', 'mysql+pymysql://', 'postgresql://', 'sqlite://')):
            raise ValueError('DATABASE_URL must be a valid database connection string')
        return v
    
    @field_validator('aws_access_key', 'aws_secret_key')
    @classmethod
    def validate_aws_credentials(cls, v):
        """Validate AWS credentials are present and reasonable length"""
        if len(v) < 16:
            raise ValueError('AWS credentials appear to be too short')
        return v
    
    @field_validator('s3_bucket')
    @classmethod
    def validate_s3_bucket(cls, v):
        """Validate S3 bucket name"""
        import re
        if not re.match(r'^[a-z0-9.-]{3,63}$', v):
            raise ValueError('S3 bucket name must be valid')
        return v

# Create global settings instance
try:
    settings = Settings()
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load configuration: {e}")
    print("Please ensure all required environment variables are set in .env file:")
    print("- DATABASE_URL")
    print("- AWS_ACCESS_KEY") 
    print("- AWS_SECRET_KEY")
    raise

def get_settings() -> Settings:
    """Get application settings"""
    return settings