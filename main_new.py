"""

FastAPI Application - Clean Architecture Implementation
Improved separation of concerns with dependency injection and layered architecture.
"""
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, validator, Field
from typing import Optional, List, Dict, Any
import asyncio
import uuid

# Import configuration and dependency injection
from dependency_container import initialize_container, get_container
from service_providers import ApplicationServiceRegistry
from api_controllers import (
    VideoProcessingController,
    SystemController, 
    HardwareController,
    ResourceController
)
from domain_services import EffectConfiguration
from config import get_settings
from logger import get_logger


# Pydantic models for API validation
class VideoProcessingRequest(BaseModel):
    """Input validation for legacy video processing"""
    video_url: HttpUrl = Field(..., description="S3 URL for the video file")
    audio_url: HttpUrl = Field(..., description="S3 URL for the audio file")  
    overlay_url: Optional[HttpUrl] = Field(None, description="Optional S3 URL for overlay image")
    normal_play: float = Field(0, ge=0, le=300, description="Duration of normal speed playback")
    slow_play: float = Field(0, ge=0, le=300, description="Duration of slow motion playback")
    effect: str = Field("slomo_boomerang", regex="^(slomo|slomo_boomerang|custom_sequence)$")
    slow_factor: float = Field(0.5, gt=0.1, le=1.0, description="Slow motion factor")
    stabilize: bool = Field(False, description="Apply video stabilization")
    
    @validator('video_url', 'audio_url', 'overlay_url')
    def validate_s3_urls(cls, v):
        if v is not None:
            url_str = str(v)
            settings = get_settings()
            if not any(domain in url_str for domain in settings.allowed_s3_domains):
                raise ValueError('URL must be from a trusted S3 domain')
        return v


class VideoEffectRequest(BaseModel):
    """Input validation for single video effect"""
    video_url: HttpUrl = Field(..., description="S3 URL for the input video file")
    effect_type: str = Field(..., regex="^(fade_in_out|blur|overlay)$")
    
    # Fade effect parameters
    fade_in_duration: Optional[float] = Field(1.0, ge=0, le=10)
    fade_out_duration: Optional[float] = Field(1.0, ge=0, le=10) 
    fade_color: Optional[str] = Field("black", regex="^(black|white|transparent)$")
    
    # Blur effect parameters
    blur_type: Optional[str] = Field("gaussian", regex="^(gaussian|motion|radial)$")
    blur_strength: Optional[float] = Field(5.0, ge=0.1, le=50.0)
    blur_duration: Optional[float] = Field(None, ge=0)
    
    # Overlay effect parameters
    overlay_url: Optional[HttpUrl] = Field(None)
    overlay_position: Optional[str] = Field("center", regex="^(center|top-left|top-right|bottom-left|bottom-right)$")
    overlay_scale: Optional[float] = Field(1.0, ge=0.1, le=2.0)
    overlay_opacity: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    
    @validator('video_url', 'overlay_url')
    def validate_s3_urls(cls, v):
        if v is not None:
            url_str = str(v)
            settings = get_settings()
            if not any(domain in url_str for domain in settings.allowed_s3_domains):
                raise ValueError('URL must be from a trusted S3 domain')
        return v


class CombinedEffectsRequest(BaseModel):
    """Input validation for combined effects"""
    video_url: HttpUrl = Field(..., description="S3 URL for the input video file")
    effects: List[Dict[str, Any]] = Field(..., min_items=1, max_items=5)
    
    @validator('video_url')
    def validate_s3_urls(cls, v):
        if v is not None:
            url_str = str(v)
            settings = get_settings()
            if not any(domain in url_str for domain in settings.allowed_s3_domains):
                raise ValueError('URL must be from a trusted S3 domain')
        return v


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str
    s3_url: Optional[str] = None
    logs: List[str] = []
    created_at: Optional[float] = None


# Application factory function
async def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    # Load configuration
    settings = get_settings()
    logger = get_logger("main")
    
    # Initialize dependency container
    container = await initialize_container()
    ApplicationServiceRegistry.register_all_providers(container)
    await container.initialize()
    
    logger.info("Application dependencies initialized", extra={'version': '3.0'})
    
    # Create FastAPI app
    app = FastAPI(
        title="Video Processing API - Clean Architecture",
        version="3.0.0",
        description="Production-ready video processing API with clean architecture",
        debug=settings.debug
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize controllers
    video_controller = VideoProcessingController()
    system_controller = SystemController()
    hardware_controller = HardwareController()
    resource_controller = ResourceController()
    
    # Store controllers in app state for access in endpoints
    app.state.controllers = {
        "video": video_controller,
        "system": system_controller,
        "hardware": hardware_controller,
        "resources": resource_controller
    }
    
    # Store container for cleanup
    app.state.container = container
    
    return app


# Create the app instance
app_instance = None


async def get_app() -> FastAPI:
    """Get or create the app instance"""
    global app_instance
    if app_instance is None:
        app_instance = await create_app()
    return app_instance


# Initialize app at module level for uvicorn
app = FastAPI(title="Loading...", description="Application is initializing...")


@app.on_event("startup")
async def startup():
    """Initialize the real app on startup"""
    global app, app_instance
    app_instance = await create_app()
    
    # Replace the placeholder app with the real one
    app.router = app_instance.router
    app.title = app_instance.title
    app.description = app_instance.description
    app.version = app_instance.version
    app.state = app_instance.state


@app.on_event("shutdown") 
async def shutdown():
    """Cleanup on shutdown"""
    if hasattr(app.state, 'container'):
        await app.state.container.cleanup()


# API Endpoints - now using clean architecture
@app.post("/process-video-s3-async/", response_model=dict)
async def process_video_legacy(
    background_tasks: BackgroundTasks,
    request: VideoProcessingRequest
):
    """Legacy video processing endpoint (backward compatibility)"""
    controller = app.state.controllers["video"]
    return await controller.process_legacy_video(background_tasks, request.dict())


@app.post("/apply-video-effect/", response_model=dict) 
async def apply_video_effect(
    background_tasks: BackgroundTasks,
    request: VideoEffectRequest
):
    """Apply a single video effect"""
    controller = app.state.controllers["video"]
    return await controller.apply_video_effects(background_tasks, request.dict())


@app.post("/apply-combined-effects/", response_model=dict)
async def apply_combined_effects(
    background_tasks: BackgroundTasks,
    request: CombinedEffectsRequest  
):
    """Apply multiple video effects in sequence"""
    controller = app.state.controllers["video"]
    return await controller.apply_combined_effects(background_tasks, request.dict())


@app.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    controller = app.state.controllers["video"]
    return await controller.get_job_status(job_id)


@app.get("/performance/status")
async def get_system_status():
    """Get comprehensive system performance status"""
    controller = app.state.controllers["system"]
    return await controller.get_system_status()


@app.get("/performance/hardware")
async def get_hardware_info():
    """Get hardware acceleration capabilities"""
    controller = app.state.controllers["hardware"]
    return await controller.get_hardware_info()


@app.get("/performance/resources")
async def get_resource_stats():
    """Get detailed resource usage statistics"""
    controller = app.state.controllers["resources"]
    return await controller.get_resource_stats()


@app.get("/performance/health")
async def get_health_check():
    """Simple health check endpoint"""
    controller = app.state.controllers["system"]
    return await controller.get_health_check()


@app.post("/performance/cleanup")
async def trigger_system_cleanup():
    """Trigger system cleanup and optimization"""
    controller = app.state.controllers["system"]
    return await controller.trigger_system_cleanup()


@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "message": "Video Processing API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)