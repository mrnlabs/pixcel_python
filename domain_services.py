"""
Domain Services - Abstraction Layer
Provides high-level business logic abstraction over infrastructure services.
"""
import asyncio
import tempfile
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncContextManager
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum

from dependency_container import ConfiguredService, inject
from service_providers import ServiceLocator
from logger import LoggedOperation


class ProcessingStatus(Enum):
    """Video processing status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VideoProcessingJob:
    """Domain model for video processing job"""
    job_id: str
    status: ProcessingStatus
    input_url: str
    output_url: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    created_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class EffectConfiguration:
    """Configuration for video effects"""
    effect_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class IVideoProcessor(ABC):
    """Abstract interface for video processing"""
    
    @abstractmethod
    async def process_video(self, job: VideoProcessingJob) -> VideoProcessingJob:
        """Process a video according to job specifications"""
        pass
    
    @abstractmethod
    async def apply_effects(self, input_url: str, effects: List[EffectConfiguration]) -> str:
        """Apply effects to a video and return output URL"""
        pass
    
    @abstractmethod
    async def get_job_status(self, job_id: str) -> Optional[VideoProcessingJob]:
        """Get the status of a processing job"""
        pass


class IJobRepository(ABC):
    """Abstract interface for job persistence"""
    
    @abstractmethod
    async def save_job(self, job: VideoProcessingJob) -> None:
        """Save a job to persistent storage"""
        pass
    
    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[VideoProcessingJob]:
        """Get a job from persistent storage"""
        pass
    
    @abstractmethod
    async def update_job_status(self, job_id: str, status: ProcessingStatus, 
                               error_message: str = None) -> None:
        """Update job status"""
        pass
    
    @abstractmethod
    async def list_jobs(self, limit: int = 100) -> List[VideoProcessingJob]:
        """List recent jobs"""
        pass


class VideoProcessingService(ConfiguredService):
    """High-level video processing domain service"""
    
    def __init__(self):
        super().__init__()
        self.services = ServiceLocator()
        self._temp_files: List[str] = []
    
    async def create_processing_job(self, 
                                   video_url: str,
                                   effects: List[EffectConfiguration],
                                   job_id: str = None) -> VideoProcessingJob:
        """Create a new video processing job"""
        import uuid
        import time
        
        if not job_id:
            job_id = str(uuid.uuid4())
        
        job = VideoProcessingJob(
            job_id=job_id,
            status=ProcessingStatus.QUEUED,
            input_url=video_url,
            parameters={
                "effects": [{"effect_type": e.effect_type, "parameters": e.parameters} 
                           for e in effects]
            },
            created_at=time.time()
        )
        
        self.logger.info(f"Created processing job {job_id}", 
                        extra={"job_id": job_id, "input_url": video_url})
        
        return job
    
    @asynccontextmanager
    async def managed_processing_context(self, job: VideoProcessingJob) -> AsyncContextManager[Dict[str, Any]]:
        """Managed context for video processing with resource cleanup"""
        temp_files = []
        context = {"temp_files": temp_files}
        
        try:
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp(prefix=f"video_job_{job.job_id}_")
            temp_files.append(temp_dir)
            context["temp_dir"] = temp_dir
            
            # Initialize performance monitoring
            performance_optimizer = self.services.performance_optimizer
            processing_context = await performance_optimizer.create_processing_context(
                job_id=job.job_id,
                video_url=job.input_url
            )
            context["processing_context"] = processing_context
            
            self.logger.info(f"Processing context created for job {job.job_id}")
            yield context
            
        finally:
            # Cleanup temporary files
            for temp_path in temp_files:
                try:
                    if os.path.isfile(temp_path):
                        os.unlink(temp_path)
                    elif os.path.isdir(temp_path):
                        import shutil
                        shutil.rmtree(temp_path)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
            
            self.logger.info(f"Processing context cleaned up for job {job.job_id}")
    
    async def process_video_with_effects(self, job: VideoProcessingJob) -> VideoProcessingJob:
        """Process video with the specified effects"""
        async with self.managed_processing_context(job) as context:
            with LoggedOperation("video_processing", job_id=job.job_id) as operation:
                try:
                    job.status = ProcessingStatus.PROCESSING
                    job.logs.append("Processing started")
                    
                    # Download input video
                    input_path = os.path.join(context["temp_dir"], "input.mp4")
                    await self.services.s3_handler.download_file(job.input_url, input_path)
                    job.logs.append("Input video downloaded")
                    
                    # Apply effects
                    output_path = os.path.join(context["temp_dir"], "output.mp4")
                    await self._apply_effects_to_video(input_path, output_path, job.parameters["effects"])
                    job.logs.append("Effects applied successfully")
                    
                    # Upload result
                    upload_key = f"processed_videos/{job.job_id}_processed.mp4"
                    job.output_url = await self.services.s3_handler.upload_file(output_path, upload_key)
                    job.logs.append(f"Output uploaded: {job.output_url}")
                    
                    job.status = ProcessingStatus.COMPLETED
                    import time
                    job.completed_at = time.time()
                    
                    self.logger.info(f"Job {job.job_id} completed successfully")
                    
                except Exception as e:
                    job.status = ProcessingStatus.FAILED
                    job.error_message = str(e)
                    job.logs.append(f"Processing failed: {e}")
                    
                    self.logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
                    raise
        
        return job
    
    async def _apply_effects_to_video(self, input_path: str, output_path: str, 
                                     effects_config: List[Dict[str, Any]]) -> None:
        """Apply effects to video file"""
        if not effects_config:
            # No effects, just copy the file
            import shutil
            shutil.copy2(input_path, output_path)
            return
        
        # Create video editor
        video_editor = self.services.create_video_editor(input_path)
        
        # Convert effect configurations to EffectConfiguration objects
        effects = [
            EffectConfiguration(effect_type=config["effect_type"], 
                              parameters=config.get("parameters", {}))
            for config in effects_config
        ]
        
        if len(effects) == 1:
            # Single effect
            effect = effects[0]
            success = await self._apply_single_effect(video_editor, effect, output_path)
        else:
            # Multiple effects - use combined effects
            combined_config = [{"type": e.effect_type, **e.parameters} for e in effects]
            success = video_editor.create_combined_effects(output_path, combined_config)
        
        if not success:
            raise Exception("Failed to apply video effects")
    
    async def _apply_single_effect(self, video_editor, effect: EffectConfiguration, 
                                  output_path: str) -> bool:
        """Apply a single effect to the video"""
        effect_type = effect.effect_type
        params = effect.parameters
        
        if effect_type == "fade_in_out":
            return video_editor.create_fade_effect(
                output_path,
                fade_type="fade_in_out",
                fade_in_duration=params.get("fade_in_duration", 1.0),
                fade_out_duration=params.get("fade_out_duration", 1.0),
                fade_color=params.get("fade_color", "black")
            )
        
        elif effect_type == "blur":
            return video_editor.create_blur_effect(
                output_path,
                blur_type=params.get("blur_type", "gaussian"),
                blur_strength=params.get("blur_strength", 5.0),
                blur_duration=params.get("blur_duration")
            )
        
        elif effect_type == "overlay":
            overlay_url = params.get("overlay_url")
            if not overlay_url:
                raise ValueError("Overlay effect requires overlay_url parameter")
                
            # Download overlay file
            overlay_path = os.path.join(os.path.dirname(output_path), "overlay.png")
            await self.services.s3_handler.download_file(overlay_url, overlay_path)
            
            return video_editor.create_overlay_effect(
                output_path,
                overlay_path,
                position=params.get("overlay_position", "center"),
                scale=params.get("overlay_scale", 1.0),
                opacity=params.get("overlay_opacity", 1.0)
            )
        
        else:
            raise ValueError(f"Unknown effect type: {effect_type}")


class JobRepository(ConfiguredService):
    """Database-backed job repository implementation"""
    
    def __init__(self):
        super().__init__()
        self.services = ServiceLocator()
    
    async def save_job(self, job: VideoProcessingJob) -> None:
        """Save a job to the database"""
        # Implementation would use database_manager to persist job
        # For now, we'll store in memory (this should be replaced with actual DB persistence)
        self.logger.info(f"Saving job {job.job_id} to database")
    
    async def get_job(self, job_id: str) -> Optional[VideoProcessingJob]:
        """Get a job from the database"""
        # Implementation would query database
        self.logger.debug(f"Retrieving job {job_id} from database")
        return None
    
    async def update_job_status(self, job_id: str, status: ProcessingStatus, 
                               error_message: str = None) -> None:
        """Update job status in database"""
        self.logger.info(f"Updating job {job_id} status to {status.value}")
    
    async def list_jobs(self, limit: int = 100) -> List[VideoProcessingJob]:
        """List recent jobs from database"""
        self.logger.debug(f"Listing {limit} recent jobs from database")
        return []


class SystemHealthService(ConfiguredService):
    """Service for monitoring system health and performance"""
    
    def __init__(self):
        super().__init__()
        self.services = ServiceLocator()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        performance_optimizer = self.services.performance_optimizer
        return await performance_optimizer.get_system_status()
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform health check"""
        status = await self.get_system_status()
        
        health_score = status.get("system_health", {}).get("score", 0)
        is_healthy = health_score >= 70
        
        return {
            "healthy": is_healthy,
            "score": health_score,
            "status": status["system_health"]["status"],
            "checks": {
                "database": status.get("database", {}).get("is_connected", False),
                "hardware": status.get("hardware", {}).get("acceleration_type", "cpu") != "cpu",
                "resources": status.get("resources", {}).get("system_memory_percent", 100) < 90
            }
        }
    
    async def cleanup_system(self) -> Dict[str, Any]:
        """Perform system cleanup"""
        performance_optimizer = self.services.performance_optimizer
        await performance_optimizer.cleanup_and_optimize()
        
        return {"message": "System cleanup completed"}