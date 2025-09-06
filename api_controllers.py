"""
API Controllers - Presentation Layer
Handles HTTP requests/responses and delegates to domain services.
Separated from FastAPI app for better testability and organization.
"""
import asyncio
from typing import Dict, Any, List
from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from dependency_container import inject, ConfiguredService
from domain_services import (
    VideoProcessingService, 
    JobRepository, 
    SystemHealthService,
    EffectConfiguration,
    ProcessingStatus
)
from service_providers import ServiceLocator
from logger import LoggedOperation


class VideoProcessingController(ConfiguredService):
    """Controller for video processing endpoints"""
    
    def __init__(self):
        super().__init__()
        self.services = ServiceLocator()
        self.processing_service = VideoProcessingService()
        self.job_repository = JobRepository()
        self._active_jobs: Dict[str, Any] = {}  # In-memory job tracking
    
    async def process_legacy_video(self, background_tasks: BackgroundTasks, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle legacy video processing endpoint (backward compatibility)"""
        
        # Check system resources
        memory_stats = self.services.resource_manager.get_memory_stats()
        if memory_stats["system_memory_percent"] > 95:
            raise HTTPException(status_code=503, detail="High memory usage")
        
        # Create job
        import uuid
        job_id = str(uuid.uuid4())
        
        # Store job in memory for status tracking
        self._active_jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "parameters": request_data,
            "logs": ["Job created and queued"],
            "created_at": asyncio.get_event_loop().time()
        }
        
        # Add background task
        background_tasks.add_task(self._process_legacy_video_background, job_id, request_data)
        
        return {
            "status": "processing",
            "job_id": job_id,
            "message": "Video processing job started"
        }
    
    async def apply_video_effects(self, background_tasks: BackgroundTasks, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply video effects using new architecture"""
        
        # Check system resources
        memory_stats = self.services.resource_manager.get_memory_stats()
        if memory_stats["system_memory_percent"] > 95:
            raise HTTPException(status_code=503, detail="High memory usage")
        
        # Parse effects from request
        effects = []
        effect_type = request_data.get("effect_type")
        if effect_type:
            effect_params = {k: v for k, v in request_data.items() 
                           if k not in ["video_url", "effect_type"]}
            effects.append(EffectConfiguration(effect_type=effect_type, parameters=effect_params))
        
        # Create job
        job = await self.processing_service.create_processing_job(
            video_url=request_data["video_url"],
            effects=effects
        )
        
        # Store job for status tracking
        self._active_jobs[job.job_id] = {
            "id": job.job_id,
            "status": job.status.value,
            "parameters": request_data,
            "logs": job.logs.copy(),
            "created_at": job.created_at
        }
        
        # Add background task
        background_tasks.add_task(self._process_video_effects_background, job)
        
        return {
            "status": "processing", 
            "job_id": job.job_id,
            "message": f"Video effect ({effect_type}) job started"
        }
    
    async def apply_combined_effects(self, background_tasks: BackgroundTasks, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple video effects"""
        
        # Check system resources
        memory_stats = self.services.resource_manager.get_memory_stats()
        if memory_stats["system_memory_percent"] > 95:
            raise HTTPException(status_code=503, detail="High memory usage")
        
        # Parse effects from request
        effects = [
            EffectConfiguration(
                effect_type=effect_config.get("type", effect_config.get("effect_type")),
                parameters=effect_config.get("parameters", {})
            )
            for effect_config in request_data["effects"]
        ]
        
        # Create job
        job = await self.processing_service.create_processing_job(
            video_url=request_data["video_url"],
            effects=effects
        )
        
        # Store job for status tracking  
        self._active_jobs[job.job_id] = {
            "id": job.job_id,
            "status": job.status.value,
            "parameters": request_data,
            "logs": job.logs.copy(),
            "created_at": job.created_at
        }
        
        # Add background task
        background_tasks.add_task(self._process_video_effects_background, job)
        
        return {
            "status": "processing",
            "job_id": job.job_id,
            "message": f"Combined effects job started with {len(effects)} effects"
        }
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a processing job"""
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job["status"],
                "s3_url": job.get("s3_url"),
                "logs": job.get("logs", [])[-10:],  # Last 10 log entries
                "created_at": job.get("created_at")
            }
        else:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    async def _process_legacy_video_background(self, job_id: str, request_data: Dict[str, Any]) -> None:
        """Background processing for legacy video processing"""
        with LoggedOperation("legacy_video_processing", job_id=job_id) as operation:
            try:
                self._active_jobs[job_id]["status"] = "processing"
                self._active_jobs[job_id]["logs"].append("Processing started")
                
                # Here you would implement the legacy processing logic
                # For now, we'll simulate processing
                await asyncio.sleep(2)
                
                # Simulate successful completion
                self._active_jobs[job_id]["status"] = "completed"
                self._active_jobs[job_id]["s3_url"] = f"https://example.com/processed/{job_id}.mp4"
                self._active_jobs[job_id]["logs"].append("Processing completed successfully")
                
            except Exception as e:
                self._active_jobs[job_id]["status"] = "failed"
                self._active_jobs[job_id]["logs"].append(f"Processing failed: {e}")
                self.logger.error(f"Legacy processing failed for job {job_id}: {e}", exc_info=True)
    
    async def _process_video_effects_background(self, job) -> None:
        """Background processing for video effects using domain services"""
        try:
            # Update in-memory tracking
            self._active_jobs[job.job_id]["status"] = "processing"
            self._active_jobs[job.job_id]["logs"] = job.logs.copy()
            
            # Process using domain service
            completed_job = await self.processing_service.process_video_with_effects(job)
            
            # Update in-memory tracking
            self._active_jobs[job.job_id]["status"] = completed_job.status.value
            self._active_jobs[job.job_id]["s3_url"] = completed_job.output_url
            self._active_jobs[job.job_id]["logs"] = completed_job.logs.copy()
            
            # Save to repository
            await self.job_repository.save_job(completed_job)
            
        except Exception as e:
            self._active_jobs[job.job_id]["status"] = "failed"
            self._active_jobs[job.job_id]["logs"].append(f"Processing failed: {e}")
            self.logger.error(f"Effects processing failed for job {job.job_id}: {e}", exc_info=True)


class SystemController(ConfiguredService):
    """Controller for system monitoring and health endpoints"""
    
    def __init__(self):
        super().__init__()
        self.health_service = SystemHealthService()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = await self.health_service.get_system_status()
            return {"status": "success", "data": status}
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")
    
    async def get_health_check(self) -> Dict[str, Any]:
        """Simple health check endpoint"""
        try:
            health_info = await self.health_service.check_health()
            
            http_status = 200 if health_info["healthy"] else 503
            response_status = "success" if health_info["healthy"] else "degraded"
            
            return JSONResponse(
                content={"status": response_status, "data": health_info},
                status_code=http_status
            )
        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return JSONResponse(
                content={
                    "status": "error",
                    "error": str(e),
                    "timestamp": asyncio.get_event_loop().time()
                },
                status_code=503
            )
    
    async def trigger_system_cleanup(self) -> Dict[str, Any]:
        """Trigger system cleanup and optimization"""
        try:
            result = await self.health_service.cleanup_system()
            return {"status": "success", "message": result["message"]}
        except Exception as e:
            self.logger.error(f"Error during system cleanup: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"System cleanup failed: {str(e)}")


class HardwareController(ConfiguredService):
    """Controller for hardware information endpoints"""
    
    def __init__(self):
        super().__init__()
        self.services = ServiceLocator()
    
    async def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware acceleration capabilities"""
        try:
            hw_manager = self.services.hardware_manager
            capabilities = hw_manager.get_capabilities()
            
            return {
                "status": "success",
                "data": {
                    "acceleration_type": capabilities.acceleration_type,
                    "device": capabilities.device,
                    "max_concurrent_streams": capabilities.max_concurrent_streams,
                    "supports_hevc": capabilities.supports_hevc,
                    "supports_av1": capabilities.supports_av1,
                    "memory_limit_mb": capabilities.memory_limit_mb,
                    "driver_version": capabilities.driver_version,
                    "optimal_concurrency": hw_manager.get_optimal_concurrency()
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting hardware info: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to get hardware info: {str(e)}")


class ResourceController(ConfiguredService):
    """Controller for resource monitoring endpoints"""
    
    def __init__(self):
        super().__init__()
        self.services = ServiceLocator()
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get detailed resource usage statistics"""
        try:
            resource_stats = self.services.resource_manager.get_memory_stats()
            
            # Add additional system stats
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            disk_usage = psutil.disk_usage('/')
            
            extended_stats = {
                **resource_stats,
                "cpu_percent": cpu_percent,
                "disk_usage": {
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": disk_usage.free / (1024**3),
                    "percent": (disk_usage.used / disk_usage.total) * 100
                },
                "active_processes_details": self.services.resource_manager.get_active_processes_info()
            }
            
            return {"status": "success", "data": extended_stats}
        except Exception as e:
            self.logger.error(f"Error getting resource stats: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to get resource stats: {str(e)}")