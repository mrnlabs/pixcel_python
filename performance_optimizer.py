"""
Performance Optimizer
Integrates all performance and resource management components for optimal video processing.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

from logger import get_logger, log_error
from hardware_manager import get_hardware_manager
from database_manager import get_database_manager
from limits_manager import get_limits_manager
from resource_manager import ResourceManager
from config import get_settings

@dataclass
class ProcessingContext:
    """Context information for video processing optimization"""
    job_id: str
    video_url: str
    estimated_duration: float
    hardware_acceleration: str
    memory_limit_mb: Optional[int] = None
    priority: int = 1  # 1 = low, 5 = high
    retry_count: int = 0

class PerformanceOptimizer:
    """
    Central performance optimization coordinator that integrates all resource management
    """
    
    def __init__(self):
        self.logger = get_logger("performance_optimizer")
        self.settings = get_settings()
        
        # Initialize managers
        self.hardware_manager = get_hardware_manager()
        self.database_manager = get_database_manager()
        self.limits_manager = get_limits_manager()
        
        # Create optimized resource manager
        optimal_concurrency = self._calculate_optimal_concurrency()
        self.resource_manager = ResourceManager(max_concurrent_processes=optimal_concurrency)
        self.resource_manager.start_monitoring()
        
        # Processing queue for job prioritization
        self.processing_queue = asyncio.PriorityQueue()
        self.active_contexts = {}
        
        self.logger.info("Performance optimizer initialized", extra={
            'optimal_concurrency': optimal_concurrency,
            'hardware_type': self.hardware_manager.get_capabilities().acceleration_type
        })
    
    def _calculate_optimal_concurrency(self) -> int:
        """Calculate optimal concurrency based on system capabilities"""
        hw_capabilities = self.hardware_manager.get_capabilities()
        
        # Base concurrency on hardware capabilities
        hardware_concurrency = hw_capabilities.max_concurrent_streams
        
        # Consider memory constraints
        memory_gb = psutil.virtual_memory().total / (1024**3)
        memory_based_concurrency = max(1, int(memory_gb / 4))  # 4GB per stream
        
        # Consider CPU cores
        cpu_cores = psutil.cpu_count(logical=False) or 2
        cpu_based_concurrency = max(1, cpu_cores // 2)
        
        # Use the most restrictive limit, but cap at configuration maximum
        optimal = min(
            hardware_concurrency,
            memory_based_concurrency, 
            cpu_based_concurrency,
            self.settings.max_concurrent_processes
        )
        
        return max(1, optimal)
    
    async def optimize_for_processing(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Optimize system configuration for video processing
        
        Args:
            context: Processing context information
            
        Returns:
            dict: Optimization recommendations and settings
        """
        try:
            # Validate limits first
            await self.limits_manager.validate_video_url(context.video_url)
            
            # Get hardware-optimized FFmpeg arguments
            hw_capabilities = self.hardware_manager.get_capabilities()
            ffmpeg_args = self.hardware_manager.get_ffmpeg_args("ultra")
            
            # Determine optimal processing parameters
            memory_stats = self.resource_manager.get_memory_stats()
            
            # Adjust based on current system load
            optimization_settings = {
                'ffmpeg_args': ffmpeg_args,
                'hardware_acceleration': hw_capabilities.acceleration_type,
                'max_threads': self._get_optimal_threads(),
                'memory_limit_mb': self._calculate_memory_limit(memory_stats),
                'processing_priority': context.priority,
                'timeout_seconds': self.limits_manager.limits.max_processing_time_seconds,
                'quality_preset': self._get_quality_preset(memory_stats, context.priority)
            }
            
            # Start timeout tracking
            self.limits_manager.start_processing_timeout(context.job_id)
            
            # Store context for monitoring
            self.active_contexts[context.job_id] = context
            
            self.logger.info("Processing optimization configured", extra={
                'job_id': context.job_id,
                'hardware_acceleration': hw_capabilities.acceleration_type,
                'memory_limit_mb': optimization_settings['memory_limit_mb'],
                'max_threads': optimization_settings['max_threads']
            })
            
            return optimization_settings
            
        except Exception as e:
            log_error(e, "processing_optimization", job_id=context.job_id)
            raise
    
    def _get_optimal_threads(self) -> int:
        """Calculate optimal thread count based on current load"""
        cpu_cores = psutil.cpu_count(logical=False) or 2
        active_processes = len(self.resource_manager.active_processes)
        
        # Reduce threads if system is busy
        if active_processes > cpu_cores // 2:
            return max(1, cpu_cores // (active_processes + 1))
        else:
            return min(cpu_cores, 8)  # Cap at 8 threads
    
    def _calculate_memory_limit(self, memory_stats: Dict[str, Any]) -> int:
        """Calculate memory limit for processing based on current usage"""
        available_memory_mb = memory_stats['system_available_mb']
        active_processes = memory_stats['active_processes']
        
        # Reserve memory for system and other processes
        reserved_memory_mb = 2048  # 2GB system reserve
        
        if active_processes > 0:
            available_per_process = (available_memory_mb - reserved_memory_mb) // (active_processes + 1)
        else:
            available_per_process = (available_memory_mb - reserved_memory_mb) // 2
        
        # Cap at reasonable limits
        return max(512, min(available_per_process, 4096))  # Between 512MB and 4GB
    
    def _get_quality_preset(self, memory_stats: Dict[str, Any], priority: int) -> str:
        """Determine quality preset based on system resources and priority"""
        memory_pressure = memory_stats['system_memory_percent']
        
        if memory_pressure > 80 or priority <= 2:
            return "medium"
        elif memory_pressure > 60 or priority <= 3:
            return "high"
        else:
            return "ultra"
    
    @asynccontextmanager
    async def managed_processing(self, context: ProcessingContext):
        """
        Context manager for managed video processing with automatic resource cleanup
        
        Usage:
            async with optimizer.managed_processing(context) as settings:
                # Process video with optimized settings
                result = await process_video(settings)
        """
        settings = None
        resource_acquired = False
        
        try:
            # Optimize for processing
            settings = await self.optimize_for_processing(context)
            
            # Acquire resources
            resource_acquired = await asyncio.to_thread(
                self.resource_manager.acquire, 
                timeout=300
            )
            
            if not resource_acquired:
                raise Exception(f"Could not acquire resources for job {context.job_id}")
            
            self.logger.info("Resources acquired for processing", extra={
                'job_id': context.job_id,
                'memory_limit_mb': settings['memory_limit_mb']
            })
            
            yield settings
            
        except Exception as e:
            log_error(e, "managed_processing", job_id=context.job_id)
            raise
            
        finally:
            # Clean up resources
            try:
                if resource_acquired:
                    self.resource_manager.release()
                
                # Stop timeout tracking
                elapsed_time = self.limits_manager.stop_processing_timeout(context.job_id)
                
                # Remove from active contexts
                self.active_contexts.pop(context.job_id, None)
                
                # Force garbage collection
                self.resource_manager.force_garbage_collection()
                
                # Clean up temporary files
                self.resource_manager.cleanup_temp_files()
                
                self.logger.info("Processing cleanup completed", extra={
                    'job_id': context.job_id,
                    'elapsed_time': elapsed_time
                })
                
            except Exception as cleanup_error:
                log_error(cleanup_error, "processing_cleanup", job_id=context.job_id)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get stats from all managers
            hw_capabilities = self.hardware_manager.get_capabilities()
            memory_stats = self.resource_manager.get_memory_stats()
            db_stats = self.database_manager.get_connection_stats()
            current_limits = self.limits_manager.get_current_limits()
            
            # Calculate system health score
            health_score = self._calculate_health_score(memory_stats)
            
            status = {
                'timestamp': time.time(),
                'system_health': {
                    'score': health_score,
                    'status': 'healthy' if health_score > 70 else 'degraded' if health_score > 40 else 'critical'
                },
                'hardware': {
                    'acceleration_type': hw_capabilities.acceleration_type,
                    'max_concurrent_streams': hw_capabilities.max_concurrent_streams,
                    'supports_hevc': hw_capabilities.supports_hevc
                },
                'resources': memory_stats,
                'database': db_stats,
                'limits': current_limits,
                'active_jobs': len(self.active_contexts),
                'queue_size': self.processing_queue.qsize()
            }
            
            return status
            
        except Exception as e:
            log_error(e, "system_status_check")
            return {'error': str(e), 'timestamp': time.time()}
    
    def _calculate_health_score(self, memory_stats: Dict[str, Any]) -> int:
        """Calculate system health score (0-100)"""
        score = 100
        
        # Deduct for memory pressure
        memory_percent = memory_stats['system_memory_percent']
        if memory_percent > 90:
            score -= 30
        elif memory_percent > 80:
            score -= 20
        elif memory_percent > 70:
            score -= 10
        
        # Deduct for high process count
        active_processes = memory_stats['active_processes']
        max_processes = self.resource_manager.max_concurrent_processes
        
        if active_processes >= max_processes:
            score -= 20
        elif active_processes > max_processes * 0.8:
            score -= 10
        
        # Deduct for database issues
        if not self.database_manager._is_connected:
            score -= 25
        
        return max(0, score)
    
    async def cleanup_and_optimize(self):
        """Perform system cleanup and optimization"""
        try:
            self.logger.info("Starting system cleanup and optimization")
            
            # Clean up temporary files
            self.resource_manager.cleanup_temp_files(force=False)
            
            # Force garbage collection
            self.resource_manager.force_garbage_collection()
            
            # Clean up expired job tracking
            expired_jobs = self.limits_manager.cleanup_expired_jobs()
            
            # Clean up expired database records
            if self.database_manager._is_connected:
                expired_db_records = await self.database_manager.cleanup_expired_jobs()
            else:
                expired_db_records = 0
            
            # Refresh hardware capabilities (in case hardware changed)
            self.hardware_manager.get_capabilities(force_refresh=True)
            
            self.logger.info("System cleanup completed", extra={
                'expired_jobs_cleaned': expired_jobs,
                'expired_db_records': expired_db_records
            })
            
        except Exception as e:
            log_error(e, "system_cleanup_and_optimization")
    
    async def shutdown(self):
        """Gracefully shutdown all managers"""
        try:
            self.logger.info("Shutting down performance optimizer")
            
            # Stop resource monitoring
            self.resource_manager.stop_monitoring()
            
            # Clean up active jobs
            for job_id in list(self.active_contexts.keys()):
                self.limits_manager.stop_processing_timeout(job_id)
            
            # Disconnect database
            await self.database_manager.disconnect()
            
            # Final cleanup
            self.resource_manager.cleanup_temp_files(force=True)
            
            self.logger.info("Performance optimizer shutdown completed")
            
        except Exception as e:
            log_error(e, "performance_optimizer_shutdown")

# Global performance optimizer instance
_performance_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer