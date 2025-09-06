"""
Service Providers for Dependency Injection
Implements ServiceProvider pattern for complex service initialization.
"""
import asyncio
from typing import Optional

from dependency_container import ServiceProvider, DIContainer, ConfiguredService
from database_manager import DatabaseManager
from hardware_manager import HardwareManager
from resource_manager import ResourceManager
from limits_manager import LimitsManager
from s3_handler import OptimizedS3Handler
from performance_optimizer import PerformanceOptimizer
from video_effects import VideoEffectsProcessor
from logger import get_logger


class DatabaseServiceProvider(ServiceProvider):
    """Service provider for database-related services"""
    
    def __init__(self):
        self.database_manager: Optional[DatabaseManager] = None
        
    async def initialize(self, container: DIContainer) -> None:
        """Initialize database services"""
        settings = container.get("settings")
        
        # Create and register database manager
        self.database_manager = DatabaseManager(settings)
        await self.database_manager.initialize()
        
        container.register_singleton("database_manager", self.database_manager)
        
        # Register database connection factory
        container.register_factory("database_connection", 
                                   lambda: self.database_manager.get_connection())
    
    async def cleanup(self) -> None:
        """Cleanup database connections"""
        if self.database_manager:
            await self.database_manager.cleanup()


class HardwareServiceProvider(ServiceProvider):
    """Service provider for hardware-related services"""
    
    def __init__(self):
        self.hardware_manager: Optional[HardwareManager] = None
        
    async def initialize(self, container: DIContainer) -> None:
        """Initialize hardware services"""
        settings = container.get("settings")
        
        # Create and register hardware manager
        self.hardware_manager = HardwareManager(settings)
        container.register_singleton("hardware_manager", self.hardware_manager)
        
        # Register hardware capabilities factory
        container.register_factory("hardware_capabilities", 
                                   lambda: self.hardware_manager.get_capabilities())
    
    async def cleanup(self) -> None:
        """Cleanup hardware resources"""
        # Hardware manager doesn't need explicit cleanup
        pass


class ResourceServiceProvider(ServiceProvider):
    """Service provider for resource management services"""
    
    def __init__(self):
        self.resource_manager: Optional[ResourceManager] = None
        
    async def initialize(self, container: DIContainer) -> None:
        """Initialize resource management services"""
        settings = container.get("settings")
        
        # Create and register resource manager
        self.resource_manager = ResourceManager(settings.max_concurrent_processes)
        container.register_singleton("resource_manager", self.resource_manager)
        
        # Register limits manager
        limits_manager = LimitsManager(settings)
        container.register_singleton("limits_manager", limits_manager)
    
    async def cleanup(self) -> None:
        """Cleanup resource management"""
        if self.resource_manager:
            await self.resource_manager.cleanup()


class StorageServiceProvider(ServiceProvider):
    """Service provider for storage services (S3, local files)"""
    
    def __init__(self):
        self.s3_handler: Optional[OptimizedS3Handler] = None
        
    async def initialize(self, container: DIContainer) -> None:
        """Initialize storage services"""
        settings = container.get("settings")
        
        # Create and register S3 handler
        self.s3_handler = OptimizedS3Handler()
        container.register_singleton("s3_handler", self.s3_handler)
        
        # Register S3 operations as factories
        container.register_factory("s3_uploader", 
                                   lambda: self.s3_handler.upload_file)
        container.register_factory("s3_downloader", 
                                   lambda: self.s3_handler.download_file)
    
    async def cleanup(self) -> None:
        """Cleanup storage resources"""
        # S3 handler doesn't need explicit cleanup
        pass


class VideoProcessingServiceProvider(ServiceProvider):
    """Service provider for video processing services"""
    
    def __init__(self):
        self.effects_processor: Optional[VideoEffectsProcessor] = None
        
    async def initialize(self, container: DIContainer) -> None:
        """Initialize video processing services"""
        settings = container.get("settings")
        
        # Get dependencies
        hardware_manager = container.get("hardware_manager")
        
        # Create and register video effects processor
        self.effects_processor = VideoEffectsProcessor(hardware_manager)
        container.register_singleton("effects_processor", self.effects_processor)
        
        # Register video editor factory (requires video path)
        from video_editor import OptimizedFFmpegVideoEditor
        container.register_factory("video_editor", 
                                   lambda video_path: OptimizedFFmpegVideoEditor(video_path))
    
    async def cleanup(self) -> None:
        """Cleanup video processing resources"""
        if self.effects_processor:
            await self.effects_processor.cleanup()


class PerformanceServiceProvider(ServiceProvider):
    """Service provider for performance monitoring and optimization"""
    
    def __init__(self):
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        
    async def initialize(self, container: DIContainer) -> None:
        """Initialize performance services"""
        settings = container.get("settings")
        
        # Get dependencies
        hardware_manager = container.get("hardware_manager")
        database_manager = container.get("database_manager")
        resource_manager = container.get("resource_manager")
        limits_manager = container.get("limits_manager")
        
        # Create and register performance optimizer
        self.performance_optimizer = PerformanceOptimizer(
            settings=settings,
            hardware_manager=hardware_manager,
            database_manager=database_manager,
            resource_manager=resource_manager,
            limits_manager=limits_manager
        )
        
        await self.performance_optimizer.initialize()
        container.register_singleton("performance_optimizer", self.performance_optimizer)
    
    async def cleanup(self) -> None:
        """Cleanup performance monitoring"""
        if self.performance_optimizer:
            await self.performance_optimizer.cleanup_and_optimize()


class ApplicationServiceRegistry:
    """Registry for all application service providers"""
    
    @staticmethod
    def register_all_providers(container: DIContainer) -> None:
        """Register all service providers with the container"""
        container.register_provider("database", DatabaseServiceProvider())
        container.register_provider("hardware", HardwareServiceProvider())
        container.register_provider("resources", ResourceServiceProvider())
        container.register_provider("storage", StorageServiceProvider())
        container.register_provider("video_processing", VideoProcessingServiceProvider())
        container.register_provider("performance", PerformanceServiceProvider())


# Convenience functions for common services
class ServiceLocator(ConfiguredService):
    """Service locator for easy access to common services"""
    
    @property
    def database_manager(self) -> DatabaseManager:
        """Get database manager"""
        return self._container.get("database_manager")
    
    @property
    def hardware_manager(self) -> HardwareManager:
        """Get hardware manager"""
        return self._container.get("hardware_manager")
    
    @property
    def resource_manager(self) -> ResourceManager:
        """Get resource manager"""
        return self._container.get("resource_manager")
    
    @property
    def limits_manager(self) -> LimitsManager:
        """Get limits manager"""
        return self._container.get("limits_manager")
    
    @property
    def s3_handler(self) -> OptimizedS3Handler:
        """Get S3 handler"""
        return self._container.get("s3_handler")
    
    @property
    def effects_processor(self) -> VideoEffectsProcessor:
        """Get effects processor"""
        return self._container.get("effects_processor")
    
    @property
    def performance_optimizer(self) -> PerformanceOptimizer:
        """Get performance optimizer"""
        return self._container.get("performance_optimizer")
    
    def __init__(self):
        super().__init__()
        from dependency_container import get_container
        self._container = get_container()
    
    def create_video_editor(self, video_path: str):
        """Create a video editor for the given video path"""
        return self._container.get("video_editor")(video_path)
    
    def get_logger(self, name: str):
        """Get a logger with the given name"""
        return self._container.get("logger")(name)