"""
Dependency Injection Container
Provides centralized dependency management and configuration injection.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar, Type, Optional, Callable
from functools import wraps
import inspect

from config import get_settings, Settings
from logger import get_logger

T = TypeVar('T')


class ServiceProvider(ABC):
    """Abstract base class for all service providers"""
    
    @abstractmethod
    async def initialize(self, container: 'DIContainer') -> None:
        """Initialize the service with dependencies"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources when shutting down"""
        pass


class SingletonMeta(type):
    """Metaclass for singleton pattern"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DIContainer(metaclass=SingletonMeta):
    """Dependency Injection Container - Singleton"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._providers: Dict[str, ServiceProvider] = {}
        self._initialized = False
        self.logger = get_logger("dependency_container")
        
    async def initialize(self) -> None:
        """Initialize all registered services"""
        if self._initialized:
            return
            
        self.logger.info("Initializing dependency container")
        
        # Register core services
        await self._register_core_services()
        
        # Initialize all service providers
        for name, provider in self._providers.items():
            self.logger.info(f"Initializing service provider: {name}")
            await provider.initialize(self)
            
        self._initialized = True
        self.logger.info("Dependency container initialization complete")
    
    async def _register_core_services(self) -> None:
        """Register core application services"""
        # Configuration service (always singleton)
        self.register_singleton("settings", get_settings())
        
        # Logger factory
        self.register_factory("logger", lambda name="default": get_logger(name))
    
    def register_singleton(self, name: str, instance: Any) -> None:
        """Register a singleton service instance"""
        self._singletons[name] = instance
        self.logger.debug(f"Registered singleton service: {name}")
    
    def register_service(self, name: str, service_class: Type[T]) -> None:
        """Register a service class (new instance per request)"""
        self._services[name] = service_class
        self.logger.debug(f"Registered service class: {name}")
    
    def register_factory(self, name: str, factory: Callable) -> None:
        """Register a factory function for creating services"""
        self._factories[name] = factory
        self.logger.debug(f"Registered factory: {name}")
    
    def register_provider(self, name: str, provider: ServiceProvider) -> None:
        """Register a service provider for complex initialization"""
        self._providers[name] = provider
        self.logger.debug(f"Registered service provider: {name}")
    
    def get(self, name: str) -> Any:
        """Get a service instance"""
        # Check singletons first
        if name in self._singletons:
            return self._singletons[name]
        
        # Check factories
        if name in self._factories:
            return self._factories[name]()
        
        # Check service classes
        if name in self._services:
            return self._services[name]()
        
        # Check providers
        if name in self._providers:
            return self._providers[name]
        
        raise KeyError(f"Service '{name}' not registered")
    
    def get_singleton(self, name: str) -> Any:
        """Get or create a singleton service"""
        if name not in self._singletons:
            if name in self._services:
                # Create singleton from service class
                self._singletons[name] = self._services[name]()
                self.logger.debug(f"Created singleton instance for: {name}")
            else:
                raise KeyError(f"Singleton service '{name}' not found")
        
        return self._singletons[name]
    
    def has(self, name: str) -> bool:
        """Check if a service is registered"""
        return (name in self._singletons or 
                name in self._factories or 
                name in self._services or
                name in self._providers)
    
    async def cleanup(self) -> None:
        """Cleanup all services and providers"""
        self.logger.info("Cleaning up dependency container")
        
        # Cleanup all providers
        for name, provider in self._providers.items():
            try:
                self.logger.debug(f"Cleaning up provider: {name}")
                await provider.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up provider {name}: {e}", exc_info=True)
        
        # Clear all registrations
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._providers.clear()
        
        self._initialized = False
        self.logger.info("Dependency container cleanup complete")


# Global container instance
container = DIContainer()


def inject(*dependencies: str):
    """Decorator for dependency injection"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Inject dependencies as keyword arguments
            for dep_name in dependencies:
                if dep_name not in kwargs:
                    kwargs[dep_name] = container.get(dep_name)
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Inject dependencies as keyword arguments
            for dep_name in dependencies:
                if dep_name not in kwargs:
                    kwargs[dep_name] = container.get(dep_name)
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class ConfiguredService:
    """Base class for services that need configuration"""
    
    def __init__(self, settings: Settings = None):
        self.settings = settings or container.get("settings")
        self.logger = container.get("logger")(self.__class__.__name__.lower())


def get_container() -> DIContainer:
    """Get the global dependency container"""
    return container


async def initialize_container() -> DIContainer:
    """Initialize and return the global container"""
    await container.initialize()
    return container