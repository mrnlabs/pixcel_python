# Video Processing API - Clean Architecture

## Architecture Overview

This document describes the refactored architecture that addresses the identified architectural issues:
- Configuration scattered across files
- No dependency injection pattern  
- Missing abstraction layers
- Poor separation of concerns

## New Architecture Components

### 1. Dependency Injection Container (`dependency_container.py`)

**Purpose**: Centralized service registration and dependency management

**Key Features**:
- Singleton pattern for global container access
- Service registration: singletons, factories, service classes, providers
- Automatic dependency injection via decorators
- Proper initialization and cleanup lifecycle management

```python
# Example usage:
@inject("settings", "logger")
async def some_function(data, settings, logger):
    logger.info("Processing data", extra={"config": settings.debug})
```

**Benefits**:
- ✅ Eliminates tight coupling between components
- ✅ Makes testing easier with dependency mocking
- ✅ Centralizes configuration and service creation
- ✅ Provides proper resource lifecycle management

### 2. Service Providers (`service_providers.py`)

**Purpose**: Complex service initialization with dependency resolution

**Service Providers**:
- `DatabaseServiceProvider`: Database connections and transactions
- `HardwareServiceProvider`: Hardware acceleration detection
- `ResourceServiceProvider`: Resource management and limits
- `StorageServiceProvider`: S3 and file storage operations
- `VideoProcessingServiceProvider`: Video effects and processing
- `PerformanceServiceProvider`: System monitoring and optimization

**Benefits**:
- ✅ Encapsulates complex initialization logic
- ✅ Manages service dependencies and startup order
- ✅ Provides clean separation between infrastructure and business logic
- ✅ Enables easy swapping of implementations for testing

### 3. Domain Services (`domain_services.py`)

**Purpose**: High-level business logic abstraction

**Key Components**:
- `VideoProcessingService`: Core business logic for video processing
- `JobRepository`: Data persistence abstraction
- `SystemHealthService`: Health monitoring and system management
- `ProcessingStatus`: Domain-specific enumerations
- `VideoProcessingJob`: Domain models

**Benefits**:
- ✅ Separates business logic from infrastructure concerns
- ✅ Provides clean interfaces for testing
- ✅ Encapsulates complex workflows with proper error handling
- ✅ Enables reusable business logic across different presentation layers

### 4. API Controllers (`api_controllers.py`)

**Purpose**: HTTP request/response handling separated from business logic

**Controllers**:
- `VideoProcessingController`: Handles video processing requests
- `SystemController`: System monitoring endpoints
- `HardwareController`: Hardware information endpoints
- `ResourceController`: Resource monitoring endpoints

**Benefits**:
- ✅ Thin presentation layer focused only on HTTP concerns
- ✅ Easy to unit test without web framework dependencies
- ✅ Clear separation between API and business logic
- ✅ Consistent error handling and response formatting

### 5. Clean Main Application (`main_new.py`)

**Purpose**: Application composition root with minimal FastAPI-specific code

**Key Features**:
- Application factory pattern
- Proper dependency injection setup
- Clean endpoint registration
- Lifecycle management (startup/shutdown)

**Benefits**:
- ✅ Minimal framework coupling
- ✅ Easy to test application composition
- ✅ Clear separation of concerns
- ✅ Proper resource management

## Architectural Improvements

### Before (Issues Fixed)

#### ❌ Configuration Scattered Across Files
```python
# In main.py
DATABASE_URL = settings.database_url

# In s3_handler.py  
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')

# In multiple files
settings = get_settings()  # Called everywhere
```

#### ❌ No Dependency Injection
```python
# Tight coupling everywhere
s3_handler = OptimizedS3Handler()  # Direct instantiation
db_manager = DatabaseManager()    # No configuration injection
```

#### ❌ Missing Abstractions
```python
# Business logic mixed with infrastructure
@app.post("/process-video/")
async def process_video(request):
    # 200+ lines of mixed concerns
    s3_handler = OptimizedS3Handler()
    video_editor = OptimizedFFmpegVideoEditor()
    # Database operations
    # File handling
    # Business logic
    # Error handling
```

#### ❌ Poor Separation of Concerns
- Single massive functions handling multiple responsibilities
- Infrastructure code mixed with business logic
- No clear boundaries between layers

### After (Clean Architecture)

#### ✅ Centralized Configuration Management
```python
# dependency_container.py
container.register_singleton("settings", get_settings())

# All services get configuration through DI
class DatabaseManager(ConfiguredService):
    def __init__(self, settings: Settings = None):
        self.settings = settings or container.get("settings")
```

#### ✅ Comprehensive Dependency Injection
```python
# service_providers.py
class DatabaseServiceProvider(ServiceProvider):
    async def initialize(self, container: DIContainer):
        settings = container.get("settings")
        db_manager = DatabaseManager(settings)
        container.register_singleton("database_manager", db_manager)

# Usage with automatic injection
@inject("database_manager", "logger")
async def some_operation(data, database_manager, logger):
    # Dependencies automatically injected
```

#### ✅ Clear Abstraction Layers
```python
# Domain layer (business logic)
class VideoProcessingService:
    async def process_video_with_effects(self, job):
        # Pure business logic, no infrastructure concerns

# Infrastructure layer (technical details)  
class S3Handler:
    async def upload_file(self, file_path, key):
        # Pure technical implementation

# Presentation layer (HTTP handling)
class VideoProcessingController:
    async def apply_video_effects(self, request):
        # HTTP-specific concerns only
```

#### ✅ Excellent Separation of Concerns
- **Presentation Layer**: HTTP request/response handling only
- **Domain Layer**: Pure business logic with no external dependencies
- **Infrastructure Layer**: Technical implementations (database, S3, FFmpeg)
- **Application Layer**: Orchestrates all layers through dependency injection

## Testing Benefits

The new architecture provides excellent testability:

```python
# Easy mocking with dependency injection
@pytest.fixture
def mock_s3_handler():
    return Mock(spec=OptimizedS3Handler)

@pytest.fixture  
def video_service(mock_s3_handler):
    container = DIContainer()
    container.register_singleton("s3_handler", mock_s3_handler)
    return VideoProcessingService()

# Clean unit testing without external dependencies
async def test_video_processing(video_service):
    job = await video_service.create_processing_job("video.mp4", [])
    assert job.status == ProcessingStatus.QUEUED
```

## Migration Path

1. **Phase 1**: Deploy new architecture alongside existing code
2. **Phase 2**: Gradually migrate endpoints to use new controllers  
3. **Phase 3**: Replace `main.py` with `main_new.py`
4. **Phase 4**: Remove legacy code and old architecture

## Performance Benefits

- **Reduced Memory Usage**: Proper resource management with cleanup
- **Better Concurrency**: Dependency injection enables better resource sharing
- **Faster Startup**: Lazy initialization of services
- **Improved Monitoring**: Centralized performance metrics

## Summary

The new clean architecture provides:

✅ **Centralized Configuration**: All configuration managed through DI container
✅ **Dependency Injection**: Loose coupling with easy testing and mocking  
✅ **Clear Abstractions**: Domain, infrastructure, and presentation layers
✅ **Separation of Concerns**: Each component has a single, well-defined responsibility
✅ **Testability**: Easy unit testing without external dependencies
✅ **Maintainability**: Clear structure that's easy to understand and modify
✅ **Scalability**: Architecture supports growth and new features

This refactoring transforms the codebase from a tightly-coupled monolith into a well-structured, maintainable, and testable application following clean architecture principles.