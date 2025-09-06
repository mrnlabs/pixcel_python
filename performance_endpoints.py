"""
Performance Monitoring Endpoints
FastAPI endpoints for monitoring and managing system performance.
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import psutil
from performance_optimizer import get_performance_optimizer
from logger import get_logger

# Create router for performance endpoints
performance_router = APIRouter(prefix="/performance", tags=["performance"])
logger = get_logger("performance_endpoints")

@performance_router.get("/status")
async def get_system_status():
    """Get comprehensive system performance status"""
    try:
        optimizer = get_performance_optimizer()
        status = await optimizer.get_system_status()
        
        return JSONResponse(content={
            "status": "success",
            "data": status
        })
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@performance_router.get("/hardware")
async def get_hardware_info():
    """Get hardware acceleration capabilities"""
    try:
        optimizer = get_performance_optimizer()
        hw_manager = optimizer.hardware_manager
        capabilities = hw_manager.get_capabilities()
        
        return JSONResponse(content={
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
        })
        
    except Exception as e:
        logger.error(f"Error getting hardware info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get hardware info: {str(e)}")

@performance_router.get("/resources")
async def get_resource_stats():
    """Get detailed resource usage statistics"""
    try:
        optimizer = get_performance_optimizer()
        resource_stats = optimizer.resource_manager.get_memory_stats()
        
        # Add additional system stats
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
            "active_processes_details": optimizer.resource_manager.get_active_processes_info()
        }
        
        return JSONResponse(content={
            "status": "success",
            "data": extended_stats
        })
        
    except Exception as e:
        logger.error(f"Error getting resource stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get resource stats: {str(e)}")

@performance_router.get("/database")
async def get_database_stats():
    """Get database connection pool statistics"""
    try:
        optimizer = get_performance_optimizer()
        db_stats = optimizer.database_manager.get_connection_stats()
        
        return JSONResponse(content={
            "status": "success",
            "data": db_stats
        })
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")

@performance_router.get("/limits")
async def get_current_limits():
    """Get current processing limits and constraints"""
    try:
        optimizer = get_performance_optimizer()
        limits = optimizer.limits_manager.get_current_limits()
        
        return JSONResponse(content={
            "status": "success",
            "data": limits
        })
        
    except Exception as e:
        logger.error(f"Error getting limits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get limits: {str(e)}")

@performance_router.post("/cleanup")
async def trigger_system_cleanup():
    """Trigger system cleanup and optimization"""
    try:
        optimizer = get_performance_optimizer()
        await optimizer.cleanup_and_optimize()
        
        return JSONResponse(content={
            "status": "success",
            "message": "System cleanup and optimization completed"
        })
        
    except Exception as e:
        logger.error(f"Error during system cleanup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"System cleanup failed: {str(e)}")

@performance_router.get("/jobs/{job_id}/stats")
async def get_job_processing_stats(job_id: str):
    """Get processing statistics for a specific job"""
    try:
        optimizer = get_performance_optimizer()
        stats = optimizer.limits_manager.get_processing_stats(job_id)
        
        if stats is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not currently processing")
        
        return JSONResponse(content={
            "status": "success",
            "data": stats
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get job stats: {str(e)}")

@performance_router.get("/health")
async def get_health_check():
    """Simple health check endpoint"""
    try:
        optimizer = get_performance_optimizer()
        status = await optimizer.get_system_status()
        
        health_info = {
            "status": status['system_health']['status'],
            "score": status['system_health']['score'],
            "timestamp": status['timestamp'],
            "uptime": psutil.boot_time(),
            "services": {
                "database": status['database']['is_connected'],
                "hardware_detection": status['hardware']['acceleration_type'] != 'cpu',
                "resource_monitoring": len(status['resources']) > 0
            }
        }
        
        http_status = 200 if status['system_health']['status'] == 'healthy' else 503
        
        return JSONResponse(content={
            "status": "success" if http_status == 200 else "degraded",
            "data": health_info
        }, status_code=http_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(content={
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }, status_code=503)

@performance_router.get("/metrics")
async def get_performance_metrics():
    """Get performance metrics in Prometheus-compatible format"""
    try:
        optimizer = get_performance_optimizer()
        status = await optimizer.get_system_status()
        
        # Format as Prometheus metrics
        metrics = []
        
        # System metrics
        metrics.append(f'video_processor_memory_percent {status["resources"]["system_memory_percent"]}')
        metrics.append(f'video_processor_active_processes {status["active_jobs"]}')
        metrics.append(f'video_processor_health_score {status["system_health"]["score"]}')
        
        # Hardware metrics
        hw_type_value = 1 if status['hardware']['acceleration_type'] != 'cpu' else 0
        metrics.append(f'video_processor_hardware_acceleration {{type="{status["hardware"]["acceleration_type"]}"}} {hw_type_value}')
        
        # Database metrics
        metrics.append(f'video_processor_db_connections {{state="active"}} {status["database"]["checked_out_connections"]}')
        metrics.append(f'video_processor_db_connections {{state="pool"}} {status["database"]["pool_size"]}')
        
        metrics_text = '\n'.join(metrics)
        
        return JSONResponse(content={
            "status": "success",
            "metrics": metrics_text,
            "format": "prometheus"
        })
        
    except Exception as e:
        logger.error(f"Error generating metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate metrics: {str(e)}")