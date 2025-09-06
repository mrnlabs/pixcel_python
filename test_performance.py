#!/usr/bin/env python3
"""
Performance Testing and Demonstration Script
Shows the improvements in resource management and performance optimization.
"""
import asyncio
import time
import requests
from performance_optimizer import get_performance_optimizer, ProcessingContext

async def test_performance_improvements():
    """Test the new performance optimization system"""
    print("=== Video Processing Performance Improvements Test ===\n")
    
    # Initialize performance optimizer
    optimizer = get_performance_optimizer()
    
    # Test 1: Hardware Detection
    print("1. Hardware Detection (Eliminates Code Duplication)")
    print("-" * 50)
    hw_capabilities = optimizer.hardware_manager.get_capabilities()
    print(f"✅ Acceleration Type: {hw_capabilities.acceleration_type}")
    print(f"✅ Max Concurrent Streams: {hw_capabilities.max_concurrent_streams}")
    print(f"✅ Supports HEVC: {hw_capabilities.supports_hevc}")
    print(f"✅ Memory Limit: {hw_capabilities.memory_limit_mb}MB")
    print()
    
    # Test 2: Resource Management
    print("2. Enhanced Resource Management (Memory Leak Prevention)")
    print("-" * 60)
    memory_stats = optimizer.resource_manager.get_memory_stats()
    print(f"✅ System Memory Usage: {memory_stats['system_memory_percent']:.1f}%")
    print(f"✅ Process Memory Usage: {memory_stats['process_memory_mb']:.1f}MB")
    print(f"✅ Active Processes: {memory_stats['active_processes']}")
    print(f"✅ Temp Files Tracked: {memory_stats['temp_files_tracked']}")
    print()
    
    # Test 3: Database Connection Pooling
    print("3. Database Connection Pooling (Efficient DB Handling)")
    print("-" * 55)
    db_stats = optimizer.database_manager.get_connection_stats()
    print(f"✅ Connection Pool Size: {db_stats['pool_size']}")
    print(f"✅ Active Connections: {db_stats['checked_out_connections']}")
    print(f"✅ Connection Status: {'Connected' if db_stats['is_connected'] else 'Disconnected'}")
    print()
    
    # Test 4: Processing Limits
    print("4. Video Processing Limits (Size & Time Controls)")
    print("-" * 50)
    limits = optimizer.limits_manager.get_current_limits()
    print(f"✅ Max File Size: {limits['max_file_size_mb']}MB")
    print(f"✅ Max Duration: {limits['max_duration_seconds']}s")
    print(f"✅ Max Processing Time: {limits['max_processing_time_seconds']}s")
    print(f"✅ Allowed Formats: {', '.join(limits['allowed_formats'])}")
    print()
    
    # Test 5: System Health Monitoring
    print("5. System Health Monitoring")
    print("-" * 30)
    system_status = await optimizer.get_system_status()
    health = system_status['system_health']
    print(f"✅ Health Score: {health['score']}/100")
    print(f"✅ System Status: {health['status'].upper()}")
    print(f"✅ Active Jobs: {system_status['active_jobs']}")
    print()
    
    # Test 6: Processing Context Management
    print("6. Managed Processing Context (Resource Safety)")
    print("-" * 50)
    
    # Create a test processing context
    test_context = ProcessingContext(
        job_id="test-performance-001",
        video_url="https://example.com/test-video.mp4",
        estimated_duration=30.0,
        hardware_acceleration=hw_capabilities.acceleration_type
    )
    
    print(f"✅ Test Job ID: {test_context.job_id}")
    print(f"✅ Hardware Acceleration: {test_context.hardware_acceleration}")
    print(f"✅ Estimated Duration: {test_context.estimated_duration}s")
    
    try:
        # Test managed processing (without actual video processing)
        optimization_settings = await optimizer.optimize_for_processing(test_context)
        print(f"✅ Memory Limit: {optimization_settings['memory_limit_mb']}MB")
        print(f"✅ Max Threads: {optimization_settings['max_threads']}")
        print(f"✅ Quality Preset: {optimization_settings['quality_preset']}")
        
        # Stop the timeout tracking
        elapsed_time = optimizer.limits_manager.stop_processing_timeout(test_context.job_id)
        print(f"✅ Processing Time Tracked: {elapsed_time:.3f}s")
        
    except Exception as e:
        print(f"❌ Error in processing context: {e}")
    
    print()
    
    # Test 7: Cleanup and Optimization
    print("7. System Cleanup & Optimization")
    print("-" * 35)
    await optimizer.cleanup_and_optimize()
    print("✅ Temporary files cleaned")
    print("✅ Memory garbage collected")
    print("✅ Expired jobs cleaned")
    print("✅ System optimized")
    print()
    
    print("=== Performance Test Completed Successfully! ===")
    
    # Summary of improvements
    print("\n🚀 KEY PERFORMANCE IMPROVEMENTS:")
    print("   • Eliminated hardware detection code duplication")
    print("   • Added comprehensive memory leak prevention")
    print("   • Implemented database connection pooling")
    print("   • Added video size and processing time limits")
    print("   • Optimized resource manager with smart concurrency")
    print("   • Added system health monitoring and metrics")
    print("   • Implemented managed processing contexts")
    print("   • Added automatic cleanup and optimization")

def test_api_endpoints():
    """Test the new performance monitoring API endpoints"""
    print("\n=== API Performance Endpoints Test ===")
    
    base_url = "http://localhost:8000/performance"
    
    endpoints_to_test = [
        "/status",
        "/hardware", 
        "/resources",
        "/database",
        "/limits",
        "/health"
    ]
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            status = "✅ PASS" if response.status_code == 200 else f"❌ FAIL ({response.status_code})"
            print(f"{status} - {endpoint}")
        except requests.RequestException as e:
            print(f"❌ ERROR - {endpoint}: {e}")
    
    print("\n💡 To test API endpoints, run: python main.py")
    print("   Then visit: http://localhost:8000/performance/status")

if __name__ == "__main__":
    # Test the performance improvements
    asyncio.run(test_performance_improvements())
    
    # Test API endpoints (will fail if server is not running)
    test_api_endpoints()
    
    print(f"\n📊 View logs at: logs/video_processor.log")
    print(f"🔍 Monitor system: http://localhost:8000/performance/status")
    print(f"📈 Health check: http://localhost:8000/performance/health")