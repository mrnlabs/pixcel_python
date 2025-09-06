import threading
import time
import psutil
import os
import gc
import weakref
from typing import Dict, Set, Optional, List
from logger import get_logger, log_error

class ResourceManager:
    """
    Manages system resources to prevent memory issues.
    Implements a semaphore system to limit concurrent processing.
    """
    
    def __init__(self, max_concurrent_processes=2):
        """
        Initialize the resource manager with enhanced memory leak prevention.
        
        Args:
            max_concurrent_processes (int): Maximum number of concurrent video processes
        """
        self.logger = get_logger("resource_manager")
        self.semaphore = threading.Semaphore(max_concurrent_processes)
        self.active_processes: Dict[int, Dict] = {}
        self._lock = threading.RLock()  # Use RLock for nested locking
        self.monitor_thread = None
        self._stop_monitor = False
        self.max_concurrent_processes = max_concurrent_processes
        
        # Enhanced resource tracking
        self.system_info = {
            'cpu_cores': psutil.cpu_count(logical=False) or 1,
            'total_memory': psutil.virtual_memory().total,
            'total_memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        # Memory leak prevention
        self.temp_file_registry: Set[str] = set()
        self.weak_refs: List[weakref.ref] = []
        self.last_gc_time = time.time()
        self.gc_interval = 30  # Force GC every 30 seconds
        
        # Resource thresholds
        self.memory_warning_threshold = 85
        self.memory_critical_threshold = 95
        self.cpu_warning_threshold = 90
        
        self.logger.info("Resource manager initialized", extra={
            'max_concurrent_processes': max_concurrent_processes,
            'cpu_cores': self.system_info['cpu_cores'],
            'total_memory_gb': self.system_info['total_memory_gb']
        })
    
    def start_monitoring(self):
        """Start the background resource monitoring thread"""
        if self.monitor_thread is None:
            self._stop_monitor = False
            self.monitor_thread = threading.Thread(target=self._monitor_resources)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop the background resource monitoring thread"""
        if self.monitor_thread is not None:
            self._stop_monitor = True
            self.monitor_thread.join(timeout=3)
            self.monitor_thread = None
            self.logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Monitor system resources and terminate processes if memory is low"""
        while not self._stop_monitor:
            try:
                # Get system memory information
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Log resource usage periodically
                if memory.percent > 70 or cpu_percent > 70:
                    print(f"Resource usage: Memory: {memory.percent}%, CPU: {cpu_percent}%")
                
                # MEMORY MANAGEMENT: Take action at different thresholds
                if memory.percent > 85:
                    print(f"WARNING: High memory usage detected: {memory.percent}%")
                    
                    # First action: Clear disk cache if on Linux (safe implementation)
                    if os.path.exists('/proc/sys/vm/drop_caches') and memory.percent > 90:
                        try:
                            # Use secure subprocess calls instead of os.system()
                            import subprocess
                            # Sync filesystem first
                            subprocess.run(['sync'], check=True, timeout=10)
                            # Try to drop caches (requires appropriate permissions)
                            with open('/proc/sys/vm/drop_caches', 'w') as f:
                                f.write('1')
                            print("Dropped system caches to free memory")
                        except (PermissionError, FileNotFoundError, subprocess.TimeoutExpired):
                            print("Could not drop system caches - insufficient permissions or timeout")
                        except Exception as e:
                            print(f"Error dropping system caches: {str(e)}")
                    
                    # Second action: Terminate the newest process if memory is critical
                    if memory.percent > 95 and self.active_processes:
                        with self._lock:
                            if self.active_processes:
                                # Find the newest process (highest start_time)
                                newest_pid = max(self.active_processes.items(), 
                                               key=lambda x: x[1]['start_time'])[0]
                                print(f"CRITICAL: Terminating process {newest_pid} due to memory pressure")
                                
                                try:
                                    process = psutil.Process(newest_pid)
                                    # Get process info before terminating
                                    process_info = {
                                        'cmd': process.cmdline(),
                                        'memory': process.memory_info().rss / (1024*1024),  # MB
                                        'cpu': process.cpu_percent()
                                    }
                                    print(f"Process using {process_info['memory']:.1f}MB of memory")
                                    
                                    # Terminate the process
                                    process.terminate()
                                    
                                    # Wait briefly and kill if not terminated
                                    try:
                                        process.wait(timeout=3)
                                    except psutil.TimeoutExpired:
                                        print(f"Process did not terminate gracefully, killing forcibly")
                                        process.kill()
                                    
                                    # Remove from our tracking
                                    if newest_pid in self.active_processes:
                                        del self.active_processes[newest_pid]
                                        
                                    # Release the semaphore to allow another process to start
                                    try:
                                        self.semaphore.release()
                                        print(f"Released semaphore after terminating process {newest_pid}")
                                    except Exception:
                                        pass
                                except Exception as e:
                                    print(f"Error terminating process: {str(e)}")
                
                # CPU MANAGEMENT: Adjust process nice values if CPU is overloaded  
                if cpu_percent > 90 and self.active_processes:
                    print(f"WARNING: High CPU usage detected: {cpu_percent}%")
                    # Reduce priority of all active processes
                    with self._lock:
                        for pid in list(self.active_processes.keys()):
                            try:
                                process = psutil.Process(pid)
                                # Lower the priority (higher nice value)
                                if hasattr(process, 'nice'):
                                    current_nice = process.nice()
                                    if current_nice < 19:  # Maximum nice value on Unix
                                        process.nice(min(current_nice + 5, 19))
                                        print(f"Reduced priority of process {pid}")
                            except Exception:
                                # Process may no longer exist
                                pass
            
            except Exception as e:
                print(f"Error in resource monitor: {str(e)}")
            
            # Sleep for a short period before checking again
            # Use shorter sleep when resources are constrained
            if memory.percent > 90 or cpu_percent > 90:
                time.sleep(1)  # Check more frequently when resources are constrained
            else:
                time.sleep(3)  # Standard check interval
    
    def acquire(self, timeout=300):
        """
        Acquire permission to start a new video process.
        
        Args:
            timeout (int): Maximum time in seconds to wait
            
        Returns:
            bool: True if acquired, False if timed out
        """
        # Check resource availability first
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            print(f"Cannot acquire resource: memory usage at {memory.percent}%")
            return False
            
        # Try to acquire the semaphore
        result = self.semaphore.acquire(timeout=timeout)
        if result:
            # Log current concurrency
            with self._lock:
                concurrency = len(self.active_processes)
            print(f"Resource acquired. Active processes: {concurrency}/{self.semaphore._value}")
        return result
    
    def release(self):
        """Release permission after a video process completes"""
        try:
            self.semaphore.release()
            print("Resource released")
        except ValueError:
            # This happens if release is called more times than acquire
            print("Warning: Attempted to release an unacquired resource")
    
    def register_process(self, pid, metadata=None):
        """Register a new FFmpeg process for tracking"""
        import time
        with self._lock:
            # Get process info
            try:
                process = psutil.Process(pid)
                mem_info = process.memory_info()
                cpu_percent = process.cpu_percent(interval=0.1)
                
                process_info = {
                    'start_time': time.time(),
                    'memory_mb': mem_info.rss / (1024 * 1024),
                    'cpu_percent': cpu_percent,
                    'metadata': metadata or {}
                }
                
                self.active_processes[pid] = process_info
                print(f"Registered process {pid} (type: {metadata.get('type', 'unknown') if metadata else 'unknown'})")
            except Exception as e:
                # Process might no longer exist
                print(f"Error registering process {pid}: {str(e)}")
    
    def unregister_process(self, pid):
        """Unregister an FFmpeg process after completion"""
        with self._lock:
            if pid in self.active_processes:
                process_info = self.active_processes[pid]
                duration = time.time() - process_info['start_time']
                print(f"Unregistered process {pid} (ran for {duration:.1f}s)")
                del self.active_processes[pid]
    
    def get_active_processes_info(self):
        """Get information about active processes"""
        with self._lock:
            info = {
                'count': len(self.active_processes),
                'processes': {}
            }
            
            # Add detailed process info for debugging
            for pid, data in self.active_processes.items():
                try:
                    process = psutil.Process(pid)
                    # Add current memory and CPU usage
                    current_data = {
                        'current_memory_mb': process.memory_info().rss / (1024 * 1024),
                        'current_cpu_percent': process.cpu_percent(interval=0),
                        'running_time': time.time() - data['start_time'],
                        'metadata': data.get('metadata', {})
                    }
                    
                    info['processes'][pid] = current_data
                except Exception:
                    # Process may have terminated
                    info['processes'][pid] = {
                        'status': 'terminated or inaccessible',
                        'metadata': data.get('metadata', {})
                    }
            
            return info
    
    def register_temp_file(self, file_path: str):
        """Register a temporary file for automatic cleanup"""
        with self._lock:
            self.temp_file_registry.add(file_path)
            self.logger.debug("Temporary file registered", extra={'file_path': file_path})
    
    def unregister_temp_file(self, file_path: str):
        """Unregister a temporary file"""
        with self._lock:
            self.temp_file_registry.discard(file_path)
    
    def cleanup_temp_files(self, force: bool = False):
        """Clean up all registered temporary files"""
        with self._lock:
            cleanup_count = 0
            files_to_remove = list(self.temp_file_registry)
            
            for file_path in files_to_remove:
                try:
                    if os.path.exists(file_path):
                        # Check if file is still being used (if not forced)
                        if not force:
                            try:
                                # Try to get file stats to see if it's locked
                                stat = os.stat(file_path)
                                # If file was created more than 1 hour ago, clean it up
                                if time.time() - stat.st_mtime > 3600:  # 1 hour
                                    os.unlink(file_path)
                                    cleanup_count += 1
                                    self.temp_file_registry.discard(file_path)
                            except (OSError, PermissionError):
                                # File might be in use, skip
                                continue
                        else:
                            os.unlink(file_path)
                            cleanup_count += 1
                            self.temp_file_registry.discard(file_path)
                    else:
                        # File doesn't exist, remove from registry
                        self.temp_file_registry.discard(file_path)
                except Exception as e:
                    self.logger.warning(f"Error cleaning temp file {file_path}", exc_info=True)
            
            if cleanup_count > 0:
                self.logger.info(f"Cleaned up {cleanup_count} temporary files")
    
    def force_garbage_collection(self):
        """Force garbage collection and cleanup weak references"""
        current_time = time.time()
        
        if current_time - self.last_gc_time > self.gc_interval:
            with self._lock:
                # Clean up dead weak references
                self.weak_refs = [ref for ref in self.weak_refs if ref() is not None]
                
                # Force garbage collection
                collected = gc.collect()
                
                self.last_gc_time = current_time
                
                if collected > 0:
                    self.logger.debug(f"Garbage collection freed {collected} objects")
    
    def add_weak_reference(self, obj):
        """Add a weak reference for tracking object lifecycle"""
        with self._lock:
            self.weak_refs.append(weakref.ref(obj))
    
    def get_memory_stats(self) -> Dict:
        """Get detailed memory statistics"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'system_memory_percent': memory.percent,
            'system_available_mb': memory.available / (1024 * 1024),
            'process_memory_mb': process.memory_info().rss / (1024 * 1024),
            'active_processes': len(self.active_processes),
            'temp_files_tracked': len(self.temp_file_registry)
        }