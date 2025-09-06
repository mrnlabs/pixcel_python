"""
Centralized Hardware Management System
Eliminates code duplication and provides comprehensive hardware detection and management.
"""
import os
import platform
import subprocess
import threading
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from logger import get_logger, log_error

@dataclass
class HardwareCapabilities:
    """Hardware capabilities information"""
    acceleration_type: Optional[str] = None
    device: Optional[str] = None
    max_concurrent_streams: int = 1
    memory_limit_mb: Optional[int] = None
    supports_hevc: bool = False
    supports_av1: bool = False
    driver_version: Optional[str] = None

class HardwareManager:
    """
    Centralized hardware detection and management system.
    Singleton pattern to ensure consistent hardware detection across the application.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(HardwareManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self.logger = get_logger("hardware_manager")
        self._capabilities = None
        self._detection_time = None
        self._cache_duration = 300  # 5 minutes cache
        self._lock = threading.RLock()
        self._initialized = True
        
        # Perform initial detection
        self.get_capabilities()
    
    def get_capabilities(self, force_refresh: bool = False) -> HardwareCapabilities:
        """
        Get hardware capabilities with caching
        
        Args:
            force_refresh: Force a fresh detection, ignoring cache
            
        Returns:
            HardwareCapabilities: Current hardware capabilities
        """
        with self._lock:
            current_time = time.time()
            
            # Use cached result if available and not expired
            if (not force_refresh and 
                self._capabilities is not None and 
                self._detection_time is not None and
                current_time - self._detection_time < self._cache_duration):
                return self._capabilities
            
            # Perform fresh detection
            self.logger.info("Detecting hardware capabilities")
            self._capabilities = self._detect_hardware()
            self._detection_time = current_time
            
            self.logger.info("Hardware detection completed", extra={
                'acceleration_type': self._capabilities.acceleration_type,
                'device': self._capabilities.device,
                'max_concurrent_streams': self._capabilities.max_concurrent_streams,
                'supports_hevc': self._capabilities.supports_hevc
            })
            
            return self._capabilities
    
    def _detect_hardware(self) -> HardwareCapabilities:
        """Perform comprehensive hardware detection"""
        system = platform.system().lower()
        capabilities = HardwareCapabilities()
        
        try:
            # Try detection methods in order of preference
            detection_methods = [
                self._detect_nvidia_nvenc,
                self._detect_intel_qsv,
                self._detect_amd_vaapi,
                self._detect_apple_videotoolbox
            ]
            
            for method in detection_methods:
                if method(capabilities, system):
                    break
            else:
                self.logger.info("No hardware acceleration detected, using CPU encoding")
                capabilities.acceleration_type = "cpu"
                capabilities.max_concurrent_streams = 2
                
        except Exception as e:
            log_error(e, "hardware_detection")
            capabilities.acceleration_type = "cpu"
            capabilities.max_concurrent_streams = 1
        
        return capabilities
    
    def _detect_nvidia_nvenc(self, capabilities: HardwareCapabilities, system: str) -> bool:
        """Detect NVIDIA NVENC capabilities"""
        if system not in ['linux', 'windows']:
            return False
            
        try:
            # Check nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_name, memory_mb, driver_version = parts[0], parts[1], parts[2]
                        
                        # Check FFmpeg NVENC support
                        ffmpeg_check = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                                    capture_output=True, text=True, timeout=10)
                        
                        if 'h264_nvenc' in ffmpeg_check.stdout:
                            capabilities.acceleration_type = "nvenc"
                            capabilities.device = "cuda"
                            capabilities.memory_limit_mb = int(memory_mb) if memory_mb.isdigit() else None
                            capabilities.driver_version = driver_version
                            capabilities.supports_hevc = 'hevc_nvenc' in ffmpeg_check.stdout
                            capabilities.supports_av1 = 'av1_nvenc' in ffmpeg_check.stdout
                            
                            # NVIDIA GPUs can typically handle multiple streams
                            capabilities.max_concurrent_streams = 4
                            
                            self.logger.info("NVIDIA NVENC detected", extra={
                                'gpu_name': gpu_name,
                                'memory_mb': memory_mb,
                                'driver_version': driver_version,
                                'supports_hevc': capabilities.supports_hevc
                            })
                            return True
                            
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            self.logger.debug(f"NVIDIA detection failed: {type(e).__name__}")
            
        return False
    
    def _detect_intel_qsv(self, capabilities: HardwareCapabilities, system: str) -> bool:
        """Detect Intel Quick Sync Video capabilities"""
        if system not in ['linux', 'windows']:
            return False
            
        try:
            # Check for Intel GPU and QSV support
            ffmpeg_check = subprocess.run(['ffmpeg', '-hide_banner', '-hwaccels'], 
                                        capture_output=True, text=True, timeout=10)
            
            if 'qsv' in ffmpeg_check.stdout.lower():
                # Verify encoders are available
                encoder_check = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                             capture_output=True, text=True, timeout=10)
                
                if 'h264_qsv' in encoder_check.stdout:
                    capabilities.acceleration_type = "qsv"
                    capabilities.device = "/dev/dri/renderD128" if system == 'linux' else None
                    capabilities.supports_hevc = 'hevc_qsv' in encoder_check.stdout
                    capabilities.supports_av1 = 'av1_qsv' in encoder_check.stdout
                    capabilities.max_concurrent_streams = 2
                    
                    self.logger.info("Intel Quick Sync Video detected", extra={
                        'supports_hevc': capabilities.supports_hevc,
                        'device': capabilities.device
                    })
                    return True
                    
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            self.logger.debug(f"Intel QSV detection failed: {type(e).__name__}")
            
        return False
    
    def _detect_amd_vaapi(self, capabilities: HardwareCapabilities, system: str) -> bool:
        """Detect AMD VAAPI capabilities"""
        if system != 'linux' or not os.path.exists('/dev/dri/renderD128'):
            return False
            
        try:
            # Check VAAPI support
            ffmpeg_check = subprocess.run(['ffmpeg', '-hide_banner', '-hwaccels'], 
                                        capture_output=True, text=True, timeout=10)
            
            if 'vaapi' in ffmpeg_check.stdout.lower():
                encoder_check = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                             capture_output=True, text=True, timeout=10)
                
                if 'h264_vaapi' in encoder_check.stdout:
                    capabilities.acceleration_type = "vaapi"
                    capabilities.device = "/dev/dri/renderD128"
                    capabilities.supports_hevc = 'hevc_vaapi' in encoder_check.stdout
                    capabilities.max_concurrent_streams = 2
                    
                    self.logger.info("AMD VAAPI detected", extra={
                        'supports_hevc': capabilities.supports_hevc
                    })
                    return True
                    
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            self.logger.debug(f"AMD VAAPI detection failed: {type(e).__name__}")
            
        return False
    
    def _detect_apple_videotoolbox(self, capabilities: HardwareCapabilities, system: str) -> bool:
        """Detect Apple VideoToolbox capabilities"""
        if system != 'darwin':
            return False
            
        try:
            # Check VideoToolbox support
            ffmpeg_check = subprocess.run(['ffmpeg', '-hide_banner', '-hwaccels'], 
                                        capture_output=True, text=True, timeout=10)
            
            if 'videotoolbox' in ffmpeg_check.stdout.lower():
                encoder_check = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                             capture_output=True, text=True, timeout=10)
                
                if 'h264_videotoolbox' in encoder_check.stdout:
                    capabilities.acceleration_type = "videotoolbox"
                    capabilities.device = None
                    capabilities.supports_hevc = 'hevc_videotoolbox' in encoder_check.stdout
                    capabilities.max_concurrent_streams = 3
                    
                    self.logger.info("Apple VideoToolbox detected", extra={
                        'supports_hevc': capabilities.supports_hevc
                    })
                    return True
                    
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            self.logger.debug(f"VideoToolbox detection failed: {type(e).__name__}")
            
        return False
    
    def get_ffmpeg_args(self, target_quality: str = "high") -> List[str]:
        """
        Get optimal FFmpeg arguments based on detected hardware
        
        Args:
            target_quality: Desired quality level (low, medium, high, ultra)
            
        Returns:
            List[str]: FFmpeg command line arguments
        """
        capabilities = self.get_capabilities()
        args = []
        
        if capabilities.acceleration_type == "nvenc":
            args.extend(['-hwaccel', 'cuda'])
            if target_quality == "ultra":
                args.extend(['-c:v', 'h264_nvenc', '-preset', 'p7', '-rc', 'vbr_hq'])
            else:
                args.extend(['-c:v', 'h264_nvenc', '-preset', 'fast'])
                
        elif capabilities.acceleration_type == "qsv":
            args.extend(['-hwaccel', 'qsv'])
            args.extend(['-c:v', 'h264_qsv', '-preset', 'veryslow' if target_quality == "ultra" else 'medium'])
            
        elif capabilities.acceleration_type == "vaapi":
            args.extend(['-hwaccel', 'vaapi', '-vaapi_device', capabilities.device])
            args.extend(['-c:v', 'h264_vaapi'])
            
        elif capabilities.acceleration_type == "videotoolbox":
            args.extend(['-hwaccel', 'videotoolbox'])
            args.extend(['-c:v', 'h264_videotoolbox'])
            
        else:
            # CPU encoding
            if target_quality == "ultra":
                args.extend(['-c:v', 'libx264', '-preset', 'slow', '-crf', '18'])
            elif target_quality == "high":
                args.extend(['-c:v', 'libx264', '-preset', 'medium', '-crf', '23'])
            else:
                args.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '28'])
        
        return args
    
    def get_optimal_concurrency(self) -> int:
        """Get optimal number of concurrent video processing streams"""
        capabilities = self.get_capabilities()
        return min(capabilities.max_concurrent_streams, 6)  # Cap at 6 for safety
    
    def supports_codec(self, codec: str) -> bool:
        """Check if a specific codec is supported by current hardware"""
        capabilities = self.get_capabilities()
        
        codec_map = {
            'hevc': capabilities.supports_hevc,
            'h265': capabilities.supports_hevc,
            'av1': capabilities.supports_av1
        }
        
        return codec_map.get(codec.lower(), True)  # Default to True for h264

# Global hardware manager instance
hardware_manager = HardwareManager()

def get_hardware_manager() -> HardwareManager:
    """Get the global hardware manager instance"""
    return hardware_manager

# Legacy compatibility functions
def detect_hardware_acceleration():
    """Legacy compatibility function"""
    capabilities = hardware_manager.get_capabilities()
    if capabilities.acceleration_type == "cpu":
        return None
    return {
        'type': capabilities.acceleration_type,
        'device': capabilities.device
    }