# Hardware-Accelerated FFmpeg Helper Functions
def get_ffmpeg_memory_flags(video_path, target_memory_mb=200):
    """
    Calculate appropriate FFmpeg flags to limit memory usage with hardware acceleration.
    
    Args:
        video_path (str): Path to the video file
        target_memory_mb (int): Target memory usage in MB
        
    Returns:
        list: FFmpeg command line options to limit memory usage
    """
    import subprocess
    import json
    import platform
    
    # Detect available hardware acceleration
    hw_accel = detect_hardware_acceleration()
    
    # Get video info to make intelligent decisions
    try:
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,codec_name',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'streams' in data and len(data['streams']) > 0:
                stream = data['streams'][0]
                width = int(stream.get('width', 0))
                height = int(stream.get('height', 0))
                codec = stream.get('codec_name', '')
                
                # Base memory flags that help in most cases
                memory_flags = [
                    '-threads', '4',  # Moderate thread count 
                ]
                
                # Add hardware acceleration if available
                if hw_accel:
                    memory_flags.extend(hw_accel)
                
                # Add codec-specific options
                if width * height > 1280 * 720:  # HD or higher
                    # Add more aggressive memory limitations for high-res videos
                    memory_flags.extend([
                        '-max_muxing_queue_size', '1024',
                        '-bufsize', f'{target_memory_mb}M',  # Increased buffer size
                    ])
                    
                    # For large videos, use frame dropping if needed
                    if width * height > 1920 * 1080:  # Full HD or higher
                        memory_flags.extend([
                            '-vsync', 'vfr',  # Variable framerate can help with memory
                            '-sws_flags', 'fast_bilinear',  # Faster scaling
                        ])
                
                return memory_flags
    except Exception as e:
        print(f"Error in get_ffmpeg_memory_flags: {str(e)}")
    
    # Default memory optimization flags if we couldn't analyze the video
    return [
        '-threads', '4',
        '-max_muxing_queue_size', '1024',
        '-bufsize', f'{target_memory_mb}M',
    ]

def detect_hardware_acceleration():
    """
    Detect available hardware acceleration on the system.
    
    Returns:
        list: FFmpeg hardware acceleration flags if available, empty list otherwise
    """
    import subprocess
    import platform
    import os
    
    # AWS instances typically use virtualized hardware that doesn't properly support
    # hardware acceleration, so we'll force CPU encoding
    print("Running on cloud server - using CPU encoding to ensure compatibility")
    return []
    
    # The below code is disabled to prevent errors with virtualized hardware
    # Uncomment and modify if you're running on hardware with proper GPU support
    """
    system = platform.system().lower()
    hw_accel = []
    
    try:
        # Check for NVIDIA GPUs (NVENC)
        if system in ['linux', 'windows']:
            try:
                # Check if nvidia-smi exists and returns successfully
                nvidia_check = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if nvidia_check.returncode == 0:
                    print("NVIDIA GPU detected, enabling NVENC acceleration")
                    # Return NVENC acceleration flags
                    return ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
            except FileNotFoundError:
                pass
        
        # Check for Intel QuickSync with additional validation
        if system in ['linux', 'windows']:
            try:
                # Check for Intel GPU via ffmpeg directly
                intel_check = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-hwaccels'],
                    capture_output=True, text=True
                )
                
                if 'qsv' in intel_check.stdout.lower():
                    # Verify the device exists and is accessible
                    device_path = '/dev/dri/renderD128'
                    if os.path.exists(device_path):
                        try:
                            # Try opening the device to verify permissions
                            with open(device_path, 'rb') as f:
                                print("Intel QuickSync detected and device is accessible")
                                return ['-hwaccel', 'qsv', '-qsv_device', '/dev/dri/renderD128']
                        except Exception as e:
                            print(f"Intel QuickSync device exists but is not accessible: {e}")
                    else:
                        print("Intel QuickSync detected but device path does not exist")
            except Exception as e:
                print(f"Error checking for Intel QuickSync: {e}")
        
        # Check for AMD GPU (Linux)
        if system == 'linux' and os.path.exists('/dev/dri/renderD128'):
            try:
                vaapi_check = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-hwaccels'],
                    capture_output=True, text=True
                )
                if 'vaapi' in vaapi_check.stdout.lower():
                    # Verify the device is accessible
                    try:
                        with open('/dev/dri/renderD128', 'rb') as f:
                            print("VAAPI compatible GPU detected and accessible")
                            return ['-hwaccel', 'vaapi', '-vaapi_device', '/dev/dri/renderD128']
                    except Exception as e:
                        print(f"VAAPI device exists but is not accessible: {e}")
            except Exception as e:
                print(f"Error checking for VAAPI: {e}")
        
        # Check for VideoToolbox on macOS
        if system == 'darwin':
            try:
                videotoolbox_check = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-hwaccels'],
                    capture_output=True, text=True
                )
                if 'videotoolbox' in videotoolbox_check.stdout.lower():
                    print("VideoToolbox detected on macOS, enabling acceleration")
                    return ['-hwaccel', 'videotoolbox']
            except Exception as e:
                print(f"Error checking for VideoToolbox: {e}")
        
    except Exception as e:
        print(f"Error in hardware acceleration detection: {str(e)}")
    """
    
    print("No compatible hardware acceleration detected, using CPU only")
    return []

def optimize_encoder_settings(width, height, quality="high", hw_encoder=None):
    """
    Get optimized encoder settings based on resolution and hardware.
    
    Args:
        width (int): Video width
        height (int): Video height
        quality (str): Desired quality level ("ultra", "high", "medium", "low")
        hw_encoder (str): Hardware encoder if detected
        
    Returns:
        dict: Dictionary with encoder settings
    """
    settings = {}
    
    # Use hardware encoder if available
    if hw_encoder == 'nvenc':
        settings['codec'] = 'h264_nvenc'
        
        if quality == "ultra":
            settings['preset'] = 'p7'  # Highest quality NVENC preset
            settings['rc'] = 'vbr_hq'
            settings['cq'] = '15'
            settings['qmin'] = '1'
            settings['qmax'] = '15'
        elif quality == "high":
            settings['preset'] = 'p6'
            settings['rc'] = 'vbr_hq'
            settings['cq'] = '20'
        elif quality == "medium":
            settings['preset'] = 'p4'
            settings['rc'] = 'vbr'
            settings['cq'] = '26'
        else:  # low
            settings['preset'] = 'p1'  # Fastest NVENC preset
            settings['rc'] = 'vbr'
            settings['cq'] = '30'
            
    elif hw_encoder == 'qsv':
        settings['codec'] = 'h264_qsv'
        
        if quality == "ultra":
            settings['preset'] = 'veryslow'
            settings['global_quality'] = '15'
        elif quality == "high":
            settings['preset'] = 'slow'
            settings['global_quality'] = '20'
        elif quality == "medium":
            settings['preset'] = 'medium'
            settings['global_quality'] = '25'
        else:  # low
            settings['preset'] = 'veryfast'
            settings['global_quality'] = '30'
            
    elif hw_encoder == 'vaapi':
        settings['codec'] = 'h264_vaapi'
        # VAAPI quality is controlled by bitrate
        
    elif hw_encoder == 'videotoolbox':
        settings['codec'] = 'h264_videotoolbox'
        
        if quality == "ultra":
            settings['profile'] = 'high'
            settings['allow_sw'] = '1'
            settings['realtime'] = '0'
            settings['max_rate'] = '0'
        else:
            settings['profile'] = 'main'
            settings['realtime'] = '1'
    
    else:
        # Software (CPU) encoding with libx264
        settings['codec'] = 'libx264'
        
        # Calculate optimal bitrate based on resolution and quality
        bitrate = calculate_optimal_bitrate(width, height, quality)
        settings['bitrate'] = f"{bitrate}k"
        
        if quality == "ultra":
            settings['preset'] = 'veryslow'
            settings['crf'] = '16'
            if supports_10bit():
                settings['pix_fmt'] = 'yuv420p10le'
            else:
                settings['pix_fmt'] = 'yuv420p'
        elif quality == "high":
            settings['preset'] = 'slow'
            settings['crf'] = '18'
            settings['pix_fmt'] = 'yuv420p'
        elif quality == "medium":
            settings['preset'] = 'medium'
            settings['crf'] = '23'
            settings['pix_fmt'] = 'yuv420p'
        else:  # low
            settings['preset'] = 'veryfast'
            settings['crf'] = '28'
            settings['pix_fmt'] = 'yuv420p'
    
    # Audio settings based on quality
    if quality == "ultra":
        settings['audio_codec'] = 'aac'
        settings['audio_bitrate'] = '320k'
        settings['audio_sample_rate'] = '48000'
    elif quality == "high":
        settings['audio_codec'] = 'aac'
        settings['audio_bitrate'] = '192k'
        settings['audio_sample_rate'] = '48000'
    elif quality == "medium":
        settings['audio_codec'] = 'aac'
        settings['audio_bitrate'] = '128k'
        settings['audio_sample_rate'] = '44100'
    else:  # low
        settings['audio_codec'] = 'aac'
        settings['audio_bitrate'] = '96k'
        settings['audio_sample_rate'] = '44100'
    
    return settings

def calculate_optimal_bitrate(width, height, quality="high"):
    """Calculate optimal bitrate based on video resolution and quality"""
    pixels = width * height
    
    # Base bitrates for different quality levels
    bitrate_factors = {
        "ultra": 1.5,
        "high": 1.0,
        "medium": 0.7,
        "low": 0.4
    }
    
    factor = bitrate_factors.get(quality.lower(), 1.0)
    
    if pixels <= 640 * 480:  # SD
        return int(3000 * factor)
    elif pixels <= 1280 * 720:  # 720p
        return int(7500 * factor)
    elif pixels <= 1920 * 1080:  # 1080p
        return int(15000 * factor)
    elif pixels <= 2560 * 1440:  # 1440p
        return int(24000 * factor)
    elif pixels <= 3840 * 2160:  # 4K
        return int(40000 * factor)
    else:  # Above 4K
        return int(75000 * factor)

def supports_10bit():
    """Check if the system supports 10-bit encoding"""
    try:
        import subprocess
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',
            '-f', 'lavfi',
            '-i', 'nullsrc=s=640x480:d=1',
            '-pix_fmt', 'yuv420p10le',
            '-f', 'null',
            '-'
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    except Exception:
        return False

def build_ffmpeg_command(input_file, output_file, settings, filters=None, start_time=None, duration=None):
    """
    Build optimized FFmpeg command with the given settings.
    
    Args:
        input_file (str): Input video file path
        output_file (str): Output video file path
        settings (dict): Encoder settings
        filters (str): FFmpeg filter chain
        start_time (float): Start time in seconds
        duration (float): Duration in seconds
        
    Returns:
        list: FFmpeg command line arguments
    """
    cmd = ['ffmpeg', '-y']
    
    # Add hardware acceleration if specified in settings
    if 'hwaccel' in settings:
        cmd.extend(['-hwaccel', settings['hwaccel']])
        if 'hwaccel_device' in settings:
            cmd.extend(['-hwaccel_device', settings['hwaccel_device']])
    
    # Add input file with trim options if needed
    if start_time is not None:
        cmd.extend(['-ss', str(start_time)])
    
    cmd.extend(['-i', input_file])
    
    if duration is not None:
        cmd.extend(['-t', str(duration)])
    
    # Add filters if specified
    if filters:
        cmd.extend(['-vf', filters])
    
    # Video codec settings
    cmd.extend(['-c:v', settings['codec']])
    
    # Add codec-specific options
    if settings['codec'] == 'libx264':
        cmd.extend(['-preset', settings['preset']])
        cmd.extend(['-crf', settings['crf']])
        if 'pix_fmt' in settings:
            cmd.extend(['-pix_fmt', settings['pix_fmt']])
        if 'bitrate' in settings:
            cmd.extend(['-b:v', settings['bitrate']])
            
    elif settings['codec'] == 'h264_nvenc':
        cmd.extend(['-preset', settings['preset']])
        cmd.extend(['-rc', settings['rc']])
        cmd.extend(['-cq', settings['cq']])
        if 'qmin' in settings:
            cmd.extend(['-qmin', settings['qmin']])
        if 'qmax' in settings:
            cmd.extend(['-qmax', settings['qmax']])
            
    elif settings['codec'] == 'h264_qsv':
        cmd.extend(['-preset', settings['preset']])
        cmd.extend(['-global_quality', settings['global_quality']])
        
    elif settings['codec'] == 'h264_vaapi':
        if 'bitrate' in settings:
            cmd.extend(['-b:v', settings['bitrate']])
            
    elif settings['codec'] == 'h264_videotoolbox':
        cmd.extend(['-profile', settings['profile']])
        cmd.extend(['-allow_sw', settings['allow_sw']])
        cmd.extend(['-realtime', settings['realtime']])
        if 'max_rate' in settings:
            cmd.extend(['-max_rate', settings['max_rate']])
    
    # Audio codec settings
    cmd.extend([
        '-c:a', settings['audio_codec'],
        '-b:a', settings['audio_bitrate'],
        '-ar', settings['audio_sample_rate']
    ])
    
    # Add output file with general options
    cmd.extend([
        '-movflags', '+faststart',  # Optimize for streaming
        output_file
    ])
    
    return cmd

def estimate_processing_time(duration, effect, width=1920, height=1080, slow_factor=0.5, quality="high"):
    """
    Estimate the video processing time based on resolution, duration and effect.
    
    Args:
        duration (float): Video duration in seconds
        effect (str): Processing effect
        width (int): Video width in pixels (default 1920)
        height (int): Video height in pixels (default 1080)
        slow_factor (float): Slow motion factor (default 0.5)
        quality (str): Quality setting (default "high")
        
    Returns:
        float: Estimated processing time in seconds
    """
    pixel_count = width * height
    
    # Base processing factors (seconds per megapixel-second)
    base_factors = {
        "ultra": 0.3,
        "high": 0.15,
        "medium": 0.08,
        "low": 0.04
    }
    
    # Effect complexity multipliers
    effect_multipliers = {
        "none": 1.0,
        "slomo": 2.5,
        "slomo_boomerang": 3.0,
        "custom_sequence": 3.5
    }
    
    base_factor = base_factors.get(quality.lower(), 0.15)
    effect_multiplier = effect_multipliers.get(effect.lower(), 1.0)
    
    # Calculate effective duration for slow motion
    if effect in ["slomo", "slomo_boomerang", "custom_sequence"]:
        effective_duration = duration / slow_factor
    else:
        effective_duration = duration
    
    # More pixels = more processing time
    pixel_factor = pixel_count / (1920 * 1080)  # Normalized to 1080p
    
    # Calculate estimated time
    estimated_seconds = base_factor * pixel_factor * effective_duration * effect_multiplier
    
    # Check if hardware acceleration is detected
    hw_accel = detect_hardware_acceleration()
    if hw_accel:
        # Hardware acceleration typically provides 2-4x speedup
        estimated_seconds /= 3.0
    
    return max(1.0, estimated_seconds)  # Ensure minimum of 1 second