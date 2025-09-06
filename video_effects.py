"""
Comprehensive Video Effects System
Implements fade in/out, overlays, blur, and other video effects using FFmpeg.
"""
import os
import tempfile
import subprocess
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from logger import get_logger, log_error, LoggedOperation
from hardware_manager import get_hardware_manager

class EffectType(Enum):
    """Supported video effect types"""
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out" 
    FADE_IN_OUT = "fade_in_out"
    BLUR = "blur"
    MOTION_BLUR = "motion_blur"
    GAUSSIAN_BLUR = "gaussian_blur"
    OVERLAY_IMAGE = "overlay_image"
    OVERLAY_VIDEO = "overlay_video"
    WATERMARK = "watermark"
    COLOR_ADJUSTMENT = "color_adjustment"
    STABILIZATION = "stabilization"

@dataclass
class FadeEffect:
    """Configuration for fade effects"""
    fade_in_duration: float = 0.0  # seconds
    fade_out_duration: float = 0.0  # seconds
    fade_color: str = "black"  # black, white, transparent
    
@dataclass
class BlurEffect:
    """Configuration for blur effects"""
    blur_type: str = "gaussian"  # gaussian, box, motion
    intensity: float = 5.0  # blur intensity
    start_time: float = 0.0  # when to start blur
    duration: float = 0.0  # duration of blur (0 = entire video)
    
@dataclass
class OverlayEffect:
    """Configuration for overlay effects"""
    overlay_path: str
    position_x: Union[int, str] = "center"  # pixels or "left", "center", "right"
    position_y: Union[int, str] = "center"  # pixels or "top", "center", "bottom"
    scale: float = 1.0  # scale factor
    opacity: float = 1.0  # 0.0 to 1.0
    start_time: float = 0.0  # when to show overlay
    duration: float = 0.0  # duration (0 = entire video)
    blend_mode: str = "normal"  # normal, multiply, overlay, screen

class VideoEffectsProcessor:
    """
    Advanced video effects processor with hardware acceleration support
    """
    
    def __init__(self):
        self.logger = get_logger("video_effects")
        self.hardware_manager = get_hardware_manager()
        self.temp_files = []
        
    def create_fade_in_effect(self, input_video: str, output_video: str, 
                            fade_duration: float = 1.0, fade_color: str = "black") -> bool:
        """
        Create fade-in effect at the beginning of video
        
        Args:
            input_video: Input video file path
            output_video: Output video file path
            fade_duration: Duration of fade-in in seconds
            fade_color: Color to fade from (black, white, transparent)
            
        Returns:
            bool: True if successful
        """
        try:
            with LoggedOperation("fade_in_effect", fade_duration=fade_duration) as op:
                # Build FFmpeg command with hardware acceleration
                hw_args = self.hardware_manager.get_ffmpeg_args("high")
                
                cmd = ['ffmpeg', '-y'] + hw_args + [
                    '-i', input_video,
                    '-filter_complex', f'fade=t=in:st=0:d={fade_duration}:color={fade_color}',
                    '-c:a', 'copy',  # Copy audio without re-encoding
                    output_video
                ]
                
                self.logger.info("Creating fade-in effect", extra={
                    'input_video': input_video,
                    'fade_duration': fade_duration,
                    'fade_color': fade_color
                })
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.logger.info("Fade-in effect created successfully")
                    return True
                else:
                    self.logger.error(f"Fade-in effect failed: {result.stderr}")
                    return False
                    
        except Exception as e:
            log_error(e, "fade_in_effect", input_video=input_video, fade_duration=fade_duration)
            return False
    
    def create_fade_out_effect(self, input_video: str, output_video: str,
                             fade_duration: float = 1.0, fade_color: str = "black") -> bool:
        """
        Create fade-out effect at the end of video
        
        Args:
            input_video: Input video file path
            output_video: Output video file path  
            fade_duration: Duration of fade-out in seconds
            fade_color: Color to fade to (black, white, transparent)
            
        Returns:
            bool: True if successful
        """
        try:
            with LoggedOperation("fade_out_effect", fade_duration=fade_duration) as op:
                # Get video duration first
                duration = self._get_video_duration(input_video)
                if duration <= 0:
                    raise ValueError("Could not determine video duration")
                
                fade_start = max(0, duration - fade_duration)
                
                hw_args = self.hardware_manager.get_ffmpeg_args("high")
                
                cmd = ['ffmpeg', '-y'] + hw_args + [
                    '-i', input_video,
                    '-filter_complex', f'fade=t=out:st={fade_start}:d={fade_duration}:color={fade_color}',
                    '-c:a', 'copy',
                    output_video
                ]
                
                self.logger.info("Creating fade-out effect", extra={
                    'input_video': input_video,
                    'fade_duration': fade_duration,
                    'fade_start': fade_start,
                    'video_duration': duration
                })
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.logger.info("Fade-out effect created successfully")
                    return True
                else:
                    self.logger.error(f"Fade-out effect failed: {result.stderr}")
                    return False
                    
        except Exception as e:
            log_error(e, "fade_out_effect", input_video=input_video, fade_duration=fade_duration)
            return False
    
    def create_fade_in_out_effect(self, input_video: str, output_video: str,
                                fade_in_duration: float = 1.0, fade_out_duration: float = 1.0,
                                fade_color: str = "black") -> bool:
        """
        Create both fade-in and fade-out effects
        
        Args:
            input_video: Input video file path
            output_video: Output video file path
            fade_in_duration: Duration of fade-in in seconds
            fade_out_duration: Duration of fade-out in seconds
            fade_color: Color for fading
            
        Returns:
            bool: True if successful
        """
        try:
            with LoggedOperation("fade_in_out_effect", 
                               fade_in=fade_in_duration, fade_out=fade_out_duration) as op:
                
                duration = self._get_video_duration(input_video)
                if duration <= 0:
                    raise ValueError("Could not determine video duration")
                
                fade_out_start = max(0, duration - fade_out_duration)
                
                hw_args = self.hardware_manager.get_ffmpeg_args("high")
                
                # Create filter that combines both fade effects
                fade_filter = f'fade=t=in:st=0:d={fade_in_duration}:color={fade_color},fade=t=out:st={fade_out_start}:d={fade_out_duration}:color={fade_color}'
                
                cmd = ['ffmpeg', '-y'] + hw_args + [
                    '-i', input_video,
                    '-filter_complex', fade_filter,
                    '-c:a', 'copy',
                    output_video
                ]
                
                self.logger.info("Creating fade-in-out effect", extra={
                    'fade_in_duration': fade_in_duration,
                    'fade_out_duration': fade_out_duration,
                    'video_duration': duration
                })
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.logger.info("Fade-in-out effect created successfully")
                    return True
                else:
                    self.logger.error(f"Fade-in-out effect failed: {result.stderr}")
                    return False
                    
        except Exception as e:
            log_error(e, "fade_in_out_effect", input_video=input_video)
            return False
    
    def create_blur_effect(self, input_video: str, output_video: str,
                         blur_type: str = "gaussian", intensity: float = 5.0,
                         start_time: float = 0.0, duration: float = 0.0) -> bool:
        """
        Create blur effect on video
        
        Args:
            input_video: Input video file path
            output_video: Output video file path
            blur_type: Type of blur (gaussian, box, motion)
            intensity: Blur intensity (1.0 to 20.0)
            start_time: When to start blur effect
            duration: Duration of blur (0 = entire video)
            
        Returns:
            bool: True if successful
        """
        try:
            with LoggedOperation("blur_effect", blur_type=blur_type, intensity=intensity) as op:
                
                video_duration = self._get_video_duration(input_video)
                if duration <= 0:
                    duration = video_duration
                
                hw_args = self.hardware_manager.get_ffmpeg_args("high")
                
                # Create appropriate blur filter
                if blur_type == "gaussian":
                    blur_filter = f'gblur=sigma={intensity}'
                elif blur_type == "box":
                    blur_filter = f'boxblur=luma_radius={int(intensity)}:luma_power=2'
                elif blur_type == "motion":
                    blur_filter = f'mblur=radius={int(intensity)}'
                else:
                    blur_filter = f'gblur=sigma={intensity}'  # Default to gaussian
                
                # Apply blur for specific time range if needed
                if start_time > 0 or duration < video_duration:
                    end_time = start_time + duration
                    filter_complex = f'[0:v]split[original][blur];[blur]{blur_filter}[blurred];[original][blurred]overlay=enable=\'between(t,{start_time},{end_time})\''
                else:
                    filter_complex = blur_filter
                
                cmd = ['ffmpeg', '-y'] + hw_args + [
                    '-i', input_video,
                    '-filter_complex', filter_complex,
                    '-c:a', 'copy',
                    output_video
                ]
                
                self.logger.info("Creating blur effect", extra={
                    'blur_type': blur_type,
                    'intensity': intensity,
                    'start_time': start_time,
                    'duration': duration
                })
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.logger.info("Blur effect created successfully")
                    return True
                else:
                    self.logger.error(f"Blur effect failed: {result.stderr}")
                    return False
                    
        except Exception as e:
            log_error(e, "blur_effect", input_video=input_video, blur_type=blur_type)
            return False
    
    def create_image_overlay(self, input_video: str, overlay_image: str, output_video: str,
                           position_x: Union[int, str] = "center", position_y: Union[int, str] = "center",
                           scale: float = 1.0, opacity: float = 1.0,
                           start_time: float = 0.0, duration: float = 0.0) -> bool:
        """
        Create image overlay effect on video
        
        Args:
            input_video: Input video file path
            overlay_image: Overlay image file path
            output_video: Output video file path
            position_x: X position (pixels, "left", "center", "right")
            position_y: Y position (pixels, "top", "center", "bottom")
            scale: Scale factor for overlay
            opacity: Opacity (0.0 to 1.0)
            start_time: When to show overlay
            duration: Duration of overlay (0 = entire video)
            
        Returns:
            bool: True if successful
        """
        try:
            with LoggedOperation("image_overlay", overlay_image=overlay_image) as op:
                
                if not os.path.exists(overlay_image):
                    raise FileNotFoundError(f"Overlay image not found: {overlay_image}")
                
                video_duration = self._get_video_duration(input_video)
                if duration <= 0:
                    duration = video_duration
                
                # Calculate position
                x_pos, y_pos = self._calculate_overlay_position(
                    input_video, overlay_image, position_x, position_y, scale
                )
                
                hw_args = self.hardware_manager.get_ffmpeg_args("high")
                
                # Build overlay filter
                overlay_filter = f'[1:v]scale=iw*{scale}:ih*{scale}[scaled_overlay];'
                
                if opacity < 1.0:
                    overlay_filter += f'[scaled_overlay]format=rgba,colorchannelmixer=aa={opacity}[transparent_overlay];'
                    overlay_name = 'transparent_overlay'
                else:
                    overlay_name = 'scaled_overlay'
                
                # Apply overlay with timing
                if start_time > 0 or duration < video_duration:
                    end_time = start_time + duration
                    overlay_filter += f'[0:v][{overlay_name}]overlay={x_pos}:{y_pos}:enable=\'between(t,{start_time},{end_time})\''
                else:
                    overlay_filter += f'[0:v][{overlay_name}]overlay={x_pos}:{y_pos}'
                
                cmd = ['ffmpeg', '-y'] + hw_args + [
                    '-i', input_video,
                    '-i', overlay_image,
                    '-filter_complex', overlay_filter,
                    '-c:a', 'copy',
                    output_video
                ]
                
                self.logger.info("Creating image overlay", extra={
                    'overlay_image': overlay_image,
                    'position': f"{position_x},{position_y}",
                    'scale': scale,
                    'opacity': opacity
                })
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.logger.info("Image overlay created successfully")
                    return True
                else:
                    self.logger.error(f"Image overlay failed: {result.stderr}")
                    return False
                    
        except Exception as e:
            log_error(e, "image_overlay", input_video=input_video, overlay_image=overlay_image)
            return False
    
    def create_video_overlay(self, input_video: str, overlay_video: str, output_video: str,
                           position_x: Union[int, str] = "center", position_y: Union[int, str] = "center",
                           scale: float = 0.3, opacity: float = 1.0,
                           start_time: float = 0.0) -> bool:
        """
        Create video overlay effect (picture-in-picture)
        
        Args:
            input_video: Main video file path
            overlay_video: Overlay video file path
            output_video: Output video file path
            position_x: X position for overlay
            position_y: Y position for overlay
            scale: Scale factor for overlay video
            opacity: Opacity of overlay
            start_time: When to start overlay
            
        Returns:
            bool: True if successful
        """
        try:
            with LoggedOperation("video_overlay", overlay_video=overlay_video) as op:
                
                if not os.path.exists(overlay_video):
                    raise FileNotFoundError(f"Overlay video not found: {overlay_video}")
                
                # Calculate position
                x_pos, y_pos = self._calculate_overlay_position(
                    input_video, overlay_video, position_x, position_y, scale
                )
                
                hw_args = self.hardware_manager.get_ffmpeg_args("high")
                
                # Build complex filter for video overlay
                filter_parts = []
                filter_parts.append(f'[1:v]scale=iw*{scale}:ih*{scale}[scaled_overlay]')
                
                if opacity < 1.0:
                    filter_parts.append(f'[scaled_overlay]format=rgba,colorchannelmixer=aa={opacity}[transparent_overlay]')
                    overlay_name = 'transparent_overlay'
                else:
                    overlay_name = 'scaled_overlay'
                
                if start_time > 0:
                    filter_parts.append(f'[0:v][{overlay_name}]overlay={x_pos}:{y_pos}:enable=\'gte(t,{start_time})\'')
                else:
                    filter_parts.append(f'[0:v][{overlay_name}]overlay={x_pos}:{y_pos}')
                
                filter_complex = ';'.join(filter_parts)
                
                cmd = ['ffmpeg', '-y'] + hw_args + [
                    '-i', input_video,
                    '-i', overlay_video,
                    '-filter_complex', filter_complex,
                    '-c:a', 'copy',  # Use audio from main video
                    '-shortest',     # End when shortest input ends
                    output_video
                ]
                
                self.logger.info("Creating video overlay", extra={
                    'overlay_video': overlay_video,
                    'position': f"{position_x},{position_y}",
                    'scale': scale
                })
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    self.logger.info("Video overlay created successfully")
                    return True
                else:
                    self.logger.error(f"Video overlay failed: {result.stderr}")
                    return False
                    
        except Exception as e:
            log_error(e, "video_overlay", input_video=input_video, overlay_video=overlay_video)
            return False
    
    def create_combined_effects(self, input_video: str, output_video: str, 
                               effects: List[Dict[str, Any]]) -> bool:
        """
        Apply multiple effects in a single pass for efficiency
        
        Args:
            input_video: Input video file path
            output_video: Output video file path
            effects: List of effect configurations
            
        Returns:
            bool: True if successful
        """
        try:
            with LoggedOperation("combined_effects", effect_count=len(effects)) as op:
                
                if not effects:
                    self.logger.warning("No effects provided, copying original video")
                    return self._copy_file(input_video, output_video)
                
                hw_args = self.hardware_manager.get_ffmpeg_args("high")
                
                # Build complex filter chain
                filter_parts = []
                input_files = [input_video]
                input_index = 0
                
                current_stream = f'[{input_index}:v]'
                
                for i, effect in enumerate(effects):
                    effect_type = effect.get('type')
                    
                    if effect_type == 'fade_in':
                        duration = effect.get('duration', 1.0)
                        color = effect.get('color', 'black')
                        current_stream = f'{current_stream}fade=t=in:st=0:d={duration}:color={color}[fade_in_{i}]'
                        current_stream = f'[fade_in_{i}]'
                        
                    elif effect_type == 'fade_out':
                        duration = effect.get('duration', 1.0)
                        color = effect.get('color', 'black')
                        video_duration = self._get_video_duration(input_video)
                        start = max(0, video_duration - duration)
                        current_stream = f'{current_stream}fade=t=out:st={start}:d={duration}:color={color}[fade_out_{i}]'
                        current_stream = f'[fade_out_{i}]'
                        
                    elif effect_type == 'blur':
                        intensity = effect.get('intensity', 5.0)
                        current_stream = f'{current_stream}gblur=sigma={intensity}[blur_{i}]'
                        current_stream = f'[blur_{i}]'
                        
                    elif effect_type == 'overlay':
                        overlay_path = effect.get('overlay_path')
                        if overlay_path and os.path.exists(overlay_path):
                            input_files.append(overlay_path)
                            overlay_index = len(input_files) - 1
                            scale = effect.get('scale', 1.0)
                            x_pos, y_pos = self._calculate_overlay_position(
                                input_video, overlay_path, 
                                effect.get('position_x', 'center'),
                                effect.get('position_y', 'center'),
                                scale
                            )
                            
                            filter_parts.append(f'[{overlay_index}:v]scale=iw*{scale}:ih*{scale}[overlay_{i}]')
                            current_stream = f'{current_stream}[overlay_{i}]overlay={x_pos}:{y_pos}[overlaid_{i}]'
                            current_stream = f'[overlaid_{i}]'
                
                # Remove the brackets from the final stream name for output
                final_stream = current_stream.strip('[]')
                
                if filter_parts:
                    filter_complex = ';'.join(filter_parts + [current_stream.replace('[', '').replace(']', '')])
                else:
                    filter_complex = current_stream.replace('[', '').replace(']', '')
                
                # Build command
                cmd = ['ffmpeg', '-y'] + hw_args
                
                # Add all input files
                for input_file in input_files:
                    cmd.extend(['-i', input_file])
                
                cmd.extend([
                    '-filter_complex', filter_complex,
                    '-c:a', 'copy',
                    output_video
                ])
                
                self.logger.info("Creating combined effects", extra={
                    'effect_count': len(effects),
                    'input_files': len(input_files)
                })
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    self.logger.info("Combined effects created successfully")
                    return True
                else:
                    self.logger.error(f"Combined effects failed: {result.stderr}")
                    return False
                    
        except Exception as e:
            log_error(e, "combined_effects", input_video=input_video, effect_count=len(effects))
            return False
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                self.logger.error(f"Could not get video duration: {result.stderr}")
                return 0.0
                
        except Exception as e:
            log_error(e, "get_video_duration", video_path=video_path)
            return 0.0
    
    def _calculate_overlay_position(self, main_video: str, overlay_path: str,
                                  position_x: Union[int, str], position_y: Union[int, str],
                                  scale: float = 1.0) -> tuple:
        """Calculate overlay position based on parameters"""
        try:
            # For simplicity, return preset positions
            # In a full implementation, you'd get actual video dimensions
            
            if isinstance(position_x, str):
                if position_x == "left":
                    x_pos = "10"
                elif position_x == "right":  
                    x_pos = "main_w-overlay_w-10"
                else:  # center
                    x_pos = "(main_w-overlay_w)/2"
            else:
                x_pos = str(position_x)
            
            if isinstance(position_y, str):
                if position_y == "top":
                    y_pos = "10"
                elif position_y == "bottom":
                    y_pos = "main_h-overlay_h-10"
                else:  # center
                    y_pos = "(main_h-overlay_h)/2"
            else:
                y_pos = str(position_y)
            
            return x_pos, y_pos
            
        except Exception as e:
            log_error(e, "calculate_overlay_position")
            return "10", "10"  # Fallback to top-left
    
    def _copy_file(self, source: str, destination: str) -> bool:
        """Copy file from source to destination"""
        try:
            import shutil
            shutil.copy2(source, destination)
            return True
        except Exception as e:
            log_error(e, "copy_file", source=source, destination=destination)
            return False
    
    def cleanup_temp_files(self):
        """Clean up any temporary files created during processing"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                self.logger.warning(f"Could not delete temp file {temp_file}: {e}")
        
        self.temp_files.clear()

# Global effects processor instance
effects_processor = VideoEffectsProcessor()

def get_effects_processor() -> VideoEffectsProcessor:
    """Get the global video effects processor instance"""
    return effects_processor