"""
Video rendering service for physics simulations.
Renders MuJoCo simulations to MP4 video files.
"""

import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import mujoco

logger = logging.getLogger(__name__)


class VideoRenderer:
    """Renders physics simulations to video."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        codec: str = 'libx264'
    ):
        """
        Initialize video renderer.

        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            codec: Video codec (default: libx264 for H.264)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec

    async def render_video(
        self,
        mjcf_xml: str,
        duration: float = 10.0,
        output_path: Optional[str] = None,
        camera_name: Optional[str] = None
    ) -> str:
        """
        Render simulation to MP4 video.

        Args:
            mjcf_xml: MuJoCo XML string
            duration: Simulation duration in seconds
            output_path: Output file path (if None, uses temp file)
            camera_name: Name of camera to use (if None, uses default)

        Returns:
            Path to generated video file
        """
        try:
            # Load model
            model = mujoco.MjModel.from_xml_string(mjcf_xml)
            data = mujoco.MjData(model)

            # Create renderer
            renderer = mujoco.Renderer(model, self.height, self.width)

            # Set camera if specified
            if camera_name:
                renderer.update_scene(data, camera=camera_name)
            else:
                renderer.update_scene(data)

            # Calculate number of frames
            dt = model.opt.timestep
            num_steps = int(duration / dt)
            frame_skip = max(1, int(1.0 / (self.fps * dt)))

            # Collect frames
            frames: List[np.ndarray] = []
            logger.info(f"Rendering {num_steps} simulation steps at {self.fps} fps")

            for step in range(num_steps):
                # Step simulation
                mujoco.mj_step(model, data)

                # Render frame at specified FPS
                if step % frame_skip == 0:
                    renderer.update_scene(data)
                    pixels = renderer.render()
                    # Convert from RGB to BGR for OpenCV/ffmpeg
                    frames.append(np.flip(pixels, axis=0))

            logger.info(f"Rendered {len(frames)} frames")

            # Generate output path
            if output_path is None:
                temp_dir = tempfile.gettempdir()
                output_path = str(Path(temp_dir) / f"simulation_{hash(mjcf_xml) % 100000}.mp4")

            # Encode to video using ffmpeg
            await self._encode_video(frames, output_path)

            logger.info(f"Video saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Video rendering failed: {e}", exc_info=True)
            raise

    async def _encode_video(self, frames: List[np.ndarray], output_path: str):
        """
        Encode frames to video using ffmpeg.

        Args:
            frames: List of RGB frame arrays
            output_path: Output video file path
        """
        try:
            # Check if ffmpeg is available
            try:
                subprocess.run(['ffmpeg', '-version'],
                             capture_output=True,
                             check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("ffmpeg not found, falling back to imageio")
                await self._encode_video_imageio(frames, output_path)
                return

            # Create temporary raw video file
            temp_raw = output_path.replace('.mp4', '_raw.yuv')

            # Write frames to raw file
            height, width = frames[0].shape[:2]
            with open(temp_raw, 'wb') as f:
                for frame in frames:
                    f.write(frame.tobytes())

            # Encode with ffmpeg
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'rgb24',
                '-r', str(self.fps),
                '-i', temp_raw,
                '-c:v', self.codec,
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr}")
                raise RuntimeError(f"ffmpeg encoding failed: {result.stderr}")

            # Clean up temp file
            Path(temp_raw).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Video encoding failed: {e}")
            raise

    async def _encode_video_imageio(self, frames: List[np.ndarray], output_path: str):
        """
        Fallback video encoding using imageio (slower but no ffmpeg dependency).

        Args:
            frames: List of RGB frame arrays
            output_path: Output video file path
        """
        try:
            import imageio

            logger.info(f"Encoding video with imageio (fps={self.fps})")
            writer = imageio.get_writer(
                output_path,
                fps=self.fps,
                codec=self.codec,
                quality=8,
                pixelformat='yuv420p'
            )

            for frame in frames:
                writer.append_data(frame)

            writer.close()

        except ImportError:
            raise RuntimeError(
                "Neither ffmpeg nor imageio available. "
                "Install imageio: pip install imageio[ffmpeg]"
            )

    async def render_gif(
        self,
        mjcf_xml: str,
        duration: float = 10.0,
        output_path: Optional[str] = None,
        camera_name: Optional[str] = None,
        fps: int = 30
    ) -> str:
        """
        Render simulation to animated GIF.

        Args:
            mjcf_xml: MuJoCo XML string
            duration: Simulation duration in seconds
            output_path: Output file path
            camera_name: Camera to use
            fps: Frames per second (GIFs typically use lower fps)

        Returns:
            Path to generated GIF file
        """
        try:
            import imageio

            # Load model
            model = mujoco.MjModel.from_xml_string(mjcf_xml)
            data = mujoco.MjData(model)

            # Create renderer
            renderer = mujoco.Renderer(model, self.height, self.width)

            if camera_name:
                renderer.update_scene(data, camera=camera_name)
            else:
                renderer.update_scene(data)

            # Calculate frames
            dt = model.opt.timestep
            num_steps = int(duration / dt)
            frame_skip = max(1, int(1.0 / (fps * dt)))

            # Collect frames
            frames = []
            for step in range(num_steps):
                mujoco.mj_step(model, data)

                if step % frame_skip == 0:
                    renderer.update_scene(data)
                    pixels = renderer.render()
                    frames.append(np.flip(pixels, axis=0))

            # Generate output path
            if output_path is None:
                temp_dir = tempfile.gettempdir()
                output_path = str(Path(temp_dir) / f"simulation_{hash(mjcf_xml) % 100000}.gif")

            # Save as GIF
            imageio.mimsave(output_path, frames, fps=fps, loop=0)

            logger.info(f"GIF saved to {output_path}")
            return output_path

        except ImportError:
            raise RuntimeError("imageio required for GIF export: pip install imageio")
        except Exception as e:
            logger.error(f"GIF rendering failed: {e}")
            raise


# Singleton instance
_video_renderer = None

def get_video_renderer() -> VideoRenderer:
    """Get singleton video renderer instance."""
    global _video_renderer
    if _video_renderer is None:
        _video_renderer = VideoRenderer()
    return _video_renderer
