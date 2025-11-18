"""
Video rendering service for physics simulations.
Renders MuJoCo simulations to MP4 video files.
"""

import asyncio
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mujoco

logger = logging.getLogger(__name__)

# Thread pool for CPU-intensive rendering tasks
_render_executor = ThreadPoolExecutor(max_workers=2)


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
            # Generate output path
            if output_path is None:
                temp_dir = tempfile.gettempdir()
                output_path = str(Path(temp_dir) / f"simulation_{hash(mjcf_xml) % 100000}.mp4")

            # Run rendering in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _render_executor,
                self._render_video_sync,
                mjcf_xml,
                duration,
                output_path,
                camera_name
            )

            logger.info(f"Video saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Video rendering failed: {e}", exc_info=True)
            raise

    def _render_video_sync(
        self,
        mjcf_xml: str,
        duration: float,
        output_path: str,
        camera_name: Optional[str]
    ):
        """
        Synchronous video rendering (runs in thread pool).
        Streams frames directly to ffmpeg to avoid memory issues.
        """
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
        expected_frames = num_steps // frame_skip

        logger.info(f"Rendering {expected_frames} frames at {self.fps} fps (streaming to ffmpeg)")

        # Check if ffmpeg is available
        ffmpeg_available = self._check_ffmpeg()

        if not ffmpeg_available:
            logger.warning("ffmpeg not found, using imageio fallback (memory intensive)")
            return self._render_with_imageio(model, data, renderer, num_steps, frame_skip, output_path)

        # Start ffmpeg process to stream frames directly
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'rgb24',
            '-r', str(self.fps),
            '-i', '-',  # Read from stdin
            '-c:v', self.codec,
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            output_path
        ]

        # Start ffmpeg process
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        try:
            frame_count = 0
            for step in range(num_steps):
                # Step simulation
                mujoco.mj_step(model, data)

                # Render frame at specified FPS
                if step % frame_skip == 0:
                    renderer.update_scene(data)
                    pixels = renderer.render()
                    # Flip vertically and write directly to ffmpeg stdin
                    flipped = np.flip(pixels, axis=0)
                    process.stdin.write(flipped.tobytes())
                    frame_count += 1

            # Close stdin to signal end of stream
            process.stdin.close()

            # Wait for ffmpeg to finish
            stdout, stderr = process.communicate(timeout=60)

            if process.returncode != 0:
                logger.error(f"ffmpeg error: {stderr.decode()}")
                raise RuntimeError(f"ffmpeg encoding failed: {stderr.decode()}")

            logger.info(f"Successfully encoded {frame_count} frames")

        except Exception as e:
            process.kill()
            process.wait()
            raise

    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                check=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _render_with_imageio(
        self,
        model,
        data,
        renderer,
        num_steps: int,
        frame_skip: int,
        output_path: str
    ):
        """
        Fallback rendering using imageio (loads frames in memory).
        Only used when ffmpeg is not available.
        """
        try:
            import imageio
        except ImportError:
            raise RuntimeError(
                "Neither ffmpeg nor imageio available. "
                "Install imageio: pip install imageio[ffmpeg]"
            )

        logger.warning("Using imageio fallback - this loads all frames in memory")

        frames = []
        for step in range(num_steps):
            mujoco.mj_step(model, data)
            if step % frame_skip == 0:
                renderer.update_scene(data)
                pixels = renderer.render()
                frames.append(np.flip(pixels, axis=0))

        logger.info(f"Encoding {len(frames)} frames with imageio")
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
            # Generate output path
            if output_path is None:
                temp_dir = tempfile.gettempdir()
                output_path = str(Path(temp_dir) / f"simulation_{hash(mjcf_xml) % 100000}.gif")

            # Run rendering in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _render_executor,
                self._render_gif_sync,
                mjcf_xml,
                duration,
                output_path,
                camera_name,
                fps
            )

            logger.info(f"GIF saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"GIF rendering failed: {e}")
            raise

    def _render_gif_sync(
        self,
        mjcf_xml: str,
        duration: float,
        output_path: str,
        camera_name: Optional[str],
        fps: int
    ):
        """
        Synchronous GIF rendering (runs in thread pool).
        Note: GIFs load frames in memory, but are typically short duration.
        """
        try:
            import imageio
        except ImportError:
            raise RuntimeError("imageio required for GIF export: pip install imageio")

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

        logger.info(f"Rendering GIF with {num_steps // frame_skip} frames at {fps} fps")

        # Collect frames (acceptable for short GIFs)
        frames = []
        for step in range(num_steps):
            mujoco.mj_step(model, data)

            if step % frame_skip == 0:
                renderer.update_scene(data)
                pixels = renderer.render()
                frames.append(np.flip(pixels, axis=0))

        # Save as GIF
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        logger.info(f"GIF encoding complete: {len(frames)} frames")


# Singleton instance
_video_renderer = None

def get_video_renderer() -> VideoRenderer:
    """Get singleton video renderer instance."""
    global _video_renderer
    if _video_renderer is None:
        _video_renderer = VideoRenderer()
    return _video_renderer
