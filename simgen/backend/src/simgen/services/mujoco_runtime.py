"""
MuJoCo Runtime: Physics simulation engine and stepper
Handles model loading, stepping, and state extraction
"""

import asyncio
import logging
import struct
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("MuJoCo not available. Install with: pip install mujoco")

logger = logging.getLogger(__name__)

class SimulationStatus(Enum):
    """Simulation state"""
    IDLE = "idle"
    LOADING = "loading"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"

@dataclass
class SimulationFrame:
    """
    Single frame of simulation data
    Optimized for binary streaming
    """
    frame_id: int
    sim_time: float
    qpos: np.ndarray      # Joint positions
    qvel: np.ndarray      # Joint velocities
    xpos: np.ndarray      # Body positions (world frame)
    xquat: np.ndarray     # Body orientations (quaternions, wxyz)
    actuator_force: Optional[np.ndarray] = None
    sensor_data: Optional[np.ndarray] = None
    contact_forces: Optional[np.ndarray] = None

    def to_binary(self) -> bytes:
        """
        Pack frame into binary format for WebSocket streaming
        Format: [frame_id(u32), sim_time(f32), qpos[], xpos[], xquat[]]
        """
        parts = [
            struct.pack('<I', self.frame_id),      # unsigned int, little-endian
            struct.pack('<f', self.sim_time),       # float, little-endian
            self.qpos.astype(np.float32).tobytes(),
            self.xpos.astype(np.float32).tobytes(),
            self.xquat.astype(np.float32).tobytes()
        ]

        # Optional data
        if self.actuator_force is not None:
            parts.append(self.actuator_force.astype(np.float32).tobytes())
        if self.sensor_data is not None:
            parts.append(self.sensor_data.astype(np.float32).tobytes())

        return b''.join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (fallback)"""
        return {
            'frame_id': self.frame_id,
            'sim_time': self.sim_time,
            'qpos': self.qpos.tolist(),
            'qvel': self.qvel.tolist(),
            'xpos': self.xpos.reshape(-1, 3).tolist(),
            'xquat': self.xquat.reshape(-1, 4).tolist(),
            'actuator_force': self.actuator_force.tolist() if self.actuator_force is not None else None,
            'sensor_data': self.sensor_data.tolist() if self.sensor_data is not None else None
        }

@dataclass
class ModelManifest:
    """
    Model metadata for client-side rendering
    Sent once at simulation start
    """
    model_name: str
    nbody: int
    nq: int  # Number of DOFs
    nv: int  # Number of velocities
    nu: int  # Number of actuators
    nsensor: int
    body_names: List[str]
    joint_names: List[str]
    actuator_names: List[str]
    sensor_names: List[str]
    geom_types: List[str]
    geom_sizes: List[List[float]]
    geom_rgba: List[List[float]]
    timestep: float
    gravity: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

class MuJoCoRuntime:
    """
    MuJoCo physics runtime with async stepping
    """

    def __init__(self,
                 timestep: float = 0.002,
                 render_fps: int = 60,
                 max_sim_time: float = 30.0,
                 headless: bool = True):
        """
        Initialize runtime

        Args:
            timestep: Physics timestep (seconds)
            render_fps: Frame streaming rate
            max_sim_time: Maximum simulation duration
            headless: Run without GUI (for servers)
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is not installed")

        self.timestep = timestep
        self.render_fps = render_fps
        self.max_sim_time = max_sim_time
        self.headless = headless

        # MuJoCo objects
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.renderer: Optional[mujoco.Renderer] = None

        # Simulation state
        self.status = SimulationStatus.IDLE
        self.frame_count = 0
        self.sim_time = 0.0
        self.real_time_factor = 1.0

        # Frame buffer for streaming
        self.frame_buffer: List[SimulationFrame] = []
        self.max_buffer_size = 120  # 2 seconds at 60 FPS

        # Callbacks
        self.frame_callback: Optional[Callable[[SimulationFrame], None]] = None
        self.error_callback: Optional[Callable[[str], None]] = None

        # Control
        self._stop_flag = False
        self._pause_flag = False

        # Configure headless rendering
        if headless:
            import os
            os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless

    def load_mjcf(self, mjcf_xml: str) -> ModelManifest:
        """
        Load model from MJCF XML string

        Returns:
            ModelManifest with model metadata
        """
        try:
            self.status = SimulationStatus.LOADING
            logger.info("Loading MJCF model...")

            # Parse XML and create model
            self.model = mujoco.MjModel.from_xml_string(mjcf_xml)
            self.data = mujoco.MjData(self.model)

            # Override timestep if specified
            if self.timestep:
                self.model.opt.timestep = self.timestep

            # Initialize renderer for headless mode
            if self.headless and not self.renderer:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)

            # Reset simulation
            mujoco.mj_resetData(self.model, self.data)

            # Create manifest
            manifest = self._create_manifest()

            self.status = SimulationStatus.IDLE
            logger.info(f"Model loaded: {manifest.nbody} bodies, {manifest.nq} DOFs")

            return manifest

        except Exception as e:
            self.status = SimulationStatus.ERROR
            logger.error(f"Failed to load MJCF: {e}")
            if self.error_callback:
                self.error_callback(str(e))
            raise

    def _create_manifest(self) -> ModelManifest:
        """Extract model metadata for client"""
        model = self.model

        # Body names
        body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
                      for i in range(model.nbody)]

        # Joint names
        joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
                       for i in range(model.njnt)]

        # Actuator names
        actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"
                          for i in range(model.nu)]

        # Sensor names
        sensor_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor_{i}"
                        for i in range(model.nsensor)]

        # Geom properties
        geom_types = []
        geom_sizes = []
        geom_rgba = []

        for i in range(model.ngeom):
            geom_types.append(model.geom_type[i])
            geom_sizes.append(model.geom_size[i].tolist())
            geom_rgba.append(model.geom_rgba[i].tolist())

        return ModelManifest(
            model_name=model.names.decode('utf-8').split('\x00')[0] if model.names else "unnamed",
            nbody=model.nbody,
            nq=model.nq,
            nv=model.nv,
            nu=model.nu,
            nsensor=model.nsensor,
            body_names=body_names,
            joint_names=joint_names,
            actuator_names=actuator_names,
            sensor_names=sensor_names,
            geom_types=geom_types,
            geom_sizes=geom_sizes,
            geom_rgba=geom_rgba,
            timestep=model.opt.timestep,
            gravity=model.opt.gravity.tolist()
        )

    def set_control(self, actuator_values: np.ndarray):
        """Set actuator control values"""
        if self.data and len(actuator_values) == self.model.nu:
            self.data.ctrl[:] = actuator_values

    def get_state(self) -> SimulationFrame:
        """Get current simulation state as frame"""
        if not self.data:
            raise RuntimeError("No simulation loaded")

        # Extract body transforms
        xpos = self.data.xpos.copy()  # Body positions
        xquat = self.data.xquat.copy()  # Body quaternions

        # Extract joint state
        qpos = self.data.qpos.copy() if self.model.nq > 0 else np.array([])
        qvel = self.data.qvel.copy() if self.model.nv > 0 else np.array([])

        # Extract actuator forces
        actuator_force = self.data.actuator_force.copy() if self.model.nu > 0 else None

        # Extract sensor data
        sensor_data = self.data.sensordata.copy() if self.model.nsensor > 0 else None

        return SimulationFrame(
            frame_id=self.frame_count,
            sim_time=self.data.time,
            qpos=qpos,
            qvel=qvel,
            xpos=xpos,
            xquat=xquat,
            actuator_force=actuator_force,
            sensor_data=sensor_data
        )

    def step(self, n_steps: int = 1) -> SimulationFrame:
        """
        Step simulation forward

        Args:
            n_steps: Number of physics steps

        Returns:
            Current frame after stepping
        """
        if not self.model or not self.data:
            raise RuntimeError("No model loaded")

        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
            self.frame_count += 1
            self.sim_time = self.data.time

        return self.get_state()

    async def run_async(self,
                        duration: Optional[float] = None,
                        callback: Optional[Callable[[SimulationFrame], None]] = None) -> None:
        """
        Run simulation asynchronously

        Args:
            duration: Simulation duration (uses max_sim_time if None)
            callback: Called for each frame
        """
        if not self.model or not self.data:
            raise RuntimeError("No model loaded")

        self.status = SimulationStatus.RUNNING
        self._stop_flag = False
        self.frame_callback = callback or self.frame_callback

        duration = duration or self.max_sim_time
        frame_interval = 1.0 / self.render_fps
        physics_steps_per_frame = max(1, int(frame_interval / self.model.opt.timestep))

        start_time = time.time()
        last_frame_time = start_time

        try:
            while self.data.time < duration and not self._stop_flag:
                # Handle pause
                if self._pause_flag:
                    self.status = SimulationStatus.PAUSED
                    await asyncio.sleep(0.1)
                    continue

                self.status = SimulationStatus.RUNNING

                # Step physics
                frame = self.step(physics_steps_per_frame)

                # Add to buffer
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) > self.max_buffer_size:
                    self.frame_buffer.pop(0)

                # Callback
                if self.frame_callback:
                    self.frame_callback(frame)

                # Frame rate limiting
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    await asyncio.sleep(frame_interval - elapsed)
                last_frame_time = time.time()

                # Calculate real-time factor
                wall_time = time.time() - start_time
                if wall_time > 0:
                    self.real_time_factor = self.data.time / wall_time

            self.status = SimulationStatus.COMPLETED
            logger.info(f"Simulation completed: {self.data.time:.2f}s simulated, "
                       f"RTF: {self.real_time_factor:.2f}x")

        except Exception as e:
            self.status = SimulationStatus.ERROR
            logger.error(f"Simulation error: {e}")
            if self.error_callback:
                self.error_callback(str(e))
            raise

    def stop(self):
        """Stop simulation"""
        self._stop_flag = True

    def pause(self):
        """Pause simulation"""
        self._pause_flag = True

    def resume(self):
        """Resume simulation"""
        self._pause_flag = False

    def reset(self):
        """Reset simulation to initial state"""
        if self.model and self.data:
            mujoco.mj_resetData(self.model, self.data)
            self.frame_count = 0
            self.sim_time = 0.0
            self.frame_buffer.clear()
            self.status = SimulationStatus.IDLE

    def render_offscreen(self, width: int = 640, height: int = 480, camera_id: int = -1) -> np.ndarray:
        """
        Render current frame offscreen

        Returns:
            RGB image as numpy array (height, width, 3)
        """
        if not self.renderer:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self.renderer.update_scene(self.data, camera=camera_id)
        pixels = self.renderer.render()

        return pixels

    def get_buffer_frames(self, n_frames: Optional[int] = None) -> List[SimulationFrame]:
        """Get recent frames from buffer"""
        if n_frames:
            return self.frame_buffer[-n_frames:]
        return self.frame_buffer.copy()

    def cleanup(self):
        """Clean up resources"""
        self.frame_buffer.clear()
        self.model = None
        self.data = None
        if self.renderer:
            self.renderer = None
        self.status = SimulationStatus.IDLE