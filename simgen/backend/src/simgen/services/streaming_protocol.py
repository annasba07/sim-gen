"""
Binary WebSocket Streaming Protocol
Efficient real-time physics frame streaming
"""

import asyncio
import logging
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Set, Any, Callable
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np

logger = logging.getLogger(__name__)

class MessageType(IntEnum):
    """WebSocket message types"""
    # Client → Server
    CONNECT = 0x01
    DISCONNECT = 0x02
    LOAD_MODEL = 0x10
    START_SIM = 0x11
    STOP_SIM = 0x12
    PAUSE_SIM = 0x13
    RESUME_SIM = 0x14
    RESET_SIM = 0x15
    SET_CONTROL = 0x20
    REQUEST_FRAME = 0x30
    REQUEST_MANIFEST = 0x31

    # Server → Client
    CONNECTED = 0x81
    MODEL_MANIFEST = 0x90
    PHYSICS_FRAME = 0x91
    STATUS_UPDATE = 0x92
    ERROR = 0x93
    PING = 0xA0
    PONG = 0xA1

@dataclass
class StreamingSession:
    """Individual streaming session"""
    session_id: str
    websocket: WebSocket
    runtime: Optional[Any] = None  # MuJoCoRuntime instance
    streaming: bool = False
    binary_mode: bool = True
    fps_target: int = 60
    last_frame_time: float = 0.0
    frame_count: int = 0
    dropped_frames: int = 0

class BinaryProtocol:
    """
    Binary protocol encoder/decoder
    All multi-byte values are little-endian
    """

    @staticmethod
    def encode_header(msg_type: MessageType, payload_size: int) -> bytes:
        """
        Encode message header
        Format: [type(u8), size(u32)]
        """
        return struct.pack('<BI', msg_type, payload_size)

    @staticmethod
    def decode_header(data: bytes) -> tuple[MessageType, int]:
        """Decode message header"""
        if len(data) < 5:
            raise ValueError("Invalid header size")
        msg_type, payload_size = struct.unpack('<BI', data[:5])
        return MessageType(msg_type), payload_size

    @staticmethod
    def encode_manifest(manifest: Dict[str, Any]) -> bytes:
        """
        Encode model manifest
        Format: [header][name_len(u16)][name][nbody(u32)][nq(u32)]...
        """
        parts = []

        # Model name
        name_bytes = manifest['model_name'].encode('utf-8')
        parts.append(struct.pack('<H', len(name_bytes)))
        parts.append(name_bytes)

        # Counts
        parts.append(struct.pack('<IIIIII',
            manifest['nbody'],
            manifest['nq'],
            manifest['nv'],
            manifest['nu'],
            manifest['nsensor'],
            len(manifest['geom_types'])
        ))

        # Timestep and gravity
        parts.append(struct.pack('<f', manifest['timestep']))
        parts.append(struct.pack('<fff', *manifest['gravity']))

        # Body names (length-prefixed strings)
        for name in manifest['body_names']:
            name_bytes = name.encode('utf-8')
            parts.append(struct.pack('<H', len(name_bytes)))
            parts.append(name_bytes)

        payload = b''.join(parts)
        header = BinaryProtocol.encode_header(MessageType.MODEL_MANIFEST, len(payload))
        return header + payload

    @staticmethod
    def encode_frame(frame_data: Dict[str, Any]) -> bytes:
        """
        Encode physics frame
        Format: [header][frame_id(u32)][sim_time(f32)][qpos][xpos][xquat]
        """
        parts = []

        # Frame metadata
        parts.append(struct.pack('<I', frame_data['frame_id']))
        parts.append(struct.pack('<f', frame_data['sim_time']))

        # Joint positions (variable length)
        qpos = np.array(frame_data['qpos'], dtype=np.float32)
        parts.append(struct.pack('<I', len(qpos)))  # Array length
        parts.append(qpos.tobytes())

        # Body positions (3D vectors)
        xpos = np.array(frame_data['xpos'], dtype=np.float32).flatten()
        parts.append(struct.pack('<I', len(xpos) // 3))  # Number of bodies
        parts.append(xpos.tobytes())

        # Body orientations (quaternions)
        xquat = np.array(frame_data['xquat'], dtype=np.float32).flatten()
        parts.append(xquat.tobytes())

        payload = b''.join(parts)
        header = BinaryProtocol.encode_header(MessageType.PHYSICS_FRAME, len(payload))
        return header + payload

    @staticmethod
    def decode_control(data: bytes) -> np.ndarray:
        """
        Decode control message
        Format: [n_actuators(u32)][values(f32[])]
        """
        if len(data) < 4:
            raise ValueError("Invalid control message")

        n_actuators = struct.unpack('<I', data[:4])[0]
        values = np.frombuffer(data[4:4 + n_actuators * 4], dtype=np.float32)
        return values

class StreamingManager:
    """
    Manages WebSocket streaming sessions
    """

    def __init__(self):
        self.sessions: Dict[str, StreamingSession] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    async def connect(self, session_id: str, websocket: WebSocket) -> StreamingSession:
        """Create new streaming session"""
        await websocket.accept()

        session = StreamingSession(
            session_id=session_id,
            websocket=websocket,
            streaming=False,
            binary_mode=True
        )

        self.sessions[session_id] = session

        # Send connection confirmation
        await self._send_status(session, "connected")

        logger.info(f"Streaming session connected: {session_id}")
        return session

    def disconnect(self, session_id: str):
        """Clean up streaming session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session.runtime:
                session.runtime.stop()
            del self.sessions[session_id]
            logger.info(f"Streaming session disconnected: {session_id}")

    async def handle_session(self, session: StreamingSession):
        """
        Main session handler
        Processes incoming messages and manages streaming
        """
        try:
            while True:
                # Receive message (binary or text)
                if session.binary_mode:
                    data = await session.websocket.receive_bytes()
                    await self._handle_binary_message(session, data)
                else:
                    data = await session.websocket.receive_json()
                    await self._handle_json_message(session, data)

        except WebSocketDisconnect:
            self.disconnect(session.session_id)
        except Exception as e:
            logger.error(f"Session error: {e}")
            await self._send_error(session, str(e))
            self.disconnect(session.session_id)

    async def _handle_binary_message(self, session: StreamingSession, data: bytes):
        """Process binary message"""
        if len(data) < 5:
            await self._send_error(session, "Invalid message format")
            return

        msg_type, payload_size = BinaryProtocol.decode_header(data)
        payload = data[5:5 + payload_size]

        if msg_type == MessageType.LOAD_MODEL:
            await self._handle_load_model(session, payload)
        elif msg_type == MessageType.START_SIM:
            await self._handle_start_sim(session)
        elif msg_type == MessageType.STOP_SIM:
            await self._handle_stop_sim(session)
        elif msg_type == MessageType.PAUSE_SIM:
            await self._handle_pause_sim(session)
        elif msg_type == MessageType.RESUME_SIM:
            await self._handle_resume_sim(session)
        elif msg_type == MessageType.RESET_SIM:
            await self._handle_reset_sim(session)
        elif msg_type == MessageType.SET_CONTROL:
            await self._handle_set_control(session, payload)
        elif msg_type == MessageType.REQUEST_MANIFEST:
            await self._send_manifest(session)
        elif msg_type == MessageType.PING:
            await session.websocket.send_bytes(
                BinaryProtocol.encode_header(MessageType.PONG, 0)
            )

    async def _handle_json_message(self, session: StreamingSession, data: Dict[str, Any]):
        """Process JSON message (fallback mode)"""
        msg_type = data.get('type')

        if msg_type == 'load_model':
            mjcf_xml = data.get('mjcf_xml')
            await self._handle_load_model(session, mjcf_xml.encode() if mjcf_xml else b'')
        elif msg_type == 'start_sim':
            await self._handle_start_sim(session)
        elif msg_type == 'stop_sim':
            await self._handle_stop_sim(session)
        elif msg_type == 'set_control':
            values = data.get('values', [])
            await self._handle_set_control(session, np.array(values, dtype=np.float32).tobytes())
        elif msg_type == 'request_manifest':
            await self._send_manifest(session)

    async def _handle_load_model(self, session: StreamingSession, payload: bytes):
        """Load MuJoCo model"""
        try:
            from .mujoco_runtime import MuJoCoRuntime

            # Decode MJCF XML
            mjcf_xml = payload.decode('utf-8')

            # Create runtime if needed
            if not session.runtime:
                session.runtime = MuJoCoRuntime(
                    render_fps=session.fps_target,
                    headless=True
                )

            # Load model
            manifest = session.runtime.load_mjcf(mjcf_xml)

            # Send manifest to client
            await self._send_manifest_data(session, manifest.to_dict())
            await self._send_status(session, "model_loaded")

        except Exception as e:
            await self._send_error(session, f"Failed to load model: {e}")

    async def _handle_start_sim(self, session: StreamingSession):
        """Start simulation streaming"""
        if not session.runtime:
            await self._send_error(session, "No model loaded")
            return

        if session.streaming:
            return  # Already streaming

        session.streaming = True

        # Frame callback for streaming
        async def stream_frame(frame):
            if session.binary_mode:
                frame_bytes = BinaryProtocol.encode_frame(frame.to_dict())
                await session.websocket.send_bytes(frame_bytes)
            else:
                await session.websocket.send_json({
                    'type': 'frame',
                    'data': frame.to_dict()
                })
            session.frame_count += 1

        # Start async simulation
        session.runtime.frame_callback = stream_frame
        asyncio.create_task(session.runtime.run_async())

        await self._send_status(session, "simulation_started")

    async def _handle_stop_sim(self, session: StreamingSession):
        """Stop simulation"""
        if session.runtime:
            session.runtime.stop()
            session.streaming = False
            await self._send_status(session, "simulation_stopped")

    async def _handle_pause_sim(self, session: StreamingSession):
        """Pause simulation"""
        if session.runtime:
            session.runtime.pause()
            await self._send_status(session, "simulation_paused")

    async def _handle_resume_sim(self, session: StreamingSession):
        """Resume simulation"""
        if session.runtime:
            session.runtime.resume()
            await self._send_status(session, "simulation_resumed")

    async def _handle_reset_sim(self, session: StreamingSession):
        """Reset simulation"""
        if session.runtime:
            session.runtime.reset()
            session.frame_count = 0
            await self._send_status(session, "simulation_reset")

    async def _handle_set_control(self, session: StreamingSession, payload: bytes):
        """Set actuator controls"""
        if not session.runtime:
            return

        try:
            values = BinaryProtocol.decode_control(payload)
            session.runtime.set_control(values)
        except Exception as e:
            await self._send_error(session, f"Invalid control data: {e}")

    async def _send_manifest(self, session: StreamingSession):
        """Send model manifest"""
        if not session.runtime or not session.runtime.model:
            await self._send_error(session, "No model loaded")
            return

        manifest = session.runtime._create_manifest()
        await self._send_manifest_data(session, manifest.to_dict())

    async def _send_manifest_data(self, session: StreamingSession, manifest: Dict[str, Any]):
        """Send manifest data"""
        if session.binary_mode:
            manifest_bytes = BinaryProtocol.encode_manifest(manifest)
            await session.websocket.send_bytes(manifest_bytes)
        else:
            await session.websocket.send_json({
                'type': 'manifest',
                'data': manifest
            })

    async def _send_status(self, session: StreamingSession, status: str):
        """Send status update"""
        if session.binary_mode:
            status_bytes = status.encode('utf-8')
            header = BinaryProtocol.encode_header(MessageType.STATUS_UPDATE, len(status_bytes))
            await session.websocket.send_bytes(header + status_bytes)
        else:
            await session.websocket.send_json({
                'type': 'status',
                'status': status
            })

    async def _send_error(self, session: StreamingSession, error: str):
        """Send error message"""
        logger.error(f"Sending error to {session.session_id}: {error}")

        if session.binary_mode:
            error_bytes = error.encode('utf-8')
            header = BinaryProtocol.encode_header(MessageType.ERROR, len(error_bytes))
            await session.websocket.send_bytes(header + error_bytes)
        else:
            await session.websocket.send_json({
                'type': 'error',
                'error': error
            })

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get session statistics"""
        if session_id not in self.sessions:
            return {}

        session = self.sessions[session_id]
        return {
            'session_id': session_id,
            'streaming': session.streaming,
            'binary_mode': session.binary_mode,
            'frame_count': session.frame_count,
            'dropped_frames': session.dropped_frames,
            'fps_target': session.fps_target
        }

# Global streaming manager instance
streaming_manager = StreamingManager()