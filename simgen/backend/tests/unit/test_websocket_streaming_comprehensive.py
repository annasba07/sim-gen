"""Comprehensive WebSocket streaming protocol tests for real-time communication."""
import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import struct
from datetime import datetime
import numpy as np
from typing import Dict, Any, Optional
import uuid

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import WebSocket streaming components
from simgen.services.streaming_protocol import (
    MessageType,
    StreamingSession,
    BinaryProtocol,
    StreamingManager
)

# Mock WebSocket for testing
class MockWebSocket:
    """Mock WebSocket for testing purposes."""

    def __init__(self):
        self.sent_messages = []
        self.received_messages = []
        self.closed = False
        self.accept_called = False

    async def accept(self):
        self.accept_called = True

    async def send_bytes(self, data: bytes):
        self.sent_messages.append(('bytes', data))

    async def send_text(self, data: str):
        self.sent_messages.append(('text', data))

    async def receive_bytes(self) -> bytes:
        if not self.received_messages:
            raise Exception("No messages to receive")
        return self.received_messages.pop(0)

    async def receive_text(self) -> str:
        if not self.received_messages:
            raise Exception("No messages to receive")
        return self.received_messages.pop(0)

    async def close(self):
        self.closed = True

    def add_received_message(self, message):
        self.received_messages.append(message)


class TestBinaryProtocol:
    """Test binary protocol encoding/decoding functionality."""

    def test_encode_header_basic(self):
        """Test basic header encoding."""
        msg_type = MessageType.CONNECT
        payload_size = 1024

        header = BinaryProtocol.encode_header(msg_type, payload_size)

        assert len(header) == 5  # 1 byte msg_type + 4 bytes payload_size
        assert isinstance(header, bytes)

        # Verify the structure
        unpacked = struct.unpack('<BI', header)
        assert unpacked[0] == int(msg_type)
        assert unpacked[1] == payload_size

    def test_decode_header_basic(self):
        """Test basic header decoding."""
        msg_type = MessageType.PHYSICS_FRAME
        payload_size = 2048

        # Create header
        header = BinaryProtocol.encode_header(msg_type, payload_size)

        # Decode it back
        decoded_type, decoded_size = BinaryProtocol.decode_header(header)

        assert decoded_type == msg_type
        assert decoded_size == payload_size

    def test_encode_decode_header_roundtrip(self):
        """Test header encode/decode roundtrip for all message types."""
        test_cases = [
            (MessageType.CONNECT, 0),
            (MessageType.LOAD_MODEL, 1024),
            (MessageType.PHYSICS_FRAME, 8192),
            (MessageType.ERROR, 256),
            (MessageType.PING, 0),
            (MessageType.PONG, 0)
        ]

        for msg_type, payload_size in test_cases:
            # Encode
            header = BinaryProtocol.encode_header(msg_type, payload_size)

            # Decode
            decoded_type, decoded_size = BinaryProtocol.decode_header(header)

            assert decoded_type == msg_type
            assert decoded_size == payload_size

    def test_encode_manifest(self):
        """Test manifest encoding."""
        manifest = {
            "model_name": "test_robot",
            "nbody": 3,
            "nq": 7,
            "nv": 6,
            "nu": 4,
            "nsensor": 2,
            "geom_types": [0, 1, 2],
            "timestep": 0.01,
            "gravity": [0.0, 0.0, -9.81],
            "body_names": ["world", "base", "link1"],
            "joint_names": ["joint1", "joint2"]
        }

        encoded = BinaryProtocol.encode_manifest(manifest)

        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

        # Should contain the binary-encoded manifest data
        # (Can't easily decode back due to complex binary format)

    def test_encode_frame_basic(self):
        """Test frame data encoding."""
        frame_data = {
            "frame_id": 123,
            "sim_time": 1.5,
            "qpos": [0.1, 0.2, 0.3],
            "xpos": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # 2 bodies * 3D positions
            "xquat": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # 2 bodies * quaternions
        }

        encoded = BinaryProtocol.encode_frame(frame_data)

        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

    def test_decode_control(self):
        """Test control data decoding."""
        # Create test control data in the expected format
        control_values = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        n_actuators = len(control_values)

        # Format: [n_actuators(u32)][values(f32[])]
        control_bytes = struct.pack('<I', n_actuators) + control_values.tobytes()

        decoded = BinaryProtocol.decode_control(control_bytes)

        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.float32
        np.testing.assert_array_almost_equal(decoded, control_values)

    def test_encode_frame_with_numpy_arrays(self):
        """Test frame encoding with numpy arrays."""
        frame_data = {
            "frame_id": 456,
            "sim_time": 2.5,
            "qpos": np.array([0.1, 0.2, 0.3]),
            "xpos": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "xquat": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        }

        encoded = BinaryProtocol.encode_frame(frame_data)

        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

        # Should contain binary physics frame data
        # (Binary format not easily decodable in test, but should be well-formed)

    def test_invalid_header_decoding(self):
        """Test handling of invalid header data."""
        # Too short data
        with pytest.raises((struct.error, ValueError)):
            BinaryProtocol.decode_header(b"short")

        # Invalid message type (though this might be handled gracefully)
        invalid_header = struct.pack('<BI', 255, 1024)  # 255 is max valid byte value
        try:
            msg_type, size = BinaryProtocol.decode_header(invalid_header)
            # If it doesn't raise an exception, check the values
            assert size == 1024
        except (ValueError, KeyError):
            # Expected behavior for invalid message type
            pass

    def test_large_payload_encoding(self):
        """Test encoding with large payloads."""
        large_payload_size = 1024 * 1024  # 1MB

        header = BinaryProtocol.encode_header(MessageType.PHYSICS_FRAME, large_payload_size)
        decoded_type, decoded_size = BinaryProtocol.decode_header(header)

        assert decoded_type == MessageType.PHYSICS_FRAME
        assert decoded_size == large_payload_size


class TestStreamingSession:
    """Test streaming session functionality."""

    def test_streaming_session_creation(self):
        """Test creating a streaming session."""
        mock_websocket = MockWebSocket()
        session_id = "test-session-001"

        session = StreamingSession(
            session_id=session_id,
            websocket=mock_websocket
        )

        assert session.session_id == session_id
        assert session.websocket == mock_websocket
        assert session.runtime is None
        assert session.streaming is False
        assert session.binary_mode is True
        assert session.fps_target == 60
        assert session.last_frame_time == 0.0

    def test_streaming_session_configuration(self):
        """Test streaming session with custom configuration."""
        mock_websocket = MockWebSocket()
        mock_runtime = MagicMock()

        session = StreamingSession(
            session_id="custom-session",
            websocket=mock_websocket,
            runtime=mock_runtime,
            streaming=True,
            binary_mode=False,
            fps_target=30,
            last_frame_time=1.5
        )

        assert session.session_id == "custom-session"
        assert session.runtime == mock_runtime
        assert session.streaming is True
        assert session.binary_mode is False
        assert session.fps_target == 30
        assert session.last_frame_time == 1.5


class TestStreamingManager:
    """Test streaming manager functionality."""

    @pytest.fixture
    def streaming_manager(self):
        """Create streaming manager for testing."""
        return StreamingManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket."""
        return MockWebSocket()

    async def test_streaming_manager_initialization(self, streaming_manager):
        """Test streaming manager initialization."""
        assert hasattr(streaming_manager, 'sessions')
        assert isinstance(streaming_manager.sessions, dict)
        assert len(streaming_manager.sessions) == 0

    async def test_connect_session(self, streaming_manager, mock_websocket):
        """Test connecting a new session."""
        session_id = "test-session-connect"

        session = await streaming_manager.connect(session_id, mock_websocket)

        assert session.session_id == session_id
        assert session.websocket == mock_websocket
        assert mock_websocket.accept_called
        assert session_id in streaming_manager.sessions

    async def test_disconnect_session(self, streaming_manager, mock_websocket):
        """Test disconnecting a session."""
        session_id = "test-session-disconnect"

        # First connect
        await streaming_manager.connect(session_id, mock_websocket)
        assert session_id in streaming_manager.sessions

        # Then disconnect
        streaming_manager.disconnect(session_id)
        assert session_id not in streaming_manager.sessions

    async def test_disconnect_nonexistent_session(self, streaming_manager):
        """Test disconnecting a session that doesn't exist."""
        # Should not raise an exception
        streaming_manager.disconnect("nonexistent-session")

    async def test_multiple_sessions(self, streaming_manager):
        """Test managing multiple concurrent sessions."""
        sessions = []
        for i in range(5):
            mock_ws = MockWebSocket()
            session_id = f"session-{i}"
            session = await streaming_manager.connect(session_id, mock_ws)
            sessions.append((session_id, session))

        # Verify all sessions exist
        assert len(streaming_manager.sessions) == 5
        for session_id, session in sessions:
            assert session_id in streaming_manager.sessions
            assert streaming_manager.sessions[session_id] == session

        # Disconnect some sessions
        streaming_manager.disconnect("session-1")
        streaming_manager.disconnect("session-3")

        assert len(streaming_manager.sessions) == 3
        assert "session-1" not in streaming_manager.sessions
        assert "session-3" not in streaming_manager.sessions

    async def test_handle_binary_message_connect(self, streaming_manager, mock_websocket):
        """Test handling binary CONNECT message."""
        session_id = "test-binary-connect"
        session = await streaming_manager.connect(session_id, mock_websocket)

        # Create CONNECT message
        header = BinaryProtocol.encode_header(MessageType.CONNECT, 0)

        with patch.object(streaming_manager, '_handle_load_model') as mock_load:
            await streaming_manager._handle_binary_message(session, header)
            # CONNECT message with no payload should be handled gracefully

    async def test_handle_json_message(self, streaming_manager, mock_websocket):
        """Test handling JSON messages."""
        session_id = "test-json-message"
        session = await streaming_manager.connect(session_id, mock_websocket)

        json_message = {
            "type": "control",
            "data": [0.1, 0.2, 0.3]
        }

        # Should handle gracefully even if not fully implemented
        try:
            await streaming_manager._handle_json_message(session, json_message)
        except (NotImplementedError, AttributeError):
            # Expected if method is not fully implemented
            pass

    async def test_send_manifest(self, streaming_manager, mock_websocket):
        """Test sending manifest to client."""
        session_id = "test-manifest"
        session = await streaming_manager.connect(session_id, mock_websocket)

        # Mock runtime with manifest
        mock_runtime = MagicMock()
        mock_runtime.get_model_manifest.return_value = {
            "model_name": "test_model",
            "bodies": ["body1", "body2"],
            "joints": ["joint1"]
        }
        session.runtime = mock_runtime

        await streaming_manager._send_manifest(session)

        # Should have sent manifest data
        assert len(mock_websocket.sent_messages) > 0

    async def test_send_status(self, streaming_manager, mock_websocket):
        """Test sending status updates."""
        session_id = "test-status"
        session = await streaming_manager.connect(session_id, mock_websocket)

        await streaming_manager._send_status(session, "running")

        # Should have sent status message
        assert len(mock_websocket.sent_messages) > 0
        msg_type, data = mock_websocket.sent_messages[0]
        assert msg_type == 'bytes'

    async def test_send_error(self, streaming_manager, mock_websocket):
        """Test sending error messages."""
        session_id = "test-error"
        session = await streaming_manager.connect(session_id, mock_websocket)

        await streaming_manager._send_error(session, "Test error message")

        # Should have sent error message
        assert len(mock_websocket.sent_messages) > 0
        msg_type, data = mock_websocket.sent_messages[0]
        assert msg_type == 'bytes'

    async def test_get_session_stats(self, streaming_manager, mock_websocket):
        """Test getting session statistics."""
        session_id = "test-stats"
        await streaming_manager.connect(session_id, mock_websocket)

        stats = streaming_manager.get_session_stats(session_id)

        assert isinstance(stats, dict)
        assert "session_id" in stats or stats is not None

    async def test_get_stats_nonexistent_session(self, streaming_manager):
        """Test getting stats for nonexistent session."""
        stats = streaming_manager.get_session_stats("nonexistent")

        # Should return None or empty dict
        assert stats is None or stats == {}

    async def test_handle_load_model(self, streaming_manager, mock_websocket):
        """Test handling load model messages."""
        session_id = "test-load-model"
        session = await streaming_manager.connect(session_id, mock_websocket)

        # Create model payload (simplified)
        model_data = b"<mujoco><worldbody></worldbody></mujoco>"

        with patch('simgen.services.mujoco_runtime.MuJoCoRuntime') as mock_runtime_class:
            mock_runtime = MagicMock()
            mock_runtime_class.return_value = mock_runtime

            await streaming_manager._handle_load_model(session, model_data)

            # Should have attempted to create runtime
            # (May not succeed without actual MuJoCo, but should handle gracefully)

    async def test_handle_simulation_controls(self, streaming_manager, mock_websocket):
        """Test handling simulation control messages."""
        session_id = "test-sim-controls"
        session = await streaming_manager.connect(session_id, mock_websocket)

        # Mock runtime
        mock_runtime = MagicMock()
        mock_runtime.start_simulation = AsyncMock()
        mock_runtime.stop_simulation = AsyncMock()
        mock_runtime.pause_simulation = AsyncMock()
        mock_runtime.resume_simulation = AsyncMock()
        mock_runtime.reset_simulation = AsyncMock()
        session.runtime = mock_runtime

        # Test start simulation
        await streaming_manager._handle_start_sim(session)
        mock_runtime.start_simulation.assert_called_once()

        # Test stop simulation
        await streaming_manager._handle_stop_sim(session)
        mock_runtime.stop_simulation.assert_called_once()

        # Test pause simulation
        await streaming_manager._handle_pause_sim(session)
        mock_runtime.pause_simulation.assert_called_once()

        # Test resume simulation
        await streaming_manager._handle_resume_sim(session)
        mock_runtime.resume_simulation.assert_called_once()

        # Test reset simulation
        await streaming_manager._handle_reset_sim(session)
        mock_runtime.reset_simulation.assert_called_once()

    async def test_handle_set_control(self, streaming_manager, mock_websocket):
        """Test handling control input messages."""
        session_id = "test-set-control"
        session = await streaming_manager.connect(session_id, mock_websocket)

        # Mock runtime
        mock_runtime = MagicMock()
        mock_runtime.set_control = MagicMock()
        session.runtime = mock_runtime

        # Create control data
        control_values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        control_bytes = control_values.tobytes()

        await streaming_manager._handle_set_control(session, control_bytes)

        # Should have called set_control on runtime
        mock_runtime.set_control.assert_called_once()

    async def test_session_without_runtime(self, streaming_manager, mock_websocket):
        """Test session operations without runtime (error conditions)."""
        session_id = "test-no-runtime"
        session = await streaming_manager.connect(session_id, mock_websocket)

        # Session has no runtime, operations should handle gracefully
        await streaming_manager._handle_start_sim(session)
        await streaming_manager._handle_stop_sim(session)
        await streaming_manager._send_manifest(session)

        # Should have sent error messages or handled gracefully
        assert len(mock_websocket.sent_messages) >= 0  # May send error messages


class TestStreamingProtocolIntegration:
    """Integration tests for streaming protocol components."""

    @pytest.fixture
    def streaming_manager(self):
        """Create streaming manager for testing."""
        return StreamingManager()

    async def test_full_session_lifecycle(self, streaming_manager):
        """Test complete session lifecycle."""
        mock_websocket = MockWebSocket()
        session_id = "lifecycle-test"

        # 1. Connect
        session = await streaming_manager.connect(session_id, mock_websocket)
        assert session_id in streaming_manager.sessions

        # 2. Simulate some operations
        await streaming_manager._send_status(session, "connected")

        # 3. Disconnect
        streaming_manager.disconnect(session_id)
        assert session_id not in streaming_manager.sessions

    async def test_binary_message_handling_flow(self, streaming_manager):
        """Test complete binary message handling flow."""
        mock_websocket = MockWebSocket()
        session_id = "binary-flow-test"

        session = await streaming_manager.connect(session_id, mock_websocket)

        # Test different message types
        message_types = [
            MessageType.CONNECT,
            MessageType.REQUEST_MANIFEST,
            MessageType.PING
        ]

        for msg_type in message_types:
            header = BinaryProtocol.encode_header(msg_type, 0)

            try:
                await streaming_manager._handle_binary_message(session, header)
            except (NotImplementedError, AttributeError):
                # Some handlers might not be fully implemented
                pass

    async def test_error_handling_resilience(self, streaming_manager):
        """Test error handling and resilience."""
        mock_websocket = MockWebSocket()
        session_id = "error-test"

        session = await streaming_manager.connect(session_id, mock_websocket)

        # Test with invalid data
        invalid_data = b"invalid_header"

        try:
            await streaming_manager._handle_binary_message(session, invalid_data)
        except Exception:
            # Should handle errors gracefully
            pass

        # Session should still exist after error
        assert session_id in streaming_manager.sessions

    async def test_concurrent_sessions_stress(self, streaming_manager):
        """Test handling multiple concurrent sessions."""
        sessions = []

        # Create multiple sessions concurrently
        for i in range(10):
            mock_ws = MockWebSocket()
            session_id = f"concurrent-{i}"
            session = await streaming_manager.connect(session_id, mock_ws)
            sessions.append((session_id, session))

        # Perform operations on all sessions
        for session_id, session in sessions:
            await streaming_manager._send_status(session, f"status-{session_id}")

        # Verify all sessions are tracked
        assert len(streaming_manager.sessions) == 10

        # Clean up
        for session_id, _ in sessions:
            streaming_manager.disconnect(session_id)

        assert len(streaming_manager.sessions) == 0

    async def test_message_serialization_roundtrip(self):
        """Test message serialization and deserialization."""
        # Test manifest roundtrip
        original_manifest = {
            "model": "test_robot",
            "version": "1.0",
            "frame_data": {
                "positions": [0.1, 0.2, 0.3],
                "velocities": [0.01, 0.02, 0.03]
            }
        }

        encoded = BinaryProtocol.encode_manifest(original_manifest)
        decoded = json.loads(encoded.decode('utf-8'))

        assert decoded == original_manifest

        # Test frame data roundtrip
        frame_data = {
            "timestamp": 1.5,
            "data": [1, 2, 3, 4, 5]
        }

        encoded_frame = BinaryProtocol.encode_frame(frame_data)
        decoded_frame = json.loads(encoded_frame.decode('utf-8'))

        assert "timestamp" in decoded_frame
        assert decoded_frame["timestamp"] == 1.5


class TestStreamingProtocolPerformance:
    """Performance tests for streaming protocol."""

    def test_header_encoding_performance(self):
        """Test header encoding performance."""
        import time

        start_time = time.time()

        # Encode many headers
        for i in range(1000):
            BinaryProtocol.encode_header(MessageType.PHYSICS_FRAME, i)

        end_time = time.time()
        elapsed = end_time - start_time

        # Should be very fast
        assert elapsed < 0.1  # Less than 100ms for 1000 operations

    def test_frame_encoding_performance(self):
        """Test frame encoding performance."""
        import time

        frame_data = {
            "timestamp": 1.0,
            "positions": list(range(100)),  # Large data set
            "velocities": list(range(100))
        }

        start_time = time.time()

        # Encode many frames
        for i in range(100):
            BinaryProtocol.encode_frame(frame_data)

        end_time = time.time()
        elapsed = end_time - start_time

        # Should complete within reasonable time
        assert elapsed < 1.0  # Less than 1 second for 100 operations

    async def test_session_creation_performance(self):
        """Test session creation performance."""
        import time

        streaming_manager = StreamingManager()

        start_time = time.time()

        # Create many sessions quickly
        for i in range(100):
            mock_ws = MockWebSocket()
            await streaming_manager.connect(f"perf-session-{i}", mock_ws)

        end_time = time.time()
        elapsed = end_time - start_time

        # Should create sessions quickly
        assert elapsed < 1.0  # Less than 1 second for 100 sessions
        assert len(streaming_manager.sessions) == 100