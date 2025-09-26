"""
Redis-based WebSocket session management for horizontal scaling.
Allows WebSocket connections to be handled by multiple backend instances.
"""

import json
import uuid
import asyncio
import logging
from typing import Dict, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import redis.asyncio as redis
from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class WebSocketSession:
    """WebSocket session data stored in Redis."""
    session_id: str
    client_id: str
    server_id: str
    connected_at: str
    last_heartbeat: str
    metadata: Dict[str, Any]
    active_simulation: Optional[str] = None


class RedisWebSocketManager:
    """
    Manages WebSocket sessions across multiple server instances using Redis.

    Features:
    - Session sharing across servers
    - Automatic failover
    - Health checks and cleanup
    - Message routing between servers
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        server_id: str = None,
        heartbeat_interval: int = 30,
        session_ttl: int = 3600
    ):
        self.redis = redis_client
        self.server_id = server_id or f"server-{uuid.uuid4().hex[:8]}"
        self.heartbeat_interval = heartbeat_interval
        self.session_ttl = session_ttl
        self.local_connections: Dict[str, WebSocket] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._pubsub_task: Optional[asyncio.Task] = None
        self._pubsub: Optional[redis.client.PubSub] = None

    async def start(self):
        """Start the session manager background tasks."""
        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Start pub/sub listener for cross-server messaging
        self._pubsub = self.redis.pubsub()
        await self._pubsub.subscribe(f"server:{self.server_id}")
        self._pubsub_task = asyncio.create_task(self._pubsub_loop())

        logger.info(f"WebSocket session manager started: {self.server_id}")

    async def stop(self):
        """Stop the session manager and cleanup."""
        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._pubsub_task:
            self._pubsub_task.cancel()

        # Unsubscribe from pub/sub
        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()

        # Mark all local sessions as disconnected
        for session_id in list(self.local_connections.keys()):
            await self.disconnect_session(session_id)

        logger.info(f"WebSocket session manager stopped: {self.server_id}")

    async def connect_session(
        self,
        websocket: WebSocket,
        client_id: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            client_id: Client identifier
            metadata: Additional session metadata

        Returns:
            Session ID
        """
        session_id = f"ws-{uuid.uuid4().hex}"

        # Store in local connections
        self.local_connections[session_id] = websocket

        # Create session data
        session = WebSocketSession(
            session_id=session_id,
            client_id=client_id,
            server_id=self.server_id,
            connected_at=datetime.utcnow().isoformat(),
            last_heartbeat=datetime.utcnow().isoformat(),
            metadata=metadata or {}
        )

        # Store in Redis
        session_key = f"session:{session_id}"
        await self.redis.setex(
            session_key,
            self.session_ttl,
            json.dumps(asdict(session))
        )

        # Add to server's session set
        await self.redis.sadd(f"server:{self.server_id}:sessions", session_id)

        # Add to global session set
        await self.redis.sadd("sessions:all", session_id)

        logger.info(f"Connected session {session_id} for client {client_id}")
        return session_id

    async def disconnect_session(self, session_id: str):
        """Disconnect and cleanup a session."""
        # Remove from local connections
        if session_id in self.local_connections:
            del self.local_connections[session_id]

        # Remove from Redis
        await self.redis.delete(f"session:{session_id}")
        await self.redis.srem(f"server:{self.server_id}:sessions", session_id)
        await self.redis.srem("sessions:all", session_id)

        logger.info(f"Disconnected session {session_id}")

    async def get_session(self, session_id: str) -> Optional[WebSocketSession]:
        """Get session data from Redis."""
        session_data = await self.redis.get(f"session:{session_id}")
        if session_data:
            data = json.loads(session_data)
            return WebSocketSession(**data)
        return None

    async def update_session(
        self,
        session_id: str,
        active_simulation: Optional[str] = None,
        metadata_update: Optional[Dict[str, Any]] = None
    ):
        """Update session data in Redis."""
        session = await self.get_session(session_id)
        if not session:
            return

        if active_simulation is not None:
            session.active_simulation = active_simulation

        if metadata_update:
            session.metadata.update(metadata_update)

        session.last_heartbeat = datetime.utcnow().isoformat()

        # Update in Redis
        await self.redis.setex(
            f"session:{session_id}",
            self.session_ttl,
            json.dumps(asdict(session))
        )

    async def send_to_session(
        self,
        session_id: str,
        message: Dict[str, Any],
        ensure_delivery: bool = False
    ):
        """
        Send message to a session (might be on different server).

        Args:
            session_id: Target session ID
            message: Message to send
            ensure_delivery: Whether to store message if session offline
        """
        # Check if session is local
        if session_id in self.local_connections:
            websocket = self.local_connections[session_id]
            try:
                await websocket.send_json(message)
                return True
            except Exception as e:
                logger.error(f"Failed to send to local session {session_id}: {e}")
                await self.disconnect_session(session_id)
                return False

        # Session is on another server, use pub/sub
        session = await self.get_session(session_id)
        if session:
            # Publish to the server handling this session
            channel = f"server:{session.server_id}"
            message_data = {
                "session_id": session_id,
                "message": message
            }
            await self.redis.publish(channel, json.dumps(message_data))
            return True

        # Session not found
        if ensure_delivery:
            # Store message for later delivery
            await self.redis.lpush(
                f"pending:{session_id}",
                json.dumps(message)
            )
            await self.redis.expire(f"pending:{session_id}", 300)  # 5 min TTL

        return False

    async def broadcast_to_simulation(
        self,
        simulation_id: str,
        message: Dict[str, Any]
    ):
        """Broadcast message to all sessions watching a simulation."""
        # Get all sessions
        all_sessions = await self.redis.smembers("sessions:all")

        for session_id in all_sessions:
            session = await self.get_session(session_id)
            if session and session.active_simulation == simulation_id:
                await self.send_to_session(session_id, message)

    async def get_server_stats(self) -> Dict[str, Any]:
        """Get statistics about sessions and servers."""
        # Get all servers
        servers = set()
        all_sessions = await self.redis.smembers("sessions:all")

        for session_id in all_sessions:
            session = await self.get_session(session_id)
            if session:
                servers.add(session.server_id)

        # Get local stats
        local_sessions = await self.redis.scard(f"server:{self.server_id}:sessions")

        return {
            "server_id": self.server_id,
            "total_servers": len(servers),
            "total_sessions": len(all_sessions),
            "local_sessions": local_sessions,
            "servers": list(servers)
        }

    async def _heartbeat_loop(self):
        """Periodically update heartbeat for local sessions."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # Update heartbeat for all local sessions
                for session_id in list(self.local_connections.keys()):
                    session = await self.get_session(session_id)
                    if session:
                        session.last_heartbeat = datetime.utcnow().isoformat()
                        await self.redis.setex(
                            f"session:{session_id}",
                            self.session_ttl,
                            json.dumps(asdict(session))
                        )

                # Clean up stale sessions (no heartbeat for 2x interval)
                stale_threshold = datetime.utcnow() - timedelta(
                    seconds=self.heartbeat_interval * 2
                )

                all_sessions = await self.redis.smembers("sessions:all")
                for session_id in all_sessions:
                    session = await self.get_session(session_id)
                    if session:
                        last_heartbeat = datetime.fromisoformat(session.last_heartbeat)
                        if last_heartbeat < stale_threshold:
                            logger.warning(f"Cleaning up stale session: {session_id}")
                            await self.redis.delete(f"session:{session_id}")
                            await self.redis.srem("sessions:all", session_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    async def _pubsub_loop(self):
        """Listen for messages from other servers."""
        while True:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1
                )

                if message and message['type'] == 'message':
                    data = json.loads(message['data'])
                    session_id = data['session_id']
                    msg = data['message']

                    # If we have this session locally, send the message
                    if session_id in self.local_connections:
                        websocket = self.local_connections[session_id]
                        try:
                            await websocket.send_json(msg)
                        except Exception as e:
                            logger.error(f"Failed to relay message to {session_id}: {e}")
                            await self.disconnect_session(session_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in pubsub loop: {e}")

    async def handle_failover(self, failed_server_id: str):
        """
        Handle sessions from a failed server.

        This would be called by a health check system when a server fails.
        """
        logger.info(f"Handling failover for server: {failed_server_id}")

        # Get sessions from failed server
        failed_sessions = await self.redis.smembers(f"server:{failed_server_id}:sessions")

        if not failed_sessions:
            return

        # Redistribute sessions among active servers
        # In a real system, you'd have better load balancing logic
        for session_id in failed_sessions:
            session = await self.get_session(session_id)
            if session:
                # Update session to this server
                session.server_id = self.server_id
                await self.redis.setex(
                    f"session:{session_id}",
                    self.session_ttl,
                    json.dumps(asdict(session))
                )

                # Add to our server's session set
                await self.redis.sadd(f"server:{self.server_id}:sessions", session_id)

        # Clean up failed server's data
        await self.redis.delete(f"server:{failed_server_id}:sessions")

        logger.info(f"Migrated {len(failed_sessions)} sessions from failed server")


# Global instance (initialized in startup)
websocket_manager: Optional[RedisWebSocketManager] = None


async def get_websocket_manager() -> RedisWebSocketManager:
    """Get the global WebSocket manager instance."""
    global websocket_manager
    if not websocket_manager:
        # Initialize Redis connection
        redis_client = redis.from_url("redis://localhost:6379/1")
        websocket_manager = RedisWebSocketManager(redis_client)
        await websocket_manager.start()
    return websocket_manager