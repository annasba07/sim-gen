"""
Real-time progress streaming via WebSockets
Provides live updates during sketch-to-physics generation
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ProgressStage(Enum):
    """Stages of the sketch-to-physics pipeline."""
    INITIALIZING = "initializing"
    DECODING_IMAGE = "decoding_image"
    VISION_ANALYSIS = "vision_analysis"
    TEXT_PROCESSING = "text_processing"
    MULTIMODAL_ENHANCEMENT = "multimodal_enhancement"
    ENTITY_EXTRACTION = "entity_extraction"
    MJCF_GENERATION = "mjcf_generation"
    SIMULATION_VALIDATION = "simulation_validation"
    RENDERING_PREVIEW = "rendering_preview"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressUpdate:
    """Progress update data structure."""
    session_id: str
    stage: ProgressStage
    progress_percent: float
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    elapsed_time: Optional[float] = None
    estimated_remaining: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class StageMetrics:
    """Metrics for individual pipeline stages."""
    stage: ProgressStage
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def complete(self, success: bool = True, error_message: str = None):
        """Mark stage as completed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message


class RealTimeProgressTracker:
    """Tracks and broadcasts real-time progress updates via WebSocket."""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.websocket_connections: Dict[str, Any] = {}
        self.stage_progress_weights = {
            ProgressStage.INITIALIZING: 5,
            ProgressStage.DECODING_IMAGE: 5,
            ProgressStage.VISION_ANALYSIS: 25,
            ProgressStage.TEXT_PROCESSING: 15,
            ProgressStage.MULTIMODAL_ENHANCEMENT: 20,
            ProgressStage.ENTITY_EXTRACTION: 10,
            ProgressStage.MJCF_GENERATION: 15,
            ProgressStage.SIMULATION_VALIDATION: 3,
            ProgressStage.RENDERING_PREVIEW: 2,
            ProgressStage.COMPLETED: 0
        }
    
    def start_session(
        self,
        session_id: str,
        total_stages: Optional[List[ProgressStage]] = None,
        websocket=None
    ):
        """Start tracking a new session."""
        if total_stages is None:
            total_stages = list(ProgressStage)
        
        self.active_sessions[session_id] = {
            "start_time": time.time(),
            "current_stage": ProgressStage.INITIALIZING,
            "stages_completed": [],
            "stage_metrics": {},
            "total_stages": total_stages,
            "websocket": websocket
        }
        
        if websocket:
            self.websocket_connections[session_id] = websocket
        
        logger.info(f"Started progress tracking for session {session_id}")
        
        # Send initial progress update
        asyncio.create_task(self.update_progress(
            session_id,
            ProgressStage.INITIALIZING,
            0,
            "Starting sketch-to-physics generation..."
        ))
    
    async def update_progress(
        self,
        session_id: str,
        stage: ProgressStage,
        stage_progress: float,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Update progress for a session and broadcast via WebSocket."""
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found for progress update")
            return
        
        session = self.active_sessions[session_id]
        
        # Calculate overall progress
        overall_progress = self._calculate_overall_progress(session_id, stage, stage_progress)
        
        # Calculate time estimates
        elapsed_time = time.time() - session["start_time"]
        estimated_remaining = self._estimate_remaining_time(overall_progress, elapsed_time)
        
        # Create progress update
        progress_update = ProgressUpdate(
            session_id=session_id,
            stage=stage,
            progress_percent=overall_progress,
            message=message,
            details=details,
            elapsed_time=elapsed_time,
            estimated_remaining=estimated_remaining
        )
        
        # Update session state
        session["current_stage"] = stage
        session["last_update"] = progress_update
        
        # Broadcast to WebSocket if connected
        await self._broadcast_progress(session_id, progress_update)
        
        logger.debug(f"Progress update: {session_id} - {stage.value} - {overall_progress:.1f}%")
    
    async def start_stage(self, session_id: str, stage: ProgressStage, message: str = None):
        """Mark the start of a pipeline stage."""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # Complete previous stage if exists
        if session["current_stage"] != ProgressStage.INITIALIZING:
            await self.complete_stage(session_id, session["current_stage"])
        
        # Start new stage
        stage_metrics = StageMetrics(stage=stage, start_time=time.time())
        session["stage_metrics"][stage] = stage_metrics
        
        default_message = f"Starting {stage.value.replace('_', ' ').title()}..."
        await self.update_progress(session_id, stage, 0, message or default_message)
    
    async def complete_stage(
        self,
        session_id: str,
        stage: ProgressStage,
        success: bool = True,
        error_message: str = None
    ):
        """Mark a stage as completed."""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        if stage in session["stage_metrics"]:
            stage_metrics = session["stage_metrics"][stage]
            stage_metrics.complete(success, error_message)
            
            if success:
                session["stages_completed"].append(stage)
                message = f"Completed {stage.value.replace('_', ' ').title()}"
            else:
                message = f"Failed {stage.value.replace('_', ' ').title()}: {error_message}"
            
            await self.update_progress(session_id, stage, 100, message)
    
    async def complete_session(
        self,
        session_id: str,
        success: bool = True,
        final_message: str = None,
        result_data: Optional[Dict[str, Any]] = None
    ):
        """Complete a session with final results."""
        if session_id not in self.active_sessions:
            return
        
        # Complete current stage
        session = self.active_sessions[session_id]
        current_stage = session["current_stage"]
        await self.complete_stage(session_id, current_stage, success)
        
        # Send completion update
        if success:
            stage = ProgressStage.COMPLETED
            message = final_message or "Simulation generation completed successfully!"
        else:
            stage = ProgressStage.ERROR
            message = final_message or "Simulation generation failed"
        
        progress_update = ProgressUpdate(
            session_id=session_id,
            stage=stage,
            progress_percent=100 if success else 0,
            message=message,
            details=result_data,
            elapsed_time=time.time() - session["start_time"]
        )
        
        await self._broadcast_progress(session_id, progress_update)
        
        # Clean up session (after delay to allow client to receive final update)
        asyncio.create_task(self._cleanup_session_delayed(session_id, delay=5))
        
        logger.info(f"Completed session {session_id}: {success}")
    
    def _calculate_overall_progress(
        self,
        session_id: str,
        current_stage: ProgressStage,
        stage_progress: float
    ) -> float:
        """Calculate overall progress percentage."""
        session = self.active_sessions[session_id]
        total_weight = sum(self.stage_progress_weights.values())
        
        # Progress from completed stages
        completed_weight = sum(
            self.stage_progress_weights[stage]
            for stage in session["stages_completed"]
        )
        
        # Progress from current stage
        current_weight = self.stage_progress_weights.get(current_stage, 0) * (stage_progress / 100)
        
        overall_progress = ((completed_weight + current_weight) / total_weight) * 100
        return min(100, max(0, overall_progress))
    
    def _estimate_remaining_time(self, progress_percent: float, elapsed_time: float) -> float:
        """Estimate remaining time based on current progress."""
        if progress_percent <= 0:
            return None
        
        total_estimated = (elapsed_time / progress_percent) * 100
        remaining = total_estimated - elapsed_time
        return max(0, remaining)
    
    async def _broadcast_progress(self, session_id: str, progress_update: ProgressUpdate):
        """Broadcast progress update via WebSocket."""
        websocket = self.websocket_connections.get(session_id)
        
        if websocket:
            try:
                message = {
                    "type": "progress_update",
                    "data": asdict(progress_update)
                }
                
                await websocket.send_text(json.dumps(message))
                logger.debug(f"Broadcasted progress to {session_id}")
                
            except Exception as e:
                logger.warning(f"Failed to broadcast progress to {session_id}: {e}")
                # Remove dead WebSocket connection
                if session_id in self.websocket_connections:
                    del self.websocket_connections[session_id]
    
    async def _cleanup_session_delayed(self, session_id: str, delay: int = 5):
        """Clean up session after delay."""
        await asyncio.sleep(delay)
        
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        if session_id in self.websocket_connections:
            del self.websocket_connections[session_id]
        
        logger.debug(f"Cleaned up session {session_id}")
    
    def get_session_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive metrics for a session."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "elapsed_time": time.time() - session["start_time"],
            "current_stage": session["current_stage"].value,
            "stages_completed": [stage.value for stage in session["stages_completed"]],
            "stage_metrics": {
                stage.value: {
                    "duration": metrics.duration,
                    "success": metrics.success,
                    "error_message": metrics.error_message
                }
                for stage, metrics in session["stage_metrics"].items()
                if metrics.duration is not None
            },
            "overall_progress": self._calculate_overall_progress(
                session_id, session["current_stage"], 100
            ) if session["current_stage"] in session["stages_completed"] else 0
        }
    
    def disconnect_websocket(self, session_id: str):
        """Handle WebSocket disconnection."""
        if session_id in self.websocket_connections:
            del self.websocket_connections[session_id]
            logger.info(f"WebSocket disconnected for session {session_id}")


# Global progress tracker instance
_progress_tracker = None

def get_progress_tracker() -> RealTimeProgressTracker:
    """Get or create global progress tracker."""
    global _progress_tracker
    
    if _progress_tracker is None:
        _progress_tracker = RealTimeProgressTracker()
    
    return _progress_tracker


# Decorator for automatic progress tracking
def track_progress(stage: ProgressStage, message: str = None):
    """Decorator to automatically track progress for async functions."""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Extract session_id from kwargs or first arg if it's a string
            session_id = kwargs.get('session_id')
            if not session_id and args and isinstance(args[0], str):
                session_id = args[0]
            
            if session_id:
                tracker = get_progress_tracker()
                await tracker.start_stage(session_id, stage, message)
                
                try:
                    result = await func(self, *args, **kwargs)
                    await tracker.complete_stage(session_id, stage, success=True)
                    return result
                except Exception as e:
                    await tracker.complete_stage(session_id, stage, success=False, error_message=str(e))
                    raise
            else:
                return await func(self, *args, **kwargs)
        
        return wrapper
    return decorator