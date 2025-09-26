"""
Real-Time Sketch Feedback API
Transforms UX from 9-step process to live interaction with instant feedback.

Addresses the core UX issue: Users need to see what the system understands
as they draw, not after completing their entire sketch.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from dataclasses import asdict

from ..core.container import get_websocket_manager, container
from ..core.interfaces import IComputerVisionPipeline, IWebSocketManager
from ..services.cv_simplified import SimplifiedCVResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/realtime", tags=["realtime"])


class SketchFeedbackManager:
    """Manages real-time sketch feedback sessions"""

    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.feedback_debouncer: Dict[str, asyncio.Task] = {}

    async def add_session(self, session_id: str, websocket: WebSocket):
        """Add a new feedback session"""
        self.active_sessions[session_id] = {
            "websocket": websocket,
            "sketch_history": [],
            "last_feedback": None,
            "confidence_trend": [],
            "detected_objects": []
        }
        logger.info(f"Added real-time feedback session: {session_id}")

    async def remove_session(self, session_id: str):
        """Remove feedback session"""
        if session_id in self.active_sessions:
            # Cancel any pending debounced feedback
            if session_id in self.feedback_debouncer:
                self.feedback_debouncer[session_id].cancel()
                del self.feedback_debouncer[session_id]

            del self.active_sessions[session_id]
            logger.info(f"Removed real-time feedback session: {session_id}")

    async def process_sketch_stroke(self, session_id: str, stroke_data: Dict[str, Any]):
        """Process new stroke data and provide immediate feedback"""
        if session_id not in self.active_sessions:
            return

        session = self.active_sessions[session_id]
        session["sketch_history"].append(stroke_data)

        # Debounce feedback to avoid overwhelming the CV pipeline
        if session_id in self.feedback_debouncer:
            self.feedback_debouncer[session_id].cancel()

        self.feedback_debouncer[session_id] = asyncio.create_task(
            self._debounced_feedback(session_id, delay=0.5)
        )

    async def _debounced_feedback(self, session_id: str, delay: float):
        """Provide feedback after a short delay to allow stroke completion"""
        try:
            await asyncio.sleep(delay)
            await self._generate_live_feedback(session_id)
        except asyncio.CancelledError:
            # Normal cancellation when new strokes arrive
            pass

    async def _generate_live_feedback(self, session_id: str):
        """Generate and send live feedback for current sketch state"""
        if session_id not in self.active_sessions:
            return

        session = self.active_sessions[session_id]
        websocket = session["websocket"]

        try:
            # Convert sketch history to image data (simplified)
            image_data = self._sketch_to_image(session["sketch_history"])

            if image_data:
                # Analyze with simplified CV pipeline
                cv_pipeline = container.get(IComputerVisionPipeline)
                result: SimplifiedCVResult = await cv_pipeline.analyze_sketch(image_data)

                # Generate human-friendly feedback
                feedback = self._generate_feedback_message(result, session)

                # Update session state
                session["last_feedback"] = feedback
                session["confidence_trend"].append(result.confidence)
                session["detected_objects"] = result.objects

                # Send feedback via WebSocket
                await websocket.send_json({
                    "type": "live_feedback",
                    "feedback": feedback,
                    "confidence": result.confidence,
                    "objects": [self._object_to_dict(obj) for obj in result.objects],
                    "processing_notes": result.processing_notes,
                    "timestamp": asyncio.get_event_loop().time()
                })

        except Exception as e:
            logger.error(f"Error generating live feedback for {session_id}: {e}")
            await websocket.send_json({
                "type": "error",
                "message": "Failed to analyze sketch",
                "details": str(e)
            })

    def _sketch_to_image(self, sketch_history: List[Dict[str, Any]]) -> Optional[bytes]:
        """Convert sketch stroke history to image data for CV analysis"""
        # This is a simplified implementation
        # In a real system, this would render the strokes to a canvas
        try:
            import numpy as np
            from PIL import Image, ImageDraw
            import io

            # Create blank canvas
            width, height = 800, 600
            image = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(image)

            # Draw all strokes
            for stroke in sketch_history:
                points = stroke.get('points', [])
                if len(points) > 1:
                    # Draw stroke as connected lines
                    for i in range(len(points) - 1):
                        x1, y1 = points[i]['x'], points[i]['y']
                        x2, y2 = points[i + 1]['x'], points[i + 1]['y']
                        draw.line([(x1, y1), (x2, y2)], fill='black', width=2)

            # Convert to bytes
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Error converting sketch to image: {e}")
            return None

    def _generate_feedback_message(self, result: SimplifiedCVResult, session: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-friendly feedback message"""
        objects = result.objects
        confidence = result.confidence

        # Determine feedback level based on confidence
        if confidence < 0.3:
            level = "low"
            message = "I'm having trouble understanding your sketch. Try making the shapes clearer."
            suggestions = ["Make shapes larger", "Use clearer strokes", "Add more detail"]
        elif confidence < 0.7:
            level = "medium"
            if objects:
                detected = [obj.object_type.value for obj in objects]
                message = f"I see {', '.join(detected)}. Keep drawing to help me understand the connections."
                suggestions = ["Add connecting lines", "Draw more objects", "Add labels if needed"]
            else:
                message = "I can see some shapes forming. Continue drawing to help me identify them."
                suggestions = ["Complete the shapes", "Add more details"]
        else:
            level = "high"
            if len(objects) >= 2:
                message = "Great! I can see multiple objects. This looks like it could be a physics system."
                suggestions = ["Add springs or joints", "Label important parts", "Ready to generate simulation"]
            else:
                message = f"I clearly see a {objects[0].object_type.value}. Add more objects to create a physics system."
                suggestions = ["Add another object", "Connect with lines", "Add forces or constraints"]

        # Detect potential physics systems
        physics_hints = self._detect_physics_patterns(objects)

        return {
            "level": level,
            "message": message,
            "suggestions": suggestions,
            "confidence": confidence,
            "physics_hints": physics_hints,
            "object_count": len(objects)
        }

    def _detect_physics_patterns(self, objects: List) -> List[str]:
        """Detect common physics patterns and provide hints"""
        hints = []

        if len(objects) >= 2:
            circles = [obj for obj in objects if obj.object_type.value == "circle"]
            rectangles = [obj for obj in objects if obj.object_type.value == "rectangle"]

            if circles and rectangles:
                hints.append("🎯 This could be a pendulum or ball-and-ramp system")
            elif len(circles) >= 2:
                hints.append("⚽ Multiple balls - collision or rolling system?")
            elif len(rectangles) >= 2:
                hints.append("📦 Multiple blocks - stacking or sliding system?")

        if len(objects) == 1:
            obj_type = objects[0].object_type.value
            if obj_type == "circle":
                hints.append("🔴 Add a ramp or surface for the ball to interact with")
            elif obj_type == "rectangle":
                hints.append("📦 Add another object or a pivot point for movement")

        return hints

    def _object_to_dict(self, obj) -> Dict[str, Any]:
        """Convert detected object to dictionary for JSON serialization"""
        return {
            "type": obj.object_type.value,
            "confidence": obj.confidence,
            "center": obj.center,
            "size": obj.size,
            "bbox": obj.bbox
        }


# Global feedback manager
feedback_manager = SketchFeedbackManager()


@router.websocket("/sketch-feedback/{session_id}")
async def sketch_feedback_websocket(
    websocket: WebSocket,
    session_id: str,
    ws_manager: IWebSocketManager = Depends(get_websocket_manager)
):
    """
    Real-time sketch feedback WebSocket endpoint.

    Provides live feedback as users draw, transforming the UX from:
    - Before: Draw → Submit → Wait → Binary Success/Failure
    - After: Draw with live feedback → Enhanced understanding → Quick iteration
    """
    await websocket.accept()
    await feedback_manager.add_session(session_id, websocket)

    # Send welcome message
    await websocket.send_json({
        "type": "connection_established",
        "message": "Real-time sketch feedback is now active!",
        "instructions": [
            "Start drawing your physics system",
            "I'll provide live feedback as you draw",
            "Watch for confidence levels and suggestions",
            "Green feedback means I understand your sketch well"
        ]
    })

    try:
        while True:
            # Receive stroke data from client
            data = await websocket.receive_json()

            if data["type"] == "stroke_start":
                # New stroke started
                await websocket.send_json({
                    "type": "stroke_acknowledged",
                    "message": "Drawing..."
                })

            elif data["type"] == "stroke_data":
                # Process stroke points
                await feedback_manager.process_sketch_stroke(session_id, data["stroke"])

            elif data["type"] == "stroke_end":
                # Stroke completed - trigger immediate feedback
                await feedback_manager._generate_live_feedback(session_id)

            elif data["type"] == "clear_canvas":
                # Canvas cleared - reset session
                if session_id in feedback_manager.active_sessions:
                    session = feedback_manager.active_sessions[session_id]
                    session["sketch_history"] = []
                    session["confidence_trend"] = []
                    session["detected_objects"] = []

                await websocket.send_json({
                    "type": "canvas_cleared",
                    "message": "Canvas cleared. Start drawing your new sketch!"
                })

            elif data["type"] == "request_analysis":
                # Manual analysis request
                await feedback_manager._generate_live_feedback(session_id)

    except WebSocketDisconnect:
        logger.info(f"Sketch feedback WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Error in sketch feedback WebSocket {session_id}: {e}")
    finally:
        await feedback_manager.remove_session(session_id)


@router.get("/feedback-stats")
async def get_feedback_stats():
    """Get statistics about real-time feedback sessions"""
    return {
        "active_sessions": len(feedback_manager.active_sessions),
        "total_sessions_data": {
            session_id: {
                "stroke_count": len(session["sketch_history"]),
                "confidence": session.get("last_feedback", {}).get("confidence", 0.0),
                "object_count": len(session["detected_objects"])
            }
            for session_id, session in feedback_manager.active_sessions.items()
        }
    }


@router.post("/test-feedback")
async def test_feedback_endpoint(image_data: Dict[str, Any]):
    """Test endpoint for feedback system without WebSocket"""
    try:
        cv_pipeline = container.get(IComputerVisionPipeline)

        # Mock image data for testing
        test_image = b"test_image_data"  # In real use, decode from base64

        result = await cv_pipeline.analyze_sketch(test_image)

        # Generate feedback
        feedback = feedback_manager._generate_feedback_message(result, {"confidence_trend": []})

        return {
            "feedback": feedback,
            "cv_result": {
                "object_count": len(result.objects),
                "confidence": result.confidence,
                "processing_notes": result.processing_notes
            }
        }
    except Exception as e:
        return {"error": str(e)}