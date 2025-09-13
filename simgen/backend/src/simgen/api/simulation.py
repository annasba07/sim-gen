import asyncio
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import logging

from ..db.base import get_async_session
from ..models.simulation import Simulation, SimulationStatus
from ..models.schemas import (
    SimulationRequest, 
    SimulationResponse, 
    SimulationListResponse,
    FeedbackRequest,
    ProgressUpdate,
    WebSocketMessage,
    SketchGenerationRequest,
    SketchTestResponse,
    SketchAnalysisResponse,
    MultiModalResponse
)
from ..services.llm_client import get_llm_client
from ..services.prompt_parser import PromptParser
from ..services.simulation_generator import SimulationGenerator


logger = logging.getLogger(__name__)
router = APIRouter()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_progress_update(self, session_id: str, progress: ProgressUpdate):
        if session_id in self.active_connections:
            try:
                websocket = self.active_connections[session_id]
                message = WebSocketMessage(
                    type="progress_update",
                    session_id=session_id,
                    data=progress.dict()
                )
                await websocket.send_json(message.dict())
            except Exception as e:
                logger.error(f"Failed to send progress update: {e}")
                self.disconnect(session_id)

connection_manager = ConnectionManager()


async def process_simulation(
    simulation_id: int,
    user_prompt: str,
    session_id: str,
    db: AsyncSession,
    max_iterations: int = 5
):
    """Background task to process simulation generation."""
    
    try:
        # Update status to processing
        await db.execute(
            update(Simulation)
            .where(Simulation.id == simulation_id)
            .values(status=SimulationStatus.PROCESSING, updated_at=datetime.utcnow())
        )
        await db.commit()
        
        # Send progress update
        await connection_manager.send_progress_update(
            session_id,
            ProgressUpdate(
                stage="parsing", 
                progress=0.2, 
                message="Parsing simulation prompt..."
            )
        )
        
        # Initialize services
        llm_client = get_llm_client()
        prompt_parser = PromptParser(llm_client)
        sim_generator = SimulationGenerator(llm_client)
        
        # Parse prompt and extract entities
        extracted_entities = await prompt_parser.parse_prompt(user_prompt)
        
        # Update simulation with extracted entities
        await db.execute(
            update(Simulation)
            .where(Simulation.id == simulation_id)
            .values(extracted_entities=extracted_entities.dict())
        )
        await db.commit()
        
        await connection_manager.send_progress_update(
            session_id,
            ProgressUpdate(
                stage="generation", 
                progress=0.4, 
                message="Generating MJCF simulation..."
            )
        )
        
        # Generate MJCF simulation (NEW: Pass user_prompt for dynamic composition)
        generation_result = await sim_generator.generate_simulation(extracted_entities, prompt=user_prompt)
        
        await connection_manager.send_progress_update(
            session_id,
            ProgressUpdate(
                stage="validation", 
                progress=0.6, 
                message="Validating physics..."
            )
        )
        
        # Execute and refine simulation (simplified for now)
        # TODO: Implement full feedback loop
        mjcf_content = generation_result.mjcf_content
        
        await connection_manager.send_progress_update(
            session_id,
            ProgressUpdate(
                stage="refinement", 
                progress=0.8, 
                message="Refining simulation quality..."
            )
        )
        
        # Update simulation with results
        generation_duration = datetime.utcnow().timestamp()  # TODO: Calculate actual duration
        
        await db.execute(
            update(Simulation)
            .where(Simulation.id == simulation_id)
            .values(
                status=SimulationStatus.COMPLETED,
                mjcf_content=mjcf_content,
                generation_method=generation_result.method,
                generation_duration=generation_duration,
                completed_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        )
        await db.commit()
        
        await connection_manager.send_progress_update(
            session_id,
            ProgressUpdate(
                stage="completed", 
                progress=1.0, 
                message="Simulation generation completed!"
            )
        )
        
        logger.info(f"Simulation {simulation_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Simulation processing failed: {e}", exc_info=True)
        
        # Update status to failed
        await db.execute(
            update(Simulation)
            .where(Simulation.id == simulation_id)
            .values(
                status=SimulationStatus.FAILED,
                error_message=str(e),
                updated_at=datetime.utcnow()
            )
        )
        await db.commit()
        
        # Send error update
        try:
            await connection_manager.send_progress_update(
                session_id,
                ProgressUpdate(
                    stage="error", 
                    progress=0.0, 
                    message=f"Generation failed: {str(e)}"
                )
            )
        except:
            pass  # Ignore WebSocket errors during error handling


@router.post("/generate", response_model=SimulationResponse)
async def generate_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session)
):
    """Generate a new physics simulation from natural language prompt."""
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Create simulation record
    simulation = Simulation(
        session_id=session_id,
        user_prompt=request.prompt,
        status=SimulationStatus.PENDING
    )
    
    db.add(simulation)
    await db.commit()
    await db.refresh(simulation)
    
    # Start background processing
    background_tasks.add_task(
        process_simulation,
        simulation.id,
        request.prompt,
        session_id,
        db,
        request.max_iterations or 5
    )
    
    logger.info(f"Started simulation generation: {simulation.id}")
    
    return SimulationResponse.from_orm(simulation)


@router.post("/test-generate")
async def test_generate_simulation(request: SimulationRequest):
    """Test simulation generation without database operations."""
    
    try:
        # Import services directly
        from ..services.prompt_parser import PromptParser
        from ..services.simulation_generator import SimulationGenerator
        from ..services.llm_client import LLMClient
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Extract entities from prompt
        prompt_parser = PromptParser(llm_client)
        entities = await prompt_parser.parse_prompt(request.prompt)
        
        # Generate simulation (NEW: Pass prompt for dynamic composition)
        sim_generator = SimulationGenerator(llm_client)
        result = await sim_generator.generate_simulation(entities, prompt=request.prompt)
        
        return {
            "status": "success" if result.success else "error",
            "prompt": request.prompt,
            "entities": {
                "objects": len(entities.objects),
                "constraints": len(entities.constraints),
                "environment": str(entities.environment)
            },
            "generation_method": result.method.value if result.method else "unknown",
            "mjcf_content": result.mjcf_content[:500] + "..." if len(result.mjcf_content) > 500 else result.mjcf_content,
            "success": result.success,
            "error": result.error_message
        }
        
    except Exception as e:
        logger.error(f"Test generation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "prompt": request.prompt,
            "error": str(e)
        }


@router.get("/{simulation_id}", response_model=SimulationResponse)
async def get_simulation(
    simulation_id: int,
    db: AsyncSession = Depends(get_async_session)
):
    """Get simulation by ID."""
    
    result = await db.execute(
        select(Simulation).where(Simulation.id == simulation_id)
    )
    simulation = result.scalar_one_or_none()
    
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    return SimulationResponse.from_orm(simulation)


@router.get("/", response_model=SimulationListResponse)
async def list_simulations(
    session_id: Optional[str] = None,
    status: Optional[SimulationStatus] = None,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_async_session)
):
    """List simulations with optional filtering."""
    
    query = select(Simulation)
    
    if session_id:
        query = query.where(Simulation.session_id == session_id)
    
    if status:
        query = query.where(Simulation.status == status)
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    query = query.order_by(Simulation.created_at.desc())
    
    result = await db.execute(query)
    simulations = result.scalars().all()
    
    # Get total count
    count_query = select(Simulation)
    if session_id:
        count_query = count_query.where(Simulation.session_id == session_id)
    if status:
        count_query = count_query.where(Simulation.status == status)
    
    count_result = await db.execute(count_query)
    total = len(count_result.scalars().all())
    
    return SimulationListResponse(
        simulations=[SimulationResponse.from_orm(sim) for sim in simulations],
        total=total,
        page=page,
        page_size=page_size
    )


@router.post("/{simulation_id}/feedback")
async def submit_feedback(
    simulation_id: int,
    feedback: FeedbackRequest,
    db: AsyncSession = Depends(get_async_session)
):
    """Submit user feedback for a simulation."""
    
    result = await db.execute(
        select(Simulation).where(Simulation.id == simulation_id)
    )
    simulation = result.scalar_one_or_none()
    
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    # Update simulation with feedback
    await db.execute(
        update(Simulation)
        .where(Simulation.id == simulation_id)
        .values(
            user_rating=feedback.rating,
            user_feedback=feedback.feedback,
            updated_at=datetime.utcnow()
        )
    )
    await db.commit()
    
    return {"message": "Feedback submitted successfully"}


@router.post("/sketch-generate", response_model=SketchTestResponse)
async def generate_from_sketch(request: SketchGenerationRequest):
    """Generate physics simulation from sketch + optional text prompt with optimized parallel processing."""
    
    import base64
    import time
    
    start_time = time.time()
    
    try:
        # Import optimized services
        from ..services.performance_optimizer import get_performance_pipeline
        from ..services.llm_client import get_llm_client
        
        # Initialize high-performance pipeline
        llm_client = get_llm_client()
        performance_pipeline = get_performance_pipeline(llm_client)
        
        # Decode the sketch image
        try:
            # Remove data URL prefix if present
            sketch_data = request.sketch_data
            if sketch_data.startswith('data:image'):
                sketch_data = sketch_data.split(',')[1]
            
            image_bytes = base64.b64decode(sketch_data)
        except Exception as e:
            return SketchTestResponse(
                status="error",
                input_data={"sketch_size": len(request.sketch_data), "prompt": request.prompt},
                processing_time=time.time() - start_time,
                success=False,
                error=f"Invalid sketch data: {str(e)}"
            )
        
        # 🚀 HIGH-PERFORMANCE PARALLEL PIPELINE
        # Runs sketch analysis + text preprocessing in parallel
        # Includes intelligent caching and performance tracking
        sketch_analysis, enhanced_result, generation_result, metrics = await performance_pipeline.generate_optimized(
            image_data=image_bytes,
            user_text=request.prompt,
            style_preferences=request.style_preferences,
            session_id=str(uuid.uuid4())  # Generate session ID for tracking
        )
        
        # Prepare response data structures
        sketch_response = None
        if sketch_analysis.success:
            sketch_response = SketchAnalysisResponse(
                success=sketch_analysis.success,
                confidence_score=sketch_analysis.confidence_score,
                physics_description=sketch_analysis.physics_description,
                detected_objects=len(sketch_analysis.extracted_entities.objects) if sketch_analysis.extracted_entities else 0,
                detected_constraints=len(sketch_analysis.extracted_entities.constraints) if sketch_analysis.extracted_entities else 0,
                raw_vision_output=sketch_analysis.raw_vision_output,
                error_message=sketch_analysis.error_message
            )
        
        multimodal_response = None
        if enhanced_result.success:
            multimodal_response = MultiModalResponse(
                sketch_analysis=sketch_response,
                enhanced_prompt=enhanced_result.enhanced_prompt,
                sketch_contribution=enhanced_result.sketch_contribution,
                text_contribution=enhanced_result.text_contribution,
                confidence_score=enhanced_result.confidence_score
            )
        
        # Prepare generation result data
        generation_data = None
        if generation_result:
            generation_data = {
                "success": generation_result.success,
                "method": generation_result.method.value if generation_result.method else "unknown",
                "mjcf_preview": generation_result.mjcf_content[:500] + "..." if len(generation_result.mjcf_content) > 500 else generation_result.mjcf_content,
                "full_mjcf_length": len(generation_result.mjcf_content),
                "error": generation_result.error_message
            }
        
        processing_time = time.time() - start_time
        
        # Add performance metrics to response
        performance_info = {
            "total_time": round(metrics.total_time, 2),
            "speedup_ratio": round(metrics.get_speedup_ratio(), 2),
            "parallel_time": round(metrics.parallel_time, 2),
            "cache_stats": {"hits": metrics.cache_hits, "misses": metrics.cache_misses},
            "optimization_enabled": True
        }
        
        return SketchTestResponse(
            status="success" if sketch_analysis.success and enhanced_result.success else "partial",
            input_data={
                "sketch_size": len(image_bytes),
                "prompt": request.prompt,
                "has_style_prefs": bool(request.style_preferences),
                "performance": performance_info  # Include performance metrics
            },
            sketch_analysis=sketch_response,
            multimodal_analysis=multimodal_response,
            generation_result=generation_data,
            processing_time=processing_time,
            success=generation_result.success if generation_result else False,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Sketch generation failed: {e}", exc_info=True)
        return SketchTestResponse(
            status="error",
            input_data={"sketch_data": "provided", "prompt": request.prompt},
            processing_time=time.time() - start_time,
            success=False,
            error=str(e)
        )


@router.get("/performance-stats")
async def get_performance_stats():
    """Get real-time performance statistics for the optimized pipeline"""
    try:
        from ..services.performance_optimizer import get_performance_pipeline
        from ..services.llm_client import get_llm_client
        
        llm_client = get_llm_client()
        pipeline = get_performance_pipeline(llm_client)
        
        stats = pipeline.get_performance_stats()
        
        return {
            "status": "success",
            "performance_stats": stats,
            "optimization_enabled": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        return {
            "status": "error", 
            "error": str(e),
            "optimization_enabled": False
        }


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time progress updates."""
    
    await connection_manager.connect(websocket, session_id)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for heartbeat
            await websocket.send_json({"type": "heartbeat", "data": data})
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(session_id)