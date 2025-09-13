from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..db.base import get_async_session
from ..models.simulation import SimulationTemplate
from ..models.schemas import TemplateResponse


router = APIRouter()


@router.get("/", response_model=List[TemplateResponse])
async def list_templates(
    category: Optional[str] = None,
    active_only: bool = True,
    db: AsyncSession = Depends(get_async_session)
):
    """List available simulation templates."""
    
    query = select(SimulationTemplate)
    
    if category:
        query = query.where(SimulationTemplate.category == category)
    
    if active_only:
        query = query.where(SimulationTemplate.is_active == True)
    
    query = query.order_by(SimulationTemplate.usage_count.desc())
    
    result = await db.execute(query)
    templates = result.scalars().all()
    
    return [TemplateResponse.from_orm(template) for template in templates]


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: int,
    db: AsyncSession = Depends(get_async_session)
):
    """Get template by ID."""
    
    result = await db.execute(
        select(SimulationTemplate).where(SimulationTemplate.id == template_id)
    )
    template = result.scalar_one_or_none()
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return TemplateResponse.from_orm(template)


@router.get("/categories/")
async def list_categories(db: AsyncSession = Depends(get_async_session)):
    """List available template categories."""
    
    result = await db.execute(
        select(SimulationTemplate.category).distinct().where(
            SimulationTemplate.is_active == True
        )
    )
    categories = result.scalars().all()
    
    return {"categories": list(categories)}