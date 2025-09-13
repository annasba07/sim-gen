"""
Multi-modal Prompt Enhancement Service

This service combines sketch analysis with text prompts to create rich,
detailed physics simulation descriptions that maximize the quality of
generated simulations.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .llm_client import LLMClient
from .sketch_analyzer import SketchAnalysisResult
from ..models.schemas import ExtractedEntities

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPromptResult:
    """Result of multi-modal prompt enhancement."""
    success: bool
    enhanced_prompt: str
    confidence_score: float
    sketch_contribution: float  # How much the sketch contributed (0-1)
    text_contribution: float    # How much the text contributed (0-1)
    combined_entities: Optional[ExtractedEntities]
    error_message: Optional[str] = None


class MultiModalEnhancer:
    """Enhances prompts by combining sketch analysis with text input."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    async def enhance_prompt(
        self,
        sketch_analysis: Optional[SketchAnalysisResult] = None,
        user_text: Optional[str] = None,
        style_preferences: Optional[Dict[str, Any]] = None
    ) -> EnhancedPromptResult:
        """
        Enhance a prompt by combining sketch analysis with text input.
        
        Args:
            sketch_analysis: Result from sketch analyzer
            user_text: User's text prompt/description
            style_preferences: Optional visual/physics style preferences
            
        Returns:
            Enhanced prompt result with combined information
        """
        try:
            if not sketch_analysis and not user_text:
                return EnhancedPromptResult(
                    success=False,
                    enhanced_prompt="",
                    confidence_score=0.0,
                    sketch_contribution=0.0,
                    text_contribution=0.0,
                    combined_entities=None,
                    error_message="Either sketch analysis or text prompt is required"
                )
            
            # Calculate contribution weights
            sketch_contribution = self._calculate_sketch_contribution(sketch_analysis)
            text_contribution = self._calculate_text_contribution(user_text)
            
            # Generate enhanced prompt
            enhanced_prompt = await self._generate_enhanced_prompt(
                sketch_analysis, user_text, style_preferences,
                sketch_contribution, text_contribution
            )
            
            # Combine entities if available
            combined_entities = self._combine_entities(sketch_analysis, user_text)
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(
                sketch_analysis, user_text, sketch_contribution, text_contribution
            )
            
            return EnhancedPromptResult(
                success=True,
                enhanced_prompt=enhanced_prompt,
                confidence_score=confidence,
                sketch_contribution=sketch_contribution,
                text_contribution=text_contribution,
                combined_entities=combined_entities
            )
            
        except Exception as e:
            logger.error(f"Multi-modal enhancement failed: {e}")
            return EnhancedPromptResult(
                success=False,
                enhanced_prompt="",
                confidence_score=0.0,
                sketch_contribution=0.0,
                text_contribution=0.0,
                combined_entities=None,
                error_message=str(e)
            )
    
    def _calculate_sketch_contribution(
        self, 
        sketch_analysis: Optional[SketchAnalysisResult]
    ) -> float:
        """Calculate how much the sketch contributes to the final prompt."""
        if not sketch_analysis or not sketch_analysis.success:
            return 0.0
        
        # Base contribution from confidence
        contribution = sketch_analysis.confidence_score * 0.6
        
        # Boost if we have extracted entities
        if sketch_analysis.extracted_entities:
            if len(sketch_analysis.extracted_entities.objects) > 0:
                contribution += 0.2
            if len(sketch_analysis.extracted_entities.constraints) > 0:
                contribution += 0.1
        
        # Boost based on physics description quality
        if len(sketch_analysis.physics_description) > 100:
            contribution += 0.1
        
        return min(contribution, 1.0)
    
    def _calculate_text_contribution(self, user_text: Optional[str]) -> float:
        """Calculate how much the text contributes to the final prompt."""
        if not user_text:
            return 0.0
        
        # Base contribution based on length and content
        base_contribution = 0.3
        
        if len(user_text) > 20:
            base_contribution += 0.2
        if len(user_text) > 50:
            base_contribution += 0.2
        
        # Boost for physics-related keywords
        physics_keywords = [
            'bounce', 'swing', 'rotate', 'slide', 'collision', 'gravity',
            'friction', 'spring', 'pendulum', 'force', 'velocity', 'acceleration'
        ]
        
        keyword_count = sum(1 for keyword in physics_keywords if keyword.lower() in user_text.lower())
        base_contribution += min(keyword_count * 0.05, 0.3)
        
        return min(base_contribution, 1.0)
    
    async def _generate_enhanced_prompt(
        self,
        sketch_analysis: Optional[SketchAnalysisResult],
        user_text: Optional[str],
        style_preferences: Optional[Dict[str, Any]],
        sketch_weight: float,
        text_weight: float
    ) -> str:
        """Generate an enhanced prompt combining all inputs."""
        
        enhancement_prompt = f"""
Create an enhanced physics simulation prompt by combining the following inputs:

SKETCH ANALYSIS (Weight: {sketch_weight:.2f}):
{sketch_analysis.physics_description if sketch_analysis and sketch_analysis.success else 'No sketch provided'}

USER TEXT (Weight: {text_weight:.2f}):
{user_text or 'No text provided'}

STYLE PREFERENCES:
{style_preferences or 'Default physics simulation style'}

Your task:
1. Combine the sketch analysis and user text into a single, comprehensive physics simulation description
2. Resolve any conflicts between sketch and text by favoring the higher-weighted input
3. Fill in missing details with reasonable physics assumptions
4. Ensure the result is suitable for high-quality MuJoCo simulation generation
5. Maintain professional visual quality expectations (realistic materials, lighting, etc.)

Requirements:
- Be specific about object properties (size, material, color, mass)
- Include spatial relationships and positioning
- Specify physics behaviors and interactions
- Include environmental details (lighting, surfaces, boundaries)
- Describe expected motion and dynamics

Output format: A detailed, professional physics simulation prompt that captures the user's intent from both sketch and text.

Focus on creating something visually impressive and physically accurate.
"""
        
        try:
            return await self.llm_client.complete(
                prompt=enhancement_prompt,
                temperature=0.3,
                max_tokens=3000
            )
        except Exception as e:
            logger.error(f"Enhanced prompt generation failed: {e}")
            
            # Fallback: combine inputs manually
            fallback_parts = []
            
            if sketch_analysis and sketch_analysis.success:
                fallback_parts.append(f"Based on sketch: {sketch_analysis.physics_description}")
            
            if user_text:
                fallback_parts.append(f"User intent: {user_text}")
            
            fallback_parts.append("Create a professional physics simulation with realistic materials and lighting.")
            
            return " ".join(fallback_parts)
    
    def _combine_entities(
        self,
        sketch_analysis: Optional[SketchAnalysisResult],
        user_text: Optional[str]
    ) -> Optional[ExtractedEntities]:
        """Combine entities from sketch analysis with any additional text-derived entities."""
        if sketch_analysis and sketch_analysis.extracted_entities:
            return sketch_analysis.extracted_entities
        
        # If no sketch entities, return None for now
        # Future enhancement: extract entities from text alone
        return None
    
    def _calculate_overall_confidence(
        self,
        sketch_analysis: Optional[SketchAnalysisResult],
        user_text: Optional[str],
        sketch_weight: float,
        text_weight: float
    ) -> float:
        """Calculate overall confidence in the enhanced prompt."""
        base_confidence = 0.4
        
        # Add confidence from sketch
        if sketch_analysis and sketch_analysis.success:
            base_confidence += sketch_weight * sketch_analysis.confidence_score * 0.4
        
        # Add confidence from text
        if user_text:
            text_conf = min(len(user_text) / 100.0, 1.0)  # Longer text = higher confidence
            base_confidence += text_weight * text_conf * 0.4
        
        # Bonus for having both inputs
        if sketch_analysis and sketch_analysis.success and user_text:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    async def create_sketch_only_prompt(
        self, 
        sketch_analysis: SketchAnalysisResult
    ) -> EnhancedPromptResult:
        """Create enhanced prompt from sketch analysis only."""
        return await self.enhance_prompt(sketch_analysis=sketch_analysis)
    
    async def create_text_only_prompt(
        self, 
        user_text: str
    ) -> EnhancedPromptResult:
        """Create enhanced prompt from text only (for comparison).""" 
        return await self.enhance_prompt(user_text=user_text)
    
    async def create_fusion_prompt(
        self,
        sketch_analysis: SketchAnalysisResult,
        user_text: str,
        style_preferences: Optional[Dict[str, Any]] = None
    ) -> EnhancedPromptResult:
        """Create the ultimate fusion prompt combining sketch + text + style."""
        return await self.enhance_prompt(
            sketch_analysis=sketch_analysis,
            user_text=user_text, 
            style_preferences=style_preferences
        )


def get_multimodal_enhancer() -> MultiModalEnhancer:
    """Get multi-modal enhancer instance with LLM client."""
    from .llm_client import get_llm_client
    return MultiModalEnhancer(get_llm_client())