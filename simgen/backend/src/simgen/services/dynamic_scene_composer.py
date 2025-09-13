#!/usr/bin/env python3
"""
Dynamic Scene Composer - Professional MuJoCo Scene Generation
Replaces hardcoded templates with intelligent, generalized scene composition
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import tempfile
import shutil

from ..models.schemas import ExtractedEntities
from ..services.llm_client import LLMClient, LLMError


logger = logging.getLogger(__name__)


@dataclass
class MenagerieModel:
    """Represents a professional MuJoCo model from Menagerie."""
    name: str
    category: str  # arm, quadruped, humanoid, hand, etc.
    xml_path: str
    assets_path: str
    description: str
    dof: int
    license: str


@dataclass
class SceneComposition:
    """Represents a dynamically composed scene."""
    main_models: List[MenagerieModel]
    environment_settings: Dict[str, Any]
    physics_constraints: List[Dict[str, Any]]
    lighting_setup: Dict[str, Any]
    camera_angles: List[Dict[str, Any]]
    materials: Dict[str, Any]


class MenagerieModelLibrary:
    """Manages the MuJoCo Menagerie model library."""
    
    def __init__(self, models_dir: str = "./menagerie_models"):
        self.models_dir = Path(models_dir)
        self.models_index: Dict[str, MenagerieModel] = {}
        self.categories = {
            'arms': ['robot arm', 'robotic arm', 'manipulator', 'arm robot'],
            'quadrupeds': ['dog robot', 'quadruped', 'four-legged robot', 'robotic dog'],
            'humanoids': ['humanoid robot', 'human robot', 'android', 'bipedal robot'],
            'hands': ['robotic hand', 'robot hand', 'gripper', 'end effector'],
            'drones': ['drone', 'quadcopter', 'flying robot', 'uav'],
            'mobile': ['mobile robot', 'wheeled robot', 'ground robot']
        }
    
    async def initialize(self) -> bool:
        """Initialize the model library by downloading Menagerie if needed."""
        try:
            if not self.models_dir.exists():
                logger.info("Downloading MuJoCo Menagerie...")
                await self._download_menagerie()
            
            await self._build_models_index()
            logger.info(f"Initialized model library with {len(self.models_index)} models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model library: {e}")
            return False
    
    async def _download_menagerie(self):
        """Download the MuJoCo Menagerie repository."""
        try:
            # Clone the repository
            cmd = [
                'git', 'clone', 
                'https://github.com/google-deepmind/mujoco_menagerie.git',
                str(self.models_dir)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Git clone failed: {stderr.decode()}")
                
            logger.info("Successfully downloaded MuJoCo Menagerie")
            
        except Exception as e:
            # Fallback: create basic structure for development
            logger.warning(f"Failed to download Menagerie: {e}")
            logger.info("Creating mock model structure for development...")
            await self._create_mock_models()
    
    async def _create_mock_models(self):
        """Create mock models for development when Menagerie isn't available."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock structure
        mock_models = [
            {'name': 'franka_panda', 'category': 'arms', 'dof': 7},
            {'name': 'ur5e', 'category': 'arms', 'dof': 6},
            {'name': 'spot', 'category': 'quadrupeds', 'dof': 12},
            {'name': 'unitree_h1', 'category': 'humanoids', 'dof': 25},
            {'name': 'shadow_hand', 'category': 'hands', 'dof': 24}
        ]
        
        for model in mock_models:
            model_dir = self.models_dir / model['name']
            model_dir.mkdir(exist_ok=True)
            
            # Create mock XML
            mock_xml = f"""<mujoco model="{model['name']}">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="{model['name']}_base">
      <geom name="{model['name']}_geom" type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>"""
            
            with open(model_dir / f"{model['name']}.xml", 'w') as f:
                f.write(mock_xml)
            
            # Create assets directory
            (model_dir / "assets").mkdir(exist_ok=True)
    
    async def _build_models_index(self):
        """Build an index of available models."""
        if not self.models_dir.exists():
            return
        
        for model_path in self.models_dir.iterdir():
            if not model_path.is_dir() or model_path.name.startswith('.'):
                continue
            
            # Look for XML files
            xml_files = list(model_path.glob("*.xml"))
            if not xml_files:
                continue
            
            main_xml = xml_files[0]  # Use first XML file found
            assets_path = model_path / "assets"
            
            # Determine category based on path/name
            category = self._determine_category(model_path.name)
            
            model = MenagerieModel(
                name=model_path.name,
                category=category,
                xml_path=str(main_xml),
                assets_path=str(assets_path) if assets_path.exists() else "",
                description=f"Professional {category} model",
                dof=self._estimate_dof(category),
                license="Professional"
            )
            
            self.models_index[model.name] = model
    
    def _determine_category(self, model_name: str) -> str:
        """Determine model category based on name."""
        name_lower = model_name.lower()
        
        if any(term in name_lower for term in ['panda', 'ur', 'kinova', 'kuka', 'arm']):
            return 'arms'
        elif any(term in name_lower for term in ['spot', 'go1', 'anymal', 'dog']):
            return 'quadrupeds'
        elif any(term in name_lower for term in ['h1', 'atlas', 'humanoid', 'human']):
            return 'humanoids'
        elif any(term in name_lower for term in ['hand', 'shadow', 'allegro', 'gripper']):
            return 'hands'
        elif any(term in name_lower for term in ['drone', 'crazyflie', 'quad']):
            return 'drones'
        else:
            return 'misc'
    
    def _estimate_dof(self, category: str) -> int:
        """Estimate degrees of freedom based on category."""
        dof_map = {
            'arms': 7,
            'quadrupeds': 12,
            'humanoids': 25,
            'hands': 20,
            'drones': 6,
            'misc': 6
        }
        return dof_map.get(category, 6)
    
    def find_models_by_semantic_match(self, description: str) -> List[MenagerieModel]:
        """Find models that semantically match a description."""
        matches = []
        description_lower = description.lower()
        
        for category, keywords in self.categories.items():
            if any(keyword in description_lower for keyword in keywords):
                # Find models in this category
                category_models = [m for m in self.models_index.values() if m.category == category]
                matches.extend(category_models[:2])  # Limit to 2 per category
        
        return matches[:3]  # Return top 3 matches


class DynamicSceneComposer:
    """Dynamically composes professional MuJoCo scenes from natural language."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.model_library = MenagerieModelLibrary()
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the scene composer."""
        if not self.initialized:
            success = await self.model_library.initialize()
            if success:
                self.initialized = True
                logger.info("Dynamic Scene Composer initialized successfully")
            return success
        return True
    
    async def compose_scene_from_prompt(self, prompt: str, entities: ExtractedEntities) -> str:
        """Compose a professional MJCF scene from natural language prompt."""
        
        if not await self.initialize():
            raise Exception("Failed to initialize scene composer")
        
        try:
            # Step 1: Analyze prompt for semantic understanding
            scene_analysis = await self._analyze_scene_requirements(prompt, entities)
            
            # Step 2: Select appropriate professional models
            selected_models = await self._select_professional_models(scene_analysis)
            
            # Step 3: Compose dynamic scene with professional quality
            mjcf_content = await self._compose_professional_mjcf(
                prompt, entities, selected_models, scene_analysis
            )
            
            return mjcf_content
            
        except Exception as e:
            logger.error(f"Scene composition failed: {e}")
            # Fallback to basic composition
            return await self._fallback_composition(prompt, entities)
    
    async def _analyze_scene_requirements(self, prompt: str, entities: ExtractedEntities) -> Dict[str, Any]:
        """Analyze the prompt to understand scene requirements."""
        
        analysis_prompt = f"""
Analyze this physics simulation prompt and extract scene requirements:

PROMPT: "{prompt}"

EXTRACTED ENTITIES: {len(entities.objects)} objects, {len(entities.constraints)} constraints

Provide a JSON analysis with:
1. "scene_type": primary simulation type (robotics, physics_demo, biomechanics, etc.)
2. "required_models": what types of professional models are needed (robotic_arm, humanoid, quadruped, etc.)
3. "physics_focus": main physics phenomena (manipulation, locomotion, collision, balance, etc.)
4. "environment_style": scene environment (laboratory, outdoor, industrial, minimal, etc.)
5. "interaction_type": how objects interact (grasping, walking, throwing, pendulum, etc.)
6. "visual_style": desired visual quality (professional, cinematic, technical, realistic)
7. "complexity_level": scene complexity (simple, moderate, complex)

Return only valid JSON.
"""
        
        try:
            response = await self.llm_client.complete(analysis_prompt, temperature=0.1)
            # Parse JSON response
            import json
            analysis = json.loads(response.strip())
            return analysis
            
        except Exception as e:
            logger.warning(f"Scene analysis failed, using defaults: {e}")
            return {
                "scene_type": "physics_demo",
                "required_models": ["basic_objects"],
                "physics_focus": "general",
                "environment_style": "minimal",
                "interaction_type": "general",
                "visual_style": "professional",
                "complexity_level": "simple"
            }
    
    async def _select_professional_models(self, scene_analysis: Dict[str, Any]) -> List[MenagerieModel]:
        """Select appropriate professional models based on scene analysis."""
        
        required_models = scene_analysis.get("required_models", [])
        selected_models = []
        
        # Map requirements to model categories
        for requirement in required_models:
            models = self.model_library.find_models_by_semantic_match(requirement)
            selected_models.extend(models)
        
        # If no specific models found, select based on scene type
        if not selected_models:
            scene_type = scene_analysis.get("scene_type", "physics_demo")
            if "robot" in scene_type.lower():
                models = self.model_library.find_models_by_semantic_match("robotic arm")
                selected_models.extend(models[:1])
        
        return selected_models[:3]  # Limit to 3 models max
    
    async def _compose_professional_mjcf(
        self, 
        prompt: str, 
        entities: ExtractedEntities, 
        selected_models: List[MenagerieModel],
        scene_analysis: Dict[str, Any]
    ) -> str:
        """Compose professional MJCF with dynamic scene generation."""
        
        composition_prompt = f"""
Generate a professional MJCF simulation based on:

PROMPT: "{prompt}"
SCENE_ANALYSIS: {json.dumps(scene_analysis, indent=2)}
AVAILABLE_MODELS: {[m.name for m in selected_models]}
ENTITIES: {len(entities.objects)} objects

Create a complete professional MJCF that:

1. PROFESSIONAL VISUAL QUALITY:
   - Use advanced lighting (3+ directional lights with shadows)
   - High-quality materials with reflectance, specular, shininess
   - Professional skybox and ground textures
   - Shadow quality 2048+ resolution
   - Atmospheric effects (haze, ambient lighting)

2. DYNAMIC SCENE COMPOSITION:
   - Intelligently place objects based on physics requirements
   - Create meaningful spatial relationships
   - Add appropriate constraints and joints
   - Include realistic physics parameters

3. PROFESSIONAL ASSETS:
   - Use professional textures and materials
   - Include multiple material types (metal, plastic, rubber, etc.)
   - Add visual details like highlights and reflections

4. PHYSICS EXCELLENCE:
   - Realistic physics parameters
   - Appropriate solver settings
   - Proper contact and friction
   - Stable simulation setup

Generate ONLY the complete MJCF XML content. Make it cinematic quality.
"""
        
        try:
            mjcf_content = await self.llm_client.complete(
                composition_prompt,
                temperature=0.2,
                max_tokens=6000
            )
            
            # Clean up the response
            mjcf_content = self._clean_mjcf_response(mjcf_content)
            
            # Enhance with professional features
            enhanced_mjcf = self._enhance_with_professional_features(mjcf_content)
            
            return enhanced_mjcf
            
        except Exception as e:
            logger.error(f"Professional composition failed: {e}")
            raise
    
    def _enhance_with_professional_features(self, mjcf_content: str) -> str:
        """Enhance MJCF with additional professional features."""
        
        # Add professional visual settings if missing
        if '<visual>' not in mjcf_content:
            visual_section = """
  <!-- PROFESSIONAL VISUAL SETTINGS -->
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="1.0 1.0 1.0" specular="0.3 0.3 0.3"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="4096" offsamples="16"/>
    <map force="0.1" zfar="50"/>
  </visual>"""
            mjcf_content = mjcf_content.replace('</mujoco>', f'{visual_section}\n</mujoco>')
        
        # Ensure professional lighting
        if 'castshadow="true"' not in mjcf_content:
            # Add shadow casting to main light
            mjcf_content = mjcf_content.replace(
                'directional="true"',
                'directional="true" castshadow="true"'
            )
        
        return mjcf_content
    
    def _clean_mjcf_response(self, response: str) -> str:
        """Clean LLM response to extract MJCF content."""
        
        # Remove markdown formatting
        if "```xml" in response:
            response = response.split("```xml")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        response = response.strip()
        
        # Ensure proper XML start
        if not response.startswith(('<?xml', '<mujoco')):
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith(('<?xml', '<mujoco')):
                    response = '\n'.join(lines[i:])
                    break
        
        return response
    
    async def _fallback_composition(self, prompt: str, entities: ExtractedEntities) -> str:
        """Fallback composition for when advanced features fail."""
        
        logger.info("Using fallback composition")
        
        # Simple professional template
        fallback_template = """<mujoco model="dynamic_simulation">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  
  <option timestep="0.002" iterations="50" solver="PGS" gravity="0 0 -9.81"/>
  
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.0 0.1 0.2" width="800" height="800"/>
    <texture name="groundplane" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="300" height="300"/>
    
    <material name="groundplane" texture="groundplane" reflectance="0.3" shininess="0.1" specular="0.4"/>
    <material name="metal" rgba="0.7 0.7 0.8 1" reflectance="0.8" shininess="0.9" specular="1.0"/>
    <material name="plastic" rgba="0.2 0.8 0.3 1" reflectance="0.5" shininess="0.7" specular="0.8"/>
  </asset>
  
  <worldbody>
    <!-- Professional Lighting -->
    <light name="sun" pos="0 0 3" dir="0 0 -1" directional="true" diffuse="1.0 1.0 0.9" specular="0.5 0.5 0.4" castshadow="true"/>
    <light name="fill1" pos="3 2 2" dir="-0.5 -0.3 -1" directional="true" diffuse="0.3 0.3 0.4" specular="0.1 0.1 0.1"/>
    <light name="fill2" pos="-2 2 1" dir="0.3 -0.3 -0.5" directional="true" diffuse="0.2 0.2 0.3" specular="0.05 0.05 0.05"/>
    
    <!-- Professional Ground -->
    <geom name="ground" type="plane" size="15 15 0.1" material="groundplane" friction="0.9 0.1 0.001"/>
    
    <!-- Dynamic Objects Based on Prompt -->"""
        
        # Add objects based on entities
        objects_xml = ""
        for i, obj in enumerate(entities.objects):
            if obj.geometry.shape == "sphere":
                objects_xml += f"""
    <body name="{obj.name}" pos="{' '.join(map(str, obj.position))}">
      <joint name="{obj.name}_free" type="free"/>
      <geom name="{obj.name}_geom" type="sphere" size="{obj.geometry.dimensions[0]}" material="plastic"/>
      <geom name="{obj.name}_highlight" type="sphere" size="{obj.geometry.dimensions[0] * 1.05}" pos="0.02 0.02 0.02" rgba="1.0 1.0 1.0 0.3"/>
    </body>"""
            else:
                # Default to box
                size_x = obj.geometry.dimensions[0] / 2 if len(obj.geometry.dimensions) > 0 else 0.1
                size_y = obj.geometry.dimensions[1] / 2 if len(obj.geometry.dimensions) > 1 else 0.1
                size_z = obj.geometry.dimensions[2] / 2 if len(obj.geometry.dimensions) > 2 else 0.1
                
                objects_xml += f"""
    <body name="{obj.name}" pos="{' '.join(map(str, obj.position))}">
      <joint name="{obj.name}_free" type="free"/>
      <geom name="{obj.name}_geom" type="box" size="{size_x} {size_y} {size_z}" material="metal"/>
    </body>"""
        
        fallback_template += objects_xml
        fallback_template += """
  </worldbody>
  
  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="1.0 1.0 1.0" specular="0.3 0.3 0.3"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="4096" offsamples="16"/>
    <map force="0.1" zfar="50"/>
  </visual>
</mujoco>"""
        
        return fallback_template