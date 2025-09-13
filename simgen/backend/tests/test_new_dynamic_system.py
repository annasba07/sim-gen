#!/usr/bin/env python3
"""
Test the new dynamic system with complex prompts
"""

import asyncio
import sys
import os
import json

# Add the backend directory to Python path
sys.path.append(os.path.dirname(__file__))

from app.services.dynamic_scene_composer import DynamicSceneComposer
from app.services.llm_client import LLMClient
from app.models.schemas import ExtractedEntities, ObjectSchema, EnvironmentSchema, GeometrySchema, MaterialSchema

async def test_new_dynamic_system():
    """Test the new dynamic composition system."""
    
    print("TESTING NEW DYNAMIC SYSTEM")
    print("=" * 50)
    print("Testing truly generalized AI simulation generation:")
    print("- No hardcoded templates")
    print("- Professional MuJoCo models from Menagerie")
    print("- Dynamic scene composition")
    print("- Semantic understanding of prompts")
    print("=" * 50)

    try:
        # Initialize the dynamic composer
        llm_client = LLMClient()
        composer = DynamicSceneComposer(llm_client)
        
        print("Initializing dynamic scene composer...")
        success = await composer.initialize()
        if not success:
            print("WARNING: Composer initialization failed, but continuing with fallback")
        else:
            print("SUCCESS: Dynamic composer initialized successfully!")
        
        # Test with various complex prompts
        test_cases = [
            {
                "name": "Robotic Manipulation", 
                "prompt": "A robotic arm grasping and manipulating objects on a table",
                "description": "Should use professional robot arm models from Menagerie"
            },
            {
                "name": "Humanoid Walking",
                "prompt": "A humanoid robot learning to walk and balance",
                "description": "Should use humanoid models with proper locomotion setup"
            },
            {
                "name": "Quadruped Robot",
                "prompt": "A four-legged robot dog moving through obstacles",
                "description": "Should use quadruped models like Spot or Go1"
            },
            {
                "name": "Complex Physics Demo",
                "prompt": "Multiple objects with different materials falling and bouncing",
                "description": "Should create realistic physics with professional materials"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTEST {i}: {test_case['name']}")
            print(f"Prompt: '{test_case['prompt']}'")
            print(f"Expected: {test_case['description']}")
            
            try:
                # Create dummy entities for testing
                dummy_entities = ExtractedEntities(
                    objects=[
                        ObjectSchema(
                            name="test_object",
                            type="rigid_body",
                            geometry=GeometrySchema(shape="sphere", dimensions=[0.1]),
                            material=MaterialSchema(density=1000, friction=0.5),
                            position=[0, 0, 1]
                        )
                    ],
                    constraints=[],
                    environment=EnvironmentSchema(
                        gravity=[0, 0, -9.81],
                        ground={"type": "plane"},
                        boundaries={"type": "none"}
                    )
                )
                
                # Test the new dynamic composition
                mjcf_content = await composer.compose_scene_from_prompt(
                    test_case['prompt'], 
                    dummy_entities
                )
                
                print(f"SUCCESS: Generated {len(mjcf_content)} characters of MJCF")
                print(f"Contains professional features: {_analyze_mjcf_quality(mjcf_content)}")
                
                # Save for inspection
                filename = f"dynamic_test_{i}_{test_case['name'].lower().replace(' ', '_')}.xml"
                with open(filename, 'w') as f:
                    f.write(mjcf_content)
                print(f"SAVED: {filename}")
                
            except Exception as e:
                print(f"FAILED: Test failed: {e}")
                
        print(f"\nDYNAMIC SYSTEM TEST COMPLETE")
        print("Check the generated XML files to see the quality improvement!")
        
    except Exception as e:
        print(f"FAILED: System test failed: {e}")
        import traceback
        traceback.print_exc()

def _analyze_mjcf_quality(mjcf_content: str) -> str:
    """Analyze MJCF content for professional features."""
    
    features = []
    
    if 'castshadow="true"' in mjcf_content:
        features.append("shadows")
    if 'specular=' in mjcf_content:
        features.append("specular materials")
    if 'reflectance=' in mjcf_content:
        features.append("reflections")  
    if 'shadowsize=' in mjcf_content:
        features.append("high-res shadows")
    if mjcf_content.count('<light') > 1:
        features.append("multi-light setup")
    if 'skybox' in mjcf_content:
        features.append("professional skybox")
    if 'texture' in mjcf_content:
        features.append("textures")
    
    return ", ".join(features) if features else "basic quality"

if __name__ == "__main__":
    asyncio.run(test_new_dynamic_system())