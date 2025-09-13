#!/usr/bin/env python3
"""
Test the new system via direct API call instead of the server
"""

import asyncio
import sys
import os
import json

# Add the backend directory to Python path
sys.path.append(os.path.dirname(__file__))

from app.services.prompt_parser import PromptParser
from app.services.simulation_generator import SimulationGenerator
from app.services.llm_client import LLMClient

async def test_new_system_directly():
    """Test the new dynamic system directly via API calls."""
    
    print("TESTING NEW DYNAMIC SYSTEM VIA DIRECT API")
    print("=" * 50)
    
    try:
        # Initialize services
        llm_client = LLMClient()
        prompt_parser = PromptParser(llm_client)
        sim_generator = SimulationGenerator(llm_client)
        
        # Test prompts that should showcase the new dynamic system
        test_prompts = [
            "A robotic arm picking up objects from a table",
            "A humanoid robot learning to walk", 
            "A bouncing ball with realistic physics",
            "Multiple spheres falling and colliding"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTEST {i}: '{prompt}'")
            
            try:
                # Parse prompt to extract entities
                print("  Parsing prompt...")
                entities = await prompt_parser.parse_prompt(prompt)
                print(f"  Found {len(entities.objects)} objects, {len(entities.constraints)} constraints")
                
                # Generate simulation using NEW dynamic system
                print("  Generating with NEW dynamic composition system...")
                result = await sim_generator.generate_simulation(entities, prompt=prompt)
                
                if result.success:
                    print(f"  SUCCESS: Generated {len(result.mjcf_content)} chars via {result.method.value}")
                    print(f"  Metadata: {result.metadata}")
                    
                    # Save for inspection  
                    filename = f"api_test_{i}_{prompt[:20].replace(' ', '_')}.xml"
                    with open(filename, 'w') as f:
                        f.write(result.mjcf_content)
                    print(f"  SAVED: {filename}")
                    
                    # Analyze quality
                    quality = _analyze_quality(result.mjcf_content)
                    print(f"  QUALITY: {quality}")
                    
                else:
                    print(f"  FAILED: {result.error_message}")
                    
            except Exception as e:
                print(f"  ERROR: {e}")
                
        print(f"\nDIRECT API TEST COMPLETE")
        
    except Exception as e:
        print(f"SYSTEM ERROR: {e}")
        import traceback
        traceback.print_exc()

def _analyze_quality(mjcf_content: str) -> str:
    """Analyze MJCF quality features."""
    
    features = []
    
    # Check for professional visual features
    if 'castshadow="true"' in mjcf_content:
        features.append("shadows")
    if 'specular=' in mjcf_content:
        features.append("materials")
    if 'shadowsize=' in mjcf_content:
        features.append("high-res")
    if mjcf_content.count('<light') > 1:
        features.append("multi-light")
        
    # Check for generalization indicators  
    if 'dynamic_composition' in mjcf_content:
        features.append("dynamic")
    if 'professional_models' in mjcf_content:
        features.append("menagerie")
        
    return ", ".join(features) if features else "basic"

if __name__ == "__main__":
    asyncio.run(test_new_system_directly())