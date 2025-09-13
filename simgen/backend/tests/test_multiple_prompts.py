#!/usr/bin/env python3
"""
Test multiple physics simulation prompts to demonstrate AI capabilities.
"""

import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.dirname(__file__))

from app.services.prompt_parser import PromptParser
from app.services.simulation_generator import SimulationGenerator
from app.services.llm_client import LLMClient


async def test_multiple_simulations():
    """Test various physics simulation prompts."""
    
    print("AI-POWERED PHYSICS SIMULATION GENERATOR")
    print("=" * 60)
    
    test_prompts = [
        "A bouncing ball with realistic physics",
        "Two spheres connected by a spring",
        "A spinning cube in zero gravity",
        "A chain of connected pendulums"
    ]
    
    # Create services
    llm_client = LLMClient()
    prompt_parser = PromptParser(llm_client)
    sim_generator = SimulationGenerator(llm_client)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/4] PROMPT: '{prompt}'")
        print("-" * 50)
        
        try:
            # Parse and generate
            entities = await prompt_parser.parse_prompt(prompt)
            result = await sim_generator.generate_simulation(entities)
            
            if result.success:
                print(f"SUCCESS! Generated {len(result.mjcf_content)} chars")
                print(f"   Objects: {len(entities.objects)}")
                print(f"   Constraints: {len(entities.constraints)}")
                print(f"   Method: {result.method}")
                
                # Show physics summary
                print("   Physics Objects:")
                for j, obj in enumerate(entities.objects):
                    print(f"     {j+1}. {obj.name} ({obj.geometry.shape}) - {obj.type}")
                
                if entities.constraints:
                    print("   Constraints:")
                    for j, const in enumerate(entities.constraints):
                        print(f"     {j+1}. {const.type} connecting {const.bodies}")
                
            else:
                print(f"FAILED: {result.error_message}")
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("AI SIMULATION GENERATION COMPLETE!")


if __name__ == "__main__":
    asyncio.run(test_multiple_simulations())