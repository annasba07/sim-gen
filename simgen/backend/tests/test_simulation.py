#!/usr/bin/env python3
"""
Direct test of simulation generation functionality.
"""

import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.dirname(__file__))

from app.services.prompt_parser import PromptParser
from app.services.simulation_generator import SimulationGenerator
from app.services.llm_client import LLMClient


async def test_simulation_generation():
    """Test the core simulation generation functionality."""
    
    print("Testing SimGen Core Functionality")
    print("=" * 50)
    
    # Test prompt
    test_prompt = "Create a simple pendulum swinging from a fixed point"
    print(f"Prompt: {test_prompt}")
    print()
    
    try:
        # Create LLM client
        print("Creating LLM client...")
        llm_client = LLMClient()
        
        # Parse entities from prompt
        print("Parsing entities from prompt...")
        prompt_parser = PromptParser(llm_client)
        entities = await prompt_parser.parse_prompt(test_prompt)
        
        print(f"   Objects found: {len(entities.objects)}")
        print(f"   Constraints found: {len(entities.constraints)}")
        print(f"   Environment: {entities.environment}")
        print()
        
        # Generate simulation
        print("Generating MJCF simulation...")
        sim_generator = SimulationGenerator(llm_client)
        result = await sim_generator.generate_simulation(entities)
        
        print(f"   Success: {result.success}")
        print(f"   Method: {result.method}")
        print(f"   MJCF length: {len(result.mjcf_content)} chars")
        
        if result.success:
            print("\nSimulation generation SUCCESS!")
            print("=" * 30)
            print("MJCF Content Preview:")
            print("-" * 30)
            # Show first 800 characters of MJCF
            preview = result.mjcf_content[:800]
            if len(result.mjcf_content) > 800:
                preview += "\n... [truncated] ..."
            print(preview)
            print("=" * 30)
        else:
            print(f"\nSimulation generation failed: {result.error_message}")
        
        return result
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(test_simulation_generation())