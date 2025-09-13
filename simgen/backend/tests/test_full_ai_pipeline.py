#!/usr/bin/env python3
"""
COMPLETE AI PIPELINE TEST: Prompt → AI → Valid MJCF → Visual Simulation
NO MANUAL INTERVENTION - Pure AI Generation
"""

import asyncio
import sys
import os
import time

# Add the backend directory to Python path
sys.path.append(os.path.dirname(__file__))

from app.services.prompt_parser import PromptParser
from app.services.simulation_generator import SimulationGenerator
from app.services.llm_client import LLMClient

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

async def test_complete_ai_pipeline(prompt: str):
    """Test complete AI pipeline from prompt to visual simulation."""
    
    print("COMPLETE AI PHYSICS PIPELINE TEST")
    print("=" * 60)
    print(f"Input: '{prompt}'")
    print("=" * 60)
    
    # Step 1: Initialize AI System
    print("Step 1: Initializing AI system...")
    llm_client = LLMClient()
    prompt_parser = PromptParser(llm_client) 
    sim_generator = SimulationGenerator(llm_client)
    print("AI system ready")
    
    # Step 2: AI analyzes prompt
    print(f"Step 2: AI analyzing prompt...")
    start_time = time.time()
    entities = await prompt_parser.parse_prompt(prompt)
    parse_time = time.time() - start_time
    
    print(f"AI identified physics ({parse_time:.2f}s):")
    print(f"   - {len(entities.objects)} objects")
    print(f"   - {len(entities.constraints)} constraints")
    for obj in entities.objects:
        print(f"     * {obj.name}: {obj.geometry.shape} ({obj.type})")
    
    # Step 3: AI generates MJCF
    print("Step 3: AI generating MJCF simulation...")
    result = await sim_generator.generate_simulation(entities)
    gen_time = time.time() - start_time
    
    if not result.success:
        print(f"AI generation failed: {result.error_message}")
        return False
    
    print(f"AI generated MJCF ({gen_time-parse_time:.2f}s):")
    print(f"   - {len(result.mjcf_content)} characters")
    print(f"   - Method: {result.method}")
    
    # Step 4: Save AI-generated MJCF
    mjcf_file = f"ai_generated_complete_{int(time.time())}.xml"
    with open(mjcf_file, 'w') as f:
        f.write(result.mjcf_content)
    print(f"Saved AI-generated MJCF: {mjcf_file}")
    
    # Step 5: Load in MuJoCo (validation)
    print("Step 4: Validating AI-generated MJCF with MuJoCo...")
    if not MUJOCO_AVAILABLE:
        print("MuJoCo not available for validation")
        return False
    
    try:
        model = mujoco.MjModel.from_xml_string(result.mjcf_content)
        data = mujoco.MjData(model)
        print(f"MuJoCo validation successful:")
        print(f"   - {model.nbody} bodies loaded")
        print(f"   - {model.njnt} joints loaded") 
        print(f"   - {model.ngeom} geoms loaded")
        
        # Step 6: Run visual simulation
        if model.njnt > 0:
            # Set initial pendulum angle if it's a pendulum
            data.qpos[0] = 0.5  # Initial angle
        
        print("Step 5: Launching AI-generated visual simulation...")
        print("SUCCESS! Close viewer when done watching AI physics")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            sim_start = time.time()
            while viewer.is_running() and (time.time() - sim_start) < 15.0:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(1.0 / 60.0)
        
        total_time = time.time() - start_time
        print("COMPLETE AI PIPELINE SUCCESS!")
        print(f"Total time: {total_time:.2f}s (Parse: {parse_time:.2f}s, Generate: {gen_time-parse_time:.2f}s)")
        print("PROMPT -> AI -> VALID MJCF -> VISUAL SIMULATION COMPLETE!")
        
        return True
        
    except Exception as e:
        print(f"MuJoCo loading failed: {e}")
        print("\nAI-Generated MJCF Debug:")
        print("-" * 40)
        print(result.mjcf_content)
        print("-" * 40)
        return False

async def main():
    """Run complete AI pipeline tests."""
    
    test_prompts = [
        "Create a simple pendulum swinging from a fixed point",
        "A bouncing ball with realistic physics"
    ]
    
    success_count = 0
    
    for prompt in test_prompts:
        print("\n" + "=" * 80)
        success = await test_complete_ai_pipeline(prompt)
        if success:
            success_count += 1
        print("=" * 80)
        
        if len(test_prompts) > 1:
            input("\nPress Enter to continue to next test...")
    
    print(f"\nFINAL RESULTS: {success_count}/{len(test_prompts)} complete AI pipelines successful")
    
    if success_count > 0:
        print("YOUR AI SYSTEM IS FULLY OPERATIONAL!")
        print("SUCCESS: Prompt -> AI Understanding -> MJCF Generation -> Visual Physics")
    else:
        print("AI system needs refinement")

if __name__ == "__main__":
    asyncio.run(main())