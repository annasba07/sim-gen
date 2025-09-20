#!/usr/bin/env python3
"""
Visual MuJoCo Simulation Runner for AI-Generated Physics
"""

import asyncio
import sys
import os
import time
import numpy as np

# Add the backend directory to Python path
sys.path.append(os.path.dirname(__file__))

from app.services.prompt_parser import PromptParser
from app.services.simulation_generator import SimulationGenerator
from app.services.llm_client import LLMClient

try:
    import mujoco
    import mujoco.viewer
    print(f"MuJoCo {mujoco.__version__} ready for visual simulation!")
except ImportError:
    print("MuJoCo not available for visual simulation")
    sys.exit(1)


class AIPhysicsVisualizer:
    """Visualize AI-generated physics simulations using MuJoCo."""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.prompt_parser = PromptParser(self.llm_client)
        self.sim_generator = SimulationGenerator(self.llm_client)
    
    async def generate_and_simulate(self, prompt: str, duration: float = 10.0):
        """Generate MJCF from prompt and run visual simulation."""
        
        print(f"\n{'='*60}")
        print(f"AI PHYSICS SIMULATION")
        print(f"{'='*60}")
        print(f"Prompt: '{prompt}'")
        print(f"Duration: {duration}s")
        print(f"{'='*60}")
        
        # Generate simulation
        print("AI analyzing prompt...")
        entities = await self.prompt_parser.parse_prompt(prompt)
        
        print(f"AI identified:")
        print(f"   - {len(entities.objects)} physics objects")
        print(f"   - {len(entities.constraints)} constraints")
        print(f"   - Environment: {entities.environment.gravity} gravity")
        
        print("AI generating MJCF simulation...")
        result = await self.sim_generator.generate_simulation(entities)
        
        if not result.success:
            print(f"Generation failed: {result.error_message}")
            return
        
        print(f"Generated {len(result.mjcf_content)} chars of MJCF")
        print(f"   Method: {result.method}")
        
        # Save MJCF to file
        mjcf_path = f"ai_generated_{int(time.time())}.xml"
        with open(mjcf_path, 'w') as f:
            f.write(result.mjcf_content)
        
        print(f"Saved to: {mjcf_path}")
        
        # Load and run simulation
        print("Loading MuJoCo simulation...")
        try:
            model = mujoco.MjModel.from_xml_string(result.mjcf_content)
            data = mujoco.MjData(model)
            
            print("Starting visual simulation...")
            print("   -> Close the viewer window to continue")
            print("   -> Use mouse to rotate/zoom the view")
            
            # Run interactive simulation
            with mujoco.viewer.launch_passive(model, data) as viewer:
                start_time = time.time()
                
                while viewer.is_running() and (time.time() - start_time) < duration:
                    # Advance physics simulation
                    mujoco.mj_step(model, data)
                    
                    # Sync with viewer (60 FPS)
                    viewer.sync()
                    time.sleep(1.0 / 60.0)
            
            print("Simulation completed!")
            
        except Exception as e:
            print(f"Simulation error: {e}")
            
            # Show MJCF for debugging
            print("\nGenerated MJCF for debugging:")
            print("-" * 40)
            print(result.mjcf_content)
            print("-" * 40)
    
    async def run_demo_simulations(self):
        """Run a demo of multiple AI-generated simulations."""
        
        demo_prompts = [
            ("Simple pendulum swinging from a fixed point", 8.0),
            ("A bouncing ball with realistic physics", 6.0),
            ("Two spheres connected by a spring", 10.0),
            ("A spinning cube in zero gravity", 8.0)
        ]
        
        print("AI PHYSICS SIMULATION DEMO")
        print("Watch AI turn language into living physics!")
        print("\nPress Enter after each simulation to continue...")
        
        for i, (prompt, duration) in enumerate(demo_prompts, 1):
            input(f"\n[{i}/{len(demo_prompts)}] Ready for: '{prompt}'? Press Enter...")
            await self.generate_and_simulate(prompt, duration)
            print(f"\nSimulation {i} complete!")
        
        print("\nDEMO COMPLETE! AI -> Physics -> Visual Reality!")


async def main():
    """Main runner."""
    
    visualizer = AIPhysicsVisualizer()
    
    if len(sys.argv) > 1:
        # Single simulation from command line
        prompt = " ".join(sys.argv[1:])
        await visualizer.generate_and_simulate(prompt, 15.0)
    else:
        # Interactive demo
        await visualizer.run_demo_simulations()


if __name__ == "__main__":
    asyncio.run(main())