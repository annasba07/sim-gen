#!/usr/bin/env python3
"""
Test visual MuJoCo simulation with AI-generated pendulum
"""

import time
import mujoco
import mujoco.viewer
import numpy as np

def run_visual_simulation(mjcf_file, duration=10.0):
    """Run visual simulation of MJCF file."""
    
    print(f"Loading AI-Generated Physics: {mjcf_file}")
    print("=" * 50)
    
    # Load model
    model = mujoco.MjModel.from_xml_path(mjcf_file)
    data = mujoco.MjData(model)
    
    print(f"Model loaded successfully!")
    print(f"Bodies: {model.nbody}")
    print(f"Joints: {model.njnt}")
    print(f"Geoms: {model.ngeom}")
    print(f"Actuators: {model.na}")
    
    # Set initial pendulum angle (45 degrees)
    if model.njnt > 0:
        data.qpos[0] = np.pi/4  # 45 degrees
        print("Set initial pendulum angle: 45 degrees")
    
    print("\nStarting visual simulation...")
    print("-> Use mouse to rotate/zoom the view")
    print("-> Close viewer window when finished")
    print("=" * 50)
    
    # Run interactive visual simulation
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running() and (time.time() - start_time) < duration:
            # Step physics simulation
            mujoco.mj_step(model, data)
            
            # Sync with viewer at 60 FPS
            viewer.sync()
            time.sleep(1.0 / 60.0)
    
    print("AI-Generated Physics Simulation Complete!")

if __name__ == "__main__":
    run_visual_simulation("fixed_pendulum.xml", 15.0)