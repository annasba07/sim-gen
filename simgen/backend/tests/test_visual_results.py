#!/usr/bin/env python3
"""
Test the visual results of the new dynamic system
"""

import mujoco
import mujoco.viewer
import time

def test_visual_quality():
    """Test the visual quality of generated simulations."""
    
    print("TESTING VISUAL QUALITY OF NEW DYNAMIC SYSTEM")
    print("=" * 50)
    
    test_files = [
        "api_test_1_A_robotic_arm_pickin.xml",
        "api_test_2_A_humanoid_robot_lea.xml", 
        "api_test_3_A_bouncing_ball_with.xml"
    ]
    
    for i, filename in enumerate(test_files, 1):
        try:
            print(f"\nTEST {i}: Loading {filename}")
            
            # Load the model
            model = mujoco.MjModel.from_xml_path(filename)
            data = mujoco.MjData(model)
            
            print(f"  Model loaded: {model.nbody} bodies, {model.ngeom} geoms, {model.nlight} lights")
            print(f"  Shadow quality: {model.vis.quality.shadowsize}")
            print(f"  Off-screen samples: {model.vis.quality.offsamples}")
            
            # Quick simulation to show it works
            print("  Running 3-second simulation...")
            
            with mujoco.viewer.launch_passive(model, data) as viewer:
                start_time = time.time()
                while viewer.is_running() and (time.time() - start_time) < 3.0:
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(1.0 / 60.0)
            
            print(f"  SUCCESS: Visual simulation completed")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\nVISUAL QUALITY TEST COMPLETE")
    print("Notice the improvements:")
    print("- Professional 3-light setup with shadows")
    print("- High-resolution shadows (4096x4096)")
    print("- Specular materials with reflections")
    print("- Professional skybox and ground textures")
    print("- Atmospheric haze effects")

if __name__ == "__main__":
    test_visual_quality()