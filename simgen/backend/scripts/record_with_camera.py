#!/usr/bin/env python3
"""
Record simulations with proper camera positioning to show the visual quality
"""

import mujoco
import numpy as np
import imageio
import time
import os

def record_simulation_with_camera(xml_file, output_name, duration=5.0, fps=30):
    """Record a simulation with proper camera setup."""
    
    try:
        print(f"Recording {xml_file} with optimized camera...")
        
        # Load model
        model = mujoco.MjModel.from_xml_path(xml_file)
        data = mujoco.MjData(model)
        
        # Create renderer with higher resolution for quality
        renderer = mujoco.Renderer(model, height=720, width=1280)
        
        # Setup camera for good viewing angle
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        
        # Position camera to see the scene
        camera.distance = 4.0  # Distance from target
        camera.elevation = -20  # Look down slightly
        camera.azimuth = 45     # Angled view
        camera.lookat[0] = 0.0  # Look at origin
        camera.lookat[1] = 0.0
        camera.lookat[2] = 0.5  # Look slightly up from ground
        
        # Setup frames list
        frames = []
        total_frames = int(duration * fps)
        dt = 1.0 / fps
        
        print(f"  Capturing {total_frames} frames at {fps}fps with enhanced visuals...")
        
        # Let physics settle for a moment
        for _ in range(100):
            mujoco.mj_step(model, data)
        
        # Record frames
        for frame_idx in range(total_frames):
            # Step simulation
            for _ in range(int(0.002 / dt)):  # Use model timestep
                mujoco.mj_step(model, data)
            
            # Update camera if we want dynamic movement
            if frame_idx > 0:
                # Slowly rotate camera for cinematic effect
                camera.azimuth += 0.5
            
            # Render frame with camera
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            frames.append(pixels.copy())
            
            if frame_idx % 30 == 0:  # Progress update every second
                progress = (frame_idx / total_frames) * 100
                print(f"    Progress: {progress:.1f}%")
        
        # Save as MP4
        output_file = f"{output_name}_hd.mp4"
        imageio.mimsave(output_file, frames, fps=fps, quality=9)
        
        print(f"  SUCCESS: Saved {output_file}")
        print(f"  Video info: {len(frames)} frames, {duration}s duration, 1280x720 HD")
        
        # Save multiple screenshots at different times
        for i, frame_time in enumerate([0.25, 0.5, 0.75]):
            frame_idx = int(frame_time * len(frames))
            screenshot_file = f"{output_name}_screenshot_{i+1}.png"
            imageio.imwrite(screenshot_file, frames[frame_idx])
            print(f"  Screenshot {i+1}: {screenshot_file}")
        
        return True, len(frames)
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, 0

def analyze_simulation_content(xml_file):
    """Analyze what's actually in the simulation."""
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_file)
        data = mujoco.MjData(model)
        
        print(f"  Model analysis:")
        print(f"    Bodies: {model.nbody} (including world)")
        print(f"    Geoms: {model.ngeom}")
        print(f"    Lights: {model.nlight}")
        print(f"    Joints: {model.njnt}")
        
        # Print body positions
        for i in range(1, model.nbody):  # Skip world body
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            pos = data.xpos[i]
            print(f"    Body '{body_name}': position {pos}")
            
    except Exception as e:
        print(f"  Analysis error: {e}")

def main():
    """Record all simulations with proper camera work."""
    
    print("RECORDING PROFESSIONAL SIMULATIONS WITH ENHANCED VISUALS")
    print("=" * 60)
    
    # Test cases to record
    recordings = [
        {
            'file': 'api_test_1_A_robotic_arm_pickin.xml',
            'name': 'robotic_arm_hd',
            'description': 'Robotic arm with objects and table'
        },
        {
            'file': 'api_test_3_A_bouncing_ball_with.xml',
            'name': 'bouncing_ball_hd', 
            'description': 'Professional bouncing ball physics'
        },
        {
            'file': 'api_test_4_Multiple_spheres_fal.xml',
            'name': 'multiple_spheres_hd',
            'description': 'Multiple spheres collision demo'
        }
    ]
    
    successful_recordings = []
    total_frames = 0
    
    for i, recording in enumerate(recordings, 1):
        print(f"\nRECORDING {i}/{len(recordings)}: {recording['description']}")
        
        # First analyze what's in the simulation
        analyze_simulation_content(recording['file'])
        
        # Then record it
        success, frame_count = record_simulation_with_camera(
            recording['file'], 
            recording['name'], 
            duration=8.0,  # Longer for better showcase
            fps=30
        )
        
        if success:
            successful_recordings.append(recording['name'])
            total_frames += frame_count
    
    print(f"\n" + "=" * 60)
    print("HD RECORDING COMPLETE!")
    print(f"Successfully recorded {len(successful_recordings)}/{len(recordings)} simulations")
    print(f"Total frames captured: {total_frames}")
    
    if successful_recordings:
        print(f"\nGenerated HD files:")
        for name in successful_recordings:
            print(f"  - {name}.mp4 (1280x720 HD video)")
            print(f"  - {name}_screenshot_1.png, _2.png, _3.png (multiple angles)")
        
        print(f"\nPROFESSIONAL QUALITY FEATURES:")
        print("- 1280x720 HD resolution")
        print("- 30fps smooth motion")
        print("- Cinematic camera movement")
        print("- Professional 3-light setup with shadows")
        print("- High-res 4096x4096 shadow mapping")
        print("- Specular materials with realistic reflections")
        print("- Professional skybox and ground textures")

if __name__ == "__main__":
    main()