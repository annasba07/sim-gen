#!/usr/bin/env python3
"""
Record simulations with standard resolution and better object positioning
"""

import mujoco
import numpy as np
import imageio
import time

def create_better_simulation():
    """Create a better demonstration simulation with proper positioning."""
    
    better_xml = """<mujoco model="professional_demo">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  
  <option timestep="0.002" iterations="50" solver="PGS" gravity="0 0 -9.81"/>
  
  <asset>
    <!-- Professional skybox -->
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.0 0.1 0.2" width="800" height="800"/>
    <texture name="groundplane" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="300" height="300"/>
    
    <!-- High-quality materials -->
    <material name="groundplane" texture="groundplane" reflectance="0.4" shininess="0.2" specular="0.6"/>
    <material name="metal" rgba="0.7 0.7 0.8 1" reflectance="0.8" shininess="0.9" specular="1.0"/>
    <material name="red_ball" rgba="0.9 0.2 0.2 1" reflectance="0.6" shininess="0.8" specular="0.9"/>
    <material name="blue_ball" rgba="0.2 0.2 0.9 1" reflectance="0.6" shininess="0.8" specular="0.9"/>
    <material name="green_ball" rgba="0.2 0.9 0.2 1" reflectance="0.6" shininess="0.8" specular="0.9"/>
  </asset>
  
  <worldbody>
    <!-- Professional lighting setup -->
    <light name="sun" pos="2 2 4" dir="-0.3 -0.3 -1" directional="true" 
           diffuse="1.0 1.0 0.9" specular="0.5 0.5 0.4" castshadow="true"/>
    <light name="fill1" pos="-2 3 2" dir="0.3 -0.5 -1" directional="true"
           diffuse="0.3 0.3 0.4" specular="0.1 0.1 0.1"/>
    <light name="fill2" pos="3 -2 1" dir="-0.5 0.3 -0.5" directional="true"
           diffuse="0.2 0.2 0.3" specular="0.05 0.05 0.05"/>
    
    <!-- Professional ground -->
    <geom name="ground" type="plane" size="5 5 0.1" material="groundplane" friction="0.9 0.1 0.001"/>
    
    <!-- Demonstration objects with proper positioning -->
    <body name="table" pos="0 0 0.4">
      <geom name="table_top" type="box" size="1.2 0.8 0.05" material="metal" pos="0 0 0"/>
      <geom name="leg1" type="cylinder" size="0.05 0.4" material="metal" pos="1.0 0.6 -0.4"/>
      <geom name="leg2" type="cylinder" size="0.05 0.4" material="metal" pos="-1.0 0.6 -0.4"/>
      <geom name="leg3" type="cylinder" size="0.05 0.4" material="metal" pos="1.0 -0.6 -0.4"/>
      <geom name="leg4" type="cylinder" size="0.05 0.4" material="metal" pos="-1.0 -0.6 -0.4"/>
    </body>
    
    <!-- Bouncing balls with different materials -->
    <body name="red_ball" pos="0.5 0.3 2.0">
      <joint name="red_ball_free" type="free"/>
      <geom name="red_ball_geom" type="sphere" size="0.1" material="red_ball" density="1000"/>
      <geom name="red_ball_highlight" type="sphere" size="0.105" pos="0.02 0.02 0.02" rgba="1.0 0.8 0.8 0.3"/>
    </body>
    
    <body name="blue_ball" pos="-0.3 0.1 2.5">
      <joint name="blue_ball_free" type="free"/>
      <geom name="blue_ball_geom" type="sphere" size="0.12" material="blue_ball" density="1200"/>
      <geom name="blue_ball_highlight" type="sphere" size="0.125" pos="0.02 0.02 0.02" rgba="0.8 0.8 1.0 0.3"/>
    </body>
    
    <body name="green_ball" pos="0.2 -0.4 3.0">
      <joint name="green_ball_free" type="free"/>
      <geom name="green_ball_geom" type="sphere" size="0.08" material="green_ball" density="800"/>
      <geom name="green_ball_highlight" type="sphere" size="0.085" pos="0.02 0.02 0.02" rgba="0.8 1.0 0.8 0.3"/>
    </body>
    
    <!-- Robotic arm representation -->
    <body name="robot_base" pos="-2 0 0.1">
      <geom name="base" type="cylinder" size="0.2 0.1" material="metal"/>
      
      <body name="arm_segment1" pos="0 0 0.1">
        <joint name="shoulder" type="hinge" axis="0 0 1" range="-180 180"/>
        <geom name="segment1" type="capsule" size="0.05 0.4" pos="0.4 0 0" material="metal"/>
        
        <body name="arm_segment2" pos="0.8 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-90 90"/>
          <geom name="segment2" type="capsule" size="0.04 0.3" pos="0.3 0 0" material="metal"/>
          
          <body name="end_effector" pos="0.6 0 0">
            <geom name="gripper" type="box" size="0.05 0.02 0.05" material="metal"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <!-- Professional visual settings -->
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="1.0 1.0 1.0" specular="0.3 0.3 0.3"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="4096" offsamples="16"/>
    <map force="0.1" zfar="50"/>
    <global offwidth="640" offheight="480"/>
  </visual>
</mujoco>"""
    
    with open("professional_demo.xml", "w") as f:
        f.write(better_xml)
    
    return "professional_demo.xml"

def record_professional_demo():
    """Record the professional quality demo."""
    
    print("CREATING & RECORDING PROFESSIONAL QUALITY DEMO")
    print("=" * 50)
    
    # Create the demo simulation
    demo_file = create_better_simulation()
    print(f"Created: {demo_file}")
    
    try:
        # Load model
        model = mujoco.MjModel.from_xml_path(demo_file)
        data = mujoco.MjData(model)
        
        print(f"Model loaded: {model.nbody} bodies, {model.ngeom} geoms, {model.nlight} lights")
        
        # Create renderer with standard resolution
        renderer = mujoco.Renderer(model, height=480, width=640)
        
        # Setup camera for cinematic view
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 6.0
        camera.elevation = -25
        camera.azimuth = 130
        camera.lookat[0] = 0.0
        camera.lookat[1] = 0.0
        camera.lookat[2] = 1.0
        
        # Record frames
        frames = []
        duration = 8.0
        fps = 30
        total_frames = int(duration * fps)
        
        print(f"Recording {total_frames} frames showing professional quality...")
        
        # Let physics settle
        for _ in range(50):
            mujoco.mj_step(model, data)
        
        # Record with camera movement
        for frame_idx in range(total_frames):
            # Step simulation
            for _ in range(2):
                mujoco.mj_step(model, data)
            
            # Slowly rotate camera for cinematic effect
            camera.azimuth += 0.3
            camera.elevation = -25 + 5 * np.sin(frame_idx * 0.02)
            
            # Render frame
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            frames.append(pixels.copy())
            
            if frame_idx % 60 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"  Progress: {progress:.1f}%")
        
        # Save video
        output_file = "professional_quality_demo.mp4"
        imageio.mimsave(output_file, frames, fps=fps, quality=9)
        print(f"SUCCESS: Saved {output_file}")
        
        # Save multiple screenshots
        for i, time_point in enumerate([0.1, 0.4, 0.7]):
            frame_idx = int(time_point * len(frames))
            screenshot = f"professional_demo_shot_{i+1}.png"
            imageio.imwrite(screenshot, frames[frame_idx])
            print(f"Screenshot: {screenshot}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = record_professional_demo()
    
    if success:
        print(f"\n" + "=" * 50)
        print("PROFESSIONAL DEMO COMPLETE!")
        print("Generated files:")
        print("  - professional_quality_demo.mp4 (cinematic 8-second demo)")
        print("  - professional_demo_shot_1.png, _2.png, _3.png (screenshots)")
        print(f"\nQUALITY FEATURES SHOWCASED:")
        print("✓ Multi-object physics (balls, table, robotic arm)")
        print("✓ Professional 3-light setup with realistic shadows")
        print("✓ High-resolution 4096x4096 shadow mapping")  
        print("✓ Specular materials with realistic reflections")
        print("✓ Professional skybox and ground textures")
        print("✓ Cinematic camera movement")
        print("✓ Smooth 30fps physics simulation")
    else:
        print("Demo recording failed!")