#!/usr/bin/env python3
"""
Cinematic Quality MuJoCo Template - Professional Visual Rendering
"""

def get_cinematic_mjcf_template() -> str:
    """Generate cinematic-quality MJCF template with professional visuals."""
    
    return """<mujoco model="cinematic_ai_simulation">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true" meshdir="." texturedir="."/>
  
  <option timestep="0.002" iterations="50" solver="PGS" gravity="{{ environment.gravity|join(' ') }}">
    <flag gravity="enable" contact="enable" frictionloss="disable" limit="enable"/>
  </option>
  
  <size nconmax="400" njmax="1000" nstack="10000000"/>
  
  <!-- PROFESSIONAL VISUAL ASSETS -->
  <asset>
    <!-- High-Quality Environment Textures -->
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.0 0.1 0.2" 
             width="800" height="800" mark="cross" markrgb="1 1 1"/>
    
    <!-- Premium Ground Textures -->
    <texture name="groundplane" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" 
             width="300" height="300" mark="edge" markrgb="0.8 0.8 0.8"/>
    <texture name="wood" type="2d" builtin="flat" rgb1="0.6 0.4 0.2" rgb2="0.5 0.3 0.1"
             width="200" height="200"/>
    
    <!-- Professional Materials -->
    <material name="groundplane" texture="groundplane" texuniform="true" 
              reflectance="0.3" shininess="0.1" specular="0.4"/>
    <material name="metal" rgba="0.7 0.7 0.8 1" reflectance="0.8" shininess="0.9" specular="1.0"/>
    <material name="wood" texture="wood" reflectance="0.1" shininess="0.1" specular="0.2"/>
    <material name="rubber" rgba="0.2 0.2 0.2 1" reflectance="0.05" shininess="0.0" specular="0.0"/>
    <material name="glass" rgba="0.8 0.9 1.0 0.3" reflectance="0.9" shininess="1.0" specular="1.0"/>
    
    <!-- Physics Materials -->
    <material name="pendulum_bob" rgba="0.8 0.2 0.2 1" reflectance="0.4" shininess="0.6" specular="0.8"/>
    <material name="pendulum_arm" rgba="0.6 0.6 0.8 1" reflectance="0.3" shininess="0.4" specular="0.6"/>
    <material name="ball_material" rgba="0.2 0.8 0.3 1" reflectance="0.7" shininess="0.8" specular="0.9"/>
  </asset>
  
  <!-- PROFESSIONAL LIGHTING SETUP -->
  <worldbody>
    <!-- Environment Lighting -->
    <light name="ambient" mode="ambient" diffuse="0.3 0.3 0.3" specular="0.0 0.0 0.0"/>
    <light name="sun" pos="0 0 3" dir="0 0 -1" directional="true" 
           diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="true"/>
    <light name="fill" pos="2 2 3" dir="-0.3 -0.3 -1" directional="true"
           diffuse="0.4 0.4 0.5" specular="0.1 0.1 0.1"/>
    
    <!-- Professional Ground Plane -->
    {% if environment.ground.type != "none" %}
    <geom name="ground" type="plane" size="20 20 0.1" material="groundplane" 
          pos="0 0 0" friction="0.8 0.1 0.001" condim="3" solref="0.02 1"/>
    {% endif %}
    
    {% for obj in objects %}
    {% if obj.name in ["FixedPoint", "Fixed Point"] %}
    <!-- CINEMATIC PENDULUM SETUP -->
    <body name="anchor" pos="{{ obj.position|join(' ') }}">
      <!-- Professional Anchor Point -->
      <geom name="anchor_base" type="cylinder" size="0.15 0.05" pos="0 0 -0.025" 
            material="metal" mass="0"/>
      <geom name="anchor_post" type="cylinder" size="0.03 0.2" pos="0 0 0.1"
            material="metal" mass="0"/>
      
      {% for other_obj in objects %}
      {% if other_obj.name in ["PendulumArm", "Pendulum Rod", "PendulumBob", "Pendulum Bob"] %}
      <!-- Professional Pendulum Arm -->
      <body name="pendulum_system" pos="0 0 0.2">
        <joint name="pendulum_hinge" type="hinge" axis="1 0 0" range="-160 160" 
               damping="0.01" frictionloss="0.001"/>
        
        <!-- Stylized Pendulum Rod -->
        <geom name="pendulum_rod" type="capsule" size="0.008 0.8" pos="0 0 -0.8"
              material="pendulum_arm" mass="0.1"/>
        
        <!-- Beautiful Pendulum Bob -->
        <body name="pendulum_bob" pos="0 0 -1.6">
          <geom name="bob_main" type="sphere" size="0.12" 
                material="pendulum_bob" mass="1.0"/>
          <geom name="bob_highlight" type="sphere" size="0.13" pos="0.02 0.02 0.02"
                rgba="1.0 0.6 0.6 0.3" mass="0"/>
        </body>
      </body>
      {% endif %}
      {% endfor %}
    </body>
    
    {% elif obj.geometry.shape == "sphere" and "ball" in obj.name.lower() %}
    <!-- CINEMATIC BALL PHYSICS -->
    <body name="{{ obj.name }}" pos="{{ obj.position|join(' ') }}">
      <joint name="{{ obj.name }}_freejoint" type="free"/>
      
      <!-- Professional Ball with Highlights -->
      <geom name="{{ obj.name }}_main" type="sphere" size="{{ obj.geometry.dimensions[0] }}" 
            material="ball_material" 
            density="{{ obj.material.density }}" 
            friction="{{ obj.material.friction }} 0.1 0.005" 
            solref="0.02 1" solimp="0.95 0.95 0.001"/>
      
      <!-- Ball Highlight Effect -->
      <geom name="{{ obj.name }}_highlight" type="sphere" size="{{ obj.geometry.dimensions[0] * 1.05 }}" 
            pos="0.02 0.02 0.02" rgba="1.0 1.0 1.0 0.2" mass="0"
            friction="0" solref="0.02 1" solimp="0.95 0.95 0.001"/>
    </body>
    
    {% else %}
    <!-- GENERAL CINEMATIC OBJECTS -->
    <body name="{{ obj.name }}" pos="{{ obj.position|join(' ') }}" euler="{{ obj.orientation|join(' ') }}">
      {% if obj.type == "rigid_body" %}
      <joint name="{{ obj.name }}_freejoint" type="free"/>
      {% endif %}
      
      {% if obj.geometry.shape == "box" %}
      <geom name="{{ obj.name }}_geom" type="box" 
            size="{{ (obj.geometry.dimensions[0]/2)|round(4) }} {{ (obj.geometry.dimensions[1]/2)|round(4) }} {{ (obj.geometry.dimensions[2]/2)|round(4) }}" 
            material="wood" density="{{ obj.material.density }}" 
            friction="{{ obj.material.friction }} 0.1 0.005" 
            solref="0.02 1" solimp="0.9 0.9 0.001"/>
      {% elif obj.geometry.shape == "sphere" %}
      <geom name="{{ obj.name }}_geom" type="sphere" size="{{ obj.geometry.dimensions[0] }}" 
            material="glass" density="{{ obj.material.density }}" 
            friction="{{ obj.material.friction }} 0.1 0.005" 
            solref="0.02 1" solimp="0.95 0.95 0.001"/>
      {% elif obj.geometry.shape == "cylinder" %}
      <geom name="{{ obj.name }}_geom" type="cylinder" 
            size="{{ obj.geometry.dimensions[0] }} {{ obj.geometry.dimensions[1] }}" 
            material="metal" density="{{ obj.material.density }}" 
            friction="{{ obj.material.friction }} 0.1 0.005" 
            solref="0.02 1" solimp="0.9 0.9 0.001"/>
      {% endif %}
    </body>
    {% endif %}
    {% endfor %}
    
    <!-- ATMOSPHERIC EFFECTS -->
    <body name="atmosphere" pos="0 0 0">
      <!-- Subtle particle effects (represented as tiny spheres) -->
      <geom name="dust1" type="sphere" size="0.001" pos="1 0.5 1.5" rgba="1 1 1 0.1" mass="0"/>
      <geom name="dust2" type="sphere" size="0.001" pos="-0.5 1 0.8" rgba="1 1 1 0.1" mass="0"/>
      <geom name="dust3" type="sphere" size="0.001" pos="0.8 -0.3 1.2" rgba="1 1 1 0.1" mass="0"/>
    </body>
  </worldbody>
  
  <!-- ADVANCED VISUAL SETTINGS -->
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <rgba haze="0.3 0.3 0.3 1"/>
    <quality shadowsize="2048" offsamples="8"/>
    <map force="0.1" zfar="30"/>
  </visual>
</mujoco>"""


def create_cinematic_demo():
    """Create a cinematic quality demo simulation."""
    
    cinematic_xml = """<mujoco model="cinematic_pendulum_demo">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  
  <option timestep="0.002" iterations="50" solver="PGS" gravity="0 0 -9.81">
    <flag gravity="enable" contact="enable" frictionloss="disable" limit="enable"/>
  </option>
  
  <size nconmax="400" njmax="1000" nstack="10000000"/>
  
  <asset>
    <!-- Cinematic Environment -->
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.0 0.1 0.2" 
             width="800" height="800" mark="cross" markrgb="1 1 1"/>
    <texture name="groundplane" type="2d" builtin="checker" rgb1="0.15 0.15 0.15" rgb2="0.25 0.25 0.25" 
             width="300" height="300" mark="edge" markrgb="0.8 0.8 0.8"/>
    
    <!-- Professional Materials -->
    <material name="groundplane" texture="groundplane" texuniform="true" 
              reflectance="0.4" shininess="0.2" specular="0.6"/>
    <material name="metal" rgba="0.7 0.7 0.8 1" reflectance="0.8" shininess="0.9" specular="1.0"/>
    <material name="pendulum_bob" rgba="0.9 0.2 0.2 1" reflectance="0.6" shininess="0.8" specular="0.9"/>
    <material name="pendulum_arm" rgba="0.6 0.6 0.9 1" reflectance="0.4" shininess="0.6" specular="0.7"/>
  </asset>
  
  <worldbody>
    <!-- Professional Lighting -->
    <light name="ambient" mode="ambient" diffuse="0.4 0.4 0.4"/>
    <light name="sun" pos="0 0 3" dir="0 0 -1" directional="true" 
           diffuse="1.0 1.0 0.9" specular="0.5 0.5 0.4" castshadow="true"/>
    <light name="fill" pos="3 2 2" dir="-0.5 -0.3 -1" directional="true"
           diffuse="0.3 0.3 0.4" specular="0.1 0.1 0.1"/>
    
    <!-- Cinematic Ground -->
    <geom name="ground" type="plane" size="15 15 0.1" material="groundplane" 
          pos="0 0 0" friction="0.9 0.1 0.001"/>
    
    <!-- Beautiful Pendulum System -->
    <body name="anchor" pos="0 0 2.5">
      <!-- Professional Anchor -->
      <geom name="anchor_base" type="cylinder" size="0.2 0.05" pos="0 0 -0.025" 
            material="metal"/>
      <geom name="anchor_post" type="cylinder" size="0.04 0.25" pos="0 0 0.125"
            material="metal"/>
      
      <!-- Pendulum Arm -->
      <body name="pendulum" pos="0 0 0.25">
        <joint name="pendulum_joint" type="hinge" axis="1 0 0" range="-170 170" 
               damping="0.02"/>
        
        <!-- Elegant Rod -->
        <geom name="rod" type="capsule" size="0.012 0.9" pos="0 0 -0.9"
              material="pendulum_arm"/>
        
        <!-- Stunning Bob -->
        <body name="bob" pos="0 0 -1.8">
          <geom name="bob_main" type="sphere" size="0.15" 
                material="pendulum_bob"/>
          <geom name="bob_shine" type="sphere" size="0.16" pos="0.03 0.03 0.03"
                rgba="1.0 0.8 0.8 0.3"/>
        </body>
      </body>
    </body>
  </worldbody>
  
  <!-- Cinematic Visuals -->
  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="1.0 1.0 1.0" specular="0.3 0.3 0.3"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="4096" offsamples="16"/>
    <map force="0.1" zfar="50"/>
  </visual>
</mujoco>"""
    
    return cinematic_xml


if __name__ == "__main__":
    # Save cinematic demo
    with open("cinematic_pendulum_demo.xml", "w") as f:
        f.write(create_cinematic_demo())
    print("Created cinematic_pendulum_demo.xml")