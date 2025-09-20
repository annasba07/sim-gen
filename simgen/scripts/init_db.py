#!/usr/bin/env python3
"""
Database initialization script for SimGen.
Run this script to set up the initial database schema and sample data.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from sqlalchemy import text
from app.db.base import async_engine, Base
from app.models.simulation import SimulationTemplate
from app.core.config import settings


async def create_tables():
    """Create all database tables."""
    print("Creating database tables...")
    
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("✓ Database tables created successfully")


async def create_sample_templates():
    """Create sample simulation templates."""
    print("Creating sample templates...")
    
    templates = [
        {
            "name": "simple_pendulum",
            "description": "A simple pendulum with adjustable mass and length",
            "category": "physics",
            "mjcf_template": """<mujoco model="simple_pendulum">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 1"/>
    
    <!-- Anchor point -->
    <body name="anchor" pos="0 0 {{ anchor_height }}">
      <geom name="anchor_geom" type="sphere" size="0.02" rgba="0.1 0.1 0.1 1"/>
      
      <!-- Pendulum mass -->
      <body name="pendulum_mass" pos="0 0 -{{ string_length }}">
        <joint name="pendulum_joint" type="hinge" axis="1 0 0"/>
        <geom name="mass_geom" type="sphere" size="{{ mass_radius }}" 
              density="{{ mass_density }}" rgba="0.8 0.2 0.2 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>""",
            "parameter_schema": {
                "type": "object",
                "properties": {
                    "anchor_height": {"type": "number", "default": 2.0, "description": "Height of anchor point (m)"},
                    "string_length": {"type": "number", "default": 1.0, "description": "Length of pendulum string (m)"},
                    "mass_radius": {"type": "number", "default": 0.1, "description": "Radius of pendulum mass (m)"},
                    "mass_density": {"type": "number", "default": 1000.0, "description": "Density of pendulum mass (kg/m³)"}
                }
            },
            "keywords": ["pendulum", "swing", "oscillation", "gravity", "simple"],
            "entity_patterns": [
                {"objects": 1, "constraints": 1, "type": "pendulum"}
            ]
        },
        
        {
            "name": "bouncing_ball",
            "description": "A ball bouncing on the ground with adjustable properties",
            "category": "physics",
            "mjcf_template": """<mujoco model="bouncing_ball">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  
  <worldbody>
    <geom name="ground" type="plane" size="10 10 0.1" 
          friction="{{ ground_friction }}" rgba="0.3 0.5 0.3 1"/>
    
    <body name="ball" pos="0 0 {{ initial_height }}">
      <joint name="ball_joint" type="free"/>
      <geom name="ball_geom" type="sphere" size="{{ ball_radius }}" 
            density="{{ ball_density }}" 
            friction="{{ ball_friction }}"
            solref="0.02 1" solimp="0.9 0.9 0.01"
            rgba="0.2 0.2 0.8 1"/>
    </body>
  </worldbody>
</mujoco>""",
            "parameter_schema": {
                "type": "object", 
                "properties": {
                    "initial_height": {"type": "number", "default": 2.0, "description": "Initial height of ball (m)"},
                    "ball_radius": {"type": "number", "default": 0.1, "description": "Radius of ball (m)"},
                    "ball_density": {"type": "number", "default": 1000.0, "description": "Density of ball (kg/m³)"},
                    "ball_friction": {"type": "number", "default": 0.5, "description": "Ball friction coefficient"},
                    "ground_friction": {"type": "number", "default": 0.8, "description": "Ground friction coefficient"}
                }
            },
            "keywords": ["ball", "bounce", "drop", "gravity", "ground"],
            "entity_patterns": [
                {"objects": 1, "constraints": 0, "type": "bouncing"}
            ]
        },
        
        {
            "name": "sliding_box",
            "description": "A box sliding down an inclined plane",
            "category": "physics",
            "mjcf_template": """<mujoco model="sliding_box">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  
  <worldbody>
    <!-- Inclined plane -->
    <body name="ramp" pos="0 0 0" euler="0 {{ ramp_angle }} 0">
      <geom name="ramp_geom" type="box" size="2 0.5 0.05" 
            friction="{{ ramp_friction }}" rgba="0.4 0.2 0.1 1"/>
    </body>
    
    <!-- Sliding box -->
    <body name="box" pos="{{ box_start_x }} 0 {{ box_start_z }}">
      <joint name="box_joint" type="free"/>
      <geom name="box_geom" type="box" 
            size="{{ box_width }} {{ box_height }} {{ box_depth }}"
            density="{{ box_density }}"
            friction="{{ box_friction }}"
            rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>""",
            "parameter_schema": {
                "type": "object",
                "properties": {
                    "ramp_angle": {"type": "number", "default": 15.0, "description": "Ramp angle in degrees"},
                    "ramp_friction": {"type": "number", "default": 0.3, "description": "Ramp friction coefficient"},
                    "box_width": {"type": "number", "default": 0.2, "description": "Box width (m)"},
                    "box_height": {"type": "number", "default": 0.2, "description": "Box height (m)"},
                    "box_depth": {"type": "number", "default": 0.2, "description": "Box depth (m)"},
                    "box_density": {"type": "number", "default": 1000.0, "description": "Box density (kg/m³)"},
                    "box_friction": {"type": "number", "default": 0.5, "description": "Box friction coefficient"},
                    "box_start_x": {"type": "number", "default": -1.5, "description": "Box starting X position"},
                    "box_start_z": {"type": "number", "default": 1.0, "description": "Box starting Z position"}
                }
            },
            "keywords": ["box", "slide", "incline", "ramp", "friction", "slope"],
            "entity_patterns": [
                {"objects": 2, "constraints": 0, "type": "sliding"}
            ]
        }
    ]
    
    # Insert templates into database
    async with async_engine.begin() as conn:
        for template_data in templates:
            query = text("""
                INSERT INTO simulation_templates 
                (name, description, category, mjcf_template, parameter_schema, keywords, entity_patterns, created_at, updated_at)
                VALUES 
                (:name, :description, :category, :mjcf_template, :parameter_schema, :keywords, :entity_patterns, :created_at, :updated_at)
                ON CONFLICT (name) DO NOTHING
            """)
            
            await conn.execute(query, {
                'name': template_data['name'],
                'description': template_data['description'], 
                'category': template_data['category'],
                'mjcf_template': template_data['mjcf_template'],
                'parameter_schema': template_data['parameter_schema'],
                'keywords': template_data['keywords'],
                'entity_patterns': template_data['entity_patterns'],
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            })
    
    print(f"✓ Created {len(templates)} sample templates")


async def verify_setup():
    """Verify the database setup."""
    print("Verifying database setup...")
    
    async with async_engine.begin() as conn:
        # Test database connectivity
        result = await conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
        
        # Check if tables exist
        tables_result = await conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """))
        
        tables = [row[0] for row in tables_result.fetchall()]
        expected_tables = ['simulations', 'simulation_templates', 'quality_assessments']
        
        for table in expected_tables:
            if table not in tables:
                raise Exception(f"Table '{table}' not found")
        
        # Check template count
        templates_result = await conn.execute(text("SELECT COUNT(*) FROM simulation_templates"))
        template_count = templates_result.scalar()
        
        print(f"✓ Database verified - {len(tables)} tables, {template_count} templates")


async def main():
    """Main initialization function."""
    print("Starting SimGen database initialization...")
    print(f"Database URL: {settings.database_url}")
    
    try:
        await create_tables()
        await create_sample_templates()
        await verify_setup()
        
        print("\n🎉 Database initialization completed successfully!")
        print("\nYou can now:")
        print("1. Start the API server: uvicorn app.main:app --reload")
        print("2. Visit http://localhost:8000/docs for API documentation")
        print("3. Try generating a simulation with the sample templates")
        
    except Exception as e:
        print(f"\n❌ Database initialization failed: {e}")
        sys.exit(1)
    
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())