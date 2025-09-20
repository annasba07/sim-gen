#!/usr/bin/env python3
"""
Simple database initialization script for SimGen.
"""

import sys
import os
from datetime import datetime

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from sqlalchemy import create_engine, text
from app.models.simulation import Base
from app.core.config import settings


def create_tables():
    """Create all database tables."""
    print("Creating database tables...")
    
    # Use sync engine for simple initialization
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    
    print("✓ Database tables created successfully")


def create_sample_templates():
    """Create sample simulation templates."""
    print("Creating sample templates...")
    
    engine = create_engine(settings.database_url)
    
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
        }
    ]
    
    # Insert templates into database
    with engine.begin() as conn:
        for template_data in templates:
            # Convert Python objects to JSON strings
            import json
            
            query = text("""
                INSERT INTO simulation_templates 
                (name, description, category, mjcf_template, parameter_schema, keywords, entity_patterns, created_at, updated_at)
                VALUES 
                (:name, :description, :category, :mjcf_template, :parameter_schema, :keywords, :entity_patterns, :created_at, :updated_at)
                ON CONFLICT (name) DO NOTHING
            """)
            
            conn.execute(query, {
                'name': template_data['name'],
                'description': template_data['description'], 
                'category': template_data['category'],
                'mjcf_template': template_data['mjcf_template'],
                'parameter_schema': json.dumps(template_data['parameter_schema']),
                'keywords': json.dumps(template_data['keywords']),
                'entity_patterns': json.dumps(template_data['entity_patterns']),
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            })
    
    print(f"✓ Created {len(templates)} sample templates")


def verify_setup():
    """Verify the database setup."""
    print("Verifying database setup...")
    
    engine = create_engine(settings.database_url)
    
    with engine.begin() as conn:
        # Test database connectivity
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
        
        # Check if tables exist
        tables_result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """))
        
        tables = [row[0] for row in tables_result.fetchall()]
        expected_tables = ['simulations', 'simulation_templates', 'quality_assessments']
        
        for table in expected_tables:
            if table not in tables:
                print(f"Warning: Table '{table}' not found")
        
        # Check template count
        templates_result = conn.execute(text("SELECT COUNT(*) FROM simulation_templates"))
        template_count = templates_result.scalar()
        
        print(f"✓ Database verified - {len(tables)} tables, {template_count} templates")


def main():
    """Main initialization function."""
    print("Starting SimGen database initialization...")
    print(f"Database URL: {settings.database_url}")
    
    try:
        create_tables()
        create_sample_templates()
        verify_setup()
        
        print("\n🎉 Database initialization completed successfully!")
        print("\nYou can now:")
        print("1. Start the API server: uvicorn app.main:app --reload")
        print("2. Visit http://localhost:8000/docs for API documentation")
        print("3. Try generating a simulation with the sample templates")
        
    except Exception as e:
        print(f"\n❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()