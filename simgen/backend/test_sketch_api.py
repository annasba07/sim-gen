#!/usr/bin/env python3
"""
Direct test of sketch-to-physics functionality without server setup
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simgen.services.sketch_analyzer import get_sketch_analyzer
from simgen.services.multimodal_enhancer import get_multimodal_enhancer
from simgen.services.simulation_generator import SimulationGenerator
from simgen.services.llm_client import get_llm_client
import base64


async def test_sketch_to_physics():
    """Test the complete sketch-to-physics pipeline"""
    
    # Robot arm sketch base64 (from our test generation)
    robot_arm_b64 = "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"
    
    print("Testing Sketch-to-Physics Pipeline...")
    print(f"Input: Robot arm sketch ({len(robot_arm_b64)} bytes base64)")
    
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(robot_arm_b64)
        print(f"OK: Decoded image: {len(image_bytes)} bytes")
        
        # Test text prompt
        text_prompt = "Make this robot arm pick up a red ball from the table"
        
        # Initialize services
        print("Initializing services...")
        sketch_analyzer = get_sketch_analyzer()
        multimodal_enhancer = get_multimodal_enhancer()
        llm_client = get_llm_client()
        sim_generator = SimulationGenerator(llm_client)
        print("OK: Services initialized")
        
        # Step 1: Analyze sketch
        print("\nStep 1: Analyzing sketch...")
        sketch_analysis = await sketch_analyzer.analyze_sketch(
            image_data=image_bytes,
            user_text=text_prompt
        )
        
        if sketch_analysis.success:
            print(f"SUCCESS: Sketch analysis successful!")
            print(f"   Confidence: {sketch_analysis.confidence_score:.2f}")
            print(f"   Physics description: {sketch_analysis.physics_description[:200]}...")
            if sketch_analysis.extracted_entities:
                print(f"   Objects detected: {len(sketch_analysis.extracted_entities.objects)}")
                print(f"   Constraints detected: {len(sketch_analysis.extracted_entities.constraints)}")
        else:
            print(f"ERROR: Sketch analysis failed: {sketch_analysis.error_message}")
            return
        
        # Step 2: Enhance prompt with multi-modal
        print("\nStep 2: Enhancing prompt with multi-modal fusion...")
        enhanced_result = await multimodal_enhancer.enhance_prompt(
            sketch_analysis=sketch_analysis,
            user_text=text_prompt
        )
        
        if enhanced_result.success:
            print(f"SUCCESS: Multi-modal enhancement successful!")
            print(f"   Confidence: {enhanced_result.confidence_score:.2f}")
            print(f"   Sketch contribution: {enhanced_result.sketch_contribution:.2f}")
            print(f"   Text contribution: {enhanced_result.text_contribution:.2f}")
            print(f"   Enhanced prompt: {enhanced_result.enhanced_prompt[:200]}...")
        else:
            print(f"ERROR: Multi-modal enhancement failed: {enhanced_result.error_message}")
            return
            
        # Step 3: Generate simulation
        print("\nStep 3: Generating MuJoCo simulation...")
        generation_result = await sim_generator.generate_simulation(
            entities=enhanced_result.combined_entities,
            prompt=enhanced_result.enhanced_prompt
        )
        
        if generation_result.success:
            print(f"SUCCESS: Simulation generation successful!")
            print(f"   Method: {generation_result.method.value if generation_result.method else 'unknown'}")
            print(f"   MJCF length: {len(generation_result.mjcf_content)} characters")
            print(f"   MJCF preview: {generation_result.mjcf_content[:300]}...")
        else:
            print(f"ERROR: Simulation generation failed: {generation_result.error_message}")
            return
            
        print("\nSUCCESS: Complete sketch-to-physics pipeline successful!")
        print("The sketch has been converted to a physics simulation!")
        
    except Exception as e:
        print(f"FATAL ERROR: Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting Direct Sketch-to-Physics Test")
    asyncio.run(test_sketch_to_physics())