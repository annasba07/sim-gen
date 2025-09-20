#!/usr/bin/env python3
"""Create a simple test sketch for the sketch-to-physics system."""

import base64
from PIL import Image, ImageDraw
import io

def create_pendulum_sketch():
    """Create a simple pendulum sketch."""
    # Create a white canvas
    width, height = 400, 300
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw pendulum
    # Fixed point at top
    pivot_x, pivot_y = 200, 50
    draw.ellipse([pivot_x-5, pivot_y-5, pivot_x+5, pivot_y+5], fill='black')
    
    # Pendulum rod (angled)
    bob_x, bob_y = 280, 200
    draw.line([pivot_x, pivot_y, bob_x, bob_y], fill='black', width=3)
    
    # Pendulum bob (circle)
    bob_radius = 15
    draw.ellipse([bob_x-bob_radius, bob_y-bob_radius, 
                  bob_x+bob_radius, bob_y+bob_radius], fill='red')
    
    # Add some dotted arc to show swing motion
    for i in range(0, 5):
        angle_offset = i * 10 - 20
        x = 200 + 150 * (angle_offset / 100)
        y = 200
        draw.ellipse([x-3, y-3, x+3, y+3], fill='gray')
    
    return image

def create_robot_arm_sketch():
    """Create a simple robot arm sketch."""
    width, height = 400, 300
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Base
    base_x, base_y = 100, 250
    draw.rectangle([base_x-20, base_y-10, base_x+20, base_y+10], fill='gray')
    
    # First arm segment
    joint1_x, joint1_y = base_x, base_y - 80
    draw.line([base_x, base_y, joint1_x, joint1_y], fill='blue', width=8)
    draw.ellipse([joint1_x-6, joint1_y-6, joint1_x+6, joint1_y+6], fill='black')
    
    # Second arm segment 
    joint2_x, joint2_y = joint1_x + 70, joint1_y - 30
    draw.line([joint1_x, joint1_y, joint2_x, joint2_y], fill='blue', width=8)
    draw.ellipse([joint2_x-6, joint2_y-6, joint2_x+6, joint2_y+6], fill='black')
    
    # End effector/gripper
    ee_x, ee_y = joint2_x + 40, joint2_y + 20
    draw.line([joint2_x, joint2_y, ee_x, ee_y], fill='green', width=6)
    draw.rectangle([ee_x-8, ee_y-4, ee_x+8, ee_y+4], fill='red')
    
    # Object to pick up (ball)
    ball_x, ball_y = 300, 240
    draw.ellipse([ball_x-12, ball_y-12, ball_x+12, ball_y+12], fill='orange')
    
    # Table
    draw.line([250, 250, 350, 250], fill='brown', width=5)
    
    return image

def create_bouncing_balls_sketch():
    """Create a sketch of bouncing balls."""
    width, height = 400, 300
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Ground line
    draw.line([50, 250, 350, 250], fill='black', width=3)
    
    # Balls at different positions and heights
    balls = [
        (100, 200, 'red'),
        (180, 150, 'blue'), 
        (260, 220, 'green'),
        (320, 180, 'purple')
    ]
    
    for x, y, color in balls:
        # Ball
        draw.ellipse([x-15, y-15, x+15, y+15], fill=color)
        # Motion lines (indicating bouncing)
        for i in range(3):
            line_y = y + (i+1) * 20
            if line_y < 250:
                draw.line([x-5, line_y, x+5, line_y], fill=color, width=2)
    
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

if __name__ == "__main__":
    # Create test sketches
    pendulum = create_pendulum_sketch()
    robot_arm = create_robot_arm_sketch()
    bouncing_balls = create_bouncing_balls_sketch()
    
    # Save locally for inspection
    pendulum.save("test_pendulum_sketch.png")
    robot_arm.save("test_robot_arm_sketch.png") 
    bouncing_balls.save("test_bouncing_balls_sketch.png")
    
    # Convert to base64 and print
    print("=== PENDULUM SKETCH BASE64 ===")
    print(image_to_base64(pendulum))
    print("\n=== ROBOT ARM SKETCH BASE64 ===")  
    print(image_to_base64(robot_arm))
    print("\n=== BOUNCING BALLS SKETCH BASE64 ===")
    print(image_to_base64(bouncing_balls))
    
    print("\n✅ Test sketches created and saved!")