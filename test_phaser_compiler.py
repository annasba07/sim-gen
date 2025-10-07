"""
Test the Phaser compiler with a simple platformer game.
Run this to verify the entire compilation pipeline works.
"""

import asyncio
import json
from pathlib import Path


# Sample game specification: Simple Platformer
PLATFORMER_SPEC = {
    "version": "1.0",
    "gameType": "platformer",
    "title": "Coin Collector",
    "description": "Jump around and collect all the coins!",

    "world": {
        "width": 800,
        "height": 600,
        "gravity": 800,
        "backgroundColor": "#87CEEB",
        "camera": {
            "follow": "player",
            "bounds": True
        }
    },

    "assets": {
        "sprites": [
            {"id": "player", "type": "sprite", "source": "placeholder", "width": 32, "height": 48},
            {"id": "platform", "type": "sprite", "source": "placeholder", "width": 200, "height": 32},
            {"id": "coin", "type": "sprite", "source": "placeholder", "width": 24, "height": 24},
            {"id": "flag", "type": "sprite", "source": "placeholder", "width": 48, "height": 64}
        ]
    },

    "entities": [
        {
            "id": "player",
            "type": "player",
            "sprite": "player",
            "x": 100,
            "y": 400,
            "physics": {"enabled": True, "bounce": 0.2}
        },
        {
            "id": "ground",
            "type": "platform",
            "sprite": "platform",
            "x": 400,
            "y": 568,
            "width": 800,
            "height": 32,
            "physics": {"enabled": True, "static": True}
        },
        {
            "id": "platform1",
            "type": "platform",
            "sprite": "platform",
            "x": 300,
            "y": 450,
            "physics": {"enabled": True, "static": True}
        },
        {
            "id": "platform2",
            "type": "platform",
            "sprite": "platform",
            "x": 500,
            "y": 350,
            "physics": {"enabled": True, "static": True}
        },
        {
            "id": "coin1",
            "type": "item",
            "sprite": "coin",
            "x": 300,
            "y": 400,
            "physics": {"enabled": True, "static": True}
        },
        {
            "id": "coin2",
            "type": "item",
            "sprite": "coin",
            "x": 500,
            "y": 300,
            "physics": {"enabled": True, "static": True}
        },
        {
            "id": "coin3",
            "type": "item",
            "sprite": "coin",
            "x": 700,
            "y": 500,
            "physics": {"enabled": True, "static": True}
        },
        {
            "id": "goal",
            "type": "item",
            "sprite": "flag",
            "x": 750,
            "y": 500,
            "physics": {"enabled": True, "static": True}
        }
    ],

    "behaviors": [
        {
            "id": "player_movement",
            "type": "movement_keyboard",
            "entityId": "player",
            "config": {
                "keys": "arrows",
                "speed": 200
            }
        },
        {
            "id": "player_jump",
            "type": "jump",
            "entityId": "player",
            "config": {
                "key": "space",
                "velocity": -400,
                "doubleJump": False
            }
        },
        {
            "id": "collect_coins",
            "type": "collect",
            "entityId": "player",
            "config": {
                "targets": ["coin1", "coin2", "coin3"],
                "scoreValue": 10,
                "destroyOnCollect": True
            }
        },
        {
            "id": "reach_goal",
            "type": "collect",
            "entityId": "player",
            "config": {
                "targets": ["goal"],
                "scoreValue": 50,
                "destroyOnCollect": False
            }
        }
    ],

    "mechanics": [
        {
            "type": "score_system",
            "config": {
                "initialScore": 0,
                "displayPosition": {"x": 20, "y": 20}
            }
        }
    ],

    "rules": [
        {
            "type": "win",
            "condition": {
                "type": "score_reaches",
                "value": 80
            },
            "action": "showMessage"
        }
    ],

    "ui": []
}


async def test_compiler():
    """Test the Phaser compiler with the platformer spec."""
    print("=" * 80)
    print("PHASER COMPILER TEST")
    print("=" * 80)

    try:
        # Import the compiler
        from simgen.backend.src.simgen.modes.games import PhaserCompiler

        print("\n✅ Compiler imported successfully")

        # Create compiler instance
        compiler = PhaserCompiler()
        print("✅ Compiler instance created")

        # Compile the game
        print("\n📦 Compiling game specification...")
        result = await compiler.compile(PLATFORMER_SPEC)

        # Check result
        if result["success"]:
            print("\n🎉 COMPILATION SUCCESSFUL!")

            # Print statistics
            print(f"\n📊 Statistics:")
            print(f"   - JavaScript code: {len(result['code']):,} bytes")
            print(f"   - HTML page: {len(result['html']):,} bytes")
            print(f"   - Assets: {len(result['assets'])} items")
            print(f"   - Warnings: {len(result['warnings'])}")

            if result['warnings']:
                print(f"\n⚠️  Warnings:")
                for warning in result['warnings']:
                    print(f"   - {warning}")

            # Save output files
            output_dir = Path("test_output")
            output_dir.mkdir(exist_ok=True)

            # Save JavaScript
            js_file = output_dir / "game.js"
            js_file.write_text(result["code"])
            print(f"\n💾 Saved JavaScript to: {js_file}")

            # Save HTML
            html_file = output_dir / "game.html"
            html_file.write_text(result["html"])
            print(f"💾 Saved HTML to: {html_file}")

            # Save spec
            spec_file = output_dir / "spec.json"
            spec_file.write_text(json.dumps(PLATFORMER_SPEC, indent=2))
            print(f"💾 Saved spec to: {spec_file}")

            # Print code preview
            print("\n" + "=" * 80)
            print("CODE PREVIEW (first 50 lines):")
            print("=" * 80)
            lines = result["code"].split("\n")[:50]
            for i, line in enumerate(lines, 1):
                print(f"{i:3d} | {line}")

            if len(result["code"].split("\n")) > 50:
                print(f"... ({len(result['code'].split('\n')) - 50} more lines)")

            print("\n" + "=" * 80)
            print("✅ TEST PASSED!")
            print("=" * 80)
            print(f"\n🎮 To play the game:")
            print(f"   1. Open {html_file.absolute()} in a web browser")
            print(f"   2. Use arrow keys to move")
            print(f"   3. Press SPACE to jump")
            print(f"   4. Collect all coins and reach the flag!")

            return True

        else:
            print("\n❌ COMPILATION FAILED!")
            print(f"\n🚨 Errors ({len(result['errors'])}):")
            for error in result['errors']:
                print(f"   - {error}")

            if result['warnings']:
                print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
                for warning in result['warnings']:
                    print(f"   - {warning}")

            return False

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION!")
        print(f"\n🚨 Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_validation():
    """Test validation of an invalid spec."""
    print("\n" + "=" * 80)
    print("VALIDATION TEST (Invalid Spec)")
    print("=" * 80)

    # Invalid spec: entity references unknown sprite
    invalid_spec = {
        "version": "1.0",
        "gameType": "platformer",
        "title": "Invalid Game",
        "world": {"width": 800, "height": 600, "gravity": 800, "backgroundColor": "#000000"},
        "assets": {"sprites": []},  # No sprites!
        "entities": [
            {
                "id": "player",
                "type": "player",
                "sprite": "unknown_sprite",  # This doesn't exist!
                "x": 100,
                "y": 100
            }
        ]
    }

    try:
        from simgen.backend.src.simgen.modes.games import PhaserCompiler

        compiler = PhaserCompiler()
        result = await compiler.compile(invalid_spec)

        if not result["success"]:
            print("\n✅ Validation correctly rejected invalid spec")
            print(f"\n📋 Errors detected:")
            for error in result['errors']:
                print(f"   - {error}")
            return True
        else:
            print("\n❌ Validation should have failed but didn't!")
            return False

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mode_registry():
    """Test that the compiler is registered in the mode system."""
    print("\n" + "=" * 80)
    print("MODE REGISTRY TEST")
    print("=" * 80)

    try:
        from simgen.backend.src.simgen.core.modes import mode_registry

        # Check that games mode exists
        games_mode = mode_registry.get_mode("games")
        if not games_mode:
            print("❌ Games mode not found in registry!")
            return False

        print(f"\n✅ Games mode found:")
        print(f"   - Name: {games_mode.name}")
        print(f"   - Icon: {games_mode.icon}")
        print(f"   - Enabled: {games_mode.enabled}")
        print(f"   - Features: {', '.join(games_mode.features)}")
        print(f"   - Engines: {', '.join(games_mode.engines)}")

        # Check that compiler is registered (this will happen when app starts)
        # For now, just verify the mode config exists
        print("\n✅ Mode registry test passed")
        print("   Note: Compiler registration happens on app startup")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n🚀 STARTING PHASER COMPILER TEST SUITE\n")

    results = []

    # Test 1: Mode Registry
    results.append(await test_mode_registry())

    # Test 2: Validation
    results.append(await test_validation())

    # Test 3: Full Compilation
    results.append(await test_compiler())

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")

    if all(results):
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThe Phaser compiler is working correctly!")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nCheck the output above for details.")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
