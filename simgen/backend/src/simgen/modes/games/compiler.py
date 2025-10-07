"""
Main Phaser compiler that orchestrates the compilation pipeline.
Implements the ModeCompiler protocol for integration with VirtualForge.
"""

import logging
from typing import Dict, List, Tuple, Union

from .parser import GameSpecParser
from .codegen import PhaserCodeGenerator
from .assets import AssetManager
from .models import GameSpec, CompilationResult

logger = logging.getLogger(__name__)


class PhaserCompiler:
    """
    Compiles game specifications to Phaser 3 code.
    Implements the ModeCompiler protocol for VirtualForge integration.

    Pipeline:
        1. Parse & validate game spec (JSON → Pydantic models)
        2. Process assets (validate, generate placeholders)
        3. Generate Phaser 3 JavaScript code
        4. Generate complete HTML wrapper
        5. Return compilation result
    """

    def __init__(self):
        self.parser = GameSpecParser()
        self.codegen = PhaserCodeGenerator()
        self.assets = AssetManager()
        logger.info("PhaserCompiler initialized")

    async def compile(self, spec: Union[dict, str], options: dict = None) -> dict:
        """
        Compile a game specification to Phaser 3 code.

        Args:
            spec: Game specification (dict or JSON string)
            options: Compilation options
                - minify: bool = False (minify output)
                - include_phaser: bool = True (include Phaser library)
                - debug: bool = False (enable Phaser debug mode)

        Returns:
            dict: {
                "success": bool,
                "code": str,              # JavaScript code
                "html": str,              # Complete HTML page
                "assets": [...],          # Asset metadata
                "errors": [...],          # Compilation errors
                "warnings": [...]         # Compilation warnings
            }
        """
        options = options or {}
        logger.info("Starting compilation...")

        # Step 1: Parse and validate
        game_spec, is_valid = await self.parser.parse(spec)

        if not is_valid:
            logger.error(f"Validation failed with {len(self.parser.errors)} errors")
            return {
                "success": False,
                "code": "",
                "html": "",
                "assets": [],
                "errors": self.parser.errors,
                "warnings": self.parser.warnings
            }

        logger.info(f"Spec validated: {game_spec.title} ({game_spec.gameType})")

        # Step 2: Process assets
        try:
            asset_metadata = await self.assets.process(game_spec)
            logger.info(f"Processed {len(asset_metadata)} assets")
        except Exception as e:
            logger.error(f"Asset processing failed: {e}")
            return {
                "success": False,
                "code": "",
                "html": "",
                "assets": [],
                "errors": [f"Asset processing error: {str(e)}"],
                "warnings": self.parser.warnings
            }

        # Step 3: Generate Phaser code
        try:
            js_code = await self.codegen.generate(game_spec)
            logger.info(f"Generated {len(js_code)} bytes of JavaScript")
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                "success": False,
                "code": "",
                "html": "",
                "assets": asset_metadata,
                "errors": [f"Code generation error: {str(e)}"],
                "warnings": self.parser.warnings
            }

        # Step 4: Generate HTML wrapper
        try:
            html = self._generate_html(js_code, game_spec, options)
            logger.info(f"Generated {len(html)} bytes of HTML")
        except Exception as e:
            logger.error(f"HTML generation failed: {e}")
            return {
                "success": False,
                "code": js_code,
                "html": "",
                "assets": asset_metadata,
                "errors": [f"HTML generation error: {str(e)}"],
                "warnings": self.parser.warnings
            }

        # Step 5: Optional minification
        if options.get("minify", False):
            try:
                js_code = self._minify(js_code)
                logger.info("Code minified")
            except Exception as e:
                logger.warning(f"Minification failed: {e}")
                # Non-fatal, continue with unminified code

        # Success!
        logger.info("Compilation successful!")
        return {
            "success": True,
            "code": js_code,
            "html": html,
            "assets": asset_metadata,
            "errors": [],
            "warnings": self.parser.warnings
        }

    async def validate(self, spec: Union[dict, str]) -> Tuple[bool, List[str]]:
        """
        Validate a game specification without compiling.

        Args:
            spec: Game specification (dict or JSON string)

        Returns:
            (is_valid, errors)
        """
        _, is_valid = await self.parser.parse(spec)
        return is_valid, self.parser.errors

    def _generate_html(self, js_code: str, spec: GameSpec, options: dict) -> str:
        """
        Generate complete HTML page with embedded game.

        Args:
            js_code: Generated JavaScript code
            spec: Game specification
            options: Compilation options

        Returns:
            Complete HTML document
        """
        title = spec.title or "Phaser Game"
        include_phaser = options.get("include_phaser", True)

        # Phaser library (CDN or local)
        if include_phaser:
            phaser_script = '<script src="https://cdn.jsdelivr.net/npm/phaser@3.80.1/dist/phaser.min.js"></script>'
        else:
            phaser_script = "<!-- Phaser library should be included externally -->"

        # Debug mode
        debug_note = ""
        if options.get("debug", False):
            debug_note = "\n        // Debug mode enabled - check console for logs"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <meta name="description" content="{spec.description or 'A game created with VirtualForge'}">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        }}

        #game-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }}

        canvas {{
            display: block;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }}

        .game-info {{
            text-align: center;
            color: white;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }}

        .game-info h1 {{
            font-size: 2em;
            margin-bottom: 0.5em;
        }}

        .game-info p {{
            font-size: 1em;
            opacity: 0.9;
        }}

        .powered-by {{
            margin-top: 20px;
            font-size: 0.9em;
            opacity: 0.7;
        }}

        .powered-by a {{
            color: white;
            text-decoration: none;
            font-weight: bold;
        }}

        .powered-by a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div id="game-container">
        <div class="game-info">
            <h1>{title}</h1>
            {f'<p>{spec.description}</p>' if spec.description else ''}
        </div>

        <!-- Game canvas will be inserted here by Phaser -->

        <div class="powered-by">
            Powered by <a href="https://virtualforge.ai" target="_blank">VirtualForge</a>
        </div>
    </div>

    {phaser_script}

    <script>{debug_note}
{js_code}
    </script>
</body>
</html>"""

    def _minify(self, code: str) -> str:
        """
        Minify JavaScript code (basic implementation).

        For production, use a proper minifier like terser.
        This is a simple version for MVP.

        Args:
            code: JavaScript code

        Returns:
            Minified code
        """
        import re

        # Remove single-line comments
        code = re.sub(r'//.*?\n', '\n', code)

        # Remove multi-line comments (preserve URLs)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # Remove extra whitespace (but preserve strings)
        # This is simplified - a real minifier would be more careful
        code = re.sub(r'\s+', ' ', code)

        # Remove spaces around operators
        code = re.sub(r'\s*([{}();,:])\s*', r'\1', code)

        return code.strip()

    def get_compiler_info(self) -> dict:
        """Get information about this compiler."""
        return {
            "name": "PhaserCompiler",
            "version": "1.0.0",
            "engine": "Phaser 3",
            "supported_game_types": ["platformer", "topdown", "puzzle", "shooter"],
            "supported_behaviors": [
                "movement_keyboard",
                "jump",
                "collect",
                "shoot",
                "follow",
                "patrol",
                "destroy_offscreen"
            ],
            "supported_mechanics": [
                "score_system",
                "health_system",
                "timer",
                "spawn_system"
            ]
        }
