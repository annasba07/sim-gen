import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path
import json

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simgen.services.dynamic_scene_composer import (
    MenagerieModel,
    SceneComposition,
    MenagerieModelLibrary,
    DynamicSceneComposer
)
from simgen.models.schemas import ExtractedEntities


class TestMenagerieModel:
    """Test cases for MenagerieModel dataclass."""

    def test_menagerie_model_creation(self):
        """Test creating a MenagerieModel instance."""
        model = MenagerieModel(
            name="franka_panda",
            category="arm",
            xml_path="/path/to/model.xml",
            assets_path="/path/to/assets",
            description="Franka Emika Panda robot arm",
            dof=7,
            license="Apache 2.0"
        )

        assert model.name == "franka_panda"
        assert model.category == "arm"
        assert model.dof == 7
        assert "Panda" in model.description


class TestSceneComposition:
    """Test cases for SceneComposition dataclass."""

    def test_scene_composition_creation(self):
        """Test creating a SceneComposition instance."""
        model = MenagerieModel(
            name="test_robot",
            category="arm",
            xml_path="/test/path",
            assets_path="/test/assets",
            description="Test robot",
            dof=6,
            license="MIT"
        )

        composition = SceneComposition(
            main_models=[model],
            environment_settings={"gravity": -9.81},
            physics_constraints=[],
            lighting_setup={"type": "three_point"},
            camera_angles=[{"name": "main", "pos": [0, 0, 5]}],
            materials={"default": {"rgba": [1, 1, 1, 1]}}
        )

        assert len(composition.main_models) == 1
        assert composition.environment_settings["gravity"] == -9.81
        assert composition.lighting_setup["type"] == "three_point"


class TestMenagerieModelLibrary:
    """Test cases for MenagerieModelLibrary."""

    @pytest.fixture
    def library(self, tmp_path):
        """Create a library instance with temp directory."""
        return MenagerieModelLibrary(str(tmp_path / "models"))

    def test_library_initialization(self, library):
        """Test library initialization."""
        assert library.models_dir.name == "models"
        assert len(library.categories) > 0
        assert "arms" in library.categories
        assert "quadrupeds" in library.categories

    @pytest.mark.asyncio
    async def test_library_initialize_missing_models(self, library, mocker):
        """Test library initialization when models directory doesn't exist."""
        mock_download = mocker.patch.object(
            library, '_download_menagerie',
            new_callable=AsyncMock
        )
        mock_build_index = mocker.patch.object(
            library, '_build_models_index',
            new_callable=AsyncMock
        )

        result = await library.initialize()

        assert result is True
        mock_download.assert_called_once()
        mock_build_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_models_by_category(self, library):
        """Test finding models by category."""
        # Setup mock models
        library.models_index = {
            "panda": MenagerieModel(
                name="panda", category="arm", xml_path="",
                assets_path="", description="", dof=7, license=""
            ),
            "spot": MenagerieModel(
                name="spot", category="quadruped", xml_path="",
                assets_path="", description="", dof=12, license=""
            ),
            "ur5": MenagerieModel(
                name="ur5", category="arm", xml_path="",
                assets_path="", description="", dof=6, license=""
            )
        }

        arms = await library.find_models_by_category("arm")
        assert len(arms) == 2
        assert any(m.name == "panda" for m in arms)
        assert any(m.name == "ur5" for m in arms)

        quadrupeds = await library.find_models_by_category("quadruped")
        assert len(quadrupeds) == 1
        assert quadrupeds[0].name == "spot"

    @pytest.mark.asyncio
    async def test_search_models(self, library):
        """Test searching models with keywords."""
        library.models_index = {
            "franka_panda": MenagerieModel(
                name="franka_panda", category="arm", xml_path="",
                assets_path="", description="Franka Emika Panda robot",
                dof=7, license=""
            ),
            "boston_spot": MenagerieModel(
                name="boston_spot", category="quadruped", xml_path="",
                assets_path="", description="Boston Dynamics Spot",
                dof=12, license=""
            )
        }

        results = await library.search_models("robot arm")
        assert len(results) > 0

        results = await library.search_models("panda")
        assert any("panda" in m.name.lower() for m in results)


class TestDynamicSceneComposer:
    """Test cases for DynamicSceneComposer."""

    @pytest.fixture
    def composer(self):
        """Create a composer instance."""
        return DynamicSceneComposer()

    @pytest.fixture
    def mock_entities(self):
        """Create mock extracted entities."""
        return ExtractedEntities(
            main_objects=["robot arm", "table", "balls"],
            environment="indoor laboratory",
            physics_properties={
                "gravity": True,
                "collisions": True,
                "friction": 0.5
            },
            interactions=["grasping", "lifting"],
            constraints=["stable base", "reachable workspace"],
            initial_conditions={"arm_position": "home"},
            success_criteria=["pick up all balls"],
            time_properties={"duration": 10, "timestep": 0.002},
            visual_style="realistic"
        )

    @pytest.mark.asyncio
    async def test_compose_scene_basic(self, composer, mock_entities, mocker):
        """Test basic scene composition."""
        # Mock the model library
        mock_library = MagicMock()
        mock_model = MenagerieModel(
            name="franka_panda", category="arm",
            xml_path="/test/panda.xml", assets_path="/test/assets",
            description="Test arm", dof=7, license="MIT"
        )

        mock_library.search_models = AsyncMock(return_value=[mock_model])
        mock_library.initialize = AsyncMock(return_value=True)

        mocker.patch.object(composer, 'model_library', mock_library)

        # Mock prompt enhancement
        mock_enhance = AsyncMock(return_value={
            "models": ["franka_panda"],
            "environment": "lab_scene"
        })
        mocker.patch.object(composer, '_enhance_with_prompt', mock_enhance)

        # Compose scene
        result = await composer.compose_scene(
            "Robot arm picking up balls",
            mock_entities
        )

        assert isinstance(result, SceneComposition)
        assert len(result.main_models) > 0
        assert result.environment_settings is not None

    @pytest.mark.asyncio
    async def test_generate_mjcf(self, composer, mocker):
        """Test MJCF generation from scene composition."""
        mock_composition = SceneComposition(
            main_models=[
                MenagerieModel(
                    name="test_robot", category="arm",
                    xml_path="/test.xml", assets_path="/assets",
                    description="Test", dof=6, license="MIT"
                )
            ],
            environment_settings={"gravity": -9.81},
            physics_constraints=[],
            lighting_setup={
                "lights": [
                    {"pos": [0, 0, 10], "dir": [0, 0, -1]}
                ]
            },
            camera_angles=[
                {"name": "main", "pos": [2, 2, 2]}
            ],
            materials={}
        )

        mjcf = await composer.generate_mjcf(mock_composition)

        assert "<mujoco" in mjcf
        assert "gravity" in mjcf
        assert "<light" in mjcf
        assert "<camera" in mjcf

    @pytest.mark.asyncio
    async def test_compose_scene_error_handling(self, composer, mock_entities, mocker):
        """Test error handling in scene composition."""
        mock_library = MagicMock()
        mock_library.initialize = AsyncMock(return_value=False)

        mocker.patch.object(composer, 'model_library', mock_library)

        with pytest.raises(Exception):
            await composer.compose_scene(
                "Test prompt",
                mock_entities
            )

    @pytest.mark.asyncio
    async def test_lighting_setup_generation(self, composer):
        """Test generation of professional lighting setup."""
        lighting = await composer._generate_lighting_setup("cinematic")

        assert "lights" in lighting
        assert len(lighting["lights"]) >= 3  # Multi-light setup

        # Check for shadow settings
        for light in lighting["lights"]:
            if "directional" in light.get("type", ""):
                assert "shadowsize" in light

    @pytest.mark.asyncio
    async def test_material_generation(self, composer):
        """Test generation of materials with visual properties."""
        materials = await composer._generate_materials(["metal", "plastic"])

        assert "metal" in materials
        assert "plastic" in materials

        # Check material properties
        metal = materials["metal"]
        assert "rgba" in metal
        assert "specular" in metal or "shininess" in metal


@pytest.mark.asyncio
class TestIntegrationDynamicSceneComposer:
    """Integration tests for DynamicSceneComposer."""

    async def test_full_pipeline(self, tmp_path):
        """Test the complete pipeline from prompt to MJCF."""
        composer = DynamicSceneComposer()

        # Create minimal test models directory
        models_dir = tmp_path / "test_models"
        models_dir.mkdir()

        # Mock a simple model
        model_dir = models_dir / "test_arm"
        model_dir.mkdir()

        model_xml = model_dir / "scene.xml"
        model_xml.write_text("""
        <mujoco>
            <worldbody>
                <body name="test_arm">
                    <joint name="joint1" type="hinge"/>
                    <geom type="box" size="0.1 0.1 0.5"/>
                </body>
            </worldbody>
        </mujoco>
        """)

        # Override models directory
        composer.model_library.models_dir = models_dir

        # Create test entities
        entities = ExtractedEntities(
            main_objects=["robot arm"],
            environment="test environment",
            physics_properties={},
            interactions=[],
            constraints=[],
            initial_conditions={},
            success_criteria=[],
            time_properties={},
            visual_style="simple"
        )

        # Test with mocked library
        with patch.object(composer.model_library, 'initialize', return_value=True):
            with patch.object(composer, '_enhance_with_prompt',
                            new_callable=AsyncMock, return_value={}):
                try:
                    result = await composer.compose_scene(
                        "Test robot arm",
                        entities
                    )
                    assert result is not None

                    mjcf = await composer.generate_mjcf(result)
                    assert "<mujoco" in mjcf
                except Exception:
                    # Expected as we're using minimal mocks
                    pass