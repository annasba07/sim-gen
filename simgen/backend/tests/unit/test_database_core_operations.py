"""Core database operation tests focusing on models and basic functionality."""
import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import tempfile
import os
import uuid

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import core database components
from simgen.core.config import settings
from simgen.db.base import Base, get_db, SessionLocal
from simgen.models.simulation import Simulation, SimulationGenerationMethod, SimulationStatus

# SQLAlchemy components
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


class TestDatabaseCoreOperations:
    """Core database operations testing with direct model access."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary SQLite database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_simulation.db")
        return f"sqlite:///{db_path}"

    @pytest.fixture
    def test_engine(self, temp_db_path):
        """Create test database engine."""
        engine = create_engine(
            temp_db_path,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False
        )
        Base.metadata.create_all(bind=engine)
        return engine

    @pytest.fixture
    def test_session(self, test_engine):
        """Create test database session."""
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    def test_database_connection_establishment(self, test_engine):
        """Test database connection can be established successfully."""
        with test_engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_simulation_table_creation(self, test_engine):
        """Test that simulation table is created with correct schema."""
        inspector = inspect(test_engine)
        tables = inspector.get_table_names()

        # Check that simulation table exists
        assert 'simulations' in tables

        # Validate simulation table schema
        columns = inspector.get_columns('simulations')
        column_names = [col['name'] for col in columns]

        expected_columns = ['id', 'session_id', 'user_prompt', 'mjcf_content', 'created_at', 'updated_at', 'generation_method', 'status']
        for expected_col in expected_columns:
            assert expected_col in column_names, f"Missing column: {expected_col}"

    def test_simulation_model_creation(self, test_session):
        """Test creating a simulation model instance."""
        simulation = Simulation(
            session_id='test-session-001',
            user_prompt='Test bouncing ball simulation',
            mjcf_content='<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>',
            generation_method=SimulationGenerationMethod.TEMPLATE_BASED,
            status=SimulationStatus.COMPLETED
        )

        test_session.add(simulation)
        test_session.commit()

        assert simulation.id is not None
        assert simulation.created_at is not None
        assert simulation.updated_at is not None

    def test_simulation_model_retrieval(self, test_session):
        """Test retrieving simulation models from database."""
        # Create test simulation
        simulation = Simulation(
            session_id='test-session-002',
            user_prompt='Test retrieval simulation',
            mjcf_content='<mujoco><worldbody><geom type="box"/></worldbody></mujoco>',
            generation_method=SimulationGenerationMethod.LLM_GENERATION,
            status=SimulationStatus.PENDING
        )
        test_session.add(simulation)
        test_session.commit()

        # Retrieve by ID
        retrieved = test_session.query(Simulation).filter(Simulation.id == simulation.id).first()
        assert retrieved is not None
        assert retrieved.user_prompt == 'Test retrieval simulation'
        assert retrieved.status == SimulationStatus.PENDING

        # Retrieve by session_id
        session_retrieved = test_session.query(Simulation).filter(
            Simulation.session_id == 'test-session-002'
        ).first()
        assert session_retrieved is not None
        assert session_retrieved.id == simulation.id

    def test_simulation_model_update(self, test_session):
        """Test updating simulation model fields."""
        # Create initial simulation
        simulation = Simulation(
            session_id='test-session-003',
            user_prompt='Test update simulation',
            mjcf_content='<mujoco><worldbody><geom type="sphere"/></worldbody></mujoco>',
            generation_method=SimulationGenerationMethod.TEMPLATE_BASED,
            status=SimulationStatus.PROCESSING
        )
        test_session.add(simulation)
        test_session.commit()

        # Update fields
        simulation.status = SimulationStatus.COMPLETED
        simulation.mjcf_content = '<mujoco><worldbody><geom type="sphere" size="0.2"/></worldbody></mujoco>'
        simulation.quality_score_overall = 8.5
        test_session.commit()

        # Verify updates
        updated = test_session.query(Simulation).filter(Simulation.id == simulation.id).first()
        assert updated.status == SimulationStatus.COMPLETED
        assert 'size="0.2"' in updated.mjcf_content
        assert updated.quality_score_overall == 8.5
        assert updated.updated_at > updated.created_at

    def test_simulation_model_deletion(self, test_session):
        """Test deleting simulation models."""
        # Create simulation to delete
        simulation = Simulation(
            session_id='test-session-004',
            user_prompt='Test deletion simulation',
            mjcf_content='<mujoco><worldbody></worldbody></mujoco>',
            generation_method=SimulationGenerationMethod.HYBRID,
            status=SimulationStatus.FAILED
        )
        test_session.add(simulation)
        test_session.commit()

        simulation_id = simulation.id

        # Delete simulation
        test_session.delete(simulation)
        test_session.commit()

        # Verify deletion
        deleted_check = test_session.query(Simulation).filter(Simulation.id == simulation_id).first()
        assert deleted_check is None

    def test_simulation_enum_values(self, test_session):
        """Test that enum values are properly handled."""
        # Test all status values
        for status in SimulationStatus:
            simulation = Simulation(
                session_id=f'test-session-{status.value}',
                user_prompt=f'Test {status.value} simulation',
                mjcf_content='<mujoco><worldbody></worldbody></mujoco>',
                generation_method=SimulationGenerationMethod.TEMPLATE_BASED,
                status=status
            )
            test_session.add(simulation)

        # Test all generation method values
        for method in SimulationGenerationMethod:
            simulation = Simulation(
                session_id=f'test-session-{method.value}',
                user_prompt=f'Test {method.value} simulation',
                mjcf_content='<mujoco><worldbody></worldbody></mujoco>',
                generation_method=method,
                status=SimulationStatus.COMPLETED
            )
            test_session.add(simulation)

        test_session.commit()

        # Verify all were saved
        total_count = test_session.query(Simulation).count()
        expected_count = len(SimulationStatus) + len(SimulationGenerationMethod)
        assert total_count >= expected_count

    def test_simulation_quality_metrics(self, test_session):
        """Test simulation quality metrics fields."""
        simulation = Simulation(
            session_id='test-session-quality',
            user_prompt='Test quality metrics simulation',
            mjcf_content='<mujoco><worldbody></worldbody></mujoco>',
            generation_method=SimulationGenerationMethod.HYBRID,
            status=SimulationStatus.COMPLETED,
            quality_score_overall=8.7,
            quality_score_physics=9.2,
            quality_score_visual=8.1,
            quality_score_functional=8.9,
            user_rating=9,
            user_feedback='Excellent physics simulation!'
        )
        test_session.add(simulation)
        test_session.commit()

        # Verify quality metrics
        retrieved = test_session.query(Simulation).filter(
            Simulation.session_id == 'test-session-quality'
        ).first()

        assert retrieved.quality_score_overall == 8.7
        assert retrieved.quality_score_physics == 9.2
        assert retrieved.quality_score_visual == 8.1
        assert retrieved.quality_score_functional == 8.9
        assert retrieved.user_rating == 9
        assert retrieved.user_feedback == 'Excellent physics simulation!'

    def test_simulation_processing_metadata(self, test_session):
        """Test simulation processing metadata fields."""
        simulation = Simulation(
            session_id='test-session-metadata',
            user_prompt='Test metadata simulation',
            mjcf_content='<mujoco><worldbody></worldbody></mujoco>',
            generation_method=SimulationGenerationMethod.LLM_GENERATION,
            status=SimulationStatus.COMPLETED,
            refinement_iterations=3,
            generation_duration=15.7,
            error_message=None,
            completed_at=datetime.now()
        )
        test_session.add(simulation)
        test_session.commit()

        # Verify processing metadata
        retrieved = test_session.query(Simulation).filter(
            Simulation.session_id == 'test-session-metadata'
        ).first()

        assert retrieved.refinement_iterations == 3
        assert retrieved.generation_duration == 15.7
        assert retrieved.error_message is None
        assert retrieved.completed_at is not None

    def test_multiple_simulations_query(self, test_session):
        """Test querying multiple simulations with various filters."""
        # Create multiple simulations
        simulations = []
        for i in range(10):
            simulation = Simulation(
                session_id=f'test-session-multi-{i}',
                user_prompt=f'Multi test simulation {i}',
                mjcf_content=f'<mujoco><worldbody><geom name="obj_{i}"/></worldbody></mujoco>',
                generation_method=SimulationGenerationMethod.TEMPLATE_BASED if i % 2 == 0 else SimulationGenerationMethod.HYBRID,
                status=SimulationStatus.COMPLETED if i < 7 else SimulationStatus.FAILED
            )
            simulations.append(simulation)
            test_session.add(simulation)

        test_session.commit()

        # Query by status
        completed_sims = test_session.query(Simulation).filter(
            Simulation.status == SimulationStatus.COMPLETED
        ).all()
        assert len(completed_sims) >= 7

        # Query by generation method
        template_sims = test_session.query(Simulation).filter(
            Simulation.generation_method == SimulationGenerationMethod.TEMPLATE_BASED
        ).all()
        assert len(template_sims) >= 5

        # Query by prompt pattern
        multi_sims = test_session.query(Simulation).filter(
            Simulation.user_prompt.like('Multi test simulation%')
        ).all()
        assert len(multi_sims) == 10

        # Complex query
        complex_results = test_session.query(Simulation).filter(
            Simulation.user_prompt.like('Multi test simulation%'),
            Simulation.status == SimulationStatus.COMPLETED,
            Simulation.generation_method == SimulationGenerationMethod.TEMPLATE_BASED
        ).all()
        assert len(complex_results) >= 3

    def test_simulation_ordering_and_pagination(self, test_session):
        """Test ordering and pagination of simulation queries."""
        # Create simulations with known timestamps
        base_time = datetime.now()
        for i in range(15):
            simulation = Simulation(
                session_id=f'test-session-order-{i}',
                user_prompt=f'Order test simulation {i:02d}',
                mjcf_content='<mujoco><worldbody></worldbody></mujoco>',
                generation_method=SimulationGenerationMethod.TEMPLATE_BASED,
                status=SimulationStatus.COMPLETED
            )
            test_session.add(simulation)

        test_session.commit()

        # Test ordering by created_at descending (newest first)
        recent_sims = test_session.query(Simulation).filter(
            Simulation.user_prompt.like('Order test simulation%')
        ).order_by(Simulation.created_at.desc()).limit(5).all()

        assert len(recent_sims) == 5
        # Verify ordering (newer timestamps should be first)
        for i in range(1, len(recent_sims)):
            assert recent_sims[i-1].created_at >= recent_sims[i].created_at

        # Test pagination
        page_1 = test_session.query(Simulation).filter(
            Simulation.user_prompt.like('Order test simulation%')
        ).order_by(Simulation.id).limit(5).all()

        page_2 = test_session.query(Simulation).filter(
            Simulation.user_prompt.like('Order test simulation%')
        ).order_by(Simulation.id).offset(5).limit(5).all()

        assert len(page_1) == 5
        assert len(page_2) == 5
        # Verify no overlap between pages
        page_1_ids = {sim.id for sim in page_1}
        page_2_ids = {sim.id for sim in page_2}
        assert page_1_ids.isdisjoint(page_2_ids)

    def test_simulation_session_grouping(self, test_session):
        """Test grouping simulations by session."""
        # Create multiple simulations for same session
        session_id = 'test-session-group'
        for i in range(5):
            simulation = Simulation(
                session_id=session_id,
                user_prompt=f'Group test simulation {i}',
                mjcf_content=f'<mujoco><worldbody><geom name="obj_{i}"/></worldbody></mujoco>',
                generation_method=SimulationGenerationMethod.TEMPLATE_BASED,
                status=SimulationStatus.COMPLETED if i < 3 else SimulationStatus.FAILED
            )
            test_session.add(simulation)

        test_session.commit()

        # Query all simulations for session
        session_sims = test_session.query(Simulation).filter(
            Simulation.session_id == session_id
        ).all()

        assert len(session_sims) == 5

        # Verify all belong to same session
        for sim in session_sims:
            assert sim.session_id == session_id

        # Count by status within session
        completed_in_session = test_session.query(Simulation).filter(
            Simulation.session_id == session_id,
            Simulation.status == SimulationStatus.COMPLETED
        ).count()

        failed_in_session = test_session.query(Simulation).filter(
            Simulation.session_id == session_id,
            Simulation.status == SimulationStatus.FAILED
        ).count()

        assert completed_in_session == 3
        assert failed_in_session == 2


class TestDatabaseTransactions:
    """Test database transaction handling and rollback scenarios."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary SQLite database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_transaction.db")
        return f"sqlite:///{db_path}"

    @pytest.fixture
    def test_engine(self, temp_db_path):
        """Create test database engine."""
        engine = create_engine(
            temp_db_path,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False
        )
        Base.metadata.create_all(bind=engine)
        return engine

    @pytest.fixture
    def test_session(self, test_engine):
        """Create test database session."""
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    def test_successful_transaction(self, test_session):
        """Test successful transaction with multiple operations."""
        simulation1 = Simulation(
            session_id='transaction-test-1',
            user_prompt='Transaction test 1',
            mjcf_content='<mujoco>test1</mujoco>',
            generation_method=SimulationGenerationMethod.LLM_GENERATION,
            status=SimulationStatus.COMPLETED
        )

        simulation2 = Simulation(
            session_id='transaction-test-2',
            user_prompt='Transaction test 2',
            mjcf_content='<mujoco>test2</mujoco>',
            generation_method=SimulationGenerationMethod.HYBRID,
            status=SimulationStatus.COMPLETED
        )

        test_session.add(simulation1)
        test_session.add(simulation2)
        test_session.commit()

        # Verify both were saved
        count = test_session.query(Simulation).filter(
            Simulation.user_prompt.like('Transaction test%')
        ).count()
        assert count == 2

    def test_transaction_rollback(self, test_session):
        """Test transaction rollback on error."""
        # Add first simulation successfully
        simulation1 = Simulation(
            session_id='rollback-test-1',
            user_prompt='Rollback test 1',
            mjcf_content='<mujoco>test1</mujoco>',
            generation_method=SimulationGenerationMethod.TEMPLATE_BASED,
            status=SimulationStatus.COMPLETED
        )
        test_session.add(simulation1)
        test_session.commit()

        # Attempt transaction that should fail
        try:
            simulation2 = Simulation(
                session_id='rollback-test-2',
                user_prompt='Rollback test 2',
                mjcf_content='<mujoco>test2</mujoco>',
                generation_method=SimulationGenerationMethod.TEMPLATE_BASED,
                status=SimulationStatus.COMPLETED
            )
            test_session.add(simulation2)

            # Force an error (try to execute invalid SQL)
            test_session.execute(text("INSERT INTO non_existent_table VALUES (1)"))
            test_session.commit()
        except Exception:
            test_session.rollback()

        # Verify rollback worked - only first simulation should exist
        count = test_session.query(Simulation).filter(
            Simulation.user_prompt.like('Rollback test%')
        ).count()
        assert count == 1

        # Verify the first simulation still exists
        remaining = test_session.query(Simulation).filter(
            Simulation.user_prompt == 'Rollback test 1'
        ).first()
        assert remaining is not None


class TestDatabaseBasicPerformance:
    """Test basic database performance characteristics."""

    @pytest.fixture
    def performance_db_setup(self):
        """Setup database with larger dataset for performance testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_performance.db")
        db_url = f"sqlite:///{db_path}"

        engine = create_engine(db_url, poolclass=StaticPool, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()

        # Create larger dataset
        simulations = []
        for i in range(50):
            simulation = Simulation(
                session_id=f'perf-session-{i // 10}',  # Group into sessions
                user_prompt=f'Performance test simulation {i}',
                mjcf_content=f'<mujoco><worldbody><geom type="sphere" size="{0.1 + i*0.01}"/></worldbody></mujoco>',
                generation_method=SimulationGenerationMethod.TEMPLATE_BASED if i % 3 == 0 else SimulationGenerationMethod.HYBRID,
                status=SimulationStatus.COMPLETED if i % 4 != 0 else SimulationStatus.FAILED,
                quality_score_overall=5.0 + (i % 6),
                refinement_iterations=i % 5
            )
            simulations.append(simulation)

        session.add_all(simulations)
        session.commit()

        try:
            yield session
        finally:
            session.close()
            engine.dispose()

    def test_bulk_query_performance(self, performance_db_setup):
        """Test performance of bulk queries."""
        session = performance_db_setup

        # Test full table scan
        start_time = datetime.now()
        all_simulations = session.query(Simulation).all()
        query_time = (datetime.now() - start_time).total_seconds()

        assert len(all_simulations) == 50
        assert query_time < 1.0  # Should complete within 1 second for 50 records

    def test_filtered_query_performance(self, performance_db_setup):
        """Test performance of filtered queries."""
        session = performance_db_setup

        # Test filtered query
        start_time = datetime.now()
        completed_simulations = session.query(Simulation).filter(
            Simulation.status == SimulationStatus.COMPLETED
        ).all()
        filtered_time = (datetime.now() - start_time).total_seconds()

        assert len(completed_simulations) > 0
        assert filtered_time < 0.5  # Filtered queries should be faster

    def test_pagination_performance(self, performance_db_setup):
        """Test performance of paginated queries."""
        session = performance_db_setup

        page_size = 10
        pages_to_test = 3

        total_time = 0
        total_records = 0

        for page in range(pages_to_test):
            start_time = datetime.now()

            page_results = session.query(Simulation).order_by(
                Simulation.id
            ).offset(page * page_size).limit(page_size).all()

            page_time = (datetime.now() - start_time).total_seconds()
            total_time += page_time
            total_records += len(page_results)

            assert len(page_results) <= page_size
            assert page_time < 0.2  # Each page should be fast

        assert total_records == pages_to_test * page_size
        assert total_time < 0.5  # Total pagination time should be reasonable

    def test_aggregation_performance(self, performance_db_setup):
        """Test performance of aggregation queries."""
        session = performance_db_setup

        # Test count aggregation
        start_time = datetime.now()
        total_count = session.query(Simulation).count()
        count_time = (datetime.now() - start_time).total_seconds()

        assert total_count == 50
        assert count_time < 0.1  # Count should be very fast

        # Test more complex aggregation (if SQLAlchemy supports it)
        start_time = datetime.now()
        completed_count = session.query(Simulation).filter(
            Simulation.status == SimulationStatus.COMPLETED
        ).count()
        filtered_count_time = (datetime.now() - start_time).total_seconds()

        assert completed_count > 0
        assert filtered_count_time < 0.2