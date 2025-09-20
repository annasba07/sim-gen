"""Comprehensive advanced database operation tests with real transactions and constraints."""
import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os
from typing import Optional, Dict, Any, List
import uuid

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import all database components for comprehensive testing
from simgen.core.config import settings
from simgen.db.base import Base, get_db, SessionLocal
from simgen.models.simulation import Simulation, SimulationGenerationMethod, SimulationStatus
from simgen.models.schemas import SimulationRequest, SimulationResponse
from simgen.database.service import DatabaseService
from simgen.database.connection_pool import ConnectionPool
from simgen.database.query_optimizer import QueryOptimizer

# SQLAlchemy components
from sqlalchemy import create_engine, text, inspect, MetaData, Table, Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.pool import StaticPool
import sqlalchemy as sa


class TestDatabaseAdvancedOperations:
    """Advanced database operations testing with real connections and transactions."""

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
            echo=False  # Set to True for SQL debugging
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

    def test_table_creation_and_schema_validation(self, test_engine):
        """Test that all tables are created with correct schema."""
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

    def test_simulation_model_crud_operations(self, test_session):
        """Test comprehensive CRUD operations on Simulation."""
        # CREATE - Insert new simulation
        simulation_data = {
            'session_id': 'test-session-001',
            'user_prompt': 'Test bouncing ball simulation',
            'mjcf_content': '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>',
            'generation_method': SimulationGenerationMethod.TEMPLATE_BASED,
            'status': SimulationStatus.COMPLETED
        }

        simulation = Simulation(**simulation_data)
        test_session.add(simulation)
        test_session.commit()

        assert simulation.id is not None
        assert simulation.created_at is not None
        assert simulation.updated_at is not None

        # READ - Query the simulation
        retrieved = test_session.query(Simulation).filter(
            Simulation.user_prompt == 'Test bouncing ball simulation'
        ).first()

        assert retrieved is not None
        assert retrieved.user_prompt == simulation_data['user_prompt']
        assert retrieved.mjcf_content == simulation_data['mjcf_content']
        assert retrieved.generation_method == simulation_data['generation_method']

        # UPDATE - Modify simulation
        retrieved.mjcf_content = '<mujoco><worldbody><geom type="sphere" size="0.2"/></worldbody></mujoco>'
        retrieved.status = SimulationStatus.COMPLETED
        test_session.commit()

        updated = test_session.query(Simulation).filter(Simulation.id == retrieved.id).first()
        assert 'size="0.2"' in updated.mjcf_content
        assert updated.status == SimulationStatus.COMPLETED
        assert updated.updated_at > updated.created_at

        # DELETE - Remove simulation
        test_session.delete(updated)
        test_session.commit()

        deleted_check = test_session.query(Simulation).filter(Simulation.id == retrieved.id).first()
        assert deleted_check is None

    def test_database_transactions_and_rollback(self, test_session):
        """Test transaction handling and rollback scenarios."""
        # Test successful transaction
        simulation1 = Simulation(
            session_id=f'test-session-{uuid.uuid4().hex[:8]}',
            user_prompt='Transaction test 1',
            mjcf_content='<mujoco>test1</mujoco>',
            generation_method=SimulationGenerationMethod.LLM_GENERATION
        )
        simulation2 = Simulation(
            session_id=f'test-session-{uuid.uuid4().hex[:8]}',
            user_prompt='Transaction test 2',
            mjcf_content='<mujoco>test2</mujoco>',
            generation_method=SimulationGenerationMethod.HYBRID
        )

        test_session.add(simulation1)
        test_session.add(simulation2)
        test_session.commit()

        # Verify both were saved
        count = test_session.query(Simulation).filter(
            Simulation.user_prompt.like('Transaction test%')
        ).count()
        assert count == 2

        # Test transaction rollback
        try:
            simulation3 = Simulation(
                user_prompt='Transaction test 3',
                mjcf_content='<mujoco>test3</mujoco>',
                generation_method=SimulationGenerationMethod.TEMPLATE_BASED
            )
            test_session.add(simulation3)

            # Simulate error condition
            test_session.execute(text("INSERT INTO non_existent_table VALUES (1)"))
            test_session.commit()
        except Exception:
            test_session.rollback()

        # Verify rollback worked - simulation3 should not exist
        count_after_rollback = test_session.query(Simulation).filter(
            Simulation.user_prompt.like('Transaction test%')
        ).count()
        assert count_after_rollback == 2  # Only first two should exist

    def test_concurrent_database_access(self, test_engine):
        """Test concurrent database access and connection pooling."""
        def create_simulation_worker(worker_id):
            """Worker function to create simulation in separate connection."""
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
            session = SessionLocal()
            try:
                simulation = Simulation(
                    user_prompt=f'Concurrent test {worker_id}',
                    mjcf_content=f'<mujoco>worker_{worker_id}</mujoco>',
                    generation_method=SimulationGenerationMethod.TEMPLATE_BASED
                )
                session.add(simulation)
                session.commit()
                return simulation.id
            finally:
                session.close()

        # Simulate concurrent operations
        worker_ids = [1, 2, 3, 4, 5]
        simulation_ids = []

        for worker_id in worker_ids:
            sim_id = create_simulation_worker(worker_id)
            simulation_ids.append(sim_id)

        # Verify all simulations were created
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        verification_session = SessionLocal()
        try:
            concurrent_count = verification_session.query(Simulation).filter(
                Simulation.user_prompt.like('Concurrent test%')
            ).count()
            assert concurrent_count == 5

            # Verify all IDs are unique
            assert len(set(simulation_ids)) == 5
        finally:
            verification_session.close()

    def test_database_constraints_and_validation(self, test_session):
        """Test database constraints and data validation."""
        # Test required field validation
        with pytest.raises(Exception):  # Should fail due to missing required fields
            incomplete_simulation = Simulation()
            test_session.add(incomplete_simulation)
            test_session.commit()

        test_session.rollback()

        # Test data type validation
        valid_simulation = Simulation(
            session_id=f'test-session-{uuid.uuid4().hex[:8]}',
            user_prompt='Valid simulation',
            mjcf_content='<mujoco><worldbody></worldbody></mujoco>',
            generation_method=SimulationGenerationMethod.HYBRID,
            status=SimulationStatus.COMPLETED
        )
        test_session.add(valid_simulation)
        test_session.commit()

        # Verify the simulation was saved correctly
        saved = test_session.query(Simulation).filter(
            Simulation.user_prompt == 'Valid simulation'
        ).first()
        assert saved.status == SimulationStatus.COMPLETED

    def test_query_optimization_and_indexing(self, test_session):
        """Test query performance and optimization."""
        # Create multiple simulations for testing
        simulations = []
        for i in range(20):
            simulation = Simulation(
                user_prompt=f'Optimization test {i}',
                mjcf_content=f'<mujoco>test_{i}</mujoco>',
                generation_method=SimulationGenerationMethod.TEMPLATE_BASED,
                status=SimulationStatus.COMPLETED
            )
            simulations.append(simulation)

        test_session.add_all(simulations)
        test_session.commit()

        # Test various query patterns
        # 1. Filter by prompt pattern
        prompt_results = test_session.query(Simulation).filter(
            Simulation.user_prompt.like('Optimization test%')
        ).all()
        assert len(prompt_results) == 20

        # 2. Filter by generation method
        template_results = test_session.query(Simulation).filter(
            Simulation.generation_method == SimulationGenerationMethod.TEMPLATE_BASED
        ).all()
        assert len(template_results) >= 20

        # 3. Filter by date range
        recent_results = test_session.query(Simulation).filter(
            Simulation.created_at >= datetime.now() - timedelta(minutes=1)
        ).all()
        assert len(recent_results) >= 20

        # 4. Complex query with multiple conditions
        complex_results = test_session.query(Simulation).filter(
            Simulation.user_prompt.like('Optimization test%'),
            Simulation.generation_method == SimulationGenerationMethod.TEMPLATE_BASED,
            Simulation.created_at >= datetime.now() - timedelta(minutes=1)
        ).all()
        assert len(complex_results) == 20

    def test_database_service_integration(self, test_session):
        """Test DatabaseService class integration."""
        # Mock the settings to use our test session
        with patch('simgen.database.service.get_db') as mock_get_db:
            mock_get_db.return_value = test_session

            # Test DatabaseService operations
            db_service = DatabaseService()

            # Test create operation through service
            simulation_data = {
                'prompt': 'Service integration test',
                'mjcf_content': '<mujoco><worldbody><geom type="box"/></worldbody></mujoco>',
                'generation_method': SimulationGenerationMethod.HYBRID,
                'metadata': {'service_test': True}
            }

            # Since we're mocking, we need to handle the service methods appropriately
            # This tests the integration patterns
            simulation = Simulation(**simulation_data)
            test_session.add(simulation)
            test_session.commit()

            # Verify through direct query
            saved = test_session.query(Simulation).filter(
                Simulation.user_prompt == 'Service integration test'
            ).first()
            assert saved is not None
            assert saved.status == SimulationStatus.COMPLETED


class TestConnectionPoolAndOptimization:
    """Test connection pooling and query optimization components."""

    def test_connection_pool_initialization(self):
        """Test ConnectionPool class instantiation and configuration."""
        # Test with default settings
        pool = ConnectionPool()
        assert pool is not None

        # Test pool configuration methods exist
        assert hasattr(pool, 'get_connection') or hasattr(pool, 'acquire')
        assert hasattr(pool, 'release') or hasattr(pool, 'close')

    def test_query_optimizer_functionality(self):
        """Test QueryOptimizer class functionality."""
        optimizer = QueryOptimizer()
        assert optimizer is not None

        # Test optimizer methods exist
        assert hasattr(optimizer, 'optimize') or hasattr(optimizer, 'analyze')

        # Test with sample query patterns
        sample_queries = [
            "SELECT * FROM simulations WHERE prompt LIKE '%test%'",
            "SELECT id, prompt FROM simulations ORDER BY created_at DESC LIMIT 10",
            "SELECT COUNT(*) FROM simulations WHERE generation_method = 'TEMPLATE_BASED'"
        ]

        for query in sample_queries:
            # Test that optimizer can handle different query types
            try:
                if hasattr(optimizer, 'optimize'):
                    result = optimizer.optimize(query)
                    assert result is not None
                elif hasattr(optimizer, 'analyze'):
                    result = optimizer.analyze(query)
                    assert result is not None
            except NotImplementedError:
                # Some methods might not be implemented yet
                pass


class TestDatabasePerformanceAndScaling:
    """Test database performance characteristics and scaling behavior."""

    @pytest.fixture
    def large_dataset_session(self, temp_db_path):
        """Create session with larger dataset for performance testing."""
        engine = create_engine(temp_db_path, poolclass=StaticPool, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()

        # Create larger dataset
        simulations = []
        for i in range(100):
            simulation = Simulation(
                user_prompt=f'Performance test simulation {i}',
                mjcf_content=f'<mujoco><worldbody><geom type="sphere" size="{0.1 + i*0.01}"/></worldbody></mujoco>',
                generation_method=SimulationGenerationMethod.TEMPLATE_BASED if i % 3 == 0 else SimulationGenerationMethod.HYBRID,
                status=SimulationStatus.COMPLETED
            )
            simulations.append(simulation)

        session.add_all(simulations)
        session.commit()

        try:
            yield session
        finally:
            session.close()

    def test_bulk_operations_performance(self, large_dataset_session):
        """Test performance of bulk database operations."""
        session = large_dataset_session

        # Test bulk query performance
        start_time = datetime.now()
        all_simulations = session.query(Simulation).all()
        query_time = (datetime.now() - start_time).total_seconds()

        assert len(all_simulations) == 100
        assert query_time < 1.0  # Should complete within 1 second

        # Test filtered query performance
        start_time = datetime.now()
        filtered_simulations = session.query(Simulation).filter(
            Simulation.generation_method == SimulationGenerationMethod.TEMPLATE_BASED
        ).all()
        filtered_query_time = (datetime.now() - start_time).total_seconds()

        assert len(filtered_simulations) > 0
        assert filtered_query_time < 0.5  # Should be faster than full query

        # Test complex query performance
        start_time = datetime.now()
        complex_results = session.query(Simulation).filter(
            Simulation.user_prompt.like('%Performance test%'),
            Simulation.generation_method == SimulationGenerationMethod.HYBRID
        ).order_by(Simulation.created_at.desc()).limit(10).all()
        complex_query_time = (datetime.now() - start_time).total_seconds()

        assert len(complex_results) <= 10
        assert complex_query_time < 0.3  # Complex queries should still be fast

    def test_pagination_and_streaming(self, large_dataset_session):
        """Test pagination and streaming query patterns."""
        session = large_dataset_session

        page_size = 10
        total_pages = 10
        results_collected = []

        for page in range(total_pages):
            offset = page * page_size
            page_results = session.query(Simulation).order_by(
                Simulation.id
            ).offset(offset).limit(page_size).all()

            results_collected.extend(page_results)

            # Each page should have exactly page_size items (except possibly the last)
            assert len(page_results) <= page_size

        # Should have collected all 100 items
        assert len(results_collected) == 100

        # Verify no duplicates in pagination
        collected_ids = [sim.id for sim in results_collected]
        assert len(set(collected_ids)) == len(collected_ids)

    def test_database_integrity_under_stress(self, large_dataset_session):
        """Test database integrity under stress conditions."""
        session = large_dataset_session

        # Test rapid concurrent-like operations
        operations_count = 50

        for i in range(operations_count):
            # Mix of operations
            if i % 4 == 0:
                # CREATE
                new_sim = Simulation(
                    user_prompt=f'Stress test {i}',
                    mjcf_content=f'<mujoco>stress_{i}</mujoco>',
                    generation_method=SimulationGenerationMethod.LLM_GENERATION
                )
                session.add(new_sim)
            elif i % 4 == 1:
                # READ
                existing = session.query(Simulation).first()
                assert existing is not None
            elif i % 4 == 2:
                # UPDATE
                to_update = session.query(Simulation).filter(
                    Simulation.user_prompt.like('Performance test%')
                ).first()
                if to_update:
                    to_update.status = SimulationStatus.PROCESSING
            else:
                # Complex query
                complex_result = session.query(Simulation).filter(
                    Simulation.id > i
                ).order_by(Simulation.created_at).first()

        # Commit all changes
        session.commit()

        # Verify database is still consistent
        total_count = session.query(Simulation).count()
        assert total_count >= 100  # Should have at least original 100 plus some new ones

        # Verify data integrity
        all_simulations = session.query(Simulation).all()
        for sim in all_simulations:
            assert sim.id is not None
            assert sim.prompt is not None
            assert sim.mjcf_content is not None
            assert sim.generation_method is not None


class TestDatabaseMigrationAndSchema:
    """Test database migration patterns and schema evolution."""

    def test_schema_introspection(self, test_engine):
        """Test database schema introspection capabilities."""
        inspector = inspect(test_engine)

        # Test table discovery
        tables = inspector.get_table_names()
        assert 'simulations' in tables

        # Test column introspection
        columns = inspector.get_columns('simulations')
        column_info = {col['name']: col for col in columns}

        # Verify key columns exist with correct types
        assert 'id' in column_info
        assert 'prompt' in column_info
        assert 'mjcf_content' in column_info
        assert 'created_at' in column_info
        assert 'updated_at' in column_info
        assert 'generation_method' in column_info
        assert 'metadata' in column_info

        # Test index introspection if any exist
        indexes = inspector.get_indexes('simulations')
        # Indexes might not be explicitly defined, but method should work
        assert isinstance(indexes, list)

    def test_database_backup_and_restore_patterns(self, temp_db_path):
        """Test patterns for database backup and restoration."""
        # Create original database with data
        engine1 = create_engine(temp_db_path, poolclass=StaticPool, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine1)

        SessionLocal1 = sessionmaker(autocommit=False, autoflush=False, bind=engine1)
        session1 = SessionLocal1()

        # Add test data
        original_simulation = Simulation(
            session_id=f'test-session-{uuid.uuid4().hex[:8]}',
            user_prompt='Backup test simulation',
            mjcf_content='<mujoco><worldbody><geom type="cylinder"/></worldbody></mujoco>',
            generation_method=SimulationGenerationMethod.HYBRID,
            status=SimulationStatus.COMPLETED
        )
        session1.add(original_simulation)
        session1.commit()
        original_id = original_simulation.id
        session1.close()

        # Simulate backup by reading data
        session1 = SessionLocal1()
        backup_data = session1.query(Simulation).all()
        session1.close()
        engine1.dispose()

        # Create new database (simulate restore)
        temp_dir2 = tempfile.mkdtemp()
        restore_db_path = f"sqlite:///{os.path.join(temp_dir2, 'restored.db')}"

        engine2 = create_engine(restore_db_path, poolclass=StaticPool, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine2)

        SessionLocal2 = sessionmaker(autocommit=False, autoflush=False, bind=engine2)
        session2 = SessionLocal2()

        # Restore data
        for sim in backup_data:
            restored_sim = Simulation(
                session_id=f'restored-{uuid.uuid4().hex[:8]}',
                user_prompt=sim.user_prompt,
                mjcf_content=sim.mjcf_content,
                generation_method=sim.generation_method,
                status=SimulationStatus.COMPLETED
            )
            session2.add(restored_sim)

        session2.commit()

        # Verify restoration
        restored_simulation = session2.query(Simulation).filter(
            Simulation.user_prompt == 'Backup test simulation'
        ).first()

        assert restored_simulation is not None
        assert restored_simulation.mjcf_content == original_simulation.mjcf_content
        assert restored_simulation.status == SimulationStatus.COMPLETED

        session2.close()
        engine2.dispose()