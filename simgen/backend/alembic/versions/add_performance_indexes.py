"""Add performance indexes and cache tables

Revision ID: add_performance_indexes
Revises:
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'add_performance_indexes'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add performance optimizations to database."""

    # Add indexes to existing simulations table
    op.create_index('idx_session_status', 'simulations', ['session_id', 'status'])
    op.create_index('idx_session_created', 'simulations', ['session_id', 'created_at'])
    op.create_index('idx_status_created', 'simulations', ['status', 'created_at'])
    op.create_index('idx_session_status_created', 'simulations', ['session_id', 'status', 'created_at'])

    # Add indexes to quality_assessments if table exists
    try:
        op.create_index('idx_sim_overall_score', 'quality_assessments', ['simulation_id', 'overall_score'])
        op.create_index('idx_created_score', 'quality_assessments', ['created_at', 'overall_score'])
    except:
        pass  # Table might not exist yet

    # Create sketch_cache table for CV results caching
    op.create_table('sketch_cache',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('image_hash', sa.String(length=64), nullable=False),
        sa.Column('cv_analysis', sa.JSON(), nullable=True),
        sa.Column('physics_spec', sa.JSON(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('hit_count', sa.Integer(), nullable=True, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('last_accessed', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('image_hash')
    )
    op.create_index('idx_hash_accessed', 'sketch_cache', ['image_hash', 'last_accessed'])
    op.create_index('idx_created_accessed', 'sketch_cache', ['created_at', 'last_accessed'])

    # Create llm_response_cache table
    op.create_table('llm_response_cache',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('prompt_hash', sa.String(length=64), nullable=False),
        sa.Column('model', sa.String(length=100), nullable=True),
        sa.Column('prompt_text', sa.Text(), nullable=True),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('response', sa.JSON(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('response_time_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('prompt_hash', 'model', 'parameters', name='uq_llm_cache')
    )
    op.create_index('idx_prompt_model', 'llm_response_cache', ['prompt_hash', 'model'])
    op.create_index('idx_expires', 'llm_response_cache', ['expires_at'])

    # Create performance_metrics table for monitoring
    op.create_table('performance_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('operation_type', sa.String(length=100), nullable=False),
        sa.Column('operation_name', sa.String(length=255), nullable=False),
        sa.Column('duration_ms', sa.Float(), nullable=False),
        sa.Column('memory_mb', sa.Float(), nullable=True),
        sa.Column('cpu_percent', sa.Float(), nullable=True),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('request_id', sa.String(length=255), nullable=True),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_op_type_timestamp', 'performance_metrics', ['operation_type', 'timestamp'])
    op.create_index('idx_session_timestamp', 'performance_metrics', ['session_id', 'timestamp'])
    op.create_index('idx_duration_timestamp', 'performance_metrics', ['duration_ms', 'timestamp'])

    # Add processing_time_ms column to simulations if not exists
    try:
        op.add_column('simulations', sa.Column('processing_time_ms', sa.Float(), nullable=True))
        op.add_column('simulations', sa.Column('frame_count', sa.Integer(), nullable=True, default=0))
    except:
        pass  # Columns might already exist

    # Create stored procedure for cleaning old cache entries
    op.execute("""
        CREATE OR REPLACE FUNCTION clean_old_cache_entries()
        RETURNS void AS $$
        BEGIN
            -- Delete sketch cache entries older than 7 days with no recent access
            DELETE FROM sketch_cache
            WHERE last_accessed < NOW() - INTERVAL '7 days'
            AND hit_count < 5;

            -- Delete expired LLM cache entries
            DELETE FROM llm_response_cache
            WHERE expires_at < NOW();

            -- Delete old performance metrics (keep 30 days)
            DELETE FROM performance_metrics
            WHERE timestamp < NOW() - INTERVAL '30 days';
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Create index for full-text search on simulations if using PostgreSQL
    try:
        op.execute("""
            CREATE INDEX idx_simulations_search ON simulations
            USING gin(to_tsvector('english', name || ' ' || COALESCE(description, '')));
        """)
    except:
        pass  # Not PostgreSQL or text search not available


def downgrade() -> None:
    """Remove performance optimizations."""

    # Drop indexes from simulations table
    op.drop_index('idx_session_status', table_name='simulations')
    op.drop_index('idx_session_created', table_name='simulations')
    op.drop_index('idx_status_created', table_name='simulations')
    op.drop_index('idx_session_status_created', table_name='simulations')

    # Drop indexes from quality_assessments
    try:
        op.drop_index('idx_sim_overall_score', table_name='quality_assessments')
        op.drop_index('idx_created_score', table_name='quality_assessments')
    except:
        pass

    # Drop cache tables
    op.drop_table('performance_metrics')
    op.drop_table('llm_response_cache')
    op.drop_table('sketch_cache')

    # Drop stored procedure
    op.execute("DROP FUNCTION IF EXISTS clean_old_cache_entries();")

    # Drop full-text search index
    try:
        op.drop_index('idx_simulations_search', table_name='simulations')
    except:
        pass

    # Remove added columns
    try:
        op.drop_column('simulations', 'processing_time_ms')
        op.drop_column('simulations', 'frame_count')
    except:
        pass