-- Phase 12: Session Handoff System
-- Run BEFORE deploying code changes

CREATE TABLE IF NOT EXISTS session_handoffs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES reasoning_sessions(id) ON DELETE SET NULL,
    project_id TEXT,
    summary TEXT NOT NULL,
    summary_embedding vector(1536),
    next_task TEXT,
    context JSONB DEFAULT '{}',
    linked_artifacts TEXT[] DEFAULT '{}',
    linked_sessions UUID[] DEFAULT '{}',
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'processed', 'archived')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

-- Index for semantic search
CREATE INDEX IF NOT EXISTS idx_session_handoffs_embedding 
    ON session_handoffs USING ivfflat (summary_embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for active handoffs lookup
CREATE INDEX IF NOT EXISTS idx_session_handoffs_status 
    ON session_handoffs(project_id, status) 
    WHERE status = 'active';

-- Index for session lookup
CREATE INDEX IF NOT EXISTS idx_session_handoffs_session 
    ON session_handoffs(session_id);
