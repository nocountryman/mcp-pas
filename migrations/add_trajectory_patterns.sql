-- Phase 34: Trajectory Learning System
-- Stores patterns from Antigravity agent trajectories for few-shot learning

CREATE TABLE IF NOT EXISTS trajectory_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cascade_id TEXT UNIQUE NOT NULL,  -- Antigravity's session ID
    session_id UUID REFERENCES reasoning_sessions(id),  -- PAS session if linked
    summary TEXT,
    summary_embedding VECTOR(1024),
    outcome TEXT CHECK (outcome IN ('success', 'partial', 'failure')),
    tool_sequence JSONB,  -- [{tool, args_hash, result_type}]
    step_count INTEGER,
    workspace_id TEXT,  -- Which workspace this came from
    captured_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vector similarity index for finding related trajectories
CREATE INDEX IF NOT EXISTS idx_trajectory_embedding 
    ON trajectory_patterns USING ivfflat (summary_embedding vector_cosine_ops)
    WITH (lists = 50);

-- Filter by outcome (success patterns vs failure patterns)
CREATE INDEX IF NOT EXISTS idx_trajectory_outcome ON trajectory_patterns(outcome);

-- Recent trajectories
CREATE INDEX IF NOT EXISTS idx_trajectory_captured ON trajectory_patterns(captured_at DESC);

COMMENT ON TABLE trajectory_patterns IS 'Phase 34: Stores Antigravity agent trajectory patterns for few-shot learning';
