-- ============================================================================
-- Scientific Reasoning MCP - Database Schema
-- Bayesian Tree of Thoughts with Semantic Search
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS ltree;

-- ============================================================================
-- Table: scientific_laws
-- Stores software engineering principles with semantic embeddings
-- ============================================================================
CREATE TABLE scientific_laws (
    id              SERIAL PRIMARY KEY,
    law_name        VARCHAR(255) NOT NULL UNIQUE,
    definition      TEXT NOT NULL,
    scientific_weight DECIMAL(3,2) NOT NULL DEFAULT 0.5 
                    CHECK (scientific_weight >= 0.0 AND scientific_weight <= 1.0),
    embedding       vector(768),
    
    -- Metadata
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor search on laws
CREATE INDEX idx_scientific_laws_embedding 
    ON scientific_laws 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- Table: reasoning_sessions
-- Tracks high-level user goals and current reasoning state
-- ============================================================================
CREATE TABLE reasoning_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    goal            TEXT NOT NULL,
    goal_embedding  vector(768),  -- For semantic session matching
    state           VARCHAR(50) NOT NULL DEFAULT 'active'
                    CHECK (state IN ('active', 'paused', 'completed', 'abandoned')),
    context         JSONB DEFAULT '{}',
    
    -- Session lineage
    parent_session_id UUID REFERENCES reasoning_sessions(id),
    
    -- Bayesian priors for this session
    prior_beliefs   JSONB DEFAULT '{}',
    
    -- Metadata
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for semantic similarity search on session goals
CREATE INDEX idx_reasoning_sessions_goal_embedding 
    ON reasoning_sessions 
    USING hnsw (goal_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- ============================================================================
-- Table: thought_nodes
-- Stores the reasoning tree with Bayesian scoring
-- ============================================================================
CREATE TABLE thought_nodes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES reasoning_sessions(id) ON DELETE CASCADE,
    
    -- Hierarchical path using ltree (e.g., 'root.hypothesis_1.sub_1')
    path            ltree NOT NULL,
    depth           INTEGER GENERATED ALWAYS AS (nlevel(path)) STORED,
    
    -- Content
    content         TEXT NOT NULL,
    node_type       VARCHAR(50) NOT NULL DEFAULT 'thought'
                    CHECK (node_type IN ('root', 'hypothesis', 'evidence', 'conclusion', 'thought')),
    
    -- Bayesian scoring
    prior_score     DECIMAL(10,8) NOT NULL DEFAULT 0.5
                    CHECK (prior_score >= 0.0 AND prior_score <= 1.0),
    likelihood      DECIMAL(10,8) NOT NULL DEFAULT 0.5
                    CHECK (likelihood >= 0.0 AND likelihood <= 1.0),
    posterior_score DECIMAL(10,8) GENERATED ALWAYS AS (
                        -- Bayes' theorem: P(H|E) ∝ P(E|H) * P(H)
                        -- Normalized assuming P(E) approximated for simple cases
                        (prior_score * likelihood) / 
                        NULLIF((prior_score * likelihood) + ((1 - prior_score) * (1 - likelihood)), 0)
                    ) STORED,
    
    -- Semantic embedding for similarity search
    embedding       vector(768),
    
    -- References to supporting laws
    supporting_laws INTEGER[] DEFAULT '{}',
    
    -- Metadata
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor search on thoughts
CREATE INDEX idx_thought_nodes_embedding 
    ON thought_nodes 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GiST index for ltree path queries (ancestors, descendants, etc.)
CREATE INDEX idx_thought_nodes_path 
    ON thought_nodes 
    USING gist (path);

-- B-tree index for session lookups
CREATE INDEX idx_thought_nodes_session 
    ON thought_nodes (session_id);

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply auto-update triggers
CREATE TRIGGER trg_scientific_laws_updated_at
    BEFORE UPDATE ON scientific_laws
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_reasoning_sessions_updated_at
    BEFORE UPDATE ON reasoning_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_thought_nodes_updated_at
    BEFORE UPDATE ON thought_nodes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Example Queries (for reference)
-- ============================================================================

-- Find similar laws by semantic search:
-- SELECT law_name, definition, 1 - (embedding <=> $1) AS similarity
-- FROM scientific_laws
-- ORDER BY embedding <=> $1
-- LIMIT 5;

-- Get all descendants of a thought node:
-- SELECT * FROM thought_nodes WHERE path <@ 'root.hypothesis_1';

-- Get direct children of a thought node:
-- SELECT * FROM thought_nodes WHERE path ~ 'root.hypothesis_1.*{1}';

-- Get ancestors of a thought node:
-- SELECT * FROM thought_nodes WHERE path @> 'root.hypothesis_1.sub_1.sub_2';

-- ============================================================================
-- Self-Learning Extension (Phase 4)
-- ============================================================================

-- Add domain-specific weights to laws
ALTER TABLE scientific_laws ADD COLUMN IF NOT EXISTS
    domain_weights JSONB DEFAULT '{}';
-- Example: {"ui_design": 0.85, "architecture": 0.6, "debugging": 0.7}

-- Table: outcome_records
-- Tracks session outcomes for learning
CREATE TABLE IF NOT EXISTS outcome_records (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES reasoning_sessions(id) ON DELETE CASCADE,
    outcome         VARCHAR(20) NOT NULL 
                    CHECK (outcome IN ('success', 'partial', 'failure')),
    confidence      DECIMAL(3,2) NOT NULL DEFAULT 1.0
                    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    winning_path    LTREE,
    notes           TEXT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_outcome_records_session 
    ON outcome_records (session_id);

CREATE INDEX IF NOT EXISTS idx_outcome_records_path 
    ON outcome_records USING gist (winning_path);

-- View: law_domain_stats
-- Aggregates success/failure rates per law per domain
CREATE OR REPLACE VIEW law_domain_stats AS
SELECT 
    sl.id as law_id,
    sl.law_name,
    -- Extract domain hint from session goal (first 50 chars as proxy)
    SUBSTRING(rs.goal, 1, 50) AS domain_hint,
    COUNT(*) FILTER (WHERE orec.outcome = 'success') as successes,
    COUNT(*) FILTER (WHERE orec.outcome = 'partial') as partials,
    COUNT(*) FILTER (WHERE orec.outcome = 'failure') as failures,
    COUNT(*) as total_outcomes,
    ROUND(
        COUNT(*) FILTER (WHERE orec.outcome = 'success')::NUMERIC / 
        NULLIF(COUNT(*), 0), 
        3
    ) as success_rate
FROM scientific_laws sl
JOIN thought_nodes tn ON sl.id = ANY(tn.supporting_laws)
JOIN outcome_records orec ON orec.session_id = tn.session_id
  AND tn.path <@ orec.winning_path
JOIN reasoning_sessions rs ON rs.id = tn.session_id
GROUP BY sl.id, sl.law_name, SUBSTRING(rs.goal, 1, 50);

-- ============================================================================
-- Interview Self-Learning Extension (Phase 6)
-- Archives interview Q&A patterns for effectiveness analysis
-- ============================================================================

-- Table: interview_history
-- Stores completed interview Q&A for pattern learning
CREATE TABLE IF NOT EXISTS interview_history (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID NOT NULL REFERENCES reasoning_sessions(id) ON DELETE CASCADE,
    
    -- Question data
    question_id         VARCHAR(50) NOT NULL,   -- Original question ID from interview
    question_text       TEXT NOT NULL,
    question_embedding  vector(768),            -- For semantic clustering
    question_category   VARCHAR(100),           -- e.g., 'scale', 'consistency', 'aesthetics'
    
    -- Answer data
    answer_given        VARCHAR(10),            -- 'A', 'B', 'C', 'skipped'
    answer_text         TEXT,                   -- Full text of chosen answer
    
    -- Flow metadata
    position_in_flow    INTEGER NOT NULL,
    was_skipped         BOOLEAN DEFAULT FALSE,
    follow_up_triggered BOOLEAN DEFAULT FALSE,
    parent_question_id  UUID REFERENCES interview_history(id),
    
    -- Domain extraction (from session goal)
    goal_domain         VARCHAR(100),
    
    -- Timestamps
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for analytics queries
CREATE INDEX IF NOT EXISTS idx_interview_history_session 
    ON interview_history(session_id);

CREATE INDEX IF NOT EXISTS idx_interview_history_domain 
    ON interview_history(goal_domain);

CREATE INDEX IF NOT EXISTS idx_interview_history_category 
    ON interview_history(question_category);

CREATE INDEX IF NOT EXISTS idx_interview_history_embedding 
    ON interview_history 
    USING hnsw (question_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- View: interview_effectiveness
-- Correlates question patterns with session outcomes
CREATE OR REPLACE VIEW interview_effectiveness AS
SELECT 
    ih.goal_domain,
    ih.question_category,
    ih.answer_given,
    COUNT(*) as times_asked,
    COUNT(*) FILTER (WHERE orec.outcome = 'success') as led_to_success,
    COUNT(*) FILTER (WHERE orec.outcome = 'failure') as led_to_failure,
    ROUND(
        COUNT(*) FILTER (WHERE orec.outcome = 'success')::NUMERIC / 
        NULLIF(COUNT(*), 0), 
        3
    ) as success_rate
FROM interview_history ih
LEFT JOIN outcome_records orec ON orec.session_id = ih.session_id
GROUP BY ih.goal_domain, ih.question_category, ih.answer_given
ORDER BY times_asked DESC;
