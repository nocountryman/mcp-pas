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

-- ============================================================================
-- v34: Auto-Tagging on Record Outcome
-- Stores suggested_tags from finalize_session for auto-application on success
-- ============================================================================
ALTER TABLE reasoning_sessions ADD COLUMN IF NOT EXISTS
    suggested_tags JSONB DEFAULT '[]'::jsonb;

-- Example: {"ui_design": 0.85, "architecture": 0.6, "debugging": 0.7}

-- ============================================================================
-- v12b: Thompson Sampling for Law Selection
-- Tracks selection/success counts for explore-exploit balancing
-- ============================================================================
ALTER TABLE scientific_laws ADD COLUMN IF NOT EXISTS
    selection_count INTEGER DEFAULT 0;
ALTER TABLE scientific_laws ADD COLUMN IF NOT EXISTS
    success_count INTEGER DEFAULT 0;

-- ============================================================================
-- v14c.1: Law Failure Modes for Targeted Critique Prompts
-- Common failure patterns to guide LLM critiques
-- ============================================================================
ALTER TABLE scientific_laws ADD COLUMN IF NOT EXISTS
    failure_modes TEXT[] DEFAULT '{}';

-- ============================================================================
-- v12a: Training Data Collection for PRM
-- Stores hypothesis+outcome pairs for future fine-tuning
-- ============================================================================
CREATE TABLE IF NOT EXISTS training_data (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hypothesis_text TEXT NOT NULL,
    goal_text       TEXT NOT NULL,
    outcome         VARCHAR(20) CHECK (outcome IN ('success', 'partial', 'failure')),
    depth           INTEGER,
    law_name        VARCHAR(255),
    session_id      UUID REFERENCES reasoning_sessions(id),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_data_outcome 
    ON training_data(outcome);

-- ============================================================================
-- v13c: Critique Calibration - Track critique accuracy for self-learning
-- Stores critique→outcome links to calibrate future critique weights
-- ============================================================================
CREATE TABLE IF NOT EXISTS critique_accuracy (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES reasoning_sessions(id) ON DELETE CASCADE,
    node_id         UUID NOT NULL REFERENCES thought_nodes(id) ON DELETE CASCADE,
    critique_severity DECIMAL(3,2), -- Original severity score (0.0-1.0)
    persona         VARCHAR(50),    -- Which critic persona (if any)
    was_top_hypothesis BOOLEAN DEFAULT false, -- Was this the winning hypothesis?
    actual_outcome  VARCHAR(20) CHECK (actual_outcome IN ('success', 'partial', 'failure')),
    critique_accurate BOOLEAN, -- Did the critique correctly predict failure?
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_critique_accuracy_session 
    ON critique_accuracy(session_id);
CREATE INDEX IF NOT EXISTS idx_critique_accuracy_persona 
    ON critique_accuracy(persona);


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
    failure_reason  TEXT,                   -- v15b: explicit failure reason for learning
    scope_embedding vector(768),            -- v15b: embedded scope for semantic matching
    failure_reason_embedding vector(768),   -- v27: embedded failure_reason for semantic search
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- v15b: Index for semantic scope search
CREATE INDEX IF NOT EXISTS idx_outcome_scope_embedding 
    ON outcome_records USING ivfflat (scope_embedding vector_cosine_ops) WITH (lists = 10);

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
-- v26: Session Tagging System
-- Enables semantic tagging and retrieval of reasoning sessions
-- ============================================================================

-- Table: session_tags
-- Many-to-many relationship between sessions and tags
CREATE TABLE IF NOT EXISTS session_tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES reasoning_sessions(id) ON DELETE CASCADE,
    tag VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(session_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_session_tags_session ON session_tags(session_id);
CREATE INDEX IF NOT EXISTS idx_session_tags_tag ON session_tags(tag);

-- Table: tag_aliases
-- Normalizes tag variations to canonical forms
CREATE TABLE IF NOT EXISTS tag_aliases (
    alias VARCHAR(100) PRIMARY KEY,
    canonical_tag VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tag_aliases_canonical ON tag_aliases(canonical_tag);

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

-- ============================================================================
-- Benchmark Framework (Phase 7)
-- Persistent test cases and results for quality measurement
-- ============================================================================

-- Table: benchmark_test_cases
-- Standardized test goals with known correct answers
CREATE TABLE IF NOT EXISTS benchmark_test_cases (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id         VARCHAR(10) NOT NULL UNIQUE,  -- e.g., 'R1', 'N2', 'B1'
    category        VARCHAR(50) NOT NULL 
                    CHECK (category IN ('refactoring', 'new_feature', 'bug_fix', 'api_design', 'edge_case')),
    goal_text       TEXT NOT NULL,
    goal_embedding  vector(768),
    
    -- Ground truth for scoring
    expected_scope  TEXT[] NOT NULL DEFAULT '{}',      -- Files/modules should be in scope
    expected_considerations TEXT[] NOT NULL DEFAULT '{}', -- Laws/concerns to consider
    anti_patterns   TEXT[] DEFAULT '{}',              -- Patterns to avoid
    
    -- Metadata
    difficulty      VARCHAR(20) DEFAULT 'medium'
                    CHECK (difficulty IN ('easy', 'medium', 'hard')),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: benchmark_results
-- Results from running the benchmark suite
CREATE TABLE IF NOT EXISTS benchmark_results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- What was tested
    test_case_id    UUID NOT NULL REFERENCES benchmark_test_cases(id),
    session_id      UUID REFERENCES reasoning_sessions(id),
    model_name      VARCHAR(100) NOT NULL,           -- e.g., 'claude-sonnet-4.5'
    prompt_variant  VARCHAR(100) NOT NULL DEFAULT 'v1', -- e.g., 'v2_two_phase'
    
    -- Metrics (all 0.0-1.0)
    scope_accuracy      DECIMAL(5,4),  -- Jaccard similarity declared vs expected
    hypothesis_relevance DECIMAL(5,4), -- Embedding similarity to goal
    critique_coverage   DECIMAL(5,4),  -- % hypotheses critiqued
    tree_depth          INTEGER,       -- Max depth reached
    
    -- Composite score (computed)
    composite_score     DECIMAL(5,2) GENERATED ALWAYS AS (
        (0.40 * COALESCE(scope_accuracy, 0) +
         0.30 * COALESCE(hypothesis_relevance, 0) +
         0.20 * COALESCE(critique_coverage, 0) +
         0.10 * LEAST(COALESCE(tree_depth::DECIMAL / 3, 0), 1.0)) * 100
    ) STORED,
    
    -- Raw data
    declared_scope  TEXT[],
    notes           TEXT,
    
    -- Metadata
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for benchmark queries
CREATE INDEX IF NOT EXISTS idx_benchmark_results_model 
    ON benchmark_results(model_name, prompt_variant);

CREATE INDEX IF NOT EXISTS idx_benchmark_results_test_case 
    ON benchmark_results(test_case_id);

-- View: benchmark_summary
-- Aggregate scores by model and prompt variant
CREATE OR REPLACE VIEW benchmark_summary AS
SELECT 
    br.model_name,
    br.prompt_variant,
    COUNT(*) as tests_run,
    ROUND(AVG(br.scope_accuracy), 3) as avg_scope_accuracy,
    ROUND(AVG(br.hypothesis_relevance), 3) as avg_relevance,
    ROUND(AVG(br.critique_coverage), 3) as avg_critique_coverage,
    ROUND(AVG(br.composite_score), 1) as avg_composite_score,
    MIN(br.composite_score) as min_score,
    MAX(br.composite_score) as max_score,
    MAX(br.created_at) as last_run
FROM benchmark_results br
GROUP BY br.model_name, br.prompt_variant
ORDER BY avg_composite_score DESC;

-- ============================================================================
-- v19: Interview Domain System
-- Domain-driven question trees with dimension tracking
-- ============================================================================

-- Table: interview_domains
-- Predefined domains (ui_design, api_design, coding_task, etc.)
CREATE TABLE IF NOT EXISTS interview_domains (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_name     VARCHAR(100) NOT NULL UNIQUE,
    description     TEXT NOT NULL,
    example_goals   TEXT[] DEFAULT '{}',    -- Sample goals for this domain
    embedding       vector(768),             -- For semantic matching
    is_active       BOOLEAN DEFAULT true,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- HNSW index for domain similarity search
CREATE INDEX IF NOT EXISTS idx_interview_domains_embedding
    ON interview_domains
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Table: interview_dimensions
-- Dimensions per domain (colors, layout, scope, constraints, etc.)
CREATE TABLE IF NOT EXISTS interview_dimensions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id       UUID NOT NULL REFERENCES interview_domains(id) ON DELETE CASCADE,
    dimension_name  VARCHAR(100) NOT NULL,
    description     TEXT,
    is_required     BOOLEAN DEFAULT true,   -- Must be answered for domain complete
    priority        INTEGER DEFAULT 50,      -- Lower = ask first
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(domain_id, dimension_name)
);

CREATE INDEX IF NOT EXISTS idx_interview_dimensions_domain
    ON interview_dimensions(domain_id);

-- Table: interview_questions
-- Question templates per dimension with choices
CREATE TABLE IF NOT EXISTS interview_questions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dimension_id    UUID NOT NULL REFERENCES interview_dimensions(id) ON DELETE CASCADE,
    question_template TEXT NOT NULL,
    question_type   VARCHAR(20) NOT NULL CHECK (question_type IN ('simple', 'complex')),
    choices         JSONB NOT NULL DEFAULT '[]',  -- [{label, description, pros, cons}]
    is_default      BOOLEAN DEFAULT true,         -- Use as default question for dimension
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_interview_questions_dimension
    ON interview_questions(dimension_id);

-- Table: interview_answers (v19)
-- Stores answers for conflict detection across sessions
CREATE TABLE IF NOT EXISTS interview_answers (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES reasoning_sessions(id) ON DELETE CASCADE,
    dimension_id    UUID NOT NULL REFERENCES interview_dimensions(id),
    question_id     UUID REFERENCES interview_questions(id),
    answer_label    VARCHAR(10),            -- 'A', 'B', 'C', etc.
    answer_text     TEXT,                   -- Full answer description
    answer_embedding vector(768),           -- For semantic conflict detection
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(session_id, dimension_id)  -- One answer per dimension per session
);

CREATE INDEX IF NOT EXISTS idx_interview_answers_session
    ON interview_answers(session_id);

CREATE INDEX IF NOT EXISTS idx_interview_answers_dimension
    ON interview_answers(dimension_id);

CREATE INDEX IF NOT EXISTS idx_interview_answers_embedding
    ON interview_answers
    USING hnsw (answer_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- View: dimension_coverage
-- Shows which dimensions are covered per session
CREATE OR REPLACE VIEW dimension_coverage AS
SELECT 
    s.id as session_id,
    d.id as domain_id,
    d.domain_name,
    dim.id as dimension_id,
    dim.dimension_name,
    dim.is_required,
    CASE WHEN a.id IS NOT NULL THEN true ELSE false END as is_covered,
    a.answer_label,
    a.answer_text
FROM reasoning_sessions s
CROSS JOIN interview_domains d
JOIN interview_dimensions dim ON dim.domain_id = d.id
LEFT JOIN interview_answers a ON a.session_id = s.id AND a.dimension_id = dim.id
WHERE d.is_active = true;


-- ============================================================================
-- v25: Conversation Logging
-- Stores free-form user text for semantic retrieval and provenance tracking
-- ============================================================================

-- Table: conversation_log
-- Captures raw user input that inspired hypotheses, feedback, extended answers
CREATE TABLE IF NOT EXISTS conversation_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES reasoning_sessions(id) ON DELETE CASCADE,
    thought_node_id UUID REFERENCES thought_nodes(id) ON DELETE SET NULL,  -- nullable link
    user_id         VARCHAR(64),    -- Optional user identifier for multi-user scenarios
    
    -- Content
    log_type        VARCHAR(30) NOT NULL CHECK (log_type IN (
        'user_input',       -- Raw user text that inspired a hypothesis
        'feedback',         -- User feedback on results
        'interview_answer', -- Extended answer beyond A/B/C
        'context'           -- Background context provided by user
    )),
    raw_text        TEXT NOT NULL,  -- Full text, no truncation (TOAST auto-compresses >2KB)
    
    -- Embeddings (support chunked for long text)
    embedding       vector(768),     -- Primary embedding (first chunk or summary)
    chunk_index     INTEGER DEFAULT 0,  -- 0 = primary, 1+ = additional chunks
    total_chunks    INTEGER DEFAULT 1,  -- Total chunks for this log entry
    
    -- Metadata
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- HNSW index for semantic search
CREATE INDEX IF NOT EXISTS idx_conversation_log_embedding
    ON conversation_log
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_conversation_log_session
    ON conversation_log(session_id);

CREATE INDEX IF NOT EXISTS idx_conversation_log_type
    ON conversation_log(log_type);

CREATE INDEX IF NOT EXISTS idx_conversation_log_user
    ON conversation_log(user_id);

-- ============================================================================
-- v30: Global Codebase Understanding
-- File indexing with semantic search and project isolation
-- ============================================================================

-- file_registry: Core file index with semantic vectors
CREATE TABLE IF NOT EXISTS file_registry (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      TEXT NOT NULL,   -- Isolation boundary (normalized from path)
    file_path       TEXT NOT NULL,   -- Relative path within project
    file_hash       TEXT NOT NULL,   -- SHA-256 for change detection
    language        TEXT,            -- Detected language (python, javascript, etc.)
    line_count      INTEGER,         -- File size in lines
    content_embedding vector(768),   -- Semantic embedding of file content
    
    -- Metadata
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(project_id, file_path)
);

-- HNSW index for semantic search on file content
CREATE INDEX IF NOT EXISTS idx_file_registry_embedding
    ON file_registry
    USING hnsw (content_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_file_registry_project
    ON file_registry(project_id);

-- file_symbols: Tree-sitter extracted structural symbols
CREATE TABLE IF NOT EXISTS file_symbols (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id         UUID NOT NULL REFERENCES file_registry(id) ON DELETE CASCADE,
    
    -- Symbol identification
    symbol_type     TEXT NOT NULL CHECK (symbol_type IN (
        'function', 'class', 'method', 'import', 'constant', 'variable'
    )),
    symbol_name     TEXT NOT NULL,
    
    -- Location in file
    line_start      INTEGER,
    line_end        INTEGER,
    
    -- Content for embedding
    signature       TEXT,            -- Function signature or class header
    docstring       TEXT,            -- Extracted docstring if present
    embedding       vector(768),     -- Semantic embedding of signature+docstring
    
    -- Hierarchy
    parent_symbol_id UUID REFERENCES file_symbols(id) ON DELETE CASCADE,
    
    -- Metadata
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- HNSW index for semantic search on symbols
CREATE INDEX IF NOT EXISTS idx_file_symbols_embedding
    ON file_symbols
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_file_symbols_file
    ON file_symbols(file_id);

CREATE INDEX IF NOT EXISTS idx_file_symbols_type
    ON file_symbols(symbol_type);
