-- ============================================================================
-- Phase 6: Project Governance Architecture
-- Migration 009: Create governance tables (Vision → Roadmap → Plans)
-- PAS Session: 18e98d43-9bd9-4c56-a8c5-036e5e9c8fd1 | Score: 0.927
-- ============================================================================

-- project_vision: High-level project vision linked to project_registry
CREATE TABLE IF NOT EXISTS project_vision (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id          TEXT NOT NULL UNIQUE,  -- Loose coupling, not FK
    
    -- Vision content
    mission             TEXT,                   -- Core mission statement
    user_needs          TEXT[],                 -- Who this serves
    strategic_goals     JSONB DEFAULT '{}',     -- {goal: priority}
    
    -- Semantic search
    embedding           vector(768),
    
    -- Metadata
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_project_vision_project_id
    ON project_vision(project_id);

CREATE INDEX IF NOT EXISTS idx_project_vision_embedding
    ON project_vision
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- roadmap_phases: Phases within a project roadmap
CREATE TABLE IF NOT EXISTS roadmap_phases (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id          TEXT NOT NULL,          -- Links to project_vision.project_id
    
    -- Phase definition
    phase_name          TEXT NOT NULL,
    description         TEXT,
    status              TEXT NOT NULL DEFAULT 'planned'
                        CHECK (status IN ('planned', 'active', 'complete', 'blocked')),
    sequence            INTEGER NOT NULL,       -- Ordering
    
    -- Metadata
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(project_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_roadmap_phases_project_id
    ON roadmap_phases(project_id);

CREATE INDEX IF NOT EXISTS idx_roadmap_phases_status
    ON roadmap_phases(status);

-- artifacts: Versioned documents (roadmaps, plans, walkthroughs)
CREATE TABLE IF NOT EXISTS artifacts (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Linking
    project_id          TEXT NOT NULL,
    session_id          UUID REFERENCES reasoning_sessions(id) ON DELETE SET NULL,
    roadmap_phase_id    UUID REFERENCES roadmap_phases(id) ON DELETE SET NULL,
    source_verbatim_log_id UUID REFERENCES conversation_log(id) ON DELETE SET NULL,
    
    -- Content
    artifact_type       TEXT NOT NULL
                        CHECK (artifact_type IN ('roadmap', 'implementation_plan', 'walkthrough', 'vision', 'other')),
    name                TEXT NOT NULL,
    content             TEXT NOT NULL,
    
    -- Versioning (SELECT MAX+1 in transaction)
    version             INTEGER NOT NULL DEFAULT 1,
    
    -- Tags for filtering
    tags                TEXT[] DEFAULT '{}',
    
    -- Semantic search
    content_embedding   vector(768),
    
    -- Metadata
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(project_id, name, version)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_project_id
    ON artifacts(project_id);

CREATE INDEX IF NOT EXISTS idx_artifacts_type
    ON artifacts(artifact_type);

CREATE INDEX IF NOT EXISTS idx_artifacts_tags
    ON artifacts USING GIN (tags);

CREATE INDEX IF NOT EXISTS idx_artifacts_session
    ON artifacts(session_id);

CREATE INDEX IF NOT EXISTS idx_artifacts_embedding
    ON artifacts
    USING hnsw (content_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- View: governance_hierarchy
-- Shows Vision → Phases → Artifacts hierarchy
CREATE OR REPLACE VIEW governance_hierarchy AS
SELECT 
    pv.project_id,
    pv.mission,
    rp.id as phase_id,
    rp.phase_name,
    rp.status as phase_status,
    rp.sequence,
    a.id as artifact_id,
    a.name as artifact_name,
    a.artifact_type,
    a.version,
    a.tags
FROM project_vision pv
LEFT JOIN roadmap_phases rp ON rp.project_id = pv.project_id
LEFT JOIN artifacts a ON a.roadmap_phase_id = rp.id
ORDER BY pv.project_id, rp.sequence, a.created_at;
