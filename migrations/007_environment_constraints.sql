-- Migration 007: Environment Constraints
-- PAS Session: fea58ef5-3773-46ac-b2d2-359f2283ba29 (Score: 0.96)
-- Phase 7c: ASPIRATIONAL constraints system

-- User preferences (global defaults, future multi-user)
CREATE TABLE IF NOT EXISTS user_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL DEFAULT 'default',
    preference_key TEXT NOT NULL,
    preference_value JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, preference_key)
);

-- Project constraints with temporal versioning
CREATE TABLE IF NOT EXISTS project_constraints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES project_registry(id) ON DELETE CASCADE,
    constraint_type TEXT NOT NULL CHECK (constraint_type IN ('philosophy', 'environment', 'quality')),
    constraint_key TEXT NOT NULL,
    constraint_data JSONB NOT NULL,
    constraint_embedding VECTOR(768),  -- For drift detection (mpnet dimensions)
    enforcement_level TEXT DEFAULT 'warn' CHECK (enforcement_level IN ('hidden', 'warn', 'block')),
    source TEXT NOT NULL CHECK (source IN ('manual', 'inferred', 'gemini_md')),
    version INT DEFAULT 1,
    valid_from TIMESTAMPTZ DEFAULT NOW(),
    valid_to TIMESTAMPTZ,  -- NULL = active
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_project_constraints_active 
    ON project_constraints(project_id, constraint_type) 
    WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS idx_project_constraints_key
    ON project_constraints(project_id, constraint_key)
    WHERE valid_to IS NULL;

-- Seed default philosophy constraints for existing projects
INSERT INTO project_constraints (project_id, constraint_type, constraint_key, constraint_data, enforcement_level, source)
SELECT 
    id as project_id,
    'philosophy',
    'no_mvp',
    'true'::jsonb,
    'block',
    'manual'
FROM project_registry
WHERE NOT EXISTS (
    SELECT 1 FROM project_constraints pc 
    WHERE pc.project_id = project_registry.id 
    AND pc.constraint_key = 'no_mvp'
    AND pc.valid_to IS NULL
);

INSERT INTO project_constraints (project_id, constraint_type, constraint_key, constraint_data, enforcement_level, source)
SELECT 
    id as project_id,
    'philosophy',
    'dual_plan',
    'true'::jsonb,
    'warn',
    'manual'
FROM project_registry
WHERE NOT EXISTS (
    SELECT 1 FROM project_constraints pc 
    WHERE pc.project_id = project_registry.id 
    AND pc.constraint_key = 'dual_plan'
    AND pc.valid_to IS NULL
);

INSERT INTO project_constraints (project_id, constraint_type, constraint_key, constraint_data, enforcement_level, source)
SELECT 
    id as project_id,
    'philosophy',
    'code_quality',
    '"production_grade"'::jsonb,
    'warn',
    'manual'
FROM project_registry
WHERE NOT EXISTS (
    SELECT 1 FROM project_constraints pc 
    WHERE pc.project_id = project_registry.id 
    AND pc.constraint_key = 'code_quality'
    AND pc.valid_to IS NULL
);
