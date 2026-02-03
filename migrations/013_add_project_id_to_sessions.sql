-- Migration: Add project_id column to reasoning_sessions
-- Purpose: Enable proper project-session relationship tracking
-- This enables session auto-detect in handoff system

-- Add project_id column
ALTER TABLE reasoning_sessions 
ADD COLUMN IF NOT EXISTS project_id VARCHAR(255);

-- Create index for project-based queries
CREATE INDEX IF NOT EXISTS idx_reasoning_sessions_project_id 
ON reasoning_sessions(project_id) WHERE project_id IS NOT NULL;

-- Migrate existing sessions: extract project_id from context JSONB
UPDATE reasoning_sessions 
SET project_id = context->>'project_id' 
WHERE context->>'project_id' IS NOT NULL 
  AND project_id IS NULL;

-- Add FK constraint (optional - projects may not always exist in registry)
-- SKIPPED: project_registry may not have all project_ids

COMMENT ON COLUMN reasoning_sessions.project_id IS 'Project this session belongs to. Enables session-project queries.';
