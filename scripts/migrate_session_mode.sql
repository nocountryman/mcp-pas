-- ============================================================================
-- Phase 8: Unified Reasoning Modes Migration
-- Adds session_mode and unconstrained columns to reasoning_sessions
-- ============================================================================

-- Add session_mode column (implementation vs research)
ALTER TABLE reasoning_sessions ADD COLUMN IF NOT EXISTS
    session_mode VARCHAR(20) DEFAULT 'implementation'
    CHECK (session_mode IN ('implementation', 'research'));

-- Add unconstrained flag for research mode (opt-out of project constraints)
ALTER TABLE reasoning_sessions ADD COLUMN IF NOT EXISTS
    unconstrained BOOLEAN DEFAULT false;

-- Extend outcome types to include knowledge_gained for research mode
ALTER TABLE outcome_records DROP CONSTRAINT IF EXISTS outcome_records_outcome_check;
ALTER TABLE outcome_records ADD CONSTRAINT outcome_records_outcome_check
    CHECK (outcome IN ('success', 'partial', 'failure', 'knowledge_gained'));

-- Backfill existing sessions as implementation mode
UPDATE reasoning_sessions 
SET session_mode = 'implementation', unconstrained = false 
WHERE session_mode IS NULL;

-- Update schema.sql reference (for documentation)
COMMENT ON COLUMN reasoning_sessions.session_mode IS 
    'Phase 8: implementation (code changes) vs research (exploration/design)';
COMMENT ON COLUMN reasoning_sessions.unconstrained IS 
    'Phase 8: If true in research mode, skip project constraint surfacing';
