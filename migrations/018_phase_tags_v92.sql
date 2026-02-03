-- v92: Add tags to roadmap_phases
-- Enables categorization like 'research', 'implementation', 'mcp', etc.

ALTER TABLE roadmap_phases ADD COLUMN IF NOT EXISTS tags TEXT[] DEFAULT '{}';

CREATE INDEX IF NOT EXISTS idx_roadmap_phases_tags ON roadmap_phases USING gin(tags);

COMMENT ON COLUMN roadmap_phases.tags IS 'Array of tags for categorization (e.g., research, mcp, implementation)';
