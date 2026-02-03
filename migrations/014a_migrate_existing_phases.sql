-- ============================================================================
-- v88: Migrate existing phases to roadmaps
-- Run AFTER 014_governance_v88.sql
-- ============================================================================

-- Create default roadmap for each project with existing phases
INSERT INTO roadmaps (project_id, title, version_tag, status)
SELECT DISTINCT 
    project_id, 
    'Default Roadmap', 
    'v1',
    'active'
FROM roadmap_phases
WHERE roadmap_id IS NULL
ON CONFLICT DO NOTHING;

-- Backfill roadmap_id for orphaned phases
UPDATE roadmap_phases rp
SET roadmap_id = rm.id
FROM roadmaps rm
WHERE rp.project_id = rm.project_id
  AND rp.roadmap_id IS NULL
  AND rm.title = 'Default Roadmap';
