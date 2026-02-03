-- v89 Governance Schema Extension
-- Adds: dual_recommendation, cross_phase_decisions, findings_data

-- 1. Add dual_recommendation to roadmap_phases
ALTER TABLE roadmap_phases 
ADD COLUMN IF NOT EXISTS dual_recommendation JSONB DEFAULT '{}'::jsonb;

COMMENT ON COLUMN roadmap_phases.dual_recommendation IS 
'Template field: {balanced: {...}, aspirational: {...}}';

-- 2. Add findings_data to artifacts (for research type)
ALTER TABLE artifacts 
ADD COLUMN IF NOT EXISTS findings_data JSONB DEFAULT NULL;

COMMENT ON COLUMN artifacts.findings_data IS 
'Structured research findings: {findings: [{source, type, text}], confidence_level: high/medium/low}';

-- 3. Create cross_phase_decisions table
CREATE TABLE IF NOT EXISTS cross_phase_decisions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id          TEXT NOT NULL,
    decision_summary    TEXT NOT NULL,
    options_considered  JSONB DEFAULT '[]'::jsonb,
    chosen_option       TEXT,
    rationale           TEXT,
    pas_node_id         UUID REFERENCES thought_nodes(id) ON DELETE SET NULL,
    phase_ids           UUID[] DEFAULT '{}',
    embedding           vector(768),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cross_phase_decisions_project 
    ON cross_phase_decisions(project_id);
CREATE INDEX IF NOT EXISTS idx_cross_phase_decisions_embedding 
    ON cross_phase_decisions USING hnsw (embedding vector_cosine_ops);

-- Add audit trigger
DROP TRIGGER IF EXISTS cross_phase_decisions_audit ON cross_phase_decisions;
CREATE TRIGGER cross_phase_decisions_audit
    AFTER INSERT OR UPDATE OR DELETE ON cross_phase_decisions
    FOR EACH ROW EXECUTE FUNCTION governance_audit_trigger();

-- 4. Update artifacts CHECK constraint to include 'research' type
ALTER TABLE artifacts DROP CONSTRAINT IF EXISTS artifacts_artifact_type_check;
ALTER TABLE artifacts ADD CONSTRAINT artifacts_artifact_type_check 
    CHECK (artifact_type = ANY (ARRAY['roadmap', 'implementation_plan', 'walkthrough', 'vision', 'research', 'other']));

