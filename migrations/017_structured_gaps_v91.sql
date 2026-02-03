-- v91: Structured Gaps Table
-- Stores gaps identified from sequential analysis for tracking and addressing

CREATE TABLE IF NOT EXISTS structured_gaps (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES reasoning_sessions(id) ON DELETE CASCADE,
    project_id      TEXT NOT NULL,
    gap_layer       TEXT NOT NULL CHECK (gap_layer IN (
        'code_structure', 'dependencies', 'data_flow', 'interfaces', 'workflows'
    )),
    gap_description TEXT NOT NULL,
    severity        TEXT DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    addressed       BOOLEAN DEFAULT FALSE,
    addressed_by    UUID REFERENCES thought_nodes(id) ON DELETE SET NULL,
    embedding       vector(768),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_structured_gaps_session ON structured_gaps(session_id);
CREATE INDEX IF NOT EXISTS idx_structured_gaps_project ON structured_gaps(project_id);
CREATE INDEX IF NOT EXISTS idx_structured_gaps_unaddressed ON structured_gaps(project_id) WHERE NOT addressed;
CREATE INDEX IF NOT EXISTS idx_structured_gaps_embedding ON structured_gaps USING hnsw (embedding vector_cosine_ops);

-- Comments
COMMENT ON TABLE structured_gaps IS 'Gaps identified from sequential analysis (v37+)';
COMMENT ON COLUMN structured_gaps.gap_layer IS 'One of: code_structure, dependencies, data_flow, interfaces, workflows';
COMMENT ON COLUMN structured_gaps.addressed_by IS 'Thought node that addressed this gap';

-- Audit trigger
CREATE TRIGGER structured_gaps_audit
    AFTER INSERT OR UPDATE OR DELETE ON structured_gaps
    FOR EACH ROW EXECUTE FUNCTION governance_audit_trigger();
