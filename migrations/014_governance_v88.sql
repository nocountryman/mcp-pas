-- ============================================================================
-- v88: Governance Schema Enhancement
-- PAS Research Session: b995dca2-6f1e-47f7-8e01-a0e5452026ce
-- PAS Implementation Session: 25456d82-d9ee-49b7-b61f-4f3e82068d41
-- ============================================================================

-- 1. Add problem_statement to project_vision
ALTER TABLE project_vision 
ADD COLUMN IF NOT EXISTS problem_statement TEXT;

-- 2. Create roadmaps table (container for phases)
CREATE TABLE IF NOT EXISTS roadmaps (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id          TEXT NOT NULL,
    title               TEXT NOT NULL,
    version_tag         TEXT,
    status              TEXT NOT NULL DEFAULT 'active' 
                        CHECK (status IN ('draft', 'active', 'archived')),
    priority_taxonomy   JSONB DEFAULT '{}'::jsonb,
    architecture_content TEXT,
    embedding           vector(768),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_roadmaps_project_id ON roadmaps(project_id);
CREATE INDEX IF NOT EXISTS idx_roadmaps_status ON roadmaps(status);
CREATE INDEX IF NOT EXISTS idx_roadmaps_embedding 
    ON roadmaps USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);

-- 3. Extend roadmap_phases with new columns
ALTER TABLE roadmap_phases 
ADD COLUMN IF NOT EXISTS roadmap_id UUID REFERENCES roadmaps(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS priority TEXT DEFAULT 'P2' 
    CHECK (priority IN ('P0', 'P1', 'P2', 'P3')),
ADD COLUMN IF NOT EXISTS goal TEXT,
ADD COLUMN IF NOT EXISTS scope_content TEXT,
ADD COLUMN IF NOT EXISTS lsp_pre_check TEXT,
ADD COLUMN IF NOT EXISTS affected_files TEXT[] DEFAULT '{}',
ADD COLUMN IF NOT EXISTS pas_session_id UUID REFERENCES reasoning_sessions(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS verification_notes TEXT;

CREATE INDEX IF NOT EXISTS idx_roadmap_phases_roadmap_id ON roadmap_phases(roadmap_id);
CREATE INDEX IF NOT EXISTS idx_roadmap_phases_priority ON roadmap_phases(priority);

-- 4. Create phase_success_criteria
CREATE TABLE IF NOT EXISTS phase_success_criteria (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phase_id                UUID NOT NULL REFERENCES roadmap_phases(id) ON DELETE CASCADE,
    criterion               TEXT NOT NULL,
    done                    BOOLEAN DEFAULT FALSE,
    verified_at             TIMESTAMPTZ,
    verified_by_session_id  UUID REFERENCES reasoning_sessions(id) ON DELETE SET NULL,
    sequence                INTEGER NOT NULL,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_phase_success_criteria_phase ON phase_success_criteria(phase_id);

-- 5. Create phase_dependencies (self-referential many-to-many)
CREATE TABLE IF NOT EXISTS phase_dependencies (
    phase_id            UUID NOT NULL REFERENCES roadmap_phases(id) ON DELETE CASCADE,
    depends_on_phase_id UUID NOT NULL REFERENCES roadmap_phases(id) ON DELETE CASCADE,
    PRIMARY KEY (phase_id, depends_on_phase_id),
    CHECK (phase_id != depends_on_phase_id)
);

-- 6. Create phase_critiques
CREATE TABLE IF NOT EXISTS phase_critiques (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phase_id                UUID NOT NULL REFERENCES roadmap_phases(id) ON DELETE CASCADE,
    critique_text           TEXT NOT NULL,
    status                  TEXT DEFAULT 'open' 
                            CHECK (status IN ('addressed', 'warning', 'open')),
    addressed_in_session_id UUID REFERENCES reasoning_sessions(id) ON DELETE SET NULL,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_phase_critiques_phase ON phase_critiques(phase_id);
CREATE INDEX IF NOT EXISTS idx_phase_critiques_status ON phase_critiques(status);

-- 7. Create governance_audit_log
CREATE TABLE IF NOT EXISTS governance_audit_log (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id          TEXT NOT NULL,
    table_name          TEXT NOT NULL,
    record_id           UUID NOT NULL,
    action              TEXT NOT NULL CHECK (action IN ('insert', 'update', 'delete')),
    old_values          JSONB,
    new_values          JSONB,
    changed_by_session_id UUID,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_governance_audit_log_project ON governance_audit_log(project_id);
CREATE INDEX IF NOT EXISTS idx_governance_audit_log_table ON governance_audit_log(table_name);
CREATE INDEX IF NOT EXISTS idx_governance_audit_log_created ON governance_audit_log(created_at);

-- 8. Create audit trigger function
CREATE OR REPLACE FUNCTION governance_audit_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO governance_audit_log (project_id, table_name, record_id, action, new_values)
        VALUES (
            COALESCE(NEW.project_id, ''),
            TG_TABLE_NAME,
            NEW.id,
            'insert',
            to_jsonb(NEW)
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO governance_audit_log (project_id, table_name, record_id, action, old_values, new_values)
        VALUES (
            COALESCE(NEW.project_id, COALESCE(OLD.project_id, '')),
            TG_TABLE_NAME,
            NEW.id,
            'update',
            to_jsonb(OLD),
            to_jsonb(NEW)
        );
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO governance_audit_log (project_id, table_name, record_id, action, old_values)
        VALUES (
            COALESCE(OLD.project_id, ''),
            TG_TABLE_NAME,
            OLD.id,
            'delete',
            to_jsonb(OLD)
        );
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- 9. Create triggers (on tables that modify governance state)
DROP TRIGGER IF EXISTS roadmaps_audit ON roadmaps;
CREATE TRIGGER roadmaps_audit
    AFTER INSERT OR UPDATE OR DELETE ON roadmaps
    FOR EACH ROW EXECUTE FUNCTION governance_audit_trigger();

DROP TRIGGER IF EXISTS roadmap_phases_audit ON roadmap_phases;
CREATE TRIGGER roadmap_phases_audit
    AFTER INSERT OR UPDATE OR DELETE ON roadmap_phases
    FOR EACH ROW EXECUTE FUNCTION governance_audit_trigger();

-- 10. Update governance_hierarchy view
CREATE OR REPLACE VIEW governance_hierarchy AS
SELECT 
    pv.project_id,
    pv.mission,
    pv.problem_statement,
    rm.id as roadmap_id,
    rm.title as roadmap_title,
    rm.version_tag,
    rm.status as roadmap_status,
    rp.id as phase_id,
    rp.phase_name,
    rp.priority,
    rp.goal,
    rp.status as phase_status,
    rp.sequence,
    a.id as artifact_id,
    a.name as artifact_name,
    a.artifact_type,
    a.version,
    a.tags
FROM project_vision pv
LEFT JOIN roadmaps rm ON rm.project_id = pv.project_id
LEFT JOIN roadmap_phases rp ON rp.roadmap_id = rm.id OR (rp.roadmap_id IS NULL AND rp.project_id = pv.project_id)
LEFT JOIN artifacts a ON a.roadmap_phase_id = rp.id
ORDER BY pv.project_id, rm.created_at, rp.sequence, a.created_at;
