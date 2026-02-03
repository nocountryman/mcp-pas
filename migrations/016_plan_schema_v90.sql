-- v90: Implementation Plan Schema Extension
-- Adds structured storage for implementation plan steps, scope, and critique linkage

-- 1. Plan steps table (normalized for status tracking)
CREATE TABLE IF NOT EXISTS plan_steps (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id     UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    step_order      INTEGER NOT NULL,
    content         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending' 
                    CHECK (status IN ('pending', 'in_progress', 'done', 'skipped')),
    pas_node_id     UUID REFERENCES thought_nodes(id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_plan_steps_artifact ON plan_steps(artifact_id);
CREATE INDEX IF NOT EXISTS idx_plan_steps_status ON plan_steps(status);

COMMENT ON TABLE plan_steps IS 'Implementation plan steps with execution tracking';

-- 2. Plan scope table (file change declarations)
CREATE TABLE IF NOT EXISTS plan_scope (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id     UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    file_path       TEXT NOT NULL,
    change_type     TEXT NOT NULL CHECK (change_type IN ('modify', 'create', 'delete')),
    lsp_refs        JSONB DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_plan_scope_artifact ON plan_scope(artifact_id);
CREATE INDEX IF NOT EXISTS idx_plan_scope_file ON plan_scope(file_path);

COMMENT ON TABLE plan_scope IS 'Scope declarations for implementation plans';
COMMENT ON COLUMN plan_scope.lsp_refs IS 'LSP reference data: [{symbol, locations, count}]';

-- 3. Critique bridge table (links steps to thought_nodes)
CREATE TABLE IF NOT EXISTS plan_step_critiques (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    step_id         UUID NOT NULL REFERENCES plan_steps(id) ON DELETE CASCADE,
    thought_node_id UUID NOT NULL REFERENCES thought_nodes(id) ON DELETE CASCADE,
    resolution_note TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(step_id, thought_node_id)
);

COMMENT ON TABLE plan_step_critiques IS 'Links plan steps to PAS critiques they address';

-- 4. Add checklist_data JSONB to artifacts
ALTER TABLE artifacts 
ADD COLUMN IF NOT EXISTS checklist_data JSONB DEFAULT '[]'::jsonb;

COMMENT ON COLUMN artifacts.checklist_data IS 
'Pre-submission checklist: [{item: string, checked: bool, note: string}]';

-- Note: plan_steps and plan_scope don't have project_id column,
-- so we don't add governance_audit_trigger to them.
-- If needed, create a separate simpler audit trigger.

