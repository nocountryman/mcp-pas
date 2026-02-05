-- SCIP Code Graph Schema
-- Migration for per-module code indexing

-- Module tracking table
CREATE TABLE IF NOT EXISTS scip_modules (
    project_id TEXT NOT NULL,
    module_id TEXT NOT NULL,
    module_path TEXT,
    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    file_count INTEGER DEFAULT 0,
    PRIMARY KEY (project_id, module_id)
);

-- Symbol occurrences (code graph data)
CREATE TABLE IF NOT EXISTS scip_occurrences (
    id SERIAL PRIMARY KEY,
    project_id TEXT NOT NULL,
    module_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_start INTEGER,
    col_start INTEGER,
    line_end INTEGER,
    col_end INTEGER,
    symbol TEXT NOT NULL,
    symbol_role INTEGER,  -- 1=definition, 8=reference, 16=import, etc.
    enclosing_symbol TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_scip_occurrences_symbol 
    ON scip_occurrences(symbol);
CREATE INDEX IF NOT EXISTS idx_scip_occurrences_module 
    ON scip_occurrences(project_id, module_id);
CREATE INDEX IF NOT EXISTS idx_scip_occurrences_file 
    ON scip_occurrences(file_path);
CREATE INDEX IF NOT EXISTS idx_scip_occurrences_role 
    ON scip_occurrences(symbol_role);
CREATE INDEX IF NOT EXISTS idx_scip_occurrences_enclosing 
    ON scip_occurrences(enclosing_symbol);

-- Cascade delete for module reindex
ALTER TABLE scip_occurrences 
    ADD CONSTRAINT fk_scip_module 
    FOREIGN KEY (project_id, module_id) 
    REFERENCES scip_modules(project_id, module_id) 
    ON DELETE CASCADE;

-- Comments
COMMENT ON TABLE scip_modules IS 'Tracks indexed modules with last index time';
COMMENT ON TABLE scip_occurrences IS 'SCIP symbol occurrences for code graph queries';
COMMENT ON COLUMN scip_occurrences.symbol_role IS 'SCIP role: 1=Definition, 8=Reference, 16=Import, 32=WriteAccess';
