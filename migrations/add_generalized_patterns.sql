-- Cross-Project Learning Enhancement
-- Add generalized pattern extraction to outcome_records

-- Add columns for generalized patterns
ALTER TABLE outcome_records 
ADD COLUMN IF NOT EXISTS generalized_pattern TEXT,
ADD COLUMN IF NOT EXISTS pattern_type TEXT;

-- Index for pattern type filtering
CREATE INDEX IF NOT EXISTS idx_outcome_pattern_type ON outcome_records(pattern_type);

-- Add embedding for generalized pattern (for cross-project semantic search)
ALTER TABLE outcome_records 
ADD COLUMN IF NOT EXISTS generalized_embedding vector(1024);

CREATE INDEX IF NOT EXISTS idx_outcome_generalized_embedding 
ON outcome_records USING ivfflat (generalized_embedding vector_cosine_ops)
WITH (lists = 10);

COMMENT ON COLUMN outcome_records.generalized_pattern IS 'Project-agnostic pattern description for cross-project learning';
COMMENT ON COLUMN outcome_records.pattern_type IS 'Category: api_timeout, import_error, schema_mismatch, dict_access, etc';
