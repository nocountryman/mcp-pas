-- Migration 008: Constraint Discovery Interview
-- Phase 7c.5: Enable interview-driven constraint discovery

-- Add constraint_mapping column to interview_questions table
-- This allows questions to map answers to structured project_constraints
ALTER TABLE interview_questions 
ADD COLUMN IF NOT EXISTS constraint_mapping JSONB DEFAULT NULL;

COMMENT ON COLUMN interview_questions.constraint_mapping IS 
'Maps answer choices to project_constraint keys. Format: {"answer_mappings": {"A": [{"key": "...", "value": ..., "type": "philosophy"}]}}';

-- Add constraint_setup domain for interview system
INSERT INTO interview_domains (domain_name, description, example_goals)
VALUES (
    'constraint_setup',
    'Interview domain for discovering project constraints (philosophy, environment, quality)',
    ARRAY['Define project constraints', 'Set up coding standards', 'Configure environment requirements']
) ON CONFLICT (domain_name) DO NOTHING;

-- Create a dimension for constraint discovery
INSERT INTO interview_dimensions (domain_id, dimension_name, description, priority)
SELECT 
    id,
    'project_constraints',
    'Dimension for discovering and configuring project-level constraints',
    1
FROM interview_domains 
WHERE domain_name = 'constraint_setup'
ON CONFLICT DO NOTHING;

-- Add sample constraint discovery questions
-- Question 1: MVP Philosophy
INSERT INTO interview_questions (
    dimension_id, 
    question_template, 
    question_type, 
    choices, 
    constraint_mapping
)
SELECT 
    id,
    'When building features, what is your philosophy on minimum viable products (MVPs)?',
    'complex',
    '[
        {"label": "A", "text": "Ship fast with minimal features, iterate based on feedback", "hidden_value": "SPEED_FOCUSED"},
        {"label": "B", "text": "Build complete, polished features from the start", "hidden_value": "QUALITY_FOCUSED"},
        {"label": "C", "text": "Balance - core functionality polished, extras can wait", "hidden_value": "PRAGMATIST"}
    ]'::jsonb,
    '{
        "answer_mappings": {
            "A": [{"key": "allow_mvp", "value": true, "type": "philosophy"}],
            "B": [{"key": "no_mvp", "value": true, "type": "philosophy"}],
            "C": [{"key": "balanced_quality", "value": true, "type": "philosophy"}]
        }
    }'::jsonb
FROM interview_dimensions 
WHERE dimension_name = 'project_constraints'
LIMIT 1;

-- Question 2: Quality Gate Strictness
INSERT INTO interview_questions (
    dimension_id, 
    question_template, 
    question_type, 
    choices, 
    constraint_mapping
)
SELECT 
    id,
    'How important is it that the AI agent follows strict quality gates before implementing changes?',
    'complex',
    '[
        {"label": "A", "text": "Critical - block changes that don''t meet quality thresholds", "hidden_value": "STRICT_SAFETY"},
        {"label": "B", "text": "Important - warn but allow override when needed", "hidden_value": "BALANCED"},
        {"label": "C", "text": "Flexible - trust the agent to make judgment calls", "hidden_value": "AUTONOMY_FOCUSED"}
    ]'::jsonb,
    '{
        "answer_mappings": {
            "A": [{"key": "quality_gate_required", "value": true, "type": "philosophy", "enforcement": "block"}],
            "B": [{"key": "quality_gate_required", "value": true, "type": "philosophy", "enforcement": "warn"}],
            "C": [{"key": "quality_gate_required", "value": false, "type": "philosophy"}]
        }
    }'::jsonb
FROM interview_dimensions 
WHERE dimension_name = 'project_constraints'
LIMIT 1;

-- Question 3: Environment Setup
INSERT INTO interview_questions (
    dimension_id, 
    question_template, 
    question_type, 
    choices, 
    constraint_mapping
)
SELECT 
    id,
    'What is your preferred terminal environment setup?',
    'complex',
    '[
        {"label": "A", "text": "Always activate venv and load .env before commands", "hidden_value": "EXPLICIT_SAFETY"},
        {"label": "B", "text": "Use absolute paths to venv binaries, skip activation", "hidden_value": "DIRECT"},
        {"label": "C", "text": "Rely on system defaults, no special setup needed", "hidden_value": "MINIMAL"}
    ]'::jsonb,
    '{
        "answer_mappings": {
            "A": [{"key": "terminal_env_activation", "value": "source .venv/bin/activate && set -a && source .env && set +a", "type": "environment", "enforcement": "block"}],
            "B": [{"key": "use_absolute_venv_paths", "value": true, "type": "environment"}],
            "C": [{"key": "terminal_env_activation", "value": null, "type": "environment"}]
        }
    }'::jsonb
FROM interview_dimensions 
WHERE dimension_name = 'project_constraints'
LIMIT 1;
