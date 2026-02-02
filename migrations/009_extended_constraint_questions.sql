-- Migration 009: Extended Constraint Discovery Questions
-- Phase 7c.5 Enhancement: Add 6 more questions for comprehensive onboarding
-- Total: 9 questions (3 existing + 6 new)

-- Question 4: Autonomy Level (CRITICAL - LLM Specialist)
INSERT INTO interview_questions (
    dimension_id, 
    question_template, 
    question_type, 
    choices, 
    constraint_mapping
)
SELECT 
    id,
    'When making changes, should I ask for permission first or just implement and show you the results?',
    'complex',
    '[
        {"label": "A", "text": "Always ask before making any code changes", "hidden_value": "ASK_FIRST"},
        {"label": "B", "text": "Just implement - I will review your work after", "hidden_value": "AUTONOMY_FOCUSED"},
        {"label": "C", "text": "Ask for significant changes, proceed on small fixes", "hidden_value": "BALANCED_AUTONOMY"}
    ]'::jsonb,
    '{
        "answer_mappings": {
            "A": [{"key": "autonomy_level", "value": "ask_always", "type": "philosophy"}],
            "B": [{"key": "autonomy_level", "value": "act_freely", "type": "philosophy"}],
            "C": [{"key": "autonomy_level", "value": "ask_significant", "type": "philosophy"}]
        }
    }'::jsonb
FROM interview_dimensions 
WHERE dimension_name = 'project_constraints'
LIMIT 1;

-- Question 5: Definition of Done (Agile Specialist)
INSERT INTO interview_questions (
    dimension_id, 
    question_template, 
    question_type, 
    choices, 
    constraint_mapping
)
SELECT 
    id,
    'What must be true before a task is considered "done"?',
    'complex',
    '[
        {"label": "A", "text": "Code works - that is enough", "hidden_value": "MINIMAL_DOD"},
        {"label": "B", "text": "Code works + tests pass", "hidden_value": "TESTED"},
        {"label": "C", "text": "Code works + tests + documented + reviewed", "hidden_value": "COMPLETE_DOD"}
    ]'::jsonb,
    '{
        "answer_mappings": {
            "A": [{"key": "definition_of_done", "value": "code_works", "type": "quality"}],
            "B": [{"key": "definition_of_done", "value": "code_and_tests", "type": "quality"}],
            "C": [{"key": "definition_of_done", "value": "full_dod", "type": "quality"}]
        }
    }'::jsonb
FROM interview_dimensions 
WHERE dimension_name = 'project_constraints'
LIMIT 1;

-- Question 6: Communication Style (PM)
INSERT INTO interview_questions (
    dimension_id, 
    question_template, 
    question_type, 
    choices, 
    constraint_mapping
)
SELECT 
    id,
    'How should I communicate with you about my work?',
    'complex',
    '[
        {"label": "A", "text": "Brief summaries - just the essentials", "hidden_value": "TERSE"},
        {"label": "B", "text": "Detailed explanations - show your reasoning", "hidden_value": "VERBOSE"},
        {"label": "C", "text": "Adaptive - brief for routine, detailed for complex", "hidden_value": "ADAPTIVE_COMM"}
    ]'::jsonb,
    '{
        "answer_mappings": {
            "A": [{"key": "communication_style", "value": "terse", "type": "philosophy"}],
            "B": [{"key": "communication_style", "value": "verbose", "type": "philosophy"}],
            "C": [{"key": "communication_style", "value": "adaptive", "type": "philosophy"}]
        }
    }'::jsonb
FROM interview_dimensions 
WHERE dimension_name = 'project_constraints'
LIMIT 1;

-- Question 7: Scope Change Handling (PM)
INSERT INTO interview_questions (
    dimension_id, 
    question_template, 
    question_type, 
    choices, 
    constraint_mapping
)
SELECT 
    id,
    'When I discover related issues while working, how should I handle them?',
    'complex',
    '[
        {"label": "A", "text": "Stay focused - only fix what was asked", "hidden_value": "STRICT_SCOPE"},
        {"label": "B", "text": "Fix small adjacent issues, flag larger ones", "hidden_value": "PRAGMATIC_SCOPE"},
        {"label": "C", "text": "Be proactive - fix everything you find", "hidden_value": "PROACTIVE"}
    ]'::jsonb,
    '{
        "answer_mappings": {
            "A": [{"key": "scope_handling", "value": "strict", "type": "philosophy"}],
            "B": [{"key": "scope_handling", "value": "pragmatic", "type": "philosophy"}],
            "C": [{"key": "scope_handling", "value": "proactive", "type": "philosophy"}]
        }
    }'::jsonb
FROM interview_dimensions 
WHERE dimension_name = 'project_constraints'
LIMIT 1;

-- Question 8: Explanation Depth (LLM Specialist)
INSERT INTO interview_questions (
    dimension_id, 
    question_template, 
    question_type, 
    choices, 
    constraint_mapping
)
SELECT 
    id,
    'How much reasoning should I show for my decisions?',
    'complex',
    '[
        {"label": "A", "text": "Just give me the answer/solution", "hidden_value": "RESULT_ONLY"},
        {"label": "B", "text": "Show the why - I want to understand your logic", "hidden_value": "SHOW_REASONING"},
        {"label": "C", "text": "Deep analysis - alternatives considered, tradeoffs explained", "hidden_value": "DEEP_ANALYSIS"}
    ]'::jsonb,
    '{
        "answer_mappings": {
            "A": [{"key": "explanation_depth", "value": "minimal", "type": "philosophy"}],
            "B": [{"key": "explanation_depth", "value": "standard", "type": "philosophy"}],
            "C": [{"key": "explanation_depth", "value": "comprehensive", "type": "philosophy"}]
        }
    }'::jsonb
FROM interview_dimensions 
WHERE dimension_name = 'project_constraints'
LIMIT 1;

-- Question 9: Code Review Requirement (Code Review Lead)
INSERT INTO interview_questions (
    dimension_id, 
    question_template, 
    question_type, 
    choices, 
    constraint_mapping
)
SELECT 
    id,
    'Should I treat my own code changes as if they need review?',
    'complex',
    '[
        {"label": "A", "text": "Yes - self-review before every commit, explain changes", "hidden_value": "SELF_REVIEW"},
        {"label": "B", "text": "Only for significant changes", "hidden_value": "REVIEW_SIGNIFICANT"},
        {"label": "C", "text": "No - just make the changes, you will review", "hidden_value": "NO_SELF_REVIEW"}
    ]'::jsonb,
    '{
        "answer_mappings": {
            "A": [{"key": "code_review_required", "value": "always", "type": "quality"}],
            "B": [{"key": "code_review_required", "value": "significant", "type": "quality"}],
            "C": [{"key": "code_review_required", "value": "none", "type": "quality"}]
        }
    }'::jsonb
FROM interview_dimensions 
WHERE dimension_name = 'project_constraints'
LIMIT 1;
