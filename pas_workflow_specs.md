# PAS (Probabilistic Abductive Scaffolding) - Complete Workflow Specification

## Purpose & Vision

PAS is a **Scientific Reasoning MCP** that improves LLM decision quality through orchestration, not by reasoning itself. The core principle is:

> **PAS doesn't reason - it creates conditions for better LLM reasoning.**

Think of PAS as a "cognitive scaffolding" system that guides an LLM to make better decisions by:
1. Providing structured friction that prevents premature convergence
2. Grounding hypotheses in scientific laws (Bayesian priors)
3. Forcing explicit critique of top hypotheses
4. Nudging the LLM toward deeper exploration when confidence is low

**Benchmark Performance**: 81.0 composite score (+43.6% improvement from baseline)

---

## Core Mechanisms

### 1. Bayesian Scoring
Every hypothesis is scored using Bayes' theorem:
```
posterior = prior × likelihood
```
- **Prior**: Semantic similarity between hypothesis and relevant scientific laws (e.g., Hyrum's Law, Postel's Law)
- **Likelihood**: LLM's stated confidence (0.0-1.0)
- **Posterior**: Combined score that ranks hypotheses

### 2. Tree Expansion (not linear chain)
Unlike linear chain-of-thought, PAS builds a tree:
```
root (goal)
├── h1 (hypothesis 1, posterior=0.82)
│   ├── h1.1 (deeper refinement, posterior=0.91)
│   └── h1.2 (alternative, posterior=0.75)
├── h2 (hypothesis 2, posterior=0.78)
└── h3 (hypothesis 3, posterior=0.65)
```

### 3. Next-Step Nudges
Every tool response includes a `next_step` suggestion guiding the LLM to:
- Critique the top hypothesis
- Expand deeper on uncertain hypotheses
- Close the learning loop with outcome recording

### 4. Self-Learning Loop (v8)
- `record_outcome()` captures success/failure
- `refresh_law_weights()` updates law priors based on outcomes
- Laws that correlate with success get higher weights over time

---

## Complete Tool Workflow

### Phase 1: Session Setup

```python
# Start a new reasoning session
start_reasoning_session(
    user_goal: str  # High-level goal or question
) -> {
    session_id: str,
    goal: str,
    state: "active"
}
```

### Phase 2: Interview (Optional Context Gathering)

```python
# Identify what clarifying questions to ask
identify_gaps(session_id) -> {
    questions_generated: int,
    message: "Call get_next_question to start"
}

# Get one question at a time
get_next_question(session_id) -> {
    question: {
        id: str,
        text: str,
        choices: [{ label: "A", description: str, pros: [], cons: [] }]
    }
}

# Submit answer
submit_answer(session_id, question_id, answer) -> {
    questions_remaining: int,
    follow_ups_injected: int | null
}

# Check if interview is complete
check_interview_complete(session_id) -> {
    is_complete: bool,
    context_summary: dict
}
```

### Phase 3: Hypothesis Generation (Expansion)

```python
# Prepare context for generating hypotheses
prepare_expansion(
    session_id: str,
    parent_node_id: str | None  # None for root-level
) -> {
    parent_content: str,
    goal: str,
    relevant_laws: [  # Semantic search results
        { law_name: str, definition: str, weight: float, similarity: float }
    ],
    instructions: "Generate 3 hypotheses with confidence..."
}

# Store generated hypotheses with Bayesian scoring
store_expansion(
    session_id: str,
    parent_node_id: str | None,
    # Hypothesis 1 (required)
    h1_text: str,
    h1_confidence: float,  # 0.0-1.0
    h1_scope: str,  # "[layer] file1.py, file2.py"
    # Hypothesis 2-3 (optional)
    h2_text: str, h2_confidence: float, h2_scope: str,
    h3_text: str, h3_confidence: float, h3_scope: str,
    # v7b: Revision tracking
    is_revision: bool = False,
    revises_node_id: str = None
) -> {
    created_nodes: [{
        node_id: str,
        path: str,  # "root.h1" or "root.h2.h1"
        content: str,
        prior_score: float,
        likelihood: float,
        posterior_score: float,
        supporting_law: str,
        declared_scope: str
    }],
    next_step: "Challenge your top hypothesis. Call prepare_critique(node_id='...')",
    confidence_nudge: str | null,  # v7a: "Low confidence detected..."
    revision_info: { is_revision: bool, revises_node_id: str } | null,
    revision_nudge: str | null
}
```

### Phase 4: Critique (Challenge Hypotheses)

```python
# Prepare context for critiquing a hypothesis
prepare_critique(node_id: str) -> {
    node_content: str,
    session_goal: str,
    current_scores: { prior, likelihood, posterior },
    supporting_laws: [{ law_name, definition }],
    instructions: "Challenge this hypothesis..."
}

# Store critique results (updates likelihood)
store_critique(
    node_id: str,
    counterargument: str,
    severity_score: float,  # 0.0-1.0 (higher = more severe)
    logical_flaws: str,  # Comma-separated
    edge_cases: str  # Comma-separated
) -> {
    score_update: {
        old_likelihood: float,
        new_likelihood: float,  # Reduced by severity
        posterior_score: float
    }
}
```

### Phase 5: Finalization (Decision)

```python
# Get final recommendation with auto-critique
finalize_session(
    session_id: str,
    top_n: int = 3,
    deep_critique: bool = False
) -> {
    recommendation: {
        node_id: str,
        content: str,
        original_score: float,
        adjusted_score: float,  # After penalties
        penalties_applied: [str]  # "unchallenged_penalty", "depth_bonus_+0.02"
    },
    runner_up: { node_id, content, adjusted_score } | null,
    decision_quality: "high" | "medium" | "low",  # Based on gap
    gap_analysis: str,  # "Clear winner (gap: 0.103)..."
    context_summary: dict | null,
    next_step: str | null,  # If decision_quality is low
    outcome_prompt: str,  # v8a: "Call record_outcome..."
    implementation_checklist: [str]  # v8c: ["[ ] Modify: server.py", ...]
}
```

### Phase 6: Learning (Post-Implementation)

```python
# Record outcome to close learning loop
record_outcome(
    session_id: str,
    outcome: "success" | "partial" | "failure",
    confidence: float = 1.0,
    notes: str = None
) -> {
    outcome_id: str,
    winning_path: str,
    attributed_nodes: int,
    laws_affected: [str],
    auto_refresh_triggered: bool,  # v8b: True every 5th outcome
    auto_refresh_result: { laws_updated: int } | null
}

# Manually refresh law weights (or auto-triggered)
refresh_law_weights(
    min_samples: int = 5,
    blend_factor: float = 0.5
) -> {
    laws_updated: int,
    updates: [{ law_name, old_weight, new_weight, success_rate }]
}
```

---

## Typical Workflow Example

```python
# 1. Start session
session = start_reasoning_session(
    user_goal="Add authentication to the API"
)

# 2. (Optional) Interview for context
identify_gaps(session.session_id)
q = get_next_question(session.session_id)
submit_answer(session.session_id, q.question.id, "B")

# 3. Generate hypotheses
ctx = prepare_expansion(session.session_id, parent_node_id=None)
# LLM reads ctx.relevant_laws and generates hypotheses

nodes = store_expansion(
    session_id=session.session_id,
    parent_node_id=None,
    h1_text="JWT-based auth with middleware",
    h1_confidence=0.85,
    h1_scope="[middleware] auth.py, [API] routes.py, [config] auth.yaml",
    h2_text="Session-based auth with Redis",
    h2_confidence=0.75,
    h2_scope="[middleware] session.py, [infra] redis.yaml",
    h3_text="OAuth2 integration with third-party",
    h3_confidence=0.6,
    h3_scope="[integration] oauth.py, [config] providers.yaml"
)
# nodes.next_step → "Challenge your top hypothesis..."

# 4. Critique top hypothesis
crit_ctx = prepare_critique(nodes.created_nodes[0].node_id)
store_critique(
    node_id=nodes.created_nodes[0].node_id,
    counterargument="JWT tokens cannot be revoked without additional infrastructure",
    severity_score=0.3,
    logical_flaws="Assumes stateless is always better",
    edge_cases="Token revocation, refresh token rotation"
)

# 5. Finalize
result = finalize_session(session.session_id)
# result.recommendation → Best hypothesis after scoring
# result.implementation_checklist → ["[ ] Modify: auth.py", ...]
# result.outcome_prompt → "Call record_outcome..."

# 6. After implementation
record_outcome(
    session_id=session.session_id,
    outcome="success",
    notes="Auth implemented successfully with JWT"
)
# Laws used in winning path get positive attribution
```

---

## Key Features (v7-v12)

### v7-v8: Foundation
| Version | Feature | Impact |
|---------|---------|--------|
| v7a | `confidence_nudge` | Infers exploration need from low confidence |
| v7b | `is_revision` + `revises_node_id` | Explicit backtracking (best: +22% score) |
| v8a | `outcome_prompt` | Guides self-learning loop |
| v8b | Auto-refresh law weights | Makes self-learning automatic |
| v8c | `implementation_checklist` | Bridges reasoning → action |

### v9-v10: Active Critique (System 2)
| Version | Feature | Impact |
|---------|---------|--------|
| v9a | Multi-Tier Critique (major/minor flaws) | Major: -0.15, Minor: -0.03 penalty |
| v9b | Constitutional Principles | 5 code + 5 UI/UX principles for critique |
| v9b.1 | UI/UX Principles | Hick's Law, WCAG, Visual Hierarchy |
| v10a | Negation Detection | Penalizes hypothesis-law contradictions |
| v10b | Critic Personas (3) | Strict Skeptic, Pragmatic Engineer, Domain Expert |

### v11-v12: MCTS-Inspired Self-Learning
| Version | Feature | Impact |
|---------|---------|--------|
| v11a | UCT Tiebreaking | Breaks close decisions (gap < 0.05) with exploration bonus |
| v11b | Law-Grounded Rollout | 20% rollout weight from scientific_weight |
| v12a | PRM Data Collection | Logs hypothesis+outcome for future fine-tuning |
| v12b | Thompson Sampling | Beta-distributed law selection (explore-exploit) |

---

## Benchmark Evolution

| Variant | Score | Key Change |
|---------|-------|------------|
| v1_baseline | 56.4 | Initial |
| v7b_is_revision | 79.4 | +40.8% from backtracking |
| **v9ab_tiered_ux** | **81.0** | Current best (tiered critique + UX) |
| v11ab_mcts | 78.8 | UCT + rollout |
| v12ab_thompson | 77.7 | Exploration variance (early stage) |

---

## Penalties Applied in Finalization

| Penalty | Trigger | Effect |
|---------|---------|--------|
| `unchallenged_penalty` | Hypothesis never critiqued | -0.10 |
| `shallow_alternatives_penalty` | <2 siblings at same level | -0.05 |
| `depth_bonus_+X` | Depth > 2 | +0.02 per level |

---

## Scientific Laws Used (Bayesian Priors)

PAS uses semantic similarity to match hypotheses with relevant laws:

- **Hyrum's Law**: API changes break someone's workflow
- **Postel's Law**: Be liberal in input, strict in output
- **Gall's Law**: Start simple, add complexity incrementally
- **Goodhart's Law**: Optimizing metrics leads to gaming
- **Occam's Razor**: Prefer simpler solutions
- **Brooks' Law**: Adding people to late projects makes them later
- **Conway's Law**: Systems mirror communication structures
- **Pareto Principle**: 80% of effects from 20% of causes
- ...and more (15+ laws in database)

---

## Research Questions for Improvement

### Active Research (v9-v12 enabled)
1. **Thompson Sampling Convergence**: How many outcomes before Beta distributions stabilize?
2. **PRM Training Threshold**: When is training_data sufficient for fine-tuning (~500+ samples)?
3. **UCT Trigger Rate**: Is 0.05 threshold too conservative? (Currently never triggers)
4. **Critique Persona Correlation**: Which persona finds most actionable flaws?

### Open Questions
5. **Prior Quality**: How can we improve semantic matching between hypotheses and laws?
6. **Exploration vs. Exploitation**: When should PAS encourage more alternatives vs. deeper refinement?
7. **Multi-Domain Transfer**: Do law weights learned in one domain transfer to others?
8. **Goodhart Risk**: Are we optimizing the benchmark or genuine reasoning quality?

---

## Goal Declaration Format

When using PAS, declare goals in this format:

```
[CONTEXT] Brief background on the problem
[GOAL] What you want to achieve
[CONSTRAINTS] Any limitations or requirements
[SCOPE] Expected affected files/modules (optional)
```

Example:
```
[CONTEXT] Our API currently accepts a dict parameter for complex options
[GOAL] Refactor to use flattened primitive parameters for LLM compatibility
[CONSTRAINTS] Must maintain backward compatibility for existing callers
[SCOPE] server.py, api.py, tests/
```

---

## Summary

PAS is an **orchestration layer** that makes LLMs produce better decisions by:

1. **Forcing alternatives** (3 hypotheses minimum)
2. **Grounding in science** (Bayesian priors from laws)
3. **Requiring critique** (challenge top hypothesis)
4. **Nudging exploration** (confidence_nudge, revision tracking)
5. **Closing the loop** (outcome recording, weight updates)

The key insight is that PAS doesn't need to be smart - it just needs to create the right conditions for the LLM to be smarter.
