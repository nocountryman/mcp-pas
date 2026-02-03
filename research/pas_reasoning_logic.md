# PAS Reasoning Logic Report

> **Purpose-Aware Scaffolding (PAS)** is a scientific reasoning system that applies Bayesian hypothesis testing and psychological laws to software development decisions.

---

## Executive Summary

PAS enforces a structured decision-making process: **Hypothesize → Critique → Execute → Learn**. Every non-trivial change must pass through this workflow, ensuring decisions are grounded in evidence and past learning is incorporated into future reasoning.

---

## Core Architecture

### The Reasoning Tree

PAS builds a **Tree of Thought (ToT)** where each node represents a hypothesis. Nodes are scored using Bayesian posteriors, and the system tracks:

- **Prior probability** (initial confidence)
- **Likelihood** (evidence from critiques)
- **Posterior** (updated score after critique)

```mermaid
graph TD
    subgraph "Reasoning Tree Structure"
        G[🎯 Goal/Session] --> H1[Hypothesis 1<br/>score: 0.85]
        G --> H2[Hypothesis 2<br/>score: 0.72]
        G --> H3[Hypothesis 3<br/>score: 0.68]
        
        H1 --> H1a[Refinement 1.1<br/>score: 0.91]
        H1 --> H1b[Refinement 1.2<br/>score: 0.78]
        
        H2 --> H2a[Refinement 2.1<br/>score: 0.65]
    end
    
    style G fill:#4CAF50,color:#fff
    style H1a fill:#2196F3,color:#fff
```

---

## Complete PAS Workflow

### Phase 1: Session Initialization

```mermaid
flowchart LR
    A[User Request] --> B{Is it trivial?}
    B -->|< 10 lines, single file| C[Direct Implementation]
    B -->|Non-trivial| D[start_reasoning_session]
    D --> E[Session Created<br/>goal stored]
```

### Phase 2: Hypothesis Generation

```mermaid
flowchart TD
    A[prepare_expansion] --> B{suggested_lookups?}
    B -->|Yes| C[find_references<br/>for each symbol]
    B -->|No| D[Generate Hypotheses]
    C --> D
    
    D --> E[store_expansion]
    E --> F[3 Hypotheses Created<br/>with scope declarations]
    
    subgraph "Hypothesis Structure"
        H1[h1_text + h1_confidence + h1_scope]
        H2[h2_text + h2_confidence + h2_scope]
        H3[h3_text + h3_confidence + h3_scope]
    end
```

### Phase 3: Critical Analysis

```mermaid
flowchart TD
    A[prepare_critique<br/>node_id] --> B[Get critique context<br/>+ relevant laws]
    B --> C[Generate counterarguments]
    C --> D[store_critique]
    
    D --> E{Severity Score}
    E -->|< 0.3| F[Minor issues<br/>proceed]
    E -->|0.3-0.7| G[Consider alternatives]
    E -->|> 0.7| H[Major flaws<br/>revise hypothesis]
    
    subgraph "Critique Output"
        CR[counterargument<br/>severity_score<br/>major_flaws<br/>minor_flaws]
    end
```

### Phase 4: Gap Analysis (Mandatory)

```mermaid
flowchart TD
    A[prepare_sequential_analysis] --> B[5-Layer Gap Check]
    
    subgraph "Gap Layers"
        L1[1. CODE STRUCTURE<br/>What code changes?]
        L2[2. DEPENDENCIES<br/>What packages assumed?]
        L3[3. DATA FLOW<br/>What data moves where?]
        L4[4. INTERFACES<br/>What APIs affected?]
        L5[5. WORKFLOWS<br/>What flows change?]
    end
    
    B --> L1 --> L2 --> L3 --> L4 --> L5
    L5 --> C[store_sequential_analysis]
    C --> D[Gaps Recorded]
```

### Phase 5: Quality Gate & Finalization

```mermaid
flowchart TD
    A[finalize_session] --> B{Quality Gate Check}
    
    B --> C{Score ≥ 0.9?}
    C -->|No| D[❌ HARD BLOCK<br/>Deepen hypotheses]
    C -->|Yes| E{Gap ≥ 0.08?}
    
    E -->|No| F[⚠️ Low diversity<br/>Explore alternatives]
    E -->|Yes| G[✅ PASSED<br/>Proceed to implementation]
    
    D --> H[prepare_expansion<br/>deeper analysis]
    F --> H
    
    G --> I[Create Implementation Plan]
    
    style D fill:#f44336,color:#fff
    style F fill:#ff9800,color:#fff
    style G fill:#4CAF50,color:#fff
```

---

## Complete End-to-End Flow

```mermaid
flowchart TD
    subgraph "1. INITIALIZATION"
        A[User Request] --> B[start_reasoning_session]
    end
    
    subgraph "2. HYPOTHESIS GENERATION"
        B --> C[prepare_expansion]
        C --> D{past_failure_warnings?}
        D -->|Yes| E[log_conversation<br/>ACKNOWLEDGE]
        D -->|No| F[Generate 3 hypotheses]
        E --> F
        F --> G[store_expansion]
    end
    
    subgraph "3. CRITICAL ANALYSIS"
        G --> H[prepare_critique<br/>top hypothesis]
        H --> I[store_critique]
    end
    
    subgraph "4. GAP ANALYSIS"
        I --> J[prepare_sequential_analysis]
        J --> K[store_sequential_analysis]
    end
    
    subgraph "5. QUALITY GATE"
        K --> L[finalize_session]
        L --> M{score ≥ 0.9<br/>gap ≥ 0.08?}
        M -->|No| N[Deepen or Expand]
        N --> C
        M -->|Yes| O[✅ APPROVED]
    end
    
    subgraph "6. EXECUTION"
        O --> P[Create Implementation Plan]
        P --> Q[Execute Changes]
        Q --> R[Verify Results]
    end
    
    subgraph "7. LEARNING"
        R --> S[record_outcome]
        S --> T[PAS learns from<br/>success/failure]
    end
    
    style O fill:#4CAF50,color:#fff
    style N fill:#ff9800,color:#fff
```

---

## Synthesis Flow (When Hypotheses Are Complementary)

```mermaid
flowchart TD
    A[finalize_session] --> B{complementarity_detected?}
    B -->|Yes| C[synthesize_hypotheses<br/>combine node_ids]
    C --> D[Hybrid Node Created]
    D --> E[prepare_critique<br/>hybrid_node_id]
    E --> F[store_critique]
    F --> G[prepare_sequential_analysis]
    G --> H[store_sequential_analysis]
    H --> I[finalize_session<br/>again]
    I --> J[record_outcome]
    
    B -->|No| K[Single winner<br/>proceed normally]
    
    style D fill:#9C27B0,color:#fff
```

---

## Tool Call Sequence (Quick Reference)

| Step | Tool | Purpose |
|------|------|---------|
| 1 | `start_reasoning_session` | Create session with goal |
| 2 | `prepare_expansion` | Get context + relevant laws |
| 3 | `find_references` | (if suggested_lookups) Scope analysis |
| 4 | `store_expansion` | Save 3 hypotheses with scores |
| 5 | `prepare_critique` | Get critique context |
| 6 | `store_critique` | Save counterarguments + severity |
| 7 | `prepare_sequential_analysis` | Get gap analysis prompts |
| 8 | `store_sequential_analysis` | Save 5-layer gap results |
| 9 | `finalize_session` | Quality gate check |
| 10 | `record_outcome` | Log success/failure for learning |

---

## Quality Gates

### Hard Blocks (Cannot Proceed)

| Check | Threshold | Action if Failed |
|-------|-----------|------------------|
| Score | ≥ 0.9 | Expand deeper, add hypotheses |
| Gap | ≥ 0.08 | Explore diverse alternatives |
| Synthesis | Must critique hybrid | Cannot skip |

### Soft Warnings

| Warning | Source | Required Action |
|---------|--------|-----------------|
| `past_failure_warnings` | Previous failures | Acknowledge in `log_conversation` |
| `preflight_warnings` | Missing checks | Complete required lookups |
| `missing_codebase_research` | No query_codebase | Research before hypothesizing |

---

## Learning Loop

```mermaid
flowchart LR
    subgraph "Outcome Recording"
        A[record_outcome] --> B{outcome}
        B -->|success| C[Attribute to<br/>winning path nodes]
        B -->|failure| D[Log failure_reason<br/>for semantic matching]
        B -->|partial| E[Mixed attribution]
    end
    
    subgraph "Weight Updates"
        C --> F[refresh_law_weights]
        D --> F
        E --> F
        F --> G[Laws that correlate<br/>with success get boosted]
    end
    
    subgraph "Future Sessions"
        G --> H[prepare_expansion<br/>surfaces relevant laws]
        H --> I[past_failure_warnings<br/>from semantic match]
    end
```

---

## Key Principles

1. **Think First, Implement Later** - No code without reasoning
2. **Critique Before Commit** - Every hypothesis must be challenged
3. **Learn From Outcomes** - Success/failure updates law weights
4. **Quality Over Speed** - Hard blocks prevent low-quality decisions
5. **Scope Awareness** - Declare affected files/modules upfront

---

*Generated: 2026-02-02 | PAS v94+*
