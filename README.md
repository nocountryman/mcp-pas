# 🏛️ PAS - Scientific Reasoning MCP Server

**P**lato • **A**ristotle • **S**ocrates

A Model Context Protocol (MCP) server that brings structured, Bayesian reasoning to AI agents. PAS implements a Tree of Thoughts architecture with scientific law grounding, enabling AI to reason more deliberately through complex problems.

[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)](https://www.postgresql.org/)
[![Version](https://img.shields.io/badge/version-v42-orange.svg)](CHANGELOG.md)

---

## 🎯 What is PAS?

PAS transforms how AI agents approach complex reasoning by:

- **Expanding** hypotheses systematically (like Plato's ideation)
- **Critiquing** each branch with counterarguments (like Socrates' dialectic)
- **Grounding** reasoning in scientific laws (like Aristotle's empiricism)

Instead of producing a single answer, PAS builds a **reasoning tree** where each hypothesis is scored using Bayesian inference and challenged through adversarial critique.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Bayesian Tree of Thoughts** | Build reasoning trees with posterior probability scoring |
| **Scientific Law Grounding** | 15+ laws (CAP Theorem, Occam's Razor, etc.) inform priors |
| **Quality Gate** | 0.9 score + 0.1 gap thresholds with improvement suggestions |
| **Negative Space Critique** | Find what's MISSING, not just what's wrong |
| **Smart Interview Pattern** | Psychology-based Q&A to gather missing context |
| **Domain Detection** | Auto-detect problem domain for tailored questions |
| **RLVR Self-Learning** | Auto-detect success/failure from terminal output |
| **Sequential Gap Analysis** | Layer-by-layer adversarial gap detection (v32/v37) |
| **Session Tagging** | Organize sessions with tags for retrieval |
| **Persistent Traits** | Learn user preferences across sessions (v22) |
| **Past Failures Surfacing** | Learn from similar past failures during critique |
| **Live Code Navigation** | Find references/definitions via Jedi (v38) |
| **Symbol Suggestions** | Auto-suggest symbols during hypothesis generation |
| **Preflight Enforcement** | Structural guardrails at hypothesis storage (v41) |
| **Self-Awareness** | PAS can introspect its own schema/tools/architecture (v40) |
| **Self-Aware Test Suite** | 4-layer tests that log failures back to PAS (v42) |

---

## 🧠 Sequential Thinking Integration

PAS is designed to work alongside `@anthropic/mcp-sequential-thinking`. While Sequential Thinking handles **linear** thought chains, PAS provides **tree-structured** reasoning with Bayesian scoring.

### Complementary Usage

| Aspect | Sequential Thinking | PAS |
|--------|---------------------|-----|
| Structure | Linear chain | Bayesian Tree |
| Scoring | None | Prior × Likelihood = Posterior |
| Self-Critique | None | Tiered penalties (major/minor flaws) |
| Quality Gate | None | 0.9 score, 0.1 gap |
| Self-Learning | None | Laws + failure memory + traits |
| Output | Final thought | Ranked recommendation + gaps |

### Recommended Pattern

```
1. Use Sequential Thinking for quick, linear reasoning
2. When uncertainty is high, escalate to PAS:
   → start_reasoning_session(user_goal="...")
   → Expand, critique, deepen until quality gate passes
3. Record outcome for RLVR learning
```

---

## 🛠️ Tools (25+ Total)

### Session Management
| Tool | Description |
|------|-------------|
| `start_reasoning_session` | Begin a new reasoning session with a goal |
| `get_session_status` | Check session state and thought count |
| `find_or_create_session` | Smart router - finds existing or creates new session |
| `complete_session` | Explicitly close a session |
| `resume_session` | Continue a completed session with inheritance |
| `tag_session` | Add tags for organization |

### Reasoning (Expand & Critique)
| Tool | Description |
|------|-------------|
| `prepare_expansion` | Get context + relevant laws for hypothesis generation |
| `store_expansion` | Store 1-3 hypotheses with Bayesian scoring |
| `prepare_critique` | Get node context + LLM prompt for counterarguments |
| `store_critique` | Store critique with tiered penalties (major/minor) |
| `search_relevant_laws` | Find scientific laws by semantic similarity |

### Sequential Analysis (v32/v37)
| Tool | Description |
|------|-------------|
| `prepare_sequential_analysis` | Get prompts for 5-layer gap analysis on top candidates |
| `store_sequential_analysis` | Store gap analysis, detect systemic gaps |

**5-Layer Gap Analysis** checks:
1. **CODE STRUCTURE** - What code changes are needed?
2. **DEPENDENCIES** - What packages/systems are assumed?
3. **DATA FLOW** - What data moves where?
4. **INTERFACES** - What APIs/contracts are affected?
5. **WORKFLOWS** - What user/system flows change?

### Tree Navigation
| Tool | Description |
|------|-------------|
| `get_reasoning_tree` | View full tree structure with scores |
| `get_best_path` | Find highest-scoring reasoning path |
| `finalize_session` | Auto-critique, quality gate, return recommendation |

### Smart Interview
| Tool | Description |
|------|-------------|
| `identify_gaps` | Generate clarifying questions based on goal |
| `get_next_question` | Get one question at a time with formatted choices |
| `submit_answer` | Answer question, trigger follow-up rules |
| `check_interview_complete` | Check if enough context gathered |

**Interview Features:**
- **Domain Detection** - Auto-detect UI, architecture, debugging, testing domains
- **Hidden Context** - Choices carry hidden metadata for trait inference
- **Follow-up Rules** - Conditional question injection based on answers
- **Progress Tracking** - Max 15 questions, 3 levels deep
- **History Archive** - Answered questions stored for self-learning

### Self-Learning (RLVR)
| Tool | Description |
|------|-------------|
| `record_outcome` | Record success/failure with semantic attribution |
| `parse_terminal_output` | Auto-detect success/failure signals from terminal |
| `refresh_law_weights` | Update law weights based on outcomes |
| `log_conversation` | Store user input for semantic search |
| `search_conversation_log` | Find past context by similarity |

**RLVR Auto-Recording:**
```json
{
  "finalize_session": {
    "terminal_output": "[terminal logs here]",
    "auto_record": true
  }
}
```
PAS automatically parses terminal output and records outcomes.

### Code Navigation (v38)
| Tool | Description |
|------|-------------|
| `sync_project` | Index a project for symbol search |
| `query_codebase` | Semantic search over indexed files |
| `find_references` | Find all references to a symbol (live Jedi) |
| `go_to_definition` | Jump to symbol definition |
| `call_hierarchy` | Build caller/callee tree |
| `import_lsif` | Import LSIF index (optional fallback) |

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 16+ with extensions:
  - `pgvector` (semantic similarity)
  - `ltree` (tree paths)

### Step 1: Clone & Setup

```bash
git clone https://github.com/nocountryman/mcp-pas.git
cd mcp-pas

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `mcp` | latest | Model Context Protocol server framework |
| `psycopg2-binary` | latest | PostgreSQL adapter |
| `pgvector` | latest | Vector similarity for embeddings |
| `pydantic` | latest | Data validation |
| `numpy` | latest | Numerical operations |
| `sentence-transformers` | latest | Local embeddings (all-MiniLM-L6-v2) |
| `tree-sitter-language-pack` | ≥0.13.0 | Symbol extraction for code indexing |
| `pyyaml` | ≥6.0 | Configuration file parsing |
| `jedi` | latest | **Live** code navigation (find references, go-to-definition) |

### Optional Tools

| Tool | Install | Purpose |
|------|---------|---------|
| `pyright` | `pip install pyright` | Generate LSIF indexes for **precision** code navigation |

**When to use LSIF vs Jedi:**
- **Jedi** (default): Live analysis, no pre-indexing, works out of the box
- **LSIF** (optional): Pre-computed index via `pyright --outputtype lsif`, faster for large codebases

```bash
# Generate LSIF index (optional)
pyright --outputtype lsif --outputfile project.lsif.json

# Import into PAS
# → import_lsif(project_id="...", lsif_path="/path/to/project.lsif.json")
```

### Step 2: Database Setup

```bash
# Create database
sudo -u postgres createdb mcp_pas

# Enable extensions
sudo -u postgres psql -d mcp_pas -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -d mcp_pas -c "CREATE EXTENSION IF NOT EXISTS ltree;"

# Run schema
psql -d mcp_pas -f schema.sql

# Seed scientific laws and domains
python seed_laws.py
python seed_domains_v2.py
```

### Step 3: Environment Configuration

```bash
cp env.template .env
# Edit .env with your database credentials
```

```env
DATABASE_URL=postgresql://user:password@localhost:5432/mcp_pas
```

> **Note:** PAS uses local embeddings via `sentence-transformers` by default. No OpenAI API key required.

### Step 4: GPU Acceleration (Optional)

PAS embedding model can run on GPU for **5-10x faster loading** (11s vs 2-5 min on CPU).

#### AMD GPUs (ROCm)

> **Note:** ROCm PyTorch wheels require Python 3.10-3.12. On rolling-release distros (Arch/CachyOS) with Python 3.14+, you'll need a separate Python 3.12 virtual environment.

**Step 1: Create Python 3.12 venv with ROCm PyTorch:**
```bash
# Create separate venv with Python 3.12
python3.12 -m venv .venv312
source .venv312/bin/activate

# Install ROCm PyTorch from official wheels
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# Install other dependencies
pip install -r requirements.txt
```

**Step 2: RDNA2 GPU workaround (RX 6000 series):**

RDNA2 GPUs (gfx1032, gfx1030) need an environment variable override:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

Add this to your MCP config:
```json
{
  "env": {
    "DATABASE_URL": "...",
    "HSA_OVERRIDE_GFX_VERSION": "10.3.0"
  }
}
```

**Verify GPU is detected:**
```bash
source .venv312/bin/activate
HSA_OVERRIDE_GFX_VERSION=10.3.0 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True AMD Radeon RX 6650 XT (or your GPU)
```

#### NVIDIA GPUs (CUDA)

On **Arch Linux / CachyOS**:
```bash
sudo pacman -S python-pytorch-cuda cuda cudnn
```

On **Ubuntu/Debian**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### Performance Comparison

| Metric | CPU | GPU (ROCm/CUDA) |
|--------|-----|-----------------|
| Model load | 2-5 min | 11-30 sec |
| Encode 100 texts | ~500ms | ~50ms |
| Memory used | System RAM | VRAM |

**No other changes needed** - once the venv uses ROCm/CUDA torch, the singleton in `utils.py` automatically loads the model to GPU.


---

## 🔌 IDE Integration

### VS Code / Cursor (with Claude/Gemini Extension)

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "pas-server": {
      "command": "/path/to/mcp-pas/.venv/bin/python",
      "args": ["/path/to/mcp-pas/server.py"],
      "env": {
        "DATABASE_URL": "postgresql://user:password@localhost:5432/mcp_pas",
        "HSA_OVERRIDE_GFX_VERSION": "10.3.0"
      }
    }
  }
}
```

> **Note:** Use `.venv312/bin/python` for GPU (ROCm) or `.venv/bin/python` for CPU-only.
> Remove `HSA_OVERRIDE_GFX_VERSION` if using NVIDIA or RDNA3+ GPUs.


**After configuration**: Restart your IDE (`Ctrl+Shift+P` → Reload Window).

---

## 🚀 Usage Walkthrough

### Basic Reasoning Flow

```
1. Start session
   → start_reasoning_session(user_goal="Design a caching layer")

2. Expand hypotheses (3 alternatives)
   → prepare_expansion(session_id="...")
   → store_expansion(session_id="...", h1_text="...", h1_confidence=0.85, ...)

3. Critique top hypothesis
   → prepare_critique(node_id="...")  # Returns llm_prompt
   → [Process prompt]
   → store_critique(node_id="...", counterargument="...", severity_score=0.3)

4. Sequential gap analysis (REQUIRED before finalize)
   → prepare_sequential_analysis(session_id="...", top_n=3)
   → [Process 5-layer prompts]
   → store_sequential_analysis(session_id="...", results="[...]")

5. Deepen if score < 0.9 or gap < 0.1
   → store_expansion(parent_node_id="...", h1_text="Improved approach...", h1_confidence=0.95)

6. Get recommendation
   → finalize_session(session_id="...")
   # Returns: quality_gate, recommendation, exhaustive_check
```

### Smart Interview Flow

```
1. Identify missing context
   → identify_gaps(session_id="...")

2. Ask questions one at a time
   → get_next_question(session_id="...")
   → [Present to user]
   → submit_answer(session_id="...", question_id="...", answer="B")

3. Check completion
   → check_interview_complete(session_id="...")
   # Once complete, hidden context is propagated to session
```

### Quality Gate (v31)

PAS won't accept low-quality decisions:

```json
{
  "quality_gate": {
    "score": 0.95,
    "score_threshold": 0.9,
    "score_ok": true,
    "gap": 0.12,
    "gap_threshold": 0.1,
    "gap_ok": true,
    "passed": true
  }
}
```

If `passed: false`, PAS suggests how to improve:
```json
{
  "score_improvement_suggestions": [
    {"lever": "score", "action": "Expand deeper with higher confidence (0.9+)"},
    {"lever": "gap", "action": "Explore more diverse alternatives"}
  ]
}
```

---

## 📊 Architecture

### Module Structure (v42)

```
server.py                  Main MCP server (~5400 lines)
├── errors.py              Exception hierarchy (~200 lines)
├── utils.py               DB, embeddings, validation (~185 lines)
├── reasoning_helpers.py   Bayesian scoring, quality (~310 lines)
├── learning_helpers.py    RLVR, terminal parsing (~240 lines)
├── interview_helpers.py   Interview flow (~220 lines)
├── codebase_helpers.py    Symbol extraction (~260 lines)
├── sessions_helpers.py    Session lifecycle (~270 lines)
├── metacognitive_helpers.py  5-stage prompting (~200 lines)
├── preflight_helpers.py   Preflight enforcement (~180 lines)
├── calibration_helpers.py CSR calibration (~180 lines)
├── purpose_helpers.py     Hierarchical inference (~150 lines)
├── hybrid_helpers.py      Hypothesis synthesis (~120 lines)
└── self_awareness_helpers.py Schema introspection (~250 lines)

tests/
├── conftest.py            Fixtures + Layer 4 failure logging
├── test_tools/            Layer 1: Static domain tests
│   ├── test_reasoning.py  16 tests (9 tools)
│   ├── test_learning.py   6 tests (4 tools)
│   ├── test_codebase.py   8 tests (8 tools)
│   ├── test_metacognitive.py 7 tests (3 tools)
│   └── test_interview.py  5 tests (4 tools)
├── scenarios/             Layer 2: YAML workflow tests
├── test_scenarios.py      YAML runner
├── test_coverage.py       Layer 3: Self-aware coverage
└── test_server.py         Legacy v22/v23 tests
```

### Database Tables

| Table | Purpose |
|-------|---------|
| `reasoning_sessions` | Session state, goal, context |
| `thought_nodes` | Hypothesis tree with Bayesian scores |
| `critique_results` | Counterarguments and penalties |
| `scientific_laws` | Grounding laws with weights |
| `outcome_records` | Success/failure for learning |
| `user_trait_profiles` | Persistent user preferences |
| `interview_history` | Archived Q&A for learning |
| `file_registry` | Indexed project files |
| `file_symbols` | Extracted symbols |
| `lsif_references` | LSIF index data |

---

## 📚 Scientific Laws Library

PAS includes 15+ established laws to ground reasoning:

- **CAP Theorem** - Distributed systems trade-offs
- **Occam's Razor** - Prefer simpler solutions
- **Conway's Law** - System structure mirrors organization
- **Hyrum's Law** - All observable behaviors will be depended on
- **Brooks' Law** - Adding people to late projects makes them later
- **Amdahl's Law** - Parallel speedup limits
- **Hofstadter's Law** - It always takes longer than expected
- *...and more*

---

## 🧪 Testing

PAS includes a **self-aware test suite** that tests itself and logs discovered bugs back into PAS for learning.

### Test Suite Architecture (v42)

| Layer | Purpose | File(s) |
|-------|---------|---------|
| **Layer 1** | Static unit tests for 38 tools | `tests/test_tools/*.py` |
| **Layer 2** | YAML-defined workflow scenarios | `tests/scenarios/*.yaml` |
| **Layer 3** | Self-aware coverage reporter | `tests/test_coverage.py` |
| **Layer 4** | Failure logging to PAS | `tests/conftest.py` hooks |

### Running Tests

```bash
# Full suite with PAS failure logging
PAS_DB_NAME=mcp_pas pytest tests/ -v

# Fast mode (no failure logging)
PAS_DB_NAME=mcp_pas PAS_LOG_FAILURES=false pytest tests/ -v

# Run only unit tests
PAS_DB_NAME=mcp_pas pytest tests/test_tools/ -v

# Run YAML scenarios
PAS_DB_NAME=mcp_pas pytest tests/test_scenarios.py -v

# Check coverage (fails if new tool lacks tests)
PAS_DB_NAME=mcp_pas pytest tests/test_coverage.py -v
```

### Self-Aware Coverage

When new tools are added, `test_coverage.py` automatically fails with:
```
Missing tests for: [new_tool_name]
```

This ensures test coverage grows with the codebase.

---

## 🤖 Model Compatibility

| Model | Status |
|-------|--------|
| Claude Opus 4.5 | ✅ Works |
| Claude Sonnet 4.5 | ✅ Works |
| Gemini 3 Flash | ✅ Works |
| Gemini 3 Pro | ⚠️ Tool call format issues |

---

## 📈 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

Current version: **v42-tests**

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Additional scientific laws
- New reasoning patterns
- Domain-specific question sets
- Performance optimizations

---

## 📄 License

GNU AGPLv3 License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Named after the three pillars of Western philosophy:
- **Plato** - Ideation and hypothesis generation
- **Aristotle** - Empirical grounding in scientific laws
- **Socrates** - Dialectic critique and questioning

*"The unexamined thought is not worth believing."*
