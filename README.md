# 🏛️ PAS - Scientific Reasoning MCP Server

**P**lato • **A**ristotle • **S**ocrates

A Model Context Protocol (MCP) server that brings structured, Bayesian reasoning to AI agents. PAS implements a Tree of Thoughts architecture with scientific law grounding, enabling AI to reason more deliberately through complex problems.

[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)](https://www.postgresql.org/)
[![Version](https://img.shields.io/badge/version-v32-orange.svg)](CHANGELOG.md)

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
| **Session Tagging** | Organize sessions with tags for retrieval |
| **Past Failures Surfacing** | Learn from similar past failures during critique |

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

### Sequential Analysis (v32)
| Tool | Description |
|------|-------------|
| `prepare_sequential_analysis` | Get prompts for gap analysis on top candidates |
| `store_sequential_analysis` | Store gap analysis, detect systemic gaps |

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
| `get_next_question` | Get one question at a time with choices |
| `submit_answer` | Answer question, trigger follow-ups |
| `check_interview_complete` | Check if enough context gathered |

### Self-Learning
| Tool | Description |
|------|-------------|
| `record_outcome` | Record success/failure with attribution |
| `parse_terminal_output` | Auto-detect success/failure signals |
| `refresh_law_weights` | Update law weights based on outcomes |
| `log_conversation` | Store user input for semantic search |
| `search_conversation_log` | Find past context by similarity |

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
OPENAI_API_KEY=sk-...  # For embeddings
```

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
        "DATABASE_URL": "postgresql://user:password@localhost:5432/mcp_pas"
      }
    }
  }
}
```

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

4. Deepen if score < 0.9 or gap < 0.1
   → store_expansion(parent_node_id="...", h1_text="Improved approach...", h1_confidence=0.95)

5. Get recommendation
   → finalize_session(session_id="...")
   # Returns: quality_gate, exhaustive_prompt, recommendation
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

## 📊 PAS vs Sequential Thinking

| Aspect | Sequential Thinking | PAS |
|--------|---------------------|-----|
| Structure | Linear chain | Bayesian Tree |
| Scoring | None | Prior × Likelihood = Posterior |
| Self-Critique | None | Tiered penalties |
| Quality Gate | None | 0.9 score, 0.1 gap |
| Self-Learning | None | Laws + failure memory |
| Output | Final thought | Ranked recommendation + gaps |

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

## 🧪 Model Compatibility

| Model | Status |
|-------|--------|
| Claude Opus 4.5 | ✅ Works |
| Claude Sonnet 4.5 | ✅ Works |
| Gemini 3 Flash | ✅ Works |
| Gemini 3 Pro | ⚠️ Tool call format issues |

---

## 📈 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

Current version: **v32**

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
