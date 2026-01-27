# 🏛️ PAS - Scientific Reasoning MCP Server

**P**lato • **A**ristotle • **S**ocrates

A Model Context Protocol (MCP) server that brings structured, Bayesian reasoning to AI agents. PAS implements a Tree of Thoughts architecture with scientific law grounding, enabling AI to reason more deliberately through complex problems.

[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)](https://www.postgresql.org/)

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
| **Smart Interview Pattern** | Structured Q&A to gather missing context |
| **Interview Self-Learning** | Archive Q&A patterns for effectiveness analysis |
| **Auto-Finalization** | Heuristic penalties + comparative critique for decisions |
| **Self-Learning** | Track outcomes to improve law weights over time |
| **Session Lifecycle** | Semantic deduplication, continuation, and inheritance |

---

## 🛠️ Tools (18 Total)

### Session Management
| Tool | Description |
|------|-------------|
| `start_reasoning_session` | Begin a new reasoning session with a goal |
| `get_session_status` | Check session state and thought count |
| `find_or_create_session` | Smart router - finds existing or creates new session |
| `complete_session` | Explicitly close a session |
| `resume_session` | Continue a completed session with inheritance |

### Reasoning (Expand & Critique)
| Tool | Description |
|------|-------------|
| `prepare_expansion` | Get context + relevant laws for hypothesis generation |
| `store_expansion` | Store 3 hypotheses with Bayesian scoring |
| `prepare_critique` | Get node context for generating counterarguments |
| `store_critique` | Store critique and update likelihood scores |
| `search_relevant_laws` | Find scientific laws by semantic similarity |

### Tree Navigation
| Tool | Description |
|------|-------------|
| `get_reasoning_tree` | View full tree structure with scores |
| `get_best_path` | Find highest-scoring reasoning path |
| `finalize_session` | Auto-critique top candidates, return recommendation |

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
| `refresh_law_weights` | Update law weights based on outcomes |

### Analytics (Views)
| View | Description |
|------|-------------|
| `interview_effectiveness` | Question→outcome correlation by domain |
| `law_domain_stats` | Law success rates per domain |

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 16+ with extensions:
  - `pgvector` (semantic similarity)
  - `ltree` (tree paths)

### Step 1: Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/mcp-pas.git
cd mcp-pas

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

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

# Seed scientific laws
python seed_laws.py
```

### Step 3: Environment Configuration

```bash
cp env.template .env
# Edit .env with your database credentials
```

```env
DATABASE_URL=postgresql://user:password@localhost:5432/mcp_pas
```

---

## 🔌 IDE Integration

### VS Code (with Claude/Gemini Extension)

Add to your MCP configuration (`~/.gemini/antigravity/mcp_config.json` or similar):

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

### Cursor / Other IDEs

Similar configuration - point to `server.py` with the virtual environment's Python interpreter.

**After configuration**: Restart your IDE for the MCP server to be detected.

---

## 🚀 Usage Walkthrough

### Basic Reasoning Flow

```
1. Start session
   → mcp_pas-server_start_reasoning_session(user_goal="Design a caching layer")

2. Expand hypotheses
   → mcp_pas-server_prepare_expansion(session_id="...")
   → mcp_pas-server_store_expansion(session_id="...", hypotheses=[...])

3. Critique top hypothesis
   → mcp_pas-server_prepare_critique(node_id="...")
   → mcp_pas-server_store_critique(node_id="...", critique={...})

4. Get recommendation
   → mcp_pas-server_finalize_session(session_id="...")
```

### Smart Interview Pattern

```
1. Identify knowledge gaps
   → mcp_pas-server_identify_gaps(session_id="...")

2. Ask questions one at a time
   → mcp_pas-server_get_next_question(session_id="...")
   → mcp_pas-server_submit_answer(session_id="...", question_id="...", answer="B")

3. Check completion
   → mcp_pas-server_check_interview_complete(session_id="...")
```

### Session Lifecycle

```
# Smart routing - finds/creates/continues automatically
→ mcp_pas-server_find_or_create_session(goal_text="Design Dill UI")

# Returns:
# - action: "existing" (active session found)
# - action: "continuation" (completed session, new linked session created)
# - action: "new" (no match, fresh session)
```

---

## 📊 Example Output

```json
{
  "recommendation": {
    "content": "Write-Through Cache with Redis: All writes update both cache and DB...",
    "original_score": 0.9494,
    "adjusted_score": 0.9494,
    "penalties_applied": []
  },
  "runner_up": {
    "content": "In-Memory Cache with TTL...",
    "adjusted_score": 0.9054
  },
  "decision_quality": "medium",
  "gap_analysis": "Moderate confidence (gap: 0.044)"
}
```

---

## 🧪 Model Compatibility

Tested with the following LLM models:

| Model | Status |
|-------|--------|
| Claude Opus 4.5 | ✅ Works |
| Claude Sonnet 4.5 | ✅ Works |
| Gemini 3 Flash | ✅ Works |
| Gemini 3 Pro | ⚠️ Tool call format issues |

---

## 📚 Scientific Laws Library

PAS includes 15+ established laws to ground reasoning:

- **CAP Theorem** - Distributed systems trade-offs
- **Occam's Razor** - Prefer simpler solutions
- **Conway's Law** - System structure mirrors organization
- **Hyrum's Law** - All observable behaviors will be depended on
- **Brooks' Law** - Adding people to late projects makes them later
- **Hofstadter's Law** - It always takes longer than expected
- *...and more*

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Additional scientific laws
- New reasoning patterns
- Performance optimizations
- Documentation improvements

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Named after the three pillars of Western philosophy:
- **Plato** - Ideation and hypothesis generation
- **Aristotle** - Empirical grounding in scientific laws
- **Socrates** - Dialectic critique and questioning

*"The unexamined thought is not worth believing."*
