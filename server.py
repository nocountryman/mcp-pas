#!/usr/bin/env python3
"""
Scientific Reasoning MCP Server
Bayesian Tree of Thoughts with Semantic Search

This server provides tools for structured scientific reasoning,
backed by PostgreSQL with pgvector for semantic search.
"""

import os
import json
import uuid
import logging
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import (
    TextContent,
    SamplingMessage,
    CreateMessageRequestParams,
    CreateMessageResult,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pas-server")

# =============================================================================
# Database Connection
# =============================================================================

def get_db_connection():
    """Create a new database connection."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    conn = psycopg2.connect(database_url, cursor_factory=RealDictCursor)
    register_vector(conn)
    return conn


def safe_close_connection(conn):
    """
    Safely close a database connection with rollback.
    
    Always rolls back before closing to prevent stuck transactions
    caused by uncommitted errors. Commit must be called explicitly.
    """
    if conn:
        try:
            conn.rollback()  # Safe to call even after commit
        except Exception:
            pass  # Ignore rollback errors
        finally:
            try:
                conn.close()
            except Exception:
                pass  # Ignore close errors


# =============================================================================
# MCP Server Setup
# =============================================================================

mcp = FastMCP(
    name="pas-server",
)


# =============================================================================
# v9b: Constitutional Principles for Critic
# =============================================================================

CONSTITUTIONAL_PRINCIPLES = [
    # Code-focused principles
    {
        "principle": "Identify unstated assumptions",
        "focus": "Implicit requirements, hidden dependencies, unverified preconditions",
        "question": "What must be true for this to work that isn't explicitly stated?",
        "domain": "code"
    },
    {
        "principle": "Check boundary conditions",
        "focus": "Empty inputs, null values, extreme sizes, negative numbers",
        "question": "What happens at the edges of valid input ranges?",
        "domain": "code"
    },
    {
        "principle": "Verify causation vs correlation",
        "focus": "Logical reasoning, cause-effect chains, spurious relationships",
        "question": "Does X actually cause Y, or are they merely correlated?",
        "domain": "code"
    },
    {
        "principle": "Question unstated dependencies",
        "focus": "Libraries, services, configurations, external state",
        "question": "What external systems/state does this rely on?",
        "domain": "code"
    },
    {
        "principle": "Consider failure modes",
        "focus": "Error handling, timeouts, partial failures, rollback",
        "question": "What happens when this fails? Is recovery possible?",
        "domain": "code"
    },
    # v9b.1: UI/UX design principles
    {
        "principle": "Apply Hick's Law (choice complexity)",
        "focus": "Decision paralysis, option overload, cognitive burden",
        "question": "Are there too many choices? Can we reduce or progressively disclose options?",
        "domain": "ui_ux"
    },
    {
        "principle": "Verify visual hierarchy",
        "focus": "Primary/secondary actions, information prominence, scan patterns",
        "question": "Is the most important element the most visually prominent? Does flow match user priority?",
        "domain": "ui_ux"
    },
    {
        "principle": "Check accessibility (WCAG)",
        "focus": "Color contrast, keyboard nav, screen readers, touch targets",
        "question": "Can users with disabilities perceive and operate this? Is contrast ≥4.5:1?",
        "domain": "ui_ux"
    },
    {
        "principle": "Test responsive breakpoints",
        "focus": "Mobile-first, tablet, desktop, large screens",
        "question": "Does this work on all screen sizes? Are touch targets ≥44px on mobile?",
        "domain": "ui_ux"
    },
    {
        "principle": "Validate feedback & affordances",
        "focus": "Loading states, hover/active states, error messages, success confirmations",
        "question": "Does the UI communicate what's happening? Are interactive elements obviously clickable?",
        "domain": "ui_ux"
    }
]


# =============================================================================
# v31: Past Failure Surfacing - Keyword Patterns
# =============================================================================

# Derived from actual outcome_records failures
KEYWORD_FAILURE_PATTERNS: dict[str, tuple[str, str]] = {
    # SCHEMA_BEFORE_CODE pattern
    'schema': ('SCHEMA_BEFORE_CODE', 'Run schema migration before deploying code changes'),
    'table': ('SCHEMA_BEFORE_CODE', 'Verify table exists in database before querying'),
    'column': ('SCHEMA_BEFORE_CODE', 'Check column exists before referencing in code'),
    'migration': ('SCHEMA_BEFORE_CODE', 'Apply database migrations before testing'),
    'alter': ('SCHEMA_BEFORE_CODE', 'Database schema change requires migration first'),
    
    # RESTART_BEFORE_VERIFY pattern
    'restart': ('RESTART_BEFORE_VERIFY', 'Restart MCP server after code changes'),
    'mcp': ('RESTART_BEFORE_VERIFY', 'MCP server needs restart to pick up new tools'),
    'server': ('RESTART_BEFORE_VERIFY', 'Server restart may be needed after changes'),
    
    # ENV_CHECK_FIRST pattern
    'venv': ('ENV_CHECK_FIRST', 'Verify .venv exists and is activated before pip commands'),
    'pip': ('ENV_CHECK_FIRST', 'Use project venv for pip installs, not system Python'),
    'install': ('ENV_CHECK_FIRST', 'Check virtual environment before installing packages'),
    
    # RESPECT_QUALITY_GATE pattern
    'quality': ('RESPECT_QUALITY_GATE', 'Do not skip quality gate checks'),
    'threshold': ('RESPECT_QUALITY_GATE', 'Respect score/gap thresholds before proceeding'),
    'gate': ('RESPECT_QUALITY_GATE', 'Quality gate must pass before implementation'),
}


def _search_relevant_failures(
    text: str,
    semantic_threshold: float = 0.55,
    limit: int = 3
) -> list[dict]:
    """
    v31: Search for past failures relevant to the given text.
    
    Combines:
    1. Semantic search on failure_reason_embedding
    2. Keyword pattern matching for deterministic warnings
    
    Returns list of warnings with source and details.
    """
    warnings = []
    text_lower = text.lower()
    
    # Part 1: Keyword pattern matching (fast, deterministic)
    seen_patterns: set[str] = set()
    for keyword, (pattern, warning_text) in KEYWORD_FAILURE_PATTERNS.items():
        if keyword in text_lower and pattern not in seen_patterns:
            warnings.append({
                'pattern': pattern,
                'source': 'keyword',
                'warning': warning_text,
                'triggered_by': keyword
            })
            seen_patterns.add(pattern)
    
    # Part 2: Semantic search (broader, probabilistic)
    if len(warnings) < limit:
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Generate embedding for the text
            embedding = get_embedding(text[:1000])
            
            cur.execute(
                """
                SELECT failure_reason, 
                       1 - (failure_reason_embedding <=> %s::vector) as similarity,
                       notes
                FROM outcome_records
                WHERE outcome = 'failure' 
                  AND failure_reason_embedding IS NOT NULL
                  AND 1 - (failure_reason_embedding <=> %s::vector) > %s
                ORDER BY failure_reason_embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, semantic_threshold, embedding, limit - len(warnings))
            )
            
            for row in cur.fetchall():
                # Avoid duplicates if semantic matches keyword pattern
                if not any(w.get('triggered_by') and w['triggered_by'] in row['failure_reason'].lower() for w in warnings):
                    warnings.append({
                        'pattern': 'SEMANTIC_MATCH',
                        'source': 'semantic',
                        'warning': row['failure_reason'],
                        'similarity': round(float(row['similarity']), 3)
                    })
            
            safe_close_connection(conn)
        except Exception as e:
            logger.warning(f"Semantic failure search failed: {e}")
    
    return warnings[:limit]


# =============================================================================
# v10b: Critic Ensemble Personas
# =============================================================================

CRITIC_PERSONAS = [
    {
        "name": "Strict Skeptic",
        "instruction": "Find any reason to reject this hypothesis. Assume it's flawed until proven otherwise. Be harsh but fair.",
        "focus_areas": ["logical fallacies", "unsupported claims", "wishful thinking"]
    },
    {
        "name": "Pragmatic Engineer",
        "instruction": "Will this work in production? Consider scale, maintenance, edge cases, and operational complexity.",
        "focus_areas": ["scalability", "maintainability", "error handling", "performance"]
    },
    {
        "name": "Domain Expert",
        "instruction": "Does this align with established patterns and scientific principles? Check against known laws and best practices.",
        "focus_areas": ["pattern adherence", "principle alignment", "industry standards"]
    }
]

AGGREGATION_GUIDANCE = "Run critique with each persona. For final severity: use MAX severity if any persona finds a critical flaw, otherwise AVERAGE across personas."


# =============================================================================
# v10a: Negation Detection Helper
# =============================================================================

NEGATION_PATTERNS = {'not', "n't", 'never', 'prohibit', 'avoid', 'without', 'disable', 'prevent', 'forbid', 'cannot', "won't", "shouldn't", "mustn't"}


def detect_negation(text: str) -> set:
    """Detect negation patterns in text for v10a NLI MVP."""
    text_lower = text.lower()
    return {p for p in NEGATION_PATTERNS if p in text_lower}


# =============================================================================
# Sampling Helper - Communicates with the Host LLM (Antigravity IDE)
# =============================================================================

class SamplingError(Exception):
    """Raised when sampling request fails or is denied."""
    pass


class SamplingDeniedError(SamplingError):
    """Raised when user denies the sampling request."""
    pass


async def sample_agent(
    prompt: str,
    context: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    system_prompt: str | None = None
) -> str:
    """
    Request text generation from the host LLM via MCP sampling.
    
    This function constructs a sampling/createMessage request and sends it
    to the connected MCP client (Antigravity IDE), which routes it to the LLM.
    
    Args:
        prompt: The user prompt to send to the LLM
        context: Optional context to prepend to the prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0-1.0)
        system_prompt: Optional system prompt override
        
    Returns:
        The generated text from the LLM
        
    Raises:
        SamplingDeniedError: If user denies the sampling request
        SamplingError: If sampling fails for any other reason
    """
    # Build the message content
    content = prompt
    if context:
        content = f"Context:\n{context}\n\n{prompt}"
    
    # Construct the sampling message
    messages = [
        SamplingMessage(
            role="user",
            content=TextContent(type="text", text=content)
        )
    ]
    
    # Build request parameters
    params = CreateMessageRequestParams(
        messages=messages,
        maxTokens=max_tokens,
        temperature=temperature,
        systemPrompt=system_prompt or "You are a scientific reasoning assistant. Provide clear, logical analysis."
    )
    
    try:
        # Get the current request context to access sampling capability
        from mcp.server.fastmcp import Context
        ctx = Context.current()  # type: ignore[attr-defined]
        
        if not ctx or not ctx.session:
            raise SamplingError("No active MCP session - cannot perform sampling")
        
        # Check if client supports sampling
        client_caps = ctx.session.client_capabilities
        if not client_caps or not client_caps.sampling:
            raise SamplingError("Client does not support sampling capability")
        
        # Send the sampling request via JSON-RPC
        logger.info(f"Requesting sample from host LLM: {prompt[:100]}...")
        
        result: CreateMessageResult = await ctx.session.create_message(
            messages=params.messages,
            max_tokens=params.maxTokens,
            temperature=params.temperature,
            system_prompt=params.systemPrompt
        )
        
        # Extract the response text
        if result.content and result.content.type == "text":
            logger.info(f"Received response: {result.content.text[:100]}...")
            return result.content.text
        else:
            raise SamplingError(f"Unexpected response content type: {result.content}")
            
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check for denial patterns
        if "denied" in error_msg or "rejected" in error_msg or "cancelled" in error_msg:
            logger.warning(f"Sampling request denied by user: {e}")
            raise SamplingDeniedError(f"Sampling request was denied: {e}")
        
        # Check for capability errors
        if "not supported" in error_msg or "capability" in error_msg:
            logger.error(f"Sampling not supported: {e}")
            raise SamplingError(f"Sampling not supported by client: {e}")
        
        # Re-raise other errors
        logger.error(f"Sampling failed: {e}")
        raise SamplingError(f"Sampling request failed: {e}")


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
async def start_reasoning_session(user_goal: str) -> dict[str, Any]:
    """
    Start a new reasoning session for the given goal.
    
    Creates a new session in the database and returns the session ID.
    Use this to begin a structured reasoning process.
    
    Args:
        user_goal: The high-level goal or question to reason about
        
    Returns:
        Dictionary with session_id and status
    """
    if not user_goal or not user_goal.strip():
        return {
            "success": False,
            "error": "User goal cannot be empty"
        }
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Generate a new session ID
        session_id = str(uuid.uuid4())
        
        # v19: Generate goal embedding for domain detection
        goal_embedding = None
        try:
            goal_embedding = get_embedding(user_goal.strip())
        except Exception as e:
            logger.warning(f"Failed to generate goal embedding: {e}")
        
        # Insert the new session
        cur.execute(
            """
            INSERT INTO reasoning_sessions (id, goal, goal_embedding, state, context)
            VALUES (%s, %s, %s, 'active', %s)
            RETURNING id, goal, state, created_at
            """,
            (session_id, user_goal.strip(), goal_embedding, json.dumps({"source": "mcp_tool"}))
        )
        
        row = cur.fetchone()
        
        # =====================================================================
        # v22 Feature 1: Load Persistent Trait Profiles
        # Query user_trait_profiles for this project and pre-populate context
        # =====================================================================
        persistent_traits = []
        user_id = None
        
        try:
            # Use goal hash as a proxy for project identity (could be enhanced)
            import hashlib
            # Generate user_id from goal pattern (simplified - in production use project path)
            user_id = hashlib.sha256(user_goal.strip()[:100].encode()).hexdigest()[:32]
            
            cur.execute("""
                SELECT trait_name, cumulative_score, last_reinforced_at
                FROM user_trait_profiles
                WHERE user_id = %s AND cumulative_score > 0
                ORDER BY cumulative_score DESC
                LIMIT 5
            """, (user_id,))
            
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            half_life_days = 30
            
            for trait_row in cur.fetchall():
                days_elapsed = (now - trait_row["last_reinforced_at"]).days
                decayed_score = trait_row["cumulative_score"] * (0.5 ** (days_elapsed / half_life_days))
                
                if decayed_score >= 0.3:  # Threshold for inclusion
                    persistent_traits.append({
                        "trait": trait_row["trait_name"],
                        "confidence": min(decayed_score, 1.0),
                        "source": "persistent",
                        "days_since_reinforced": days_elapsed
                    })
            
            if persistent_traits:
                # Update session context with persistent traits AND user_id
                cur.execute("""
                    UPDATE reasoning_sessions 
                    SET context = context || %s::jsonb
                    WHERE id = %s
                """, (json.dumps({"persistent_traits": persistent_traits, "user_id": user_id}), session_id))
                conn.commit()
                logger.info(f"v22: Loaded {len(persistent_traits)} persistent traits for session")
            else:
                # v22: Still store user_id for trait persistence on record_outcome
                cur.execute("""
                    UPDATE reasoning_sessions 
                    SET context = context || %s::jsonb
                    WHERE id = %s
                """, (json.dumps({"user_id": user_id}), session_id))
                conn.commit()
                
        except Exception as e:
            logger.warning(f"v22: Failed to load persistent traits: {e}")
        
        conn.commit()
        
        logger.info(f"Created reasoning session: {session_id}")
        
        response = {
            "success": True,
            "session_id": str(row["id"]),
            "goal": row["goal"],
            "state": row["state"],
            "created_at": row["created_at"].isoformat(),
            "message": f"Reasoning session started. Use this session_id for subsequent reasoning operations."
        }
        
        if persistent_traits:
            response["persistent_traits"] = persistent_traits
            response["message"] += f" Loaded {len(persistent_traits)} persistent trait(s) from history."
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def get_session_status(session_id: str) -> dict[str, Any]:
    """
    Get the current status of a reasoning session.
    
    Args:
        session_id: The UUID of the session to check
        
    Returns:
        Dictionary with session details and thought count
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get session details
        cur.execute(
            """
            SELECT id, goal, state, context, prior_beliefs, created_at, updated_at
            FROM reasoning_sessions
            WHERE id = %s
            """,
            (session_id,)
        )
        
        session = cur.fetchone()
        if not session:
            return {
                "success": False,
                "error": f"Session {session_id} not found"
            }
        
        # Count thoughts in this session
        cur.execute(
            """
            SELECT COUNT(*) as thought_count,
                   MAX(depth) as max_depth
            FROM thought_nodes
            WHERE session_id = %s
            """,
            (session_id,)
        )
        
        stats = cur.fetchone()
        
        return {
            "success": True,
            "session_id": str(session["id"]),
            "goal": session["goal"],
            "state": session["state"],
            "thought_count": stats["thought_count"],
            "max_depth": stats["max_depth"] or 0,
            "created_at": session["created_at"].isoformat(),
            "updated_at": session["updated_at"].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def search_relevant_laws(query: str, limit: int = 5) -> dict[str, Any]:
    """
    Search for relevant scientific laws using semantic similarity.
    
    Uses pgvector to find laws that are semantically similar to the query.
    
    Args:
        query: The text to search for relevant laws
        limit: Maximum number of results (default: 5)
        
    Returns:
        List of relevant laws with similarity scores
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        # Generate embedding for the query
        model = SentenceTransformer('all-mpnet-base-v2')
        query_embedding = model.encode([query])[0].tolist()
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Search for similar laws
        cur.execute(
            """
            SELECT 
                id,
                law_name,
                definition,
                scientific_weight,
                1 - (embedding <=> %s::vector) as similarity
            FROM scientific_laws
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, limit)
        )
        
        laws = []
        for row in cur.fetchall():
            laws.append({
                "id": row["id"],
                "law_name": row["law_name"],
                "definition": row["definition"],
                "scientific_weight": float(row["scientific_weight"]),
                "similarity": round(float(row["similarity"]), 4)
            })
        
        return {
            "success": True,
            "query": query,
            "results": laws,
            "count": len(laws)
        }
        
    except Exception as e:
        logger.error(f"Failed to search laws: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# Reasoning Tools - Expansion & Critique (Phase 1: Prompt-Driven)
# =============================================================================

def get_embedding(text: str) -> list[float]:
    """Generate a 768-dim embedding for the given text."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')
    return model.encode([text])[0].tolist()


@mcp.tool()
async def prepare_expansion(session_id: str, parent_node_id: str | None = None) -> dict[str, Any]:
    """
    Prepare context for thought expansion. Returns parent content, session goal, 
    and relevant scientific laws so the LLM can generate hypotheses.
    
    After calling this, generate 3 hypotheses and pass them to store_expansion().
    
    Args:
        session_id: The reasoning session UUID
        parent_node_id: Parent thought UUID (None to expand from root/goal)
        
    Returns:
        Context dict with parent_content, goal, relevant_laws for hypothesis generation
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get session
        cur.execute("SELECT id, goal, state FROM reasoning_sessions WHERE id = %s", (session_id,))
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": f"Session {session_id} not found"}
        
        if session["state"] != "active":
            return {"success": False, "error": f"Session is {session['state']}, not active"}
        
        # Get parent node or use goal as root
        if parent_node_id:
            cur.execute(
                "SELECT id, path, content FROM thought_nodes WHERE id = %s AND session_id = %s",
                (parent_node_id, session_id)
            )
            parent = cur.fetchone()
            if not parent:
                return {"success": False, "error": f"Parent node {parent_node_id} not found"}
            parent_content = parent["content"]
            parent_path = parent["path"]
        else:
            parent_content = session["goal"]
            parent_path = None
            
            # Check for existing root
            cur.execute(
                "SELECT id, path FROM thought_nodes WHERE session_id = %s AND path = 'root'",
                (session_id,)
            )
            existing_root = cur.fetchone()
            if existing_root:
                parent_node_id = str(existing_root["id"])
                parent_path = "root"
        
        # Find relevant laws for this context
        embedding = get_embedding(parent_content)
        cur.execute(
            """
            SELECT id, law_name, definition, scientific_weight,
                   1 - (embedding <=> %s::vector) as similarity
            FROM scientific_laws WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector LIMIT 3
            """,
            (embedding, embedding)
        )
        
        laws = [{"id": r["id"], "law_name": r["law_name"], "definition": r["definition"],
                 "weight": float(r["scientific_weight"]), "similarity": round(float(r["similarity"]), 4)}
                for r in cur.fetchall()]
        
        # =====================================================================
        # v22 Feature 2: Law Weight Boosting Based on User Traits
        # Boost law weights that correlate with user's persistent/latent traits
        # =====================================================================
        # Get session context for traits
        cur.execute("SELECT context FROM reasoning_sessions WHERE id = %s", (session_id,))
        context_row = cur.fetchone()
        session_context = context_row["context"] if context_row and context_row["context"] else {}
        
        # Combine persistent and latent traits
        all_trait_names = []
        persistent_traits = session_context.get("persistent_traits", [])
        latent_traits = session_context.get("latent_traits", [])
        
        for t in persistent_traits:
            if t.get("trait"):
                all_trait_names.append(t["trait"])
        for t in latent_traits:
            if t.get("trait"):
                all_trait_names.append(t["trait"])
        
        # Apply boosting if we have traits
        boosted_laws = []
        if all_trait_names and laws:
            cur.execute("""
                SELECT law_id, SUM(boost_factor) as total_boost
                FROM trait_law_correlations
                WHERE trait_name = ANY(%s)
                GROUP BY law_id
            """, (all_trait_names,))
            
            boosts = {row["law_id"]: row["total_boost"] for row in cur.fetchall()}
            
            for law in laws:
                if law["id"] in boosts:
                    # Cap boost at 50% to prevent runaway
                    boost = min(boosts[law["id"]], 0.5)
                    original_weight = law["weight"]
                    law["weight"] = round(law["weight"] * (1 + boost), 4)
                    law["boosted_by"] = boost
                    boosted_laws.append(law["law_name"])
                    logger.debug(f"v22: Boosted {law['law_name']} from {original_weight} to {law['weight']}")
        
        # =====================================================================
        # v21 Phase 3: Trait-Aware Instructions
        # Read latent_traits from session context and append dynamic guidance
        # =====================================================================
        base_instructions = "Consider: What is requested? What files/modules might be affected? For each hypothesis, declare SCOPE as specific file paths. Optionally prefix with layer if helpful: [API] routes.py, [DB] models.py, [tests] test_auth.py. Generate 3 hypotheses with confidence (0.0-1.0). Call store_expansion(h1_text=..., h1_confidence=..., h1_scope='[layer] file1.py, file2.py', ...)."
        
        # v22: latent_traits already retrieved in Feature 2 section above
        
        # Trait-to-guidance mapping
        TRAIT_GUIDANCE = {
            "RISK_AVERSE": "User prefers SAFE, REVERSIBLE approaches. Prioritize hypotheses with clear rollback strategies and minimal blast radius.",
            "MINIMALIST": "User values SIMPLICITY. Favor minimal implementations that solve the core problem without unnecessary complexity.",
            "CONTROL_ORIENTED": "User wants EXPLICIT CONTROL. Prefer solutions that give user visibility and manual override options.",
            "AUTOMATION_TRUSTING": "User trusts AUTOMATION. Feel free to propose AI/automated solutions without excessive manual oversight.",
            "ACCESSIBILITY_FOCUSED": "User prioritizes ACCESSIBILITY. Consider beginner-friendly approaches with good documentation.",
            "PRAGMATIST": "User seeks BALANCED solutions. Blend theory with practicality, avoid extremes.",
            "EXPLICIT_CONTROL": "User wants TRANSPARENCY. Prefer explicit configuration over magic/convention.",
            "SPEED_FOCUSED": "User prioritizes PERFORMANCE. Consider efficiency and fast execution in hypotheses.",
            "SAFETY_CONSCIOUS": "User values ERROR PREVENTION. Include validation and safety checks in proposed solutions.",
            "AUTONOMY_FOCUSED": "User wants FREEDOM and flexibility. Avoid overly prescriptive or locked-down solutions.",
        }
        
        # Build trait guidance section
        trait_guidance = ""
        if latent_traits:
            guidance_lines = []
            for trait_info in latent_traits:
                trait_name = trait_info.get("trait")
                confidence = trait_info.get("confidence", 0)
                if trait_name in TRAIT_GUIDANCE and confidence >= 0.7:
                    guidance_lines.append(f"- {TRAIT_GUIDANCE[trait_name]}")
            
            if guidance_lines:
                trait_guidance = "\n\nUSER PREFERENCES DETECTED:\n" + "\n".join(guidance_lines)
                logger.info(f"v21 Phase 3: Added {len(guidance_lines)} trait guidance(s) to instructions")
        
        instructions = base_instructions + trait_guidance
        
        # =====================================================================
        # v31: Past Failure Surfacing
        # Search for relevant past failures to warn agent before hypothesis generation
        # =====================================================================
        past_failure_warnings = _search_relevant_failures(parent_content)
        
        response = {
            "success": True,
            "session_id": session_id,
            "parent_node_id": parent_node_id,
            "parent_path": parent_path,
            "parent_content": parent_content,
            "goal": session["goal"],
            "relevant_laws": laws,
            "instructions": instructions,
            "latent_traits": latent_traits if latent_traits else None  # v21: Include for transparency
        }
        
        # v31: Add warnings if any found
        if past_failure_warnings:
            response["past_failure_warnings"] = past_failure_warnings
            logger.info(f"v31: Surfaced {len(past_failure_warnings)} failure warning(s)")
        
        return response
        
    except Exception as e:
        logger.error(f"prepare_expansion failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def store_expansion(
    session_id: str,
    parent_node_id: str | None,
    # Hypothesis 1 (required)
    h1_text: str,
    h1_confidence: float,
    h1_scope: Optional[str] = None,
    # Hypothesis 2 (optional)
    h2_text: Optional[str] = None,
    h2_confidence: Optional[float] = None,
    h2_scope: Optional[str] = None,
    # Hypothesis 3 (optional)
    h3_text: Optional[str] = None,
    h3_confidence: Optional[float] = None,
    h3_scope: Optional[str] = None,
    # v7b: Revision tracking (borrowed from sequential thinking)
    is_revision: bool = False,
    revises_node_id: Optional[str] = None,
    # v25: Conversation logging
    source_text: Optional[str] = None,
    user_id: Optional[str] = None
) -> dict[str, Any]:
    """
    Store generated hypotheses with Bayesian scoring.
    
    Uses flattened parameters for universal LLM compatibility.
    At least one hypothesis (h1) is required, h2 and h3 are optional.
    
    Args:
        session_id: The reasoning session UUID
        parent_node_id: Parent node UUID (None if expanding from goal)
        h1_text: First hypothesis text (required)
        h1_confidence: First hypothesis confidence 0.0-1.0 (required)
        h1_scope: First hypothesis affected scope (comma-separated files/modules)
        h2_text: Second hypothesis text (optional)
        h2_confidence: Second hypothesis confidence 0.0-1.0 (optional)
        h2_scope: Second hypothesis affected scope
        h3_text: Third hypothesis text (optional)
        h3_confidence: Third hypothesis confidence 0.0-1.0 (optional)
        h3_scope: Third hypothesis affected scope
        is_revision: True if these hypotheses revise previous thinking (v7b)
        revises_node_id: Node ID being revised, if is_revision is True
        source_text: v25 - Raw user input that inspired this hypothesis (logged for semantic search)
        user_id: v25 - Optional user identifier for multi-user scenarios
        
    Returns:
        Created nodes with Bayesian posterior scores, declared scopes, and revision info
    """
    try:
        # Build hypotheses list from flattened params
        hypotheses = []
        if h1_text:
            hypotheses.append({"hypothesis": h1_text, "confidence": h1_confidence or 0.5, "scope": h1_scope})
        if h2_text:
            hypotheses.append({"hypothesis": h2_text, "confidence": h2_confidence or 0.5, "scope": h2_scope})
        if h3_text:
            hypotheses.append({"hypothesis": h3_text, "confidence": h3_confidence or 0.5, "scope": h3_scope})
        
        if not hypotheses:
            return {"success": False, "error": "At least one hypothesis (h1_text) is required"}
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Verify session
        cur.execute("SELECT goal FROM reasoning_sessions WHERE id = %s AND state = 'active'", (session_id,))
        if not cur.fetchone():
            return {"success": False, "error": "Session not found or not active"}
        
        # Determine parent path
        if parent_node_id:
            cur.execute("SELECT path FROM thought_nodes WHERE id = %s", (parent_node_id,))
            parent = cur.fetchone()
            if not parent:
                return {"success": False, "error": "Parent node not found"}
            parent_path = parent["path"]
        else:
            # Create or get root
            cur.execute("SELECT id, path FROM thought_nodes WHERE session_id = %s AND path = 'root'", (session_id,))
            root = cur.fetchone()
            if root:
                parent_path = "root"
                parent_node_id = str(root["id"])
            else:
                # Create root
                cur.execute("SELECT goal FROM reasoning_sessions WHERE id = %s", (session_id,))
                goal = cur.fetchone()["goal"]
                root_id = str(uuid.uuid4())
                root_emb = get_embedding(goal)
                cur.execute(
                    "INSERT INTO thought_nodes (id, session_id, path, content, node_type, prior_score, likelihood, embedding) VALUES (%s, %s, 'root', %s, 'root', 0.5, 0.5, %s)",
                    (root_id, session_id, goal, root_emb)
                )
                parent_path = "root"
                parent_node_id = root_id
        
        created_nodes = []
        for i, hyp in enumerate(hypotheses[:3]):
            hypothesis_text = str(hyp.get("hypothesis", ""))
            _conf = hyp.get("confidence")
            llm_confidence = float(_conf) if _conf is not None else 0.5  # type: ignore[arg-type]
            declared_scope = hyp.get("scope")
            
            if not hypothesis_text:
                continue
            
            # Generate embedding and find similar law
            hyp_emb = get_embedding(hypothesis_text)
            
            # v13a: Multi-Law Matching - fetch top-3 similar laws for ensemble prior
            cur.execute(
                """
                SELECT id, law_name, definition, scientific_weight, selection_count, success_count,
                       1 - (embedding <=> %s::vector) as similarity
                FROM scientific_laws WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector LIMIT 3
                """,
                (hyp_emb, hyp_emb)
            )
            laws = cur.fetchall()
            
            # v13a: Filter by similarity threshold (>0.2) and compute weighted ensemble prior
            MIN_SIMILARITY = 0.2
            matching_laws = [l for l in laws if l["similarity"] >= MIN_SIMILARITY]
            
            if matching_laws:
                # v35: Use extracted pure helper for ensemble prior
                prior, supporting_law_ids, law_name = _compute_ensemble_prior(
                    matching_laws, hypothesis_text
                )
                supporting_law = supporting_law_ids
                
                # v12b: Track law selection (DB update must stay inline)
                for law in matching_laws:
                    cur.execute(
                        "UPDATE scientific_laws SET selection_count = selection_count + 1 WHERE id = %s",
                        (law["id"],)
                    )
            else:
                prior = 0.5
                supporting_law = []
                law_name = None
            
            likelihood = llm_confidence
            
            # Insert node
            node_id = str(uuid.uuid4())
            new_path = f"{parent_path}.h{i+1}"
            
            cur.execute(
                """
                INSERT INTO thought_nodes (id, session_id, path, content, node_type, prior_score, likelihood, embedding, supporting_laws, declared_scope)
                VALUES (%s, %s, %s, %s, 'hypothesis', %s, %s, %s, %s, %s)
                RETURNING id, path, prior_score, likelihood, posterior_score
                """,
                (node_id, session_id, new_path, hypothesis_text, prior, likelihood, hyp_emb, supporting_law, declared_scope)
            )
            node = cur.fetchone()
            
            created_nodes.append({
                "node_id": str(node["id"]),
                "path": node["path"],
                "content": hypothesis_text[:150] + "..." if len(hypothesis_text) > 150 else hypothesis_text,
                "prior_score": float(node["prior_score"]),
                "likelihood": float(node["likelihood"]),
                "posterior_score": float(node["posterior_score"]) if node["posterior_score"] else None,
                "supporting_law": law_name,
                "declared_scope": declared_scope
            })
        
        # =====================================================================
        # v25: Conversation Logging - Store source_text if provided
        # =====================================================================
        conversation_log_id = None
        if source_text and created_nodes:
            try:
                first_node_id = created_nodes[0]["node_id"]
                source_embedding = get_embedding(source_text[:2000])  # Truncate for embedding (max useful context)
                
                cur.execute(
                    """
                    INSERT INTO conversation_log (session_id, thought_node_id, user_id, log_type, raw_text, embedding)
                    VALUES (%s, %s, %s, 'user_input', %s, %s)
                    RETURNING id
                    """,
                    (session_id, first_node_id, user_id, source_text, source_embedding)
                )
                log_entry = cur.fetchone()
                conversation_log_id = str(log_entry["id"]) if log_entry else None
                logger.info(f"v25: Created conversation_log entry {conversation_log_id} for source_text ({len(source_text)} chars)")
            except Exception as log_err:
                logger.warning(f"v25: Failed to create conversation_log entry: {log_err}")
                # Don't fail the whole operation for logging failure
        
        conn.commit()
        
        # Workflow nudge: Find top hypothesis and suggest critique
        top_node = max(created_nodes, key=lambda n: n.get("posterior_score") or 0) if created_nodes else None
        next_step = None
        if top_node:
            next_step = f"Challenge your top hypothesis. Call prepare_critique(node_id='{top_node['node_id']}')"
        
        # v7a: Confidence nudge (borrowed from sequential thinking meta-cognition)
        confidence_nudge = None
        if created_nodes:
            avg_confidence = sum(n.get("likelihood", 0.5) for n in created_nodes) / len(created_nodes)
            if avg_confidence < 0.65:
                confidence_nudge = f"Low confidence detected (avg: {avg_confidence:.2f}). Consider: (1) expand deeper on uncertain hypothesis, (2) add alternative perspectives, (3) gather more context before deciding."
        
        # v7b: Revision tracking response
        revision_info = None
        revision_nudge = None
        if is_revision:
            revision_info = {
                "is_revision": True,
                "revises_node_id": revises_node_id,
                "message": "Revision noted. Original hypothesis preserved for comparison."
            }
            if top_node:
                revision_nudge = f"Revision recorded. Consider critiquing the revised hypothesis. Call prepare_critique(node_id='{top_node['node_id']}')"
        
        # =====================================================================
        # v21: Scope-Based Failure Matching
        # Query historical failures that touched similar files/scopes
        # =====================================================================
        scope_warnings = []
        try:
            # Collect all declared scopes from this expansion
            all_scopes = []
            for node in created_nodes:
                scope = node.get("declared_scope")
                if scope:
                    # Parse comma-separated scope items
                    all_scopes.extend([s.strip() for s in scope.split(',') if s.strip()])
            
            if all_scopes:
                # Query failures that mention similar files/scopes
                scope_patterns = ['%' + s.split(':')[0] + '%' for s in all_scopes if s]  # Extract file/module names
                
                cur.execute("""
                    SELECT DISTINCT o.failure_reason, s.goal, t.declared_scope
                    FROM outcome_records o
                    JOIN reasoning_sessions s ON o.session_id = s.id
                    JOIN thought_nodes t ON t.session_id = s.id
                    WHERE o.outcome = 'failure'
                    AND o.failure_reason IS NOT NULL
                    AND t.declared_scope IS NOT NULL
                    AND (
                        -- Match if any scope pattern overlaps
                        t.declared_scope ILIKE ANY(%s)
                    )
                    ORDER BY o.created_at DESC
                    LIMIT 3
                """, (scope_patterns,))
                
                related_failures = cur.fetchall()
                for r in related_failures:
                    scope_warnings.append({
                        "past_goal": r["goal"][:60] + "..." if len(r["goal"]) > 60 else r["goal"],
                        "past_scope": r["declared_scope"][:80] if r["declared_scope"] else None,
                        "failure_reason": r["failure_reason"][:120] + "..." if len(r["failure_reason"]) > 120 else r["failure_reason"]
                    })
                
                if scope_warnings:
                    logger.info(f"v21: Found {len(scope_warnings)} scope-related historical failures")
        except Exception as e:
            conn.rollback()  # Clear any transaction issues
            logger.warning(f"v21: Scope-based failure matching failed: {e}")
        
        return {
            "success": True, 
            "session_id": session_id, 
            "created_nodes": created_nodes, 
            "count": len(created_nodes),
            "next_step": next_step,
            "confidence_nudge": confidence_nudge,
            "revision_info": revision_info,
            "revision_nudge": revision_nudge,
            "scope_warnings": scope_warnings if scope_warnings else None  # v21: Historical failure warnings
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"store_expansion failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def prepare_critique(
    node_id: str,
    critique_mode: str = "standard"  # v31a: 'standard' or 'negative_space'
) -> dict[str, Any]:
    """
    Prepare context for critiquing a thought node. Returns node content and 
    supporting laws so the LLM can generate counterarguments.
    
    After calling this, generate a critique and pass it to store_critique().
    
    Args:
        node_id: The thought node UUID to critique
        critique_mode: v31a - 'standard' (what's wrong?) or 'negative_space' (what's missing?)
        
    Returns:
        Context dict with node_content, supporting_laws for critique generation
    """

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            """
            SELECT t.id, t.session_id, t.path, t.content, t.prior_score, t.likelihood, t.posterior_score, t.supporting_laws, s.goal
            FROM thought_nodes t JOIN reasoning_sessions s ON t.session_id = s.id WHERE t.id = %s
            """,
            (node_id,)
        )
        node = cur.fetchone()
        if not node:
            return {"success": False, "error": f"Node {node_id} not found"}
        
        # Get supporting laws
        laws_text = []
        if node["supporting_laws"]:
            # v14c.1: Include failure_modes for targeted critique
            cur.execute("SELECT law_name, definition, failure_modes FROM scientific_laws WHERE id = ANY(%s)", (node["supporting_laws"],))
            for law in cur.fetchall():
                laws_text.append({
                    "law_name": law["law_name"], 
                    "definition": law["definition"],
                    "failure_modes": law["failure_modes"] or []  # v14c.1
                })
        
        # v16c: LLM Critique Synthesis - generate suggested critique
        # v31a: Support critique_mode for different critique perspectives
        suggested_critique = None
        negative_space_gaps = None  # v31a: Gaps found in negative_space mode
        
        # v31a: Select prompt based on critique mode
        if critique_mode == "negative_space":
            critique_prompt = f"""Analyze what this hypothesis does NOT address:

Hypothesis: {node["content"]}

Session Goal: {node["goal"]}

For EACH component of the goal, answer:
1. Is this component fully addressed by the hypothesis?
2. What aspects are MISSING or assumed?
3. What adjacent concerns are NOT considered?

Return ONLY a valid JSON object:
{{
  "gaps": [
    {{"component": "...", "addressed": true/false, "missing": "what's not covered"}},
    ...
  ],
  "blind_spots": ["things taken for granted"],
  "boundary_issues": ["where the hypothesis ends but the goal continues"],
  "overall_coverage": 0.8
}}

Focus on OMISSIONS, not flaws in what IS proposed."""
            system = "You are a gap analyst. Find what's MISSING, not what's wrong. Return only valid JSON."
        else:
            # Standard critique mode (existing behavior)
            critique_prompt = f"""Critique this hypothesis:
{node["content"]}

Session Goal: {node["goal"]}

Consider these scientific laws: {', '.join(l.get('law_name', '') for l in laws_text) if laws_text else 'None'}

Return ONLY a valid JSON object:
{{
  "counterargument": "Main counter-argument that challenges this hypothesis",
  "severity": 0.5,
  "major_flaws": ["flaw1", "flaw2"],
  "minor_flaws": ["minor issue"],
  "edge_cases": ["edge case 1"]
}}

Be specific and constructive. Focus on what could go wrong."""
            system = "You are a hypothesis critic. Return only valid JSON."

        
        # v32 FIX: MCP sampling not supported by clients
        # Instead of calling LLM directly, return prompts for agent to process
        # Agent will call LLM externally and pass results to store_critique
        llm_prompt = {
            "prompt": critique_prompt,
            "system": system,
            "expected_format": "JSON object with counterargument/severity/flaws" if critique_mode == "standard" else "JSON object with gaps/blind_spots/boundary_issues"
        }
        logger.info(f"v32: Returning prompt for agent to process (mode: {critique_mode})")

        
        # v27 -> v31: Surface past failures matching this hypothesis content
        # v31 Enhancement: Add keyword pattern matching in addition to semantic search
        past_failures = []
        
        # v31: First check keyword patterns (fast, deterministic)
        keyword_warnings = _search_relevant_failures(node["content"], semantic_threshold=0.55, limit=3)
        for w in keyword_warnings:
            if w.get('source') == 'keyword':
                past_failures.append({
                    "pattern": w.get('pattern'),
                    "warning": w.get('warning'),
                    "source": "keyword",
                    "triggered_by": w.get('triggered_by')
                })
        
        # v27: Then add semantic matches
        try:
            # Get embedding for current node content
            node_embedding = get_embedding(node["content"][:1000])
            cur.execute("""
                SELECT o.failure_reason, s.goal,
                       1 - (o.failure_reason_embedding <=> %s::vector) as similarity
                FROM outcome_records o
                JOIN reasoning_sessions s ON o.session_id = s.id
                WHERE o.outcome != 'success'
                AND o.failure_reason_embedding IS NOT NULL
                AND 1 - (o.failure_reason_embedding <=> %s::vector) > 0.5
                ORDER BY similarity DESC
                LIMIT 3
            """, (node_embedding, node_embedding))
            for r in cur.fetchall():
                # Avoid duplicates from keyword matches
                if not any(r["failure_reason"][:50] in str(pf.get('warning', '')) for pf in past_failures):
                    past_failures.append({
                        "failure_reason": r["failure_reason"][:200],
                        "session_goal": r["goal"][:100],
                        "similarity": round(float(r["similarity"]), 3),
                        "source": "semantic"
                    })
            if past_failures:
                logger.info(f"v31: Found {len(past_failures)} failure warning(s) for node {node_id}")
        except Exception as e:
            logger.warning(f"v31 past_failures lookup failed: {e}")
        
        # v29: Assumption Surfacing - return prompt for client LLM to extract assumptions
        # NOTE: MCP sampling via create_message is not supported by all clients,
        # so we return the prompt for the calling LLM to process directly
        assumption_extraction_prompt = f"""Analyze this hypothesis and extract 2-3 IMPLICIT ASSUMPTIONS that must be true for it to work:

Hypothesis: {node["content"]}

For each assumption, answer:
1. What is the assumption? (something taken for granted)
2. When could this assumption be FALSE? (challenge it)
3. What happens if this assumption fails? (the risk)

Focus on hidden dependencies, preconditions, and things taken for granted.
Format your response as a numbered list."""
        

        return {
            "success": True,
            "node_id": node_id,
            "path": node["path"],
            "node_content": node["content"],
            "session_goal": node["goal"],
            # v31a: Include critique mode used
            "critique_mode": critique_mode,
            "current_scores": {
                "prior": float(node["prior_score"]),
                "likelihood": float(node["likelihood"]),
                "posterior": float(node["posterior_score"]) if node["posterior_score"] else None
            },
            "supporting_laws": laws_text,
            # v32: LLM prompt for agent to process (replaces broken MCP sampling)
            "llm_prompt": llm_prompt,
            # v16c/v31a: These are now populated by agent after processing llm_prompt
            "suggested_critique": None,
            "negative_space_gaps": None,
            # v27: Past failures matching this hypothesis
            "past_failures": past_failures,
            # v29: Assumption Surfacing - prompt for client LLM to extract and challenge assumptions
            "assumption_extraction_prompt": assumption_extraction_prompt,
            # v32: Updated instructions for two-step workflow
            "instructions": f"Process the llm_prompt to generate critique. Mode: {critique_mode}. After generating, call store_critique(node_id='{node_id}', counterargument=..., severity_score=..., logical_flaws='...', edge_cases='...').",
            # v9b: Constitutional Principles for structured critique
            "constitutional_principles": CONSTITUTIONAL_PRINCIPLES,
            # v10b: Critic Ensemble Personas
            "critic_personas": CRITIC_PERSONAS,
            "aggregation_guidance": AGGREGATION_GUIDANCE
        }

        
    except Exception as e:
        logger.error(f"prepare_critique failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()

async def store_critique(
    node_id: str,
    counterargument: str,
    severity_score: float,
    logical_flaws: str = "",
    edge_cases: str = "",
    major_flaws: str = "",  # v9a: Major flaws that invalidate hypothesis
    minor_flaws: str = ""   # v9a: Minor concerns worth noting
) -> dict[str, Any]:
    """
    Store critique results and update node likelihood.
    
    Uses flattened parameters for universal LLM compatibility.
    
    Args:
        node_id: The thought node UUID
        counterargument: Main counter-argument text
        severity_score: Impact score 0.0-1.0 (higher = more severe critique)
        logical_flaws: Comma or newline separated list of flaws (optional)
        edge_cases: Comma or newline separated list of edge cases (optional)
        major_flaws: v9a - Major issues that would invalidate the hypothesis (comma-separated)
        minor_flaws: v9a - Minor concerns worth noting but not blocking (comma-separated)
        
    Returns:
        Updated node scores
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT likelihood FROM thought_nodes WHERE id = %s", (node_id,))
        node = cur.fetchone()
        if not node:
            return {"success": False, "error": "Node not found"}
        
        severity = float(severity_score)
        old_likelihood = float(node["likelihood"])
        
        # v9a: Tiered penalty calculation
        MAJOR_PENALTY = 0.15  # Each major flaw has significant impact
        MINOR_PENALTY = 0.03  # Each minor flaw has small impact
        
        major_count = len([f for f in major_flaws.replace('\n', ',').split(',') if f.strip()])
        minor_count = len([f for f in minor_flaws.replace('\n', ',').split(',') if f.strip()])
        
        if major_count > 0 or minor_count > 0:
            # v9a: Use tiered penalty calculation
            total_penalty = (major_count * MAJOR_PENALTY) + (minor_count * MINOR_PENALTY)
            new_likelihood = max(0.1, old_likelihood * (1 - min(total_penalty, 0.8)))
            tier_breakdown = {
                "major_count": major_count,
                "minor_count": minor_count,
                "total_penalty": round(total_penalty, 3),
                "method": "tiered"
            }
        else:
            # Legacy: fallback to severity_score
            new_likelihood = max(0.1, old_likelihood * (1 - severity * 0.5))
            tier_breakdown = {
                "major_count": 0,
                "minor_count": 0,
                "total_penalty": round(severity * 0.5, 3),
                "method": "legacy_severity"
            }
        
        cur.execute(
            "UPDATE thought_nodes SET likelihood = %s, updated_at = NOW() WHERE id = %s RETURNING prior_score, likelihood, posterior_score",
            (new_likelihood, node_id)
        )
        updated = cur.fetchone()
        conn.commit()
        
        # Count flaws/cases for summary (simple split)
        flaw_count = len([f for f in logical_flaws.replace('\n', ',').split(',') if f.strip()])
        case_count = len([c for c in edge_cases.replace('\n', ',').split(',') if c.strip()])
        
        return {
            "success": True,
            "node_id": node_id,
            "critique_summary": {
                "counterargument": counterargument[:200],
                "flaw_count": flaw_count,
                "edge_case_count": case_count,
                "severity_score": severity
            },
            "score_update": {
                "old_likelihood": old_likelihood,
                "new_likelihood": float(updated["likelihood"]),
                "posterior_score": float(updated["posterior_score"]) if updated["posterior_score"] else None
            },
            # v9a: Tier breakdown for transparency
            "tier_breakdown": tier_breakdown
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"store_critique failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def prepare_sequential_analysis(
    session_id: str,
    top_n: int = 5
) -> dict[str, Any]:
    """
    v32: Prepare adversarial gap analysis prompts for top candidates.
    
    Returns prompts for agent to process. Agent should analyze each
    candidate and call store_sequential_analysis with results.
    
    Args:
        session_id: The reasoning session UUID
        top_n: Number of top candidates to analyze (max 5)
        
    Returns:
        Prompts for each candidate for agent to process
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get session goal
        cur.execute(
            "SELECT goal FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": "Session not found"}
        
        # Get top candidates
        cur.execute(
            """
            SELECT id, content, posterior_score
            FROM thought_nodes
            WHERE session_id = %s AND node_type = 'hypothesis'
            ORDER BY posterior_score DESC
            LIMIT %s
            """,
            (session_id, min(top_n, 5))
        )
        candidates = cur.fetchall()
        
        if not candidates:
            return {"success": False, "error": "No candidates to analyze"}
        
        # v32 FIX: Return prompts for agent to process (no MCP sampling)
        candidate_prompts = []
        for candidate in candidates:
            seq_prompt = f"""What gaps exist between this hypothesis and the goal?

GOAL: {session["goal"]}
HYPOTHESIS: {candidate["content"]}

Return ONLY valid JSON:
{{"gaps": ["gap1", "gap2"], "revisions_needed": ["revision1"]}}

Focus on what's MISSING, not what's wrong."""
            
            candidate_prompts.append({
                "node_id": str(candidate["id"]),
                "content": candidate["content"][:100],
                "score": float(candidate["posterior_score"]) if candidate["posterior_score"] else 0.0,
                "prompt": seq_prompt
            })
        
        logger.info(f"v32: Returning {len(candidate_prompts)} prompts for sequential analysis")
        
        return {
            "success": True,
            "session_id": session_id,
            "system_prompt": "Find what's MISSING. Be adversarial. Return only valid JSON.",
            "candidate_prompts": candidate_prompts,
            "instructions": f"Process each prompt and call store_sequential_analysis(session_id='{session_id}', results='[{{\"node_id\": \"...\", \"gaps\": [...], \"revisions_needed\": [...]}}]')"
        }
        
    except Exception as e:
        logger.error(f"prepare_sequential_analysis failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def store_sequential_analysis(
    session_id: str,
    results: str  # JSON array: [{"node_id": "...", "gaps": [...], "revisions_needed": [...]}]
) -> dict[str, Any]:
    """
    v32: Store sequential analysis results from agent.
    
    Called after agent processes prompts from prepare_sequential_analysis.
    Aggregates systemic gaps and stores for finalize_session.
    
    Args:
        session_id: The reasoning session UUID
        results: JSON array of gap analysis per candidate
        
    Returns:
        Confirmation with systemic gaps detected
    """
    try:
        # Parse results
        parsed_results = json.loads(results)
        
        if not isinstance(parsed_results, list):
            return {"success": False, "error": "results must be a JSON array"}
        
        # Collect all gaps for systemic detection
        all_gaps = []
        for result in parsed_results:
            gaps = result.get("gaps", [])
            if isinstance(gaps, list):
                all_gaps.extend(gaps)
        
        # Detect systemic gaps (>50% candidates have same gap)
        systemic_gaps = []
        if all_gaps and len(parsed_results) > 1:
            from collections import Counter
            gap_counts = Counter(all_gaps)
            threshold = len(parsed_results) * 0.5
            systemic_gaps = [g for g, c in gap_counts.items() if c >= threshold]
        
        logger.info(f"v32: Stored sequential analysis - {len(parsed_results)} candidates, {len(systemic_gaps)} systemic gaps")
        
        return {
            "success": True,
            "session_id": session_id,
            "candidates_analyzed": len(parsed_results),
            "sequential_analysis": parsed_results,
            "systemic_gaps": systemic_gaps,
            "next_step": f"Call finalize_session(session_id='{session_id}')"
        }
        
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {e}"}
    except Exception as e:
        logger.error(f"store_sequential_analysis failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_reasoning_tree(session_id: str, max_depth: int = 10) -> dict[str, Any]:
    """
    Get the full reasoning tree for a session.
    
    Args:
        session_id: The reasoning session UUID
        max_depth: Maximum tree depth to retrieve
        
    Returns:
        All thought nodes ordered by path
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            """
            SELECT id, path, content, node_type, depth, prior_score, likelihood, posterior_score, created_at
            FROM thought_nodes WHERE session_id = %s AND depth <= %s ORDER BY path
            """,
            (session_id, max_depth)
        )
        
        nodes = []
        for r in cur.fetchall():
            nodes.append({
                "id": str(r["id"]), "path": r["path"], "content": r["content"][:100] + "..." if len(r["content"]) > 100 else r["content"],
                "type": r["node_type"], "depth": r["depth"],
                "posterior": float(r["posterior_score"]) if r["posterior_score"] else None
            })
        
        return {"success": True, "session_id": session_id, "nodes": nodes, "count": len(nodes)}
        
    except Exception as e:
        logger.error(f"get_reasoning_tree failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def get_best_path(session_id: str) -> dict[str, Any]:
    """
    Find the highest-scoring reasoning path through the tree.
    
    Args:
        session_id: The reasoning session UUID
        
    Returns:
        The best leaf node and its ancestor chain
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Find best leaf (node with highest posterior that has no children)
        cur.execute(
            """
            SELECT t1.id, t1.path, t1.content, t1.posterior_score, t1.depth
            FROM thought_nodes t1
            WHERE t1.session_id = %s
            AND NOT EXISTS (SELECT 1 FROM thought_nodes t2 WHERE t2.session_id = t1.session_id AND t2.path <@ t1.path AND t2.path != t1.path)
            ORDER BY t1.posterior_score DESC NULLS LAST LIMIT 1
            """,
            (session_id,)
        )
        best = cur.fetchone()
        if not best:
            return {"success": False, "error": "No thoughts in session"}
        
        # Get ancestors
        cur.execute(
            "SELECT id, path, content, posterior_score, depth FROM thought_nodes WHERE session_id = %s AND %s <@ path ORDER BY depth",
            (session_id, best["path"])
        )
        
        path_nodes = [{"id": str(r["id"]), "path": r["path"], "content": r["content"][:80], "score": float(r["posterior_score"]) if r["posterior_score"] else None} for r in cur.fetchall()]
        
        return {
            "success": True,
            "best_node_id": str(best["id"]),
            "best_score": float(best["posterior_score"]) if best["posterior_score"] else None,
            "path": path_nodes
        }
        
    except Exception as e:
        logger.error(f"get_best_path failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# Interview Tools - Smart Question Flow (Phase 2)
# =============================================================================

DEFAULT_INTERVIEW_CONFIG = {
    "max_questions": 15,
    "max_depth": 3,
    "questions_answered": 0,
    "questions_remaining": 0
}

# =============================================================================
# v20: Adaptive Depth Quality Thresholds
# =============================================================================

DEFAULT_QUALITY_THRESHOLDS = {
    "gap_score": 0.10,           # Winner must be ≥10% better than runner-up
    "critique_coverage": 0.66,   # ≥66% of top candidates must be critiqued
    "min_depth": 2,              # Must explore at least 2 levels deep
    "max_confidence_variance": 0.25,  # Variance ≤0.25 for stability
    "max_iterations": 5          # Safeguard: max expansion cycles
}


def get_interview_context(session_context: dict) -> dict:
    """Extract or initialize interview context from session."""
    if not session_context:
        session_context = {}
    
    if "interview" not in session_context:
        session_context["interview"] = {
            "config": DEFAULT_INTERVIEW_CONFIG.copy(),
            "pending_questions": [],
            "answer_history": []
        }
    return session_context["interview"]


def archive_interview_to_history(cur, session_id: str, goal: str, interview: dict):
    """
    Archive completed interview Q&A to interview_history table for self-learning.
    
    Extracts domain from goal and stores each answered question with its context.
    """
    from engine import get_embedding
    
    # Extract domain from goal (first meaningful word after common prefixes)
    goal_lower = goal.lower()
    domain = "general"
    domain_keywords = {
        "ui": "ui_design", "ux": "ui_design", "dashboard": "ui_design", "widget": "ui_design",
        "cache": "architecture", "api": "architecture", "database": "architecture", "schema": "architecture",
        "debug": "debugging", "fix": "debugging", "error": "debugging", "bug": "debugging",
        "test": "testing", "verify": "testing", "validate": "testing",
        "design": "design", "architect": "architecture", "pattern": "patterns"
    }
    for keyword, dom in domain_keywords.items():
        if keyword in goal_lower:
            domain = dom
            break
    
    pending = interview.get("pending_questions", [])
    answered_questions = [q for q in pending if q.get("answered")]
    
    for position, q in enumerate(answered_questions, 1):
        try:
            # Get embedding for semantic clustering
            embedding = get_embedding(q.get("question_text", ""))
            
            # Determine if follow-up was triggered
            follow_up_triggered = False
            for rule in q.get("follow_up_rules", []):
                if rule.get("when_answer") == q.get("answer"):
                    follow_up_triggered = True
                    break
            
            # Find answer text from choices
            answer_text = None
            answer_given = q.get("answer")
            if answer_given and "choices" in q:
                for choice in q["choices"]:
                    if choice.get("label") == answer_given:
                        answer_text = choice.get("text", "")
                        break
            
            cur.execute(
                """
                INSERT INTO interview_history 
                (session_id, question_id, question_text, question_embedding, 
                 question_category, answer_given, answer_text, position_in_flow,
                 was_skipped, follow_up_triggered, goal_domain)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    q.get("id", f"q_{position}"),
                    q.get("question_text", ""),
                    embedding,
                    q.get("category", "general"),
                    answer_given,
                    answer_text,
                    position,
                    False,  # was_skipped
                    follow_up_triggered,
                    domain
                )
            )
        except Exception as e:
            logger.warning(f"Failed to archive question {q.get('id')}: {e}")
            continue
    
    logger.info(f"Archived {len(answered_questions)} interview questions for session {session_id}")


@mcp.tool()
async def identify_gaps(session_id: str) -> dict[str, Any]:
    """
    Analyze session goal and context to identify clarifying questions needed.
    
    Generates structured questions with multiple choice answers, pros/cons,
    and follow-up rules. Questions are prioritized and limited by safety config.
    
    Args:
        session_id: The reasoning session UUID
        
    Returns:
        Interview config and generated questions, or indication that no gaps exist
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get session
        cur.execute("SELECT goal, context FROM reasoning_sessions WHERE id = %s", (session_id,))
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": f"Session {session_id} not found"}
        
        goal = session["goal"]
        context = session["context"] or {}
        interview = get_interview_context(context)
        
        # Check if interview already has questions
        if interview["pending_questions"]:
            return {
                "success": True,
                "session_id": session_id,
                "status": "interview_in_progress",
                "interview_config": interview["config"],
                "pending_count": len([q for q in interview["pending_questions"] if not q.get("answered")]),
                "message": "Interview already has pending questions. Use get_next_question to continue."
            }
        
        # v16d.2: Query historical failures for similar goals (domain-agnostic via embedding)
        historical_questions: list[dict[str, Any]] = []
        try:
            cur.execute("""
                SELECT s.goal, o.notes, o.failure_reason
                FROM outcome_records o
                JOIN reasoning_sessions s ON o.session_id = s.id
                WHERE o.outcome != 'success'
                AND (o.notes IS NOT NULL OR o.failure_reason IS NOT NULL)
                AND s.goal_embedding IS NOT NULL
                AND 1 - (s.goal_embedding <=> (SELECT goal_embedding FROM reasoning_sessions WHERE id = %s)) > 0.6
                LIMIT 3
            """, (session_id,))
            
            for row in cur.fetchall():
                reason = row["failure_reason"] or row["notes"]
                if reason:
                    historical_questions.append({
                        "id": f"hist_{len(historical_questions)+1}",
                        "question_text": f"A similar goal '{row['goal'][:50]}...' failed because: {reason[:100]}. How will you address this?",
                        "question_type": "open",
                        "priority": 5,  # High priority - show first
                        "depth": 1,
                        "depends_on": [],
                        "follow_up_rules": [],
                        "answered": False,
                        "answer": None,
                        "source": "historical_failure"
                    })
        except Exception as e:
            conn.rollback()  # Clear aborted transaction state
            logger.warning(f"v16d.2 historical query failed: {e}")
        
        # =====================================================================
        # v19: Domain Detection + Dimension-Based Questions
        # Detect goal domain, load dimensions, generate structured questions
        # =====================================================================
        domain_questions = []
        detected_domains: list[dict[str, Any]] = []
        try:
            # Query domains by embedding similarity to goal
            cur.execute("""
                SELECT id, domain_name, description,
                       1 - (embedding <=> (SELECT goal_embedding FROM reasoning_sessions WHERE id = %s)) as similarity
                FROM interview_domains
                WHERE is_active = true
                AND embedding IS NOT NULL
                ORDER BY embedding <=> (SELECT goal_embedding FROM reasoning_sessions WHERE id = %s)
                LIMIT 2
            """, (session_id, session_id))
            
            similar_domains = cur.fetchall()
            
            # Pick domains with similarity > 0.5 (or top 1 if none pass threshold)
            for row in similar_domains:
                if row["similarity"] > 0.5 or not detected_domains:
                    detected_domains.append({
                        "id": str(row["id"]),
                        "name": row["domain_name"],
                        "similarity": row["similarity"]
                    })
            
            if detected_domains:
                logger.info(f"v19: Detected domains: {[d['name'] for d in detected_domains]}")
                
                # Store detected domains in session context
                context["detected_domains"] = detected_domains
                
                # Load dimensions for detected domains, ordered by priority
                # Use string IDs - PostgreSQL will cast them to UUIDs
                domain_ids = [d["id"] for d in detected_domains]
                cur.execute("""
                    SELECT dim.id, dim.dimension_name, dim.description, dim.is_required, dim.priority,
                           d.domain_name,
                           q.question_template, q.question_type, q.choices
                    FROM interview_dimensions dim
                    JOIN interview_domains d ON dim.domain_id = d.id
                    LEFT JOIN interview_questions q ON q.dimension_id = dim.id AND q.is_default = true
                    WHERE dim.domain_id = ANY(%s::uuid[])
                    ORDER BY dim.priority ASC
                """, (domain_ids,))
                
                dimensions = cur.fetchall()
                
                # Track which dimensions we're asking about
                dimension_coverage = {}
                
                for dim in dimensions:
                    dim_id = str(dim["id"])
                    dimension_coverage[dim_id] = {
                        "name": dim["dimension_name"],
                        "domain": dim["domain_name"],
                        "is_required": dim["is_required"],
                        "covered": False
                    }
                    
                    # Generate question if template exists
                    if dim["question_template"]:
                        choices = dim["choices"] if dim["choices"] else []
                        domain_questions.append({
                            "id": f"dim_{dim['dimension_name']}",
                            "dimension_id": dim_id,
                            "question_text": dim["question_template"],
                            "question_type": dim["question_type"],
                            "choices": choices,
                            "priority": dim["priority"],
                            "depth": 1,
                            "depends_on": [],
                            "follow_up_rules": [],
                            "answered": False,
                            "answer": None,
                            "source": "domain_dimension",
                            "domain": dim["domain_name"],
                            "dimension": dim["dimension_name"],
                            "is_required": dim["is_required"]
                        })
                
                # Store dimension coverage in context for tracking
                context["dimension_coverage"] = dimension_coverage
                
                logger.info(f"v19: Generated {len(domain_questions)} dimension questions for {len(detected_domains)} domain(s)")
        
        except Exception as e:
            conn.rollback()  # Clear aborted transaction state
            logger.warning(f"v19 domain detection failed: {e}")
        
        # Analyze goal to generate questions
        # v16d.1: LLM-generated goal-derived questions
        # v17c: Focus on business context only, not code quality
        # v18: Smart LLM Gating - return [] for specific goals
        # v19: Skip LLM questions if domain detection succeeded
        # v21: Hidden Context Question Design - laddering, trade-offs, consequence-framing
        goal_questions: list[dict[str, Any]] = []
        try:
            goal_prompt = f"""Analyze this goal and generate clarifying questions if needed:

GOAL: {goal}

DECISION RULES:
1. If goal is SPECIFIC (names files/functions, has clear constraints) → return []
2. If goal is AMBIGUOUS (multiple valid interpretations) → return 1-2 questions max
3. If goal is OPEN-ENDED (design/architecture/planning) → return 2-3 questions max

EXAMPLES OF SPECIFIC GOALS (return []):
- "Fix bug in auth.py line 45"
- "Add logging to UserService.create_user()"
- "Refactor parse_terminal_output to handle '0 failed'"

=== HIDDEN CONTEXT QUESTION DESIGN (v21) ===
Use psychology-based techniques to extract more information per question:

1. CONSEQUENCE FRAMING (Laddering):
   - Frame choices as CONSEQUENCES, not features
   - BAD: "Which database?" → "PostgreSQL / MySQL / MongoDB"
   - GOOD: "When data integrity fails, what's the worst outcome?"
          → "Users lose trust / Debugging becomes hard / Schema breaks"

2. TRADE-OFF BUNDLES (Conjoint):
   - Force implicit priority decisions
   - BAD: "Is performance important?" (everyone says yes)
   - GOOD: "Which trade-off can you live with?"
          → "50ms slower but consistent / Fast but occasional stale reads"

3. SCENARIO COMPLETION (Projective):
   - Present a situation and ask what happens next
   - GOOD: "A critical bug appears at 2am before launch. When you investigate, you find..."
          → Options reveal risk tolerance, debugging philosophy

4. COGNITIVE LOAD (Keep it simple):
   - Options < 15 words each
   - Use concrete scenarios, not abstractions
   - Gut reaction = true preference

WHAT EACH CHOICE REVEALS:
- Every option should reveal an underlying VALUE (risk tolerance, priority, philosophy)
- Include a "hidden_value" field describing what the choice reveals
- Example: {{"label": "A", "description": "Fast but complex", "hidden_value": "performance_priority"}}

Return format:
[{{"question_text": "...", "choices": [{{"label": "A", "description": "...", "hidden_value": "...", "pros": ["..."], "cons": ["..."]}}]}}]

Return [] if goal is specific enough. Return ONLY the JSON array."""
            
            # v32 FIX: MCP sampling not supported by clients
            # Return prompt for agent to process - agent stores questions via store_gaps_questions
            llm_question_prompt = {
                "prompt": goal_prompt,
                "system": "You are a structured question generator. Return only valid JSON arrays.",
                "max_questions": 3
            }
            logger.info(f"v32: Returning question generation prompt for agent")
            
            # Don't wait for LLM - just return the prompt for agent to process
            # Agent will call store_gaps_questions after processing
        except Exception as e:
            logger.warning(f"v16d.1 goal question prompt setup failed: {e}")
        
        # v19: Prefer domain questions over LLM-generated questions
        # Priority: 1) domain dimensions, 2) goal-derived LLM, 3) catch-all
        if domain_questions:
            # Domain detection succeeded - use dimension-based questions
            questions = domain_questions
            logger.info(f"v19: Using {len(domain_questions)} domain-based questions")
        elif goal_questions:
            # Fallback to LLM-generated questions
            questions = goal_questions
        else:
            questions = []
        
        # v18: If no questions at all and no historical, offer catch-all
        if not questions and not historical_questions:
            goal_preview = goal[:50] + "..." if len(goal) > 50 else goal
            questions = [{
                "id": "q_catchall",
                "question_text": f"Is there anything specific about '{goal_preview}' I should know before proceeding?",
                "question_type": "open",
                "choices": [],
                "optional": True,
                "priority": 100,
                "depth": 1,
                "depends_on": [],
                "follow_up_rules": [],
                "answered": False,
                "answer": None,
                "source": "catchall"
            }]
            logger.info("v18: Using catch-all question")
        
        # v16d.2: Prepend historical questions (higher priority)
        questions = historical_questions + questions
        
        # Update interview context
        interview["pending_questions"] = questions
        interview["config"]["questions_remaining"] = len(questions)
        
        # Save to session
        cur.execute(
            "UPDATE reasoning_sessions SET context = %s, updated_at = NOW() WHERE id = %s",
            (json.dumps(context), session_id)
        )
        conn.commit()
        
        return {
            "success": True,
            "session_id": session_id,
            "interview_config": interview["config"],
            "questions_generated": len(questions),
            "detected_domains": [d["name"] for d in detected_domains] if detected_domains else None,
            "message": "Interview questions generated. Call get_next_question to start."
        }
        
    except Exception as e:
        logger.error(f"identify_gaps failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def get_next_question(session_id: str) -> dict[str, Any]:
    """
    Get the next unanswered question from the interview queue.
    
    Returns ONE question at a time with formatted choices including pros/cons.
    Includes progress indicator showing current position.
    
    Args:
        session_id: The reasoning session UUID
        
    Returns:
        Single question with choices, or indication that interview is complete
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT context FROM reasoning_sessions WHERE id = %s", (session_id,))
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": "Session not found"}
        
        context = session["context"] or {}
        interview = get_interview_context(context)
        
        # =====================================================================
        # v23: Check if early termination suggested
        # =====================================================================
        if interview.get("early_termination_suggested") and not interview.get("early_exit_declined"):
            remaining = len([q for q in interview.get("pending_questions", []) if not q.get("answered")])
            return {
                "success": True,
                "session_id": session_id,
                "early_exit_offered": True,
                "questions_remaining": remaining,
                "reason": interview.get("early_termination_reason", "diminishing_returns"),
                "message": "We've learned enough about your preferences. Continue (submit any answer) or proceed to prepare_expansion."
            }
        
        # Find first unanswered, unlocked question
        pending = interview.get("pending_questions", [])
        answered_ids = {q["id"] for q in pending if q.get("answered")}
        
        for q in sorted(pending, key=lambda x: x.get("priority", 999)):
            if q.get("answered"):
                continue

            # Check dependencies
            deps = q.get("depends_on", [])
            if all(d in answered_ids for d in deps):
                # This question is ready
                total_questions = len(pending)
                answered_count = interview["config"].get("questions_answered", 0)
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "progress": f"Question {answered_count + 1} of ~{total_questions}",
                    "question": {
                        "id": q["id"],
                        "text": q["question_text"],
                        "type": q.get("question_type", "single_choice"),
                        "choices": q["choices"]
                    }
                }
        
        # No more questions
        return {
            "success": True,
            "session_id": session_id,
            "interview_complete": True,
            "message": "All questions answered. Ready for prepare_expansion."
        }
        
    except Exception as e:
        logger.error(f"get_next_question failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def submit_answer(session_id: str, question_id: str, answer: str) -> dict[str, Any]:
    """
    Submit an answer to an interview question.
    
    Updates the question as answered, stores in history, triggers context propagation
    including potential follow-up question injection (respecting depth/count limits).
    
    Args:
        session_id: The reasoning session UUID
        question_id: The question ID being answered
        answer: The selected answer (e.g., "A", "B", "C")
        
    Returns:
        Confirmation with any injected follow-up questions
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT context FROM reasoning_sessions WHERE id = %s", (session_id,))
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": "Session not found"}
        
        context = session["context"] or {}
        interview = get_interview_context(context)
        pending = interview.get("pending_questions", [])
        config = interview["config"]
        
        # v23: If user continues after early exit offer, mark as declined
        if interview.get("early_termination_suggested"):
            interview["early_exit_declined"] = True
            logger.info("v23: User declined early exit, continuing interview")

        
        # Find the question
        question = None
        for q in pending:
            if q["id"] == question_id:
                question = q
                break
        
        if not question:
            return {"success": False, "error": f"Question {question_id} not found"}
        
        if question.get("answered"):
            return {"success": False, "error": "Question already answered"}
        
        # Mark answered
        question["answered"] = True
        question["answer"] = answer
        config["questions_answered"] = config.get("questions_answered", 0) + 1
        
        # v21: Extract hidden_value from selected choice
        hidden_value = None
        answer_description = None
        for choice in question.get("choices", []):
            if choice.get("label") == answer:
                hidden_value = choice.get("hidden_value")
                answer_description = choice.get("description")
                break
        
        # Store in history (v21: now includes hidden_value for latent trait inference)
        interview["answer_history"].append({
            "question_id": question_id,
            "question_text": question["question_text"],
            "answer": answer,
            "answer_description": answer_description,
            "hidden_value": hidden_value,  # v21: reveals underlying values/priorities
            "timestamp": str(uuid.uuid4())[:8]  # Simple timestamp proxy
        })
        
        # =====================================================================
        # v19: Track dimension coverage for domain-based questions
        # =====================================================================
        dimension_covered = None
        if question.get("source") == "domain_dimension" and question.get("dimension_id"):
            dim_id = question["dimension_id"]
            dimension_coverage = context.get("dimension_coverage", {})
            
            if dim_id in dimension_coverage:
                dimension_coverage[dim_id]["covered"] = True
                dimension_covered = dimension_coverage[dim_id]["name"]
                logger.info(f"v19: Dimension '{dimension_covered}' covered")
            
            # Store answer in interview_answers for conflict detection
            try:
                # Find the answer text from choices
                answer_text = answer
                for choice in question.get("choices", []):
                    if choice.get("label") == answer:
                        answer_text = choice.get("description", answer)
                        break
                
                cur.execute("""
                    INSERT INTO interview_answers (session_id, dimension_id, question_id, answer_label, answer_text)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (session_id, dimension_id) DO UPDATE SET
                        answer_label = EXCLUDED.answer_label,
                        answer_text = EXCLUDED.answer_text,
                        created_at = NOW()
                """, (session_id, dim_id, question.get("id"), answer, answer_text))
                
                logger.info(f"v19: Persisted answer for dimension {dim_id}")
            except Exception as e:
                conn.rollback()  # Clear aborted transaction state before continuing
                logger.warning(f"v19: Failed to persist answer: {e}")
        
        # Check for follow-up injection
        injected = []
        follow_up_rules = question.get("follow_up_rules", [])
        for rule in follow_up_rules:
            if rule.get("when_answer") == answer:
                new_q = rule["inject"].copy()
                
                # Safety checks
                if new_q.get("depth", 1) > config["max_depth"]:
                    logger.warning(f"Skipping injection: depth {new_q['depth']} exceeds max {config['max_depth']}")
                    continue
                    
                if len(pending) >= config["max_questions"]:
                    logger.warning(f"Skipping injection: max questions {config['max_questions']} reached")
                    continue
                
                # Inject the question
                new_q["answered"] = False
                new_q["answer"] = None
                pending.append(new_q)
                injected.append(new_q["id"])
        
        # Update remaining count
        config["questions_remaining"] = len([q for q in pending if not q.get("answered")])
        
        # =====================================================================
        # v23: Track evidence delta for plateau detection
        # =====================================================================
        evidence_history = interview.get("evidence_history", [])
        
        # Calculate current evidence (count of hidden_values detected)
        previous_total = sum(evidence_history) if evidence_history else 0
        current_hidden_values: dict[str, int] = {}
        for entry in interview.get("answer_history", []):
            hv = entry.get("hidden_value")
            if hv:
                current_hidden_values[hv] = current_hidden_values.get(hv, 0) + 1
        current_total = sum(current_hidden_values.values())
        
        # Delta = how many new hidden_value "hits" this answer contributed
        evidence_delta = 1 if hidden_value else 0  # Simple: count if this answer had a hidden_value
        evidence_history.append(evidence_delta)
        interview["evidence_history"] = evidence_history[-5:]  # Keep last 5
        
        # Check for plateau (last 3 answers with no new trait info)
        if len(evidence_history) >= 3:
            recent_gain = sum(evidence_history[-3:])
            if recent_gain == 0 and config.get("questions_answered", 0) >= 3:
                interview["early_termination_suggested"] = True
                interview["early_termination_reason"] = "diminishing_returns"
                logger.info("v23: Early termination suggested - no trait evidence in last 3 answers")
        
        # Save to session
        cur.execute(
            "UPDATE reasoning_sessions SET context = %s, updated_at = NOW() WHERE id = %s",
            (json.dumps(context), session_id)
        )

        conn.commit()
        
        return {
            "success": True,
            "session_id": session_id,
            "question_id": question_id,
            "answer_recorded": answer,
            "dimension_covered": dimension_covered,  # v19
            "questions_remaining": config["questions_remaining"],
            "follow_ups_injected": injected if injected else None,
            "message": f"Answer recorded. {config['questions_remaining']} questions remaining."
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"submit_answer failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def check_interview_complete(session_id: str) -> dict[str, Any]:
    """
    Check if the interview has gathered enough context to proceed.
    
    Args:
        session_id: The reasoning session UUID
        
    Returns:
        Whether interview is complete and summary of gathered context
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT goal, context FROM reasoning_sessions WHERE id = %s", (session_id,))
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": "Session not found"}
        
        goal = session["goal"]
        context = session["context"] or {}
        interview = get_interview_context(context)
        pending = interview.get("pending_questions", [])
        
        unanswered = [q for q in pending if not q.get("answered")]
        answered = [q for q in pending if q.get("answered")]
        
        is_complete = len(unanswered) == 0 and len(answered) > 0
        
        # Build context summary from answers
        context_summary = {}
        for q in answered:
            # Find the selected choice details
            selected = None
            for c in q.get("choices", []):
                if c["label"] == q.get("answer"):
                    selected = c["description"]
                    break
            context_summary[q["id"]] = {
                "question": q["question_text"],
                "answer": q.get("answer"),
                "selected": selected
            }
        
        # =====================================================================
        # v21 Phase 2: Latent Trait Inference from Hidden Values
        # Aggregates hidden_values from answer_history and infers user traits
        # =====================================================================
        latent_traits: list[dict[str, Any]] = []
        hidden_value_counts: dict[str, int] = {}
        
        if is_complete:
            # Aggregate hidden_values from answer history
            answer_history = interview.get("answer_history", [])
            for entry in answer_history:
                hv = entry.get("hidden_value")
                if hv:
                    hidden_value_counts[hv] = hidden_value_counts.get(hv, 0) + 1
            
            # v35: Use extracted pure helper for trait inference
            latent_traits = _infer_traits_from_hidden_values(hidden_value_counts)
            
            # =================================================================
            # v22 Feature 3: Hybrid Inference - Semantic Fallback
            # For answers without hidden_value or unmatched patterns, use
            # semantic similarity against trait_exemplars
            # =================================================================
            unmatched_descriptions = []
            for entry in answer_history:
                hv = entry.get("hidden_value")
                desc = entry.get("answer_description")
                
                # If no hidden_value OR hidden_value didn't match any rule pattern
                if desc and (not hv or not any(p in (hv or "").upper() for p, _, _, _ in TRAIT_RULES)):
                    unmatched_descriptions.append(desc)
            
            semantic_matches = []
            if unmatched_descriptions:
                try:
                    # Check if trait_exemplars table has embeddings
                    cur.execute("SELECT COUNT(*) FROM trait_exemplars WHERE embedding IS NOT NULL")
                    exemplar_count = cur.fetchone()[0]
                    
                    if exemplar_count > 0:
                        from sentence_transformers import SentenceTransformer
                        model = SentenceTransformer('all-mpnet-base-v2')
                        
                        for desc_text in unmatched_descriptions[:5]:  # Limit to 5 for performance
                            desc_embedding = model.encode(desc_text).tolist()
                            
                            cur.execute("""
                                SELECT trait_name, 1 - (embedding <=> %s::vector) as similarity
                                FROM trait_exemplars
                                WHERE embedding IS NOT NULL
                                ORDER BY embedding <=> %s::vector
                                LIMIT 1
                            """, (desc_embedding, desc_embedding))
                            
                            result = cur.fetchone()
                            if result and result["similarity"] > 0.65:  # Threshold for semantic match
                                semantic_matches.append({
                                    "trait": result["trait_name"],
                                    "similarity": round(float(result["similarity"]), 3),
                                    "source_text": desc_text[:50]
                                })
                                
                                # Add to trait counts with reduced confidence
                                trait_name = result["trait_name"]
                                existing = next((t for t in latent_traits if t["trait"] == trait_name), None)
                                if existing:
                                    existing["evidence_count"] = existing.get("evidence_count", 1) + 1
                                    existing["semantic_boost"] = True
                                else:
                                    latent_traits.append({
                                        "trait": trait_name,
                                        "confidence": 0.6,  # Lower confidence for semantic matches
                                        "evidence_count": 1,
                                        "source": "semantic"
                                    })
                                
                                logger.info(f"v22 Semantic match: '{desc_text[:30]}...' → {trait_name} (sim: {result['similarity']:.2f})")
                                
                except ImportError:
                    logger.debug("v22: sentence-transformers not available for hybrid inference")
                except Exception as e:
                    logger.warning(f"v22 Hybrid inference failed: {e}")
            
            # Store in session context for downstream use
            if latent_traits or hidden_value_counts:
                context["latent_traits"] = latent_traits
                context["hidden_value_counts"] = hidden_value_counts
                logger.info(f"v21: Inferred {len(latent_traits)} latent traits from {sum(hidden_value_counts.values())} hidden_values")
        
        # Archive interview to history for self-learning
        if is_complete and not interview.get("archived"):
            try:
                archive_interview_to_history(cur, session_id, goal, interview)
                interview["archived"] = True
                cur.execute(
                    "UPDATE reasoning_sessions SET context = %s WHERE id = %s",
                    (json.dumps(context), session_id)
                )
                conn.commit()
            except Exception as e:
                logger.warning(f"Failed to archive interview: {e}")
                conn.rollback()
        
        return {
            "success": True,
            "session_id": session_id,
            "is_complete": is_complete,
            "questions_answered": len(answered),
            "questions_remaining": len(unanswered),
            "context_summary": context_summary if is_complete else None,
            "latent_traits": latent_traits if latent_traits else None,  # v21 Phase 2
            "hidden_value_counts": hidden_value_counts if hidden_value_counts else None,  # v21 Phase 2
            "archived_for_learning": interview.get("archived", False),
            "message": "Ready for prepare_expansion" if is_complete else f"{len(unanswered)} questions remaining"
        }
        
    except Exception as e:
        logger.error(f"check_interview_complete failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# Finalization Tool - Auto-Critique & Recommendation (Phase 3)
# =============================================================================

HEURISTIC_PENALTIES = {
    "unchallenged": 0.10,      # Never critiqued
    "shallow_alternatives": 0.05,  # <2 siblings at same level
    "monoculture": 0.05,       # v13b: All siblings match same law
}

DEPTH_BONUS_PER_LEVEL = 0.02  # Reward deeper refinement

# =============================================================================
# v11a: UCT-based selection for close decisions
# =============================================================================
UCT_EXPLORATION_C = 1.4  # Standard exploration constant
UCT_THRESHOLD = 0.05     # Apply UCT when gap < threshold

# =============================================================================
# v11b: Law-grounded rollout configuration  
# =============================================================================
ROLLOUT_WEIGHT = 0.2     # Blend: final = (1-weight)*posterior + weight*rollout


# =============================================================================
# v35: Latent Trait Inference Constants and Helper
# =============================================================================

TRAIT_RULES = [
    # (hidden_value_pattern, min_count, trait_name, confidence)
    ("SAFETY", 2, "RISK_AVERSE", 0.8),
    ("SIMPLICITY", 2, "MINIMALIST", 0.75),
    ("POWER_USER", 2, "CONTROL_ORIENTED", 0.8),
    ("AI_", 2, "AUTOMATION_TRUSTING", 0.7),
    ("BEGINNER", 2, "ACCESSIBILITY_FOCUSED", 0.75),
    ("BALANCE", 3, "PRAGMATIST", 0.7),
    ("EXPLICIT", 2, "EXPLICIT_CONTROL", 0.8),
    ("PERFORMANCE", 2, "SPEED_FOCUSED", 0.75),
    ("ERROR_PREVENTION", 2, "SAFETY_CONSCIOUS", 0.8),
    ("FREEDOM", 2, "AUTONOMY_FOCUSED", 0.75),
]


def _infer_traits_from_hidden_values(
    hidden_value_counts: dict[str, int]
) -> list[dict[str, Any]]:
    """
    v35: Pure function to infer latent traits from hidden value patterns.
    
    Args:
        hidden_value_counts: Dict of hidden_value -> occurrence count
    
    Returns:
        List of trait dicts with trait, confidence, evidence_count
    """
    latent_traits = []
    
    for pattern, min_count, trait_name, confidence in TRAIT_RULES:
        matching_count = sum(
            count for hv, count in hidden_value_counts.items() 
            if pattern in hv.upper()
        )
        if matching_count >= min_count:
            latent_traits.append({
                "trait": trait_name,
                "confidence": confidence,
                "evidence_count": matching_count
            })
    
    return latent_traits


# =============================================================================
# Pure Utility: Heuristic Penalty Application (v35 extraction)
# =============================================================================

def _apply_heuristic_penalties(
    node: dict[str, Any],
    sibling_counts: dict[str, int],
    law_diversity: dict[str, tuple[int, int]]
) -> tuple[float, list[str]]:
    """
    Pure function: Apply penalties to candidate score.
    
    Args:
        node: Candidate node from DB (needs posterior_score, likelihood, depth, path)
        sibling_counts: Dict of parent_path -> count of siblings at that level
        law_diversity: Dict of parent_path -> (unique_laws, total_siblings)
    
    Returns:
        Tuple of (adjusted_score, list_of_penalties_applied)
    """
    original_score = float(node["posterior_score"])
    adjusted_score = original_score
    penalties: list[str] = []
    
    # Unchallenged penalty: if likelihood is suspiciously round
    likelihood = float(node["likelihood"])
    if str(likelihood).endswith(('0', '5')) and likelihood in [0.8, 0.85, 0.9, 0.95, 0.88, 0.92]:
        adjusted_score -= HEURISTIC_PENALTIES["unchallenged"]
        penalties.append("unchallenged_penalty")
    
    # Depth bonus (deeper = more refined)
    depth = int(node["depth"])
    if depth > 2:
        bonus = (depth - 2) * DEPTH_BONUS_PER_LEVEL
        adjusted_score += bonus
        if bonus > 0:
            penalties.append(f"depth_bonus_+{bonus:.2f}")
    
    # Shallow alternatives penalty: if <2 sibling hypotheses at same level
    node_path = str(node["path"])
    parent_path = ".".join(node_path.split(".")[:-1]) if "." in node_path else ""
    sibling_count = sibling_counts.get(parent_path, 1)
    if sibling_count < 2:
        adjusted_score -= HEURISTIC_PENALTIES["shallow_alternatives"]
        penalties.append("shallow_alternatives_penalty")
    
    # v13b: Monoculture penalty - if all siblings match same law, penalize
    if parent_path in law_diversity:
        unique_laws, total_siblings = law_diversity[parent_path]
        if total_siblings >= 2 and unique_laws == 1:
            adjusted_score -= HEURISTIC_PENALTIES["monoculture"]
            penalties.append("monoculture_penalty")
    
    # Ensure score stays in valid range
    adjusted_score = max(0.1, min(1.0, adjusted_score))
    
    return adjusted_score, penalties


def _get_outcome_multiplier(outcome: str) -> float:
    """
    v35: Pure function for outcome-based weighting.
    
    Returns multiplier for trait persistence based on outcome.
    """
    if outcome == "success":
        return 1.2
    elif outcome == "failure":
        return 0.8
    return 1.0


def _compute_critique_accuracy(
    node_path: str,
    winning_path: str,
    outcome: str
) -> Optional[bool]:
    """
    v35: Pure function to determine if critique was accurate.
    
    Args:
        node_path: Path of the critiqued node
        winning_path: Path of the winning hypothesis
        outcome: Session outcome ('success', 'partial', 'failure')
    
    Returns:
        True if critique was accurate, False if not, None if indeterminate
    """
    is_winner = str(node_path).startswith(str(winning_path)) or str(winning_path).startswith(str(node_path))
    
    if is_winner:
        # If winner was critiqued and outcome is failure, critique was accurate
        return outcome == 'failure'
    else:
        # If non-winner was critiqued and outcome is success, critique may have been accurate
        return outcome == 'success'


def _compute_law_effective_weight(
    base_weight: float,
    selection_count: int,
    success_count: int,
    negation_asymmetry: bool
) -> tuple[float, float]:
    """
    v35: Pure function for Thompson sampling law weight.
    
    Args:
        base_weight: Scientific weight from law
        selection_count: Times this law was selected
        success_count: Times it led to success
        negation_asymmetry: If hypothesis negation mismatches law
    
    Returns:
        Tuple of (effective_weight, thompson_sample)
    """
    import random
    
    alpha = success_count + 1
    beta_param = selection_count - success_count + 1
    thompson_sample = random.betavariate(alpha, beta_param)
    
    effective_weight = 0.7 * base_weight + 0.3 * thompson_sample
    negation_penalty = 0.15 if negation_asymmetry else 0.0
    effective_weight = max(0.1, effective_weight - negation_penalty)
    
    return effective_weight, thompson_sample


def _compute_ensemble_prior(
    matching_laws: list[dict],
    hypothesis_text: str
) -> tuple[float, list[int], str | None]:
    """
    v35: Pure function to compute ensemble prior from multiple laws.
    
    Args:
        matching_laws: List of law dicts with similarity, scientific_weight, etc
        hypothesis_text: The hypothesis text (for negation check)
    
    Returns:
        Tuple of (prior, supporting_law_ids, primary_law_name)
    """
    if not matching_laws:
        return 0.5, [], None
    
    total_weighted = 0.0
    total_similarity = 0.0
    supporting_law_ids = []
    law_names = []
    
    for law in matching_laws:
        # Negation detection
        hyp_negations = detect_negation(hypothesis_text)
        law_negations = detect_negation(law["definition"]) if law.get("definition") else set()
        negation_asymmetry = bool(hyp_negations) != bool(law_negations)
        
        # Compute effective weight
        effective_weight, _ = _compute_law_effective_weight(
            float(law["scientific_weight"]),
            law.get("selection_count", 0) or 0,
            law.get("success_count", 0) or 0,
            negation_asymmetry
        )
        
        similarity = float(law["similarity"])
        total_weighted += effective_weight * similarity
        total_similarity += similarity
        
        supporting_law_ids.append(law["id"])
        law_names.append(law["law_name"])
    
    # Ensemble prior = Σ(weight × similarity) / Σ(similarity)
    prior = (total_weighted / total_similarity * 0.6) + (matching_laws[0]["similarity"] * 0.4)
    prior = max(0.1, min(0.95, prior))
    
    return prior, supporting_law_ids, law_names[0] if law_names else None


# v35b: Pure helper for tag generation
DOMAIN_PATTERNS: list[tuple[str, list[str]]] = [
    ("database", ["db", "schema", "table", "sql", "postgres", "column"]),
    ("api", ["api", "endpoint", "route", "request", "response"]),
    ("backend", ["server", "backend", "service", "logic"]),
    ("frontend", ["ui", "ux", "frontend", "component", "page"]),
    ("testing", ["test", "pytest", "verify", "validation"]),
    ("bugfix", ["bug", "fix", "error", "failure"]),
    ("feature", ["feature", "implement", "add", "new"]),
    ("refactor", ["refactor", "cleanup", "improve", "optimize"]),
]

def _generate_suggested_tags(goal: str, winning_content: str, max_tags: int = 5) -> list[str]:
    """Generate suggested tags based on goal and winning content. Pure function."""
    combined_text = f"{goal} {winning_content}".lower()
    tags = []
    for tag, keywords in DOMAIN_PATTERNS:
        if any(kw in combined_text for kw in keywords):
            tags.append(tag)
    return tags[:max_tags]


# v35b: Pure helper for decision quality computation
def _compute_decision_quality(
    gap: float,
    rec_conf: float,
    run_conf: float,
    uct_applied: bool = False
) -> tuple[str, str]:
    """
    Compute decision quality and gap analysis message. Pure function.
    Returns (decision_quality, gap_analysis).
    """
    if gap < 0.03:
        decision_quality = "low"
        gap_analysis = f"Very close decision (gap: {gap:.3f}). Consider both options."
    elif gap < 0.10:
        decision_quality = "medium"
        gap_analysis = f"Moderate confidence (gap: {gap:.3f}). Winner is better but runner-up has merit."
    else:
        decision_quality = "high"
        gap_analysis = f"Clear winner (gap: {gap:.3f}). High confidence in recommendation."
    
    # v14a.1: Confidence-weighted gap
    weighted_gap = gap * min(rec_conf, run_conf)
    
    if decision_quality == "high" and weighted_gap < 0.05:
        decision_quality = "medium"
        gap_analysis += f" [v14a.1: Downgraded - low confidence (wgap: {weighted_gap:.3f})]"
    elif decision_quality == "medium" and weighted_gap < 0.02:
        decision_quality = "low"
        gap_analysis += f" [v14a.1: Downgraded - low confidence (wgap: {weighted_gap:.3f})]"
    
    if uct_applied:
        gap_analysis += " [v11a: UCT exploration applied]"
    
    return decision_quality, gap_analysis


# v35b: Pure helper for UCT tiebreaking
def _apply_uct_tiebreaking(
    rec_score: float,
    rec_depth: int,
    run_score: float,
    run_depth: int
) -> tuple[bool, bool]:
    """
    Apply UCT exploration bonus for close decisions. Pure function.
    Returns (should_swap, uct_was_applied).
    """
    import math
    
    gap = rec_score - run_score
    if gap >= UCT_THRESHOLD:
        return False, False
    
    rec_visits = max(1, rec_depth)
    run_visits = max(1, run_depth)
    total_visits = rec_visits + run_visits
    
    rec_uct = rec_score + UCT_EXPLORATION_C * math.sqrt(math.log(total_visits) / rec_visits)
    run_uct = run_score + UCT_EXPLORATION_C * math.sqrt(math.log(total_visits) / run_visits)
    
    return run_uct > rec_uct, True


# v35b: Pure helper to build processed candidate dict
def _build_processed_candidate(
    node: dict,
    adjusted_score: float,
    penalties: list[str],
    rollout_score: float
) -> dict[str, Any]:
    """Build a processed candidate dict for finalize_session. Pure function."""
    original_score = float(node["posterior_score"])
    depth = int(node["depth"])
    final_score = (1 - ROLLOUT_WEIGHT) * adjusted_score + ROLLOUT_WEIGHT * rollout_score
    
    return {
        "node_id": str(node["id"]),
        "path": node["path"],
        "content": node["content"],
        "depth": depth,
        "prior_score": round(float(node["prior_score"]), 4),
        "likelihood": round(float(node["likelihood"]), 4),
        "confidence": round(float(node["prior_score"]) * float(node["likelihood"]), 4),
        "original_score": round(original_score, 4),
        "adjusted_score": round(adjusted_score, 4),
        "rollout_score": round(rollout_score, 4),
        "final_score": round(final_score, 4),
        "penalties_applied": penalties
    }


# =============================================================================
# v20: Adaptive Depth Quality Metrics
# =============================================================================

def compute_quality_metrics(
    cur,
    session_id: str,
    candidates: list,
    gap: float,
    thresholds: Optional[dict] = None
) -> dict:
    """
    Compute 4 quality signals to determine if reasoning is sufficient.
    
    Returns quality_sufficient flag, metrics breakdown, and suggestions.
    """
    import statistics
    
    thresholds = thresholds or DEFAULT_QUALITY_THRESHOLDS
    
    # 1. Gap score (already computed by caller)
    gap_ok = gap >= thresholds["gap_score"]
    
    # 2. Critique coverage - % of top candidates that have been critiqued
    # Critiqued nodes have likelihood reduced below their original confidence
    critiqued_count = 0
    uncritiqued_nodes = []
    for c in candidates[:3]:  # Check top 3
        node_id = c.get("node_id") or c.get("id")
        likelihood = c.get("likelihood", 0.5)
        # A node is considered critiqued if likelihood < initial confidence
        # (store_critique reduces likelihood, initial is typically 0.7-0.9)
        if likelihood < 0.8:  # Threshold: critiqued if reduced
            critiqued_count += 1
        else:
            if node_id:
                uncritiqued_nodes.append(str(node_id))
    
    critique_coverage = critiqued_count / max(len(candidates[:3]), 1)
    critique_ok = critique_coverage >= thresholds["critique_coverage"]
    
    # 3. Tree depth - maximum depth reached
    cur.execute("""
        SELECT MAX(depth) as max_depth FROM thought_nodes WHERE session_id = %s
    """, (session_id,))
    depth_result = cur.fetchone()
    max_depth = depth_result["max_depth"] if depth_result and depth_result["max_depth"] else 0
    depth_ok = max_depth >= thresholds["min_depth"]
    
    # 4. Confidence variance - stability of scores
    confidences = [c.get("likelihood", 0.5) for c in candidates]
    if len(confidences) > 1:
        variance = statistics.variance(confidences)
    else:
        variance = 0
    variance_ok = variance <= thresholds["max_confidence_variance"]
    
    # Compute overall sufficiency
    quality_sufficient = gap_ok and critique_ok and depth_ok and variance_ok
    
    # Generate suggestions for improvement
    suggestions = []
    if not critique_ok and uncritiqued_nodes:
        suggestions.append({
            "action": "critique",
            "node_id": uncritiqued_nodes[0],
            "reason": f"Top candidate unchallenged ({critiqued_count}/{len(candidates[:3])} critiqued)"
        })
    if not depth_ok:
        best_node_id = candidates[0].get("node_id") or candidates[0].get("id") if candidates else None
        if best_node_id:
            suggestions.append({
                "action": "expand",
                "node_id": str(best_node_id),
                "reason": f"Tree too shallow (depth {max_depth}, need ≥{thresholds['min_depth']})"
            })
    if not gap_ok:
        suggestions.append({
            "action": "expand_alternatives",
            "node_id": None,
            "reason": f"Gap too small ({gap:.3f}, need ≥{thresholds['gap_score']}). Explore more options."
        })
    if not variance_ok:
        suggestions.append({
            "action": "refine_confidences",
            "node_id": None,
            "reason": f"High uncertainty (variance {variance:.3f}). Add evidence or critique."
        })
    
    return {
        "sufficient": quality_sufficient,
        "metrics": {
            "gap_score": round(gap, 4),
            "gap_ok": gap_ok,
            "critique_coverage": round(critique_coverage, 2),
            "critique_ok": critique_ok,
            "tree_depth": max_depth,
            "depth_ok": depth_ok,
            "confidence_variance": round(variance, 4),
            "variance_ok": variance_ok
        },
        "suggestions": suggestions,
        "thresholds": thresholds
    }


@mcp.tool()
async def finalize_session(
    session_id: str,
    top_n: int = 3,
    deep_critique: bool = False,
    terminal_output: Optional[str] = None,  # v17b: RLVR auto-record
    # v31b: Exhaustive gap check (sequential-thinking style)
    exhaustive_check: bool = True,
    # v31d: Quality thresholds
    min_score_threshold: float = 0.9,
    min_gap_threshold: float = 0.1,
    # v33: Quality gate enforcement (opt-out, not opt-in)
    skip_quality_gate: bool = False
) -> dict[str, Any]:
    """
    Finalize a reasoning session by auto-critiquing top hypotheses.
    
    Applies heuristic penalties (unchallenged, shallow tree), compares
    top candidates, and returns a final recommendation with confidence.
    
    Args:
        session_id: The reasoning session UUID
        top_n: Number of top candidates to consider (default: 3)
        deep_critique: If True, returns critique requests for LLM
        terminal_output: v17b - Raw terminal output to auto-parse and record outcome
        exhaustive_check: v31b - Run layer-by-layer gap analysis on recommendation
        min_score_threshold: v31d - Minimum score required (default: 0.9)
        min_gap_threshold: v31d - Minimum gap between top candidates (default: 0.1)
        skip_quality_gate: v33 - If True, bypass enforcement (for debugging only)
        
    Returns:
        Final recommendation with adjusted score and decision quality
    """
    try:

        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get session info
        cur.execute(
            "SELECT goal, context FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": "Session not found"}
        
        # Get all thought nodes ordered by posterior score
        cur.execute(
            """
            SELECT id, path, content, node_type, depth,
                   prior_score, likelihood, posterior_score,
                   supporting_laws
            FROM thought_nodes
            WHERE session_id = %s AND node_type = 'hypothesis'
            ORDER BY posterior_score DESC
            LIMIT %s
            """,
            (session_id, top_n)
        )
        candidates = cur.fetchall()
        
        if not candidates:
            return {
                "success": False, 
                "error": "No hypotheses found. Run prepare_expansion first."
            }
        
        # Get sibling counts for shallow_alternatives penalty
        # (count how many siblings each node has at its level)
        cur.execute(
            """
            SELECT SUBPATH(path, 0, NLEVEL(path) - 1) as parent_path, 
                   COUNT(*) as sibling_count
            FROM thought_nodes
            WHERE session_id = %s AND node_type = 'hypothesis'
            GROUP BY SUBPATH(path, 0, NLEVEL(path) - 1)
            """,
            (session_id,)
        )
        sibling_counts = {str(row["parent_path"]): row["sibling_count"] for row in cur.fetchall()}
        
        # v13b: Get unique law counts per parent path for monoculture detection
        cur.execute(
            """
            SELECT SUBPATH(path, 0, NLEVEL(path) - 1) as parent_path,
                   COUNT(DISTINCT supporting_laws[1]) as unique_laws,
                   COUNT(*) as total_siblings
            FROM thought_nodes
            WHERE session_id = %s AND node_type = 'hypothesis' 
                  AND supporting_laws IS NOT NULL AND array_length(supporting_laws, 1) > 0
            GROUP BY SUBPATH(path, 0, NLEVEL(path) - 1)
            """,
            (session_id,)
        )
        law_diversity = {str(row["parent_path"]): (row["unique_laws"], row["total_siblings"]) 
                        for row in cur.fetchall()}
        
        # Check if nodes have been critiqued (likelihood changed from initial)
        # A critiqued node will have likelihood != 0.X initial value
        
        # Process each candidate
        processed = []
        for node in candidates:
            # v35: Use extracted pure helper for penalty calculations
            adjusted_score, penalties = _apply_heuristic_penalties(
                node, sibling_counts, law_diversity
            )
            
            # v11b: Calculate law-grounded rollout score
            supporting_law_ids = node["supporting_laws"] or []
            rollout_score = 0.5  # Default neutral
            if supporting_law_ids:
                cur.execute(
                    """
                    SELECT AVG(scientific_weight) as avg_weight
                    FROM scientific_laws
                    WHERE id = ANY(%s) AND scientific_weight IS NOT NULL
                    """,
                    (supporting_law_ids,)
                )
                rollout_result = cur.fetchone()
                if rollout_result and rollout_result["avg_weight"]:
                    rollout_score = float(rollout_result["avg_weight"])
            
            # v35b: Build processed dict via helper
            processed.append(_build_processed_candidate(
                node, adjusted_score, penalties, rollout_score
            ))
        
        # Sort by final score (includes rollout blending)
        processed.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Deep critique mode - return requests for LLM
        if deep_critique and len(processed) >= 1:
            critique_requests = []
            for p in processed[:2]:  # Top 2 only
                # Get supporting laws for context
                cur.execute(
                    """
                    SELECT sl.law_name, sl.definition
                    FROM scientific_laws sl
                    JOIN thought_nodes tn ON sl.id = ANY(tn.supporting_laws)
                    WHERE tn.id = %s
                    LIMIT 3
                    """,
                    (p["node_id"],)
                )
                laws = cur.fetchall()
                critique_requests.append({
                    "node_id": p["node_id"],
                    "content": p["content"],
                    "supporting_laws": [{"name": l["law_name"], "definition": l["definition"]} for l in laws]
                })
            
            return {
                "success": True,
                "needs_critique": True,
                "message": "Generate critiques for these nodes, then call store_critique for each, then call finalize_session again.",
                "critique_requests": critique_requests
            }
        
        # Compare top 2 (comparative critique)
        recommendation = processed[0]
        runner_up = processed[1] if len(processed) > 1 else None
        
        # v11a/v35b: UCT tiebreaking for close decisions (using pure helper)
        uct_applied = False
        if runner_up:
            gap = recommendation["final_score"] - runner_up["final_score"]
            
            # v35b: Apply UCT tiebreaking via helper
            should_swap, uct_applied = _apply_uct_tiebreaking(
                recommendation["final_score"], recommendation["depth"],
                runner_up["final_score"], runner_up["depth"]
            )
            if should_swap:
                recommendation, runner_up = runner_up, recommendation
                processed[0], processed[1] = processed[1], processed[0]
                gap = recommendation["final_score"] - runner_up["final_score"]
            
            # v35b: Compute decision quality via helper
            rec_conf = recommendation.get("confidence", 0.5)
            run_conf = runner_up.get("confidence", 0.5)
            decision_quality, gap_analysis = _compute_decision_quality(
                gap, rec_conf, run_conf, uct_applied
            )
        else:
            decision_quality = "medium"
            gap_analysis = "Only one candidate available."
            gap = 0  # Single candidate, no gap
        
        # v20: Compute adaptive depth quality metrics
        quality_result = compute_quality_metrics(
            cur=cur,
            session_id=session_id,
            candidates=processed,
            gap=gap
        )
        
        # Get interview context summary if available
        context = session["context"] or {}
        interview = context.get("interview", {})
        context_summary = None
        if interview.get("answer_history"):
            context_summary = {
                h["question_id"]: h["answer"] 
                for h in interview["answer_history"]
            }
        
        # Determine next_step guidance based on decision quality and depth
        winner_depth = recommendation.get("depth", 2)
        next_step = None
        if decision_quality == "low" and winner_depth < 4:
            next_step = f"Decision is close. Expand the winning hypothesis deeper. Call prepare_expansion(session_id='{session_id}', parent_node_id='{recommendation['node_id']}')"
        elif decision_quality == "medium" and winner_depth < 3:
            next_step = f"Consider refining the recommendation. Call prepare_expansion(session_id='{session_id}', parent_node_id='{recommendation['node_id']}')"
        
        # v8a: Outcome prompt to close self-learning loop
        # v17c: Focus on business value, not code quality (RLVR handles that)
        outcome_prompt = f"Did this solve your business problem? record_outcome(session_id='{session_id}', outcome='success'|'partial'|'failure'). Note: Code quality is validated automatically by RLVR."
        
        # v8c: Implementation checklist - bridge reasoning to action
        implementation_checklist = []
        winning_content = recommendation["content"]
        winning_scope = recommendation.get("declared_scope", "")
        
        # Parse scope to generate checklist items
        if winning_scope:
            scope_items = [s.strip() for s in winning_scope.split(",") if s.strip()]
            for item in scope_items:
                implementation_checklist.append(f"[ ] Modify: {item}")
        if not implementation_checklist:
            implementation_checklist.append("[ ] Implement the recommended approach")
        
        # Add standard checklist items
        implementation_checklist.extend([
            "[ ] Write/update tests",
            "[ ] Verify changes work as expected"
        ])
        
        # v26/v35b: Suggest tags based on goal and recommendation (using pure helper)
        suggested_tags: list[str] = []
        try:
            suggested_tags = _generate_suggested_tags(session["goal"], winning_content)
            
            if suggested_tags:
                logger.info(f"v26: Suggested tags for session {session_id}: {suggested_tags}")
                # v34: Store suggested_tags in DB for auto-application on record_outcome
                try:
                    cur.execute(
                        "UPDATE reasoning_sessions SET suggested_tags = %s WHERE id = %s",
                        (json.dumps(suggested_tags), session_id)
                    )
                    conn.commit()
                except Exception as e:
                    logger.warning(f"v34 suggested_tags DB write failed: {e}")
        except Exception as e:
            logger.warning(f"v26 tag suggestion failed: {e}")
        
        # v15b: Query past failures with similar goals (domain-agnostic via semantic similarity)
        past_failures = []
        try:
            if session.get("goal_embedding"):
                cur.execute("""
                    SELECT o.notes, o.failure_reason, s.goal 
                    FROM outcome_records o 
                    JOIN reasoning_sessions s ON o.session_id = s.id
                    WHERE o.outcome != 'success' 
                    AND (o.notes IS NOT NULL OR o.failure_reason IS NOT NULL)
                    AND s.goal_embedding IS NOT NULL
                    AND 1 - (s.goal_embedding <=> %s) > 0.7
                    LIMIT 3
                """, (session["goal_embedding"],))
                for r in cur.fetchall():
                    reason = r["failure_reason"] or r["notes"]
                    if reason:
                        past_failures.append({"goal": r["goal"][:100], "reason": reason})
        except Exception as e:
            logger.warning(f"v15b past_failures lookup failed: {e}")
        
        # v16b.1: Compute confidence calibration warning (LLM nudge, not adjustment)
        calibration_warning = None
        try:
            cur.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE o.outcome = 'success') as successes,
                    COUNT(*) as total
                FROM outcome_records o
                JOIN thought_nodes t ON t.session_id = o.session_id
                WHERE t.likelihood >= 0.8
                AND o.created_at > NOW() - INTERVAL '30 days'
            """)
            cal = cur.fetchone()
            if cal and cal["total"] >= 5:
                success_rate = cal["successes"] / cal["total"]
                if success_rate < 0.7:
                    calibration_warning = f"⚠️ Calibration: High-confidence (≥0.8) hypotheses succeed only {success_rate:.0%} of the time in recent sessions"
        except Exception as e:
            logger.warning(f"v16b.1 calibration query failed: {e}")
        
        # v15b: Construct scope_guidance (domain-agnostic)
        winning_scope = recommendation.get("declared_scope", "") or ""
        scope_guidance = {
            "context": {
                "goal": session["goal"],
                "scope": winning_scope,
                "recommendation": recommendation["content"][:200]
            },
            "prompt": "What validation or follow-up steps are needed for this specific context? If this is pure reasoning with no action items, respond 'No follow-up needed'.",
            "past_failures": past_failures,
            "calibration_warning": calibration_warning  # v16b.1
        }
        
        # v17b: RLVR Auto-Recording - parse terminal output and auto-record outcome
        rlvr_result = None
        if terminal_output and terminal_output.strip():
            try:
                rlvr_result = await parse_terminal_output(
                    session_id=session_id,
                    terminal_text=terminal_output,
                    auto_record=True
                )
                logger.info(f"v17b: RLVR auto-recorded for session {session_id}: {rlvr_result.get('signal')}")
            except Exception as e:
                logger.warning(f"v17b RLVR auto-record failed: {e}")
                rlvr_result = {"success": False, "error": str(e)}
        
        # v31b: Exhaustive Check - layer-by-layer gap analysis
        # v32 FIX: MCP sampling not supported - return prompt for agent to process
        exhaustive_gaps = None
        exhaustive_prompt = None
        if exhaustive_check and recommendation:
            exhaustive_prompt = {
                "prompt": f"""Analyze this recommendation for what it does NOT address:

Recommendation: {recommendation["content"]}
Goal: {session["goal"]}

For EACH of these layers, identify what's covered vs missing:
1. CODE STRUCTURE: What code changes are needed? Any not mentioned?
2. DEPENDENCIES: What packages/systems are assumed but not stated?
3. DATA FLOW: What data moves where? Any gaps in the data path?
4. INTERFACES: What APIs/contracts are affected? Any missing?
5. WORKFLOWS: What user/system flows change? Any not addressed?

Return ONLY a valid JSON object:
{{
  "gaps": [
    {{"layer": "...", "covered": "...", "missing": "..."}}
  ],
  "critical_gaps": ["gaps that could cause failure"],
  "overall_coverage": 0.8
}}

Focus on OMISSIONS, not flaws.""",
                "system": "You are a gap analyst. Find what's MISSING, not what's wrong. Return only valid JSON.",
                "instructions": "Process this prompt and review the gaps. If critical gaps found, consider deepening your analysis."
            }
            logger.info(f"v32: Returning exhaustive_prompt for agent to process")

        
        # v32: Sequential analysis moved to run_sequential_analysis tool
        # Call run_sequential_analysis(session_id) BEFORE finalize_session for gap analysis
        sequential_analysis: list[dict[str, Any]] = []  # Populated by run_sequential_analysis if called
        systemic_gaps: list[str] = []
        
        # v31d: Quality Gate - check score and gap thresholds
        winner_score = recommendation["adjusted_score"]
        runner_up_score = runner_up["adjusted_score"] if runner_up else 0.0
        gap = winner_score - runner_up_score if runner_up else 1.0
        
        quality_gate = {
            "score": round(winner_score, 4),
            "score_threshold": min_score_threshold,
            "score_ok": winner_score >= min_score_threshold,
            "gap": round(gap, 4),
            "gap_threshold": min_gap_threshold,
            "gap_ok": gap >= min_gap_threshold,
            "passed": winner_score >= min_score_threshold and gap >= min_gap_threshold
        }
        
        # v31e: Score Improvement Suggestions
        score_improvement_suggestions = []
        if not quality_gate["passed"]:
            if not quality_gate["score_ok"]:
                score_improvement_suggestions.append({
                    "lever": "score",
                    "current": round(winner_score, 3),
                    "threshold": min_score_threshold,
                    "action": "Expand deeper with higher confidence (0.9+) or address critique penalties"
                })
            if not quality_gate["gap_ok"]:
                score_improvement_suggestions.append({
                    "lever": "gap",
                    "current": round(gap, 3),
                    "threshold": min_gap_threshold,
                    "action": "Explore more diverse alternatives to differentiate the best solution"
                })
        
        # v33: Quality gate enforcement
        quality_gate_enforced = quality_gate["passed"] or skip_quality_gate
        
        # v33: If gate not passed AND not skipped, prefix [UNVERIFIED] and log
        recommendation_content = recommendation["content"]
        if not quality_gate_enforced:
            recommendation_content = f"[UNVERIFIED] {recommendation['content']}"
            # Log enforcement violation to DB
            try:
                cur.execute("""
                    UPDATE reasoning_sessions 
                    SET context = COALESCE(context, '') || E'\n[ENFORCEMENT VIOLATION at ' || NOW()::text || ']'
                    WHERE id = %s
                """, (session_id,))
                conn.commit()
                logger.warning(f"v33: Quality gate not enforced for session {session_id} (score={winner_score:.3f}, gap={gap:.3f})")
            except Exception as log_err:
                logger.error(f"Failed to log enforcement violation: {log_err}")
        
        return {
            "success": True,
            "session_id": session_id,
            "recommendation": {
                "node_id": recommendation["node_id"],
                "content": recommendation_content,  # v33: may have [UNVERIFIED] prefix
                "original_score": recommendation["original_score"],
                "adjusted_score": recommendation["adjusted_score"],
                "penalties_applied": recommendation["penalties_applied"]
            },
            # v33: Quality gate enforcement status
            "quality_gate_enforced": quality_gate_enforced,
            "runner_up": {
                "node_id": runner_up["node_id"],
                "content": runner_up["content"],
                "adjusted_score": runner_up["adjusted_score"]
            } if runner_up else None,
            "decision_quality": decision_quality,
            "gap_analysis": gap_analysis,
            "candidates_evaluated": len(processed),
            "context_summary": context_summary,
            "next_step": next_step,
            "outcome_prompt": outcome_prompt if not rlvr_result else None,  # v17b: suppress if auto-recorded
            "implementation_checklist": implementation_checklist,
            "scope_guidance": scope_guidance,  # v15b
            "rlvr_result": rlvr_result,  # v17b: auto-outcome detection result
            # v20: Adaptive depth quality metrics
            "quality_sufficient": quality_result["sufficient"],
            "quality_breakdown": quality_result["metrics"],
            "deepen_suggestions": quality_result["suggestions"] if not quality_result["sufficient"] else [],
            # v26: Suggested tags for session organization
            "suggested_tags": suggested_tags,
            # v31b/v32: Exhaustive gap check - prompt for agent to process if needed
            "exhaustive_gaps": exhaustive_gaps,
            "exhaustive_prompt": exhaustive_prompt,  # v32: LLM prompt for agent to process
            # v31d: Quality gate (score + gap thresholds)
            "quality_gate": quality_gate,
            # v31e: How to improve if gate not passed
            "score_improvement_suggestions": score_improvement_suggestions,
            # v32: Always-on sequential analysis
            "sequential_analysis": sequential_analysis,
            "systemic_gaps": systemic_gaps
        }


        
    except Exception as e:
        logger.error(f"finalize_session failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# Self-Learning Tools (Phase 4)
# =============================================================================


@mcp.tool()
async def tag_session(
    session_id: str,
    tags: str  # Comma-separated tags for universal LLM compatibility
) -> dict[str, Any]:
    """
    Add tags to a reasoning session for organization and retrieval.
    
    Tags are normalized (lowercase, trimmed) and deduplicated.
    If a tag has an alias defined, the canonical form is used.
    
    Args:
        session_id: The reasoning session UUID
        tags: Comma-separated list of tags (e.g., "backend, database, v27")
        
    Returns:
        Confirmation with normalized tags applied
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Parse and normalize tags
        tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
        if not tag_list:
            return {"success": False, "error": "No valid tags provided"}
        
        # Check aliases and normalize
        normalized_tags = []
        for tag in tag_list:
            cur.execute(
                "SELECT canonical_tag FROM tag_aliases WHERE alias = %s",
                (tag,)
            )
            alias_row = cur.fetchone()
            canonical = alias_row["canonical_tag"] if alias_row else tag
            if canonical not in normalized_tags:
                normalized_tags.append(canonical)
        
        # Insert tags (ignore duplicates)
        inserted = 0
        for tag in normalized_tags:
            try:
                cur.execute(
                    """
                    INSERT INTO session_tags (session_id, tag)
                    VALUES (%s, %s)
                    ON CONFLICT (session_id, tag) DO NOTHING
                    """,
                    (session_id, tag)
                )
                if cur.rowcount > 0:
                    inserted += 1
            except Exception as e:
                logger.warning(f"v26 tag insert failed for '{tag}': {e}")
        
        conn.commit()
        
        # Get all tags for session
        cur.execute(
            "SELECT tag FROM session_tags WHERE session_id = %s ORDER BY created_at",
            (session_id,)
        )
        all_tags = [r["tag"] for r in cur.fetchall()]
        
        logger.info(f"v26: Tagged session {session_id} with {inserted} new tags")
        
        return {
            "success": True,
            "session_id": session_id,
            "tags_added": inserted,
            "all_tags": all_tags,
            "message": f"Session tagged with {len(all_tags)} tags"
        }
        
    except Exception as e:
        logger.error(f"tag_session failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def record_outcome(
    session_id: str,
    outcome: str,
    confidence: float = 1.0,
    notes: Optional[str] = None,
    failure_reason: Optional[str] = None,  # v15b: explicit failure reason for learning
    keep_open: bool = False
) -> dict[str, Any]:
    """
    Record the outcome of a reasoning session for learning.
    
    Stores the outcome with attribution to the winning path nodes.
    This data enables PAS to learn which laws correlate with success.
    
    Args:
        session_id: The reasoning session UUID
        outcome: 'success', 'partial', or 'failure'
        confidence: How confident is this assessment (0.0-1.0)
        notes: Optional notes about why this outcome
        failure_reason: v15b - Explicit reason for failure (enables semantic learning)
        keep_open: If True, don't auto-complete even on success/failure
        
    Returns:
        Confirmation with stats about attributed nodes
    """

    if outcome not in ('success', 'partial', 'failure'):
        return {"success": False, "error": "outcome must be 'success', 'partial', or 'failure'"}
    
    confidence = max(0.0, min(1.0, confidence))
    
    try:
        conn = get_db_connection()
        conn.rollback()  # Defensive: ensure clean transaction state
        cur = conn.cursor()
        
        # Get the best path for this session
        cur.execute(
            """
            SELECT id, path, content, supporting_laws
            FROM thought_nodes
            WHERE session_id = %s AND node_type = 'hypothesis'
            ORDER BY posterior_score DESC
            LIMIT 1
            """,
            (session_id,)
        )
        best_node = cur.fetchone()
        
        if not best_node:
            return {"success": False, "error": "No hypotheses found in session"}
        
        winning_path = best_node["path"]
        
        # v15b: Get scope embedding for learning (from best node's declared_scope)
        scope_embedding = None
        try:
            cur.execute(
                "SELECT declared_scope FROM thought_nodes WHERE id = %s",
                (best_node["id"],)
            )
            scope_row = cur.fetchone()
            if scope_row and scope_row.get("declared_scope"):
                scope_embedding = get_embedding(scope_row["declared_scope"])
        except Exception as e:
            logger.warning(f"v15b scope embedding failed: {e}")
            conn.rollback()  # Clear aborted transaction state before continuing
        
        # v27: Embed failure_reason for semantic search (if provided)
        failure_reason_embedding = None
        if failure_reason:
            try:
                failure_reason_embedding = get_embedding(failure_reason[:2000])
                logger.info(f"v27: Embedded failure_reason for session {session_id}")
            except Exception as e:
                logger.warning(f"v27 failure_reason embedding failed: {e}")
        
        # Record the outcome with v15b + v27 fields
        cur.execute(
            """
            INSERT INTO outcome_records (session_id, outcome, confidence, winning_path, notes, failure_reason, scope_embedding, failure_reason_embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, created_at
            """,
            (session_id, outcome, confidence, winning_path, notes, failure_reason, scope_embedding, failure_reason_embedding)
        )
        record = cur.fetchone()
        
        # Count attributed nodes and laws
        cur.execute(
            """
            SELECT COUNT(*) as node_count,
                   ARRAY_AGG(DISTINCT unnest) as laws
            FROM thought_nodes, UNNEST(supporting_laws)
            WHERE session_id = %s AND path <@ %s
            """,
            (session_id, winning_path)
        )
        stats = cur.fetchone()
        
        # v12b: Update success_count for Thompson Sampling on success outcomes
        if outcome == 'success' and stats and stats["laws"]:
            attributed_laws = [l for l in stats["laws"] if l is not None]
            if attributed_laws:
                cur.execute(
                    "UPDATE scientific_laws SET success_count = success_count + 1 WHERE id = ANY(%s)",
                    (attributed_laws,)
                )
        
        # v12a: Log training data for PRM fine-tuning
        cur.execute(
            """
            SELECT content, path, supporting_laws FROM thought_nodes
            WHERE session_id = %s AND path <@ %s AND node_type = 'hypothesis'
            """,
            (session_id, winning_path)
        )
        for node in cur.fetchall():
            # Get first supporting law name
            law_name = None
            if node["supporting_laws"]:
                cur.execute("SELECT law_name FROM scientific_laws WHERE id = %s", (node["supporting_laws"][0],))
                law_row = cur.fetchone()
                law_name = law_row["law_name"] if law_row else None
            
            cur.execute(
                """
                INSERT INTO training_data (hypothesis_text, goal_text, outcome, depth, law_name, session_id)
                VALUES (%s, (SELECT goal FROM reasoning_sessions WHERE id = %s), %s, %s, %s, %s)
                """,
                (node["content"], session_id, outcome, len(str(node["path"]).split(".")), law_name, session_id)
            )
        
        # v13c: Track critique accuracy for self-learning calibration
        # Find all critiqued nodes in session (nodes where likelihood was modified from initial)
        cur.execute(
            """
            SELECT tn.id, tn.path, tn.likelihood, tn.posterior_score
            FROM thought_nodes tn
            WHERE tn.session_id = %s 
              AND tn.node_type = 'hypothesis'
              AND tn.likelihood NOT IN (0.8, 0.85, 0.9, 0.95, 0.88, 0.92, 0.75, 0.7)
            """,
            (session_id,)
        )
        critiqued_nodes = cur.fetchall()
        
        for cnode in critiqued_nodes:
            # v35: Use extracted pure helper for critique accuracy
            is_winner = str(cnode["path"]).startswith(str(winning_path)) or str(winning_path).startswith(str(cnode["path"]))
            critique_accurate = _compute_critique_accuracy(cnode["path"], winning_path, outcome)
            
            cur.execute(
                """
                INSERT INTO critique_accuracy 
                    (session_id, node_id, critique_severity, was_top_hypothesis, actual_outcome, critique_accurate)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (session_id, cnode["id"], 1.0 - float(cnode["likelihood"]), is_winner, outcome, critique_accurate)
            )
        
        # Auto-complete session if outcome is definitive (not partial) and not keep_open
        session_completed = False
        if outcome in ('success', 'failure') and not keep_open:
            cur.execute(
                "UPDATE reasoning_sessions SET state = 'completed' WHERE id = %s",
                (session_id,)
            )
            session_completed = True
        
        # =====================================================================
        # v22 Feature 1b: Persist Traits to user_trait_profiles
        # Store session traits with outcome-based weighting
        # =====================================================================
        traits_persisted = 0
        try:
            # Get session context for user_id and latent_traits
            cur.execute("SELECT context FROM reasoning_sessions WHERE id = %s", (session_id,))
            ctx_row = cur.fetchone()
            session_context = ctx_row["context"] if ctx_row and ctx_row["context"] else {}
            
            user_id = session_context.get("user_id")
            latent_traits = session_context.get("latent_traits", [])
            
            if user_id and latent_traits:
                # v35: Use extracted pure helper
                outcome_multiplier = _get_outcome_multiplier(outcome)
                
                for trait_info in latent_traits:
                    trait_name = trait_info.get("trait")
                    if not trait_name:
                        continue
                    
                    evidence = trait_info.get("confidence", 0.5) * outcome_multiplier
                    
                    cur.execute("""
                        INSERT INTO user_trait_profiles (user_id, trait_name, cumulative_score, evidence_count)
                        VALUES (%s, %s, %s, 1)
                        ON CONFLICT (user_id, trait_name) DO UPDATE SET
                            cumulative_score = user_trait_profiles.cumulative_score + %s,
                            evidence_count = user_trait_profiles.evidence_count + 1,
                            last_reinforced_at = now()
                    """, (user_id, trait_name, evidence, evidence))
                    traits_persisted += 1
                
                if traits_persisted > 0:
                    logger.info(f"v22: Persisted {traits_persisted} traits for user {user_id[:8]}...")
                    
        except Exception as e:
            logger.warning(f"v22: Failed to persist traits: {e}")
        
        conn.commit()
        
        # v8b: Auto-trigger refresh_law_weights after sufficient outcomes
        auto_refresh_result = None
        MIN_SAMPLES_THRESHOLD = 5
        try:
            cur.execute("SELECT COUNT(*) FROM outcome_records")
            total_outcomes = cur.fetchone()[0]
            if total_outcomes >= MIN_SAMPLES_THRESHOLD and total_outcomes % MIN_SAMPLES_THRESHOLD == 0:
                # Trigger refresh on every Nth outcome (5, 10, 15, etc.)
                auto_refresh_result = await refresh_law_weights(min_samples=MIN_SAMPLES_THRESHOLD)
        except Exception as refresh_err:
            logger.warning(f"Auto-refresh check failed: {refresh_err}")
        
        # =====================================================================
        # v34: Auto-apply suggested_tags on success/partial outcomes
        # =====================================================================
        auto_tagged = None
        if outcome in ('success', 'partial'):
            try:
                cur.execute(
                    "SELECT suggested_tags FROM reasoning_sessions WHERE id = %s",
                    (session_id,)
                )
                tags_row = cur.fetchone()
                if tags_row and tags_row.get("suggested_tags"):
                    suggested = tags_row["suggested_tags"]
                    if isinstance(suggested, str):
                        suggested = json.loads(suggested)
                    if suggested:
                        # Apply tags using existing tag_session logic
                        for tag in suggested:
                            # Normalize tag
                            normalized = tag.strip().lower()
                            if normalized:
                                cur.execute(
                                    """
                                    INSERT INTO session_tags (session_id, tag)
                                    VALUES (%s, %s)
                                    ON CONFLICT (session_id, tag) DO NOTHING
                                    """,
                                    (session_id, normalized)
                                )
                        conn.commit()
                        auto_tagged = suggested
                        logger.info(f"v34: Auto-tagged session {session_id} with {suggested}")
            except Exception as e:
                logger.warning(f"v34 auto-tagging failed: {e}")
        
        return {
            "success": True,
            "session_id": session_id,
            "outcome_id": str(record["id"]),
            "outcome": outcome,
            "confidence": confidence,
            "winning_path": winning_path,
            "attributed_nodes": stats["node_count"] or 0,
            "laws_affected": stats["laws"] or [],
            "session_completed": session_completed,
            "auto_refresh_triggered": auto_refresh_result is not None,
            "auto_refresh_result": auto_refresh_result,
            "auto_tagged": auto_tagged,  # v34
            "message": f"Outcome recorded. {stats['node_count'] or 0} nodes in winning path." + 
                      (" Session auto-completed." if session_completed else "") +
                      (f" Auto-refreshed {auto_refresh_result.get('laws_updated', 0)} law weights." if auto_refresh_result else "") +
                      (f" Auto-tagged: {auto_tagged}" if auto_tagged else "")
        }
        
    except Exception as e:
        logger.error(f"record_outcome failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# v17a: RLVR Auto-Outcome Detection
# =============================================================================

import re

# Domain-agnostic patterns for success/failure detection
SUCCESS_PATTERNS = [
    r'\bPASS(?:ED)?\b',
    r'\bOK\b',
    r'\bSUCCESS(?:FUL)?\b',
    r'✓',
    r'\bAll tests passed\b',
    r'\bBuild succeeded\b',
    r'\bexit code 0\b',
    r'\b0 failed\b',
    r'\bno errors\b',
    r'\bcompleted successfully\b',
]

FAILURE_PATTERNS = [
    r'\bFAIL(?:ED|URE)?\b',
    r'\bERROR\b',
    r'\bException\b',
    r'✗',
    r'\bAssertionError\b',
    r'\bBuild failed\b',
    r'exit code [1-9]\d*',
    r'\bTraceback\b',
    r'\bSyntaxError\b',
    r'\bTypeError\b',
    r'\bValueError\b',
    r'\bAttributeError\b',
    r'\bImportError\b',
    r'\bRuntimeError\b',
    r'\bCRITICAL\b',
    r'\bFATAL\b',
]

# v17b.2: Patterns to extract failure reasons for semantic learning
# Each pattern has a named group 'reason' to capture the error message
FAILURE_REASON_PATTERNS = [
    # Python exceptions with message: "ValueError: invalid literal for int()"
    r'(?P<type>\w+Error):\s*(?P<reason>.+?)(?:\n|$)',
    # Python exceptions: "Exception: something went wrong"
    r'(?P<type>Exception):\s*(?P<reason>.+?)(?:\n|$)',
    # Assertion failures: "AssertionError: expected X but got Y"
    r'AssertionError:\s*(?P<reason>.+?)(?:\n|$)',
    # pytest/jest assertion: "assert x == y" or "Expected X to equal Y"
    r'(?:assert|Assert)\s+(?P<reason>.+?)(?:\n|$)',
    r'Expected\s+(?P<reason>.+?)(?:\n|$)',
    # Build errors: "error: cannot find module 'X'"
    r'error:\s*(?P<reason>.+?)(?:\n|$)',
    # Compilation errors: "fatal error: file not found"
    r'fatal error:\s*(?P<reason>.+?)(?:\n|$)',
    # npm/node errors: "Error: Cannot find module"
    r'Error:\s*(?P<reason>.+?)(?:\n|$)',
    # Generic test failure with name: "FAILED test_foo - reason"
    r'FAILED\s+\S+\s*[-:]\s*(?P<reason>.+?)(?:\n|$)',
]


@mcp.tool()
async def parse_terminal_output(
    session_id: str,
    terminal_text: str,
    auto_record: bool = False
) -> dict[str, Any]:
    """
    Parse terminal output for success/failure signals (v17a RLVR).
    
    Uses domain-agnostic regex patterns to detect test pass/fail,
    build success/error, and runtime crashes. Returns structured
    signal detection with confidence scoring.
    
    Args:
        session_id: The reasoning session UUID
        terminal_text: Raw terminal output to analyze
        auto_record: If True and signal is clear, auto-call record_outcome
        
    Returns:
        Detected signal with confidence and matches
    """
    if not terminal_text or not terminal_text.strip():
        return {
            "success": True,
            "signal": "unknown",
            "confidence": 0.0,
            "matches": [],
            "message": "Empty terminal output provided"
        }
    
    # Case-insensitive matching
    success_matches = []
    failure_matches = []
    
    for pattern in SUCCESS_PATTERNS:
        matches = re.findall(pattern, terminal_text, re.IGNORECASE)
        if matches:
            success_matches.extend(matches)
    
    for pattern in FAILURE_PATTERNS:
        matches = re.findall(pattern, terminal_text, re.IGNORECASE)
        if matches:
            failure_matches.extend(matches)
    
    # v17a.2: Filter out false-positive failures from success contexts
    # "0 failed", "passed", etc. should not count as failure signals
    false_positive_context = re.search(r'\b0\s+failed\b', terminal_text, re.IGNORECASE)
    if false_positive_context:
        # Remove one 'failed' match for each '0 failed' context found
        zero_failed_count = len(re.findall(r'\b0\s+failed\b', terminal_text, re.IGNORECASE))
        for _ in range(zero_failed_count):
            for i, m in enumerate(failure_matches):
                if m.lower() == 'failed':
                    failure_matches.pop(i)
                    break
    
    # Determine signal and confidence
    success_count = len(success_matches)
    failure_count = len(failure_matches)
    total = success_count + failure_count
    
    if total == 0:
        signal = "unknown"
        confidence = 0.0
        all_matches = []
    elif failure_count > 0 and success_count == 0:
        signal = "failure"
        confidence = min(0.95, 0.7 + (failure_count * 0.05))
        all_matches = failure_matches
    elif success_count > 0 and failure_count == 0:
        signal = "success"
        confidence = min(0.95, 0.7 + (success_count * 0.05))
        all_matches = success_matches
    else:
        # Mixed signals - failure takes precedence
        if failure_count >= success_count:
            signal = "failure"
            confidence = 0.6
        else:
            signal = "success"
            confidence = 0.5
        all_matches = failure_matches + success_matches
    
    # v17b.2: Extract failure reason for semantic learning
    failure_reason = None
    if signal == "failure":
        for pattern in FAILURE_REASON_PATTERNS:
            match = re.search(pattern, terminal_text, re.IGNORECASE)
            if match:
                try:
                    failure_reason = match.group('reason').strip()[:200]  # Limit length
                    break
                except IndexError:
                    continue
    
    result = {
        "success": True,
        "session_id": session_id,
        "signal": signal,
        "confidence": round(confidence, 2),
        "success_matches": success_matches[:5],  # Limit to first 5
        "failure_matches": failure_matches[:5],
        "total_success_signals": success_count,
        "total_failure_signals": failure_count,
        "failure_reason": failure_reason,  # v17b.2
        "message": f"Detected {signal} signal with {confidence:.0%} confidence"
    }
    
    # Auto-record if enabled and confidence is high
    if auto_record and signal != "unknown" and confidence >= 0.7:
        try:
            outcome_result = await record_outcome(
                session_id=session_id,
                outcome=signal,
                confidence=confidence,
                notes=f"Auto-recorded by v17a RLVR. Matches: {all_matches[:3]}",
                failure_reason=failure_reason  # v17b.2: pass extracted reason
            )
            result["auto_recorded"] = True
            result["outcome_result"] = outcome_result
            result["message"] = str(result["message"]) + f" Auto-recorded as {signal}."
            if failure_reason:
                result["message"] = str(result["message"]) + " Failure reason extracted."
        except Exception as e:
            result["auto_recorded"] = False
            result["auto_record_error"] = str(e)
    else:
        result["auto_recorded"] = False
        if auto_record and signal == "unknown":
            result["message"] = str(result["message"]) + " Signal too ambiguous for auto-record."
        elif auto_record and confidence < 0.7:
            result["message"] = str(result["message"]) + f" Confidence too low ({confidence:.0%}) for auto-record."
    
    return result


@mcp.tool()
async def refresh_law_weights(
    min_samples: int = 5,
    blend_factor: float = 0.5
) -> dict[str, Any]:
    """
    Update law weights based on accumulated outcome data.
    
    Computes success rates from outcome_records and blends with
    original scientific_weight. Only updates laws with sufficient data.
    
    Args:
        min_samples: Minimum outcomes required before updating (default: 5)
        blend_factor: How much new data influences weight (0-1, default: 0.5)
        
    Returns:
        Number of laws updated and their new weights
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get success rates for laws with sufficient data
        cur.execute(
            """
            SELECT 
                law_id,
                law_name,
                total_outcomes,
                success_rate
            FROM law_domain_stats
            WHERE total_outcomes >= %s
            """,
            (min_samples,)
        )
        stats = cur.fetchall()
        
        if not stats:
            return {
                "success": True,
                "laws_updated": 0,
                "message": f"No laws have {min_samples}+ recorded outcomes yet."
            }
        
        updated = []
        for row in stats:
            # Blend new success rate with original weight
            cur.execute(
                "SELECT scientific_weight FROM scientific_laws WHERE id = %s",
                (row["law_id"],)
            )
            original = cur.fetchone()
            if not original:
                continue
                
            old_weight = float(original["scientific_weight"])
            new_weight = (float(row["success_rate"]) * blend_factor) + (old_weight * (1 - blend_factor))
            new_weight = round(max(0.1, min(1.0, new_weight)), 3)
            
            # Update the weight
            cur.execute(
                """
                UPDATE scientific_laws 
                SET scientific_weight = %s,
                    domain_weights = domain_weights || %s
                WHERE id = %s
                """,
                (new_weight, json.dumps({"_last_update": str(row["success_rate"])}), row["law_id"])
            )
            
            updated.append({
                "law_name": row["law_name"],
                "samples": row["total_outcomes"],
                "success_rate": float(row["success_rate"]),
                "old_weight": old_weight,
                "new_weight": new_weight
            })
        
        conn.commit()
        
        return {
            "success": True,
            "laws_updated": len(updated),
            "updates": updated,
            "message": f"Updated {len(updated)} law weights based on {sum(u['samples'] for u in updated)} outcomes."
        }
        
    except Exception as e:
        logger.error(f"refresh_law_weights failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# Session Lifecycle Tools (Phase 5)
# =============================================================================

@mcp.tool()
async def find_or_create_session(
    goal_text: str,
    similarity_threshold: float = 0.8
) -> dict[str, Any]:
    """
    Smart session router - finds existing sessions or creates new ones.
    
    Uses semantic similarity to detect related sessions:
    - If >threshold similarity to ACTIVE session: returns existing
    - If >threshold similarity to COMPLETED session: creates continuation
    - Otherwise: creates new session
    
    Args:
        goal_text: The goal/question for reasoning
        similarity_threshold: Cosine similarity threshold (default: 0.8)
        
    Returns:
        Session info with action taken (existing/continuation/new)
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Generate embedding for the goal
        goal_embedding = get_embedding(goal_text)
        
        # Search for similar sessions
        cur.execute(
            """
            SELECT id, goal, state, 
                   1 - (goal_embedding <=> %s::vector) as similarity
            FROM reasoning_sessions
            WHERE goal_embedding IS NOT NULL
            ORDER BY goal_embedding <=> %s::vector
            LIMIT 5
            """,
            (goal_embedding, goal_embedding)
        )
        similar = cur.fetchall()
        
        # Check for matches above threshold
        best_match = None
        for row in similar:
            if row["similarity"] and row["similarity"] >= similarity_threshold:
                best_match = row
                break
        
        if best_match:
            if best_match["state"] == "active":
                # Return existing active session
                return {
                    "success": True,
                    "action": "existing",
                    "session_id": str(best_match["id"]),
                    "goal": best_match["goal"],
                    "similarity": round(best_match["similarity"], 3),
                    "message": f"Found existing active session (similarity: {best_match['similarity']:.2f})"
                }
            else:
                # Create continuation of completed session
                cur.execute(
                    """
                    INSERT INTO reasoning_sessions (goal, goal_embedding, parent_session_id, context)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (goal_text, goal_embedding, best_match["id"], 
                     json.dumps({"continued_from": str(best_match["id"])}))
                )
                new_session = cur.fetchone()
                
                # Copy root node from parent as context
                cur.execute(
                    """
                    INSERT INTO thought_nodes (session_id, path, content, node_type)
                    SELECT %s, 'root', content, 'root'
                    FROM thought_nodes
                    WHERE session_id = %s AND path = 'root'
                    """,
                    (new_session["id"], best_match["id"])
                )
                
                conn.commit()
                return {
                    "success": True,
                    "action": "continuation",
                    "session_id": str(new_session["id"]),
                    "parent_session_id": str(best_match["id"]),
                    "goal": goal_text,
                    "similarity": round(best_match["similarity"], 3),
                    "message": f"Created continuation of completed session (similarity: {best_match['similarity']:.2f})"
                }
        
        # No match - create new session
        cur.execute(
            """
            INSERT INTO reasoning_sessions (goal, goal_embedding)
            VALUES (%s, %s)
            RETURNING id, created_at
            """,
            (goal_text, goal_embedding)
        )
        new_session = cur.fetchone()
        
        # Create root node
        cur.execute(
            """
            INSERT INTO thought_nodes (session_id, path, content, node_type, embedding)
            VALUES (%s, 'root', %s, 'root', %s)
            """,
            (new_session["id"], goal_text, goal_embedding)
        )
        
        conn.commit()
        return {
            "success": True,
            "action": "new",
            "session_id": str(new_session["id"]),
            "goal": goal_text,
            "message": "Created new reasoning session"
        }
        
    except Exception as e:
        logger.error(f"find_or_create_session failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def complete_session(
    session_id: str,
    notes: Optional[str] = None
) -> dict[str, Any]:
    """
    Explicitly complete a reasoning session.
    
    Use this to manually close sessions that shouldn't auto-complete
    or to close abandoned sessions.
    
    Args:
        session_id: The reasoning session UUID
        notes: Optional notes about why session is being closed
        
    Returns:
        Confirmation of completion
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            """
            UPDATE reasoning_sessions 
            SET state = 'completed', 
                context = context || %s
            WHERE id = %s
            RETURNING state
            """,
            (json.dumps({"completion_notes": notes or "Manually completed"}), session_id)
        )
        result = cur.fetchone()
        
        if not result:
            return {"success": False, "error": "Session not found"}
        
        conn.commit()
        return {
            "success": True,
            "session_id": session_id,
            "state": "completed",
            "message": "Session marked as completed"
        }
        
    except Exception as e:
        logger.error(f"complete_session failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def resume_session(
    session_id: str
) -> dict[str, Any]:
    """
    Resume a completed or paused session.
    
    Creates a continuation session that inherits context from the original.
    Use this when you want to build on previous reasoning.
    
    Args:
        session_id: The session UUID to resume
        
    Returns:
        New continuation session info
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get original session
        cur.execute(
            "SELECT id, goal, goal_embedding, state FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        original = cur.fetchone()
        
        if not original:
            return {"success": False, "error": "Session not found"}
        
        if original["state"] == "active":
            return {
                "success": True,
                "action": "already_active",
                "session_id": session_id,
                "message": "Session is already active"
            }
        
        # Create continuation session
        cur.execute(
            """
            INSERT INTO reasoning_sessions (goal, goal_embedding, parent_session_id, context)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (original["goal"], original["goal_embedding"], session_id,
             json.dumps({"resumed_from": session_id}))
        )
        new_session = cur.fetchone()
        
        # Copy best hypothesis from parent as starting context
        cur.execute(
            """
            INSERT INTO thought_nodes (session_id, path, content, node_type, prior_score, likelihood)
            SELECT %s, 'root', content, 'root', posterior_score, posterior_score
            FROM thought_nodes
            WHERE session_id = %s
            ORDER BY posterior_score DESC
            LIMIT 1
            """,
            (new_session["id"], session_id)
        )
        
        conn.commit()
        return {
            "success": True,
            "action": "resumed",
            "session_id": str(new_session["id"]),
            "parent_session_id": session_id,
            "goal": original["goal"],
            "message": f"Created continuation session, inheriting from {session_id}"
        }
        
    except Exception as e:
        logger.error(f"resume_session failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# v25: Conversation Logging Tools
# =============================================================================

@mcp.tool()
async def search_conversation_log(
    query: str,
    session_id: Optional[str] = None,
    log_type: Optional[str] = None,
    limit: int = 10
) -> dict[str, Any]:
    """
    Search conversation logs by semantic similarity.
    
    Finds past user inputs, feedback, and context that match the query.
    Useful for understanding what the user has said across sessions.
    
    Args:
        query: The search text (will be embedded for semantic matching)
        session_id: Optional - limit search to specific session
        log_type: Optional - filter by type ('user_input', 'feedback', 'interview_answer', 'context')
        limit: Maximum results to return (default: 10)
        
    Returns:
        Matching log entries with their linked sessions and thought nodes
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Generate query embedding
        query_embedding = get_embedding(query)
        
        # Build query based on filters
        base_query = """
            SELECT 
                cl.id,
                cl.session_id,
                cl.thought_node_id,
                cl.user_id,
                cl.log_type,
                cl.raw_text,
                cl.created_at,
                1 - (cl.embedding <=> %s::vector) as similarity,
                rs.goal as session_goal,
                tn.content as linked_thought
            FROM conversation_log cl
            JOIN reasoning_sessions rs ON cl.session_id = rs.id
            LEFT JOIN thought_nodes tn ON cl.thought_node_id = tn.id
            WHERE cl.embedding IS NOT NULL
        """
        params: list[Any] = [query_embedding]
        
        if session_id:
            base_query += " AND cl.session_id = %s"
            params.append(session_id)
        
        if log_type:
            base_query += " AND cl.log_type = %s"
            params.append(log_type)
        
        base_query += " ORDER BY cl.embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, limit])
        
        cur.execute(base_query, params)
        results = cur.fetchall()
        
        entries = []
        for r in results:
            entries.append({
                "id": str(r["id"]),
                "session_id": str(r["session_id"]),
                "thought_node_id": str(r["thought_node_id"]) if r["thought_node_id"] else None,
                "user_id": r["user_id"],
                "log_type": r["log_type"],
                "raw_text": r["raw_text"][:500] + "..." if len(r["raw_text"]) > 500 else r["raw_text"],
                "full_text_length": len(r["raw_text"]),
                "similarity": round(float(r["similarity"]), 4),
                "session_goal": r["session_goal"][:100] + "..." if len(r["session_goal"]) > 100 else r["session_goal"],
                "linked_thought": r["linked_thought"][:150] + "..." if r["linked_thought"] and len(r["linked_thought"]) > 150 else r["linked_thought"],
                "created_at": r["created_at"].isoformat()
            })
        
        return {
            "success": True,
            "query": query,
            "count": len(entries),
            "entries": entries,
            "filters_applied": {
                "session_id": session_id,
                "log_type": log_type
            }
        }
        
    except Exception as e:
        logger.error(f"search_conversation_log failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def log_conversation(
    session_id: str,
    raw_text: str,
    log_type: str = "context",
    thought_node_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> dict[str, Any]:
    """
    Manually log free-form text to a session's conversation log.
    
    Use this to capture user context, feedback, or notes that aren't
    directly tied to hypothesis creation.
    
    Args:
        session_id: The reasoning session UUID
        raw_text: The text to log (no truncation)
        log_type: One of 'user_input', 'feedback', 'interview_answer', 'context' (default: 'context')
        thought_node_id: Optional - link to a specific thought node
        user_id: Optional - user identifier for multi-user scenarios
        
    Returns:
        Created log entry ID
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Verify session exists
        cur.execute("SELECT id FROM reasoning_sessions WHERE id = %s", (session_id,))
        if not cur.fetchone():
            return {"success": False, "error": "Session not found"}
        
        # Verify thought_node if provided
        if thought_node_id:
            cur.execute("SELECT id FROM thought_nodes WHERE id = %s", (thought_node_id,))
            if not cur.fetchone():
                return {"success": False, "error": "Thought node not found"}
        
        # Generate embedding (truncate for embedding, but store full text)
        embedding = get_embedding(raw_text[:2000])
        
        cur.execute(
            """
            INSERT INTO conversation_log (session_id, thought_node_id, user_id, log_type, raw_text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, created_at
            """,
            (session_id, thought_node_id, user_id, log_type, raw_text, embedding)
        )
        result = cur.fetchone()
        conn.commit()
        
        return {
            "success": True,
            "log_id": str(result["id"]),
            "session_id": session_id,
            "log_type": log_type,
            "text_length": len(raw_text),
            "created_at": result["created_at"].isoformat(),
            "message": f"Logged {len(raw_text)} characters to conversation_log"
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"log_conversation failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)

# =============================================================================
# v30: Global Codebase Understanding Tools
# =============================================================================

import hashlib
from pathlib import Path

# Language extensions for tree-sitter
LANGUAGE_EXTENSIONS = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.jsx': 'javascript',
    '.tsx': 'typescript',
    '.go': 'go',
    '.rs': 'rust',
    '.java': 'java',
    '.c': 'c',
    '.cpp': 'cpp',
    '.cs': 'c_sharp',
    '.rb': 'ruby',
    '.php': 'php',
}

# Patterns to skip during sync
SKIP_PATTERNS = {
    '__pycache__', 'node_modules', '.git', '.venv', 'venv',
    'dist', 'build', '.egg-info', '.tox', '.pytest_cache',
    '.mypy_cache', '.ruff_cache', 'coverage', 'htmlcov'
}


def _normalize_project_id(project_path: str) -> str:
    """Derive consistent project_id from path."""
    # Get the final directory name, lowercase alphanumeric only
    path = Path(project_path).resolve()
    name = path.name.lower()
    # Remove non-alphanumeric chars
    return ''.join(c for c in name if c.isalnum() or c == '_')


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file content."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract_symbols(content: str, language: str) -> list[dict]:
    """Extract function/class symbols using tree-sitter."""
    try:
        import tree_sitter_languages
    except ImportError:
        logger.warning("tree-sitter-languages not installed, skipping symbol extraction")
        return []
    
    try:
        parser = tree_sitter_languages.get_parser(language)
        tree = parser.parse(content.encode())
        
        symbols = []
        
        # Walk the tree looking for function/class definitions
        def walk_node(node, parent_name=None):
            node_type = node.type
            
            # Python-specific
            if node_type == 'function_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    sym = {
                        'type': 'function',
                        'name': name_node.text.decode(),
                        'line_start': node.start_point[0] + 1,
                        'line_end': node.end_point[0] + 1,
                        'signature': content[node.start_byte:node.end_byte].split('\n')[0],
                    }
                    # Extract docstring if present
                    if node.child_count > 0:
                        for child in node.children:
                            if child.type == 'expression_statement':
                                expr = child.child(0)
                                if expr and expr.type == 'string':
                                    sym['docstring'] = expr.text.decode().strip('"""\'\'\'')
                                    break
                    symbols.append(sym)
            
            elif node_type == 'class_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    sym = {
                        'type': 'class',
                        'name': name_node.text.decode(),
                        'line_start': node.start_point[0] + 1,
                        'line_end': node.end_point[0] + 1,
                        'signature': content[node.start_byte:node.end_byte].split('\n')[0],
                    }
                    symbols.append(sym)
            
            # JavaScript/TypeScript function
            elif node_type in ('function_declaration', 'method_definition', 'arrow_function'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols.append({
                        'type': 'function',
                        'name': name_node.text.decode(),
                        'line_start': node.start_point[0] + 1,
                        'line_end': node.end_point[0] + 1,
                        'signature': content[node.start_byte:node.end_byte].split('\n')[0][:200],
                    })
            
            # Recurse children
            for child in node.children:
                walk_node(child, parent_name)
        
        walk_node(tree.root_node)
        return symbols
        
    except Exception as e:
        logger.warning(f"Symbol extraction failed for {language}: {e}")
        return []


@mcp.tool()
async def sync_project(
    project_path: str,
    project_id: Optional[str] = None,
    max_files: int = 500,
    max_file_size_kb: int = 100
) -> dict[str, Any]:
    """
    Index a project directory for codebase understanding.
    
    Walks the directory, computes file hashes, extracts symbols via tree-sitter,
    and stores embeddings for semantic search. Uses project_id for isolation.
    
    Args:
        project_path: Absolute path to project root
        project_id: Optional custom project ID (auto-derived from path if not provided)
        max_files: Maximum files to index (default 500)
        max_file_size_kb: Skip files larger than this (default 100KB)
    
    Returns:
        Sync results with counts and any errors
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        path = Path(project_path).resolve()
        if not path.exists():
            return {"success": False, "error": f"Path does not exist: {project_path}"}
        
        # Normalize project_id
        pid = project_id or _normalize_project_id(project_path)
        
        # Get existing files for this project
        cur.execute(
            "SELECT file_path, file_hash FROM file_registry WHERE project_id = %s",
            (pid,)
        )
        existing = {row['file_path']: row['file_hash'] for row in cur.fetchall()}
        
        stats = {
            'files_scanned': 0,
            'files_added': 0,
            'files_updated': 0,
            'files_unchanged': 0,
            'files_skipped': 0,
            'symbols_extracted': 0,
            'errors': []
        }
        
        # Walk directory
        for file_path in path.rglob('*'):
            if stats['files_scanned'] >= max_files:
                break
            
            # Skip directories and non-files
            if not file_path.is_file():
                continue
            
            # Skip hidden and unwanted directories
            rel_parts = file_path.relative_to(path).parts
            if any(p in SKIP_PATTERNS or p.startswith('.') for p in rel_parts[:-1]):
                continue
            
            # Skip large files
            try:
                size_kb = file_path.stat().st_size / 1024
                if size_kb > max_file_size_kb:
                    stats['files_skipped'] += 1
                    continue
            except OSError:
                continue
            
            # Check extension for language support
            ext = file_path.suffix.lower()
            language = LANGUAGE_EXTENSIONS.get(ext)
            if not language:
                stats['files_skipped'] += 1
                continue
            
            stats['files_scanned'] += 1
            rel_path = str(file_path.relative_to(path))
            
            try:
                # Compute hash
                file_hash = _compute_file_hash(file_path)
                
                # Check if unchanged
                if rel_path in existing and existing[rel_path] == file_hash:
                    stats['files_unchanged'] += 1
                    continue
                
                # Read content
                content = file_path.read_text(encoding='utf-8', errors='replace')
                line_count = content.count('\n') + 1
                
                # Generate embedding for file content (chunk if needed)
                # Use first 2000 chars for embedding
                embedding = get_embedding(content[:2000])
                
                # Upsert file_registry
                cur.execute(
                    """
                    INSERT INTO file_registry (project_id, file_path, file_hash, language, line_count, content_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (project_id, file_path) DO UPDATE SET
                        file_hash = EXCLUDED.file_hash,
                        language = EXCLUDED.language,
                        line_count = EXCLUDED.line_count,
                        content_embedding = EXCLUDED.content_embedding,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    (pid, rel_path, file_hash, language, line_count, embedding)
                )
                file_id = cur.fetchone()['id']
                
                if rel_path in existing:
                    stats['files_updated'] += 1
                else:
                    stats['files_added'] += 1
                
                # Extract and store symbols
                symbols = _extract_symbols(content, language)
                
                # Clear old symbols for this file
                cur.execute("DELETE FROM file_symbols WHERE file_id = %s", (file_id,))
                
                for sym in symbols:
                    # Generate embedding for signature+docstring
                    embed_text = sym.get('signature', '') + '\n' + sym.get('docstring', '')
                    sym_embedding = get_embedding(embed_text[:500])
                    
                    cur.execute(
                        """
                        INSERT INTO file_symbols (file_id, symbol_type, symbol_name, line_start, line_end, signature, docstring, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (file_id, sym['type'], sym['name'], sym.get('line_start'), 
                         sym.get('line_end'), sym.get('signature'), sym.get('docstring'), sym_embedding)
                    )
                    stats['symbols_extracted'] += 1
                
            except Exception as e:
                stats['errors'].append(f"{rel_path}: {str(e)[:100]}")
                if len(stats['errors']) > 10:
                    break
        
        conn.commit()
        
        return {
            "success": True,
            "project_id": pid,
            "project_path": str(path),
            **stats,
            "message": f"Synced {stats['files_added']} new, {stats['files_updated']} updated files"
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"sync_project failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def query_codebase(
    query: str,
    project_id: str,
    top_k: int = 10,
    cross_project: bool = False,
    search_symbols: bool = True,
    search_files: bool = True
) -> dict[str, Any]:
    """
    Semantic search over indexed codebase.
    
    Searches file content and symbols using pgvector similarity.
    Uses project_id for isolation unless cross_project=True.
    
    Args:
        query: Natural language query (e.g., "finalize session logic")
        project_id: Project to search (required unless cross_project=True)
        top_k: Maximum results per category (default 10)
        cross_project: If True, search all projects (default False)
        search_symbols: Include symbol matches (default True)
        search_files: Include file content matches (default True)
    
    Returns:
        Matching files and symbols with scores
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Generate query embedding
        query_embedding = get_embedding(query)
        
        results = {
            "success": True,
            "query": query,
            "project_id": project_id if not cross_project else "ALL",
            "files": [],
            "symbols": []
        }
        
        # Search files
        if search_files:
            if cross_project:
                cur.execute(
                    """
                    SELECT project_id, file_path, language, line_count,
                           1 - (content_embedding <=> %s::vector) as similarity
                    FROM file_registry
                    WHERE content_embedding IS NOT NULL
                    ORDER BY content_embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, top_k)
                )
            else:
                cur.execute(
                    """
                    SELECT project_id, file_path, language, line_count,
                           1 - (content_embedding <=> %s::vector) as similarity
                    FROM file_registry
                    WHERE project_id = %s AND content_embedding IS NOT NULL
                    ORDER BY content_embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, project_id, query_embedding, top_k)
                )
            
            for row in cur.fetchall():
                results["files"].append({
                    "project_id": row['project_id'],
                    "file_path": row['file_path'],
                    "language": row['language'],
                    "line_count": row['line_count'],
                    "similarity": float(row['similarity'])
                })
        
        # Search symbols
        if search_symbols:
            if cross_project:
                cur.execute(
                    """
                    SELECT fr.project_id, fr.file_path, 
                           fs.symbol_type, fs.symbol_name, fs.line_start, fs.line_end,
                           fs.signature, fs.docstring,
                           1 - (fs.embedding <=> %s::vector) as similarity
                    FROM file_symbols fs
                    JOIN file_registry fr ON fs.file_id = fr.id
                    WHERE fs.embedding IS NOT NULL
                    ORDER BY fs.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, top_k)
                )
            else:
                cur.execute(
                    """
                    SELECT fr.project_id, fr.file_path, 
                           fs.symbol_type, fs.symbol_name, fs.line_start, fs.line_end,
                           fs.signature, fs.docstring,
                           1 - (fs.embedding <=> %s::vector) as similarity
                    FROM file_symbols fs
                    JOIN file_registry fr ON fs.file_id = fr.id
                    WHERE fr.project_id = %s AND fs.embedding IS NOT NULL
                    ORDER BY fs.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, project_id, query_embedding, top_k)
                )
            
            for row in cur.fetchall():
                results["symbols"].append({
                    "project_id": row['project_id'],
                    "file_path": row['file_path'],
                    "symbol_type": row['symbol_type'],
                    "symbol_name": row['symbol_name'],
                    "line_start": row['line_start'],
                    "line_end": row['line_end'],
                    "signature": row['signature'][:200] if row['signature'] else None,
                    "similarity": float(row['similarity'])
                })
        
        results["file_count"] = len(results["files"])
        results["symbol_count"] = len(results["symbols"])
        
        return results
        
    except Exception as e:
        logger.error(f"query_codebase failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Run the MCP server."""
    logger.info("Starting PAS (Scientific Reasoning) MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
