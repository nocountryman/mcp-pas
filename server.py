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
from typing import Any
from contextlib import asynccontextmanager

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
        ctx = Context.current()
        
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
        conn.commit()
        
        logger.info(f"Created reasoning session: {session_id}")
        
        return {
            "success": True,
            "session_id": str(row["id"]),
            "goal": row["goal"],
            "state": row["state"],
            "created_at": row["created_at"].isoformat(),
            "message": f"Reasoning session started. Use this session_id for subsequent reasoning operations."
        }
        
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
        
        return {
            "success": True,
            "session_id": session_id,
            "parent_node_id": parent_node_id,
            "parent_path": parent_path,
            "parent_content": parent_content,
            "goal": session["goal"],
            "relevant_laws": laws,
            "instructions": "Consider: What is requested? What files/modules might be affected? For each hypothesis, declare SCOPE as specific file paths. Optionally prefix with layer if helpful: [API] routes.py, [DB] models.py, [tests] test_auth.py. Generate 3 hypotheses with confidence (0.0-1.0). Call store_expansion(h1_text=..., h1_confidence=..., h1_scope='[layer] file1.py, file2.py', ...)."
        }
        
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
    h1_scope: str = None,
    # Hypothesis 2 (optional)
    h2_text: str = None,
    h2_confidence: float = None,
    h2_scope: str = None,
    # Hypothesis 3 (optional)
    h3_text: str = None,
    h3_confidence: float = None,
    h3_scope: str = None,
    # v7b: Revision tracking (borrowed from sequential thinking)
    is_revision: bool = False,
    revises_node_id: str = None
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
            hypothesis_text = hyp.get("hypothesis", "")
            llm_confidence = float(hyp.get("confidence", 0.5))
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
                import random
                
                total_weighted = 0.0
                total_similarity = 0.0
                supporting_law_ids = []
                law_names = []
                
                for law in matching_laws:
                    # v10a: Negation detection
                    hyp_negations = detect_negation(hypothesis_text)
                    law_negations = detect_negation(law["definition"]) if law["definition"] else set()
                    negation_asymmetry = bool(hyp_negations) != bool(law_negations)
                    negation_penalty = 0.15 if negation_asymmetry else 0.0
                    
                    # v12b: Thompson Sampling per law
                    selection_count = law.get("selection_count", 0) or 0
                    success_count = law.get("success_count", 0) or 0
                    alpha = success_count + 1
                    beta_param = selection_count - success_count + 1
                    thompson_sample = random.betavariate(alpha, beta_param)
                    
                    base_weight = float(law["scientific_weight"])
                    effective_weight = 0.7 * base_weight + 0.3 * thompson_sample
                    effective_weight = max(0.1, effective_weight - negation_penalty)
                    
                    # Weighted contribution: weight × similarity
                    similarity = float(law["similarity"])
                    total_weighted += effective_weight * similarity
                    total_similarity += similarity
                    
                    supporting_law_ids.append(law["id"])
                    law_names.append(law["law_name"])
                    
                    # v12b: Track law selection
                    cur.execute(
                        "UPDATE scientific_laws SET selection_count = selection_count + 1 WHERE id = %s",
                        (law["id"],)
                    )
                
                # v13a: Ensemble prior = Σ(weight × similarity) / Σ(similarity)
                prior = (total_weighted / total_similarity * 0.6) + (matching_laws[0]["similarity"] * 0.4)
                prior = max(0.1, min(0.95, prior))  # Clamp to valid range
                supporting_law = supporting_law_ids
                law_name = law_names[0]  # Primary law for display
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
                INSERT INTO thought_nodes (id, session_id, path, content, node_type, prior_score, likelihood, embedding, supporting_laws)
                VALUES (%s, %s, %s, %s, 'hypothesis', %s, %s, %s, %s)
                RETURNING id, path, prior_score, likelihood, posterior_score
                """,
                (node_id, session_id, new_path, hypothesis_text, prior, likelihood, hyp_emb, supporting_law)
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
        
        return {
            "success": True, 
            "session_id": session_id, 
            "created_nodes": created_nodes, 
            "count": len(created_nodes),
            "next_step": next_step,
            "confidence_nudge": confidence_nudge,
            "revision_info": revision_info,
            "revision_nudge": revision_nudge
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
async def prepare_critique(node_id: str) -> dict[str, Any]:
    """
    Prepare context for critiquing a thought node. Returns node content and 
    supporting laws so the LLM can generate counterarguments.
    
    After calling this, generate a critique and pass it to store_critique().
    
    Args:
        node_id: The thought node UUID to critique
        
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
        suggested_critique = None
        try:
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
            
            response = await mcp.request_context.session.create_message(
                CreateMessageRequestParams(
                    messages=[SamplingMessage(role="user", content=TextContent(type="text", text=critique_prompt))],
                    max_tokens=500,
                    system_prompt="You are a hypothesis critic. Return only valid JSON."
                )
            )
            
            if response and response.content and response.content.text:
                response_text = response.content.text.strip()
                # Handle markdown code blocks
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.strip()
                suggested_critique = json.loads(response_text)
                logger.info(f"v16c: Generated suggested critique for node {node_id}")
        except Exception as e:
            logger.warning(f"v16c critique generation failed: {e}")
        
        return {
            "success": True,
            "node_id": node_id,
            "path": node["path"],
            "node_content": node["content"],
            "session_goal": node["goal"],
            "current_scores": {
                "prior": float(node["prior_score"]),
                "likelihood": float(node["likelihood"]),
                "posterior": float(node["posterior_score"]) if node["posterior_score"] else None
            },
            "supporting_laws": laws_text,
            # v16c: LLM-generated critique suggestion
            "suggested_critique": suggested_critique,
            # v14c.1: Enhanced instructions with failure mode guidance
            "instructions": "Challenge this hypothesis. Use the failure_modes from supporting laws as targeted challenge prompts. What could go wrong? What would make it STRONGER? Generate critique with counterargument and severity_score (0.0-1.0). Call store_critique(node_id=..., counterargument=..., severity_score=..., logical_flaws='flaw1, flaw2', edge_cases='case1, case2').",
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
        historical_questions = []
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
        detected_domains = []
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
        goal_questions = []
        try:
            goal_prompt = f"""Analyze this goal and decide if clarifying questions are needed:

GOAL: {goal}

DECISION RULES:
1. If goal is SPECIFIC (names files/functions, has clear constraints) → return []
2. If goal is AMBIGUOUS (multiple valid interpretations) → return 1-2 questions max
3. If goal is OPEN-ENDED (design/architecture/planning) → return 2-3 questions max

EXAMPLES OF SPECIFIC GOALS (return []):
- "Fix bug in auth.py line 45"
- "Add logging to UserService.create_user()"
- "Refactor parse_terminal_output to handle '0 failed'"

EXAMPLES OF AMBIGUOUS GOALS (return 1-2 questions):
- "Improve performance" → ask about which endpoints, target latency
- "Add authentication" → ask about auth type, user scope
- "Fix the login issue" → ask for repro steps

For each question, focus on BUSINESS CONTEXT only:
- Scope boundaries (what's in/out?)
- Success criteria (how will you know it works?)
- User context (who is this for?)

Do NOT ask about code quality, architecture, or testing strategy.

Return format:
[{{"question_text": "...", "choices": [{{"label": "A", "description": "...", "pros": ["..."], "cons": ["..."]}}]}}]

Return [] if the goal is already specific enough. Return ONLY the JSON array."""
            
            # Use sampling to generate questions
            from mcp.types import SamplingMessage, TextContent
            response = await mcp.request_context.session.create_message(
                CreateMessageRequestParams(
                    messages=[SamplingMessage(role="user", content=TextContent(type="text", text=goal_prompt))],
                    max_tokens=1000,
                    system_prompt="You are a structured question generator. Return only valid JSON arrays."
                )
            )
            
            if response and response.content and response.content.text:
                # Try to parse JSON from response
                import json as json_parser
                response_text = response.content.text.strip()
                # Handle potential markdown code blocks
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.strip()
                
                parsed = json_parser.loads(response_text)
                for i, q in enumerate(parsed[:3]):  # Limit to 3 questions
                    goal_questions.append({
                        "id": f"goal_{i+1}",
                        "question_text": q.get("question_text", ""),
                        "question_type": "single_choice",
                        "choices": q.get("choices", []),
                        "priority": 3 + i,  # Priority 3-5, after historical but before generic
                        "depth": 1,
                        "depends_on": [],
                        "follow_up_rules": [],
                        "answered": False,
                        "answer": None,
                        "source": "goal_derived"
                    })
                logger.info(f"v16d.1: Generated {len(goal_questions)} goal-derived questions")
        except Exception as e:
            logger.warning(f"v16d.1 goal question generation failed: {e}")
        
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
        
        # Store in history
        interview["answer_history"].append({
            "question_id": question_id,
            "question_text": question["question_text"],
            "answer": answer,
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
# v20: Adaptive Depth Quality Metrics
# =============================================================================

def compute_quality_metrics(
    cur,
    session_id: str,
    candidates: list,
    gap: float,
    thresholds: dict = None
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
    critiqued_count = 0
    uncritiqued_nodes = []
    for c in candidates[:3]:  # Check top 3
        node_id = c.get("node_id") or c.get("id")
        if node_id:
            cur.execute("""
                SELECT COUNT(*) as cnt FROM thought_critiques WHERE node_id = %s
            """, (str(node_id),))
            result = cur.fetchone()
            if result and result["cnt"] > 0:
                critiqued_count += 1
            else:
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
    terminal_output: str = None  # v17b: RLVR auto-record
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
            original_score = float(node["posterior_score"])
            adjusted_score = original_score
            penalties = []
            
            # Unchallenged penalty: if likelihood is suspiciously round
            # (never critiqued nodes keep their initial likelihood)
            likelihood = float(node["likelihood"])
            # Check if likelihood looks "untouched" (ends in .0, .5, etc.)
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
                # If multiple siblings but only 1 unique law = monoculture
                if total_siblings >= 2 and unique_laws == 1:
                    adjusted_score -= HEURISTIC_PENALTIES["monoculture"]
                    penalties.append("monoculture_penalty")
            
            # Ensure score stays in valid range
            adjusted_score = max(0.1, min(1.0, adjusted_score))
            
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
            
            # v11b: Blend adjusted score with rollout score
            final_score = (1 - ROLLOUT_WEIGHT) * adjusted_score + ROLLOUT_WEIGHT * rollout_score
            
            processed.append({
                "node_id": str(node["id"]),
                "path": node["path"],
                "content": node["content"],
                "depth": depth,
                "prior_score": round(float(node["prior_score"]), 4),      # v14a.1
                "likelihood": round(float(node["likelihood"]), 4),        # v14a.1
                "confidence": round(float(node["prior_score"]) * float(node["likelihood"]), 4),  # v14a.1
                "original_score": round(original_score, 4),
                "adjusted_score": round(adjusted_score, 4),
                "rollout_score": round(rollout_score, 4),  # v11b
                "final_score": round(final_score, 4),      # v11b
                "penalties_applied": penalties
            })
        
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
        
        # v11a: UCT tiebreaking for close decisions
        uct_applied = False
        if runner_up:
            gap = recommendation["final_score"] - runner_up["final_score"]
            avg = (recommendation["final_score"] + runner_up["final_score"]) / 2
            relative_gap = gap / avg if avg > 0 else 0
            
            # v11a: If gap < threshold, apply UCT exploration bonus
            if gap < UCT_THRESHOLD:
                import math
                # Count approx visits based on depth (deeper = more iterations)
                rec_visits = max(1, recommendation["depth"])
                run_visits = max(1, runner_up["depth"])
                total_visits = rec_visits + run_visits
                
                # UCT: Q + C * sqrt(ln(N)/n)
                rec_uct = recommendation["final_score"] + UCT_EXPLORATION_C * math.sqrt(math.log(total_visits) / rec_visits)
                run_uct = runner_up["final_score"] + UCT_EXPLORATION_C * math.sqrt(math.log(total_visits) / run_visits)
                
                # If UCT favors runner-up, swap
                if run_uct > rec_uct:
                    recommendation, runner_up = runner_up, recommendation
                    processed[0], processed[1] = processed[1], processed[0]
                    uct_applied = True
                    gap = recommendation["final_score"] - runner_up["final_score"]
            
            if gap < 0.03:
                decision_quality = "low"
                gap_analysis = f"Very close decision (gap: {gap:.3f}). Consider both options."
            elif gap < 0.10:
                decision_quality = "medium"
                gap_analysis = f"Moderate confidence (gap: {gap:.3f}). Winner is better but runner-up has merit."
            else:
                decision_quality = "high"
                gap_analysis = f"Clear winner (gap: {gap:.3f}). High confidence in recommendation."
            
            # v14a.1: Confidence-weighted gap - lower confidence reduces decision quality
            rec_conf = recommendation.get("confidence", 0.5)
            run_conf = runner_up.get("confidence", 0.5)
            weighted_gap = gap * min(rec_conf, run_conf)
            
            # v14a.1: Downgrade decision quality if weighted gap is too low
            if decision_quality == "high" and weighted_gap < 0.05:
                decision_quality = "medium"
                gap_analysis += f" [v14a.1: Downgraded - low confidence (wgap: {weighted_gap:.3f})]"
            elif decision_quality == "medium" and weighted_gap < 0.02:
                decision_quality = "low"
                gap_analysis += f" [v14a.1: Downgraded - low confidence (wgap: {weighted_gap:.3f})]"
            
            if uct_applied:
                gap_analysis += " [v11a: UCT exploration applied]"
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
        
        return {
            "success": True,
            "session_id": session_id,
            "recommendation": {
                "node_id": recommendation["node_id"],
                "content": recommendation["content"],
                "original_score": recommendation["original_score"],
                "adjusted_score": recommendation["adjusted_score"],
                "penalties_applied": recommendation["penalties_applied"]
            },
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
            "deepen_suggestions": quality_result["suggestions"] if not quality_result["sufficient"] else []
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
async def record_outcome(
    session_id: str,
    outcome: str,
    confidence: float = 1.0,
    notes: str = None,
    failure_reason: str = None,  # v15b: explicit failure reason for learning
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
        
        # Record the outcome with v15b fields
        cur.execute(
            """
            INSERT INTO outcome_records (session_id, outcome, confidence, winning_path, notes, failure_reason, scope_embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, created_at
            """,
            (session_id, outcome, confidence, winning_path, notes, failure_reason, scope_embedding)
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
            is_winner = str(cnode["path"]).startswith(str(winning_path)) or str(winning_path).startswith(str(cnode["path"]))
            # Critique was accurate if: node was critiqued AND (failure if winner OR success if not winner)
            # Simple heuristic: critique severity implied caution, was it warranted?
            critique_accurate = None
            if is_winner:
                # If winner was critiqued and outcome is failure, critique was accurate
                critique_accurate = (outcome == 'failure')
            else:
                # If non-winner was critiqued and outcome is success, critique may have been accurate
                critique_accurate = (outcome == 'success')
            
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
            "message": f"Outcome recorded. {stats['node_count'] or 0} nodes in winning path." + 
                      (" Session auto-completed." if session_completed else "") +
                      (f" Auto-refreshed {auto_refresh_result.get('laws_updated', 0)} law weights." if auto_refresh_result else "")
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
            result["message"] += f" Auto-recorded as {signal}."
            if failure_reason:
                result["message"] += f" Failure reason extracted."
        except Exception as e:
            result["auto_recorded"] = False
            result["auto_record_error"] = str(e)
    else:
        result["auto_recorded"] = False
        if auto_record and signal == "unknown":
            result["message"] += " Signal too ambiguous for auto-record."
        elif auto_record and confidence < 0.7:
            result["message"] += f" Confidence too low ({confidence:.0%}) for auto-record."
    
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
    notes: str = None
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
# Entry Point
# =============================================================================

def main():
    """Run the MCP server."""
    logger.info("Starting PAS (Scientific Reasoning) MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
