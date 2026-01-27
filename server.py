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


# =============================================================================
# MCP Server Setup
# =============================================================================

mcp = FastMCP(
    name="pas-server",
)


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
        
        # Insert the new session
        cur.execute(
            """
            INSERT INTO reasoning_sessions (id, goal, state, context)
            VALUES (%s, %s, 'active', %s)
            RETURNING id, goal, state, created_at
            """,
            (session_id, user_goal.strip(), json.dumps({"source": "mcp_tool"}))
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
            conn.close()


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
            conn.close()


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
            conn.close()


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
            "instructions": "Before generating hypotheses, briefly consider: What is explicitly requested? Is this creation or modification? (if relevant) What else might be affected? Keep this brief. Then generate 3 hypotheses with confidence (0.0-1.0). If confidence < 0.7, note what would increase it. Call store_expansion(h1_text=..., h1_confidence=..., h2_text=..., h2_confidence=..., h3_text=..., h3_confidence=...)."
        }
        
    except Exception as e:
        logger.error(f"prepare_expansion failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            conn.close()


@mcp.tool()
async def store_expansion(
    session_id: str,
    parent_node_id: str | None,
    # Hypothesis 1 (required)
    h1_text: str,
    h1_confidence: float,
    # Hypothesis 2 (optional)
    h2_text: str = None,
    h2_confidence: float = None,
    # Hypothesis 3 (optional)
    h3_text: str = None,
    h3_confidence: float = None
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
        h2_text: Second hypothesis text (optional)
        h2_confidence: Second hypothesis confidence 0.0-1.0 (optional)
        h3_text: Third hypothesis text (optional)
        h3_confidence: Third hypothesis confidence 0.0-1.0 (optional)
        
    Returns:
        Created nodes with Bayesian posterior scores
    """
    try:
        # Build hypotheses list from flattened params
        hypotheses = []
        if h1_text:
            hypotheses.append({"hypothesis": h1_text, "confidence": h1_confidence or 0.5})
        if h2_text:
            hypotheses.append({"hypothesis": h2_text, "confidence": h2_confidence or 0.5})
        if h3_text:
            hypotheses.append({"hypothesis": h3_text, "confidence": h3_confidence or 0.5})
        
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
            
            if not hypothesis_text:
                continue
            
            # Generate embedding and find similar law
            hyp_emb = get_embedding(hypothesis_text)
            cur.execute(
                """
                SELECT id, law_name, scientific_weight, 1 - (embedding <=> %s::vector) as similarity
                FROM scientific_laws WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector LIMIT 1
                """,
                (hyp_emb, hyp_emb)
            )
            law = cur.fetchone()
            
            # Calculate Bayesian scores
            if law:
                prior = (float(law["scientific_weight"]) * 0.6) + (float(law["similarity"]) * 0.4)
                supporting_law = [law["id"]]
                law_name = law["law_name"]
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
                "supporting_law": law_name
            })
        
        conn.commit()
        
        # Workflow nudge: Find top hypothesis and suggest critique
        top_node = max(created_nodes, key=lambda n: n.get("posterior_score") or 0) if created_nodes else None
        next_step = None
        if top_node:
            next_step = f"Challenge your top hypothesis. Call prepare_critique(node_id='{top_node['node_id']}')"
        
        return {
            "success": True, 
            "session_id": session_id, 
            "created_nodes": created_nodes, 
            "count": len(created_nodes),
            "next_step": next_step
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"store_expansion failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            conn.close()


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
            cur.execute("SELECT law_name, definition FROM scientific_laws WHERE id = ANY(%s)", (node["supporting_laws"],))
            for law in cur.fetchall():
                laws_text.append({"law_name": law["law_name"], "definition": law["definition"]})
        
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
            "instructions": "Challenge this hypothesis: What could be wrong? What would make it STRONGER? Generate critique with counterargument and severity_score (0.0-1.0). Balance criticism with constructive improvement. Call store_critique(node_id=..., counterargument=..., severity_score=..., logical_flaws='flaw1, flaw2', edge_cases='case1, case2')."
        }
        
    except Exception as e:
        logger.error(f"prepare_critique failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            conn.close()


@mcp.tool()

async def store_critique(
    node_id: str,
    counterargument: str,
    severity_score: float,
    logical_flaws: str = "",
    edge_cases: str = ""
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
        new_likelihood = max(0.1, old_likelihood * (1 - severity * 0.5))
        
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
            }
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"store_critique failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            conn.close()


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
            conn.close()


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
            conn.close()


# =============================================================================
# Interview Tools - Smart Question Flow (Phase 2)
# =============================================================================

DEFAULT_INTERVIEW_CONFIG = {
    "max_questions": 15,
    "max_depth": 3,
    "questions_answered": 0,
    "questions_remaining": 0
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
        
        # Analyze goal to generate questions
        # For now, generate standard UX/design questions as template
        # In future: use LLM to generate context-specific questions
        
        questions = [
            {
                "id": "q1",
                "question_text": "What is the primary platform for this solution?",
                "question_type": "single_choice",
                "choices": [
                    {"label": "A", "description": "Desktop-first web application",
                     "pros": ["Maximum screen space", "Complex UI possible"], 
                     "cons": ["Not mobile accessible"]},
                    {"label": "B", "description": "Mobile-first responsive",
                     "pros": ["Accessible anywhere", "Touch-optimized"], 
                     "cons": ["Limited screen space"]},
                    {"label": "C", "description": "Desktop + Tablet hybrid",
                     "pros": ["Best of both", "Flexible deployment"], 
                     "cons": ["More design work required"]}
                ],
                "priority": 10,
                "depth": 1,
                "depends_on": [],
                "follow_up_rules": [
                    {
                        "when_answer": "C",
                        "inject": {
                            "id": "q1a",
                            "question_text": "Which should be the PRIMARY design target?",
                            "question_type": "single_choice",
                            "choices": [
                                {"label": "A", "description": "Desktop-first, adapt to tablet",
                                 "pros": ["Easier complex layouts"], "cons": ["May feel cramped on tablet"]},
                                {"label": "B", "description": "Tablet-first, scale to desktop",
                                 "pros": ["Touch-friendly base"], "cons": ["May underuse desktop space"]}
                            ],
                            "priority": 15,
                            "depth": 2,
                            "depends_on": ["q1"]
                        }
                    }
                ],
                "answered": False,
                "answer": None
            },
            {
                "id": "q2",
                "question_text": "What is the primary user expertise level?",
                "question_type": "single_choice",
                "choices": [
                    {"label": "A", "description": "Beginners / Students",
                     "pros": ["Simple interface", "Guided experience"], 
                     "cons": ["May frustrate power users"]},
                    {"label": "B", "description": "Professionals",
                     "pros": ["Dense information", "Advanced features"], 
                     "cons": ["Steep learning curve"]},
                    {"label": "C", "description": "Mixed (beginners to pros)",
                     "pros": ["Wide audience"], 
                     "cons": ["Must support progressive disclosure"]}
                ],
                "priority": 20,
                "depth": 1,
                "depends_on": [],
                "follow_up_rules": [],
                "answered": False,
                "answer": None
            },
            {
                "id": "q3",
                "question_text": "What visual style best fits the product?",
                "question_type": "single_choice",
                "choices": [
                    {"label": "A", "description": "Clean / Minimal (like Linear, Apple)",
                     "pros": ["Timeless", "Professional"], "cons": ["Can feel cold"]},
                    {"label": "B", "description": "Playful / Approachable (like Canva, Notion)",
                     "pros": ["Welcoming", "Reduces anxiety"], "cons": ["May not feel serious"]},
                    {"label": "C", "description": "Dark / Professional (like Figma, DaVinci)",
                     "pros": ["Familiar to pros"], "cons": ["Can intimidate beginners"]},
                    {"label": "D", "description": "Organic / Warm (earthy, handcrafted)",
                     "pros": ["Unique, indie spirit"], "cons": ["Niche taste"]}
                ],
                "priority": 30,
                "depth": 1,
                "depends_on": [],
                "follow_up_rules": [],
                "answered": False,
                "answer": None
            }
        ]
        
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
            "message": "Interview questions generated. Call get_next_question to start."
        }
        
    except Exception as e:
        logger.error(f"identify_gaps failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            conn.close()


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
            conn.close()


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
            conn.close()


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
            conn.close()


# =============================================================================
# Finalization Tool - Auto-Critique & Recommendation (Phase 3)
# =============================================================================

HEURISTIC_PENALTIES = {
    "unchallenged": 0.10,      # Never critiqued
    "shallow_alternatives": 0.05,  # <2 siblings at same level
}

DEPTH_BONUS_PER_LEVEL = 0.02  # Reward deeper refinement


@mcp.tool()
async def finalize_session(
    session_id: str,
    top_n: int = 3,
    deep_critique: bool = False
) -> dict[str, Any]:
    """
    Finalize a reasoning session by auto-critiquing top hypotheses.
    
    Applies heuristic penalties (unchallenged, shallow tree), compares
    top candidates, and returns a final recommendation with confidence.
    
    Args:
        session_id: The reasoning session UUID
        top_n: Number of top candidates to consider (default: 3)
        deep_critique: If True, returns critique requests for LLM
        
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
            
            # Ensure score stays in valid range
            adjusted_score = max(0.1, min(1.0, adjusted_score))
            
            processed.append({
                "node_id": str(node["id"]),
                "path": node["path"],
                "content": node["content"],
                "depth": depth,
                "original_score": round(original_score, 4),
                "adjusted_score": round(adjusted_score, 4),
                "penalties_applied": penalties
            })
        
        # Sort by adjusted score
        processed.sort(key=lambda x: x["adjusted_score"], reverse=True)
        
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
        
        if runner_up:
            gap = recommendation["adjusted_score"] - runner_up["adjusted_score"]
            avg = (recommendation["adjusted_score"] + runner_up["adjusted_score"]) / 2
            relative_gap = gap / avg if avg > 0 else 0
            
            if gap < 0.03:
                decision_quality = "low"
                gap_analysis = f"Very close decision (gap: {gap:.3f}). Consider both options."
            elif gap < 0.10:
                decision_quality = "medium"
                gap_analysis = f"Moderate confidence (gap: {gap:.3f}). Winner is better but runner-up has merit."
            else:
                decision_quality = "high"
                gap_analysis = f"Clear winner (gap: {gap:.3f}). High confidence in recommendation."
        else:
            decision_quality = "medium"
            gap_analysis = "Only one candidate available."
        
        # Get interview context summary if available
        context = session["context"] or {}
        interview = context.get("interview", {})
        context_summary = None
        if interview.get("answer_history"):
            context_summary = {
                h["question_id"]: h["answer"] 
                for h in interview["answer_history"]
            }
        
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
            "context_summary": context_summary
        }
        
    except Exception as e:
        logger.error(f"finalize_session failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            conn.close()


# =============================================================================
# Self-Learning Tools (Phase 4)
# =============================================================================

@mcp.tool()
async def record_outcome(
    session_id: str,
    outcome: str,
    confidence: float = 1.0,
    notes: str = None,
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
        keep_open: If True, don't auto-complete even on success/failure
        
    Returns:
        Confirmation with stats about attributed nodes
    """

    if outcome not in ('success', 'partial', 'failure'):
        return {"success": False, "error": "outcome must be 'success', 'partial', or 'failure'"}
    
    confidence = max(0.0, min(1.0, confidence))
    
    try:
        conn = get_db_connection()
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
        
        # Record the outcome
        cur.execute(
            """
            INSERT INTO outcome_records (session_id, outcome, confidence, winning_path, notes)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, created_at
            """,
            (session_id, outcome, confidence, winning_path, notes)
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
        
        # Auto-complete session if outcome is definitive (not partial) and not keep_open
        session_completed = False
        if outcome in ('success', 'failure') and not keep_open:
            cur.execute(
                "UPDATE reasoning_sessions SET state = 'completed' WHERE id = %s",
                (session_id,)
            )
            session_completed = True
        
        conn.commit()
        
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
            "message": f"Outcome recorded. {stats['node_count'] or 0} nodes in winning path." + 
                      (" Session auto-completed." if session_completed else "")
        }
        
    except Exception as e:
        logger.error(f"record_outcome failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            conn.close()


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
            conn.close()


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
            conn.close()


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
            conn.close()


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
            conn.close()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Run the MCP server."""
    logger.info("Starting PAS (Scientific Reasoning) MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
