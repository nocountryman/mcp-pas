#!/usr/bin/env python3
"""
Reasoning Engine - Core logic for Bayesian Tree of Thoughts

This module implements the "Thinking" and "Antithesis" modes:
- expand_thought: Generates child hypotheses using LLM + scientific law grounding
- critique_thought: Attacks a specific node to find weaknesses
"""

import os
import json
import uuid
import logging
import re
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("pas-engine")

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


# Use singleton embedding model from utils
from pas.utils import get_embedding


# =============================================================================
# Thought Expansion - "Thesis" Mode
# =============================================================================

EXPANSION_PROMPT = """You are a scientific reasoning assistant. Given the current thought, generate exactly 3 possible next steps or hypotheses.

Current Thought:
{parent_content}

Context/Goal:
{session_goal}

For each hypothesis, provide:
1. A clear statement of the hypothesis
2. Your confidence score (0.0-1.0) in this direction

Format your response as a JSON array:
[
  {{"hypothesis": "First possible direction...", "confidence": 0.8}},
  {{"hypothesis": "Second possible direction...", "confidence": 0.7}},
  {{"hypothesis": "Third possible direction...", "confidence": 0.6}}
]

Only output the JSON array, no other text."""


async def expand_thought(
    session_id: str,
    parent_node_id: str | None = None,
    sample_agent_fn = None
) -> dict[str, Any]:
    """
    Expand a thought node by generating 3 child hypotheses using LLM + scientific grounding.
    
    Steps:
    1. Retrieve parent thought from DB
    2. Use sample_agent to generate 3 possible next steps
    3. For each step:
       - Search scientific_laws for most similar law
       - Calculate Bayesian posterior: (law_weight * llm_confidence) / marginal
       - Insert into thought_nodes with correct ltree path
    
    Args:
        session_id: The reasoning session UUID
        parent_node_id: The parent thought UUID (None for root)
        sample_agent_fn: The sample_agent function to call LLM
        
    Returns:
        Dictionary with created child nodes
    """
    if sample_agent_fn is None:
        from server import sample_agent
        sample_agent_fn = sample_agent
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get session details
        cur.execute(
            "SELECT id, goal, state FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": f"Session {session_id} not found"}
        
        if session["state"] != "active":
            return {"success": False, "error": f"Session is {session['state']}, not active"}
        
        # Get parent thought (or create root context)
        if parent_node_id:
            cur.execute(
                "SELECT id, path, content, depth FROM thought_nodes WHERE id = %s AND session_id = %s",
                (parent_node_id, session_id)
            )
            parent = cur.fetchone()
            if not parent:
                return {"success": False, "error": f"Parent node {parent_node_id} not found"}
            
            parent_content = parent["content"]
            parent_path = parent["path"]
        else:
            # Create root node if none exists
            parent_content = session["goal"]
            parent_path = None
            
            # Check if root already exists
            cur.execute(
                "SELECT id, path FROM thought_nodes WHERE session_id = %s AND path = 'root'",
                (session_id,)
            )
            existing_root = cur.fetchone()
            if existing_root:
                parent_node_id = str(existing_root["id"])
                parent_path = existing_root["path"]
            else:
                # Create root node
                root_id = str(uuid.uuid4())
                root_embedding = get_embedding(parent_content)
                cur.execute(
                    """
                    INSERT INTO thought_nodes 
                    (id, session_id, path, content, node_type, prior_score, likelihood, embedding)
                    VALUES (%s, %s, 'root', %s, 'root', 0.5, 0.5, %s)
                    RETURNING id, path
                    """,
                    (root_id, session_id, parent_content, root_embedding)
                )
                root = cur.fetchone()
                parent_node_id = str(root["id"])
                parent_path = "root"
                conn.commit()
        
        # Generate child hypotheses via LLM
        prompt = EXPANSION_PROMPT.format(
            parent_content=parent_content,
            session_goal=session["goal"]
        )
        
        try:
            llm_response = await sample_agent_fn(
                prompt=prompt,
                system_prompt="You are a scientific reasoning assistant. Output only valid JSON.",
                temperature=0.7
            )
        except Exception as e:
            return {"success": False, "error": f"LLM sampling failed: {e}"}
        
        # Parse LLM response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\[[\s\S]*\]', llm_response)
            if json_match:
                hypotheses = json.loads(json_match.group())
            else:
                hypotheses = json.loads(llm_response)
        except json.JSONDecodeError as e:
            return {
                "success": False, 
                "error": f"Failed to parse LLM response as JSON: {e}",
                "raw_response": llm_response[:500]
            }
        
        if not isinstance(hypotheses, list) or len(hypotheses) == 0:
            return {"success": False, "error": "LLM returned empty or invalid hypotheses"}
        
        # Process each hypothesis
        created_nodes = []
        
        for i, hyp in enumerate(hypotheses[:3]):  # Limit to 3
            hypothesis_text = hyp.get("hypothesis", "")
            llm_confidence = float(hyp.get("confidence", 0.5))
            
            if not hypothesis_text:
                continue
            
            # Generate embedding for the hypothesis
            hyp_embedding = get_embedding(hypothesis_text)
            
            # Find most similar scientific law
            cur.execute(
                """
                SELECT 
                    id, law_name, definition, scientific_weight,
                    1 - (embedding <=> %s::vector) as similarity
                FROM scientific_laws
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (hyp_embedding, hyp_embedding)
            )
            law = cur.fetchone()
            
            # Calculate Bayesian posterior
            if law:
                law_weight = float(law["scientific_weight"])
                law_similarity = float(law["similarity"])
                
                # Prior: weighted combination of law weight and similarity
                prior = (law_weight * 0.6) + (law_similarity * 0.4)
                
                # Likelihood: LLM confidence
                likelihood = llm_confidence
                
                # Marginal (normalization): P(E) = P(E|H)*P(H) + P(E|¬H)*P(¬H)
                marginal = (likelihood * prior) + ((1 - likelihood) * (1 - prior))
                
                # Posterior via Bayes' theorem
                if marginal > 0:
                    posterior = (likelihood * prior) / marginal
                else:
                    posterior = 0.5
                
                supporting_law_id = law["id"]
                supporting_law_name = law["law_name"]
            else:
                prior = 0.5
                likelihood = llm_confidence
                posterior = llm_confidence  # Fallback
                supporting_law_id = None
                supporting_law_name = None
            
            # Build ltree path
            node_id = str(uuid.uuid4())
            child_label = f"h{i+1}"  # h1, h2, h3
            new_path = f"{parent_path}.{child_label}"
            
            # Insert the new thought node
            cur.execute(
                """
                INSERT INTO thought_nodes 
                (id, session_id, path, content, node_type, prior_score, likelihood, embedding, supporting_laws)
                VALUES (%s, %s, %s, %s, 'hypothesis', %s, %s, %s, %s)
                RETURNING id, path, prior_score, likelihood, posterior_score
                """,
                (
                    node_id, session_id, new_path, hypothesis_text,
                    prior, likelihood, hyp_embedding,
                    [supporting_law_id] if supporting_law_id else []
                )
            )
            
            new_node = cur.fetchone()
            
            created_nodes.append({
                "node_id": str(new_node["id"]),
                "path": new_node["path"],
                "content": hypothesis_text[:200] + "..." if len(hypothesis_text) > 200 else hypothesis_text,
                "prior_score": float(new_node["prior_score"]),
                "likelihood": float(new_node["likelihood"]),
                "posterior_score": float(new_node["posterior_score"]) if new_node["posterior_score"] else None,
                "supporting_law": supporting_law_name,
                "llm_confidence": llm_confidence
            })
        
        conn.commit()
        
        return {
            "success": True,
            "session_id": session_id,
            "parent_node_id": parent_node_id,
            "parent_path": parent_path,
            "created_nodes": created_nodes,
            "count": len(created_nodes)
        }
        
    except Exception as e:
        conn.rollback()
        logger.error(f"expand_thought failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


# =============================================================================
# Thought Critique - "Antithesis" Mode
# =============================================================================

CRITIQUE_PROMPT = """You are a rigorous scientific critic. Your job is to find weaknesses, flaws, and counterarguments to the following hypothesis.

Hypothesis to Attack:
{node_content}

Original Goal:
{session_goal}

Supporting Scientific Law:
{supporting_law}

Instructions:
1. Identify the strongest counterargument to this hypothesis
2. Find potential logical flaws or unstated assumptions
3. Consider edge cases where this hypothesis would fail
4. Rate the severity of issues found (0.0 = minor, 1.0 = fatal flaw)

Format your response as JSON:
{{
  "counterargument": "The main counterargument is...",
  "logical_flaws": ["Flaw 1", "Flaw 2"],
  "edge_cases": ["Edge case 1", "Edge case 2"],
  "severity_score": 0.6,
  "recommendation": "reject|weak|moderate|strong|accept"
}}

Only output the JSON, no other text."""


async def critique_thought(
    node_id: str,
    sample_agent_fn = None
) -> dict[str, Any]:
    """
    Critique a thought node by having the LLM attack it ("Antithesis" mode).
    
    This function:
    1. Retrieves the target thought and its context
    2. Asks the LLM to find counterarguments and weaknesses
    3. Updates the node's likelihood based on critique severity
    4. Returns the critique analysis
    
    Args:
        node_id: The thought node UUID to critique
        sample_agent_fn: The sample_agent function to call LLM
        
    Returns:
        Dictionary with critique results and updated scores
    """
    if sample_agent_fn is None:
        from server import sample_agent
        sample_agent_fn = sample_agent
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get the target node
        cur.execute(
            """
            SELECT t.id, t.session_id, t.path, t.content, t.prior_score, t.likelihood, 
                   t.posterior_score, t.supporting_laws,
                   s.goal as session_goal
            FROM thought_nodes t
            JOIN reasoning_sessions s ON t.session_id = s.id
            WHERE t.id = %s
            """,
            (node_id,)
        )
        node = cur.fetchone()
        
        if not node:
            return {"success": False, "error": f"Node {node_id} not found"}
        
        # Get supporting law details
        supporting_law_text = "No supporting law found"
        if node["supporting_laws"]:
            cur.execute(
                "SELECT law_name, definition FROM scientific_laws WHERE id = ANY(%s)",
                (node["supporting_laws"],)
            )
            laws = cur.fetchall()
            if laws:
                supporting_law_text = "; ".join([
                    f"{l['law_name']}: {l['definition'][:100]}..." 
                    for l in laws
                ])
        
        # Generate critique via LLM
        prompt = CRITIQUE_PROMPT.format(
            node_content=node["content"],
            session_goal=node["session_goal"],
            supporting_law=supporting_law_text
        )
        
        try:
            llm_response = await sample_agent_fn(
                prompt=prompt,
                system_prompt="You are a rigorous scientific critic. Output only valid JSON.",
                temperature=0.5  # Lower temperature for more focused critique
            )
        except Exception as e:
            return {"success": False, "error": f"LLM sampling failed: {e}"}
        
        # Parse critique response
        try:
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                critique = json.loads(json_match.group())
            else:
                critique = json.loads(llm_response)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse critique response: {e}",
                "raw_response": llm_response[:500]
            }
        
        # Extract critique components
        counterargument = critique.get("counterargument", "")
        logical_flaws = critique.get("logical_flaws", [])
        edge_cases = critique.get("edge_cases", [])
        severity_score = float(critique.get("severity_score", 0.5))
        recommendation = critique.get("recommendation", "moderate")
        
        # Update likelihood based on critique severity
        # Higher severity = lower likelihood
        current_likelihood = float(node["likelihood"])
        severity_penalty = severity_score * 0.5  # Max 50% reduction
        new_likelihood = max(0.1, current_likelihood * (1 - severity_penalty))
        
        # Update the node
        cur.execute(
            """
            UPDATE thought_nodes 
            SET likelihood = %s, updated_at = NOW()
            WHERE id = %s
            RETURNING id, prior_score, likelihood, posterior_score
            """,
            (new_likelihood, node_id)
        )
        updated_node = cur.fetchone()
        conn.commit()
        
        return {
            "success": True,
            "node_id": node_id,
            "node_path": node["path"],
            "critique": {
                "counterargument": counterargument,
                "logical_flaws": logical_flaws,
                "edge_cases": edge_cases,
                "severity_score": severity_score,
                "recommendation": recommendation
            },
            "score_update": {
                "prior_score": float(updated_node["prior_score"]),
                "old_likelihood": current_likelihood,
                "new_likelihood": float(updated_node["likelihood"]),
                "posterior_score": float(updated_node["posterior_score"]) if updated_node["posterior_score"] else None
            }
        }
        
    except Exception as e:
        conn.rollback()
        logger.error(f"critique_thought failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


# =============================================================================
# Tree Traversal Utilities
# =============================================================================

def get_thought_tree(session_id: str, max_depth: int = 10) -> dict[str, Any]:
    """
    Retrieve the full thought tree for a session.
    
    Returns a nested structure of all thoughts organized by their ltree paths.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(
            """
            SELECT id, path, content, node_type, depth,
                   prior_score, likelihood, posterior_score,
                   supporting_laws, created_at
            FROM thought_nodes
            WHERE session_id = %s AND depth <= %s
            ORDER BY path
            """,
            (session_id, max_depth)
        )
        
        nodes = []
        for row in cur.fetchall():
            nodes.append({
                "id": str(row["id"]),
                "path": row["path"],
                "content": row["content"],
                "node_type": row["node_type"],
                "depth": row["depth"],
                "prior_score": float(row["prior_score"]),
                "likelihood": float(row["likelihood"]),
                "posterior_score": float(row["posterior_score"]) if row["posterior_score"] else None,
                "created_at": row["created_at"].isoformat()
            })
        
        return {
            "success": True,
            "session_id": session_id,
            "nodes": nodes,
            "count": len(nodes)
        }
        
    except Exception as e:
        logger.error(f"get_thought_tree failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


def get_best_path(session_id: str) -> dict[str, Any]:
    """
    Find the highest-scoring path through the thought tree.
    
    Uses posterior_score to identify the most promising reasoning chain.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get leaf nodes (nodes with no children)
        cur.execute(
            """
            WITH leaf_nodes AS (
                SELECT t1.* 
                FROM thought_nodes t1
                WHERE t1.session_id = %s
                AND NOT EXISTS (
                    SELECT 1 FROM thought_nodes t2 
                    WHERE t2.session_id = t1.session_id 
                    AND t2.path <@ t1.path 
                    AND t2.path != t1.path
                )
            )
            SELECT id, path, content, posterior_score, depth
            FROM leaf_nodes
            ORDER BY posterior_score DESC NULLS LAST
            LIMIT 1
            """,
            (session_id,)
        )
        
        best_leaf = cur.fetchone()
        if not best_leaf:
            return {"success": False, "error": "No thoughts found in session"}
        
        # Get all ancestors of this leaf
        cur.execute(
            """
            SELECT id, path, content, posterior_score, depth
            FROM thought_nodes
            WHERE session_id = %s AND %s <@ path
            ORDER BY depth
            """,
            (session_id, best_leaf["path"])
        )
        
        path_nodes = []
        for row in cur.fetchall():
            path_nodes.append({
                "id": str(row["id"]),
                "path": row["path"],
                "content": row["content"],
                "posterior_score": float(row["posterior_score"]) if row["posterior_score"] else None,
                "depth": row["depth"]
            })
        
        return {
            "success": True,
            "session_id": session_id,
            "best_leaf_id": str(best_leaf["id"]),
            "best_score": float(best_leaf["posterior_score"]) if best_leaf["posterior_score"] else None,
            "path": path_nodes
        }
        
    except Exception as e:
        logger.error(f"get_best_path failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        conn.close()
