"""GEMINI.md ↔ Database bidirectional sync with hybrid LLM/embedding approach.

Phase 7c: Environment Constraints
PAS Session: fea58ef5-3773-46ac-b2d2-359f2283ba29
"""

from typing import Literal, Optional, Any
from pathlib import Path
import json

from pas.utils import get_db_connection, get_embedding


EXTRACTION_PROMPT = """Analyze this GEMINI.md and extract process constraints as structured JSON.

GEMINI.md content:
---
{content}
---

Return ONLY valid JSON (no markdown fences):
{{
  "constraints": [
    {{
      "type": "philosophy|environment|quality",
      "key": "no_mvp|dual_plan|planning_depth|code_quality|assumption_policy|...",
      "value": <any JSON value - boolean, string, or number>,
      "enforcement": "hidden|warn|block",
      "source_section": "Rule heading or description"
    }}
  ]
}}

Focus on extracting:
- Explicit rules (e.g., "NO MVP", "always generate 2 plans")
- Process preferences (e.g., planning_depth, assumption handling)
- Quality standards (e.g., code_quality level)
"""

# Section marker for idempotent export
EXPORT_SECTION_MARKER = "## PAS-Exported Constraints"

EXPORT_TEMPLATE = """{marker}

> Auto-generated from database. Last synced: {timestamp}

### Philosophy Constraints

| Key | Value | Enforcement |
|-----|-------|-------------|
{philosophy_rows}

### Environment Constraints

| Key | Value | Enforcement |
|-----|-------|-------------|
{environment_rows}

### Quality Constraints

| Key | Value | Enforcement |
|-----|-------|-------------|
{quality_rows}
"""


def parse_gemini_constraints(gemini_path: Path) -> dict[str, Any]:
    """
    Return extraction prompt for agent LLM processing.
    
    Agent should process the extraction_prompt with their LLM,
    then call store_extracted_constraints() with results.
    """
    if not gemini_path.exists():
        return {"success": False, "error": f"GEMINI.md not found: {gemini_path}"}
    
    content = gemini_path.read_text()
    
    # Truncate to reasonable size for LLM context
    max_chars = 12000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n... [truncated]"
    
    return {
        "success": True,
        "extraction_prompt": EXTRACTION_PROMPT.format(content=content),
        "file_path": str(gemini_path),
        "content_length": len(content),
        "next_step": "Process extraction_prompt with your LLM, then call store_extracted_constraints(project_id, constraints)"
    }


def store_extracted_constraints(project_id: str, constraints: list[dict]) -> dict[str, Any]:
    """
    Persist LLM-extracted constraints with embeddings.
    
    Uses temporal versioning: existing constraints are expired (valid_to = NOW)
    and new versions are inserted.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    stored = 0
    errors = []
    
    # Get project UUID from project_id string
    cur.execute("SELECT id FROM project_registry WHERE project_id = %s", (project_id,))
    row = cur.fetchone()
    if not row:
        return {"success": False, "error": f"Project not found: {project_id}"}
    
    project_uuid = row["id"]
    
    for c in constraints:
        try:
            # Generate embedding for drift detection
            constraint_text = json.dumps({"key": c["key"], "value": c["value"]})
            embedding = get_embedding(constraint_text)
            
            # Get current max version for this constraint
            cur.execute("""
                SELECT MAX(version) as max_ver FROM project_constraints 
                WHERE project_id = %s AND constraint_key = %s
            """, (project_uuid, c["key"]))
            version_row = cur.fetchone()
            new_version = (version_row["max_ver"] or 0) + 1
            
            # Expire existing active version
            cur.execute("""
                UPDATE project_constraints 
                SET valid_to = NOW()
                WHERE project_id = %s AND constraint_key = %s AND valid_to IS NULL
            """, (project_uuid, c["key"]))
            
            # Insert new version
            cur.execute("""
                INSERT INTO project_constraints 
                    (project_id, constraint_type, constraint_key, constraint_data, 
                     constraint_embedding, enforcement_level, source, version)
                VALUES (%s, %s, %s, %s, %s, %s, 'gemini_md', %s)
            """, (
                project_uuid, 
                c.get("type", "philosophy"), 
                c["key"], 
                json.dumps(c["value"]),
                embedding, 
                c.get("enforcement", "warn"),
                new_version
            ))
            stored += 1
            
        except Exception as e:
            errors.append({"key": c.get("key", "unknown"), "error": str(e)})
    
    conn.commit()
    return {
        "success": len(errors) == 0,
        "stored_count": stored,
        "errors": errors if errors else None
    }


def detect_drift(project_id: str, new_constraints: list[dict], threshold: float = 0.15) -> dict[str, Any]:
    """
    Compare new constraints against DB using embedding similarity.
    
    Returns drifts where the semantic meaning has changed significantly
    (similarity < 1 - threshold).
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get project UUID
    cur.execute("SELECT id FROM project_registry WHERE project_id = %s", (project_id,))
    row = cur.fetchone()
    if not row:
        return {"success": False, "error": f"Project not found: {project_id}"}
    
    project_uuid = row["id"]
    drifts = []
    
    for c in new_constraints:
        constraint_text = json.dumps({"key": c["key"], "value": c["value"]})
        new_embedding = get_embedding(constraint_text)
        
        # Compare against stored constraint using pgvector cosine distance
        cur.execute("""
            SELECT constraint_key, constraint_data, 
                   1 - (constraint_embedding <=> %s::vector) as similarity
            FROM project_constraints
            WHERE project_id = %s AND constraint_key = %s AND valid_to IS NULL
              AND constraint_embedding IS NOT NULL
        """, (new_embedding, project_uuid, c["key"]))
        
        row = cur.fetchone()
        if row:
            similarity = row["similarity"] or 0
            if similarity < (1 - threshold):
                drifts.append({
                    "key": c["key"],
                    "drift_score": round(1 - similarity, 3),
                    "db_value": row["constraint_data"],
                    "file_value": c["value"],
                    "requires_resolution": True
                })
    
    return {
        "success": True,
        "drifts": drifts,
        "drift_detected": len(drifts) > 0,
        "resolution_prompt": _build_resolution_prompt(drifts) if drifts else None
    }


def _build_resolution_prompt(drifts: list[dict]) -> str:
    """Build a prompt for agent LLM to resolve drift conflicts."""
    lines = ["The following constraints have drifted between GEMINI.md and database:\n"]
    
    for d in drifts:
        lines.append(f"**{d['key']}** (drift: {d['drift_score']:.0%})")
        lines.append(f"  - DB value: {d['db_value']}")
        lines.append(f"  - File value: {d['file_value']}")
        lines.append("")
    
    lines.append("Which source should be authoritative? Options:")
    lines.append("A) Use GEMINI.md values (file overwrites DB)")
    lines.append("B) Use DB values (DB overwrites file)")
    lines.append("C) Keep both, resolve manually")
    
    return "\n".join(lines)


def resolve_constraint_drift(
    project_id: str,
    project_path: str,
    resolution: Literal["file_to_db", "db_to_file", "skip"],
    drifted_keys: Optional[list[str]] = None,
    file_constraints: Optional[list[dict]] = None
) -> dict[str, Any]:
    """
    Apply conflict resolution for drifted constraints.
    
    Args:
        project_id: Project identifier
        project_path: Path to project root (for db_to_file)
        resolution: How to resolve:
            - file_to_db: GEMINI.md values overwrite DB
            - db_to_file: DB values overwrite GEMINI.md
            - skip: Leave unresolved for manual handling
        drifted_keys: Optional list of specific keys to resolve (all if None)
        file_constraints: Required for file_to_db - the extracted constraints from GEMINI.md
        
    Returns:
        Resolution results with counts and paths affected
    """
    if resolution == "skip":
        return {
            "success": True,
            "resolution": "skip",
            "message": "No changes made. Please resolve manually."
        }
    
    if resolution == "file_to_db":
        # User chose GEMINI.md as source of truth
        if not file_constraints:
            return {
                "success": False,
                "error": "file_constraints required for file_to_db resolution"
            }
        
        # Filter to only drifted keys if specified
        to_store = file_constraints
        if drifted_keys:
            to_store = [c for c in file_constraints if c.get("key") in drifted_keys]
        
        # Store using temporal versioning
        result = store_extracted_constraints(project_id, to_store)
        return {
            "success": result.get("success", False),
            "resolution": "file_to_db",
            "constraints_updated": result.get("stored_count", 0),
            "errors": result.get("errors")
        }
    
    elif resolution == "db_to_file":
        # User chose DB as source of truth
        export_result = export_constraints_to_markdown(project_id)
        if not export_result.get("success"):
            return export_result
        
        write_result = write_gemini_export(project_path, export_result["export_content"], "append")
        return {
            "success": write_result.get("success", False),
            "resolution": "db_to_file", 
            "constraints_exported": export_result.get("constraint_count", 0),
            "path": write_result.get("path")
        }
    
    return {"success": False, "error": f"Unknown resolution: {resolution}"}


def export_constraints_to_markdown(project_id: str) -> dict[str, Any]:
    """
    Export database constraints as formatted markdown.
    
    Queries project_constraints table and formats as markdown table
    grouped by constraint_type (philosophy/environment/quality).
    
    Returns:
        export_content: Pre-formatted markdown block
        constraint_count: Number of constraints exported
        project_path: Path to project for write operations
    """
    from datetime import datetime
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get project UUID and path
    cur.execute("""
        SELECT id, project_path FROM project_registry WHERE project_id = %s
    """, (project_id,))
    row = cur.fetchone()
    if not row:
        return {"success": False, "error": f"Project not found: {project_id}"}
    
    project_uuid = row["id"]
    project_path = row["project_path"]
    
    # Get active constraints grouped by type
    cur.execute("""
        SELECT constraint_type, constraint_key, constraint_data, enforcement_level
        FROM project_constraints
        WHERE project_id = %s AND valid_to IS NULL
        ORDER BY constraint_type, constraint_key
    """, (project_uuid,))
    
    constraints = cur.fetchall()
    
    # Group by type
    by_type: dict[str, list] = {"philosophy": [], "environment": [], "quality": []}
    for c in constraints:
        ctype = c["constraint_type"]
        if ctype in by_type:
            by_type[ctype].append(c)
    
    def format_rows(items: list) -> str:
        if not items:
            return "| (none) | - | - |"
        return "\n".join(
            f"| `{c['constraint_key']}` | `{c['constraint_data']}` | {c['enforcement_level']} |"
            for c in items
        )
    
    content = EXPORT_TEMPLATE.format(
        marker=EXPORT_SECTION_MARKER,
        timestamp=datetime.now().isoformat(),
        philosophy_rows=format_rows(by_type["philosophy"]),
        environment_rows=format_rows(by_type["environment"]),
        quality_rows=format_rows(by_type["quality"]),
    )
    
    return {
        "success": True,
        "export_content": content,
        "constraint_count": len(constraints),
        "project_path": project_path
    }


def write_gemini_export(project_path: str, content: str, mode: str = "append") -> dict[str, Any]:
    """
    Write exported constraints to GEMINI.md.
    
    Supports idempotent section replacement - if the export section already
    exists, it will be replaced rather than duplicated.
    
    Args:
        project_path: Project root path containing GEMINI.md
        content: Markdown content to write (from export_constraints_to_markdown)
        mode: 'append' (add/replace section) or 'replace' (overwrite entire file)
    
    Returns:
        Success status with file path and mode used
    """
    gemini_path = Path(project_path) / "GEMINI.md"
    
    if mode == "replace":
        gemini_path.write_text(content)
        return {"success": True, "path": str(gemini_path), "mode": "replaced"}
    
    elif mode == "append":
        if gemini_path.exists():
            existing = gemini_path.read_text()
            
            # Check if section already exists - replace it idempotently
            if EXPORT_SECTION_MARKER in existing:
                # Find section start
                start_idx = existing.index(EXPORT_SECTION_MARKER)
                # Find next h2 or end of file
                rest = existing[start_idx + len(EXPORT_SECTION_MARKER):]
                next_h2 = rest.find("\n## ")
                if next_h2 == -1:
                    # No next section, replace to end
                    new_content = existing[:start_idx].rstrip() + "\n\n" + content
                else:
                    # Replace until next section
                    end_idx = start_idx + len(EXPORT_SECTION_MARKER) + next_h2
                    new_content = existing[:start_idx].rstrip() + "\n\n" + content + "\n" + existing[end_idx + 1:]
                gemini_path.write_text(new_content)
                return {"success": True, "path": str(gemini_path), "mode": "replaced_section"}
            else:
                # Append to end
                gemini_path.write_text(existing.rstrip() + "\n\n" + content)
                return {"success": True, "path": str(gemini_path), "mode": "appended"}
        else:
            # Create new file
            gemini_path.write_text(content)
            return {"success": True, "path": str(gemini_path), "mode": "created"}
    
    return {"success": False, "error": f"Unknown mode: {mode}"}


async def sync_gemini_constraints(

    project_id: str,
    project_path: str,
    direction: Literal["file_to_db", "db_to_file", "detect_only"] = "detect_only"
) -> dict[str, Any]:
    """
    Main sync tool for GEMINI.md ↔ database constraints.
    
    Args:
        project_id: Project identifier (e.g., 'mcp-pas')
        project_path: Path to project root containing GEMINI.md
        direction: Sync direction
            - detect_only: Just detect drift, don't sync
            - file_to_db: GEMINI.md overwrites DB
            - db_to_file: DB overwrites GEMINI.md (not implemented yet)
    
    Returns:
        Sync results with extraction_prompt if needed
    """
    gemini_path = Path(project_path) / "GEMINI.md"
    
    if not gemini_path.exists():
        return {"success": False, "error": f"GEMINI.md not found at {gemini_path}"}
    
    if direction == "detect_only":
        # Return extraction prompt for agent to process
        parse_result = parse_gemini_constraints(gemini_path)
        if not parse_result["success"]:
            return parse_result
        
        return {
            "success": True,
            "mode": "detect_only",
            "extraction_prompt": parse_result["extraction_prompt"],
            "next_steps": [
                "1. Process extraction_prompt with your LLM to get constraints",
                "2. Call detect_drift(project_id, constraints) to check for changes",
                "3. If drift detected, call sync_gemini_constraints with direction='file_to_db'"
            ]
        }
    
    elif direction == "file_to_db":
        # Agent must provide extracted constraints via store_extracted_constraints
        return {
            "success": True,
            "mode": "file_to_db",
            "message": "Use store_extracted_constraints(project_id, constraints) with LLM-extracted constraints",
            "extraction_prompt": parse_gemini_constraints(gemini_path).get("extraction_prompt")
        }
    
    elif direction == "db_to_file":
        # Export constraints from database as formatted markdown
        export_result = export_constraints_to_markdown(project_id)
        if not export_result["success"]:
            return export_result
        
        return {
            "success": True,
            "mode": "db_to_file",
            "dry_run": True,
            "export_content": export_result["export_content"],
            "constraint_count": export_result["constraint_count"],
            "project_path": export_result["project_path"],
            "next_step": "Review export_content, then call write_gemini_export(project_path, export_content, mode='append')"
        }

    
    return {"success": False, "error": f"Unknown direction: {direction}"}
