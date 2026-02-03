"""
Phase 12: MCP Primitives Extraction

Parses Python files for @mcp.tool() and @mcp.resource() decorators
and extracts metadata for indexing.
"""
import ast
import re
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("pas-server")


def extract_mcp_primitives(file_path: str, project_id: str) -> list[dict[str, Any]]:
    """
    Parse Python file for MCP decorator patterns.
    
    Finds:
    - @mcp.tool() decorators
    - @mcp.resource("uri") decorators
    
    Returns list of:
    {
        'primitive_type': 'tool' | 'resource',
        'name': function name,
        'description': docstring,
        'line_number': decorator line,
        'parameters': {param: type_hint},
        'uri_pattern': 'pas://...' for resources (None for tools)
    }
    """
    path = Path(file_path)
    if not path.exists() or path.suffix != ".py":
        return []
    
    try:
        source = path.read_text()
        tree = ast.parse(source)
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        return []
    
    primitives = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            primitive = _extract_from_function(node, source)
            if primitive:
                primitives.append(primitive)
    
    return primitives


def _extract_from_function(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str) -> dict | None:
    """Extract primitive info from a decorated function."""
    for decorator in node.decorator_list:
        # Handle @mcp.tool()
        if isinstance(decorator, ast.Call):
            if _is_mcp_decorator(decorator, "tool"):
                return {
                    "primitive_type": "tool",
                    "name": node.name,
                    "description": ast.get_docstring(node) or "",
                    "line_number": decorator.lineno,
                    "parameters": _extract_parameters(node),
                    "uri_pattern": None,
                }
            
            # Handle @mcp.resource("uri")
            if _is_mcp_decorator(decorator, "resource"):
                uri_pattern = None
                if decorator.args and isinstance(decorator.args[0], ast.Constant):
                    uri_pattern = decorator.args[0].value
                
                return {
                    "primitive_type": "resource",
                    "name": node.name,
                    "description": ast.get_docstring(node) or "",
                    "line_number": decorator.lineno,
                    "parameters": _extract_parameters(node),
                    "uri_pattern": uri_pattern,
                }
    
    return None


def _is_mcp_decorator(decorator: ast.Call, decorator_name: str) -> bool:
    """Check if decorator is @mcp.<decorator_name>()."""
    func = decorator.func
    
    # @mcp.tool()
    if isinstance(func, ast.Attribute):
        if func.attr == decorator_name:
            if isinstance(func.value, ast.Name) and func.value.id == "mcp":
                return True
    
    return False


def _extract_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    """Extract function parameters with type hints."""
    params = {}
    
    for arg in node.args.args:
        param_name = arg.arg
        if param_name == "self":
            continue
        
        # Get type annotation if present
        if arg.annotation:
            type_hint = _annotation_to_string(arg.annotation)
        else:
            type_hint = "Any"
        
        params[param_name] = type_hint
    
    return params


def _annotation_to_string(annotation: ast.expr) -> str:
    """Convert AST annotation to string representation."""
    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Constant):
        return str(annotation.value)
    elif isinstance(annotation, ast.Subscript):
        # Handle things like list[str], dict[str, Any]
        if isinstance(annotation.value, ast.Name):
            base = annotation.value.id
            slice_str = _annotation_to_string(annotation.slice)
            return f"{base}[{slice_str}]"
    elif isinstance(annotation, ast.Tuple):
        parts = [_annotation_to_string(elt) for elt in annotation.elts]
        return ", ".join(parts)
    elif isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        # Handle X | Y type unions
        left = _annotation_to_string(annotation.left)
        right = _annotation_to_string(annotation.right)
        return f"{left} | {right}"
    
    return "Any"


def upsert_mcp_primitives(cur, project_id: str, primitives: list[dict], file_path: str) -> int:
    """
    Upsert primitives into mcp_primitives table.
    
    Returns count of upserted records.
    """
    import json
    
    count = 0
    for prim in primitives:
        cur.execute("""
            INSERT INTO mcp_primitives 
                (project_id, primitive_type, name, description, file_path, line_number, parameters, uri_pattern)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (project_id, primitive_type, name) 
            DO UPDATE SET 
                description = EXCLUDED.description,
                file_path = EXCLUDED.file_path,
                line_number = EXCLUDED.line_number,
                parameters = EXCLUDED.parameters,
                uri_pattern = EXCLUDED.uri_pattern
        """, (
            project_id,
            prim["primitive_type"],
            prim["name"],
            prim["description"][:2000] if prim["description"] else None,  # Truncate long docstrings
            file_path,
            prim["line_number"],
            json.dumps(prim["parameters"]),
            prim["uri_pattern"],
        ))
        count += 1
    
    return count
