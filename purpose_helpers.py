"""
PAS Purpose Inference Helpers (v40 Phase 1)

Pure functions for hierarchical purpose detection:
- FUNCTION level: What does each function do?
- MODULE level: What does this file accomplish?
- PROJECT level: How does this contribute to the system?

Based on ICPC 2024 program comprehension research.
"""

from typing import Dict, Any, Optional, List
import re
import json


# Template for LLM purpose inference
PURPOSE_PROMPT_TEMPLATE = """Analyze this code file and answer these questions:

**FILE**: {file_path}
{sibling_context}
**CONTENT**:
```
{content}
```

---

Answer at 3 levels:

1. **FUNCTION PURPOSES** (for each key function):
   - Function name: What problem does it solve?

2. **MODULE PURPOSE** (this file as a whole):
   - PROBLEM: What technical problem does this file solve?
   - USER NEED: What user need does it address?
   - PHILOSOPHY: Why does this file exist in the architecture?

3. **PROJECT CONTRIBUTION**:
   - How does this file contribute to the overall system?

---

Return ONLY valid JSON:
{{
  "function_purposes": [
    {{"name": "func_name", "purpose": "..."}}
  ],
  "module_purpose": {{
    "problem": "...",
    "user_need": "...",
    "philosophy": "..."
  }},
  "project_contribution": "..."
}}
"""


# v40 Phase 1.5: Module aggregation prompt
MODULE_AGGREGATION_PROMPT = """Synthesize a module-level purpose from these file purposes:

**DIRECTORY**: {directory_path}

**FILES IN THIS MODULE**:
{file_purposes}

---

Answer:
1. **MODULE PURPOSE**: What does this module (directory) accomplish as a whole?
2. **USER NEED**: What user need does this module address?
3. **ARCHITECTURE ROLE**: How does this module fit in the project architecture?

Return ONLY valid JSON:
{{
  "module_purpose": "...",
  "user_need": "...",
  "architecture_role": "..."
}}
"""


# v43: Project-level purpose inference template
PROJECT_PURPOSE_PROMPT_TEMPLATE = """Analyze this project and answer these questions:

**PROJECT**: {project_id}
**PATH**: {project_path}

**TOP MODULES** (most significant directories/files):
{module_summaries}

**FILE STATISTICS**:
- Total files: {file_count}
- Languages: {languages}

---

Answer at the PROJECT level:

1. **MISSION**: What is the primary purpose of this entire project? (1-2 sentences)
2. **USER NEEDS**: What user needs does this project address? (list 3-5)
3. **MUST-HAVE MODULES**: What modules/components must exist for this project to fulfill its mission? (list 3-7)
4. **DETECTED DOMAIN**: What domain does this project belong to? (e.g., 'backend', 'frontend', 'ml', 'devops', 'library', 'cli')
5. **DOMAIN CONFIDENCE**: How confident are you in the domain detection? (0.0 to 1.0)

---

Return ONLY valid JSON:
{{
  "mission": "...",
  "user_needs": ["need1", "need2", "need3"],
  "must_have_modules": ["module1", "module2", "module3"],
  "detected_domain": "...",
  "domain_confidence": 0.9
}}
"""


def build_purpose_prompt(
    file_path: str, 
    content: str, 
    max_content_length: int = 8000,
    sibling_files: Optional[List[str]] = None  # v40 Phase 1.5: Hybrid sibling-aware
) -> str:
    """
    Build LLM prompt for purpose inference.
    
    Args:
        file_path: Path to the file being analyzed
        content: File content (will be truncated if too long)
        max_content_length: Max chars of content to include
        sibling_files: Optional list of sibling filenames in same directory
        
    Returns:
        Formatted prompt string
    """
    # Truncate content if too long
    if len(content) > max_content_length:
        content = content[:max_content_length] + "\n\n... [truncated]"
    
    # v40 Phase 1.5: Add sibling context for module awareness
    sibling_context = ""
    if sibling_files:
        sibling_list = ", ".join(sibling_files[:10])  # Limit to 10
        sibling_context = f"\n**SIBLING FILES** (same module): {sibling_list}\n"
    
    return PURPOSE_PROMPT_TEMPLATE.format(
        file_path=file_path,
        sibling_context=sibling_context,
        content=content
    )


def build_module_aggregation_prompt(
    directory_path: str,
    file_purposes: List[Dict[str, Any]]
) -> str:
    """
    Build LLM prompt for module-level purpose aggregation.
    
    Args:
        directory_path: Path to the directory/module
        file_purposes: List of {file, problem, user_need} from file purposes
        
    Returns:
        Formatted aggregation prompt
    """
    purpose_lines = []
    for fp in file_purposes[:10]:  # Limit to 10 files
        file_name = fp.get("file", "unknown").split("/")[-1]
        problem = fp.get("problem", "Unknown purpose")
        purpose_lines.append(f"- `{file_name}`: {problem}")
    
    file_purposes_text = "\n".join(purpose_lines) if purpose_lines else "No file purposes available"
    
    return MODULE_AGGREGATION_PROMPT.format(
        directory_path=directory_path,
        file_purposes=file_purposes_text
    )


def parse_module_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM response for module aggregation into structured data.
    
    Args:
        response: Raw LLM response (should be JSON)
        
    Returns:
        Parsed module purpose dict or None if parsing fails
    """
    try:
        # Handle markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            response = json_match.group(1)
        
        response = response.strip()
        data = json.loads(response)
        
        if not isinstance(data, dict):
            return None
        
        return {
            "module_purpose": data.get("module_purpose", ""),
            "user_need": data.get("user_need", ""),
            "architecture_role": data.get("architecture_role", "")
        }
        
    except (json.JSONDecodeError, AttributeError):
        return None


# v43: Project-level purpose helpers
def build_project_purpose_prompt(
    project_id: str,
    project_path: str,
    module_summaries: List[Dict[str, str]],
    file_count: int,
    languages: List[str]
) -> str:
    """
    Build LLM prompt for project-level purpose inference.
    
    Args:
        project_id: Project identifier
        project_path: Absolute path to project
        module_summaries: List of {file, problem} for top files
        file_count: Total files in project
        languages: List of detected languages
        
    Returns:
        Formatted prompt string
    """
    # Format module summaries
    if module_summaries:
        lines = []
        for mod in module_summaries[:10]:  # Limit to 10
            file_name = mod.get("file", "unknown").split("/")[-1]
            problem = mod.get("problem", "Unknown purpose")[:100]
            lines.append(f"- `{file_name}`: {problem}")
        module_text = "\n".join(lines)
    else:
        module_text = "No module purposes available yet"
    
    # Format languages
    lang_text = ", ".join(languages[:5]) if languages else "Unknown"
    
    return PROJECT_PURPOSE_PROMPT_TEMPLATE.format(
        project_id=project_id,
        project_path=project_path,
        module_summaries=module_text,
        file_count=file_count,
        languages=lang_text
    )


def parse_project_purpose_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM response for project purpose into structured data.
    
    Args:
        response: Raw LLM response (should be JSON)
        
    Returns:
        Parsed project purpose dict or None if parsing fails
    """
    try:
        # Handle markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            response = json_match.group(1)
        
        response = response.strip()
        data = json.loads(response)
        
        if not isinstance(data, dict):
            return None
        
        # Validate and extract purpose hierarchy
        purpose_hierarchy = {
            "mission": data.get("mission", ""),
            "user_needs": data.get("user_needs", []),
            "must_have_modules": data.get("must_have_modules", [])
        }
        
        # Validate types
        if not isinstance(purpose_hierarchy["user_needs"], list):
            purpose_hierarchy["user_needs"] = [str(purpose_hierarchy["user_needs"])]
        if not isinstance(purpose_hierarchy["must_have_modules"], list):
            purpose_hierarchy["must_have_modules"] = [str(purpose_hierarchy["must_have_modules"])]
        
        return {
            "purpose_hierarchy": purpose_hierarchy,
            "detected_domain": data.get("detected_domain", ""),
            "domain_confidence": float(data.get("domain_confidence", 0.0))
        }
        
    except (json.JSONDecodeError, AttributeError, ValueError):
        return None




def parse_purpose_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM response into structured purpose data.
    
    Args:
        response: Raw LLM response (should be JSON)
        
    Returns:
        Parsed purpose dict or None if parsing fails
    """
    try:
        # Try to extract JSON from response
        # Handle case where LLM wraps in markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            response = json_match.group(1)
        
        # Clean up common issues
        response = response.strip()
        
        data = json.loads(response)
        
        # Validate required fields
        if not isinstance(data, dict):
            return None
        
        # Ensure expected structure
        return {
            "function_purposes": data.get("function_purposes", []),
            "module_purpose": data.get("module_purpose", {}),
            "project_contribution": data.get("project_contribution", "")
        }
        
    except (json.JSONDecodeError, AttributeError):
        return None


def extract_function_signatures(content: str) -> List[Dict[str, str]]:
    """
    Extract function signatures from Python content for purpose context.
    
    Args:
        content: Python file content
        
    Returns:
        List of {name, signature} dicts
    """
    functions = []
    
    # Match Python function definitions
    pattern = r'(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)'
    
    for match in re.finditer(pattern, content):
        name = match.group(1)
        params = match.group(2).strip()
        
        # Skip private/dunder methods for purpose analysis
        if name.startswith('__') and name.endswith('__'):
            continue
        if name.startswith('_') and not name.startswith('__'):
            continue
            
        functions.append({
            "name": name,
            "signature": f"def {name}({params})"
        })
    
    return functions


def validate_purpose_cache(
    cache: Optional[Dict[str, Any]], 
    current_hash: str
) -> bool:
    """
    Check if cached purpose is still valid.
    
    Args:
        cache: Cached purpose data (may be None)
        current_hash: Current file hash
        
    Returns:
        True if cache is valid, False if needs refresh
    """
    if cache is None:
        return False
    
    cached_hash = cache.get("file_hash")
    if cached_hash != current_hash:
        return False
    
    # Check for required fields
    purposes = cache.get("purposes")
    if not purposes:
        return False
    
    return True


def build_purpose_context_entry(
    file_path: str,
    purposes: Dict[str, Any],
    status: str = "cached"
) -> Dict[str, Any]:
    """
    Build a purpose context entry for prepare_expansion response.
    
    Args:
        file_path: Path to the file
        purposes: Parsed purpose data
        status: "cached" or "needs_inference"
        
    Returns:
        Purpose context entry dict
    """
    module_purpose = purposes.get("module_purpose", {})
    
    return {
        "file": file_path,
        "status": status,
        "problem": module_purpose.get("problem", ""),
        "user_need": module_purpose.get("user_need", ""),
        "philosophy": module_purpose.get("philosophy", ""),
        "project_contribution": purposes.get("project_contribution", "")
    }


def summarize_purposes_for_prompt(purpose_contexts: List[Dict[str, Any]]) -> str:
    """
    Generate a summary of purposes for inclusion in hypothesis prompts.
    
    Args:
        purpose_contexts: List of purpose context entries
        
    Returns:
        Formatted string for prompt injection
    """
    if not purpose_contexts:
        return ""
    
    lines = ["**Scope Context** (what these files do):"]
    
    for ctx in purpose_contexts:
        if ctx.get("status") == "cached":
            problem = ctx.get("problem", "Unknown")
            lines.append(f"- `{ctx['file']}`: {problem}")
    
    return "\n".join(lines)
