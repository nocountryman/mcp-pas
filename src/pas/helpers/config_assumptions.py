"""
v45f Config Assumptions Helpers

Extracts implicit assumptions from configuration files.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_config_file(config_path: str) -> Dict[str, Any]:
    """Parse YAML or JSON config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    content = path.read_text()
    
    if path.suffix in ('.yaml', '.yml'):
        import yaml
        return yaml.safe_load(content) or {}
    elif path.suffix == '.json':
        import json
        return json.loads(content)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def extract_assumptions(config: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
    """
    Extract assumptions from config using heuristics.
    
    Heuristics:
    - Numeric thresholds → performance/quality expectations
    - Time values (days, hours, seconds) → scheduling/timing requirements
    - Path/URL strings → deployment/environment requirements
    - Boolean flags → feature toggles/constraints
    """
    assumptions = []
    
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recurse into nested config
            assumptions.extend(extract_assumptions(value, full_key))
        elif isinstance(value, (int, float)):
            assumptions.append(extract_numeric_assumption(full_key, value))
        elif isinstance(value, str):
            assumption = extract_string_assumption(full_key, value)
            if assumption:
                assumptions.append(assumption)
        elif isinstance(value, bool):
            assumptions.append({
                "key": full_key,
                "value": value,
                "type": "feature_flag",
                "assumption": f"Feature '{full_key}' is {'enabled' if value else 'disabled'} by default"
            })
    
    return [a for a in assumptions if a]  # Filter None


def extract_numeric_assumption(key: str, value: float) -> Dict[str, Any]:
    """Extract assumption from numeric value."""
    key_lower = key.lower()
    
    # Threshold detection
    if any(t in key_lower for t in ('threshold', 'min', 'max', 'limit')):
        if 0 < value <= 1:
            return {
                "key": key,
                "value": value,
                "type": "threshold",
                "assumption": f"Expects values {'above' if 'min' in key_lower else 'below'} {value:.0%}"
            }
        return {
            "key": key,
            "value": value,
            "type": "threshold",
            "assumption": f"Bounds value at {value}"
        }
    
    # Time detection
    if any(t in key_lower for t in ('days', 'hours', 'seconds', 'timeout', 'interval')):
        return {
            "key": key,
            "value": value,
            "type": "timing",
            "assumption": f"Expects operation within {value} time units"
        }
    
    # Count/size detection
    if any(t in key_lower for t in ('count', 'size', 'max', 'depth', 'top_n')):
        return {
            "key": key,
            "value": value,
            "type": "capacity",
            "assumption": f"Limits to {int(value)} items"
        }
    
    return {"key": key, "value": value, "type": "numeric", "assumption": f"Configured to {value}"}


def extract_string_assumption(key: str, value: str) -> Optional[Dict[str, Any]]:
    """Extract assumption from string value."""
    # Path detection
    if '/' in value or value.startswith('.'):
        return {
            "key": key,
            "value": value,
            "type": "path",
            "assumption": f"Expects path '{value}' to exist"
        }
    
    # URL detection
    if value.startswith(('http://', 'https://', 'postgresql://', 'redis://')):
        return {
            "key": key,
            "value": value[:50] + "..." if len(value) > 50 else value,
            "type": "connection",
            "assumption": f"Requires network access to external service"
        }
    
    # Environment variable detection
    if re.match(r'\$\{?\w+\}?', value):
        return {
            "key": key,
            "value": value,
            "type": "environment",
            "assumption": f"Requires environment variable {value}"
        }
    
    return None  # Skip plain strings


def build_enrichment_prompt(assumptions: List[Dict], config_path: str) -> str:
    """Build LLM prompt for semantic enrichment."""
    assumption_list = "\n".join(
        f"- {a['key']}: {a['assumption']} (type: {a['type']})"
        for a in assumptions[:25]
    )
    
    return f"""Analyze these extracted configuration assumptions:

## Config File: {config_path}

## Detected Assumptions
{assumption_list}

Please enrich with:
1. **domain_context**: What system/domain do these assumptions serve?
2. **deployment_requirements**: What infrastructure is implied?
3. **operational_constraints**: What runtime behaviors are expected?

Return as JSON:
{{
  "enriched_assumptions": [...],
  "domain_context": "...",
  "deployment_requirements": ["..."],
  "operational_constraints": ["..."]
}}"""
