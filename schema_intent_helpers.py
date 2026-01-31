"""
v45e Schema Intent Helpers

Extracts domain entities and relationships from database schema.
Uses information_schema introspection (not DDL parsing).
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def extract_schema_entities(tables: Dict[str, List[Dict]], relationships: List[Dict]) -> Dict[str, Any]:
    """
    Extract domain entities from schema info.
    
    Uses heuristics to classify tables:
    - Entity: has 'id' column + foreign key references TO it
    - Configuration: name ends with '_config', '_settings', 'configuration'  
    - Audit: name ends with '_log', '_history', 'audit'
    - Junction: only contains foreign key columns (+ optional timestamps)
    
    Args:
        tables: Dict from get_schema_info {table_name: [columns]}
        relationships: List from get_schema_info [{from_table, to_table, ...}]
        
    Returns:
        {entities: [...], relationships: [...], classifications: [...]}
    """
    entities = []
    classifications = []
    
    # Build FK reference counts (tables referenced by FKs = likely entities)
    fk_targets: Dict[str, int] = {}
    for rel in relationships:
        target = rel.get("to_table")
        if target:
            fk_targets[target] = fk_targets.get(target, 0) + 1
    
    for table_name, columns in tables.items():
        column_names = [c.get("column", "") for c in columns]
        
        # Classification heuristics
        classification = classify_table(table_name, column_names, fk_targets.get(table_name, 0))
        
        entity = {
            "name": table_name,
            "classification": classification,
            "column_count": len(columns),
            "is_referenced": fk_targets.get(table_name, 0) > 0,
            "reference_count": fk_targets.get(table_name, 0)
        }
        
        entities.append(entity)
        classifications.append({"table": table_name, "type": classification})
    
    # Format relationships
    formatted_rels = [
        {
            "from_entity": r.get("from_table"),
            "to_entity": r.get("to_table"),
            "from_column": r.get("from_column"),
            "to_column": r.get("to_column")
        }
        for r in relationships
    ]
    
    return {
        "entities": entities,
        "relationships": formatted_rels,
        "classifications": classifications,
        "stats": {
            "total_tables": len(entities),
            "total_relationships": len(formatted_rels)
        }
    }


def classify_table(table_name: str, column_names: List[str], fk_reference_count: int) -> str:
    """Classify table type using naming heuristics."""
    name_lower = table_name.lower()
    
    # Audit/logging tables
    if any(name_lower.endswith(suffix) for suffix in ('_log', '_history', '_audit', 'audit_trail')):
        return "audit"
    
    # Configuration tables
    if any(name_lower.endswith(suffix) for suffix in ('_config', '_settings', '_configuration', 'settings')):
        return "configuration"
    
    # Junction/association tables (typically 2-word with underscore and few columns)
    if '_' in table_name and len(column_names) <= 4:
        # Check if most columns are FKs (id + 2 FKs + maybe timestamp)
        if all(c.endswith('_id') or c in ('id', 'created_at', 'updated_at') for c in column_names):
            return "junction"
    
    # Domain entity (has id and is referenced by other tables)
    if 'id' in column_names and fk_reference_count > 0:
        return "entity"
    
    # Standalone table
    if 'id' in column_names:
        return "standalone"
    
    return "unknown"


def build_enrichment_prompt(entities: List[Dict], relationships: List[Dict]) -> str:
    """
    Build LLM prompt for semantic enrichment of extracted entities.
    
    Returns prompt for optional LLM analysis to add:
    - Business domain context
    - Entity descriptions
    - Relationship semantics
    """
    table_list = "\n".join(f"- {e['name']} ({e['classification']}): {e['column_count']} columns" 
                           for e in entities[:30])  # Limit for context
    
    rel_list = "\n".join(f"- {r['from_entity']} → {r['to_entity']}" 
                         for r in relationships[:20])
    
    return f"""Analyze this database schema and provide semantic enrichment:

## Tables
{table_list}

## Relationships
{rel_list}

For each entity, provide:
1. **domain_context**: What business domain does this belong to?
2. **description**: Human-readable description of the entity's purpose
3. **key_attributes**: Most important columns and their meaning

Return as JSON:
{{
  "enriched_entities": [
    {{"name": "table_name", "domain_context": "...", "description": "...", "key_attributes": ["..."]}}
  ],
  "domain_summary": "Overall domain description"
}}"""
