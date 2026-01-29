#!/usr/bin/env python3
"""
Add hidden_value fields to ALL interview question choices.
Updates the database directly (not the seed file).

Taxonomy mapping to traits:
- SAFETY_* → RISK_AVERSE trait
- SIMPLICITY_* → MINIMALIST trait  
- CONTROL_* / EXPLICIT_* → CONTROL_ORIENTED trait
- AI_* / AUTOMATION_* → AUTOMATION_TRUSTING trait
- BEGINNER_* / ACCESSIBILITY_* → ACCESSIBILITY_FOCUSED trait
- BALANCE_* / PRAGMATIC_* → PRAGMATIST trait
- PERFORMANCE_* / SPEED_* → SPEED_FOCUSED trait
- FREEDOM_* / AUTONOMY_* → AUTONOMY_FOCUSED trait
- ERROR_PREVENTION_* → SAFETY_CONSCIOUS trait
"""

import os
import json
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# Pattern-based hidden_value inference
# Order matters - first match wins
PATTERNS = [
    # Safety/Risk patterns → RISK_AVERSE
    (r"(unlimited undo|full history|version history|rollback)", "SAFETY_FULL"),
    (r"(snapshots|recent only|last few)", "SAFETY_MODERATE"),
    (r"(confirm|confirmation dialog)", "SAFETY_CONFIRM"),
    
    # Error Prevention → SAFETY_CONSCIOUS
    (r"(validation|prevent|smart defaults|real-time)", "ERROR_PREVENTION"),
    (r"(guardrails|ai catches)", "AI_SAFETY"),
    (r"(warnings only)", "FREEDOM_WITH_WARNINGS"),
    
    # Simplicity → MINIMALIST
    (r"(minimal|simple|basic|clean|one task)", "SIMPLICITY_FIRST"),
    (r"(3-5 items|2-3\)|less)", "MINIMALIST_CHOICE"),
    
    # Control → CONTROL_ORIENTED
    (r"(manual|explicit|user sets|full control|user enters)", "EXPLICIT_CONTROL"),
    (r"(mode toggle|pro mode|advanced toggle)", "USER_MODE_CONTROL"),
    (r"(custom|create any)", "CUSTOM_CONTROL"),
    
    # AI/Automation → AUTOMATION_TRUSTING
    (r"(ai auto|ai detect|ai tag|ai suggest|ai struct|ai inference)", "AI_DRIVEN"),
    (r"(adaptive|personalized|learns from)", "AI_ADAPTIVE"),
    (r"(magic|auto-breakdown|pre-fill)", "AI_MAGIC"),
    
    # Expert/Power → POWER_USER
    (r"(power user|expert|pro |advanced|keyboard shortcut)", "POWER_USER"),
    (r"(all visible|everything shown|dense|extensive|10\+)", "COMPREHENSIVE"),
    (r"(native apps|native features)", "NATIVE_POWER"),
    
    # Beginner → ACCESSIBILITY_FOCUSED
    (r"(beginner|accessible|lower learning|easy to)", "BEGINNER_FRIENDLY"),
    (r"(progressive|unlock|reveal)", "PROGRESSIVE_LEARNING"),
    (r"(guided|tutorial|teaching|hints)", "LEARNING_GUIDED"),
    
    # Balance → PRAGMATIST
    (r"(balance|hybrid|best of both|contextual)", "BALANCED_APPROACH"),
    (r"(standard|5-7 items|4-6\)|moderate)", "PRAGMATIC_STANDARD"),
    (r"(good for most|works across)", "PRAGMATIC_UNIVERSAL"),
    
    # Performance → SPEED_FOCUSED
    (r"(fast|performance|efficient|instant|quick|blazing)", "PERFORMANCE_FIRST"),
    (r"(lightweight|less overhead)", "PERFORMANCE_LIGHTWEIGHT"),
    
    # Freedom → AUTONOMY_FOCUSED
    (r"(freedom|flexible|any order|open world|no restriction)", "FREEDOM_FIRST"),
    (r"(user chooses|user choice|self-directed)", "AUTONOMY_USER"),
    
    # Domain-specific film
    (r"(industry standard|stripboard|above-the-line)", "INDUSTRY_STANDARD"),
    (r"(simplified|plain language|beginner)", "SIMPLIFIED_APPROACH"),
    
    # Tech choices
    (r"(react|vue|svelte|next\.js)", "TECH_FRONTEND"),
    (r"(python|node|go|rust)", "TECH_BACKEND"),
    (r"(postgres|mongo|sqlite)", "TECH_DATABASE"),
    
    # Visual/Design
    (r"(dark mode|cinematic|premium)", "VISUAL_PREMIUM"),
    (r"(light mode|clean|minimal)", "VISUAL_CLEAN"),
    (r"(vibrant|bold|playful)", "VISUAL_PLAYFUL"),
    
    # Accessibility
    (r"(wcag aaa|maximum accessibility)", "ACCESSIBILITY_MAXIMUM"),
    (r"(wcag aa|standard compliance)", "ACCESSIBILITY_STANDARD"),
    (r"(best effort|no strict)", "ACCESSIBILITY_MINIMAL"),
    
    # Collaboration
    (r"(real-time sync|live cursors)", "COLLAB_REALTIME"),
    (r"(async|comment)", "COLLAB_ASYNC"),
    (r"(turn-based|one person)", "COLLAB_SEQUENTIAL"),
    
    # Mobile
    (r"(pwa|installable web)", "MOBILE_PWA"),
    (r"(native apps|ios/android)", "MOBILE_NATIVE"),
    (r"(responsive web)", "MOBILE_RESPONSIVE"),
    (r"(offline|queue changes)", "OFFLINE_CAPABLE"),
]


def infer_hidden_value(description: str, label: str) -> str:
    """Infer hidden_value from choice description."""
    desc_lower = description.lower()
    
    for pattern, hidden_value in PATTERNS:
        if re.search(pattern, desc_lower):
            return hidden_value
    
    # Fallback based on choice position
    if label == "A":
        return "OPTION_PRIMARY"
    elif label == "B":
        return "OPTION_BALANCED"
    elif label == "C":
        return "OPTION_ALTERNATIVE"
    else:
        return "OPTION_OTHER"


def main():
    """Update all interview questions with hidden_value."""
    conn_str = os.getenv("DATABASE_URL", "postgresql://mcp_admin:12345@localhost:5432/mcp_pas")
    
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get all questions
    cur.execute("SELECT id, dimension_id, choices FROM interview_questions")
    questions = cur.fetchall()
    
    print(f"Found {len(questions)} questions")
    
    updated_count = 0
    already_has_count = 0
    
    for q in questions:
        choices = q["choices"]
        if not choices:
            continue
        
        modified = False
        for choice in choices:
            if "hidden_value" not in choice or not choice["hidden_value"]:
                hidden_value = infer_hidden_value(
                    choice.get("description", ""),
                    choice.get("label", "X")
                )
                choice["hidden_value"] = hidden_value
                modified = True
            else:
                already_has_count += 1
        
        if modified:
            # Update the question
            cur.execute(
                "UPDATE interview_questions SET choices = %s WHERE id = %s",
                (json.dumps(choices), q["id"])
            )
            updated_count += 1
    
    conn.commit()
    
    print(f"Updated {updated_count} questions with hidden_value")
    print(f"Already had hidden_value: {already_has_count} choices")
    
    # Verify
    cur.execute("""
        SELECT d.name as domain_name, COUNT(*) as total,
               SUM(CASE WHEN c->>'hidden_value' IS NOT NULL THEN 1 ELSE 0 END) as with_hv
        FROM interview_questions q
        JOIN interview_dimensions dim ON q.dimension_id = dim.id
        JOIN interview_domains d ON dim.domain_id = d.id,
             jsonb_array_elements(q.choices) as c
        GROUP BY d.name
        ORDER BY d.name
    """)
    
    print("\n=== Coverage by Domain ===")
    for row in cur.fetchall():
        print(f"{row['domain_name']}: {row['with_hv']}/{row['total']} choices have hidden_value")
    
    cur.close()
    conn.close()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
