"""
PAS Metacognitive Helpers (v40 Phase 2)

5-Stage Metacognitive Prompting based on arXiv:2308.05342v4:
1. Understanding - restate problem
2. Preliminary Judgment - initial hypothesis
3. Critical Evaluation - challenge assumptions
4. Final Decision - synthesize
5. Confidence Expression - calibrated uncertainty
"""

from typing import Dict, Any, Optional, List


# Metacognitive stage definitions
METACOGNITIVE_STAGES = {
    0: {
        "name": "Not Started",
        "prompt": None,
        "required_output": None
    },
    1: {
        "name": "Understanding",
        "prompt": """**STAGE 1: UNDERSTANDING**

Before generating hypotheses, restate the problem in your own words:
- What is the core problem we're trying to solve?
- What are the key constraints?
- What would success look like?

Provide your restatement, then call store_expansion with your hypotheses.""",
        "required_output": "problem_restatement"
    },
    2: {
        "name": "Preliminary Judgment",
        "prompt": """**STAGE 2: PRELIMINARY JUDGMENT**

Generate 2-3 initial hypotheses with confidence scores (0.0-1.0).
For each hypothesis:
- State the approach clearly
- Assign initial confidence based on your certainty
- Declare the scope (files/modules affected)

Call store_expansion with your hypotheses.""",
        "required_output": "hypotheses"
    },
    3: {
        "name": "Critical Evaluation",
        "prompt": """**STAGE 3: CRITICAL EVALUATION**

Challenge each hypothesis:
- What assumptions are we making?
- What could go wrong?
- What are we missing?

Call prepare_critique on your top hypothesis, then store_critique.""",
        "required_output": "critiques"
    },
    4: {
        "name": "Final Decision",
        "prompt": """**STAGE 4: FINAL DECISION**

Synthesize insights from stages 1-3:
- Which hypothesis survived critique?
- What modifications are needed?
- What's the recommended path forward?

Call finalize_session to get the recommendation.""",
        "required_output": "recommendation"
    },
    5: {
        "name": "Confidence Expression",
        "prompt": """**STAGE 5: CALIBRATED CONFIDENCE**

Express your final confidence using this calibration scale:

| Score | Meaning | When to Use |
|-------|---------|-------------|
| 0.90+ | Virtually certain | Strong evidence, tested approach |
| 0.70-0.89 | Likely | Good evidence, some uncertainty |
| 0.50-0.69 | Possible | Mixed evidence, notable risks |
| <0.50 | Uncertain | Weak evidence, needs more analysis |

State your calibrated confidence and reasoning.""",
        "required_output": "calibrated_confidence"
    }
}


def get_stage_info(stage: int) -> Dict[str, Any]:
    """Get information about a metacognitive stage."""
    if stage not in METACOGNITIVE_STAGES:
        return {"error": f"Invalid stage: {stage}. Valid stages: 0-5"}
    return METACOGNITIVE_STAGES[stage]


def get_stage_prompt(stage: int) -> Optional[str]:
    """Get the prompt for a specific stage."""
    info = METACOGNITIVE_STAGES.get(stage, {})
    return info.get("prompt")


def validate_stage_progression(current: int, target: int) -> Dict[str, Any]:
    """
    Validate that stage progression is valid.
    
    Rules:
    - Can advance by 1 from current
    - Can skip to stage 3 (Critical) from 1 (fast path)
    - Cannot go backwards more than 1 stage
    """
    if target == current + 1:
        return {"valid": True, "reason": "Normal progression"}
    
    if current == 1 and target == 3:
        return {"valid": True, "reason": "Fast path: skip Preliminary (hypotheses already in goal)"}
    
    if target == current:
        return {"valid": True, "reason": "Repeat current stage"}
    
    if target < current:
        return {"valid": False, "reason": f"Cannot go backwards from stage {current} to {target}"}
    
    if target > current + 1:
        return {"valid": False, "reason": f"Cannot skip from stage {current} to {target}"}
    
    return {"valid": False, "reason": "Unknown progression"}


def get_calibration_guidance(confidence: float) -> str:
    """
    Get calibration guidance for a confidence score.
    
    Args:
        confidence: Score between 0.0 and 1.0
        
    Returns:
        Calibration guidance string
    """
    if confidence >= 0.90:
        return "Virtually certain - strong evidence, tested approach"
    elif confidence >= 0.70:
        return "Likely - good evidence, some uncertainty"
    elif confidence >= 0.50:
        return "Possible - mixed evidence, notable risks"
    else:
        return "Uncertain - weak evidence, needs more analysis"


def format_stage_status(stage: int) -> str:
    """Format current stage for display."""
    info = METACOGNITIVE_STAGES.get(stage, {})
    name = info.get("name", "Unknown")
    return f"Stage {stage}/5: {name}"


def get_stages_summary() -> List[Dict[str, str]]:
    """Get summary of all stages for display."""
    return [
        {"stage": i, "name": info["name"]}
        for i, info in METACOGNITIVE_STAGES.items()
        if i > 0
    ]
