"""
Phase 31: Multi-Model Verification via Gemini CLI

Uses Gemini CLI as subprocess for adversarial critique of hypotheses.
Zero API cost - leverages existing Gemini subscription.
"""
import subprocess
import json
import logging
from typing import Optional

logger = logging.getLogger("pas-server")

GEMINI_MODEL = "gemini-3-pro-preview"
VERIFICATION_TIMEOUT = 120  # seconds


async def verify_with_gemini(hypothesis: str, context: Optional[str] = None) -> dict:
    """
    Call Gemini CLI as subprocess for adversarial critique.
    
    Args:
        hypothesis: The hypothesis text to verify
        context: Optional additional context (goal, constraints)
        
    Returns:
        {
            "model": "gemini-3-pro-preview",
            "major_flaws": [...],
            "minor_issues": [...],
            "confidence": 0.0-1.0,
            "raw_critique": "..."
        }
    """
    prompt = f'''You are a skeptical code reviewer performing adversarial verification.

Your job is to find logical flaws, edge cases, and missing requirements in this hypothesis.
Be critical but fair. Look for:
1. Logical contradictions
2. Missing edge cases  
3. Unstated assumptions
4. Dependency risks
5. Scope creep indicators

HYPOTHESIS:
{hypothesis}
'''
    if context:
        prompt += f'''
CONTEXT:
{context}
'''
    
    prompt += '''
Return your analysis as JSON:
{
  "major_flaws": ["list of critical issues that would invalidate the hypothesis"],
  "minor_issues": ["list of concerns worth noting but not blocking"],
  "confidence": 0.0 to 1.0 (how confident are you in this hypothesis after critique),
  "summary": "one-line summary of your verdict"
}
'''
    
    try:
        # Run Gemini CLI as subprocess
        result = subprocess.run(
            ["gemini", "-m", GEMINI_MODEL, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=VERIFICATION_TIMEOUT
        )
        
        if result.returncode != 0:
            logger.warning(f"Gemini CLI failed: {result.stderr[:200]}")
            return {
                "model": GEMINI_MODEL,
                "major_flaws": [],
                "minor_issues": [],
                "confidence": 0.5,
                "error": result.stderr[:500]
            }
        
        raw_output = result.stdout.strip()
        
        # Try to extract JSON from output
        critique = _parse_gemini_response(raw_output)
        critique["model"] = GEMINI_MODEL
        critique["raw_critique"] = raw_output[:2000]
        
        return critique
        
    except subprocess.TimeoutExpired:
        logger.warning("Gemini CLI timed out")
        return {
            "model": GEMINI_MODEL,
            "major_flaws": [],
            "minor_issues": [],
            "confidence": 0.5,
            "error": "timeout"
        }
    except FileNotFoundError:
        logger.error("Gemini CLI not found - install with: pip install google-genai")
        return {
            "model": GEMINI_MODEL,
            "major_flaws": [],
            "minor_issues": [],
            "confidence": 0.5,
            "error": "gemini CLI not installed"
        }


def _parse_gemini_response(output: str) -> dict:
    """Extract JSON from Gemini CLI output."""
    # Try to find JSON block
    try:
        # Look for JSON in code blocks
        if "```json" in output:
            start = output.find("```json") + 7
            end = output.find("```", start)
            json_str = output[start:end].strip()
            return json.loads(json_str)
        
        # Try parsing whole output as JSON
        if output.startswith("{"):
            return json.loads(output)
        
        # Fallback: extract structured data manually
        return {
            "major_flaws": [],
            "minor_issues": [],
            "confidence": 0.5,
            "summary": output[:500]
        }
        
    except json.JSONDecodeError:
        return {
            "major_flaws": [],
            "minor_issues": [],
            "confidence": 0.5,
            "summary": output[:500]
        }
