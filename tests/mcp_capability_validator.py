"""
MCP Capability Validator - Phase 16a

Test server to validate Antigravity's support for MCP primitives:
- Resources (@mcp.resource)
- Prompts (@mcp.prompt)  
- Sampling (ctx.sample)
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-validator")


# -----------------------------------------------------------------------------
# Test 1: Resources
# -----------------------------------------------------------------------------
@mcp.resource("test://hello")
def test_resource() -> str:
    """Static resource to test resource support."""
    return "Resource works! This text was served from test://hello"


# -----------------------------------------------------------------------------
# Test 2: Prompts
# -----------------------------------------------------------------------------
@mcp.prompt("test_prompt")
def test_prompt(name: str = "World") -> str:
    """Simple prompt template to test prompt support."""
    return f"Hello, {name}! This prompt template is working correctly."


# -----------------------------------------------------------------------------
# Test 3: Sampling
# -----------------------------------------------------------------------------
from mcp.server.fastmcp import Context

@mcp.tool()
async def test_sampling(ctx: Context) -> dict:
    """
    Tool that requests an LLM completion via MCP sampling.
    
    This tests if Antigravity supports the sampling/createMessage protocol.
    """
    try:
        result = await ctx.sample(
            messages=[{
                "role": "user",
                "content": {"type": "text", "text": "Respond with exactly: 'Sampling works!'"}
            }],
            max_tokens=20
        )
        return {
            "success": True,
            "sampling_supported": True,
            "llm_response": str(result)
        }
    except NotImplementedError:
        return {
            "success": False,
            "sampling_supported": False,
            "error": "Sampling not implemented by client"
        }
    except AttributeError as e:
        return {
            "success": False,
            "sampling_supported": False,
            "error": f"Context doesn't have sample method: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "sampling_supported": "unknown",
            "error": f"{type(e).__name__}: {e}"
        }


# -----------------------------------------------------------------------------
# Health check tool
# -----------------------------------------------------------------------------
@mcp.tool()
def validator_status() -> dict:
    """Check what capabilities this validator exposes."""
    return {
        "server": "mcp-validator",
        "capabilities": {
            "resources": ["test://hello"],
            "prompts": ["test_prompt"],
            "sampling_tool": "test_sampling"
        },
        "how_to_test": {
            "resources": "Call list_resources, then read_resource(uri='test://hello')",
            "prompts": "Call list_prompts, then use test_prompt",
            "sampling": "Call test_sampling tool"
        }
    }


if __name__ == "__main__":
    mcp.run()
