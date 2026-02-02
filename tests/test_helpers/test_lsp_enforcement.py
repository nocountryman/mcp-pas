"""
Tests for Phase 3: LSP Enforcement in validate_plan.

Tests the LSP section check and scope validation.
"""
import pytest
import uuid


@pytest.fixture
def test_session(db_connection):
    """Create a test session with proper UUID."""
    conn = db_connection
    cur = conn.cursor()
    
    session_id = str(uuid.uuid4())
    
    # Create session - note: column is 'state' not 'status'
    cur.execute("""
        INSERT INTO reasoning_sessions (id, goal, state, context)
        VALUES (%s, 'Test LSP enforcement', 'active', '{}')
        RETURNING id
    """, (session_id,))
    conn.commit()
    
    yield session_id
    
    # Cleanup
    cur.execute("DELETE FROM thought_nodes WHERE session_id = %s", (session_id,))
    cur.execute("DELETE FROM reasoning_sessions WHERE id = %s", (session_id,))
    conn.commit()


@pytest.mark.asyncio
async def test_validate_plan_without_lsp_impact(test_session):
    """Test that lsp_section_check is None when no lsp_impact provided."""
    from pas.server import validate_plan
    
    result = await validate_plan(
        session_id=test_session,
        plan_text="# Test Plan\n\nSome content here.",
        lsp_impact=None
    )
    
    # Without lsp_impact, lsp_section_check should not be in response
    assert result.get("lsp_section_check") is None


@pytest.mark.asyncio
async def test_validate_plan_with_lsp_section_present(test_session):
    """Test detection of LSP Impact Analysis section."""
    from pas.server import validate_plan
    
    plan_with_section = """
# Implementation Plan

## Proposed Changes
Add new function to utils.py

## LSP Impact Analysis
- Symbol `get_user` is referenced in 3 files
- No external callers outside scope
"""
    
    result = await validate_plan(
        session_id=test_session,
        plan_text=plan_with_section,
        lsp_impact={"lsp_available": True, "callers_outside_scope": []}
    )
    
    assert result.get("lsp_section_check") is not None
    assert result["lsp_section_check"]["has_lsp_section"] is True
    assert result["lsp_section_check"]["warning"] is None


@pytest.mark.asyncio
async def test_validate_plan_missing_lsp_section(test_session):
    """Test warning when lsp_impact provided but no section."""
    from pas.server import validate_plan
    
    plan_without_section = """
# Implementation Plan

## Proposed Changes
Add new function to utils.py

## Verification
Run tests
"""
    
    result = await validate_plan(
        session_id=test_session,
        plan_text=plan_without_section,
        lsp_impact={"lsp_available": True, "callers_outside_scope": []}
    )
    
    assert result["lsp_section_check"]["has_lsp_section"] is False
    assert result["lsp_section_check"]["warning"] is not None
    assert "LSP Impact Analysis" in result["lsp_section_check"]["warning"]


@pytest.mark.asyncio
async def test_validate_plan_scope_warnings(test_session):
    """Test scope warnings for unaddressed external callers."""
    from pas.server import validate_plan
    
    plan_text = """
# Implementation Plan

## LSP Impact Analysis
Analyzed symbol usage.
"""
    
    lsp_impact = {
        "lsp_available": True,
        "callers_outside_scope": ["/path/to/external_caller.py"]
    }
    
    result = await validate_plan(
        session_id=test_session,
        plan_text=plan_text,
        lsp_impact=lsp_impact
    )
    
    assert result.get("scope_warnings") is not None
    assert len(result["scope_warnings"]) == 1
    assert "external_caller.py" in result["scope_warnings"][0]["file"]


@pytest.mark.asyncio  
async def test_validate_plan_scope_warning_addressed(test_session):
    """Test no scope warning when external caller is in plan."""
    from pas.server import validate_plan
    
    plan_text = """
# Implementation Plan

## LSP Impact Analysis
External caller: external_caller.py will need update.
"""
    
    lsp_impact = {
        "lsp_available": True,
        "callers_outside_scope": ["/path/to/external_caller.py"]
    }
    
    result = await validate_plan(
        session_id=test_session,
        plan_text=plan_text,
        lsp_impact=lsp_impact
    )
    
    # External caller mentioned in plan, so no scope warning
    assert result.get("scope_warnings") is None or len(result.get("scope_warnings", [])) == 0
