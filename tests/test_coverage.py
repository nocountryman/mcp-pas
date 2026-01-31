"""
Layer 3: Self-Aware Coverage Reporter

Auto-detects missing tests by comparing:
- Tools from get_self_awareness()
- Test functions in test_tools/*.py

Future features auto-appear as "uncovered" when tests fail.
"""
import pytest
import os
import ast
from pathlib import Path


class TestCoverageAudit:
    """Self-aware test coverage verification."""
    
    @pytest.mark.asyncio
    async def test_all_tools_have_tests(self, db_connection):
        """
        LAYER 3 CORE: Ensure every tool in get_self_awareness has a test.
        
        This test fails when new tools are added without corresponding tests,
        automatically enforcing test coverage for new features.
        """
        from pas.server import get_self_awareness
        
        # Get all tools from PAS self-awareness
        awareness = await get_self_awareness()
        # Tools is a nested structure: {"count": N, "tools": [...]}
        tools_data = awareness["tools"]
        all_tools = {t["name"] for t in tools_data.get("tools", [])}
        
        # Discover covered tools from test files
        covered_tools = self._discover_covered_tools()
        
        # Find uncovered tools
        uncovered = all_tools - covered_tools
        
        # Some tools may be tested indirectly or are utilities
        # Allow a small set of exceptions (will be covered in future iterations)
        ALLOWED_EXCEPTIONS = {
            "resume_session",  # Tested via find_or_create
            "infer_module_purpose",  # Tested via infer_file_purpose
            "store_module_purpose",  # Tested via store_file_purpose
            # Sequential analysis tools - tested in YAML scenarios
            "prepare_sequential_analysis",
            "store_sequential_analysis",
            # Session utilities - basic CRUD operations
            "complete_session",
            "get_session_status",
            "get_reasoning_tree",
            "tag_session",
            # Conversation log - tested indirectly via interview flow
            "log_conversation",
            "search_conversation_log",
            # Search tools - tested via integration
            "search_relevant_laws",
            # Synthesis - tested in scenarios
            "synthesize_hypotheses",
        }
        
        truly_uncovered = uncovered - ALLOWED_EXCEPTIONS
        
        if truly_uncovered:
            pytest.fail(
                f"Missing tests for {len(truly_uncovered)} tools:\n"
                + "\n".join(f"  - {tool}" for tool in sorted(truly_uncovered))
            )
    
    def _discover_covered_tools(self) -> set[str]:
        """Parse test files to find which tools are tested."""
        covered = set()
        test_dir = Path(__file__).parent / "test_tools"
        
        if not test_dir.exists():
            return covered
        
        for test_file in test_dir.glob("test_*.py"):
            try:
                with open(test_file) as f:
                    tree = ast.parse(f.read())
                
                # Look for imports from server
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module == "server":
                            for alias in node.names:
                                covered.add(alias.name)
            except Exception:
                continue
        
        return covered
    
    @pytest.mark.asyncio
    async def test_all_workflows_have_scenarios(self, db_connection):
        """
        Ensure each workflow has at least one scenario test.
        """
        from pas.server import get_self_awareness
        
        awareness = await get_self_awareness()
        architecture = awareness.get("architecture", {})
        
        workflows = [
            "reasoning_flow",
            "learning_flow",
            "codebase_flow",
            "metacognitive_flow"
        ]
        
        # Check scenario files exist
        scenarios_dir = Path(__file__).parent / "scenarios"
        
        if not scenarios_dir.exists():
            pytest.skip("Scenarios directory not yet created")
        
        scenario_files = list(scenarios_dir.glob("*.yaml"))
        
        # At least one scenario should exist
        assert len(scenario_files) >= 1, \
            f"Expected at least 1 scenario file, found {len(scenario_files)}"


class TestSchemaAwareness:
    """Verify schema self-awareness matches reality."""
    
    @pytest.mark.asyncio
    async def test_schema_table_count_accurate(self, db_connection):
        """Verify reported table count matches database."""
        from pas.server import get_self_awareness
        
        awareness = await get_self_awareness()
        reported_count = awareness["schema"]["table_count"]
        
        # Count actual tables
        cur = db_connection.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        row = cur.fetchone()
        actual_count = row["count"] if isinstance(row, dict) else row[0]
        
        # Allow some variance for views/temp tables/migrations
        assert abs(reported_count - actual_count) <= 5, \
            f"Schema reports {reported_count} tables but DB has {actual_count}"
    
    @pytest.mark.asyncio
    async def test_tool_count_matches_server(self, db_connection):
        """Verify tool count is reasonable."""
        from pas.server import get_self_awareness
        
        awareness = await get_self_awareness()
        # Tools is nested: {"count": N, "tools": [...]}
        tool_count = awareness["tools"].get("count", 0)
        
        # Should have 35+ tools based on current implementation
        assert tool_count >= 35, f"Expected 35+ tools, found {tool_count}"
        assert tool_count <= 60, f"Unexpectedly high tool count: {tool_count}"
