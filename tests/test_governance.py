"""Tests for v88/v89 governance schema functions.

PAS Research Session: b995dca2-6f1e-47f7-8e01-a0e5452026ce
PAS Implementation Session: 25456d82-d9ee-49b7-b61f-4f3e82068d41
v89 Session: 8a0e1440-84e5-4d3a-968c-dc0ca0062151
"""
import pytest
import uuid

from pas.helpers.governance import (
    create_roadmap, get_roadmaps, update_roadmap,
    create_phase_success_criterion, update_success_criterion,
    get_phase_success_criteria, create_phase_dependency,
    get_phase_dependencies, create_phase_critique, update_phase_critique,
    get_phase_critiques, create_roadmap_phase,
    # v89 additions
    create_cross_phase_decision, get_cross_phase_decisions,
    update_phase_dual_recommendation, store_research_findings
)



@pytest.fixture
def test_project_id():
    """Generate unique project ID for test isolation."""
    return f"test-governance-v88-{uuid.uuid4().hex[:8]}"


class TestRoadmaps:
    """Tests for roadmap CRUD operations."""
    
    def test_create_roadmap(self, test_project_id):
        """Test roadmap creation."""
        roadmap_id = create_roadmap(
            project_id=test_project_id,
            title="Test Roadmap v88",
            version_tag="v1.0",
            priority_taxonomy={"P0": "Critical", "P1": "High"}
        )
        assert roadmap_id is not None
        assert len(roadmap_id) == 36  # UUID format
        
        # Verify retrieval
        roadmaps = get_roadmaps(test_project_id)
        assert len(roadmaps) >= 1
        assert any(r['title'] == "Test Roadmap v88" for r in roadmaps)

    def test_update_roadmap(self, test_project_id):
        """Test roadmap update."""
        roadmap_id = create_roadmap(
            project_id=test_project_id,
            title="Update Test Roadmap"
        )
        
        result = update_roadmap(roadmap_id, status="archived")
        assert result is True
        
        roadmaps = get_roadmaps(test_project_id, status="archived")
        assert any(r['id'] == roadmap_id for r in roadmaps)


class TestPhaseSuccessCriteria:
    """Tests for phase success criteria CRUD."""
    
    def test_phase_success_criteria_crud(self, test_project_id):
        """Test success criteria CRUD."""
        # Create a phase first
        phase = create_roadmap_phase(
            test_project_id, "Test Criteria Phase", "Test", "planned"
        )
        phase_id = phase['id']
        
        # Create criteria
        criterion_id = create_phase_success_criterion(
            phase_id, "Tests pass", 1
        )
        assert criterion_id is not None
        
        # Get criteria
        criteria = get_phase_success_criteria(phase_id)
        assert len(criteria) >= 1
        assert criteria[0]['criterion'] == "Tests pass"
        assert criteria[0]['done'] is False
        
        # Mark done
        result = update_success_criterion(criterion_id, done=True)
        assert result is True
        
        criteria = get_phase_success_criteria(phase_id)
        matching = [c for c in criteria if c['id'] == criterion_id]
        assert len(matching) == 1
        assert matching[0]['done'] is True


class TestPhaseDependencies:
    """Tests for phase dependency management."""
    
    def test_phase_dependencies(self, test_project_id):
        """Test phase dependency creation."""
        phase1 = create_roadmap_phase(test_project_id, "Dep Phase 1", "", "planned")
        phase2 = create_roadmap_phase(test_project_id, "Dep Phase 2", "", "planned")
        
        phase1_id = phase1['id']
        phase2_id = phase2['id']
        
        # Create dependency (phase2 depends on phase1)
        result = create_phase_dependency(phase2_id, phase1_id)
        assert result is True
        
        # Get dependencies
        deps = get_phase_dependencies(phase2_id)
        assert len(deps) >= 1
        assert any(d['id'] == phase1_id for d in deps)


class TestPhaseCritiques:
    """Tests for phase critique tracking."""
    
    def test_phase_critiques(self, test_project_id):
        """Test phase critique tracking."""
        phase = create_roadmap_phase(
            test_project_id, "Critique Phase", "", "planned"
        )
        phase_id = phase['id']
        
        # Create critique
        critique_id = create_phase_critique(
            phase_id, "Missing error handling", "open"
        )
        assert critique_id is not None
        
        # Get open critiques
        critiques = get_phase_critiques(phase_id, status="open")
        assert len(critiques) >= 1
        assert any(c['critique_text'] == "Missing error handling" for c in critiques)
        
        # Update to addressed
        result = update_phase_critique(critique_id, "addressed")
        assert result is True
        
        # Verify status change
        critiques = get_phase_critiques(phase_id, status="addressed")
        assert any(c['id'] == critique_id for c in critiques)


# =============================================================================
# v89 Tests
# =============================================================================

class TestCrossPhaseDecisions:
    """Tests for cross-phase decision tracking."""
    
    def test_create_and_get_decision(self, test_project_id):
        """Test creating and retrieving cross-phase decisions."""
        decision_id = create_cross_phase_decision(
            project_id=test_project_id,
            decision_summary="Use JSONB for flexible metadata",
            options_considered=["Separate tables", "JSONB column", "Hybrid"],
            chosen_option="Hybrid",
            rationale="Best balance of flexibility and querying"
        )
        assert decision_id is not None
        
        decisions = get_cross_phase_decisions(test_project_id)
        assert len(decisions) >= 1
        match = [d for d in decisions if d['id'] == decision_id]
        assert len(match) == 1
        assert match[0]['decision_summary'] == "Use JSONB for flexible metadata"
        assert match[0]['chosen_option'] == "Hybrid"


class TestDualRecommendation:
    """Tests for phase dual recommendation updates."""
    
    def test_update_dual_recommendation(self, test_project_id):
        """Test updating phase dual_recommendation."""
        phase = create_roadmap_phase(
            test_project_id, "Dual Rec Phase", "", "planned"
        )
        phase_id = phase['id']
        
        result = update_phase_dual_recommendation(
            phase_id=phase_id,
            balanced={"approach": "minimal", "risk": "low"},
            aspirational={"approach": "full", "risk": "medium"}
        )
        assert result is True


class TestResearchFindings:
    """Tests for research findings storage."""
    
    def test_store_research_findings(self, test_project_id):
        """Test storing findings on a research artifact."""
        from pas.helpers.governance import store_artifact
        
        # Create research artifact first
        artifact_result = store_artifact(
            project_id=test_project_id,
            name="Test Research Doc",
            content="# Research\n\nFindings...",
            artifact_type="research"
        )
        artifact_id = artifact_result["id"]
        assert artifact_id is not None
        
        # Store findings
        result = store_research_findings(
            artifact_id=artifact_id,
            findings=[
                {"source": "paper1", "type": "primary", "text": "Finding A"},
                {"source": "paper2", "type": "secondary", "text": "Finding B"}
            ],
            confidence_level="high"
        )
        assert result is True


# ============================================================================
# v90: Implementation Plan Functions Tests
# ============================================================================

class TestPlanSteps:
    """Tests for plan step tracking."""
    
    def test_add_and_get_steps(self, test_project_id):
        """Test adding and retrieving plan steps."""
        from pas.helpers.governance import store_artifact, add_plan_step, get_plan_steps
        
        # Create implementation plan artifact
        artifact = store_artifact(
            project_id=test_project_id,
            name="Test Plan v90",
            content="# Test Plan\n\nSteps...",
            artifact_type="implementation_plan"
        )
        artifact_id = artifact["id"]
        
        # Add steps
        step1_id = add_plan_step(artifact_id, "Create migration", 1)
        step2_id = add_plan_step(artifact_id, "Add helper functions", 2, status="pending")
        
        assert step1_id is not None
        assert step2_id is not None
        
        # Get steps
        steps = get_plan_steps(artifact_id)
        assert len(steps) == 2
        assert steps[0]["step_order"] == 1
        assert steps[0]["content"] == "Create migration"
        assert steps[1]["step_order"] == 2
    
    def test_update_step_status(self, test_project_id):
        """Test updating step status."""
        from pas.helpers.governance import store_artifact, add_plan_step, update_step_status, get_plan_steps
        
        artifact = store_artifact(
            project_id=test_project_id,
            name="Test Plan Status",
            content="# Test",
            artifact_type="implementation_plan"
        )
        artifact_id = artifact["id"]
        
        step_id = add_plan_step(artifact_id, "Test step", 1)
        
        # Update status
        result = update_step_status(step_id, "done")
        assert result is True
        
        steps = get_plan_steps(artifact_id)
        assert steps[0]["status"] == "done"


class TestPlanScope:
    """Tests for plan scope declarations."""
    
    def test_add_and_get_scope(self, test_project_id):
        """Test adding and retrieving scope declarations."""
        from pas.helpers.governance import store_artifact, add_plan_scope, get_plan_scope
        
        artifact = store_artifact(
            project_id=test_project_id,
            name="Test Plan Scope",
            content="# Test",
            artifact_type="implementation_plan"
        )
        artifact_id = artifact["id"]
        
        # Add scope
        scope1_id = add_plan_scope(artifact_id, "src/helper.py", "modify")
        scope2_id = add_plan_scope(
            artifact_id, 
            "src/new_file.py", 
            "create",
            lsp_refs=[{"symbol": "func1", "count": 3}]
        )
        
        assert scope1_id is not None
        assert scope2_id is not None
        
        # Get scope
        scope = get_plan_scope(artifact_id)
        assert len(scope) == 2


class TestPlanChecklist:
    """Tests for plan checklist tracking."""
    
    def test_update_checklist(self, test_project_id):
        """Test updating plan checklist."""
        from pas.helpers.governance import store_artifact, update_plan_checklist
        
        artifact = store_artifact(
            project_id=test_project_id,
            name="Test Plan Checklist",
            content="# Test",
            artifact_type="implementation_plan"
        )
        artifact_id = artifact["id"]
        
        checklist = [
            {"item": "PAS score >= 0.9", "checked": True, "note": "Score: 0.95"},
            {"item": "Tests added", "checked": False, "note": None}
        ]
        
        result = update_plan_checklist(artifact_id, checklist)
        assert result is True

