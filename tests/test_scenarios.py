"""
Layer 2: YAML Scenario Test Runner

Executes declarative test scenarios defined in YAML files.
Supports variable interpolation and step-by-step assertions.
"""
import pytest
import yaml
import os
from pathlib import Path


def load_scenarios():
    """Load all YAML scenario files."""
    scenarios_dir = Path(__file__).parent / "scenarios"
    
    if not scenarios_dir.exists():
        return []
    
    scenarios = []
    for yaml_file in scenarios_dir.glob("*.yaml"):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
            if data:
                data["_file"] = yaml_file.name
                scenarios.append(data)
    
    return scenarios


SCENARIOS = load_scenarios()


class ScenarioRunner:
    """Executes a single scenario with variable tracking."""
    
    def __init__(self, scenario: dict):
        self.scenario = scenario
        self.variables = {}
    
    async def run(self):
        """Execute all steps in the scenario."""
        from server import (
            start_reasoning_session, prepare_expansion, store_expansion,
            prepare_critique, store_critique, finalize_session, record_outcome
        )
        
        # Tool mapping
        tools = {
            "start_reasoning_session": start_reasoning_session,
            "prepare_expansion": prepare_expansion,
            "store_expansion": store_expansion,
            "prepare_critique": prepare_critique,
            "store_critique": store_critique,
            "finalize_session": finalize_session,
            "record_outcome": record_outcome,
        }
        
        results = []
        
        for i, step in enumerate(self.scenario.get("steps", [])):
            tool_name = step["tool"]
            args = self._interpolate_args(step.get("args", {}))
            
            if tool_name not in tools:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            # Execute tool
            result = await tools[tool_name](**args)
            results.append(result)
            
            # Store variables for interpolation
            if step.get("store"):
                for var_name, path in step["store"].items():
                    self.variables[var_name] = self._extract_value(result, path)
            
            # Auto-store common values
            if "session_id" in result:
                self.variables["session_id"] = result["session_id"]
            if "created_nodes" in result and result["created_nodes"]:
                self.variables["node_id"] = result["created_nodes"][0].get("node_id")
            
            # Run assertions
            if step.get("assert"):
                self._check_assertions(result, step["assert"], i + 1)
        
        return results
    
    def _interpolate_args(self, args: dict) -> dict:
        """Replace $variable references with stored values."""
        interpolated = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                var_name = value[1:]
                if var_name in self.variables:
                    interpolated[key] = self.variables[var_name]
                else:
                    raise ValueError(f"Variable not found: {var_name}")
            else:
                interpolated[key] = value
        return interpolated
    
    def _extract_value(self, obj: dict, path: str):
        """Extract value from nested dict using dot notation."""
        parts = path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]
            else:
                return None
        return current
    
    def _check_assertions(self, result: dict, assertions: list, step_num: int):
        """Verify assertions against result."""
        for assertion in assertions:
            # Simple equality check: field == value
            if "==" in assertion:
                field, expected = assertion.split("==")
                field = field.strip()
                expected = expected.strip()
                
                actual = self._extract_value(result, field)
                
                # Handle special values
                if expected == "true":
                    expected = True
                elif expected == "false":
                    expected = False
                elif expected == "null":
                    expected = None
                elif expected.startswith('"') and expected.endswith('"'):
                    # Handle quoted strings
                    expected = expected[1:-1]
                elif expected.startswith("'") and expected.endswith("'"):
                    # Handle single-quoted strings
                    expected = expected[1:-1]
                elif expected.isdigit():
                    # Handle integer values
                    expected = int(expected)
                elif expected.replace(".", "", 1).isdigit():
                    # Handle float values
                    expected = float(expected)
                
                assert actual == expected, \
                    f"Step {step_num}: {field} == {actual}, expected {expected}"
            
            # Existence check: field != null
            elif "!= null" in assertion:
                field = assertion.replace("!= null", "").strip()
                actual = self._extract_value(result, field)
                assert actual is not None, \
                    f"Step {step_num}: {field} is null"
            
            # Length check: field.length >= N
            elif ".length >=" in assertion:
                parts = assertion.split(".length >=")
                field = parts[0].strip()
                expected_len = int(parts[1].strip())
                actual = self._extract_value(result, field)
                assert len(actual) >= expected_len, \
                    f"Step {step_num}: {field} length is {len(actual)}, expected >= {expected_len}"


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.get("name", s.get("_file", "unknown")))
async def test_scenario(scenario, db_connection):
    """Run a YAML-defined test scenario."""
    runner = ScenarioRunner(scenario)
    await runner.run()
