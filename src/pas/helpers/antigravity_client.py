"""
Phase 32: Antigravity Context API Client

Integrates with Antigravity's internal Connect RPC API to access:
- Agent trajectories (GetAllCascadeTrajectories, GetCascadeTrajectory)
- User memories (GetUserMemories)
- MCP server states (GetMcpServerStates)

Based on verified reverse-engineering research from Feb 2026.
"""
import subprocess
import requests
import re
import logging
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger("pas-server")

# Connect RPC protocol constants
SERVICE_PREFIX = "exa.language_server_pb.LanguageServerService"
CONNECT_PROTOCOL_VERSION = "1"
DEFAULT_TIMEOUT = 5  # seconds


@dataclass
class AntigravityInstance:
    """Represents a discovered Antigravity language server instance."""
    pid: str
    port: int
    csrf_token: str
    workspace_id: Optional[str] = None


class AntigravityClient:
    """
    Client for Antigravity's Connect RPC API.
    
    Auto-discovers active language server instances and provides
    access to trajectories, memories, and MCP state.
    """
    
    def __init__(self, workspace_filter: Optional[str] = None):
        """
        Initialize client with auto-discovery.
        
        Args:
            workspace_filter: Optional substring to filter by workspace path
        """
        self.instances: List[AntigravityInstance] = []
        self.primary: Optional[AntigravityInstance] = None
        self._discover(workspace_filter)
    
    def _discover(self, workspace_filter: Optional[str] = None) -> None:
        """Auto-discover all active Antigravity language server instances."""
        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)
        except Exception as e:
            logger.warning(f"Failed to scan processes: {e}")
            return
        
        for line in result.stdout.split('\n'):
            if 'language_server' not in line or '--csrf_token' not in line:
                continue
            
            # Apply workspace filter if specified
            if workspace_filter and workspace_filter not in line:
                continue
            
            # Extract CSRF token
            token_match = re.search(r'--csrf_token\s+([a-f0-9-]+)', line)
            if not token_match:
                continue
            
            token = token_match.group(1)
            pid = line.split()[1]
            
            # Extract workspace ID
            ws_match = re.search(r'--workspace_id\s+([^\s]+)', line)
            ws_id = ws_match.group(1) if ws_match else None
            
            # Find listening ports for this PID
            try:
                lsof = subprocess.run(
                    ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-p", pid],
                    capture_output=True, text=True, timeout=5
                )
            except Exception:
                continue
            
            for port_line in lsof.stdout.split('\n'):
                port_match = re.search(r':(\d+)\s+\(LISTEN\)', port_line)
                if not port_match:
                    continue
                
                port = int(port_match.group(1))
                
                # Verify this is the Connect RPC port via Heartbeat
                if self._verify_port(port, token):
                    instance = AntigravityInstance(
                        pid=pid,
                        port=port,
                        csrf_token=token,
                        workspace_id=ws_id
                    )
                    self.instances.append(instance)
                    
                    # Set first discovered as primary
                    if not self.primary:
                        self.primary = instance
                    break  # Found the active port for this PID
    
    def _verify_port(self, port: int, token: str) -> bool:
        """Verify if a port responds to Connect RPC Heartbeat."""
        try:
            url = f"http://127.0.0.1:{port}/{SERVICE_PREFIX}/Heartbeat"
            headers = self._build_headers(token)
            resp = requests.post(url, headers=headers, json={}, timeout=1)
            return "lastExtensionHeartbeat" in resp.text
        except Exception:
            return False
    
    def _build_headers(self, token: str) -> Dict[str, str]:
        """Build Connect RPC headers with CSRF token."""
        return {
            "Content-Type": "application/json",
            "X-Codeium-Csrf-Token": token,
            "X-Csrf-Token": token,  # v108 dual-key support
            "Connect-Protocol-Version": CONNECT_PROTOCOL_VERSION,
            "Cookie": f"csrf_token={token}"
        }
    
    def call(self, method: str, data: Optional[Dict] = None, 
             instance: Optional[AntigravityInstance] = None,
             _retry: bool = True) -> Dict[str, Any]:
        """
        Call an Antigravity API method.
        
        Args:
            method: RPC method name (e.g., "GetUserMemories")
            data: Optional request payload
            instance: Specific instance to use (defaults to primary)
            _retry: Internal flag to prevent infinite retry loops
            
        Returns:
            JSON response from the API
        """
        inst = instance or self.primary
        if not inst:
            return {"error": "No Antigravity instance discovered"}
        
        url = f"http://127.0.0.1:{inst.port}/{SERVICE_PREFIX}/{method}"
        headers = self._build_headers(inst.csrf_token)
        payload = data or {"metadata": {"ideName": "antigravity"}}
        
        try:
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=DEFAULT_TIMEOUT
            )
            return response.json()
        except requests.exceptions.Timeout:
            return {"error": f"Timeout calling {method}"}
        except requests.exceptions.ConnectionError:
            # Phase 34: Retry with re-discovery on connection failure
            if _retry:
                logger.info(f"Connection failed to port {inst.port}, re-discovering...")
                self.instances = []
                self.primary = None
                self._discover()
                if self.primary:
                    return self.call(method, data, self.primary, _retry=False)
            return {"error": f"Connection failed to port {inst.port}"}
        except Exception as e:
            return {"error": str(e)}

    
    # Convenience methods for common operations
    
    def get_user_memories(self) -> Dict[str, Any]:
        """Get stored agent rules and guidelines."""
        return self.call("GetUserMemories")
    
    def get_all_trajectories(self) -> Dict[str, Any]:
        """Get list of all agent session trajectories."""
        return self.call("GetAllCascadeTrajectories")
    
    def get_trajectory(self, cascade_id: str) -> Dict[str, Any]:
        """Get detailed steps for a specific agent session.
        
        Note: May timeout for large sessions (1000+ steps).
        Use get_trajectory_steps() for reliable retrieval.
        """
        return self.call("GetCascadeTrajectory", {"cascadeId": cascade_id})
    
    def get_trajectory_steps(self, cascade_id: str, limit: int = 100) -> List[Dict]:
        """Get trajectory steps with reliable retrieval.
        
        Uses GetCascadeTrajectorySteps which handles large sessions.
        
        Args:
            cascade_id: The cascade/session ID
            limit: Max steps to return (default 100)
            
        Returns:
            List of step dicts with tool calls
        """
        result = self.call("GetCascadeTrajectorySteps", {
            "cascadeId": cascade_id,
            "limit": limit
        })
        steps = result.get("steps", [])
        if isinstance(steps, dict):
            steps = list(steps.values())
        return steps if isinstance(steps, list) else []
    
    def get_mcp_states(self) -> Dict[str, Any]:
        """Get status of all registered MCP servers."""
        return self.call("GetMcpServerStates")
    
    def get_user_status(self) -> Dict[str, Any]:
        """Get user profile, tier, and quota info."""
        return self.call("GetUserStatus")


# Singleton for reuse
_client: Optional[AntigravityClient] = None


def get_antigravity_client(workspace_filter: Optional[str] = None, force_refresh: bool = False) -> AntigravityClient:
    """Get or create Antigravity client singleton.
    
    Args:
        workspace_filter: Optional substring to filter by workspace
        force_refresh: If True, clear cached client and re-discover
    """
    global _client
    if force_refresh or _client is None:
        _client = AntigravityClient(workspace_filter)
    return _client


def reset_antigravity_client():
    """Clear the singleton to force re-discovery on next call."""
    global _client
    _client = None
