"""
AgentX Top-Level Config Module
Re-exports from agentx.config for root-level script execution.
"""
from agentx.config import AgentXConfig, DEFAULT_CONFIG, get_config

__all__ = ["AgentXConfig", "DEFAULT_CONFIG", "get_config"]
