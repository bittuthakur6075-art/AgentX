"""
AgentX: Zero-Framework Autonomous Agentic Engine & Control System
==================================================================
Production-grade modular AI agent system written from scratch in pure Python.
Direct LLM REST API interfacing, custom ReAct loop, safe tool sandbox, and vector memory.
"""

from agentx.config import AgentXConfig, DEFAULT_CONFIG, get_config
from agentx.tools import (
    agentx_tool,
    ToolRegistry,
    AgentXTool,
    DEFAULT_REGISTRY,
    get_default_registry,
    math_evaluator,
    web_scraper,
    db_query_tool,
    code_executor,
)
from agentx.memory import (
    VectorMemory,
    DEFAULT_MEMORY,
    get_default_memory,
    SlidingWindowChunker,
    TextChunk,
    SearchResult,
    memory_search,
)
from agentx.engine import (
    AgentXEngine,
    create_engine,
    AgentXResponse,
    ReActStep,
    ChatMessage,
    ChatContextManager,
)

__version__ = "1.0.0"
__all__ = [
    # Engine & Context
    "AgentXEngine",
    "create_engine",
    "AgentXResponse",
    "ReActStep",
    "ChatMessage",
    "ChatContextManager",
    # Tools & Registry
    "agentx_tool",
    "ToolRegistry",
    "AgentXTool",
    "DEFAULT_REGISTRY",
    "get_default_registry",
    "math_evaluator",
    "web_scraper",
    "db_query_tool",
    "code_executor",
    # Memory & RAG
    "VectorMemory",
    "DEFAULT_MEMORY",
    "get_default_memory",
    "SlidingWindowChunker",
    "TextChunk",
    "SearchResult",
    "memory_search",
    # Configuration
    "AgentXConfig",
    "DEFAULT_CONFIG",
    "get_config",
]
