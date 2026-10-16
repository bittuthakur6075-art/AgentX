"""
AgentX Configuration Module
===========================
Enterprise configuration management and environment setup for the AgentX engine.
Supports zero-framework HTTP REST integration with OpenAI-compatible LLM endpoints,
Groq, Ollama, DeepSeek, Anthropic, or any standard function-calling provider.
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

# Attempt to load dotenv if available
try:
    from dotenv import load_dotenv
    # Look for .env in current directory or parent directory
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass


@dataclass
class AgentXConfig:
    """
    Central configuration container for AgentX engine and runtime components.
    All properties can be overridden programmatically or via environment variables.
    """
    # LLM REST API Endpoint and Credentials
    api_key: str = field(
        default_factory=lambda: (
            os.getenv("AGENTX_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or ""
        )
    )
    api_base_url: str = field(
        default_factory=lambda: (
            os.getenv("AGENTX_API_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
    )
    model: str = field(
        default_factory=lambda: os.getenv("AGENTX_MODEL") or "gpt-4o-mini"
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("AGENTX_TEMPERATURE", "0.2"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("AGENTX_MAX_TOKENS", "2048"))
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("AGENTX_REQUEST_TIMEOUT", "45"))
    )

    # ReAct Execution Loop Settings
    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("AGENTX_MAX_ITERATIONS", "10"))
    )
    stop_sequences: list[str] = field(
        default_factory=lambda: ["Observation:", "OBSERVATION:"]
    )

    # Context Window & Memory Management
    max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("AGENTX_MAX_CONTEXT_TOKENS", "8000"))
    )
    chars_per_token_estimate: float = 4.0  # Conservative estimate for English text
    preserve_recent_messages: int = field(
        default_factory=lambda: int(os.getenv("AGENTX_PRESERVE_RECENT_MESSAGES", "6"))
    )

    # Vector Memory & Embeddings
    embedding_api_key: Optional[str] = field(
        default_factory=lambda: (
            os.getenv("AGENTX_EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or None
        )
    )
    embedding_base_url: str = field(
        default_factory=lambda: (
            os.getenv("AGENTX_EMBEDDING_BASE_URL")
            or os.getenv("AGENTX_API_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("AGENTX_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(os.getenv("AGENTX_EMBEDDING_DIM", "1536"))
    )

    # Native Tool Parameters
    sqlite_db_path: str = field(
        default_factory=lambda: os.getenv("AGENTX_SQLITE_DB", "agentx_sandbox.db")
    )
    code_execution_timeout: int = field(
        default_factory=lambda: int(os.getenv("AGENTX_CODE_TIMEOUT", "10"))
    )
    web_scraper_timeout: int = field(
        default_factory=lambda: int(os.getenv("AGENTX_SCRAPER_TIMEOUT", "15"))
    )
    web_scraper_max_length: int = field(
        default_factory=lambda: int(os.getenv("AGENTX_SCRAPER_MAX_LEN", "5000"))
    )

    # Logging and Diagnostics
    log_level: str = field(
        default_factory=lambda: os.getenv("AGENTX_LOG_LEVEL", "INFO")
    )
    enable_verbose_output: bool = field(
        default_factory=lambda: os.getenv("AGENTX_VERBOSE", "true").lower() in ("1", "true", "yes")
    )

    def get_auth_headers(self) -> Dict[str, str]:
        """Generate HTTP authorization headers for OpenAI-compatible REST APIs."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AgentX-Autonomous-Engine/1.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_embedding_headers(self) -> Dict[str, str]:
        """Generate HTTP headers for embedding API endpoints."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AgentX-Memory-Engine/1.0"
        }
        key = self.embedding_api_key or self.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def validate_api_access(self) -> tuple[bool, str]:
        """Validate whether API configuration has necessary parameters."""
        if not self.api_key:
            return False, (
                "No API Key found. Set OPENAI_API_KEY or AGENTX_API_KEY environment variable, "
                "or pass `api_key` directly to AgentXConfig."
            )
        if not self.api_base_url:
            return False, "api_base_url cannot be empty."
        return True, "Configuration valid."


# Global default configuration instance
DEFAULT_CONFIG = AgentXConfig()


def get_config(**kwargs: Any) -> AgentXConfig:
    """
    Factory function to retrieve or customize configuration.
    Accepts keyword arguments matching AgentXConfig fields.
    """
    if not kwargs:
        return DEFAULT_CONFIG
    return AgentXConfig(**kwargs)
