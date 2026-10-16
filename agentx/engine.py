"""
AgentX ReAct Execution Engine
=============================
Zero-framework ReAct (Reason + Act) autonomous agent execution engine.
Implements the continuous cycle: Observe -> Think -> Call Tool -> Evaluate Tool Output -> Respond.
Features:
  - Low-level direct HTTP payload handler for LLM REST APIs (OpenAI, Groq, DeepSeek, Ollama, etc.).
  - Explicit chat context manager with proactive context window trimming.
  - Dual-mode tool calling: Native Function Calling API + Fallback ReAct text parser.
  - Streaming execution trace generator for real-time observability.
"""

from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import requests

from agentx.config import AgentXConfig, DEFAULT_CONFIG
from agentx.tools import ToolRegistry, DEFAULT_REGISTRY
from agentx.memory import VectorMemory, DEFAULT_MEMORY

# Setup engine logger
logger = logging.getLogger("AgentXEngine")


# ============================================================================
# 1. EXPLICIT CHAT CONTEXT MANAGER & TOKEN TRIMMER
# ============================================================================

@dataclass
class ChatMessage:
    """Represents a standardized chat message across all LLM REST APIs."""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to standard OpenAI REST payload dictionary."""
        payload: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        elif self.role == "assistant" and self.tool_calls:
            payload["content"] = ""  # Some endpoints require non-null content

        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        return payload

    def estimate_tokens(self, chars_per_token: float = 4.0) -> int:
        """Estimate token footprint of this message."""
        total_chars = len(self.role) + 4
        if self.content:
            total_chars += len(self.content)
        if self.name:
            total_chars += len(self.name)
        if self.tool_calls:
            total_chars += len(json.dumps(self.tool_calls))
        return max(1, int(total_chars / chars_per_token))


class ChatContextManager:
    """
    Explicit chat context window manager with intelligent trimming.
    Guarantees that system prompt and initial user goals are never discarded
    while dynamically trimming intermediate tool outputs when approaching context bounds.
    """

    def __init__(self, config: AgentXConfig):
        self.config = config
        self._messages: List[ChatMessage] = []

    def add_message(
        self,
        role: str,
        content: Optional[str] = None,
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> ChatMessage:
        """Append a message to the context buffer and enforce trimming limits."""
        msg = ChatMessage(
            role=role,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls
        )
        self._messages.append(msg)
        self.trim_context()
        return msg

    def get_messages(self) -> List[ChatMessage]:
        """Return shallow copy of current message history."""
        return list(self._messages)

    def get_payload_messages(self) -> List[Dict[str, Any]]:
        """Return message list ready for low-level HTTP payload dispatch."""
        return [m.to_dict() for m in self._messages]

    def total_estimated_tokens(self) -> int:
        """Compute cumulative token footprint of all active messages."""
        return sum(m.estimate_tokens(self.config.chars_per_token_estimate) for m in self._messages)

    def trim_context(self) -> int:
        """
        Trim intermediate conversation history if total tokens exceed max_context_tokens.
        Preserves:
          1. System prompt (index 0)
          2. Initial user query (index 1)
          3. Most recent N messages (configured via config.preserve_recent_messages)
        Returns the number of messages trimmed.
        """
        limit = self.config.max_context_tokens
        current_tokens = self.total_estimated_tokens()

        if current_tokens <= limit or len(self._messages) <= 4:
            return 0

        trimmed_count = 0
        preserve_tail = max(2, self.config.preserve_recent_messages)

        # Trimming candidates are between index 2 and (len - preserve_tail)
        while self.total_estimated_tokens() > limit and len(self._messages) > (2 + preserve_tail):
            # Locate first non-system message eligible for trimming
            # To preserve tool_call pairings, if trimming an assistant message with tool_calls,
            # we also trim its corresponding tool output.
            target_idx = 2
            removed = self._messages.pop(target_idx)
            trimmed_count += 1

            # If removed message had a tool_call, remove matching tool response if it is next
            if removed.tool_calls and target_idx < len(self._messages) and self._messages[target_idx].role == "tool":
                self._messages.pop(target_idx)
                trimmed_count += 1

        if trimmed_count > 0:
            # Insert a lightweight notice so the LLM knows history was compressed
            notice = ChatMessage(
                role="system",
                content=f"[Context Manager: {trimmed_count} prior intermediate interaction steps were summarized to preserve token window.]"
            )
            self._messages.insert(2, notice)

        return trimmed_count

    def clear(self) -> None:
        """Clear all messages from context."""
        self._messages.clear()


# ============================================================================
# 2. EXECUTION TRACE & RESPONSE DATA STRUCTURES
# ============================================================================

@dataclass
class ReActStep:
    """Detailed record of an individual ReAct cycle step."""
    step_number: int
    observation: Optional[str] = None
    thought: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    tool_raw_arguments: Optional[str] = None
    tool_output: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "observation": self.observation,
            "thought": self.thought,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "tool_output": self.tool_output,
            "duration_seconds": round(self.duration_seconds, 3)
        }


@dataclass
class AgentXResponse:
    """Final output package generated by the AgentX ReAct Engine."""
    final_answer: str
    trace: List[ReActStep]
    total_iterations: int
    total_tool_calls: int
    finish_reason: str
    execution_time_seconds: float
    token_usage_estimate: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "trace": [s.to_dict() for s in self.trace],
            "total_iterations": self.total_iterations,
            "total_tool_calls": self.total_tool_calls,
            "finish_reason": self.finish_reason,
            "execution_time_seconds": round(self.execution_time_seconds, 2),
            "token_usage_estimate": self.token_usage_estimate
        }


# ============================================================================
# 3. LOW-LEVEL HTTP PAYLOAD DISPATCHER (ZERO-FRAMEWORK)
# ============================================================================

class LowLevelLLMClient:
    """
    Direct REST API client for OpenAI-compatible LLM endpoints.
    Uses Python requests directly without LangChain, LlamaIndex, or third-party wrappers.
    """

    def __init__(self, config: AgentXConfig):
        self.config = config

    def call_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto"
    ) -> Dict[str, Any]:
        """
        Execute an HTTP POST request to the /chat/completions endpoint.
        Returns the parsed raw JSON response dictionary.
        """
        endpoint = f"{self.config.api_base_url}/chat/completions"
        headers = self.config.get_auth_headers()

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Include tools schema if tools are available
        if tools_schema:
            payload["tools"] = tools_schema
            payload["tool_choice"] = tool_choice

        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.config.request_timeout
            )

            if resp.status_code != 200:
                err_text = resp.text
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("error", {}).get("message", err_text)
                except Exception:
                    err_msg = err_text
                raise RuntimeError(f"LLM API Error (HTTP {resp.status_code}): {err_msg}")

            return resp.json()

        except requests.exceptions.Timeout:
            raise TimeoutError(f"HTTP request to LLM endpoint '{endpoint}' timed out after {self.config.request_timeout}s.")
        except requests.exceptions.ConnectionError as conn_err:
            raise ConnectionError(f"Failed to connect to LLM endpoint '{endpoint}': {str(conn_err)}")


# ============================================================================
# 4. SYSTEM PROMPT BUILDER
# ============================================================================

AGENTX_SYSTEM_PROMPT_TEMPLATE = """You are AgentX, a production-grade Autonomous AI Systems Architect and Control Engine.
You operate on an explicit ReAct (Reasoning + Acting) execution model:
1. OBSERVE: Analyze the user goal or the factual observation returned from a previous tool execution.
2. THINK: Reason step-by-step about what specific information is missing or what computation is required.
3. ACT: Call the most appropriate tool with valid arguments.
4. EVALUATE: Validate tool outputs against the objective before formulating the final answer.

OPERATIONAL GUIDELINES:
- Always prefer calling native tools (math_evaluator, web_scraper, db_query_tool, code_executor, memory_search) for computations, web queries, factual lookups, and code runs instead of guessing.
- For math or equations, always use `math_evaluator`.
- For current web pages or docs, use `web_scraper`.
- For structured data queries, use `db_query_tool`.
- For custom scripts or algorithmic tests, use `code_executor`.
- For company knowledge base or prior documents, use `memory_search`.
- When you have obtained sufficient information to fully answer the request, synthesize a clear, comprehensive final answer.
"""


# ============================================================================
# 5. CORE REACT EXECUTION ENGINE
# ============================================================================

class AgentXEngine:
    """
    Core AgentX Autonomous Engine.
    Executes the ReAct loop: Observe -> Think -> Call Tool -> Evaluate Tool Output -> Respond.
    Zero-framework architecture interfacing directly with REST APIs.
    """

    def __init__(
        self,
        config: Optional[AgentXConfig] = None,
        registry: Optional[ToolRegistry] = None,
        memory: Optional[VectorMemory] = None,
        custom_system_prompt: Optional[str] = None
    ):
        self.config = config or DEFAULT_CONFIG
        self.registry = registry or DEFAULT_REGISTRY
        self.memory = memory or DEFAULT_MEMORY
        self.system_prompt = custom_system_prompt or AGENTX_SYSTEM_PROMPT_TEMPLATE
        self.llm_client = LowLevelLLMClient(self.config)
        self.context_manager = ChatContextManager(self.config)

    def _parse_text_react_fallback(self, content: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """
        Fallback parser for non-function-calling models or completion models that
        output traditional ReAct text patterns:
        Thought: <reasoning>
        Action: <tool_name>
        Action Input: <json or string>
        """
        action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", content, re.IGNORECASE)
        if not action_match:
            return None

        tool_name = action_match.group(1).strip()
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", content, re.IGNORECASE | re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        # Extract Action Input
        input_match = re.search(r"Action Input:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
        raw_args = input_match.group(1).strip() if input_match else "{}"

        # Clean code fences or markdown blocks if present
        raw_args = re.sub(r"^```(?:json)?\s*", "", raw_args)
        raw_args = re.sub(r"\s*```$", "", raw_args)

        parsed_args: Dict[str, Any] = {}
        try:
            parsed_args = json.loads(raw_args)
        except Exception:
            # If not valid JSON, pass raw string under a primary parameter (e.g. 'expression' or 'query' or 'code')
            tool_obj = self.registry.get(tool_name)
            if tool_obj:
                props = tool_obj.schema.get("function", {}).get("parameters", {}).get("properties", {})
                if props:
                    first_prop = list(props.keys())[0]
                    parsed_args = {first_prop: raw_args}
                else:
                    parsed_args = {"input": raw_args}
            else:
                parsed_args = {"input": raw_args}

        return (thought, tool_name, parsed_args)

    def run_stream(self, user_query: str) -> Generator[ReActStep, None, AgentXResponse]:
        """
        Generator execution mode: Yields each ReActStep as it occurs in real time.
        Enables streaming observability for rich web applications and live terminal UIs.
        """
        start_time = time.time()
        self.context_manager.clear()

        # Step 0: Initialize context with System Prompt and User Goal
        self.context_manager.add_message(role="system", content=self.system_prompt)
        self.context_manager.add_message(role="user", content=user_query)

        steps: List[ReActStep] = []
        iteration = 0
        total_tool_calls = 0
        finish_reason = "max_iterations_reached"
        final_answer = ""
        schemas = self.registry.get_schemas()

        while iteration < self.config.max_iterations:
            iteration += 1
            step_start = time.time()
            step = ReActStep(step_number=iteration)

            # Step 1: OBSERVE & THINK -> Dispatch to LLM API
            try:
                raw_response = self.llm_client.call_chat_completion(
                    messages=self.context_manager.get_payload_messages(),
                    tools_schema=schemas if schemas else None
                )
            except Exception as api_err:
                step.thought = f"LLM Dispatch Failure: {str(api_err)}"
                step.duration_seconds = time.time() - step_start
                steps.append(step)
                yield step
                finish_reason = "api_error"
                final_answer = f"Error communicating with LLM endpoint: {str(api_err)}"
                break

            choice = raw_response.get("choices", [{}])[0]
            message_data = choice.get("message", {})
            content = message_data.get("content") or ""
            tool_calls = message_data.get("tool_calls") or []

            # Step 2: EVALUATE MODEL INTENT (Native Tool Call vs. ReAct Text vs. Direct Response)
            if tool_calls:
                # Native Function Calling detected
                first_tool = tool_calls[0]
                call_id = first_tool.get("id", f"call_{iteration}")
                func_data = first_tool.get("function", {})
                tool_name = func_data.get("name", "")
                raw_arguments = func_data.get("arguments", "{}")

                # Parse arguments
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                except Exception:
                    arguments = {"raw": raw_arguments}

                step.thought = content.strip() if content else f"Invoking tool `{tool_name}` to retrieve required data."
                step.tool_name = tool_name
                step.tool_arguments = arguments
                step.tool_raw_arguments = raw_arguments

                # Append assistant tool_call message to context
                self.context_manager.add_message(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls
                )

                # Step 3: CALL TOOL (Execute Tool in Registry)
                tool_exec_output = self.registry.execute(tool_name, arguments)
                step.tool_output = tool_exec_output
                total_tool_calls += 1

                # Step 4: EVALUATE TOOL OUTPUT (Feed result back into context)
                self.context_manager.add_message(
                    role="tool",
                    content=tool_exec_output,
                    tool_call_id=call_id,
                    name=tool_name
                )

                step.duration_seconds = time.time() - step_start
                steps.append(step)
                yield step
                continue

            # Check if model output textual ReAct Action format
            text_react = self._parse_text_react_fallback(content)
            if text_react:
                thought, tool_name, arguments = text_react
                step.thought = thought
                step.tool_name = tool_name
                step.tool_arguments = arguments

                # Record assistant message
                self.context_manager.add_message(role="assistant", content=content)

                # Execute tool
                tool_exec_output = self.registry.execute(tool_name, arguments)
                step.tool_output = tool_exec_output
                total_tool_calls += 1

                # Feed back as user/system observation
                self.context_manager.add_message(
                    role="user",
                    content=f"Observation: {tool_exec_output}"
                )

                step.duration_seconds = time.time() - step_start
                steps.append(step)
                yield step
                continue

            # Step 5: RESPOND (Final Answer produced)
            step.thought = "Final response formulated. Objective satisfied."
            final_answer = content
            self.context_manager.add_message(role="assistant", content=final_answer)
            finish_reason = "completed"

            step.duration_seconds = time.time() - step_start
            steps.append(step)
            yield step
            break

        total_duration = time.time() - start_time
        if finish_reason == "max_iterations_reached" and not final_answer:
            final_answer = (
                f"Agent reached maximum reasoning limit of {self.config.max_iterations} iterations. "
                "Synthesizing best available answer based on gathered tool observations."
            )

        return AgentXResponse(
            final_answer=final_answer,
            trace=steps,
            total_iterations=iteration,
            total_tool_calls=total_tool_calls,
            finish_reason=finish_reason,
            execution_time_seconds=total_duration,
            token_usage_estimate=self.context_manager.total_estimated_tokens()
        )

    def run(self, user_query: str) -> AgentXResponse:
        """
        Synchronous execution entry point.
        Executes the ReAct loop to completion and returns the final AgentXResponse.
        """
        generator = self.run_stream(user_query)
        try:
            while True:
                next(generator)
        except StopIteration as stop:
            return stop.value


# ============================================================================
# 6. ENGINE FACTORY
# ============================================================================

def create_engine(
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any
) -> AgentXEngine:
    """
    Factory to construct a configured AgentXEngine instance.
    """
    cfg_kwargs = {}
    if api_key:
        cfg_kwargs["api_key"] = api_key
    if api_base_url:
        cfg_kwargs["api_base_url"] = api_base_url
    if model:
        cfg_kwargs["model"] = model
    cfg_kwargs.update(kwargs)

    config = AgentXConfig(**cfg_kwargs) if cfg_kwargs else DEFAULT_CONFIG
    return AgentXEngine(config=config)
