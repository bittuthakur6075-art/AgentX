"""
agentx_engine.py - AgentX ReAct Execution Engine
================================================
Zero-framework autonomous execution engine.
Implements the ReAct cycle: Observe -> Think -> Act (Call Tool) -> Evaluate -> Respond.
Uses direct low-level HTTP REST requests without high-level agent libraries.
"""

from __future__ import annotations
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import requests

from agentx_tools import ToolRegistry, DEFAULT_TOOL_REGISTRY
from agentx_memory import VectorStore, DEFAULT_VECTOR_STORE

logger = logging.getLogger("AgentXEngine")


# ============================================================================
# 1. CHAT MESSAGE & CONTEXT STATE MANAGER
# ============================================================================

@dataclass
class ChatMessage:
    """Represents a standard chat message across LLM REST endpoints."""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        elif self.role == "assistant" and self.tool_calls:
            d["content"] = ""
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d

    def estimate_tokens(self, chars_per_token: float = 4.0) -> int:
        total = len(self.role) + 4
        if self.content:
            total += len(self.content)
        if self.name:
            total += len(self.name)
        if self.tool_calls:
            total += len(json.dumps(self.tool_calls))
        return max(1, int(total / chars_per_token))


class ChatContextManager:
    """
    Explicit chat context manager with context window trimming.
    Guarantees that system prompt and initial user goals are never discarded
    while dynamically pruning intermediate tool observations when token capacity is reached.
    """

    def __init__(self, max_tokens: int = 8000, preserve_recent: int = 4):
        self.max_tokens = max_tokens
        self.preserve_recent = preserve_recent
        self._messages: List[ChatMessage] = []

    def add_message(
        self,
        role: str,
        content: Optional[str] = None,
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> ChatMessage:
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
        return list(self._messages)

    def get_payload_messages(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._messages]

    def total_estimated_tokens(self) -> int:
        return sum(m.estimate_tokens() for m in self._messages)

    def trim_context(self) -> int:
        """Prune older intermediate interactions when exceeding token limit."""
        if self.total_estimated_tokens() <= self.max_tokens or len(self._messages) <= 4:
            return 0

        trimmed = 0
        # Preserve system (index 0), initial prompt (index 1), and recent tail messages
        while self.total_estimated_tokens() > self.max_tokens and len(self._messages) > (2 + self.preserve_recent):
            target_idx = 2
            removed = self._messages.pop(target_idx)
            trimmed += 1
            # If removing assistant tool_call, also remove corresponding tool result
            if removed.tool_calls and target_idx < len(self._messages) and self._messages[target_idx].role == "tool":
                self._messages.pop(target_idx)
                trimmed += 1

        if trimmed > 0:
            notice = ChatMessage(
                role="system",
                content=f"[Context Manager: {trimmed} older intermediate reasoning steps trimmed to preserve token context.]"
            )
            self._messages.insert(2, notice)

        return trimmed

    def clear(self) -> None:
        self._messages.clear()


# ============================================================================
# 2. STEP TRACE & EXECUTION STATE
# ============================================================================

@dataclass
class ReActStep:
    """Detailed record of a single ReAct step."""
    step_number: int
    observation: Optional[str] = None
    thought: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "tool_output": self.tool_output,
            "duration_seconds": round(self.duration_seconds, 3)
        }


@dataclass
class AgentXResponse:
    """Complete response package from the AgentX ReAct Engine."""
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
# 3. LOW-LEVEL HTTP REST PAYLOAD DISPATCHER
# ============================================================================

class LowLevelLLMClient:
    """Direct HTTP REST client for /v1/chat/completions using pure requests."""

    def __init__(self, api_key: str, api_base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini", temperature: float = 0.2, max_tokens: int = 2048, timeout: int = 45):
        self.api_key = api_key
        self.api_base_url = api_base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def call_chat_completion(self, messages: List[Dict[str, Any]], tools_schema: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        endpoint = f"{self.api_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AgentX-ReAct-Engine/1.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools_schema:
            payload["tools"] = tools_schema
            payload["tool_choice"] = "auto"

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                err_text = resp.text
                try:
                    err_json = resp.json()
                    err_text = err_json.get("error", {}).get("message", err_text)
                except Exception:
                    pass
                raise RuntimeError(f"LLM API Error (HTTP {resp.status_code}): {err_text}")
            return resp.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"REST call to LLM '{endpoint}' timed out after {self.timeout}s.")
        except requests.exceptions.ConnectionError as c_err:
            raise ConnectionError(f"Failed to connect to LLM endpoint '{endpoint}': {c_err}")


# ============================================================================
# 4. AGENTX REACT EXECUTION ENGINE
# ============================================================================

SYSTEM_PROMPT = """You are AgentX, an autonomous AI system architect and control engine.
You operate on an explicit ReAct (Reasoning + Acting) loop:
1. OBSERVE: Review the user objective or the latest tool output.
2. THINK: Reason step-by-step on what exact calculation, query, or data lookup is needed next.
3. ACT: Call the most suitable registered tool with valid arguments.
4. EVALUATE: Validate the tool output against the original directive.
5. RESPOND: Synthesize a clear, comprehensive final answer once all required facts are gathered.

Guidelines:
- Never guess numerical, mathematical, or scientific calculations. Always invoke `math_evaluator`.
- For structured data lookups, inspect tables via `db_query_tool`.
- For current external webpages, fetch via `web_scraper`.
- For Python algorithms, execute via `code_executor`.
- For indexed company documentation, query via `memory_search`.
"""


class AgentXEngine:
    """
    Custom AgentX ReAct Execution Engine.
    Executes the loop: Observe -> Think -> Call Tool -> Evaluate Tool Output -> Respond.
    Zero-framework architecture interfacing directly with LLM REST APIs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_iterations: int = 10,
        registry: Optional[ToolRegistry] = None,
        memory: Optional[VectorStore] = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AGENTX_API_KEY") or ""
        self.api_base_url = api_base_url
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations

        self.registry = registry or DEFAULT_TOOL_REGISTRY
        self.memory = memory or DEFAULT_VECTOR_STORE
        self.llm_client = LowLevelLLMClient(
            api_key=self.api_key,
            api_base_url=self.api_base_url,
            model=self.model,
            temperature=self.temperature
        )
        self.context = ChatContextManager()

    def _parse_text_react_fallback(self, content: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Fallback parser for models returning standard ReAct text format."""
        act_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", content, re.IGNORECASE)
        if not act_match:
            return None

        tool_name = act_match.group(1).strip()
        th_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", content, re.IGNORECASE | re.DOTALL)
        thought = th_match.group(1).strip() if th_match else ""

        inp_match = re.search(r"Action Input:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
        raw_input = inp_match.group(1).strip() if inp_match else "{}"
        raw_input = re.sub(r"^```(?:json)?\s*", "", raw_input)
        raw_input = re.sub(r"\s*```$", "", raw_input)

        try:
            parsed_args = json.loads(raw_input)
        except Exception:
            tool_obj = self.registry.get(tool_name)
            if tool_obj:
                props = tool_obj.schema.get("function", {}).get("parameters", {}).get("properties", {})
                first_key = list(props.keys())[0] if props else "input"
                parsed_args = {first_key: raw_input}
            else:
                parsed_args = {"input": raw_input}

        return (thought, tool_name, parsed_args)

    def run_stream(self, user_query: str) -> Generator[ReActStep, None, AgentXResponse]:
        """Generator running the ReAct loop, yielding each ReActStep in real time."""
        t0 = time.time()
        self.context.clear()
        self.context.add_message("system", SYSTEM_PROMPT)
        self.context.add_message("user", user_query)

        steps: List[ReActStep] = []
        iteration = 0
        tool_call_count = 0
        finish_reason = "max_iterations_reached"
        final_answer = ""
        schemas = self.registry.get_schemas()

        while iteration < self.max_iterations:
            iteration += 1
            step_t0 = time.time()
            step = ReActStep(step_number=iteration)

            # 1. OBSERVE & THINK -> Dispatch HTTP REST Call
            try:
                resp_json = self.llm_client.call_chat_completion(
                    messages=self.context.get_payload_messages(),
                    tools_schema=schemas if schemas else None
                )
            except Exception as api_err:
                step.thought = f"LLM Dispatch Error: {api_err}"
                step.duration_seconds = time.time() - step_t0
                steps.append(step)
                yield step
                finish_reason = "api_error"
                final_answer = f"Error communicating with LLM endpoint: {api_err}"
                break

            msg_data = resp_json.get("choices", [{}])[0].get("message", {})
            content = msg_data.get("content") or ""
            tool_calls = msg_data.get("tool_calls") or []

            # 2. ACT (Tool Call Detected)
            if tool_calls:
                call_info = tool_calls[0]
                call_id = call_info.get("id", f"call_{iteration}")
                f_data = call_info.get("function", {})
                tool_name = f_data.get("name", "")
                raw_args = f_data.get("arguments", "{}")

                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {"raw": raw_args}

                step.thought = content.strip() if content else f"Invoking tool `{tool_name}` to retrieve factual data."
                step.tool_name = tool_name
                step.tool_arguments = args

                self.context.add_message("assistant", content=content, tool_calls=tool_calls)

                # 3. EVALUATE TOOL OUTPUT
                tool_out = self.registry.execute(tool_name, args)
                step.tool_output = tool_out
                tool_call_count += 1

                self.context.add_message("tool", content=tool_out, tool_call_id=call_id, name=tool_name)
                step.duration_seconds = time.time() - step_t0
                steps.append(step)
                yield step
                continue

            # Fallback for text ReAct
            text_react = self._parse_text_react_fallback(content)
            if text_react:
                thought, tool_name, args = text_react
                step.thought = thought
                step.tool_name = tool_name
                step.tool_arguments = args

                self.context.add_message("assistant", content=content)
                tool_out = self.registry.execute(tool_name, args)
                step.tool_output = tool_out
                tool_call_count += 1

                self.context.add_message("user", content=f"Observation: {tool_out}")
                step.duration_seconds = time.time() - step_t0
                steps.append(step)
                yield step
                continue

            # 4. RESPOND (Final Answer)
            step.thought = "Objective satisfied. Synthesizing final response."
            final_answer = content
            self.context.add_message("assistant", content=final_answer)
            finish_reason = "completed"

            step.duration_seconds = time.time() - step_t0
            steps.append(step)
            yield step
            break

        elapsed = time.time() - t0
        if finish_reason == "max_iterations_reached" and not final_answer:
            final_answer = f"Maximum reasoning limit of {self.max_iterations} cycles reached. Unable to finalize goal."

        return AgentXResponse(
            final_answer=final_answer,
            trace=steps,
            total_iterations=iteration,
            total_tool_calls=tool_call_count,
            finish_reason=finish_reason,
            execution_time_seconds=elapsed,
            token_usage_estimate=self.context.total_estimated_tokens()
        )

    def run(self, user_query: str) -> AgentXResponse:
        """Synchronously execute ReAct loop to completion."""
        gen = self.run_stream(user_query)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            return stop.value
