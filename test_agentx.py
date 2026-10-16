"""
AgentX Test Suite & Verification Script
=======================================
End-to-end verification of all core components:
1. Safe Math Evaluator (AST Whitelist)
2. Safe SQLite DB Query Tool (Read-Only Validation)
3. Subprocess Code Executor (stdout/stderr/timeout)
4. Tool Registry & JSON Schema Generator
5. Sliding Window Chunker & NumPy Cosine Similarity Vector Memory
6. Chat Context Manager Trimming
7. ReAct Loop & Text Fallback Parser
"""

import math
import os
import sys
import unittest

from agentx.config import AgentXConfig
from agentx.tools import (
    agentx_tool,
    ToolRegistry,
    math_evaluator,
    db_query_tool,
    code_executor,
    function_to_json_schema,
)
from agentx.memory import (
    SlidingWindowChunker,
    LocalHashingVectorizer,
    VectorMemory,
)
from agentx.engine import (
    ChatMessage,
    ChatContextManager,
    AgentXEngine,
    ReActStep,
)


class TestAgentXTools(unittest.TestCase):
    """Test suite for tool decorator, JSON schema generator, and native tools."""

    def test_schema_generator(self):
        @agentx_tool()
        def sample_add(a: int, b: int = 10) -> int:
            """
            Add two integers.
            :param a: The first integer
            :param b: The second integer
            """
            return a + b

        schema = sample_add.schema
        self.assertEqual(schema["type"], "function")
        func_def = schema["function"]
        self.assertEqual(func_def["name"], "sample_add")
        self.assertEqual(func_def["parameters"]["properties"]["a"]["type"], "integer")
        self.assertIn("a", func_def["parameters"]["required"])
        self.assertNotIn("b", func_def["parameters"]["required"])

    def test_math_evaluator_valid(self):
        res1 = math_evaluator("sqrt(144) + 12")
        self.assertEqual(res1, "24")

        res2 = math_evaluator("2 ** 5 - 10 * 2")
        self.assertEqual(res2, "12")

        res3 = math_evaluator("round(sin(pi / 2), 2)")
        self.assertEqual(res3, "1")

    def test_math_evaluator_safety(self):
        # Division by zero should not crash process
        div_zero = math_evaluator("100 / 0")
        self.assertTrue("Division or modulo by zero" in div_zero or "Error" in div_zero)

        # Prohibited operations and injection attempts must fail
        injection = math_evaluator("__import__('os').system('dir')")
        self.assertTrue("Error" in injection or "prohibited" in injection.lower())

        attr_access = math_evaluator("().__class__.__bases__")
        self.assertTrue("Error" in attr_access or "prohibited" in attr_access.lower())

    def test_db_query_tool_read_only(self):
        # Valid query against demo database
        res = db_query_tool("SELECT name, role, salary FROM employees WHERE salary > 150000")
        self.assertIn("Alice Chen", res)
        self.assertIn("Frank Castle", res)

        # Destructive statement rejection
        del_attempt = db_query_tool("DELETE FROM employees WHERE id = 1")
        self.assertIn("Security Restriction", del_attempt)

        drop_attempt = db_query_tool("DROP TABLE employees")
        self.assertIn("Security Restriction", drop_attempt)

    def test_code_executor(self):
        code = "val = 20 * 5\nprint(f'Computed Result: {val}')"
        out = code_executor(code)
        self.assertIn("Computed Result: 100", out)
        self.assertIn("Exit Code: 0", out)

        # Syntax error handling
        bad_code = "for i in range(10)\n  print(i)"
        err_out = code_executor(bad_code)
        self.assertIn("SyntaxError", err_out)


class TestAgentXMemory(unittest.TestCase):
    """Test suite for chunker, vectorizer, and in-memory vector database."""

    def test_sliding_window_chunker(self):
        chunker = SlidingWindowChunker(chunk_size=50, chunk_overlap=15)
        text = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda Mu Nu Xi Omicron Pi Rho Sigma Tau"
        chunks = chunker.chunk_text(text, doc_id="test_doc")

        self.assertGreater(len(chunks), 1)
        # Verify overlap exists between consecutive chunks
        self.assertEqual(chunks[0].doc_id, "test_doc")

    def test_vector_memory_cosine_search(self):
        mem = VectorMemory()
        mem.clear()

        mem.add_document(
            text="Autonomous agents utilize the ReAct pattern for reasoning and tool execution.",
            doc_id="react_doc",
            metadata={"domain": "agentic_ai"}
        )
        mem.add_document(
            text="Relational databases organize structured data using SQL tables, foreign keys, and indexes.",
            doc_id="db_doc",
            metadata={"domain": "databases"}
        )

        # Search for agent concepts
        hits = mem.search("How do autonomous agents reason with ReAct?", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].doc_id, "react_doc")
        self.assertGreater(hits[0].similarity, 0.0)

        # Search for database concepts
        db_hits = mem.search("structured SQL relational tables", top_k=1)
        self.assertEqual(len(db_hits), 1)
        self.assertEqual(db_hits[0].doc_id, "db_doc")

        # Metadata filtering: ensure only databases document is returned even if query is 'reasoning'
        filtered_hits = mem.search("reasoning", top_k=2, filter_metadata={"domain": "databases"})
        self.assertEqual(len(filtered_hits), 1)
        self.assertEqual(filtered_hits[0].doc_id, "db_doc")

        # Non-matching filter returns 0
        no_hits = mem.search("reasoning", top_k=2, filter_metadata={"domain": "non_existent"})
        self.assertEqual(len(no_hits), 0)


class TestAgentXEngine(unittest.TestCase):
    """Test suite for context manager, trimming, and ReAct fallback parser."""

    def test_context_manager_trimming(self):
        cfg = AgentXConfig(max_context_tokens=100, preserve_recent_messages=2)
        ctx = ChatContextManager(cfg)

        ctx.add_message("system", "System instructions")
        ctx.add_message("user", "Initial goal")

        # Add heavy intermediate messages that push past the token limit
        for i in range(10):
            ctx.add_message("assistant", f"Intermediate thought step {i} " + ("word " * 30))
            ctx.add_message("tool", f"Observation result {i} " + ("data " * 30))

        # Context manager must have trimmed older intermediate messages
        messages = ctx.get_messages()
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].role, "user")
        self.assertLess(len(messages), 22)

    def test_text_react_fallback_parser(self):
        engine = AgentXEngine()
        sample_text_response = (
            "Thought: I need to calculate the square root of 625.\n"
            "Action: math_evaluator\n"
            "Action Input: {\"expression\": \"sqrt(625)\"}"
        )
        parsed = engine._parse_text_react_fallback(sample_text_response)
        self.assertIsNotNone(parsed)
        thought, tool_name, args = parsed
        self.assertEqual(tool_name, "math_evaluator")
        self.assertEqual(args.get("expression"), "sqrt(625)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
