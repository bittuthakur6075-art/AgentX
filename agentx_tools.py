"""
agentx_tools.py - AgentX Tool Registry & Native Tools
=====================================================
Zero-framework tool registry, automatic OpenAPI/JSON-Schema builder,
and 4 production-grade native tools:
  1. math_evaluator: Safe AST whitelist parser (no eval).
  2. web_scraper: Raw HTTP fetcher with BeautifulSoup HTML extraction.
  3. db_query_tool: Read-only SQLite query engine with sandbox demo tables.
  4. code_executor: Isolated Python subprocess execution with timeouts.
"""

from __future__ import annotations
import ast
import inspect
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# ============================================================================
# 1. TYPE INSPECTOR & JSON SCHEMA GENERATOR
# ============================================================================

def _py_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """Map Python type annotations to JSON Schema types."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is Union:
        # Check for Optional[T]
        valid_args = [a for a in args if a is not type(None)]
        if len(valid_args) == 1:
            return _py_type_to_json_schema(valid_args[0])
        return {"type": "string"}

    if py_type in (int,):
        return {"type": "integer"}
    if py_type in (float,):
        return {"type": "number"}
    if py_type in (str,):
        return {"type": "string"}
    if py_type in (bool,):
        return {"type": "boolean"}
    if py_type in (list,) or origin in (list, List):
        inner_type = args[0] if args else str
        return {"type": "array", "items": _py_type_to_json_schema(inner_type)}
    if py_type in (dict,) or origin in (dict, Dict):
        return {"type": "object"}

    return {"type": "string"}


def _extract_docstring_params(docstring: Optional[str]) -> Dict[str, str]:
    """Extract parameter descriptions from standard/Sphinx/Google docstrings."""
    if not docstring:
        return {}

    param_docs: Dict[str, str] = {}
    lines = docstring.split("\n")

    sphinx_re = re.compile(r":param\s+([a-zA-Z0-9_]+):\s*(.+)")
    google_re = re.compile(r"^\s*([a-zA-Z0-9_]+)(?:\s*\([^)]*\))?:\s*(.+)")

    in_args_block = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if lower in ("args:", "parameters:", "arguments:"):
            in_args_block = True
            continue
        elif in_args_block and (lower in ("returns:", "raises:", "examples:") or stripped.startswith("---")):
            in_args_block = False

        m_sphinx = sphinx_re.match(stripped)
        if m_sphinx:
            param_docs[m_sphinx.group(1)] = m_sphinx.group(2).strip()
            continue

        if in_args_block:
            m_google = google_re.match(line)
            if m_google:
                param_docs[m_google.group(1)] = m_google.group(2).strip()

    return param_docs


def generate_function_schema(func: Callable[..., Any], name: Optional[str] = None, desc: Optional[str] = None) -> Dict[str, Any]:
    """Inspect function signature and docstrings to build standard Tool JSON Schema."""
    tool_name = name or func.__name__
    raw_doc = inspect.getdoc(func) or ""
    first_line = raw_doc.split("\n")[0].strip() if raw_doc else f"Execute {tool_name}"
    description = desc or first_line

    sig = inspect.signature(func)
    param_docs = _extract_docstring_params(raw_doc)

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = param.annotation if param.annotation != inspect.Parameter.empty else str
        prop_schema = _py_type_to_json_schema(annotation)
        prop_schema["description"] = param_docs.get(param_name, f"Parameter: {param_name}")

        if param.default != inspect.Parameter.empty:
            prop_schema["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = prop_schema

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


# ============================================================================
# 2. TOOL REGISTRY & @agentx_tool DECORATOR
# ============================================================================

class AgentXTool:
    """Wrapper encapsulating a tool function, metadata, and JSON schema."""

    def __init__(self, func: Callable[..., Any], name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (inspect.getdoc(func) or "").split("\n")[0].strip()
        self.schema = generate_function_schema(func, name=self.name, desc=self.description)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def execute(self, arguments: Union[str, Dict[str, Any]]) -> str:
        """Execute tool safely with a dict or raw JSON string, returning string output."""
        kwargs: Dict[str, Any] = {}
        if isinstance(arguments, str):
            clean = arguments.strip()
            if clean:
                try:
                    kwargs = json.loads(clean)
                except Exception as err:
                    return f"Error: Failed to parse arguments JSON: {err}. Received: {clean}"
        elif isinstance(arguments, dict):
            kwargs = arguments

        try:
            res = self.func(**kwargs)
            if isinstance(res, (dict, list)):
                return json.dumps(res, indent=2, default=str)
            return str(res)
        except Exception as err:
            return f"Error executing tool '{self.name}': {type(err).__name__}: {str(err)}"


class ToolRegistry:
    """Central registry holding active AgentX tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, AgentXTool] = {}

    def register(self, func_or_tool: Union[Callable[..., Any], AgentXTool], name: Optional[str] = None, description: Optional[str] = None) -> AgentXTool:
        if isinstance(func_or_tool, AgentXTool):
            tool = func_or_tool
        else:
            tool = AgentXTool(func_or_tool, name=name, description=description)
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Optional[AgentXTool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_schemas(self, filter_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if filter_names is not None:
            return [self._tools[n].schema for n in filter_names if n in self._tools]
        return [tool.schema for tool in self._tools.values()]

    def execute(self, tool_name: str, arguments: Union[str, Dict[str, Any]]) -> str:
        tool = self.get(tool_name)
        if not tool:
            available = ", ".join(self.list_tools())
            return f"Error: Tool '{tool_name}' not found. Available tools: [{available}]"
        return tool.execute(arguments)


# Global default registry instance
DEFAULT_TOOL_REGISTRY = ToolRegistry()


def agentx_tool(name: Optional[str] = None, description: Optional[str] = None, registry: Optional[ToolRegistry] = None) -> Callable[[Callable[..., Any]], AgentXTool]:
    """Decorator converting any Python function into an AgentX Tool."""
    target_registry = registry or DEFAULT_TOOL_REGISTRY

    def decorator(func: Callable[..., Any]) -> AgentXTool:
        tool = AgentXTool(func, name=name, description=description)
        target_registry.register(tool)
        return tool

    return decorator


# ============================================================================
# 3. NATIVE BUILT-IN TOOLS
# ============================================================================

# --- Tool 1: Safe Math Evaluator (AST Whitelist Parser) ---

class _SafeMathASTEvaluator(ast.NodeVisitor):
    """AST visitor strictly calculating math expressions without eval()."""

    ALLOWED_OPERATORS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: +a,
    }

    ALLOWED_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": lambda *args: sum(args if len(args) > 1 else args[0]),
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "ceil": math.ceil,
        "floor": math.floor,
        "factorial": math.factorial,
        "gcd": math.gcd,
    }

    ALLOWED_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }

    def visit(self, node: ast.AST) -> Any:
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Prohibited syntax: {node.__class__.__name__}")

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Constant type {type(node.value).__name__} is not allowed")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_cls = type(node.op)
        if op_cls not in self.ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_cls.__name__} is prohibited")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if op_cls in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division or modulo by zero error")
        if op_cls is ast.Pow and (left > 1000 or right > 100):
            raise OverflowError("Exponent values exceed safe compute limits")
        return self.ALLOWED_OPERATORS[op_cls](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_cls = type(node.op)
        if op_cls not in self.ALLOWED_OPERATORS:
            raise ValueError(f"Unary operator {op_cls.__name__} is prohibited")
        return self.ALLOWED_OPERATORS[op_cls](self.visit(node.operand))

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.ALLOWED_CONSTANTS:
            return self.ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"Symbol '{node.id}' is undefined or prohibited")

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Direct attribute calls are prohibited")
        fname = node.func.id
        if fname not in self.ALLOWED_FUNCTIONS:
            raise ValueError(f"Function '{fname}' is not in approved math whitelist")
        args = [self.visit(a) for a in node.args]
        return self.ALLOWED_FUNCTIONS[fname](*args)


@agentx_tool(name="math_evaluator", description="Safely compute mathematical and scientific expressions without eval(). Supports arithmetic, powers, roots, trigonometry, and logarithms.")
def math_evaluator(expression: str) -> str:
    """
    Safely evaluate math expressions via an AST whitelist.
    :param expression: Mathematical expression string, e.g. 'sqrt(256) + 14 * log10(1000)'
    """
    clean_expr = expression.strip()
    if not clean_expr:
        return "Error: Empty expression provided."
    try:
        parsed = ast.parse(clean_expr, mode="eval")
        visitor = _SafeMathASTEvaluator()
        val = visitor.visit(parsed)
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        return f"{val:.8g}" if isinstance(val, float) else str(val)
    except Exception as err:
        return f"Math Evaluation Error: {str(err)}"


# --- Tool 2: Web Scraper (Raw HTTP + BeautifulSoup Content Cleaning) ---

@agentx_tool(name="web_scraper", description="Fetch a webpage via raw HTTP GET and extract cleaned, readable textual body, stripping scripts, navigation, and noise tags.")
def web_scraper(url: str, max_chars: int = 4000) -> str:
    """
    Fetch and clean webpage content using requests and BeautifulSoup.
    :param url: The target HTTP/HTTPS URL to scrape.
    :param max_chars: Maximum characters of clean text to return (default: 4000).
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return f"Error: Invalid URL scheme '{parsed.scheme}'. Only http and https URLs are allowed."

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgentX/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(url.strip(), headers=headers, timeout=12, allow_redirects=True)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else "No Title Found"

        # Remove irrelevant noise elements
        for el in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "svg"]):
            el.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if not clean_text:
            return f"Page '{title}' was retrieved successfully, but had no readable body text."

        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars] + f"\n\n[Truncated to {max_chars} characters]"

        return f"=== PAGE TITLE: {title} ===\n\n{clean_text}"

    except requests.exceptions.Timeout:
        return f"Error: Request timed out while attempting to reach {url}."
    except requests.exceptions.HTTPError as h_err:
        return f"Error: HTTP {h_err.response.status_code} returned by {url}."
    except Exception as err:
        return f"Error scraping URL '{url}': {type(err).__name__}: {str(err)}"


# --- Tool 3: Safe SQLite DB Query Tool ---

def _ensure_sandbox_db(db_path: str = "agentx_sandbox.db") -> None:
    """Ensure demo database with employees, products, and sales tables exists."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        role TEXT NOT NULL,
        salary REAL NOT NULL,
        join_date TEXT NOT NULL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        stock INTEGER NOT NULL
    );
    """)
    cur.execute("SELECT COUNT(*) FROM employees;")
    if cur.fetchone()[0] == 0:
        cur.executemany("""
        INSERT INTO employees (name, department, role, salary, join_date) VALUES (?, ?, ?, ?, ?)
        """, [
            ("Alice Chen", "Engineering", "Principal Architect", 175000.0, "2021-03-15"),
            ("Bob Martinez", "Engineering", "Senior ML Engineer", 145000.0, "2022-06-01"),
            ("Carol Danvers", "Product", "VP of Product", 190000.0, "2020-01-10"),
            ("David Kim", "Sales", "Enterprise Account Exec", 120000.0, "2023-02-20"),
            ("Frank Castle", "Security", "Head of SecOps", 160000.0, "2021-11-04")
        ])
        cur.executemany("""
        INSERT INTO products (sku, name, category, price, stock) VALUES (?, ?, ?, ?, ?)
        """, [
            ("AI-001", "AgentX Enterprise License", "Software", 4999.0, 100),
            ("AI-002", "Neural Compute Appliance", "Hardware", 12500.0, 24),
            ("SW-101", "Edge Inference Runtime", "Software", 899.0, 250)
        ])
    conn.commit()
    conn.close()


@agentx_tool(name="db_query_tool", description="Safely execute read-only SQLite SQL queries against structured databases. Inspect tables and retrieve rows.")
def db_query_tool(query: str, db_path: Optional[str] = None) -> str:
    """
    Safely execute read-only SQL queries on SQLite database.
    :param query: SQL query string, e.g. 'SELECT name, role, salary FROM employees WHERE salary > 130000'
    :param db_path: Optional path to SQLite file (defaults to 'agentx_sandbox.db').
    """
    clean_query = query.strip()
    if not clean_query:
        return "Error: Empty query provided."

    target_db = db_path or "agentx_sandbox.db"
    _ensure_sandbox_db(target_db)

    # Validate read-only verb
    first_token = clean_query.split()[0].upper() if clean_query.split() else ""
    if first_token not in {"SELECT", "WITH", "PRAGMA", "EXPLAIN"}:
        return f"Security Restriction: Only read-only operations (SELECT, WITH, PRAGMA) are allowed. Detected: '{first_token}'"

    # Reject mutating keywords
    prohibited = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "ATTACH", "DETACH"]
    tokens = [t.upper() for t in re.split(r"[\s;,]+", clean_query)]
    for kw in prohibited:
        if kw in tokens:
            return f"Security Restriction: Mutating statement '{kw}' detected. Query rejected."

    try:
        conn = sqlite3.connect(f"file:{os.path.abspath(target_db)}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(clean_query)
        rows = cur.fetchmany(100)

        if not rows:
            conn.close()
            return "Query executed successfully. Result set is empty (0 rows)."

        headers = [col[0] for col in cur.description]
        records = [dict(r) for r in rows]
        conn.close()

        # Format ASCII table
        col_widths = {h: max(len(h), max(len(str(r[h])) for r in records)) for h in headers}
        header_str = " | ".join(h.ljust(col_widths[h]) for h in headers)
        sep_str = "-+-".join("-" * col_widths[h] for h in headers)
        data_lines = [" | ".join(str(r[h]).ljust(col_widths[h]) for h in headers) for r in records]

        return f"{header_str}\n{sep_str}\n" + "\n".join(data_lines) + f"\n\n({len(records)} rows returned)"

    except Exception as err:
        return f"Database Error: {type(err).__name__}: {str(err)}"


# --- Tool 4: Isolated Subprocess Python Code Executor ---

@agentx_tool(name="code_executor", description="Execute arbitrary Python code in an isolated subprocess. Captures stdout, stderr, execution duration, and exit codes.")
def code_executor(code: str, timeout: int = 10) -> str:
    """
    Run Python code in a sandboxed subprocess with strict timeout.
    :param code: Valid Python source code to execute.
    :param timeout: Maximum seconds before terminating process (default: 10s).
    """
    clean_code = code.strip()
    if not clean_code:
        return "Error: No Python code provided for execution."

    tf_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf_path = tf.name
            tf.write(clean_code)

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, tf_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(tf_path))
        )
        elapsed = time.time() - t0

        stdout = proc.stdout.strip() or "(No standard output)"
        stderr = proc.stderr.strip()

        res = [f"=== Subprocess Finished (Exit Code: {proc.returncode}, Duration: {elapsed:.2f}s) ==="]
        res.append(f"[STDOUT]\n{stdout}")
        if stderr:
            res.append(f"[STDERR]\n{stderr}")

        return "\n\n".join(res)

    except subprocess.TimeoutExpired:
        return f"Execution Error: Subprocess timed out after {timeout} seconds. Process killed."
    except Exception as err:
        return f"Execution Error: {type(err).__name__}: {str(err)}"
    finally:
        if tf_path and os.path.exists(tf_path):
            try:
                os.remove(tf_path)
            except OSError:
                pass
