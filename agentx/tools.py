"""
AgentX Tool Registry & Schema Builder
=====================================
Zero-framework tool decorator, automatic JSON Schema generator, and production-grade
native tools (Safe Math Evaluator, Web Scraper, Safe SQLite Query, and Subprocess Code Executor).
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# ============================================================================
# 1. TYPE INSPECTOR & JSON SCHEMA GENERATOR
# ============================================================================

def _python_type_to_json_type(py_type: Any) -> Dict[str, Any]:
    """
    Map Python type annotations to OpenAPI / JSON Schema types.
    Handles primitives, Optionals, Lists, Dicts, and Unions.
    """
    # Handle typing constructs (Optional, Union)
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is Union:
        # Check if it's Optional[T] (Union[T, NoneType])
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_json_type(non_none_args[0])
        return {"type": "string", "description": f"Union of {[a.__name__ for a in non_none_args if hasattr(a, '__name__')]}"}

    if py_type in (int,):
        return {"type": "integer"}
    if py_type in (float,):
        return {"type": "number"}
    if py_type in (str,):
        return {"type": "string"}
    if py_type in (bool,):
        return {"type": "boolean"}
    if py_type in (list, list) or origin in (list, List):
        item_type = args[0] if args else str
        return {
            "type": "array",
            "items": _python_type_to_json_type(item_type)
        }
    if py_type in (dict, dict) or origin in (dict, Dict):
        return {"type": "object"}

    # Default fallback
    return {"type": "string"}


def _parse_docstring_param_descriptions(docstring: Optional[str]) -> Dict[str, str]:
    """
    Parse parameter descriptions from Google, Sphinx, or standard style docstrings.
    """
    if not docstring:
        return {}

    param_descriptions: Dict[str, str] = {}
    lines = docstring.split("\n")

    # Pattern for Sphinx: :param name: description
    sphinx_pattern = re.compile(r":param\s+([a-zA-Z0-9_]+):\s*(.+)")
    # Pattern for Google/NumPy style: param_name (type): description or param_name: description
    google_pattern = re.compile(r"^\s*([a-zA-Z0-9_]+)(?:\s*\([^)]*\))?:\s*(.+)")

    in_args_section = False
    for line in lines:
        stripped = line.strip()

        # Check section headers
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args_section = True
            continue
        elif in_args_section and (stripped.lower() in ("returns:", "raises:", "example:", "examples:") or stripped.startswith("---")):
            in_args_section = False

        sphinx_match = sphinx_pattern.match(stripped)
        if sphinx_match:
            param_descriptions[sphinx_match.group(1)] = sphinx_match.group(2).strip()
            continue

        if in_args_section:
            google_match = google_pattern.match(line)
            if google_match:
                param_descriptions[google_match.group(1)] = google_match.group(2).strip()

    return param_descriptions


def function_to_json_schema(func: Callable[..., Any], name_override: Optional[str] = None, desc_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert a Python function signature and docstring into a standard OpenAI-compatible
    Tool JSON Schema format.
    """
    func_name = name_override or func.__name__
    raw_doc = inspect.getdoc(func) or ""

    # Clean function summary description
    first_line = raw_doc.split("\n")[0].strip() if raw_doc else f"Execute {func_name}"
    description = desc_override or first_line

    sig = inspect.signature(func)
    param_docs = _parse_docstring_param_descriptions(raw_doc)

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        # Skip *args or **kwargs
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = param.annotation if param.annotation != inspect.Parameter.empty else str
        schema_prop = _python_type_to_json_type(annotation)

        # Attach parameter description
        prop_desc = param_docs.get(param_name, f"Parameter: {param_name}")
        schema_prop["description"] = prop_desc

        # If parameter has a default value, record it and do not mark as required
        if param.default != inspect.Parameter.empty:
            schema_prop["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = schema_prop

    return {
        "type": "function",
        "function": {
            "name": func_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


# ============================================================================
# 2. TOOL REGISTRY & DECORATOR
# ============================================================================

class AgentXTool:
    """Wrapper encapsulating a registered tool function, its metadata, and schema."""

    def __init__(self, func: Callable[..., Any], name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (inspect.getdoc(func) or "").split("\n")[0].strip()
        self.schema = function_to_json_schema(func, name_override=self.name, desc_override=self.description)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def execute(self, arguments: Union[str, Dict[str, Any]]) -> str:
        """
        Execute tool with either a dictionary of keyword arguments or a raw JSON string.
        Returns serialized string output suitable for feeding back into the ReAct loop.
        """
        parsed_args: Dict[str, Any] = {}
        if isinstance(arguments, str):
            clean_str = arguments.strip()
            if clean_str:
                try:
                    parsed_args = json.loads(clean_str)
                except Exception as json_err:
                    return f"Error: Failed to parse arguments JSON: {json_err}. Input was: {clean_str}"
        elif isinstance(arguments, dict):
            parsed_args = arguments

        try:
            result = self.func(**parsed_args)
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2, default=str)
            return str(result)
        except Exception as exec_err:
            return f"Error executing tool '{self.name}': {type(exec_err).__name__}: {str(exec_err)}"


class ToolRegistry:
    """
    Central repository for managing, discovering, and executing AgentX tools.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, AgentXTool] = {}

    def register(self, func_or_tool: Union[Callable[..., Any], AgentXTool], name: Optional[str] = None, description: Optional[str] = None) -> AgentXTool:
        """Register a function or tool in the registry."""
        if isinstance(func_or_tool, AgentXTool):
            tool = func_or_tool
        else:
            tool = AgentXTool(func_or_tool, name=name, description=description)

        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Optional[AgentXTool]:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self._tools

    def list_tools(self) -> List[str]:
        """Return list of all registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self, filter_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return list of standard tool schemas for LLM payload generation."""
        if filter_names is not None:
            return [self._tools[k].schema for k in filter_names if k in self._tools]
        return [tool.schema for tool in self._tools.values()]

    def execute(self, tool_name: str, arguments: Union[str, Dict[str, Any]]) -> str:
        """Execute a registered tool by name with arguments."""
        tool = self.get(tool_name)
        if not tool:
            available = ", ".join(self.list_tools())
            return f"Error: Tool '{tool_name}' not found. Available tools: [{available}]"
        return tool.execute(arguments)


# Global default registry
DEFAULT_REGISTRY = ToolRegistry()


def agentx_tool(name: Optional[str] = None, description: Optional[str] = None, registry: Optional[ToolRegistry] = None) -> Callable[[Callable[..., Any]], AgentXTool]:
    """
    Decorator to turn any standard Python function into an AgentX Tool
    with automatic JSON Schema generation and registration.

    Example:
        @agentx_tool()
        def multiply(a: float, b: float) -> float:
            '''Multiply two numbers together.'''
            return a * b
    """
    target_registry = registry or DEFAULT_REGISTRY

    def decorator(func: Callable[..., Any]) -> AgentXTool:
        tool = AgentXTool(func, name=name, description=description)
        target_registry.register(tool)
        return tool

    return decorator


# ============================================================================
# 3. BUILT-IN NATIVE TOOLS
# ============================================================================

# --- Tool 1: Safe Math Evaluator (AST Whitelist Parser) ---

class _SafeMathVisitor(ast.NodeVisitor):
    """
    AST visitor that strictly evaluates mathematical expressions without eval().
    Supports basic arithmetic, powers, roots, standard math functions, and constants.
    """
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

    ALLOWED_FUNCTIONS: Dict[str, Callable[..., Any]] = {
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
        "atan2": math.atan2,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "ceil": math.ceil,
        "floor": math.floor,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "degrees": math.degrees,
        "radians": math.radians,
    }

    ALLOWED_CONSTANTS: Dict[str, float] = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
    }

    def visit(self, node: ast.AST) -> Any:
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Unsupported expression construct: {node.__class__.__name__}")

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Literal constants of type {type(node.value).__name__} are not permitted")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_type = type(node.op)
        if op_type not in self.ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} is not allowed")
        left = self.visit(node.left)
        right = self.visit(node.right)
        # Prevent division by zero crash
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Math error: Division or modulo by zero")
        # Prevent runaway exponentiation attacks
        if op_type is ast.Pow and (left > 1000 or right > 100):
            raise OverflowError("Exponent values too large for safe computation")
        return self.ALLOWED_OPERATORS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_type = type(node.op)
        if op_type not in self.ALLOWED_OPERATORS:
            raise ValueError(f"Unary operator {op_type.__name__} is not allowed")
        operand = self.visit(node.operand)
        return self.ALLOWED_OPERATORS[op_type](operand)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.ALLOWED_CONSTANTS:
            return self.ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"Variable or symbol '{node.id}' is undefined or prohibited")

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Method calls and attribute access are strictly prohibited")
        func_name = node.func.id
        if func_name not in self.ALLOWED_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not in the approved mathematical whitelist")
        args = [self.visit(arg) for arg in node.args]
        return self.ALLOWED_FUNCTIONS[func_name](*args)


@agentx_tool(name="math_evaluator", description="Safely evaluate mathematical and scientific expressions without eval(). Supports arithmetic, powers, trigonometry, logarithms, and constants like pi and e.")
def math_evaluator(expression: str) -> str:
    """
    Safely parse and compute mathematical expressions using an AST whitelist.

    :param expression: Mathematical expression string, e.g. 'sqrt(144) + 12 * sin(pi / 2)'
    :return: Formatted numerical result or detailed error message.
    """
    clean_expr = expression.strip()
    if not clean_expr:
        return "Error: Empty expression provided."

    try:
        # Parse expression into AST
        parsed = ast.parse(clean_expr, mode="eval")
        visitor = _SafeMathVisitor()
        result = visitor.visit(parsed)

        # Format clean integer vs float representation
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return f"{result:.8g}" if isinstance(result, float) else str(result)
    except Exception as err:
        return f"Math Evaluation Error: {str(err)}"


# --- Tool 2: Web Scraper (Raw HTTP Fetcher + BeautifulSoup Content Cleaning) ---

@agentx_tool(name="web_scraper", description="Fetch a web page via raw HTTP GET and extract cleaned, readable text content, stripping ads, navigation, and script tags.")
def web_scraper(url: str, max_chars: int = 4000) -> str:
    """
    Fetch and clean webpage content using requests and BeautifulSoup.

    :param url: The target HTTP/HTTPS URL to scrape.
    :param max_chars: Maximum characters of clean text to return (default: 4000).
    :return: Cleaned text extracted from the webpage or error description.
    """
    parsed = urlparse(url.strip())
    if not parsed.scheme or parsed.scheme not in ("http", "https"):
        return f"Error: Invalid URL scheme '{parsed.scheme}'. Only http and https URLs are supported."

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 AgentX/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        response = requests.get(url.strip(), headers=headers, timeout=12, allow_redirects=True)
        response.raise_for_status()

        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract page title
        title = soup.title.string.strip() if soup.title and soup.title.string else "No Title Found"

        # Remove irrelevant and noise tags
        for element in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "svg", "button", "iframe"]):
            element.decompose()

        # Extract textual content
        text = soup.get_text(separator="\n")

        # Clean excess blank lines and whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if not clean_text:
            return f"Page '{title}' was fetched successfully, but contained no readable text body."

        # Truncate to limit
        total_len = len(clean_text)
        truncated_note = ""
        if total_len > max_chars:
            clean_text = clean_text[:max_chars]
            truncated_note = f"\n\n[Content truncated. Displayed {max_chars} of {total_len} characters. Full URL: {url}]"

        return f"=== PAGE TITLE: {title} ===\n\n{clean_text}{truncated_note}"

    except requests.exceptions.Timeout:
        return f"Error: Request timed out while attempting to connect to {url}."
    except requests.exceptions.HTTPError as http_err:
        return f"Error: HTTP {http_err.response.status_code} returned by {url}."
    except Exception as err:
        return f"Error scraping URL '{url}': {type(err).__name__}: {str(err)}"


# --- Tool 3: Safe SQLite DB Query Tool ---

def _init_sandbox_db(db_path: str) -> None:
    """Initialize demo tables in sandbox database if empty or newly created."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        role TEXT NOT NULL,
        salary REAL NOT NULL,
        join_date TEXT NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        stock INTEGER NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        units INTEGER NOT NULL,
        total_amount REAL NOT NULL,
        sale_date TEXT NOT NULL,
        FOREIGN KEY(product_id) REFERENCES products(id)
    );
    """)

    # Seed demo data if employees table is empty
    cursor.execute("SELECT COUNT(*) FROM employees;")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
        INSERT INTO employees (name, department, role, salary, join_date) VALUES (?, ?, ?, ?, ?)
        """, [
            ("Alice Chen", "Engineering", "Principal Architect", 175000.0, "2021-03-15"),
            ("Bob Martinez", "Engineering", "Senior ML Engineer", 145000.0, "2022-06-01"),
            ("Carol Danvers", "Product", "VP of Product", 190000.0, "2020-01-10"),
            ("David Kim", "Sales", "Enterprise Account Exec", 120000.0, "2023-02-20"),
            ("Emma Watson", "Engineering", "Frontend Lead", 138000.0, "2022-09-15"),
            ("Frank Castle", "Security", "Head of SecOps", 160000.0, "2021-11-04")
        ])

        cursor.executemany("""
        INSERT INTO products (sku, name, category, price, stock) VALUES (?, ?, ?, ?, ?)
        """, [
            ("AI-001", "AgentX Enterprise License", "Software", 4999.0, 100),
            ("AI-002", "Neural Compute Appliance", "Hardware", 12500.0, 24),
            ("AI-003", "RAG Vector Accelerator Card", "Hardware", 2800.0, 50),
            ("SW-101", "Edge Inference Runtime", "Software", 899.0, 250),
            ("CS-500", "Custom Agent Fine-tuning Pack", "Services", 7500.0, 15)
        ])

        cursor.executemany("""
        INSERT INTO sales (product_id, units, total_amount, sale_date) VALUES (?, ?, ?, ?)
        """, [
            (1, 2, 9998.0, "2026-10-01"),
            (2, 1, 12500.0, "2026-10-03"),
            (3, 4, 11200.0, "2026-10-05"),
            (4, 10, 8990.0, "2026-10-08"),
            (5, 1, 7500.0, "2026-10-12")
        ])

    conn.commit()
    conn.close()


@agentx_tool(name="db_query_tool", description="Safely execute read-only SQLite SQL queries against structured databases. Inspect tables and retrieve rows cleanly formatted.")
def db_query_tool(query: str, db_path: Optional[str] = None) -> str:
    """
    Safely execute read-only SQL queries on SQLite database.

    :param query: SQL query string to run, e.g. 'SELECT name, department, salary FROM employees WHERE salary > 130000'
    :param db_path: Optional custom path to SQLite database. Defaults to 'agentx_sandbox.db'.
    :return: Formatted query results table or error message.
    """
    clean_query = query.strip()
    if not clean_query:
        return "Error: Empty query provided."

    target_db = db_path or os.getenv("AGENTX_SQLITE_DB", "agentx_sandbox.db")
    # Ensure database exists with demo schema if using default sandbox
    if target_db == "agentx_sandbox.db" and not os.path.exists(target_db):
        _init_sandbox_db(target_db)
    elif not os.path.exists(target_db):
        _init_sandbox_db(target_db)

    # Security check: Disallow mutating or dangerous statements
    first_token = clean_query.split()[0].upper() if clean_query.split() else ""
    allowed_verbs = {"SELECT", "WITH", "PRAGMA", "EXPLAIN"}

    if first_token not in allowed_verbs:
        return f"Security Restriction: Only read-only operations ({', '.join(allowed_verbs)}) are allowed. Received statement starting with: '{first_token}'"

    # Block destructive multi-statement injections
    prohibited_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "ATTACH", "DETACH"]
    query_tokens = [t.upper() for t in re.split(r"[\s;,]+", clean_query)]
    for kw in prohibited_keywords:
        if kw in query_tokens:
            return f"Security Restriction: Destructive statement '{kw}' detected in SQL text. Execution aborted."

    try:
        # Open in URI read-only mode if supported
        uri_path = f"file:{os.path.abspath(target_db)}?mode=ro"
        try:
            conn = sqlite3.connect(uri_path, uri=True, timeout=5)
        except Exception:
            conn = sqlite3.connect(target_db, timeout=5)

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(clean_query)

        rows = cursor.fetchmany(100)
        if not rows:
            conn.close()
            return "Query executed successfully. Result set is empty (0 rows)."

        headers = [col[0] for col in cursor.description]
        records = [dict(row) for row in rows]
        conn.close()

        # Build clean ASCII / Markdown table output
        col_widths = {h: max(len(h), max(len(str(r[h])) for r in records)) for h in headers}
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        divider_line = "-+-".join("-" * col_widths[h] for h in headers)
        data_lines = [
            " | ".join(str(r[h]).ljust(col_widths[h]) for h in headers)
            for r in records
        ]

        total_rows = len(records)
        summary = f"\n({total_rows} row{'s' if total_rows != 1 else ''} returned)"
        return f"{header_line}\n{divider_line}\n" + "\n".join(data_lines) + summary

    except sqlite3.Error as sql_err:
        return f"SQLite Database Error: {str(sql_err)}"
    except Exception as err:
        return f"Error executing database query: {type(err).__name__}: {str(err)}"


# --- Tool 4: Isolated Subprocess Python Code Executor ---

@agentx_tool(name="code_executor", description="Execute arbitrary Python code in an isolated subprocess. Captures stdout, stderr, and returns execution metrics.")
def code_executor(code: str, timeout: int = 10) -> str:
    """
    Run Python code in a sandboxed subprocess and return stdout and stderr.

    :param code: Valid Python source code to execute.
    :param timeout: Execution timeout in seconds (default: 10s).
    :return: Formatted execution output containing stdout, stderr, and exit status.
    """
    clean_code = code.strip()
    if not clean_code:
        return "Error: No Python code provided for execution."

    # Write code to a temporary file
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
            temp_file = tf.name
            tf.write(clean_code)

        start_time = time.time()
        # Execute using current python interpreter in isolated environment
        proc = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(temp_file))
        )
        elapsed = time.time() - start_time

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        output_parts: List[str] = [
            f"=== Execution Completed (Exit Code: {proc.returncode}, Duration: {elapsed:.2f}s) ==="
        ]
        if stdout:
            output_parts.append(f"[STDOUT]\n{stdout}")
        else:
            output_parts.append("[STDOUT]\n(No standard output)")

        if stderr:
            output_parts.append(f"[STDERR]\n{stderr}")

        return "\n\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return f"Execution Error: Code run exceeded timeout limit of {timeout} seconds. Process was terminated."
    except Exception as err:
        return f"Execution Error: Failed to execute subprocess: {type(err).__name__}: {str(err)}"
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass


# ============================================================================
# 4. MODULE INITIALIZATION
# ============================================================================

def get_default_registry() -> ToolRegistry:
    """Retrieve global default ToolRegistry pre-loaded with native tools."""
    return DEFAULT_REGISTRY
