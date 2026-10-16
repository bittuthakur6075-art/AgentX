"""
app.py - AgentX Autonomous Platform Web Interface
==================================================
Developer-first Streamlit Dashboard with Dark-Mode Slate/Cyberpunk Indigo palette:
  - Background: #0F172A
  - Surfaces & Cards: #1E293B with #334155 glass borders
  - Accents: #6366F1 (Indigo Glow) & #10B981 (Emerald Badges)
Features:
  - 2-Column responsive layout (25% Left Control Panel, 75% Main Workbench).
  - Dynamic tool routing toggles (Calculator, Scraper, SQL, Python Subprocess, Memory RAG).
  - Document ingestion for in-memory Vector DB (NumPy cosine similarity).
  - Real-time reasoning accordions: [Reasoning...], [Tool Called], [Result Received].
  - Clean audit timeline and markdown response synthesis.

Run via:
    streamlit run app.py
"""

from __future__ import annotations
import os
import sys
import time
from typing import Optional

import streamlit as st

# Ensure current directory is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentx_tools import (
    ToolRegistry,
    DEFAULT_TOOL_REGISTRY,
    math_evaluator,
    web_scraper,
    db_query_tool,
    code_executor,
)
from agentx_memory import (
    VectorStore,
    DEFAULT_VECTOR_STORE,
    memory_search,
)
from agentx_engine import (
    AgentXEngine,
    AgentXResponse,
    ReActStep,
)


# ============================================================================
# 1. STREAMLIT PAGE CONFIG & DEVELOPER-FIRST DARK THEME CSS
# ============================================================================

st.set_page_config(
    page_title="AgentX Autonomous Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DARK_SLATE_THEME_CSS = """
<style>
/* Base Viewport & Dark Slate Background (#0F172A) */
.stApp {
    background-color: #0F172A !important;
    color: #F8FAFC !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Inter', -apple-system, monospace;
}

/* 25% Left Control Panel Sidebar (#0B1120) */
section[data-testid="stSidebar"] {
    background-color: #0B1120 !important;
    border-right: 1px solid #334155 !important;
}

section[data-testid="stSidebar"] div.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Glassmorphism Card System (#1E293B with #334155 borders) */
.agentx-card {
    background: rgba(30, 41, 59, 0.75);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    transition: all 0.2s ease-in-out;
}

.agentx-card:hover {
    border-color: #6366F1;
    box-shadow: 0 0 16px rgba(99, 102, 241, 0.2);
}

/* Reactive Status Badges */
.badge-active {
    display: inline-flex;
    align-items: center;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid #10B981;
    color: #10B981;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 9999px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.badge-executing {
    display: inline-flex;
    align-items: center;
    background: rgba(99, 102, 241, 0.18);
    border: 1px solid #6366F1;
    color: #A5B4FC;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 9999px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    animation: glowPulse 1.5s infinite;
}

.badge-idle {
    display: inline-flex;
    align-items: center;
    background: rgba(148, 163, 184, 0.15);
    border: 1px solid #64748B;
    color: #94A3B8;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 9999px;
}

@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
    50% { box-shadow: 0 0 14px 4px rgba(99, 102, 241, 0.3); }
}

/* Typography & Cyberpunk Gradient */
.agentx-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #FFFFFF 0%, #A5B4FC 50%, #6366F1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.agentx-subtitle {
    color: #94A3B8;
    font-size: 0.95rem;
    margin-bottom: 1.25rem;
}

/* Inputs & Form Controls */
div[data-baseweb="input"], div[data-baseweb="textarea"] {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    color: #F8FAFC !important;
}

div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 1px #6366F1 !important;
}

/* Primary Indigo Button */
button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
    border: none !important;
    box-shadow: 0 0 14px rgba(99, 102, 241, 0.4) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"]:hover {
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.65) !important;
    transform: translateY(-1px);
}
</style>
"""

st.markdown(DARK_SLATE_THEME_CSS, unsafe_allow_html=True)


# ============================================================================
# 2. SESSION STATE MANAGEMENT
# ============================================================================

if "agent_status" not in st.session_state:
    st.session_state.agent_status = "IDLE"
if "session_history" not in st.session_state:
    st.session_state.session_history = []


# ============================================================================
# 3. LEFT CONTROL PANEL (25% Sidebar Width)
# ============================================================================

with st.sidebar:
    # 1. Platform Brand & Dynamic Status Badge
    st.markdown('<div class="agentx-title">⚡ AgentX</div>', unsafe_allow_html=True)
    st.caption("Zero-Framework Autonomous Engine v1.0")

    status = st.session_state.agent_status
    if status == "EXECUTING":
        st.markdown('Status: <span class="badge-executing">● EXECUTING REACT LOOP</span>', unsafe_allow_html=True)
    elif status == "ACTIVE":
        st.markdown('Status: <span class="badge-active">● ENGINE ACTIVE</span>', unsafe_allow_html=True)
    else:
        st.markdown('Status: <span class="badge-idle">○ SYSTEM IDLE</span>', unsafe_allow_html=True)

    st.divider()

    # 2. Control Matrix (API Key, Base URL, Model, Hyperparameters)
    st.markdown("#### ⚙️ Control Matrix")
    api_key_input = st.text_input(
        "REST API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        placeholder="sk-...",
        help="OpenAI, Groq, Together, DeepSeek, or local LLM key."
    )
    api_base_url = st.text_input(
        "Base REST URL",
        value=os.getenv("AGENTX_API_BASE_URL", "https://api.openai.com/v1"),
        help="Endpoint for /chat/completions (e.g. https://api.openai.com/v1 or http://localhost:11434/v1)"
    )
    model_choice = st.selectbox(
        "Model Brain",
        ["gpt-4o-mini", "gpt-4o", "llama-3.3-70b-versatile", "deepseek-chat", "custom"],
        index=0
    )
    if model_choice == "custom":
        model_choice = st.text_input("Custom Model ID", value="gpt-4o-mini")

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    max_cycles = st.slider("Max ReAct Cycles", min_value=1, max_value=20, value=10, step=1)

    # 3. Active Tool Routing Toggles
    st.divider()
    st.markdown("#### 🛠️ Active Tool Routing")
    toggle_math = st.checkbox("Calculator (`math_evaluator`)", value=True)
    toggle_scraper = st.checkbox("Web Scraper (`web_scraper`)", value=True)
    toggle_db = st.checkbox("SQLite Query (`db_query_tool`)", value=True)
    toggle_code = st.checkbox("Python Subprocess (`code_executor`)", value=True)
    toggle_memory = st.checkbox("Vector Memory (`memory_search`)", value=True)

    # 4. Document Ingestion for In-Memory Vector Store
    st.divider()
    st.markdown("#### 🧠 In-Memory Vector RAG")
    doc_upload = st.file_uploader("Upload Document (.txt, .md)", type=["txt", "md"])
    if doc_upload is not None:
        raw_doc_text = doc_upload.read().decode("utf-8")
        if st.button("Chunk & Index into Memory"):
            doc_id = doc_upload.name
            c_ids = DEFAULT_VECTOR_STORE.add_document(
                text=raw_doc_text,
                doc_id=doc_id,
                metadata={"source": doc_id, "uploaded_at": str(time.time())}
            )
            st.success(f"Indexed {len(c_ids)} chunks for '{doc_id}'.")

    st.caption(f"Vector Database: **{DEFAULT_VECTOR_STORE.count()}** chunks indexed")


# ============================================================================
# 4. MAIN WORKBENCH (75% Responsive Canvas)
# ============================================================================

st.markdown('<div class="agentx-title">Autonomous Execution Workbench</div>', unsafe_allow_html=True)
st.markdown('<div class="agentx-subtitle">Zero-Framework Agentic Orchestration • Live ReAct Reasoning Accordion • Real-Time Stream Output</div>', unsafe_allow_html=True)

# Preset Quick Directives
st.markdown("**⚡ Sample Autonomous Instructions:**")
col_p1, col_p2, col_p3 = st.columns(3)
preset_query = None

with col_p1:
    if st.button("🧮 Compound Math AST"):
        preset_query = "Calculate the compound interest on a principal of $15,000 at an annual interest rate of 6.5% compounded monthly over 7 years using the formula P * (1 + r/n)**(n*t)."
with col_p2:
    if st.button("📊 SQLite Audit Query"):
        preset_query = "Query the employees database table to find all engineers earning more than $140,000, and compute their average salary."
with col_p3:
    if st.button("🐍 Subprocess Fibonacci Primes"):
        preset_query = "Execute a Python script to compute the first 12 Fibonacci numbers and check which of them are prime numbers."

user_input = st.text_area(
    "Agent Directive:",
    value=preset_query or "",
    placeholder="Enter high-level goal (e.g. Scrape a webpage, query employees SQLite database, run Python code, or evaluate mathematical formulas)...",
    height=95
)

col_run, col_reset = st.columns([4, 1])
with col_run:
    run_directive = st.button("🚀 Dispatch AgentX Directive", type="primary", use_container_width=True)
with col_reset:
    if st.button("🧹 Clear History", use_container_width=True):
        st.session_state.session_history = []
        st.session_state.agent_status = "IDLE"
        st.rerun()

# Build filtered tool registry based on toggles
active_tool_names = []
if toggle_math:
    active_tool_names.append("math_evaluator")
if toggle_scraper:
    active_tool_names.append("web_scraper")
if toggle_db:
    active_tool_names.append("db_query_tool")
if toggle_code:
    active_tool_names.append("code_executor")
if toggle_memory:
    active_tool_names.append("memory_search")

runtime_registry = ToolRegistry()
for name in active_tool_names:
    t = DEFAULT_TOOL_REGISTRY.get(name)
    if t:
        runtime_registry.register(t)

# Execution Flow
if run_directive:
    if not user_input.strip():
        st.warning("Please provide an instruction or directive.")
    elif not api_key_input.strip():
        st.error("API Key is required to call the LLM endpoint. Enter your key in the Left Control Panel.")
        st.info("Tip: You can use OpenAI, Groq, Together, DeepSeek, or run Ollama locally at http://localhost:11434/v1 without an API key.")
    else:
        st.session_state.agent_status = "EXECUTING"

        engine = AgentXEngine(
            api_key=api_key_input.strip(),
            api_base_url=api_base_url.strip(),
            model=model_choice,
            temperature=float(temperature),
            max_iterations=int(max_cycles),
            registry=runtime_registry,
            memory=DEFAULT_VECTOR_STORE
        )

        st.markdown("### 🔍 Live ReAct Execution Trace")
        trace_area = st.container()

        generator = engine.run_stream(user_input.strip())
        final_resp: Optional[AgentXResponse] = None

        try:
            while True:
                step: ReActStep = next(generator)
                with trace_area:
                    card_title = f"Step {step.step_number}: {step.tool_name or 'Synthesizing Response'}"
                    with st.expander(f"🧩 {card_title} ({step.duration_seconds:.2f}s)", expanded=True):
                        if step.thought:
                            st.markdown(f"**🧠 [Reasoning...]**\n\n{step.thought}")
                        if step.tool_name:
                            st.markdown(f"**⚙️ [Tool Called: `{step.tool_name}`]**")
                            st.json(step.tool_arguments or {})
                        if step.tool_output:
                            st.markdown("**📊 [Result Received]**")
                            st.code(step.tool_output, language="text")
        except StopIteration as stop:
            final_resp = stop.value

        st.session_state.agent_status = "ACTIVE"

        if final_resp:
            st.success(
                f"✅ Execution Satisfied in {final_resp.execution_time_seconds:.2f}s • "
                f"{final_resp.total_iterations} ReAct cycles • "
                f"{final_resp.total_tool_calls} tool executions"
            )
            st.markdown("### 🏁 Synthesized Response")
            st.markdown(final_resp.final_answer)

            # Record in session history audit trail
            st.session_state.session_history.append({
                "time": time.strftime("%H:%M:%S"),
                "query": user_input.strip(),
                "answer": final_resp.final_answer,
                "cycles": final_resp.total_iterations,
                "tool_calls": final_resp.total_tool_calls
            })

# Render Session History Timeline
if st.session_state.session_history and not run_directive:
    st.divider()
    st.markdown("### 📜 Session History Timeline")
    for item in reversed(st.session_state.session_history):
        with st.container():
            st.markdown(f"**Directive ({item['time']} | {item['cycles']} cycles | {item['tool_calls']} tool calls):**")
            st.info(item["query"])
            st.markdown(item["answer"])
            st.markdown("---")
