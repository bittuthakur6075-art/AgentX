# ⚡ AgentX: Zero-Framework Autonomous Agentic Engine & Control System

> **A production-grade, modular Agentic AI system written in pure Python from scratch without high-level agent libraries (no LangChain, LlamaIndex, CrewAI, or AutoGen).**

AgentX interfaces directly with low-level LLM REST APIs, executes a native **ReAct (Reason + Act)** loop, features a custom in-memory vector database with NumPy cosine similarity, and isolates tool execution with AST whitelisting and sub-process sandboxing.

---

## 🏗️ Technical Architecture

```mermaid
flowchart TD
    User([User Prompt / Instruction]) --> Engine[AgentX ReAct Execution Engine]
    
    subgraph Engine_Core [AgentX Engine Core]
        ContextMgr[Chat Context Manager\nProactive Window Trimming]
        PayloadHandler[Low-Level HTTP REST Handler\nOpenAI / Groq / DeepSeek / Ollama]
        ReActLoop[ReAct Loop Controller\nObserve -> Think -> Act -> Evaluate]
    end
    
    Engine --> ContextMgr
    ContextMgr --> PayloadHandler
    PayloadHandler --> ReActLoop
    
    subgraph Tool_Registry [Tool Registry & Schemas]
        Registry[ToolRegistry\n@agentx_tool Decorator]
        MathTool[math_evaluator\nAST Whitelist Parser]
        WebTool[web_scraper\nrequests + BeautifulSoup]
        DBTool[db_query_tool\nSafe SQLite Read-Only]
        CodeTool[code_executor\nIsolated Subprocess Runner]
    end

    subgraph Memory_RAG [Custom Vector Memory & RAG]
        Chunker[Sliding Window Chunker\nConfigurable Overlap]
        Embeddings[Dual-Mode Embeddings\nREST API + Subword Hashing]
        VectorStore[In-Memory Vector DB\nNumPy Matrix Cosine Similarity]
        MetaIndex[Metadata Inverted Index\nFast Key-Value Filtering]
    end

    ReActLoop -->|Call Tool| Registry
    Registry --> MathTool
    Registry --> WebTool
    Registry --> DBTool
    Registry --> CodeTool
    Registry -->|memory_search| VectorStore

    MathTool -->|Observation| ContextMgr
    WebTool -->|Observation| ContextMgr
    DBTool -->|Observation| ContextMgr
    CodeTool -->|Observation| ContextMgr
    VectorStore -->|Context Chunks| ContextMgr
    
    ReActLoop -->|Final Answer| Response([Structured AgentXResponse])
```

---

## 📁 Repository Structure

```
AgentX/
├── agentx/                          # Core AgentX Framework Package
│   ├── __init__.py                  # Public exports and versioning
│   ├── config.py                    # Enterprise configuration & API key management
│   ├── tools.py                     # Tool decorator, JSON schema builder & native tools
│   ├── memory.py                    # In-memory vector DB, sliding chunker & cosine similarity
│   └── engine.py                    # Core ReAct loop, context manager & low-level HTTP handler
├── config.py                        # Root re-export of agentx.config
├── tools.py                         # Root re-export of agentx.tools
├── memory.py                        # Root re-export of agentx.memory
├── engine.py                        # Root re-export of agentx.engine
├── app.py                           # Dual UI: Interactive Streamlit Dashboard & FastAPI Gateway
├── test_agentx.py                   # Comprehensive unit & integration test suite
├── requirements.txt                 # Minimum production dependencies
└── README.md                        # Architectural specification and usage guide
```

---

## 🚀 Key Modules & Capabilities

### 1. ReAct Execution Engine (`agentx/engine.py`)
- **Observe -> Think -> Call Tool -> Evaluate Tool Output -> Respond**:
  Maintains execution state, measures step latency, and synthesizes final answers.
- **Low-Level HTTP Payload Handler**:
  Direct raw `requests.post` to `/v1/chat/completions` (OpenAI, Groq, Together, DeepSeek, Local Ollama, vLLM). No opaque third-party wrappers.
- **Explicit Context Window Manager**:
  Estimates token consumption and applies proactive FIFO/compression trimming to intermediate tool observations while strictly safeguarding the system instructions and initial user query.
- **Dual-Mode Execution**:
  Seamlessly supports native model function calling (`tool_calls`) as well as traditional ReAct text parsing (`Thought: ... Action: ... Action Input: ...`).

### 2. Tool Registry & Schema Builder (`agentx/tools.py`)
- **`@agentx_tool` Decorator**:
  Inspects standard Python function signatures, type annotations, and docstrings to automatically generate standard JSON Schemas compatible with OpenAI function calling specifications.
- **Built-in Native Tools**:
  1. `math_evaluator`: Safe Python expression parser built on Python's `ast` module. Disallows `eval()`, attribute navigation, or system calls. Whitelists arithmetic, powers, roots, trigonometry, and logarithms.
  2. `web_scraper`: HTTP client with `BeautifulSoup` cleaning. Strips scripts, styles, navigation, headers, footers, and noise tags.
  3. `db_query_tool`: Safe SQLite executor for structured data retrieval. Validates read-only statements (`SELECT`, `WITH`, `PRAGMA`), rejecting mutations (`DROP`, `DELETE`, `UPDATE`, `INSERT`). Pre-seeds an enterprise sandbox database with `employees`, `products`, and `sales`.
  4. `code_executor`: Isolated sub-process Python code runner capturing `stdout`, `stderr`, execution duration, and exit codes with strict timeout limits.

### 3. Custom Vector Memory & RAG (`agentx/memory.py`)
- **NumPy Cosine Similarity**:
  Vectorized batch matrix multiplication for instant similarity ranking:
  $$\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \|\mathbf{B}\|_2}$$
- **Sliding Window Chunker**:
  Slices long documents into overlapping segments with configurable `chunk_size` and `chunk_overlap`, snapping to whitespace boundaries.
- **Dual-Mode Embedding Engine**:
  - Direct REST API embeddings (`/v1/embeddings`).
  - Zero-dependency local subword TF-IDF hashing vectorizer for offline and local operation.
- **Metadata Inverted Index**:
  Indexes documents by metadata attributes for pre-filtered vector queries.

---

## 🛠️ Getting Started

### 1. Installation

```bash
# Clone or navigate to the repository
cd C:\Code\AgentX

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory (or pass parameters directly to `AgentXConfig`):

```ini
# LLM Endpoint Configuration
OPENAI_API_KEY=your_api_key_here
AGENTX_API_BASE_URL=https://api.openai.com/v1
AGENTX_MODEL=gpt-4o-mini
AGENTX_MAX_ITERATIONS=10
AGENTX_TEMPERATURE=0.1

# Optional Vector Embedding Endpoint (falls back automatically to built-in local vectorizer)
AGENTX_EMBEDDING_API_KEY=your_api_key_here
AGENTX_EMBEDDING_MODEL=text-embedding-3-small
```

---

## 💻 Programmatic Usage Examples

### Example 1: Creating Custom Tools & Running the Engine

```python
from agentx.config import AgentXConfig
from agentx.tools import agentx_tool, DEFAULT_REGISTRY
from agentx.engine import AgentXEngine

# Define a custom tool with standard type hints and docstring
@agentx_tool()
def currency_converter(amount: float, from_curr: str, to_curr: str) -> str:
    """
    Convert monetary amounts between major currencies.
    :param amount: Amount of money to convert.
    :param from_curr: 3-letter currency code, e.g. USD, EUR.
    :param to_curr: Target 3-letter currency code, e.g. GBP, JPY.
    """
    rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 152.0}
    if from_curr.upper() in rates and to_curr.upper() in rates:
        in_usd = amount / rates[from_curr.upper()]
        converted = in_usd * rates[to_curr.upper()]
        return f"{amount} {from_curr.upper()} = {converted:.2f} {to_curr.upper()}"
    return f"Unsupported currency pair: {from_curr} to {to_curr}"

# Initialize engine
config = AgentXConfig(
    api_key="your_api_key",
    model="gpt-4o-mini"
)
engine = AgentXEngine(config=config, registry=DEFAULT_REGISTRY)

# Execute ReAct loop
response = engine.run("What is $2,500 USD in EUR, and what is the compound growth if invested at 7% for 5 years?")

print("=== Final Answer ===")
print(response.final_answer)

print("\n=== Execution Trace ===")
for step in response.trace:
    print(f"Step {step.step_number}: {step.tool_name} -> Output: {step.tool_output}")
```

### Example 2: Ingesting Documents & Semantic RAG Search

```python
from agentx.memory import VectorMemory

# Initialize in-memory vector database
memory = VectorMemory(chunk_size=400, chunk_overlap=80)

# Ingest technical documentation
memory.add_document(
    text="AgentX avoids heavyweight frameworks and relies on raw HTTP REST requests. "
         "Context management uses an explicit FIFO token trimmer to preserve system prompts.",
    doc_id="agentx_architecture_spec",
    metadata={"topic": "architecture", "version": "1.0"}
)

# Perform vectorized similarity search
results = memory.search("How does AgentX manage its context window?", top_k=2)

for hit in results:
    print(f"Match [{hit.similarity:.4f}]: {hit.text}")
```

---

## 🖥️ User Interfaces

### 1. Interactive Streamlit Dashboard

Run the rich web interface featuring interactive ReAct trace visualization, tool sandboxing, vector memory inspection, and database viewer:

```bash
streamlit run app.py
```

### 2. Production FastAPI REST Gateway

Launch the REST server with automatic OpenAPI Swagger documentation:

```bash
# Start server
python app.py --api --port 8000

# Or via uvicorn
uvicorn app:app --host 127.0.0.1 --port 8000
```

#### API Endpoints:
- `GET /`: Health check and system metadata.
- `GET /api/tools`: List all registered tools and their generated JSON schemas.
- `POST /api/chat`: Execute autonomous ReAct run (`query`, optional `api_key`, `model`).
- `POST /api/tools/execute`: Manually run a registered tool sandbox.
- `POST /api/memory/add`: Ingest documents into Vector Memory.
- `POST /api/memory/search`: Semantic Cosine search across chunks.
- `DELETE /api/memory/clear`: Reset vector database.

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

---

## 🧪 Running Tests

Verify the entire test suite covering math safety, read-only SQL validation, subprocess isolation, sliding chunker, and the context manager:

```bash
python test_agentx.py
```
