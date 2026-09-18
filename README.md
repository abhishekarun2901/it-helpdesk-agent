# IT Helpdesk Agent

An internal IT support chatbot built with FastAPI, LangGraph, Milvus-Lite RAG, and an out-of-process MCP diagnostic server.

---

## 1. Architecture & Design Choices

### Explicit Decision & Routing Logic (LangGraph)
- **Pre-Execution Routing (`router_node`)**: An initial classification step inspects the incoming user request before entering the agent loop:
  - `escalate`: Immediately rejects unauthorized actions (e.g., accessing payroll/HR records) at the graph level without invoking LLM tools.
  - `clarify`: Detects ambiguous or uninterpretable input and requests clarifying details directly.
  - `proceed`: Passes legitimate IT inquiries to the reasoning loop.
- **Agent Reasoning Loop (`agent_node` <-> `tool_node`)**: ReAct pattern bound to native and MCP tools with checkpointer memory persistence.

### Knowledge Retrieval (RAG) with Milvus-Lite
- **Embedded Vector Store**: Uses embedded Milvus-Lite persisted locally at `data/milvus_demo.db` with HNSW indexing (`M=8`, `efConstruction=64`).
- **Semantic Chunking**: Employs `SemanticChunker` paired with `sentence-transformers/all-MiniLM-L6-v2` to maintain complete policy context boundaries.
- **Untrusted Context Guarding**: Retrieved documents and log outputs are wrapped in explicit boundary tags (`<untrusted_retrieved_data>` and `<untrusted_mcp_data>`), preventing prompt injection attacks from external data.

### Model Context Protocol (MCP) Integration
- **Out-of-Process Server (`app/mcp_server.py`)**: A dedicated MCP server running on standard STDIO transport exposing `read_diagnostic_logs`.
- **Unified Client (`app/mcp_client.py`)**: Uses the official `mcp` client SDK to connect, inspect, and invoke the server tool via LangChain's tool execution path.

### Asynchronous API Endpoints (FastAPI)
- `GET /health`: Health check verification.
- `WS /ws/chat`: Real-time streaming WebSocket endpoint maintaining conversational session state across turns.
- `POST /trigger` & `GET /trigger/{job_id}`: Out-of-band asynchronous task execution simulating webhooks/scheduled runs with polling support.

---

## 2. Setup & Installation

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Environment configuration
# Create a .env file with your Groq credentials:
# GROQ_API_KEY="your-groq-api-key"