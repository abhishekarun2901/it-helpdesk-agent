import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from pydantic import BaseModel

from app.graph import helpdesk_agent
from app.vectorstore import close_milvus_client, init_vectorstore

logger = logging.getLogger(__name__)
job_store: dict[str, dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initializing Milvus-Lite Knowledge Base...")
    init_vectorstore(force_reload=False)
    print("[Startup] Knowledge base ready.")
    yield
    print("[Shutdown] Closing Milvus client and releasing file locks...")
    close_milvus_client()

app = FastAPI(
    title="IT Helpdesk Agent Service",
    description="FastAPI service hosting LangGraph agent with Milvus-Lite RAG and MCP tools",
    lifespan=lifespan,
)

class TriggerPayload(BaseModel):
    query: str

class TriggerResponse(BaseModel):
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: str | None = None

@app.get("/health")
async def health_check() -> dict[str, str]:
    """Trivial health check endpoint."""
    return {"status": "healthy"}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    Live streaming WebSocket chat endpoint.
    Maintains session history across turns via LangGraph thread_id.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    try:
        while True:
            user_input = await websocket.receive_text()
            if not user_input.strip():
                continue

            input_data = {"messages": [HumanMessage(content=user_input)]}

            async for chunk, _metadata in helpdesk_agent.astream(
                input_data, config=config, stream_mode="messages"
            ):
                if isinstance(chunk, (AIMessage, AIMessageChunk)) and chunk.content:
                    await websocket.send_text(str(chunk.content))

    except WebSocketDisconnect:
        pass

async def process_headless_agent_run(job_id: str, query: str):
    """Executes the agent graph in an out-of-band asynchronous task."""
    config = {"configurable": {"thread_id": job_id}, "recursion_limit": 10}
    input_data = {"messages": [HumanMessage(content=query)]}

    try:
        result = await helpdesk_agent.ainvoke(input_data, config=config)
        final_message = result["messages"][-1]
        output_content = (
            final_message.content if isinstance(final_message, AIMessage) else str(final_message)
        )

        job_store[job_id] = {
            "status": "completed",
            "result": output_content,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Headless agent run failed for job %s: %s", job_id, exc)
        job_store[job_id] = {
            "status": "failed",
            "result": f"Execution error: {exc!s}",
        }

@app.post("/trigger", response_model=TriggerResponse, status_code=202)
async def trigger_job(payload: TriggerPayload, background_tasks: BackgroundTasks):
    """Asynchronously triggers an agent run decoupled from interactive sessions."""
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "processing", "result": None}
    background_tasks.add_task(process_headless_agent_run, job_id, payload.query)
    return TriggerResponse(job_id=job_id, status="processing")

@app.get("/trigger/{job_id}", response_model=JobStatusResponse)
async def get_trigger_status(job_id: str):
    """Poll the status or output of an async trigger job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job ID not found")

    job_data = job_store[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job_data["status"],
        result=job_data.get("result"),
    )