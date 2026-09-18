import asyncio
import os
import sys
from pathlib import Path

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

from app.config import get_settings

settings = get_settings()

class ReadLogInput(BaseModel):
    log_file_name: str = Field(description="Name of the log file to inspect, e.g., 'auth.log' or 'system.log'")
    search_keyword: str = Field(default="", description="Keyword to filter log lines")

async def _invoke_mcp_tool_async(log_file_name: str, search_keyword: str = "") -> str:
    server_script = Path(__file__).resolve().parent / "mcp_server.py"
    project_root = Path(__file__).resolve().parent.parent

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{existing_pythonpath}" if existing_pythonpath else str(project_root)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        env=env,
    )

    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.call_tool(
            "read_diagnostic_logs",
            arguments={"log_file_name": log_file_name, "search_keyword": search_keyword},
        )

        is_err = getattr(result, "is_error", getattr(result, "isError", False))
        if is_err:
            return f"MCP Tool Execution Error: {result.content}"

        return "\n".join([item.text for item in result.content if hasattr(item, "text")])

def search_system_logs_sync(log_file_name: str, search_keyword: str = "") -> str:
    """Synchronous fallback wrapper."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _invoke_mcp_tool_async(log_file_name, search_keyword)).result()
    return asyncio.run(_invoke_mcp_tool_async(log_file_name, search_keyword))

def get_mcp_tool() -> StructuredTool:
    """Returns the LangChain-wrapped MCP filesystem tool supporting both sync and async."""
    return StructuredTool.from_function(
        func=search_system_logs_sync,
        coroutine=_invoke_mcp_tool_async,
        name="search_system_logs",
        description="Reads diagnostic and authentication logs via an out-of-process MCP server.",
        args_schema=ReadLogInput,
    )

search_system_logs = search_system_logs_sync