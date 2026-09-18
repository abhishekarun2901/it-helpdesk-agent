import sys
from pathlib import Path

# Safeguard sys.path for standalone process runs
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mcp.server.mcpserver import MCPServer

from app.config import get_settings

settings = get_settings()
mcp_server = MCPServer(name="FilesystemLogServer")

@mcp_server.tool(name="read_diagnostic_logs")
def read_diagnostic_logs(log_file_name: str, search_keyword: str = "") -> str:
    """
    Reads IT system and authentication log files from the server log directory.
    Use this to diagnose system crashes, login lockouts, network timeouts, or printer faults.
    Common log files: 'auth.log', 'system.log'.
    """
    logs_dir = settings.LOGS_DIR
    target_file = (logs_dir / log_file_name).resolve()

    if not str(target_file).startswith(str(logs_dir.resolve())):
        return "Security error: Access denied outside the designated log directory."

    if not target_file.exists() or not target_file.is_file():
        available = [f.name for f in logs_dir.glob("*.log")] if logs_dir.exists() else []
        return f"File '{log_file_name}' not found. Available logs: {available}"

    try:
        with open(target_file, encoding="utf-8") as f:
            lines = f.readlines()

        if search_keyword:
            terms = search_keyword.lower().split()
            matches = [
                line.strip()
                for line in lines
                if all(term in line.lower() for term in terms)
            ]
            if not matches:
                return f"No entries matching '{search_keyword}' in {log_file_name}."
            content = "\n".join(matches)
        else:
            content = "".join(lines[-25:])

        return (
            f'<untrusted_mcp_data source="{log_file_name}">\n'
            f'{content.strip()}\n'
            f'</untrusted_mcp_data>'
        )
    except Exception as exc:  # noqa: BLE001
        return f"Failed reading log file {log_file_name}: {exc!s}"

if __name__ == "__main__":
    mcp_server.run(transport="stdio")