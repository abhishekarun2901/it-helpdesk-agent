from langchain_core.tools import tool

from app.vectorstore import get_retriever


@tool
def lookup_policy_kb(query: str) -> str:
    """Searches internal company IT policies, FAQs, and documentation."""
    retriever = get_retriever(k=2)
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant policy documents found."
    return "\n---\n".join([
        f'<untrusted_retrieved_data source="{d.metadata.get("source", "unknown")}">\n'
        f'{d.page_content}\n'
        f'</untrusted_retrieved_data>'
        for d in docs
    ])

@tool
def create_ticket(issue: str, priority: str = "medium") -> str:
    """Creates a new IT support ticket for hardware, device, or account incidents."""
    import random
    ticket_id = f"TICK-{random.randint(1000, 9999)}"
    return f"Ticket {ticket_id} created successfully with {priority} priority for issue: {issue}"

@tool
def check_system_status(service_name: str) -> str:
    """Checks the operational health and status of an IT service."""
    service = service_name.lower()
    if "email" in service or "exchange" in service:
        return "Service 'Corporate Email' is OPERATIONAL but degraded (12s sync delay)."
    if "vpn" in service:
        return "Service 'VPN Gateway' is OPERATIONAL with 99.98% uptime."
    return f"Service '{service_name}' is OPERATIONAL. No incidents reported."

@tool
def reset_password(user_id: str) -> str:
    """Initiates a secure password reset sequence for a given user ID."""
    return f"Password reset link generated and dispatched to registered phone/email for user {user_id}."

def get_all_tools():
    return [lookup_policy_kb, create_ticket, check_system_status, reset_password]

def get_tool_map():
    return {t.name: t for t in get_all_tools()}