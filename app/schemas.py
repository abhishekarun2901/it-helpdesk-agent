from enum import Enum

from pydantic import BaseModel, Field, field_validator


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CreateTicketInput(BaseModel):
    issue: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Detailed description of the hardware or software incident.",
    )
    priority: TicketPriority = Field(
        default=TicketPriority.MEDIUM,
        description="Severity level of the support ticket.",
    )

    @field_validator("issue")
    @classmethod
    def clean_issue(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Issue description cannot be empty or whitespace.")
        return cleaned

class CheckSystemStatusInput(BaseModel):
    service_name: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Name of the infrastructure service (e.g., 'email', 'vpn', 'ad').",
    )

class ResetPasswordInput(BaseModel):
    user_id: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9._-]{3,30}$",
        description="Target enterprise username or employee identifier.",
    )

class LookupPolicyInput(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=250,
        description="Search query against internal IT policy documents.",
    )

# WebSocket Message Schemas
class WebSocketIncomingMessage(BaseModel):
    content: str = Field(..., min_length=1, max_length=1000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Message cannot be whitespace only.")
        return cleaned

class WebSocketOutgoingMessage(BaseModel):
    type: str = "token"
    content: str