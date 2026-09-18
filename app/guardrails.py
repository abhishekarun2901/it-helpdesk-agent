import re

from pydantic import BaseModel, Field, field_validator

# Block patterns indicating jailbreaks or prompt injection
INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"(?i)system\s*prompt",
    r"(?i)you\s+are\s+now\s+(in\s+)?(developer\s+mode|dan)",
    r"(?i)reveal\s+(internal|system)\s+(instructions|keys|passwords)",
]

# Sensitive patterns that must never leak to the client
LEAKAGE_PATTERNS = [
    r"gsk_[A-Za-z0-9]{32,}",                # Groq API Keys
    r"(?i)groq_api_key",                     # Key reference
    r"(?i)<untrusted_[a-z_]+>",              # Internal untrusted wrapper tags
    r"(?i)</untrusted_[a-z_]+>",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",  # PII email leaks
]

class UserQueryInput(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="User IT query")

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or pure whitespace.")
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, cleaned):
                raise ValueError("Security Violation: Adversarial prompt injection pattern detected.")
        return cleaned

class GuardrailResponse(BaseModel):
    safe: bool
    sanitized_content: str

def sanitize_agent_output(content: str) -> GuardrailResponse:
    """Output guardrail: redact secrets, internal wrapper tags, or leaked API keys."""
    sanitized = content
    for pattern in LEAKAGE_PATTERNS:
        sanitized = re.sub(pattern, "[REDACTED]", sanitized)

    return GuardrailResponse(safe=True, sanitized_content=sanitized)