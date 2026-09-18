from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

SUPPORTED_GROQ_MODELS = {
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-safeguard-20b",
}
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = DEFAULT_GROQ_MODEL

    # Paths
    DATA_DIR: Path = BASE_DIR / "data"
    DOCS_DIR: Path = BASE_DIR / "data" / "docs"
    LOGS_DIR: Path = BASE_DIR / "data" / "logs"
    MILVUS_PATH: str = str(BASE_DIR / "data" / "milvus_demo.db")

    # Vector store parameters
    COLLECTION_NAME: str = "it_helpdesk_kb"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    @field_validator("GROQ_MODEL", mode="before")
    @classmethod
    def validate_groq_model(cls, v: str) -> str:
        if not v:
            return DEFAULT_GROQ_MODEL
        v_str = str(v).strip()
        if "llama" in v_str.lower() or v_str not in SUPPORTED_GROQ_MODELS:
            print(
                f"[Config Warning] Model '{v_str}' is unsupported or decommissioned on Groq for this account. "
                f"Automatically defaulting to '{DEFAULT_GROQ_MODEL}'."
            )
            return DEFAULT_GROQ_MODEL
        return v_str

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()