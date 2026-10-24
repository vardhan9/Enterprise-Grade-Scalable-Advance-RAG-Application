import os
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)


class Settings:
    # Load environment variables from .env file
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_COLLECTION_NAME = "enterprise-rag"
    GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

    # --- CHUNKING (used before embedding / indexing) ---
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))

    # --- LLM GATEWAY (PORTKEY) ---
    # Accept either PORTKEY_API_KEY or legacy lowercase portkey_api_key
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY") or os.getenv("portkey_api_key")
    GROQ_SLUG = os.getenv("GROQ_SLUG", "rag-enterprise-openrouter")
    GROQ_SLUG_2 = os.getenv("GROQ_SLUG_2", "marthala-groq-2")

    # Short OpenRouter-style id (e.g. qwen/qwen3.8-27b or deepseek/deepseek-r1)
    # or a full Portkey route starting with @
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

    @classmethod
    def portkey_chat_model(cls) -> str:
        """Full Portkey model string used by ChatOpenAI / guardrails."""
        m = (cls.LLM_MODEL or "").strip()
        if m.startswith("@"):
            return m
        return f"@{cls.GROQ_SLUG}/{m}"


settings = Settings()


def set_env_value(key: str, value, env_path: Path | None = None) -> None:
    """Upsert KEY=value in .env and refresh os.environ + Settings class attrs."""
    path = env_path or _ENV_PATH
    str_value = str(value)
    line = f"{key}={str_value}"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()
    found = False
    out = []
    for raw in lines:
        if raw.strip().startswith(f"{key}="):
            out.append(line)
            found = True
        else:
            out.append(raw)
    if not found:
        if out and out[-1].strip():
            out.append("")
        out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.environ[key] = str_value

    cast = {
        "CHUNK_SIZE": int,
        "CHUNK_OVERLAP": int,
        "EMBEDDING_BATCH_SIZE": int,
        "LLM_MAX_TOKENS": int,
        "LLM_TEMPERATURE": float,
    }.get(key, str)
    if hasattr(Settings, key):
        setattr(Settings, key, cast(str_value))
