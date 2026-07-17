import os
from dotenv import load_dotenv

load_dotenv()
class Settings:
    # Load environment variables from .env file
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_COLLECTION_NAME = "enterprise-rag"
    GROQ_MODEL_NAME="llama-3.3-70b-versatile"


    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG =  "marthala-groq"     # primary: @rag/llama-3.3-70b-versatile
    GROQ_SLUG_2 = "marthala-groq-2"  # fallback: @brag/llama-3.1-8b-instant

settings = Settings()