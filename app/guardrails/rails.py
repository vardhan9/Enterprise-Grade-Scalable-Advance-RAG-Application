import re

from langchain_openai import ChatOpenAI
import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails
from portkey_ai import PORTKEY_GATEWAY_URL

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, JAILBREAK_PATTERNS, OFF_TOPIC_PATTERNS


_rails: LLMRails | None = None

from dataclasses import dataclass


@dataclass
class GuardResult:
    blocked: bool
    rail: str | None = None
    response: str | None = None

def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses llama-3.1-8b-instant for fast intent classification at the gate —
    the heavier llama-3.3-70b-versatile is reserved for the RAG pipeline.
    """
    global _rails

    guard_llm = ChatOpenAI(
    api_key=settings.PORTKEY_API_KEY,
    base_url=PORTKEY_GATEWAY_URL,
    model=f"@{settings.GROQ_SLUG}/qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=30,       # a canonical form is short — cap it hard
   
)

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised (llama-3.1-8b-instant).")
    
def parse_rail_response(content: str) -> tuple[bool, str | None, str]:
    """
    Returns:
        fired: whether a rail fired
        rail: rail name, e.g. OFF_TOPIC
        response: clean user-facing response
    """

    match = re.match(r"^\[RAIL:([A-Z_]+)\]\s*", content)

    if not match:
        return False, None, content.strip()

    rail = match.group(1)

    response = content[match.end():].strip()

    return True, rail, response

def deterministic_guard(message: str) -> tuple[bool, str | None]:
    text = " ".join(message.lower().split())

    for pattern in JAILBREAK_PATTERNS:
        if pattern in text:
            return True, (
                "I maintain consistent guidelines regardless of how I am prompted. "
                "I am here to help with Kubernetes, Intel, and networking. "
                "What can I help you with?"
            )

    for pattern in OFF_TOPIC_PATTERNS:
        if pattern in text:
            return True, (
                "I'm an Enterprise IT Assistant focused on Kubernetes, "
                "Intel hardware, and networking. I can't help with that — "
                "but ask me anything technical!"
            )

    return False, None

def guard(message: str) -> GuardResult:

    # ─────────────────────────────────────────────
    # Layer 1 — deterministic guard
    # ─────────────────────────────────────────────
    fired, response = deterministic_guard(message)

    if fired:
        logfire.info(
            f"🛡️ Deterministic guard fired | query='{message[:80]}'"
        )

        return GuardResult(
            blocked=True,
            rail="DETERMINISTIC",
            response=response,
        )

    # ─────────────────────────────────────────────
    # Layer 2 — NeMo Guardrails
    # ─────────────────────────────────────────────
    if _rails is None:
        logfire.warning(
            "⚠️ Guardrails not initialised — skipping gate."
        )

        return GuardResult(blocked=False)

    with logfire.span("🛡️ Guardrails Check"):

        result = _rails.generate(
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ]
        )

        # NeMo returns:
        # {"role": "assistant", "content": "..."}
        content = (
            result.get("content", "")
            if isinstance(result, dict)
            else str(result)
        )

        # Parse our [RAIL:XXX] marker
        fired, rail, response = parse_rail_response(content)

        if fired:
            logfire.info(
                f"🛡️ Guardrails fired | "
                f"rail={rail} | "
                f"query='{message[:80]}'"
            )

            return GuardResult(
                blocked=True,
                rail=rail,
                response=response,
            )

        logfire.info("✅ Guardrails passed.")

        return GuardResult(blocked=False)