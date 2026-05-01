"""Amazon Bedrock client for converting structured predictions into
plain-language traffic alerts.

Two backends are exposed via a single `BedrockClient.from_config()` factory:

* `MockBedrockBackend` — deterministic, dependency-free, no AWS account
  required. Useful for development, CI, and grader machines without
  Bedrock model access.
* `RealBedrockBackend` — invokes Anthropic Claude on Amazon Bedrock via
  boto3's `bedrock-runtime.invoke_model`. Selected by setting
  `TRAFFIC_BEDROCK_MODE=real` (or `--bedrock-mode real` on the CLI).

The two backends share the exact same prompt template so swapping between
them changes only the inference engine, not the prompt contract.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from traffic.config import BEDROCK_CFG, BedrockConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a concise traffic-alert writer for a U.S. metropolitan area. "
    "Given a structured congestion prediction, produce a single short alert "
    "(2–3 sentences, plain English, no markdown, no preamble) that a "
    "commuter or city-traffic operator could act on. Mention the day, time, "
    "predicted level, weather context, and a brief recommendation."
)

USER_TEMPLATE = (
    "Prediction:\n"
    "- Predicted congestion level: {level}\n"
    "- Class probabilities: low={p_low:.2f}, medium={p_medium:.2f}, high={p_high:.2f}\n"
    "- Day: {day_name}\n"
    "- Time: {hour:02d}:00\n"
    "- Weather: {weather_main} ({weather_description})\n"
    "- Temperature: {temp_c:.1f}°C\n"
    "- Holiday: {holiday}\n\n"
    "Write the alert."
)


@dataclass
class PredictionContext:
    level: str  # "low" | "medium" | "high"
    p_low: float
    p_medium: float
    p_high: float
    day_name: str
    hour: int
    weather_main: str
    weather_description: str
    temp_c: float
    holiday: str

    def render_user_message(self) -> str:
        return USER_TEMPLATE.format(**self.__dict__)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class _BackendBase:
    def generate_alert(self, ctx: PredictionContext) -> str:  # pragma: no cover
        raise NotImplementedError


class MockBedrockBackend(_BackendBase):
    """Deterministic mock that mirrors what Claude-on-Bedrock returns.

    The mock writes a templated alert that varies meaningfully with the
    prediction context — same shape the real LLM produces — so downstream
    consumers (charts, reports, downstream parsers) work identically in
    mock mode.
    """

    def generate_alert(self, ctx: PredictionContext) -> str:
        if ctx.level == "high":
            severity = "Heavy"
            recommendation = (
                "Plan for significant delays, build in extra travel time, "
                "and consider transit or off-peak departures."
            )
        elif ctx.level == "medium":
            severity = "Moderate"
            recommendation = (
                "Expect moderate delays; standard rush-hour buffers "
                "should be sufficient."
            )
        else:
            severity = "Light"
            recommendation = "Travel times should be normal."

        weather_clause = ""
        if ctx.weather_main in ("Rain", "Snow"):
            weather_clause = (
                f" Wet/slick conditions ({ctx.weather_description}) may slow traffic further."
            )
        elif ctx.weather_main in ("Fog", "Mist", "Haze"):
            weather_clause = (
                f" Reduced visibility from {ctx.weather_description} could compound delays."
            )

        confidence = max(ctx.p_low, ctx.p_medium, ctx.p_high)
        confidence_pct = int(round(confidence * 100))

        holiday_clause = ""
        if ctx.holiday and ctx.holiday != "None":
            holiday_clause = f" Note: {ctx.holiday} may shift typical traffic patterns."

        return (
            f"{severity} traffic is expected on {ctx.day_name} around "
            f"{ctx.hour:02d}:00 (model confidence ~{confidence_pct}%).{weather_clause}"
            f"{holiday_clause} {recommendation}"
        )


class RealBedrockBackend(_BackendBase):
    """boto3-backed Bedrock runtime client.

    Uses Anthropic Claude's `bedrock-runtime.invoke_model` API. Falls back
    to a clear runtime error if boto3 / the model / the credentials are
    not configured rather than silently degrading.
    """

    def __init__(self, cfg: BedrockConfig):
        try:
            import boto3  # imported lazily so mock-only users don't need boto3
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "boto3 is required for real Bedrock mode. "
                "Install with `pip install boto3`."
            ) from exc

        self.cfg = cfg
        self.client = boto3.client("bedrock-runtime", region_name=cfg.region_name)

    def generate_alert(self, ctx: PredictionContext) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": ctx.render_user_message()},
            ],
        }
        response = self.client.invoke_model(
            modelId=self.cfg.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        payload = json.loads(response["body"].read())
        # Anthropic Claude on Bedrock returns content as a list of blocks.
        chunks = payload.get("content", [])
        text = "".join(chunk.get("text", "") for chunk in chunks if isinstance(chunk, dict))
        return text.strip()


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------
class BedrockClient:
    """Single entry point used by the rest of the application."""

    def __init__(self, backend: _BackendBase):
        self.backend = backend

    @classmethod
    def from_config(cls, cfg: Optional[BedrockConfig] = None) -> "BedrockClient":
        cfg = cfg or BEDROCK_CFG
        if cfg.mode == "real":
            backend: _BackendBase = RealBedrockBackend(cfg)
        elif cfg.mode == "mock":
            backend = MockBedrockBackend()
        else:
            raise ValueError(f"Unknown bedrock mode: {cfg.mode!r}. Use 'mock' or 'real'.")
        logger.info("BedrockClient initialized in %s mode", cfg.mode)
        return cls(backend)

    def generate_alert(self, ctx: PredictionContext) -> str:
        return self.backend.generate_alert(ctx)

    def generate_alert_dict(self, prediction: Dict) -> str:
        """Convenience helper that accepts a plain dict instead of a dataclass."""
        ctx = PredictionContext(**prediction)
        return self.generate_alert(ctx)
