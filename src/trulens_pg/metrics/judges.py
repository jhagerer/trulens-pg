"""Judge (LLM provider) configuration: OpenAI, Azure OpenAI, LiteLLM."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from trulens_pg.config import Settings
from trulens_pg.errors import ValidationError

PROVIDERS = ("openai", "azure_openai", "litellm")


@dataclass(frozen=True)
class JudgeConfig:
    """Judge settings stored with each metric version and recorded with each result.

    Attributes:
        provider: ``openai``, ``azure_openai`` or ``litellm``.
        model: Model name (Azure: deployment name).
        temperature: Sampling temperature.
        extra: Further keyword arguments for the provider class (no secrets:
            API keys come from the provider's own environment variables).
    """

    provider: str
    model: str
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValidationError(
                f"Judge provider must be one of {PROVIDERS}, got {self.provider!r}."
            )
        if not self.model:
            raise ValidationError("Judge model must not be empty.")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> JudgeConfig:
        known = {"provider", "model", "temperature"}
        return cls(
            provider=str(data.get("provider", "")).lower(),
            model=str(data.get("model", "")),
            temperature=float(data.get("temperature", 0.0) or 0.0),
            extra={k: v for k, v in data.items() if k not in known and k != "extra"}
            | dict(data.get("extra") or {}),
        )

    @classmethod
    def from_env(cls, settings: Settings | None = None) -> JudgeConfig | None:
        """Read ``TRULENS_PG_JUDGE_PROVIDER``, ``_MODEL`` and ``_TEMPERATURE``."""
        judge = (settings or Settings.from_env()).judge
        if not judge.get("provider") or not judge.get("model"):
            return None
        return cls.from_mapping(
            {k: v for k, v in judge.items() if k in {"provider", "model", "temperature"}}
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.extra:
            data["extra"] = dict(self.extra)
        return data

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}@t={self.temperature:g}"


def build_judge(config: JudgeConfig) -> Any:
    """Create the TruLens provider object for ``config``."""
    try:
        if config.provider == "openai":
            from trulens.providers.openai import OpenAI

            return OpenAI(model_engine=config.model, **config.extra)
        if config.provider == "azure_openai":
            from trulens.providers.openai import AzureOpenAI

            return AzureOpenAI(deployment_name=config.model, **config.extra)
        from trulens.providers.litellm import LiteLLM

        return LiteLLM(model_engine=config.model, **config.extra)
    except ImportError as exc:  # pragma: no cover - optional extras
        extra = "litellm" if config.provider == "litellm" else "openai"
        raise ImportError(f'Install the judge extra: pip install "trulens-pg[{extra}]"') from exc


def judge_info(judge: Any, config: JudgeConfig | None = None) -> dict[str, Any]:
    """Provider, model and temperature of a judge, for recording with results."""
    if config is not None:
        return config.to_dict()
    if judge is None:
        return {}
    model = getattr(judge, "model_engine", None) or getattr(judge, "deployment_name", None)
    return {
        "provider": type(judge).__name__.lower(),
        "model": str(model) if model else None,
        "temperature": 0.0,
    }


def judge_model_label(info: Mapping[str, Any]) -> str | None:
    if not info:
        return None
    return f"{info.get('provider')}:{info.get('model')}"


def resolve_judge(
    judge: Any | None, stored: Mapping[str, Any] | None = None
) -> tuple[Any | None, dict[str, Any]]:
    """Pick the judge for a metric: argument, then stored config, then environment.

    Args:
        judge: A TruLens provider object, a `JudgeConfig`, a mapping, or ``None``.
        stored: The judge stored with the metric version.

    Returns:
        ``(provider object or None, judge info dict)``.
    """
    if isinstance(judge, JudgeConfig):
        return build_judge(judge), judge.to_dict()
    if isinstance(judge, Mapping):
        cfg = JudgeConfig.from_mapping(judge)
        return build_judge(cfg), cfg.to_dict()
    if judge is not None:
        info = judge_info(judge)
        if stored and stored.get("temperature") is not None:
            info["temperature"] = float(stored["temperature"])
        return judge, info
    if stored and stored.get("provider") and stored.get("model"):
        cfg = JudgeConfig.from_mapping(stored)
        return build_judge(cfg), cfg.to_dict()
    env_cfg = JudgeConfig.from_env()
    if env_cfg is not None:
        return build_judge(env_cfg), env_cfg.to_dict()
    return None, {}
