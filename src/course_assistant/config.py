"""Configuration for the class service endpoints (dobolyi.com).

Values come from environment variables, optionally seeded from a local
``.env`` file (python-dotenv). API keys NEVER appear in any repr, log
message, error, or module output — they only flow into outbound request
headers inside ``clients``.

Security rules (assignment requirement):
- Real keys live ONLY in a local, gitignored ``.env``.
- Committed examples (``.env.example``) use dummy values.
- Never log or print key material.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

from dotenv import load_dotenv

load_dotenv()  # reads .env from CWD upward; no-op when absent


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


@dataclass(frozen=True)
class ServiceConfig:
    """Connection settings for one class service. The API key is secret."""

    name: str
    base_url: str
    api_key: str
    model: str

    def __repr__(self) -> str:  # redacted by design — never prints the key
        key = "***" if self.api_key else "(empty)"
        return (
            f"ServiceConfig(name={self.name!r}, base_url={self.base_url!r}, "
            f"api_key={key!r}, model={self.model!r})"
        )


# (env prefix, config attribute, human name) — role map from class-endpoints.txt
_SERVICES: tuple[tuple[str, str, str], ...] = (
    ("CLASS_VISION_LLM", "vision_llm", "Vision LLM"),
    ("CLASS_TEXT_EMBED", "text_embed", "Text embeddings"),
    ("CLASS_VISUAL_EMBED", "visual_embed", "Visual embeddings"),
    ("CLASS_RERANK", "rerank", "Multimodal reranking"),
    ("CLASS_PARSE", "parse", "Document parsing"),
)


@dataclass(frozen=True)
class Config:
    vision_llm: ServiceConfig
    text_embed: ServiceConfig
    visual_embed: ServiceConfig
    rerank: ServiceConfig
    parse: ServiceConfig

    @property
    def services(self) -> tuple[ServiceConfig, ...]:
        return (self.vision_llm, self.text_embed, self.visual_embed, self.rerank, self.parse)

    @property
    def keys_present(self) -> bool:
        """True when every service has a non-empty API key (integration-ready)."""
        return all(s.api_key for s in self.services)

    def require_keys(self) -> "Config":
        """Raise :class:`ConfigError` when any service lacks an API key.

        Entry points that talk to class services call this so the app fails
        with a clear, key-free message instead of a 401 deep in a request.
        """
        missing = [s.name for s in self.services if not s.api_key]
        if missing:
            raise ConfigError(
                "Class services are not configured for: " + ", ".join(missing)
                + ". Add the real API keys to your local .env (see .env.example)."
            )
        return self

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Config":
        env = dict(os.environ if env is None else env)
        missing = sorted(
            f"{prefix}_BASE_URL" for prefix, _, _ in _SERVICES if not env.get(f"{prefix}_BASE_URL")
        )
        if missing:
            raise ConfigError(
                "Missing class-service configuration: " + ", ".join(missing)
                + ". Copy .env.example to .env and fill in the values (see README)."
            )
        kwargs: dict[str, ServiceConfig] = {}
        for prefix, attr, name in _SERVICES:
            kwargs[attr] = ServiceConfig(
                name=name,
                base_url=env[f"{prefix}_BASE_URL"].rstrip("/"),
                api_key=env.get(f"{prefix}_API_KEY", ""),
                model=env.get(f"{prefix}_MODEL", ""),
            )
        return cls(**kwargs)


def load_config(env: Optional[Mapping[str, str]] = None) -> Config:
    """Build the configuration (use this instead of Config.from_env directly)."""
    return Config.from_env(env)
