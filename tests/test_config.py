"""Tests for config loading — no live services involved."""

import pytest

from course_assistant.config import Config, ConfigError

ENV_OK = {
    "CLASS_VISION_LLM_BASE_URL": "http://dobolyi.com:9001/v1",
    "CLASS_VISION_LLM_API_KEY": "",
    "CLASS_VISION_LLM_MODEL": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
    "CLASS_TEXT_EMBED_BASE_URL": "http://dobolyi.com:9002",
    "CLASS_TEXT_EMBED_API_KEY": "",
    "CLASS_TEXT_EMBED_MODEL": "nvidia/Nemotron-3-Embed-1B-BF16",
    "CLASS_VISUAL_EMBED_BASE_URL": "http://dobolyi.com:9003/v1",
    "CLASS_VISUAL_EMBED_API_KEY": "",
    "CLASS_VISUAL_EMBED_MODEL": "Qwen/Qwen3-VL-Embedding-2B",
    "CLASS_RERANK_BASE_URL": "http://dobolyi.com:9004",
    "CLASS_RERANK_API_KEY": "",
    "CLASS_RERANK_MODEL": "Qwen/Qwen3-VL-Reranker-2B",
    "CLASS_PARSE_BASE_URL": "http://dobolyi.com:9005/v1",
    "CLASS_PARSE_API_KEY": "",
    "CLASS_PARSE_MODEL": "dots.mocr",
}


def test_from_env_builds_all_five_services():
    cfg = Config.from_env(ENV_OK)
    assert len(cfg.services) == 5
    assert cfg.vision_llm.base_url == "http://dobolyi.com:9001/v1"
    assert cfg.text_embed.base_url == "http://dobolyi.com:9002"
    assert cfg.parse.base_url == "http://dobolyi.com:9005/v1"
    assert cfg.parse.model == "dots.mocr"


def test_missing_base_url_is_ok_offline_but_require_keys_raises():
    # Offline mode must boot with no env at all (fresh clone, no .env).
    env = {k: v for k, v in ENV_OK.items() if k != "CLASS_RERANK_BASE_URL"}
    cfg = Config.from_env(env)
    assert cfg.rerank.base_url == ""
    assert not cfg.keys_present
    with pytest.raises(ConfigError, match="Multimodal reranking"):
        cfg.require_keys()


def test_no_env_at_all_is_offline():
    cfg = Config.from_env({})
    assert all(s.base_url == "" and s.api_key == "" for s in cfg.services)
    assert not cfg.keys_present


def test_empty_api_keys_are_allowed_but_keys_present_is_false():
    cfg = Config.from_env(ENV_OK)
    assert not cfg.keys_present


def test_keys_present_when_all_services_have_keys():
    env = dict(ENV_OK)
    for prefix in ("CLASS_VISION_LLM", "CLASS_TEXT_EMBED", "CLASS_VISUAL_EMBED",
                   "CLASS_RERANK", "CLASS_PARSE"):
        env[f"{prefix}_API_KEY"] = "k"
    assert Config.from_env(env).keys_present


def test_repr_never_leaks_api_key():
    env = dict(ENV_OK)
    env["CLASS_VISION_LLM_API_KEY"] = "supersecret-key-123456789"
    cfg = Config.from_env(env)
    assert "supersecret" not in repr(cfg.vision_llm)
    assert "supersecret" not in repr(cfg)
    assert "***" in repr(cfg.vision_llm)


def test_base_url_trailing_slash_stripped():
    env = dict(ENV_OK)
    env["CLASS_TEXT_EMBED_BASE_URL"] = "http://dobolyi.com:9002/"
    assert Config.from_env(env).text_embed.base_url == "http://dobolyi.com:9002"


def test_require_keys_raises_when_missing_and_ok_when_present():
    with pytest.raises(ConfigError, match="Vision LLM"):
        Config.from_env(ENV_OK).require_keys()
    env = dict(ENV_OK)
    for prefix in ("CLASS_VISION_LLM", "CLASS_TEXT_EMBED", "CLASS_VISUAL_EMBED",
                   "CLASS_RERANK", "CLASS_PARSE"):
        env[f"{prefix}_API_KEY"] = "k"
    assert Config.from_env(env).require_keys() is not None
