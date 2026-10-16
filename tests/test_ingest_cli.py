"""Regression tests for the material-ingestion CLI embedding mode."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from course_assistant.ingest.embeddings import HashEmbedder


def _load_ingest_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ingest.py"
    spec = importlib.util.spec_from_file_location("course_ingest_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(tmp_path, *, offline=False):
    return argparse.Namespace(
        vector_dir=str(tmp_path / "vectors"),
        real_embeddings=False,
        offline_embeddings=offline,
    )


def test_build_index_uses_class_embeddings_by_default_when_configured(tmp_path, monkeypatch):
    ingest = _load_ingest_module()
    configured = SimpleNamespace(
        keys_present=True,
        text_embed=object(),
        visual_embed=object(),
    )
    monkeypatch.setattr("course_assistant.config.load_config", lambda: configured)

    index = ingest.build_index(_args(tmp_path))

    assert not isinstance(index.text_embedder.__self__, HashEmbedder)
    assert not isinstance(index.image_embedder.__self__, HashEmbedder)


def test_build_index_offline_flag_explicitly_uses_hash_embeddings(tmp_path, monkeypatch):
    ingest = _load_ingest_module()
    configured = SimpleNamespace(
        keys_present=True,
        text_embed=object(),
        visual_embed=object(),
    )
    monkeypatch.setattr("course_assistant.config.load_config", lambda: configured)

    index = ingest.build_index(_args(tmp_path, offline=True))

    assert isinstance(index.text_embedder.__self__, HashEmbedder)
    assert isinstance(index.image_embedder.__self__, HashEmbedder)
