"""End-to-end tests for the ingest CLI: add (dedupe), list, remove cascade."""

import subprocess
import sys
from pathlib import Path

import pytest

from test_formats import make_pptx

REPO_ROOT = Path(__file__).resolve().parents[1]
INGEST = REPO_ROOT / "scripts" / "ingest.py"


def run(*args, **kw):
    return subprocess.run(
        [sys.executable, str(INGEST), *args],
        capture_output=True, text=True, **kw,
    )


@pytest.fixture
def workspace(tmp_path):
    materials = tmp_path / "materials"
    materials.mkdir()
    deck = make_pptx(materials / "week9.pptx", slides=2)
    return {
        "materials": materials,
        "deck": deck,
        "vector_dir": str(tmp_path / "vec"),
        "images_root": str(tmp_path / "images"),
        "registry": str(tmp_path / "registry.json"),
    }


def common_flags(ws):
    return ["--vector-dir", ws["vector_dir"], "--images-root", ws["images_root"],
            "--registry", ws["registry"]]


def test_add_twice_is_deduped_and_remove_cascades(workspace):
    ws = workspace
    r = run("ingest", str(ws["deck"]), *common_flags(ws))
    assert r.returncode == 0, r.stderr
    assert "2 pages, 2 chunks, 2 visual vectors" in r.stdout

    # second load: no-op
    r2 = run("ingest", str(ws["deck"]), *common_flags(ws))
    assert "duplicate" in r2.stdout

    # registry + index populated
    rl = run("list", *common_flags(ws))
    assert "week9.pptx" in rl.stdout

    # remove by filename -> everything gone
    rr = run("remove", "week9.pptx", *common_flags(ws))
    assert rr.returncode == 0, rr.stderr
    assert not Path(ws["registry"]).exists() or '"doc_id"' not in Path(ws["registry"]).read_text().replace(" ", "")
    assert list(Path(ws["images_root"]).iterdir()) == []  # images dir cascade


def test_unsupported_file_reports_format_and_continues(workspace):
    ws = workspace
    bad = ws["materials"] / "notes.xyz"
    bad.write_text("nope")
    r = run("ingest", str(ws["deck"]), str(bad), *common_flags(ws))
    assert r.returncode == 0
    assert ".xyz" in r.stdout and "We accept" in r.stdout  # format message
    assert "week9.pptx" in [e["filename"] for e in __import__("json").loads(Path(ws["registry"]).read_text() or "{}").values()]


def test_remove_unknown_target_reports(workspace):
    ws = workspace
    r = run("remove", "nope.pptx", *common_flags(ws))
    assert r.returncode == 1
    assert "no registered document" in r.stdout
