"""cli.py's pure logic (form building, output formatting) - no network, no subprocess. httpx is
already a test dependency (FastAPI's TestClient uses it), so cli.py imports cleanly here too."""
import argparse
import sys
from pathlib import Path

import cli


def _args(**overrides):
    base = dict(image="a.png", image2=None, query=None, adapter="general", modality="optical",
                modality2="optical", backend=cli.DEFAULT_BACKEND, token=None, json=False, timeout=180.0)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_form_single_image_defaults_the_query(tmp_path):
    img = tmp_path / "scene.png"
    img.write_bytes(b"pretend-png-bytes")
    data, files = cli.build_form(_args(image=str(img)))
    assert data == {"query": cli.DEFAULT_QUERY, "adapter": "general", "modality_a": "optical", "chat_history": "[]"}
    assert len(files) == 1
    assert files[0] == ("images", ("scene.png", b"pretend-png-bytes"))


def test_build_form_scan_defaults_to_extract_features(tmp_path):
    img = tmp_path / "scene.png"
    img.write_bytes(b"x")
    data, _ = cli.build_form(_args(image=str(img), adapter="mining"))
    assert data["query"] == "Extract features."


def test_build_form_pair_adds_the_second_image_and_modality_b(tmp_path):
    a, b = tmp_path / "before.tif", tmp_path / "after.tif"
    a.write_bytes(b"a"); b.write_bytes(b"b")
    data, files = cli.build_form(_args(image=str(a), image2=str(b), modality="sar", modality2="sar", query="what changed?"))
    assert data["query"] == "what changed?"
    assert data["modality_a"] == "sar" and data["modality_b"] == "sar"
    assert [f[1][0] for f in files] == ["before.tif", "after.tif"]


def test_summarize_reports_answer_trace_warnings_and_links():
    result = {
        "answer": "It is a forest.",
        "agent_execution_trace": {
            "task": "SINGLE_IMAGE_VQA", "routing_reason": "1 image uploaded",
            "nodes_traversed": ["RuleBasedRouter", "SingleImageSpecialist"],
            "execution_status": "completed", "validation_status": "not performed",
            "warnings": ["Band order is not declared; assumed R, G, B."],
        },
        "visual_evidence_url": "/reports/abc/evidence.png",
        "report_download_url": "/reports/abc/report.pdf",
        "report_error": None,
    }
    out = cli.summarize(result)
    assert "It is a forest." in out
    assert "RuleBasedRouter -> SingleImageSpecialist" in out
    assert "WARNING: Band order is not declared; assumed R, G, B." in out
    assert "/reports/abc/evidence.png" in out
    assert "/reports/abc/report.pdf" in out


def test_summarize_shows_the_report_error_when_there_is_no_pdf():
    result = {
        "answer": "ok", "agent_execution_trace": {"task": "SINGLE_IMAGE_VQA", "routing_reason": "x", "nodes_traversed": []},
        "report_download_url": None, "report_error": "PDF generation failed: ValueError",
    }
    out = cli.summarize(result)
    assert "Report unavailable: PDF generation failed: ValueError" in out


def test_summarize_prints_the_confidence_only_when_the_trace_has_one():
    trace = {"task": "SINGLE_IMAGE_VQA", "routing_reason": "x", "nodes_traversed": []}
    with_score = cli.summarize({"answer": "ok", "agent_execution_trace": {**trace, "telemetry": {"confidence": 0.812}}})
    assert "Confidence: 0.812 (mean token probability; not a correctness score)" in with_score
    without = cli.summarize({"answer": "ok", "agent_execution_trace": {**trace, "telemetry": {"confidence": None}}})
    assert "Confidence" not in without


def test_main_exits_readably_when_the_image_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cli.py", "--image", "no-such-file.png"])
    try:
        cli.main()
        raised = False
    except SystemExit as e:
        raised = True
        assert "No such file" in str(e.args[0])
    assert raised
