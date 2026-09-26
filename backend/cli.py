#!/usr/bin/env python
"""Thin CLI for SatQuery-AI: hits an already-running backend's /analyze endpoint so a query can be
tried from a terminal without the frontend. Start the backend first (see CLAUDE.md, "Running the
tests" section neighbours the run instructions), then:

    python cli.py --image scene.png --query "What is here?"
    python cli.py --image before.tif --image2 after.tif --modality sar        # change detection
    python cli.py --image optical.png --image2 sar.png --modality2 sar         # fusion
    python cli.py --image scene.png --adapter mining                          # feature scan

This is a plain HTTP client, not a second copy of the pipeline: one image means single-image VQA (or
a scan, with --adapter), two images of the same declared modality means change detection, two images
of different modalities means fusion - exactly main_api.py's own routing rule. Prints the answer, the
execution trace (task, routing, stages, warnings) and where the evidence image / PDF report can be
downloaded from the server.
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("This CLI needs httpx (already a test dependency): pip install httpx")

DEFAULT_BACKEND = "http://127.0.0.1:8000"
DEFAULT_QUERY = "Describe the scene."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="Image A: the single image, or the Before/optical image of a pair")
    p.add_argument("--image2", help="Image B: the After/SAR image of a pair (omit for single-image analysis)")
    p.add_argument("--query", help=f"The question to ask (default for a scan: extract features; otherwise: {DEFAULT_QUERY!r})")
    p.add_argument("--adapter", default="general", choices=["general", "mining", "deforestation", "agriculture"],
                    help="A feature-scan adapter (single image only); 'general' is a free-text question")
    p.add_argument("--modality", default="optical", choices=["optical", "sar"], help="Sensor type of --image")
    p.add_argument("--modality2", default="optical", choices=["optical", "sar"], help="Sensor type of --image2")
    p.add_argument("--backend", default=DEFAULT_BACKEND, help=f"Backend base URL (default: {DEFAULT_BACKEND})")
    p.add_argument("--token", help="Bearer token of a signed-in account (required unless the backend runs with REQUIRE_SIGN_IN=0); the run is added to its report history")
    p.add_argument("--json", action="store_true", help="Print the raw JSON response instead of a formatted summary")
    p.add_argument("--timeout", type=float, default=180.0, help="Request timeout in seconds (model inference can be slow)")
    return p.parse_args()


def build_form(args):
    query = args.query or ("Extract features." if args.adapter != "general" else DEFAULT_QUERY)
    data = {"query": query, "adapter": args.adapter, "modality_a": args.modality, "chat_history": "[]"}
    files = [("images", (Path(args.image).name, Path(args.image).read_bytes()))]
    if args.image2:
        data["modality_b"] = args.modality2
        files.append(("images", (Path(args.image2).name, Path(args.image2).read_bytes())))
    return data, files


def summarize(result):
    trace = result.get("agent_execution_trace", {})
    lines = [
        "", f"Answer: {result.get('answer', '')}", "",
        f"Task: {trace.get('task')}   Routing: {trace.get('routing_reason')}",
        f"Stages: {' -> '.join(trace.get('nodes_traversed', []))}",
        f"Status: {trace.get('execution_status')}   Validation: {trace.get('validation_status')}",
    ]
    telemetry = trace.get("telemetry") or {}
    if telemetry.get("confidence") is not None:
        lines.append(f"Confidence: {telemetry['confidence']} (mean token probability; not a correctness score)")
    for w in trace.get("warnings", []) or []:
        lines.append(f"WARNING: {w}")
    if result.get("visual_evidence_url"):
        lines.append(f"Evidence image: {result['visual_evidence_url']}")
    if result.get("report_download_url"):
        lines.append(f"PDF report: {result['report_download_url']}")
    elif result.get("report_error"):
        lines.append(f"Report unavailable: {result['report_error']}")
    scan = result.get("scan")
    if scan and scan.get("headline"):
        lines.append(f"Scan headline: {scan['headline']}")
    return "\n".join(lines)


def main():
    args = parse_args()
    for path in (args.image, args.image2):
        if path and not Path(path).is_file():
            sys.exit(f"No such file: {path}")

    data, files = build_form(args)
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else None

    print(f"POSTing to {args.backend}/analyze ...", file=sys.stderr)
    try:
        response = httpx.post(f"{args.backend}/analyze", data=data, files=files, headers=headers, timeout=args.timeout)
    except httpx.ConnectError:
        sys.exit(f"Could not reach {args.backend}. Is the backend running? (uvicorn main_api:app, from backend/)")
    except httpx.TimeoutException:
        sys.exit(f"Timed out after {args.timeout}s. Model inference can be slow on CPU; try --timeout with a larger value.")

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        sys.exit(f"HTTP {response.status_code}: {detail}")

    result = response.json()
    print(json.dumps(result, indent=2) if args.json else summarize(result))


if __name__ == "__main__":
    main()
