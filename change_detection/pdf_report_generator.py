"""
pdf_report_generator.py
=========================
Member 6 (part 2): Auditable PDF report generation.

Generates a clean, professional 1-page audit document from:
  - the user query
  - the text answer
  - the agent execution trace (JSON)
  - a visual evidence image (mask / bounding boxes / overlay)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import cv2


def generate_pdf_report(
    output_path: str,
    query: str,
    answer: str,
    execution_trace: dict,
    evidence_image: np.ndarray | None = None,
    confidence_score: float | None = None,
) -> str:
    """
    Args:
        output_path: where to write the .pdf.
        query: original user query.
        answer: text answer from the pipeline.
        execution_trace: the agent_execution_trace dict (from agent_controller.py).
        evidence_image: optional HxWx3 uint8 RGB array (mask/boxes/overlay) to embed.
        confidence_score: optional float 0-1.

    Returns:
        output_path (for chaining / API response).
    """
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontSize=18, spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=16,
    )
    heading_style = ParagraphStyle(
        "HeadingCustom", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6,
    )
    body_style = styles["BodyText"]

    story = []

    story.append(Paragraph("SATQUERY AI — Analysis Audit Report", title_style))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"Generated: {timestamp}", subtitle_style))

    story.append(Paragraph("User Query", heading_style))
    story.append(Paragraph(query, body_style))

    story.append(Paragraph("Answer", heading_style))
    story.append(Paragraph(answer, body_style))

    if confidence_score is not None:
        story.append(Paragraph("Confidence Score", heading_style))
        story.append(Paragraph(f"{confidence_score:.2%}", body_style))

    if evidence_image is not None:
        story.append(Paragraph("Visual Evidence", heading_style))
        tmp_img_path = output_path.replace(".pdf", "_evidence.png")
        cv2.imwrite(tmp_img_path, cv2.cvtColor(evidence_image, cv2.COLOR_RGB2BGR))
        story.append(RLImage(tmp_img_path, width=4.5 * inch, height=4.5 * inch))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Auditable Execution Trace", heading_style))
    trace_rows = [["Field", "Value"]]
    for k, v in execution_trace.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, default=str)
        trace_rows.append([str(k), str(v)])

    table = Table(trace_rows, colWidths=[1.8 * inch, 4.2 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
    ]))
    story.append(table)

    doc.build(story)
    return output_path


# --------------------------------------------------------------------------- #
# Standalone smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mock_evidence = np.full((300, 300, 3), 180, dtype=np.uint8)
    cv2.rectangle(mock_evidence, (60, 60), (240, 240), (255, 0, 0), thickness=4)

    mock_trace = {
        "task_selected": "CHANGE_DETECTION",
        "models_used": ["OpenCV-SSIM", "Qwen2-VL (mock)"],
        "inputs_validated": {"query_provided": True, "image_count_sufficient": True},
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    path = generate_pdf_report(
        output_path="mock_audit_report.pdf",
        query="What changed between these two satellite images?",
        answer="14.35% of the area changed, consistent with new construction.",
        execution_trace=mock_trace,
        evidence_image=mock_evidence,
        confidence_score=0.87,
    )
    print("PDF generated at:", path)
