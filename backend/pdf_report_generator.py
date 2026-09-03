import io
import json
from datetime import datetime
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_pdf_report(
    query: str,
    answer: str,
    confidence_score: float,
    agent_execution_trace: Dict[str, Any],
    image_source: Optional[Any] = None,  # file path, bytes, or file-like object
    output_path: Optional[str] = None,
) -> io.BytesIO:
    """
    Generates a 1-page professional audit document.
    Returns an in-memory BytesIO buffer (or writes to output_path if provided).
    """
    buffer = io.BytesIO()
    target = output_path if output_path else buffer

    # Set 0.4 inch (28.8pt) margins to guarantee single-page budget
    doc = SimpleDocTemplate(
        target,
        pagesize=letter,
        leftMargin=28,
        rightMargin=28,
        topMargin=28,
        bottomMargin=28,
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography & Palette
    primary_color = colors.HexColor("#0F172A")    # Slate 900
    accent_color = colors.HexColor("#2563EB")     # Blue 600
    card_bg = colors.HexColor("#F8FAFC")          # Slate 50
    border_color = colors.HexColor("#E2E8F0")     # Slate 200
    text_muted = colors.HexColor("#64748B")       # Slate 500

    styles.add(ParagraphStyle("ReportTitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=primary_color))
    styles.add(ParagraphStyle("ReportMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=10, textColor=text_muted, alignment=2))
    styles.add(ParagraphStyle("SectionHeading", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=primary_color, spaceAfter=4))
    styles.add(ParagraphStyle("BodyDark", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#1E293B")))
    styles.add(ParagraphStyle("LabelText", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=text_muted))
    styles.add(ParagraphStyle("TraceCode", parent=styles["Normal"], fontName="Courier", fontSize=7, leading=8.5, textColor=colors.HexColor("#0F172A")))

    story = []

    # --- Header Block ---
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    header_data = [
        [
            Paragraph("<b>AGENTIC AI SYSTEM</b><br/><font size=9 color='#2563EB'>Audit & Inspection Report</font>", styles["ReportTitle"]),
            Paragraph(f"Generated: {timestamp}<br/>Confidence: <b>{confidence_score * 100:.1f}%</b><br/>Status: <b>VERIFIED</b>", styles["ReportMeta"]),
        ]
    ]
    header_table = Table(header_data, colWidths=[3.2 * inch, 4.3 * inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1, color=accent_color, spaceBefore=4, spaceAfter=8))

    # --- Query & Summary Section ---
    summary_data = [
        [Paragraph("USER QUERY", styles["LabelText"])],
        [Paragraph(f"<i>\"{query}\"</i>", styles["BodyDark"])],
        [Spacer(1, 4)],
        [Paragraph("SYNTHESIZED ANSWER / INFERENCE", styles["LabelText"])],
        [Paragraph(answer, styles["BodyDark"])],
    ]
    summary_table = Table(summary_data, colWidths=[7.5 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), card_bg),
        ("BOX", (0, 0), (-1, -1), 0.75, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    # --- Two-Column Layout: Visual Evidence & Execution Trace ---
    story.append(Paragraph("INSPECTION EVIDENCE & EXECUTION TELEMETRY", styles["SectionHeading"]))

    # Left Column: Visual Evidence
    left_elements = []
    if image_source:
        try:
            # Constrain image within bounding area (3.6in width x 3.1in height)
            img = Image(image_source, width=3.6 * inch, height=3.1 * inch, kind="proportional")
            left_elements.append(img)
        except Exception:
            left_elements.append(Paragraph("<i>[Image rendering unavailable]</i>", styles["BodyDark"]))
    else:
        left_elements.append(Paragraph("<i>No visual evidence supplied.</i>", styles["BodyDark"]))

    # Right Column: Formatted Trace
    trace_pretty = json.dumps(agent_execution_trace, indent=2)
    # Truncate if raw string exceeds safe printable line count
    if len(trace_pretty) > 1200:
        trace_pretty = trace_pretty[:1197] + "..."
    
    trace_escaped = trace_pretty.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>").replace(" ", "&nbsp;")
    right_elements = [
        Paragraph("<b>Execution Trace (Telemetry Log)</b>", styles["LabelText"]),
        Spacer(1, 4),
        Paragraph(trace_escaped, styles["TraceCode"]),
    ]

    col_data = [[left_elements, right_elements]]
    two_col_table = Table(col_data, colWidths=[3.7 * inch, 3.8 * inch])
    two_col_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, 0), card_bg),
        ("BACKGROUND", (1, 0), (1, 0), card_bg),
        ("BOX", (0, 0), (0, 0), 0.75, border_color),
        ("BOX", (1, 0), (1, 0), 0.75, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(two_col_table)
    story.append(Spacer(1, 8))

    # --- Footer Block ---
    footer_text = Paragraph(
        "Confidential Audit Record • Automated Agent Execution Pipeline • Page 1 of 1",
        styles["ReportMeta"],
    )
    story.append(footer_text)

    # Build PDF
    doc.build(story)
    
    if not output_path:
        buffer.seek(0)
        return buffer
    return buffer