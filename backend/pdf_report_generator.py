import io
import json
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as _canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _safe(text: Any) -> str:
    """Make arbitrary text safe for ReportLab's Paragraph.

    Paragraph parses XML-like markup, so a raw '<', '>' or '&' in model output (e.g. "cover < 10%")
    raises a parse error. Escape first, THEN turn newlines into <br/> (otherwise the tag itself
    would be escaped) so multi-line answers keep their structure.
    """
    text = "" if text is None else str(text)
    return escape(text).replace("\r\n", "\n").replace("\n", "<br/>")


class _NumberedCanvas(_canvas.Canvas):
    """Draws a footer with the real page number and page count on every page. The total is only
    known once the whole document is laid out, so pages are held back until save()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_pages = []

    def showPage(self):
        self._saved_pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_pages)
        for state in self._saved_pages:
            self.__dict__.update(state)
            self.setFont("Helvetica", 7)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawCentredString(
                letter[0] / 2, 18,
                f"Confidential Audit Record \u2022 Automated Agent Execution Pipeline \u2022 Page {self._pageNumber} of {total}",
            )
            super().showPage()
        super().save()


def _report_status(trace: Dict[str, Any]) -> str:
    """Status shown in the header, taken from what the pipeline reported. Nothing here checks the
    answer for correctness, so the word 'verified' is never used."""
    status = str(trace.get("execution_status") or "not reported").upper()
    if trace.get("warnings"):
        status += " WITH WARNINGS"
    return status


def generate_pdf_report(
    query: str,
    answer: str,
    agent_execution_trace: Dict[str, Any],
    image_source: Optional[Any] = None,
    output_path: Optional[str] = None,
    chat_history: Optional[list] = None, # <-- Added chat_history parameter
) -> io.BytesIO:
    buffer = io.BytesIO()
    target = output_path if output_path else buffer

    doc = SimpleDocTemplate(
        target,
        pagesize=letter,
        leftMargin=28, rightMargin=28, topMargin=28, bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor("#0F172A")
    accent_color = colors.HexColor("#2563EB")
    card_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#E2E8F0")
    text_muted = colors.HexColor("#64748B")

    styles.add(ParagraphStyle("ReportTitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=primary_color))
    styles.add(ParagraphStyle("ReportMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=10, textColor=text_muted, alignment=2))
    styles.add(ParagraphStyle("SectionHeading", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=primary_color, spaceAfter=4))
    styles.add(ParagraphStyle("BodyDark", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#1E293B")))
    styles.add(ParagraphStyle("UserMsg", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=accent_color))
    styles.add(ParagraphStyle("AssistantMsg", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=11, textColor=primary_color))
    styles.add(ParagraphStyle("LabelText", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=text_muted))
    styles.add(ParagraphStyle("TraceCode", parent=styles["Normal"], fontName="Courier", fontSize=7, leading=8.5, textColor=colors.HexColor("#0F172A")))

    story = []

    # --- Header Block ---
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header_data = [[
        Paragraph("<b>SATQUERY AI SYSTEM</b><br/><font size=9 color='#2563EB'>Audit & Inspection Report</font>", styles["ReportTitle"]),
        Paragraph(f"Generated: {timestamp}<br/>Status: <b>{_safe(_report_status(agent_execution_trace))}</b>", styles["ReportMeta"]),
    ]]
    header_table = Table(header_data, colWidths=[3.2 * inch, 4.3 * inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1, color=accent_color, spaceBefore=4, spaceAfter=8))

    # --- FULL CHAT CONVERSATION BLOCK ---
    story.append(Paragraph("AUDIT CONVERSATION HISTORY", styles["SectionHeading"]))
    
    summary_data = []
    if chat_history and len(chat_history) > 0:
        for msg in chat_history:
            role = msg.get("role", "user").upper()
            text = _safe(msg.get("text", msg.get("content", "")))
            if role == "SYSTEM":
                # Pipeline/status entries (e.g. "Initiating mining scan...") are not something the user said
                style, prefix = styles["LabelText"], "<b>SYSTEM:</b> "
            elif role == "USER":
                style, prefix = styles["UserMsg"], "<b>USER:</b> "
            else:
                style, prefix = styles["AssistantMsg"], "<b>AI ASSISTANT:</b> "
            summary_data.append([Paragraph(f"{prefix}{text}", style)])
    else:
        # Fallback to single prompt/answer if no history is provided
        summary_data = [
            [Paragraph("USER QUERY", styles["LabelText"])],
            [Paragraph(f"<i>\"{_safe(query)}\"</i>", styles["BodyDark"])],
            [Spacer(1, 4)],
            [Paragraph("SYNTHESIZED ANSWER / INFERENCE", styles["LabelText"])],
            [Paragraph(_safe(answer), styles["BodyDark"])],
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

    left_elements = []
    if image_source:
        try:
            img = Image(image_source, width=3.6 * inch, height=3.1 * inch, kind="proportional")
            left_elements.append(img)
        except Exception:
            left_elements.append(Paragraph("<i>[Image rendering unavailable]</i>", styles["BodyDark"]))
    else:
        left_elements.append(Paragraph("<i>No visual evidence supplied.</i>", styles["BodyDark"]))

    # Temporal-order statement gets its own paragraph (the JSON dump below is length-truncated)
    trace_for_dump = json.loads(json.dumps(agent_execution_trace))
    temporal = (trace_for_dump.get("telemetry") or {}).pop("temporal_order", None)
    # Supporting data of a comparison (shifts, scan scores, agreement) is printed as its own section
    # below instead of as raw JSON; the on-screen answer stays short.
    comparison_details = (trace_for_dump.get("telemetry") or {}).pop("comparison_details", None)
    summary_lines = []
    if trace_for_dump.get("task"):
        summary_lines.append(f"<b>Task:</b> {_safe(trace_for_dump['task'])} ({_safe(trace_for_dump.get('routing', 'rule-based'))} routing: {_safe(trace_for_dump.get('routing_reason', ''))})")
    if trace_for_dump.get("nodes_traversed"):
        summary_lines.append("<b>Stages run:</b> " + _safe(" \u2192 ".join(trace_for_dump["nodes_traversed"])))
    model_used = (trace_for_dump.get("telemetry") or {}).get("model_used")
    if model_used:
        summary_lines.append(f"<b>Model:</b> {_safe(model_used)}")
    if trace_for_dump.get("validation_status"):
        summary_lines.append(f"<b>Validation:</b> {_safe(trace_for_dump['validation_status'])}")
    for w in trace_for_dump.get("warnings") or []:
        summary_lines.append(f"<b>Warning:</b> {_safe(w)}")

    right_elements = [
        Paragraph("<b>Run Summary</b>", styles["LabelText"]),
        Spacer(1, 4),
        *[item for line in summary_lines for item in (Paragraph(line, styles["BodyDark"]), Spacer(1, 3))],
        *([Paragraph("<b>Temporal order:</b> " + _safe(temporal.get("summary", "")), styles["BodyDark"]), Spacer(1, 4)]
          if isinstance(temporal, dict) and temporal.get("summary") else []),
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

    if isinstance(comparison_details, dict) and comparison_details:
        story.append(Paragraph("COMPARISON DETAILS", styles["SectionHeading"]))
        rows = []
        for title, lines in comparison_details.items():
            rows.append([Paragraph(_safe(str(title).upper()), styles["LabelText"])])
            rows.extend([Paragraph(_safe(line), styles["BodyDark"])] for line in (lines or []))
            rows.append([Spacer(1, 3)])
        details_table = Table(rows, colWidths=[7.5 * inch])
        details_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), card_bg),
            ("BOX", (0, 0), (-1, -1), 0.75, border_color),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(details_table)
        story.append(Spacer(1, 8))

    # Full trace, never truncated: audit data must not be silently cut. Preformatted text flows
    # across pages, so a long trace just makes the report longer.
    story.append(Paragraph("FULL EXECUTION TRACE", styles["SectionHeading"]))
    trace_lines = []
    for line in json.dumps(trace_for_dump, indent=2, ensure_ascii=False).split("\n"):
        indent = len(line) - len(line.lstrip(" "))
        trace_lines.extend(textwrap.wrap(line, width=118, subsequent_indent=" " * (indent + 4),
                                         drop_whitespace=False, replace_whitespace=False) or [""])
    story.append(Preformatted("\n".join(trace_lines), styles["TraceCode"]))

    doc.build(story, canvasmaker=_NumberedCanvas)
    
    if not output_path:
        buffer.seek(0)
        return buffer
    return buffer