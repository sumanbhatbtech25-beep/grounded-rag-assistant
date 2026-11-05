"""Export a conversation (questions, cited answers, sources) as Markdown or PDF."""
from __future__ import annotations

import io
import re
from datetime import datetime
from xml.sax.saxutils import escape

from rag.pipeline import Answer

STATUS_LABEL = {
    "answered": "Grounded answer",
    "refused": "Not found in documents",
    "no_context": "No relevant source",
    "unverified": "Withheld (no valid citation)",
    "error": "Error",
}


def _span(c) -> str:
    hl = c.highlight_lines
    return f"line {hl[0]}" if len(hl) == 1 else f"lines {hl[0]}-{hl[-1]}"


def _cited_text(c) -> list[str]:
    hl = set(c.highlight_lines)
    return [t for j, t in enumerate(c.lines) if c.line_start + j in hl]


def to_markdown(history: list[tuple[str, Answer]], title: str = "CiteSight conversation") -> str:
    out = [f"# {title}", "", f"_Exported {datetime.now():%d %b %Y, %H:%M}_", ""]
    for n, (q, ans) in enumerate(history, start=1):
        out += [f"## Q{n}. {q}", "", ans.answer, ""]
        m = ans.metrics
        out.append(f"*{STATUS_LABEL.get(ans.status, ans.status)} · {m.get('total_ms', 0) / 1000:.2f} s"
                   + (f" · grounding {m.get('grounding', 0):.0%}" if ans.status == "answered" else "")
                   + f" · model {m.get('model', '')}*")
        out.append("")
        if ans.citations:
            out.append("**Sources**")
            out.append("")
            for c in ans.citations:
                out.append(f"- **[S{c.sid}]** {c.source}, page {c.page}, {_span(c)}")
                for t in _cited_text(c):
                    out.append(f"  > {t}")
            out.append("")
        out.append("---")
        out.append("")
    return "\n".join(out)


def to_pdf(history: list[tuple[str, Answer]], title: str = "CiteSight conversation") -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

    navy, amber, muted = colors.HexColor("#1F4B7A"), colors.HexColor("#E0A526"), colors.HexColor("#6B7280")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Times-Bold", fontSize=20, textColor=navy,
                        alignment=0, spaceAfter=2)
    meta = ParagraphStyle("meta", parent=ss["Normal"], fontSize=8.5, textColor=muted, spaceAfter=10)
    q_style = ParagraphStyle("q", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=11.5,
                             textColor=colors.HexColor("#1B2430"), spaceBefore=6, spaceAfter=4)
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=10.5, leading=15)
    src = ParagraphStyle("src", parent=ss["Normal"], fontSize=9, leading=12, leftIndent=8, spaceBefore=3)
    quote = ParagraphStyle("quote", parent=ss["Normal"], fontName="Helvetica-Oblique", fontSize=9,
                           leading=12, leftIndent=18, textColor=colors.HexColor("#3C4655"),
                           backColor=colors.HexColor("#FCF3D9"), borderPadding=(2, 4, 2, 4), spaceBefore=2)

    def cite_tags(text: str) -> str:
        return re.sub(r"\[S(\d+)\]", r'<font color="#1F4B7A"><b>[S\1]</b></font>', escape(text)).replace("\n", "<br/>")

    story = [Paragraph(escape(title), h1),
             Paragraph(f"Exported {datetime.now():%d %b %Y, %H:%M} · {len(history)} question(s) · "
                       "every answer cited to page and line", meta),
             HRFlowable(width="100%", color=amber, thickness=1.5, spaceAfter=8)]
    for n, (q, ans) in enumerate(history, start=1):
        m = ans.metrics
        story.append(Paragraph(f"Q{n}. {escape(q)}", q_style))
        story.append(Paragraph(cite_tags(ans.answer), body))
        story.append(Paragraph(
            f"{STATUS_LABEL.get(ans.status, ans.status)} · {m.get('total_ms', 0) / 1000:.2f} s"
            + (f" · grounding {m.get('grounding', 0):.0%}" if ans.status == "answered" else ""), meta))
        for c in ans.citations:
            story.append(Paragraph(f'<font color="#1F4B7A"><b>[S{c.sid}]</b></font> '
                                   f"{escape(c.source)} — page {c.page}, {_span(c)}", src))
            for t in _cited_text(c):
                story.append(Paragraph(escape(t), quote))
        story += [Spacer(1, 6), HRFlowable(width="100%", color=colors.HexColor("#E0DCD1"), thickness=0.6),
                  Spacer(1, 4)]

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title=title, author="CiteSight",
                      leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm
                      ).build(story)
    return buf.getvalue()
