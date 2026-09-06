"""把 REPORT-zh.md 渲染为 PDF(中文, reportlab)。

覆盖: 标题/章节/小节标题、段落、无序列表、表格、代码块、页脚页码。
渲染前做 PDF 友好替换: Unicode 连字符->ASCII '-'、上标->^。
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "REPORT-zh.md"
OUT_DIR = ROOT / "outputs" / "pdf"
OUT = OUT_DIR / "REPORT-zh.pdf"

# ---------- 字体 ----------
pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))
try:
    pdfmetrics.registerFont(TTFont("SimSun", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))
    BODY_FONT = "SimSun"
except Exception:  # noqa: BLE001
    BODY_FONT = "SimHei"
HEAD_FONT = "SimHei"

# ---------- 样式 ----------
styles = {
    "title": ParagraphStyle("title", fontName=HEAD_FONT, fontSize=20, leading=28, alignment=1, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName=BODY_FONT, fontSize=10, leading=15, alignment=1, textColor=colors.HexColor("#444444")),
    "h1": ParagraphStyle("h1", fontName=HEAD_FONT, fontSize=15, leading=21, spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName=HEAD_FONT, fontSize=12.5, leading=18, spaceBefore=8, spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName=HEAD_FONT, fontSize=11, leading=16, spaceBefore=6, spaceAfter=4),
    "body": ParagraphStyle("body", fontName=BODY_FONT, fontSize=10, leading=16, alignment=TA_LEFT, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName=BODY_FONT, fontSize=10, leading=16, leftIndent=10, spaceAfter=2),
    "code": ParagraphStyle("code", fontName=HEAD_FONT, fontSize=8.5, leading=12, leftIndent=4),
    "cell": ParagraphStyle("cell", fontName=BODY_FONT, fontSize=8.5, leading=12),
    "cellh": ParagraphStyle("cellh", fontName=HEAD_FONT, fontSize=9, leading=12),
}


def clean(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    sup = {"⁻": "^-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
    for k, v in sup.items():
        text = text.replace(k, v)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.strip()


def add_para(flow, text: str, style: str = "body") -> None:
    if text.strip():
        flow.append(Paragraph(clean(text), styles[style]))


def build() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    flow = []
    first_h1 = True
    i = 0
    in_code = False
    code_buf = []
    table_buf = []

    def flush_table() -> None:
        nonlocal table_buf
        if not table_buf:
            return
        rows = []
        for raw in table_buf:
            cells = [c.strip() for c in raw.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            rows.append(cells)
        if rows:
            ncol = max(len(r) for r in rows)
            rows = [r + [""] * (ncol - len(r)) for r in rows]
            avail = A4[0] - 2 * 40 * mm
            cw = max(14 * mm, avail / ncol)
            data = [[Paragraph(c, styles["cellh"] if ri == 0 else styles["cell"]) for c in r] for ri, r in enumerate(rows)]
            t = Table(data, colWidths=[cw] * ncol, repeatRows=1)
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ]))
            flow.append(t)
            flow.append(Spacer(1, 6))
        table_buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                flow.append(Preformatted("\n".join(code_buf), styles["code"]))
                flow.append(Spacer(1, 5))
                code_buf = []
                in_code = False
            else:
                flush_table()
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            flush_table()
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_buf.append(lines[i])
                i += 1
            flush_table()
            continue
        if stripped == "---":
            i += 1
            continue
        if stripped.startswith("# "):
            flush_table()
            if first_h1:
                first_h1 = False
                add_para(flow, stripped[2:], "title")
                flow.append(Spacer(1, 6))
            else:
                flow.append(Paragraph(clean(stripped[2:]), styles["h1"]))
                flow.append(Spacer(1, 4))
            i += 1
            continue
        if stripped.startswith("## "):
            flush_table()
            add_para(flow, stripped[3:], "h2")
            i += 1
            continue
        if stripped.startswith("### "):
            flush_table()
            add_para(flow, stripped[4:], "h3")
            i += 1
            continue
        if re.match(r"^\s*[-*] ", line):
            add_para(flow, re.sub(r"^\s*[-*] ", "", line), "bullet")
            i += 1
            continue
        if re.match(r"^\s*\d+\. ", line):
            add_para(flow, re.sub(r"^\s*\d+\. ", "", line), "bullet")
            i += 1
            continue
        if stripped:
            add_para(flow, stripped)
        i += 1
    flush_table()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("SimHei", 8)
        canvas.drawCentredString(A4[0] / 2, 12 * mm, f"- {doc.page} -")
        canvas.restoreState()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                           leftMargin=40 * mm, rightMargin=40 * mm,
                           topMargin=35 * mm, bottomMargin=35 * mm,
                           title="Synapse-Net 双原语联想基底 研究报告",
                           author="Synapse-Net")
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
