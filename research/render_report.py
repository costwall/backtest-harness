"""
research/render_report.py
=========================
Конвертер отчёта Markdown → PDF (reportlab, без внешних бинарей — работает на
корп-Windows). Переиспользуемо: каждый клиентский отчёт (заполненный
REPORT_TEMPLATE.md) рендерится одной командой в готовый для Fiverr PDF.

Поддержанное подмножество Markdown (ровно то, что в наших отчётах):
  # H1   ## H2   > blockquote   | таблицы |   - списки   **bold** *italic* `code`
  --- горизонтальная линия.  HTML-комментарии <!-- ... --> пропускаются.

Бренд: зелёный акцент (#1e8e3e) под лого «cost wall».

Запуск:  python -m research.render_report research/REPORT_SAMPLE.md out.pdf
"""
from __future__ import annotations

import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

ACCENT = colors.HexColor("#1e8e3e")     # зелёный бренда
DARK = colors.HexColor("#202124")
GREY = colors.HexColor("#5f6368")
LIGHT = colors.HexColor("#f1f3f4")


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", parent=base["Title"], textColor=ACCENT,
                             fontSize=20, spaceAfter=4, leading=24),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], textColor=DARK,
                             fontSize=13, spaceBefore=12, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5,
                               leading=13, alignment=TA_LEFT, textColor=DARK),
        "quote": ParagraphStyle("quote", parent=base["BodyText"], fontSize=9.5,
                                leading=13, textColor=GREY, leftIndent=8,
                                borderColor=ACCENT, borderWidth=0, italic=True),
        "cell": ParagraphStyle("cell", fontSize=8.5, leading=11, textColor=DARK),
        "cellh": ParagraphStyle("cellh", fontSize=8.5, leading=11,
                                textColor=colors.white, fontName="Helvetica-Bold"),
    }


def _inline(text: str) -> str:
    """Markdown inline → reportlab markup (HTML-подмножество)."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"(?<![\*\w])\*(?!\*)(.+?)(?<!\*)\*(?![\*\w])", r"<i>\1</i>", text)
    return text


def _table(rows: list[list[str]], st: dict) -> Table:
    header = [Paragraph(_inline(c), st["cellh"]) for c in rows[0]]
    body = [[Paragraph(_inline(c), st["cell"]) for c in r] for r in rows[1:]]
    t = Table([header] + body, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dadce0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _footer(canvas, doc) -> None:
    """Фирменная подпись внизу каждой страницы (бренд + источник + номер)."""
    canvas.saveState()
    w = A4[0]
    y = 11 * mm
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(0.6)
    canvas.line(18 * mm, y + 4 * mm, w - 18 * mm, y + 4 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, y, "costwall  ·  honest, costs-included strategy validation")
    canvas.drawRightString(
        w - 18 * mm, y, f"github.com/costwall/backtest-harness   ·   p.{doc.page}")
    canvas.restoreState()


def render(md_path: str, pdf_path: str) -> None:
    lines = open(md_path, encoding="utf-8").read().splitlines()
    st = _styles()
    def _is_block(ln: str) -> bool:
        """Строка начинает блочный элемент (НЕ продолжение абзаца)?"""
        s = ln.lstrip()
        return (not ln.strip()
                or s.startswith(("#", ">", "|", "- ", "---", "<!--")))

    flow: list = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.lstrip().startswith("<!--"):           # пропустить HTML-коммент
            while i < n and "-->" not in lines[i]:
                i += 1
            i += 1
        elif line.startswith("# "):
            flow.append(Paragraph(_inline(line[2:]), st["h1"]))
            i += 1
        elif line.startswith("## "):
            flow.append(Paragraph(_inline(line[3:]), st["h2"]))
            i += 1
        elif line.startswith("---"):
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.6, color=ACCENT))
            i += 1
        elif line.startswith("> "):                    # склеить blockquote
            buf = []
            while i < n and lines[i].startswith("> "):
                buf.append(lines[i][2:].strip())
                i += 1
            flow.append(Spacer(1, 2))
            flow.append(Paragraph(_inline(" ".join(buf)), st["quote"]))
        elif line.lstrip().startswith("|"):            # собрать таблицу
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[\s:-]+$", "".join(cells)):   # пропустить |---|
                    rows.append(cells)
                i += 1
            if rows:
                flow.append(Spacer(1, 3))
                flow.append(_table(rows, st))
                flow.append(Spacer(1, 3))
        elif line.lstrip().startswith("- "):           # собрать список (пункт может
            items = []                                 # переноситься на след. строки)
            while i < n and lines[i].lstrip().startswith("- "):
                buf = [lines[i].lstrip()[2:].strip()]
                i += 1
                while i < n and lines[i].strip() and not _is_block(lines[i]):
                    buf.append(lines[i].strip())
                    i += 1
                items.append(ListItem(Paragraph(_inline(" ".join(buf)), st["body"]),
                                      leftIndent=10))
            flow.append(ListFlowable(items, bulletType="bullet", start="•",
                                     bulletColor=ACCENT, bulletFontSize=7))
        elif line.strip():                             # склеить абзац до пустой/блочной
            buf = []
            while i < n and lines[i].strip() and not _is_block(lines[i]):
                buf.append(lines[i].strip())
                i += 1
            flow.append(Paragraph(_inline(" ".join(buf)), st["body"]))
        else:
            flow.append(Spacer(1, 4))
            i += 1

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=20 * mm,
        title="Strategy Edge Report — costwall",
    )
    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m research.render_report <in.md> <out.pdf>")
        sys.exit(1)
    render(sys.argv[1], sys.argv[2])
    print(f"written {sys.argv[2]}")
