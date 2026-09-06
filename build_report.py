"""组装 REPORT-zh.md: 按章节顺序合并各中文草稿。

纯机械合并 + 标题规范化(去掉工作草稿标记)。不改动正文内容。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SOURCES = [
    ("DRAFT-ch1-intro-abstract.md", "ch1"),
    ("DRAFT-ch2-architecture.md", "ch2"),
    ("DRAFT-ch3-method.md", "ch3"),
    ("PAPER-zh-DRAFT.md", "ch4"),
    ("DRAFT-ch5-boundaries.md", "ch5"),
    ("DRAFT-ch6-discussion.md", "ch6"),
    ("DRAFT-ch7-limitations.md", "ch7"),
]


def norm_ch1(text: str) -> str:
    lines = text.splitlines()
    out = []
    skip_tail = False
    for line in lines:
        if line.startswith("# 摘要与第 1 章"):
            continue
        if line.strip() == "## 正文顺序说明":
            skip_tail = True
        if skip_tail:
            continue
        if line.strip() == "## 摘要":
            out.append("# 摘要")
            continue
        if line.strip() == "## 1.1 背景与问题":
            out.append("# 1 引言")
        out.append(line)
    return "\n".join(out).strip()


def norm_chapter(text: str, num: str, title: str) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        if line.startswith("# 第") and "章" in line and "（中文草稿）" in line:
            out.append(f"# {num} {title}")
            continue
        if line.startswith("本章状态："):
            continue
        out.append(line)
    return "\n".join(out).strip()


def norm_ch4(text: str) -> str:
    start = text.find("写作规则：")
    if start != -1:
        sep = text.find("---", start)
        if sep != -1:
            text = text[:start] + text[sep + 3 :]
    lines = text.splitlines()
    out = []
    first = True
    for line in lines:
        if first and line.startswith("#"):
            out.append("# 4 结果")
            first = False
            continue
        out.append(line)
    return "\n".join(out).strip()


def main() -> None:
    parts = [
        "# Synapse-Net：双原语联想基底 —— 开源研究报告（中文草稿）",
        "",
        "版本：中文草稿（对应 v0.9.0 代码快照）　日期：2026-09-06",
        "章节来源：DRAFT-ch1..ch7 与 Results 4.1–4.8；所有结论见 CLAIMS-REGISTRY.md。",
        "",
    ]
    for fname, kind in SOURCES:
        text = (ROOT / fname).read_text(encoding="utf-8")
        if kind == "ch1":
            body = norm_ch1(text)
        elif kind == "ch2":
            body = norm_chapter(text, "2", "系统架构与数学定义")
        elif kind == "ch3":
            body = norm_chapter(text, "3", "实验方法")
        elif kind == "ch4":
            body = norm_ch4(text)
        elif kind == "ch5":
            body = norm_chapter(text, "5", "边界与负结果")
        elif kind == "ch6":
            body = norm_chapter(text, "6", "讨论")
        else:
            body = norm_chapter(text, "7", "局限性")
        parts.append(body)
        parts.append("")
        parts.append("---")
        parts.append("")
    (ROOT / "REPORT-zh.md").write_text("\n".join(parts), encoding="utf-8")
    print("REPORT-zh.md assembled")


if __name__ == "__main__":
    main()
