"""措辞规范扫描 (会审第二遍)。

检查:
  1) 4.1-4.8 小节齐全;
  2) 每小节包含预期 E 编号;
  3) 证据引用格式 = "证据：E#，outputs/xxx.json", 且 JSON 文件存在;
  4) 术语/数字格式警告(不阻断): 外部 stub 前缀、λ/hop_decay/阈值预设表述、
     小数写法一致性、"联想层"等候选统一词。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DRAFT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "REPORT-zh.md")
if not DRAFT.exists():
    DRAFT = ROOT / "PAPER-zh-DRAFT.md"
OUT = ROOT / "outputs"

EXPECTED = {
    "4.1": ["E1", "E4"],
    "4.2": ["E2", "E3", "E10", "E9", "E14"],
    "4.3": ["E5", "E15", "E16"],
    "4.4": ["E6a", "E6b", "E8", "E11"],
    "4.5": ["E7", "E13"],
    "4.6": ["E12", "E18"],
    "4.7": ["E17", "E19"],
    "4.8": ["E20", "E21", "E22"],
}


def main() -> int:
    text = DRAFT.read_text(encoding="utf-8")
    fails: list[str] = []
    warns: list[str] = []

    # 1) 小节齐全
    sections = re.findall(r"^## (4\.\d) ", text, flags=re.M)
    for s in EXPECTED:
        if s not in sections:
            fails.append(f"缺少小节 {s}")

    # 2) 每小节 E 编号覆盖
    parts = re.split(r"^## (4\.\d) ", text, flags=re.M)
    sec_map = {}
    for i in range(1, len(parts), 2):
        sec_map[parts[i]] = parts[i + 1]
    for sec, codes in EXPECTED.items():
        body = sec_map.get(sec, "")
        for code in codes:
            if code not in body:
                fails.append(f"{sec} 缺少 {code} 引用")

    # 3) 证据引用格式 + JSON 存在性
    refs = re.findall(r"证据：(E\d[^）\n]*)", text)
    for ref in refs:
        if not re.search(r"E\d", ref):
            fails.append(f"证据引用缺 E 编号: {ref[:40]}")
        for m in re.finditer(r"outputs/([\w.\-]+\.json)", ref):
            fname = m.group(1)
            if not (OUT / fname).exists():
                fails.append(f"JSON 不存在: {fname}")

    # 4) 术语/格式警告
    if "联想层内" in text:
        warns.append("术语候选: '联想层内' -> 建议统一为 '联想基底' 表述")
    plain_stub = [m.start() for m in re.finditer(r"stub", text)]
    for pos in plain_stub:
        ctx = text[max(0, pos - 10):pos]
        if "外部" not in ctx:
            warns.append(f"裸 'stub'(前文无'外部') 出现于位置 {pos}: ...{ctx}...")
            break
    for param, label in (("λ", "λ"), ("hop_decay", "hop_decay"), ("阈值", "阈值")):
        if label not in text and param != "hop_decay":
            fails.append(f"缺少参数表述: {label}")
    if "固定预设系数" in text and "hop_decay" not in text:
        warns.append("每跳系数未用参数名 hop_decay, 建议补全参数名")
    for pattern, note in (
        (r"0\.\d0(?![0-9])", "尾随零小数如 0.90"),
        (r"1e-5|1E-5", "科学计数法 1e-5 写法"),
    ):
        hits = re.findall(pattern, text)
        if hits:
            warns.append(f"{note}: {len(hits)} 处 -> 建议统一格式")

    for w in warns:
        print("WARN -", w)
    for f in fails:
        print("FAIL -", f)
    print(f"\n汇总: {len(fails)} FAIL, {len(warns)} WARN")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
