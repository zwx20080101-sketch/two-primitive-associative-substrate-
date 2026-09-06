"""Synapse-Net V0 runner.

Usage:
    python main.py              # run all experiments
    python main.py exp1_chain   # run one experiment

Every run writes outputs/<exp_id>.json next to the console report.
Exit code is 1 if any selected experiment FAILs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments import ALL

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def fmt_activation(data: dict) -> str:
    act = data.get("activation") or data.get("activation_A") or data.get("full_activation") or {}
    if not act:
        return ""
    items = sorted(act.items())
    lines = []
    for i in range(0, len(items), 8):
        chunk = items[i : i + 8]
        lines.append("   " + "   ".join(f"{k}:{v:.6f}" for k, v in chunk))
    return "\n".join(lines)


def fmt_counts(data: dict) -> str:
    counts = data.get("counts") or {}
    if not counts:
        return ""
    items = list(counts.items())
    lines = []
    for i in range(0, len(items), 5):
        chunk = items[i : i + 5]
        lines.append("   " + "   ".join(f"{k}:{v}" for k, v in chunk))
    return "\n".join(lines)


def print_report(result: dict) -> None:
    title = f"[{result['id']}] {result['title']}"
    print(title)
    print("-" * len(title))
    for c in result["checks"]:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['name']}")
        if c.get("detail"):
            print(f"         {c['detail']}")
    act_txt = fmt_activation(result.get("data", {}))
    if act_txt:
        print(f"  激活输出:\n{act_txt}")
    counts_txt = fmt_counts(result.get("data", {}))
    if counts_txt:
        print(f"  连接次数 N (被一起提起的次数):\n{counts_txt}")
    print(f"  ==> 结果: {'PASS' if result['passed'] else 'FAIL'}\n")


def main() -> int:
    selected = sys.argv[1:] or ["all"]
    ids = sorted(ALL) if "all" in selected else [s for s in selected if s in ALL]
    unknown = [s for s in selected if s not in ALL and s != "all"]
    if unknown:
        print(f"未知实验: {unknown}", file=sys.stderr)
        print(f"可用: all, {', '.join(sorted(ALL))}", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_passed = True
    for exp_id in ids:
        result = ALL[exp_id].run()
        print_report(result)
        out_file = OUT_DIR / f"{exp_id}.json"
        out_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        all_passed = all_passed and result["passed"]

    print(f"汇总: {'全部 PASS' if all_passed else '存在 FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
