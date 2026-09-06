"""E2: repeated experience strength - N changes later propagation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention import attention_filter
from experiments.checker import Checker
from synapse_net import SynapseNet


def run() -> dict:
    net = SynapseNet()
    for _ in range(100):
        net.learn(["A", "B"])
    for _ in range(100):
        net.learn(["B", "C"])
    for _ in range(100):
        net.learn(["C", "D"])
    for _ in range(2):
        net.learn(["A", "X"])
    c = Checker()

    c.check(
        "P3: [A,B]x100 -> N(A,B)=100",
        net.connection_count("A", "B") == 100,
        f"N(A,B)={net.connection_count('A', 'B')}",
    )
    c.check(
        "弱经验 [A,X]x2 -> N(A,X)=2",
        net.connection_count("A", "X") == 2,
        f"N(A,X)={net.connection_count('A', 'X')}",
    )

    w_ab = net.effective_weight("A", "B")
    w_ax = net.effective_weight("A", "X")
    c.check(
        "P3: 有效强度 W(A,B)=100 > W(A,X)=2",
        w_ab == 100 and w_ax == 2 and w_ab > w_ax,
        f"W(A,B)={w_ab}, W(A,X)={w_ax}",
    )
    c.check(
        "归一化强度: strength(A->B)=1.0 > strength(A->X)=0.02",
        net.strength("A", "B") == 1.0 and abs(net.strength("A", "X") - 0.02) < 1e-12,
        f"s(A->B)={net.strength('A', 'B')}, s(A->X)={net.strength('A', 'X')}",
    )

    acts = net.activate("A")
    c.check(
        "底层全量输出必须包含弱节点 X",
        "X" in acts,
        f"keys={sorted(acts)}",
    )
    expected = {"A": 1.0, "B": 0.9, "C": 0.81, "D": 0.729, "X": 0.018}
    mismatch = {
        k: (acts.get(k), v)
        for k, v in expected.items()
        if abs(acts.get(k, 0.0) - v) > 1e-9
    }
    c.check(
        "逐节点数值符合公式 (B=0.9, C=0.81, D=0.729, X=0.018)",
        not mismatch,
        f"observed={ {k: round(acts[k], 6) for k in sorted(acts)} }",
    )
    c.check(
        "B 明显强于 X (重复经验改变传播)",
        acts["B"] > acts["X"] * 10,
        f"a(B)={acts['B']:.6f}, a(X)={acts['X']:.6f}",
    )

    high = attention_filter(acts, threshold=0.5)
    low = attention_filter(acts, threshold=0.01)
    c.check(
        "threshold=0.5: X 被过滤 (想不起 X)",
        set(high) == {"A", "B", "C", "D"},
        f"visible={sorted(high)}",
    )
    c.check(
        "threshold=0.01: X 重新可访问 (X 一直存在)",
        "X" in low,
        f"visible={sorted(low)}",
    )

    return {
        "id": "exp2_strength",
        "title": "E2 重复经验强度: 底层保存全部, 阈值决定可访问",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "network": {
                "N(A,B)": net.connection_count("A", "B"),
                "N(B,C)": net.connection_count("B", "C"),
                "N(C,D)": net.connection_count("C", "D"),
                "N(A,X)": net.connection_count("A", "X"),
            },
            "full_activation": {k: round(v, 6) for k, v in sorted(acts.items())},
            "visible_high_threshold": sorted(high),
            "visible_low_threshold": sorted(low),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
