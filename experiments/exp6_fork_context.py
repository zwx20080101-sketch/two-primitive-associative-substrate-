"""E6a: 分叉点实验 - 底层只给强度分布, 不判对错。

原则(经讨论锁定): 底层不知道"哪个结尾是对的"。蛇的例子里, 底层只存过
"蛇-咬-疼-恐惧"的共现; "恐惧"是外部模块消费激活路径的结果。
所以本实验所有判据都是"相对强度/结构", 不是"答对/答错"。

四个子场景(每个都从全新空网络开始):
  S1 无上下文等强: [A,B,C]x100 + [A,B,D]x100
      -> A,B 处 C 与 D 等强(诚实歧义, 底层不选边);
  S2 频率偏置:     [A,B,C]x100 + [A,B,D]x10
      -> C 明显强于 D(重复连接强度决定偏置, 不是规则);
  S3 上下文偏置:   [X1,A,B,C]x100 + [X2,A,B,D]x100
      -> X1 下 C 占优(与 X1 同窗共现过), X2 下 D 占优;
  S4 L1 边界观察:  同一批经验只喂 L1 相邻顺序
      -> X1 在场时 B 的分叉处 C/D 仍等强:
         两两顺序不携带上下文, 若顺序预测需要上下文 -> L2 候选。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention import attention_filter
from experiments.checker import Checker
from order_layer import OrderLayer
from synapse_net import SynapseNet


def _ratio(acts: dict, a: str, b: str) -> float:
    return acts[a] / acts[b]


def run() -> dict:
    c = Checker()
    data: dict = {}

    # ---------------- S1: 无上下文, 等次数 ----------------
    net1 = SynapseNet()
    for _ in range(100):
        net1.learn(["A", "B", "C"])
        net1.learn(["A", "B", "D"])
    c.check(
        "S1 结构: N(A,B)=200, N(B,C)=N(B,D)=100, 无 X",
        net1.connection_count("A", "B") == 200
        and net1.connection_count("B", "C") == 100
        and net1.connection_count("B", "D") == 100,
        f"N(A,B)={net1.connection_count('A', 'B')}, "
        f"N(B,C)={net1.connection_count('B', 'C')}, "
        f"N(B,D)={net1.connection_count('B', 'D')}",
    )
    s1 = net1.activate(["A", "B"])
    r1 = _ratio(s1, "C", "D")
    c.check(
        "S1 无上下文: C 与 D 等强 (ratio=1, 诚实歧义, 底层不选边)",
        math.isclose(r1, 1.0, rel_tol=1e-9),
        f"C={s1['C']:.6f}, D={s1['D']:.6f}, ratio={r1:.6f}",
    )
    data["s1"] = {"activation": {k: round(v, 6) for k, v in sorted(s1.items())}}

    # ---------------- S2: 频率偏置 (100 vs 10) ----------------
    net2 = SynapseNet()
    for _ in range(100):
        net2.learn(["A", "B", "C"])
    for _ in range(10):
        net2.learn(["A", "B", "D"])
    s2 = net2.activate(["A", "B"])
    r2 = _ratio(s2, "C", "D")
    c.check(
        "S2 频率: C:D ≈ 10:1 (经验次数不同 -> 强度不同, 不是谁对)",
        math.isclose(r2, 10.0, rel_tol=0.01),
        f"C={s2['C']:.6f}, D={s2['D']:.6f}, ratio={r2:.3f}",
    )
    hi2 = attention_filter(s2, threshold=0.5)
    c.check(
        "S2 阈值 0.5: 只剩 C 可见 (被消费的是强路径, 弱路径仍存在)",
        set(hi2) == {"A", "B", "C"} and "D" not in hi2,
        f"visible={sorted(hi2)}",
    )
    data["s2"] = {"activation": {k: round(v, 6) for k, v in sorted(s2.items())}}

    # ---------------- S3: 上下文偏置 (X1 vs X2) ----------------
    net3 = SynapseNet()
    for _ in range(100):
        net3.learn(["X1", "A", "B", "C"])
        net3.learn(["X2", "A", "B", "D"])
    s3_x1 = net3.activate(["X1", "A", "B"])
    s3_x2 = net3.activate(["X2", "A", "B"])
    c.check(
        "S3 X1 下: C 显著强于 D (C 曾与 X1 同窗共现)",
        s3_x1["C"] > 1.5 * s3_x1["D"],
        f"C={s3_x1['C']:.6f}, D={s3_x1['D']:.6f}, ratio={_ratio(s3_x1, 'C', 'D'):.3f}",
    )
    c.check(
        "S3 X2 下: D 显著强于 C (对称)",
        s3_x2["D"] > 1.5 * s3_x2["C"],
        f"D={s3_x2['D']:.6f}, C={s3_x2['C']:.6f}, ratio={_ratio(s3_x2, 'D', 'C'):.3f}",
    )
    hi3_x1 = attention_filter(s3_x1, threshold=0.7)
    hi3_x2 = attention_filter(s3_x2, threshold=0.7)
    c.check(
        "S3 阈值 0.7: X1 下只放出 C, X2 下只放出 D",
        set(hi3_x1) == {"X1", "A", "B", "C"}
        and set(hi3_x2) == {"X2", "A", "B", "D"},
        f"X1 visible={sorted(hi3_x1)}; X2 visible={sorted(hi3_x2)}",
    )
    data["s3"] = {
        "X1_activation": {k: round(v, 6) for k, v in sorted(s3_x1.items())},
        "X2_activation": {k: round(v, 6) for k, v in sorted(s3_x2.items())},
    }

    # ---------------- S4: L1 相邻顺序的能力边界 ----------------
    order = OrderLayer()
    for _ in range(100):
        order.learn(["X1", "A", "B", "C"])
        order.learn(["X2", "A", "B", "D"])
    s4 = order.activate(["X1", "A", "B"])
    r4 = _ratio(s4, "C", "D")
    c.check(
        "S4 L1 边界: X1 在场, B 的分叉处 C/D 仍等强 (两两顺序不带上下文)",
        math.isclose(r4, 1.0, rel_tol=1e-9),
        f"C={s4['C']:.6f}, D={s4['D']:.6f}, ratio={r4:.6f}",
    )
    data["s4"] = {
        "note": "若要求顺序预测也利用上下文 -> L2 候选",
        "activation": {k: round(v, 6) for k, v in sorted(s4.items())},
    }

    return {
        "id": "exp6_fork_context",
        "title": "E6a 分叉与上下文: 底层只给强度分布, 不判对错",
        "passed": c.passed,
        "checks": c.checks,
        "data": data,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
