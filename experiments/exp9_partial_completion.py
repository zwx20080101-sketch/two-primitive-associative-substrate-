"""E9: 部分输入补全 - 缺口回填, 不许无中生有。

训练: 五连链 A-B-C-D-E, 相邻对各 100 次。

判据(预先注册):
  1) 打 A: B=0.9, C=0.81, D=0.729, E=0.6561 (全链回忆);
  2) 打 A+C+E: 缺的 B、D 被补出(0.9), 五节点全在, 无新节点;
  3) 打 A+E: 中间 C 出现(0.81) - 两端会合, 但 V0 max 规则不叠加(记录);
  4) 闭包: 对从未学过的 F activate -> KeyError, 无法唤醒不存在的节点;
  5) 补出的都是训练时存在的节点。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.checker import Checker
from synapse_net import SynapseNet

CHAIN = ["A", "B", "C", "D", "E"]


def run() -> dict:
    net = SynapseNet()
    for _ in range(100):
        for a, b in zip(CHAIN, CHAIN[1:]):
            net.learn([a, b])
    c = Checker()

    c.check(
        "训练结构: 5 nodes / 4 edges",
        net.node_count() == 5 and net.edge_count() == 4,
        f"nodes={net.node_count()}, edges={net.edge_count()}",
    )

    # 1) 只给开头
    acts_a = net.activate("A")
    expected1 = {"B": 0.9, "C": 0.81, "D": 0.729, "E": 0.9 ** 4}
    ok1 = all(math.isclose(acts_a[k], v, rel_tol=1e-9) for k, v in expected1.items())
    c.check(
        "1) 打 A: 全链回忆 B=0.9 C=0.81 D=0.729 E=0.6561",
        ok1,
        ", ".join(f"{k}={acts_a[k]:.4f}" for k in CHAIN),
    )

    # 2) 隔一个给一个 (A, C, E): 缺口 B、D 补出
    acts_gap = net.activate(["A", "C", "E"])
    ok2 = (
        set(acts_gap) == set(CHAIN)
        and math.isclose(acts_gap["B"], 0.9, rel_tol=1e-9)
        and math.isclose(acts_gap["D"], 0.9, rel_tol=1e-9)
    )
    c.check(
        "2) 打 A+C+E: 缺口 B、D 被补出(0.9), 五节点全在",
        ok2,
        ", ".join(f"{k}={acts_gap[k]:.4f}" for k in sorted(acts_gap)),
    )

    # 3) 只给首尾 A+E: 中间 C 出现, max 不叠加(记录)
    acts_ends = net.activate(["A", "E"])
    c.check(
        "3) 打 A+E: 中间 C=0.81 被补出 (两端会合, max 规则不叠加, 记录)",
        math.isclose(acts_ends["C"], 0.81, rel_tol=1e-9),
        ", ".join(f"{k}={acts_ends[k]:.4f}" for k in sorted(acts_ends)),
    )

    # 4) 闭包硬判据: 从未学过的 F 无法被唤醒
    raised = False
    try:
        net.activate("F")
    except KeyError:
        raised = True
    c.check(
        "4) 闭包: activate(F) 对未学过的 F 直接报错, 无法无中生有",
        raised and "F" not in net.nodes,
        f"F in nodes: {'F' in net.nodes}, activate raised KeyError: {raised}",
    )

    # 5) 补出的都是训练时存在的节点
    all_out = set(acts_a) | set(acts_gap) | set(acts_ends)
    c.check(
        "5) 补全只取回已存节点: 所有输出 ⊆ 训练节点集合",
        all_out <= set(CHAIN),
        f"output={sorted(all_out)}, learned={CHAIN}",
    )

    return {
        "id": "exp9_partial_completion",
        "title": "E9 部分输入补全: 缺口回填, 不许无中生有",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "network": {
                "nodes": net.node_count(),
                "edges": net.edge_count(),
            },
            "from_A": {k: round(v, 6) for k, v in sorted(acts_a.items())},
            "from_A_C_E": {k: round(v, 6) for k, v in sorted(acts_gap.items())},
            "from_A_E": {k: round(v, 6) for k, v in sorted(acts_ends.items())},
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
