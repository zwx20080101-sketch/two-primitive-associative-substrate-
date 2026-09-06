"""E14: 多路证据 - 两路信号同达, 应不应该比单路强? (max vs sum)

宪章四问: 基底参与=传播会合规则; 外部 stub=无; 结论归谁=基底传播公式;
做到哪步停=给出差异数据与冻结建议, 由用户拍板。

训练:
  单路: A-B、B-T x100         -> T 只有一条路
  双路: A-B、B-T、A-D、D-T x100 -> T 有两条独立路

规则对比(实验层实现, 不改任何层文件):
  max 规则: a(v)=max(到达); 双路 T = 单路 T = 0.81 (会合无增益);
  sum 规则: 单次贡献求和、封顶 1.0; 双路 T = 1.0, 单路 T = 0.81 (有增益);
  循环安全: 三角形环上 sum 规则必须终止且值 <= 1.0。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.checker import Checker
from synapse_net import SynapseNet


def _max_activate(net: SynapseNet, seed: str) -> dict[str, float]:
    """现状规则: 与 L0 一致的 max 吸收。"""
    return net.activate(seed)


def _sum_activate(
    net: SynapseNet,
    seed: str,
    cap: float = 1.0,
    hop_decay: float = 0.9,
) -> tuple[dict[str, float], int]:
    """会合求和规则(实验层): 单次发射、冻结于出队时刻。

    策略: 每个节点只发射一次(值在出队时冻结);
    已发射的节点不再接收后续贡献——防止无向边回环把整图泵到封顶。
    同一波到达的多路贡献在目标出队前累加并封顶 cap。
    """
    from collections import deque

    acts = {seed: 1.0}
    fired: set[str] = set()
    queue = deque([seed])
    queued = {seed}
    fired_count = 0
    while queue:
        u = queue.popleft()
        queued.discard(u)
        if u in fired:
            continue
        fired.add(u)  # 冻结: 之后不再接收贡献
        fired_count += 1
        au = acts[u]
        for v in net._adj.get(u, ()):
            if v in fired:
                continue
            arrival = au * net.strength(u, v) * hop_decay
            if arrival > 1e-12:
                acts[v] = min(cap, acts.get(v, 0.0) + arrival)
                if v not in queued:
                    queued.add(v)
                    queue.append(v)
    return acts, fired_count


def _train(net: SynapseNet, pairs: list[tuple[str, str]], times: int = 100) -> None:
    for a, b in pairs:
        for _ in range(times):
            net.learn([a, b])


def run() -> dict:
    c = Checker()

    net_single = SynapseNet()
    _train(net_single, [("A", "B"), ("B", "T")])
    net_dual = SynapseNet()
    _train(net_dual, [("A", "B"), ("B", "T"), ("A", "D"), ("D", "T")])

    # 1) max 规则
    max_single = _max_activate(net_single, "A")["T"]
    max_dual = _max_activate(net_dual, "A")["T"]
    c.check(
        "1) max 规则: 双路 T = 单路 T = 0.81 (会合无增益, 现状)",
        math.isclose(max_single, 0.81, rel_tol=1e-9)
        and math.isclose(max_dual, 0.81, rel_tol=1e-9),
        f"单路={max_single:.4f}, 双路={max_dual:.4f}",
    )

    # 2) sum 规则
    sum_single, _ = _sum_activate(net_single, "A")
    sum_dual, iters = _sum_activate(net_dual, "A")
    c.check(
        "2) sum 规则: 双路 T = 1.0 (封顶), 单路 T = 0.81 (会合有增益)",
        math.isclose(sum_single["T"], 0.81, rel_tol=1e-9)
        and math.isclose(sum_dual["T"], 1.0, rel_tol=1e-9),
        f"单路={sum_single['T']:.4f}, 双路={sum_dual['T']:.4f}",
    )

    # 3) 差异量化
    ratio_max = max_dual / max_single
    ratio_sum = sum_dual["T"] / sum_single["T"]
    c.check(
        "3) 双路/单路比值: max=1.0, sum≈1.23",
        math.isclose(ratio_max, 1.0, rel_tol=1e-9)
        and math.isclose(ratio_sum, 1.0 / 0.81, rel_tol=1e-3),
        f"max 比值={ratio_max:.4f}, sum 比值={ratio_sum:.4f}",
    )

    # 4) 成本清单(静态): 若全局换成 sum 会影响的既有实验
    affected = {
        "E9": "打 A+E 时中间 C 会从 0.81 升到封顶 1.0 (会合点判据要重基线)",
        "E8": "打狗时 猫 会收到 咬->猫 与 追->猫 两路 -> 从 0.81 升到 1.0",
        "E6a-S3": "C 收到 X1 直达 + B 转来两路 -> 比值从 2.0 变化",
    }
    c.check(
        "4) 成本清单已记录: 换 sum 需要重基线的实验 = E9/E8/E6a-S3",
        len(affected) == 3,
        "; ".join(f"{k}: {v}" for k, v in affected.items()),
    )

    # 5) 循环安全: 三角形环上 sum 必须终止且值封顶
    net_cycle = SynapseNet()
    _train(net_cycle, [("A", "B"), ("B", "C"), ("C", "A")], times=10)
    cycle_acts, cycle_iters = _sum_activate(net_cycle, "A")
    terminated = cycle_iters < 1000
    bounded = all(v <= 1.0 + 1e-9 for v in cycle_acts.values())
    c.check(
        "5) 循环安全: sum 在三角形环上终止且所有值 <= 1.0 (封顶)",
        terminated and bounded,
        f"iterations={cycle_iters}, acts={ {k: round(v, 6) for k, v in sorted(cycle_acts.items())} }",
    )

    return {
        "id": "exp14_multi_path_evidence",
        "title": "E14 多路证据: max vs sum 会合规则对比 (不改层文件)",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "max_rule": {
                "single_T": round(max_single, 6),
                "dual_T": round(max_dual, 6),
                "dual_single_ratio": round(ratio_max, 6),
            },
            "sum_rule": {
                "single_T": round(sum_single["T"], 6),
                "dual_T": round(sum_dual["T"], 6),
                "dual_single_ratio": round(ratio_sum, 6),
                "converged_iterations_dual": iters,
            },
            "affected_if_global_sum": affected,
            "cycle_safety": {
                "terminated": terminated,
                "iterations": cycle_iters,
                "bounded": bounded,
            },
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
