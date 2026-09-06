"""E10: 新旧程度 - 同样次数, 更久没见 -> 检索变弱, 记忆永存。

训练(两条相同结构的网络, 唯一差别是 λ):
  [A,B] x100  在 t=1..100     (最旧)
  [A,C] x100  在 t=901..1000  (中等)
  [A,D] x100  在 t=1901..2000 (最新)
三条边 N 都=100, 只有 last_seen 不同。

判据(预先注册):
  S1 N 全等: N(A,B)=N(A,C)=N(A,D)=100;
  S2 last_seen: 100 / 1000 / 2000;
  S3 λ=0 对照: B=C=D=0.9 (没有时间衰减, 新旧一样强);
  S4 λ=0.005: D=0.9 > C≈0.006 > B≈7e-5 (越旧越难想起);
  S5 阈值: 0.5 只见 D; 1e-6 时 B/C/D 全在 (弱但存在);
  S6 同一网络把 λ 调回 0: B 立刻回 0.9 (衰减只在检索时刻);
  S7 存储不变: N/节点/连接全程未删。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention import attention_filter
from experiments.checker import Checker
from synapse_net import SynapseNet

LAMBDA = 0.005


def _train(net: SynapseNet) -> None:
    for i in range(1, 101):
        net.learn(["A", "B"], t=float(i))
    for i in range(1, 101):
        net.learn(["A", "C"], t=float(900 + i))
    for i in range(1, 101):
        net.learn(["A", "D"], t=float(1900 + i))


def run() -> dict:
    net_flat = SynapseNet()                    # λ=0: 对照 (等价于全部旧实验)
    net_decay = SynapseNet(time_decay_lambda=LAMBDA)
    _train(net_flat)
    _train(net_decay)
    c = Checker()

    # ---------- S1: N 全等 ----------
    ns = {k: net_decay.connection_count("A", k) for k in ("B", "C", "D")}
    c.check(
        "S1 次数全等: N(A,B)=N(A,C)=N(A,D)=100 (唯一变量是时间)",
        ns["B"] == ns["C"] == ns["D"] == 100,
        str(ns),
    )

    # ---------- S2: last_seen ----------
    ls = {k: net_decay.last_seen("A", k) for k in ("B", "C", "D")}
    c.check(
        "S2 时间戳: last_seen = 100 / 1000 / 2000",
        ls["B"] == 100.0 and ls["C"] == 1000.0 and ls["D"] == 2000.0,
        str(ls),
    )

    # ---------- S3: λ=0 对照 ----------
    acts_flat = net_flat.activate("A")
    ok3 = all(math.isclose(acts_flat[k], 0.9, rel_tol=1e-9) for k in ("B", "C", "D"))
    c.check(
        "S3 λ=0 对照: B=C=D=0.9 (没有时间衰减, 新旧一样强)",
        ok3,
        f"B={acts_flat['B']:.4f}, C={acts_flat['C']:.4f}, D={acts_flat['D']:.4f}",
    )

    # ---------- S4: λ=0.005 ----------
    acts_decay = net_decay.activate("A")
    exp_b = 0.9 * math.exp(-LAMBDA * (2000.0 - 100.0))   # Δt=1900
    exp_c = 0.9 * math.exp(-LAMBDA * (2000.0 - 1000.0))  # Δt=1000
    ok4 = (
        math.isclose(acts_decay["D"], 0.9, rel_tol=1e-9)
        and math.isclose(acts_decay["C"], exp_c, rel_tol=1e-9)
        and math.isclose(acts_decay["B"], exp_b, rel_tol=1e-9)
        and acts_decay["D"] > acts_decay["C"] > acts_decay["B"]
    )
    c.check(
        "S4 λ=0.005: D=0.9 > C≈0.006 > B≈7e-5 (越旧越难想起)",
        ok4,
        f"B={acts_decay['B']:.8f}, C={acts_decay['C']:.6f}, D={acts_decay['D']:.4f}",
    )

    # ---------- S5: 阈值视角 ----------
    hi = attention_filter(acts_decay, threshold=0.5)
    lo = attention_filter(acts_decay, threshold=1e-6)
    c.check(
        "S5 阈值: 0.5 只见 D; 1e-6 时 B/C/D 都在 (弱但存在)",
        set(hi) == {"A", "D"}
        and {"B", "C", "D"} <= set(lo),
        f"0.5 -> {sorted(hi)}; 1e-6 -> {sorted(lo)}",
    )

    # ---------- S6: 同一网络把 λ 调回 0 ----------
    net_decay.lambda_ = 0.0
    acts_back = net_decay.activate("A")
    ok6 = all(math.isclose(acts_back[k], 0.9, rel_tol=1e-9) for k in ("B", "C", "D"))
    c.check(
        "S6 同一网络 λ 调回 0: B/C/D 立刻全部回 0.9 (衰减只在检索时刻)",
        ok6,
        f"B={acts_back['B']:.4f}, C={acts_back['C']:.4f}, D={acts_back['D']:.4f}",
    )

    # ---------- S7: 存储不变 ----------
    c.check(
        "S7 存储不变: N 仍=100, 4 nodes / 3 edges 未删",
        net_decay.connection_count("A", "B") == 100
        and net_decay.node_count() == 4
        and net_decay.edge_count() == 3,
        f"N(A,B)={net_decay.connection_count('A', 'B')}, "
        f"nodes={net_decay.node_count()}, edges={net_decay.edge_count()}",
    )

    return {
        "id": "exp10_recency_lambda",
        "title": "E10 新旧程度: 同样次数, 更久没见 -> 检索变弱但记忆永存",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "lambda": LAMBDA,
            "N": ns,
            "last_seen": ls,
            "flat_activation": {k: round(v, 6) for k, v in sorted(acts_flat.items())},
            "decay_activation": {k: round(v, 9) for k, v in sorted(acts_decay.items())},
            "back_to_flat_activation": {k: round(v, 6) for k, v in sorted(acts_back.items())},
            "visible_0.5": sorted(hi),
            "visible_1e-6": sorted(lo),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
