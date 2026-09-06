"""E5: L1 方向/顺序层 - HELLO 正走 vs 倒走。

反复"说出 HELLO" 100 次(相邻字母对按先后喂给 L0 与 L1):
  L0(冻结): 只记录 H-e, e-l, l-o 的共现, 无向、对称;
  L1(本层): 记录 H->e, e->l, l->o 各 100 次, 反方向 0 次。

预期:
  - 从 H 打: 正向一路顺, o 仍接近 0.9^3;
  - 从 o 打: 反向第一跳 l 还容易, 越往深处越弱(费劲), 低阈值才能倒完整;
  - L0 本身仍然完全对称(倒走和正走一样强)——顺序不对称只来自 L1。

注: HELLO 的双写 l 在节点层是同一个节点; L1 自 E13 起会把 l->l
    记录为"重复标记"(同节点相邻二次到达), 链为 H-e-l-o 外加 l->l。
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

HELLO = ["H", "e", "l", "l", "o"]  # 双写 l 在节点层折叠为同一 "l"


def _walk(layer, start: str) -> list[str]:
    """读取端走读器: 每步取激活最强的未访问邻居 (与 E4 同规则)。"""
    seq = [start]
    seen = {start}
    current = start
    while True:
        acts = layer.activate(current)
        candidates = {k: v for k, v in acts.items() if k not in seen and v > 1e-9}
        if not candidates:
            break
        nxt = max(sorted(candidates), key=lambda k: candidates[k])
        seq.append(nxt)
        seen.add(nxt)
        current = nxt
    return seq


def run(passes: int = 100) -> dict:
    bottom = SynapseNet()          # L0: 冻结底层, 每次全新空网络
    order = OrderLayer()           # L1: 方向/顺序层
    c = Checker()

    c.check(
        "L0 与 L1 都从空白启动",
        bottom.node_count() == 0 and bottom.edge_count() == 0
        and order.node_count() == 0 and order.edge_count() == 0,
        f"L0: {bottom.node_count()} nodes / {bottom.edge_count()} edges; "
        f"L1: {order.node_count()} nodes / {order.edge_count()} directed pairs",
    )

    for _ in range(passes):
        # L0(冻结): 相邻字母对的"同时/共现", 只建无向连接
        for a, b in zip(HELLO, HELLO[1:]):
            if a != b:
                bottom.learn([a, b])
        # L1: 一次完整说出 HELLO 的顺序事件
        order.learn(HELLO)

    c.check(
        "经验后: L0 有 4 nodes / 3 无向连接; L1 有 4 nodes / 4 方向记录(含 l->l 重复标记)",
        bottom.node_count() == 4 and bottom.edge_count() == 3
        and order.node_count() == 4 and order.edge_count() == 4,
        f"L0: {bottom.node_count()} / {bottom.edge_count()}; "
        f"L1: {order.node_count()} / {order.edge_count()}",
    )

    counts = {
        "H->e": order.count("H", "e"),
        "e->l": order.count("e", "l"),
        "l->l": order.count("l", "l"),
        "l->o": order.count("l", "o"),
        "e->H": order.count("e", "H"),
        "l->e": order.count("l", "e"),
        "o->l": order.count("o", "l"),
    }
    c.check(
        "方向分开记账: 顺向与 l->l 重复标记各 100 次, 反向各 0 次",
        counts["H->e"] == counts["e->l"] == counts["l->o"] == passes
        and counts["l->l"] == passes
        and counts["e->H"] == counts["l->e"] == counts["o->l"] == 0,
        str(counts),
    )

    fwd = order.activate("H")
    expected_o = 0.9 ** 3
    c.check(
        "从 H 打: 正向一路顺, 到达 o (0.9^3)",
        math.isclose(fwd["o"], expected_o, rel_tol=1e-9),
        "H:1.0 -> e:0.9 -> l:0.81 -> o:0.729",
    )

    rev = order.activate("o")
    expected_e = 0.9 * (1 / 101) * 0.9   # 反向第二跳只剩 1/101 保底强度
    expected_H = expected_e * (1 / 101) * 0.9
    c.check(
        "从 o 打: 第一跳 l 仍顺 (末尾好回忆)",
        math.isclose(rev["l"], 0.9, rel_tol=1e-9),
        f"a(l)={rev['l']:.9f}",
    )
    c.check(
        "反向越走越费劲: e 已近消失, H 几乎不可达",
        rev["e"] < 0.01 and rev["H"] < 0.0001,
        f"a(e)={rev['e']:.6f}, a(H)={rev['H']:.8f} (公式预期 "
        f"e={expected_e:.6f}, H={expected_H:.8f})",
    )

    visible_high = attention_filter(rev, threshold=0.5)
    visible_low = attention_filter(rev, threshold=1e-5)
    c.check(
        "threshold=0.5 时倒背只够得着 o、l; 降到 1e-5 才倒完整",
        set(visible_high) == {"o", "l"} and set(visible_low) == {"o", "l", "e", "H"},
        f"0.5 -> {''.join(sorted(visible_high))}; "
        f"1e-5 -> {''.join(sorted(visible_low))}",
    )

    bottom_rev = bottom.activate("o")
    c.check(
        "冻结的 L0 仍完全对称: 从 o 打, H 照样 0.729",
        math.isclose(bottom_rev["H"], 0.729, rel_tol=1e-9),
        f"L0 a(H)={bottom_rev['H']:.9f}",
    )

    fwd_walk = _walk(order, "H")
    rev_walk = _walk(order, "o")
    c.check(
        "走读器: H 出发正向走通 H-e-l-o",
        fwd_walk == ["H", "e", "l", "o"],
        "-".join(fwd_walk),
    )
    c.check(
        "走读器: o 出发反向仍能倒走 o-l-e-H (只是每步都弱)",
        rev_walk == ["o", "l", "e", "H"],
        "-".join(rev_walk),
    )

    return {
        "id": "exp5_direction_hello",
        "title": "E5 L1 方向层: HELLO 正走顺、倒走费力 (顺序只来自经验统计)",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "counts": counts,
            "activation": {k: round(v, 9) for k, v in sorted(fwd.items())},
            "reverse_activation": {k: round(v, 9) for k, v in sorted(rev.items())},
            "bottom_reverse_activation": {
                k: round(v, 9) for k, v in sorted(bottom_rev.items())
            },
            "forward_walk": "-".join(fwd_walk),
            "reverse_walk": "-".join(rev_walk),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
