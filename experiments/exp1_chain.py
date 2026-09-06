"""E1: V0 main experiment - 26-letter chain from co-occurrence alone."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.checker import Checker
from synapse_net import SynapseNet

ALPHABET = [chr(ord("A") + i) for i in range(26)]


def build_chain_net() -> SynapseNet:
    """Learn [A,B], [B,C], ..., [Y,Z], each exactly once."""
    net = SynapseNet()
    for a, b in zip(ALPHABET, ALPHABET[1:]):
        net.learn([a, b])
    return net


def run() -> dict:
    net = build_chain_net()
    c = Checker()

    c.check(
        "P1: 26 个事件后得到 26 nodes / 25 connections",
        net.node_count() == 26 and net.edge_count() == 25,
        f"nodes={net.node_count()}, connections={net.edge_count()}",
    )
    c.check(
        "P2: A-B 存在 (count=1)",
        net.connection_count("A", "B") == 1,
        f"N(A,B)={net.connection_count('A', 'B')}",
    )
    c.check(
        "P2: A-Z 不存在 (从未共现)",
        net.connection_count("A", "Z") == 0,
        "N(A,Z)=0",
    )

    acts = net.activate("A")
    missing = [ch for ch in ALPHABET if ch not in acts]
    c.check(
        "activate(A) 全链可达 (A..Z 全部出现)",
        not missing,
        f"missing={missing or '无'}",
    )

    vals = [acts[ch] for ch in ALPHABET]
    strictly_decreasing = all(vals[i] > vals[i + 1] for i in range(25))
    c.check(
        "激活值沿 A->Z 严格单调递减",
        strictly_decreasing,
        "A..Z: " + ", ".join(f"{v:.6f}" for v in vals),
    )

    ratios_ok = all(
        math.isclose(vals[i + 1] / vals[i], 0.9, rel_tol=1e-9)
        for i in range(25)
    )
    c.check(
        "P4: 每跳精确满足 a(n+1) = a(n) x strength x hop_decay",
        ratios_ok,
        f"ratio={vals[1] / vals[0]:.9f} (期望 0.9)",
    )
    z_expected = 0.9 ** 25
    c.check(
        "Z 到达预期深度: 0.9^25",
        math.isclose(acts["Z"], z_expected, rel_tol=1e-9),
        f"a(Z)={acts['Z']:.9f}, 期望={z_expected:.9f}",
    )

    return {
        "id": "exp1_chain",
        "title": "E1 主实验: 26 字母链 (纯共现 + 激活扩散)",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "network": {
                "nodes": net.node_count(),
                "connections": net.edge_count(),
                "N(A,B)": net.connection_count("A", "B"),
                "N(A,Z)": net.connection_count("A", "Z"),
            },
            "activation": {ch: round(acts[ch], 9) for ch in ALPHABET},
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
