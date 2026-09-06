"""E0: empty network + single-event co-occurrence binding."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention import attention_filter
from experiments.checker import Checker
from synapse_net import SynapseNet


def run() -> dict:
    net = SynapseNet()
    c = Checker()

    c.check(
        "空网络启动: 0 nodes / 0 connections",
        net.node_count() == 0 and net.edge_count() == 0,
        f"nodes={net.node_count()}, connections={net.edge_count()}",
    )

    net.learn(["A", "B"])
    c.check(
        "learn([A,B]) 后: 2 nodes / 1 connection",
        net.node_count() == 2 and net.edge_count() == 1,
        f"nodes={net.node_count()}, connections={net.edge_count()}",
    )
    c.check(
        "A-B 存在且 count=1",
        net.connection_count("A", "B") == 1,
        f"N(A,B)={net.connection_count('A', 'B')}",
    )
    c.check(
        "A-Z 不存在 (无共现则无连接)",
        net.connection_count("A", "Z") == 0,
        "N(A,Z)=0",
    )

    acts = net.activate("A")
    expect_keys = {"A", "B"}
    c.check(
        "activate(A) 只唤醒 A、B",
        set(acts) == expect_keys,
        f"keys={sorted(acts)}",
    )
    c.check(
        "B 按公式到达: 1.0 x strength(1.0) x hop_decay(0.9)",
        abs(acts.get("B", 0.0) - 0.9) < 1e-9,
        f"a(B)={acts.get('B', 0.0):.9f}",
    )

    net.learn(["A", "B"])
    c.check(
        "再次 learn([A,B]): count 累计到 2 (记忆增强)",
        net.connection_count("A", "B") == 2,
        f"N(A,B)={net.connection_count('A', 'B')}",
    )

    low = attention_filter(acts, threshold=0.01)
    c.check(
        "读取端纯函数: 阈值 0.01 仍可见 A、B",
        set(low) == {"A", "B"},
        f"visible={sorted(low)}",
    )

    return {
        "id": "exp0_smoke",
        "title": "E0 冒烟: 空网络与单事件绑定",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "start": {"nodes": 0, "connections": 0},
            "after_first_learn": {
                "nodes": net.node_count(),
                "connections": net.edge_count(),
                "N(A,B)": net.connection_count("A", "B"),
            },
            "activation_after_first_learn": acts,
            "after_second_learn": {"N(A,B)": net.connection_count("A", "B")},
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
