"""E3: threshold state switch - same bottom layer, different visible content."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention import attention_filter
from experiments.checker import Checker
from experiments.exp1_chain import ALPHABET, build_chain_net
from synapse_net import SynapseNet


def run() -> dict:
    net = build_chain_net()
    c = Checker()

    before = net.state_snapshot()
    acts = net.activate("A")
    visible = {
        thr: attention_filter(acts, threshold=thr)
        for thr in (0.5, 0.1, 0.001)
    }
    after = net.state_snapshot()

    c.check(
        "读取端对底层零副作用 (state 逐字节一致)",
        before == after,
        "",
    )

    expected_sizes = {0.5: 7, 0.1: 22, 0.001: 26}
    for thr, expect in expected_sizes.items():
        c.check(
            f"threshold={thr}: 可见节点数 = {expect}",
            len(visible[thr]) == expect,
            f"visible {len(visible[thr])} 个: {''.join(sorted(visible[thr]))}",
        )

    c.check(
        "高阈值只保留链头, 低阈值放行到链尾 (认知状态剧变, 网络不变)",
        set(visible[0.5]) < set(visible[0.1]) < set(visible[0.001]),
        "0.5 < 0.1 < 0.001",
    )

    return {
        "id": "exp3_threshold",
        "title": "E3 阈值状态开关: 同一底层输出的三种认知内容",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "activation_A": {ch: round(acts[ch], 6) for ch in ALPHABET},
            "visible_sets": {
                str(thr): "".join(sorted(v)) for thr, v in visible.items()
            },
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
