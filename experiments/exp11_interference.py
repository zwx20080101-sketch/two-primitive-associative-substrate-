"""E11: 多词干扰 - 共享词多了, 单句还能不能干净取回。

训练(相邻对, 每句 x50):
  L1:  狗咬猫
  L2:  狗咬猫, 狗追老鼠
  L5:  狗咬猫, 狗追老鼠, 狗抓鸟, 狗舔水, 狗嗅花
带情景网络: 5 句各配唯一 X_i, 整窗共现 x50: [X_i, 狗, V_i, O_i]

判据(预先注册):
  1) 拥挤度: 打狗 -> 并行 0.9 候选数 = 句子数 (1/2/5), 无偏好;
  2) 从独特动词(抓)打: threshold 0.85 只见 {抓,狗,鸟} - 独特处依然干净;
  3) 无情景对照: 单给狗, 5 个候选全亮 (threshold 0.85) - 歧义集中在共享枢纽;
  4) 带情景: 从 X_i 打, threshold 0.85 只放出 {X_i,狗,V_i,O_i} - 情景可分离;
  5) 干扰不破坏记忆: N 仍=50, 节点/边不丢;
  6) 闭包: 全程无新节点。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention import attention_filter
from experiments.checker import Checker
from synapse_net import SynapseNet

SENTENCES = [
    ("狗", "咬", "猫"),
    ("狗", "追", "老鼠"),
    ("狗", "抓", "鸟"),
    ("狗", "舔", "水"),
    ("狗", "嗅", "花"),
]


def _train_pairs(net: SynapseNet, sentences, times: int = 50) -> None:
    for subj, verb, obj in sentences:
        for _ in range(times):
            net.learn([subj, verb])
            net.learn([verb, obj])


def run() -> dict:
    c = Checker()
    verbs = [v for _, v, _ in SENTENCES]
    objects = [o for _, _, o in SENTENCES]

    # ---------- 1) 拥挤度: 1 / 2 / 5 句 ----------
    counts_by_level = []
    for k in (1, 2, 5):
        net = SynapseNet()
        _train_pairs(net, SENTENCES[:k])
        acts = net.activate("狗")
        strong = [v for v in verbs[:k] if math.isclose(acts[v], 0.9, rel_tol=1e-9)]
        weak = [o for o in objects[:k] if math.isclose(acts[o], 0.81, rel_tol=1e-9)]
        counts_by_level.append(len(strong))
        if k == 5:
            net5 = net
            acts5 = acts
    c.check(
        "1) 拥挤度: 打狗, 并行 0.9 候选数 = 句子数 (1/2/5)",
        counts_by_level == [1, 2, 5],
        f"L1={counts_by_level[0]}, L2={counts_by_level[1]}, L5={counts_by_level[2]}",
    )
    all_equal = all(
        math.isclose(acts5[v], 0.9, rel_tol=1e-9) for v in verbs
    )
    c.check(
        "1b) 无偏好: 五个候选全部 0.9 (底层不选句)",
        all_equal,
        ", ".join(f"{v}={acts5[v]:.4f}" for v in verbs),
    )

    # ---------- 2) 从独特动词打: 本句独占恢复 ----------
    acts_zhua = net5.activate("抓")
    hi_zhua = attention_filter(acts_zhua, threshold=0.85)
    c.check(
        "2) 从独特动词'抓'打: threshold 0.85 只见 {抓,狗,鸟}",
        set(hi_zhua) == {"抓", "狗", "鸟"},
        f"visible={sorted(hi_zhua)}",
    )

    # ---------- 3) 无情景对照: 单给狗 ----------
    hi_dog = attention_filter(acts5, threshold=0.85)
    c.check(
        "3) 无情景: 单给狗, 5 个动词候选全亮 (歧义集中在共享枢纽)",
        set(hi_dog) == {"狗", "咬", "追", "抓", "舔", "嗅"},
        f"visible={sorted(hi_dog)}",
    )

    # ---------- 4) 带情景: X_i 整窗共现后可分离 ----------
    net_ctx = SynapseNet()
    for i, (subj, verb, obj) in enumerate(SENTENCES, start=1):
        for _ in range(50):
            net_ctx.learn([f"X{i}", subj, verb, obj])
    ctx_ok = True
    details = []
    for i, (subj, verb, obj) in enumerate(SENTENCES, start=1):
        acts = net_ctx.activate(f"X{i}")
        visible = attention_filter(acts, threshold=0.85)
        expected = {f"X{i}", subj, verb, obj}
        details.append(f"X{i}->{''.join(sorted(visible))}")
        ctx_ok = ctx_ok and set(visible) == expected
    c.check(
        "4) 带情景: 从 X_i 打, 0.85 只放出本句 {X_i,狗,V_i,O_i}",
        ctx_ok,
        "; ".join(details),
    )

    # ---------- 5) 干扰不破坏记忆 ----------
    n_ok = (
        all(net5.connection_count("狗", v) == 50 for v in verbs)
        and all(net5.connection_count(v, o) == 50 for v, o in zip(verbs, objects))
    )
    c.check(
        "5) 干扰不破坏记忆: N 仍=50; L5 为 11 nodes/10 edges",
        n_ok
        and net5.node_count() == 11
        and net5.edge_count() == 10,
        f"nodes={net5.node_count()}, edges={net5.edge_count()}",
    )

    # ---------- 6) 闭包 ----------
    learned5 = set(net5.nodes)
    learned_ctx = set(net_ctx.nodes)
    c.check(
        "6) 闭包: 所有输出节点都曾是学过的节点",
        set(acts5) <= learned5
        and set(net_ctx.activate("X1")) <= learned_ctx,
        "",
    )

    return {
        "id": "exp11_interference",
        "title": "E11 多词干扰: 歧义集中在共享枢纽, 情景可分离",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "crowding_candidates": {"L1": counts_by_level[0], "L2": counts_by_level[1], "L5": counts_by_level[2]},
            "from_dog_L5_0.85": sorted(hi_dog),
            "from_zhua_L5_0.85": sorted(hi_zhua),
            "context_visible_sets": {f"X{i}": d.split("->")[1] for i, d in enumerate(details, start=1)},
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
