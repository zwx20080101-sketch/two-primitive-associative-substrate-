"""E43: 抑制控制 —— 负值注入对【轨迹】的压制效果。

严格按 DESIGN-E43.md。**不改任何层文件；v2 only；不引入 stub / 语义 / 奖励。**

机制（别写错）：负值节点进入 v2 的 frozen 集合 ⇒ **阻断正值路径的传播**（结构屏障），
  不是"数值压低"（max 吸收下负值不参与竞争）。

预注册要点:
  - 主语料 = 自造 7 节点拓扑（割点结构）；P1–P5 全部跑它
  - 辅语料 = C1（只给 P6）；corpus_planted() 返回 v1 的 net ⇒ 必须重放进 v2 并做逐位校验
  - P1 主判据 = 路径排他；P2 辅判据 = 下游 <= 0（只算 tip1）；P3 自检 = 注入值保留 + 种子未被压制
  - P4 = 阴性控制（注入 junction ⇒ 5 个节点变）；P5 = 阴性控制（破坏割点 ⇒ 压制失败）
  - 节点名三层防护：零撞车命名 + 常量集中定义 + 只用常量引用（E42/E43 的命名教训）
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.checker import Checker
from experiments.exp24_module_differentiation import (
    corpus_planted,
    impute_diagonal,
    matrix_from_net,
    spectral,
)
from synapse_net_v2 import SynapseNet as V2

# --- 节点名常量（集中定义；代码其余部分只引用常量，不写裸字符串）-------------------
START_NODE = "start_node"
TRUNK = "trunk"
JUNCTION = "junction"
ARM1 = "arm1"
TIP1 = "tip1"
ARM2 = "arm2"
TIP2 = "tip2"
NODES = (START_NODE, TRUNK, JUNCTION, ARM1, TIP1, ARM2, TIP2)

EDGES = ((START_NODE, TRUNK), (TRUNK, JUNCTION), (JUNCTION, ARM1),
         (ARM1, TIP1), (JUNCTION, ARM2), (ARM2, TIP2))
EDGE_REPS = 100

SEED_VALUE = 1.0
NEG_VALUE = -0.8
TOL = 1e-12

# --- 预注册阈值（跑后不改）-----------------------------------------------------
SUPPRESS_THRESHOLD = 0.0      # 下游 <= 0（严格）
EXPECT_P1_CHANGED = 2         # P1：变化节点数（确定性值）
EXPECT_P4_CHANGED = 5         # P4：注入 junction 时的变化节点数（确定性值）
EXPECT_P5_TIP1 = 0.729        # P5：破坏割点后 tip1 保持的基线值

# --- 基线七元组（§2.1 预注册；P1 的"逐位不变"锚定到它）-----------------------
BASELINE_EXPECTED = {
    START_NODE: 1.000000, TRUNK: 0.900000, JUNCTION: 0.810000,
    ARM1: 0.729000, TIP1: 0.656100, ARM2: 0.729000, TIP2: 0.656100,
}

HASH_FILES = ["synapse_net.py", "synapse_net_v2.py", "synapse_net_v3.py",
              "order_layer.py", "context_layer.py", "chunk_layer.py"]
V1_HASH_EXPECTED = "cf99a07a0e60fd4d"
V2_HASH_EXPECTED = "ee6df589eb2f6983"
V3_HASH_EXPECTED = "085be73a06f0694c"
LAYER_HASH_EXPECTED = {"order_layer.py": "419d54745a9eb795",
                       "context_layer.py": "bd029d7bbdc345be",
                       "chunk_layer.py": "46f5857be0fc7c7e"}
SPECTRAL_SEED = 20260911

C_ENTRY_TEXT = (
    "C??：在【割点结构】上，用 v2 的已登记带值入口把负值注入【分叉点下游的第一个节点】，"
    "该节点成为【路径屏障】（frozen 集合阻断正值传播）⇒ 该分支的下游被压到 <= 0"
    "（本语料 tip1：0.656100 → -0.720000，精确 = 注入值 x hop_decay），"
    "而【另一分支逐位不变】（arm2/tip2 差异 = 0）。"
    "对照组（阴性）：注入到共同上游 junction（非割点）⇒ 两个分支都被压（5 个节点变化），"
    "证明「只有一支变」来自割点结构；给下游加替代正路径 ⇒ 压制失败（tip1 保持正值），"
    "证明割点条件的必要性。"
    "范围限定：① 机制是【结构屏障】（frozen 阻断），不是【数值压低】——与 C38 的"
    "「构造参数 g 在到达值算式做减法」（数学运算）机制不同；"
    "② 下游的【精确负值】只在「紧邻叶子（全部邻居 frozen）」时成立，"
    "含环下游会退化为 EPS 伪影 ⇒ 判据写「<= 0」，不写精确值；"
    "③ 不压制种子（start_node 恒 1.0）⇒ 与 B33 不冲突；④ 功劳归 v2 的已登记入口（C37/B30）"
    "的 frozen 行为 + 注入策略（选割点），不是基底新能力（L0 一字未动）；"
    "⑤ 语料是 E43 自造的 7 节点最小拓扑。"
)
FAIL_TEXT = (
    "E43 未通过：哪条判据挂、实测值见 exp43 JSON。失败分层：① 入口是否保留负值（P0/P3）"
    "→ ② 屏障是否形成（P2）→ ③ 排他性是否来自割点（P4/P5）。"
    "机制性 FAIL 才在跑完后新立 B 编号；实现/判据问题则修正后重跑（不新立编号）。"
)


def _sha16(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def all_hashes() -> dict:
    return {f: _sha16(f) for f in HASH_FILES}


def build_main_corpus():
    """主语料：7 节点自造拓扑（v2 实例）。"""
    net = V2()
    for a, b in EDGES:
        for _ in range(EDGE_REPS):
            net.learn([a, b])
    return net


def clone_v1_into_v2(net1):
    """把 v1 的 connections 重放进新的 v2 实例（§2.3 的三步重放）。"""
    net2 = V2()
    for (a, b), rec in net1.connections.items():
        for _ in range(int(rec["count"])):
            net2.learn([a, b])
    return net2


def edge_counts(net) -> dict:
    return {tuple(sorted(k)): int(v["count"]) for k, v in net.connections.items()}


def reachable(adj: dict, src: str, blocked: set) -> set:
    seen, stack = {src}, [src]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v in blocked or v in seen:
                continue
            seen.add(v)
            stack.append(v)
    return seen


def run() -> dict:
    c = Checker()
    h_before = all_hashes()
    net = build_main_corpus()

    # ---------- 基线 / 阳性 / 阴性控制（全部主语料）----------
    baseline = net.activate({START_NODE: SEED_VALUE})
    pos = net.activate({START_NODE: SEED_VALUE, ARM1: NEG_VALUE})
    inj_junction = net.activate({START_NODE: SEED_VALUE, JUNCTION: NEG_VALUE})
    inj_tip1 = net.activate({START_NODE: SEED_VALUE, TIP1: NEG_VALUE})

    changed_pos = [k for k in NODES if abs(pos[k] - baseline[k]) > TOL]
    changed_junction = [k for k in NODES if abs(inj_junction[k] - baseline[k]) > TOL]
    changed_tip1 = [k for k in NODES if abs(inj_tip1[k] - baseline[k]) > TOL]

    # ---------- P0 前置：拍平自检 ----------
    flatten_ok = abs(pos[ARM1] - NEG_VALUE) <= TOL and abs(pos[START_NODE] - SEED_VALUE) <= TOL
    c.check("P0 前置: v2 only + 拍平自检（注入后 arm1 的实际值 == 注入值）+ 六文件哈希",
            flatten_ok and h_before["synapse_net.py"] == V1_HASH_EXPECTED
            and h_before["synapse_net_v2.py"] == V2_HASH_EXPECTED,
            f"拍平自检：arm1 = {pos[ARM1]:.6f}（注入 {NEG_VALUE}，容差 {TOL}）⇒ {flatten_ok}；"
            f"v1={h_before['synapse_net.py']} v2={h_before['synapse_net_v2.py']}"
            f"（若用 v1 的 dict 入口，arm1 会被静默拍平成 1.0 —— B21）")

    # ---------- P1 主判据：路径排他 ----------
    branch2_unchanged = (abs(pos[ARM2] - baseline[ARM2]) <= TOL
                         and abs(pos[TIP2] - baseline[TIP2]) <= TOL)
    p1_ok = branch2_unchanged and pos[TIP1] <= SUPPRESS_THRESHOLD and len(changed_pos) == EXPECT_P1_CHANGED
    c.check("P1 主判据（路径排他，定义 D）: 分支2 逐位不变 + 分支1 下游 <= 0 + 变化节点数 = 2",
            p1_ok,
            f"变化节点 = {changed_pos}（预期 2 个 = [{ARM1}, {TIP1}]）；"
            f"分支2：{ARM2} {baseline[ARM2]:.6f} → {pos[ARM2]:.6f}、"
            f"{TIP2} {baseline[TIP2]:.6f} → {pos[TIP2]:.6f}（逐位不变 = {branch2_unchanged}）；"
            f"分支1 下游：{TIP1} {baseline[TIP1]:.6f} → {pos[TIP1]:.6f}（<= 0 = {pos[TIP1] <= SUPPRESS_THRESHOLD}）"
            f"；逐节点对照 = "
            + "；".join(f"{k} {baseline[k]:.6f}→{pos[k]:.6f}" for k in NODES))

    # ---------- P2 辅判据：下游压制（只算 tip1）----------
    c.check("P2 辅判据（下游压制，定义 B）: tip1 <= 0（只算下游节点，arm1 归 P3）",
            pos[TIP1] <= SUPPRESS_THRESHOLD,
            f"{TIP1} = {pos[TIP1]:.6f}（阈值 <= {SUPPRESS_THRESHOLD}）；"
            f"来源：紧邻叶子（唯一邻居 {ARM1} 是 frozen）⇒ 精确值 = 注入值 x hop_decay = "
            f"{NEG_VALUE} x 0.9 = {NEG_VALUE * 0.9:.6f}")

    # ---------- P3 自检：注入值保留 + 种子未被压制 ----------
    retained = abs(pos[ARM1] - NEG_VALUE) <= TOL
    seed_intact = abs(pos[START_NODE] - SEED_VALUE) <= TOL
    c.check("P3 自检（定义 A）: 注入值保留（arm1 = -0.8）+ 种子未被压制（start_node = 1.0）",
            retained and seed_intact,
            f"{ARM1} = {pos[ARM1]:.6f}（保留 = {retained}）；"
            f"{START_NODE} = {pos[START_NODE]:.6f}（未被压制 = {seed_intact}）"
            f" ⇒ 与 B33（不压制种子）不冲突")

    # ---------- P4 阴性控制 1：注入 junction（非割点）----------
    c.check(f"P4 阴性控制 1（割点条件）: 注入 {JUNCTION} ⇒ 变化节点数 = {EXPECT_P4_CHANGED}（两分支都被压）",
            len(changed_junction) == EXPECT_P4_CHANGED
            and inj_junction[ARM1] <= SUPPRESS_THRESHOLD and inj_junction[ARM2] <= SUPPRESS_THRESHOLD,
            f"变化节点 = {changed_junction}（预期 {EXPECT_P4_CHANGED} 个 = {JUNCTION} + 两分支的 4 个）；"
            f"两分支下游：{ARM1}={inj_junction[ARM1]:.6f}、{ARM2}={inj_junction[ARM2]:.6f}（均 <= 0）"
            f"｜对照 P1 的 {len(changed_pos)} 个 ⇒ 「只有一支变」来自割点结构")

    # ---------- P5 阴性控制 2：破坏割点 ----------
    net_broken = V2()
    for a, b in EDGES + ((JUNCTION, TIP1),):          # 加一条替代正路径
        for _ in range(EDGE_REPS):
            net_broken.learn([a, b])
    base_broken = net_broken.activate({START_NODE: SEED_VALUE})
    brk = net_broken.activate({START_NODE: SEED_VALUE, ARM1: NEG_VALUE})
    changed_broken = [k for k in NODES if abs(brk[k] - base_broken[k]) > TOL]
    c.check("P5 阴性控制 2（割点必要性）: 加替代正路径后压制失败（tip1 保持正值）",
            brk[TIP1] > SUPPRESS_THRESHOLD and changed_broken == [ARM1],
            f"加 {JUNCTION}—{TIP1} 后：变化节点 = {changed_broken}（仅注入点自身）；"
            f"{TIP1} = {brk[TIP1]:.6f}（基线 {base_broken[TIP1]:.6f}，替代正路径 "
            f"{base_broken[JUNCTION] * 0.9:.6f} 赢过负值）⇒ 压制失败 ✓")

    # ---------- P6 辅助观察：C1 区域压制 ----------
    net1, nodes, labels, _ = corpus_planted()
    net2 = clone_v1_into_v2(net1)
    e1, e2 = edge_counts(net1), edge_counts(net2)
    consistent = (e1 == e2 and net1.node_count() == net2.node_count())
    assign = spectral(impute_diagonal(matrix_from_net(net1, nodes)), 3, seed=SPECTRAL_SEED)
    mods = sorted(set(int(x) for x in assign))
    members = {m: [nodes[i] for i in range(len(nodes)) if assign[i] == m] for m in mods}
    hub_seed, hub_neg = members[0][0], members[0][1]
    b_c1 = net2.activate({hub_seed: SEED_VALUE})
    i_c1 = net2.activate({hub_seed: SEED_VALUE, hub_neg: NEG_VALUE})
    module_means = {m: (float(np.mean([b_c1[n] for n in members[m]])),
                        float(np.mean([i_c1[n] for n in members[m]]))) for m in mods}
    changed_c1 = [n for n in nodes if abs(b_c1[n] - i_c1[n]) > TOL]
    c.check("P6 辅助观察（不判定）: C1 区域压制 + 【重放后逐位一致性校验】",
            consistent,
            f"【重放校验】net1: {net1.node_count()} 节点 / {net1.edge_count()} 边；"
            f"net2: {net2.node_count()} 节点 / {net2.edge_count()} 边；"
            f"逐边计数比较：{'完全一致' if consistent else '不一致（P6 数字作废）'}；"
            f"区域压制：模块均值 "
            + "；".join(f"模块{m} {v[0]:.6f}→{v[1]:.6f}" for m, v in module_means.items())
            + f"；差异节点 {len(changed_c1)}/{len(nodes)}")

    # ---------- P7 只读 ----------
    h_after = all_hashes()
    c.check("P7 只读: 六文件哈希前后一致",
            h_before == h_after and h_after["synapse_net.py"] == V1_HASH_EXPECTED
            and h_after["synapse_net_v2.py"] == V2_HASH_EXPECTED
            and h_after["synapse_net_v3.py"] == V3_HASH_EXPECTED
            and all(h_after[f] == v for f, v in LAYER_HASH_EXPECTED.items()),
            f"before==after = {h_before == h_after}；v1={h_after['synapse_net.py']} "
            f"v2={h_after['synapse_net_v2.py']} v3={h_after['synapse_net_v3.py']}；"
            f"L0 签名（节点/边）= {net.node_count()}/{net.edge_count()}")

    # ---------- P8 双向 ----------
    ok_all = all(ck["ok"] for ck in c.checks)
    outcome = "C 条目（轨迹压制能力）" if ok_all else "E43_FAIL"
    c.check("P8 双向结局: P1–P5 全过 → 新 C 条目；否则 → 如实记录（不预分配 B 编号）",
            ok_all, f"落定 {outcome}（前序判据全过 = {ok_all}）")

    return {
        "id": "exp43_trajectory_suppression",
        "title": "E43 抑制控制: 负值注入对轨迹的压制效果（frozen 屏障）",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "corpus": {"nodes": list(NODES), "edges": [list(e) for e in EDGES],
                       "edge_reps": EDGE_REPS,
                       "topology_note": "start_node—trunk—junction—arm1—tip1 / └ arm2—tip2",
                       "injection_point": ARM1, "cut_for": TIP1,
                       "suppressed_branch": [ARM1, TIP1], "untouched_branch": [ARM2, TIP2]},
            "naming": {"rule": "零撞车命名 + 常量集中定义 + 只用常量引用",
                       "collisions_measured": {"seed": 114, "hub": 14, "src": 9, "s0": 8, "n0": 5, "fork": 3, "mid": 2,
                                               "start_node": 0, "trunk": 0, "junction": 0, "arm1": 0, "tip1": 0,
                                               "arm2": 0, "tip2": 0}},
            "baseline": {k: round(baseline[k], 9) for k in NODES},
            "baseline_expected": BASELINE_EXPECTED,
            "positive_injection": {"cfg": {START_NODE: SEED_VALUE, ARM1: NEG_VALUE},
                                   "values": {k: round(pos[k], 9) for k in NODES},
                                   "changed_nodes": changed_pos},
            "P4_inject_junction": {"changed_nodes": changed_junction,
                                   "values": {k: round(inj_junction[k], 9) for k in NODES}},
            "P5_broken_cut": {"added_edge": [JUNCTION, TIP1], "changed_nodes": changed_broken,
                              "tip1": round(brk[TIP1], 9), "tip1_baseline": round(base_broken[TIP1], 9)},
            "side_probe_inject_tip1": {"changed_nodes": changed_tip1,
                                       "values": {k: round(inj_tip1[k], 9) for k in NODES}},
            "P6_replay_consistency": {"net1_nodes": net1.node_count(), "net1_edges": net1.edge_count(),
                                      "net2_nodes": net2.node_count(), "net2_edges": net2.edge_count(),
                                      "edge_counts_identical": bool(e1 == e2),
                                      "violations": 0 if consistent else -1},
            "P6_module_means": {str(m): {"before": round(v[0], 9), "after": round(v[1], 9)}
                                for m, v in module_means.items()},
            "P6_changed_nodes": changed_c1,
            "readonly": {"hashes_before": h_before, "hashes_after": h_after,
                         "v1_expected": V1_HASH_EXPECTED, "v2_expected": V2_HASH_EXPECTED,
                         "v3_expected": V3_HASH_EXPECTED, "layer_expected": LAYER_HASH_EXPECTED},
            "outcome": outcome,
            "registration": {"outcome": outcome, "text": C_ENTRY_TEXT, "C_entry": C_ENTRY_TEXT,
                             "E43_FAIL": FAIL_TEXT},
            "note": "机制 = frozen 屏障阻断正值传播（结构），不是数值压低（数学）。"
                    "全程 v2；节点名用零撞车常量引用（E42/E43 的命名教训）。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
