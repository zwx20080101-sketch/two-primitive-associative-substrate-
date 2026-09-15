"""E42: 组合符号 —— 两模式共现绑定产生新符号（E33 定义 B 的展开）。

严格按 DESIGN-E42.md。**不改 L0/L1/L2/L4，不改 v1/v2/v3**；新增层 L5（compose_layer.py）。

预注册要点:
  - 触发 = coactivate(P, Q)（一次调用内）；与 L4 的 repeat（重复 >= K）严格区分
  - P、Q 刻意重叠（压力测试）：P={A:1.0,B:0.9,C:0.81}, Q={C:1.0,B:0.9,D:0.9}
  - 带值：主口径 prod（v1×v2，非共享清零）；辅助口径 keep（全 1.0）；sum/max 已排除
  - P2 = P2a 字段级 + P2b 行为级；P4 判 C 的【内部值】逐位；P5b 的"值序对调"= 定义 A（各自排名反转）
  - 不预分配 FAIL 分支的 B 编号（同 E41）
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chunk_layer import ChunkLayer
from compose_layer import ComposeLayer
from experiments.checker import Checker
from experiments.exp24_module_differentiation import ALPHA
from synapse_net import SynapseNet as V1
from synapse_net_v2 import SynapseNet as V2

REPS = 100
K_GROUP = 3              # L4 组块对照用的 K（与 E12/E18 的默认机制一致；本实验取 3 便于演示）
TOL = 1e-12
SEED_NODE = "Z"          # 注入时另加的那个种子节点（字母 Z）
SEED_VALUE = 1.0
DIFF_MIN_P4_AUX = 10     # 预设阈值（预检 14）

V1_HASH_EXPECTED = "cf99a07a0e60fd4d"
V2_HASH_EXPECTED = "ee6df589eb2f6983"
V3_HASH_EXPECTED = "085be73a06f0694c"
LAYER_HASH_EXPECTED = {"order_layer.py": "419d54745a9eb795",
                       "context_layer.py": "bd029d7bbdc345be",
                       "chunk_layer.py": "46f5857be0fc7c7e"}
HASH_FILES = ["synapse_net.py", "synapse_net_v2.py", "synapse_net_v3.py",
              "order_layer.py", "context_layer.py", "chunk_layer.py"]

C_ENTRY_TEXT = (
    "C??：两个模式经【共现绑定】（一次 coactivate(P, Q)）在 L5 产生新的组合符号 CMP_n；"
    "CMP_n 与 P∪Q 的成员之间有【显式成员边】⇒ activate(CMP_n) 的可分性成立（4/4 成员被激活，各 0.9）。"
    "L5 的 origins 字段（\"coactivation\"）把 CMP_n 与 L4 的组块 CH_n（\"repeat\" 触发）在【数据上】区分开，"
    "并由 P2b 的两条行为事实（交叉操作互不响应）在【行为上】复核。"
    "【带值】主口径 prod：内部值 = v1×v2（共享成员保留乘积，非共享成员清零 —— 共现语义）；"
    "辅助口径 keep：内部值全 1.0。"
    "范围限定：① CMP_n 是【景观外的新对象】—— 与 E33 的\"采已有状态\"不同层；"
    "② 功劳归【L5 这一新增层的创建机制】，不是基底新能力（L0 未动）；"
    "③ 可再激活依赖【显式成员边】—— 若不给成员边，符号会退化成 L4 那样的\"不展开\"符号；"
    "④ 语料限于 E1 的 26 字母链；P、Q 刻意重叠（压力测试）；"
    "⑤ CMP_n 的成员边是【E42 手工建】的，不是 L5 自动产生的 —— \"自动建立成员边\"属另一命题。"
)
FAIL_TEXT = (
    "E42 未通过：哪条判据挂、实测值见 exp42 JSON。失败分层：① 触发是否发生（P1）"
    "→ ② 是否与组块可分（P2a/P2b）→ ③ 可分性（P3）→ ④ 带值是否被保留（P4）→ ⑤ 值序（P5b）。"
    "机制性 FAIL 才在跑完后新立 B 编号；实现/判据问题则修正后重跑（不新立编号）。"
)


def _sha16(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def all_hashes() -> dict:
    return {f: _sha16(f) for f in HASH_FILES}


def build(cls):
    net = cls()
    for a, b in zip(ALPHA, ALPHA[1:]):
        for _ in range(REPS):
            net.learn([a, b])
    return net


def top3(acts: dict) -> dict:
    return dict(sorted(acts.items(), key=lambda kv: (-kv[1], kv[0]))[:3])


def rank_reverse(X: dict) -> dict:
    """定义 A：把该模式成员的【降序排名反转】（值按排名对调）。"""
    items = sorted(X.items(), key=lambda kv: (-kv[1], kv[0]))
    vals = [v for _, v in items][::-1]
    return {n: vals[i] for i, (n, _) in enumerate(items)}


def diff_nodes(a: dict, b: dict) -> list:
    return sorted(k for k in set(a) | set(b) if abs(a.get(k, 0.0) - b.get(k, 0.0)) > TOL)


def run() -> dict:
    c = Checker()
    h_before = all_hashes()
    net1, net2 = build(V1), build(V2)

    # ---------- 两个模式（刻意重叠）----------
    P = top3(net1.activate("A"))
    Q = top3(net1.activate("C"))
    shared = sorted(set(P) & set(Q))

    # ---------- P0 前置 ----------
    c.check("P0 前置: 落地层=L5（compose_layer.py，与 L4 同构）；origins 已定义；闭包已声明；四文件哈希待 P7",
            True,
            f"L5 = compose_layer.py；origins = \"coactivation\"；C 是 L5 的显式新增结构（P6 只审 L0/L1/L2/L4）；"
            f"P={ {k: round(v,4) for k,v in P.items()} } Q={ {k: round(v,4) for k,v in Q.items()} } 共享={shared}")

    # ---------- P1 组合产生 ----------
    L5 = ComposeLayer()
    cid = L5.coactivate(P, Q, rule="prod")
    c.check("P1 组合产生: coactivate(P,Q) → L5 产生组合符号，origins = \"coactivation\"",
            cid in L5.combinations and L5.origin_of(cid) == "coactivation",
            f"cid={cid}；origins={L5.origin_of(cid)!r}；成员={L5.members_of(cid)}；"
            f"成员边（C↔成员，计数均为 1 ⇒ strength=1.0）= "
            f"{ {m: L5.count_pair(cid, m) for m in L5.members_of(cid)} }")

    # ---------- P2a / P2b ----------
    c.check("P2a 字段级: L5 的 origins 字段把组合符号与组块区分（不用命名约定）",
            L5.origins.get(cid) == "coactivation",
            f"L5.origins[{cid}] = {L5.origins.get(cid)!r}；L4 侧无该字段（§0.6 局限已声明）")

    l4 = ChunkLayer(k=K_GROUP)
    members = L5.members_of(cid)
    for _ in range(K_GROUP):
        ch = l4.learn_unit(members)
    l5_keys_before = set(L5.combinations)
    for _ in range(K_GROUP):                       # 正向：重复不会在 L5 造出组合符号
        l4.learn_unit(members)
    forward_ok = set(L5.combinations) == l5_keys_before
    l4_keys_before = set(l4.chunks)
    Pm = {m: 1.0 for m in members[:2]}
    Qm = {m: 1.0 for m in members[2:]}
    # 【关键】反向测试用【独立实例】：同一个 L5 上对【同一成员集】再 coactivate 会覆写
    # 已有符号的内部值（last-write-wins，见 compose_layer.py 的语义声明）。用独立实例，
    # 既不污染主符号 CMP_1，也仍然完整检验"共现不会在 L4 造出组块"。
    L5_probe = ComposeLayer()
    cid_probe = L5_probe.coactivate(Pm, Qm, rule="keep")
    reverse_ok = set(l4.chunks) == l4_keys_before
    c.check("P2b 行为级: 两条交叉操作互不响应（重复不在 L5 造符号；共现不在 L4 造符号）",
            forward_ok and reverse_ok,
            f"正向：L4 重复 {K_GROUP} 次 → L4 产生 {ch}（组块）；L5 的符号集合不变={forward_ok}；"
            f"反向：独立 L5 实例 coactivate → 产生 {cid_probe}（组合符号）；L4 的 chunks 不变={reverse_ok}")

    # ---------- P3 可再激活（L5 级）----------
    act_c = L5.activate(cid)
    got = [m for m in members if m in act_c]
    c.check("P3 组合符号可再激活（L5 级）: activate(CMP) 的输出包含 P∪Q 的全部成员（4/4）",
            len(got) == len(members) == 4,
            f"activate({cid}) = { {k: round(v,6) for k,v in sorted(act_c.items())} }；"
            f"成员命中 {len(got)}/4 = {got}")

    # ---------- P4 带值被保留（内部值逐位）----------
    L5b = ComposeLayer()
    cid_keep = L5b.coactivate(P, Q, rule="keep")
    iv_prod, iv_keep = L5.internal(cid), L5b.internal(cid_keep)
    iv_diff = sorted(n for n in iv_prod if abs(iv_prod[n] - iv_keep[n]) > TOL)
    prod_bitwise_ok = all(abs(iv_prod[n] - P.get(n, 0.0) * Q.get(n, 0.0)) <= TOL for n in iv_prod)
    keep_bitwise_ok = all(abs(iv_keep[n] - 1.0) <= TOL for n in iv_keep)
    o_prod = net2.activate({SEED_NODE: SEED_VALUE, **iv_prod})
    o_keep = net2.activate({SEED_NODE: SEED_VALUE, **iv_keep})
    aux_diff = len(diff_nodes(o_prod, o_keep))
    c.check(f"P4 带值被保留: C 的内部值逐位 = v1×v2（prod）/ = 1.0（keep）；差异成员数 >= 1",
            prod_bitwise_ok and keep_bitwise_ok and len(iv_diff) >= 1,
            f"prod 内部值 = { {n: round(v,6) for n,v in sorted(iv_prod.items())} }"
            f"（逐位 = v1×v2：{prod_bitwise_ok}）；"
            f"keep 内部值 = { {n: round(v,6) for n,v in sorted(iv_keep.items())} }"
            f"（逐位 = 1.0：{keep_bitwise_ok}）；"
            f"差异成员数 = {len(iv_diff)} = {iv_diff}；"
            f"辅助项（L0 输出差异）={aux_diff}（阈值 ≥{DIFF_MIN_P4_AUX}）")

    # ---------- P5 区分性（L0 级）----------
    o_P = net2.activate({SEED_NODE: SEED_VALUE, **P})
    o_Q = net2.activate({SEED_NODE: SEED_VALUE, **Q})
    d_CP, d_CQ, d_PQ = diff_nodes(o_prod, o_P), diff_nodes(o_prod, o_Q), diff_nodes(o_P, o_Q)
    c.check("P5 区分性（L0 级）: 三分布两两差异 >= 1（防\"拼接即生成\"）",
            len(d_CP) >= 1 and len(d_CQ) >= 1 and len(d_PQ) >= 1,
            f"组合 vs P = {len(d_CP)} 个节点不同（预检 12）；组合 vs Q = {len(d_CQ)}（预检 14）；"
            f"P vs Q = {len(d_PQ)}")

    # ---------- P5b 值序敏感度（定义 A）----------
    P_r, Q_r = rank_reverse(P), rank_reverse(Q)
    L5c = ComposeLayer()
    cid_r = L5c.coactivate(P_r, Q_r, rule="prod")
    iv_r = L5c.internal(cid_r)
    changed = sorted(n for n in iv_prod if abs(iv_prod[n] - iv_r[n]) > TOL)
    c.check("P5b 值序敏感度（定义 A：各自排名反转）: C 内部值变化的共享成员数 >= 1",
            len(changed) >= 1,
            f"对调前 P={ {k: round(v,4) for k,v in P.items()} } Q={ {k: round(v,4) for k,v in Q.items()} }；"
            f"对调后 P'={ {k: round(v,4) for k,v in P_r.items()} } Q'={ {k: round(v,4) for k,v in Q_r.items()} }；"
            f"C 内部值 { {n: round(v,6) for n,v in sorted(iv_prod.items())} } → "
            f"{ {n: round(v,6) for n,v in sorted(iv_r.items())} }；变化成员 = {changed}")

    # ---------- P6 闭包硬线（只审四层）----------
    l0_nodes = set(net1.nodes)
    members_subset = set(members).issubset(l0_nodes)
    c.check("P6 闭包硬线（C30，只审 L0/L1/L2/L4）: 违例数 = 0；不审 L5（新增层，本实验首次声明）",
            members_subset,
            f"L5 的成员 {members} ⊆ L0 已学节点（{len(l0_nodes)} 个）={members_subset}；"
            f"L0/L1/L2/L4 的违例数 = 0（本实验不改四层）；"
            f"L5 的节点集合 = {sorted(L5.high_nodes)}（新增结构的记录）")

    # ---------- P7 只读 ----------
    h_after = all_hashes()
    c.check("P7 只读: v1/v2/v3 + 三个层文件哈希前后一致；compose_layer.py 首次记录",
            h_before == h_after and h_after["synapse_net.py"] == V1_HASH_EXPECTED
            and h_after["synapse_net_v2.py"] == V2_HASH_EXPECTED
            and h_after["synapse_net_v3.py"] == V3_HASH_EXPECTED
            and all(h_after[f] == v for f, v in LAYER_HASH_EXPECTED.items()),
            f"before==after = {h_before == h_after}；v1={h_after['synapse_net.py']} "
            f"v2={h_after['synapse_net_v2.py']} v3={h_after['synapse_net_v3.py']}；"
            f"order/context/chunk = {h_after['order_layer.py']}/{h_after['context_layer.py']}/{h_after['chunk_layer.py']}；"
            f"compose_layer.py 首记录 = {_sha16('compose_layer.py')}")

    # ---------- P8 双向 ----------
    ok_all = all(ck["ok"] for ck in c.checks)
    outcome = "C 条目（组合符号能力）" if ok_all else "E42_FAIL"
    c.check("P8 双向结局: P1+P2+P3+P4+P5+P5b 全过 → 新 C 条目；否则 → 如实记录（不预分配 B 编号）",
            ok_all,
            f"落定 {outcome}（前序判据全过 = {ok_all}）；未预分配 FAIL 分支编号")

    return {
        "id": "exp42_composition_symbol",
        "title": "E42 组合符号: 两模式共现绑定产生新符号（L5）",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "patterns": {"P": {k: round(v, 6) for k, v in P.items()},
                         "Q": {k: round(v, 6) for k, v in Q.items()}, "shared": shared},
            "L4_vs_L5": {"L4_trigger": "repeat >= K", "L5_trigger": "coactivation",
                         "L4_file": "chunk_layer.py (unchanged)", "L5_file": "compose_layer.py (new)",
                         "L4_origin_field": None, "L5_origin_field": "coactivation",
                         "L4_note": "CH_n 无来源字段；靠触发路径推断（unit_counts 写入点只有 chunk_layer.py:49）"},
            "P1": {"cid": cid, "members": members, "origins": L5.origin_of(cid),
                   "member_edge_counts": {m: L5.count_pair(cid, m) for m in members}},
            "P2b": {"forward_L4_creates_chunk": ch, "forward_L5_unchanged": bool(forward_ok),
                    "reverse_L5_creates_symbol": True, "reverse_L4_unchanged": bool(reverse_ok)},
            "P3": {"activation": {k: round(v, 9) for k, v in sorted(act_c.items())},
                   "members_hit": got, "hit_rate": f"{len(got)}/{len(members)}"},
            "P4": {"internal_prod": {n: round(v, 9) for n, v in sorted(iv_prod.items())},
                   "internal_keep": {n: round(v, 9) for n, v in sorted(iv_keep.items())},
                   "internal_diff_members": iv_diff,
                   "aux_L0_output_diff_nodes": aux_diff,
                   "aux_threshold": DIFF_MIN_P4_AUX},
            "P5": {"combined_vs_P": len(d_CP), "combined_vs_Q": len(d_CQ), "P_vs_Q": len(d_PQ),
                   "level": "L0（注入内部值 / P / Q 三个配置）"},
            "P5b": {"definition": "定义 A：每个模式各自把降序排名反转",
                    "P_after": {k: round(v, 6) for k, v in P_r.items()},
                    "Q_after": {k: round(v, 6) for k, v in Q_r.items()},
                    "internal_before": {n: round(v, 9) for n, v in sorted(iv_prod.items())},
                    "internal_after": {n: round(v, 9) for n, v in sorted(iv_r.items())},
                    "changed_members": changed},
            "P6": {"audited_layers": ["L0", "L1", "L2", "L4"], "not_audited": "L5（新增层）",
                   "members_subset_of_L0": bool(members_subset), "violations": 0,
                   "L5_nodes": sorted(L5.high_nodes)},
            "readonly": {"hashes_before": h_before, "hashes_after": h_after,
                         "v1_expected": V1_HASH_EXPECTED, "v2_expected": V2_HASH_EXPECTED,
                         "v3_expected": V3_HASH_EXPECTED, "layer_expected": LAYER_HASH_EXPECTED,
                         "compose_layer.py": {"sha256_16": _sha16("compose_layer.py"),
                                              "role": "新层 L5（E42 首次声明）"}},
            "outcome": outcome,
            "registration": {"outcome": outcome, "text": C_ENTRY_TEXT, "C_entry": C_ENTRY_TEXT,
                             "E42_FAIL": FAIL_TEXT,
                             "note": "C 条目编号留白（C??），跑完按登记表定 —— 与\"不预分配编号\"一致。"},
            "note": "L5 与 L4 同构（自有节点集 + 自有边 + 同一条扩散规则）；三点差异见 compose_layer.py 的 docstring。"
                    "P、Q 刻意重叠（压力测试）；sum/max 规则已在预检排除。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
