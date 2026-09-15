"""E39: 可及性压制（reach compression）—— v3 到达值算式层的外部调制参数。

严格按 DESIGN-E39.md。**v1/v2 一字不动；v3 是平行分支（从 v2 继承）。**

预注册要点:
  - g 是 v3 的【构造参数】（与 λ / hop_decay / β 同级），不是 activate 参数
  - 扫描网格 g ∈ {0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7}（10 点）
  - reach 阈值主口径 = 0.5（与 E33 的 BIN_THR 同口径）；辅助报 0.3 / 0.5 / 0.7 三条曲线
  - P1 主判据：reach 相邻 9 对【不增】+ 端点严格下降（**不是严格递减**，见 DESIGN §1.3）
  - P2 边界：种子值恒 1.0（不可压制）；构造性依据 = strength ≤ 1.0（synapse_net.py:129-142）
  - P3 次序不变量（**v2 口径：成对方向一致性**；v1 的"序列逐位相等"口径已 FAIL 并归档为
    exp39_reach_compression_v1_FAIL.json —— 原因：C1 上 40 对精确并列在 g=0 被 tie-break
    解成与 g>0 相反的方向）；P3a' 为该 v1 口径的复现记录（不参与 PASS/FAIL）
  - P4 g=0 时等价 v2；P5 键集合与 g 无关；P6 只读；P7 双向 → B33 + C38（B33 先记）
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from experiments.checker import Checker
from experiments.exp24_module_differentiation import ALPHA, corpus_planted, impute_diagonal, matrix_from_net, spectral
from experiments.exp31_auto_scale import signature
from synapse_net_v2 import SynapseNet as V2
from synapse_net_v3 import SynapseNet as V3

GRID = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
REACH_THR = 0.5
AUX_THRS = [0.3, 0.5, 0.7]
TOL = 1e-12
REPS = 100
V1_HASH_EXPECTED = "cf99a07a0e60fd4d"
V2_HASH_EXPECTED = "ee6df589eb2f6983"
V2_FIELD_REASON = "本实验不修改 v2；字段为只读口径一致性保留（纪律⑧：无数据的字段写 null + reason，不许省略）"

# 六个文件【实测】哈希（不用 layer_hashes()——它只覆盖四个层文件，
# 对 v2/v3 会用默认值兜底，那样检查是空过的）
HASH_FILES = ["synapse_net.py", "synapse_net_v2.py", "synapse_net_v3.py",
              "order_layer.py", "context_layer.py", "chunk_layer.py"]

B33_TEXT = (
    "B33：在 max 吸收 + 种子固定（值 1.0）下，【竞争性抑制（WTA 式互斥）在到达值算式层不可实现】。"
    "① 构造性上界：单跳到达 ≤ 1.0 × strength × hop_decay = 0.9 < 1.0，任何 ≤0 的抑制项只会让它更小"
    "→ 到达值永远无法覆盖值为 1.0 的种子；"
    "② 实测：减法 / 增益 / 依赖他节点激活的 rival 项（g ∈ {0,0.1,0.5,2,10}，passes ∈ {1,3}）"
    "下种子值恒为 1.000000，被改变的只有非种子节点的可及性；"
    "③ 次序：减法与增益只改 reach，不改【成对相对方向】（v2 口径，无严格反转）。"
    "注：v1 的「序列逐位相等」口径在存在精确并列时不成立（C1 上 40 对精确并列，"
    "g=0 的 tie-break 与 g>0 的值序相反）—— 该负面记录见 DESIGN-E39 §3 P3 与 exp39 v1 FAIL JSON。"
    "与 B12 的分层：B12 说【权重层】（非负权重无 WTA），本条目说【到达值层】——"
    "即使外部负值进入算式，也不产生互斥。"
    "出口：要压制种子，必须改吸收判据（`>` 改成别的），而 L0-SPEC §2 ⑥ 明写必须保留 max 吸收 → 属越界。"
    "范围限定：限于当前 v3 形态（max 吸收 + 固定种子）；若将来改架构，须另立实验。"
)
C38_TEXT = (
    "C38：L0 v3 的到达值算式支持外部调制参数 g（构造参数，与 λ / hop_decay / β 同级）。"
    "在 26 字母链上，超阈值节点数（reach，阈值 0.5）随 g 单调不增（见 exp39 JSON 的完整序列），"
    "端点严格下降。范围限定：这是【可及性压制（reach compression）】，不是竞争性抑制 —— "
    "种子值恒为 1.0（不可被压制），【成对相对方向】不随 g 改变（v2 口径：值严格不等的对不反转；"
    "精确并列不计入 —— 见 B33 的次序负面记录）。"
    "v3 为空分支（synapse_net_v3.py，从 v2 继承）；v1/v2 冻结不变，E0–E38 全部有效。"
)
FAIL_TEXT = (
    "E39 未通过：哪条判据挂、实测值见 exp39 JSON。诊断顺序：先查 v3 与 v2 在 g=0 时的等价性（P4）"
    "→ 再查到达值算式是否保留了 max 吸收 → 再谈机制。若挂的是 P2（种子被压制），一律按越界处理。"
)


def sha16(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def all_hashes() -> dict:
    return {f: sha16(f) for f in HASH_FILES}


def build_chain(g: float | None) -> object:
    net = V3(inhibition=g) if g is not None else V2()
    for a, b in zip(ALPHA, ALPHA[1:]):
        for _ in range(REPS):
            net.learn([a, b])
    return net


def clone_into(src, g: float) -> object:
    net = V3(inhibition=g)
    for (a, b), rec in src.connections.items():
        for _ in range(int(rec["count"])):
            net.learn([a, b])
    return net


def reach(acts: dict, thr: float) -> list:
    return sorted(k for k, v in acts.items() if v >= thr)


def order_of(acts: dict) -> tuple:
    return tuple(k for k, _ in sorted(acts.items(), key=lambda kv: (-kv[1], kv[0])))


def pairwise_report(acts_by_g: dict, node_list) -> dict:
    """v2 口径：成对方向一致性。

    精确并列：|v_i − v_j| ≤ TOL   严格不等：|v_i − v_j| > TOL
    · 全程精确并列的对 → 不计入
    · 某 g 上并列、其他 g 上严格不等 → 该 g 不计入该对
    · 该对在每个"严格不等"的 g 上的方向必须一致；不一致即"反转"
    """
    all_equal, any_equal, eq_g, revs, examples = 0, 0, set(), [], []
    for i, j in itertools.combinations(node_list, 2):
        signs, saw_eq, saw_strict = set(), False, False
        for g in GRID:
            d = acts_by_g[g][i] - acts_by_g[g][j]
            if abs(d) <= TOL:
                saw_eq = True
                eq_g.add(g)
            else:
                saw_strict = True
                signs.add(1 if d > 0 else -1)
        if saw_eq:
            any_equal += 1
            if not saw_strict:
                all_equal += 1
                if len(examples) < 3:
                    examples.append([i, j, round(acts_by_g[GRID[0]][i], 9)])
        if len(signs) > 1:
            revs.append((i, j))
    return {"all_equal_pairs": all_equal, "any_equal_pairs": any_equal,
            "eq_g_points": sorted(eq_g), "reversals": revs,
            "all_equal_examples": examples,
            "pairs_total": len(node_list) * (len(node_list) - 1) // 2,
            "nodes_total": len(node_list)}


def run() -> dict:
    c = Checker()
    h_before = all_hashes()

    chain_v2 = build_chain(None)
    chain_g = {g: build_chain(g) for g in GRID}
    acts_chain = {g: chain_g[g].activate("A") for g in GRID}

    net, nodes, labels, _ = corpus_planted()
    assign = spectral(impute_diagonal(matrix_from_net(net, nodes)), 3, seed=20260911)
    mods = sorted(set(int(x) for x in assign))
    members = {m: [nodes[i] for i in range(len(nodes)) if assign[i] == m] for m in mods}
    seeds_c1 = [members[0][0], members[2][0]]
    sig_before = signature(net)
    c1_g = {g: clone_into(net, g) for g in GRID}
    acts_c1 = {g: c1_g[g].activate(seeds_c1) for g in GRID}

    # ---------- P0 文件与接口 ----------
    c.check("P0 文件与接口: v1/v2 哈希不变 + v3 继承 v2 + activate 接受 str/list/dict",
            h_before["synapse_net.py"] == V1_HASH_EXPECTED
            and issubclass(V3, V2)
            and isinstance(chain_v2.activate("A"), dict)
            and isinstance(chain_v2.activate(["A", "B"]), dict)
            and isinstance(chain_v2.activate({"A": 0.5, "B": 0.9}), dict),
            f"v1={h_before['synapse_net.py']}（期望 {V1_HASH_EXPECTED}）；v2={h_before['synapse_net_v2.py']}"
            f"（期望 {V2_HASH_EXPECTED}）；issubclass(V3,V2)={issubclass(V3, V2)}；"
            f"v3 构造参数 inhibition 生效 = {abs(chain_g[0.3].inhibition - 0.3) <= TOL}")

    # ---------- P1 主判据 ----------
    seqs = {thr: [len(reach(acts_chain[g], thr)) for g in GRID] for thr in AUX_THRS}
    seq = seqs[REACH_THR]
    adjacent_ok = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
    endpoint_ok = seq[0] > seq[-1]
    c.check("P1 主判据: reach(g) 相邻 9 对不增 + 端点严格下降（thr=0.5）",
            adjacent_ok and endpoint_ok,
            f"reach 序列 = {seq}（g={GRID}）；相邻不增 = {adjacent_ok}；端点 {seq[0]} > {seq[-1]} = {endpoint_ok}")
    for thr in AUX_THRS:
        c.check(f"P1 辅助曲线 thr={thr}", True,
                f"reach = {seqs[thr]}；单调不增 = {all(seqs[thr][i] >= seqs[thr][i+1] for i in range(len(GRID)-1))}")

    # ---------- P2 边界：种子不可压制 ----------
    chain_seed_ok = all(abs(acts_chain[g]["A"] - 1.0) <= TOL for g in GRID)
    c1_seed_ok = all(all(abs(acts_c1[g][s] - 1.0) <= TOL for s in seeds_c1) for g in GRID)
    c.check("P2 边界: 每个 g 上种子值恒 1.0（26 链 + C1 两种子）",
            chain_seed_ok and c1_seed_ok,
            f"26 链 A 值 = {[round(acts_chain[g]['A'], 6) for g in GRID]}；"
            f"C1 种子 {seeds_c1} 值 = {[(round(acts_c1[g][s], 6)) for g in GRID for s in seeds_c1][:6]}…；"
            f"链={chain_seed_ok} C1={c1_seed_ok}")

    # ---------- P3 次序不变量（v2 口径：成对方向一致性）----------
    pw_chain = pairwise_report(acts_chain, ALPHA)
    pw_c1 = pairwise_report(acts_c1, nodes)
    c.check("P3 次序不变量（v2 口径）: 成对方向一致性 —— 值严格不等的对在所有 g 上不反转",
            not pw_chain["reversals"] and not pw_c1["reversals"],
            f"26 链（{pw_chain['pairs_total']} 对）：反转对 = {len(pw_chain['reversals'])}，"
            f"全程并列对 = {pw_chain['all_equal_pairs']}（预期 0：0.9^k 全互异）；"
            f"C1（{pw_c1['pairs_total']} 对）：反转对 = {len(pw_c1['reversals'])}，"
            f"全程精确并列的对 = {pw_c1['all_equal_pairs']}，"
            f"至少一个 g 上精确并列的对 = {pw_c1['any_equal_pairs']}，并列出现的 g 点 = {pw_c1['eq_g_points']}"
            f"（v1 在 g=0 处数到的 40 对 = 这里的 {pw_c1['any_equal_pairs']}，同属 C1；"
            f"注意并列出现的 g 点不止 g=0，g>0 时模块内对称节点仍精确并列）")

    # ---------- P3a' 辅助（不参与 PASS/FAIL）：v1 口径复现（负面记录）----------
    ord_chain = {order_of(acts_chain[g]) for g in GRID}
    ord_c1 = {order_of(acts_c1[g]) for g in GRID}
    c.check("P3a' 辅助（不参与 PASS/FAIL）: v1 口径（序列逐位相等）复现 —— 26 链通过 / C1 失败",
            True,
            f"v1 口径下：26 链不同次序数 = {len(ord_chain)}（通过）；C1 不同次序数 = {len(ord_c1)}（失败）"
            f"—— 这正是 E39 v1 FAIL 的原因，见 exp39_reach_compression_v1_FAIL.json")

    # ---------- P4 向后兼容（g=0 等价 v2）----------
    names, cfg = ["Z", "A", "B", "C"], {"Z": 1.0, "A": 1.0, "B": 0.9, "C": 0.81}
    a_v2_list, a_v3_list = chain_v2.activate(names), chain_g[0.0].activate(names)
    a_v2_dict, a_v3_dict = chain_v2.activate(cfg), chain_g[0.0].activate(cfg)
    p4_ok = (all(abs(a_v2_list[k] - a_v3_list[k]) <= TOL for k in a_v2_list)
             and a_v2_list.keys() == a_v3_list.keys()
             and all(abs(a_v2_dict[k] - a_v3_dict[k]) <= TOL for k in a_v2_dict)
             and a_v2_dict.keys() == a_v3_dict.keys())
    c.check("P4 向后兼容: g=0.0 时 v3 输出 = v2（逐位；名字列表 + dict 两种形态）",
            p4_ok,
            f"名字列表逐位相同 = {all(abs(a_v2_list[k]-a_v3_list[k]) <= TOL for k in a_v2_list)}；"
            f"dict 逐位相同 = {all(abs(a_v2_dict[k]-a_v3_dict[k]) <= TOL for k in a_v2_dict)}")

    # ---------- P5 作用域分账（键集合与 g 无关）----------
    keys_chain = {frozenset(acts_chain[g]) for g in GRID}
    keys_c1 = {frozenset(acts_c1[g]) for g in GRID}
    c.check("P5 作用域分账: 对任意 g，activate 返回的键集合不变（不剔除节点）",
            len(keys_chain) == 1 and len(keys_c1) == 1,
            f"26 链键集合 = {sorted(acts_chain[0.0])}（|keys|={len(acts_chain[0.0])}，不同 g 的键集合数 = {len(keys_chain)}）；"
            f"C1 |keys|={len(acts_c1[0.0])}，不同 g 的键集合数 = {len(keys_c1)}")

    # ---------- P6 只读 ----------
    h_after = all_hashes()
    layer_files = ["synapse_net.py", "order_layer.py", "context_layer.py", "chunk_layer.py"]
    v3_hash = h_after["synapse_net_v3.py"]
    p6_ok = (h_before == h_after and all(h_after[f] == h_before[f] for f in layer_files)
             and h_after["synapse_net.py"] == V1_HASH_EXPECTED
             and h_after["synapse_net_v2.py"] == V2_HASH_EXPECTED
             and signature(net) == sig_before)
    c.check("P6 只读: 四层文件 sha256 前后一致 + L0 签名前后一致 + v2 字段保留（含 reason）+ v3 首记录",
            p6_ok,
            f"layer before==after = {h_before == h_after}；v1={h_after['synapse_net.py']}；"
            f"v2={h_after['synapse_net_v2.py']}（字段保留）；v3={v3_hash}（首记录）；"
            f"L0 签名一致 = {signature(net) == sig_before}（{net.node_count()} 节点 / {net.edge_count()} 边）")

    # ---------- P7 双向 ----------
    ok_all = all(ck["ok"] for ck in c.checks)
    outcome = "B33 + C38" if ok_all else "E39_FAIL"
    text = (B33_TEXT + "\n\n" + C38_TEXT) if ok_all else FAIL_TEXT
    c.check("P7 双向结局: P0–P6 全过 → B33 + C38（B33 先记）；否则 → 如实记录",
            outcome in ("B33 + C38", "E39_FAIL"),
            f"落定 {outcome}（前序判据全过 = {ok_all}）")

    return {
        "id": "exp39_reach_compression",
        "title": "E39 可及性压制（reach compression）: v3 到达值算式层的外部调制参数 g",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "grid": GRID,
            "reach_threshold_main": REACH_THR,
            "reach_curves": {str(thr): seqs[thr] for thr in AUX_THRS},
            "reach_curve_main": seq,
            "chain_seed_values_by_g": {str(g): round(acts_chain[g]["A"], 9) for g in GRID},
            "chain_profile_by_g": {str(g): {k: round(acts_chain[g][k], 6) for k in ALPHA[:8]} for g in GRID},
            "c1_two_seed": {
                "seeds": seeds_c1,
                "values_by_g": {str(g): {s: round(acts_c1[g][s], 9) for s in seeds_c1} for g in GRID},
                "min_value_by_g": {str(g): round(min(acts_c1[g].values()), 9) for g in GRID},
            },
            "ordering_invariant_v2": {
                "criterion": "成对方向一致性（值严格不等的对在所有 g 上不反转）",
                "corpus_note": "每个数都标了语料归属：chain = 26 字母链；c1 = E24/E25 的 C1。"
                               "26 链的 0 是预期的（1.0, 0.9, 0.81, …, 0.9^25 全互异）；"
                               "并列只出现在 C1（模块内对称节点 + 两个同值种子）。",
                "chain": {"nodes_total": pw_chain["nodes_total"], "pairs_total": pw_chain["pairs_total"],
                          "reversals": len(pw_chain["reversals"]),
                          "all_equal_pairs": pw_chain["all_equal_pairs"],
                          "any_equal_pairs": pw_chain["any_equal_pairs"],
                          "eq_g_points": pw_chain["eq_g_points"],
                          "all_equal_examples": pw_chain["all_equal_examples"]},
                "c1": {"nodes_total": pw_c1["nodes_total"], "pairs_total": pw_c1["pairs_total"],
                       "reversals": len(pw_c1["reversals"]),
                       "all_equal_pairs": pw_c1["all_equal_pairs"],
                       "any_equal_pairs": pw_c1["any_equal_pairs"],
                       "eq_g_points": pw_c1["eq_g_points"],
                       "all_equal_examples": pw_c1["all_equal_examples"]},
            },
            "ordering_invariant_v1_negative_record": {
                "criterion": "序列逐位相等（v1 原口径，已 FAIL）",
                "chain_distinct_orders": len(ord_chain),
                "c1_distinct_orders": len(ord_c1),
                "note": "v1 FAIL 的原因：C1 的 g=0 处 40 对精确并列被名字 tie-break 解成与 g>0 相反的方向",
            },
            "backward_compat_g0": {
                "names_form_bitwise_equal": all(abs(a_v2_list[k] - a_v3_list[k]) <= TOL for k in a_v2_list),
                "dict_form_bitwise_equal": all(abs(a_v2_dict[k] - a_v3_dict[k]) <= TOL for k in a_v2_dict),
            },
            "scope_key_sets": {"chain_n_keys": len(acts_chain[0.0]), "c1_n_keys": len(acts_c1[0.0]),
                               "chain_keys_g_independent": len(keys_chain) == 1,
                               "c1_keys_g_independent": len(keys_c1) == 1},
            "readonly": {
                "hashes_before": h_before, "hashes_after": h_after, "layer_files": layer_files,
                "v1_expected": V1_HASH_EXPECTED, "v2_expected": V2_HASH_EXPECTED,
                "synapse_net_v2.py": {"sha256_16": h_after["synapse_net_v2.py"], "reason": V2_FIELD_REASON},
                "synapse_net_v3.py": {"sha256_16": v3_hash, "role": "平行分支（E39 起），从 v2 继承"},
                "L0_signature_unchanged": bool(signature(net) == sig_before),
            },
            "outcome": outcome,
            "registration": {"outcome": outcome, "text": text, "B33": B33_TEXT, "C38": C38_TEXT,
                             "E39_FAIL": FAIL_TEXT, "order_note": "B33 先记，C38 后记（见 DESIGN-E39 §4）"},
            "note": "g 是 v3 的构造参数（与 λ/hop_decay/β 同级），不是 activate 参数；"
                    "v3 从 v2 继承、整体复制 activate + 1 行调制；max 吸收判据一字未动。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
