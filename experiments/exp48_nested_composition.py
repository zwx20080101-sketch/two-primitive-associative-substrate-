"""E48: 组合的嵌套（combine(C, R) → D）。

严格按 DESIGN-E48.md。**不改任何已验收文件；L0 一字不动；不新增层。**

预注册要点:
  - symbol_a = coactivate(P, Q)；symbol_b = coactivate({symbol_a: 1.0}, R)
  - rule 主口径 = keep（全 1.0）—— 因为 prod 在【不相交输入】下退化（D 的内部值全 0）
  - 判据顺序：P1（输入层是否出现新量）【先判】→ P2（级联）→ P3（0.9^d）→ P4/P5（结构）
  - P2 的"出现" = key 在输出字典里且值 > 0；P3 判精确值 0.9^d
  - 刻意设计：四个符号的成员集不重叠（避免触发 B35 的 last-write-wins）

【命名（DESIGN §0.4 的零撞车清单）】符号的局部变量名用 symbol_a / symbol_b / symbol_c /
  symbol_d —— 不用 C / D：语料节点里就有一个叫 "C" 的节点，`C = coactivate(...)` 会与它同名
  （E42 型撞车）。除改名外另加【跑前断言】（符号 id 不得撞 L0 节点名），与 P0 的跑后审计双保险。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from compose_layer import ComposeLayer
from experiments.checker import Checker
from experiments.exp24_module_differentiation import ALPHA
from synapse_net_v2 import SynapseNet as V2

# --- 语料与模式（常量集中定义；避免裸字符串散落）---------------------------------
SEED_A = "A"          # P 的来源种子
SEED_C = "C"          # Q 的来源种子
SEED_Z = "Z"          # R 的来源种子
SEED_M = "M"          # 二层嵌套用的第二个内层模式的来源种子
REPS = 100
HOP_DECAY = 0.9
TOL = 1e-12
RULE_MAIN = "keep"
RULE_AUX = "prod"

# --- 一层组合符号的 L0 成员（预检实测；P3 用 d=1）----------------------------
SYMBOL_A_L0_MEMBERS = ("A", "B", "C", "D")

HASH_FILES = ["synapse_net.py", "synapse_net_v2.py", "synapse_net_v3.py",
              "order_layer.py", "context_layer.py", "chunk_layer.py",
              "compose_layer.py", "experiments/exp34_working_memory.py"]
EXPECTED = {"synapse_net.py": "cf99a07a0e60fd4d",
            "synapse_net_v2.py": "ee6df589eb2f6983",
            "synapse_net_v3.py": "085be73a06f0694c",
            "order_layer.py": "419d54745a9eb795",
            "context_layer.py": "bd029d7bbdc345be",
            "chunk_layer.py": "46f5857be0fc7c7e",
            "compose_layer.py": "81e350497e9e9960",
            "experiments/exp34_working_memory.py": "e821f0df7ca165c3"}

C_ENTRY_TEXT = (
    "C??：E42 的组合符号可作为模式【再次传入 combine】（**符号身份进入组合，而非它的成员**）；"
    "嵌套不膨胀层级（D 仍在 L5 同一实例，不需要 L6）；L0 可及强度随嵌套深度按 **0.9^d** 衰减"
    "（d=1 → 0.9、d=2 → 0.81，实测逐位吻合）。"
    "范围限定：① 这是 L5 的【层内递归】，不是基底新能力（L0 一字未动，节点/边签名逐位不变）；"
    "② 衰减律 0.9^d 是 hop_decay 的**数学必然**，不是涌现；"
    "③ 不声称「系统学会了递归」、「等于泛化」、「产生新机制」；"
    "④ 语料限于 E1 的 26 字母链；成员集刻意不重叠（避免 B35 覆写）。"
)
FAIL_TEXT = (
    "E48 未通过：哪条判据挂、实测值见 exp48 JSON。"
    "诊断：P1 挂（输入仍是投影拼接）→ 同 E47，可行但无新内容，不立条目；"
    "P2 挂 → 级联未发生；P2 过而 P3 挂 → 结构对、强度错（先查 L5 成员边与 hop_decay，再谈机制）。"
)


def _sha16(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def all_hashes() -> dict:
    return {f: _sha16(f) for f in HASH_FILES}


def top3(acts: dict) -> dict:
    return dict(sorted(acts.items(), key=lambda kv: (-kv[1], kv[0]))[:3])


def visible(acts: dict, node: str) -> bool:
    """P2 的"出现"定义：key 在输出字典里，且值 > 0。"""
    return node in acts and acts[node] > 0


def run() -> dict:
    c = Checker()
    h_before = all_hashes()

    net = V2()
    for a, b in zip(ALPHA, ALPHA[1:]):
        for _ in range(REPS):
            net.learn([a, b])
    l0_nodes = set(net.nodes)
    L5 = ComposeLayer()

    P = top3(net.activate({SEED_A: 1.0}))
    Q = top3(net.activate({SEED_C: 1.0}))
    R = top3(net.activate({SEED_Z: 1.0}))

    # ---------- 组合：一层 symbol_a / 二层 symbol_b（主口径 keep）----------
    symbol_a = L5.coactivate(P, Q, rule=RULE_MAIN)
    inp_keys = {symbol_a} | set(R)
    symbol_b = L5.coactivate({symbol_a: 1.0}, R, rule=RULE_MAIN)

    # 二层结构样例：symbol_c = combine(R, S)，symbol_d = combine(symbol_a, symbol_c)
    S = top3(net.activate({SEED_M: 1.0}))
    symbol_c = L5.coactivate(R, S, rule=RULE_MAIN)
    symbol_d = L5.coactivate({symbol_a: 1.0}, {symbol_c: 1.0}, rule=RULE_MAIN)

    # 【防护：跑前拦截】符号 id 不得与语料节点名同名 —— 与 P0 的跑后审计构成双保险（E42 教训）
    _symbol_ids = (symbol_a, symbol_b, symbol_c, symbol_d)
    assert set(_symbol_ids).isdisjoint(l0_nodes), (
        f"符号 id 撞车 L0 节点名：{sorted(set(_symbol_ids) & l0_nodes)}")
    assert "C" not in _symbol_ids, f"符号 id 撞车语料节点 'C'：{_symbol_ids}"

    acts_a = L5.activate(symbol_a)
    acts_b = L5.activate(symbol_b)

    # ---------- P0 前置 ----------
    sym_l0_overlap = sorted(s for s in L5.high_nodes if s in l0_nodes and s.startswith("CMP_"))
    c.check("P0 前置: L5 接口对输入类型无检查（只取键）+ 语料就绪 + 八文件哈希",
            set(L5.members_of(symbol_b)) == {symbol_a} | set(R)
            and h_before["synapse_net_v2.py"] == EXPECTED["synapse_net_v2.py"]
            and h_before["compose_layer.py"] == EXPECTED["compose_layer.py"],
            f"coactivate(P,Q) 的成员 = set(P)|set(Q)（未校验键的类型）；"
            f"symbol_b 的成员 = {L5.members_of(symbol_b)}；v2={h_before['synapse_net_v2.py']}、"
            f"compose_layer={h_before['compose_layer.py']}；"
            f"【撞车审计（E42 教训）】符号 id 与 L0 节点集的交集 = {sym_l0_overlap}（空 = 无撞车）")

    # ---------- P1 输入层：出现新量（先判）----------
    l5_symbols_in_input = sorted(n for n in inp_keys if n in L5.high_nodes and n not in l0_nodes)
    c.check("P1 输入层（本实验的成立条件）: combine 的输入里含【至少一个 L5 符号】",
            len(l5_symbols_in_input) >= 1,
            f"输入键集 = {sorted(inp_keys)}；其中 L5 符号（∈ L5.high_nodes 且 ∉ L0 节点集）= "
            f"{l5_symbols_in_input}；一层符号 = {symbol_a} 在 L0 里吗 = {symbol_a in l0_nodes}")

    # ---------- P2 输出层：级联可分（两跳）----------
    need = [symbol_a] + list(SYMBOL_A_L0_MEMBERS)
    ok2 = all(visible(acts_b, n) for n in need)
    c.check("P2 输出层: activate(symbol_b) 里 CMP_1 与 A/B/C/D 都【出现】（key 在且值 > 0）",
            ok2,
            f"判据对象 = {need}；实际值 = "
            + "；".join(f"{n}={acts_b.get(n, float('nan')):.6f}" for n in need)
            + f"；一跳层 X/Y/Z = "
            + "；".join(f"{n}={acts_b.get(n, float('nan')):.6f}" for n in ("X", "Y", "Z"))
            + f"；【报告②供 P3 对照】CMP_1={acts_b.get(symbol_a, float('nan')):.6f}、"
            + "A/B/C/D=" + "/".join(f"{acts_b.get(n, float('nan')):.6f}" for n in SYMBOL_A_L0_MEMBERS))

    # ---------- P3 定量关系 0.9^d（数学必然）----------
    d1 = acts_a.get("A", float("nan"))         # 一层符号的 L0 成员（d=1）
    d2 = acts_b.get("A", float("nan"))         # 二层符号的 L0 成员（d=2）
    exp1, exp2 = HOP_DECAY ** 1, HOP_DECAY ** 2
    c.check("P3 定量关系: 嵌套深度 d 的 L0 可及强度 = 0.9^d（|实际 − 0.9^d| ≤ 1e-12）",
            abs(d1 - exp1) <= TOL and abs(d2 - exp2) <= TOL,
            f"d=1: 实际 {d1:.12f} vs 0.9^1={exp1:.12f}（差 {abs(d1-exp1):.3e}）；"
            f"d=2: 实际 {d2:.12f} vs 0.9^2={exp2:.12f}（差 {abs(d2-exp2):.3e}）"
            f"【标注】这是 hop_decay 的数学必然，不是涌现")

    # ---------- P4 层级不膨胀 ----------
    syms = L5.combination_ids()
    c.check("P4 层级不膨胀: 一层/二层/二层样例的 4 个符号都在【同一个 L5 实例】⇒ 不需要 L6",
            all(s in syms for s in (symbol_a, symbol_b, symbol_c, symbol_d)),
            f"L5 的全部符号 = {syms}；每个符号的成员集 = "
            + "；".join(f"{s}:{L5.members_of(s)}" for s in syms)
            + f"；实例数 = 1（全程只用一个 ComposeLayer）")

    # ---------- P5 B35 不触发（刻意设计）----------
    sets = [frozenset(L5.members_of(s)) for s in syms]
    c.check("P5 B35 不触发（刻意设计）: 各符号成员集两两不同 ⇒ 无 last-write-wins 覆写",
            len(set(sets)) == len(sets),
            f"成员集 = {[sorted(s) for s in sets]}；两两不同 = {len(set(sets)) == len(sets)}"
            f"（设计声明：刻意选不重叠的成员集，避免嵌套与覆写两件事混在一起）")

    # ---------- 辅助：prod 口径的退化 + 级联一致性 ----------
    # 【比较口径】"rule 不影响级联"这一条必须【拓扑匹配】：同一符号集、只差 rule。
    # 若两实例的符号集不同（如只建一层/二层 vs 建满 4 个符号），比较的就是"拓扑"而非"rule"。
    # 本文件第一版正是这么写的（aux 实例只含两个符号），报出 False —— 定位为拓扑差异
    # （主实例多出 CMP_3/CMP_4 与 L/M/N），与 rule 无关；此处改为一对【拓扑匹配】的实例。
    L5_m = ComposeLayer()
    symbol_a_m = L5_m.coactivate(P, Q, rule=RULE_AUX)
    symbol_b_m = L5_m.coactivate({symbol_a_m: 1.0}, R, rule=RULE_AUX)
    symbol_c_m = L5_m.coactivate(R, S, rule=RULE_AUX)
    L5_m.coactivate({symbol_a_m: 1.0}, {symbol_c_m: 1.0}, rule=RULE_AUX)
    aux_matched = (L5_m.activate(symbol_b_m) == L5.activate(symbol_b))

    # 最小见证：两个【只含一层/二层符号】的干净实例，只差 rule（隔离拓扑与 rule 两个变量）
    L5_keep, L5_prod = ComposeLayer(), ComposeLayer()
    symbol_a_keep = L5_keep.coactivate(P, Q, rule=RULE_MAIN)
    symbol_b_keep = L5_keep.coactivate({symbol_a_keep: 1.0}, R, rule=RULE_MAIN)
    symbol_a_prod = L5_prod.coactivate(P, Q, rule=RULE_AUX)
    symbol_b_prod = L5_prod.coactivate({symbol_a_prod: 1.0}, R, rule=RULE_AUX)
    aux_clean = (L5_keep.activate(symbol_b_keep) == L5_prod.activate(symbol_b_prod))

    c.check("辅助（不判定）: prod 口径下二层符号的内部值全 0（不相交输入）但级联结果与 keep 逐位相同",
            True,
            f"prod: 二层符号的内部值 = {L5_m.internal(symbol_b_m)}；"
            f"keep: 二层符号的内部值 = {L5.internal(symbol_b)}；"
            f"【拓扑匹配】两口径的 activate(二层符号) 逐位相同 = {aux_matched}（同一符号集，只差 rule）；"
            f"【最小见证】只含一层/二层符号的两实例逐位相同 = {aux_clean}")

    # ---------- P6 只读 ----------
    h_after = all_hashes()
    c.check("P6 只读: 八文件哈希前后一致（含 compose_layer 与 exp34）+ L0 签名前后一致",
            h_before == h_after and all(h_after[f] == v for f, v in EXPECTED.items())
            and net.node_count() == len(l0_nodes) and net.edge_count() == 25,
            f"before==after = {h_before == h_after}；"
            + "；".join(f"{f}={h_after[f]}" for f in ("synapse_net.py", "synapse_net_v2.py",
                                                      "synapse_net_v3.py", "compose_layer.py"))
            + f"；L0 签名 = {net.node_count()} 节点 / {net.edge_count()} 边（未调用 learn/activate 之外无写入）")

    # ---------- P7 双向 ----------
    ok_all = all(ck["ok"] for ck in c.checks)
    outcome = "C 条目（符号可再次传入 combine）" if ok_all else "E48_FAIL"
    c.check("P7 双向结局: P1–P5 全过 → 新 C 条目；否则 → 如实记录（不预分配 B 编号）",
            ok_all, f"落定 {outcome}（前序判据全过 = {ok_all}）")

    return {
        "id": "exp48_nested_composition",
        "title": "E48 组合的嵌套: combine(symbol_a, R) → symbol_b（符号作为模式）",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "corpus": {"kind": "E1 26 字母链", "nodes": net.node_count(), "edges": net.edge_count(),
                       "hop_decay": HOP_DECAY, "reps": REPS},
            "patterns": {"P": {k: round(v, 6) for k, v in P.items()},
                         "Q": {k: round(v, 6) for k, v in Q.items()},
                         "R": {k: round(v, 6) for k, v in R.items()},
                         "S": {k: round(v, 6) for k, v in S.items()}},
            "rule": {"main": RULE_MAIN, "aux": RULE_AUX,
                     "why_keep_main": "prod 在不相交输入下退化（内部值全 0）；P1/P2/P3 与 rule 无关（实测级联逐位相同）"},
            "P1_input_layer": {"input_keys": sorted(inp_keys),
                               "l5_symbols_in_input": l5_symbols_in_input,
                               "symbol_a_in_L0": bool(symbol_a in l0_nodes),
                               "symbol_ids": list(_symbol_ids),
                               "symbol_ids_collide_L0": sorted(set(_symbol_ids) & l0_nodes)},
            "P2_cascade": {"activation": {k: round(v, 9) for k, v in sorted(acts_b.items(), key=lambda kv: -kv[1])},
                           "required_visible": need,
                           "actual": {n: round(acts_b.get(n, float('nan')), 9) for n in need},
                           "one_hop": {n: round(acts_b.get(n, float('nan')), 9) for n in ("X", "Y", "Z")}},
            "P3_decay": {"d1_actual": round(d1, 12), "d1_expected": exp1,
                         "d2_actual": round(d2, 12), "d2_expected": exp2,
                         "note": "hop_decay^d 是数学必然，不是涌现"},
            "P4_symbols": {"ids": syms, "members": {s: L5.members_of(s) for s in syms},
                           "instances": 1, "needs_L6": False},
            "P5_B35": {"member_sets": [sorted(s) for s in sets], "pairwise_distinct": len(set(sets)) == len(sets),
                       "note": "刻意设计：不重叠，避免 last-write-wins"},
            "aux_prod_degenerate": {"inner_internal_prod": L5_m.internal(symbol_b_m),
                                    "inner_internal_keep": L5.internal(symbol_b),
                                    "topology_matched_identical": bool(aux_matched),
                                    "clean_pair_identical": bool(aux_clean),
                                    "note": "第一版比较用『aux 只含前两个符号』对『主实例含 4 个符号』=> "
                                            "比较的是拓扑而非 rule；改为拓扑匹配后为 True。"},
            "readonly": {"hashes_before": h_before, "hashes_after": h_after, "expected": EXPECTED},
            "outcome": outcome,
            "registration": {"outcome": outcome, "text": C_ENTRY_TEXT, "C_entry": C_ENTRY_TEXT,
                             "E48_FAIL": FAIL_TEXT,
                             "note": "C 条目编号留白（C??），跑完按登记表定。"},
            "note": "E48 是 E47 判定规则（新内容必须在输入层判）的第一个反例/正例："
                    "输入层确实出现了 E41/E42/E47 都没有的量（L5 符号本身作为模式）。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
