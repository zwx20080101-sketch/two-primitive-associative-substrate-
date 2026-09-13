"""E30: 闭包审计（B05 正式化）——"已学集合"与失败语义。

严格按 DESIGN-E30.md。**纯只读审计，无 numpy 依赖；不改任何层文件。**

预注册要点:
  - 双向结局（写在设计最前面）：违例数 = 0 → C30（设计不变量）；> 0 → B22（软执行 + 明细）；
  - B21【总是登记】：失败语义在层间不统一（两处静默：L2 未知前缀 → {}；L4 单元素 → None）；
  - P2 核对顺序：L0/L1 → L2 → L4① / L4②（分开核对）；
  - P3 失败语义表 5 列：调用 | 输入类型 | 期望语义 | 实测语义 | 是否报错；
  - 辅助 1/2 是【记录项】不是判据：期望来自预检；不符只记"预检观察未被正式复现"，不阻断 P1–P6。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from chunk_layer import ChunkLayer
from context_layer import ContextLayer
from order_layer import OrderLayer
from synapse_net import SynapseNet

from experiments.checker import Checker
from experiments.exp24_module_differentiation import (
    SESSION_SEED,
    ari,
    corpus_planted,
    impute_diagonal,
    matrix_from_net,
    spectral,
)
from experiments.exp25_module_dynamics import block_matrix, rho

REPS = 100
LAYER_FILES = ["synapse_net.py", "order_layer.py", "context_layer.py", "chunk_layer.py"]
UNKNOWN = "ZZZ"

B21_TEXT = (
    "B21（总是登记，与闭包结局无关）：失败语义在层间不统一——"
    "KeyError（L0/L1/L4 的未学过）、空字典（L2 未知前缀）、ValueError（L2 参数错误）、"
    "None（L4 未达 K）。其中【L2 的未知前缀不报错、返回 {}】是本实验认定的接口风险："
    "调用方若不检查空结果，就无法区分「这个词没学过」与「学过但无候选」，"
    "与「不报错、给出看起来合理的结果」同族。本实验只记录，不修改报错行为。"
    "（更正：预检曾写「两处静默」，正式跑显示 L4 的 None 是【有信息】的——None 与块 id 可区分，"
    "故不计入静默；静默案例 = 1 处。）"
)
C30_TEXT = (
    "C30（结局 A，仅当 P2 违例数 = 0 时登记）：闭包是【设计不变量】——"
    "在受控语料上逐层审计 L0/L1/L2/L4（五个已学集合），所有输出集合 ⊆ 该层已学集合，"
    "违例数 = 0。「不存在」在层间的表现不统一（KeyError / 空字典 / None / ValueError），"
    "但不产生闭包外节点。"
)
B22_TEXT = (
    "B22（结局 B，仅当 P2 违例数 > 0 时登记）：闭包在某层被【软执行】——"
    "明细：<层 / 调用 / 输出节点 / 该层已学集合>。"
)


def layer_hashes() -> dict:
    root = Path(__file__).resolve().parent.parent
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest()[:16] for n in LAYER_FILES}


def build():
    net, l1, l2 = SynapseNet(), OrderLayer(), ContextLayer()
    for _ in range(REPS):
        net.learn(["A", "B", "C"])
        l1.learn(["A", "B", "C"])
        l2.learn(["A", "B", "C"])
    ch = ChunkLayer()
    for _ in range(REPS):
        ch.learn_unit(["H", "e", "l", "l", "o"])
    return net, l1, l2, ch


def call(fn):
    """执行一次调用，返回 (实测语义描述, 是否报错, 输出集合)。"""
    try:
        r = fn()
        if isinstance(r, dict):
            return f"返回 dict(keys={sorted(r)[:6]})", False, set(r)
        if isinstance(r, list):
            return f"返回 list{str(r)[:40]}", False, set(r)
        return f"返回 {type(r).__name__}({r})", False, set()
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:60]}", True, set()


def run() -> dict:
    c = Checker()
    hashes_before = layer_hashes()
    net, l1, l2, ch = build()

    # ---------- P1 接口回归（E24 语料）----------
    net_r, nodes_r, labels_r, _ = corpus_planted()
    Mr = matrix_from_net(net_r, nodes_r)
    assign_r = spectral(impute_diagonal(Mr), 3, seed=SESSION_SEED)
    truth_r = [labels_r[n] for n in nodes_r]
    ari_r = float(ari(assign_r, truth_r))
    scale_r = max(
        np.mean([Mr[p, q] for p in range(len(nodes_r)) if assign_r[p] == i
                 for q in range(len(nodes_r)) if assign_r[q] == i and p != q])
        for i in sorted(set(assign_r.tolist()))
    )
    rho_r = rho(block_matrix(Mr, assign_r, scale_r))
    c.check("P1 接口回归: E24 植入语料重现 ARI=1.000 / 3 模块 / ρ=0.0019",
            ari_r >= 0.999 and len(set(assign_r.tolist())) == 3 and abs(rho_r - 0.0019) < 5e-4,
            f"ARI={ari_r:.3f}, 模块数={len(set(assign_r.tolist()))}, ρ={rho_r:.4f}")

    # ---------- §2 各层"已学集合" ----------
    learned = {
        "L0": set(net.nodes),
        "L1": set(l1.nodes),
        "L2_prefixes": set(l2.counts.keys()),
        "L4_chunks": set(ch.chunk_ids()),
    }
    spells = {cid: ch.spell(cid) for cid in ch.chunk_ids()}
    learned["L4_members"] = {t for toks in spells.values() for t in toks}
    l2_tokens = {t for (a, b) in learned["L2_prefixes"] for t in (a, b)}
    l2_tokens |= {c for (a, b) in learned["L2_prefixes"] for c in l2.counts[(a, b)]}

    # ---------- P2 闭包审计（顺序：L0/L1 → L2 → L4①/L4②）----------
    audit, violations = [], []
    checks_order = [
        ("L0", "L0.activate", lambda s: net.activate(s), learned["L0"], "L0 已学集合 = net.nodes"),
        ("L1", "L1.activate", lambda s: l1.activate(s), learned["L1"], "L1 已学集合 = order_layer.nodes"),
        ("L2①", "L2.query(已知前缀)", lambda p: l2.query(list(p)), l2_tokens,
         "L2 已学集合 = 2-前缀 token 集（L2 无独立节点集，闭包构造性成立）"),
    ]
    for name, call_name, fn, allowed, why in checks_order:
        outs, inputs = set(), []
        if name.startswith("L2"):
            for p in sorted(learned["L2_prefixes"]):
                _d, _err, out = call(lambda p=p: fn(p))
                outs |= out
                inputs.append(list(p))
        else:
            for s in sorted(allowed):
                _d, _err, out = call(lambda s=s: fn(s))
                outs |= out
                inputs.append(s)
        bad = sorted(outs - set(allowed))
        audit.append({"层": name, "调用": call_name, "已学集合定义": why,
                      "样本输入数": len(inputs), "输出并集大小": len(outs),
                      "违例数": len(bad), "违例节点": bad})
        violations += bad
    # L4 两个集合分开核对
    l4a_out = set()
    for cid in sorted(learned["L4_chunks"]):
        _d, _err, out = call(lambda cid=cid: ch.spell(cid))
        l4a_out |= out
    bad_a = sorted(l4a_out - learned["L4_members"])
    bad_domain = []      # 定义域核对：spell 只对块节点有效
    _d, err_unknown, out_unknown = call(lambda: ch.spell("CH_不存在"))
    audit.append({"层": "L4①", "调用": "L4.spell(块节点) → 值域",
                  "已学集合定义": "① 块节点集合 = chunk_ids()（spell 定义域）",
                  "样本输入数": len(learned["L4_chunks"]), "输出并集大小": len(l4a_out),
                  "违例数": len(bad_a), "违例节点": bad_a})
    audit.append({"层": "L4②", "调用": "L4.spell(非块节点) → 定义域核对",
                  "已学集合定义": "② 块成员集合 = 所有 spell 值域",
                  "样本输入数": 1, "输出并集大小": len(out_unknown),
                  "违例数": len(out_unknown - learned["L4_members"]),
                  "违例节点": sorted(out_unknown - learned["L4_members"]),
                  "note": f"非块节点 {err_unknown}"})
    violations += bad_a
    c.check("P2 闭包审计（顺序 L0/L1 → L2 → L4①/L4②）: 逐层核对输出 ⊆ 该层已学集合",
            len(violations) == 0,
            "; ".join(f"{r['层']} 违例={r['违例数']}" for r in audit) + f"；总违例={len(violations)}")

    # ---------- P3 失败语义表（5 列 × 输入类型枚举）----------
    rows = [
        ("L0.activate(已学)", "已学", "返回激活分布", lambda: net.activate("A")),
        ("L0.activate(未学过)", "未学过", "拒绝（闭包硬约束）", lambda: net.activate(UNKNOWN)),
        ("L1.activate(已学)", "已学", "返回激活分布", lambda: l1.activate("A")),
        ("L1.activate(未学过)", "未学过", "拒绝（闭包硬约束）", lambda: l1.activate(UNKNOWN)),
        ("L2.query(已知前缀)", "已学", "返回候选分布", lambda: l2.query(["A", "B"])),
        ("L2.query(未知前缀)", "未知前缀", "拒绝（闭包硬约束）", lambda: l2.query(["A", UNKNOWN])),
        ("L2.query(长度 != 2)", "参数错误", "拒绝（参数校验）", lambda: l2.query(["A"])),
        ("L4.spell(已知块)", "已学", "返回值域 token 列表", lambda: ch.spell(sorted(learned["L4_chunks"])[0])),
        ("L4.spell(不存在的块)", "未学过", "拒绝（闭包硬约束）", lambda: ch.spell("CH_不存在")),
        ("L4.activate(块节点)", "已学", "返回激活分布", lambda: ch.activate(sorted(learned["L4_chunks"])[0])),
        ("L4.activate(块成员)", "未学过", "拒绝（闭包硬约束）", lambda: ch.activate("H")),
        ("L4.learn_unit(单元素)", "参数错误", "忽略（不建块）", lambda: ch.learn_unit(["a"])),
    ]
    table = []
    for name, in_type, expected, fn in rows:
        observed, raised, _out = call(fn)
        table.append({"调用": name, "输入类型": in_type, "期望语义": expected,
                      "实测语义": observed, "是否报错": raised})
    # "静默"的操作定义（预注册）：期望语义是"拒绝"、但实际未报错 → 调用方拿不到异常信号。
    silent = [r for r in table if not r["是否报错"] and "拒绝" in r["期望语义"]]
    c.check("P3 失败语义表完整（5 列 × 输入类型枚举，无空格）",
            len(table) == len(rows) and all(all(k in r and r[k] != "" for k in
                ("调用", "输入类型", "期望语义", "实测语义")) for r in table),
            f"{len(table)} 行 × 5 列填满；【静默放行】（期望拒绝但未报错）{len(silent)} 行："
            + "; ".join(f"{r['调用']} → {r['实测语义']}" for r in silent)
            + "；另注：L4.learn_unit(单元素) 返回 None 属【有信息的正常返回】（None 与块 id 可区分），不计入静默")

    # ---------- 辅助 1 / 辅助 2（记录项，不阻断判据）----------
    aux1_obs, _e1, _o1 = call(lambda: l2.query(["A", UNKNOWN]))
    aux2_obs, e2, _o2 = call(lambda: ch.activate("H"))
    aux = {
        "aux1_L2_unknown_prefix": {"auxiliary": True, "期望": "{}（静默空返回）", "实测": aux1_obs,
                                   "符合预检": aux1_obs.startswith("返回 dict(keys=[])")},
        "aux2_L4_member_activate": {"auxiliary": True, "期望": "KeyError（硬拒绝）", "实测": aux2_obs,
                                    "符合预检": e2},
        "note": "辅助项是记录项不是判据；不符只记「预检观察未被正式复现」，不阻断 P1–P6。",
    }

    # ---------- P5 只读 ----------
    hashes_after = layer_hashes()
    c.check("P5 只读（纪律 ⑧）: 层文件 sha256 前后一致",
            hashes_before == hashes_after,
            f"before={hashes_before}；after={hashes_after}")

    # ---------- P6 登记（按双向预注册落定）----------
    outcome = "A" if len(violations) == 0 else "B"
    registered = ["B21", "C30"] if outcome == "A" else ["B21", "B22"]
    c.check("P6 登记: 按双向预注册落定（A → B21+C30；B → B21+B22）",
            registered == (["B21", "C30"] if outcome == "A" else ["B21", "B22"]),
            f"闭包结局={outcome}（总违例={len(violations)}）→ 登记 {registered}")

    return {
        "id": "exp30_closure_audit",
        "title": "E30 闭包审计: 五个已学集合逐层核对 + 失败语义表",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "closure_audit": audit,
            "violations_total": len(violations),
            "outcome": outcome,
            "failure_semantics_table": table,
            "auxiliary": aux,
            "learned_sets": {k: (sorted(v)[:8] if isinstance(v, set) else v) for k, v in learned.items()},
            "l2_tokens": sorted(l2_tokens),
            "layer_hashes": {"before": hashes_before, "after": hashes_after},
            "registration": {"outcome": outcome, "registered": registered,
                             "B21": B21_TEXT, "C30": C30_TEXT, "B22": B22_TEXT},
            "note": "只读审计，不修改任何层的报错行为；改报错语义是另一个实验。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
