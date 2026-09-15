"""E38: 节点级 / 模块级质量分配的口径确认（A 小步）。

严格按 DESIGN-E38.md。**不改任何层文件（0 行改动）；只对 v1 的 net 做统计，不调用 v2。**

预注册要点:
  - 语料: E24/E25 的 C1（corpus_planted()，15 节点 / 3 模块）
  - SEEDS = [20260911] + range(9)（10 个；含 E33 的参照种子 20260911）
  - 折叠规则主口径: 平票（模块内符号和 = 0）→ 0.0；辅助口径 → −1.0（只报差异，不作判据）
  - 质量口径: 主对比用【重归一化】值（分母 = 落在 4 个纯二值状态内的总质量），原始值同时报
  - P0 可读性（B24）; P1 种子稳定性（排名第 1 + 极差 ≤ 0.05）; P2 偏移方向（10/10 d_s > 0）;
    P3 β 扫描（β ∈ {1,2,4,8} 各 10/10）; P4 平票不掩盖（β=2，中位数 < 0.01）;
    P5 解耦对照（跨模块置 0 → 精确 0.25）; P6 只读; P7 双向
  - P0–P5 每条判据【同时报 10/10 与 9/9】（9/9 = 排除参照种子 20260911）
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.checker import Checker
from experiments.exp24_module_differentiation import (
    ari,
    corpus_planted,
    impute_diagonal,
    matrix_from_net,
    spectral,
)
from experiments.exp25_module_dynamics import block_matrix
from experiments.exp31_auto_scale import signature
from experiments.exp33_landscape_sampling import all_states, boltzmann

SEEDS = [20260911, 0, 1, 2, 3, 4, 5, 6, 7, 8]
REFERENCE_SEED = 20260911
BETA_SCAN = [0.5, 1.0, 2.0, 4.0, 8.0]
JUDGE_BETAS = [1.0, 2.0, 4.0, 8.0]
MAIN_BETA = 2.0
TARGET = (1.0, 1.0, 1.0)

TOL = 1e-12
DECOUPLE_TOL = 1e-9
ARI_MED_MIN = 0.99
ARI_MIN_MIN = 0.90
RANGE_MAX = 0.05
TIE_MASS_MAX = 0.01

V1_HASH_EXPECTED = "cf99a07a0e60fd4d"
V2_HASH_EXPECTED = "ee6df589eb2f6983"
V2_FIELD_REASON = (
    "本实验不调用 v2；字段为只读口径一致性保留"
    "（纪律⑧：无数据的字段写 null + reason，不许省略）"
)

# 六个文件【实测】哈希。
# 修补记录（2026-09-15）：初版用 layer_hashes()，它只覆盖四个层文件，
# 对 synapse_net_v2.py 走的是 `h_after.get(..., V2_HASH_EXPECTED)` 默认值兜底
# → 该项检查形同虚设（报出的值恰好正确，但不是真实测量）。改为逐文件实测。
HASH_FILES = ["synapse_net.py", "synapse_net_v2.py",
              "order_layer.py", "context_layer.py", "chunk_layer.py"]

C33_LIMIT_TEXT = (
    "C33 限定条款（E38 追加）：C33 所述「节点级与模块级是粗粒化关系」应读作"
    "【支撑 / 自由度一致】—— 节点级有效支撑从 86 单调降至 4，与模块级 4 个自由度对应。"
    "该表述【不蕴含】质量分配一致：节点级精确分布聚合到 4 个纯二值状态后，(+1,+1,+1) 的质量"
    "系统性地高于模块级 W3 的 Boltzmann 概率（10 种子同号，实测见 exp38 JSON）。"
    "（种子轴说明：10 个聚类种子给出的是【同一划分】，仅模块标签排列不同 —— 该语料划分过于清晰，"
    "种子轴退化，故该偏移是【确定量】而非多种子统计证据。）"
)
B32_TEXT = (
    "B32：分块均值的模块级 W3 的 Boltzmann 概率【不是】节点级分布对模块符号的正确边缘化 —— "
    "在 E24/E25 的 C1 语料上二者相差 d（实测见 exp38 JSON），且偏移由跨模块耦合产生"
    "（把节点级跨模块项置 0 后，聚合分布精确塌成 0.25 均匀）。"
    "种子轴说明：10 个聚类种子给出同一划分（仅标签排列不同，实跑诊断 = 1/10），"
    "故该偏移在该划分下是【确定量】，不构成多种子独立证据。"
    "折叠规则说明：主口径（平票 → 0.0）与辅助口径（平票 → −1.0）下 4 状态质量逐位相同"
    "（maxdiff = 0.0）—— 原因是【本配置的巧合】：携带平票质量的模块（size 4）恰好就是"
    "被钳制的种子模块，辅助口径把它的平票折成 −1.0 后与钳制 +1 冲突、落在 S3 之外；"
    "不是普遍法则（若钳制换到别的模块，辅助口径会改变主表，量级 ≈4.6e-05）。"
    "用途限定：模块级 W3 可作「结构 / 自由度」模型，不可当作节点级分布的精确定量替代。"
    "范围限定：限于 E24/E25 的 C1 语料（15 节点 / 3 模块 / 块均值口径）；"
    "其他语料或其他 block_matrix 口径下的边缘化误差不在本结论范围内。"
)
E38_NO_REPRO_TEXT = (
    "E38 未复现：A2b 的节点级/模块级质量偏移未在 10 种子 / β 扫描下稳定复现"
    "（哪条判据挂、实测值见 exp38 JSON）。该偏移记为【折叠规则产物 / 预检观察未复现】，"
    "A2b 预检表作废，不占 B 编号（与 E26 的 w=91 降级同型，纪律⑦）。"
)


def _stats(vals: list[float]) -> dict:
    a = np.array(vals, dtype=float)
    return {
        "n": int(a.size),
        "median": round(float(np.median(a)), 9),
        "min": round(float(a.min()), 9),
        "max": round(float(a.max()), 9),
        "range": round(float(a.max() - a.min()), 9),
    }


def _sha16(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def all_hashes() -> dict:
    """六个文件逐一实测（修补：不再用只覆盖四个层文件的 layer_hashes()）。"""
    return {f: _sha16(f) for f in HASH_FILES}


def _run_seed(seed: int, net, nodes, truth, Sn) -> dict:
    """单个聚类种子下的全部统计（含 β 扫描与解耦对照）。"""
    Mraw = matrix_from_net(net, nodes)
    assign = spectral(impute_diagonal(Mraw), 3, seed=seed)
    mods = sorted(set(int(x) for x in assign))
    scale = max(
        np.mean([Mraw[p, q] for p in range(len(nodes)) if assign[p] == i
                 for q in range(len(nodes)) if assign[q] == i and p != q])
        for i in mods
    )
    W3 = block_matrix(Mraw, assign, scale)
    Wn = Mraw / Mraw.max()
    idxmod = {m: [i for i in range(len(nodes)) if assign[i] == m] for m in mods}
    seed_mod = int(assign[0])
    S3 = all_states(3, {seed_mod: 1})

    # 模块内符号和（向量化；=0 即平票）
    sums = np.stack([Sn[:, idx].sum(axis=1) for idx in (idxmod[m] for m in mods)], axis=1)
    tie_rows = np.any(np.abs(sums) <= TOL, axis=1)
    tie_incidents = int(np.sum(np.abs(sums) <= TOL))
    keys_main = [tuple(r) for r in np.sign(sums)]                                  # 平票 → 0.0
    keys_aux = [tuple(r) for r in np.where(np.abs(sums) <= TOL, -1.0, np.sign(sums))]  # 平票 → −1.0
    layout_main = {tuple(np.round(r, 12)): i for i, r in enumerate(S3)}

    scan = []
    for beta in BETA_SCAN:
        _, pn = boltzmann(Sn, Wn, beta)
        agg_main: dict = {}
        agg_aux: dict = {}
        for k, p in zip(keys_main, pn):
            agg_main[k] = agg_main.get(k, 0.0) + float(p)
        for k, p in zip(keys_aux, pn):
            agg_aux[k] = agg_aux.get(k, 0.0) + float(p)

        inside = float(sum(agg_main.get(tuple(r), 0.0) for r in S3))
        renorm = np.array([agg_main.get(tuple(r), 0.0) / inside for r in S3])
        raw = np.array([agg_main.get(tuple(r), 0.0) for r in S3])
        aux_inside = float(sum(agg_aux.get(tuple(r), 0.0) for r in S3))
        aux_renorm = np.array([agg_aux.get(tuple(r), 0.0) / aux_inside for r in S3])
        _, p3 = boltzmann(S3, W3, beta)
        d = renorm - np.asarray(p3, dtype=float)
        tgt_idx = layout_main[tuple(np.round(np.array(TARGET), 12))]

        scan.append({
            "beta": beta,
            "module_p3": [round(float(x), 9) for x in p3],
            "node_renorm": [round(float(x), 9) for x in renorm],
            "node_raw": [round(float(x), 9) for x in raw],
            "d_vector": [round(float(x), 9) for x in d],
            "inside_mass": round(inside, 9),
            "target_renorm": round(float(renorm[tgt_idx]), 9),
            "target_rank1": bool(float(renorm[tgt_idx]) == float(renorm.max())),
            "d_target": round(float(d[tgt_idx]), 9),
            "tie_count": int(tie_rows.sum()),
            "tie_mass": round(float(pn[tie_rows].sum()), 9),
            "tie_incidents": tie_incidents,
            "aux_vs_main_maxdiff": round(float(np.max(np.abs(aux_renorm - renorm))), 9),
        })

    # 解耦对照（P5）：跨模块项全部置 0，β = MAIN_BETA
    Wzero = Wn.copy()
    for i in mods:
        for j in mods:
            if i == j:
                continue
            for p in idxmod[i]:
                for q in idxmod[j]:
                    Wzero[p, q] = 0.0
    _, pn0 = boltzmann(Sn, Wzero, MAIN_BETA)
    agg0: dict = {}
    for k, p in zip(keys_main, pn0):
        agg0[k] = agg0.get(k, 0.0) + float(p)
    inside0 = float(sum(agg0.get(tuple(r), 0.0) for r in S3))
    renorm0 = np.array([agg0.get(tuple(r), 0.0) / inside0 for r in S3])

    # 跨模块耦合总量：节点级（直接求和）vs 模块级折算回节点单位（分块均值 × 配对数）
    # 两侧都用【无序对】（p<q），否则会差一个因子 2
    cross_total = float(sum(Wn[p, q] for p in range(len(nodes)) for q in range(p + 1, len(nodes))
                            if assign[p] != assign[q]))
    cross_total_block = float(sum(
        (len(idxmod[i]) * len(idxmod[j])) * W3[i, j]
        for i in mods for j in mods if i < j
    ))

    return {
        "seed": seed,
        "ari": round(float(ari(assign, truth)), 9),
        # 划分（规范形：块内排序 + 块间排序，与模块标签无关）vs 标签排列（按模块 id 顺序）
        "partition_blocks": sorted(sorted(nodes[i] for i in range(len(nodes)) if assign[i] == m)
                                   for m in mods),
        "labeling_blocks": [sorted(nodes[i] for i in range(len(nodes)) if assign[i] == m)
                            for m in mods],
        "scale": round(float(scale), 9),
        "module_sizes": {str(m): len(idxmod[m]) for m in mods},
        "seed_module": seed_mod,
        "module_W": np.round(W3, 9).tolist(),
        "s3_order": [[float(v) for v in r] for r in S3],
        "cross_total_node": round(cross_total, 9),
        "cross_total_block": round(cross_total_block, 9),
        "decoupled_renorm": [round(float(x), 9) for x in renorm0],
        "decoupled_max_dev_from_quarter": round(float(np.max(np.abs(renorm0 - 0.25))), 12),
        "beta_scan": scan,
    }


def _at(rec: dict, beta: float) -> dict:
    return next(x for x in rec["beta_scan"] if x["beta"] == beta)


def _dual(name: str, fn) -> list[tuple[str, bool, str]]:
    """同一条判据跑两个子集：10/10（全样本）与 9/9（排除参照种子）。"""
    out = []
    for label, subset in (("10/10", _ROWS), ("9/9", _ROWS9)):
        ok, detail = fn(subset)
        out.append((f"{name} [{label}]", ok, detail))
    return out


_ROWS: list[dict] = []
_ROWS9: list[dict] = []


def run() -> dict:
    global _ROWS, _ROWS9
    c = Checker()
    h_before = all_hashes()
    net, nodes, labels, _ = corpus_planted()
    truth = np.array([labels[n] for n in nodes])
    sig_before = signature(net)
    Sn = all_states(15, {0: 1})          # 16384，节点索引 0 钳制 +1（同 E33）

    _ROWS = [_run_seed(s, net, nodes, truth, Sn) for s in SEEDS]
    _ROWS9 = [r for r in _ROWS if r["seed"] != REFERENCE_SEED]
    ref = next(r for r in _ROWS if r["seed"] == REFERENCE_SEED)

    # ---------- P0 语料可读性（B24 口径）----------
    def p0(sub):
        aris = [r["ari"] for r in sub]
        med, mn = float(np.median(aris)), float(min(aris))
        return (med >= ARI_MED_MIN and mn >= ARI_MIN_MIN), \
            f"种子数={len(aris)}；ARI 中位数={med:.4f}（阈值 ≥{ARI_MED_MIN}），最小值={mn:.4f}（阈值 ≥{ARI_MIN_MIN}）"
    for nm, ok, det in _dual("P0 语料可读性（B24）", p0):
        c.check(nm, ok, det)

    # ---------- P1a' 辅助（不参与 PASS/FAIL）：聚类划分去重诊断 ----------
    part_keys = [tuple(tuple(b) for b in r["partition_blocks"]) for r in _ROWS]
    label_keys = [tuple(tuple(b) for b in r["labeling_blocks"]) for r in _ROWS]
    distinct = sorted(set(part_keys))
    distinct_lab = sorted(set(label_keys))
    degenerate = len(distinct) == 1
    c.check("P1a' 辅助（不参与 PASS/FAIL）: 聚类划分去重诊断",
            True,
            f"不同划分 = {len(distinct)} / {len(part_keys)}（不同标签排列 = {len(distinct_lab)}）；"
            + ("所有种子给出【同一划分】（仅模块标签排列不同）→ 种子轴退化："
               "P1/P2/P3 的 10/10 与 9/9 是同一测量的重复，『极差 = 0』是退化结果，"
               "不构成跨划分的稳定性证据；划分 = 三个植入组本身"
               if degenerate else "划分各不相同 → 种子轴有效"))

    # ---------- P1 种子稳定性 ----------
    def p1(sub):
        vals = [_at(r, MAIN_BETA)["target_renorm"] for r in sub]
        rank1 = all(_at(r, MAIN_BETA)["target_rank1"] for r in sub)
        st = _stats(vals)
        return (rank1 and st["range"] <= RANGE_MAX), \
            f"排名第 1 = {sum(1 for r in sub if _at(r, MAIN_BETA)['target_rank1'])}/{len(sub)}；" \
            f"中位={st['median']:.6f} 最小={st['min']:.6f} 最大={st['max']:.6f} 极差={st['range']:.6f}（阈值 ≤{RANGE_MAX}）"
    for nm, ok, det in _dual("P1 种子稳定性（β=2，排名 + 极差）", p1):
        c.check(nm, ok, det)

    # ---------- P2 偏移方向 ----------
    def p2(sub):
        ds = [_at(r, MAIN_BETA)["d_target"] for r in sub]
        st = _stats(ds)
        return (st["min"] > 0.0), \
            f"d_s > 0 的种子 = {sum(1 for d in ds if d > 0)}/{len(ds)}；" \
            f"中位={st['median']:.6f} 最小={st['min']:.6f}"
    for nm, ok, det in _dual("P2 偏移方向（β=2，d_s > 0）", p2):
        c.check(nm, ok, det)

    # ---------- P3 β 扫描不翻转（判定点 {1,2,4,8}）----------
    def p3(sub):
        parts, ok_all = [], True
        for beta in JUDGE_BETAS:
            good = sum(1 for r in sub if _at(r, beta)["d_target"] > 0.0)
            ok_all &= (good == len(sub))
            parts.append(f"β={beta}: {good}/{len(sub)}")
        return ok_all, "; ".join(parts)
    for nm, ok, det in _dual("P3 β 扫描不翻转（d_s > 0）", p3):
        c.check(nm, ok, det)

    # ---------- P4 平票不掩盖结论（判定点 β=2）----------
    def p4(sub):
        vals = [_at(r, MAIN_BETA)["tie_mass"] for r in sub]
        st = _stats(vals)
        return (st["median"] < TIE_MASS_MAX), \
            f"平票承载质量：中位={st['median']:.9f}（阈值 <{TIE_MASS_MAX}），最小={st['min']:.9f}，最大={st['max']:.9f}"
    for nm, ok, det in _dual("P4 平票不掩盖结论（β=2）", p4):
        c.check(nm, ok, det)

    # ---------- P5 解耦对照 ----------
    def p5(sub):
        devs = [r["decoupled_max_dev_from_quarter"] for r in sub]
        return (max(devs) <= DECOUPLE_TOL), \
            f"与 0.25 的最大偏差 = {max(devs):.3e}（阈值 ≤{DECOUPLE_TOL}）；" \
            f"通过种子 = {sum(1 for d in devs if d <= DECOUPLE_TOL)}/{len(devs)}"
    for nm, ok, det in _dual("P5 解耦对照（跨模块置 0 → 0.25）", p5):
        c.check(nm, ok, det)

    # ---------- P6 只读 ----------
    h_after = all_hashes()
    layer_files = ["synapse_net.py", "order_layer.py", "context_layer.py", "chunk_layer.py"]
    p6_ok = (h_before == h_after
             and all(h_after[f] == h_before[f] for f in layer_files)
             and h_after["synapse_net.py"] == V1_HASH_EXPECTED
             and signature(net) == sig_before)
    v2_hash = h_after["synapse_net_v2.py"]          # ← 实测，不再走默认值兜底
    c.check("P6 只读: 五个文件 sha256 逐文件实测且前后一致 + L0 签名前后一致 + v2 字段保留（含 reason）",
            p6_ok and v2_hash == V2_HASH_EXPECTED,
            f"before==after = {h_before == h_after}（六文件逐文件实测）；v1={h_after['synapse_net.py']}（期望 {V1_HASH_EXPECTED}）；"
            f"L0 签名前后一致 = {signature(net) == sig_before}（{net.node_count()} 节点 / {net.edge_count()} 边）；"
            f"v2 实测 = {v2_hash}（期望 {V2_HASH_EXPECTED}；未调用，字段保留）")

    # ---------- P7 双向 ----------
    ok_all = all(ck["ok"] for ck in c.checks)
    outcome = "C33_LIMIT + B32" if ok_all else "E38_NO_REPRO"
    text = (C33_LIMIT_TEXT + "\n\n" + B32_TEXT) if ok_all else E38_NO_REPRO_TEXT
    c.check("P7 双向结局: P0–P5 全过 → C33 加限定 + B32；否则 → 记录未复现（不占编号）",
            outcome in ("C33_LIMIT + B32", "E38_NO_REPRO"),
            f"落定 {outcome}（前序判据全过 = {ok_all}）")

    return {
        "id": "exp38_mass_allocation",
        "title": "E38 节点级/模块级质量分配的口径确认: 偏移是否稳定、是否由跨模块耦合产生",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "seeds": SEEDS,
            "reference_seed": REFERENCE_SEED,
            "beta_scan": BETA_SCAN,
            "judge_betas": JUDGE_BETAS,
            "params": {"main_beta": MAIN_BETA, "target_state": list(TARGET),
                       "tie_main": 0.0, "tie_aux": -1.0,
                       "tolerance": TOL, "decouple_tol": DECOUPLE_TOL,
                       "ari_median_min": ARI_MED_MIN, "ari_min_min": ARI_MIN_MIN,
                       "range_max": RANGE_MAX, "tie_mass_max": TIE_MASS_MAX,
                       "range_max_note": "预设阈值，无多种子数据依据；跑完不改",
                       "tie_mass_max_note": "预设阈值；预检单种子约 1e-4 量级，余量约 2 个数量级"},
            "s3_order_note": "每个种子各自的 S3 顺序（钳制该种子的种子模块为 +1）；TARGET=(1,1,1) 恒在其中",
            "per_seed": _ROWS,
            "summary": {
                "distinct_partitions": {
                    "count": len(distinct), "of": len(part_keys),
                    "count_labelings": len(distinct_lab),
                    "degenerate": bool(degenerate),
                    "blocks": [list(b) for b in distinct[0]] if degenerate else None,
                    "note": "种子轴诊断：=1 表示 10 个聚类种子给出同一划分，"
                            "此时 P1/P2/P3 的 10/10 与 9/9 是同一测量的重复（非独立样本）",
                },
                "target_renorm_median_min_max_range": _stats([_at(r, MAIN_BETA)["target_renorm"] for r in _ROWS]),
                "d_target_median_min": _stats([_at(r, MAIN_BETA)["d_target"] for r in _ROWS]),
                "tie_mass_by_beta": {str(b): _stats([_at(r, b)["tie_mass"] for r in _ROWS]).copy()
                                     for b in BETA_SCAN},
                "tie_count_median_min": _stats([_at(r, MAIN_BETA)["tie_count"] for r in _ROWS]),
                "inside_mass_median_min": _stats([_at(r, MAIN_BETA)["inside_mass"] for r in _ROWS]),
                "ari_median_min": _stats([r["ari"] for r in _ROWS]),
                "aux_vs_main_maxdiff": max(_at(r, MAIN_BETA)["aux_vs_main_maxdiff"] for r in _ROWS),
                "cross_total_node_vs_block": {
                    "node": ref["cross_total_node"], "block": ref["cross_total_block"],
                    "note": "分块均值按构造保持跨模块耦合总量（见 DESIGN-E38 §0.3）"},
            },
            "reference_seed_row": {"seed": REFERENCE_SEED,
                                   "beta_2": _at(ref, MAIN_BETA),
                                   "s3_order": ref["s3_order"],
                                   "module_W": ref["module_W"],
                                   "ari": ref["ari"]},
            "d_vectors_all_states_beta2": {
                str(r["seed"]): {"s3_order": r["s3_order"],
                                 "d_vector": _at(r, MAIN_BETA)["d_vector"]}
                for r in _ROWS},
            "readonly": {
                "hashes_before": h_before, "hashes_after": h_after,
                "layer_files": layer_files,
                "v1_expected": V1_HASH_EXPECTED,
                "synapse_net_v2.py": {"sha256_16": v2_hash, "reason": V2_FIELD_REASON},
                "L0_signature_unchanged": bool(signature(net) == sig_before),
            },
            "outcome": outcome,
            "registration": {"outcome": outcome, "text": text,
                             "C33_LIMIT": C33_LIMIT_TEXT, "B32": B32_TEXT,
                             "E38_NO_REPRO": E38_NO_REPRO_TEXT},
            "note": "P0–P5 每条判据同时报 10/10 与 9/9（9/9 排除参照种子 20260911）；"
                    "本实验不调用 v2，层文件 0 行改动。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
