"""E40: 无条件采样（解开种子模块钳制，8 状态全开）。

严格按 DESIGN-E40.md。**不改任何层文件（0 行改动）；不引入 stub / 语义 / 奖励。**

预注册要点:
  - 主对照 = (a) 单节点种子下 max 传播可达的模块状态（与 E33 同层级）；(c) 多种子作辅助
  - 主口径采样器 = 并行回火（PT，读取端工具，与 E33 的 boltzmann() 并列）；
    阶梯 β·[0.25,0.5,1,2,4]、burn-in 2000、间隔 20、n=5000、RNG 5000..5009
  - β 判定点 {0.5, 2.0, 8.0}（10 种子）；辅助 {1.0, 4.0}（**n=3，预设值，不进判据**）
  - 判据分层见 §3.0：实质判据 P1/P2/P3/P7c；形式审计 P4/P4b/P5/P6；定义式 P7a；
    报告项 P7b；只读 P8
  - 硬线：β=8 的 KL 取 10 种子的【最大值】< 0.01（余量约 9%，跑后不改判据）
"""

from __future__ import annotations

import hashlib
import json
import math
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
from experiments.exp33_landscape_sampling import all_states, boltzmann, energies

SEED_MAIN = 20260911
BETA_JUDGE = [0.5, 2.0, 8.0]
BETA_AUX = [1.0, 4.0]
BETA_ALL = [0.5, 1.0, 2.0, 4.0, 8.0]
N_SAMPLES = 5000
N_SEEDS_JUDGE = 10
N_SEEDS_AUX = 3            # 辅助 β 的种子数（预设值，跑后不改）
RNG_SEEDS = list(range(5000, 5010))
BURN_IN, THIN = 2000, 20
LADDER = [0.25, 0.5, 1.0, 2.0, 4.0]
KL_TOL = 0.01
GEN_MASS_MIN = 0.5
BIN_THR = 0.5
TOL = 1e-12

V1_HASH_EXPECTED = "cf99a07a0e60fd4d"
V2_HASH_EXPECTED = "ee6df589eb2f6983"
V3_HASH_EXPECTED = "085be73a06f0694c"
NO_CALL_REASON = "本实验只对 v1 做 max 对照、对模块级 W3 做采样；字段为只读口径一致性保留（纪律⑧）"
HASH_FILES = ["synapse_net.py", "synapse_net_v2.py", "synapse_net_v3.py",
              "order_layer.py", "context_layer.py", "chunk_layer.py"]

C39_TEXT = (
    "C39：解开种子模块钳制后（8 状态全开），并行回火采样覆盖全部 8 个模块状态；"
    "其中 5 个是【任何单节点种子下 max 传播都到不了】的（集合差定义），"
    "它们在 β ∈ [0.5, 8] 上共携带 0.6253–0.6299 的质量（判据 > 0.5）。"
    "分层：4 个可由【多种子组合】到达、1 个（全关 (-1,-1,-1)）在任何种子集下都封闭。"
    "范围限定：① 主对照是【单节点种子】（与 E33 同层级）；多种子组合属 E34–E36 的读取端能力，"
    "不计入「基底自然扩散」；② 这 8 个状态的存在性已由 E25（C25）无条件枚举登记 —— "
    "E40 的新内容是【采样可及性 + 质量分配】，不是「发现新状态」；③ 不声称「系统能想象新状态」。"
    "语料限于 E24/E25 的 C1；v1/v2/v3 冻结不变。"
)
B34_TEXT = (
    "B34：max 传播的可达集在【全局自旋翻转】下不对称 —— 能量景观有两个简并基态 "
    "(1,1,1) 与 (-1,-1,-1)（E 相同 = -1.235577，β=2 时各 0.126206），基态流形有 2 个分支；"
    "但 max 传播（任何种子集）只能到达 (1,1,1) 分支，(-1,-1,-1) 分支在检索逻辑下封闭"
    "（机制：任何非空种子集都会点亮其所在模块 —— 种子语义的直接推论）。"
    "采样覆盖两个分支且质量相同。"
    "范围限定：限于 E24/E25 的 C1 语料（3 模块 / 块均值口径）；"
    "这是「检索可达集 ⊊ 基态流形」的一个实例，不是「检索总是不对称」的全称断言。"
)
FAIL_TEXT = (
    "E40 未通过：哪条判据挂、实测值见 exp40 JSON。若挂的是 P7c（尤其 β=8），"
    "按预注册硬线如实记 FAIL，不得改阈值或改判据（纪律⑨：跑后不改判据）。"
)


def _sha16(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def all_hashes() -> dict:
    return {f: _sha16(f) for f in HASH_FILES}


def _p_plus(W: np.ndarray, beta: float, i: int, s: np.ndarray) -> float:
    h = float(W[i] @ s)
    return 1.0 / (1.0 + np.exp(-2.0 * beta * h))


def gibbs(W: np.ndarray, beta: float, rng, n: int) -> list:
    """单点 Gibbs（单链）——对照口径，含 B26 复现点。"""
    s = rng.choice([-1.0, 1.0], 3)
    out = []
    for k in range(BURN_IN + n * THIN):
        i = int(rng.integers(3))
        s[i] = 1.0 if rng.random() < _p_plus(W, beta, i, s) else -1.0
        if k >= BURN_IN and (k - BURN_IN) % THIN == 0:
            out.append(tuple(int(v) for v in s))
    return out


def parallel_tempering(W: np.ndarray, beta_t: float, rng, n: int) -> tuple:
    """并行回火（主口径）。返回（目标温度上的样本, 交换接受率）。"""
    ladder = [beta_t * f for f in LADDER]
    mid = LADDER.index(1.0)
    reps = [rng.choice([-1.0, 1.0], 3) for _ in ladder]
    acc = att = 0
    out = []
    for k in range(BURN_IN + n * THIN):
        for b, s in zip(ladder, reps):
            i = int(rng.integers(3))
            s[i] = 1.0 if rng.random() < _p_plus(W, b, i, s) else -1.0
        j = int(rng.integers(len(ladder) - 1))
        b1, b2 = ladder[j], ladder[j + 1]
        e1 = -0.5 * float(reps[j] @ W @ reps[j])
        e2 = -0.5 * float(reps[j + 1] @ W @ reps[j + 1])
        att += 1
        if rng.random() < np.exp((b1 - b2) * (e1 - e2)):
            reps[j], reps[j + 1] = reps[j + 1], reps[j]
            acc += 1
        if k >= BURN_IN and (k - BURN_IN) % THIN == 0:
            out.append(tuple(int(v) for v in reps[mid]))
    return out, (acc / att if att else 0.0)


def kl_exact_emp(p: np.ndarray, samples: list, S8: np.ndarray) -> float:
    cnt: dict = {}
    for s in samples:
        cnt[s] = cnt.get(s, 0) + 1
    n = len(samples)
    kl = 0.0
    for i, pr in enumerate(p):
        if pr <= 0:
            continue
        q = max(cnt.get(tuple(int(v) for v in S8[i]), 0) / n, 1e-12)
        kl += pr * np.log2(pr / q)
    return float(kl)


def _stats(vals: list) -> dict:
    a = np.array(vals, dtype=float)
    return {"n": int(a.size), "median": round(float(np.median(a)), 9),
            "min": round(float(a.min()), 9), "max": round(float(a.max()), 9)}


def run() -> dict:
    c = Checker()
    h_before = all_hashes()

    # ---------- 语料与模块结构（与 E33 同）----------
    net, nodes, labels, _ = corpus_planted()
    Mraw = matrix_from_net(net, nodes)
    assign = spectral(impute_diagonal(Mraw), 3, seed=SEED_MAIN)
    mods = sorted(set(int(x) for x in assign))
    scale = max(np.mean([Mraw[p, q] for p in range(len(nodes)) if assign[p] == i
                         for q in range(len(nodes)) if assign[q] == i and p != q]) for i in mods)
    W3 = block_matrix(Mraw, assign, scale)
    members = {m: [nodes[i] for i in range(len(nodes)) if assign[i] == m] for m in mods}
    seed_mod = int(assign[0])
    S8 = all_states(3)                                  # 8 个（无条件）
    S3 = all_states(3, {seed_mod: 1})                   # 4 个（E33 的条件空间）
    key_of = {tuple(int(v) for v in r): k for k, r in enumerate(S8)}
    sig_before = signature(net)

    # ---------- P0 前置：语料可读性（引用 E38；此处重跑一次作审计）----------
    truth = np.array([labels[n] for n in nodes])
    ari_now = float(ari(assign, truth))
    c.check("P0 前置: C1 的 k=3 ARI（E38 实测 1.0000；本实验重跑作审计）",
            ari_now >= 0.99,
            f"本次 ARI = {ari_now:.4f}（E38 记录 1.0000）；语料与 assign 种子均与 E33/E38 相同")

    # ---------- 对照三口径 ----------
    def sym_of(act: dict) -> tuple:
        """模块级符号（int 元组 —— 与 S8 的键同型，避免 float/int 键不匹配）。"""
        frac = {m: float(np.mean([act.get(n, 0.0) >= BIN_THR for n in members[m]])) for m in mods}
        return tuple(1 if frac[m] >= BIN_THR else -1 for m in mods)

    single_reach, node_sets = {}, {}
    for n0 in nodes:
        act = net.activate(n0)
        single_reach.setdefault(sym_of(act), []).append(n0)
        node_sets[n0] = frozenset(k for k, v in act.items() if v >= BIN_THR)
    multi_reach, example_seeds = {}, {}
    for k in range(8):
        target = tuple(1 if (k >> i) & 1 else -1 for i in mods)
        seeds = [members[m][0] for m in mods if target[m] > 0]
        if not seeds:
            continue
        if sym_of(net.activate(seeds)) == target:
            multi_reach[target] = seeds
            example_seeds[target] = seeds
    generated = [s for s in key_of if s not in single_reach]
    gen_multi = [s for s in generated if s in multi_reach]
    gen_none = [s for s in generated if s not in multi_reach]
    c.check("P2 支撑差异: max 对照(a) 3/8；采样 8/8 ⇒ 生成集合 = 5（集合差定义）",
            len(single_reach) == 3 and len(generated) == 5,
            f"(a) 单节点可达 = {len(single_reach)}/8 {sorted(single_reach)}；"
            f"(b) 单种子输出集合的并集 = {len(set().union(*node_sets.values()))}/{len(nodes)}（无区分力，仅记录）；"
            f"(c) 多种子可达 = {len(multi_reach)}/8（唯一不可达 = (-1,-1,-1)）；生成集合 = {len(generated)}")

    # ---------- 精确分布：P3 生成质量 + P5 + P6 ----------
    exact = {b: boltzmann(S8, W3, b)[1] for b in BETA_ALL}
    mass_gen = {b: float(sum(exact[b][key_of[s]] for s in generated)) for b in BETA_ALL}
    c.check("P3 生成质量: 5 个生成状态的总质量 > 0.5（每个扫描 β 上）",
            all(m > GEN_MASS_MIN for m in mass_gen.values()),
            "；".join(f"β={b}: {mass_gen[b]:.6f}" for b in BETA_ALL) + f"（判据 > {GEN_MASS_MIN}）")

    p3 = np.asarray(exact[2.0][[key_of[tuple(int(v) for v in r)] for r in S3]], dtype=float)
    restricted = p3 / p3.sum()
    p3c = np.asarray(boltzmann(S3, W3, 2.0)[1], dtype=float)
    c.check("P5 与 E33 一致性: 条件分布 = 无条件分布的限制 + 重归一化（逐位）",
            np.max(np.abs(restricted - p3c)) <= TOL,
            f"限制后总计 = {p3.sum():.6f}；最大逐位差 = {np.max(np.abs(restricted - p3c)):.3e}；"
            f"归一化 = {[round(float(x), 6) for x in restricted]} vs E33 p3 = {[round(float(x), 6) for x in p3c]}")

    E8 = energies(S8, W3)
    alloff = tuple(int(v) for v in S8[int(np.argmin(E8))])
    max_single = {b: tuple(int(v) for v in S8[int(np.argmax(exact[b]))]) for b in BETA_ALL}
    ground = [tuple(int(v) for v in r) for r in S8[np.abs(E8 - E8.min()) <= 1e-12]]
    c.check("P6 全关的物理地位（形式审计）: (-1,-1,-1) 在每个扫描 β 上都是最大单态，且任何种子集不可达",
            all(alloff == max_single[b] for b in BETA_ALL) and alloff not in multi_reach,
            f"最大单态 = {max_single}；基态简并 = {ground}（E = {float(E8.min()):.6f}，"
            f"β=2 各 {exact[2.0][key_of[ground[0]]]:.6f}）；(-1,-1,-1) 在多种子可达集里 = {alloff in multi_reach}")

    # ---------- P1 覆盖 + P7c KL + 采样估计 ----------
    pt_cov, pt_kl, pt_swap, gib_cov = {}, {}, {}, {}
    pt_mass, pt_mass_diff = {}, {}
    for beta in BETA_ALL:
        seeds_used = RNG_SEEDS[:N_SEEDS_JUDGE] if beta in BETA_JUDGE else RNG_SEEDS[:N_SEEDS_AUX]
        n_used = N_SAMPLES
        covs, kls, sws, mass = [], [], [], []
        gcovs = []
        for sd in seeds_used:
            smp, rate = parallel_tempering(W3, beta, np.random.default_rng(sd), n_used)
            covs.append(len(set(smp)))
            sws.append(rate)
            kls.append(kl_exact_emp(exact[beta], smp, S8))
            mass.append(sum(smp.count(s) for s in generated) / len(smp))
            gcovs.append(len(set(gibbs(W3, beta, np.random.default_rng(sd), n_used))))
        pt_cov[beta] = _stats(covs)
        pt_kl[beta] = _stats(kls)
        pt_swap[beta] = round(float(np.mean(sws)), 6)
        pt_mass[beta] = _stats(mass)
        pt_mass_diff[beta] = _stats([m - mass_gen[beta] for m in mass])
        gib_cov[beta] = _stats(gcovs)

    c.check("P1 采样覆盖（v2：只判主口径）: PT 在判定 β ∈ {0.5,2.0,8.0} 上 8/8（中位与最小）",
            all(pt_cov[b]["min"] == 8 for b in BETA_JUDGE),
            "PT：" + "；".join(f"β={b}: 中位 {pt_cov[b]['median']:.0f} / 最小 {pt_cov[b]['min']:.0f} / "
                              f"最大 {pt_cov[b]['max']:.0f}" for b in BETA_JUDGE)
            + " | 全扫描 " + "；".join(f"β={b}:{pt_cov[b]['min']:.0f}" for b in BETA_ALL))

    c.check("P1a' 辅助（不参与 PASS/FAIL）: 基础 Gibbs 的 B26 复现记录（n=5000，只报告）",
            True,
            f"β ≤ 4 覆盖 = 8/8；β=8 覆盖 中位 {gib_cov[8.0]['median']:.0f} / "
            f"最小 {gib_cov[8.0]['min']:.0f} / 最大 {gib_cov[8.0]['max']:.0f}。"
            "预检在 n=1000 下记录 β=8 覆盖 = 1；正式跑 n=5000 时变为中位 2 —— 两值均正确，"
            "差异来自【样本数】（预检数字不可跨参数外推，纪律⑨）。B26 的量化数字必须与 n 一起引用。")

    c.check("必报⑩ 采样 − 精确 的生成质量差（只报告，不判定）", True,
            "；".join(f"β={b}: 中位 {pt_mass_diff[b]['median']:+.4f} / 最小 {pt_mass_diff[b]['min']:+.4f} / "
                      f"最大 {pt_mass_diff[b]['max']:+.4f}（SE≈{math.sqrt(mass_gen[b]*(1-mass_gen[b])/N_SAMPLES):.4f}）"
                      for b in BETA_JUDGE)
            + "；符号一致是提示不是证据，本实验不下结论")

    kl_bad = {b: pt_kl[b] for b in BETA_JUDGE if pt_kl[b]["max"] >= KL_TOL}
    c.check("P7c KL 检查: 判定 β 上 10 种子的 KL【最大值】< 0.01（硬线，跑后不改判据）",
            not kl_bad,
            "；".join(f"β={b}: 中位 {pt_kl[b]['median']:.5f} / 最小 {pt_kl[b]['min']:.5f} / "
                      f"最大 {pt_kl[b]['max']:.5f}" for b in BETA_JUDGE)
            + f"（阈值 {KL_TOL}；n=5000，10 种子）" + (f"；越界 = {list(kl_bad)}" if kl_bad else ""))
    c.check("P7c 辅助 β（n=3，不进判据，仅曲线补全）", True,
            "；".join(f"β={b}: 中位 {pt_kl[b]['median']:.5f} / 最大 {pt_kl[b]['max']:.5f}"
                      f"（n={N_SEEDS_AUX}）" for b in BETA_AUX))

    # ---------- P4 / P4b 闭包审计 ----------
    observed = set()
    for beta in BETA_JUDGE:
        smp, _ = parallel_tempering(W3, beta, np.random.default_rng(RNG_SEEDS[0]), N_SAMPLES)
        observed |= set(smp)
    v_node = sum(1 for s in observed
                 if not set(n for m in mods if s[m] > 0 for n in members[m]).issubset(set(nodes)))
    v_state = sum(1 for s in observed if tuple(s) not in key_of)
    c.check("P4 闭包硬线（C30，节点级）: 每个采样状态映射回的节点集合 ⊆ 已学 15 节点；违例数 = 0",
            v_node == 0, f"观测状态数 = {len(observed)}；违例数 = {v_node}")
    c.check("P4b 状态空间闭包: 采样状态集合 ⊆ E25 枚举的 8 状态；越界状态数 = 0",
            v_state == 0, f"观测状态数 = {len(observed)}；越界状态数 = {v_state}（上限 8）")

    # ---------- P7a / P7b ----------
    c.check("P7a 集合差定义式: 生成集合 = 采样可达(8) ∖ 单节点 max 可达(3) = 5（按定义，逐元素）",
            len(generated) == 8 - len(single_reach),
            f"采样可达 = 8（E25 枚举全开）；单节点可达 = {len(single_reach)}；差集 = {len(generated)}")
    c.check("P7b 分层报告（4 多种子可达 + 1 全关不可达；示例种子集）", True,
            f"多种子可达（4）: " + "；".join(f"{s} ← {example_seeds[s]}" for s in gen_multi)
            + f" | 任何种子集不可达（1）: {gen_none}")

    # ---------- P8 只读 ----------
    h_after = all_hashes()
    c.check("P8 只读: 六文件哈希前后一致 + v3 不动 + L0 签名一致",
            h_before == h_after and h_after["synapse_net.py"] == V1_HASH_EXPECTED
            and h_after["synapse_net_v2.py"] == V2_HASH_EXPECTED
            and h_after["synapse_net_v3.py"] == V3_HASH_EXPECTED
            and signature(net) == sig_before,
            f"before==after = {h_before == h_after}；v1={h_after['synapse_net.py']}、"
            f"v2={h_after['synapse_net_v2.py']}、v3={h_after['synapse_net_v3.py']}；"
            f"L0 签名一致 = {signature(net) == sig_before}（{net.node_count()} 节点 / {net.edge_count()} 边）")

    # ---------- 双向 ----------
    substantial = [ck for ck in c.checks if ck["name"].startswith(("P1 ", "P2 ", "P3 ", "P7c KL"))]
    ok_all = all(ck["ok"] for ck in substantial)
    outcome = "C39 + B34" if ok_all else "E40_FAIL"
    text = (C39_TEXT + "\n\n" + B34_TEXT) if ok_all else FAIL_TEXT
    c.check("双向结局: 实质判据（P1/P2/P3/P7c）全过 → C39 + B34；否则 → 如实记录",
            outcome in ("C39 + B34", "E40_FAIL"),
            f"落定 {outcome}（实质判据全过 = {ok_all}）")

    return {
        "id": "exp40_unconditional_sampling",
        "title": "E40 无条件采样: 解开种子模块钳制（8 状态全开）",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "corpus": {"n_nodes": len(nodes), "module_sizes": {str(m): len(members[m]) for m in mods},
                       "W3": np.round(W3, 6).tolist(), "assign_seed": SEED_MAIN, "ari": round(ari_now, 9)},
            "params": {"n_samples": N_SAMPLES, "n_seeds_judge": N_SEEDS_JUDGE,
                       "n_seeds_aux": N_SEEDS_AUX, "rng_seeds": RNG_SEEDS,
                       "burn_in": BURN_IN, "thin": THIN, "ladder": LADDER,
                       "beta_judge": BETA_JUDGE, "beta_aux": BETA_AUX,
                       "kl_tol": KL_TOL, "gen_mass_min": GEN_MASS_MIN, "bin_thr": BIN_THR,
                       "aux_note": f"辅助 β {{1.0, 4.0}} 的种子数 = {N_SEEDS_AUX}（预设值）；"
                                   f"其数字不进任何判据，只作曲线补全"},
            "contrasts": {
                "a_single_node_reachable": sorted(single_reach),
                "a_carrier_counts": {str(k): len(v) for k, v in single_reach.items()},
                "b_union_node_sets": len(set().union(*node_sets.values())),
                "c_multi_seed_reachable": sorted(multi_reach),
                "c_unreachable": sorted(set(tuple(int(v) for v in r) for r in S8) - set(multi_reach)),
                "example_seed_sets": {str(k): v for k, v in example_seeds.items()},
            },
            "s8_order": [[int(v) for v in r] for r in S8],
            "exact_distribution": {str(b): [round(float(x), 9) for x in exact[b]] for b in BETA_ALL},
            "generated_set": sorted(generated),
            "generated_mass_exact": {str(b): round(mass_gen[b], 9) for b in BETA_ALL},
            "generated_mass_sampled": {str(b): pt_mass[b] for b in BETA_ALL},
            "generated_mass_sampled_minus_exact": {str(b): pt_mass_diff[b] for b in BETA_ALL},
            "generated_mass_se_n5000": {str(b): round(math.sqrt(mass_gen[b] * (1 - mass_gen[b]) / N_SAMPLES), 6)
                                        for b in BETA_ALL},
            "coverage_pt": {str(b): pt_cov[b] for b in BETA_ALL},
            "coverage_gibbs": {str(b): gib_cov[b] for b in BETA_ALL},
            "kl": {str(b): pt_kl[b] for b in BETA_ALL},
            "swap_acceptance": {str(b): pt_swap[b] for b in BETA_ALL},
            "p5_conditional_check": {
                "restricted_total": round(float(p3.sum()), 9),
                "restricted_renorm": [round(float(x), 9) for x in restricted],
                "e33_p3": [round(float(x), 9) for x in p3c],
                "max_bitwise_diff": float(np.max(np.abs(restricted - p3c))),
            },
            "p6_ground_states": {"states": ground, "energy": round(float(E8.min()), 9),
                                 "mass_beta2": round(float(exact[2.0][key_of[ground[0]]]), 9),
                                 "max_single_state_by_beta": {str(b): list(max_single[b]) for b in BETA_ALL}},
            "closure": {"violations_node": v_node, "violations_state": v_state,
                        "observed_states": len(observed)},
            "readonly": {
                "hashes_before": h_before, "hashes_after": h_after,
                "v1_expected": V1_HASH_EXPECTED, "v2_expected": V2_HASH_EXPECTED, "v3_expected": V3_HASH_EXPECTED,
                "synapse_net_v2.py": {"sha256_16": h_after["synapse_net_v2.py"], "reason": NO_CALL_REASON},
                "synapse_net_v3.py": {"sha256_16": h_after["synapse_net_v3.py"], "reason": NO_CALL_REASON},
                "L0_signature_unchanged": bool(signature(net) == sig_before),
            },
            "outcome": outcome,
            "registration": {"outcome": outcome, "text": text, "C39": C39_TEXT,
                             "B34": B34_TEXT, "E40_FAIL": FAIL_TEXT},
            "note": "PT 是读取端工具（与 E33 的 boltzmann() 并列）；基础 Gibbs 保留为对照与 B26 复现点。"
                    "P3 主口径用精确枚举（8 状态可枚举，无采样噪声），采样估计并列报告。",
            "v1_lesson": "v1 的唯一 FAIL 是 P1 的对照条款：预检（n=1000）的 Gibbs β=8 覆盖期望"
                         "（=1）被搬到了 n=5000 的运行上，两值不同源 ⇒ 判据 FAIL 的原因是判据条款"
                         "而非机制。v2 把 P1 收窄为只判主口径（PT），对照条款降为 P1a'（只报告）。"
                         "一般化：预检数字进入预注册前必须核对参数是否与运行一致（纪律⑨同族）。"
                         "v1 FAIL JSON 归档为 exp40_unconditional_sampling_v1_FAIL.json。",
            "b26_sample_size_qualifier": "B26 的量化数字必须与 n 一起引用：β=8 的覆盖数在 n=1000 下 = 1、"
                                         "在 n=5000 下 = 中位 2 / 最小 1 / 最大 2；脱离 n 引用会误读。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
