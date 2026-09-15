"""E35: 工作记忆的串联 —— 携带的是"重注入"还是"图传播"？

严格按 DESIGN-E35.md。**无 numpy 依赖；不改任何层文件；不新增层。**

预注册要点:
  - W 复用 E34 的同一实现（import，不重写）：k=3，tie-break = 值降序、同值节点名升序；
  - P_a（对照，构造性）：每步重注入 → seed1 恒 1.0、W 集合从步 2 起恒定（不作正结果）；
  - P_b（实证）：只注入一次 → 痕迹退化为纯图衰减 0.9^13 = 0.254186583（容差 1e-9）；
  - P_c（平权性）：被携带成员的值全为 1.0；变体3 里新种子 D/E 挤不进 W；
  - 双向：P_b + P_c 通过 → C35；否则 → B28。
"""

from __future__ import annotations

import json
from pathlib import Path

from synapse_net import SynapseNet

from experiments.checker import Checker
from experiments.exp31_auto_scale import layer_hashes
from experiments.exp34_working_memory import WorkingMemory, signature

ALPHA = [chr(ord("A") + i) for i in range(26)]
REPS = 100
K = 3
N_STEPS = 5
DECAY = 0.9 ** 13          # 0.9^13 = 0.254186583（Z→M 共 13 跳）
TOL = 1e-9

C35_TEXT = (
    "C35：W 栈可以在多步链路中携带 N 个痕迹，但"
    "(a) 只在【每步重注入】时维持——停止重注入后痕迹立即退化为纯图衰减（0.9^13 = 0.254187），"
    "W 不留下任何持续影响；"
    "(b) 痕迹之间【无优先级、无时间顺序】（被携带成员的值全部 1.0）；"
    "(c) 成员由 tie-break 决定，新痕迹可能挤不进去（变体3：D/E 均未进 W）。"
    "⇒ 这是【载体】行为，不是【状态整合】。"
    "范围限定：26 字母链语料；W 仍是实验侧读取端工具（同 E34）。"
    "边界登记：要区分「哪个痕迹来自哪一步」，至少需要 ① 顺序标签（写入 W 时记时间戳）"
    "或 ② 值域编码（保留种子原始激活值）；两者都不在 E35 范围内——E36 的入口。"
)
B28_TEXT = (
    "B28：多步串联的行为与「重注入 + 值抹平 + tie-break」三者决定的三条性质不符"
    "（具体偏差见 JSON）；W 栈的携带在该语料上需要重新定位机制。"
)


def build_chain() -> SynapseNet:
    net = SynapseNet()
    for a, b in zip(ALPHA, ALPHA[1:]):
        for _ in range(REPS):
            net.learn([a, b])
    return net


def run() -> dict:
    c = Checker()
    hashes_before = layer_hashes()
    net = build_chain()
    sig_before = signature(net)
    seed1 = "M"

    # ---------- P_a：每步重注入（构造性对照）----------
    chain_a, prev = [], None
    for step in range(1, N_STEPS + 1):
        seeds = [seed1] if step == 1 else ["Z"] + prev.seeds()
        out = net.activate(seeds)
        w = WorkingMemory(K)
        w.hold(out)
        chain_a.append({"step": step, "seeds": list(seeds), "W": w.snapshot(),
                        "W_seeds": w.seeds(), "seed1_value": round(float(out[seed1]), 9)})
        prev = w
    w_sets_constant = len({tuple(c["W_seeds"]) for c in chain_a[1:]}) == 1
    seed1_all_one = all(abs(c["seed1_value"] - 1.0) <= TOL for c in chain_a)

    # ---------- 变体4：步数扫描（每步重注入）----------
    scan4 = []
    for n in (2, 3, 5, 8):
        prev = None
        for step in range(1, n + 1):
            seeds = ["Z"] if step == 1 else ["Z"] + prev.seeds()
            out = net.activate(seeds)
            w = WorkingMemory(K)
            w.hold(out)
            prev = w
        scan4.append({"N": n, "W": prev.seeds(), "seed1_Z_value": round(float(out["Z"]), 9)})

    # ---------- P_b：只注入一次 ----------
    w1 = WorkingMemory(K)
    w1.hold(net.activate(seed1))
    out2 = net.activate(["Z"] + w1.seeds())
    w2 = WorkingMemory(K)
    w2.hold(out2)
    chain_b = [{"step": 2, "seeds": ["Z"] + w1.seeds(),
                "seed1_value": round(float(out2[seed1]), 9)}]
    for step in (3, 5):
        out = net.activate(["Z"])          # 不再注入
        chain_b.append({"step": step, "seeds": ["Z"], "seed1_value": round(float(out[seed1]), 9)})
    pb_vals = [r["seed1_value"] for r in chain_b if r["step"] >= 3]
    pb_ok = all(abs(v - DECAY) <= TOL for v in pb_vals)

    # ---------- P_c：平权性 ----------
    carried_values = [sorted(c["W"].values()) for c in chain_a[1:]]
    all_one = all(all(abs(v - 1.0) <= TOL for v in vals) for vals in carried_values)
    chain3 = []
    prev = None
    for step, seed in enumerate(["A", "B", "C", "D", "E"], start=1):
        seeds = [seed] + (prev.seeds() if prev else [])
        out = net.activate(seeds)
        w = WorkingMemory(K)
        w.hold(out)
        chain3.append({"step": step, "seed": seed, "seeds": list(seeds), "W_seeds": w.seeds()})
        prev = w
    last_w = set(chain3[-1]["W_seeds"])
    pc_ok = all_one and ("D" not in last_w) and ("E" not in last_w)

    # ---------- P1a / P1b ----------
    hashes_after = layer_hashes()
    c.check("P1a 接口不改: 四个层文件 sha256 前后一致",
            hashes_before == hashes_after,
            f"before={hashes_before}；after={hashes_after}")
    sig_after = signature(net)
    c.check("P1b L0 不改: 节点数/边数/边集合签名逐位相同",
            sig_before == sig_after,
            f"{sig_before['nodes']} 节点 / {sig_before['edges']} 边；签名相同={sig_before == sig_after}")

    c.check("P_a（对照，构造性，不作正结果）: 每步重注入 → seed1 恒 1.0、W 集合从步2起恒定",
            seed1_all_one and w_sets_constant,
            f"seed1 值逐步={[r['seed1_value'] for r in chain_a]}；"
            f"W 集合从步2起恒定={w_sets_constant}（内容={chain_a[-1]['W_seeds']}）")

    c.check("P_b 实证: 只注入一次后，痕迹 = 纯图衰减 0.9^13 = 0.254186583",
            pb_ok,
            f"步3/步5 的 seed1={pb_vals}；期望 {DECAY:.9f}；"
            f"最大偏差={max(abs(v - DECAY) for v in pb_vals):.3e}（容差 {TOL:g}）")

    c.check("P_c 平权性: ① 被携带成员值全为 1.0；② 变体3 里 D/E 均未挤入 W",
            pc_ok,
            f"① 各步被携带值={carried_values}（全 1.0={all_one}）；"
            f"② 变体3 最终 W={chain3[-1]['W_seeds']}，D 在={'D' in last_w}，E 在={'E' in last_w}")

    ok = pb_ok and pc_ok
    outcome = "C35" if ok else "B28"
    text = C35_TEXT if ok else B28_TEXT
    c.check("P5 双向结局: P_b + P_c 通过 → C35；否则 → B28",
            outcome in ("C35", "B28"),
            f"落定 {outcome}（P_b={pb_ok}, P_c={pc_ok}）")

    return {
        "id": "exp35_working_memory_chain",
        "title": "E35 工作记忆串联: 携带的是重注入还是图传播",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "spec": {"K": K, "N_STEPS": N_STEPS,
                     "tie_break": "值降序；同值按节点名升序（预注册）",
                     "decay_reference": {"formula": "0.9^13", "value": round(DECAY, 9),
                                         "derivation": "M=第13个字母，Z=第26个；Z→M 共 26-13=13 跳"}},
            "Pa_reinjection_chain": chain_a,
            "Pa_W_sets_constant_from_step2": bool(w_sets_constant),
            "Pb_single_injection": {"chain": chain_b, "expected": round(DECAY, 9),
                                    "max_deviation": max(abs(v - DECAY) for v in pb_vals)},
            "Pc_equality": {"carried_values_per_step": carried_values,
                            "all_carried_are_1.0": bool(all_one),
                            "variant3_chain": chain3,
                            "variant3_final_W": chain3[-1]["W_seeds"],
                            "D_in_final_W": bool("D" in last_w), "E_in_final_W": bool("E" in last_w)},
            "aux_variant4_step_scan": scan4,
            "readonly": {"layer_hashes_before": hashes_before, "layer_hashes_after": hashes_after,
                         "L0_signature_same": sig_before == sig_after,
                         "L0_nodes": sig_before["nodes"], "L0_edges": sig_before["edges"]},
            "outcome": outcome,
            "registration": {"outcome": outcome, "text": text, "C35": C35_TEXT, "B28": B28_TEXT},
            "note": "W 复用 E34 的同一实现（import）；W 不进 LAYERS、不改 BOUNDARY。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
