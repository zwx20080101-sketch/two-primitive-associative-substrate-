"""E41: 工作记忆的值域编码（痕迹的排序可测）。

严格按 DESIGN-E41.md。**W 层一字不动（exp34 不改）；不改任何层文件。**

预注册要点:
  - 抹平不在 W 层、在注入调用点（§0.1）：W.items 从 E34 起就是 dict[str,float]，
    seeds() 丢值、snapshot() 带值 ⇒ 本实验只改调用点（1 行），W 层 0 行。
  - 形态 A（dict 直接注入）唯一合理；形态 B 经实测排除。
  - P2 顺序可测 ≥ 10（预检 13）；P5 恒等式 X = 5；P3 扩散节点【逐节点列】。
  - 本实验【不预分配 FAIL 分支的 B 编号】；机制性 FAIL 才在跑完后新立 B 编号。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.checker import Checker
from experiments.exp24_module_differentiation import ALPHA
from experiments.exp34_working_memory import WorkingMemory
from synapse_net import SynapseNet as V1
from synapse_net_v2 import SynapseNet as V2

REPS = 100
TOL = 1e-12
K = 3
SEED = "Z"
DIFF_MIN_P2 = 10            # 预设阈值（预检 13）
NON_SEED_MIN_P5 = 5         # 预设阈值 X（预检 11）

V1_HASH_EXPECTED = "cf99a07a0e60fd4d"
V2_HASH_EXPECTED = "ee6df589eb2f6983"
V3_HASH_EXPECTED = "085be73a06f0694c"
LAYER_HASH_EXPECTED = {"order_layer.py": "419d54745a9eb795",
                       "context_layer.py": "bd029d7bbdc345be",
                       "chunk_layer.py": "46f5857be0fc7c7e"}
HASH_FILES = ["synapse_net.py", "synapse_net_v2.py", "synapse_net_v3.py",
              "order_layer.py", "context_layer.py", "chunk_layer.py"]
W_FILE = "experiments/exp34_working_memory.py"

C40_TEXT = (
    "C40：W 层保存的痕迹值经【已登记的带值种子入口】（v2，C37 + B30）注入基底时，"
    "痕迹的【排序可测】—— 同节点集、值序对调（A/C 互换）→ 基底输出在 13 个节点上不同，"
    "其中 11 个是【非种子（扩散）节点】（D…N，逐节点值见 JSON）。"
    "⇒ C36 的「分类 ≠ 排序」里【排序轴】的限定被解除（分类轴不变）。"
    "范围限定：① 这叫【值序可测】，不是【时间序可测】—— C35 的「无时间顺序、无优先级」"
    "边界【仍然成立】；② 功劳归外部 W 层工具（从 E34 起就把值存在 items 里）+ v2 的已登记入口，"
    "【不是基底新能力】；③ 本实验不改 W 层（0 行），只改注入调用点（1 行）；"
    "v1/v2/v3 一字不动。语料限于 E1 的 26 字母链。"
)
C36_UPDATE_TEXT = (
    "【C36 的状态更新（E41，2026-09-16）】"
    "C36 原文的排序轴限定（「排序差异需要值域编码，不在本实验」）在 E41 下被【解除】，"
    "前提是走 E37 已登记的带值入口（C37 + B30）：同集合、值序对调 → 基底输出在 13 个节点上不同"
    "（含 11 个扩散节点）。C36 的原文保留（记录 E36 当时的真实判断）；本条款只更新状态，"
    "不改 C36 的判据与范围（【分类可测】那一轴不变）。"
    "【与 E38 对 C33 的做法方向相反】：E38 是【收窄】（加限定），E41 是【解除】（去限定）——"
    "两者都不改原条目的判据，只更新其适用状态。"
)
OVERREACH_STATUS_TEXT = (
    "【越权嫌疑的状态说明（E41，2026-09-16）】"
    "C36 写「越权嫌疑」时（E36，2026-09-14），带值 mapping 入口【尚不存在】。"
    "E37（2026-09-15）已由用户拍板落地并登记为 C37 + B30。"
    "E41 走的是【已登记的合法入口】，不是新改基底。"
    "故「越权嫌疑」这一分句在 E41 之后【不适用】—— 但保留原文，作为时间戳证据。"
)
C35_ANNOTATION_TEXT = (
    "【C35 的状态注解（E41，2026-09-16）】"
    "C35 原文的「无优先级」记录的是 E35 当时的观察（被携带成员的值全部 1.0，注入口抹平）。"
    "E41 之后，该表述的精确含义是：W 自身不排序；排序信息来自被保存的 value，注入时由入口保留。"
    "C35 原文保留；本注解只澄清语境，不改 C35 的判据与范围。"
)
FAIL_TEXT = (
    "E41 未通过：哪条判据挂、实测值见 exp41 JSON。失败分层：① W 层是否保存了值（P0/P1）"
    "→ ② 注入是否传了值（P1）→ ③ 基底是否保留值（B30 已确认）。"
    "若 FAIL 是【机制性】的，跑完后新立 B 编号；若是【实现/判据问题】，修正后重跑（不新立编号）。"
)


def _sha16(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def all_hashes() -> dict:
    return {f: _sha16(f) for f in HASH_FILES}


def build(cls):
    net = cls()
    for a, b in zip(ALPHA, ALPHA[1:]):
        for _ in range(REPS):
            net.learn([a, b])
    return net


def run() -> dict:
    c = Checker()
    h_before = all_hashes()
    w_hash_before = _sha16(W_FILE)
    net1, net2 = build(V1), build(V2)

    # ---------- P0 前置 ----------
    acts_a = net1.activate("A")
    W = WorkingMemory(k=K)
    W.hold(acts_a)
    w_items = {n: round(float(v), 9) for n, v in W.items.items()}
    w_seeds = W.seeds()
    w_snapshot = {n: round(float(v), 9) for n, v in W.snapshot().items()}
    c.check("P0 前置: W 层可跑通（hold/seeds/snapshot 与 E34 行为一致）+ v2 带值入口可用",
            set(w_items) == set(w_seeds) == set(w_snapshot) and len(w_items) == K
            and isinstance(net2.activate({"Z": 1.0}), dict),
            f"W.items={w_items}；W.seeds()={w_seeds}（值被丢弃）；W.snapshot()={w_snapshot}（带值出口）；"
            f"v2.activate(dict) 可用")

    # ---------- 注入与对照 ----------
    cfg1 = {SEED: 1.0, **W.snapshot()}
    cfg2 = {SEED: 1.0, "A": 0.81, "B": 0.9, "C": 1.00}
    o1, o2 = net2.activate(cfg1), net2.activate(cfg2)
    seeds_union = set(cfg1) | set(cfg2)
    diff_nodes = [k for k in ALPHA if abs(o1[k] - o2[k]) > TOL]
    diff_seed = [k for k in diff_nodes if k in seeds_union]
    diff_spread = [k for k in diff_nodes if k not in seeds_union]
    flat1 = net2.activate([SEED] + W.seeds())
    flat2 = net2.activate([SEED, "C", "B", "A"])
    flat_diff = [k for k in ALPHA if abs(flat1[k] - flat2[k]) > TOL]

    # ---------- P1 值域保留 ----------
    p1_ok = all(abs(o1[n] - W.items[n]) <= TOL for n in W.items)
    c.check("P1 值域保留: 注入后种子节点取值 = W 保存的原始值（逐位；依据 B30 常量语义）",
            p1_ok,
            "；".join(f"{n}: W={W.items[n]:.6f} → 输出={o1[n]:.6f}" for n in sorted(W.items)))

    # ---------- P2 顺序可测（主判据）----------
    c.check(f"P2 顺序可测: 同集合、值序对调 ⇒ 差异节点数 ≥ {DIFF_MIN_P2}（预设阈值）",
            len(diff_nodes) >= DIFF_MIN_P2 and len(flat_diff) == 0,
            f"形态 A 差异节点数 = {len(diff_nodes)}（阈值 ≥{DIFF_MIN_P2}；预检 13）"
            + ("（= 13，与预检一致 ✓）" if len(diff_nodes) == 13 else
               f"（与预检 13 相差 {len(diff_nodes) - 13}；v2 是确定性的，偏离提示实现有出入）")
            + f"；抹平对照（list 形式）差异节点数 = {len(flat_diff)}（E36 结论复现）")

    # ---------- P3 差异分层（逐节点列）----------
    spread_rows = [(k, round(o1[k], 9), round(o2[k], 9)) for k in diff_spread]
    c.check("P3 差异分层: 种子层（平凡，= 输入差）与扩散层（真传播）分开；扩散层【逐节点列出】",
            len(diff_spread) >= 1 and len(spread_rows) == len(diff_spread),
            f"种子层 {len(diff_seed)} 个 = {diff_seed}"
            f"（A 1.0→0.81、C 0.81→1.0，即输入差本身）；"
            f"扩散层 {len(diff_spread)} 个 = {[r[0] for r in spread_rows]}；"
            f"逐节点（节点, config1, config2）= {spread_rows}")

    # ---------- P4 与 E36 的对照 ----------
    c.check("P4 与 E36 的对照（报告项）: 分类轴（E36 可测）不变；排序轴 E36 不可测 → E41 可测",
            True,
            f"分类轴：E36 已证可测（seeds(step=1) 只返回步1 的 3 个节点；C36 登记）——本实验不变；"
            f"排序轴：E36 不可测（抹平注入下差异 = {len(flat_diff)}）→ E41 可测（差异 = {len(diff_nodes)}）"
            f" ⇒ C36 的排序轴限定【被解除】（C36 原文保留，加状态更新条款）")

    # ---------- P5 恒等式检查 ----------
    non_seed_diff = sum(1 for k in ALPHA if k not in set(cfg1) and abs(o1[k] - o2[k]) > TOL)
    seed_out_eq_in = all(abs(o1[n] - cfg1[n]) <= TOL for n in cfg1)
    c.check(f"P5 恒等式检查: 注入种子数 < 总数 且 至少 {NON_SEED_MIN_P5} 个非种子节点上输出 ≠ 输入",
            len(cfg1) < len(ALPHA) and non_seed_diff >= NON_SEED_MIN_P5,
            f"注入种子 {len(cfg1)} / {len(ALPHA)} 节点（非种子 {len(ALPHA)-len(cfg1)} 个）⇒ 非恒等式；"
            f"非种子节点中 config1≠config2 的个数 = {non_seed_diff}（阈值 ≥{NON_SEED_MIN_P5}，预检 11）；"
            f"种子节点上「输出 == 输入」= {seed_out_eq_in}（B30 常量语义的已知性质，不是陷阱）")

    # ---------- P6 只读 ----------
    h_after = all_hashes()
    w_hash_after = _sha16(W_FILE)
    p6_ok = (h_before == h_after and h_after["synapse_net.py"] == V1_HASH_EXPECTED
             and h_after["synapse_net_v2.py"] == V2_HASH_EXPECTED
             and h_after["synapse_net_v3.py"] == V3_HASH_EXPECTED
             and all(h_after[f] == v for f, v in LAYER_HASH_EXPECTED.items())
             and w_hash_before == w_hash_after)
    c.check("P6 只读: 六文件哈希前后一致 + exp34 的 W 层哈希前后一致",
            p6_ok,
            f"六文件 before==after = {h_before == h_after}；"
            f"v1={h_after['synapse_net.py']} v2={h_after['synapse_net_v2.py']} v3={h_after['synapse_net_v3.py']}；"
            f"order/context/chunk = {h_after['order_layer.py']}/{h_after['context_layer.py']}/{h_after['chunk_layer.py']}；"
            f"exp34_working_memory.py = {w_hash_after}（前后一致 = {w_hash_before == w_hash_after}，W 层 0 行改动）")

    # ---------- P7 双向 ----------
    ok_all = all(ck["ok"] for ck in c.checks)
    outcome = "C40 + C36 状态更新 + C35 注解 + 越权嫌疑状态说明" if ok_all else "E41_FAIL"
    text = ("\n\n".join([C40_TEXT, C36_UPDATE_TEXT, OVERREACH_STATUS_TEXT, C35_ANNOTATION_TEXT])
            if ok_all else FAIL_TEXT)
    c.check("P7 双向结局: P1+P2+P3+P5 全过 → C40 + 三条状态条款；否则 → 如实记录（不预分配 B 编号）",
            ok_all,
            f"落定 {'C40 + 三条状态条款' if ok_all else 'E41_FAIL'}（前序判据全过 = {ok_all}）"
            + ("；未预分配 FAIL 分支编号（机制性 FAIL 才在跑完后新立）" if ok_all else ""))

    return {
        "id": "exp41_seed_value_encoding",
        "title": "E41 工作记忆的值域编码: 痕迹的排序可测（抹平在调用点，不在 W 层）",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "corpus": {"kind": "E1 26 字母链", "nodes": len(ALPHA), "reps": REPS},
            "params": {"k": K, "seed": SEED, "diff_min_p2": DIFF_MIN_P2,
                       "non_seed_min_p5": NON_SEED_MIN_P5, "tolerance": TOL,
                       "form": "形态 A（dict 直接注入）—— 形态 B 经实测排除（cfgA == cfgB）"},
            "W_layer": {"file": W_FILE, "items": w_items, "seeds_returns": w_seeds,
                        "snapshot": w_snapshot,
                        "note": "items 从 E34 起就是 dict[str,float]；seeds() 丢值；snapshot() 带值出口已存在"},
            "configs": {"config1": {k: round(v, 9) for k, v in cfg1.items()},
                        "config2": {k: round(v, 9) for k, v in cfg2.items()}},
            "P2_order": {"diff_nodes_total": len(diff_nodes), "diff_nodes": diff_nodes,
                         "flattened_control_diff": len(flat_diff)},
            "P3_layers": {"seed_layer": diff_seed, "seed_layer_values": {k: [round(o1[k], 9), round(o2[k], 9)]
                                                                        for k in diff_seed},
                          "spread_layer": [r[0] for r in spread_rows],
                          "spread_layer_rows": [{"node": r[0], "config1": r[1], "config2": r[2]}
                                                for r in spread_rows]},
            "P4_axes": {"classification": {"experiment": "E36", "measurable": True, "registered": "C36"},
                        "ordering": {"experiment_before": "E36", "measurable_before": False,
                                     "experiment_after": "E41", "measurable_after": True,
                                     "note": "C36 原文保留，加状态更新条款（解除排序轴限定）"}},
            "P5_identity": {"injected_seeds": len(cfg1), "total_nodes": len(ALPHA),
                            "non_seed_differing": non_seed_diff,
                            "seed_nodes_output_equals_input": bool(seed_out_eq_in)},
            "E37_vs_E41_13_note": "E37 的差异集是 {B,C} ∪ D..N（对调 B/C）；E41 的是 {A,C} ∪ D..N（对调 A/C）。"
                                  "数目同为 13，集合不同 —— 两实验的对照配置不同，不是同源。",
            "readonly": {"hashes_before": h_before, "hashes_after": h_after,
                         "v1_expected": V1_HASH_EXPECTED, "v2_expected": V2_HASH_EXPECTED,
                         "v3_expected": V3_HASH_EXPECTED, "layer_expected": LAYER_HASH_EXPECTED,
                         "W_file": {"path": W_FILE, "sha256_16": w_hash_after,
                                    "unchanged": bool(w_hash_before == w_hash_after)}},
            "outcome": outcome,
            "registration": {"outcome": outcome, "text": text, "C40": C40_TEXT,
                             "C36_update": C36_UPDATE_TEXT,
                             "overreach_status": OVERREACH_STATUS_TEXT,
                             "C35_annotation": C35_ANNOTATION_TEXT,
                             "E41_FAIL": FAIL_TEXT,
                             "no_preallocated_B": True},
            "note": "抹平不在 W 层、在注入调用点：W 层 0 行改动，只改调用点 1 行。"
                    "本实验不预分配 FAIL 分支的 B 编号（机制性 FAIL 才在跑完后新立）。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
