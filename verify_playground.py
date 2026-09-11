"""P5a 校验：网页内嵌快照 ↔ 冻结 JSON 逐字段一致（V1）+ 面板边界（V3/V4/V5/V6）。

用法: python verify_playground.py
退出码: 1 表示存在不一致。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "playground.html"

BANNED = ["意识", "理解", "自由能", "灵魂", "主观体验"]

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def main() -> int:
    html = HTML.read_text(encoding="utf-8")

    m = re.search(r'<script id="analysisSnapshot" type="application/json">(.*?)</script>', html, re.S)
    if not m:
        check("V1 内嵌快照存在", False, "未找到 analysisSnapshot")
        return report()
    snap = json.loads(m.group(1))

    d24 = json.loads((ROOT / "outputs/exp24_module_differentiation.json").read_text(encoding="utf-8"))["data"]
    d25 = json.loads((ROOT / "outputs/exp25_module_dynamics.json").read_text(encoding="utf-8"))["data"]
    d26 = json.loads((ROOT / "outputs/exp26_module_coordination.json").read_text(encoding="utf-8"))["data"]

    # ---------- V1 逐字段一致 ----------
    pairs = [
        ("S1.modules", snap["S1"]["modules"], d24["C1"]["modules_spectral"]),
        ("S1.assign", snap["S1"]["assign"], d24["C1"]["assign_spectral"]),
        ("S1.truth", snap["S1"]["truth"], d24["C1"]["truth"]),
        ("S1.ari_spectral", snap["S1"]["ari_spectral"], d24["C1"]["ARI_spectral"]),
        ("S1.stability_ari", snap["S1"]["stability_ari"], d24["C1"]["stability_ARI_spectral"]),
        ("S1.within_between_ratio", snap["S1"]["within_between_ratio"], d24["C1"]["within_between_ratio"]),
        ("S1.identifiability_min_gap", snap["S1"]["identifiability_min_gap"], d24["C1"]["identifiability_min_gap"]),
        ("S1.null_enrichment", snap["S1"]["null_enrichment"], d24["C4_null"]["enrichment"]),
        ("S2.rho", snap["S2"]["rho"], d25["rho_C1"]),
        ("S2.block_matrix", snap["S2"]["block_matrix"], d25["W_C1"]),
        ("S2.basins_per_module", snap["S2"]["basins_per_module"], d25["C1"]["basins_per_module"]),
        ("S2.state_separation", snap["S2"]["state_separation"], d25["C1"]["state_separation"]),
        ("S2.joint_attractor_patterns", snap["S2"]["joint_attractor_patterns"], d25["C1"]["joint_attractor_patterns"]),
        ("S2.hysteresis_width", snap["S2"]["hysteresis_width"], d25["C1"]["hysteresis_width"]),
        ("S2.up_transition", snap["S2"]["up_transition"], d25["C1"]["up_transition"]),
        ("S2.down_transition", snap["S2"]["down_transition"], d25["C1"]["down_transition"]),
        ("S2.control_width", snap["S2"]["control_width"], d25["C2"]["hysteresis_width"]),
        ("S2.unclassified", snap["S2"]["unclassified"], d25["C1"]["unclassified_ratio"]),
        ("S3.w_values", snap["S3"]["w_values"], d26["dose_response"]["w_values"]),
        ("S3.EI_int_mean", snap["S3"]["EI_int_mean"], d26["dose_response"]["EI_int_mean"]),
        ("S3.EI_int_per_seed", snap["S3"]["EI_int_per_seed"], d26["dose_response"]["EI_int_per_seed"]),
        ("S3.monotone_per_seed", snap["S3"]["monotone_per_seed"], d26["dose_response"]["monotone_per_seed"]),
        ("S3.min_EI_int_w65", snap["S3"]["min_EI_int_w65"], d26["dose_response"]["min_EI_int_w65"]),
        ("S3.C2_prime_EI_int", snap["S3"]["C2_prime_EI_int"], d26["C2_prime_EI_int"]),
        ("S3.ablation_EI_int", snap["S3"]["ablation_EI_int"], d26["ablation_EI_int"]),
        ("S3.experience_curve", snap["S3"]["experience_curve"], d26["experience_curve"]),
        ("S3.asym_C1", snap["S3"]["asym_C1"], d26["asym_C1_prime"]),
        ("S3.rho_C1", snap["S3"]["rho_C1"], d26["rho_C1"]),
    ]
    bad = [(n, a, b) for n, a, b in pairs if a != b]
    check("V1 逐字段一致（网页快照 == outputs/*.json）", not bad,
          f"{len(pairs) - len(bad)}/{len(pairs)} 字段一致" + (f"；不一致：{bad[:3]}" if bad else ""))

    # ---------- V6 数据来源可追溯 ----------
    prov_ok, prov_detail = True, []
    for key in ("exp24", "exp25", "exp26"):
        p = snap["provenance"].get(key, {})
        good = bool(re.fullmatch(r"[0-9a-f]{7,40}", str(p.get("commit", "")))) and bool(p.get("tag")) and bool(p.get("file"))
        prov_ok &= good
        prov_detail.append(f"{key}:commit={p.get('commit')},tag={p.get('tag')}")
    check("V6 三个面板都标注 commit + tag + 文件", prov_ok, "; ".join(prov_detail))

    # ---------- 面板边界 ----------
    panel = re.search(r'<div class="panel" id="analysisPanel">(.*?)<script id="analysisSnapshot"', html, re.S)
    body = panel.group(1) if panel else ""
    inputs = re.findall(r"<input[^>]*>", body)
    only_checks = all('type="checkbox"' in i for i in inputs)
    check("面板无自由输入控件（w 只能在记录点之间切换）", only_checks and len(inputs) == 3,
          f"面板内 input 数={len(inputs)}，全部为 checkbox={only_checks}")

    check("V3 规模守卫文案存在（>12 节点禁用页内试算）", "12 节点" in html and "禁用" in html,
          "guard 文案检查")

    check("V4 自检为字段级（字段/期望/实际 三列）",
          all(k in html for k in ("字段", "期望", "实际")) and "selfCheckRows" in html,
          "自检表列检查")

    low = html.lower()
    hits = [w for w in BANNED if w in body or w in low]
    check("V5 无越权措辞（意识/理解/自由能/灵魂/主观体验）", not hits, f"命中={hits or '无'}")

    check("滞回示意已标注来源限制", "原始 I 扫描轨迹未入库" in html, "E25 快照局限标注")

    return report()


def report() -> int:
    failed = 0
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} - {name}" + (f"（{detail}）" if detail else ""))
        failed += 0 if ok else 1
    print(f"\n汇总: {len(results) - failed}/{len(results)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
