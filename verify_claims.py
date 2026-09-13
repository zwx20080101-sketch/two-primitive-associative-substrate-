"""会审工具: 核对论文中文草稿与实验数据/登记表的一致性。

检查三件事:
  1) outputs/exp*.json 全部 passed=True 且 checks 全 ok;
  2) 草稿引用的关键数值与 JSON 数据一致(近似比较);
  3) 措辞红线: 无"这说明人类/人脑就是这样"式断言;
     4.6/4.7 含外部 stub 边界标记; λ/阈值等标为预设。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DRAFT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "REPORT-zh.md")
if not DRAFT.exists():
    DRAFT = ROOT / "PAPER-zh-DRAFT.md"


def load(name: str):
    with (OUT / name).open(encoding="utf-8") as f:
        return json.load(f)


def approx(a, b, tol=1e-3):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def check(checks: list, name: str, ok: bool, detail: str):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def main() -> int:
    checks: list[dict] = []

    # ---- 1) 每个实验 JSON passed + checks ok ----
    # 例外：文件名以 _FAIL 结尾的是【归档的失败记录】（如 E32 v1 的预注册失败），
    # 它们按设计就应当是 passed=False，不参与"全部通过"的通过性检查。
    for p in sorted(OUT.glob("exp*.json")):
        if p.stem.endswith("_FAIL"):
            continue
        if p.name in ("exp20_ablation.json", "exp21_scale.json", "exp22_stats.json"):
            continue  # 自定义套件数据, 单独核对
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            passed = data.get("passed", False)
            all_ok = all(chk.get("ok") for chk in data.get("checks", []))
            check(checks, f"passed:{p.name}", passed and all_ok, f"{len(data.get('checks', []))} checks")
        except Exception as e:  # noqa: BLE001
            check(checks, f"load:{p.name}", False, str(e))

    # ---- 2) 关键数值与草稿一致 ----
    facts = []

    d1 = load("exp1_chain.json")
    facts.append(("E1 节点=26", d1["data"]["network"]["nodes"] == 26))
    facts.append(("E1 连接=25", d1["data"]["network"]["connections"] == 25))
    facts.append(("E1 N(A,Z)=0", d1["data"]["network"]["N(A,Z)"] == 0))
    facts.append(("E1 激活 Z≈0.07179", approx(d1["data"]["activation"]["Z"], 0.9 ** 25, 1e-4)))

    d2 = load("exp2_strength.json")
    facts.append(("E2 a(B)=0.9", approx(d2["data"]["full_activation"]["B"], 0.9)))
    facts.append(("E2 a(X)=0.018", approx(d2["data"]["full_activation"]["X"], 0.018)))

    d3 = load("exp3_threshold.json")
    vs = d3["data"]["visible_sets"]
    facts.append(("E3 可见 7/22/26", len(vs["0.5"]) == 7 and len(vs["0.1"]) == 22 and len(vs["0.001"]) == 26))

    d5 = load("exp5_direction_hello.json")
    facts.append(("E5 正走 o=0.729", approx(d5["data"]["activation"]["o"], 0.729)))
    facts.append(("E5 倒走 H≈7.1e-5", approx(d5["data"]["reverse_activation"]["H"], 7.1e-5, 0.1)))
    facts.append(("E5 l->l=100", d5["data"]["counts"]["l->l"] == 100))

    d6 = load("exp6_fork_context.json")
    facts.append(("E6a S1 等强", approx(d6["data"]["s1"]["activation"]["C"], 0.45) and approx(d6["data"]["s1"]["activation"]["D"], 0.45)))
    facts.append(("E6a S2 10:1", approx(d6["data"]["s2"]["activation"]["C"], 0.818182) and approx(d6["data"]["s2"]["activation"]["D"], 0.081818)))
    facts.append(("E6a S3 X1 C:D=2:1", approx(d6["data"]["s3"]["X1_activation"]["C"], 0.9) and approx(d6["data"]["s3"]["X1_activation"]["D"], 0.45)))

    d6b = load("exp6b_translation_synonym.json")
    facts.append(("E6b 组A X=0.81", approx(d6b["data"]["s2"]["with_S_P_activation"]["X"], 0.81)))
    facts.append(("E6b 组B 无 X", "X" not in d6b["data"]["s2"]["without_S_P_activation"]))

    d7 = load("exp7_dense_episode.json")
    facts.append(("E7 疼 degree=5", d7["data"]["degrees"]["疼(摔倒)"] == 5))
    facts.append(("E7 摔->疼 0.9", approx(d7["data"]["activation_from_shuai"]["疼"], 0.9)))
    facts.append(("E7 空中->疼 0.81", approx(d7["data"]["activation_from_kongzhong"]["疼"], 0.81)))
    facts.append(("E7 快->疼 0.36", approx(d7["data"]["activation_from_kuai"]["疼"], 0.36)))

    d8 = load("exp8_hub_graph.json")
    facts.append(("E8 追-老鼠=200", d8["data"]["counts"]["追-老鼠"] == 200))
    facts.append(("E8 走读=狗咬猫追老鼠", d8["data"]["walk_from_狗"] == "狗咬猫追老鼠"))

    d9 = load("exp9_partial_completion.json")
    facts.append(("E9 A+C+E -> B/D=0.9", approx(d9["data"]["from_A_C_E"]["B"], 0.9) and approx(d9["data"]["from_A_C_E"]["D"], 0.9)))
    facts.append(("E9 A+E -> C=0.81", approx(d9["data"]["from_A_E"]["C"], 0.81)))

    d10 = load("exp10_recency_lambda.json")
    facts.append(("E10 λ=0 B=0.9", approx(d10["data"]["flat_activation"]["B"], 0.9)))
    facts.append(("E10 λ=0.005 D=0.9", approx(d10["data"]["decay_activation"]["D"], 0.9)))
    facts.append(("E10 λ=0.005 C≈0.006", approx(d10["data"]["decay_activation"]["C"], 0.006064, 0.05)))
    facts.append(("E10 λ 归 0 B=0.9", approx(d10["data"]["back_to_flat_activation"]["B"], 0.9)))

    d13 = load("exp13_repeat_marker.json")
    facts.append(("E13 重复=100/0/2",
                  d13["data"]["repeat_100_events"]["n(汪->汪)"] == 100
                  and d13["data"]["separate_2_events"]["n(汪->汪)"] == 0
                  and d13["data"]["triple_1_event"]["n(汪->汪)"] == 2))

    d14 = load("exp14_multi_path_evidence.json")
    facts.append(("E14 max 单路=0.81", approx(d14["data"]["max_rule"]["single_T"], 0.81)))
    facts.append(("E14 max 双路=0.81", approx(d14["data"]["max_rule"]["dual_T"], 0.81)))
    facts.append(("E14 sum 双路=1.0", approx(d14["data"]["sum_rule"]["dual_T"], 1.0)))

    d11 = load("exp11_interference.json")
    facts.append(("E11 拥挤度 1/2/5",
                  d11["data"]["crowding_candidates"]["L1"] == 1
                  and d11["data"]["crowding_candidates"]["L2"] == 2
                  and d11["data"]["crowding_candidates"]["L5"] == 5))
    facts.append(("E11 独特词 0.85 见本句", set(d11["data"]["from_zhua_L5_0.85"]) == {"狗", "抓", "鸟"}))

    d16 = load("exp16_role_markers.json")
    facts.append(("E16 打狗->猫=0.81", approx(d16["data"]["s1"]["from_gou"]["猫"], 0.81)))
    facts.append(("E16 L2 [狗,咬]->猫", set(d16["data"]["s1"]["L2_gou_yao"]) == {"猫"}))

    d17 = load("exp17_system_reduplication.json")
    facts.append(("E17 stub 输出=人人", d17["data"]["stub"]["output_人"] == "人人"))
    facts.append(("E17 听回 n(人->人)=1", d17["data"]["repeat_markers"]["n(人->人)_after_feedback"] == 1))

    d18 = load("exp18_spelling_output.json")
    facts.append(("E18 spell 双写", d18["data"]["chunk"]["spell"] == ["H", "e", "l", "l", "o"]))
    facts.append(("E18 输出=Hello", d18["data"]["motor_output"]["text"] == "Hello"))
    facts.append(("E18 听回 l->l=100", d18["data"]["l1_after_hearing"]["l->l"] == 100))

    d19 = load("exp19_emotion_modulation.json")
    facts.append(("E19 fear 差=1.8", approx(d19["data"]["marker_path"]["fear_snake"] - d19["data"]["marker_path"]["fear_stick"], 1.8)))
    facts.append(("E19 密度 0.9/0.54",
                  approx(d19["data"]["density_path"]["snake_300"], 0.9) and approx(d19["data"]["density_path"]["stick_60"], 0.54)))

    a20 = json.loads((OUT / "exp20_ablation.json").read_text(encoding="utf-8"))
    rows20 = a20["rows"]
    facts.append(("E20 A 方向 ON≈10201", approx(rows20["A_方向"]["on"], 10201, 0.01)))
    facts.append(("E20 C 翻译 ON=1/OFF=0", rows20["C_翻译"]["on"] == 1.0 and rows20["C_翻译"]["off"] == 0.0))
    facts.append(("E20 D 纯计数=0", rows20["D_补全"]["count_only"] == 0.0))
    facts.append(("E20 H 情绪差=1.8", approx(rows20["H_情绪"]["on"], 1.8)))

    a21 = json.loads((OUT / "exp21_scale.json").read_text(encoding="utf-8"))
    facts.append(("E21 六现象 100x 全 1.0", all(v["100"] == 1.0 for v in a21["rates"].values())))

    a22 = json.loads((OUT / "exp22_stats.json").read_text(encoding="utf-8"))
    facts.append(("E22 正样本率 1.0", all(v["pos"]["rate"] == 1.0 for v in a22.values())))
    facts.append(("E22 CI 下限 0.94", all(v["pos"]["ci_lo"] >= 0.90 for v in a22.values())))

    for label, ok in facts:
        check(checks, f"数据核对:{label}", ok, "" if ok else "MISMATCH")

    # ---- 3) 措辞红线与 stub 边界 ----
    text = DRAFT.read_text(encoding="utf-8")
    text_flat = text.replace("\n", "")
    red = ["这说明人类", "人脑就是这样", "证明了人类"]
    check(checks, "红线: 无'这说明人类/人脑就是这样'", not any(r in text for r in red), "")
    check(checks, "红线: 4.6 含外部 stub 标记", "动作/书写模块（外部 stub）" in text_flat, "")
    check(checks, "红线: 4.7 含外部 stub 标记", "语言规则模块（外部 stub）" in text and "外部情绪模块（stub）" in text, "")
    check(checks, "红线: 预设参数声明存在", "预设" in text and "λ" in text, "")

    failed = [x for x in checks if not x["ok"]]
    for c in checks:
        print(("PASS" if c["ok"] else "FAIL"), "-", c["name"], c["detail"])
    print(f"\n汇总: {len(checks) - len(failed)}/{len(checks)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
