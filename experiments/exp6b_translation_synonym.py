"""E6b: 翻译/同义实验 - 表面不同, 靠共现与情景连成"同一个东西"。

场景(全部只用冻结的 L0, 不加新层):
  S1 符号-读音互译 (用户方案):
     反复学 [A, 唉] x100、[B, 必] x100
     -> 打 A 出 唉, 打 唉 回 A; A 不会串到 B/必;
  S2 同义经情景:
     组A: [S,P,Q,R]x100 + [S,X,Y,Z]x100 (共享情景 S)
     -> 打 P 能经 S 取到 X(另一个表面), 反之亦然;
     组B(对照): 两串经验没有 S
     -> 打 P 时 X 完全不出现, 两簇分开;
  S3 因果联想链 (用户方案):
     [加速,摔倒]x100 + [摔倒,疼]x100
     -> 打 加速: 摔倒 0.9 先出, 疼 0.81 顺出;
        threshold=0.85 只见摔倒, 降到 0.7 才见疼。

原则: 底层不存"翻译"或"因果"概念, 只存共现; 反馈(疼)也是普通节点,
本身只是链条的一部分。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention import attention_filter
from experiments.checker import Checker
from synapse_net import SynapseNet


def run() -> dict:
    c = Checker()
    data: dict = {}

    # ---------------- S1: 符号-读音互译 ----------------
    net1 = SynapseNet()
    for _ in range(100):
        net1.learn(["A", "唉"])
        net1.learn(["B", "必"])
    s1_a = net1.activate("A")
    c.check(
        "S1 打 A: 出读音 唉 (0.9)",
        "唉" in s1_a and math.isclose(s1_a["唉"], 0.9, rel_tol=1e-9),
        f"a(唉)={s1_a.get('唉', 0):.6f}",
    )
    c.check(
        "S1 A 不串线: B/必 不出现 (只和共现过的相连)",
        "B" not in s1_a and "必" not in s1_a,
        f"keys={sorted(s1_a)}",
    )
    s1_back = net1.activate("唉")
    c.check(
        "S1 打 唉: 回字母 A (双向互译)",
        "A" in s1_back and "B" not in s1_back and "必" not in s1_back,
        f"keys={sorted(s1_back)}",
    )
    data["s1"] = {
        "A_activation": {k: round(v, 6) for k, v in sorted(s1_a.items())},
        "ai_activation": {k: round(v, 6) for k, v in sorted(s1_back.items())},
    }

    # ---------------- S2: 同义经情景 (组A 有 S / 组B 无 S) ----------------
    netS = SynapseNet()
    for _ in range(100):
        netS.learn(["S", "P", "Q", "R"])
        netS.learn(["S", "X", "Y", "Z"])
    s2_p = netS.activate("P")
    s2_x = netS.activate("X")
    c.check(
        "S2 组A: 打 P -> 经 S 取到另一表面 X (0.9^2)",
        "X" in s2_p and math.isclose(s2_p["X"], 0.81, rel_tol=1e-9),
        f"a(X)={s2_p.get('X', 0):.6f}",
    )
    c.check(
        "S2 组A: 打 X -> 回 P (互译)",
        "P" in s2_x and math.isclose(s2_x["P"], 0.81, rel_tol=1e-9),
        f"a(P)={s2_x.get('P', 0):.6f}",
    )

    netN = SynapseNet()
    for _ in range(100):
        netN.learn(["P", "Q", "R"])
        netN.learn(["X", "Y", "Z"])
    s2n_p = netN.activate("P")
    c.check(
        "S2 组B(对照): 没有 S, 打 P 时 X 完全不出现 (两簇分开)",
        "X" not in s2n_p,
        f"keys={sorted(s2n_p)}",
    )
    data["s2"] = {
        "with_S_P_activation": {k: round(v, 6) for k, v in sorted(s2_p.items())},
        "with_S_X_activation": {k: round(v, 6) for k, v in sorted(s2_x.items())},
        "without_S_P_activation": {k: round(v, 6) for k, v in sorted(s2n_p.items())},
    }

    # ---------------- S3: 因果联想链 (加速 -> 摔倒 -> 疼) ----------------
    net3 = SynapseNet()
    for _ in range(100):
        net3.learn(["加速", "摔倒"])
        net3.learn(["摔倒", "疼"])
    s3 = net3.activate("加速")
    c.check(
        "S3 打 加速: 摔倒 0.9 直接出",
        math.isclose(s3["摔倒"], 0.9, rel_tol=1e-9),
        f"a(摔倒)={s3['摔倒']:.6f}",
    )
    c.check(
        "S3 再顺出 疼 0.81 (摔倒与疼共现过, 不是加速里存了疼)",
        math.isclose(s3["疼"], 0.81, rel_tol=1e-9),
        f"a(疼)={s3['疼']:.6f}",
    )
    hi = attention_filter(s3, threshold=0.85)
    lo = attention_filter(s3, threshold=0.7)
    c.check(
        "S3 阈值 0.85 只见摔倒; 0.7 才连疼也放出",
        set(hi) == {"加速", "摔倒"} and "疼" in lo,
        f"0.85 -> {sorted(hi)}; 0.7 -> {sorted(lo)}",
    )
    data["s3"] = {"activation": {k: round(v, 6) for k, v in sorted(s3.items())}}

    return {
        "id": "exp6b_translation_synonym",
        "title": "E6b 翻译/同义: 互译、经情景聚类、因果联想链",
        "passed": c.passed,
        "checks": c.checks,
        "data": data,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
