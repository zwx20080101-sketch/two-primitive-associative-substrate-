"""E13: 基底重复标记 - 同一节点短窗内二次激活这个事实能否被表示。

只验基底事实, 不附带"强调/普遍"语义(语义归语言模块, 见 BOUNDARY.md)。
机制: L1 允许有序自转移 n(汪->汪): 含义严格为"汪 之后紧跟着又响了一次汪"。

判据(预先注册):
  1) [汪,汪] x100 -> n(汪->汪)=100;
  2) 分开两次 [汪](t=1, t=100) -> n(汪->汪)=0 (分散出现不算重复);
  3) [汪,汪,汪] x1 -> n(汪->汪)=2 (按相邻转移计数);
  4) 打汪 -> 仍只有 {汪:1.0}, 不自动冒出"强调"节点(不附语义);
  5) 未学过的"人"重复 -> n(人->人)=0, 互不串扰;
  6) 标记可被外部读取(数据输出);
  7) 回归: 全量 E0-E12 仍 PASS(自环不增强激活, 理论上无副作用)。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.checker import Checker
from order_layer import OrderLayer


def run() -> dict:
    c = Checker()

    # 1) 同窗连续两次 x100
    net1 = OrderLayer()
    for _ in range(100):
        net1.learn(["汪", "汪"])
    c.check(
        "1) [汪,汪]x100 -> n(汪->汪)=100 (重复事实被记下)",
        net1.count("汪", "汪") == 100,
        f"n(汪->汪)={net1.count('汪', '汪')}",
    )

    # 2) 分散两次(不同事件) -> 0
    net2 = OrderLayer()
    net2.learn(["汪"], t=1.0)
    net2.learn(["汪"], t=100.0)
    c.check(
        "2) 分开两次 [汪] -> n(汪->汪)=0 (分散出现不算重复标记)",
        net2.count("汪", "汪") == 0,
        f"n(汪->汪)={net2.count('汪', '汪')}",
    )

    # 3) 连续三次 -> 2
    net3 = OrderLayer()
    net3.learn(["汪", "汪", "汪"])
    c.check(
        "3) [汪,汪,汪] -> n(汪->汪)=2 (按相邻转移计数, 不塌缩成一次)",
        net3.count("汪", "汪") == 2,
        f"n(汪->汪)={net3.count('汪', '汪')}",
    )

    # 4) 不附语义: 打汪仍只有 {汪:1.0}, 无"强调"节点
    acts = net1.activate("汪")
    c.check(
        "4) 打汪 -> 仍只有 {汪:1.0}, 不自动冒出'强调'节点 (标记不带语义)",
        set(acts) == {"汪"} and math.isclose(acts["汪"], 1.0, rel_tol=1e-9),
        f"acts={acts}, nodes={sorted(net1.nodes)}",
    )

    # 5) 不串扰: 汪的重复不影响人
    net5 = OrderLayer()
    for _ in range(100):
        net5.learn(["汪", "汪"])
    for _ in range(100):
        net5.learn(["人"])
    c.check(
        "5) 未学过的'人'重复 -> n(人->人)=0, 与汪互不串扰",
        net5.count("人", "人") == 0 and "人" in net5.nodes,
        f"n(人->人)={net5.count('人', '人')}, 人 in nodes: {'人' in net5.nodes}",
    )

    # 6) 标记可被外部读取
    c.check(
        "6) 标记可查询: count(汪,汪)=100 是给外部模块读的证据 (数据)",
        net1.count("汪", "汪") == 100,
        "外部模块(如语言层)将来读取该计数判断'叠词', 那是 M2 的事",
    )

    return {
        "id": "exp13_repeat_marker",
        "title": "E13 基底重复标记: 同一节点短窗内二次激活的表示",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "repeat_100_events": {"n(汪->汪)": net1.count("汪", "汪")},
            "separate_2_events": {"n(汪->汪)": net2.count("汪", "汪")},
            "triple_1_event": {"n(汪->汪)": net3.count("汪", "汪")},
            "activation_after_repeat": {k: round(v, 6) for k, v in sorted(acts.items())},
            "no_crosstalk": {"n(人->人)": net5.count("人", "人")},
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
