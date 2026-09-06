"""M2-A: 叠词外推(系统级) - 基底 + 语言规则 stub 的合成。

宪章四问: 基底=重复标记/规则标记绑定/特征检索;
外部 stub=语言规则模块(判定+渲染); 结论归谁=系统整体(功劳记账);
停点=产出"人人"且闭包不破, 语义真伪不评判。

训练输入:
  教过的叠词: [天,天]x100, [夜,夜]x100        -> L1: n(天->天)=100, n(夜->夜)=100
  规则反馈:   [天,R]x100, [夜,R]x100          -> L0: R 绑定 天/夜 (R=语言模块打的"这是叠词")
  资格特征:   [天,F],[夜,F],[人,F] x100       -> L0: F 绑定 天/夜/人 (名词类)
  反例:       [的]x100                        -> 的 无 F 绑定

流程: 教例子 -> stub 读证据外推"人人"、拒绝"的的" -> 输出被听回 ->
      L1 记录 n(人->人)=1 (新记忆由外部输出变新经验, 部件全旧)。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.checker import Checker
from language_stub import ReduplicationStub
from order_layer import OrderLayer
from synapse_net import SynapseNet


def run() -> dict:
    l0 = SynapseNet()
    l1 = OrderLayer()
    stub = ReduplicationStub(rule_marker="R", feature="F")
    c = Checker()

    # ---------- 训练 ----------
    for _ in range(100):
        l1.learn(["天", "天"])
        l1.learn(["夜", "夜"])
        l0.learn(["天", "R"])
        l0.learn(["夜", "R"])
        l0.learn(["天", "F"])
        l0.learn(["夜", "F"])
        l0.learn(["人", "F"])
        l0.learn(["的"])

    # 1) 教过的叠词有重复标记
    c.check(
        "1) 教过的叠词: n(天->天)=n(夜->夜)=100 (基底重复标记)",
        l1.count("天", "天") == 100 and l1.count("夜", "夜") == 100,
        f"n(天->天)={l1.count('天', '天')}, n(夜->夜)={l1.count('夜', '夜')}",
    )

    # 2) 规则标记绑定
    r_ok = (
        l0.connection_count("天", "R") == 100
        and l0.connection_count("夜", "R") == 100
        and l0.connection_count("的", "R") == 0
    )
    c.check(
        "2) 规则标记 R 绑定 天/夜, 不绑定'的'",
        r_ok,
        f"N(天,R)={l0.connection_count('天', 'R')}, "
        f"N(夜,R)={l0.connection_count('夜', 'R')}, "
        f"N(的,R)={l0.connection_count('的', 'R')}",
    )

    # 3) 资格特征检索
    acts_f = l0.activate("F")
    c.check(
        "3) 打 F -> 天/夜/人 都在 (基底提供'像谁'的线索)",
        all(math.isclose(acts_f.get(w, 0.0), 0.9, rel_tol=1e-9) for w in ("天", "夜", "人")),
        f"天={acts_f.get('天', 0):.2f}, 夜={acts_f.get('夜', 0):.2f}, "
        f"人={acts_f.get('人', 0):.2f}",
    )

    # 4) 闭包(外推前): 没人教过人人
    c.check(
        "4) 外推前: n(人->人)=0 (没人教过, 基底绝无标记)",
        l1.count("人", "人") == 0,
        f"n(人->人)={l1.count('人', '人')}",
    )

    # 5) stub 外推
    can_ren = stub.can_reduplicate(l0, l1, "人")
    can_de = stub.can_reduplicate(l0, l1, "的")
    out_ren = stub.render("人") if can_ren else ""
    out_de = stub.render("的") if can_de else ""
    c.check(
        "5) stub 外推: 人 -> 人人; 的 -> 拒绝 (特征类约束)",
        can_ren and not can_de and out_ren == "人人" and out_de == "",
        f"人 判定={can_ren}, 输出={out_ren!r}; 的 判定={can_de}, 输出={out_de!r}",
    )

    # 6) 回写记忆: 输出被听回 -> n(人->人)=1
    l1.learn(["人", "人"])
    c.check(
        "6) 回写记忆: 说出'人人'被听回后 n(人->人)=1 (新记忆=外部输出->新经验)",
        l1.count("人", "人") == 1,
        f"n(人->人)={l1.count('人', '人')}",
    )

    # 7) 功劳记账
    ledger = [
        {"步骤": "重复标记 n(天->天)/n(夜->夜)", "谁干的": "基底 L1", "内容": "只记'同节点相邻再现'"},
        {"步骤": "规则标记 R 绑定", "谁干的": "语言模块注入标记 + 基底绑定", "内容": "R=这是叠词(输入侧)"},
        {"步骤": "特征类 F 检索(天/夜/人)", "谁干的": "基底 L0 绑定 + 激活", "内容": "提供'像谁'线索"},
        {"步骤": "外推决策(人->人人), 拒绝的", "谁干的": "语言规则 stub(外部)", "内容": "读证据做判定"},
        {"步骤": "渲染输出'人人'", "谁干的": "stub(外部)", "内容": "字符串输出"},
        {"步骤": "回写 n(人->人)=1", "谁干的": "输出被听回 + 基底 L1 记录", "内容": "新记忆, 部件全旧"},
        {"步骤": "最终'人人'", "谁干的": "系统整体", "内容": "基底+语言stub 相辅相成"},
    ]
    c.check(
        "7) 功劳记账: 七步归属写清 (基底/stub/系统)",
        len(ledger) == 7 and all(row["谁干的"] for row in ledger),
        "重复标记=基底; 规则判定+渲染=stub; 最终'人人'=系统整体",
    )

    # 8) 闭包: 无"人人"单节点
    no_new = "人人" not in l0.nodes and "人人" not in l1.nodes
    c.check(
        "8) 闭包: 基底没有发明'人人'节点 (输出只是字符串渲染)",
        no_new,
        f"L0 nodes={sorted(l0.nodes)}, L1 nodes={sorted(l1.nodes)}",
    )

    return {
        "id": "exp17_system_reduplication",
        "title": "M2-A 叠词外推(系统级): 基底 + 语言规则 stub -> 人人",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "repeat_markers": {
                "n(天->天)": l1.count("天", "天"),
                "n(夜->夜)": l1.count("夜", "夜"),
                "n(人->人)_before": 0,
                "n(人->人)_after_feedback": l1.count("人", "人"),
            },
            "stub": {
                "taught_words": sorted(stub.taught_words(l0)),
                "eligible_words": sorted(stub.eligible_words(l0)),
                "can_人": can_ren,
                "can_的": can_de,
                "output_人": out_ren,
                "output_的": out_de,
            },
            "ledger": ledger,
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
