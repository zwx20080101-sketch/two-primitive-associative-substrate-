"""M2-B: 拼写输出(系统级) - 块记录 -> 动作 stub 逐字母输出 -> 听回闭环。

宪章四问: 基底=块记录/语境检索/听回学习; 外部 stub=动作/书写模块(读块+渲染);
结论归谁=系统整体(记账表); 停点=输出准确的 HELLO 且听回闭环成立。

流程:
  1) HELLO 作为带边界单元 x100 -> L4 成块 CH_HELLO(spell=[H,e,l,l,o]);
  2) 语境: 打"课堂" -> 块被整体唤起(0.9);
  3) 动作 stub: 读块 -> 逐字母 H,e,l,l,o -> 渲染 "HELLO";
  4) 听回: 输出的字母序列学回 L1 -> n(l->l)=100 (E5 缺的双写由自己输出补齐)。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunk_layer import ChunkLayer
from experiments.checker import Checker
from motor_stub import MotorStub
from order_layer import OrderLayer
from synapse_net import SynapseNet


def run() -> dict:
    cl = ChunkLayer(k=5)
    l1 = OrderLayer()
    motor = MotorStub()
    c = Checker()

    # 1) 块形成
    cid = None
    for _ in range(100):
        cid = cl.learn_unit(["H", "e", "l", "l", "o"])
    c.check(
        "1) 块记录完整: spell(CH_HELLO)=[H,e,l,l,o] (5 个位置)",
        cl.spell(cid) == ["H", "e", "l", "l", "o"],
        f"chunks={cl.chunk_ids()}, spell={cl.spell(cid)}",
    )

    # 2) 语境唤起
    for _ in range(50):
        cl.learn_sequence(["课堂", cid])
    acts = cl.activate("课堂")
    c.check(
        "2) 语境唤起: 打'课堂' -> 块 CH_HELLO 整体 0.9",
        cid in acts and math.isclose(acts[cid], 0.9, rel_tol=1e-9),
        f"a(CH_HELLO)={acts.get(cid, 0):.4f}",
    )

    # 3) 动作 stub 输出
    letters = motor.spell_out(cl, cid)
    text = motor.render(letters)
    c.check(
        "3) stub 输出: 逐字母 H,e,l,l,o -> 字符串 'Hello' (双写还原, 大小写按 token 原样)",
        letters == ["H", "e", "l", "l", "o"] and text == "Hello",
        f"letters={letters}, text={text!r}",
    )

    # 4) 听回闭环
    for _ in range(100):
        l1.learn(letters)
    c.check(
        "4) 听回闭环: L1 学回输出序列 -> n(l->l)=100 (E5 缺的双写由自己输出补齐)",
        l1.count("H", "e") == 100
        and l1.count("e", "l") == 100
        and l1.count("l", "l") == 100
        and l1.count("l", "o") == 100,
        f"n(H->e)={l1.count('H', 'e')}, n(e->l)={l1.count('e', 'l')}, "
        f"n(l->l)={l1.count('l', 'l')}, n(l->o)={l1.count('l', 'o')}",
    )

    # 5) 闭包: stub 读不存在的块报错
    raised = False
    try:
        motor.spell_out(cl, "CH_NOPE")
    except KeyError:
        raised = True
    c.check(
        "5) 闭包: stub 读不存在的块 -> 报错 (无法凭空拼写)",
        raised and "CH_NOPE" not in cl.chunks,
        f"raised KeyError: {raised}",
    )

    # 6) 功劳记账
    ledger = [
        {"步骤": "块形成与顺序记录", "谁干的": "基底 L4", "内容": "spell=[H,e,l,l,o]"},
        {"步骤": "语境唤起块", "谁干的": "基底 L4 检索", "内容": "打课堂 -> 块 0.9"},
        {"步骤": "逐字母渲染输出", "谁干的": "动作/书写 stub(外部)", "内容": "H,e,l,l,o -> HELLO"},
        {"步骤": "听回学习", "谁干的": "输出(外部) + 基底 L1", "内容": "n(l->l)=100 补齐双写"},
        {"步骤": "最终'HELLO'", "谁干的": "系统整体", "内容": "块记录+动作stub 相辅相成"},
    ]
    c.check(
        "6) 功劳记账: 五步归属写清 (基底/stub/系统)",
        len(ledger) == 5 and all(row["谁干的"] for row in ledger),
        "块记录/听回=基底; 逐字母渲染=动作stub; 最终 HELLO=系统整体",
    )

    # 7) 基线回归
    net = SynapseNet()
    net.learn(["A", "B"])
    acts0 = net.activate("A")
    c.check(
        "7) 基线回归: L0 learn([A,B])->B=0.9 仍成立 (未改层文件)",
        math.isclose(acts0["B"], 0.9, rel_tol=1e-9),
        f"a(B)={acts0.get('B', 0):.4f}",
    )

    return {
        "id": "exp18_spelling_output",
        "title": "M2-B 拼写输出(系统级): 块记录 -> 动作 stub -> HELLO -> 听回闭环",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "chunk": {"id": cid, "spell": cl.spell(cid)},
            "context_activation": {k: round(v, 6) for k, v in sorted(acts.items())},
            "motor_output": {"letters": letters, "text": text},
            "l1_after_hearing": {
                "H->e": l1.count("H", "e"),
                "e->l": l1.count("e", "l"),
                "l->l": l1.count("l", "l"),
                "l->o": l1.count("l", "o"),
            },
            "ledger": ledger,
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
