"""E12: 组块与嵌套 (L4 组块层)。

子场景:
  S1 整词与双写: HELLO 作为带边界单元重复 100 次 -> CH_HELLO 成块;
     块记录保留双写: spell(CH) = [H,e,l,l,o]; 平面 L1 链只能 H-e-l-o;
  S2 块的复用: 块 红苹果 出现在"我喜欢红苹果"和"他买了红苹果"两句中,
     同一个块节点被两句共享, 可整体唤起并可展开;
  S3 嵌套句: 块 咬猫 成块(C1), 主句在块级学成 [C1, 的, 狗, 追, 老鼠]
     (咬了猫的狗追了老鼠): 狗-追 主谓在块级相邻; 从句块可独立展开;
     对照平面链: 没有独立块节点, "咬猫"无法作为整体被复用。

判据全部预先注册; 不改 L0/L1; 块只由见过的成员组成(闭包)。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunk_layer import ChunkLayer
from experiments.checker import Checker
from synapse_net import SynapseNet


def _walk(net, start: str) -> list[str]:
    seq = [start]
    seen = {start}
    current = start
    while True:
        acts = net.activate(current)
        candidates = {k: v for k, v in acts.items() if k not in seen and v > 1e-9}
        if not candidates:
            break
        nxt = max(sorted(candidates), key=lambda k: candidates[k])
        seq.append(nxt)
        seen.add(nxt)
        current = nxt
    return seq


def _degree_net(net, node: str) -> int:
    return sum(1 for a, b in net.connections if node in (a, b))


def run() -> dict:
    c = Checker()
    data: dict = {}

    # ================= S1: HELLO 整词与双写 =================
    cl1 = ChunkLayer(k=5)
    for _ in range(100):
        cl1.learn_unit(["H", "e", "l", "l", "o"])
    c1_ids = cl1.chunk_ids()
    c.check(
        "S1 成块: HELLO 出现 100 次 -> 恰好 1 个块 (K=5)",
        len(c1_ids) == 1,
        f"chunks={c1_ids}",
    )
    hello_spell = cl1.spell(c1_ids[0]) if c1_ids else []
    c.check(
        "S1 双写: spell(CH_HELLO) = [H,e,l,l,o] (5 个位置, 双写还原)",
        hello_spell == ["H", "e", "l", "l", "o"],
        f"spell={hello_spell}",
    )
    binds = {m: cl1.member_bind.get((c1_ids[0], m), 0) for m in "Helo"}
    c.check(
        "S1 成员绑定: 去重成员 H/e/l/o 各绑定 96 次 (第 5 次起)",
        binds == {"H": 96, "e": 96, "l": 96, "o": 96},
        str(binds),
    )

    # 平面 L1 链对照 (E5 的构造)
    flat = SynapseNet()
    for _ in range(100):
        flat.learn(["H", "e"])
        flat.learn(["e", "l"])
        flat.learn(["l", "o"])
    flat_walk = _walk(flat, "H")
    c.check(
        "S1 对照: 平面 L1 链只能走 H-e-l-o, 还原不了双写 (局限记录)",
        flat_walk == ["H", "e", "l", "o"] and len(flat_walk) == 4,
        f"flat walk={flat_walk}",
    )

    # 块可当原子参与高层事件
    for _ in range(50):
        cl1.learn_sequence(["课堂", c1_ids[0]])
    acts_class = cl1.activate("课堂")
    c.check(
        "S1 原子性: 打'课堂' -> 块 CH_HELLO 作为整体被唤起 (0.9)",
        c1_ids[0] in acts_class and math.isclose(acts_class[c1_ids[0]], 0.9, rel_tol=1e-9),
        f"a(CH_HELLO)={acts_class.get(c1_ids[0], 0):.4f}",
    )
    data["s1"] = {
        "chunk": c1_ids[0],
        "spell": hello_spell,
        "member_binds": binds,
        "flat_walk": flat_walk,
    }

    # ================= S2: 块的复用 =================
    cl2 = ChunkLayer(k=5)
    c2_id = None
    for _ in range(100):
        c2_id = cl2.learn_unit(["红", "苹果"])
    for _ in range(100):
        cl2.learn_sequence(["我", "喜欢", c2_id])
        cl2.learn_sequence(["他", "买了", c2_id])
    c.check(
        "S2 成块: 红苹果 成块, 且两句共用同一个块节点",
        len(cl2.chunk_ids()) == 1
        and cl2.count_pair("喜欢", c2_id) == 100
        and cl2.count_pair("买了", c2_id) == 100,
        f"chunk={c2_id}, N(喜欢,{c2_id})={cl2.count_pair('喜欢', c2_id)}, "
        f"N(买了,{c2_id})={cl2.count_pair('买了', c2_id)}",
    )
    c.check(
        "S2 块度: 块节点在两句中 degree=2 (喜欢/买了)",
        cl2.degree(c2_id) == 2,
        f"degree={cl2.degree(c2_id)}",
    )
    acts_like = cl2.activate("喜欢")
    acts_buy = cl2.activate("买了")
    c.check(
        "S2 取回: 从'喜欢'或'买了'都能整体唤起同一块 (0.9)",
        c2_id in acts_like
        and c2_id in acts_buy
        and math.isclose(acts_like[c2_id], 0.9, rel_tol=1e-9)
        and math.isclose(acts_buy[c2_id], 0.9, rel_tol=1e-9),
        f"a(块|喜欢)={acts_like[c2_id]:.4f}, a(块|买了)={acts_buy[c2_id]:.4f}",
    )
    c.check(
        "S2 展开: spell(块) = [红,苹果]",
        cl2.spell(c2_id) == ["红", "苹果"],
        f"spell={cl2.spell(c2_id)}",
    )
    data["s2"] = {"chunk": c2_id, "spell": cl2.spell(c2_id)}

    # ================= S3: 嵌套句 =================
    cl3 = ChunkLayer(k=5)
    c1_id = None
    for _ in range(100):
        c1_id = cl3.learn_unit(["咬", "猫"])
    for _ in range(100):
        cl3.learn_sequence([c1_id, "的", "狗", "追", "老鼠"])
    c.check(
        "S3 成块: 咬猫 成块 C1, spell=[咬,猫]",
        cl3.spell(c1_id) == ["咬", "猫"],
        f"C1={c1_id}, spell={cl3.spell(c1_id)}",
    )
    c.check(
        "S3 主谓: 块级序列 [C1,的,狗,追,老鼠] 中 狗-追 相邻 (N=100)",
        cl3.count_pair("狗", "追") == 100 and cl3.count_pair(c1_id, "的") == 100,
        f"N(狗,追)={cl3.count_pair('狗', '追')}, N(C1,的)={cl3.count_pair(c1_id, '的')}",
    )
    acts_dog = cl3.activate("狗")
    c.check(
        "S3 跨块依赖: 打狗 -> 追=0.9 (主谓在块级没被从句打断)",
        math.isclose(acts_dog["追"], 0.9, rel_tol=1e-9),
        f"a(追|狗)={acts_dog['追']:.4f}",
    )
    walk_nested = _walk(cl3, c1_id)
    c.check(
        "S3 整句取回: 从句首 C1 走读整句 C1-的-狗-追-老鼠",
        walk_nested == [c1_id, "的", "狗", "追", "老鼠"],
        "-".join(walk_nested),
    )

    # 平面链对照
    flat2 = SynapseNet()
    for _ in range(100):
        for a, b in zip(["咬", "猫", "的", "狗", "追", "老鼠"], ["咬", "猫", "的", "狗", "追", "老鼠"][1:]):
            flat2.learn([a, b])
    c.check(
        "S3 对照: 平面链没有独立块节点; '咬'只是 1 度字母, 展开不出独立'咬猫'",
        "CH" not in "".join(flat2.list_nodes())
        and _degree_net(flat2, "咬") == 1,
        f"咬 degree={_degree_net(flat2, '咬')}, nodes={flat2.list_nodes()}",
    )
    data["s3"] = {
        "chunk": c1_id,
        "spell": cl3.spell(c1_id),
        "walk": walk_nested,
    }

    return {
        "id": "exp12_chunk_nesting",
        "title": "E12 组块与嵌套: L4 组块层 (整词/双写/块复用/嵌套句)",
        "passed": c.passed,
        "checks": c.checks,
        "data": data,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
