"""Synapse-Net L0 v2：带值种子入口（**平行分支**，不修改 v1）。

与 v1 的唯一差异：`activate` 的种子入口支持 `dict[str, float]`，且**种子值常量**
（传播不抬高、不降低）。名字列表形式向后兼容（仍全部 1.0）。

实现形态：**子类覆盖**——只重写 `activate`，存储层（learn / connections / nodes）
与其他读取助手全部继承 v1。这样 v2 的 diff 就是"种子入口"这一段，
不会产生第二份 learn/strength 实现（B17 那类漂移风险的来源）。

规格见 L0-SPEC.md §3；影响评估见 DESIGN-E37-IMPACT.md；判据见 DESIGN-E37.md。
"""

from __future__ import annotations

import math
from collections import deque

from synapse_net import SynapseNet as _SynapseNetV1

EPS = 1e-12


class SynapseNet(_SynapseNetV1):
    """v1 + 带值种子（常量语义）。其余行为与 v1 完全一致。"""

    def activate(self, start, hop_decay: float = 0.9) -> dict:
        """三种输入：

        - `str`            → 单种子，值 1.0
        - `list[str]`      → 多种子，全部 1.0（**与 v1 完全一致，向后兼容**）
        - `dict[str,float]`→ 带值种子，值被保留且**常量**（传播不改变它）
        """
        if isinstance(start, dict):
            seed_values = {str(k): float(v) for k, v in start.items()}
        elif isinstance(start, str):
            seed_values = {start: 1.0}
        else:
            seed_values = {s: 1.0 for s in start}
        if not seed_values:
            return {}
        for s in seed_values:
            if s not in self.nodes:
                raise KeyError(f"node not in network: {s!r} (learn() it first)")

        now = self._clock
        acts: dict = dict(seed_values)
        frozen = set(seed_values)          # 常量语义：种子永不被传播改写
        queue = deque(seed_values)
        queued = set(seed_values)

        while queue:
            u = queue.popleft()
            queued.discard(u)
            au = acts[u]
            for v in self._adj.get(u, ()):
                if v in frozen:
                    continue
                arrival = au * self.strength(u, v, now) * hop_decay
                if arrival > acts.get(v, -math.inf) + EPS:
                    acts[v] = arrival
                    if v not in queued:
                        queued.add(v)
                        queue.append(v)
        return acts
