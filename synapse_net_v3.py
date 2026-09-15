"""Synapse-Net L0 v3：到达值算式层的外部调制参数（**平行分支**，不修改 v1/v2）。

与 v2 的唯一差异：构造参数 `inhibition = g`，作用在到达值算式上

    arrival = au * strength(u, v) * hop_decay - g

**max 吸收判据一字不动**（`arrival > acts.get(v, -math.inf) + EPS`）—— 这是 L0-SPEC §2 ⑥
的硬线；把"取最大"改成别的即越界。

实现形态（见 DESIGN-E39 §1.2）：**子类覆盖**——v2.activate 是单体方法、没有可注入的钩子，
故 v3 只能整体复制该方法（v2 的 39 行）+ 1 行调制。v1/v2 一个字节都不动。

规格见 L0-SPEC.md §2（约定 ⑥）；判据见 DESIGN-E39.md。
"""

from __future__ import annotations

import math
from collections import deque

from synapse_net_v2 import SynapseNet as _SynapseNetV2

EPS = 1e-12


class SynapseNet(_SynapseNetV2):
    """v2 + 到达值算式层的外部调制参数 g（构造参数，默认 0.0 = 与 v2 逐位等价）。"""

    def __init__(self, time_decay_lambda: float = 0.0, inhibition: float = 0.0):
        super().__init__(time_decay_lambda)
        self.inhibition = float(inhibition)

    def activate(self, start, hop_decay: float = 0.9) -> dict:
        """与 v2.activate 逐行相同，只多一个调制项 `- self.inhibition`。

        三种输入（同 v2）：
        - `str`            → 单种子，值 1.0
        - `list[str]`      → 多种子，全部 1.0（与 v1 完全一致，向后兼容）
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
                # ↓↓↓ v3 的唯一改动：到达值减去外部调制 g（max 吸收判据不变）
                arrival = au * self.strength(u, v, now) * hop_decay - self.inhibition
                # ↑↑↑ 上行是 v2 的 `arrival = au * self.strength(u, v, now) * hop_decay` 加一个调制项
                if arrival > acts.get(v, -math.inf) + EPS:
                    acts[v] = arrival
                    if v not in queued:
                        queued.add(v)
                        queue.append(v)
        return acts
