"""Layer 1 (L1): 方向/顺序层 (Direction / Order Layer).

V0 底层 (synapse_net.py) 已冻结, 永不再修改。本层是加在它上面的第一层:

  - L0 (冻结) 负责"同时/共现": 谁和谁一起出现过 -> 永久无向连接 N_ab;
  - L1 负责"先后/顺序": 相邻到达的字母对按实际方向分别计数;
    同一节点相邻再次到达也计数(自转移 = 重复标记, 如 汪->汪)。

L1 与 L0 的关系, 就像"左手传给右手"和"右手传给左手"要分开记账:

  directed_count[(u, v)]  = u 先于 v 相邻出现的次数
  directed_count[(v, u)]  = v 先于 u 相邻出现的次数

两者不假定相等; 没发生过的方向计数为 0, 但保留 floor(弱可逆保底),
使"倒背"依然可能, 只是不顺、耗能大。

传播公式 (u 作为信号源流向 v):
  raw(u->v)     = directed_count[(u, v)] + floor
  strength(u->v)= raw(u->v) / max_{w 与 u 相邻} raw(u->w)
  arrival       = a(u) x strength(u->v) x hop_decay

即: 传播"顺不顺" = 该方向真实发生过多少次 / 从 u 出发最强的那条路。
"""

from __future__ import annotations

import math
from collections import deque

EPS = 1e-12


class OrderLayer:
    def __init__(self, floor: float = 1.0):
        self.nodes: set[str] = set()
        # directed_count[(u, v)] = 事件中 u 紧挨着先于 v 出现的次数
        self.directed_count: dict[tuple[str, str], int] = {}
        # 邻接索引(纯结构优化, 不改语义): 出/入边集合
        self._out: dict[str, set[str]] = {}
        self._in: dict[str, set[str]] = {}
        self.floor = float(floor)
        self._clock = 0.0

    # ------------------------------------------------------------------
    # 学习: 一次"按先后到达"的事件
    # ------------------------------------------------------------------
    def learn(self, ordered_event: list[str], t: float | None = None) -> None:
        """记录相邻到达对 (a, b), 即 a 先 b 后, n(a->b) += 1。

        允许有序自转移: a==b 时记录"同一节点相邻再次出现"(重复标记),
        不携带任何语义, 只标记时间性重复这一事实;
        同一事件内出现 A,B,A 这类往返时, 两个方向各自计数。
        """
        if t is None:
            self._clock += 1.0
            t = self._clock
        else:
            t = float(t)
            if t > self._clock:
                self._clock = t

        self.nodes.update(ordered_event)
        for a, b in zip(ordered_event, ordered_event[1:]):
            key = (a, b)
            self.directed_count[key] = self.directed_count.get(key, 0) + 1
            self._out.setdefault(a, set()).add(b)
            self._in.setdefault(b, set()).add(a)

    # ------------------------------------------------------------------
    # 读取: 方向强度与激活扩散 (纯读取, 不改记忆)
    # ------------------------------------------------------------------
    def neighbors(self, u: str) -> set[str]:
        """u 的可达邻居 = 出边目标 ∪ 入边来源(与原扫描语义一致, O(度))。"""
        return self._out.get(u, set()) | self._in.get(u, set())

    def count(self, a: str, b: str) -> int:
        """n(a->b): a 先于 b 相邻出现的次数 (没发生过 = 0)。"""
        return self.directed_count.get((a, b), 0)

    def strength(self, u: str, v: str) -> float:
        """从 u 流向 v 的方向强度: 次数越多越顺, 反方向只有 floor 保底。"""
        raw = self.directed_count.get((u, v), 0) + self.floor
        best = raw
        for w in self.neighbors(u):
            best = max(
                best,
                self.directed_count.get((u, w), 0) + self.floor,
            )
        return raw / best if best > 0 else 0.0

    def activate(self, start: str | list[str], hop_decay: float = 0.9) -> dict[str, float]:
        """沿方向强度扩散; 顺向一路顺, 反向越走越弱。"""
        seeds = [start] if isinstance(start, str) else list(start)
        if not seeds:
            return {}
        for s in seeds:
            if s not in self.nodes:
                raise KeyError(f"node not in order layer: {s!r}")

        acts: dict[str, float] = {s: 1.0 for s in seeds}
        queue = deque(seeds)
        queued = set(seeds)
        while queue:
            u = queue.popleft()
            queued.discard(u)
            au = acts[u]
            for v in self.neighbors(u):
                arrival = au * self.strength(u, v) * hop_decay
                if arrival > acts.get(v, -math.inf) + EPS:
                    acts[v] = arrival
                    if v not in queued:
                        queued.add(v)
                        queue.append(v)
        return acts

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        """有方向记录的对数 (每个方向一条)。"""
        return len(self.directed_count)
