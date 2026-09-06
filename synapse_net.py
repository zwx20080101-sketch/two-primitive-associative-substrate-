"""Synapse-Net V0 reference implementation.

Only two axioms live at this layer:

  learn(event, t=None)
      Co-occurrence binding: every unordered pair inside one event is
      recorded once (N_ab += 1, last_seen = t).  Nodes are created on
      first sight.  Connections are permanent: nothing is ever decayed,
      weakened-in-place or deleted from this structure.

  activate(start, hop_decay=0.9)
      Activation diffusion: seeds fire at 1.0 and signals travel along
      connections.  The FULL activation map is returned, including weak
      memories.  Forgetting / focusing is a read-side concern, never
      implemented here.

Retrieval-time effective weight:  W_ab(t) = N_ab * exp(-lambda * dt_ab)
With the V0 default lambda = 0, W equals N and time plays no role yet.
"""

from __future__ import annotations

import math
from collections import deque

EPS = 1e-12


class SynapseNet:
    """Minimal associative cognitive substrate (V0).

    Internal state:
      nodes[label]       = {"first_seen", "last_seen", "appear_count"}
      connections[(a,b)] = {"count", "last_seen"}   (a < b, undirected)
      _adj[label]        = set of neighbor labels
    """

    def __init__(self, time_decay_lambda: float = 0.0):
        self.nodes: dict[str, dict] = {}
        self.connections: dict[tuple[str, str], dict] = {}
        self._adj: dict[str, set[str]] = {}
        self._clock: float = 0.0
        self.lambda_ = float(time_decay_lambda)

    # ------------------------------------------------------------------
    # Axiom 1: co-occurrence binding
    # ------------------------------------------------------------------
    def learn(self, event: list[str], t: float | None = None) -> None:
        """Bind all unordered pairs inside one event (one co-occurrence window)."""
        if t is None:
            self._clock += 1.0
            t = self._clock
        else:
            t = float(t)
            if t > self._clock:
                self._clock = t

        labels = list(dict.fromkeys(event))  # dedupe, keep first-seen order
        for label in labels:
            node = self.nodes.get(label)
            if node is None:
                self.nodes[label] = {"first_seen": t, "last_seen": t, "appear_count": 1}
                self._adj[label] = set()
            else:
                node["last_seen"] = t
                node["appear_count"] += 1

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                key = (a, b) if a < b else (b, a)
                rec = self.connections.get(key)
                if rec is None:
                    self.connections[key] = {"count": 1, "last_seen": t}
                    self._adj[a].add(b)
                    self._adj[b].add(a)
                else:
                    rec["count"] += 1
                    rec["last_seen"] = t

    # ------------------------------------------------------------------
    # Axiom 2: activation diffusion
    # ------------------------------------------------------------------
    def activate(self, start: str | list[str], hop_decay: float = 0.9) -> dict[str, float]:
        """Return the FULL activation map, weak connections included.

        a(v) = best-path product over  hop_decay * strength(u->v)  edges,
        with seeds fixed at 1.0.  A node re-fires only when its activation
        strictly improves; propagation is monotone and terminates.
        """
        seeds = [start] if isinstance(start, str) else list(start)
        if not seeds:
            return {}
        for s in seeds:
            if s not in self.nodes:
                raise KeyError(f"node not in network: {s!r} (learn() it first)")

        now = self._clock
        acts: dict[str, float] = {s: 1.0 for s in seeds}
        queue = deque(seeds)
        queued = set(seeds)

        while queue:
            u = queue.popleft()
            queued.discard(u)
            au = acts[u]
            for v in self._adj.get(u, ()):
                arrival = au * self.strength(u, v, now) * hop_decay
                if arrival > acts.get(v, -math.inf) + EPS:
                    acts[v] = arrival
                    if v not in queued:
                        queued.add(v)
                        queue.append(v)
        return acts

    # ------------------------------------------------------------------
    # Retrieval helpers (never mutate the network)
    # ------------------------------------------------------------------
    def _effective_weight(self, rec: dict, t: float) -> float:
        dt = max(0.0, t - rec["last_seen"])
        return rec["count"] * math.exp(-self.lambda_ * dt)

    def effective_weight(self, a: str, b: str, t: float | None = None) -> float:
        """W_ab(t) = N_ab * exp(-lambda * dt).  With lambda=0 this is N_ab."""
        key = (a, b) if a < b else (b, a)
        rec = self.connections.get(key)
        return 0.0 if rec is None else self._effective_weight(rec, self._clock if t is None else t)

    def strength(self, u: str, v: str, t: float | None = None) -> float:
        """Normalized edge strength as seen FROM u:
        W_uv / max over u's neighbors of W_uw  (0 if no connection).
        """
        now = self._clock if t is None else t
        key = (u, v) if u < v else (v, u)
        rec = self.connections.get(key)
        if rec is None:
            return 0.0
        best = 0.0
        for w in self._adj.get(u, ()):
            k = (u, w) if u < w else (w, u)
            best = max(best, self._effective_weight(self.connections[k], now))
        return self._effective_weight(rec, now) / best if best > 0 else 0.0

    def connection_count(self, a: str, b: str) -> int:
        """N_ab: permanent co-occurrence count (0 when the pair never met)."""
        key = (a, b) if a < b else (b, a)
        rec = self.connections.get(key)
        return 0 if rec is None else rec["count"]

    def last_seen(self, a: str, b: str) -> float:
        key = (a, b) if a < b else (b, a)
        rec = self.connections.get(key)
        return float("nan") if rec is None else rec["last_seen"]

    def list_nodes(self) -> list[str]:
        return sorted(self.nodes)

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.connections)

    def state_snapshot(self) -> dict:
        """Deep-ish copy used to prove that read-side layers change nothing."""
        return {
            "nodes": {k: dict(v) for k, v in self.nodes.items()},
            "connections": {k: dict(v) for k, v in self.connections.items()},
            "clock": self._clock,
        }
