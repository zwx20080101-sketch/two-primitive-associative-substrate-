# 纯公式权威清单（FORMULAS-REF）

只列公式，无文字说明。权威来源：FORMULAS.md 与代码实现。

## L0 共现绑定（learn）

```text
对 E 内无序对 (u,v), u≠v:
  N_uv ← N_uv + 1
  last_seen_uv ← t
```

## L0 检索强度

```text
W_uv(t) = N_uv · exp(−λ · max(0, t − last_seen_uv))
```

## L0 传播强度（每源 max 归一化）

```text
s(u→v; t) = W_uv(t) / max_{w ∈ N(u)} W_uw(t)
```

## L0 激活扩散

```text
种子: a(s) = 1.0
arrival(u→v) = a(u) · s(u→v) · hop_decay
a(v) = max(a(v), arrival(u→v))
```

## L1 有序相邻计数

```text
n(x_i → x_{i+1}) ← n(x_i → x_{i+1}) + 1    (允许 x_i == x_{i+1})
```

## L1 方向强度

```text
raw(u→v)      = n(u→v) + floor
strength(u→v) = raw(u→v) / max_{w 与 u 相邻} raw(u→w)
```

## L2 三元组计数与查询

```text
n((x_i, x_{i+1}) → x_{i+2}) ← +1          (滑动相邻窗口)

strength(c) = n((a,b)→c) / max_c n((a,b)→c)
```

## L4 成块与绑定

```text
unit_count(seq) ← +1
当 unit_count(seq) == K:  创建 CH_n, record(CH_n) = list(seq)
对 seq 去重成员 m: member_bind[(CH_n, m)] ← +1
```

## 读取端

```text
attention: 保留 activation ≥ threshold 的节点
top_k:     按 (−activation, label) 取前 k
sum 候选:  a(v) = min(1.0, Σ 已发射邻居贡献)   (单次发射, 出队冻结)
```

## 参数默认值

```text
hop_decay = 0.9   λ = 0.0(可用 0.005)   K = 5   floor = 1.0
threshold ∈ (0,1]   sum cap = 1.0   L2 前缀长度 = 2
```
