# Synapse-Net 接口契约（INTERFACE）

冻结日期：2026-09-06　依据：代码签名 + FORMULAS.md + BOUNDARY.md

本文件是"输入怎么进、查询怎么出、层与外部模块怎么协作"的**唯一权威文档**。
代码签名与本文件冲突时，以本文件为准（需先改文档、经用户确认，再改代码）。

---

## 0. 总体数据流

```text
               外部世界/传感器/语言模块产出的事件
                          │
                          ▼
   ┌────────── 事件驱动者 (driver) ──────────┐
   │ 同一事件按语义分发给不同层:              │
   │  L0.learn(同步集合)   ← 同时出现的节点  │
   │  L1.learn(有序流)     ← 谁先谁后        │
   │  L2.learn(有序流)     ← 双前缀统计      │
   │  L4.learn_unit(单元)  ← 带边界整体      │
   └─────────────────────────────────────────┘
                          │
       查询: activate / query / spell
                          │
                          ▼
             全量激活图 / 候选分布 / 块记录
                          │
     读取端: attention_filter / 走读器 / stub(外部)
```

约定：**时间戳由驱动者统一传入**，保证跨层事件可对齐；
各层内部时钟只作"本层未显式传 t"时的兜底。

---

## 1. 时间与时钟规则

1. `learn(..., t=None)`：不传 t 时本层时钟 +1；传 t 时本层时钟取
   `max(clock, t)`（时钟单调）。
2. 各层时钟相互独立；需要跨层对齐的事件，驱动者必须给各层传**同一个 t**。
3. t 为 float（整数自动转 float）。时间只影响 last_seen 记录与
   L0 检索衰减 W=N·e^(−λΔt)；L1/L2/L4 当前不因 t 改变计数以外的行为。

---

## 2. 事件编码约定

| 层 | 方法 | 事件语义 | 注意事项 |
| --- | --- | --- | --- |
| L0 | `learn(event: list[str], t)` | 一个同步共现窗口：窗口内节点**不分先后**，两两成连 | 列表内部去重；u≠v，无自环 |
| L1 | `learn(ordered_event: list[str], t)` | 有序到达流：相邻对按方向计数 | 保留重复相邻转移（`[汪,汪,汪]`→2 个 n(汪→汪)） |
| L2 | `learn(ordered_event: list[str], t)` | 有序到达流：滑动相邻三元组 | 只计相邻；长度<3 不产生记录 |
| L4 | `learn_unit(unit: list, t)` | 一个**带边界整体**（词/短语），可含重复位置 | 同一单元重复 ≥K 次成块；spell 保留重复 |
| L4 | `learn_sequence(atoms: list[str], t)` | 块级序列：原子=普通标签或 CH_n | 相邻无序对计数（块级绑定） |

标签约定：`str`；大小写敏感；`CH_n` 为机器块号（不透明、无语义）；
驱动者负责把"同一个物理事件"的同步面、顺序面、边界面分别喂给对应层。

---

## 3. 查询契约

| 层/工具 | 方法 | 输入 | 输出 | 语义 |
| --- | --- | --- | --- | --- |
| L0 | `activate(start, hop_decay=0.9)` | 单个或列表种子（必须已学过） | `dict[label→float]` | **全量激活图**（含弱连接，值∈[0,1]）；未知种子 KeyError |
| L1 | `activate(start, hop_decay=0.9)` | 同上（L1 节点） | 同上 | 方向软图上扩散 |
| L2 | `query(prefix)` | 恰好 2 个标签的前缀 | `dict[cand→strength∈(0,1]]` | 条件计数分布（非扩散）；未知前缀→空 dict |
| L2 | `candidates(prefix)` | 同上 | `dict[cand→int]` | 原始计数视图 |
| L4 | `activate(start, hop_decay=0.9)` | 块级原子种子 | 同 L0 | 块级图扩散 |
| L4 | `spell(chunk_id)` | CH_n | 有序成员表（含重复） | 未知块 KeyError |
| 读取端 | `attention_filter(acts, threshold=None, top_k=None)` | 任意激活图 | 过滤后 dict | 纯函数；threshold∈(0,1]；top_k≥0 |

只读保证：所有查询/过滤**不改变任何层状态**（E3 已验证 state 不变）。

---

## 4. 只读观测 API（供外部模块与实验断言）

| 层 | 方法 |
| --- | --- |
| L0 | `connection_count(a,b)`、`effective_weight(a,b,t)`、`strength(u,v,t)`、`last_seen(a,b)`、`list_nodes()`、`node_count()`、`edge_count()`、`state_snapshot()` |
| L1 | `count(a,b)`、`strength(u,v)`、`neighbors(u)`、`node_count()`、`edge_count()` |
| L2 | `count(a,b,c)`、`prefix_types()`、`triple_types()`、`nodes` |
| L4 | `count_pair(a,b)`、`chunk_ids()`、`degree(node)`、`member_bind` |

外部模块只允许：调用只读 API 读证据 + 通过公开 learn 喂回新经验；
**不得直接修改层内部结构**（connections/nodes/counts 等一律视为私有）。

---

## 5. 错误语义

| 情况 | 行为 |
| --- | --- |
| activate 未学过的种子 | `KeyError`（闭包） |
| spell 未知块 | `KeyError` |
| query 前缀长度 ≠ 2 | `ValueError` |
| attention threshold ∉ (0,1] | `ValueError` |
| top_k < 0 | `ValueError` |

---

## 6. 外部 stub 调用契约（语言/动作/情绪）

三个占位 stub（`language_stub.py`、`motor_stub.py`、`emotion_stub.py`）遵守：

1. 只读基底证据（邻居、计数、激活、块记录）；
2. 通过公开 learn 接口把"输出被听到/看到"作为新经验喂回；
3. 不修改层内部结构；
4. 系统级实验必须输出**功劳记账表**（步骤 / 谁干的 / 内容），
   区分基底贡献与外部贡献；
5. stub 为占位实现，可整体替换为真实模块，接口签名保持稳定。

---

## 7. 组合示例（driver 模式）

把一次"说出 HELLO"事件喂给系统：

```python
t = next_time()                      # 驱动者维护全局单调时间
l0.learn(["H", "e", "l", "l", "o"], t=t)   # 同步面(窗口内共现)
l1.learn(["H", "e", "l", "l", "o"], t=t)   # 顺序面(含 l->l 重复标记)
l2.learn(["H", "e", "l", "l", "o"], t=t)   # 三元组统计
l4.learn_unit(["H", "e", "l", "l", "o"], t=t)  # 边界整体(成块/拼写)
```

查询拼写：

```python
acts = l4.activate("课堂")          # 语境唤起块级图
cid  = pick_chunk(acts)             # 外部选择(读取端)
text = MotorStub().render(MotorStub().spell_out(l4, cid))  # -> "Hello"
```
