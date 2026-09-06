# V2 论文大纲（PAPER-OUTLINE）

日期：2026-09-06　底线：**科学真实**——每一条结论必须指向实验文件与数据；
不追求一鸣惊人；任何没有证据支撑的表述一律不写。

## 0. 候选题目

- EN：A Two-Primitive Associative Substrate: Co-occurrence Binding and
  Activation Diffusion with Specialized Read-Out Layers
  (A Minimal Reference Implementation and Controlled Reproduction Study)
- 中文：双原语联想基底：共现绑定与激活扩散及其专用读取层——
  一个最小参考实现与受控复现实验

## 1. 摘要骨架（每条都对应正文证据）

1. 问题：仅用"共现绑定 + 激活扩散"，外加分层表示与外部读取层，能覆盖多少联想/时序/语言式/价值式行为？
2. 方法：预先注册判据的确定性 toy 实验（E0–E19）+ 消融/对照（E20）
   + 规模一致性（E21）+ 随机输入族抽样与 Wilson CI（E22）；
3. 结果（保守表述）：在所考察的受控输入族内，机制足以产生
   联想链、补全、方向不对称、时间衰减、共享情景绑定、组块/双写、
   角色区分、叠词外推（配合外部 stub）与价值读取差异；
4. 边界：3 前缀以上、真递归/系统性泛化、真实感知运动不在范围内；
5. 声明等级：充分性演示 + 抽样一致性；**不是人脑等同性主张**。

## 2. 章节安排

| 章 | 内容 | 主要证据来源 |
| --- | --- | --- |
| 1 Introduction | 动机、与联想扩散/组块文献的关系、本文定位 | 无实验数据（引用综述） |
| 2 Architecture & Formulas | L0–L4 + 读取端 + stub；公式 | FORMULAS/INTERFACE/PARAMETERS |
| 3 Method | 实验协议、判据先行、确定性、统计方法 | DESIGN/STATUS 方法说明 |
| 4 Results | 现象族结果（见 4.x） | outputs/exp*.json |
| 5 Boundaries | 诚实边界与负结果清单 | CLAIMS-REGISTRY 边界行 |
| 6 Discussion | 支持的结论 / 不支持的结论 / 与人脑关系的明确限定 | CLAIMS-REGISTRY |
| 7 Limitations | toy 规模、给定 token、无真实感知等 | — |
| 8 Reproducibility | 命令、脚本、数据、版本快照 | README/main.py/outputs |

### 4. 结果小节（每个数字/现象都登记）

- 4.1 结构自组织：E1（26 字母链）、E4（双向等强）
- 4.2 检索强度与阈值：E2、E3、E10（λ 检索衰减、记忆永存）
- 4.3 方向与顺序：E5（正走顺/倒走费力）、E15（双前缀）、E16（角色）
- 4.4 语义拓扑：E6b（翻译绑定）、E8（共享枢纽/并行候选/组合≠直连）
- 4.5 时间与事件：E7（脉冲密度/预警）、E13（重复标记）
- 4.6 组块与拼写：E12、E18（双写还原与听回闭环）
- 4.7 系统级：E17（叠词外推+功劳记账）、E19（情绪读取端）
- 4.8 证据综合：E20（消融表）、E21（规模一致性）、E22（Wilson CI）

## 3. 图表清单（每个图/表必须有 artifact 对应）

| 图表 | 内容 | 来源 artifact |
| --- | --- | --- |
| Fig.1 架构图 | L0–L4/读取端/stub 数据流 | INTERFACE.md 总体数据流 |
| Table 1 实验总表 | E0–E22 现象/判据/结果 | STATUS.md §3 + outputs/*.json |
| Table 2 消融表 | 八现象 ON/OFF/基线 | outputs/exp20_ablation.json |
| Fig.2 规模曲线 | 六现象 1×/10×/100× 通过率 | outputs/exp21_scale.json |
| Table 3 统计表 | 随机输入族成功率 + Wilson CI | outputs/exp22_stats.json |
| Table 4 功劳记账 | 系统级场景谁做了什么 | E17/E18/E19 ledger |
| Table 5 边界清单 | 失败/超出范围现象 | CLAIMS-REGISTRY 边界行 |

## 4. 科学诚信硬规则（写入正文与复现包）

1. 每一条主张 = 登记表编号 → 实验文件 → 数据 JSON，三件可追溯；
2. 表述模板："在 [输入族 X] 下，机制 M 以 [确定性/抽样率 r(CI)] 产生 Y"；
   禁止写超出输入族的全称结论；
3. 边界与负结果必须出现在正文（如 L1 无上下文分叉、3 前缀上限、
   max 不叠加、sum 拓扑≠底层、平面链无双写）；
4. 统计方法如实描述：确定性主证据 + M=60 随机输入族 + Wilson 95% CI；
   判据在跑之前注册（已在实验文件与 DESIGN 中留存），无事后挑选；
5. 不写"人脑就是这样"，只写"与若干已知现象的定性模式一致"；
6. 版本快照：V2 发布前 git init + tag v1.0，代码/文档/数据一并归档。

## 5. 复现包清单

- 代码：层文件 + 外部 stub + 实验 E0–E22 + benchmark
- 数据：outputs/*.json（已齐全，22 实验 + 3 套件 + benchmark）
- 文档：README/DESIGN/STATUS/FORMULAS/INTERFACE/PARAMETERS/
  BENCHMARK/BOUNDARY/ROADMAP/REVIEW/GATE/RESEARCH/CLAIMS
- 运行：`python main.py`（全量）、`python main.py expN`、`python benchmark.py`
