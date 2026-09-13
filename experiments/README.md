# experiments/ 索引

运行：`python main.py`（全量，逐项 PASS/FAIL）或 `python main.py exp5_direction_hello`（单个）。
每个实验把结果写成 `outputs/<id>.json`；所有结论编号见 [CLAIMS-REGISTRY.md](../CLAIMS-REGISTRY.md)。

实验分**两轮**（划分与产物编号见 [README.md](../README.md)「两轮实验阶段」）：
第一轮 = 单层能力目录（E0–E22），第二轮 = 组合律 + S1/S2/S3（E23–E26）。

---

## 第一轮：E0–E22（单层能力目录）

**产物**：C01–C22 / B01–B07　**设计文档**：[DESIGN.md](../DESIGN.md)、[REPORT-zh.md](../REPORT-zh.md)

主题：一次加一层、每层一个现象；后面追加消融（E20）、规模（E21）、统计（E22）。

| 文件 | 主题 | 对应登记 |
| --- | --- | --- |
| `exp0_smoke.py` | 冒烟：空网络与单事件绑定 | — |
| `exp1_chain.py` | 26 字母链：纯共现自组织出顺序链 | C01 |
| `exp2_strength.py` | 重复经验强度：底层全量保留，阈值决定可访问 | C02 |
| `exp3_threshold.py` | 阈值状态开关：同一底层输出的三种认知内容 | C02 |
| `exp4_walk_alphabet.py` | 逐对强化 + 链式走读（双向对称） | C03 |
| `exp5_direction_hello.py` | L1 方向层：Hello 正走顺、倒走费力 | C03 / C04 |
| `exp6_fork_context.py` | 分叉与上下文：底层只给强度分布，不判对错 | C05 / C06 |
| `exp6b_translation_synonym.py` | 翻译/同义：经共享情景互译、因果联想链 | C07 |
| `exp7_dense_episode.py` | 密集瞬间：一次摔倒=时间轴上的脉冲流 | C08 |
| `exp8_hub_graph.py` | 连接是图不是线：共享枢纽支撑多条句子 | C09 |
| `exp9_partial_completion.py` | 部分输入补全：缺口回填，不许无中生有 | C10 |
| `exp10_recency_lambda.py` | 新旧程度：λ 检索衰减，N 永久 | C11 |
| `exp11_interference.py` | 多词干扰：歧义集中在共享枢纽，情景可分离 | C12 |
| `exp12_chunk_nesting.py` | L4 组块：整词/双写/块复用/嵌套句 | C13 |
| `exp13_repeat_marker.py` | 基底重复标记：同窗二次激活（叠词的基底部分） | C14 |
| `exp14_multi_path_evidence.py` | 多路证据：max vs sum 会合规则对比（读取端） | C15 |
| `exp15_higher_order_context.py` | L2 三元组：双前缀 → 下一个 | C16 |
| `exp16_role_markers.py` | 角色标记归属：语序/格标记区分 狗咬猫 vs 猫咬狗 | C17 |
| `exp17_system_reduplication.py` | M2-A 叠词外推（系统级：基底 + 语言 stub） | C18 |
| `exp18_spelling_output.py` | M2-B 拼写输出（块记录 → 动作 stub → 听回闭环） | C13 |
| `exp19_emotion_modulation.py` | M2-C 情绪调制（读取端增益，无需写侧调制轴） | C19 |
| `exp20_ablation_suite.py` | 消融套件：机制 ON/OFF/随机/纯计数 | C20（另被 C04/C06/C07/C10/C13/C17/C19 引用为 E20A–H） |
| `exp21_scale_consistency.py` | 规模一致性：六现象 1x/10x/100x | C21 |
| `exp22_sampling_statistics.py` | 随机输入族抽样成功率 + Wilson 95% CI | C22 |

---

## 第二轮：E23–E26（组合律 + S1/S2/S3）

**产物**：C23–C26 / B08–B19　**设计文档**：[DESIGN-E23.md](../DESIGN-E23.md)、
[DESIGN-E24.md](../DESIGN-E24.md)、[DESIGN-E25.md](../DESIGN-E25.md)、
[DESIGN-E26.md](../DESIGN-E26.md)

主题：多层同时打开时的合成与归因（E23），以及从 L0 统计里能读出的宏观结构
——**S1 模块分化 / S2 模块内动力学 / S3 模块间协调**（E24/E25/E26）。
三层都是**读取端分析**，不写回记忆。

| 文件 | 主题 | 对应登记 |
| --- | --- | --- |
| `exp23_composition.py` | 组合律：L1+L2+L4 同时开启 → 现象共存且可分别归因 | C23 / B08 |
| `exp24_module_differentiation.py` | **S1 模块分化**：ICA 预注册失败 → 谱聚类读出恢复植入模块 | C24 / B09 / B10 / B11（另 B19） |
| `exp25_module_dynamics.py` | **S2 模块内动力学**：模块平均激活的双稳 + 滞回 | C25 / B12 / B13 / B14 |
| `exp26_module_coordination.py` | **S3 模块间协调**：EI 整合量 + 剂量响应（精确枚举 2^12） | C26 / B15 / B16 / B17 / B18 |

展示层（不在实验结果内）：[playground.html](../playground.html) 的 S1–S3 **只读分析面板**
（P5a，快照回放 + 字段级自检 + commit 溯源），设计见 [DESIGN-P5.md](../DESIGN-P5.md)；
校验脚本 [verify_playground.py](../verify_playground.py)。

---

## 边界量化：E27 / E28 / E29 / E30（不属于两轮能力目录）

**产物**：C27–C30 / B01–B05（升级/指针）/ B20–B21　**设计文档**：[DESIGN-E27.md](../DESIGN-E27.md)、[DESIGN-E28.md](../DESIGN-E28.md)、[DESIGN-E29.md](../DESIGN-E29.md)、[DESIGN-E30.md](../DESIGN-E30.md)

主题：把第一轮的边界从"观测记录"升级为带数字的律。**纯计数，无 numpy 依赖；不改任何层文件。**

| 文件 | 主题 | 对应登记 |
| --- | --- | --- |
| `exp27_context_window.py` | 上下文窗口边界：`k_min = k*`；现有层天花板 = 2；probe-k 为实验侧只读探针（不是新层） | C27 / B01（升级） |
| `exp28_readout_modes.py` | 读取端 max/sum 系统对比：max 精确不叠加（`max(a,a)=a`）、sum 封顶且**顺序依赖**（T5 三角）；T4 环为 sum 终止性自检 | C28 / B02（指针）/ B03（升级） |
| `exp29_hierarchy_scales.py` | 层级 = **多尺度读出**：k=2 读超模块、k=4 读子模块；消融跨子模块边后层级塌陷；L0 统计跨 k 逐位不变 | C29 / B20 |
| `exp30_closure_audit.py` | **闭包审计**：五个已学集合逐层核对（违例 = 0）；失败语义表（KeyError / `{}` / None / ValueError 并存） | C30 / B21（B05 指针） |

---

## 依赖与约定

- 第一轮：仅标准库；`exp20/21/22` 另有自带数据文件（`outputs/exp20_ablation.json`、
  `exp21_scale.json`、`exp22_stats.json`）。
- 第二轮：`exp24/25/26` 依赖 **numpy**（谱聚类、精确 EI 枚举）。
- 纪律（必读）：[NEXT-PLAN.md](../NEXT-PLAN.md) §5 + [CLAIMS-REGISTRY.md](../CLAIMS-REGISTRY.md)
  「写作纪律」——预注册不写方向倾向；预检观察 ≠ 结论文；
  块对角/零耦合语料上整合量必须精确为 0；信息论量不得为负。
