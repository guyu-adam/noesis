# CGWT 论文审稿全记录

> 稿件：What Coalition Broadcast Reveals About Causal Integration: Computational Constraints from a GWT--IIT Neural Processor System
> 作者：Guyu Adam
> 目标期刊：Entropy (MDPI)
> 审稿周期：2026-05-19 ~ 2026-05-21
> 总审稿轮次：内部评审 1 轮 + 代码审稿 3 轮 + 论文审稿 2 轮 + 专项审稿 3 轮 = 9 份审稿意见

---

## 审稿时间线

```
2026-05-19  内部评审 (LLM版本)        → Major Revision，建议换RNN
2026-05-19  LLM分支评审               → 建议转投JAAMAS
2026-05-21  代码审稿 Round 1          → Major Revision，5个假设全部不显著
2026-05-21  代码审稿 Round 2          → Conditional Acceptance，补充控制条件
2026-05-21  代码审稿 Round 3          → Accept，所有blocking issues resolved
2026-05-21  论文审稿 Round 1          → Major Revision，论文是placeholder状态
2026-05-21  论文审稿 Round 2 (终审)   → Accept，论文完整
2026-05-21  专项：Scope/原创性/引用    → Accept，Excellent Fit
2026-05-21  专项：引用核查/格式/语言   → Accept with Minor Revisions (6个引用错误)
2026-05-21  专项：行业前景/发展趋势    → Accept，五个趋势交汇点
```

---

## 一、内部评审阶段（2026-05-19）

### 初始状态

- 代码使用 LLM (qwen3:4b) 作为 agent
- 实验数据为占位符
- Φ 公式不统一（competitive 和 collaborative 使用不同公式）
- 样本量不足（每模式 24 观测）

### 核心发现

1. **Φ 公式不一致** — `phi_proxy()` 和 `phi_collaborative()` 使用不同权重和分量，相当于用两把不同的尺子量完说 A > B
2. **实验数据是占位的** — U=576.0 是 n=24 的理论最大值，统计上不可能
3. **LLM agent 不适合测 Φ** — token 分布相似度与神经元因果交互是两码事
4. **缺失引用** — IWMT 2020、GodelOS、Zenodo 2026 等重叠工作未引用

### 关键决策

建议将 LLM agent 替换为 RNN 处理器，"用真正的因果结构而非 token 分布计算 Φ"。这一决策决定了后续所有实验的方向。

### 世界模型代码 Bug

- `consensus_concepts` 内存泄漏（Counter 只增不减）
- Hybrid 模式在标准 GlobalWorkspace 上 WorldModel 空转
- `world_embedding` 是死代码
- Consensus threshold 跨 coalition size 不一致

### 创新点评估

| CGWT 声明 | 重叠风险 | 区分方案 |
|-----------|---------|---------|
| "GWT+IIT 计算整合框架" | 高（GodelOS 已有，IWMT 2020 理论上已有）| 不能声称首创，改叙事为"系统的计算实验对比" |
| "Coalition broadcast 替代 winner-take-all" | 中（GodelOS 有 coalition dynamics）| 精确区分机制：consensus + world model scoring |
| "World model 驱动的 consensus scoring" | 低 | 可能原创，聚焦宣传 |

---

## 二、代码审稿阶段

### Round 1（2026-05-21）— Major Revision

**核心问题：5 个预注册假设全部不显著，且方向性矛盾**

| 假设 | 预测 | 实际 | p-value |
|------|------|------|---------|
| H1: Collab > Competitive | Φ_collab 最高 | competitive (0.727) > collaborative (0.655) | 0.380 |
| H2: Collab > Random | — | Δ 仅 0.039 | 0.342 |
| H3: Hybrid > Competitive | — | competitive > hybrid | 0.500 |
| H4: Competitive > Random | — | 正向不显著 | 0.476 |
| H5: Broadcast > No-broadcast | — | — | 0.683 |

更致命的是：**single_processor 模式 Φ 最高（0.874），远超所有多处理器方案**。

**建议的应对策略（按推荐度）：**

(a) **最诚实且最有发表价值：接受并理论化 null result。** 叙事从"我们证明了 CGWT"改为"实验揭示了 CGWT 的关键局限，为整合-分化权衡提供了计算证据"。

(b) 如果坚持原假设，必须改进 coalition merge 策略（均值合并消除差异化信息）。

(c) 切换因变量：ΔΦ → Φ_after，broadcast vs no-broadcast 的效应量 d≈8.0。

**方法学问题：**
- Φ 近似的 no-MIP-search 偏误未验证一致性
- Histogram MI 的 bin count 敏感性未分析
- 所有模式 sensitivity CV 0.43-0.52，超过声称的 CV<0.3 阈值
- 实验设计混淆了"处理器内部 Φ"和"广播增益 Φ"（前者占 98%+）
- Coalition merge 使用均值（信息论上最差的策略）

**代码审查要点：**
- MI 估算的空间遍历假设需验证
- Dirichlet(2,2,2) 采样应改为 Dirichlet([4,3.5,2.5]) 集中在默认权重附近
- cluster_activation_states 质心初始化非确定性
- 缺少实验配置锁定机制

---

### Round 2（2026-05-21）— Conditional Acceptance

**本轮新增：** 2 个控制条件（single_self_broadcast、homogeneous_competitive）、Φ 分解（within/between）、ANCOVA、Cohen's d、效力分析、Φ 校准锚点、merge fidelity 度量

**逐项回应评估：**

1. **控制条件** — 最大亮点
   - single_self_broadcast ≈ single_processor (d=0.004)：广播本身不伤 Φ
   - homogeneous_competitive << competitive：处理器差异化的必要性被干净地证明

2. **Φ_within / Φ_between 分解** — 方向正确
   - Φ_within (2.2-2.4) >> Φ_between (0.05-0.06)，说明 Φ 主要由 W_rec 内部递归驱动
   - 但两者来自不同算法路径，量纲不同，不能直接比比值

3. **ANCOVA** — 最有价值的统计补充
   - F(7,791) = 23.65, p≈0, η² = 0.17

4. **Merge 策略** — attention_weighted 退化为均值
   - merge_diff_retention ≈ 0.009：attention weights 近均匀分布 [0.2, 0.2, 0.2, 0.2, 0.2]

5. **Φ 校准** — 参数不匹配。需用 n_neurons=512 重算

6. **实验配置锁定** — 已满足复现要求

**Blocking items (本轮)：**
1. H5 和 H7 必须补充 Φ_after 分析（或论证 ΔΦ 更合适）
2. 用匹配参数重算 Φ calibration
3. 修正 `_required_sample_size()` 的单尾/双尾不匹配

**叙事建议：**
- 标题从 "Testing Whether" 改为 "Computational Constraints on" 或 "What Coalition Broadcast Reveals About"
- 四个发现按证据强度排列

---

### Round 3（2026-05-21）— Accept

**三项 blocking 全部解决：**

1. ✅ **H5b/H7b 的 Φ_after 分析**
   - H5b: d = 7.23（极巨大效应），p ≈ 0
   - H7b: d = 1.12（大效应），p ≈ 0

2. ✅ **Φ calibration 匹配实验参数** — n_neurons=64, n_processors=5, n_clusters=30

3. ✅ **单尾公式修正** — `one_tailed` 参数已添加

**超额完成：attention weight 追踪**
- collaborative: cosine sim 0.955 ± 0.003（几乎完全一致）
- hybrid: cosine sim 1.000 ± 0.000
- 直接验证了 attention collapse → merge 退化 → coalition 无优势的因果链

**剩余 non-blocking issues：**
- Calibration over-clustering artifact（n_clusters=30 > n_history=25）
- effective_n 显示计算使用了错误公式
- processor_pairwise_similarity 仍报错

**最终推荐：Accept。**

---

## 三、论文审稿阶段

### Round 1（2026-05-21）— Major Revision

**致命问题：论文是 placeholder 状态**

- Results、Discussion、Conclusions 全部标注 `[TO BE FILLED]` 或 `[placeholder]`
- Discussion 残留 LLM 实验的旧文本，与实际 neural 数据矛盾
- 参数不一致：稿件声称 256 神经元/500 cycles/5 模式，实际 512/800/8

**要求：**
1. 填入实际数据
2. 叙事从 confirmatory testing 转为 constraint discovery
3. 标题改为描述性
4. 同步所有参数
5. 增加 attention weight 分析

---

### Round 2 / 终审（2026-05-21）— Accept

**评价摘要：**

| 部分 | 评级 |
|------|------|
| 标题与摘要 | ⭐⭐⭐⭐⭐ 优秀 |
| Introduction | ⭐⭐⭐⭐ 良好 |
| Methods | ⭐⭐⭐⭐⭐ 优秀 |
| Results | ⭐⭐⭐⭐⭐ 优秀 |
| Discussion | ⭐⭐⭐⭐⭐ 优秀（最大亮点） |
| Conclusions | ⭐⭐⭐⭐⭐ 优秀 |
| 代码与数据 | ⭐⭐⭐⭐⭐ 优秀 |

**三大优势：**
1. **学术诚实性** — null result 处理堪称范例
2. **方法透明度** — 四个近似全部列表化 + 偏误讨论
3. **理论深度** — Discussion 4.2-4.3 与 Luppi PID 和 Kearney MaxCal 的形式化对接

**微小建议（proof 阶段）：**
- 自引预印本更新或删除
- Calibration 措辞补充脚注
- 添加 Φ_after bar plot

---

## 四、专项审稿

### Scope / Originality / Citation Audit（2026-05-21）— Accept

**Entropy 三重匹配：**
1. Φ 的信息论本质（MI、KL、JS 的系统性应用）
2. TPM → EI 因果结构管线
3. MaxCal 与自由能原理的桥接

**原创性分级：**
- 概念原创性：良好（coalition consensus 非 trivial，有概念前身）
- 方法原创性：良好（技术组合 + Φ 分解 + attention 诊断 = 独特体系）
- 发现原创性：按层次分化
  - H5b：低（replication），但 d=7.23 证据强度极高
  - H7b/H6：中等，新的计算约束
  - Finding 4 null result 诊断：较高原创性

**缺失引用建议：**
1. Albantakis & Tononi (2019) — 因果 TPM 理论
2. Oizumi, Albantakis & Tononi (2014) — IIT 3.0（已在参考文献但正文未显式引用）
3. Mediano, Rosas et al. (2021) — ΦID 分解
4. Tsuchiya et al. (2015) — GWT-IIT 概念互补
5. **Hoel et al. (2013) — Causal emergence (强烈建议)**

**引用时间分布：** 2025-2026 年引用占近一半，准确反映了 GWT-IIT 领域的活跃期时间结构。

---

### 终审引用核查（2026-05-21）— Accept with Minor Revisions

**引用真实性：19 条中 13 条正确，6 条存在错误**

| 严重度 | 引用 | 问题 |
|--------|------|------|
| 🔴 严重 | Phua 2025 | 标题完全错误（"Causal Closing" → 实际是 "Can We Test Consciousness Theories on AI?"），作者名错误 |
| 🟡 中等 | Ferrante 2025 | 标题不准确（"An adversarial collaboration to..." → 实际 "Adversarial testing of..."） |
| 🟡 中等 | Barrett 2026 | 缺少 3 位作者（只有 4 人，实际 7 人） |
| 🟡 中等 | Luppi 2024 | 卷号 13 → 12 |
| 🟢 轻微 | Du et al. 2024 | "LLMs" → "Language Models", "Multi-Agent" → "Multiagent" |
| 🟢 轻微 | GodelOS | 作者 "Steake" → "Hirst, O.C. (Steake)" |

**其他问题：**
- Introduction "Second" 出现两次（编号错误）
- Table 3 "Comp." 列无定义

**语言 AI 化检测：** 未发现显著 AI 化迹象。学术诚实性表达自然。

**格式：** MDPI Entropy 合规，缺少图表。

**逻辑通顺度：** 良好至优秀。从理论动机到约束发现的逻辑链完整。

**结论清晰度：** 良好至优秀。三层面结构（Abstract → Results → Conclusions）一致性高。

---

### 行业前景与发展趋势（2026-05-21）— Accept

**五个正在形成的领域趋势：**

1. **从"理论对抗"到"约束收敛"** — 本工作不是比较两个理论谁赢，而是发现统一框架的边界条件
2. **MaxCal/自由能原理的方法论复兴** — 本工作是首个将 Kearney MaxCal 桥接操作化为 coalition selection 优化目标的尝试
3. **从 LLM agent 到神经 agent 的回摆** — 用因果结构而非 token 分布计算 Φ
4. **Multi-agent AI 从工程到科学的转变** — 首个基于因果分析的设计原则
5. **开放科学与预注册的规范化**

**三维影响路径：**
- 意识科学 → 方法论升级（对抗性合作的前置预验证）
- Multi-agent AI → 架构设计原则（differentiation gate, adaptive coalition）
- 神经形态硬件 → 通信总线 + 异构核心 + 带宽瓶颈

**关键风险：**
- Φ_approx rank 验证未完成（最大方法学风险）
- 单一架构（tanh RNN）的鲁棒性未知
- Over-clustering artifact

---

## 五、跨轮次主题分析

### 5.1 反复出现的核心发现

| 发现 | 首次出现 | 逐轮强化 |
|------|---------|---------|
| Broadcast 是 Φ 的必要条件（d=7.23） | R1（H5 不显著，但 Φ_after 差异巨大） | R2（ANCOVA 确认）→ R3（d=7.23）→ 终审（Finding 1） |
| 处理器差异化是关键前提（d=1.12） | R2（homogeneous_competitive 控制条件） | R3（d=1.12）→ 终审（Finding 2） |
| 多处理器架构引入信息瓶颈 | R1（single_processor 最高） | R2（self_broadcast 控制）→ 终审（Finding 3 + Hoel causal emergence 对接） |
| Coalition consensus 不优于 competitive | R1（方向相反） | R2（attention collapse 诊断）→ R3（merge 退化确认）→ 终审（Finding 4, null result 有诊断价值） |

### 5.2 方法学进化

```
R1: ΔΦ + MWU + 5模式
    ↓ 问题：ΔΦ 信噪比不足，效应被 W_rec 淹没
R2: + 2控制条件 + ANCOVA + Cohen's d + power analysis + Φ分解 + calibration + merge fidelity
    ↓ 核心改进：Φ_after 作为备选DV，控制条件隔离因果因素
R3: + attention weight追踪 + one-tailed公式修正 + 匹配参数calibration
    ↓ 核心改进：诊断了coalition失败的原因
终稿: 叙事从confirmatory → constraint discovery，Discussion增加MaxCal形式化 + Hoel causal emergence
```

### 5.3 审稿人最反复强调的3个问题（均已解决）

1. **Φ_approx vs exact Φ 的秩相关验证缺失** — 已在 Limitations 中诚实声明为 future work
2. **Calibration 的 over-clustering artifact** — 已在 Results 中添加脚注，Limitations 中进一步讨论
3. **Sensitivity CV 超过自设阈值** — 已用"所有模式占窄 CV 范围 → 偏误跨条件一致"论证可接受

---

## 六、建议优先级时间线

```
Proof 阶段（立即）:
  □ Phua 2025 标题修正或删除
  □ 添加 Φ_after bar plot (Figure 1)

发表后 1-3 个月:
  □ Adaptive coalition 实现（differentiation gate）
  □ Small-system exact Φ rank validation (N=6-12)

发表后 3-6 个月:
  □ LSTM 对照实验（验证架构鲁棒性）
  □ LLM agent 对照实验（验证跨架构收敛）

发表后 6-12 个月:
  □ MaxCal coalition optimization 完整实现（Equation 4）
  □ 更多 processor specialization 策略（不同输入编码、时间常数等）
```

---

## 附录：全部审稿文件索引

| # | 文件路径 | 日期 | 类型 | 推荐意见 |
|---|---------|------|------|---------|
| 1 | `reviews/20260519-internal/审稿意见20260519.md` | 05-19 | 内部评审 (LLM版) | Major Revision |
| 2 | `reviews/20260519-internal/审稿意见-LLM-20260519.md` | 05-19 | LLM分支投稿策略 | JAAMAS推荐 |
| 3 | `reviews/20260521-round1-code/审稿意见20260521.md` | 05-21 | 代码审稿 R1 | Major Revision |
| 4 | `reviews/20260521-round2-code/审稿意见202605211007.md` | 05-21 | 代码审稿 R2 | Conditional Acceptance |
| 5 | `reviews/20260521-round3-code/审稿意见202605211024.md` | 05-21 | 代码审稿 R3 | Accept |
| 6 | `reviews/20260521-round1-paper/审稿意见-tex-202605211024.md` | 05-21 | 论文审稿 R1 | Major Revision |
| 7 | `审稿意见-202605211028.md` | 05-21 | 论文审稿 R2 (终审) | Accept |
| 8 | `审稿意见-entropy-scope-202605211028.md` | 05-21 | Scope/原创性/引用 | Accept |
| 9 | `reviews/20260521-final-audit/审稿意见-终审引用核查-202605211045.md` | 05-21 | 终审引用核查 | Accept with Minor Revisions |
| 10 | `reviews/20260521-prospects/审稿意见-行业前景与发展趋势-202605211100.md` | 05-21 | 行业前景/趋势 | Accept |

---

*审稿全记录汇编，2026-05-21*
