# 修改意见

针对你的实际情况：单作者、无机构、一篇中文核心、投 Entropy。

---

## 1. 重新定位论文叙事

当前论文的核心问题不是数据，是**叙事和数据的错位**。

三轮实验跑完后的稳定事实：
- 七个多处理器模式 Φ 全部落在 1.022–1.049，无论怎么广播
- 多处理器 Φ 是单处理器的约 2.2 倍
- 差异化有微弱但稳健的效应（clustered d=0.13）
- Φ_between（广播增益）在所有模式下都是常数 0.05

当前论文的 Introduction 还在用 hypothesis-testing 框架（H1-H7, 声称要验证 CGWT）。但你的数据不支持这个叙事——你做的是 honest science，那就按 honest science 写。

**建议改为**：这篇论文的贡献不是"验证了 CGWT"，而是**搭建了一个可诊断的计算测试台，发现了 Φ_approx 在 RNN 系统中实际测什么**。三个经验约束（广播无效应、Φ 随神经元数线性增长、差异化有微弱效应）是副产品，诊断工具包（Φ 分解 + 滑动窗口 TPM + 聚类统计 + coalition 追踪）是核心贡献。

具体操作：
- 把 Introduction 里的 H1-H7 列表精简，移到 Methods 作为"预注册方向"，不再作为论文主干
- Abstract 同步到新数据
- Discussion 第一段不再说"最重要的结果是 null result"，改为"我们测量了 Φ_approx 对广播机制的敏感性，发现在当前实现中它由处理器内部递归结构主导"
- 论文标题可以考虑加副标题："A Diagnostic Toolkit and Empirical Constraints"

---

## 2. Abstract 必须改

当前 abstract 还是旧数字（Φ=0.63-0.86, d=7.43, single > multi）。这是最大的硬伤——审稿人一眼看到 abstract 和 Results 表格对不上，直接拒。

改为：所有多处理器模式 Φ 落在 1.022-1.049，广播机制解释零方差；Φ 由神经元总数主导；差异化效应 d=0.13；coalition 形成成功但不增加 Φ；两项实现缺陷通过自行开发的诊断工具发现；核心贡献是方法论而非经验发现。

---

## 3. 补两个小但重要的问题

### 3.1 标定数据用同规模跑
当前 calibration 用 64 neurons，实验用 512。审稿人会问：如果 Φ 对神经元数敏感，为什么用不同规模标定？直接用 512 neuron × 5 processor 跑一遍 calibration，让数字可以和实验直接对比。

### 3.2 处理器功能性相似度分析
`processor_pairwise_similarity` 表已经在 metadata 里了。reasoner_vs_integrator = 0.20，其他 9 对都 < 0.07。这说明"结构性差异化"不等同于"功能性差异化"——面对相同 stimulus，不同 W_rec 的处理器输出高度相似。这解释了为什么 coalition 没有 Φ 优势。把这个分析写进 Results，比你现在的"attention weight collapse"解释更有说服力。

---

## 4. Φ 近似未验证这个问题，诚实面对

审稿人最可能抓住不放的点：你的 Φ_approx 没和真 Φ 做过验证。

你现在论文里已经写了"validation on N=6-12 binary units is future work"。但这对独立研究者来说不够——审稿人会想"你既然承认没验证，我凭什么信你的数字"。

**不需要真的跑验证实验。** 但需要在 Limitations 里加一段更具体的讨论：
- 说明为什么没做（真 Φ 对 N>16 的系统就不可计算了，这是 IIT 的已知问题不是你的问题）
- 讨论 Φ_approx 的三个 component（EI, MI_integration, differentiation）各自可能对应真 Φ 的什么部分
- 给出一个 falsifiable prediction：如果 Φ_approx 确实捕获了因果结构，那么在 N=10 的二值系统中 Φ_approx 和真 Φ 的 rank correlation 应 ≥0.8

这样审稿人知道你意识到了问题，有验证计划，而且承认局限性。这比藏着掖着好。

---

## 5. Cover letter

对独立研究者来说，cover letter 可能比论文本身还重要。审稿人的第一反应是"这人没有机构背书，我怎么信"。你的 cover letter 需要直接消解这个顾虑。

建议结构：
1. 第一段：一句话说这篇论文做什么
2. 第二段：诚实说明你是独立研究者，所有代码、数据、实验配置公开在 GitHub 上（附 commit hash），审稿人可以完全重现
3. 第三段：说明这篇论文的贡献不是声称重大发现，而是提供了一个可诊断的计算框架和一套经验约束。明确写"All data and code are open-source (MIT license)"
4. 最后一段：申请 fee waiver（独立研究者、无基金资助）

不要用 ChatGPT 生成的 cover letter 模板。审稿编辑每天都看几十封模板，一眼就能识别。用你自己的话写。

---

## 6. 修改优先级

| 优先级 | 事项 | 工作量 |
|--------|------|--------|
| P0 | 改 Abstract | 30 分钟 |
| P0 | 改 Introduction 叙事定位 | 2 小时 |
| P0 | 写 cover letter | 1 小时 |
| P1 | 补 processor pairwise similarity 分析 | 1 小时 |
| P1 | 用 512 neuron 重跑 calibration | 跑一次实验 |
| P1 | 加强 Limitations 里的 Φ 验证讨论 | 1 小时 |
| P2 | 改 Discussion 措辞 | 2 小时 |
| P2 | 标题考虑微调 | 15 分钟 |

总共大约一天的工作量。不需要新实验，不需要新代码，只改文本。
