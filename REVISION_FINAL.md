# 投稿前必须修改的问题

---

## 问题1（致命）：编造了不存在的"同行评审"历史

论文中 **7 处**提到"peer review"和"anonymous reviewers at Entropy"，但文章从未投过 Entropy，不存在审稿人。真投出去，编辑或审稿人一眼能识别这是虚假陈述，可能直接 desk reject。

### 需修改位置

**第263行** — `added in response to peer review`
→ 改为 `added as methodological controls`

**第463-469行** — `During peer review, two implementation issues were identified and corrected before the results presented here were obtained (see Section~4.4 for details): (1) coalition formation used...`
→ 改为 `During development, two implementation issues were identified and corrected before the results presented here were obtained (see Section~4.4 for details): (1) coalition formation used...`

**第582行** — `Two implementation issues were discovered during peer review and corrected`
→ `Two implementation issues were discovered during development and corrected`

**第740行** — `The two implementation issues corrected during peer review were discovered`
→ `The two implementation issues corrected during development were discovered`

**第845行** — `during peer review`
→ `during development`

**第890行** — `corrected during peer review`
→ `corrected during development`

**第928-936行** — 整个 Acknowledgments 段落需要重写。原文感谢了不存在的"anonymous reviewers at Entropy"。

建议改为：
```latex
\noindent\textbf{Acknowledgments:} The author thanks the developers
of PyTorch, CuPy, NumPy, and SciPy for the open-source tools that
made this work possible. The development of the diagnostic toolkit
benefited substantially from systematic internal validation and
debugging, during which two non-trivial implementation issues were
identified and corrected.
```

---

## 问题2（重要）：上下文注入权重公式写错了

**第254-255行**：
```
$h_0' = 0.7 \cdot h_0 + 0.3 \cdot \text{ws}$
```

这是修 bug 之前的旧权重。代码（`neural_base.py:155-156`）中实际是：
```python
alpha = float(os.environ.get("NOESIS_CTX_WEIGHT", "0.6"))
h = (1.0 - alpha) * h + alpha * cv
```

即 `alpha = 0.6`，`h_new = 0.4 * h_old + 0.6 * ws`。

**改为**：`$h_0' = 0.4 \cdot h_0 + 0.6 \cdot \text{ws}$`

---

## 问题3（小）：表1数值四舍五入不一致

| 模式 | 实际值 | 表里写 | 应改为 |
|------|--------|--------|--------|
| Random | 1.049234 | 1.046 | 1.049 |
| Adaptive | 1.049182 | 1.046 | 1.049 |

对应的 `std_phi_after` 也需检查（实际 0.193/0.194，表里写 0.192，应统一）。

---

## 问题4（小）：LaTeX 格式问题

**第804-808行**（Limitations 第4条）：
- `\text{rec}$` → 缺少 `$` 或括号，应为 `$W_\text{rec}$`
- `\in \{15, 20, 25, 30, 35\}$` → 缺少 `$` 开头，应为 `$T \in \{15, 20, 25, 30, 35\}$`

---

## 优先级

| 优先级 | 事项 | 工作量 |
|--------|------|--------|
| P0 | 删除/替换所有"peer review"和"anonymous reviewers" | 15分钟 |
| P0 | 修正上下文注入权重公式 | 1分钟 |
| P2 | 统一表1数值 | 5分钟 |
| P2 | 修 LaTeX 格式问题 | 5分钟 |

**不需要重跑实验，纯文本修改，约30分钟。**
