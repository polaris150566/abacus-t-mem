# ControlNet + CFG 用于 WY ring 生成的方案设计

## 背景与动机

### 根本问题

当前训练目标（序列恢复率）与真实需求（膜蛋白界面 WY ring 的热力学正确性）存在根本性错位：

- native 序列不是唯一正确答案，恢复率高不等于设计好
- 训练数据里 WY ring 只是软热力学偏好（约 4 kcal/mol），不是硬约束
- 模型无法超越训练分布的天花板，加再多 WY loss 权重也只是在表面抬指标

### 为什么之前的注入方案失败

以 concat 为例：concat_proj 权重训练后趋近于零（mean abs diff < 0.002），
模型学会了"绕过"膜信息，因为不用它照样能优化 loss。
adaLN 的 mem_norm 完全没有更新（weight=1, bias=0），同样被旁路。

---

## 核心思路

### ControlNet 学习偏置

不让模型从头学如何在 WY ring 位置生成 WY，而是让一个独立的可训练分支
专门学习"在 interface 位置，相对于无条件预测，W/Y 的 logit 应该偏移多少"。

### CFG 在推理阶段放大效应

训练时同时学有条件和无条件预测，推理时：

    logits_final = logits_uncond + scale * (logits_cond - logits_uncond)

scale > 1 可以把条件方向的偏置放大到超出训练分布，突破数据天花板。

---

## 架构设计

### 条件信号

使用已有的 3-state fasta_mem mask：
- 0 = aqueous（水溶液区）
- 1 = interface（膜-水界面，WY ring 所在位置）
- 2 = core（膜疏水核心区）

将其 one-hot 编码为 3 维向量，作为额外节点特征注入 ControlNet 分支输入。

### ControlNet 分支结构

```
原始 abacust（完全冻结）：
  prot_encoder_pifold → encoder_layers[0,1,2] → decoder_layers → logits

ControlNet 可训练分支：
  (结构特征 + 3-state mask one-hot) 
    → 复制的 encoder_layers[0,1,2]（可训练，从冻结参数初始化）
    → 每层输出经过 zero-init Linear
    → 加回冻结主干对应 encoder_layers 的输出
```

零初始化 Linear 保证：
- 训练初期分支输出严格为零，冻结主干行为不变
- 梯度只能让偏置从零往上学，无法退化回零

### 参数量考虑

只复制 encoder_layers（3层），不复制 prot_encoder_pifold，
参数量增加较小，训练速度影响可控。

---

## 训练方案

### CFG dropout

训练时以 p=0.15 的概率将 mask 置零（null condition），
强制模型同时学习：
- 有条件：知道哪里是 interface，在那里偏向 W/Y
- 无条件：正常序列设计，不知道膜位置

### WY loss

保留 W/Y loss 权重（建议 2.0），为 ControlNet 分支提供明确的训练信号方向。
没有这个信号，分支不知道"偏向 W/Y"是对的方向。

### 从 noise_650M 开始训练

和之前所有版本保持一致，从 checkpoint_best_noise650M.pt 初始化，
冻结部分保持不动，只有 ControlNet 分支可训练。

---

## 推理方案

跑两次前向：
1. 用真实 3-state mask → logits_cond
2. 用 null mask（全零）→ logits_uncond

合并：
    logits_final = logits_uncond + scale * (logits_cond - logits_uncond)

scale 建议从 2.0 开始，可以尝试 3.0~5.0 看 WY 比例变化。

---

## 为什么这比之前方案更合理

| 问题 | 之前方案 | 本方案 |
|------|---------|--------|
| 条件信号被绕过 | concat_proj/adaLN 可以趋近零 | zero-init 保证无法退化 |
| 数据天花板 | WY loss 抬指标但受分布限制 | CFG scale > 1 可超出训练分布 |
| 梯度传播 | 膜信号注入点梯度稀释 | 独立分支专门优化条件偏置 |
| 预训练知识保留 | finetune 可能遗忘 | 主干完全冻结 |

---

## 潜在风险

1. scale 过大时可能导致非 WY ring 位置也被强行推向 W/Y，需要 mask 精确
2. ControlNet 分支的训练信号强度仍受数据分布限制，只是比之前更能被利用
3. 推理时跑两次前向，速度减半（可接受）

---

## 后续可能的扩展

- 条件信号从 3-state mask 换成逐残基 Wimley-White 界面转移能（更有物理意义）
- 结合 MCMC resampling：用 CFG 的 logits 差值作为能量函数做后验采样
- 两阶段：先用 ControlNet 生成候选，再用膜能量函数筛选

---

## 参考

- ControlNet 原论文：Adding Conditional Control to Text-to-Image Diffusion Models (Zhang et al., 2023)
- Wimley-White 界面疏水性标度：Wimley & White, Nature Structural Biology, 1996
- WY ring 物理机制：芳香环与磷脂头基氢键 + 与胆碱的阳离子-π 相互作用，约 4 kcal/mol
- 当前训练数据：0124_cluster_dict_b64，膜蛋白结构来自 PDB + TMDET 标注
