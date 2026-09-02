# 03 · 深度学习基础

> 这一章决定你能不能进"深度学习相关的组"。反向传播、归一化、优化器、梯度问题是四个必考区，Transformer 放在 [04 大模型](04-llm.md)。

## 本章高频考点

> 星级是**主观判断**，反映常见面经里的印象分布，不是统计数据。用来排复习优先级即可。

| 考点 | 频率 | 典型问法 |
|---|---|---|
| 反向传播推导 | ★★★ | 手推一个两层网络 |
| 梯度消失/爆炸 | ★★★ | 原因 + 5 种解法 |
| BN / LN / RMSNorm | ★★★ | 为什么 NLP 用 LN 不用 BN |
| 激活函数对比 | ★★★ | ReLU 死亡问题、GELU/SwiGLU |
| 优化器 Adam/AdamW | ★★★ | Adam 公式；AdamW 和 L2 的区别 |
| Dropout | ★★★ | 训练/推理不一致怎么处理 |
| 参数初始化 | ★★ | Xavier/He 的推导思想 |
| 学习率调度 + warmup | ★★★ | 为什么需要 warmup |
| 残差连接 | ★★★ | 为什么能解决退化问题 |

---

## 1. 反向传播

### 1.1 核心就是链式法则

对两层网络 $z_1=W_1x+b_1,\ a_1=\sigma(z_1),\ z_2=W_2a_1+b_2,\ \hat y=\text{softmax}(z_2)$，交叉熵损失 $L$：

$$\delta_2 = \frac{\partial L}{\partial z_2}=\hat y - y$$
$$\frac{\partial L}{\partial W_2}=\delta_2 a_1^T,\quad \frac{\partial L}{\partial b_2}=\delta_2$$
$$\delta_1=\frac{\partial L}{\partial z_1}=(W_2^T\delta_2)\odot\sigma'(z_1)$$
$$\frac{\partial L}{\partial W_1}=\delta_1 x^T$$

**Softmax + 交叉熵的梯度就是 $\hat y - y$，这个结论必须记住**（和 LR 一致，指数族的性质）。

**追问：Softmax 的数值稳定性怎么处理？**
减去最大值：$\text{softmax}(x_i)=\frac{e^{x_i-\max x}}{\sum_j e^{x_j-\max x}}$，防止 $e^{x}$ 上溢。实现中还会用 `log_softmax` 避免先 exp 再 log。

### 1.2 计算图与自动微分

- 前向：构建计算图，缓存中间激活值（这是显存的大头）。
- 反向：拓扑逆序累加梯度。
- **梯度检查点（gradient checkpointing）**：不缓存全部激活，反向时重算，用约 30% 的额外时间换 $O(\sqrt{n})$ 的显存〔Chen et al., 2016〕。大模型训练必用。

---

## 2. 激活函数

| 激活 | 表达式 | 优点 | 缺点 |
|---|---|---|---|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | 有概率含义 | 梯度消失（最大导数 0.25）、非零均值 |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | 零均值 | 仍梯度消失 |
| ReLU | $\max(0,x)$ | 计算快、正区间梯度恒为 1、稀疏激活 | **神经元死亡**、非零均值 |
| LeakyReLU | $\max(\alpha x,x)$ | 解决死亡 | 多一个超参 |
| ELU / SELU | 负区间指数 | 零均值、自归一化 | 慢 |
| **GELU** | $x\cdot\Phi(x)$ | 平滑、BERT/GPT 标配 | 略慢 |
| **SwiGLU** | $(\text{Swish}(xW)\odot xV)W_2$ | LLaMA 系标配，效果最好 | 参数量 ×1.5（所以 FFN 隐层取 $\frac83 d$ 补偿） |

**ReLU 神经元死亡**：某次大梯度更新后使 $w^Tx+b<0$ 对所有输入成立 → 梯度永远为 0 → 该神经元再也不更新。解法：小学习率、LeakyReLU、正确初始化。

**为什么需要非线性激活？** 否则多层线性变换的复合仍是线性变换，$W_2W_1x = Wx$，深度失去意义。

---

## 3. 归一化（极高频）

### 3.1 各种 Norm 的归一化维度

设输入形状 $(N, C, H, W)$ 或 NLP 的 $(B, T, D)$：

| Norm | 归一化维度 | 场景 |
|---|---|---|
| BatchNorm | 对每个通道 C，在 (N,H,W) 上算均值方差 | CV |
| LayerNorm | 对每个样本，在 (C,H,W) 或 (D) 上算 | NLP/Transformer |
| InstanceNorm | 对每个样本每个通道，在 (H,W) 上算 | 风格迁移 |
| GroupNorm | 通道分组，组内 (H,W) 上算 | 小 batch 的检测/分割 |
| RMSNorm | 只除以 RMS，不减均值，无 $\beta$ | LLaMA 等大模型，省 7-10% 计算 |

### 3.2 BatchNorm

$$\hat x = \frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\quad y=\gamma\hat x+\beta$$

**作用**：① 缓解内部协变量偏移（原论文说法，后来被质疑）；② **真正的作用是让损失面更平滑（Lipschitz 常数更小），允许更大学习率**〔Santurkar et al., NeurIPS 2018〕；③ 有轻微正则效果（batch 内的噪声）。

**训练 vs 推理**：训练用当前 batch 统计量并用动量更新 running_mean/var；推理用 running 统计量。**这是 BN 最容易出 bug 的地方 —— 忘记 `model.eval()`**。

**BN 的缺点**：
- batch 小时统计量不准（检测/分割任务 batch 常为 2-4）→ 用 GN。
- 序列长度可变时，不同位置的 batch 内样本数不同 → NLP 不用 BN。
- 训练/推理不一致；分布式训练需要 SyncBN。

**Q：为什么 NLP 用 LayerNorm 不用 BatchNorm？（必问）**
① 序列长度可变，padding 位置会污染 batch 统计量；② NLP 中同一维度在不同样本间不具有"同类特征"的语义（CV 中同一通道是同一种卷积核的响应，可比；NLP 中 batch 内不同句子的第 t 个词毫无关系）；③ 推理时 batch 可能为 1；④ LN 对每个样本独立，天然适合自回归解码。

**Q：BN 放在激活前还是后？** 原论文是 conv → BN → ReLU，实践中也这么用。

**Q：有 BN 时还需要 bias 吗？** 不需要。BN 会减去均值，bias 被消掉，所以 `conv(bias=False)` + BN 是标准写法。

### 3.3 Pre-LN vs Post-LN（Transformer 必问）

- **Post-LN**（原始 Transformer）：`x + Sublayer(x)` 再 LN。梯度需穿过 LN，深层训练不稳定，**必须配 warmup**。
- **Pre-LN**（GPT-2 之后主流）：`x + Sublayer(LN(x))`。残差路径是干净的恒等映射，梯度可直达底层，训练稳定，可以不用 warmup 或用很短的 warmup。代价是最终效果略差一点点，所以有 DeepNorm、Sandwich-LN 等折中方案。

---

## 4. 优化器

### 4.1 演进脉络

| 优化器 | 核心思想 |
|---|---|
| SGD | $\theta \leftarrow \theta - \eta g$ |
| Momentum | $v=\beta v+g$，累积历史梯度，抑制震荡、冲过小坑 |
| Nesterov | 先按动量走一步再算梯度，"前瞻" |
| AdaGrad | $\eta/\sqrt{\sum g^2}$，稀疏特征友好；但分母单调增，后期学不动 |
| RMSProp | 用指数移动平均代替累加，解决 AdaGrad 学习率衰减过快 |
| **Adam** | Momentum + RMSProp + 偏差修正 |
| **AdamW** | 把权重衰减从梯度里拿出来，直接作用于参数 |

### 4.2 Adam 公式（要能默写）

$$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$$
$$\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t}$$
$$\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

默认 $\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$。大模型常用 $\beta_2=0.95$（对梯度尖峰更鲁棒）。

**偏差修正为什么需要？** $m_0=0$ 导致前期 $m_t$ 被严重低估（偏向 0），除以 $1-\beta_1^t$ 校正。$t$ 大时该因子趋近 1，自动失效。

### 4.3 AdamW vs Adam + L2（高频）

Adam + L2：把 $\lambda\theta$ 加到梯度里 → 它会被 $\sqrt{\hat v_t}$ 自适应缩放 → **梯度大的参数得到的衰减反而小**，正则效果被扭曲。

AdamW（解耦权重衰减）：
$$\theta_t = \theta_{t-1}-\eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\theta_{t-1}\right)$$
衰减项不经过自适应缩放。**现在训练 Transformer 一律用 AdamW。**

**Q：Adam 效果好为什么 CV 里还常用 SGD+Momentum？**
Adam 收敛快但常收敛到 sharp minima，泛化略差；SGD 调好了最终精度更高。实践中的折中：前期 Adam 后期切 SGD，或直接用 AdamW + cosine 调度。

**Q：Adam 的显存开销？** 每个参数需要额外存 $m$ 和 $v$（fp32），即 8 bytes/参数；加上 fp32 主权重和梯度，混合精度训练下约 **16 bytes/参数**。7B 模型全参微调 ≈ 112GB 优化器+参数状态，这就是为什么需要 ZeRO / LoRA（见 [10 工程](10-engineering.md)）。

### 4.4 学习率调度

- **Warmup**：从 0 线性升到峰值 lr（常见 2000-10000 步或总步数的 1-5%）。
  **为什么需要？** ① 训练初期参数随机，梯度方向噪声大，大 lr 会把参数带偏；② Adam 前期 $\hat v$ 估计不准，自适应步长方差大；③ Post-LN 结构初期梯度尺度失衡。
- **Cosine 退火**：$\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos(\frac{t}{T}\pi))$。目前最主流。
- 其他：Step decay、多项式衰减、ReduceLROnPlateau、One-Cycle。

### 4.5 梯度裁剪

按范数裁剪：若 $\|g\|>c$ 则 $g\leftarrow c\cdot\frac{g}{\|g\|}$。大模型训练标配（常取 1.0），防止某个坏 batch 直接把模型训崩。

---

## 5. 梯度问题（必问）

### 5.1 梯度消失/爆炸的原因

反向传播是连乘：$\frac{\partial L}{\partial W_1}\propto \prod_{l} W_l^T\text{diag}(\sigma'(z_l))$。
- 若每项 <1，连乘后指数级趋近 0 → **梯度消失**。
- 若每项 >1，指数级增长 → **梯度爆炸**。

### 5.2 解决方案（分类记忆）

| 层面 | 手段 |
|---|---|
| 激活函数 | 用 ReLU 系（正区间导数恒为 1）代替 Sigmoid/Tanh |
| 结构 | **残差连接**、DenseNet、LSTM 的门控（加性更新代替连乘） |
| 归一化 | BN/LN 把每层输入拉回稳定分布 |
| 初始化 | Xavier / He，让每层输出方差保持一致 |
| 训练技巧 | 梯度裁剪（治爆炸）、warmup、更小学习率 |

### 5.3 残差连接为什么有效（必问）

$y = x + F(x)$，则 $\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x}$。**梯度里有一个恒等项 $I$，保证梯度至少能原样传回去，不会因连乘归零。**

另一个视角：ResNet 解决的是**退化问题**（深层网络训练误差反而更高），而不只是梯度消失（那个 BN 已经缓解了）。恒等映射让"多加的层至少能学成什么都不做"，保证深模型不劣于浅模型。

---

## 6. 正则化

### 6.1 Dropout

训练时以概率 $p$ 置零神经元。**推理时不 drop**，为保证期望一致有两种做法：
- 原始版：推理时输出 ×$(1-p)$。
- **Inverted Dropout**（现在的实现）：训练时输出除以 $(1-p)$，推理时什么都不做。

**Dropout 为什么有效？** ① 防止神经元共适应；② 相当于训练了 $2^n$ 个子网络的集成；③ 等价于给权重加噪声，是一种自适应正则。

**Q：Dropout 和 BN 一起用会怎样？**
会冲突。Dropout 改变了神经元输出的方差，训练时 BN 统计的是 dropout 后的方差，推理时 dropout 关闭方差变了 → **方差偏移**。解法：把 Dropout 放在所有 BN 之后（网络末端），或只用其一。Transformer 用 LN 所以没这个问题。

**Q：Dropout 在 Transformer 里加在哪？** attention 权重上、FFN 中间、每个子层输出（残差相加前）。

### 6.2 其他正则

- **Label Smoothing**：$y_{soft}=(1-\epsilon)y+\epsilon/K$。防止模型对预测过度自信，改善校准；但会损害知识蒸馏的教师模型质量。
- **Mixup**：$\tilde x=\lambda x_i+(1-\lambda)x_j$，标签同样混合。**CutMix**：区域替换。
- **Stochastic Depth**：随机丢弃整层（DropPath），ViT/ConvNeXt 标配。
- **Early Stopping**：监控验证集，patience 轮不提升就停。
- **权重衰减**：见 AdamW。

---

## 7. 参数初始化

**为什么不能全 0？** 所有神经元对称，梯度相同，永远学到一样的东西（对称性无法打破）。LR 可以全 0 因为只有一层。

**为什么不能太大/太小？** 太大 → 激活值饱和/爆炸；太小 → 信号逐层衰减为 0。

- **Xavier/Glorot**：$\text{Var}(W)=\frac{2}{n_{in}+n_{out}}$。假设激活是线性/tanh 的，让前向后向的方差都保持。
- **He/Kaiming**：$\text{Var}(W)=\frac{2}{n_{in}}$。针对 ReLU —— ReLU 砍掉一半，方差减半，所以分子是 2。
- Transformer 常用 $\mathcal{N}(0, 0.02)$，并对残差分支的输出投影按 $1/\sqrt{2L}$ 缩放（GPT-2 做法），防止深层残差累积方差。

---

## 8. CNN 基础（放这里，CV 细节见 [06](06-cv-multimodal.md)）

### 8.1 核心概念

- **参数量**：$k\times k\times C_{in}\times C_{out} + C_{out}$
- **FLOPs**：$\approx 2\times k^2\times C_{in}\times C_{out}\times H_{out}\times W_{out}$
  （这里把一次乘加算作 2 次浮点运算。**很多论文报的是 MACs / MAdds，不乘这个 2**，相差一倍 —— 面试报数字时先说清用的是哪种口径）
- **输出尺寸**：$H_{out}=\lfloor\frac{H+2p-k}{s}\rfloor+1$
- **感受野**：$RF_l = RF_{l-1} + (k_l-1)\prod_{i<l}s_i$

### 8.2 卷积的两大先验

**局部连接**（局部相关性）和 **权值共享**（平移不变性）。相比全连接，参数量大幅下降且有归纳偏置 —— 这也是小数据集上 CNN 优于 ViT 的原因。

### 8.3 常见变体

- **1×1 卷积**：跨通道信息融合 + 升降维（bottleneck 结构的关键，大幅降 FLOPs）。
- **深度可分离卷积**（MobileNet）：Depthwise（每通道独立卷积）+ Pointwise（1×1）。计算量降为 $\frac{1}{C_{out}}+\frac{1}{k^2}$，约 1/8~1/9。
- **空洞卷积**：不增加参数扩大感受野，用于分割。
- **转置卷积**：上采样，注意棋盘效应（可改用 上采样+卷积）。

### 8.4 池化

Max Pooling 保留最强响应（纹理），Average Pooling 保留整体（背景）。**Global Average Pooling** 替代全连接层，大幅减参数且强制通道与类别对应（CAM 的基础）。

---

## 9. 调参与训练排查（面试爱问的实操题）

**Q：模型不收敛/loss 是 NaN 怎么排查？**
1. 先用极小数据集（如 8 条）过拟合 —— 如果连这都学不会，说明代码有 bug（标签错位、忘记 zero_grad、loss 写错）。
2. 检查学习率是否过大 → 降 10 倍试。
3. 检查数据：是否有 NaN/Inf、归一化是否做了、标签范围是否正确。
4. NaN 常见来源：`log(0)`（加 eps）、除零、fp16 溢出（用 loss scaling 或 bf16）、学习率过大导致爆炸。
5. 加梯度裁剪，打印梯度范数看是否爆炸。
6. 检查是否忘了 `optimizer.zero_grad()` 或 `model.train()`。

**Q：训练 loss 下降但验证 loss 上升？** 过拟合，见 [02 §7.1](02-machine-learning.md)。

**Q：训练 loss 一直不降？** 欠拟合或有 bug：学习率太小、模型容量不够、特征无信息、优化器配置错误、数据没 shuffle。

**Q：训练集和验证集 loss 都很低但线上效果差？** 分布偏移、数据泄漏、线上线下特征不一致（见 [02 §7.4](02-machine-learning.md)）。

---

## 延伸阅读

- *Deep Learning*（Goodfellow, Bengio, Courville, 2016）—— 反向传播、正则化、优化的系统参考
- *Batch Normalization*（Ioffe & Szegedy, ICML 2015）与 *How Does Batch Normalization Help Optimization?*（Santurkar et al., NeurIPS 2018）—— 后者推翻了「内部协变量偏移」的解释，改为损失面平滑
- *Layer Normalization*（Ba, Kiros, Hinton, 2016）
- *Root Mean Square Layer Normalization*（Zhang & Sennrich, NeurIPS 2019）—— RMSNorm
- *Adam: A Method for Stochastic Optimization*（Kingma & Ba, ICLR 2015）
- *Decoupled Weight Decay Regularization*（Loshchilov & Hutter, ICLR 2019）—— AdamW 与 Adam+L2 的区别
- *Deep Residual Learning for Image Recognition*（He et al., CVPR 2016）—— 残差与退化问题
- *Identity Mappings in Deep Residual Networks*（He et al., ECCV 2016）—— Pre-activation，Pre-LN 思路的来源
- *On Layer Normalization in the Transformer Architecture*（Xiong et al., ICML 2020）—— Pre-LN 为什么可以不用 warmup
- *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*（Srivastava et al., JMLR 2014）
- *Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift*（Li et al., CVPR 2019）—— 方差偏移
- *Delving Deep into Rectifiers*（He et al., ICCV 2015）—— He 初始化
- *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*（Keskar et al., ICLR 2017）
- *Accurate, Large Minibatch SGD*（Goyal et al., 2017）—— 学习率线性缩放与 warmup

---

## 自测清单

- [ ] 能手推两层网络的反向传播
- [ ] 能说出 softmax+CE 的梯度是 $\hat y-y$ 并解释为什么
- [ ] 能讲清 BN 的训练/推理差异，和为什么 NLP 用 LN
- [ ] 能默写 Adam 公式，说清 AdamW 与 Adam+L2 的区别
- [ ] 能解释 warmup 的三个动机
- [ ] 能从五个层面回答梯度消失
- [ ] 能解释残差连接为什么有效（梯度视角 + 退化问题视角）
- [ ] 知道 Inverted Dropout 和 Dropout+BN 的冲突
- [ ] 能算卷积的参数量、FLOPs、输出尺寸、感受野
- [ ] 有一套 loss 变 NaN 的排查流程
