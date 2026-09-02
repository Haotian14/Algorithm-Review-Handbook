# 06 · CV 与多模态

> CV 岗现在的面试重心已经从"backbone 细节"转向"ViT + 扩散模型 + 多模态大模型"，但经典检测/分割仍是必考。

## 本章高频考点

> 星级是**主观判断**，反映常见面经里的印象分布，不是统计数据。用来排复习优先级即可。

| 考点 | 频率 | 典型问法 |
|---|---|---|
| ResNet 的残差 | ★★★ | 为什么能训到 100+ 层 |
| 目标检测两阶段 vs 单阶段 | ★★★ | Faster R-CNN 完整流程 |
| NMS / IoU / Anchor | ★★★ | 手写 NMS 和 IoU |
| Focal Loss | ★★★ | 解决什么问题，公式 |
| ViT | ★★★ | 和 CNN 的归纳偏置差异 |
| 扩散模型 | ★★★ | 前向/反向过程，为什么比 GAN 稳 |
| CLIP | ★★★ | 对比学习目标，零样本怎么做 |
| 多模态大模型（LLaVA/Qwen-VL） | ★★★ | 图像 token 怎么接进 LLM |
| 分割 | ★★ | 语义/实例/全景的区别 |
| 数据增强 | ★★ | 常见增强及其作用 |
| 对抗样本与鲁棒性 | ★★ | FGSM/PGD、对抗训练、为什么会存在 |

---

## 1. 经典 CNN 演进

| 网络 | 关键贡献 |
|---|---|
| **AlexNet** (2012) | ReLU、Dropout、GPU 训练、数据增强 |
| **VGG** (2014) | 堆叠 3×3 小卷积（两个 3×3 感受野=5×5 但参数更少、非线性更多） |
| **GoogLeNet/Inception** | 多尺度并行分支；1×1 卷积降维减少计算 |
| **ResNet** (2015) | 残差连接，解决**退化问题**，能训到 152 层 |
| **DenseNet** | 密集连接，特征复用，参数少但显存占用大 |
| **MobileNet** | 深度可分离卷积，移动端 |
| **ShuffleNet** | 分组卷积 + channel shuffle |
| **SENet** | 通道注意力：Squeeze(GAP) → Excitation(FC-ReLU-FC-Sigmoid) → Scale |
| **EfficientNet** | 复合缩放：深度/宽度/分辨率按固定比例联合放大 |
| **ConvNeXt** (2022) | 用 ViT 的训练技巧和设计（大 kernel、LN、GELU、倒瓶颈）现代化 ResNet，证明 CNN 不弱于 ViT |

**Q：ResNet 为什么用 1×1-3×3-1×1 的 bottleneck？**
先用 1×1 降维（256→64），在低维空间做 3×3 卷积，再用 1×1 升回去。相比直接两个 3×3（256 通道），FLOPs 降低约 8 倍。

**Q：ResNet 的 shortcut 维度不匹配怎么办？** 用 1×1 卷积 + stride 做投影（option B），或补零（option A）。

---

## 2. 目标检测

### 2.1 两阶段：Faster R-CNN 流程（要能完整讲）

1. **Backbone**（ResNet+FPN）提特征图。
2. **RPN**：在特征图每个位置铺 $k$ 个 anchor（3 尺度 × 3 长宽比），分两支 —— 前景/背景二分类 + bbox 回归偏移量。
3. **筛选 Proposal**：按分数排序，NMS 后取 top-N（训练 2000，测试 300）。
4. **RoI Align**：把不同大小的 proposal 池化成固定尺寸（7×7）。
5. **检测头**：分类（C+1 类）+ 精细 bbox 回归。

**Q：RoI Pooling 和 RoI Align 的区别？（必问）**
RoI Pooling 有**两次量化取整**（proposal 坐标 → 特征图网格，以及划分 bin），导致特征与原图位置**错位**，对小目标和分割影响大。RoI Align 用**双线性插值**在浮点坐标上采样，不取整，Mask R-CNN 靠这个显著提升了 mask AP〔Mask R-CNN, ICCV 2017〕。

### 2.2 单阶段

- **YOLO 系列**：直接在特征图上回归，速度快。演进要点：v1 网格回归 → v2 引入 anchor + BN → v3 多尺度 + Darknet53 → v4/v5 大量 trick（Mosaic、CIoU、CSP） → v8/v11 anchor-free + 解耦头 + TAL 标签分配。
- **SSD**：多尺度特征图上做检测。
- **RetinaNet**：FPN + **Focal Loss**，证明单阶段也能达到两阶段精度。
- **FCOS**：anchor-free，逐像素预测到四边的距离 + centerness。
- **DETR**：Transformer + 集合预测 + 匈牙利匹配，**去掉了 NMS 和 anchor**；缺点是收敛慢（500 epoch），Deformable DETR 用可变形注意力把它降到 50 epoch。

### 2.3 Focal Loss（必问）

$$FL(p_t)=-\alpha_t(1-p_t)^{\gamma}\log(p_t)$$

**解决的问题**：单阶段检测器的**前景/背景极度不平衡**（10^4~10^5 个 anchor 中只有几十个正样本）。海量易分负样本（背景，$p_t\to1$）虽然单个 loss 小，但数量太多，累加后主导了梯度。

**机制**：$(1-p_t)^\gamma$ 是调制因子。$p_t$ 大（易分）时该因子趋近 0，loss 被大幅压低；$p_t$ 小（难分）时因子接近 1，loss 基本不变。$\gamma=2$ 时，$p_t=0.9$ 的样本 loss 缩小 100 倍。$\alpha$ 额外平衡正负样本的类别权重。

### 2.4 IoU 家族

- **IoU** = 交/并。缺点：不相交时恒为 0，没有梯度，也无法反映距离远近。
- **GIoU**：$IoU - \frac{|C\setminus(A\cup B)|}{|C|}$（$C$ 是最小外接框），解决不相交问题。
- **DIoU**：加中心点距离惩罚 $\frac{\rho^2(b,b^{gt})}{c^2}$，收敛更快；**DIoU-NMS** 在 NMS 中也考虑中心距离，减少遮挡场景的误删。
- **CIoU**：DIoU + 长宽比一致性项。

### 2.5 NMS 与标签分配

**NMS 流程**：按分数排序 → 取最高分的框 → 删除与它 IoU > 阈值的框 → 重复。代码见 [09](09-coding-ml-scratch.md)。
**问题**：密集遮挡场景会误删真实目标。
**改进**：Soft-NMS（不直接删，按 IoU 衰减分数）、DIoU-NMS、Weighted NMS、Cluster-NMS（可并行）。

**标签分配演进**（近年重点）：
静态（IoU 阈值）→ ATSS（按统计自适应定阈值）→ **SimOTA/TAL**（把分配建模成最优传输/动态 top-k，同时考虑分类分数和 IoU）。核心思想：**哪些 anchor 该当正样本应该由当前模型的预测质量动态决定**，而不是固定几何规则。

---

## 3. 分割

| 任务 | 输出 | 代表 |
|---|---|---|
| 语义分割 | 每像素类别，不分实例 | FCN、U-Net、DeepLab、SegFormer |
| 实例分割 | 每个物体独立 mask | Mask R-CNN、SOLO |
| 全景分割 | 语义 + 实例统一 | Panoptic FPN、Mask2Former |
| 可提示分割 | 点/框/文本提示出 mask | **SAM / SAM2** |

**关键结构**：
- **U-Net**：编码器-解码器 + **skip connection** 融合浅层的高分辨率细节和深层的语义信息。医学图像标配，也是扩散模型的骨干。
- **空洞卷积/ASPP**（DeepLab）：不降分辨率地扩大感受野，多尺度并行。
- **转置卷积 vs 上采样+卷积**：前者有棋盘效应，后者更常用。

**评价指标**：mIoU（各类 IoU 平均）、Dice 系数 $\frac{2|A\cap B|}{|A|+|B|}$（医学常用，等价于 F1）、PQ（全景质量）。

---

## 4. Vision Transformer

### 4.1 ViT 基本结构

图像切成 16×16 patch → 线性投影成 token → 加位置编码 + [CLS] → 标准 Transformer Encoder → [CLS] 接分类头。

**Q：ViT 和 CNN 的核心差异？（必问）**
CNN 有**局部性**和**平移不变性**两个强归纳偏置，ViT 几乎没有（只在 patch 切分时有一点局部性）。
后果：**ViT 需要大数据**（JFT-300M 级别）才能超过 CNN；小数据集上 CNN 更好。解法：强数据增强 + 蒸馏（DeiT）、引入层次结构和局部窗口（Swin）、混合架构。

**Q：Swin Transformer 的贡献？**
① **窗口注意力（W-MSA）**：注意力限制在局部窗口内，复杂度从 $O(n^2)$ 降到 $O(n)$；② **移位窗口（SW-MSA）**：交替移位让窗口间有信息交流；③ **层次化结构**（patch merging 逐层降分辨率），可以像 CNN 一样做检测分割的 backbone。

### 4.2 自监督预训练

- **对比学习**：MoCo（动量编码器 + 队列维护大量负样本）、SimCLR（大 batch + 强增强 + 投影头）、BYOL/SimSiam（**无需负样本**，靠 predictor + stop-gradient 防坍缩）、DINO（自蒸馏，涌现出分割能力）。
- **掩码建模**：**MAE**（掩码 75% 的 patch，只把可见 patch 送编码器，轻量解码器重建像素）、BEiT（重建离散 token）、SimMIM。
- **MAE 为什么掩码率能到 75%（远高于 NLP 的 15%）？**〔MAE, CVPR 2022〕 图像信息高度冗余，掩码少了模型可以靠邻域插值蒙混过关，学不到语义。

---

## 5. 生成模型

### 5.1 三条路线对比

| | GAN | VAE | Diffusion |
|---|---|---|---|
| 原理 | 对抗博弈 | 变分推断 | 逐步去噪 |
| 生成质量 | 高但易模式坍缩 | 模糊 | **最高** |
| 训练稳定性 | 差 | 好 | **好** |
| 采样速度 | 快（一步） | 快 | 慢（多步，可蒸馏加速） |
| 似然估计 | 无 | 有下界 | 有 |

### 5.2 扩散模型（DDPM）

**前向加噪**（固定，无参数）：
$$q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)$$
重参数后可**一步到位**：$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$，其中 $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$。这是训练能高效的关键。

**反向去噪**（学习）：训练一个网络 $\epsilon_\theta(x_t,t)$ 预测加入的噪声，损失极其简单：
$$L=\mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\ t)\|^2\right]$$

**Q：为什么扩散比 GAN 稳定？** 它把"从噪声到图像"这个极难的一步映射拆解成 T 个极简单的去噪步骤，每步是有监督的回归问题，没有对抗博弈，也就没有纳什均衡不收敛和模式坍缩的问题。

**Q：怎么加速采样？** DDIM（确定性非马尔可夫采样，50 步即可）、DPM-Solver（把反向过程视为 ODE 用高阶数值解法，10-20 步）、**一致性模型/蒸馏**（LCM、SDXL-Turbo，1-4 步）。

**Q：Stable Diffusion 的关键改动？**
**Latent Diffusion** —— 先用 VAE 把图像压到 $64\times64\times4$ 的隐空间，在隐空间做扩散，大幅降低计算量〔Latent Diffusion, CVPR 2022〕。条件注入靠 **cross-attention**（文本经 CLIP text encoder 编码后作为 K,V）。

**Q：CFG（Classifier-Free Guidance）是什么？**
训练时按一定概率丢弃条件，让同一个模型既能有条件也能无条件预测；采样时：
$$\tilde\epsilon = \epsilon_\theta(x_t,\varnothing) + w\cdot(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing))$$
$w$ 越大越贴合提示词但多样性下降、易过饱和。

**新趋势**：DiT（用 Transformer 替代 U-Net，可 scale，Sora 的基础）、Flow Matching / Rectified Flow（更直的概率路径，训练更简单采样更快，SD3、Flux 采用）。

---

## 6. 多模态

### 6.1 CLIP（必问）

**训练**：图文对，图像编码器和文本编码器分别编码，在 batch 内做对比学习 —— $N$ 个图文对构成 $N\times N$ 相似度矩阵，对角线是正样本，用对称的 InfoNCE 损失：
$$L=\frac12\left[\text{CE}(\text{logits}, \text{arange}(N)) + \text{CE}(\text{logits}^T,\text{arange}(N))\right]$$
带可学习温度系数 $\tau$。

**零样本分类**：把类别名填进模板 "a photo of a {class}"，编码成文本向量，与图像向量算相似度取最大。

**Q：CLIP 为什么强？** 4 亿图文对的规模 + 自然语言监督（不受限于固定类别集）+ 对比学习目标简单可 scale。

**Q：CLIP 的局限？** 细粒度识别弱、不会数数、对空间关系和否定词理解差、无法生成、对 OCR 类任务依赖于训练数据碰巧覆盖。

### 6.2 多模态大模型（VLM）

主流范式：**视觉编码器 + 连接器（projector）+ LLM**。

- **LLaVA**：CLIP ViT → 一个 MLP → 映射到 LLM 的 token 空间，和文本 token 拼接。两阶段训练：① 冻结视觉和 LLM，只训 projector 做对齐；② 解冻 LLM 做指令微调。**简单但极其有效**。
- **BLIP-2 / Q-Former**：用一组可学习 query 通过交叉注意力从视觉特征中抽取固定数量（如 32 个）的 token，压缩视觉信息，减少 LLM 的负担。
- **Flamingo**：在 LLM 层间插入 gated cross-attention，冻结 LLM 主体。
- **Qwen-VL / InternVL**：支持动态分辨率（把大图切成多个子图 + 缩略图）、原生多语言 OCR。

**Q：图像 token 太多怎么办？** 高分辨率图像会产生几千个 token，挤占上下文且慢。解法：Q-Former 压缩、pixel shuffle / patch merge 降采样、AnyRes 动态切分、token 剪枝（丢弃注意力低的视觉 token）。

**Q：多模态的幻觉？** 模型会描述图中不存在的物体（object hallucination），因为 LLM 的语言先验太强（"厨房里通常有冰箱"）。评测用 POPE；缓解靠更细粒度的对齐数据、负样本训练、视觉对比解码（VCD）。

---

## 7. 数据增强与训练技巧

| 增强 | 作用 |
|---|---|
| 翻转/裁剪/缩放/旋转 | 基础几何不变性 |
| 颜色抖动/灰度 | 光照鲁棒 |
| **Mixup** | 线性插值样本和标签，正则 + 校准 |
| **CutMix** | 区域替换，比 Mixup 更适合分类+定位 |
| **Mosaic** (YOLO) | 四图拼接，丰富小目标和上下文 |
| **RandAugment / AutoAugment** | 自动搜索增强策略 |
| **Copy-Paste** | 实例分割中把物体粘贴到别的图 |
| **随机擦除 / Cutout** | 模拟遮挡 |

**注意**：增强要与任务匹配：数字识别不能上下翻转（6/9）、医学影像不能改变颜色语义、检测任务几何增强要同步变换标注框。

---

## 8. 对抗样本与鲁棒性

**现象**：给图片加上人眼看不出的微小扰动（通常约束 $\|\delta\|_\infty\le\epsilon$，如 $8/255$），
分类器就以高置信度给出完全错误的类别。这不是个别 bug，是深度模型的**系统性脆弱**。

**FGSM（一步攻击）**：沿损失对输入的梯度方向走一步
$$x'=x+\epsilon\cdot\text{sign}(\nabla_x L(f(x),y))$$

**PGD（多步，最强的一阶攻击）**：小步长多次迭代，每步投影回 $\epsilon$-球内
$$x^{t+1}=\Pi_{\|x'-x\|_\infty\le\epsilon}\left(x^{t}+\alpha\cdot\text{sign}(\nabla_x L)\right)$$
随机起点 + 多次重启能进一步增强。**评测防御方法时必须用 PGD 而不是 FGSM**，否则容易得出虚假的鲁棒性。

**Q：为什么会存在对抗样本？**
两种主流解释：① **局部线性假设**〔Goodfellow et al., ICLR 2015〕—— 高维空间中即使每维扰动极小，
沿梯度方向的累积效应也足以翻转 logits；② **非鲁棒特征**〔Ilyas et al., NeurIPS 2019〕—— 数据里本就存在
人类看不见但**真实可泛化**的高频统计特征，模型合理地利用了它们，攻击正是在操纵这些特征。
第二种解释更被接受：**对抗样本是特征，不是 bug。**

**防御**：

| 方法 | 说明 |
|---|---|
| **对抗训练** | 训练时就用 PGD 生成的对抗样本，本质是 min-max 优化 $\min_\theta\mathbb{E}\max_{\|\delta\|\le\epsilon}L$〔Madry et al., ICLR 2018〕。**目前唯一被广泛验证有效的方法**，代价是训练慢几倍、干净样本准确率下降 |
| 随机平滑 | 加高斯噪声后多次投票，能给出**可证明的**鲁棒半径，但半径小、推理贵 |
| 输入变换/去噪 | JPEG 压缩、随机缩放等。**大多在自适应攻击下失效** —— 只是让梯度不好算（梯度混淆），不是真鲁棒 |

**Q：鲁棒性和准确率有冲突吗？** 有。对抗训练通常会掉几个点的干净准确率，
因为它强迫模型放弃那些"有用但不鲁棒"的特征。这是一个**要按场景取舍**的 trade-off，不是纯粹的改进。

**业务里的"对抗"往往更土也更常见**：内容审核场景中，对手不是在算梯度，
而是在图上加噪点、拉伸变形、把违规文字写成变体字或嵌进图片。
防御靠的是**数据增强覆盖已知变体 + 快速迭代的对抗闭环（人审产出新标签回流训练）**，
而不是学术的 PGD 对抗训练。**面试答这题时能区分"学术对抗"和"业务对抗"是加分项。**


---

## 延伸阅读

- *Deep Residual Learning*（He et al., CVPR 2016）/ *Densely Connected Convolutional Networks*（Huang et al., CVPR 2017）
- *Squeeze-and-Excitation Networks*（Hu et al., CVPR 2018）
- *A ConvNet for the 2020s*（Liu et al., CVPR 2022）—— ConvNeXt
- *Faster R-CNN*（Ren et al., NeurIPS 2015）/ *Mask R-CNN*（He et al., ICCV 2017）—— RoI Align 的量化误差分析
- *Focal Loss for Dense Object Detection*（Lin et al., ICCV 2017）—— RetinaNet
- *Distance-IoU Loss*（Zheng et al., AAAI 2020）—— DIoU 与 CIoU
- *End-to-End Object Detection with Transformers*（Carion et al., ECCV 2020）—— DETR
- *Bridging the Gap Between Anchor-based and Anchor-free Detection (ATSS)*（Zhang et al., CVPR 2020）
- *U-Net*（Ronneberger et al., MICCAI 2015）/ *Segment Anything*（Kirillov et al., ICCV 2023）
- *An Image is Worth 16x16 Words*（Dosovitskiy et al., ICLR 2021）—— ViT
- *Swin Transformer*（Liu et al., ICCV 2021）
- *Masked Autoencoders Are Scalable Vision Learners*（He et al., CVPR 2022）—— 75% 掩码率的实验依据
- *Denoising Diffusion Probabilistic Models*（Ho et al., NeurIPS 2020）
- *Denoising Diffusion Implicit Models*（Song et al., ICLR 2021）—— DDIM
- *High-Resolution Image Synthesis with Latent Diffusion Models*（Rombach et al., CVPR 2022）—— Stable Diffusion，隐空间压缩倍数的出处
- *Classifier-Free Diffusion Guidance*（Ho & Salimans, 2022）
- *Scalable Diffusion Models with Transformers*（Peebles & Xie, ICCV 2023）—— DiT
- *Learning Transferable Visual Models From Natural Language Supervision*（Radford et al., ICML 2021）—— CLIP
- *Visual Instruction Tuning*（Liu et al., NeurIPS 2023）—— LLaVA 两阶段训练
- *BLIP-2*（Li et al., ICML 2023）—— Q-Former
- *Evaluating Object Hallucination in Large Vision-Language Models*（Li et al., EMNLP 2023）—— POPE
- *Explaining and Harnessing Adversarial Examples*（Goodfellow et al., ICLR 2015）—— FGSM 与线性假设
- *Towards Deep Learning Models Resistant to Adversarial Attacks*（Madry et al., ICLR 2018）—— PGD 与对抗训练的 min-max 框架
- *Adversarial Examples Are Not Bugs, They Are Features*（Ilyas et al., NeurIPS 2019）—— 非鲁棒特征

---

## 自测清单

- [ ] 能完整讲 Faster R-CNN 的五个步骤
- [ ] 能说清 RoI Pooling 和 RoI Align 的量化误差问题
- [ ] 能写 Focal Loss 公式并解释调制因子
- [ ] 能手写 IoU 和 NMS
- [ ] 能说清 GIoU/DIoU/CIoU 各解决什么
- [ ] 能讲 ViT 缺少归纳偏置的后果和 Swin 的三个改进
- [ ] 能写 DDPM 的前向公式和训练损失
- [ ] 能解释 Latent Diffusion 和 CFG
- [ ] 能讲 CLIP 的对比学习目标和零样本流程
- [ ] 能讲 LLaVA 的两阶段训练和图像 token 过多的解法
- [ ] 能写 FGSM/PGD，并说清为什么评测防御要用 PGD
- [ ] 能解释对抗样本的"非鲁棒特征"视角和鲁棒性-准确率的取舍
