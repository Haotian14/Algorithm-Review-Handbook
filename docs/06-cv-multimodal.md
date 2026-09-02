# 06 · CV 与多模态

> CV 岗现在的面试重心已经从"backbone 细节"转向"ViT + 扩散模型 + 多模态大模型"，但经典检测/分割仍是必考。

## 本章高频考点

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
RoI Pooling 有**两次量化取整**（proposal 坐标 → 特征图网格，以及划分 bin），导致特征与原图位置**错位**，对小目标和分割影响大。RoI Align 用**双线性插值**在浮点坐标上采样，不取整，Mask R-CNN 靠这个把 mask AP 提升了约 10%。

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
- **MAE 为什么掩码率能到 75%（远高于 NLP 的 15%）？** 图像信息高度冗余，掩码少了模型可以靠邻域插值蒙混过关，学不到语义。

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
**Latent Diffusion** —— 先用 VAE 把图像压到 $64\times64\times4$ 的隐空间，在隐空间做扩散，计算量降低约 48 倍。条件注入靠 **cross-attention**（文本经 CLIP text encoder 编码后作为 K,V）。

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
