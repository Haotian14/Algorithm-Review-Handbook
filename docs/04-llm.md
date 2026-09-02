# 04 · 大模型 / LLM

> 目前算法岗最热的方向。校招面试的深度通常到"能讲清楚原理 + 知道主流方案的取舍"，不要求你训过 100B 模型，但**Transformer 的每一个细节都可能被追到底**。

## 本章高频考点

> 星级是**主观判断**，反映常见面经里的印象分布，不是统计数据。用来排复习优先级即可。

| 考点 | 频率 | 典型问法 |
|---|---|---|
| Self-Attention 原理 + 手写 | ★★★ | 为什么除以 $\sqrt{d_k}$ |
| MHA / MQA / GQA / MLA | ★★★ | KV Cache 显存怎么算 |
| 位置编码：绝对/RoPE/ALiBi | ★★★ | RoPE 怎么外推 |
| Encoder-only vs Decoder-only | ★★★ | 为什么现在都用 Decoder-only |
| 预训练 → SFT → RLHF/DPO | ★★★ | PPO 四个模型是什么 |
| LoRA / QLoRA / PEFT | ★★★ | LoRA 为什么有效，秩怎么选 |
| KV Cache / PagedAttention / continuous batching | ★★★ | 推理为什么慢 |
| 量化 / 蒸馏 / 投机解码 | ★★ | INT8 vs INT4 vs FP8 |
| RAG | ★★★ | 检索效果差怎么优化 |
| Agent / Function Calling | ★★ | ReAct 流程 |
| 幻觉与评测 | ★★ | 怎么评价一个 LLM |
| MoE | ★★ | 路由怎么做，负载均衡 |

---

## 1. Transformer 架构

### 1.1 Self-Attention

$$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Q：为什么除以 $\sqrt{d_k}$？（几乎必问）**
假设 $q,k$ 各分量独立、均值 0 方差 1，则 $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ 的方差是 $d_k$。$d_k$ 大时点积数值大 → softmax 进入饱和区 → 输出接近 one-hot → **梯度接近 0**。除以 $\sqrt{d_k}$ 把方差拉回 1。

**Q：为什么要 Q、K、V 三个不同的矩阵？为什么不能 Q=K？**
Q=K 会让注意力矩阵对称，且每个 token 对自己的注意力必然最大（自己和自己内积最大），削弱表达能力。分开投影让"我在找什么"（Q）和"我能提供什么"（K）解耦，V 则承载真正传递的信息。

**Q：多头注意力的作用？**
在不同子空间并行捕捉不同的关系模式（语法/指代/位置等）。总参数量与单头 $d$ 维相同（每头 $d/h$），是"分组"不是"加倍"。可类比 CNN 的多通道。

**Q：Attention 的复杂度？**
时间/空间都是 $O(n^2d)$（$n$ 序列长度）。这是长上下文的核心瓶颈。
优化方向：稀疏注意力（Longformer/BigBird）、线性注意力（Linformer/Performer）、**FlashAttention**（不降低复杂度，但用分块 + 在线 softmax 避免实例化 $n\times n$ 矩阵，把 HBM 读写从 $O(n^2)$ 降到 $O(n^2/M)$，是 **IO 感知**的精确注意力，显存降为线性）。

**Q：Attention 里的 mask 有哪几种？**
① **Padding mask**：屏蔽 pad 位置（把 logits 置 $-\infty$）；② **Causal mask**（下三角）：Decoder 防止看到未来；③ Prefix LM mask：前缀双向、生成部分单向（GLM/UniLM）。

### 1.2 完整 Transformer Block（Pre-LN 版）

```
x = x + Attention(LN(x))
x = x + FFN(LN(x))
```

FFN：$\text{FFN}(x)=W_2\,\text{act}(W_1x)$，隐层维度通常 $4d$。LLaMA 用 SwiGLU：$W_2(\text{Swish}(W_1x)\odot W_3x)$，隐层取 $\frac83 d$ 保持参数量相当。

**Q：FFN 的作用？** ① 提供非线性；② 参数量的大头（约占 2/3），被认为是模型的"知识存储"（key-value memory 假说）；③ Attention 做 token 间混合，FFN 做通道间混合。

### 1.3 三种架构对比

| 架构 | 代表 | 注意力 | 适合 |
|---|---|---|---|
| Encoder-only | BERT, RoBERTa | 双向 | 理解类：分类、NER、检索 |
| Decoder-only | GPT, LLaMA, Qwen | 单向因果 | 生成，现在的主流 |
| Encoder-Decoder | T5, BART | 编码双向 + 解码交叉注意力 | 翻译、摘要 |

**Q：为什么现在大模型都是 Decoder-only？（高频）**
1. **训练效率高**：每个位置都能当作预测目标，一次前向产生 $n$ 个训练信号；Encoder-Decoder 结构复杂，MLM 只有 15% 的位置贡献损失。
2. **零样本泛化更好**：实验（如 Google 的 architecture 对比）表明纯 LM 目标的 Decoder-only 在无微调时表现最好。
3. **注意力矩阵满秩**：因果 mask 使注意力矩阵为下三角，是满秩的，理论表达能力更强；双向注意力矩阵可能低秩。
4. **工程简单**：结构统一，方便 scale、方便 KV Cache 复用、方便做 in-context learning。
5. 生成任务天然统一：所有任务都可以转成 "prompt → continuation"。

---

## 2. 位置编码（必问）

Attention 本身是置换不变的（打乱输入顺序，输出只是同样打乱），必须注入位置信息。

| 方案 | 做法 | 优缺点 |
|---|---|---|
| 正弦绝对编码 | $PE_{(pos,2i)}=\sin(pos/10000^{2i/d})$，加到 embedding | 无参数、理论可外推，但实际外推差 |
| 可学习绝对编码 | BERT 的做法 | 简单，但完全不能超过训练长度 |
| 相对位置编码 | T5 的 bias、Transformer-XL | 建模相对距离，外推较好，但慢 |
| **RoPE** | 对 Q、K 做旋转 | LLaMA/Qwen/GLM 标配，相对位置 + 可外推 |
| **ALiBi** | 在 attention logits 上加 $-m\cdot\|i-j\|$ 线性偏置 | 极简、外推强，但长距离信息衰减 |

### 2.1 RoPE 详解

思想：把 $d$ 维向量两两配对成 $d/2$ 个复数，对位置 $m$ 的向量旋转角度 $m\theta_i$，$\theta_i=10000^{-2i/d}$。

关键性质：
$$\langle f(q,m), f(k,n)\rangle = g(q,k,m-n)$$
旋转后的内积只依赖**相对位置 $m-n$** —— 用绝对位置的实现，达到相对位置的效果。这是 RoPE 优雅的地方。

**Q：RoPE 怎么做长度外推？**
- **位置插值（PI）**：把位置索引压缩 $m \to m/s$，让超长位置落回训练见过的范围。需要少量微调。
- **NTK-aware 缩放**：不均匀缩放 —— 高频维度（局部信息）少缩放，低频维度（全局信息）多缩放，即调大 base：$\theta_i = (10000\cdot s^{d/(d-2)})^{-2i/d}$。**无需微调就有效果**。
- **YaRN**：NTK 分段插值 + 注意力温度调整，目前效果最好。
- 直接**调大 base**（如 10000 → 1000000）再继续预训练，是 LLaMA-3 / Qwen 长上下文的常规做法。

---

## 3. 注意力的 KV 优化（推理必问）

### 3.1 KV Cache

自回归解码时，每生成一个 token 都要对前面所有 token 做 attention。之前 token 的 K、V 不会变，缓存下来避免重复计算，把每步复杂度从 $O(n^2)$ 降到 $O(n)$。

**KV Cache 显存计算（要会算）**：
$$\text{Bytes} = 2 \times L \times n_{kv\_heads} \times d_{head} \times \text{seq\_len} \times \text{batch} \times \text{dtype\_bytes}$$
（前面的 2 是 K 和 V）

举例：LLaMA-7B（$L=32$，32 头，$d_{head}=128$），fp16，batch=1，4K 上下文：
$2\times32\times32\times128\times4096\times2 \approx 2.1$ GB。**batch 一大就爆显存** —— 这是 MQA/GQA 出现的原因。

### 3.2 MHA → MQA → GQA → MLA

| 方案 | K/V 头数 | KV Cache | 效果 |
|---|---|---|---|
| MHA | $h$ | 100% | 基线 |
| MQA | 1 | $1/h$ | 省显存多，效果掉点明显 |
| **GQA** | $g$（如 8） | $g/h$ | 折中，LLaMA-2/3 采用 |
| **MLA** | 低秩压缩到潜空间 | 更小 | DeepSeek-V2/V3，效果不降反升 |

**MLA 的思路**：把 K、V 联合压缩成一个低维潜向量 $c_{KV}$ 缓存，推理时再上投影恢复；并用矩阵吸收技巧避免显式还原。相当于对 KV Cache 做了低秩分解。

### 3.3 推理阶段的两个阶段（必问）

- **Prefill（预填充）**：处理输入 prompt，所有 token 并行计算，**计算密集（compute-bound）**，GPU 利用率高。指标：TTFT（首 token 延迟）。
- **Decode（解码）**：逐 token 生成，每步只算 1 个 token 但要读全部权重和 KV Cache，**访存密集（memory-bound）**，算力利用率极低。指标：TPOT（每 token 延迟）。

**结论**：LLM 推理慢的根本原因是 decode 阶段是访存瓶颈 —— 所以优化方向是减少访存（量化、MQA/GQA）和提高并行度（continuous batching、投机解码），而不是单纯堆算力。

### 3.4 推理系统优化

- **Continuous Batching**（vLLM/TGI）：不等整个 batch 都生成完，某个序列结束就立刻换新请求进来，吞吐提升数倍。
- **PagedAttention**：借鉴操作系统虚拟内存分页，把 KV Cache 分成固定大小 block 非连续存储，消除内存碎片，显存利用率从约 40% 提到 90% 以上〔vLLM, SOSP 2023〕，还能让不同请求共享公共前缀（system prompt）。
- **投机解码（Speculative Decoding）**：小模型（draft）一次猜 $k$ 个 token，大模型一次前向并行验证，接受最长正确前缀。**输出分布与原模型严格一致**（靠拒绝采样保证），加速 2-3×。变体：Medusa（多头并行预测）、EAGLE（在特征层面做草稿）。
- **Prefix Caching**：多轮对话/相同 system prompt 复用已算好的 KV。
- **Chunked Prefill**：把长 prompt 切块与 decode 混合调度，平衡 TTFT 和吞吐。

---

## 4. 训练流程

### 4.1 三/四阶段范式

```
预训练 (Pretrain) → 继续预训练/领域适配 (CPT) → 指令微调 (SFT) → 偏好对齐 (RLHF/DPO)
```

**预训练**：自回归语言建模 $\max\sum\log P(x_t|x_{<t})$，万亿 token 级数据。关键在**数据**：去重（MinHash/SimHash）、质量过滤（分类器 + 规则 + 困惑度）、配比（代码/数学能提升推理能力）、去污染（防止测试集泄漏）。

**SFT**：用 (instruction, response) 对做监督微调。**只对 response 部分计算 loss**（prompt 部分 mask 掉）—— 这是个高频细节考点。质量远比数量重要 —— LIMA 用 1000 条高质量数据即达到可观的对齐效果〔LIMA, NeurIPS 2023〕。

**对齐**：让模型输出符合人类偏好（有用、无害、诚实）。

### 4.2 RLHF（PPO）三步

1. **训 SFT 模型**（初始策略）。
2. **训奖励模型 RM**：收集人工偏好对 $(x, y_w, y_l)$，损失
   $$L=-\log\sigma(r_\theta(x,y_w)-r_\theta(x,y_l))$$
   （Bradley-Terry 模型）。RM 通常用 SFT 模型改造，最后接标量头。
3. **PPO 优化策略**：
   $$\max_\pi \mathbb{E}[r_\phi(x,y)] - \beta\,\text{KL}[\pi_\theta(y|x)\|\pi_{ref}(y|x)]$$

**Q：PPO 阶段有几个模型？（高频）** 四个：
- **Actor**（要训练的策略，可训练）
- **Critic**（价值函数，估计状态价值，可训练）
- **Reward Model**（打分，冻结）
- **Reference Model**（SFT 模型，冻结，算 KL 防止跑偏）

显存开销约是 SFT 的 4 倍，这是 RLHF 工程难度大的原因。

**Q：为什么要 KL 惩罚？** 防止 **reward hacking** —— 策略钻 RM 的空子（比如输出重复的讨好性文本），同时保持语言能力不退化。

### 4.3 DPO（必问，因为它简单又主流）

**核心洞察**：RLHF 的最优策略有闭式解
$$\pi^*(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$$
反解出 $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x)$，代入 Bradley-Terry 损失后 $Z(x)$ 抵消，得到：

$$L_{DPO}=-\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

**优点**：不需要显式 RM，不需要在线采样，只要两个模型（policy + ref），像 SFT 一样训练，稳定。
**缺点**：离线方法，受限于偏好数据的分布；容易同时降低 $y_w$ 和 $y_l$ 的概率（只要相对差距拉大就行）；对数据质量敏感。

**衍生**：IPO（防过拟合）、KTO（只需好/坏单标签，不需成对）、ORPO（无需 ref model，SFT 和对齐一步到位）、SimPO（用长度归一化的平均对数概率，无需 ref）。

### 4.4 GRPO 与推理模型（新热点）

GRPO（DeepSeek）：**去掉 Critic**，对同一个 prompt 采样一组 $G$ 个回答，用组内奖励的均值和标准差做归一化当作 advantage：
$$A_i=\frac{r_i-\text{mean}(r)}{\text{std}(r)}$$
省掉了和 Actor 同规模的 Critic 模型，显存和实现复杂度都大幅下降。

**R1 类推理模型的关键**：用**可验证奖励**（数学答案对错、代码能否通过测试）做大规模 RL，模型自发涌现出长链思考、自我验证、回溯等行为。这类奖励不会被 hack，是能 scale 的关键。

---

## 5. 参数高效微调（PEFT）

### 5.1 LoRA（必问）

$$W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r}BA,\quad B\in\mathbb{R}^{d\times r},A\in\mathbb{R}^{r\times k},\ r\ll\min(d,k)$$

- $A$ 用高斯初始化，$B$ **初始化为 0** → 训练开始时 $\Delta W=0$，不破坏预训练模型。
- 只训练 $A,B$，参数量降到原来的 0.1%~1%。
- **推理时可以合并** $W_0+BA$ 进权重，**零额外延迟**（这是 LoRA 相对 Adapter 的核心优势）。

**Q：LoRA 为什么有效？**
假设是"模型适配下游任务时的权重更新矩阵具有低**本征秩**（intrinsic rank）"—— 大模型已经学好了通用表示，微调只需在一个低维子空间内调整。

**Q：$r$ 和 $\alpha$ 怎么选？**
$r$ 常取 8/16/32/64；任务与预训练差异越大、数据越多，$r$ 越大。$\alpha$ 常取 $2r$ 或 $16$，缩放因子 $\alpha/r$ 控制更新幅度（相当于 LoRA 分支的学习率）。

**Q：LoRA 加在哪些模块？**
原论文只加 $W_q, W_v$。实践表明**加到所有线性层（q,k,v,o + FFN 的 gate/up/down）效果最好**〔QLoRA, NeurIPS 2023〕。

**Q：LoRA 和全参微调的差距？**
数据量小时接近甚至更好（正则效果）；数据量大或需要注入大量新知识时会落后。LoRA 更适合"学格式/风格/任务范式"，不适合"灌新知识"。

### 5.2 其他 PEFT 方法

| 方法 | 做法 | 特点 |
|---|---|---|
| **QLoRA** | 基座 4-bit NF4 量化冻结 + LoRA 训练（fp16） | 单卡 24G 微调 33B；配合 Double Quantization、Paged Optimizer |
| **DoRA** | 把权重分解为幅度和方向，只对方向用 LoRA | 效果更接近全参 |
| Adapter | 层间插入小 bottleneck 模块 | 有推理延迟 |
| Prefix/Prompt Tuning | 在 KV 或输入前加可学习的虚拟 token | 占用上下文长度，训练不稳定 |
| BitFit | 只训 bias | 极省但效果有限 |
| **AdaLoRA** | 按重要性动态分配各层的秩 | 更优的参数分配 |

**Q：什么时候用 LoRA，什么时候全参？**
数据 < 万级、只是学任务格式、多任务需要切换适配器 → LoRA。
需要注入大量领域知识、有充足算力、追求极致效果 → 全参（或先 CPT 再 SFT）。

### 5.3 灾难性遗忘

微调后通用能力下降。缓解：混入通用数据（比例 SFT : 通用 ≈ 1:1~1:5，**经验值，需按任务实测**）、更小学习率、LoRA（限制参数改动范围）、正则化（L2-SP、EWC）、模型融合（把微调前后权重做加权平均）。

---

## 6. 模型压缩与加速

### 6.1 量化

| 方案 | 类型 | 说明 |
|---|---|---|
| **GPTQ** | PTQ，权重 4bit | 基于二阶信息逐层做误差补偿，快，精度好 |
| **AWQ** | PTQ，权重 4bit | 发现约 1% 的「显著权重」由激活幅度决定〔AWQ, MLSys 2024〕，对其按通道缩放保护，不做混合精度 |
| **SmoothQuant** | PTQ，W8A8 | 把激活的异常值难度"迁移"一部分到权重上，让两者都好量化 |
| **LLM.int8()** | 混合精度 | 离群特征维度走 fp16，其余 int8 |
| **QAT** | 训练时量化 | 效果最好但成本高 |
| **FP8** | 硬件原生（H100） | 训练推理都可用，逐渐成为主流 |

**Q：为什么 LLM 量化难？** 激活值中存在**系统性离群值（outlier）**，某些特征维度的幅度比其他大 100 倍，直接按张量量化会把正常值全压到几个 bin 里。所以主流方案都在处理离群值。

**Q：量化对什么影响最大？** 权重量化到 4bit 通常损失很小；激活量化更难；KV Cache 量化到 int8/int4 能大幅提升吞吐但影响长文本质量。

### 6.2 知识蒸馏

- **Logits 蒸馏**：$L=\alpha\,\text{CE}(y,\hat y_s)+(1-\alpha)T^2\,\text{KL}(\text{softmax}(z_t/T)\|\text{softmax}(z_s/T))$。温度 $T$ 放大"暗知识"（错误类别之间的相对概率）；乘 $T^2$ 是因为软标签梯度会缩小 $1/T^2$。
- **特征蒸馏**：对齐中间层表示（需要投影层对齐维度）。
- **黑盒蒸馏 / 数据蒸馏**：用强模型生成数据训小模型（Alpaca、Distilabel 路线），LLM 时代最常用。
- **On-policy 蒸馏**：让学生自己采样，教师对学生的输出打分/纠正（MiniLLM、GKD），比纯离线数据蒸馏效果好。

### 6.3 MoE

用稀疏激活换参数量：每层有 $N$ 个专家 FFN，router 为每个 token 选 top-$k$（常见 $k$=1 或 2）。
- **总参数量大、激活参数量小** → 训练/推理 FLOPs 与稠密小模型相当，但容量大得多。
- **负载均衡**：需要辅助损失（auxiliary loss）防止所有 token 都路由到少数专家；DeepSeek-V3 用无辅助损失的 bias 调整方案。
- **挑战**：显存占用仍是全部专家（推理需全部加载）、通信开销大（All-to-All）、微调容易不稳定。
- **细粒度专家 + 共享专家**（DeepSeekMoE）：把专家切细增加组合数，同时保留常驻共享专家学通用知识。

---

## 7. RAG（检索增强生成）

### 7.1 基础链路

```
文档 → 切分(chunking) → 向量化(embedding) → 存入向量库
查询 → 改写 → 向量检索 + 关键词检索(BM25) → 融合(RRF) → 重排(rerank) → 拼 prompt → 生成 → 引用/校验
```

### 7.2 每个环节的优化点（面试爱问"效果不好怎么办"）

**切分**：固定长度切分会打断语义 → 按语义/标题层级切分；用 **overlap**（10-20%）；**小块检索、大块喂给模型**（small-to-big）；父子块索引。

**Embedding**：选中文效果好的模型（BGE、GTE、Conan）；领域数据可以微调（用 (query, positive, hard negatives) 做对比学习）；注意查询和文档要用同一空间，非对称检索模型要加正确的指令前缀。

**检索**：
- **混合检索**：向量（语义）+ BM25（关键词，对专有名词/数字/型号更准），用 RRF 融合：$\text{score}=\sum_i\frac{1}{k+\text{rank}_i}$。
- **查询改写**：多查询生成（Multi-Query）、HyDE（先让 LLM 生成假设答案再拿去检索，解决 query 和 doc 的语义不对称）、子问题分解。
- **重排**：用 Cross-Encoder（如 bge-reranker）对 top-50 精排取 top-5。Cross-Encoder 让 query 和 doc 交互，比双塔准很多但慢，所以只用在重排。

**生成**：明确要求"只根据给定资料回答，无法回答就说不知道"；要求输出引用来源；上下文顺序注意 **"Lost in the Middle"**（长上下文中间的信息容易被忽略，把最相关的放开头和结尾）。

**评测**：检索侧看 Recall@k / MRR / NDCG；生成侧看忠实度（faithfulness）、答案相关性、上下文精确率（RAGAS 框架）。

### 7.3 向量检索 ANN

| 索引 | 原理 | 特点 |
|---|---|---|
| **HNSW** | 多层可导航小世界图 | 召回高、查询快，内存占用大，主流选择 |
| IVF-PQ | 倒排聚类 + 乘积量化 | 内存友好，适合超大规模 |
| Flat | 暴力 | 精确但慢，小数据集可用 |
| ScaNN | 各向异性量化 | Google 方案，精度高 |

**相似度**：内积 / 余弦 / L2。归一化后余弦与内积等价，L2 与余弦单调相关。

---

## 8. Agent 与 Function Calling

**ReAct 范式**：`Thought → Action → Observation` 循环，直到得出 Final Answer。让模型显式写出推理过程再决定调工具。

**Function Calling**：把工具描述（名称、参数 JSON Schema）放进 prompt/特殊字段，模型输出结构化的调用请求，外部执行后把结果回填。

**关键工程问题**：
- 工具太多（>20）时选错 → 先做工具检索（把工具描述向量化，按 query 召回 top-k 再给模型）。
- 多步任务错误累积 → 加反思（Reflexion）、加校验器、限制最大步数、支持回退。
- 上下文爆炸 → 对历史做摘要压缩，只保留关键状态。
- **评测难**：端到端成功率 + 每步工具调用准确率 + 平均步数/成本。

**Multi-Agent**：Planner/Executor/Critic 分工。收益在于分解复杂任务，代价是延迟和成本成倍上升 —— 面试时能说出这个 trade-off 会加分。

---

## 9. 幻觉、评测与解码

### 9.1 幻觉

**成因**：训练数据中的错误/过时信息；自回归的"必须往下说"的压力；SFT 教会模型回答超出其知识边界的问题；RLHF 奖励"看起来自信"的回答；长尾知识记忆不牢。

**缓解**：RAG 引入外部知识；要求输出引用；让模型学会说"不知道"（SFT 数据里加拒答样本）；自洽性检查（Self-Consistency 多次采样投票）；事实性 RL；解码时约束（约束解码/结构化输出）。

### 9.2 解码策略

| 策略 | 说明 |
|---|---|
| Greedy | 每步取 argmax，确定性但重复、无趣 |
| Beam Search | 保留 $k$ 条路径，适合翻译等有唯一答案的任务，开放生成上会很枯燥 |
| Temperature | $p_i\propto\exp(z_i/T)$，$T$ 小更确定，$T$ 大更随机 |
| **Top-k** | 只在概率最高的 k 个中采样 |
| **Top-p (nucleus)** | 累积概率达到 p 的最小集合中采样，比 top-k 自适应 |
| Repetition / Presence Penalty | 抑制重复 |
| Min-p | 按最高概率的比例设阈值，长尾更稳 |

**实践**：确定性任务（代码、抽取）用 $T\approx0$；创意生成用 $T=0.7\sim1.0$ + top_p=0.9。

### 9.3 评测

- **基座能力**：MMLU / CMMLU / C-Eval（知识）、GSM8K / MATH（数学）、HumanEval / MBPP（代码）、BBH（推理）。
- **对话/对齐**：MT-Bench、AlpacaEval、Arena Elo（人类盲测投票，最可信）。
- **长文本**：LongBench、大海捞针（NIAH）。
- **方法论**：注意**数据污染**（测试集混进训练数据）；LLM-as-a-Judge 有**位置偏见**（偏好第一个）、**长度偏见**（偏好长回答）、**自我偏好**，要做位置交换 + 打分标准细化。

---

## 10. Scaling Law 与训练工程

**Chinchilla 结论**：给定算力预算 $C\approx 6ND$（$N$ 参数量，$D$ 数据量 token 数），最优配比是 $D\approx 20N$〔Chinchilla, NeurIPS 2022〕。之前的模型（GPT-3）普遍**训练不足**。
现在实践中因为推理成本占大头，会**远超 Chinchilla 最优**地喂数据（LLaMA-3 8B 用了 15T token，约 1800 tokens/param），换取小模型的高性能。

**训练常见问题**：
- **Loss spike**：梯度裁剪、降 lr、跳过坏 batch、回滚到之前的 checkpoint 换数据顺序继续、用 bf16 而非 fp16。
- **数值精度**：bf16 动态范围与 fp32 相同（8 位指数），精度低但不易溢出，是大模型训练首选；fp16 需要 loss scaling。
- 分布式并行策略见 [10 工程能力](10-engineering.md)。

---

## 延伸阅读

> 本章是全书时效性最差的一章。以下论文是基础，但具体方案半年就会变，面试前请对一遍最新进展。

**架构**
- *Attention Is All You Need*（Vaswani et al., NeurIPS 2017）
- *RoFormer: Enhanced Transformer with Rotary Position Embedding*（Su et al., 2021）—— RoPE
- *YaRN: Efficient Context Window Extension of Large Language Models*（Peng et al., 2023）—— NTK 分段插值外推
- *Train Short, Test Long: Attention with Linear Biases*（Press et al., ICLR 2022）—— ALiBi
- *GLU Variants Improve Transformer*（Shazeer, 2020）—— SwiGLU
- *Fast Transformer Decoding: One Write-Head is All You Need*（Shazeer, 2019）—— MQA
- *GQA: Training Generalized Multi-Query Transformer Models*（Ainslie et al., EMNLP 2023）
- *DeepSeek-V2 / DeepSeek-V3 技术报告*（2024）—— MLA、细粒度 MoE、无辅助损失负载均衡

**推理**
- *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*（Dao et al., NeurIPS 2022）
- *Efficient Memory Management for Large Language Model Serving with PagedAttention*（Kwon et al., SOSP 2023）—— vLLM，显存利用率 40%→90% 的出处
- *Fast Inference from Transformers via Speculative Decoding*（Leviathan et al., ICML 2023）

**训练与对齐**
- *Training language models to follow instructions with human feedback*（Ouyang et al., NeurIPS 2022）—— InstructGPT，RLHF 三步法
- *Direct Preference Optimization*（Rafailov et al., NeurIPS 2023）—— DPO 的闭式解推导
- *DeepSeekMath: Pushing the Limits of Mathematical Reasoning*（Shao et al., 2024）—— GRPO
- *DeepSeek-R1*（2025）—— 可验证奖励的大规模 RL
- *LIMA: Less Is More for Alignment*（Zhou et al., NeurIPS 2023）—— 「1000 条高质量 SFT 数据」结论的出处
- *Training Compute-Optimal Large Language Models*（Hoffmann et al., NeurIPS 2022）—— Chinchilla，D≈20N

**微调与压缩**
- *LoRA: Low-Rank Adaptation of Large Language Models*（Hu et al., ICLR 2022）
- *QLoRA: Efficient Finetuning of Quantized LLMs*（Dettmers et al., NeurIPS 2023）—— 也是「LoRA 加到所有线性层更好」的实验来源
- *GPTQ: Accurate Post-Training Quantization*（Frantar et al., ICLR 2023）
- *AWQ: Activation-aware Weight Quantization*（Lin et al., MLSys 2024）—— 「约 1% 显著权重」的出处
- *SmoothQuant*（Xiao et al., ICML 2023）
- *LLM.int8()*（Dettmers et al., NeurIPS 2022）—— 激活离群值现象

**RAG 与评测**
- *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*（Lewis et al., NeurIPS 2020）
- *Precise Zero-Shot Dense Retrieval without Relevance Labels*（Gao et al., ACL 2023）—— HyDE
- *Lost in the Middle: How Language Models Use Long Contexts*（Liu et al., TACL 2024）
- *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*（Zheng et al., NeurIPS 2023）—— 位置偏见与长度偏见
- *ReAct: Synergizing Reasoning and Acting in Language Models*（Yao et al., ICLR 2023）

---

## 自测清单

- [ ] 能手写 Multi-Head Attention（见 [09](09-coding-ml-scratch.md)）并解释 $\sqrt{d_k}$
- [ ] 能说出 Decoder-only 主流化的 4 个理由
- [ ] 能讲 RoPE 的相对位置性质和 3 种外推方法
- [ ] 能算 KV Cache 显存，并说清 MQA/GQA/MLA 的取舍
- [ ] 能区分 prefill 和 decode，并说明 LLM 推理为什么是访存瓶颈
- [ ] 能说出 PPO 的四个模型，以及 DPO 怎么把 RM 消掉的
- [ ] 能讲 GRPO 相对 PPO 的改动
- [ ] 能讲 LoRA 的原理、B 为什么初始化为 0、r/α 怎么选
- [ ] 能说出 3 种量化方案的核心思想和 LLM 量化难在哪
- [ ] 能给出 RAG 效果差时的至少 6 条优化手段
- [ ] 能说清 Top-k / Top-p / Temperature 的区别与适用场景
- [ ] 知道 Chinchilla 结论以及为什么实践中会超配数据
