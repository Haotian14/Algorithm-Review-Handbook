# 07 · NLP 基础

> LLM 时代这些仍然要会。BERT 系模型在检索、分类、NER 上仍是工业主力，而 tokenizer、词向量、RNN 是理解 Transformer 的前置。

## 本章高频考点

| 考点 | 频率 | 典型问法 |
|---|---|---|
| BPE / WordPiece / SentencePiece | ★★★ | 为什么用子词，中文怎么切 |
| Word2Vec 两种模型 + 两种加速 | ★★★ | 负采样和层次 softmax |
| RNN / LSTM / GRU | ★★ | LSTM 三个门，怎么缓解梯度消失 |
| BERT 预训练任务 | ★★★ | MLM 的 80/10/10 为什么 |
| BERT 变体 | ★★ | RoBERTa/ALBERT/ELECTRA 改了什么 |
| 文本表示与向量检索 | ★★★ | Sentence-BERT，为什么 CLS 直接用效果差 |
| NER / 文本分类实践 | ★★ | CRF 的作用 |

---

## 1. 分词与 Tokenizer

### 1.1 为什么用子词（subword）

- 词级：词表巨大、OOV 严重（未登录词只能变 UNK）。
- 字符级：序列太长，语义粒度太细。
- **子词**：折中 —— 常用词保留完整，罕见词拆成有意义的片段（`unhappiness` → `un` + `happi` + `ness`），词表可控且无 OOV。

### 1.2 三种算法

| 算法 | 思路 | 使用者 |
|---|---|---|
| **BPE** | 从字符开始，反复合并**频率最高**的相邻对 | GPT 系、LLaMA（BBPE） |
| **WordPiece** | 合并使**语言模型似然提升最大**的对（近似为 $\frac{P(ab)}{P(a)P(b)}$ 最大） | BERT |
| **Unigram LM** | 从大词表开始，反复删除对总似然损失最小的子词 | ALBERT、T5、多数 SentencePiece 模型 |

**SentencePiece** 是一个工具库（可实现 BPE 或 Unigram），特点是把空格也当普通字符（用 `▁` 表示），**语言无关、可逆**，不需要预分词，中日韩友好。

**Byte-level BPE (BBPE)**：在 UTF-8 字节上做 BPE，词表基础单元只有 256 个，**任何字符都不会 OOV**，GPT-2 之后成为标准。代价是中文一个字要 2-3 个 token（中文 LLM 会专门扩充中文词表）。

**常见追问**：
- **词表大小怎么权衡？** 大词表 → 序列短、推理快、但 Embedding 和输出 softmax 层参数量大、低频 token 训练不充分。中文 LLM 常用 6-15 万。
- **为什么 LLM 数不清 "strawberry 里有几个 r"？** token 是子词单元，模型看不到字符级组成。
- **数字怎么处理？** LLaMA 把数字按单个数字切分（0-9 各一个 token），有利于算术能力。

---

## 2. 词向量

### 2.1 Word2Vec

- **CBOW**：用上下文预测中心词，训练快，对高频词好。
- **Skip-gram**：用中心词预测上下文，对低频词和小数据集更好（每个词产生更多训练样本）。

**两种加速**（必问）：
1. **层次 Softmax**：用 Huffman 树组织词表，把 $|V|$ 分类变成 $\log|V|$ 次二分类。高频词路径短。
2. **负采样（Negative Sampling）**：把多分类改成二分类 —— 正样本是真实的 (中心词, 上下文词)，随机采 $k$ 个负样本，用 sigmoid 做二分类。
   采样分布：$P(w)\propto f(w)^{3/4}$。**3/4 次方是为了适度提升低频词被采到的概率**（相对纯频率而言），同时又不至于像均匀分布那样过度采样噪声词。

**Q：Word2Vec 的缺陷？** 静态词向量，一词一义 —— "苹果"在"吃苹果"和"苹果手机"中是同一个向量。这就是 ELMo/BERT 上下文相关表示的动机。

### 2.2 其他

- **GloVe**：基于全局共现矩阵做矩阵分解，损失 $\sum f(X_{ij})(w_i^T\tilde w_j+b_i+\tilde b_j-\log X_{ij})^2$，结合了全局统计和局部窗口。
- **FastText**：词表示为字符 n-gram 的和，能处理 OOV 和形态丰富的语言，分类任务极快。

---

## 3. RNN / LSTM / GRU

### 3.1 RNN 的问题

$h_t=\tanh(W_hh_{t-1}+W_xx_t)$。反向传播时梯度包含 $\prod W_h^T\text{diag}(\tanh')$，连乘导致**梯度消失/爆炸**，无法建模长依赖。

### 3.2 LSTM

三个门 + 一个细胞状态：
- **遗忘门** $f_t=\sigma(W_f[h_{t-1},x_t])$：决定丢弃多少旧记忆。
- **输入门** $i_t$ + 候选 $\tilde C_t=\tanh(\cdot)$：决定写入多少新信息。
- **细胞状态更新** $C_t=f_t\odot C_{t-1}+i_t\odot\tilde C_t$
- **输出门** $o_t$：$h_t=o_t\odot\tanh(C_t)$

**Q：LSTM 怎么缓解梯度消失？（必答）**
细胞状态的更新是**加性**的：$\frac{\partial C_t}{\partial C_{t-1}}=f_t$。当遗忘门接近 1 时，梯度可以近乎无损地沿细胞状态这条"高速公路"传递，避免了 RNN 中的连乘衰减。注意是**缓解**不是根除。

### 3.3 GRU

合并成两个门：更新门 $z_t$（同时承担遗忘和输入）、重置门 $r_t$。$h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$。
参数少约 1/4，训练快，小数据集上常与 LSTM 相当。

### 3.4 Seq2Seq 与注意力的由来

Encoder-Decoder 把整个源句压缩成一个固定向量 → **信息瓶颈**，长句效果差。
Bahdanau Attention：解码每一步动态地对编码器所有隐状态加权求和。这正是后来 Transformer 中 Attention 的思想源头 —— Transformer 的贡献是**去掉 RNN，只保留注意力**，从而可并行。

---

## 4. BERT 及其变体

### 4.1 BERT 预训练

- **MLM（掩码语言模型）**：随机遮 15% 的 token。被选中的 token：**80% 替换为 [MASK]，10% 替换为随机词，10% 保持不变**。
  **为什么这样设计？（必问）** 因为下游微调时输入里没有 [MASK]，如果 100% 用 [MASK] 会造成**预训练-微调不一致**。10% 随机词强迫模型对每个 token 都保持上下文表示能力（不能只在看到 MASK 时才思考），10% 不变让模型偏向真实分布。
- **NSP（下一句预测）**：判断两句是否连续。后来 RoBERTa 证明 **NSP 基本无用甚至有害**（因为负样本来自不同文档，任务退化成主题预测，太简单）。

**输入表示** = Token Embedding + Segment Embedding + Position Embedding（可学习的绝对位置）。

### 4.2 变体对比

| 模型 | 改动 |
|---|---|
| **RoBERTa** | 去掉 NSP、**动态掩码**（每个 epoch 换掩码位置）、更大 batch、更多数据、更长训练、BBPE |
| **ALBERT** | 词嵌入因式分解（$V\times H \to V\times E + E\times H$）、跨层参数共享、用 SOP 替代 NSP（预测句序是否颠倒，更难更有用） |
| **ELECTRA** | 用小生成器替换部分 token，判别器做**全部位置**的"是否被替换"二分类 → 样本效率远高于只用 15% 位置的 MLM |
| **DeBERTa** | 解耦内容和位置的注意力 + 增强的掩码解码器 |
| **中文** | MacBERT（用同义词替换 MASK）、ERNIE（实体/短语级掩码）、全词掩码 WWM |

---

## 5. 文本表示与语义检索

**Q：为什么直接用 BERT 的 [CLS] 做句向量效果很差？（高频）**
① BERT 的 [CLS] 是为 NSP 训练的，没有直接优化"语义相似句子的向量应该接近"；② BERT 的原生句向量存在**各向异性**（anisotropy）—— 向量分布在一个狭窄的锥形区域内，任意两句的余弦相似度都很高，区分度差；高频词还会主导方向。

**解法**：
- **Sentence-BERT**：孪生网络 + 有监督的三元组/分类目标微调，让相似句向量接近。
- **SimCSE**：无监督版本极简 —— 同一句话过两次带 dropout 的编码器得到两个略有差异的向量作为正对，batch 内其他句子为负样本，做对比学习。有监督版用 NLI 数据的 entailment 作正例、contradiction 作 hard negative。
- **后处理**：BERT-flow / BERT-whitening 把分布变换成各向同性。
- **现代方案**：BGE / GTE / E5 系列 —— 大规模弱监督对比预训练 + 高质量有监督微调 + 指令前缀。

**双塔 vs 交叉编码器**：双塔（Bi-Encoder）可预计算、快，用于召回；交叉编码器（Cross-Encoder）让 query 和 doc 在每层交互，准但每对都要算一次，用于重排。这与 [05 推荐](05-recsys-ads.md) 的召回/精排是同一套思想。

---

## 6. 典型 NLP 任务

### 6.1 NER（序列标注）

标注方案：BIO / BIOES。
经典架构：BiLSTM + CRF，或 BERT + CRF。

**Q：为什么要加 CRF？（必问）**
Softmax 逐位置独立分类，无法建模**标签之间的约束**，可能输出 `I-PER` 直接跟在 `O` 后面这种非法序列。CRF 学习标签转移矩阵，用 Viterbi 解码全局最优路径，保证输出合法。

**其他范式**：Span-based（枚举片段做分类，天然支持嵌套实体）、MRC-based（把 NER 转成阅读理解）、生成式（LLM 直接输出结构化结果）。

### 6.2 文本分类

短文本可用 FastText/TextCNN（多尺寸卷积核当 n-gram 特征提取器）；一般任务 BERT 微调；LLM 时代小样本直接用 prompt + few-shot 或用 LLM 造数据蒸馏给小模型。

**长文本怎么办？** 截断（头 256 + 尾 256 常比只取头好）、滑窗分段后聚合、用长上下文模型（Longformer）、先抽取关键句再分类。

### 6.3 文本相似度/匹配

- 表示型（双塔）vs 交互型（ESIM、BERT 交叉）。
- 评价：Spearman 相关系数（STS 任务）、准确率。

---

## 自测清单

- [ ] 能说清 BPE / WordPiece / Unigram 的区别，以及 BBPE 的好处
- [ ] 能讲 Word2Vec 的两种结构和两种加速，以及 3/4 次方的含义
- [ ] 能说清 LSTM 三个门和"加性更新"为什么缓解梯度消失
- [ ] 能解释 BERT MLM 的 80/10/10 设计动机
- [ ] 能说出 RoBERTa/ALBERT/ELECTRA 各改了什么
- [ ] 能解释 BERT 句向量各向异性问题和 SimCSE 的做法
- [ ] 能解释 NER 中 CRF 的必要性
