# 02 · 经典机器学习

> 这一章是算法岗的"必答题库"。哪怕你面的是 LLM 岗，LR、树模型、评估指标也一定会被问。这里的每一条都要能推、能讲、能举反例。

## 本章高频考点

> 星级是**主观判断**，反映常见面经里的印象分布，不是统计数据。用来排复习优先级即可。

| 考点 | 频率 | 典型问法 |
|---|---|---|
| LR 推导 + 为什么用交叉熵不用 MSE | ★★★ | 完整推导梯度 |
| SVM 与核函数 | ★★★ | 为什么要对偶？核函数解决什么？ |
| GBDT / XGBoost / LightGBM 区别 | ★★★ | 三者的工程优化点 |
| RF vs GBDT | ★★★ | 偏差方差角度 |
| AUC 的含义与计算 | ★★★ | 手写 AUC；AUC 为什么对不平衡稳健 |
| 过拟合处理 | ★★★ | 列举 8 种以上 |
| 特征工程 | ★★ | 类别特征怎么处理 |
| 不平衡数据 | ★★★ | 采样/权重/阈值/指标四个层面 |

---

## 1. 逻辑回归（LR）

### 1.1 完整推导

模型：$\hat y = \sigma(w^Tx+b)$，$\sigma(z)=\frac{1}{1+e^{-z}}$。

似然（伯努利）：$L=\prod_i \hat y_i^{y_i}(1-\hat y_i)^{1-y_i}$

负对数似然 = 交叉熵：$J = -\frac{1}{n}\sum_i [y_i\log\hat y_i + (1-y_i)\log(1-\hat y_i)]$

关键中间结果：$\sigma'(z)=\sigma(z)(1-\sigma(z))$

梯度：
$$\frac{\partial J}{\partial w} = \frac{1}{n}\sum_i (\hat y_i - y_i)x_i$$

**这个形式非常干净 —— 误差 × 输入。线性回归的梯度也是这个形式，这不是巧合**：都属于指数族 + 正则链接函数（GLM）。

### 1.2 为什么用交叉熵而不是 MSE？（必问）

如果用 MSE：$J=\frac12(\hat y-y)^2$，则 $\frac{\partial J}{\partial w}=(\hat y-y)\sigma'(z)x$。

多出来的 $\sigma'(z)=\sigma(1-\sigma)$ 在 $z$ 很大或很小时趋近 0 → **梯度消失**。即预测错得越离谱（$z$ 绝对值大），梯度反而越小，学不动。

交叉熵的梯度里没有 $\sigma'$，错得越多梯度越大。另外 MSE + sigmoid 是非凸的，交叉熵 + sigmoid 是凸的。

### 1.3 LR 常见追问

**Q：LR 为什么是线性模型？**
决策边界 $w^Tx+b=0$ 是超平面，对特征线性。sigmoid 只是把 logit 映射成概率，不改变边界的线性性。

**Q：LR 能处理非线性吗？**
能，但要靠人：特征交叉、离散化分桶、多项式特征。这就是为什么工业界 LR 时代靠"人肉特征工程"，也是 FM/Wide&Deep 出现的动机（[05 推荐](05-recsys-ads.md)）。

**Q：LR 为什么在工业界（尤其广告 CTR）长盛不衰？**
① 可解释，权重直接看；② 训练/预测极快，支持超大规模稀疏特征（十亿级）；③ 天然输出概率，方便做 ECPM 计算和校准；④ 易于在线学习（FTRL）。

**Q：LR 中特征离散化的好处？**
① 引入非线性（每个分桶一个权重）；② 对异常值鲁棒（年龄 300 落在最后一桶）；③ 稀疏向量内积快；④ 特征可以单独调整，模型更稳定；⑤ 便于做特征交叉。

**Q：多分类怎么做？**
Softmax 回归（多项 LR），或 OvR / OvO。Softmax 各类别互斥；多标签用多个 sigmoid。

**Q：LR 参数能否初始化为 0？**
可以。LR 是凸的且没有对称性问题（不像神经网络的隐层神经元会对称）。

---

## 2. SVM

### 2.1 核心思想

最大化间隔：$\min \frac12\|w\|^2\ \text{s.t.}\ y_i(w^Tx_i+b)\ge 1$。
几何间隔 $= \frac{2}{\|w\|}$，最大化间隔 ⟺ 最小化 $\|w\|^2$。

### 2.2 为什么要转对偶？（必问）

三个理由：
1. **引入核函数**：对偶形式里样本只以内积 $x_i^Tx_j$ 出现，可以直接换成 $K(x_i,x_j)$，无需显式映射到高维。
2. 复杂度从与特征维度 $d$ 相关变成与样本数 $n$ 相关，$d\gg n$ 时更划算。
3. 对偶问题的约束更简单（只有 $0\le\alpha_i\le C$ 和 $\sum\alpha_iy_i=0$）。

对偶形式：$\max_\alpha \sum_i\alpha_i - \frac12\sum_{i,j}\alpha_i\alpha_jy_iy_jK(x_i,x_j)$

由 KKT 互补松弛，只有落在间隔边界上/内的样本 $\alpha_i>0$，即**支持向量**。预测只依赖支持向量。

### 2.3 核函数

| 核 | 表达式 | 说明 |
|---|---|---|
| 线性 | $x_i^Tx_j$ | 高维稀疏文本首选 |
| 多项式 | $(\gamma x_i^Tx_j+r)^d$ | 度数高易过拟合 |
| RBF/高斯 | $\exp(-\gamma\|x_i-x_j\|^2)$ | 默认选择，对应无穷维空间 |
| Sigmoid | $\tanh(\gamma x_i^Tx_j+r)$ | 少用 |

**核函数的合法条件**：Mercer 条件 —— 核矩阵（Gram 矩阵）对任意样本集都半正定。

**RBF 的 $\gamma$ 怎么影响？** $\gamma$ 大 → 单个样本影响范围小 → 决策边界弯曲 → 过拟合。$C$ 大 → 惩罚误分类重 → 间隔小 → 过拟合。

### 2.4 软间隔与 Hinge Loss

引入松弛变量 $\xi_i$：$\min \frac12\|w\|^2 + C\sum\xi_i$。等价的无约束形式：
$$\min \frac12\|w\|^2 + C\sum_i \max(0, 1-y_i f(x_i))$$
即 **L2 正则 + Hinge Loss**。

**Hinge vs 交叉熵**：Hinge 在分类正确且间隔 >1 时梯度为 0（只关心边界附近的点，稀疏解）；交叉熵永远有梯度（所有样本都参与，输出可解释为概率）。

### 2.5 LR vs SVM（经典对比题）

| 维度 | LR | SVM |
|---|---|---|
| 损失 | LogLoss | Hinge Loss |
| 关注样本 | 全部（远离边界的也有小梯度） | 仅支持向量 |
| 输出 | 概率 | 距离（需 Platt scaling 才有概率） |
| 非线性 | 靠特征工程 | 靠核函数 |
| 大数据 | 更适合（可在线学习） | 核 SVM 是 $O(n^2)\sim O(n^3)$，不适合 |
| 异常值 | 敏感（所有点都计损失） | 相对鲁棒 |

---

## 3. 树模型

### 3.1 决策树三兄弟

| 算法 | 分裂准则 | 特征类型 | 树型 |
|---|---|---|---|
| ID3 | 信息增益 | 只支持离散 | 多叉 |
| C4.5 | 信息增益率 | 连续+离散 | 多叉 |
| CART | 分类用基尼，回归用平方误差 | 连续+离散 | **二叉** |

**信息增益的缺陷**：偏向取值多的特征（极端情况：用 ID 做特征，信息增益最大但毫无泛化性）。C4.5 用信息增益率 = 信息增益 / 特征自身的熵，惩罚取值多的特征。

**基尼指数**：$\text{Gini}=1-\sum_k p_k^2$。在**自然对数**下它是熵的一阶泰勒近似（用 $\ln p\approx p-1$ 代入 $-\sum p\ln p$ 即得 $1-\sum p^2$），计算不含对数所以更快，这是 CART 选它的原因。

**剪枝**：预剪枝（max_depth、min_samples_leaf、min_impurity_decrease）快但可能欠拟合；后剪枝（CART 的代价复杂度剪枝 $C_\alpha(T)=C(T)+\alpha|T|$）效果好但慢。

### 3.2 随机森林（Bagging）

两重随机性：
1. **样本随机**：Bootstrap 有放回抽样（每棵树约用 63.2% 的样本，$1-(1-1/n)^n \to 1-1/e$）。剩下的 36.8% 是 **OOB 样本**，可直接当验证集。
2. **特征随机**：每次分裂只从随机 $m$ 个特征中选（分类常取 $\sqrt{d}$）。

目的是**降低树之间的相关性**。因为 $\text{Var}(\bar X)=\rho\sigma^2+\frac{1-\rho}{T}\sigma^2$，$T\to\infty$ 时方差下界由相关系数 $\rho$ 决定，所以降低 $\rho$ 比加树更重要。

### 3.3 GBDT（Boosting）

核心：**用新树拟合当前模型的负梯度**（对平方损失就是残差）。

第 $m$ 轮：$F_m(x)=F_{m-1}(x)+\nu\cdot h_m(x)$，$h_m$ 拟合 $r_{im}=-\left[\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$，$\nu$ 是学习率（shrinkage）。

**为什么 GBDT 用回归树而不是分类树？** 因为要拟合的是连续的负梯度值，即使做分类任务也是如此。

**为什么 GBDT 不适合高维稀疏特征？** 树的分裂基于特征取值划分，one-hot 后每个特征只有 0/1，一次分裂只能切出极少样本，增益低且容易过拟合；LR/FM 在这类特征上表现更好。这是"CTR 场景为什么用 LR/FM 而不是纯 GBDT"的答案。

### 3.4 XGBoost 的改进（高频）

1. **二阶泰勒展开**：$L \approx \sum_i [g_i f(x_i) + \frac12 h_i f^2(x_i)] + \Omega(f)$，用一阶 $g$ + 二阶 $h$，收敛更快、更准。
2. **正则项进结构**：$\Omega(f)=\gamma T + \frac12\lambda\sum_j w_j^2$，$T$ 是叶子数。
3. **叶子最优权重有闭式解**：$w_j^*=-\frac{\sum_{i\in I_j}g_i}{\sum_{i\in I_j}h_i+\lambda}$，对应的结构分数
   $$\text{Obj}=-\frac12\sum_j\frac{G_j^2}{H_j+\lambda}+\gamma T$$
   分裂增益 $= \frac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma$。**这个公式要会写。**
4. **缺失值自动处理**：为每个分裂学一个默认方向。
5. **列采样**（借鉴 RF）、**近似分位数分裂**、**块存储支持并行**（并行的是特征维度的增益计算，不是树之间）。

### 3.5 LightGBM 的改进（高频）

1. **Histogram 直方图算法**：把连续特征离散成 256 个 bin，复杂度从 $O(\#data\times\#feature)$ 降到 $O(\#bin\times\#feature)$；且父节点直方图减去一个子节点即得另一子节点（**直方图作差加速**）。
2. **Leaf-wise 生长**（XGBoost 是 level-wise）：每次选增益最大的叶子分裂，同样叶子数下损失更低，但容易过拟合，需 `max_depth` 限制。
3. **GOSS**（单边梯度采样）：保留大梯度样本，小梯度样本随机采样并放大权重。
4. **EFB**（互斥特征捆绑）：把很少同时非零的稀疏特征绑成一个，降维。
5. **原生类别特征支持**：按类别的梯度统计排序后找最优切分，不用 one-hot。

**一句话总结三者**：XGBoost = 二阶导 + 正则化的精细版；LightGBM = 为速度和内存做工程优化的版本；CatBoost = 用 Ordered Target Statistics 解决类别特征的**目标泄漏**并用 Ordered Boosting 减小预测偏移。

### 3.6 RF vs GBDT（必答）

| | 随机森林 | GBDT |
|---|---|---|
| 集成方式 | Bagging，并行 | Boosting，串行 |
| 主要降低 | 方差 | 偏差 |
| 基学习器 | 深树（低偏差高方差） | 浅树（高偏差低方差） |
| 对异常值 | 鲁棒 | 敏感（会一直拟合它） |
| 树多了会过拟合吗 | 基本不会 | 会 |
| 数据要求 | 不需要归一化 | 不需要归一化 |

---

## 4. 无监督

### 4.1 K-Means

流程：初始化 k 个中心 → 分配样本到最近中心 → 重算中心 → 迭代至收敛。本质是 EM 的硬分配版本，最小化 SSE $\sum_i\|x_i-\mu_{c_i}\|^2$。

**缺点与对策**：
- 需要指定 k → 肘部法、轮廓系数、Gap Statistic。
- 对初始化敏感 → **K-Means++**：第一个中心随机，后续中心以正比于 $D(x)^2$ 的概率选（离已有中心越远越可能被选中）。
- 只能发现球形簇 → 用 DBSCAN / 谱聚类 / GMM。
- 对异常值敏感 → K-Medoids。
- 必须归一化（基于欧氏距离）。

**复杂度**：$O(n\cdot k\cdot d\cdot t)$。

### 4.2 其他聚类

- **DBSCAN**：基于密度（eps, min_samples），能发现任意形状簇 + 自动识别噪声点，不用指定 k；但对密度不均的数据和高维数据效果差。
- **层次聚类**：不用指定 k，能出树状图，但 $O(n^3)$ 太慢。
- **GMM**：软分配，用 EM 求解，能建模椭球簇；K-Means 是 GMM 协方差为 $\sigma^2 I$ 且 $\sigma\to0$ 的特例。

### 4.3 EM 算法

E 步：用当前参数计算隐变量的后验 $Q(z)=p(z|x,\theta^{old})$。
M 步：最大化 $\sum_z Q(z)\log\frac{p(x,z|\theta)}{Q(z)}$（ELBO）。
保证似然单调不减，但只收敛到局部最优。VAE 就是 EM 的变分近似版本。

---

## 5. 评估指标（极高频）

### 5.1 混淆矩阵派生

- Precision $=\frac{TP}{TP+FP}$：预测为正的里面对了多少。**关心误报**时用（如推送、风控拦截）。
- Recall $=\frac{TP}{TP+FN}$：真正的正例找回了多少。**关心漏报**时用（如癌症筛查、欺诈检测）。
- F1 $=\frac{2PR}{P+R}$：调和平均，惩罚偏科。$F_\beta$ 中 $\beta>1$ 偏向 recall。

### 5.2 ROC 与 AUC（必问）

- ROC：横轴 FPR $=\frac{FP}{FP+TN}$，纵轴 TPR $=\frac{TP}{TP+FN}$，遍历阈值画出的曲线。
- **AUC 的概率含义**：随机取一个正样本和一个负样本，模型给正样本打分高于负样本的概率。
- **秩公式**（手算/手写用）：
  $$\text{AUC}=\frac{\sum_{i\in\text{正}}\text{rank}_i - \frac{M(M+1)}{2}}{M\cdot N}$$
  $M$ 正样本数，$N$ 负样本数，rank 是按分数**升序**的排名（从 1 开始）。
  **关键细节**：分数相同的样本必须取**平均秩**，否则结果是错的 —— 手写 AUC 时大多数人栽在这里，实现见 [09 §3](09-coding-ml-scratch.md)。

**Q：AUC 为什么对类别不平衡不敏感？**
TPR 只用正样本算，FPR 只用负样本算，二者都是"组内比例"，改变正负样本比例不改变各自的比例。而 Precision 的分母 $TP+FP$ 跨了两类，所以 PR 曲线对不平衡敏感。

**Q：什么时候该用 PR-AUC 而不是 ROC-AUC？**
极度不平衡且我们只关心正类时（如 0.1% 的欺诈）。此时负样本极多，FPR 分母巨大，即使 FP 增加很多 FPR 也几乎不动，ROC 看起来虚高；PR 曲线能真实反映精度下降。

**Q：AUC 的缺陷？**
① 只看排序不看绝对值 —— 广告出价需要**校准**（calibration）后的真实概率，此时要看 LogLoss/COPC；② 全局 AUC 掩盖用户内部的排序质量 → 推荐系统用 **GAUC**（按用户分组算 AUC 再加权平均）。权重用**曝光数**还是**点击数**两种口径都有人用，阿里 DIN 原文用的是曝光数；组内只有单一类别时该组 AUC 无定义，要跳过而不是记 0。

### 5.3 回归指标

MSE（对异常值敏感）、MAE（鲁棒但不可导于 0）、RMSE、Huber（小误差平方大误差线性，兼顾）、MAPE（相对误差，真值为 0 时爆炸）、$R^2$。

---

## 6. 特征工程

### 6.1 类别特征处理

| 方法 | 说明 | 风险 |
|---|---|---|
| One-Hot | 维度爆炸但无序假设正确 | 高基数不可用 |
| Label Encoding | 引入了虚假的序关系 | 只适合树模型 |
| Target Encoding | 用该类别下 label 均值 | **目标泄漏**，必须用 K-fold 或加平滑：$\frac{n\cdot\bar y_c+m\cdot\bar y}{n+m}$ |
| Embedding | 深度模型标配，学出语义 | 需要足够数据 |
| Hashing Trick | 固定维度，省内存 | 哈希冲突 |
| Frequency Encoding | 用出现频次 | 简单有效 |

### 6.2 连续特征处理

归一化（Min-Max，受异常值影响）vs 标准化（Z-score，更常用）vs 分位数变换 vs 对数变换（长尾数据，如价格、播放量）vs 分桶离散化。

**哪些模型需要归一化？** 需要：基于距离（KNN、K-Means、SVM）、基于梯度（LR、NN，加速收敛）、有正则项的（否则惩罚不公平）、PCA。不需要：树模型（只看划分点的相对顺序）。

### 6.3 特征选择

- **Filter**：方差过滤、卡方检验、互信息、相关系数。快但不考虑模型。
- **Wrapper**：递归特征消除（RFE）。准但慢。
- **Embedded**：L1 正则、树模型的 feature importance、SHAP。实用首选。

**注意**：树模型的默认 feature_importance（基于分裂增益）**偏向高基数特征**，更可靠的是 permutation importance 或 SHAP。

---

## 7. 常见工程问题

### 7.1 过拟合怎么办（列举题，至少说 8 条）

数据侧：增加数据、数据增强、清洗噪声标签。
模型侧：降低复杂度（减层/减深度/减参数）、正则化 L1/L2、Dropout、BatchNorm（有轻微正则效果）、参数共享。
训练侧：早停、学习率调度、标签平滑、Mixup/CutMix。
集成侧：Bagging、模型平均、SWA。
验证侧：交叉验证选超参、确保训练/验证分布一致（时序数据要按时间切分，不能随机切）。

### 7.2 数据不平衡（四个层面回答）

1. **数据层面**：过采样（SMOTE 在特征空间插值生成新样本，注意会在噪声附近生成脏数据）、欠采样（EasyEnsemble 多次欠采样做集成，避免信息损失）。
2. **算法层面**：class_weight / `pos_weight`、Focal Loss $-(1-p_t)^\gamma\log p_t$（降低易分样本权重）。
3. **阈值层面**：不用 0.5，用 PR 曲线找最优阈值；或做概率校准。
4. **指标层面**：别看 accuracy，看 PR-AUC / F1 / 召回@固定精度。

**追问**："采样后概率会失真吗？" 会。负采样率为 $r$ 时，需要校准：$p = \frac{p'}{p'+(1-p')/r}$〔Facebook, ADKDD 2014〕。广告 CTR 场景必做。

### 7.3 数据泄漏（Data Leakage）

典型来源：① 用全量数据做归一化/填充（应只用训练集统计量）；② Target Encoding 不做 K-fold；③ 时序数据随机切分导致用未来预测过去；④ 特征里混入了标签的衍生物（如"是否退款"预测"是否投诉"）；⑤ 同一用户的样本同时出现在训练和测试集。

### 7.4 线上线下不一致

原因：特征穿越（离线用了线上取不到的未来特征）、训练与推理的特征计算逻辑不一致（**特征工程代码应共用一套**）、样本分布漂移、延迟反馈（转化回流晚导致标签不准）。
排查方法：对同一批请求 dump 线上特征与离线特征做逐字段 diff。

---

## 延伸阅读

- *XGBoost: A Scalable Tree Boosting System*（Chen & Guestrin, KDD 2016）—— 二阶泰勒展开、结构分数、分裂增益公式的原始出处
- *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*（Ke et al., NeurIPS 2017）—— GOSS 与 EFB
- *CatBoost: unbiased boosting with categorical features*（Prokhorenkova et al., NeurIPS 2018）—— Ordered Target Statistics 如何避免目标泄漏
- *Greedy Function Approximation: A Gradient Boosting Machine*（Friedman, 2001）—— GBDT 原始论文
- *Random Forests*（Breiman, 2001）—— 两重随机性与 OOB
- *SMOTE: Synthetic Minority Over-sampling Technique*（Chawla et al., 2002）
- *The Relationship Between Precision-Recall and ROC Curves*（Davis & Goadrich, ICML 2006）—— 不平衡数据下该看 PR 还是 ROC
- *A Unified Approach to Interpreting Model Predictions*（Lundberg & Lee, NeurIPS 2017）—— SHAP

---

## 自测清单

- [ ] 能推 LR 梯度并解释为什么不用 MSE
- [ ] 能说清 SVM 为什么对偶、核函数解决什么问题
- [ ] 能写出 XGBoost 的分裂增益公式
- [ ] 能讲 LightGBM 的三大优化（Histogram / Leaf-wise / GOSS+EFB）
- [ ] 能说清 RF 和 GBDT 在偏差方差上的分工
- [ ] 能解释 AUC 的概率含义 + 秩公式 + 为什么对不平衡稳健
- [ ] 能讲 GAUC 及其动机
- [ ] 能一口气列 8 种以上防过拟合手段
- [ ] 能从四个层面回答数据不平衡
- [ ] 能举出 5 种数据泄漏的场景
