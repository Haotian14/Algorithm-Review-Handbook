# 09 · 手撕代码：ML / DL 组件

> **这一章是算法岗区别于开发岗的核心考点。** 相当一部分算法岗面试会让你手写 attention、AUC、KMeans、NMS 这类组件（主观判断，无统计支撑）。
>
> 本章所有代码都在 `code/ml_components.py` 中，带自测用例，`python code/ml_components.py` 可直接跑通。

## 出现频率排行

> 排序是**主观判断**，不是统计结果。

| 组件 | 频率 | 常出现的岗位 |
|---|---|---|
| Self-Attention / MHA | ★★★ | LLM / NLP / 通用 |
| AUC | ★★★ | 推荐 / 广告 / 通用 |
| K-Means | ★★★ | 通用 |
| NMS + IoU | ★★★ | CV |
| Softmax / 交叉熵（数值稳定） | ★★★ | 通用 |
| 逻辑回归（含梯度推导） | ★★★ | 通用 |
| LayerNorm / BatchNorm | ★★ | 通用 |
| 卷积（im2col 或直接三重循环） | ★★ | CV |
| PCA | ★★ | 通用 |
| Focal Loss | ★★ | CV |
| RoPE | ★★ | LLM |
| 蓄水池抽样 / 按权重采样 | ★★ | 通用 |
| Dropout（含 inverted） | ★ | 通用 |

---

## 1. 数值稳定的基础算子

**面试官第一个看的就是你有没有减最大值。** 忘了这一步基本就是"写过但没理解"。

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)   # 关键：防止 exp 上溢
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def sigmoid(x):
    """正负分支分别处理，避免 exp(大正数) 溢出。"""
    pos = x >= 0
    out = np.empty_like(x, dtype=np.float64)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def cross_entropy(logits, labels):
    """logits:(N,C)  labels:(N,) 整数标签"""
    p = softmax(logits, axis=-1)
    n = logits.shape[0]
    return -np.mean(np.log(p[np.arange(n), labels] + 1e-12))
```

**追问**：为什么实际框架用 `log_softmax` 而不是 `log(softmax(x))`？
因为 `softmax` 的结果可能下溢到 0，再取 log 得到 `-inf`。`log_softmax(x) = x - max - log(sum(exp(x-max)))` 全程不需要先算出概率值，更稳定。

---

## 2. Self-Attention / Multi-Head Attention（最高频）

### 2.1 NumPy 版

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q:(...,n,d_k) K:(...,m,d_k) V:(...,m,d_v)；mask 中 True 表示需要屏蔽"""
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)   # 除 sqrt(d_k) 防止 softmax 饱和
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    attn = softmax(scores, axis=-1)
    return attn @ V, attn


def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=False):
    """x:(B,T,D)，四个权重均为 (D,D)"""
    B, T, D = x.shape
    d_head = D // n_heads

    def split(t):                                   # (B,T,D) -> (B,h,T,d_head)
        return t.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)

    Q, K, V = split(x @ Wq), split(x @ Wk), split(x @ Wv)

    mask = None
    if causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)   # 上三角为 True = 屏蔽未来

    out, attn = scaled_dot_product_attention(Q, K, V, mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)       # 多头拼回
    return out @ Wo, attn
```

### 2.2 PyTorch 版（更常被要求）

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)   # 融合成一个矩阵更快
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B,T,D) -> (B,h,T,d_k)
        q, k, v = [t.view(B, T, self.h, self.d_k).transpose(1, 2) for t in (q, k, v)]

        scores = q @ k.transpose(-2, -1) / self.d_k ** 0.5       # (B,h,T,T)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = self.drop(scores.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


# 因果 mask（下三角以上为 True 表示屏蔽）
def causal_mask(T, device=None):
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
```

### 2.3 带 KV Cache 的增量解码（进阶，问到就是加分项）

```python
class CachedAttention(nn.Module):
    """自回归解码：每步只算新 token 的 q，k/v 拼接到缓存上。"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, cache=None):
        B, T, D = x.shape                       # 解码阶段 T == 1
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.h, self.d_k).transpose(1, 2) for t in (q, k, v)]

        if cache is not None:
            k = torch.cat([cache['k'], k], dim=2)    # 沿序列维拼接
            v = torch.cat([cache['v'], v], dim=2)
        new_cache = {'k': k, 'v': v}

        # 解码时 q 只有一个位置，天然只能看到已缓存的历史，无需 causal mask
        attn = (q @ k.transpose(-2, -1) / self.d_k ** 0.5).softmax(-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out), new_cache
```

**必答追问**：
- 为什么除以 $\sqrt{d_k}$ → 见 [04 §1.1](04-llm.md)。
- mask 为什么填 `-inf` 而不是 0？因为要在 softmax **之前**填，softmax 后才变成 0；填 0 会让被屏蔽位置得到相当大的权重。
- 实现中用 `-1e9` 而不是 `-inf` 是为了在 fp16 下避免 NaN（整行全被 mask 时 `-inf` 会产生 `0/0`）。

---

## 3. AUC（推荐/广告必考）

```python
def auc_score(y_true, y_score):
    """秩公式：AUC = (sum(正样本的秩) - M(M+1)/2) / (M*N)，正确处理并列分数。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_score)
    order = np.argsort(y_score, kind="mergesort")   # 升序
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:                                    # 并列分数取平均秩
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0                   # 秩从 1 开始
        ranks[order[i:j + 1]] = avg
        i = j + 1
    M = int(np.sum(y_true == 1)); N = n - M
    if M == 0 or N == 0:
        return float("nan")
    return (ranks[y_true == 1].sum() - M * (M + 1) / 2.0) / (M * N)
```

**面试的两个坑**：① **并列分数必须取平均秩**，否则结果不对（大部分人写的版本会在这里挂）；② 单类样本时 AUC 无定义，要处理。

**GAUC（推荐场景，常追问）**：

一个能说明它为什么必要的反例 —— 用户 a 打分整体高、正样本多，用户 b 打分整体低、正样本少，
两组**组内**的负样本分数都压过正样本：

| 用户 | label | score |
|---|---|---|
| a | 1, 1, 1, 0 | 0.90, 0.85, 0.80, **0.95** |
| b | 1, 0, 0, 0 | 0.20, **0.35, 0.30, 0.25** |

全局 AUC = 9/16 ≈ **0.56**，看着还行；但两组的组内 AUC 都是 0，**GAUC = 0**。
模型学到的只是「a 这类用户分数高」，对同一个用户该推哪条完全排反了 —— 这正是线上体验崩掉而离线 AUC 不报警的典型形态。

```python
def gauc(y_true, y_score, user_ids):
    """按用户分组算 AUC 再加权平均。组内只有单一类别时该组无定义，必须跳过而不是记 0。"""
    from collections import defaultdict
    groups = defaultdict(lambda: ([], []))
    for u, y, s in zip(user_ids, y_true, y_score):
        groups[u][0].append(y); groups[u][1].append(s)
    total_w, total = 0.0, 0.0
    for ys, ss in groups.values():
        a = auc_score(ys, ss)
        if np.isnan(a):          # 组内只有一类，跳过
            continue
        w = len(ys)              # 曝光数加权（DIN 原文口径）；也有用点击数 sum(ys) 的
        total += a * w; total_w += w
    return total / total_w if total_w else float("nan")
```

---

## 4. IoU 与 NMS（CV 必考）

```python
def iou(a, b):
    """框格式 (x1, y1, x2, y2)"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)       # 关键：不相交时截断为 0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, thresh=0.5):
    """返回保留框的下标，向量化实现。"""
    boxes = np.asarray(boxes, dtype=float); scores = np.asarray(scores, dtype=float)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ious = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][ious <= thresh]               # 只保留 IoU 低的
    return keep
```

**追问：Soft-NMS 怎么改？** 不直接删除，而是衰减分数：
```python
scores[order[1:]] *= np.exp(-ious ** 2 / sigma)   # 高斯衰减版
```
然后重新排序继续，最后按分数阈值过滤。

**追问：多类别 NMS？** 按类别分别做；工程上常用 trick：给每个类别的框坐标加上 `cls_id * offset`（一个大于图像尺寸的偏移），让不同类的框永远不相交，就可以一次性做完。

---

## 5. K-Means（含 K-Means++）

```python
def kmeans(X, k, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # K-Means++ 初始化：以正比于 D(x)^2 的概率选下一个中心
    centers = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centers.append(X[rng.choice(n, p=probs)])
    centers = np.array(centers, dtype=float)

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dist = ((X[:, None, :] - centers[None]) ** 2).sum(-1)    # (n,k)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):                   # 收敛判据
            break
        labels = new_labels
        for j in range(k):
            if np.any(labels == j):                              # 防止空簇除零
                centers[j] = X[labels == j].mean(axis=0)
    return labels, centers
```

**面试要点**：必须写 K-Means++ 初始化（或至少说出来）；必须处理**空簇**；必须有收敛判据；说清复杂度 $O(nkdt)$。

---

## 6. 逻辑回归（含梯度推导）

```python
class LogisticRegression:
    def __init__(self, lr=0.1, n_iter=1000, l2=0.0):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.n_iter):
            p = sigmoid(X @ self.w + self.b)
            err = p - y                                  # 交叉熵梯度 = 预测 - 真实
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)
```

写的时候**边写边说**："交叉熵对 logit 的梯度正好是 $\hat y - y$，因为 sigmoid 的导数被交叉熵的 $1/\hat y$ 约掉了" —— 见 [02 §1.2](02-machine-learning.md)。

---

## 7. 归一化层

```python
def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """最后一维归一化，每个样本独立 —— 与 batch 无关。"""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    if gamma is not None: x_hat = x_hat * gamma
    if beta is not None:  x_hat = x_hat + beta
    return x_hat

def rms_norm(x, gamma=None, eps=1e-6):
    """RMSNorm：不减均值，省一次 reduce。LLaMA 系使用。"""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    out = x / rms
    return out * gamma if gamma is not None else out

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """训练模式；推理模式要换成 running_mean / running_var。"""
    mu, var = x.mean(axis=0), x.var(axis=0)               # 注意是在 batch 维度
    return gamma * (x - mu) / np.sqrt(var + eps) + beta
```

**追问：BN 的反向传播？** 要点是 $\mu$ 和 $\sigma^2$ 都依赖 $x$，梯度有三条路径：
$$\frac{\partial L}{\partial x_i}=\frac{1}{\sqrt{\sigma^2+\epsilon}}\left(\frac{\partial L}{\partial \hat x_i}-\frac1N\sum_j\frac{\partial L}{\partial \hat x_j}-\frac{\hat x_i}{N}\sum_j\frac{\partial L}{\partial \hat x_j}\hat x_j\right)$$

---

## 8. RoPE 旋转位置编码

```python
def rope(x, base=10000.0):
    """x:(B,T,D)，D 为偶数。前后半维配对旋转（HuggingFace 风格）。"""
    B, T, D = x.shape
    half = D // 2
    theta = base ** (-np.arange(0, half) * 2.0 / D)       # 各维不同频率
    ang = np.arange(T)[:, None] * theta[None, :]          # (T, half)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin,
                           x1 * sin + x2 * cos], axis=-1)
```

自测点：**旋转是正交变换，不改变向量的模长** —— `np.linalg.norm(rope(x)) == np.linalg.norm(x)`。这是验证实现正确性的最快方法。

---

## 9. Focal Loss

```python
def focal_loss(probs, labels, alpha=0.25, gamma=2.0):
    """二分类版本。probs 是预测为正类的概率。"""
    pt = np.where(labels == 1, probs, 1 - probs)          # 预测正确类别的概率
    at = np.where(labels == 1, alpha, 1 - alpha)
    return -np.mean(at * (1 - pt) ** gamma * np.log(pt + 1e-12))
```

PyTorch 版（数值更稳，直接吃 logits）：
```python
def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    at = alpha * targets + (1 - alpha) * (1 - targets)
    return (at * (1 - pt) ** gamma * bce).mean()
```

---

## 10. 卷积（手写 im2col 版）

```python
def conv2d(x, w, b=None, stride=1, padding=0):
    """x:(N,C,H,W)  w:(F,C,kh,kw) —— im2col + 矩阵乘，比三重循环快得多。"""
    N, C, H, W = x.shape
    F_, _, kh, kw = w.shape
    x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1

    cols = np.zeros((N, C * kh * kw, out_h * out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]
            cols[:, :, i * out_w + j] = patch.reshape(N, -1)

    out = w.reshape(F_, -1) @ cols                # (F, C*kh*kw) @ (N, C*kh*kw, L)
    out = out.transpose(1, 0, 2).reshape(N, F_, out_h, out_w)
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out
```

**必答**：输出尺寸公式 $\lfloor\frac{H+2p-k}{s}\rfloor+1$、参数量 $k^2C_{in}C_{out}+C_{out}$、FLOPs $\approx 2k^2C_{in}C_{out}H_{out}W_{out}$。

---

## 11. PCA

```python
def pca(X, k):
    """中心化 -> SVD -> 取前 k 个主成分。直接对 X 做 SVD 比构造协方差矩阵更稳。"""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k]                                # 主成分方向（行向量）
    var = (S ** 2) / (len(X) - 1)                 # 各主成分的方差
    return Xc @ comps.T, comps, var[:k] / var.sum()
```

---

## 12. 采样

```python
def reservoir_sampling(stream, k, seed=0):
    """未知长度数据流中等概率取 k 个，空间 O(k)。"""
    rng = np.random.default_rng(seed)
    pool = []
    for i, item in enumerate(stream):
        if i < k:
            pool.append(item)
        else:
            j = rng.integers(0, i + 1)            # [0, i]
            if j < k:
                pool[j] = item
    return pool


def weighted_sample(weights, rng=None):
    """按权重采样一个下标：前缀和 + 二分，O(log n)。"""
    import bisect
    rng = rng or np.random.default_rng(0)
    prefix = np.cumsum(weights)
    return bisect.bisect_right(prefix, rng.random() * prefix[-1])
```

**蓄水池抽样的正确性证明**（会被要求）：
对第 $i$ 个元素（$i>k$），它被选入的概率是 $k/i$；此后第 $j$ 轮（$j>i$）它不被替换的概率是 $1-\frac{k}{j}\cdot\frac{1}{k}=\frac{j-1}{j}$。
所以最终留下的概率 $=\frac{k}{i}\prod_{j=i+1}^{n}\frac{j-1}{j}=\frac{k}{i}\cdot\frac{i}{n}=\frac{k}{n}$。

---

## 13. Dropout（Inverted）

```python
def dropout(x, p=0.5, training=True, rng=None):
    """Inverted Dropout：训练时除以 (1-p)，推理时什么都不做。"""
    if not training or p == 0:
        return x
    rng = rng or np.random.default_rng()
    mask = (rng.random(x.shape) >= p) / (1.0 - p)     # 缩放放在训练侧
    return x * mask
```

---

## 14. 其他可能被问到的

```python
# 手写 Adam（一步更新）
def adam_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    m_hat = m / (1 - b1 ** t)                # 偏差修正
    v_hat = v / (1 - b2 ** t)
    p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
    return p, m, v

# 手写 InfoNCE（对比学习，CLIP/SimCSE 都用它）
def info_nce(z1, z2, tau=0.07):
    """z1,z2:(N,D) 已归一化的两个视角的表示；对角线是正样本对。"""
    logits = z1 @ z2.T / tau
    labels = np.arange(len(z1))
    return 0.5 * (cross_entropy(logits, labels) + cross_entropy(logits.T, labels))

# 手写 Beam Search（生成任务）
def beam_search(step_fn, start, beam=3, max_len=10, eos=None):
    """step_fn(seq) -> (candidate_tokens, log_probs)"""
    beams = [(0.0, [start])]
    for _ in range(max_len):
        cand = []
        for score, seq in beams:
            if eos is not None and seq[-1] == eos:
                cand.append((score, seq)); continue
            toks, lps = step_fn(seq)
            for t, lp in zip(toks, lps):
                cand.append((score + lp, seq + [t]))
        beams = sorted(cand, key=lambda x: -x[0])[:beam]
    return beams[0][1]
```

---

## 自测清单

- [ ] 能默写数值稳定的 softmax / sigmoid / cross_entropy
- [ ] 能在 10 分钟内写出 PyTorch 版 MHA（含 causal mask）
- [ ] 能写带 KV Cache 的增量解码
- [ ] 能写 AUC 并**正确处理并列分数**，能写 GAUC
- [ ] 能写 IoU + NMS，并说出 Soft-NMS 和多类别 NMS 的改法
- [ ] 能写 K-Means++ 并处理空簇
- [ ] 能写 LR 并解释梯度为什么是 $\hat y - y$
- [ ] 能写 LayerNorm / RMSNorm / BatchNorm 并说清差异
- [ ] 能写 RoPE 并用模长不变性验证
- [ ] 能写 Focal Loss（numpy 和 PyTorch 两版）
- [ ] 能写卷积并算出输出尺寸/参数量/FLOPs
- [ ] 能写蓄水池抽样并证明其正确性
- [ ] 能写 Adam 一步更新和 InfoNCE
