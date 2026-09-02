"""面试手撕：机器学习/深度学习核心组件的 numpy 实现。

每个函数都是面试现场可以在 10 分钟内写完的量级。
运行 `python code/ml_components.py` 会跑一遍自测。
"""
import numpy as np


# ---------------------------------------------------------------- 基础算子
def softmax(x, axis=-1):
    """数值稳定的 softmax：减去最大值防止 exp 上溢。"""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(x):
    """数值稳定的 sigmoid：正负分支分别处理，避免 exp 溢出。"""
    pos = x >= 0
    out = np.empty_like(x, dtype=np.float64)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """LayerNorm：在最后一维上归一化，每个样本独立。"""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    if gamma is not None:
        x_hat = x_hat * gamma
    if beta is not None:
        x_hat = x_hat + beta
    return x_hat


def rms_norm(x, gamma=None, eps=1e-6):
    """RMSNorm：不减均值，只除以均方根。LLaMA 系使用。"""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    out = x / rms
    return out * gamma if gamma is not None else out


def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """BatchNorm（训练模式）：x 形状 (N, D)，在 batch 维度上统计。"""
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta


# ---------------------------------------------------------------- 注意力
def scaled_dot_product_attention(Q, K, V, mask=None):
    """单头注意力。Q:(..., n, d_k)  K:(..., m, d_k)  V:(..., m, d_v)

    mask: 可广播到 (..., n, m)，True 表示"需要屏蔽"。
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)   # 除以 sqrt(d_k) 防止 softmax 饱和
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    attn = softmax(scores, axis=-1)
    return attn @ V, attn


def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=False):
    """多头自注意力。x:(B, T, D)，权重均为 (D, D)。"""
    B, T, D = x.shape
    d_head = D // n_heads

    def split(t):                       # (B,T,D) -> (B,h,T,d_head)
        return t.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)

    Q, K, V = split(x @ Wq), split(x @ Wk), split(x @ Wv)

    mask = None
    if causal:                          # 下三角因果 mask，屏蔽未来
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)

    out, attn = scaled_dot_product_attention(Q, K, V, mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)     # 拼回多头
    return out @ Wo, attn


def rope(x, base=10000.0):
    """旋转位置编码 RoPE。x: (B, T, D)，D 必须为偶数。"""
    B, T, D = x.shape
    half = D // 2
    theta = base ** (-np.arange(0, half) * 2.0 / D)      # (half,)
    pos = np.arange(T)[:, None]                          # (T,1)
    ang = pos * theta[None, :]                           # (T, half)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin,
                           x1 * sin + x2 * cos], axis=-1)


# ---------------------------------------------------------------- 损失函数
def cross_entropy(logits, labels):
    """多分类交叉熵。logits:(N,C) labels:(N,) 整数标签。"""
    p = softmax(logits, axis=-1)
    n = logits.shape[0]
    return -np.mean(np.log(p[np.arange(n), labels] + 1e-12))


def focal_loss(probs, labels, alpha=0.25, gamma=2.0):
    """二分类 Focal Loss。probs:(N,) 预测为正类的概率。"""
    pt = np.where(labels == 1, probs, 1 - probs)
    at = np.where(labels == 1, alpha, 1 - alpha)
    return -np.mean(at * (1 - pt) ** gamma * np.log(pt + 1e-12))


# ---------------------------------------------------------------- 评估指标
def auc_score(y_true, y_score):
    """秩公式计算 AUC，正确处理并列分数（取平均秩）。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_score)
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:                        # 并列分数取平均秩
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0       # 秩从 1 开始
        ranks[order[i:j + 1]] = avg
        i = j + 1
    M = int(np.sum(y_true == 1))
    N = n - M
    if M == 0 or N == 0:
        return float("nan")
    return (ranks[y_true == 1].sum() - M * (M + 1) / 2.0) / (M * N)


def iou(box_a, box_b):
    """两个框的 IoU，格式 (x1, y1, x2, y2)。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, thresh=0.5):
    """非极大值抑制，返回保留框的下标。boxes:(N,4)"""
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ious = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][ious <= thresh]
    return keep


# ---------------------------------------------------------------- 经典模型
class LogisticRegression:
    """手写逻辑回归（批量梯度下降）。"""

    def __init__(self, lr=0.1, n_iter=1000, l2=0.0):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = sigmoid(X @ self.w + self.b)
            err = p - y                                  # 梯度就是 (预测 - 真实)
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def kmeans(X, k, n_iter=100, seed=0):
    """K-Means（含 K-Means++ 初始化）。返回 (labels, centers)。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # ---- K-Means++ 初始化：距离越远越可能被选中
    centers = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centers.append(X[rng.choice(n, p=probs)])
    centers = np.array(centers, dtype=float)

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dist = ((X[:, None, :] - centers[None]) ** 2).sum(-1)   # (n, k)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):                  # 收敛
            break
        labels = new_labels
        for j in range(k):
            if np.any(labels == j):
                centers[j] = X[labels == j].mean(axis=0)
    return labels, centers


def pca(X, k):
    """PCA：中心化 -> SVD -> 取前 k 个主成分。返回 (降维结果, 主成分, 解释方差比)。"""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k]
    var = (S ** 2) / (len(X) - 1)
    return Xc @ comps.T, comps, var[:k] / var.sum()


# ---------------------------------------------------------------- 采样
def reservoir_sampling(stream, k, seed=0):
    """蓄水池抽样：从未知长度的数据流中等概率取 k 个。"""
    rng = np.random.default_rng(seed)
    pool = []
    for i, item in enumerate(stream):
        if i < k:
            pool.append(item)
        else:
            j = rng.integers(0, i + 1)
            if j < k:
                pool[j] = item
    return pool


def weighted_sample(weights, rng=None):
    """按权重随机采样一个下标：前缀和 + 二分，O(log n)。"""
    import bisect
    rng = rng or np.random.default_rng(0)
    prefix = np.cumsum(weights)
    target = rng.random() * prefix[-1]
    return bisect.bisect_right(prefix, target)


# ---------------------------------------------------------------- 自测
def _test():
    rng = np.random.default_rng(0)

    # softmax / sigmoid 数值稳定性
    assert np.allclose(softmax(np.array([1000.0, 1000.0])), [0.5, 0.5])
    assert np.isfinite(sigmoid(np.array([-1000.0, 1000.0]))).all()

    # LayerNorm 输出均值 0 方差 1
    x = rng.normal(size=(4, 8))
    ln = layer_norm(x)
    assert abs(ln.mean()) < 1e-6 and abs(ln.std() - 1) < 1e-3

    # 注意力：因果 mask 下第 0 个位置只能看到自己
    B, T, D, H = 2, 5, 8, 2
    xs = rng.normal(size=(B, T, D))
    Ws = [rng.normal(size=(D, D)) * 0.1 for _ in range(4)]
    out, attn = multi_head_attention(xs, *Ws, n_heads=H, causal=True)
    assert out.shape == (B, T, D)
    assert np.allclose(attn[:, :, 0, 1:], 0)             # 未来位置被屏蔽
    assert np.allclose(attn.sum(-1), 1.0)                # 每行注意力和为 1

    # RoPE：旋转不改变向量模长
    xr = rng.normal(size=(1, 6, 4))
    assert np.allclose(np.linalg.norm(rope(xr), axis=-1), np.linalg.norm(xr, axis=-1))

    # AUC：完美分类器 = 1.0，随机 = 0.5 附近，全并列 = 0.5
    assert auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc_score([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert abs(auc_score([0, 1, 1, 0], [0.1, 0.4, 0.35, 0.8]) - 0.5) < 1e-9

    # IoU / NMS
    assert abs(iou((0, 0, 2, 2), (1, 1, 3, 3)) - 1 / 7) < 1e-9
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (50, 50, 60, 60)]
    assert nms(boxes, [0.9, 0.8, 0.7], 0.5) == [0, 2]

    # 逻辑回归：线性可分数据应完美拟合
    Xp = np.vstack([rng.normal(2, 0.5, (50, 2)), rng.normal(-2, 0.5, (50, 2))])
    yp = np.r_[np.ones(50), np.zeros(50)]
    lr = LogisticRegression(lr=0.5, n_iter=800).fit(Xp, yp)
    assert (lr.predict(Xp) == yp).mean() == 1.0

    # KMeans：三个远离的簇应被正确分开
    Xk = np.vstack([rng.normal(c, 0.3, (40, 2)) for c in ([0, 0], [6, 6], [0, 6])])
    labels, centers = kmeans(Xk, 3, seed=1)
    assert len(set(labels[:40])) == 1 and len(set(labels[40:80])) == 1

    # PCA：第一主成分应捕获绝大部分方差
    Xa = rng.normal(size=(200, 1)) @ np.array([[3.0, 1.0]]) + rng.normal(0, 0.05, (200, 2))
    _, _, ratio = pca(Xa, 2)
    assert ratio[0] > 0.99

    # 蓄水池抽样：长期频率应接近均匀
    counts = np.zeros(10)
    for s in range(3000):
        for v in reservoir_sampling(range(10), 3, seed=s):
            counts[v] += 1
    assert counts.std() / counts.mean() < 0.05

    # Focal Loss 应小于普通 BCE（易分样本被降权）
    pr = np.array([0.9, 0.95, 0.85])
    lb = np.array([1, 1, 1])
    bce = -np.mean(np.log(pr))
    assert focal_loss(pr, lb, alpha=1.0) < bce

    print("all tests passed")


if __name__ == "__main__":
    _test()
