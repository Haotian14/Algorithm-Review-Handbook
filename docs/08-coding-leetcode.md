# 08 · 手撕代码：算法题

> 算法岗手撕的强度普遍是 **LeetCode 中等**，少数公司会出 Hard。真正拉开差距的不是难题，而是**能不能在 15 分钟内边讲思路边写出 bug-free 的中等题**。

## 面试手撕的正确姿势

1. **先复述题目和边界**：输入范围？有重复元素吗？空输入怎么办？—— 这一步就能拿印象分。
2. **先说思路再动手**：暴力解 → 复杂度 → 优化点 → 优化后复杂度。面试官通常会在这里给提示。
3. **写代码时讲出来**：沉默写码是大忌。
4. **写完主动测试**：自己举一个正常例子 + 一个边界例子跑一遍。
5. **说复杂度**：时间和空间都要说。

---

## 1. 必背模板

### 1.1 二分查找（三种形态，一定要写对）

```python
# 形态 1：查找目标值（闭区间）
def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# 形态 2：左边界（第一个 >= target 的位置），等价于 bisect_left
def lower_bound(nums, target):
    lo, hi = 0, len(nums)          # 左闭右开
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo                       # 可能等于 len(nums)

# 形态 3：二分答案（求满足 check 的最小值）
def binary_search_answer(lo, hi, check):
    while lo < hi:
        mid = (lo + hi) // 2
        if check(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

**记忆要点**：只要能把区间定义（闭区间 vs 左闭右开）想清楚，循环条件和更新方式就唯一确定了。**别背，要推。**

### 1.2 滑动窗口

```python
def sliding_window(s):
    from collections import defaultdict
    window = defaultdict(int)
    left = 0
    ans = 0
    for right, ch in enumerate(s):
        window[ch] += 1
        while <窗口不合法>:          # 收缩条件
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1
        ans = max(ans, right - left + 1)
    return ans
```

适用：连续子数组/子串的最长/最短/计数问题，且窗口具有**单调性**（扩大窗口使条件更难满足）。

### 1.3 回溯

```python
def backtrack(path, choices):
    if <满足结束条件>:
        res.append(path[:])          # 注意拷贝！
        return
    for i, c in enumerate(choices):
        if <剪枝条件>:               # 去重/合法性判断
            continue
        path.append(c)               # 做选择
        backtrack(path, <新的选择列表>)
        path.pop()                   # 撤销选择
```

**去重技巧**：先排序，然后 `if i > start and nums[i] == nums[i-1]: continue`（同层去重）。

### 1.4 并查集

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:      # 按秩合并
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.count -= 1
        return True
```

### 1.5 单调栈

```python
# 下一个更大元素
def next_greater(nums):
    n = len(nums)
    res = [-1] * n
    stack = []                       # 存下标，栈内元素单调递减
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            res[stack.pop()] = x
        stack.append(i)
    return res
```

用途：下一个更大/更小元素、柱状图最大矩形、接雨水、每日温度。

### 1.6 Dijkstra（堆优化）

```python
import heapq
def dijkstra(graph, start, n):
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue                 # 过期数据，跳过
        for v, w in graph[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist
```

### 1.7 拓扑排序（BFS/Kahn）

```python
from collections import deque
def topo_sort(n, edges):
    graph = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:               # u -> v
        graph[u].append(v)
        indeg[v] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else []   # 空表示有环
```

### 1.8 Trie

```python
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False

    def insert(self, word):
        node = self
        for ch in word:
            if ch not in node.children:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):
        node = self
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end
```

### 1.9 快速排序 + 快速选择（手撕高频）

```python
import random

def quick_sort(nums, lo, hi):
    if lo >= hi:
        return
    p = partition(nums, lo, hi)
    quick_sort(nums, lo, p - 1)
    quick_sort(nums, p + 1, hi)

def partition(nums, lo, hi):
    idx = random.randint(lo, hi)          # 随机化，防止有序数组退化成 O(n^2)
    nums[idx], nums[hi] = nums[hi], nums[idx]
    pivot = nums[hi]
    i = lo
    for j in range(lo, hi):
        if nums[j] < pivot:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    nums[i], nums[hi] = nums[hi], nums[i]
    return i

def find_kth_largest(nums, k):
    """第 k 大 = 下标 n-k 的元素，平均 O(n)"""
    target = len(nums) - k
    lo, hi = 0, len(nums) - 1
    while True:
        p = partition(nums, lo, hi)
        if p == target:
            return nums[p]
        elif p < target:
            lo = p + 1
        else:
            hi = p - 1
```

**必答追问**：快排最坏 $O(n^2)$（已排序数组 + 固定 pivot），随机化后期望 $O(n\log n)$；不稳定；平均空间 $O(\log n)$（递归栈）。

### 1.10 归并排序（求逆序对也靠它）

```python
def merge_sort(nums):
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    res, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:        # <= 保证稳定性
            res.append(left[i]); i += 1
        else:
            res.append(right[j]); j += 1
    res += left[i:]
    res += right[j:]
    return res
```

### 1.11 堆（Top-K）

```python
import heapq
def top_k(nums, k):
    """维护大小为 k 的小顶堆，O(n log k)"""
    heap = []
    for x in nums:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)
    return sorted(heap, reverse=True)
```

**Top-K 的三种解法对比**：全排序 $O(n\log n)$；堆 $O(n\log k)$（**数据流场景只能用这个**）；快速选择 $O(n)$ 平均（需要能随机访问全部数据）。

### 1.12 LRU Cache（超高频手撕）

```python
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
```

⚠️ 面试官常要求**手写双向链表 + 哈希表**版本，不许用 OrderedDict。核心：哈希表存 key → 节点，双向链表维护顺序，用虚拟头尾节点简化边界。

---

## 2. 动态规划专题

### 2.1 解题四步法

1. 定义状态 `dp[i]` / `dp[i][j]` 的**确切含义**（这一步错了后面全错）。
2. 写状态转移方程。
3. 确定初始条件和边界。
4. 确定遍历顺序（保证计算 `dp[i]` 时依赖项已算好）。

### 2.2 常见类型速查

| 类型 | 状态定义 | 代表题 |
|---|---|---|
| 线性 DP | `dp[i]` = 以 i 结尾/前 i 个的最优 | 最大子数组和、打家劫舍、LIS |
| 背包 | `dp[i][j]` = 前 i 个物品容量 j 的最优 | 0-1 背包、完全背包、分割等和子集 |
| 区间 DP | `dp[i][j]` = 区间 [i,j] 的最优 | 戳气球、最长回文子序列 |
| 两个序列 | `dp[i][j]` = A 前 i 个和 B 前 j 个 | 编辑距离、LCS |
| 状态机 DP | `dp[i][state]` | 买卖股票系列 |
| 树形 DP | 后序遍历返回多个状态 | 打家劫舍 III、二叉树最大路径和 |
| 数位 DP | 按位构造 + 记忆化 | 统计特殊数字 |

### 2.3 背包模板（必须分清）

```python
# 0-1 背包：每个物品最多一次 → 容量倒序遍历
dp = [0] * (W + 1)
for i in range(n):
    for w in range(W, weights[i] - 1, -1):
        dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

# 完全背包：每个物品无限次 → 容量正序遍历
for i in range(n):
    for w in range(weights[i], W + 1):
        dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
```

**为什么 0-1 背包要倒序？** 一维数组中 `dp[w - weights[i]]` 如果正序遍历就已经被本轮更新过了（相当于物品用了多次）。倒序保证读到的是上一轮的值。**这是面试官最爱追问的点。**

**组合数 vs 排列数**（零钱兑换 II vs 组合总和 IV）：
- 求**组合数**：外层物品，内层容量。
- 求**排列数**：外层容量，内层物品。

### 2.4 编辑距离（模板题）

```python
def min_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1],   # 替换
                                   dp[i-1][j],     # 删除
                                   dp[i][j-1])     # 插入
    return dp[m][n]
```

### 2.5 LIS 的 $O(n\log n)$ 解法

```python
import bisect
def length_of_lis(nums):
    tails = []          # tails[i] = 长度为 i+1 的上升子序列的最小结尾
    for x in nums:
        idx = bisect.bisect_left(tails, x)   # 严格递增用 bisect_left
        if idx == len(tails):
            tails.append(x)
        else:
            tails[idx] = x
    return len(tails)
```

⚠️ `tails` 本身**不是**某个真实的最长上升子序列，只是长度对应的最小结尾数组。

---

## 3. 高频题单（按题型）

### 3.1 必刷 60 题（校招算法岗覆盖率 >80%）

**数组/双指针/滑窗**
1. 两数之和 · 2. 三数之和 · 3. 盛最多水的容器 · 4. 接雨水 · 5. 无重复字符的最长子串 · 6. 最小覆盖子串 · 7. 找到字符串中所有字母异位词 · 8. 长度最小的子数组 · 9. 移动零 · 10. 合并区间 · 11. 轮转数组 · 12. 除自身以外数组的乘积

**哈希/前缀和**
13. 和为 K 的子数组 · 14. 最长连续序列 · 15. 字母异位词分组 · 16. 和可被 K 整除的子数组

**链表**
17. 反转链表（迭代 + 递归） · 18. K 个一组翻转链表 · 19. 环形链表 I/II · 20. 合并两个有序链表 · 21. 合并 K 个升序链表 · 22. 相交链表 · 23. 排序链表 · 24. 删除链表倒数第 N 个节点 · 25. 两数相加 · 26. LRU 缓存

**二叉树**
27. 中序/前序/后序遍历（迭代版） · 28. 层序遍历 · 29. 最大深度 · 30. 翻转二叉树 · 31. 对称二叉树 · 32. 二叉树的直径 · 33. 验证二叉搜索树 · 34. 二叉搜索树第 K 小 · 35. 最近公共祖先 · 36. 二叉树最大路径和 · 37. 从前序与中序构造二叉树 · 38. 路径总和 III · 39. 二叉树右视图 · 40. 二叉树展开为链表

**回溯**
41. 全排列（含去重） · 42. 子集 · 43. 组合总和 · 44. 括号生成 · 45. 单词搜索 · 46. 分割回文串 · 47. N 皇后

**二分**
48. 搜索旋转排序数组 · 49. 寻找两个正序数组的中位数 · 50. 在排序数组中查找元素的第一个和最后一个位置 · 51. 搜索二维矩阵 II

**图/BFS/DFS**
52. 岛屿数量 · 53. 腐烂的橘子 · 54. 课程表 I/II · 55. 实现 Trie

**DP**
56. 爬楼梯 · 57. 最大子数组和 · 58. 打家劫舍 · 59. 零钱兑换 · 60. 最长递增子序列 · 61. 编辑距离 · 62. 不同路径 · 63. 分割等和子集 · 64. 最长回文子串 · 65. 买卖股票的最佳时机 I/II/III

**栈/堆/贪心**
66. 有效的括号 · 67. 最小栈 · 68. 每日温度 · 69. 柱状图中最大的矩形 · 70. 数组中的第 K 个最大元素 · 71. 前 K 个高频元素 · 72. 数据流的中位数 · 73. 跳跃游戏 I/II · 74. 划分字母区间

**排序**
75. 手写快排 · 76. 手写归并 · 77. 手写堆排序 · 78. 颜色分类（荷兰国旗）

### 3.2 算法岗特色题（比纯开发岗更常出）

- **蓄水池抽样**（数据流采样）
- **随机数生成**（rand5 → rand7、按权重随机 = 前缀和 + 二分）
- **Top-K 频率统计**（海量数据 + 内存受限 → 分治哈希 + 堆）
- **矩阵运算**（旋转矩阵、螺旋遍历、矩阵乘法）
- **数值计算**（Pow(x,n) 快速幂、Sqrt(x) 二分/牛顿法）
- **概率/期望题**（见 [01 数学](01-math-basics.md)）
- **手写 ML 组件**（见 [09](09-coding-ml-scratch.md)，这才是算法岗的真正区分点）

---

## 4. 复杂度速查

| 结构/算法 | 时间 | 空间 |
|---|---|---|
| 快排 | 平均 $O(n\log n)$，最坏 $O(n^2)$ | $O(\log n)$ |
| 归并 | $O(n\log n)$ | $O(n)$ |
| 堆排序 | $O(n\log n)$ | $O(1)$ |
| 建堆 | $O(n)$（不是 $n\log n$！） | — |
| 哈希表 | 平均 $O(1)$，最坏 $O(n)$ | $O(n)$ |
| 二分 | $O(\log n)$ | $O(1)$ |
| Dijkstra（堆） | $O((V+E)\log V)$ | $O(V)$ |
| 并查集（路径压缩+按秩） | 近似 $O(\alpha(n))\approx O(1)$ | $O(n)$ |
| Trie | 插入/查询 $O(L)$ | $O(\text{总字符数}\times\Sigma)$ |

**面试常见追问**：
- 稳定排序有哪些？归并、插入、冒泡、计数、基数。快排、堆排、选择排序**不稳定**。
- 为什么 Python 的 `sort` / Java 的 `Arrays.sort(Object[])` 用 TimSort？因为它是稳定的且对部分有序数据接近 $O(n)$。
- 建堆为什么是 $O(n)$？自底向上建堆时，深度为 $h$ 的节点只有 $n/2^{h+1}$ 个，加权求和 $\sum h\cdot n/2^{h+1}$ 收敛到 $O(n)$。

---

## 5. Python 面试提速技巧

```python
from collections import defaultdict, Counter, deque, OrderedDict
import heapq, bisect, functools, itertools, math

Counter(nums).most_common(k)          # Top-K 频率
heapq.nlargest(k, nums)               # 直接取前 k 大
bisect.bisect_left(arr, x)            # 左边界二分
deque(maxlen=k)                       # 固定长度队列
functools.lru_cache(maxsize=None)     # 记忆化搜索（DP 递归版）
itertools.accumulate(nums)            # 前缀和
math.inf / -math.inf                  # 边界初始化
sorted(items, key=lambda x: (-x[1], x[0]))   # 多级排序
```

⚠️ 注意：`heapq` 是小顶堆，求最大值时取负数入堆；递归深度默认 1000，深递归要 `sys.setrecursionlimit`。

---

## 自测清单

- [ ] 三种二分能不看模板独立推出来
- [ ] 手写快排 + 快速选择，能说清随机化的必要性
- [ ] 手写归并排序，知道稳定性从哪来
- [ ] 手写 LRU（双向链表 + 哈希版）
- [ ] 能解释 0-1 背包为什么倒序遍历
- [ ] 能区分背包问题中组合数和排列数的遍历顺序
- [ ] 能写 LIS 的 $O(n\log n)$ 解法并解释 tails 的含义
- [ ] 能写并查集（路径压缩 + 按秩合并）
- [ ] 能说清建堆为什么是 $O(n)$
- [ ] 必刷题单里的中等题能在 15 分钟内 bug-free 写完
