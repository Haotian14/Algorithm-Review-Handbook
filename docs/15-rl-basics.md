# 15 · 强化学习基础

> 校招很少有纯 RL 岗，但 RL 的基本功现在是**大模型方向的必修课** —— RLHF、GRPO、推理模型的可验证奖励 RL 全都建立在这套框架上。
> [04 §4.2](04-llm.md) 讲了 RLHF 有哪四个模型，这一章补上**为什么是这四个、PPO 到底在优化什么**。
> 推荐和广告方向则会在序列决策、探索利用、约束出价上碰到它。

## 本章高频考点

> 星级是**主观判断**，反映常见面经里的印象分布，不是统计数据。用来排复习优先级即可。

| 考点 | 频率 | 典型问法 |
|---|---|---|
| MDP 五要素与贝尔曼方程 | ★★ | 写出贝尔曼最优方程 |
| Q-learning vs SARSA | ★★ | on-policy 和 off-policy 的区别 |
| DQN 的两个关键技巧 | ★★ | 经验回放和目标网络各解决什么 |
| 策略梯度与 baseline | ★★★ | 为什么减 baseline 不改变期望却降方差 |
| **PPO 的 clip 目标** | ★★★ | ratio 是什么，为什么要裁剪 |
| **GAE** | ★★ | $\lambda$ 在权衡什么 |
| 与 RLHF / GRPO 的衔接 | ★★★ | 语言模型怎么建成 MDP，GRPO 省掉了什么 |
| 探索与利用 | ★★ | UCB / Thompson Sampling |
| 离线 RL 与 OPE | ★★ | 推荐场景为什么不能在线试错 |

---

## 1. MDP：把问题形式化

**五要素** $(\mathcal{S},\mathcal{A},P,R,\gamma)$：状态、动作、转移概率 $P(s'|s,a)$、奖励 $R(s,a)$、折扣因子 $\gamma\in[0,1)$。

**目标**：找策略 $\pi(a|s)$ 最大化期望累计折扣回报 $\mathbb{E}\left[\sum_{t}\gamma^t r_t\right]$。

**$\gamma$ 在干什么？** ① 数学上保证无限期回报收敛；② 业务上表达"未来的收益不如现在值钱"。
$\gamma$ 越大越看重长期 —— 推荐场景里 $\gamma$ 的选择直接对应"你有多在乎长期留存 vs 当下点击"。

**两个值函数**：
$$V^\pi(s)=\mathbb{E}_\pi\left[\sum_t \gamma^t r_t \mid s_0=s\right],\qquad Q^\pi(s,a)=\mathbb{E}_\pi\left[\sum_t\gamma^t r_t\mid s_0=s,a_0=a\right]$$

**贝尔曼期望方程**（当前值 = 即时奖励 + 折扣后的后继值）：
$$V^\pi(s)=\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)\left[R(s,a)+\gamma V^\pi(s')\right]$$

**贝尔曼最优方程**（把"按策略平均"换成"取最大"）：
$$Q^*(s,a)=\mathbb{E}_{s'}\left[r+\gamma\max_{a'}Q^*(s',a')\right]$$

**优势函数** $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$：这个动作比"这个状态下的平均水平"好多少。
**后面所有策略梯度方法的核心都是它** —— 我们要放大"比平均好"的动作的概率。

---

## 2. 值方法

### 2.1 Q-learning vs SARSA（经典对比题）

**Q-learning**（off-policy）：
$$Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]$$

**SARSA**（on-policy）：
$$Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma Q(s',a')-Q(s,a)\right]$$

**唯一的差别在 TD 目标里用 $\max_{a'}$ 还是用实际执行的 $a'$。**

| | Q-learning | SARSA |
|---|---|---|
| 类型 | **off-policy**：学的是最优策略，行为可以是别的（如 $\epsilon$-greedy） | **on-policy**：学的就是当前正在执行的策略 |
| 行为特点 | 更"激进"，会学出理论最优路径 | 更"保守"，会把探索时的失误算进来 |
| 经典例子 | 悬崖行走中走贴着悬崖的最短路 | 走离悬崖远一点的安全路（因为探索时会掉下去） |

**一句话**：**off-policy 学"最优怎么走"，on-policy 学"我这样走会怎样"。**
这个区分很重要 —— 它决定了能不能复用旧数据（off-policy 可以，on-policy 每次更新后旧数据就失效了）。

### 2.2 DQN

用神经网络近似 $Q(s,a;\theta)$，损失是 TD 误差的平方。但直接这么做会发散，DQN 靠两个技巧稳住：

1. **经验回放（Replay Buffer）**：把交互存起来随机采样。
   解决 ① 相邻样本高度相关（违反 SGD 的 i.i.d. 假设）；② 数据利用率低（一条数据用一次就扔）。
2. **目标网络（Target Network）**：TD 目标用一个**周期性同步的旧网络**算：
   $$y = r+\gamma\max_{a'}Q(s',a';\theta^-)$$
   解决"**目标随着参数一起动**"的问题 —— 否则就是在追自己的尾巴，训练极不稳定。

**改进三件套**：
- **Double DQN**：$\max$ 操作会**系统性高估** $Q$（对含噪估计取最大，期望上偏高）。
  解法是解耦"选动作"和"评估"：$y=r+\gamma\,Q(s',\arg\max_{a'}Q(s',a';\theta);\theta^-)$。
- **Dueling DQN**：把网络拆成 $Q=V(s)+A(s,a)$，很多状态下动作选谁都差不多，单独学 $V$ 更高效。
- **PER（优先经验回放）**：按 TD 误差大小优先采样"学得最不好"的样本，要用重要性权重修正引入的偏差。

**局限**：DQN 只能处理**离散动作**。连续动作要用 DDPG/TD3/SAC，或直接用策略梯度。

---

## 3. 策略梯度

### 3.1 策略梯度定理

直接把策略参数化成 $\pi_\theta(a|s)$ 并对目标做梯度上升：
$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a|s)\cdot Q^{\pi}(s,a)\right]$$

**直觉**：$\nabla\log\pi$ 是"让这个动作更可能发生"的方向，乘上它的好坏 $Q$ —— **好动作就加大概率，坏动作就减小概率**。

**REINFORCE**：用一整条轨迹的实际回报 $G_t$ 代替 $Q$。无偏，但**方差极大**（一条轨迹的回报噪声很大）。

### 3.2 Baseline 与优势（高频追问）

减去一个只依赖状态的基线 $b(s)$：
$$\nabla J=\mathbb{E}\left[\nabla\log\pi_\theta(a|s)\left(G_t-b(s)\right)\right]$$

**Q：为什么减 baseline 不改变期望，却能降方差？（必答）**
不改变期望是因为
$$\mathbb{E}_{a\sim\pi}\left[\nabla\log\pi_\theta(a|s)\,b(s)\right]=b(s)\sum_a \pi_\theta(a|s)\frac{\nabla\pi_\theta(a|s)}{\pi_\theta(a|s)}=b(s)\,\nabla\sum_a\pi_\theta(a|s)=b(s)\,\nabla 1=0$$
（概率和恒为 1，其梯度为 0）。

降方差是因为**去掉了与动作无关的共同分量**：如果某个状态下所有动作的回报都是 +100，
不减基线时所有动作的概率都被推高（只是幅度不同），信号全是噪声；
减去 $b(s)=V(s)$ 之后，剩下的正是优势 $A(s,a)$ —— **只反映"这个动作相对好在哪"**。

**取 $b(s)=V(s)$ 就得到 Actor-Critic**：Actor 是策略 $\pi_\theta$，Critic 是价值网络 $V_\phi$。
**这就是 RLHF 里为什么会有一个 Critic。**

### 3.3 GAE（广义优势估计）

优势的估计要在偏差和方差之间取舍：
- 用一步 TD 误差 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$：方差小，但依赖 $V$ 的准确性，**偏差大**。
- 用整条轨迹的回报：无偏，但**方差大**。

GAE 用指数加权把所有 $n$ 步估计融合起来：
$$A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}$$

**$\lambda$ 就是这个 trade-off 的旋钮**：$\lambda=0$ 退化成一步 TD（低方差高偏差），
$\lambda=1$ 退化成蒙特卡洛（无偏高方差）。实践常取 $\lambda=0.95$。
**能说清"$\lambda$ 在权衡偏差和方差"就答到点子上了。**

---

## 4. PPO（最该掌握的一个）

### 4.1 要解决的问题

策略梯度是 on-policy 的：更新一次策略，之前采的数据就不能用了 —— **样本效率极低**。
想复用数据就要用重要性采样，定义比率
$$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$
目标变成 $\mathbb{E}[r_t(\theta)A_t]$。但**比率偏离 1 太多时，重要性采样的方差会爆炸，一次更新就可能把策略毁掉**。

**TRPO** 的解法是加一个硬约束：$\max \mathbb{E}[r_tA_t]\ \text{s.t.}\ \mathbb{E}[\text{KL}(\pi_{old}\|\pi_\theta)]\le\delta$。
效果好，但要算二阶信息（Fisher 矩阵 + 共轭梯度），实现复杂。

### 4.2 PPO-Clip：用裁剪代替约束

$$L^{CLIP}(\theta)=\mathbb{E}_t\left[\min\Big(r_t(\theta)A_t,\ \ \text{clip}\big(r_t(\theta),1-\epsilon,1+\epsilon\big)A_t\Big)\right]$$

常取 $\epsilon=0.2$。**这个式子要能解释清楚**：

- $A_t>0$（这是个好动作，想提高其概率）：一旦 $r_t>1+\epsilon$，裁剪项把目标**封顶**，梯度消失 → **不再继续推高**。
- $A_t<0$（坏动作，想降低其概率）：一旦 $r_t<1-\epsilon$，同样被截住 → **不再继续压低**。
- 取 $\min$ 使得这个目标是**真实目标的悲观下界**，从而"更新过头就没有收益"，起到了信任域的效果。

**一句话**：**PPO 用一个一阶、好实现的裁剪，近似了 TRPO 的信任域约束 —— 每次只走一小步，不让新策略离旧策略太远。**

**完整损失**还包括两项：
$$L=L^{CLIP}-c_1\underbrace{(V_\phi(s)-V^{target})^2}_{\text{Critic 的值函数损失}}+c_2\underbrace{H[\pi_\theta]}_{\text{熵奖励，鼓励探索防早熟}}$$

**实现细节（问到就是加分）**：一批数据重复训练多个 epoch（这正是 clip 存在的意义）、
优势做 batch 内标准化、值函数也可以做 clip、注意 KL 早停。

---

## 5. 从 PPO 到 RLHF 与 GRPO

### 5.1 语言模型怎么变成 MDP

| MDP 要素 | 在 LLM 里对应什么 |
|---|---|
| 状态 $s_t$ | prompt + 已经生成的 token 前缀 |
| 动作 $a_t$ | 下一个 token（动作空间 = 整个词表，几万到十几万维） |
| 转移 | 确定性的 —— 就是把 token 拼接上去 |
| 奖励 | **只在序列结束时由 RM 给一个标量**，中间步骤没有奖励（稀疏奖励） |

**因为奖励极度稀疏，Critic 的价值就凸显出来了** —— 它把"最终这条回答得了多少分"分摊到每一个 token 上，
告诉模型是哪几个 token 拉高或拉低了分数。这就是 [04 §4.2](04-llm.md) 里第二个可训练模型的作用。

实际优化的是带 KL 惩罚的奖励：
$$\tilde r_t = r_\phi(x,y)\cdot\mathbb{1}[t=T] - \beta\,\log\frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$$
KL 项**逐 token 生效**，防止策略为了讨好 RM 而跑偏（reward hacking）和语言能力退化。

至此四个模型就齐了：**Actor（$\pi_\theta$）、Critic（$V_\phi$）、Reward Model（$r_\phi$）、Reference（$\pi_{ref}$）**。

### 5.2 GRPO 省掉了什么

Critic 和 Actor 同规模，显存和实现成本都很高。**GRPO 的思路是：既然要的只是一个 baseline，为什么不用采样来估？**

对同一个 prompt 采样 $G$ 个回答，用**组内奖励的均值当 baseline**、标准差做归一化：
$$A_i=\frac{r_i-\text{mean}(r_{1..G})}{\text{std}(r_{1..G})}$$

- 省掉了整个 Critic 网络（显存和工程复杂度大降）。
- 代价：需要对每个 prompt 多次采样（采样成本上升），且优势是**序列级**的（组内每个 token 共享同一个优势值），比 Critic 给的 token 级信号粗。
- 在**可验证奖励**（数学答案对错、代码是否通过测试）场景下特别合适 —— 奖励本身准确且不会被 hack，组内比较就足够了。

**顺带一提**：DPO（[04 §4.3](04-llm.md)）走的是完全不同的路 —— **它根本不做 RL**，
而是用闭式解把奖励消掉，变成一个类似 SFT 的监督目标。
**面试时能把「PPO / GRPO / DPO 三条路线」摆清楚，比背某一个的公式更有价值**：
PPO 在线采样 + Critic，GRPO 在线采样 + 组内 baseline，DPO 离线偏好对 + 无采样。

---

## 6. 探索与利用

**为什么必须探索**：只按当前估计取最优（纯利用）会永远错过更好的选项，
因为没试过的东西估计值不准。

| 策略 | 做法 | 特点 |
|---|---|---|
| $\epsilon$-greedy | 以 $\epsilon$ 概率随机 | 简单，但探索是**无差别**的，会一直去试明显很差的选项 |
| **UCB** | 选 $\bar x_a + c\sqrt{\frac{\ln t}{n_a}}$ | **对不确定性加成** —— 试得少的选项加分多，有理论遗憾界 |
| **Thompson Sampling** | 从后验分布采样再取最大 | 工业界常用，天然处理延迟反馈，实现简单效果好 |
| 内在奖励 | 好奇心 / 计数 / RND | 用于奖励极度稀疏的深度 RL |

这套东西在推荐冷启动里就是 [05 §5](05-recsys-ads.md) 的 E&E，**是同一件事的两个名字**。

---

## 7. 推荐 / 广告里的 RL（以及为什么它难落地）

**为什么想用 RL**：推荐本质是**序列决策** —— 这一刷推什么会影响用户接下来看什么、明天还来不来。
监督学习每次只优化"这一条的点击率"，是**贪心的**；RL 才能优化长期累计收益（时长、留存）。

**为什么难落地（这才是面试想听的）**：

1. **不能在线试错**：随机探索的代价是真实的用户体验和收入，不像游戏可以无限重开。
2. **奖励难定义**：留存是长期、稀疏、受无数外部因素影响的信号，噪声远大于信号。
3. **状态空间巨大且部分可观测**：用户真实兴趣观测不到。
4. **离线评估难**：日志是旧策略产生的，无法直接评估新策略的表现。

**所以实际用的是**：
- **离线 RL（Batch RL）**：只从历史日志学，不与环境交互。核心难点是**分布偏移** ——
  模型会对日志里没出现过的动作给出过高估计（外推误差），因此需要 **BCQ / CQL** 这类
  "**约束策略不要偏离数据分布太远**"的方法。**注意这与 PPO 的 clip、RLHF 的 KL 惩罚是同一个思想母题：别走太远。**
- **离线策略评估 OPE**：用重要性采样（IPS）、双重稳健（DR）估计新策略的线上表现 ——
  和 [14 §6](14-causal-uplift.md) 的 IPW / AIPW 是**同一套数学**，因为二者都在回答反事实问题。
- **约束出价**：广告在预算和 ROI 约束下的出价控制，常建模成带约束的 MDP，实践中 PID 控制器往往就够用且更稳。

**一个诚实的判断**：多数团队的主力仍是监督学习 + 多目标融合，RL 用在重排、探索和出价这类
**动作空间小、反馈相对快**的环节。面试时**别把 RL 说成万金油**，能讲出它的适用边界更显成熟。

---

## 延伸阅读

> 未附链接：成稿环境无法联网核验，标题和年份均可直接搜索到原文。

- *Reinforcement Learning: An Introduction*（Sutton & Barto, 2nd ed. 2018）—— MDP、TD、策略梯度的标准教材
- *Human-level control through deep reinforcement learning*（Mnih et al., Nature 2015）—— DQN 的经验回放与目标网络
- *Deep Reinforcement Learning with Double Q-learning*（van Hasselt et al., AAAI 2016）—— max 高估问题
- *Trust Region Policy Optimization*（Schulman et al., ICML 2015）
- *Proximal Policy Optimization Algorithms*（Schulman et al., 2017）—— PPO-Clip
- *High-Dimensional Continuous Control Using Generalized Advantage Estimation*（Schulman et al., ICLR 2016）—— GAE 与 $\lambda$
- *Training language models to follow instructions with human feedback*（Ouyang et al., NeurIPS 2022）—— RLHF 里 PPO 的具体用法
- *DeepSeekMath*（Shao et al., 2024）—— GRPO
- *Conservative Q-Learning for Offline RL*（Kumar et al., NeurIPS 2020）—— CQL 与分布偏移
- *Offline Reinforcement Learning: Tutorial, Review, and Perspectives*（Levine et al., 2020）

---

## 自测清单

- [ ] 能写出贝尔曼最优方程，并解释 $\gamma$ 的两层含义
- [ ] 能说清 Q-learning 和 SARSA 的唯一差别，以及 on/off-policy 的实际后果
- [ ] 能说出 DQN 的经验回放和目标网络各解决什么问题
- [ ] 能证明减 baseline 不改变梯度期望，并解释它为什么降方差
- [ ] 能解释 GAE 里 $\lambda$ 在权衡什么
- [ ] **能默写 PPO 的 clip 目标，并分正负优势两种情况解释裁剪的作用**
- [ ] 能说清 PPO 与 TRPO 的关系
- [ ] 能把语言模型建成 MDP，说清状态、动作、奖励分别是什么
- [ ] 能讲 GRPO 省掉 Critic 的代价，以及 PPO / GRPO / DPO 三条路线的区别
- [ ] 能说出推荐场景用 RL 的四个困难，以及离线 RL 的分布偏移问题
