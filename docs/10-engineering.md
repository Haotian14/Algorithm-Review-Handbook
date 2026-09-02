# 10 · 工程能力

> 算法岗实际工作里有一多半是工程。校招面试问工程通常是为了验证"你是真的自己训过模型，还是只跑过 notebook"。这些问题答得好，可信度会大幅提升。

## 本章高频考点

> 星级是**主观判断**，反映常见面经里的印象分布，不是统计数据。用来排复习优先级即可。

| 考点 | 频率 | 典型问法 |
|---|---|---|
| PyTorch 训练循环细节 | ★★★ | zero_grad 忘了会怎样 |
| 显存占用分析 | ★★★ | 训练一个 7B 模型要多少显存 |
| OOM 怎么解决 | ★★★ | 列举 6 种以上 |
| 混合精度训练 | ★★★ | fp16 vs bf16，loss scaling |
| 分布式：DP/DDP/ZeRO | ★★★ | DDP 为什么比 DP 快 |
| 并行策略：TP/PP/SP | ★★ | 什么时候用张量并行 |
| Dataset / DataLoader | ★★ | num_workers 怎么设 |
| 推理部署 | ★★ | ONNX / TensorRT / vLLM |
| 实验管理与复现 | ★★ | 怎么保证可复现 |

---

## 1. PyTorch 训练循环的坑

```python
model.train()
for epoch in range(epochs):
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)   # ① 必须清零，否则梯度累加
        with torch.autocast('cuda', dtype=torch.bfloat16):   # ② 混合精度
            out = model(x)
            loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # ③ 梯度裁剪
        optimizer.step()
        scheduler.step()                        # ④ 注意是每 step 还是每 epoch

model.eval()
with torch.no_grad():                           # ⑤ 验证时关梯度，省显存
    ...
```

**高频细节题**：

| 问题 | 答案 |
|---|---|
| 忘了 `zero_grad()` 会怎样？ | 梯度不断累加，相当于用越来越大的有效 batch 且梯度失真，训练会崩 |
| `zero_grad(set_to_none=True)` 有什么好处？ | 把 `.grad` 置为 `None` 而不是 0 张量，省显存、省一次 memset，是现在的默认值 |
| 忘了 `model.eval()` 会怎样？ | Dropout 仍在丢弃、BN 用当前 batch 统计量 → 验证指标抖动且偏低 |
| `torch.no_grad()` 和 `model.eval()` 的区别？ | 前者不建计算图（省显存/加速），后者改变 Dropout/BN 的行为。**两个都要**，缺一不可 |
| `loss.backward()` 能调用两次吗？ | 不能，计算图会被释放。需要 `retain_graph=True`（但通常说明你的写法有问题） |
| `.detach()` 和 `.data` 的区别？ | `.detach()` 安全（原地修改会被检测报错），`.data` 会绕过 autograd 检查，不要用 |
| 为什么 loss 要 `.item()` 后再累加？ | 直接累加张量会保留整个计算图，导致显存泄漏 |
| 梯度累积怎么写？ | `loss = loss / accum_steps; loss.backward()`，每 `accum_steps` 步才 `optimizer.step()` 和 `zero_grad()` |

---

## 2. 显存分析（必会算）

### 2.1 训练显存的四大块

| 组成 | 大小（以 $\Phi$ 个参数、混合精度 + Adam 为例） |
|---|---|
| 模型参数 | fp16: $2\Phi$ |
| 梯度 | fp16: $2\Phi$ |
| 优化器状态 | fp32 主权重 $4\Phi$ + Adam 的 $m$ $4\Phi$ + $v$ $4\Phi$ = $12\Phi$ |
| **小计（静态）** | **$16\Phi$ 字节** |
| 激活值 | 与 batch × 序列长度 × 层数 × 隐层维度成正比，**动态且常是大头** |

**例题：全参微调 7B 模型需要多少显存？**
静态部分 $16\times 7\times10^9 = 112$ GB，加上激活值和碎片，需要 2 张 80G A100 起步（还得用 ZeRO 切分）。这就是 LoRA/QLoRA 存在的意义 —— LoRA 只训 0.1% 的参数，优化器状态几乎归零，7B 单卡 24G 就能跑。

**推理显存**：参数（fp16 为 $2\Phi$）+ KV Cache（计算公式见 [04 §3.1](04-llm.md)）+ 少量激活。7B fp16 推理约 14GB + KV Cache。

### 2.2 OOM 怎么解决（列举题）

按"效果损失从小到大"排序：
1. **减小 batch size** + 用**梯度累积**保持有效 batch 不变（几乎无损）。
2. **混合精度**（bf16/fp16），显存直接减半。
3. **梯度检查点（activation checkpointing）**：不缓存中间激活，反向时重算。显存降到 $O(\sqrt{L})$，时间多 ~30%。
4. **优化器状态切分（ZeRO-1/2/3）** 或用 8-bit 优化器（bitsandbytes）。
5. **CPU/NVMe offload**（ZeRO-Offload），慢但能跑。
6. **减小输入尺寸/序列长度**，或用 FlashAttention 消除 $n^2$ 的注意力矩阵。
7. **PEFT**（LoRA/QLoRA）替代全参微调。
8. 工程排查：`torch.cuda.empty_cache()` 清碎片、检查是否累加了带图的张量、验证阶段是否忘了 `no_grad`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片。

---

## 3. 混合精度

| 格式 | 位数分配 | 特点 |
|---|---|---|
| fp32 | 1+8+23 | 基准 |
| **fp16** | 1+5+10 | 精度高但**动态范围小**，易溢出 → 需要 **loss scaling** |
| **bf16** | 1+8+7 | 动态范围与 fp32 相同，精度低但不溢出 → **大模型首选**，无需 loss scaling |
| fp8 | E4M3 / E5M2 | H100+ 原生支持，训练推理都在用 |

**Loss Scaling 原理**：fp16 下小梯度会下溢为 0。把 loss 乘一个大系数 $S$，梯度同比放大到 fp16 可表示范围，更新前再除以 $S$。动态 loss scaling 会在检测到 inf/nan 时自动减半系数并跳过该步。

**混合精度为什么还要 fp32 主权重？** 参数更新量往往比参数本身小几个数量级，fp16 下 `w + tiny_update` 会因舍入而无变化（stale weights）。所以保留一份 fp32 主权重做累加。

---

## 4. 分布式训练

### 4.1 数据并行

| 方式 | 机制 | 问题 |
|---|---|---|
| **DP** (`DataParallel`) | 单进程多线程，主卡收集所有输出算 loss 再散发 | 受 GIL 限制、**主卡显存和通信瓶颈**、负载不均。**已废弃** |
| **DDP** (`DistributedDataParallel`) | 每卡一个进程，各自前反向，用 **Ring All-Reduce** 同步梯度 | 通信量与卡数无关、可重叠通信和计算（梯度分桶，边算边同步）。**标准做法** |

**Q：DDP 为什么比 DP 快？** ① 多进程无 GIL 争抢；② 只同步梯度（All-Reduce），不搬运数据和输出；③ 通信与反向计算重叠；④ 无中心节点瓶颈。

**Ring All-Reduce 通信量**：每卡传输 $2\frac{N-1}{N}\times$ 参数量，与卡数几乎无关（这是它优于 Parameter Server 的关键）。

### 4.2 ZeRO（DeepSpeed / FSDP）

数据并行的问题是每张卡都存一份完整的参数、梯度、优化器状态。ZeRO 把它们切分到各卡：

| 阶段 | 切分内容 | 显存（每卡） | 通信量 |
|---|---|---|---|
| ZeRO-1 | 优化器状态 | $4\Phi+\frac{12\Phi}{N}$ | 与 DDP 相同 |
| ZeRO-2 | + 梯度 | $2\Phi+\frac{14\Phi}{N}$ | 与 DDP 相同 |
| ZeRO-3 | + 参数 | $\frac{16\Phi}{N}$ | 约 1.5× DDP |

ZeRO-3（= PyTorch FSDP）在前向/反向时按需 All-Gather 出需要的层参数，用完立即释放。**用通信换显存。**

### 4.3 模型并行

- **张量并行 TP**（Megatron）：把单个矩阵乘按行/列切到多卡。MLP 的第一个矩阵按列切、第二个按行切，中间不需要通信，只在最后 All-Reduce 一次。**通信频繁，只在单机内（NVLink）用**。
- **流水线并行 PP**：按层切到不同卡，用 micro-batch 填充流水线减少 **bubble**（气泡率 $\approx\frac{p-1}{m+p-1}$，$p$ 是 stage 数，$m$ 是 micro-batch 数）。调度：GPipe、1F1B、交错式 1F1B。
- **序列并行 SP**：把序列维度切分，处理超长上下文（Ring Attention / Ulysses）。
- **专家并行 EP**：MoE 的专家分布到不同卡，需要 All-to-All 通信。

**3D/4D 并行**：大规模训练通常组合使用 —— **机内 TP，机间 PP，最外层 DP + ZeRO**。原则是把通信量大的并行放在带宽高的层级。

---

## 5. 数据流水线

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,          # 经验值：CPU 核数的一半到全部；太大会争抢内存和 IO
    pin_memory=True,        # 锁页内存，加速 H2D 拷贝（配合 non_blocking=True）
    persistent_workers=True,# 避免每个 epoch 重建 worker 进程
    prefetch_factor=2,
    drop_last=True,         # BN 场景下避免最后一个 batch 太小
)
```

**Q：GPU 利用率低怎么排查？**
1. `nvidia-smi dmon` 或 `py-spy` 看是不是卡在数据加载 → 加 `num_workers`、把解码/增强搬到 GPU（DALI）、用更快的存储格式（webdataset / LMDB / 内存映射）。
2. 是否有频繁的 CPU-GPU 同步（`.item()`、`.cpu()`、`print(loss)` 在循环里）。
3. batch 太小导致 kernel launch 开销占比高 → 增大 batch 或用 CUDA Graph。
4. 用 `torch.profiler` 定位瓶颈算子。

**大规模数据**：用流式读取（IterableDataset / webdataset），避免一次性加载；多机训练要用 `DistributedSampler` 保证各卡不重复。

---

## 6. 推理部署

### 6.1 通用模型

```
训练框架 (PyTorch) → 导出 (ONNX / TorchScript) → 优化 (TensorRT / OpenVINO) → 服务 (Triton)
```

优化手段：算子融合（Conv+BN+ReLU 融成一个 kernel）、量化（INT8 需要校准集）、图优化（常量折叠、死代码消除）、动态 batch、多流并发。

**Conv + BN 融合**：推理时 BN 是固定的线性变换，可以直接吸收进卷积权重：
$$W' = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}W,\quad b'=\frac{\gamma(b-\mu)}{\sqrt{\sigma^2+\epsilon}}+\beta$$

### 6.2 LLM 推理

用 **vLLM / SGLang / TensorRT-LLM**，核心优化见 [04 §3](04-llm.md)：PagedAttention、continuous batching、投机解码、prefix caching、量化。

**服务指标**：TTFT（首 token 延迟，影响体感）、TPOT（每 token 延迟）、吞吐（tokens/s）、并发数。注意**延迟和吞吐是矛盾的** —— 增大 batch 提升吞吐但拉高单请求延迟，要按业务场景取舍（对话优先延迟，离线批处理优先吞吐）。

---

## 7. 实验管理与可复现

```python
def set_seed(seed):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True    # 会变慢
    torch.backends.cudnn.benchmark = False
```

**注意**：即便如此，**多卡训练、原子加操作、cuDNN 的非确定性算法**仍可能导致结果有微小差异。面试时能说出"完全 bit-wise 复现在 GPU 上很难，我们保证的是统计意义上的复现"会显得很专业。

**实验管理**：用 W&B / TensorBoard / MLflow 记录指标；用 Hydra / OmegaConf 管配置；每次实验记录 **代码 commit + 配置 + 数据版本 + 随机种子**；一次只改一个变量。

---

## 8. 常见工程八股速答

| 问题 | 要点 |
|---|---|
| Python GIL 的影响 | 多线程不能并行计算，所以 DataLoader 用多**进程**；IO 密集可用多线程 |
| `nn.Module` 的 `__call__` 和 `forward` | `__call__` 会触发 hook，所以要用 `model(x)` 而不是 `model.forward(x)` |
| `register_buffer` 的用途 | 存非参数状态（如 BN 的 running_mean、causal mask），会随 `state_dict` 保存和 `.to(device)` 迁移，但不参与梯度 |
| 权重共享（tied embedding） | 输入 embedding 和输出 lm_head 共享权重，省参数且效果不降 |
| 梯度累积和大 batch 等价吗 | 数学上对无状态层等价；但 BN 不等价（BN 只在当前 micro-batch 内统计） |
| checkpoint 存什么 | model.state_dict + optimizer.state_dict + scheduler + epoch/step + scaler + 随机数状态 |
| 学习率和 batch size 的关系 | 线性缩放法则：batch 扩大 $k$ 倍，lr 也扩大 $k$ 倍，并配合 warmup |
| 为什么大模型训练用 bf16 | 动态范围等同 fp32，不需要 loss scaling，训练更稳 |

---

## 延伸阅读

- *Mixed Precision Training*（Micikevicius et al., ICLR 2018）—— fp32 主权重与 loss scaling
- *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*（Rajbhandari et al., SC 2020）—— 三个阶段的显存/通信表
- *ZeRO-Offload*（Ren et al., ATC 2021）
- *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*（Zhao et al., VLDB 2023）
- *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*（Shoeybi et al., 2019）—— 张量并行的切分方式
- *GPipe*（Huang et al., NeurIPS 2019）与 *Efficient Large-Scale Language Model Training on GPU Clusters*（Narayanan et al., SC 2021）—— 流水线 bubble 与 1F1B
- *Training Deep Nets with Sublinear Memory Cost*（Chen et al., 2016）—— 梯度检查点，O(√n) 显存与约 30% 额外时间的出处
- *PyTorch Distributed: Experiences on Accelerating Data Parallel Training*（Li et al., VLDB 2020）—— DDP 的梯度分桶与通信重叠

---

## 自测清单

- [ ] 能说出训练循环里 5 个以上的常见坑
- [ ] 能算 7B 模型全参微调的显存，并解释 $16\Phi$ 的来源
- [ ] 能列 6 种以上 OOM 的解法并说明各自的代价
- [ ] 能说清 fp16 和 bf16 的差异，以及为什么需要 fp32 主权重
- [ ] 能解释 DDP 优于 DP 的四个原因和 Ring All-Reduce
- [ ] 能说清 ZeRO 三个阶段切了什么、显存和通信的取舍
- [ ] 能区分 TP / PP / SP / EP 及其部署位置
- [ ] 有一套 GPU 利用率低的排查流程
- [ ] 能说出 LLM 推理服务的关键指标和延迟/吞吐的矛盾
