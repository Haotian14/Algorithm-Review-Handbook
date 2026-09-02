# 算法岗面试复习手册

面向**校招 / 实习**的算法岗面试复习材料，覆盖大模型 LLM、推荐搜索广告、CV 多模态、通用机器学习四个方向，并包含系统设计、SQL 与大数据、手撕代码、因果推断与 Uplift、强化学习基础、项目与行为面。

**17 章系统笔记 · 201 道高频自测题 · 可运行的手撕代码**

> **在线阅读**：https://haotian14.github.io/Algorithm-Review-Handbook/

---

## 内容可信度与时效（重要）

内容由 AI 依据公开技术资料整理，**不含任何公司内部信息或面试真题原文**。三点请先知道：

1. **主观判断已明确标注。** 各章的 ★ 星级、「高频/必问」字样全部是主观估计，没有统计支撑，
   用来排复习优先级即可，不要当事实复述给面试官。
2. **关键结论挂了出处。** 出自论文的结论标了 `〔论文, 年份〕`，每章末尾有「延伸阅读」列出完整标题。
   未附链接是因为成稿环境无法联网核验 —— 与其给可能失效或指错的链接，不如给可搜索的准确标题。
3. **每章标了时效。** `慢`（数学/经典 ML/算法题，几年不变）、`中`、`快`（大模型，
   **建议面试前再对一遍最新进展**）。网页版会在章节头部显示。

详见 [00 章的「关于本手册」](docs/00-roadmap.md)。

## 这本手册解决什么问题

市面上的面经要么是零散的问题列表，要么是教科书式的知识罗列。这本手册按**面试官真实的提问方式**组织：

```
问题 → 答案骨架 → 追问
```

因为面试的实际形态就是「一个问题，你答一层，他往下追一层」。知识型章节（01–07、10、14、15）大量标注了追问方向，L1 答对只是及格，追问答得上才是加分；手撕、系统设计、SQL、行为面这几章是模板和答题框架式的，形态不同。

网页版还提供：全书搜索（⌘K）、随机抽题闪卡、章节掌握度追踪、亮/暗色主题。

---

## 目录

| # | 章节 | 内容 |
|---|---|---|
| 00 | [复习路线图](docs/00-roadmap.md) | 考什么、3 个月 / 1 个月 / 1 周三种排期、常见误区 |
| 01 | [数学与统计基础](docs/01-math-basics.md) | PCA 推导、MLE 与交叉熵、偏差方差、概率面试题 |
| 02 | [经典机器学习](docs/02-machine-learning.md) | LR/SVM/GBDT/XGBoost/LightGBM、AUC、特征工程、SHAP、不平衡数据 |
| 03 | [深度学习基础](docs/03-deep-learning.md) | 反向传播、BN/LN、Adam/AdamW、梯度问题、残差、调参排查 |
| 04 | [大模型 LLM](docs/04-llm.md) | Transformer、RoPE、KV Cache、RLHF/DPO/GRPO、LoRA、量化、RAG、Agent、安全与越狱 |
| 05 | [推荐搜索广告](docs/05-recsys-ads.md) | 全链路、双塔、负采样、CTR 演进、GNN 召回、MMoE/PLE/ESMM、GAUC、AB、搜索相关性 |
| 06 | [CV 与多模态](docs/06-cv-multimodal.md) | 检测分割、Focal Loss、ViT/Swin、扩散模型、CLIP、VLM、对抗样本 |
| 07 | [NLP 基础](docs/07-nlp-basics.md) | 分词、Word2Vec、LSTM、BERT 系、语义检索与句向量 |
| 08 | [手撕：算法题](docs/08-coding-leetcode.md) | 九大模板、DP 专题、78 道高频题单（带 LeetCode 题号）、复杂度速查 |
| 09 | [手撕：ML 组件](docs/09-coding-ml-scratch.md) | attention / AUC / NMS / K-Means / RoPE 等，**代码可直接跑** |
| 10 | [工程能力](docs/10-engineering.md) | 显存计算、OOM、混合精度、DDP/ZeRO/TP/PP、推理部署、隐私与联邦学习 |
| 11 | [算法系统设计](docs/11-system-design.md) | 通用答题框架、推荐/RAG 系统设计、指标体系、特征平台、容量估算 |
| 12 | [SQL 与大数据](docs/12-sql-bigdata.md) | 窗口函数、留存漏斗、数据倾斜、Spark/Flink、样本构造 |
| 13 | [项目与行为面](docs/13-project-behavioral.md) | 项目深挖六层套路、STAR-R、简历写法、反问 |
| 14 | [因果推断与 Uplift](docs/14-causal-uplift.md) | 相关 vs 因果、ATE/CATE、Uplift 四象限、S/T/X-Learner、Qini、观测数据去偏 |
| 15 | [强化学习基础](docs/15-rl-basics.md) | MDP、DQN、策略梯度、**PPO 的 clip 与 GAE**、与 RLHF/GRPO 的衔接、离线 RL |
| 16 | [高频题库](docs/16-question-bank.md) | 201 道自测题（由 `data/questions.json` 生成），★ 分级见下 |

---

## 手撕代码可以直接运行

`code/ml_components.py` 里的实现全部带自测用例：

```bash
pip install numpy
python code/ml_components.py     # → all tests passed
```

包含：数值稳定的 softmax/sigmoid/交叉熵、多头注意力（含 causal mask）、RoPE、LayerNorm/RMSNorm/BatchNorm、
AUC（正确处理并列分数）、IoU、NMS、Focal Loss、逻辑回归、K-Means++、PCA、蓄水池抽样、按权重采样。

---

## 本地预览网页版

```bash
node scripts/build.mjs      # 把 docs/*.md 打包进 site/assets/content.js
node scripts/serve.mjs      # → http://localhost:5173
```

或者 `npm run dev` 一步完成。

站点是**纯静态、零运行时依赖**：marked / KaTeX / highlight.js 和字体都已 vendor 到 `site/vendor/`，
不请求任何外部 CDN，断网也能用。

---

## 部署

站点已经上线：**https://haotian14.github.io/Algorithm-Review-Handbook/**

`.github/workflows/deploy.yml` 会在推送到 `main` 时自动构建并发布，平时不用管。

<details>
<summary>如果你 fork 了这个仓库，首次部署需要手动开一次开关</summary>

1. 打开仓库 **Settings → Pages**
2. **Source** 选 **GitHub Actions**（不是 Deploy from a branch）
3. 回到 **Actions** 页签，重跑一次 `Deploy to GitHub Pages`

这一步没法在 workflow 里自动化：创建 Pages 站点需要仓库的 `administration` 权限，
而 workflow 的 `GITHUB_TOKEN` 拿不到（试过 `configure-pages` 的 `enablement: true`，
报 `Resource not accessible by integration`）。
</details>

站点用的都是相对路径，所以放在子路径下也能正常工作。想换成自定义域名，在 `site/` 下加一个 `CNAME` 文件即可。

也可以直接把 `site/` 目录丢给 Vercel / Netlify / Cloudflare Pages —— 无需构建命令，输出目录填 `site`。

---

## 怎么改内容

| 想改什么 | 改哪里 |
|---|---|
| 章节正文 | `docs/*.md`，然后 `node scripts/build.mjs` |
| 题库 | `data/questions.json`（会自动生成 `docs/16-question-bank.md` 和网页抽题数据） |
| 新增一章 | 建 `docs/<编号>-<名字>.md`，并在 `scripts/build.mjs` 的 `CHAPTERS` 里登记（漏登记会构建报错） |
| 站点样式 | `site/assets/style.css`（顶部一组 CSS 变量控制整套配色） |
| 站点逻辑 | `site/assets/app.js` |

题库条目格式：

```json
{ "c": "llm", "l": 3, "q": "问题", "a": "答案要点，支持 Markdown 和 $LaTeX$" }
```

`c` 是方向 id，`l` 是重要程度：

| `l` | 含义 | 占比 |
|---|---|---|
| 3 | **必答**：几乎每场都会问到，答不上直接扣大分 | 约 1/4，刻意精选 |
| 2 | **高频**：常见考点，建议掌握 | 约 45% |
| 1 | **加分项**：特定方向或追问才会问到 | 约 30% |

分级是**主观判断**。之所以严格控制 ★★★ 的比例，是因为如果一半的题都标成「必答」，这个标记就失去了排优先级的作用。

---

## 快捷键

| 键 | 作用 |
|---|---|
| `⌘K` / `Ctrl+K` / `/` | 全书搜索 |
| `←` `→` | 上一章 / 下一章 |
| 抽题面板内 `空格` | 看答案 / 下一题 |
| 抽题面板内 `1` `2` | 标记「还不熟」/「会了」 |
| `Esc` | 关闭浮层 |

进度和答题记录存在浏览器 localStorage 里，只在本机保留。

---

## 声明

内容基于公开资料和常见面试考点整理，不含任何公司的内部信息或真题原文。
技术领域更新很快，涉及具体模型和方案的部分请结合最新进展判断。

MIT License。
