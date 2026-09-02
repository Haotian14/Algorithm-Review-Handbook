# 算法岗面试复习手册

面向**校招 / 实习**的算法岗面试复习材料，覆盖大模型 LLM、推荐搜索广告、CV 多模态、通用机器学习四个方向。

**13 章系统笔记 · 130 道高频自测题 · 可运行的手撕代码**

> 🌐 **网页版**：`https://<你的用户名>.github.io/Algorithm-Review-Handbook/`
> （首次部署需按下方「部署」一节在仓库设置里打开 GitHub Pages）

---

## 这本手册解决什么问题

市面上的面经要么是零散的问题列表，要么是教科书式的知识罗列。这本手册按**面试官真实的提问方式**组织：

```
问题 → 答案骨架 → 追问
```

因为面试的实际形态就是「一个问题，你答一层，他往下追一层」。手册里每个知识点都标注了追问方向，L1 答对只是及格，追问答得上才是加分。

网页版还提供：全书搜索（⌘K）、随机抽题闪卡、章节掌握度追踪、亮/暗色主题。

---

## 目录

| # | 章节 | 内容 |
|---|---|---|
| 00 | [复习路线图](docs/00-roadmap.md) | 考什么、3 个月 / 1 个月 / 1 周三种排期、常见误区 |
| 01 | [数学与统计基础](docs/01-math-basics.md) | PCA 推导、MLE 与交叉熵、偏差方差、概率面试题 |
| 02 | [经典机器学习](docs/02-machine-learning.md) | LR/SVM/GBDT/XGBoost/LightGBM、AUC、特征工程、不平衡数据 |
| 03 | [深度学习基础](docs/03-deep-learning.md) | 反向传播、BN/LN、Adam/AdamW、梯度问题、残差、调参排查 |
| 04 | [大模型 LLM](docs/04-llm.md) | Transformer、RoPE、KV Cache、RLHF/DPO/GRPO、LoRA、量化、RAG、Agent |
| 05 | [推荐搜索广告](docs/05-recsys-ads.md) | 全链路、双塔、负采样、CTR 演进、MMoE/PLE/ESMM、GAUC、AB 测试 |
| 06 | [CV 与多模态](docs/06-cv-multimodal.md) | 检测分割、Focal Loss、ViT/Swin、扩散模型、CLIP、VLM |
| 07 | [NLP 基础](docs/07-nlp-basics.md) | 分词、Word2Vec、LSTM、BERT 系、语义检索与句向量 |
| 08 | [手撕：算法题](docs/08-coding-leetcode.md) | 九大模板、DP 专题、78 道高频题单、复杂度速查 |
| 09 | [手撕：ML 组件](docs/09-coding-ml-scratch.md) | attention / AUC / NMS / K-Means / RoPE 等，**代码可直接跑** |
| 10 | [工程能力](docs/10-engineering.md) | 显存计算、OOM、混合精度、DDP/ZeRO/TP/PP、推理部署 |
| 11 | [项目与行为面](docs/11-project-behavioral.md) | 项目深挖六层套路、STAR-R、简历写法、反问 |
| 12 | [高频题库](docs/12-question-bank.md) | 130 道自测题（由 `data/questions.json` 生成） |

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

仓库自带 GitHub Actions，推送即自动构建并发布到 GitHub Pages。

**首次需要手动开一次开关**（只需一次，之后每次推送都会自动部署）：

1. 打开仓库 **Settings → Pages**
2. **Source** 选 **GitHub Actions**（不是 Deploy from a branch）
3. 回到 **Actions** 页签，重跑一次 `Deploy to GitHub Pages`
4. 访问 `https://<你的用户名>.github.io/Algorithm-Review-Handbook/`

> 这一步没法在 workflow 里自动化：创建 Pages 站点需要仓库的 `administration` 权限，
> 而 workflow 的 `GITHUB_TOKEN` 拿不到（试过 `configure-pages` 的 `enablement: true`，
> 报 `Resource not accessible by integration`）。

站点用的都是相对路径，所以放在子路径下也能正常工作。想换成自定义域名，在 `site/` 下加一个 `CNAME` 文件即可。

也可以直接把 `site/` 目录丢给 Vercel / Netlify / Cloudflare Pages —— 无需构建命令，输出目录填 `site`。

---

## 怎么改内容

| 想改什么 | 改哪里 |
|---|---|
| 章节正文 | `docs/*.md`，然后 `node scripts/build.mjs` |
| 题库 | `data/questions.json`（会自动生成 `docs/12-question-bank.md` 和网页抽题数据） |
| 新增一章 | 建 `docs/<编号>-<名字>.md`，并在 `scripts/build.mjs` 的 `CHAPTERS` 里登记（漏登记会构建报错） |
| 站点样式 | `site/assets/style.css`（顶部一组 CSS 变量控制整套配色） |
| 站点逻辑 | `site/assets/app.js` |

题库条目格式：

```json
{ "c": "llm", "l": 3, "q": "问题", "a": "答案要点，支持 Markdown 和 $LaTeX$" }
```

`c` 是方向 id，`l` 是重要程度（3 = 必答 / 2 = 高频 / 1 = 加分项）。

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
