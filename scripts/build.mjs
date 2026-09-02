/**
 * 构建脚本（零依赖）：
 *   1. 根据 data/questions.json 生成 docs/12-question-bank.md
 *   2. 把 docs/*.md 全部打包成 site/assets/content.js，供静态站点直接使用
 *      （打包成 JS 而不是 fetch markdown，是为了让站点在 file:// 下也能打开）
 *
 * 用法：node scripts/build.mjs
 */
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DOCS = join(ROOT, 'docs');

// ----------------------------------------------------------------- 章节元信息
const CHAPTERS = [
  { id: '00-roadmap',            title: '复习路线图',     short: '路线图',   icon: '🗺️', group: '开始',   desc: '考什么、怎么排期、常见误区' },
  { id: '01-math-basics',        title: '数学与统计基础', short: '数学基础', icon: '∑',  group: '基础',   desc: '线代、概率、优化、概率面试题' },
  { id: '02-machine-learning',   title: '经典机器学习',   short: '机器学习', icon: '🌲', group: '基础',   desc: 'LR/SVM/树模型/评估指标/特征工程' },
  { id: '03-deep-learning',      title: '深度学习基础',   short: '深度学习', icon: '◈',  group: '基础',   desc: '反向传播/归一化/优化器/正则/调参' },
  { id: '04-llm',                title: '大模型 LLM',     short: '大模型',   icon: '🤖', group: '方向',   desc: 'Transformer/RLHF/PEFT/推理加速/RAG' },
  { id: '05-recsys-ads',         title: '推荐搜索广告',   short: '推搜广',   icon: '🎯', group: '方向',   desc: '召回粗排精排/CTR 演进/多目标/AB' },
  { id: '06-cv-multimodal',      title: 'CV 与多模态',    short: 'CV',       icon: '👁', group: '方向',   desc: '检测分割/ViT/扩散模型/CLIP/VLM' },
  { id: '07-nlp-basics',         title: 'NLP 基础',       short: 'NLP',      icon: '文', group: '方向',   desc: '分词/词向量/RNN/BERT 系/语义检索' },
  { id: '08-coding-leetcode',    title: '手撕：算法题',   short: '算法题',   icon: '⌨', group: '手撕',   desc: '模板 + DP 专题 + 高频题单' },
  { id: '09-coding-ml-scratch',  title: '手撕：ML 组件',  short: 'ML 组件',  icon: '⚙',  group: '手撕',   desc: 'attention/AUC/NMS/KMeans 可运行实现' },
  { id: '10-engineering',        title: '工程能力',       short: '工程',     icon: '🔧', group: '进阶',   desc: '显存/混合精度/分布式/部署' },
  { id: '11-project-behavioral', title: '项目与行为面',   short: '项目面',   icon: '💬', group: '进阶',   desc: '项目深挖套路/STAR/反问/简历' },
  { id: '12-question-bank',      title: '高频题库',       short: '题库',     icon: '★',  group: '进阶',   desc: '130 道自测题，支持随机抽查' },
];

// ----------------------------------------------------- 1. 生成 12-question-bank.md
const bank = JSON.parse(readFileSync(join(ROOT, 'data', 'questions.json'), 'utf8'));
const catName = Object.fromEntries(bank.categories.map((c) => [c.id, c]));
const stars = (n) => '★'.repeat(n) + '☆'.repeat(3 - n);

let md = `# 12 · 高频题库

> 共 **${bank.questions.length}** 道题，按方向分组。**先盖住答案自己说一遍，再对照要点。**
> 网页版支持随机抽题、按方向/难度筛选和掌握度追踪 —— 用那个刷效率更高。

| 星级 | 含义 |
|---|---|
| ★★★ | 必答，答不上直接扣大分 |
| ★★☆ | 高频，建议掌握 |
| ★☆☆ | 加分项，答出来会让人眼前一亮 |

`;

for (const cat of bank.categories) {
  const qs = bank.questions.filter((q) => q.c === cat.id);
  if (!qs.length) continue;
  qs.sort((a, b) => b.l - a.l);
  md += `\n---\n\n## ${cat.name}（${qs.length} 题）\n\n> 详细讲解见 [${cat.chapter}](${cat.chapter}.md)\n\n`;
  qs.forEach((q, i) => {
    md += `### ${i + 1}. ${q.q}\n\n\`${stars(q.l)}\`  ${q.a}\n\n`;
  });
}
writeFileSync(join(DOCS, '12-question-bank.md'), md);

// -------------------------------------------------- 2. 打包 markdown 到 content.js
const files = readdirSync(DOCS).filter((f) => f.endsWith('.md'));
const known = new Set(CHAPTERS.map((c) => c.id));
for (const f of files) {
  const id = f.replace(/\.md$/, '');
  if (!known.has(id)) throw new Error(`docs/${f} 未在 CHAPTERS 中登记`);
}
for (const c of CHAPTERS) {
  if (!files.includes(`${c.id}.md`)) throw new Error(`缺少 docs/${c.id}.md`);
}

const chapters = CHAPTERS.map((c) => {
  const body = readFileSync(join(DOCS, `${c.id}.md`), 'utf8');
  return { ...c, body, words: body.replace(/\s/g, '').length };
});

const out =
  '/* 由 scripts/build.mjs 自动生成，请勿手工修改。改内容请编辑 docs/*.md 后重新构建。 */\n' +
  'window.__HANDBOOK__ = ' +
  JSON.stringify({ chapters, bank }) +
  ';\n';
writeFileSync(join(ROOT, 'site', 'assets', 'content.js'), out);

const total = chapters.reduce((s, c) => s + c.words, 0);
console.log(`✓ docs/12-question-bank.md  (${bank.questions.length} 题)`);
console.log(`✓ site/assets/content.js    (${chapters.length} 章 / 约 ${(total / 1000).toFixed(1)}k 字 / ${(out.length / 1024).toFixed(0)} KB)`);
