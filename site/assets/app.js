/* 算法岗面试复习手册 —— 前端应用（零构建，原生 ES2020） */
(() => {
'use strict';

const DATA = window.__HANDBOOK__;
const CH = DATA.chapters;
const BANK = DATA.bank;
const CAT = Object.fromEntries(BANK.categories.map(c => [c.id, c]));
const QS = BANK.questions.map((q, i) => ({ ...q, id: `q${i}` }));

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => s.replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));

/* ============================================================ 本地存储 */
const KEY = 'aih.v1';
const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } };
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch {} };
const store = Object.assign({ theme: null, done: [], marks: {} }, load());
const done = new Set(store.done);

/* ============================================================ 主题 */
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  const meta = $('meta[name=theme-color]');
  if (meta) meta.content = t === 'dark' ? '#121211' : '#faf9f7';
}
applyTheme(store.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
$('#btn-theme').onclick = () => {
  store.theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  applyTheme(store.theme); save();
};

/* ============================================================ Markdown 渲染 */
/* 先把公式抠出来占位，避免 marked 把 x_i 里的下划线当成斜体，渲染完再交给 KaTeX。 */
function protectMath(src) {
  const box = [];
  // 奇数下标是代码（围栏或行内），原样保留
  const parts = src.split(/(```[\s\S]*?```|`[^`\n]*`)/g);
  for (let i = 0; i < parts.length; i += 2) {
    parts[i] = parts[i]
      .replace(/\$\$([\s\S]+?)\$\$/g, (_, c) => (box.push({ d: true,  c }), `@@KMATH${box.length - 1}@@`))
      .replace(/\$([^$\n]+?)\$/g,      (_, c) => (box.push({ d: false, c }), `@@KMATH${box.length - 1}@@`));
  }
  return [parts.join(''), box];
}

function restoreMath(html, box) {
  return html.replace(/@@KMATH(\d+)@@/g, (m, i) => {
    const it = box[+i];
    if (!it) return m;
    try {
      return katex.renderToString(it.c, { displayMode: it.d, throwOnError: false, strict: false });
    } catch {
      return `<code>${esc(it.c)}</code>`;
    }
  });
}

marked.setOptions({ gfm: true, breaks: false, headerIds: false, mangle: false });

function renderMarkdown(src) {
  const [protectedSrc, box] = protectMath(src);
  let html = marked.parse(protectedSrc);
  html = restoreMath(html, box);

  const wrap = document.createElement('div');
  wrap.className = 'prose';
  wrap.innerHTML = html;

  // 站内 .md 链接改成 hash 路由
  $$('a[href$=".md"]', wrap).forEach(a => {
    const id = a.getAttribute('href').replace(/^.*\//, '').replace(/\.md$/, '');
    if (CH.some(c => c.id === id)) a.setAttribute('href', `#/c/${id}`);
  });
  $$('a[href^="http"]', wrap).forEach(a => { a.target = '_blank'; a.rel = 'noopener'; });

  // 标题锚点
  $$('h2, h3, h4', wrap).forEach((h, i) => { h.id = h.id || `sec-${i}`; });

  // 表格加横向滚动容器
  $$('table', wrap).forEach(t => {
    const d = document.createElement('div');
    d.className = 'table-wrap';
    t.replaceWith(d); d.appendChild(t);
  });

  // 任务清单变成真复选框（纯展示用，不持久化）
  $$('li', wrap).forEach(li => {
    const m = li.innerHTML.match(/^\s*\[( |x|X)\]\s*/);
    if (m) li.innerHTML = `<input type="checkbox"${/x/i.test(m[1]) ? ' checked' : ''}>` + li.innerHTML.slice(m[0].length);
  });

  // 代码高亮 + 复制按钮
  $$('pre > code', wrap).forEach(code => {
    if (window.hljs) {
      try { hljs.highlightElement(code); } catch {}
    }
    const pre = code.parentElement;
    const box2 = document.createElement('div');
    box2.className = 'code-block';
    pre.replaceWith(box2); box2.appendChild(pre);
    const btn = document.createElement('button');
    btn.className = 'copy-btn'; btn.type = 'button'; btn.textContent = '复制';
    btn.onclick = async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        btn.textContent = '已复制'; btn.classList.add('ok');
        setTimeout(() => { btn.textContent = '复制'; btn.classList.remove('ok'); }, 1400);
      } catch { btn.textContent = '复制失败'; }
    };
    box2.appendChild(btn);
  });

  return wrap;
}

/* ============================================================ 侧栏导航 */
function renderNav() {
  const nav = $('#nav');
  nav.innerHTML = '';
  const groups = [];
  CH.forEach(c => {
    let g = groups.find(x => x.name === c.group);
    if (!g) groups.push(g = { name: c.group, items: [] });
    g.items.push(c);
  });
  groups.forEach(g => {
    const box = document.createElement('div');
    box.className = 'nav-group';
    box.innerHTML = `<div class="nav-group-title">${esc(g.name)}</div>` + g.items.map(c => `
      <a class="nav-item${done.has(c.id) ? ' done' : ''}" href="#/c/${c.id}" data-id="${c.id}">
        <span class="nav-num">${c.id.slice(0, 2)}</span>
        <span class="nav-label">${esc(c.title)}</span>
        <span class="nav-check">✓</span>
      </a>`).join('');
    nav.appendChild(box);
  });
  $('#build-info').textContent = `${CH.length} 章 · ${QS.length} 道自测题 · ${QS.filter(q => q.l === 3).length} 道必答`;
}

function syncProgress() {
  const pct = Math.round(done.size / CH.length * 100);
  const C = 2 * Math.PI * 19;
  const fg = $('#ring-fg');
  fg.style.strokeDashoffset = String(C * (1 - pct / 100));
  fg.style.opacity = pct === 0 ? '0' : '1';   // 0% 时不要露出圆头小点
  $('#ring-label').textContent = pct + '%';
  $('#progress-text').textContent = `${done.size} / ${CH.length} 章已掌握`;
  $$('.nav-item').forEach(a => a.classList.toggle('done', done.has(a.dataset.id)));
}

function toggleDone(id) {
  done.has(id) ? done.delete(id) : done.add(id);
  store.done = [...done]; save(); syncProgress();
}

$('#btn-reset').onclick = () => {
  if (!confirm('确定要清空复习进度和答题记录吗？')) return;
  done.clear(); store.done = []; store.marks = {}; save(); syncProgress(); route();
};

/* ============================================================ 视图：首页 */
function viewHome() {
  const totalWords = CH.reduce((s, c) => s + c.words, 0);
  const groups = [];
  CH.forEach(c => {
    let g = groups.find(x => x.name === c.group);
    if (!g) groups.push(g = { name: c.group, items: [] });
    g.items.push(c);
  });

  const el = document.createElement('div');
  el.className = 'home';
  el.innerHTML = `
    <section class="hero">
      <span class="hero-eyebrow">校招 / 实习 · 全方向覆盖</span>
      <h1>算法岗面试<em>复习手册</em></h1>
      <p>把大模型、推荐搜索广告、CV 多模态和通用机器学习的面试考点整理成一套可以按章推进的复习材料。
         每个知识点按「<b>问题 → 答案骨架 → 追问</b>」组织，因为面试官就是这么问的。</p>
      <div class="hero-stats">
        <div class="stat"><b>${CH.length}</b><span>章系统笔记</span></div>
        <div class="stat"><b>${QS.length}</b><span>道高频自测题</span></div>
        <div class="stat"><b>${(totalWords / 1000).toFixed(0)}k</b><span>字内容</span></div>
        <div class="stat"><b>${QS.filter(q => q.l === 3).length}</b><span>道必答题</span></div>
      </div>
    </section>
    ${groups.map(g => `
      <div class="section-head"><h2>${esc(g.name)}</h2><span>${g.items.length} 章</span></div>
      <div class="cards">
        ${g.items.map(c => `
          <a class="card${done.has(c.id) ? ' done' : ''}" href="#/c/${c.id}">
            <div class="card-top">
              <span class="card-icon">${c.icon}</span>
              <b>${esc(c.title)}</b>
              <span class="card-num">${c.id.slice(0, 2)}</span>
            </div>
            <p>${esc(c.desc)}</p>
            <div class="card-foot">
              <span>约 ${(c.words / 1000).toFixed(1)}k 字</span>
              <span class="done-badge">✓ 已掌握</span>
            </div>
          </a>`).join('')}
      </div>`).join('')}
  `;
  return el;
}

/* ============================================================ 视图：章节 */
function viewChapter(c) {
  const idx = CH.indexOf(c);
  const prev = CH[idx - 1], next = CH[idx + 1];

  const art = document.createElement('article');
  const head = document.createElement('div');
  head.className = 'chapter-head';
  head.innerHTML = `
    <div class="chapter-kicker">
      <span>${esc(c.group)}</span><span class="dot"></span>
      <span>第 ${c.id.slice(0, 2)} 章</span><span class="dot"></span>
      <span>约 ${(c.words / 1000).toFixed(1)}k 字</span>
    </div>
    <button class="mark-btn${done.has(c.id) ? ' on' : ''}" id="mark">
      <svg viewBox="0 0 24 24"><path d="M20 6L9 17l-5-5"/></svg>
      <span>${done.has(c.id) ? '已掌握' : '标记为已掌握'}</span>
    </button>`;
  art.appendChild(head);
  art.appendChild(renderMarkdown(c.body));

  const foot = document.createElement('div');
  foot.className = 'chapter-nav';
  if (prev) foot.innerHTML += `<a href="#/c/${prev.id}"><small>← 上一章</small><b>${esc(prev.title)}</b></a>`;
  if (next) foot.innerHTML += `<a class="next" href="#/c/${next.id}"><small>下一章 →</small><b>${esc(next.title)}</b></a>`;
  art.appendChild(foot);

  head.querySelector('#mark').onclick = e => {
    toggleDone(c.id);
    const on = done.has(c.id);
    e.currentTarget.classList.toggle('on', on);
    e.currentTarget.querySelector('span').textContent = on ? '已掌握' : '标记为已掌握';
  };
  return art;
}

/* ============================================================ 右侧 TOC + 滚动高亮 */
let spy = [];
function buildToc(root) {
  const toc = $('#toc');
  const hs = $$('h2, h3', root);
  if (hs.length < 3) { toc.innerHTML = ''; spy = []; return; }
  toc.innerHTML = `<div class="toc-title">本章目录</div>` + hs.map(h =>
    `<a href="#${h.id}" data-t="${h.id}" class="${h.tagName === 'H3' ? 'lv3' : ''}">${esc(h.textContent)}</a>`
  ).join('');
  $$('a', toc).forEach(a => {
    a.onclick = e => {
      e.preventDefault();
      const t = document.getElementById(a.dataset.t);
      if (t) { window.scrollTo({ top: t.offsetTop - 72, behavior: 'smooth' }); history.replaceState(null, '', location.hash.split('#')[1] ? location.hash : location.hash); }
    };
  });
  spy = hs.map(h => ({ id: h.id, el: h }));
}

function onScroll() {
  const doc = document.documentElement;
  const max = doc.scrollHeight - innerHeight;
  $('#reading-bar').style.width = (max > 0 ? Math.min(100, scrollY / max * 100) : 0) + '%';
  if (!spy.length) return;
  let cur = spy[0].id;
  for (const s of spy) { if (s.el.getBoundingClientRect().top <= 110) cur = s.id; else break; }
  $$('#toc a').forEach(a => a.classList.toggle('active', a.dataset.t === cur));
}
addEventListener('scroll', onScroll, { passive: true });

/* ============================================================ 路由 */
function route() {
  const m = location.hash.match(/^#\/c\/([\w-]+)/);
  const main = $('#main');
  main.innerHTML = '';
  closeSidebar();

  if (m) {
    const c = CH.find(x => x.id === m[1]);
    if (c) {
      main.appendChild(viewChapter(c));
      buildToc(main);
      $$('.nav-item').forEach(a => a.classList.toggle('active', a.dataset.id === c.id));
      document.title = `${c.title} · 算法岗面试复习手册`;
      scrollTo(0, 0); onScroll();
      return;
    }
  }
  main.appendChild(viewHome());
  $('#toc').innerHTML = ''; spy = [];
  $$('.nav-item').forEach(a => a.classList.remove('active'));
  document.title = '算法岗面试复习手册';
  scrollTo(0, 0); onScroll();
}
addEventListener('hashchange', route);

/* ============================================================ 移动端抽屉 */
function openSidebar()  { $('#sidebar').classList.add('open');  $('#scrim').classList.add('on'); }
function closeSidebar() { $('#sidebar').classList.remove('open'); $('#scrim').classList.remove('on'); }
$('#btn-menu').onclick = () => $('#sidebar').classList.contains('open') ? closeSidebar() : openSidebar();
$('#scrim').onclick = closeSidebar;

/* ============================================================ 搜索 */
const INDEX = (() => {
  const out = [];
  const stripMath = t => t.replace(/\$\$[\s\S]*?\$\$/g, ' ').replace(/\$[^$\n]*\$/g, ' ');
  for (const c of CH) {
    if (c.id === '12-question-bank') continue;   // 题库单独入索引，避免重复
    // 按二级/三级标题切成小节，命中后可以直接定位
    const blocks = c.body.split(/\n(?=#{2,3} )/);
    for (const b of blocks) {
      const hm = b.match(/^#{1,3} (.+)/);
      const heading = hm ? hm[1].trim() : c.title;
      const text = stripMath(b.replace(/^#{1,3} .+/, '').replace(/```[\s\S]*?```/g, ' '))
                    .replace(/[|>#*`_\[\]()-]/g, ' ')
                    .replace(/\s+/g, ' ').trim();
      if (text.length < 12) continue;
      out.push({ cid: c.id, cshort: c.short, heading, text, low: (heading + ' ' + text).toLowerCase() });
    }
  }
  // 题库也进索引
  for (const q of QS) {
    out.push({ cid: '12-question-bank', cshort: '题库', heading: stripMath(q.q).replace(/[*`]/g, ''),
               text: stripMath(q.a).replace(/[*`]/g, '').replace(/\s+/g, ' ').trim(),
               low: (q.q + ' ' + q.a).toLowerCase() });
  }
  return out;
})();

function search(query) {
  const toks = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!toks.length) return [];
  const hits = [];
  for (const it of INDEX) {
    let score = 0, ok = true;
    for (const t of toks) {
      if (!it.low.includes(t)) { ok = false; break; }
      score += it.heading.toLowerCase().includes(t) ? 6 : 1;
    }
    if (ok) hits.push({ it, score });
  }
  return hits.sort((a, b) => b.score - a.score).slice(0, 40).map(h => h.it);
}

function snippet(text, toks) {
  const low = text.toLowerCase();
  let at = -1;
  for (const t of toks) { const i = low.indexOf(t); if (i >= 0 && (at < 0 || i < at)) at = i; }
  const from = Math.max(0, at - 40);
  let s = esc(text.slice(from, from + 190));
  if (from > 0) s = '…' + s;
  for (const t of toks) {
    s = s.replace(new RegExp(`(${t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'), '<mark>$1</mark>');
  }
  return s;
}

const so = $('#search-overlay'), si = $('#search-input'), sr = $('#search-results');
let selIdx = 0;

function openSearch() {
  so.hidden = false; si.value = ''; selIdx = 0;
  sr.innerHTML = `<div class="sr-empty">输入关键词搜索 ${CH.length} 章正文与 ${QS.length} 道题库</div>`;
  si.focus();
}
function closeSearch() { so.hidden = true; }

function runSearch() {
  const q = si.value.trim();
  const toks = q.toLowerCase().split(/\s+/).filter(Boolean);
  if (!q) { sr.innerHTML = `<div class="sr-empty">输入关键词搜索 ${CH.length} 章正文与 ${QS.length} 道题库</div>`; return; }
  const res = search(q);
  selIdx = 0;
  if (!res.length) { sr.innerHTML = `<div class="sr-empty">没有找到「${esc(q)}」相关内容</div>`; return; }
  sr.innerHTML = res.map((r, i) => `
    <a class="sr-item${i === 0 ? ' sel' : ''}" href="#/c/${r.cid}" data-i="${i}">
      <div class="sr-top"><span class="sr-chap">${esc(r.cshort)}</span><span class="sr-head">${esc(r.heading)}</span></div>
      <div class="sr-snip">${snippet(r.text, toks)}</div>
    </a>`).join('') + `<div class="sr-hint">↑↓ 选择 · Enter 打开 · Esc 关闭</div>`;
  $$('.sr-item', sr).forEach(a => a.onclick = closeSearch);
}

si.addEventListener('input', runSearch);
$('#btn-search').onclick = openSearch;
so.addEventListener('click', e => { if (e.target === so) closeSearch(); });

si.addEventListener('keydown', e => {
  const items = $$('.sr-item', sr);
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    if (!items.length) return;
    selIdx = (selIdx + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
    items.forEach((a, i) => a.classList.toggle('sel', i === selIdx));
    items[selIdx].scrollIntoView({ block: 'nearest' });
  } else if (e.key === 'Enter') {
    e.preventDefault();
    if (items[selIdx]) { location.hash = items[selIdx].getAttribute('href'); closeSearch(); }
  }
});

/* ============================================================ 随机抽题 */
const qo = $('#quiz-overlay');
const QF = { cats: new Set(), levels: new Set() };
let deck = [], cur = null, revealed = false;

function pool() {
  return QS.filter(q =>
    (!QF.cats.size   || QF.cats.has(q.c)) &&
    (!QF.levels.size || QF.levels.has(q.l)));
}

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

function nextQuestion() {
  if (!deck.length) deck = shuffle(pool().slice());
  cur = deck.pop() || null;
  revealed = false;
  renderQuiz();
}

function renderFilters() {
  const f = $('#quiz-filters');
  f.innerHTML =
    BANK.categories.map(c => {
      const n = QS.filter(q => q.c === c.id).length;
      return `<button class="chip${QF.cats.has(c.id) ? ' on' : ''}" data-cat="${c.id}">${esc(c.name)} ${n}</button>`;
    }).join('') +
    [3, 2, 1].map(l => `<button class="chip${QF.levels.has(l) ? ' on' : ''}" data-lv="${l}">${'★'.repeat(l)}</button>`).join('');

  $$('[data-cat]', f).forEach(b => b.onclick = () => {
    const id = b.dataset.cat;
    QF.cats.has(id) ? QF.cats.delete(id) : QF.cats.add(id);
    deck = []; renderFilters(); nextQuestion();
  });
  $$('[data-lv]', f).forEach(b => b.onclick = () => {
    const l = +b.dataset.lv;
    QF.levels.has(l) ? QF.levels.delete(l) : QF.levels.add(l);
    deck = []; renderFilters(); nextQuestion();
  });
}

function inlineMd(s) {
  const [p, box] = protectMath(s);
  let html = marked.parseInline(p);
  return restoreMath(html, box);
}

function renderQuiz() {
  const body = $('#quiz-body'), foot = $('#quiz-actions'), stat = $('#quiz-stat');
  const p = pool();

  if (!cur) {
    body.innerHTML = `<div class="quiz-empty">当前筛选条件下没有题目，换个方向或星级试试。</div>`;
    foot.innerHTML = ''; stat.textContent = '';
    return;
  }

  body.innerHTML = `
    <div class="q-meta">
      <span class="q-tag">${esc(CAT[cur.c].name)}</span>
      <span class="q-stars">${'★'.repeat(cur.l)}${'☆'.repeat(3 - cur.l)}</span>
    </div>
    <div class="q-text">${inlineMd(cur.q)}</div>
    ${revealed
      ? `<div class="q-answer">${inlineMd(cur.a)}</div>
         <a class="q-link" href="#/c/${CAT[cur.c].chapter}" data-go>去看这一章的详细讲解 →</a>`
      : `<div class="q-hidden" id="reveal">点击或按空格查看答案要点</div>`}
  `;

  const rv = $('#reveal', body);
  if (rv) rv.onclick = () => { revealed = true; renderQuiz(); };
  const go = $('[data-go]', body);
  if (go) go.onclick = () => { qo.hidden = true; };

  foot.innerHTML = revealed
    ? `<button class="pill-btn" data-mark="weak">还不熟 (1)</button>
       <button class="pill-btn accent" data-mark="ok">会了 (2)</button>`
    : `<button class="pill-btn" data-skip>跳过</button>
       <button class="pill-btn accent" data-reveal>看答案 (空格)</button>`;

  const rb = $('[data-reveal]', foot); if (rb) rb.onclick = () => { revealed = true; renderQuiz(); };
  const sb = $('[data-skip]', foot);   if (sb) sb.onclick = nextQuestion;
  $$('[data-mark]', foot).forEach(b => b.onclick = () => {
    store.marks[cur.id] = b.dataset.mark; save(); nextQuestion();
  });

  const marks = Object.entries(store.marks);
  const okN = marks.filter(([, v]) => v === 'ok').length;
  const weakN = marks.filter(([, v]) => v === 'weak').length;
  stat.textContent = `候选 ${p.length} 题 · 本轮剩 ${deck.length} · 已标会 ${okN} · 待加强 ${weakN}`;
}

function openQuiz() { qo.hidden = false; renderFilters(); if (!cur) nextQuestion(); else renderQuiz(); }
$('#btn-quiz').onclick = openQuiz;
$('#quiz-close').onclick = () => { qo.hidden = true; };
qo.addEventListener('click', e => { if (e.target === qo) qo.hidden = true; });

/* ============================================================ 快捷键 */
addEventListener('keydown', e => {
  const typing = /^(INPUT|TEXTAREA)$/.test(e.target.tagName);
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openSearch(); return; }
  if (e.key === 'Escape') { closeSearch(); qo.hidden = true; closeSidebar(); return; }
  if (typing) return;
  if (e.key === '/') { e.preventDefault(); openSearch(); return; }

  if (!qo.hidden) {
    if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); if (!revealed) { revealed = true; renderQuiz(); } else nextQuestion(); }
    else if (e.key === '1' && revealed) { store.marks[cur.id] = 'weak'; save(); nextQuestion(); }
    else if (e.key === '2' && revealed) { store.marks[cur.id] = 'ok';   save(); nextQuestion(); }
    else if (e.key.toLowerCase() === 'n') nextQuestion();
    return;
  }

  // 章节间翻页
  const m = location.hash.match(/^#\/c\/([\w-]+)/);
  if (!m) return;
  const i = CH.findIndex(c => c.id === m[1]);
  if (e.key === 'ArrowLeft'  && CH[i - 1]) location.hash = `#/c/${CH[i - 1].id}`;
  if (e.key === 'ArrowRight' && CH[i + 1]) location.hash = `#/c/${CH[i + 1].id}`;
});

/* ============================================================ 启动 */
renderNav();
syncProgress();
route();
})();
