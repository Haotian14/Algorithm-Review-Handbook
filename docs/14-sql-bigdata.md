# 14 · SQL 与大数据

> 很多人栽在这一章 —— 准备了三个月的模型，结果一面上来让写一道窗口函数 SQL。
> 算法岗每天都要从数仓里捞数据、做样本、算指标，**SQL 是基本功而不是加分项**。

## 本章高频考点

> 星级是**主观判断**，反映常见面经里的印象分布，不是统计数据。用来排复习优先级即可。

| 考点 | 频率 | 典型问法 |
|---|---|---|
| 窗口函数 | ★★★ | 分组 Top-N、留存、同环比 |
| 留存率 / 漏斗 | ★★★ | 写一个次日留存 |
| JOIN 与数据倾斜 | ★★★ | 某个 key 特别大怎么办 |
| 行列转换 | ★★ | 宽表转长表 |
| 连续 N 天问题 | ★★ | 连续登录 3 天的用户 |
| MapReduce / Spark 原理 | ★★ | shuffle 是什么，为什么慢 |
| 批流架构 | ★★ | Lambda vs Kappa |
| Hive 调优 | ★★ | 小文件、倾斜、谓词下推 |

---

## 1. 窗口函数（必须熟到不用想）

```sql
<函数>() OVER (
    PARTITION BY 分组列
    ORDER BY 排序列
    ROWS BETWEEN <起点> AND <终点>   -- 可选，滑动窗口
)
```

### 1.1 三个排名函数的区别（高频）

```sql
SELECT
    name, score,
    ROW_NUMBER() OVER (ORDER BY score DESC) AS rn,    -- 1,2,3,4  不重复
    RANK()       OVER (ORDER BY score DESC) AS rk,    -- 1,2,2,4  并列跳号
    DENSE_RANK() OVER (ORDER BY score DESC) AS drk    -- 1,2,2,3  并列不跳号
FROM scores;
```

**记忆**：`ROW_NUMBER` 强行不并列；`RANK` 并列后跳号（像比赛名次）；`DENSE_RANK` 并列后不跳号。

### 1.2 分组 Top-N（最经典的一道）

**每个部门薪资最高的 3 个人：**

```sql
SELECT dept_id, emp_id, salary
FROM (
    SELECT dept_id, emp_id, salary,
           DENSE_RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rk
    FROM employee
) t
WHERE rk <= 3;
```

**注意**：用 `DENSE_RANK` 还是 `ROW_NUMBER` 取决于「薪资并列算不算同一名」—— **面试时要主动问这个**。

### 1.3 偏移函数与滑动窗口

```sql
-- 每个用户相邻两次行为的时间间隔
SELECT user_id, event_time,
       LAG(event_time)  OVER (PARTITION BY user_id ORDER BY event_time) AS prev_time,
       LEAD(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS next_time
FROM events;

-- 7 日移动平均（含当天，往前 6 天）
SELECT dt, dau,
       AVG(dau) OVER (ORDER BY dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7
FROM daily_active;

-- 累计求和
SELECT dt, revenue,
       SUM(revenue) OVER (ORDER BY dt ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum
FROM daily_revenue;
```

**追问**：`ROWS` 和 `RANGE` 有什么区别？
`ROWS` 按**物理行数**取窗口；`RANGE` 按**值的范围**取，排序列值相同的行会被算作同一组。
做移动平均一般要用 `ROWS`，用 `RANGE` 在有重复日期时结果会不对。

---

## 2. 高频业务题

### 2.1 次日留存率

```sql
-- 口径：某天首次活跃的用户中，第二天还活跃的比例
WITH first_day AS (           -- 每个用户的首次活跃日
    SELECT user_id, MIN(dt) AS first_dt
    FROM user_active
    GROUP BY user_id
)
SELECT
    f.first_dt,
    COUNT(DISTINCT f.user_id)                                        AS new_users,
    COUNT(DISTINCT a.user_id)                                        AS retained,
    COUNT(DISTINCT a.user_id) * 1.0 / COUNT(DISTINCT f.user_id)      AS retention_d1
FROM first_day f
LEFT JOIN user_active a
       ON f.user_id = a.user_id
      AND a.dt = DATE_ADD(f.first_dt, 1)     -- 关键：只关联第二天
GROUP BY f.first_dt;
```

**面试要点**：
- **必须先问口径** —— 「留存」是相对首次活跃还是相对当天活跃？新用户还是全量用户？
- 分母要用 `COUNT(DISTINCT)`，防止一天多条记录重复计数
- 整数相除会截断，记得 `* 1.0`（Hive/MySQL 都要注意）
- 要 N 日留存就把 `DATE_ADD(..., 1)` 换成 `n`，或用 `DATEDIFF` 分组一次算出多个

### 2.2 连续登录 N 天（经典技巧题）

**核心技巧**：日期减去行号，连续的日期会得到**同一个常量**。

```sql
WITH d AS (
    SELECT DISTINCT user_id, dt FROM user_active      -- 先去重
),
g AS (
    SELECT user_id, dt,
           DATE_SUB(dt, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY dt)) AS grp
    FROM d
)
SELECT user_id, MIN(dt) AS start_dt, MAX(dt) AS end_dt, COUNT(*) AS days
FROM g
GROUP BY user_id, grp
HAVING COUNT(*) >= 3;
```

**为什么成立**：日期每天 +1，行号每行 +1，两者相减对连续区间恒定；一旦断开，差值就跳变。
**面试时一定要把这个原理说出来**，只写代码不解释会被认为是背的。

### 2.3 漏斗转化

```sql
-- 曝光 → 点击 → 下单 三步转化
SELECT
    COUNT(DISTINCT CASE WHEN step >= 1 THEN user_id END) AS impression,
    COUNT(DISTINCT CASE WHEN step >= 2 THEN user_id END) AS click,
    COUNT(DISTINCT CASE WHEN step >= 3 THEN user_id END) AS order_cnt,
    COUNT(DISTINCT CASE WHEN step >= 2 THEN user_id END) * 1.0
        / COUNT(DISTINCT CASE WHEN step >= 1 THEN user_id END) AS ctr
FROM user_funnel;
```

**追问：要求步骤必须按顺序发生怎么办？**
用窗口函数取每个用户各步骤的最早时间，再判断时间是否递增；
或者用 `LAG` 检查前一步是否在当前步之前完成。这类「有序漏斗」在真实场景里更常见。

### 2.4 行列转换

```sql
-- 长转宽（行转列）
SELECT user_id,
       MAX(CASE WHEN item = 'A' THEN cnt END) AS a_cnt,
       MAX(CASE WHEN item = 'B' THEN cnt END) AS b_cnt
FROM t GROUP BY user_id;

-- 宽转长（列转行，Hive）
SELECT user_id, item, cnt
FROM wide_table
LATERAL VIEW explode(map('A', a_cnt, 'B', b_cnt)) tmp AS item, cnt;
```

### 2.5 去重与近似去重

```sql
COUNT(DISTINCT user_id)                    -- 精确，但 shuffle 到单个 reducer，数据大时会卡死
APPROX_COUNT_DISTINCT(user_id)             -- HyperLogLog，误差约 2%，速度快几个量级
```

**面试要点**：能主动说出「`COUNT(DISTINCT)` 在超大表上是性能杀手，因为所有值要汇聚到一个 reducer 去重；
UV 这类指标可以用 HLL 近似」，是有实战经验的信号。

---

## 3. 数据倾斜（大数据面试必问）

**现象**：99% 的 task 几分钟跑完，剩下 1 个跑几小时甚至 OOM。

**根因**：shuffle 后某个 key 的数据量远大于其他 key，落到同一个 reducer。

**常见来源**：`NULL` 值、爬虫/机器人的超级用户 ID、热门商品 ID、默认值 `-1` / `unknown`。

### 解法（按场景选，面试要能说出对应关系）

| 场景 | 解法 |
|---|---|
| 大表 JOIN 小表 | **Map Join**（把小表广播到每个 mapper，完全避免 shuffle）—— 首选 |
| 空值/默认值倾斜 | 提前过滤掉，或给 NULL 拼随机后缀打散 |
| 单个 key 极大 | **加盐**：key 拼随机数 `key_rand(0,n)` 打散聚合，再二次聚合去掉盐 |
| GROUP BY 倾斜 | 两阶段聚合：先局部预聚合（加盐），再全局聚合 |
| 少数几个热 key | 热 key 单独拎出来处理，其余走正常链路，最后 UNION |
| 参数级缓解 | `hive.groupby.skewindata=true`、Spark AQE 的自动倾斜 JOIN 优化 |

**加盐的具体写法**：

```sql
-- 第一阶段：加盐局部聚合
SELECT CONCAT(key, '_', CAST(FLOOR(RAND() * 10) AS STRING)) AS salted_key,
       SUM(v) AS partial
FROM t GROUP BY 1
-- 第二阶段：去盐全局聚合
-- SELECT SPLIT(salted_key,'_')[0] AS key, SUM(partial) FROM ... GROUP BY 1
```

---

## 4. MapReduce / Spark 原理

### 4.1 shuffle 为什么是瓶颈

Map 端输出按 key 分区、排序、溢写磁盘 → 网络传输到 Reduce 端 → Reduce 端归并排序。
**涉及磁盘 IO + 网络 IO + 序列化**，是分布式计算里最贵的一步。
所有大数据调优的核心思路都是：**减少 shuffle 的数据量，或干脆避免 shuffle**。

### 4.2 Spark 核心概念

| 概念 | 说明 |
|---|---|
| **RDD / DataFrame** | 不可变的分布式数据集；DataFrame 有 schema，能被 Catalyst 优化，**优先用它** |
| **窄依赖 / 宽依赖** | 窄依赖（map/filter）父分区一对一，可流水线执行；**宽依赖（groupBy/join）触发 shuffle**，是 Stage 的分界 |
| **DAG / Stage** | 按宽依赖切分 Stage，Stage 内并行执行 |
| **惰性求值** | transformation 只建 DAG，遇到 action 才真正执行 |
| **persist / cache** | 多次复用的中间结果要缓存，否则每个 action 都会重算整条链路 |
| **广播变量** | 小表广播到各 executor，实现 Map Join |
| **AQE** | 自适应查询执行：运行时根据实际数据量调整分区数、自动处理倾斜 JOIN、动态切换 JOIN 策略 |

**Spark 为什么比 MapReduce 快？**
① 中间结果在内存而不是每步落磁盘；② DAG 调度能把多个窄依赖算子合并成一个 Stage 流水线执行，
减少落盘次数；③ Catalyst / Tungsten 做了执行计划优化和内存管理。
**注意**：不是「Spark 全在内存」—— 内存放不下照样溢写磁盘。

### 4.3 常见 OOM 与调优

| 问题 | 排查方向 |
|---|---|
| Driver OOM | 用了 `collect()` 把大结果拉回 driver；广播了太大的表 |
| Executor OOM | 单分区数据过大（倾斜）；缓存太多；`executor-memory` 不足 |
| 跑得慢 | 分区数不合理（太少并行度不够，太多调度开销大）；shuffle 量大；小文件太多 |
| 小文件问题 | 输出前 `coalesce` / `repartition` 合并；Hive 开启 merge 参数 |

---

## 5. 批流架构

| 架构 | 结构 | 优缺点 |
|---|---|---|
| **Lambda** | 批处理层（准确）+ 流处理层（低延迟）并行，服务层合并 | 准确性和实时性兼顾，但**两套代码两套逻辑，维护成本高、易不一致** |
| **Kappa** | 只有流处理，历史数据靠重放消息队列 | 一套代码，简洁；但重放大量历史数据成本高，且要求消息保留足够久 |

**流处理的三个语义**：at-most-once（可能丢）、at-least-once（可能重）、
**exactly-once**（Flink 靠 checkpoint + 两阶段提交实现，需要下游支持幂等或事务）。

**Flink 的两个关键概念**：
- **事件时间 vs 处理时间**：算指标要用**事件时间**，否则数据延迟到达会算错
- **Watermark**：用来判断「事件时间小于 T 的数据应该都到齐了」，从而触发窗口计算。
  迟到太久的数据进侧输出流单独处理

**和算法岗的关系**：实时特征、实时样本拼接（曝光流 join 点击流）、实时指标监控都靠这套。
面试被问到「实时特征怎么算」，答案就是 Flink 消费行为流做窗口聚合，写入 Redis/Feature Store。

---

## 6. 算法岗最常写的几类查询

```sql
-- ① 构造训练样本：曝光表 left join 点击表，未点击即负样本
SELECT i.request_id, i.user_id, i.item_id, i.features,
       CASE WHEN c.item_id IS NOT NULL THEN 1 ELSE 0 END AS label
FROM impression i
LEFT JOIN click c
       ON i.request_id = c.request_id AND i.item_id = c.item_id
WHERE i.dt = '2026-09-01';

-- ② 用户近 30 天行为序列（注意截断，防止超长序列）
SELECT user_id,
       CONCAT_WS(',', COLLECT_LIST(item_id)) AS seq   -- Hive
FROM (
    SELECT user_id, item_id,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts DESC) AS rn
    FROM behavior
    WHERE dt BETWEEN '2026-08-03' AND '2026-09-01'
) t
WHERE rn <= 50
GROUP BY user_id;

-- ③ AB 实验指标对比
SELECT exp_group,
       COUNT(DISTINCT user_id)                          AS uv,
       SUM(click) * 1.0 / SUM(impression)               AS ctr,
       SUM(duration) * 1.0 / COUNT(DISTINCT user_id)    AS avg_duration
FROM exp_log
WHERE dt BETWEEN '2026-09-01' AND '2026-09-07'
GROUP BY exp_group;
```

**样本构造里最容易出的错**：
- 时间窗口跨界导致**标签穿越**（用了曝光之后才发生的信息当特征）
- `LEFT JOIN` 写成 `INNER JOIN` → 负样本全丢，模型只见过正样本
- 关联键不完整（只按 item_id 不带 request_id）→ 一次曝光匹配到多次点击，标签重复放大

---

## 7. SQL 面试的答题习惯

1. **先问口径**：留存/活跃/转化怎么定义？时间窗口多长？要不要去重？
2. **先说思路再写**：「我打算先用窗口函数给每个分组排名，再过滤」
3. **写完自己走一遍**：举 2-3 行数据口述结果
4. **主动说边界**：NULL 怎么处理？除零？重复记录？
5. **主动说性能**：这个 `COUNT(DISTINCT)` 在大表上会慢，可以换 HLL

---

## 延伸阅读

> 未附链接：成稿环境无法联网核验，标题可直接搜索到原文。

- *MapReduce: Simplified Data Processing on Large Clusters*（Dean & Ghemawat, OSDI 2004）
- *Resilient Distributed Datasets*（Zaharia et al., NSDI 2012）—— Spark RDD 原始论文
- *Spark SQL: Relational Data Processing in Spark*（Armbrust et al., SIGMOD 2015）—— Catalyst 优化器
- *Apache Flink: Stream and Batch Processing in a Single Engine*（Carbone et al., 2015）
- *Lightweight Asynchronous Snapshots for Distributed Dataflows*（Carbone et al., 2015）—— Flink checkpoint 与 exactly-once
- *Designing Data-Intensive Applications*（Kleppmann, 2017）—— 批流、一致性语义的系统性参考
- *The Dataflow Model*（Akidau et al., VLDB 2015）—— 事件时间与 watermark 的理论基础

---

## 自测清单

- [ ] 能说清 ROW_NUMBER / RANK / DENSE_RANK 的区别，并知道分组 Top-N 该选哪个
- [ ] 能默写次日留存的 SQL，并说出三个容易出错的点
- [ ] 能写连续登录 N 天，并解释「日期减行号」为什么成立
- [ ] 能说清 ROWS 和 RANGE 的区别
- [ ] 能列出 5 种以上数据倾斜的解法，并对应到具体场景
- [ ] 能解释 shuffle 为什么慢、宽窄依赖如何切分 Stage
- [ ] 能说清 Spark 比 MapReduce 快的三个原因（且不说错成「全在内存」）
- [ ] 能对比 Lambda 和 Kappa 架构，说清 exactly-once 怎么实现
- [ ] 能写出训练样本构造的 SQL，并说出标签穿越的风险
