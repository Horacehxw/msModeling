---
name: project-status
description: Generate perf-database project progress dashboard. Default=concise console view; ALL=full PDF report; TCX/ZH/HDY/etc=person-focused view
---

# 算子性能数据库 — 项目进展看板

为 SE 生成可视化进度报告，整合代码、文档、日报、站会纪要等多维信息源。

## 参数解析

| 调用方式 | 行为 |
|---------|------|
| `/project-status` (无参数) | **简化版**: 直接在 console 输出。聚焦整体进度、TOP 风险、需要 SE 关注的决策点、下一步建议。~200 行以内 |
| `/project-status ALL` | **完整版**: 生成完整项目看板，保存到 `docs/perf_database/daily_project_status/进展看板_{YYYYMMDD}.md` (同时备份到 `/Users/horacehxw/Documents/hxw-华为/蚂蚁仿真器项目/AI进展总结\`)。格式美观，面向全体项目成员 |
| `/project-status <人员代号>` | **个人版**: 在 console 输出。包含全局摘要 + 该成员的专项关注事项、阻塞、下一步建议。代号: TCX/ZH/ZZY/HDY/LJW/DSH/QCX/HXW |

## 数据采集步骤

**所有模式共用**。使用 Agent 工具并行采集以下 5 类信息源。

### Step 1: 并行采集 (启动 3-4 个 Agent)

**Agent 1: 代码仓库分析**
```
1. git fetch --all
2. git log --oneline -20 feat/perf-database (主线进展)
3. git branch -r --list 'gitcode/*' | 逐分支统计领先/落后 feat/perf-database 的 commit 数
4. git log --oneline gitcode-ascend/develop --not feat/perf-database (主仓新增)
5. pytest tests/perf_database/ --ignore=tests/perf_database/test_reference_data_e2e.py -q (测试状态)
6. git diff --stat remotes/gitcode/develop..feat/perf-database | tail -3 (变更规模)
```

**Agent 2: 文档分析 (Design Doc 为合规基准)**
```
1. 读取 docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.4.md
   - §4 核心模块: 逐模块检查实现 vs spec 偏差
   - §7.5 评估指标: M1-M6 实现状态
   - §8 开发计划: Phase 交付物 vs 实际完成
   - §9.1 融合 Gap: 各项状态
   - §1.4 验收标准: E2E <15%, 覆盖 >90%, 工具链 9 个
2. 读取 docs/perf_database/WORK_PLAN_Q1.md (v3.2)
   - 所有检查点的完成状态 (✅/🔄/⏳)
   - Phase 时间线 vs 当前日期
   - 风险表 R1-R14
3. 读取最新的 E2E 验证报告 (reports/phase1-e2e-*/phase1_e2e_*_verification_report_zh.md)
   - M1-M6 指标数据
   - MISS 根因分类
   - Phase 2/3 TODO 优先级
4. 读取 CHANGELOG (docs/perf_database/CHANGELOG_*.md)
5. 读取 docs/perf_database/METRICS_GUIDE.md (M1-M6 指标定义和用法)
```

**Agent 4: M1-M6 指标计算 (迭代优化)** (如果 profiling 数据可用)

⚠️ **核心原则: 迭代搜索最优 TC 参数以最大化 M4 (per-shape hit rate)**

目标: 找到使 M4 最高的 (nq, ql) 参数组合。不再依赖一次推导，而是:
1. 从 profiling 提取 M-dim 分布 + 从 perf database CSV 提取可用 shape 清单
2. 生成候选 (nq, ql)，优先选 CSV 中存在的 M-dim
3. 运行 TC → 分析 m4_miss_shape_list → 调整参数 → 重试
4. 最多 MAX_ITER 次迭代，选择 M4 最高的参数

```
# ==========================================
# 配置 (每次更新 profiling 数据时检查)
# ==========================================
PROF_BASE="/Users/horacehxw/Data/Profiling/Profiling-0325-final-vllm-new"
PROF_QWEN3="$PROF_BASE/profiler-qwen3-0325"
PROF_DSV3="$PROF_BASE/profiler-dsv3-0325"

DATA_DIR="$(pwd)/tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5"

M4_ACCEPT=0.80   # M4 高于此值时停止迭代
MAX_ITER=3        # 每个场景最多迭代次数

# ==========================================
# Step 0: Auto-discover profiling scenarios
# ==========================================
# 规则: 优先 rank0，优先最高 concurrency (最接近生产环境)
# 自动扫描目录，不依赖硬编码场景名
#
# 目录结构 (0325):
#   profiler-qwen3-0325/
#     profiler-qwen3-input4096-output1-concurrency{1,16}-rank{0,8}         (PF)
#     profiler-qwen3-input4096-output1536-concurrency{1,16}-rrate{1,2}-rank{0,8}  (DC)
#   profiler-dsv3-0325/
#     profiler-dsv3-input4096-output1[-concurrency1]-rank{0,8}             (PF)
#     profiler-dsv3-input4096-output1536-concurrency{1,8}-rrate{1,4}-rank{0,8}    (DC)

python3.10 -c "
import os, re
from collections import defaultdict

def discover_scenarios(prof_dir, model_prefix):
    '''Discover profiling scenarios, preferring rank0 + highest concurrency.'''
    scenarios = defaultdict(list)
    for d in sorted(os.listdir(prof_dir)):
        full = os.path.join(prof_dir, d)
        if not os.path.isdir(full): continue
        if not d.startswith(f'profiler-{model_prefix}-'): continue
        m = re.match(
            rf'profiler-{model_prefix}-input(\d+)-output(\d+)'
            r'(?:-concurrency(\d+))?(?:-rrate(\d+))?-rank(\d+)',
            d
        )
        if not m: continue
        inp, out, conc, rrate, rank = m.groups()
        conc = int(conc) if conc else 0
        rrate = int(rrate) if rrate else 0
        rank = int(rank)
        phase = 'PF' if out == '1' else 'DC'
        scenarios[phase].append((conc, rrate, rank, full))
    best = {}
    for phase, entries in scenarios.items():
        rank0 = [e for e in entries if e[2] == 0]
        if not rank0: rank0 = entries
        rank0.sort(key=lambda e: (-e[0], -e[1]))
        best[phase] = rank0[0][3]
    return best

# Output trace paths as shell variable assignments
for model, prefix, prof_dir in [
    ('QWEN3', 'qwen3', '$PROF_QWEN3'),
    ('DSV3', 'dsv3', '$PROF_DSV3'),
]:
    scenarios = discover_scenarios(prof_dir, prefix)
    for phase in ['PF', 'DC']:
        if phase in scenarios:
            print(f'{model}_{phase}_TRACE={scenarios[phase]}')
"
# 将上面的输出保存为 shell 变量, 后续 M6 使用

# ==========================================
# Step 0.5: Extract M-dim + CSV shape inventory
# ==========================================
# 两个目标:
#   1. 从 profiling kernel_details.csv 提取每个场景的 M-dim 分布
#   2. 从 perf database CSV 提取可用 M-dim 集合
# 两者的交集决定了候选 (nq, ql)

python3.10 -c "
import csv, os
from collections import Counter

DATA_DIR = '$DATA_DIR'

# --- 从 profiling 提取 M-dim 分布 ---
# 使用 Step 0 auto-discover 的 trace 路径
# AI 在执行时需要用实际发现的路径替换下面的占位符

print('=== Profiling M-dim distribution ===')
# 对 4 个场景逐一提取:
#   Qwen3 (BF16): 主计算 kernel = MatMulV2
#   DSv3 (W8A8):  主计算 kernel = QuantBatchMatmulV3
for name, kd_path, kernel_type in [
    ('Qwen3_PF', '<QWEN3_PF_TRACE>/kernel_details.csv', 'MatMulV2'),
    ('Qwen3_DC', '<QWEN3_DC_TRACE>/kernel_details.csv', 'MatMulV2'),
    ('DSv3_PF',  '<DSV3_PF_TRACE>/kernel_details.csv',  'QuantBatchMatmulV3'),
    ('DSv3_DC',  '<DSV3_DC_TRACE>/kernel_details.csv',  'QuantBatchMatmulV3'),
]:
    m_dims = Counter()
    try:
        with open(kd_path) as f:
            for row in csv.DictReader(f):
                if row.get('Type','').strip() == kernel_type:
                    shapes = row.get('Input Shapes','').strip('\"')
                    parts = shapes.split(';')
                    if parts:
                        first_dims = parts[0].strip('()').split(',')
                        m_dims[first_dims[0].strip()] += 1
    except FileNotFoundError:
        print(f'{name}: TRACE NOT FOUND at {kd_path}')
        continue
    dominant = m_dims.most_common(1)[0] if m_dims else ('?', 0)
    total = sum(m_dims.values())
    pct = dominant[1]/total*100 if total else 0
    print(f'{name}: {kernel_type} dominant M={dominant[0]} ({pct:.0f}%, {dominant[1]}/{total})')
    for m, count in m_dims.most_common(5):
        print(f'  M={m}: {count}')

# --- 从 perf database CSV 提取可用 M-dim ---
print()
print('=== CSV shape inventory ===')
for csv_file in ['MatMulV2.csv', 'QuantBatchMatmulV3.csv']:
    csv_path = os.path.join(DATA_DIR, csv_file)
    if not os.path.exists(csv_path):
        print(f'{csv_file}: NOT FOUND')
        continue
    m_dims = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            shapes = row.get('input_shapes', row.get('Input Shapes', '')).strip('\"')
            parts = shapes.split(';')
            if parts:
                dims = parts[0].strip('()').split(',')
                d = dims[0].strip()
                if d.isdigit():
                    m_dims.add(int(d))
    print(f'{csv_file}: {len(m_dims)} unique M-dims = {sorted(m_dims)[:30]}')
"

# ==========================================
# Step 1: 候选参数生成 + 迭代优化
# ==========================================
#
# 候选生成规则:
#   Prefill: nq * ql 应等于 (或在 block-padding 后等于) profiling dominant M-dim
#     候选列表 (假设 dominant M = D):
#       (1, D), (2, D//2), (4, D//4), (8, D//8), (16, D//16)
#       对每个还需尝试 block-padding 变体: ql±{16,32,64}
#     优先选择: nq*ql 的 M-dim 在 CSV shape inventory 中存在的候选
#
#   Decode: ql=1 固定, nq = profiling dominant M-dim
#     BF16 (Qwen3): NPU pad 到 {16,32,64}, 优先匹配 CSV 中存在的值
#     INT8 (DSv3): QuantBatchMatmulV3 不 pad M, 直接用 dominant M
#     候选: dominant_M, 以及 ceil(dominant_M / 16)*16 等 pad 变体
#
# 迭代优化循环 (伪代码, AI 需转化为实际执行):
#
#   best_m4 = 0; best_nq = nq; best_ql = ql
#   for iter in range(MAX_ITER):
#     1. 运行 TC (text_generate) → export-metrics JSON
#     2. 解析 JSON: m4 = r['m4']['m4_per_shape_hr']
#     3. if m4 > best_m4: best_m4 = m4; best_nq = nq; best_ql = ql
#     4. if m4 >= M4_ACCEPT: break (accepted)
#     5. 分析 r['m4']['m4_miss_shape_list'] + r['misses']:
#        - 分类 MISS 原因: shape_not_found vs kernel_not_mapped vs other
#        - 如果大部分是 kernel_not_mapped → 参数调整无效, break
#        - 如果大部分是 shape_not_found → 提取 dominant MISS M-dim
#     6. 调整 nq/ql 瞄准 MISS 中 dominant M-dim:
#        - PF: 尝试 nq=1 ql=dominant_miss_M (或其他分解)
#        - DC: 尝试 nq=dominant_miss_M ql=1
#     7. 继续下一轮
#   用 best 结果的 metrics JSON 作为最终报告

# TC 命令模板 (AI 需要根据 Step 0/0.5 结果填充 NQ/QL):

# --- Qwen3 Prefill ---
# --quantize-linear-action DISABLED (BF16)
# --enable-flashcomm-v1 对标 vLLM ENABLE_FLASHCOMM1=1 (仅 Prefill)
# Decode 不加 --enable-flashcomm-v1: FlashComm 仅用于 Prefill 的 allGather/reduceScatter,
# Decode 走 allReduce, 加此 flag 会改变 op graph 导致 M3 降低 ~18pp
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries $NQ --query-length $QL --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile --profiling-database "$DATA_DIR" \
  --enable-flashcomm-v1 \
  --export-metrics results/qwen3_prefill_metrics.json --log-level info

# --- Qwen3 Decode ---
# 注意: 不加 --enable-flashcomm-v1 (Decode 不使用 FlashComm)
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries $NQ --query-length 1 --context-length 4096 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile --profiling-database "$DATA_DIR" \
  --export-metrics results/qwen3_decode_metrics.json --log-level info

# --- DSv3 Prefill ---
# --enable-shared-expert-tp 对标 vLLM enable_expert_parallel 下的 shared expert TP 分片
# 不加此 flag 时 shared expert shapes 不按 TP 分片, 导致 M3 降低 ~10pp
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries $NQ --query-length $QL --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile --profiling-database "$DATA_DIR" \
  --enable-shared-expert-tp \
  --export-metrics results/dsv3_prefill_metrics.json --log-level info

# --- DSv3 Decode ---
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries $NQ --query-length 1 --context-length 4096 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile --profiling-database "$DATA_DIR" \
  --enable-shared-expert-tp \
  --export-metrics results/dsv3_decode_metrics.json --log-level info

# ==========================================
# Step 1.5: MISS 分析 + 参数调整 (每次迭代后执行)
# ==========================================
python3.10 -c "
import json, sys
from collections import Counter

path = sys.argv[1]  # e.g., results/qwen3_prefill_metrics.json
r = json.load(open(path))
m4 = r['m4']['m4_per_shape_hr']
miss_shapes = r['m4'].get('m4_miss_shape_list', [])
misses = r.get('misses', [])

print(f'M4 = {m4:.4f} ({m4*100:.1f}%)')
print(f'MISS count: {len(miss_shapes)}')

# Classify MISS reasons
shape_miss = 0
kernel_miss = 0
other_miss = 0
for m in misses:
    reason = m.get('reason', '').lower()
    if 'shape' in reason or 'not found' in reason:
        shape_miss += 1
    elif 'kernel' in reason or 'mapping' in reason or 'no entry' in reason:
        kernel_miss += 1
    else:
        other_miss += 1

print(f'MISS breakdown: shape_not_found={shape_miss} kernel_not_mapped={kernel_miss} other={other_miss}')

if kernel_miss > shape_miss:
    print('VERDICT: STOP — most MISSes are kernel coverage gaps, parameter tuning cannot help')
else:
    # Extract dominant MISS M-dim
    miss_m_dims = Counter()
    for m in miss_shapes:
        if m.get('shape') and m['shape'][0]:
            s = m['shape'][0]
            dim0 = s[0] if isinstance(s, list) else s
            miss_m_dims[dim0] += 1
    if miss_m_dims:
        top_miss = miss_m_dims.most_common(3)
        print(f'Top MISS M-dims: {top_miss}')
        print(f'VERDICT: ADJUST nq/ql to target M={top_miss[0][0]}')
    else:
        print('VERDICT: STOP — no actionable MISS pattern')

# Also show top MISS func_names for diagnosis
miss_funcs = Counter(m.get('func_name', '?') for m in misses)
print()
print('Top MISS ops:')
for fn, cnt in miss_funcs.most_common(10):
    print(f'  {fn}: {cnt}')
" results/qwen3_prefill_metrics.json

# ==========================================
# Step 2: M6 计算 (用迭代优化选出的最佳参数)
# ==========================================
# M6 = TC_prediction / real_per_fwd
# M6=1.0 完美, >1 高估, <1 低估. Phase 3 目标: 0.85-1.15
#
# compute_m6.py 自动用 anchor kernel (ArgMaxV2) 切分 forward passes, 取 Stage/N 为分母
# --profiler-output 使用 Step 0 auto-discover 的 trace 路径 (rank0, highest concurrency)
#
# AI 必须用 Step 0 实际发现的路径替换 $XXX_TRACE 变量:

python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/qwen3_prefill_metrics.json \
  --profiler-output "$QWEN3_PF_TRACE"

python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/qwen3_decode_metrics.json \
  --profiler-output "$QWEN3_DC_TRACE"

python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/dsv3_prefill_metrics.json \
  --profiler-output "$DSV3_PF_TRACE"

python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/dsv3_decode_metrics.json \
  --profiler-output "$DSV3_DC_TRACE"

# ==========================================
# Step 3: 汇总输出 (含迭代优化记录)
# ==========================================
python3.10 -c "
import json
print('场景                  M1     M2     M3     M4     M5     TC(ms)')
print('-' * 70)
for name, path in [
    ('Qwen3 PF', 'results/qwen3_prefill_metrics.json'),
    ('Qwen3 DC', 'results/qwen3_decode_metrics.json'),
    ('DSv3 PF',  'results/dsv3_prefill_metrics.json'),
    ('DSv3 DC',  'results/dsv3_decode_metrics.json'),
]:
    r = json.load(open(path))
    m1=r['m1']['m1_raw_op_count_hr']
    m2=r['m2']['m2_fused_op_hr']
    m3=r['m3']['m3_fused_op_hr_no_zc']
    m4=r['m4']['m4_per_shape_hr']
    m5=r['m5']['m5_simulated_latency_coverage']
    tc=r['m6_input']['tc_predicted_total_s']*1000
    print(f'{name:<22} {m1:>5.1%} {m2:>5.1%} {m3:>5.1%} {m4:>5.1%} {m5:>5.1%} {tc:>8.1f}')
"

python3.10 -c "
import json
print()
print('场景                  TC(ms)   Real(ms)   M6      判断')
print('-' * 65)
for name, path in [
    ('Qwen3 PF', 'results/qwen3_prefill_m6.json'),
    ('Qwen3 DC', 'results/qwen3_decode_m6.json'),
    ('DSv3 PF',  'results/dsv3_prefill_m6.json'),
    ('DSv3 DC',  'results/dsv3_decode_m6.json'),
]:
    r = json.load(open(path))
    tc = r['tc_predicted_us']/1e3
    real = r['real_per_fwd_us']/1e3
    m6 = tc / real if real > 0 else 0
    flag = 'OK' if 0.85 <= m6 <= 1.15 else ('HIGH' if m6 > 1.15 else 'LOW')
    n_fwd = r.get('n_forward_passes', '?')
    print(f'{name:<22} {tc:>8.1f} {real:>8.1f} {m6:>6.3f}   {flag}  (N={n_fwd})')
"
```

注意: 如果 profiling 数据路径不可访问（如 Profiling 数据目录不存在），M6 无法计算，跳过并注明。
如果 results/*_metrics.json 已存在且日期为当天且迭代参数未变更，可直接读取而不重新运行迭代。
**如果数据版本变更了 (DATA_DIR 或 PROF_BASE 改变)，必须重新运行完整迭代。**
**迭代优化的目标是 M4，但最终报告必须展示 M1-M6 全部指标 + 每个场景的迭代记录 (nq/ql/M4 per iter)。**

**Agent 3: 团队动态**
```
1. 读取日报: /Users/horacehxw/Documents/hxw-华为/蚂蚁仿真器项目/日报\日报汇总.txt
2. 读取所有站会纪要 PDF: /Users/horacehxw/Documents/hxw-华为/蚂蚁仿真器项目/日报/智能纪要*.pdf
   先列出所有纪要文件，然后逐个提取:
   ls "/Users/horacehxw/Documents/hxw-华为/蚂蚁仿真器项目/日报/"智能纪要*.pdf
   对每个 PDF 使用 text 模式提取 (低 context 开销):
   python3.10 ~/.claude/scripts/read_pdf.py "<pdf_path>" --mode text
   如需查看图表/流程图, 改用 image 模式:
   python3.10 ~/.claude/scripts/read_pdf.py "<pdf_path>" --mode image --pages 1
   然后用 Read 工具读取输出的 PNG 文件
   - 提取每人进展、阻塞、风险信号
   - 提取站会决策和 action items
   - 提取待办清单
3. 交叉验证: 日报说的 vs 站会纪要 vs 代码实际变更
```

### Step 2: 交叉分析

对采集到的数据进行交叉验证:
1. **实现 vs Design Doc**: 每个已完成任务是否符合 spec？偏差即为风险
2. **计划 vs 实际**: Work Plan 的检查点日期 vs 实际完成日期，识别延期趋势
3. **日报 vs 代码**: 日报说"完成"但代码未合入？日报未提到但代码有变更？
4. **指标趋势**: M1-M6 当前值 vs Phase 目标，差距分析。重点关注:
   - M3 vs >50% 目标 (Phase 2)
   - M5 vs >80% 目标 (Phase 3)
   - M6 vs 0.85-1.15 目标 (Phase 3): M6<1 说明覆盖不足（需更多 microbench 数据），M6>1 说明 microbench 偏高
   - M4 MISS shape list → 指导 microbench 数据采集优先级
5. **TC-Profiling 对齐验证** (M6 可信度):
   - 从 profiling kernel_details.csv 提取 MatMulV2 dominant M-dim (= 实际 batch tokens)
   - 与 TC 命令的 nq×ql (= TC 仿真 batch tokens) 对比
   - 如不一致, M6 不可信, 需标注原因和修正建议

### Step 2.5: 指标 Gap 逐项分析 (必须)

**对每个场景的每个 MISS 指标**, 逐一分析:
1. 列出该指标的 **当前值 vs 目标值** 和 **gap 大小**
2. 识别 **最大 gap** 的根因 (哪些算子/shape MISS? 是数据缺失、融合 gap、还是建模不支持?)
3. 检查 **是否有对应的人在解决** (对照 Work Plan 任务分配 + 日报)
4. 给出 **应该怎么解决** (补数据? 新 pass? 改查询逻辑? 上游修复?)

输出格式:
```
| 场景+指标 | 当前 | 目标 | Gap | 最大根因 | Owner | 解决路径 | 预期增益 |
```

### Step 3: 按模式生成输出

---

## 输出模板: 简化版 (默认, console)

直接在当前 console 回复，不生成文件。格式如下:

```markdown
# 项目状态速览 | {YYYY-MM-DD}

## 进度: Phase {N} {状态} | 距交付 {X} 天
{一句话当前状态}

## 时间线
{ASCII 时间线图，标注 Phase 起止、当前位置、里程碑}
示例:
Phase 1 [3.6━━━━━━━3.13] ✅ GO
Phase 2 [3.16━▶━━━━3.20]   ← 今天在这里
Phase 3      [3.19━━━━3.23]
交付                    3.23 🎯

## 关键指标 (M1-M6)
{完整 6 指标表 + 进度条}
示例:
| 场景 | M1 | M2 | M3 | M4 | M5 | M6 (ratio) |
|------|:--:|:--:|:--:|:--:|:--:|:----------:|
| Qwen3 PF | 77.8% | 47.4% | 23.1% | 47.4% | 52.4% | 0.531 |
| Qwen3 DC | 84.1% | 63.2% | 46.2% | 63.2% | 59.5% | 1.607 |
| DSv3 PF  | 70.6% | 50.0% | 15.4% | 41.2% | 71.9% | 0.531 |
| DSv3 DC  | 71.8% | 52.3% | 19.2% | 43.1% | 56.3% | 0.303 |

Phase 目标进度 (M2-M6 全列):
| 指标 | 目标 | Qwen3 PF | Qwen3 DC | DSv3 PF | DSv3 DC | 进度 |
|------|:---:|:--------:|:--------:|:-------:|:-------:|------|
| M2   | (GO/NO-GO) | 47.4% | 63.2% | 50.0% | 52.3% | ▓▓▓▓▓░░░░░ |
| M3   | >50% | 23.1% | 46.2% | 15.4% | 19.2% | ▓▓░░░░░░░░ |
| M4   | (诊断) | 47.4% | 63.2% | 41.2% | 43.1% | ▓▓▓▓░░░░░░ |
| M5   | >80% | 52.4% | 59.5% | 71.9% | 56.3% | ▓▓▓▓▓░░░░░ |
| M6   | 0.85-1.15 | 0.531 | 1.607 | 0.531 | 0.303 | ▓▓▓░░░░░░░ |

如有历史数据，附趋势:
M3 趋势 (Qwen3 PF):  Phase1 → Phase2 → 当前
                       31.2% → ?      → ?

指标说明 (简):
- M1: 原始 HIT 率 (debug 用, 被 zero_cost 膨胀)
- M2: 融合算子 HIT 率 (GO/NO-GO 门槛, 含 zero_cost)
- M3: 计算算子 HIT 率 (核心进度, 排除 zero_cost) — Phase 2 验收
- M4: per-shape HIT 率 (缺口诊断, miss list 指导 microbench 采集)
- M5: 仿真延迟覆盖 (延迟加权, 大算子优先) — Phase 3 验收
- M6: empirical E2E ratio (vs 真实 per-fwd, 目标 0.85–1.15) — Phase 3 验收

**摘要规则**: 摘要/速览中必须展示 **M2-M6** (5 个指标), M1 因 zero_cost 膨胀可省略。
完整看板的指标表必须展示 **M1-M6** 全部 6 个指标。

## 验证命令与数据源 (必须)
{列出本次报告用到的完整 TC 命令和数据源, 便于复现}
{必须包含迭代优化记录: 每个场景的 iter/nq/ql/M4}
示例:
```
DATA_DIR=tensor_cast/.../vllm0.18.0_torch2.9.0_cann8.5
PROF_QWEN3=/Users/.../Profiling-0325-final-vllm-new/profiler-qwen3-0325

# Qwen3 Prefill — 迭代优化结果
# iter 1: nq=2 ql=1032 → M4=67.2% (< 80%, analyzing MISS...)
#   MISS breakdown: shape_not_found=12 kernel_not_mapped=3
#   Top MISS M-dim: 2064 (8 occurrences)
# iter 2: nq=1 ql=2064 → M4=81.3% ✅ (accepted, >= 80%)
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 1 --query-length 2064 ...
  → TC M-dim: 2064, Profiling dominant M-dim: 2064 (✅ 匹配)

# M6
python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/qwen3_prefill_metrics.json \
  --profiler-output "$PROF_QWEN3/profiler-qwen3-input4096-output1-concurrency16-rank0"
```
每个场景必须注明: TC batch tokens, profiling M-dim, 是否匹配, 迭代次数。

## 指标 Gap 逐项分析 (必须)
{对每个未达标指标, 列出 gap、最大根因、owner、解决路径}
示例:
| 场景+指标 | 当前 | 目标 | Gap | 最大根因 | Owner | 解决路径 | 预期增益 |
|-----------|:----:|:----:|:---:|---------|:-----:|---------|:-------:|
| DSv3 PF M3 | 37.5% | >50% | -12.5pp | DFC fused_kernel_gap (5 ops) | LJW | DFC fusion pass | +15-30pp |
| DSv3 DC M3 | 29.2% | >50% | -20.8pp | DFC + shape_coverage_gap | LJW+TCX | DFC pass + microbench | +20-35pp |
| Qwen3 PF M4 | 67.2% | >80% | -12.8pp | MISS M-dim 2064 在 CSV 中缺失 | TCX | 补采 microbench M=2064 shape | +10-15pp |

## 风险矩阵
{用 ASCII 矩阵可视化 TOP 风险的 影响×概率 分布}
示例:
影响 ↑
 高  │ R5●        R11●
 中  │    R10●  R3  R7
 低  │              R13
     └──────────────────→ 概率
       低    中    高

## TOP 风险详情
1. 🔴 {风险描述} — {影响} — {缓解}
2. 🟡 ...
3. 🟡 ...

## 需要 SE 决策/关注
- {具体决策点，如: "通信建模方案 P/D 分开 vs 固定开销，HDY 待定论"}
- {需要推动的外部依赖}

## 团队负载一览
{紧凑表格: 每人当前主要任务 + 阻塞状态 + 依赖关系}
示例:
| 成员 | 主要任务 | 状态 | 阻塞 |
|------|---------|:----:|------|
| TCX | Microbench | 🟢 | 等 QCX NPU 接口 |
| ZH  | 查询接口  | 🟢 | — |
| HDY | Profiling 采集 | 🟡 | comm gap 1.66-20.6x |

## 本周进展亮点
- {3-5 条核心进展}

## Git 分支状态
{与上游的 ahead/behind 可视化}
示例:
feat/perf-database vs gitcode-ascend/develop:
  ours ████████████████████ +120 commits ahead
  upstream ██ +9 commits (需 rebase)

## 下一步建议 (按优先级)
1. {最重要的行动}
2. ...
3. ...
```

**原则**: 300 行以内。保留需要 SE 知道和决策的信息，用 ASCII 可视化增强直观性。不展开任务列表细节。

**可视化要素清单** (尽量包含):
- 时间线图 (Phase 进度 + 当前位置)
- 指标进度条 (▓░ 或 █ 风格)
- 指标趋势 (如有历史数据)
- 风险矩阵 (影响×概率 scatter)
- 团队负载表 (含阻塞状态)
- Git ahead/behind 条形图

---

## 输出模板: 完整版 (ALL, 保存文件)

生成完整看板 Markdown，保存到:
- **主路径** (git 仓库内): `docs/perf_database/daily_project_status/进展看板_{YYYYMMDD}.md`
- **备份路径**: `/Users/horacehxw/Documents/hxw-华为/蚂蚁仿真器项目/AI进展总结\进展看板_{YYYYMMDD}.md`

**结构** (按重要性排序，决策层信息在前，执行细节在后):

```markdown
# 算子性能数据库项目看板
**日期** | **分支** | **交付日** | **Design Doc 版本** | **数据版本**

## 一、执行摘要
{3-5 句话总结当前状态，核心矛盾，关键判断}

## 二、关键指标 (M1-M6)
{M1-M5: 百分比指标表 (4 场景) + M6: ratio 列 (1.0=完美, 目标 0.85-1.15)}
{M6 需要 ASCEND_PROFILER_OUTPUT 数据 + --export-metrics JSON，如不可用则注明}
{Phase 目标进度 + 收益路径估算}

## 二-A、验证命令与数据源 (必须)
{列出每个场景用到的:
 1. 完整 TC 命令 (含所有参数)
 2. DATA_DIR 路径
 3. PROFILING 路径 (M6 用)
 4. TC batch tokens (nq×ql) vs profiling dominant M-dim
 5. 是否匹配 → M6 是否可信
 6. 迭代优化记录: iter/nq/ql/M4/MISS分析}

## 二-B、指标 Gap 逐项分析 (必须)
{对每个未达标的 场景×指标 组合:
 1. 当前值 vs 目标值, gap 大小
 2. 最大 gap 根因 (哪些算子 MISS? 哪种 root cause?)
 3. 是否有对应 owner 在解决 (对照 Work Plan + 日报)
 4. 应该怎么解决 (补数据/新 pass/改查询/上游修复)
 5. 预期增益 (解决后 M3/M5 能提升多少 pp)}

## 三、TOP 风险 (按影响排序)
{风险表: #/风险/影响/状态/缓解}

## 四、时间线
{ASCII 时间线 + 里程碑表}

## 五、本周目标与任务分配
{P0 事项表 + 各人本周任务表 (来自日报+站会)}

## 六、下一步建议
{按优先级的 5-6 条建议}

## 七、Git 分支全景
{gitcode-ascend/develop 主仓新增 commit 分析 + rebase 建议}
{gitcode 功能分支状态表 (领先/落后/是否已合入)}
{关键未合入分支提醒}

## 八、Phase {N} 完成详情 (折叠)
{任务完成清单}
{MISS 根因分类}
{代码与测试现状}
{日报关键事件时间线}
```

**格式要求**: 面向全体项目成员分发。使用中文。表格对齐。ASCII 图表清晰。重要数字加粗。

---

## 输出模板: 个人版 (<人员代号>, console)

直接在 console 回复。包含:

```markdown
# {姓名} 专项看板 | {YYYY-MM-DD}

## 全局状态 (所有人需知)
{2-3 句话: 当前 Phase, 距交付天数, 核心矛盾}
{M1-M6 指标表 (精简, 4 场景)}
{未达标指标 Gap 摘要: 最大 gap + 对应 owner + 解决路径}

## 你的任务状态
| 任务 | 截止 | 状态 | 说明 |
{从 Work Plan + 日报提取该人的任务}

## 你的阻塞与风险
{从日报/站会提取该人的阻塞项和风险信号}

## 依赖你的下游任务
{谁在等你的产出? 哪些任务依赖你完成?}

## 你需要关注的协作点
{站会 action items 中分配给你的待办}
{需要与谁对齐什么}

## 建议下一步 (按优先级)
1. {最重要}
2. ...
3. ...
```

## 人员代号映射

| 代号 | 姓名 | 职责域 | Work Plan 任务 |
|------|------|--------|---------------|
| **HXW** | 贺骁武 | SE, spec review + 决策 + 进展管理 | 全局 |
| **TCX** | 唐楚笑 | 数据层: 工具链 + Microbench + Attention + 插值 | D1-D4, E1-E5, H5 |
| **ZH** | 祝豪 | 查询引擎: _lookup_compute/comm/composite | B1-B2, G1-G2 |
| **ZZY** | 张震宇 | Qwen3 op_mapping: BF16 验证 + Decode + 自动化 | C1-C5, H2, H4 |
| **HDY** | 胡定一 | DSV3 op_mapping + HCCL + Profiling 分析 | C6-C10, H1, H3 |
| **LJW** | 魏宇昊 | 融合 Pass: DispatchFFNCombine | F1-F2 |
| **XJT** | 许锦涛 | (已交接给 LJW) CLI + Compile Pass | A1-A3 ✅ |
| **DSH**/**CY** | 丁世浩(从云) | 客户 database 分支 | — |
| **QCX** | 钱晨希 | NPU 算子专家支持 | — |

## 关键文件路径

| 文件 | 用途 |
|------|------|
| `docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.4.md` | Design Doc (合规基准) |
| `docs/perf_database/WORK_PLAN_Q1.md` | Work Plan v3.2 (任务+时间线) |
| `docs/perf_database/CHANGELOG_*.md` | 变更日志 |
| `docs/perf_database/reports/phase1-e2e-*/phase1_e2e_*_verification_report_zh.md` | E2E 验证报告 |
| `/Users/horacehxw/Documents/hxw-华为/蚂蚁仿真器项目/日报\日报汇总.txt` | 飞书日报汇总 |
| `/Users/horacehxw/Documents/hxw-华为/蚂蚁仿真器项目/日报\智能纪要*.pdf` | 站会纪要 |
| `docs/perf_database/daily_project_status/` | **主输出目录** (git 仓库内，日期命名) |
| `/Users/horacehxw/Documents/hxw-华为/蚂蚁仿真器项目/AI进展总结\` | 完整看板备份目录 |
| `tensor_cast/performance_model/profiling_database/` | 核心实现代码 |
| `tools/perf_data_collection/` | 数据采集工具链 |

## 注意事项

- **Design Doc 是一切的标准**: 每个模块的实现状态都要对照 Design Doc spec 检查。合规分析作为内部方法论使用，不单独成章输出到报告。仅当发现 spec vs 实现偏差时，将偏差写入风险章节
- **日报是非结构化的**: 飞书聊天记录格式，多人交错，需要仔细解析上下文
- **站会纪要由 AI 生成**: 质量参差，需与日报交叉验证
- **gitcode-ascend/develop 只关注主分支**: 分析它比我们多了什么功能，是否需要 rebase
- **gitcode 是我们团队的 fork**: 所有 remote 分支都要 fetch 后分析
- **不要自动 commit**: 生成的文件不入 git
- **中文输出**: 面向中文团队
