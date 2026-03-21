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

**Agent 4: M1-M6 指标计算** (如果 profiling 数据可用)
```
运行 4 个场景的 TC profiling 命令，采集 M1-M6 全量指标:

DATA_DIR="$(pwd)/tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5"

# M1-M5 (在线): 4 个场景各跑一次 TC profiling + --export-metrics

# Qwen3 Prefill: --enable-flashcomm-v1 对标 vLLM ENABLE_FLASHCOMM1=1
# FlashComm 将 all_reduce→rms_norm 替换为 reduce_scatter→rms_norm→all_gather
# 效果: M3 33%→50%, M5 62%→88%, 但 reduce_scatter CSV 有膨胀风险 (35ms/call vs 真实 ~2ms)
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 10 --query-length 4104 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile --perf-database "$DATA_DIR" \
  --enable-flashcomm-v1 \
  --export-metrics results/qwen3_prefill_metrics.json --log-level info

# Qwen3 Decode: 暂不加 --enable-flashcomm-v1
# 原因: vLLM cudagraph FULL_DECODE_ONLY 模式下 decode 可能不走 flashcomm 路径
# decode profiling 中是否有 hcom_reduceScatter_ 待确认, 如有则加此 flag
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 16 --query-length 1 --context-length 4096 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile --perf-database "$DATA_DIR" \
  --export-metrics results/qwen3_decode_metrics.json --log-level info

# DSv3 Prefill: 对标 profiler-dsv3-input2048-output1 (QBM batch=2048)
# 参数推导: profiling kernel_details 中 QuantBatchMatmulV3 batch dim=2048
# vLLM max-num-batched-tokens=2048, 单请求 ISL=2048 → nq=1 ql=2048
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 1 --query-length 2048 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile --perf-database "$DATA_DIR" \
  --export-metrics results/dsv3_prefill_metrics.json --log-level info

# DSv3 Decode: 对标 profiler-dsv3-input4096-output1536-concurrency8 (QBM batch=5)
# 参数推导: profiling kernel_details 中 QuantBatchMatmulV3 batch dim=5
# vLLM max-num-seqs=8, DP=2 → per-rank ~5 queries → nq=10 (dp_size=2, 10/2=5)
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 10 --query-length 1 --context-length 4096 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile --perf-database "$DATA_DIR" \
  --export-metrics results/dsv3_decode_metrics.json --log-level info

# M6 (半离线): TC Prediction Ratio = tc_full_prediction / real_per_fwd
# M6=1.0 完美, >1 高估, <1 低估. Phase 3 目标: 0.85-1.15
#
# M6 计算方法论 (2026-03-20 修正):
#   分子: TC 全量预测 (empirical + analytic fallback), 从 --export-metrics JSON 的
#         m6_input.tc_predicted_total_s 获取
#   分母: 真实单个 forward pass 耗时, 通过以下步骤获取:
#     1. 从 kernel_details.csv 中识别 forward pass 边界 (寻找每 forward pass 出现恰好一次的
#        anchor kernel, 如 ArgMaxV2/ApplyTopKTopPCustom 等 sampling kernel, 或
#        DispatchFFNCombine/FusedInferAttentionScore 等模型特有 kernel)
#     2. AI 分析每个 forward pass 的结构 (batch dim, kernel 组成)
#     3. 确认所有 forward pass 结构一致后, 用 Stage / n_forward_passes 作为分母
#     4. 如不一致, 需找到与 TC 仿真参数匹配的那一个 forward pass
#
# TC 参数推导方法论:
#   1. 从 profiling kernel_details.csv 的 QuantBatchMatmulV3 (主计算 kernel) 读 batch dim
#   2. 根据 batch dim 和 DP/TP 配置反推 TC 的 nq/ql 参数
#   3. 用 anchor kernel 切分 forward passes, 验证每个 pass 结构是否一致
#
# Qwen3 profiling 数据 (0314):
PROF_QWEN3="/Users/horacehxw/Data/Profiling/Profiling-0317-full/profiler-qwen3-0314"
# DSv3 profiling 数据 (0319, 最新):
PROF_DSV3="/Users/horacehxw/Data/Profiling/Profiling-0320-DSv3/profiler-dsv3-0319"
# Phase 1 E2E 基线 (0313, Qwen3 M6 用):
PROF_BASE="/Users/horacehxw/Data/Profiling/Profiling-0313-phase1-e2e-test"

python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/qwen3_prefill_metrics.json \
  --profiler-output "$PROF_BASE/profilier_prefill_qwen32b-input4096-output1/$(ls $PROF_BASE/profilier_prefill_qwen32b-input4096-output1/)/ASCEND_PROFILER_OUTPUT"

python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/qwen3_decode_metrics.json \
  --profiler-output "$PROF_BASE/profilier_decode_qwen32b-input4k/$(ls $PROF_BASE/profilier_decode_qwen32b-input4k/)/ASCEND_PROFILER_OUTPUT"

# DSv3 M6: 使用新 0319 profiling 数据 (通信模式已从 reduceScatter 变为 allReduce)
python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/dsv3_prefill_metrics.json \
  --profiler-output "$PROF_DSV3/profiler-dsv3-input2048-output1"

python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/dsv3_decode_metrics.json \
  --profiler-output "$PROF_DSV3/profiler-dsv3-input4096-output1536-concurrency8-rrate4"

# 汇总 M1-M5
python3.10 -c "
import json
print('场景                  M1     M2     M3     M4     M5     TC(ms)')
print('-' * 70)
for name, path in [
    ('Qwen3 PF (+FC)', 'results/qwen3_prefill_metrics.json'),
    ('Qwen3 DC',       'results/qwen3_decode_metrics.json'),
    ('DSv3 PF',        'results/dsv3_prefill_metrics.json'),
    ('DSv3 DC',        'results/dsv3_decode_metrics.json'),
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

# ==========================================
# M6 计算: AI 分析 profiling forward pass
# ==========================================
# compute_m6.py 会自动寻找 anchor kernel (默认 ArgMaxV2) 做 forward pass 切分并取平均。
# 但 **必须** 由 AI (即你) 在使用前先验证:
#
# 步骤 1: 分析 profiling 中每个 forward pass 的结构
#   对每个 profiling 场景, 用 kernel_details.csv 做以下分析:
#   - 识别 anchor kernel 做 forward pass 边界切分 (每 fwd 恰好出现一次的 kernel type)
#   - 检查每个 forward pass 的 kernel 组成 (DFC/QBM/FIA/RING_MLA 数量)
#   - 提取每个 forward pass 的 QuantBatchMatmulV3 batch dim (= 实际 batch size)
#   - 确认: 所有 forward pass 结构是否一致 (batch dim, kernel 数量, 算子类型)
#
# 步骤 2: 匹配 TC 仿真的 forward pass
#   - 如果所有 forward pass 结构一致 → 可以用 Stage/N 作为 M6 分母
#   - 如果不一致 (混合 prefill+decode, 不同 batch size):
#     → 找到 batch dim 匹配 TC 参数的那一个 forward pass
#     → 用该 forward pass 的 kernel duration sum 作为 M6 分母
#     → 注意: kernel sum 含 compute-comm overlap, 应优先用 step_trace Stage/N
#
# 步骤 3: 计算 M6
#   M6 = TC_prediction / real_per_fwd
#   - 分子: m6_input.tc_predicted_total_s (混合预测: empirical for HIT + analytic for MISS)
#   - 分母: AI 确认后的单个 forward pass 真实耗时
#
# 示例 (已验证的 forward pass 结构):
#   DSv3 PF (input2048): 12 个一致的纯 prefill pass, QBM batch=2048, Stage/12=295ms
#   DSv3 DC (c8): 62 个一致的纯 decode pass, QBM batch=5, Stage/62=51ms
#   Qwen3 PF (input4096): 5 个纯 prefill pass, Stage/5=1147ms

# 自动计算 (仅在 AI 确认 forward pass 结构后使用):
python3.10 -c "
import json
print()
print('场景                  TC(ms)   Real(ms)   M6      判断')
print('-' * 65)
for name, path in [
    ('Qwen3 PF (+FC)', 'results/qwen3_prefill_m6.json'),
    ('Qwen3 DC',       'results/qwen3_decode_m6.json'),
    ('DSv3 PF',        'results/dsv3_prefill_m6.json'),
    ('DSv3 DC',        'results/dsv3_decode_m6.json'),
]:
    r = json.load(open(path))
    tc = r['tc_predicted_us']/1e3
    real = r['real_per_fwd_us']/1e3
    m6 = tc / real if real > 0 else 0
    flag = 'OK' if 0.85 <= m6 <= 1.15 else ('HIGH' if m6 > 1.15 else 'LOW')
    n_fwd = r.get('n_forward_passes', '?')
    print(f'{name:<22} {tc:>8.1f} {real:>8.1f} {m6:>6.3f}   {flag}  (N={n_fwd})')
print()
print('注意: M6 因 microbench CSV 膨胀 (R10) 暂不可信。Phase 3 验收以 M3+M5 为主指标。')
print('      如果 forward pass 结构未经 AI 验证, M6 分母可能不准确。')
"
```

注意: 如果 profiling 数据路径不可访问（如 Profiling 数据目录不存在），M6 无法计算，跳过并注明。
如果 results/*.json 已存在且日期为当天，可直接读取而不重新运行 TC。

**Agent 3: 团队动态**
```
1. 读取日报: /Users/horacehxw/Documents/hxw-华为/蚂蚁仿真器项目/日报\日报汇总.txt
2. 读取最近 3 天站会纪要 PDF: /Users/horacehxw/Documents/hxw-华为/蚂蚁仿真器项目/日报\智能纪要*.pdf
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
| `tensor_cast/performance_model/perf_database/` | 核心实现代码 |
| `tools/perf_data_collection/` | 数据采集工具链 |

## 注意事项

- **Design Doc 是一切的标准**: 每个模块的实现状态都要对照 Design Doc spec 检查。合规分析作为内部方法论使用，不单独成章输出到报告。仅当发现 spec vs 实现偏差时，将偏差写入风险章节
- **日报是非结构化的**: 飞书聊天记录格式，多人交错，需要仔细解析上下文
- **站会纪要由 AI 生成**: 质量参差，需与日报交叉验证
- **gitcode-ascend/develop 只关注主分支**: 分析它比我们多了什么功能，是否需要 rebase
- **gitcode 是我们团队的 fork**: 所有 remote 分支都要 fetch 后分析
- **不要自动 commit**: 生成的文件不入 git
- **中文输出**: 面向中文团队
