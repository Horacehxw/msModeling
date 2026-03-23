---
name: weekly-report
description: Generate Chinese weekly report (鍛ㄦ姤) from work plan, design doc, and daily standup notes
version: 1.1.0
source: local-session-analysis
analyzed_commits: 4
---

# Weekly Report Generation Skill (鍛ㄦ姤)

Generate a Chinese weekly report summarizing team progress, risks, and next steps for the perf-database project.

## When to Use

- Every Thursday/Friday to prepare the weekly report
- Before team meetings requiring progress summaries

## Prerequisites

- Updated work plan (`docs/perf_database/WORK_PLAN_Q1.md`)
- Updated design doc (`docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v*.md`)
- Daily standup notes (鏃ユ姤) for the reporting week
- Previous weekly report (if exists): `D:\File\hxw-鍗庝负\铓傝殎浠跨湡鍣ㄩ」鐩甛鍛ㄦ姤\鍛ㄦ姤_{YYYYMMDD}.md`
- Weekly report template: `D:\File\hxw-鍗庝负\铓傝殎浠跨湡鍣ㄩ」鐩甛鍛ㄦ姤\鍛ㄦ姤妯℃澘.txt`

## Audience & Tone

**Target audience**: Leadership / PM who are NOT involved in daily project execution. They need to understand:
1. **What** we are building and **why** (1-2 sentence background per goal)
2. **What** was accomplished this week (core contributions, not every PR)
3. **What** comes next (prioritized, with timeline)
4. **What** might go wrong and what help is needed (risks/blockers)

**Writing principles**:
- Lead each delivery goal with a brief **> 鑳屾櫙** block explaining the goal in plain language
- Summarize by **functional milestone**, not by individual PRs or task IDs
- Use task IDs (A1, C3, etc.) sparingly 鈥?only when referencing work plan items for traceability, not as primary structure
- Prefer concrete numbers (coverage %, error ratios, checkpoint counts) over vague descriptions
- Keep each bullet to 1-2 sentences; avoid implementation details (code paths, class names)
- A reader should understand each bullet WITHOUT reading the design doc or work plan

## Workflow

### Step 1: Read Previous Report

Read the most recent weekly report to understand baseline and avoid repeating unchanged items.

### Step 2: Gather Sources (Parallel)

Launch parallel reads of:

1. **Work plan**: `docs/perf_database/WORK_PLAN_Q1.md` 鈥?extract task status, phase progress, risks
2. **Design doc**: Latest `OPERATOR_PERF_DATABASE_DESIGN_zh_v*.md` 鈥?extract architecture decisions, gap analysis
3. **Daily standup notes**: Read all 鏃ユ姤 files for the reporting week (typically Mon-Fri)
4. **CHANGELOG** (if generated this week): `docs/perf_database/CHANGELOG_*.md`

### Step 2b: Gather M1-M6 Metrics (if profiling data available)

Run TC profiling + compute_m6 for the standard 4 scenarios to get current M1-M6 values.
See `docs/perf_database/METRICS_GUIDE.md` for commands and interpretation.

Key metrics to include in the report:
- **M3** (Fused Op HR excl zero_cost): core progress indicator, Phase 2 target >50%
- **M5** (Simulated Latency Coverage): analytic-weighted coverage, Phase 3 target >80%
- **M6** (Empirical E2E Ratio): empirical_hit / real_per_fwd, Phase 3 target 0.85-1.15
  - M6 = 1.0 perfect; >1 overestimate; <1 underestimate (coverage gap)

If metrics changed significantly from last week, highlight the delta and root cause.

### Step 3: Identify Core Contributions

For each Q1 delivery goal:

1. Count completed checkpoints vs total from work plan 鈫?progress %
2. Identify **3-6 core contributions** that represent meaningful milestones (NOT every small PR/task)
3. Classify each contribution into a functional category (e.g., 绔埌绔祦绋? 绠楀瓙鏄犲皠瑕嗙洊, 鏁版嵁閲囬泦, 鍏抽敭闂瑙ｅ喅, 鎬ц兘鐡堕瀹氫綅)
4. For risk items that changed status (new/mitigated/escalated), highlight as a contribution if significant

**What counts as a "core contribution"**:
- A capability that didn't exist before (e.g., "绔埌绔祦绋嬫墦閫?)
- A significant coverage/accuracy improvement with numbers (e.g., "瑕嗙洊鐜囦粠 40% 鈫?98%")
- A high-risk item resolved or mitigated
- A critical bug fix with measurable impact (e.g., "淇 8x 楂樹及闂")

**What to omit**:
- Routine document updates, code review, minor refactors
- Task-level details that only matter within the team (e.g., "B2 鏂囨。鍒濈瀹屾垚")
- Work-in-progress items without a clear outcome

### Step 4: Generate Report

Follow the template structure. The standard format is:

```markdown
鍛ㄦ姤锛歿YYYY.M.DD}
---
銆愪富瑕佽繘灞曘€?
Q1 浜や粯鐩爣 1锛歿goal description}锛坽quantitative target}锛?
> 鑳屾櫙锛歿1-2 sentences explaining what this goal is about in plain language}

- 宸插畬鎴愶細
  1. {functional milestone}: {what changed + quantitative result}
  2. ...锛?-6 items max锛?
- 鏁翠綋杩涘睍锛氱害 {N}%锛圥hase X {M}/{T} 妫€鏌ョ偣瀹屾垚锛?
  - {1-sentence summary of status + main gaps}

Q1 浜や粯鐩爣 2锛歿goal description}
> 鑳屾櫙锛?..
- 宸插畬鎴愶細
  1. ...
- 鏁翠綋杩涘睍锛氱害 {N}%
  - ...

---
銆愪笅涓€姝ュ伐浣溿€?
1. {milestone}锛坽deadline}锛夛細{what + expected outcome}
2. ...

---
銆愰闄╂眰鍔┿€?
1. {risk name}锛坽risk ID}锛寋severity change if any}锛夛細{impact + mitigation status}
2. ...
```

### Step 5: Content Guidelines

**銆愪富瑕佽繘灞曘€憇ection**:
- Start each delivery goal with a `> 鑳屾櫙` block for non-project readers
- Group by functional milestone, not by person or task ID
- 3-6 core contributions per goal 鈥?quality over quantity
- Include M1-M6 metrics where relevant (see METRICS_GUIDE.md for definitions)
- Highlight risk status changes inline (e.g., "鍘熼珮椋庨櫓椤癸紝宸茬紦瑙?)
- End with progress % and 1-sentence gap summary

**銆愪笅涓€姝ュ伐浣溿€憇ection**:
- 3-5 items max, ordered by priority/timeline
- Each item: milestone name + deadline + expected outcome
- Person names optional (omit if not relevant to leadership)
- Focus on next 1-2 weeks

**銆愰闄╂眰鍔┿€憇ection**:
- Include risk ID from work plan (R10, R11, etc.)
- Show severity changes (e.g., 楂樷啋宸茬紦瑙? to highlight progress
- For mitigated risks: use ~~strikethrough~~ and briefly state the resolution
- For active risks: describe impact on timeline + what help is needed
- Order by severity (high 鈫?medium 鈫?low)

### Step 6: Save Report

Save to: `D:\File\hxw-鍗庝负\铓傝殎浠跨湡鍣ㄩ」鐩甛鍛ㄦ姤\鍛ㄦ姤_{YYYYMMDD}.md`

(This is a local-only file, NOT committed to git)

## Privacy Convention

- In git-tracked files: use abbreviations (XJT, LJW, ZH, ZZY, HDY, TCX)
- In local-only reports (鍛ㄦ姤): abbreviations are also preferred for consistency

## Quality Checklist

- [ ] Each delivery goal has a `> 鑳屾櫙` block readable by non-project people
- [ ] Contributions are functional milestones (not task-level items)
- [ ] 3-6 items per goal, no more
- [ ] All Q1 delivery goals covered with progress %
- [ ] Next steps have deadlines and expected outcomes
- [ ] Risks show severity and status changes
- [ ] Numbers/metrics are sourced from actual data (not estimated)
- [ ] A leadership reader can understand every bullet without the work plan
- [ ] Chinese language throughout (except technical terms)

