---
name: weekly-report
description: Generate Chinese weekly report (周报) from work plan, design doc, and daily standup notes
version: 1.1.0
source: local-session-analysis
analyzed_commits: 4
---

# Weekly Report Generation Skill (周报)

Generate a Chinese weekly report summarizing team progress, risks, and next steps for the perf-database project.

## When to Use

- Every Thursday/Friday to prepare the weekly report
- Before team meetings requiring progress summaries

## Prerequisites

- Updated work plan (`docs/perf_database/WORK_PLAN_Q1.md`)
- Updated design doc (`docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v*.md`)
- Daily standup notes (日报) for the reporting week
- Previous weekly report (if exists): `D:\File\hxw-华为\蚂蚁仿真器项目\周报\周报_{YYYYMMDD}.md`
- Weekly report template: `D:\File\hxw-华为\蚂蚁仿真器项目\周报\周报模板.txt`

## Audience & Tone

**Target audience**: Leadership / PM who are NOT involved in daily project execution. They need to understand:
1. **What** we are building and **why** (1-2 sentence background per goal)
2. **What** was accomplished this week (core contributions, not every PR)
3. **What** comes next (prioritized, with timeline)
4. **What** might go wrong and what help is needed (risks/blockers)

**Writing principles**:
- Lead each delivery goal with a brief **> 背景** block explaining the goal in plain language
- Summarize by **functional milestone**, not by individual PRs or task IDs
- Use task IDs (A1, C3, etc.) sparingly — only when referencing work plan items for traceability, not as primary structure
- Prefer concrete numbers (coverage %, error ratios, checkpoint counts) over vague descriptions
- Keep each bullet to 1-2 sentences; avoid implementation details (code paths, class names)
- A reader should understand each bullet WITHOUT reading the design doc or work plan

## Workflow

### Step 1: Read Previous Report

Read the most recent weekly report to understand baseline and avoid repeating unchanged items.

### Step 2: Gather Sources (Parallel)

Launch parallel reads of:

1. **Work plan**: `docs/perf_database/WORK_PLAN_Q1.md` — extract task status, phase progress, risks
2. **Design doc**: Latest `OPERATOR_PERF_DATABASE_DESIGN_zh_v*.md` — extract architecture decisions, gap analysis
3. **Daily standup notes**: Read all 日报 files for the reporting week (typically Mon-Fri)
4. **CHANGELOG** (if generated this week): `docs/perf_database/CHANGELOG_*.md`

### Step 3: Identify Core Contributions

For each Q1 delivery goal:

1. Count completed checkpoints vs total from work plan → progress %
2. Identify **3-6 core contributions** that represent meaningful milestones (NOT every small PR/task)
3. Classify each contribution into a functional category (e.g., 端到端流程, 算子映射覆盖, 数据采集, 关键问题解决, 性能瓶颈定位)
4. For risk items that changed status (new/mitigated/escalated), highlight as a contribution if significant

**What counts as a "core contribution"**:
- A capability that didn't exist before (e.g., "端到端流程打通")
- A significant coverage/accuracy improvement with numbers (e.g., "覆盖率从 40% → 98%")
- A high-risk item resolved or mitigated
- A critical bug fix with measurable impact (e.g., "修复 8x 高估问题")

**What to omit**:
- Routine document updates, code review, minor refactors
- Task-level details that only matter within the team (e.g., "B2 文档初稿完成")
- Work-in-progress items without a clear outcome

### Step 4: Generate Report

Follow the template structure. The standard format is:

```markdown
周报：{YYYY.M.DD}
---
【主要进展】
Q1 交付目标 1：{goal description}（{quantitative target}）
> 背景：{1-2 sentences explaining what this goal is about in plain language}

- 已完成：
  1. {functional milestone}: {what changed + quantitative result}
  2. ...（3-6 items max）
- 整体进展：约 {N}%（Phase X {M}/{T} 检查点完成）
  - {1-sentence summary of status + main gaps}

Q1 交付目标 2：{goal description}
> 背景：...
- 已完成：
  1. ...
- 整体进展：约 {N}%
  - ...

---
【下一步工作】
1. {milestone}（{deadline}）：{what + expected outcome}
2. ...

---
【风险求助】
1. {risk name}（{risk ID}，{severity change if any}）：{impact + mitigation status}
2. ...
```

### Step 5: Content Guidelines

**【主要进展】section**:
- Start each delivery goal with a `> 背景` block for non-project readers
- Group by functional milestone, not by person or task ID
- 3-6 core contributions per goal — quality over quantity
- Include quantitative metrics (coverage %, error ratios, checkpoint counts)
- Highlight risk status changes inline (e.g., "原高风险项，已缓解")
- End with progress % and 1-sentence gap summary

**【下一步工作】section**:
- 3-5 items max, ordered by priority/timeline
- Each item: milestone name + deadline + expected outcome
- Person names optional (omit if not relevant to leadership)
- Focus on next 1-2 weeks

**【风险求助】section**:
- Include risk ID from work plan (R10, R11, etc.)
- Show severity changes (e.g., 高→已缓解) to highlight progress
- For mitigated risks: use ~~strikethrough~~ and briefly state the resolution
- For active risks: describe impact on timeline + what help is needed
- Order by severity (high → medium → low)

### Step 6: Save Report

Save to: `D:\File\hxw-华为\蚂蚁仿真器项目\周报\周报_{YYYYMMDD}.md`

(This is a local-only file, NOT committed to git)

## Privacy Convention

- In git-tracked files: use abbreviations (XJT, LJW, ZH, ZZY, HDY, TCX)
- In local-only reports (周报): abbreviations are also preferred for consistency

## Quality Checklist

- [ ] Each delivery goal has a `> 背景` block readable by non-project people
- [ ] Contributions are functional milestones (not task-level items)
- [ ] 3-6 items per goal, no more
- [ ] All Q1 delivery goals covered with progress %
- [ ] Next steps have deadlines and expected outcomes
- [ ] Risks show severity and status changes
- [ ] Numbers/metrics are sourced from actual data (not estimated)
- [ ] A leadership reader can understand every bullet without the work plan
- [ ] Chinese language throughout (except technical terms)
