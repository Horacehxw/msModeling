---
name: weekly-report
description: Generate Chinese weekly report (周报) from work plan, design doc, and daily standup notes
version: 1.0.0
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
- Weekly report template: `D:\File\hxw-华为\蚂蚁仿真器项目\周报\周报模板.txt`

## Workflow

### Step 1: Read Template

Read the weekly report template to understand the required format and sections.

### Step 2: Gather Sources (Parallel)

Launch parallel reads of:

1. **Work plan**: `docs/perf_database/WORK_PLAN_Q1.md` — extract task status, phase progress, risks
2. **Design doc**: Latest `OPERATOR_PERF_DATABASE_DESIGN_zh_v*.md` — extract architecture decisions, gap analysis
3. **Daily standup notes**: Read all 日报 files for the reporting week (typically Mon-Fri)
4. **CHANGELOG** (if generated this week): `docs/perf_database/CHANGELOG_*.md`

### Step 3: Compute Progress Metrics

For each Q1 delivery goal (交付目标):

1. Count completed checkpoints vs total checkpoints from work plan
2. Calculate percentage (e.g., 12/18 = ~55%)
3. Identify key completed items for the "已完成" list
4. Identify remaining gaps

### Step 4: Generate Report

Follow the template structure exactly. The standard format is:

```markdown
周报：{YYYY.M.DD}
---
【主要进展】
Q1 交付目标 1：{goal description}
- 已完成：
  1. {category}: {details with person abbreviation and task IDs}
  2. ...
- 整体进展：约 {N}%
  - {summary of what's done and what remains}

Q1 交付目标 2：{goal description}
- 已完成：
  1. ...
- 整体进展：约 {N}%
  - ...

---
【下一步工作】
1. {next step with owner, deadline, and expected outcome}
2. ...

---
【风险求助】
1. {risk ID}（{risk name}，{severity}）：{description with impact}
2. ...
```

### Step 5: Content Guidelines

**【主要进展】section**:
- Group by Q1 delivery goal (交付目标), not by person
- Within each goal, organize completed items by functional category (e.g., CLI集成, 算子映射, 通信查询, 数据采集, bug修复)
- Include task IDs (A1, B2, C3, etc.) for traceability
- Include quantitative metrics where available (coverage %, performance ratios)
- Note key technical findings/decisions inline
- End each goal with overall progress % and gap summary

**【下一步工作】section**:
- Ordered by priority/timeline
- Each item: what + who + when + expected outcome
- Focus on next 1-2 weeks

**【风险求助】section**:
- Include risk ID from work plan (R10, R11, etc.)
- Classify severity (高风险/中风险/低风险)
- Describe impact on timeline and dependencies
- Note any mitigation actions already taken

### Step 6: Save Report

Save to: `D:\File\hxw-华为\蚂蚁仿真器项目\周报\周报_{YYYYMMDD}.md`

(This is a local-only file, NOT committed to git)

## Privacy Convention

- In git-tracked files: use abbreviations (XJT, LJW, ZH, ZZY, HDY, TCX)
- In local-only reports (周报): abbreviations are also preferred for consistency

## Quality Checklist

- [ ] All Q1 delivery goals covered with progress %
- [ ] Task IDs referenced match work plan
- [ ] Next steps have owners and deadlines
- [ ] Risks include severity and impact description
- [ ] Numbers/metrics are sourced from actual data (not estimated)
- [ ] Report follows template format exactly
- [ ] Chinese language throughout (except technical terms)
