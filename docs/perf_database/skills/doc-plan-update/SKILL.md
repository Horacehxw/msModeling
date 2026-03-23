---
name: doc-plan-update
description: Update perf-database design doc and work plan with version bumps, changelogs, and archiving
version: 1.0.0
source: local-session-analysis
analyzed_commits: 4
---

# Design Doc & Work Plan Update Skill

Update `OPERATOR_PERF_DATABASE_DESIGN_zh` and `WORK_PLAN_Q1` with new findings, decisions, and progress 鈥?then generate a CHANGELOG and archive old versions.

## When to Use

- End of week or sprint when multiple tasks have landed
- After major architectural decisions or personnel changes
- Before milestone reviews (e.g., mini E2E, Phase gate)

## Prerequisites

- Access to daily standup notes (鏃ユ姤) and PR history
- Access to design doc and work plan (current versions)
- A fresh branch from the dev main branch (e.g., `feat/perf-database`)

## Workflow

### Step 1: Gather Context (Parallel)

Launch 3 parallel explore agents:

1. **Git history agent**: Scan all feature branches for recent commits, PRs, and code changes relevant to the perf-database subsystem.
2. **Doc/report agent**: Read existing design doc, work plan, and any reports in `docs/perf_database/reports/`.
3. **Daily notes agent**: Read daily standup notes (鏃ユ姤) from the reporting period. Extract key findings, decisions, blockers, and status updates per person.

Key sources:
- Local feature branches (git log, git diff)
- `docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v*.md`
- `docs/perf_database/WORK_PLAN_Q1.md`
- Daily standup notes directory

### Step 2: Identify Deltas

From gathered context, categorize changes into:

| Category | Examples |
|----------|---------|
| **New additions** | New sections, new assumptions, new tasks |
| **Modifications** | Version targets, parameter renames, status updates |
| **Architecture decisions** | Interface changes, data model changes |
| **Personnel changes** | Handoffs, new assignments |
| **Risk updates** | New risks, closed risks, changed severity |
| **Task status** | Checkpoints completed, new checkpoints added |

### Step 3: Create Branch

```bash
# Create fresh branch from dev main branch
git checkout -b docs/update-v{NEW_VERSION} feat/perf-database
```

If `feat/perf-database` is checked out in another worktree, use:
```bash
git checkout -b docs/update-v{NEW_VERSION} $(git rev-parse feat/perf-database)
```

### Step 4: Archive Old Versions

```bash
mkdir -p docs/perf_database/archive
git mv docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v{OLD}.md docs/perf_database/archive/
cp docs/perf_database/WORK_PLAN_Q1.md docs/perf_database/archive/WORK_PLAN_Q1_v{OLD}.md
git add docs/perf_database/archive/
git commit -m "docs(perf-db): archive design doc v{OLD} and work plan v{OLD}"
```

### Step 5: Update Design Doc (Surgical Edits)

**Principle**: Preserve existing document structure. Make targeted edits to specific sections, not full rewrites.

Typical edit points:
- **Header**: Version number, date
- **搂1.4 (Target)**: Backend versions (vllm, torch, CANN)
- **搂2.x (Architecture)**: New fusion ops, pipeline changes
- **搂3.3 (Data layout)**: New version directories, naming conventions
- **搂4.x (Query dispatch)**: Parameter renames, new dispatch rules, input handling
- **搂4.5 (op_mapping)**: cann_version, new fields
- **搂4.7 (Data quality)**: Collection tool notes
- **搂9.x (Gap analysis)**: Updated gap items, closed items, new items

Copy old version to new filename, then apply edits:
```bash
cp docs/perf_database/archive/OPERATOR_PERF_DATABASE_DESIGN_zh_v{OLD}.md \
   docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v{NEW}.md
# Apply targeted edits using Edit tool
```

Commit: `docs(perf-db): design doc v{NEW} 鈥?{summary of key changes}`

### Step 6: Update Work Plan (In-Place Edits)

Edit `WORK_PLAN_Q1.md` directly:
- **Header**: Version number, target CANN version
- **搂1.x (Assumptions)**: New assumptions
- **搂3.x (Personnel)**: Handoffs, availability changes
- **Phase tasks**: Add 鉁?馃攧 status markers to completed/in-progress checkpoints
- **New task sections**: Add new task IDs with owners, deadlines, sub-checkpoints
- **搂6+ (Future phases)**: Scope changes (e.g., task reassignment)
- **搂8 (Risks)**: Close resolved risks, add new risks with probability

Commit: `docs(perf-db): work plan v{NEW} 鈥?{summary of key changes}`

### Step 7: Generate CHANGELOG

Create `docs/perf_database/CHANGELOG_{YYYYMMDD}.md` with:

```markdown
# CHANGELOG {YYYY-MM-DD}

璁捐鏂囨。 v{OLD} 鈫?v{NEW} + 宸ヤ綔璁″垝 v{OLD} 鈫?v{NEW}

---

## 璁捐鏂囨。鍙樻洿 (v{OLD} 鈫?v{NEW})

### 鏂板
- **搂X.Y**: {description}

### 鍙樻洿
- **搂X.Y**: {old} 鈫?{new}

### 鏋舵瀯鍐崇瓥
1. **{Decision}**: {rationale}

---

## 宸ヤ綔璁″垝鍙樻洿 (v{OLD} 鈫?v{NEW})

### 浜哄憳鍙樺姩
### 鐩爣璋冩暣
### Phase N 杩涘睍锛堟埅鑷?{date}锛?
### 鏂板浠诲姟
### 鏂板椋庨櫓
### 宸插叧闂闄?
### 鏍稿績鍋囪鏂板
```

Commit: `docs(perf-db): CHANGELOG {date} (v{NEW} delta summary)`

### Step 8: Review

- Verify all version references are consistent
- Check that task status markers match reality
- Ensure no privacy-sensitive names in git-tracked files (use abbreviations)

## Privacy Convention

In all git-tracked files, use abbreviated names (e.g., XJT, LJW, ZH, ZZY, HDY, TCX) instead of full names.

## Output Files

| File | Action |
|------|--------|
| `docs/perf_database/archive/OPERATOR_PERF_DATABASE_DESIGN_zh_v{OLD}.md` | Moved |
| `docs/perf_database/archive/WORK_PLAN_Q1_v{OLD}.md` | Copied |
| `docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v{NEW}.md` | Created |
| `docs/perf_database/WORK_PLAN_Q1.md` | Edited in-place |
| `docs/perf_database/CHANGELOG_{YYYYMMDD}.md` | Created |

