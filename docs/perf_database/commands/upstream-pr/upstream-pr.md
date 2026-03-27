---
name: upstream-pr
description: Build and update upstream PR branches (PR-A/PR-B) for submitting perf-database to gitcode-ascend/develop. Handles sync, squash, cleanup, verification, force-push updates, and co-author management.
---

# Upstream PR Builder

Build, update, or rebuild the upstream PR branches for submitting perf-database features to `gitcode-ascend/develop`.

## Usage

| Command | Action |
|---------|--------|
| `/upstream-pr build` | Full pipeline: sync develop → build PR-A/B branches → verify → push |
| `/upstream-pr update` | Sync latest feat/perf-database, rebuild PR-A/B, force-push |
| `/upstream-pr status` | Show current PR-A/B status, diff stats, pending team MRs |
| `/upstream-pr verify` | Run unit tests + functional tests + linter on PR-A/B branches |
| `/upstream-pr sync` | Merge develop into feat/perf-database (resolve conflicts, push sync branch) |

---

## Principles

### Develop → feat/perf-database Merge Principles

When syncing upstream develop into our branch, **基于对整个项目的理解和工程实践来判断**，而不是机械地选边：

1. **理解变更的全链路影响** — 每个 develop 改动，追踪它在我们代码里的所有下游引用（op 定义 → 调用方 → 编译 pass → op_mapping → 测试）。重命名一个 op 意味着整条链路都要更新
2. **不引入冗余** — 如果 develop 重命名了一个 op（如 permute_tokens → init_routing_v2），不要同时保留旧名 op 定义（除非有明确的向后兼容需求）。op_mapping 可以保留旧名映射（数据层兼容）
3. **一个功能只有一个模块** — 不允许 re-export wrapper 或双模块共存
4. **feat/perf-database 只做 merge 不做 rebase** — 团队成员的分支永远不 break
5. **Unit test 不能 fail, M1→M6 不能严重回退**
6. **拿不准就停下来问用户** — 而不是先做一个"看起来合理"的选择然后事后修复

### PR 构建原则

PR-A/B 的拆分不是机械的文件分组。构建时 **launch sub-agent 分析依赖**：
1. 分析每个文件的 import 依赖链，确认 PR-A 和 PR-B 各自 import-complete
2. 检查 tests 的 fixture/import 依赖，确保每个 PR 的测试独立可运行
3. 确认 PR-A 的 `--compile --profiling-database` 端到端路径完整（从 CLI → compile passes → model_runner → ProfilingDataSource → CSV）
4. 确认 PR-B 的 tools/ 不 import tensor_cast（已验证，但每次 build 需重新检查）

### Commit Message 原则

- No Chinese comments or person name abbreviations in commit body
- Co-authored-by lines use git email format only
- Squash merge: 1 PR = 1 commit

---

## Architecture

### PR Split

| | PR-A: Core Functionality | PR-B: Offline Toolchain |
|---|---|---|
| **Content** | profiling_database/ impl + empirical.py M1-M6 + DFC pass + FlashComm pass + dispatch_ffn_combine op + op_mapping×3 + CSV data (LFS) + model_runner/CLI + runtime tests | tools/perf_data_collection/ + tests/tools/ + run_comm_bench.sh |
| **Dependency** | Independent | Independent (tools do NOT import tensor_cast) |
| **CI** | Unit + functional + lint | Unit + lint |

### Merge Strategy

- **Squash merge**: Each PR = 1 commit. Original history preserved on `feat/perf-database`.
- **Parallel review**: Both PRs submitted simultaneously. PR-B as Draft.
- **Serial merge**: PR-A first (logical: functionality before tools), then PR-B.
- **Bug fix during review**: Team continues merging to feat/perf-database. Periodically rebuild + force push PR-A/B.

---

## Build Process (`/upstream-pr build`)

### Step 0: Sync develop → feat/perf-database (if needed)

Check if develop has commits not yet in feat/perf-database. If so, run `/upstream-pr sync` first.

```bash
git fetch gitcode && git fetch gitcode-ascend
BEHIND=$(git log --oneline gitcode/feat/perf-database..gitcode-ascend/develop | wc -l | tr -d ' ')
if [ "$BEHIND" -gt 0 ]; then
  echo "⚠️ feat/perf-database is $BEHIND commits behind develop. Run /upstream-pr sync first."
  exit 1
fi
```

### Step 1: Create worktrees

```bash
# Clean up old worktrees if they exist
git worktree remove .claude/worktrees/pr-upstream-a 2>/dev/null
git worktree remove .claude/worktrees/pr-upstream-b 2>/dev/null
git branch -D pr/perf-db-a pr/perf-db-b 2>/dev/null

# Create fresh
git worktree add .claude/worktrees/pr-upstream-a --track -b pr/perf-db-a gitcode/feat/perf-database
git worktree add .claude/worktrees/pr-upstream-b --track -b pr/perf-db-b gitcode/feat/perf-database
```

### Step 2: Dependency analysis (sub-agent)

**Launch a sub-agent** to analyze the dependency graph:

```
Agent task: In each worktree, analyze:
1. PR-A files: trace import chain from model_runner.py → ProfilingDataSource → data_source.py.
   Confirm all imports resolve within PR-A's file set.
2. PR-A compile path: confirm dispatch_ffn_combine_pass.py and flashcomm_v1_pass.py
   are registered in compile_backend.py and the corresponding ops exist in fused_moe.py.
3. PR-B files: confirm tools/ has ZERO imports from tensor_cast/.
4. Tests: confirm each PR's tests import only from files within that PR.
5. Report any cross-PR dependencies found.
```

### Step 3: PR-A cleanup + squash

In the PR-A worktree:

```bash
cd .claude/worktrees/pr-upstream-a

# Remove files NOT for upstream
git rm -r docs/perf_database/ CLAUDE.md .github/copilot-instructions.md 2>/dev/null

# Remove PR-B files (tools)
git rm -r tools/perf_data_collection/ tests/tools/ 2>/dev/null

# Clean .gitignore — remove internal entries, keep universally useful ones
# (edit manually, keep: .DS_Store, __pycache__, coverage, .conda/.venv, generated_microbench/, PROF_*/, temp/)

git add -A && git commit -m "chore: prepare PR-A branch for upstream"

# Squash all commits into 1
git reset --soft gitcode-ascend/develop
git commit -m "<PR-A commit message — see Co-author List below>"
```

### Step 4: PR-B cleanup + squash

In the PR-B worktree:

```bash
cd .claude/worktrees/pr-upstream-b

# Remove files NOT for upstream
git rm -r docs/perf_database/ CLAUDE.md .github/copilot-instructions.md 2>/dev/null

# Revert all non-tools changes back to develop state
git checkout gitcode-ascend/develop -- tensor_cast/ tests/test_tensor_cast/ tests/perf_database/ .gitattributes .gitignore docs/

git add -A && git commit -m "chore: prepare PR-B branch for upstream"

# Squash
git reset --soft gitcode-ascend/develop
git commit -m "<PR-B commit message — see Co-author List below>"
```

### Step 5: Verify (comprehensive)

**⚠️ NEVER run full tensor_cast test suite — machine will break.**

#### PR-A verification:

```bash
cd .claude/worktrees/pr-upstream-a
git lfs pull

# 1. Unit tests
python3.10 -m pytest tests/perf_database/ -q --tb=line                    # ~3s, 285 tests
python3.10 -m pytest tests/test_tensor_cast/test_empirical.py -q          # ~2s
python3.10 -m pytest tests/test_tensor_cast/test_dfc_pass.py -q           # ~2min

# 2. Functional tests — analytic mode (unchanged behavior)
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 --device TEST_DEVICE --log-level warning
# Expected: [analytic] output with TPS

# 3. Functional tests — profiling mode (new feature)
DATA_DIR="tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5"
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 1 --query-length 4112 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile --profiling-database "$DATA_DIR" \
  --enable-flashcomm-v1 --log-level warning
# Expected: [empirical] output with TPS, ~155ms execution time

# 4. Linter
lintrunner -a
```

#### PR-B verification:

```bash
cd .claude/worktrees/pr-upstream-b

# 1. Unit tests
python3.10 -m pytest tests/tools/ -q --tb=line

# 2. Import isolation check
python3.10 -c "
import ast, sys, pathlib
for f in pathlib.Path('tools/perf_data_collection').rglob('*.py'):
    tree = ast.parse(f.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, 'module', '') or ''
            if 'tensor_cast' in mod:
                print(f'ERROR: {f}:{node.lineno} imports tensor_cast')
                sys.exit(1)
print('OK: no tensor_cast imports in tools/')
"

# 3. Linter
lintrunner -a
```

### Step 6: Push

```bash
git push gitcode pr/perf-db-a --force
git push gitcode pr/perf-db-b --force
```

---

## Sync Process (`/upstream-pr sync`)

When `gitcode-ascend/develop` has new commits not yet in `feat/perf-database`:

### Step 1: Create sync worktree

```bash
git worktree add .claude/worktrees/sync-develop -b sync/develop-YYYYMMDD gitcode/feat/perf-database
cd .claude/worktrees/sync-develop
```

### Step 2: Merge develop

```bash
git merge gitcode-ascend/develop
# Resolve conflicts following the merge principles:
# - Interface naming from develop
# - Feature logic from our implementation
# - One module per feature
```

### Step 3: Verify

```bash
python3.10 -m pytest tests/perf_database/ -q --tb=line
python3.10 -c "from tensor_cast.core.model_runner import ModelRunner; print('OK')"
# Functional test: analytic + profiling mode
```

### Step 4: Push sync branch and notify

```bash
git push gitcode sync/develop-YYYYMMDD
echo "⚠️ Sync branch pushed to gitcode/sync/develop-YYYYMMDD"
echo "→ Create MR to gitcode/feat/perf-database for team review"
echo "→ After MR merged, run /upstream-pr update to rebuild PR-A/B"
```

---

## Update Process (`/upstream-pr update`)

When team merges new bug fixes to feat/perf-database:

1. `git fetch gitcode`
2. Compare `gitcode/feat/perf-database` with last build base
3. If new commits exist:
   - Remove old worktrees
   - Re-run full build process (Steps 1-6)
   - Force push updates
4. Notify user of changes since last build

---

## Co-author Lists

### PR-A:

```
Co-authored-by: tt0cool <xujintao8@h-partners.com>
Co-authored-by: Secluded_Ocean <tangchuxiao0709@qq.com>
Co-authored-by: goodluck008 <goodluck008.qq.com>
Co-authored-by: hudingyi <hudingyi@hudingyideMacBook-Air.local>
Co-authored-by: stormchasingg <stormchasingg@noreply.gitcode.com>
Co-authored-by: zhangzhenyu <a>
Co-authored-by: 星北 <xingbei.gc@antgroup.com>
Co-authored-by: Kudo__shinichi <liuning119@huawei.com>
```

### PR-B:

```
Co-authored-by: Secluded_Ocean <tangchuxiao0709@qq.com>
Co-authored-by: hudingyi <hudingyi@hudingyideMacBook-Air.local>
Co-authored-by: zhangzhenyu <a>
```

---

## File Lists

### PR-A Inclusion

```
tensor_cast/performance_model/profiling_database/  (*.py + data/)
tensor_cast/performance_model/empirical.py
tensor_cast/performance_model/comm_analytic.py
tensor_cast/compilation/freezing_passes/dispatch_ffn_combine_pass.py
tensor_cast/compilation/passes/flashcomm_v1_pass.py
tensor_cast/compilation/compile_backend.py  (modified)
tensor_cast/ops/fused_moe.py
tensor_cast/layers/moe_layer.py
tensor_cast/ops/mla.py
tensor_cast/core/model_runner.py
tensor_cast/core/user_config.py
tensor_cast/scripts/text_generate.py
tensor_cast/config.py
tensor_cast/model_config.py
tests/perf_database/  (all)
tests/test_tensor_cast/test_empirical.py
tests/test_tensor_cast/test_dfc_pass.py
tests/test_tensor_cast/test_flashcomm_v1_pass.py
tests/test_tensor_cast/test_compile_passes_spike.py
tests/test_tensor_cast/test_runtime.py  (modified)
tests/test_tensor_cast/test_text_generate.py  (modified)
tests/test_tensor_cast/test_parallel_moe.py  (modified)
.gitattributes
.gitignore  (modified)
docs/en/tensor_cast_instruct.md  (modified)
```

### PR-B Inclusion

```
tools/perf_data_collection/  (all *.py + *.sh)
tests/tools/  (all)
```

### Exclusion (Neither PR)

```
docs/perf_database/          # Internal dev docs, skills, dashboards
CLAUDE.md                    # Claude Code config
.claude/                     # Claude Code config
.github/copilot-instructions.md
results/                     # Ephemeral metrics output
```

---

## Notes

- **Never run full tensor_cast test suite locally** — machine will break
- **Module naming**: `profiling_database` (mainline dir), `DataSourcePerformanceModel` (mainline ABC)
- **CLI flag**: `--profiling-database` (not `--perf-database`)
- After PR merges to develop: `feat/perf-database merge gitcode-ascend/develop` (squash merge produces "phantom conflicts" where both sides have identical content from different commits — manually resolve each conflict, verify content matches, do NOT use `-X` auto-resolve)
- Future dev: continue on feat/perf-database → periodically merge develop → cut new clean branch → squash → new PR
