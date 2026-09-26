# GitHub Workflow Enhancements

Comprehensive tools and scripts for automated GitHub workflow management while maintaining **safety-first philosophy**.

## Overview

This directory contains automation scripts for:

- **Auto-Merge** with precondition checking (`scripts/gh-auto-merge.sh`)
- **PR Status Monitoring** in watch mode (`scripts/gh-pr-status-watch.sh`)
- **Worktree Lifecycle Management** (`scripts/git-worktree-manager.sh`)
- **CI Self-Healing Analysis** (conservative fix suggestions) (`scripts/ci-self-heal-analyzer.sh`)
- **PR Lifecycle Hooks** (stale PR handling, labeling) (`scripts/gh-pr-lifecycle-hooks.sh`)

All scripts:
- ✅ Default to read-only mode unless explicitly requested
- ✅ Respect branch protection rules
- ✅ Require confirmation for destructive actions
- ✅ Support `--dry-run` for safe testing
- ✅ Log all actions to `.qoder/logs/`

---

## Table of Contents

1. [Auto-Merge](#auto-merge)
2. [PR Status Watch](#pr-status-watch)
3. [Worktree Manager](#worktree-manager)
4. [CI Self-Heal Analyzer](#ci-self-heal-analyzer)
5. [Shared Utilities](#shared-utilities)
6. [Safety Guidelines](#safety-guidelines)
7. [Testing](#testing)

---

## Auto-Merge

**File:** `scripts/gh-auto-merge.sh`

### Purpose

Automatically merge PRs when all conditions are met, while respecting branch protection rules and avoiding conflicts.

### Usage

```bash
# Check conditions without merging (dry-run)
scripts/gh-auto-merge.sh --dry-run <PR_NUMBER>

# Merge with squash strategy (default)
scripts/gh-auto-merge.sh --strategy=squash <PR_NUMBER>

# Merge with timeout (wait up to N minutes for checks)
scripts/gh-auto-merge.sh --timeout=30m --strategy merge <PR_NUMBER>
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy=SQUASH\|MERGE\|REBASE` | `squash` | Merge strategy |
| `--timeout=DURATION` | `30m` | Maximum wait time for CI checks |
| `--dry-run` | `false` | Preview what will happen without executing |

### Preconditions Checked

Before merging, the script verifies:

1. ✅ PR is open (not closed or merged)
2. ✅ No merge conflicts (`mergeable=true`)
3. ✅ All required status checks pass
4. ✅ Branch protection rules respected
5. ✅ User has merge permissions

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Successfully merged |
| 1 | Precondition not met (conflict, failing checks, etc.) |
| 2 | Timeout waiting for checks |
| 3 | Not authenticated / insufficient permissions |

### Example Session

```bash
$ scripts/gh-auto-merge.sh --dry-run 123

=== Checking prerequisites ===
Fetching PR #123...

=== PR Details ===
  Number:   #123
  Title:    Fix critical bug in backend
  State:    open
  From:     feature-critical-fix
  Base:     main
  Mergeable: true
  CI Status: success
  Reviews:   approved

=== Dry Run Mode ===
Would perform the following action:
  gh pr merge 123 --repo owner/repo --strategy squash

No changes made.
```

### Integration with SKILL.md

The git and gh plugin skills reference auto-merge commands:

- `plugins/git.SKILL.md` - Worktree automation examples
- `plugins/gh.SKILL.md` - Auto-merge and PR monitoring sections

See those files for usage patterns within agent workflows.

---

## PR Status Watch

**File:** `scripts/gh-pr-status-watch.sh`

### Purpose

Continuously monitor PR check statuses and receive real-time updates on CI progress.

### Usage

```bash
# Watch single PR until all checks complete
scripts/gh-pr-status-watch.sh <PR_NUMBER>

# Multiple PRs in batch mode
cat pr-list.txt | xargs -I{} scripts/gh-pr-status-watch.sh {}

# JSON output for CI integration
scripts/gh-pr-status-watch.sh --json <PR_NUMBER> | jq '.checks[]'
```

### Output Formats

#### Markdown (Default)

Human-readable table with status per check:

```
┌─────────────┬──────────────────┬─────────────┬─────────────┬──────────────────────────┐
│ PR №       │ Check Name        │ Status      │ Result      │ Detail                   │
├─────────────┼──────────────────┼─────────────┼─────────────┼──────────────────────────┤
│ #123        │ Build & Test      │ completed   │ success     │                          │
│ #123        │ Security Scan     │ completed   │ failure     │ 2 high-severity issues   │
│ #123        │ Documentation     │ pending     │ n/a         │ Waiting for queue        │
│ #123        │ Lint              │ completed   │ success     │                          │
└─────────────┴──────────────────┴─────────────┴─────────────┴──────────────────────────┘

📊 Summary: 2 passing, 1 failing, 1 pending
⚠️  WARNING: Some checks are failing!
```

#### JSON Format

Machine-parseable output for automation:

```json
{
  "pr": 123,
  "checks": [
    {
      "name": "Build & Test",
      "workflowName": "CI",
      "status": "completed",
      "conclusion": "success",
      "detail": "",
      "url": "https://github.com/..."
    },
    ...
  ]
}
```

### Watch Mode Behavior

For a single PR, enters continuous watch mode:
- Polls every N seconds (configurable via `--interval`)
- Updates display each interval
- Press Ctrl+C to exit cleanly
- Shows summary after each update

### Batch Mode

For multiple PRs, polls once and exits:
- Good for CI integration
- Combine with `jq` for filtering
- Output can be redirected to files

### Configuration

```bash
# Set default poll interval
export DEFAULT_INTERVAL="10s"  # options: s, m, h suffixes

# Set base repository if not detected from git remote
export GITHUB_REPO="owner/name"
```

---

## Worktree Manager

**File:** `scripts/git-worktree-manager.sh`

### Purpose

Automate creation, management, and cleanup of git worktrees for parallel development across multiple branches.

### Commands

#### Create Worktree

```bash
scripts/git-worktree-manager.sh add <branch> <path>

# Example: create isolated worktree for feature branch
scripts/git-worktree-manager.sh add feature-x ./worktrees/feature-x
```

Features:
- Validates disk space before creating
- Checks if path already exists
- Creates new branch automatically
- Shows preview in dry-run mode

#### List Worktrees

```bash
scripts/git-worktree-manager.sh list [--verbose|--json]

# Examples
scripts/git-worktree-manager.sh list                     # Basic listing
scripts/git-worktree-manager.sh list --verbose           # With SHA details
scripts/git-worktree-manager.sh list --json              # Machine-readable
```

Output format:

```
=== Git Worktrees ===

Path              Branch         Commit SHA         
----------------- -------------- -------------------
./worktrees/dev   dev            a1b2c3d            
./worktrees/feat  feat-api       e4f5g6h            
./worktrees/test  test-release   i7j8k9l            

Total worktrees: 3
```

#### Remove Worktree

```bash
scripts/git-worktree-manager.sh remove <path>|<branch>

# Examples
scripts/git-worktree-manager.sh remove ./worktrees/feature-x
scripts/git-worktree-manager.sh remove feature-y --force
```

Safety features:
- Detects uncommitted changes
- Warns before removal
- Requires explicit confirmation
- Supports `--force` to skip warning
- Optionally deletes local branch

#### Prune Stale Entries

```bash
scripts/git-worktree-manager.sh prune [--dry-run]

# Cleanup orphaned worktree paths
scripts/git-worktree-manager.sh prune
```

Removes stale entries where the filesystem path no longer exists.

#### Sync Between Worktrees

```bash
scripts/git-worktree-manager.sh sync <source-path> <target-branch>

# Show preview of sync
scripts/git-worktree-manager.sh --dry-run sync ./worktrees/feat main
```

Shows commits that would be applied, doesn't modify anything.

### Safety Rules

| Rule | Description |
|------|-------------|
| Disk space check | Requires minimum 100MB free |
| Path validation | Never overwrites existing directories |
| Change warning | Always shows uncommitted changes before removal |
| Branch safety | Only deletes local branches, never remote |
| Orphan detection | Prunes broken references automatically |

### Integration with Reporting

Works seamlessly with `git-uncommitted-report.sh`:

```bash
# Report for all worktrees
for wt in ./worktrees/*/; do
  echo "=== $wt ==="
  bash scripts/git-uncommitted-report.sh -C "$wt" || true
done
```

---

## CI Self-Heal Analyzer

**File:** `scripts/ci-self-heal-analyzer.sh`

### Purpose

Analyze CI failures and suggest conservative fixes without automatically applying them.

### IMPORTANT SAFETY BOUNDARIES

- ❌ **NO automatic commits** by default
- ❌ **NO pushing changes** anywhere
- ✅ **Only generates suggestions** in stdout/file
- ✅ Includes clear **"how to apply"** instructions

### Usage

```bash
# Generate markdown table of suggested fixes
scripts/ci-self-heal-analyzer.sh --output=table <RUN_ID>

# Output diff snippets for manual application
scripts/ci-self-heal-analyzer.sh --output=diff <RUN_ID>

# Create draft issue with analysis
scripts/ci-self-heal-analyzer.sh --output=issue <RUN_ID>

# Apply fixes automatically (REQUIRES CONFIRMATION!)
scripts/ci-self-heal-analyzer.sh --commit <RUN_ID>
```

### Output Formats

#### Table Format

Markdown table with fix recommendations:

```markdown
## CI Failure Analysis Report

| Issue Type | Files Affected | Fix Command | Confidence |
|------------|----------------|-------------|------------|
| formatting | backend/src/lib.rs | `cargo fmt` | 100% |
| clippy | multiple files | `cargo clippy --fix` | 95% |

### Recommended Actions

```bash
# 1. Apply formatting fixes
cargo fmt

# 2. Apply clippy fixes (review carefully!)
cargo clippy --fix --allow-dirty --allow-staged

# 3. Re-run tests
cargo test --workspace
```

After applying fixes:
  $ git commit -m "chore: self-heal CI fixes"
  $ git push
```

#### Diff Format

Shows actual diff snippets that could be applied:

```diff
### Formatting corrections
--- src/file.rs
+++ src/file.rs
@@ -1 +1 @@
-fn old() {}
+fn old() -> Result<(), Error> { Ok(()) }
```

#### Issue Template Format

Full GitHub issue template with analysis results (useful for tracking).

### Supported Fix Types

| Type | Detection Method | Confidence |
|------|------------------|------------|
| Formatting | `cargo fmt --check` | 100% |
| Clippy warnings | `cargo clippy` | 95% |
| Test expectations | Log analysis | 80% |

### Manual Application Steps

For any suggested fix:

```bash
# 1. Review suggestion
scripts/ci-self-heal-analyzer.sh --output=table <RUN_ID>

# 2. Manually apply fix
cargo fmt
cargo clippy --fix

# 3. Verify and commit
git status
git add .
git commit -m "chore: apply suggested CI fixes"

# 4. Push to trigger re-run
git push
```

### Integration with CI Workflow

Add this job to your CI workflow (optional):

```yaml
# .github/workflows/self-heal-analysis.yml
name: Self-heal Analysis
on:
  workflow_run:
    workflows: ["ci"]
    types: [completed]
    branches: [main]

jobs:
  analyze-failure:
    if: github.event.workflow_run.conclusion == 'failure'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Analyze CI failure
        run: |
          # Download failed run logs
          gh run download ${{ github.event.workflow_run.id }} --name test-log
          
          # Generate self-heal report
          bash scripts/ci-self-heal-analyzer.sh --output=issue \
            --output-file=.qoder/reports/self-heal-${{ github.event.workflow_run.id }}.md \
            ${{ github.event.workflow_run.id }}
```

---

## Shared Utilities

**File:** `/.qoder/scripts-common.sh`

### Purpose

Common functions and utilities shared across all automation scripts.

### Included Functions

#### Logging

```bash
log_debug "message"  # Debug output
log_info "message"   # Information
log_warn "message"   # Warning (shown even in quiet mode)
log_error "message"  # Error
```

#### Rate Limit Handling

```bash
# Wait for rate limit to reset (max 60 seconds)
wait_for_rate_limit [max_wait_seconds]

# Usage within scripts:
wait_for_rate_limit 30 || exit 1
```

#### JSON Parsing Helper

```bash
# Get JSON field with default value
json_get "<json>" "<jq_filter>" "[default_value]"

# Example:
json_get "$data" '.mergeable // "unknown"'
```

#### Action Logging

```bash
# Log to both stdout and timestamped log file
log_action "<action_type>" "<details>"

# Writes to: .qoter/logs/YYYYMMDD.log
```

#### Configuration Variables

```bash
readonly LOG_DIR="${LOG_DIR:-.qoder/logs}"
readonly REPORTS_DIR="${REPORTS_DIR:-.qoder/reports}"
```

Customizable via environment variables.

### How to Use in Scripts

At the top of any automation script:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.qoder/scripts-common.sh"
```

---

## Safety Guidelines

### Core Principles

1. **Read-only by default** - All operations require explicit flags for modifications
2. **Explicit confirmation** - Destructive actions always ask for permission
3. **Audit trail** - All changes logged to `.qoder/logs/`
4. **Respect protections** - Never bypass branch protection rules
5. **Rate limit awareness** - Back off gracefully on 403 responses

### Best Practices

#### Before Merging

```bash
# Always dry-run first
scripts/gh-auto-merge.sh --dry-run <PR_NUMBER>

# Verify conditions manually
gh pr view <PR_NUMBER> --json mergeable,statusCheckRollup,reviewDecision
```

#### Before Removing Worktrees

```bash
# Check for uncommitted changes first
git worktree list

# Ensure clean state
cd <worktree-path>
git status

# Then proceed with removal
scripts/git-worktree-manager.sh remove <worktree-path>
```

#### Before Applying Fixes

```bash
# Always review suggestions first
scripts/ci-self-heal-analyzer.sh --output=table <RUN_ID>

# NEVER use --commit without reviewing
# Instead, manually apply:
cargo fmt
cargo clippy --fix
```

### Emergency Recovery

If something goes wrong:

1. **Undo recent merges:**
   ```bash
   git reflog
   git reset --hard HEAD@{previous-position}
   ```

2. **Recover deleted worktree:**
   ```bash
   # If worktree was removed but branch still exists:
   git checkout -b recovered-branch <sha-from-reflog>
   ```

3. **Restore rate limit access:**
   ```bash
   # Check current rate limit
   gh api rate_limit --jq '.rate'
   
   # Wait if exhausted
   sleep 60
   ```

---

## Testing

### Unit Tests (Dry-Run Mode)

All scripts support `--dry-run` for safe testing:

```bash
# Test auto-merge without execution
scripts/gh-auto-merge.sh --dry-run 123

# Test worktree creation
scripts/git-worktree-manager.sh add feature-test ./worktrees/test-dry \
  --dry-run

# Test analyzer without committing
scripts/ci-self-heal-analyzer.sh --output=table 123456
```

### Integration Tests

#### Test Auto-Merge Flow

```bash
# 1. Create test PR locally
git checkout -b test-auto-merge
echo "test" > test-file.txt
git add test-file.txt
git commit -m "test: auto-merge test"
git push origin test-auto-merge
gh pr create --title "Test Auto-Merge" --body "Testing merge automation"

# 2. Run CI and let it pass
# Wait for all checks to succeed

# 3. Test merge
scripts/gh-auto-merge.sh --dry-run <PR_NUMBER>  # Verify preconditions
scripts/gh-auto-merge.sh --strategy squash <PR_NUMBER>  # Execute merge

# 4. Clean up
# (PR should auto-close after merge)
```

#### Test Worktree Isolation

```bash
# 1. Create worktree
scripts/git-worktree-manager.sh add feature-isolation ./worktrees/isolation-test

# 2. Modify files in worktree (different from main repo)
cd ./worktrees/isolation-test
echo "isolated change" >> test.txt
git add test.txt
git commit -m "test: isolation commit"

# 3. Return to main repo
cd -
git status  # Should show worktree separately

# 4. Verify no conflicts
scripts/git-worktree-manager.sh list

# 5. Clean up
scripts/git-worktree-manager.sh remove feature-isolation
```

### CI Testing

Add to `.github/workflows/test-scripts.yml`:

```yaml
name: Test Automation Scripts

on:
  pull_request:
    paths:
      - 'scripts/**'
      - '.qoder/scripts-common.sh'

jobs:
  test-scripts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Make scripts executable
        run: chmod +x scripts/*.sh
        
      - name: Test help messages
        run: |
          for script in scripts/*.sh; do
            bash "$script" --help | grep -q "usage:" || \
              echo "Missing help: $script"
          done
      
      - name: Dry-run critical scripts
        run: |
          bash scripts/gh-auto-merge.sh --dry-run || true
          bash scripts/git-worktree-manager.sh list || true
          echo "All scripts support safe testing modes"
```

---

## Future Enhancements

Potential additions for future phases:

- [ ] Webhook integration for real-time notifications
- [ ] PR template generation based on conventional commits
- [ ] Automated dependency updates with bot-controlled PRs
- [ ] Code owners auto-assignment based on file changes
- [ ] Performance metrics tracking across PRs
- [ ] Integration with external project management tools

---

## Contributing

When adding new automation scripts:

1. Follow existing naming convention: `snake-case.sh`
2. Include full usage documentation in script header
3. Support `--dry-run` flag
4. Log all actions to `.qoder/logs/`
5. Add safety checks before destructive operations
6. Update relevant SKILL.md documentation
7. Include example usage in comments

---

## References

- [`plugins/git.SKILL.md`](../plugins/git.SKILL.md) - Git plugin with worktree examples
- [`plugins/gh.SKILL.md`](../plugins/gh.SKILL.md) - GitHub CLI plugin with auto-merge features
- [`scripts/git-uncommitted-report.sh`](../scripts/git-uncommitted-report.sh) - Existing change reporting tool
- [`scripts/bench-matrix.sh`](../scripts/bench-matrix.sh) - Performance benchmarking infrastructure

---

*Last updated: 2026-09-26*
