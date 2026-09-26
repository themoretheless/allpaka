---
name: ci-self-heal-analyzer
description: Консервативный анализ неудач CI и предложение исправлений без автоматического применения. Использовать когда нужно автоматически определить причины failing tests, formatting issues, или clippy warnings — но применить fixes вручную.
---

# CI Self-Heal Analyzer

Conservative analysis tool для CI failure diagnosis с suggestions without auto-applying.

## Когда использовать

- После неудачного CI run — получить automated diagnosis
- Поиск root causes для failing tests или build failures
- Автоматическое обнаружение formatting/clippy issues
- Генерация fix recommendations для team members
- Integration в issue tracking workflow (draft PR creation)

## ⚠️ Critical Safety Boundaries

**ЭТОТ СКРИПТ НЕ ПРИМЕНЯЕТ ИСПРАВЛЕНИЯ АВТОМАТИЧЕСКИ!**

| ❌ NO | ✅ DO |
|-------|------|
| Automatic commits | Only generate suggestions |
| Pushing changes anywhere | Manual application by user |
| Silent modifications | Clear "how to apply" instructions |
| Force-fixes | Conservative detection with confidence scores |

###例外

`--commit` flag still requires interactive confirmation before applying ANY changes.

## Установка требований

```sh
# Проверить авторизацию для GitHub API
gh auth status

# Если не авторизован
gh auth login
```

## Основные команды

### Table Output (Recommended)

Markdown table с рекомендациями по fix'ам:

```bash
scripts/ci-self-heal-analyzer.sh --output=table <RUN_ID>
```

**Пример вывода:**
```markdown
## CI Failure Analysis Report

| Issue Type | Files Affected | Fix Command | Confidence |
|------------|----------------|-------------|------------|
| formatting | backend/src/lib.rs | `cargo fmt` | 100% |
| clippy | src/cli.rs | `cargo clippy --fix` | 95% |

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

### Diff Output

Diff snippets которые можно применить:

```bash
scripts/ci-self-heal-analyzer.sh --output=diff <RUN_ID>
```

**Output example:**
```diff
### Formatting corrections
--- src/file.rs
+++ src/file.rs
@@ -1 +1 @@
-fn old() {}
+fn old() -> Result<(), Error> { Ok(()) }
```

### Issue Template

GitHub issue template для tracking:

```bash
scripts/ci-self-heal-analyzer.sh --output=issue <RUN_ID>
```

Creates full markdown report как draft issue ready for review.

## Детальное применение

### Basic Usage

```bash
# Get analysis after CI failure
SCRIPTS_DIR=$(git rev-parse --show-toplevel)/scripts

# From CI job
bash $SCRIPTS_DIR/ci-self-heal-analyzer.sh --output=table $GITHUB_RUN_ID

# Or from local
bash $SCRIPTS_DIR/ci-self-heal-analyzer.sh --output=table 123456
```

### Analyze Specific Failure Type

```bash
# Focus on formatting issues only
bash scripts/ci-self-heal-analyzer.sh --output=table $RUN_ID | \
  grep -A 10 "formatting"

# Get only diff hints
bash scripts/ci-self-heal-analyzer.sh --output=diff $RUN_ID | \
  cargo apply --dry-run
```

## Application Workflow

### Step-by-Step Safe Application

```bash
# 1. Generate analysis
ANALYSIS=$(bash scripts/ci-self-heal-analyzer.sh --output=table $RUN_ID)

# 2. Review recommendations
echo "$ANALYSIS" | less

# 3. Manually apply each suggestion
cargo fmt
cargo clippy --fix --allow-dirty --allow-staged

# 4. Verify and commit
git status
git add .
git commit -m "chore: apply self-heal suggestions"

# 5. Trigger re-run
git push
```

### Automated (With Confirmation)

```bash
# With explicit --commit flag AND interactive prompt
bash scripts/ci-self-heal-analyzer.sh --commit $RUN_ID

# Prompts: "Confirm application of all fixes? [y/N]: "
# User must type 'y' to proceed
```

## Integration in CI/CD

### GitHub Actions (Diagnostic Mode)

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: cargo test
      
  # Optional: Self-heal diagnostic on failure
  self-heal-diagnostic:
    if: failure() && github.event.pull_request.number
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Analyze CI failure
        run: |
          RUN_ID=${{ github.run_id }}
          
          echo "# CI Failure Analysis" > .qoder/reports/self-heal.md
          bash scripts/ci-self-heal-analyzer.sh --output=table $RUN_ID >> .qoder/reports/self-heal.md
          
      - name: Upload analysis artifact
        uses: actions/upload-artifact@v4
        with:
          name: self-heal-analysis
          path: .qoder/reports/self-heal.md
```

### Comment on PR Automatically

```yaml
# .github/workflows/pr-self-heal-comment.yml
name: Self-Heal Comment on Failure
on:
  pull_request:
    types: [closed]

jobs:
  comment-analysis:
    if: github.event.pull_request.merged == false
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Generate fix suggestions
        id: analysis
        run: |
          RESULTS=$(bash scripts/ci-self-heal-analyzer.sh --output=table $GITHUB_RUN_ID)
          echo "::notice::Self-heal analysis completed"
          echo "$RESULTS" > analysis-report.md
      
      - name: Comment on PR
        uses: octokit/request-action@v2.x
        with:
          route: post
          owner: ${{ github.repository_owner }}
          repo: ${{ github.event.repository.name }}
          issue_number: ${{ github.event.pull_request.number }}
          body: |
            ## CI Self-Heal Suggestions
            
            Generated analysis for failed CI:
            
            ```bash
            $(cat analysis-report.md)
            ```
            
            **Please review and apply manually.** This tool suggests fixes but does not auto-commit.
```

## Detection Algorithms

### 1. Formatting Issues (`cargo fmt`)

```rust
// Checks for missing braces, wrong indentation, etc.
fn broken_style() {     // Would be detected
    let x=1;           // Missing spaces
}                      // Extra blank line
```

Detection method:
```bash
cargo fmt --check --all 2>&1 | grep "^diff"
```

Confidence: **100%** — binary pass/fail check

### 2. Clippy Warnings

```rust
// Detects unnecessary allocations, better alternatives
let vec = Vec::new();  // Suggested: Vec::new() → Default::default()
```

Detection method:
```bash
cargo clippy --all-targets 2>&1 | grep -E "(warning:|help:)"
```

Confidence: **95%** — may require manual review of suggestions

### 3. Test Expectation Mismatches

```rust
#[test]
fn test_output() {
    assert_eq!(actual, expected);  // Detects when actual != expected
}
```

Detection method:
```bash
# Parse test logs for panic messages
grep -A 5 "thread.*panicked\|assertion failed" test-output.log
```

Confidence: **80%** — requires human judgment for expected value updates

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Successfully analyzed, suggestions generated |
| 1 | No automatic fixes detected or error during analysis |
| 2 | Authentication/access error (not logged into GitHub CLI) |

## Configuration

### Environment Variables

```bash
# Set repository if not in workspace
export GITHUB_REPO="owner/name"

# Custom output directory
export OUTPUT_DIR=".qoder/reports"
```

### Timeout for Operations

```bash
# If running on slow runner, increase timeout
export MAX_ANALYSIS_TIME="60s"
```

## Examples

### Daily CI Health Check

```bash
#!/bin/bash
# scripts/daily-ci-health.sh

LAST_RUN=$(gh run list --limit 1 --repo owner/repo --json databaseId --jq '.[0].databaseId')

if [[ $LAST_RUN ]]; then
  echo "Analyzing latest CI run..."
  bash scripts/ci-self-heal-analyzer.sh --output=table "$LAST_RUN" > daily-ci-report.md
  
  if grep -q "failing" daily-ci-report.md; then
    echo "⚠️ CI health issues detected"
    cat daily-ci-report.md
  else
    echo "✅ CI is healthy"
  fi
fi
```

### Weekly Summary Report

```python
# weekly-ci-summary.py
import subprocess
from datetime import datetime

def analyze_week():
    results = []
    
    for day in range(7):
        date = datetime.now().replace(day=datetime.now().day - day)
        
        runs = subprocess.run(
            ['gh', 'run', 'list', '--date', date.strftime('%Y-%m-%d'), '--json', 'databaseId,conclusion'],
            capture_output=True, text=True
        )
        
        for run in json.loads(runs.stdout):
            if run['conclusion'] == 'failure':
                result = subprocess.run(
                    ['bash', 'scripts/ci-self-heal-analyzer.sh', '--output=table', str(run['databaseId'])],
                    capture_output=True, text=True
                )
                results.append({
                    'date': date,
                    'analysis': result.stdout
                })
    
    return results
```

## Comparison with Manual Debugging

### Manual Approach (Time-consuming)

```bash
# Look at CI logs
gh run view $RUN_ID --log-failed

# Manally identify patterns
grep "error:" ci-log.txt
grep "warning:" ci-log.txt

# Try fixes
cargo fmt
cargo clippy
# Hope it works...
```

### Self-Heal Approach (Automated)

```bash
# Single command analysis
bash scripts/ci-self-heal-analyzer.sh --output=table $RUN_ID

# Gets structured recommendations with confidence scores
# And clear "how to apply" instructions
```

Преимущества:
- ✅ Consistent diagnostics across all failures
- ✅ Machine-readable output format
- ✅ Confidence scoring for trustworthiness
- ✅ Clear application instructions
- ✅ Audit trail for team knowledge base

## Best Practices

1. **Always review before applying**: Never use `--commit` without reading suggestions first
2. **Check confidence scores**: Lower confidence means more manual review needed
3. **Use as team knowledge base**: Store analyses in shared docs for future reference
4. **Combine with manual inspection**: Auto-analysis complements human debugging, doesn't replace it
5. **Share findings**: Add valuable suggestions to team wiki/documentation

## Troubleshooting

### No fixes detected

```bash
$ bash scripts/ci-self-heal-analyzer.sh --output=table 123456
| No automatic fixes detected | - | Manual inspection required | - |
```

**Meaning:** Сложная ошибка требует ручного анализа (logic errors, dependency issues, infrastructure problems)

### Authentication error

```bash
Error: Failed to fetch CI run #123456
Make sure you have access to this repository
```

**Solution:**
```bash
gh auth login --scopes workflow,read:org
```

### Timeout on large repos

```bash
Analysis timed out waiting for cargo clippy output
```

**Solution:**
```bash
# Limit scope
cargo clippy --package specific-package --features feature-name
```

## Related Skills

- **gh pr checks**: Quick manual check of individual PR status
- **git-uncommitted-report**: Report changes after applying fixes
- **gh-auto-merge**: Auto-merge после successful CI с помощью self-heal
