#!/usr/bin/env bash
# Run every Studio UI test. Each one loads slices of crates/allpaka-chat/web/app.js
# into a node:vm context with stub DOM objects and asserts on the result, so
# they need node and nothing else. Paths inside the tests are relative to the
# repository root; this script changes there first.
#
# usage: scripts/studio-ui-tests/run.sh [name-substring]
set -uo pipefail

cd "$(dirname "$0")/../.."
filter=${1:-}
passed=0
failed=()
for test in scripts/studio-ui-tests/test-studio-*-ui.js; do
    [[ -n "$filter" && "$test" != *"$filter"* ]] && continue
    if output=$(node "$test" 2>&1); then
        passed=$((passed + 1))
    else
        failed+=("$test")
        printf 'FAIL %s\n%s\n' "$test" "$output" >&2
    fi
done
printf '%d passed, %d failed\n' "$passed" "${#failed[@]}"
[[ ${#failed[@]} -eq 0 ]]
