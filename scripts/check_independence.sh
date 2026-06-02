#!/usr/bin/env bash
# Independence / leakage guard for LuminaST (issue #112).
#
# Enforces the policy documented in BASELINE_COMPARISON.md ("Leakage
# Prevention"): shipped code must not bake in the prior-art baseline's brand
# strings or provenance identifiers. Exits non-zero on any hit so it can be
# wired into CI. Run from the repo root:
#
#     bash scripts/check_independence.sh
#
set -euo pipefail

# Brand / provenance strings that must never appear in shipped code. The
# pattern is case-sensitive and matches the documented policy in
# BASELINE_COMPARISON.md (DeepSpatial, stPainter, the upstream author handle,
# the legacy project name, and the baseline DOI).
PATTERN='DeepSpatial|stPainter|yyh030806|Gene Diffusion Transformer|10\.64898/2026\.02\.11\.704553'

# Code zones that must stay brand-free. docs/ is intentionally NOT scanned:
# legitimate prior-art *citations* live there per the policy ("Prior Art and
# Audit Boundary").
ZONES=(src scripts tests experiments)

# Documented exemptions, mirroring tests/test_issue_121_independence_guard.py:
#   * results_contract.py — vendored byte-identical from the parent
#     orchestration repo (SHA-pinned by tests/test_contract_schema.py); its
#     docstring example references the baseline on-disk layout and cannot be
#     edited without breaking byte-identity.
#   * compare_outputs.py — the explicitly-gated audit-comparison script that
#     legitimately points at an external baselines/stPainter-original clone.
#   * the guard test + this script itself necessarily contain the pattern.
EXCLUDES=(
  --exclude-dir=__pycache__
  --exclude-dir='*.egg-info'
  --exclude='check_independence.sh'
  --exclude='results_contract.py'
  --exclude='compare_outputs.py'
  --exclude='test_issue_121_independence_guard.py'
)

EXISTING_ZONES=()
for z in "${ZONES[@]}"; do
  [ -d "$z" ] && EXISTING_ZONES+=("$z")
done

HITS=$(grep -rnE "$PATTERN" "${EXISTING_ZONES[@]}" "${EXCLUDES[@]}" 2>/dev/null || true)

if [ -n "$HITS" ]; then
  echo "FAIL: baseline brand / provenance references found in shipped code:" >&2
  echo "$HITS" >&2
  exit 1
fi

echo "PASS: no baseline brand references in ${EXISTING_ZONES[*]}."
