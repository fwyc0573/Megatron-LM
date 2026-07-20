#!/usr/bin/env bash
# Run the strict V21 verifier with shell-level fail-fast propagation.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
EXPECTED_STATUS=VERIFIER_PENDING

while (($# > 0)); do
    case "$1" in
        --expected-status)
            (($# >= 2)) || { printf 'missing value for --expected-status\n' >&2; exit 2; }
            EXPECTED_STATUS=$2
            shift 2
            ;;
        --repo-root)
            (($# >= 2)) || { printf 'missing value for --repo-root\n' >&2; exit 2; }
            REPO_ROOT=$(cd -- "$2" && pwd)
            shift 2
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

printf 'SESSION47_FINAL_VERIFICATION_V21=START\n'
DOC_OUTPUT=$(bash "${REPO_ROOT}/tests/unit/test_sc26_ae_docs_contract.sh")
printf '%s\n' "${DOC_OUTPUT}"
grep -Fxq 'DOC_CONTRACT_STATUS=PASS' <<<"${DOC_OUTPUT}"
grep -Fxq 'PUBLIC_ENTRY_COUNT=9' <<<"${DOC_OUTPUT}"
grep -Fxq 'PAPER_SUGGESTION_COUNT=10' <<<"${DOC_OUTPUT}"
python3 "${REPO_ROOT}/tests/integration/sc26_ae_v21_verifier.py" \
    --repo-root "${REPO_ROOT}" \
    --expected-status "${EXPECTED_STATUS}"
printf 'SESSION47_FINAL_VERIFICATION_V21=PASS\n'
