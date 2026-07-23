## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-27 | Added test layout documentation during restructuring |
| 2026-03-15 | Added self-contained GPT-6.7B slowdown workflow reference |

# Test Layout

- `tests/unit/`: isolated module/unit checks
- `tests/integration/`: cross-module integration checks
- `tests/performance/`: performance and accuracy comparison scripts
- `tests/e2e/`: full workflow smoke and end-to-end checks
  - `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh`: self-contained Phase 8c/8d workflow using trace-shaped schedule generation and case-local targeted `NCU` collection

Notes:
- Legacy ad-hoc test scripts were consolidated from project root.
- Some historical exploratory scripts are stored in `tests/unit/topology/` and are excluded from default pytest collection via `tests/unit/topology/conftest.py`.
