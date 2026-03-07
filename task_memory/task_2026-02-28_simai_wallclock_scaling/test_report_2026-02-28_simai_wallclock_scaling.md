## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Initialized test report for SimAI wall-clock scaling implementation |
| 2026-02-28 | Recorded unit/integration validation results for implementation phase |
| 2026-02-28 | Added queue shell validation (syntax + start/status behavior) |
| 2026-03-01 | Added formal-phase blocker evidence for 8192-GPU topology generation |
| 2026-03-01 | Added topology-template feasibility probe for 8192 GPUs |
| 2026-03-01 | Added unit validation for AlibabaHPN topology filename resolution fix |
| 2026-03-02 | Recorded detached 8192 rerun startup evidence without adding redundant tests |
| 2026-03-03 | Added final completion evidence for 8192 measurement and auto-plot outputs |
| 2026-03-03 | Added startup evidence for isolated new-plan rerun (PP=16 for >=1024) |
| 2026-03-03 | Added runner hotfix note and watcher safeguard evidence for new-plan rerun |
| 2026-03-04 | Added completion evidence for isolated new-plan run outputs and plots |

# Test Report: SimAI Wall-clock Scaling

**Date**: 2026-02-28  
**Environment**: `python3 --version` -> `Python 3.10.12`

## 1. Test Script Information

- Scripts under test:
  - `SimAI/tests/performance/simai_wallclock_scaling/measure_wallclock.py`
  - `SimAI/tests/performance/simai_wallclock_scaling/plot_wallclock.py`
  - `SimAI/tests/performance/simai_wallclock_scaling/run_wallclock_queue.sh`
  - `SimAI/tests/unit/test_simai_wallclock_scaling.py`

- Reproducible commands:

```bash
python3 -m unittest discover -s SimAI/tests/unit -p 'test_simai_wallclock_scaling.py' -v
python3 SimAI/tests/performance/simai_wallclock_scaling/measure_wallclock.py --simai-root SimAI --phase sensitivity --dry-run
python3 SimAI/tests/performance/simai_wallclock_scaling/measure_wallclock.py --simai-root SimAI --phase formal --dry-run
python3 SimAI/tests/performance/simai_wallclock_scaling/plot_wallclock.py
bash -n SimAI/tests/performance/simai_wallclock_scaling/run_wallclock_queue.sh
SimAI/tests/performance/simai_wallclock_scaling/run_wallclock_queue.sh start
SimAI/tests/performance/simai_wallclock_scaling/run_wallclock_queue.sh status
SimAI/tests/performance/simai_wallclock_scaling/run_wallclock_queue.sh logs
python3 SimAI/tests/performance/simai_wallclock_scaling/measure_wallclock.py --simai-root SimAI --phase formal
python3 - <<'PY'
import subprocess, tempfile, pathlib
root=pathlib.Path("SimAI/astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py").resolve()
for topo in ["Spectrum-X","AlibabaHPN","DCN+"]:
    with tempfile.TemporaryDirectory() as td:
        cmd=["python3", str(root), "-topo", topo, "-g", "8192", "-gps", "8", "-gt", "H800", "-bw", "400Gbps", "-nvbw", "2400Gbps"]
        p=subprocess.run(cmd,cwd=td,capture_output=True,text=True)
        print(topo, p.returncode)
PY
```

## 2. Validation Criteria

- Unit tests must cover:
  - feasible PP enumeration
  - DP integrality fail-fast
  - formal PP selection policy with 20% threshold
  - strict CSV schema enforcement
  - workload command construction
  - simulation command non-zero fail-fast
  - plotting output generation with synthetic input CSV
- Sensitivity dry-run must enumerate expected 6 configs (8/16/32 all feasible PP).
- Formal dry-run without measured sensitivity rows must fail-fast with explicit error.
- Plot script must fail-fast when result CSV has no measured rows.
- Queue script must:
  - start in detached mode via `nohup` when no measurement process exists;
  - fail-fast avoid duplicate launch when another `measure_wallclock.py` process exists;
  - expose readable status/log output.
- Formal phase must complete all 11 scale points; if a required topology cannot be generated, script must fail fast with actionable stderr evidence.

## 3. Test Results and Evidence

| Test / Command | Result | Details |
|----------------|--------|---------|
| `unittest discover` | PASS | `Ran 8 tests ... OK` |
| sensitivity dry-run | PASS | Enumerated 6 configs; no execution side effects |
| formal dry-run without sensitivity data | PASS (expected fail-fast) | Raised `ValueError` requiring measured sensitivity CSV rows |
| plot CLI with empty formal CSV | PASS (expected fail-fast) | Raised `ValueError: CSV is empty` |
| queue shell syntax check | PASS | `bash -n` exit code 0 |
| queue `start` under existing measurement | PASS (expected skip) | Detected running `measure_wallclock.py`, skipped launching duplicate queue |
| queue `status` | PASS | Reported `SKIPPED existing_measure_process` with runtime file paths |
| formal real run to finish missing point | FAIL-FAST (blocking issue) | 8192 topology generation rejected by template capacity guard |
| topology template probe (8192) | PASS | `AlibabaHPN` succeeds; `Spectrum-X` and `DCN+` fail with capacity error |
| unit tests after topology resolver patch | PASS | `Ran 9 tests ... OK` |
| detached formal completion with `AlibabaHPN` | PASS | 8192 row written and plotting command executed in same detached pipeline |

### Key Output Excerpts

- Unit tests:
  - `Ran 8 tests in 0.856s`
  - `OK`
- Sensitivity dry-run:
  - `[phase] sensitivity configs=6`
  - listed configs: `(8,pp=1)`, `(16,pp=1/2)`, `(32,pp=1/2/4)`
- Formal dry-run fail-fast:
  - `ValueError: Sensitivity CSV is required before formal phase ... Run with --phase sensitivity first.`
- Plot fail-fast on empty CSV:
  - `ValueError: CSV is empty: .../wallclock_scaling.csv`
- Queue duplicate-guard:
  - `[start] detected existing measure_wallclock.py process; skip launching a second queue.`
  - `[status] state_file=... SKIPPED existing_measure_process`
- Formal blocker (8192):
  - `ValueError: Number of GPU exceeds the capacity of Rail_Optimized_SingleToR(One Pod)`
  - Emitted from `gen_Topo_Template.py` with `-topo Spectrum-X -g 8192 -gps 8 ...`
- Template probe:
  - `Spectrum-X rc=1` (capacity error)
  - `DCN+ rc=1` (capacity error)
  - `AlibabaHPN rc=0` (topology file generated)
- Post-fix unit run:
  - `test_ensure_topology_file_accepts_template_with_extra_tokens ... ok`
  - `Ran 9 tests in 0.618s`

## 4. Failure Handling

- No unexpected test failures occurred.
- Two intentional fail-fast scenarios were executed and validated:
  - formal stage without measured sensitivity rows
  - plotting stage without measured formal rows

## 5. Full Measurement Validation (Final)

Final execution command used for completion:

```bash
setsid -f bash -lc "python3 SimAI/tests/performance/simai_wallclock_scaling/measure_wallclock.py --simai-root SimAI --phase formal --topology-template AlibabaHPN && python3 SimAI/tests/performance/simai_wallclock_scaling/plot_wallclock.py" \
  > SimAI/tests/performance/simai_wallclock_scaling/results/runtime/formal_8192_and_plot.log 2>&1
```

Final status:
- Sensitivity CSV complete: 6 measured rows.
- Formal CSV complete: 11 measured rows including `8192,8,8,128,46642.760256`.
- Plot artifacts generated:
  - `SimAI/tests/performance/simai_wallclock_scaling/results/wallclock_scaling.png`
  - `SimAI/tests/performance/simai_wallclock_scaling/results/wallclock_sensitivity.png`

Key completion evidence from log:
- `[done] total_gpus=8192 tp=8 pp=8 dp=128 wallclock_seconds=46642.760256`
- `[plot] scaling png: .../wallclock_scaling.png`
- `[plot] sensitivity png: .../wallclock_sensitivity.png`

## 6. 2026-03-02 Continuation Note (No Redundant Test Re-run)

- Per user instruction, no duplicate unit/integration suites were re-run in this checkpoint.
- Only execution-state verification commands were used:
  - `ps -ef | grep -E 'measure_wallclock.py|ns3.36.1-AstraS' | grep -v grep`
  - `wc -l .../wallclock_scaling.csv` and `tail .../wallclock_scaling.csv`
- Detached runtime command started:
  - `setsid -f bash -lc "python3 ... --phase formal --topology-template AlibabaHPN && python3 .../plot_wallclock.py" > .../formal_8192_and_plot.log 2>&1`
- Evidence:
  - Active PIDs observed: measure `1908582`, NS3 `1908856`
  - Runtime log path: `SimAI/tests/performance/simai_wallclock_scaling/results/runtime/formal_8192_and_plot.log`

## 7. 2026-03-03 Completion Check

- Process check:
  - `ps -ef | grep -E 'measure_wallclock.py|ns3.36.1-AstraS|plot_wallclock.py' | grep -v grep`
  - Output empty (all long-running jobs finished).
- CSV check:
  - `wc -l .../wallclock_scaling.csv` -> `12` (header + 11 rows)
  - includes 8192 row with measured wall-clock seconds.
- Artifact check:
  - `ls -l SimAI/tests/performance/simai_wallclock_scaling/results/wallclock_*.png`
  - both expected PNG files exist.

## 8. 2026-03-03 New-plan Rerun Startup (Isolated Outputs)

- User request scope:
  - Reuse existing `<1024` data.
  - Rerun only `1024/2048/4096/8192` with updated `PP=16`.
  - Run in background without overwriting existing records.
- Execution command (detached):
  - `setsid -f bash -lc "SimAI/tests/performance/simai_wallclock_scaling/results/new_plan_pp16_ge1024_2026-03-03/run_new_plan.sh" > SimAI/tests/performance/simai_wallclock_scaling/results/new_plan_pp16_ge1024_2026-03-03/run.log 2>&1`
- Isolation evidence:
  - New output directory:
    - `SimAI/tests/performance/simai_wallclock_scaling/results/new_plan_pp16_ge1024_2026-03-03/`
  - Existing historical CSV (`results/wallclock_scaling.csv`) remains unchanged.
- Runtime evidence:
  - Process observed:
    - `bash .../run_new_plan.sh`
    - NS3 process on world_size=1024, tp=8, pp=16

## 9. 2026-03-03 New-plan Runner Hotfix

- Issue:
  - New-plan runner contained malformed print expression for `run_summary.json` path.
- Fix:
  - Updated `run_new_plan.sh` print line to use `results_dir / 'run_summary.json'`.
- Safeguard:
  - Started detached watcher process to rerun runner with `resume=True` after initial worker exits, so already completed rows are skipped and plotting still completes.

## 10. 2026-03-04 New-plan Completion Check

- Process check:
  - `ps -ef | grep -E 'run_new_plan.sh|measure_wallclock.py|ns3.36.1-AstraS|plot_wallclock.py' | grep -v grep`
  - Output empty (new-plan background tasks finished).
- New-plan CSV check:
  - `wc -l .../wallclock_scaling_newplan_pp16_ge1024.csv` -> `12` (header + 11 rows)
  - `>=1024` rows present with `PP=16` and computed `DP={8,16,32,64}`.
- New-plan plot check:
  - `ls -l .../wallclock_scaling_newplan_pp16_ge1024.png .../wallclock_sensitivity_reused.png`
  - both PNG outputs exist.
- New-plan log evidence:
  - `[done] total_gpus=1024 tp=8 pp=16 dp=8 wallclock_seconds=3459.977473`
  - `[done] total_gpus=2048 tp=8 pp=16 dp=16 wallclock_seconds=7196.982990`
  - `[done] total_gpus=4096 tp=8 pp=16 dp=32 wallclock_seconds=14806.385846`
  - `[done] total_gpus=8192 tp=8 pp=16 dp=64 wallclock_seconds=34282.179494`
  - `[plot] scaling png: .../wallclock_scaling_newplan_pp16_ge1024.png`
