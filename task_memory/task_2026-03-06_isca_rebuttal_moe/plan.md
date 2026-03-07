## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-06 | Initialized ISCA rebuttal MoE task plan |
| 2026-03-06 | Completed routing-skew validation and rebuttal-text integration |

# Plan

## Goal
Implement MoE-focused rebuttal updates for the ISCA 2026 paper draft by adding a minimal but convincing routing-skew validation path, updating paper text, and storing reproducible evidence under this task directory.

## Scope
- Add the smallest code support needed to run a controlled MoE routing-skew experiment.
- Generate a concise experiment report with reproducible commands and measured outputs.
- Revise the rebuttal paper draft to directly answer MoE-related reviewer concerns.

## Steps
- [x] Audit current MoE routing injection points and baseline evidence.
- [x] Add a failing test for routing-skew control parsing / generation.
- [x] Implement controlled routing-skew support for MoE tracing.
- [x] Run focused validation and record the results.
- [x] Update rebuttal text in the paper draft.
- [x] Summarize evidence and remaining limitations.

## Acceptance Criteria
- A reproducible routing-skew experiment path exists in-repo.
- Paper text explicitly addresses modern MoE generalization, MoE all-to-all modeling, routing skew, and capacity-factor scope.
- A Markdown test report under this task directory records commands, environment, and outcomes.

## Status
**Completed** — routing-skew support, paired validation, paper-text patches, and evidence capture are all in place.
