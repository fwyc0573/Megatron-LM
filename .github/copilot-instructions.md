# Agent Instruction Rules

## 1. Salutation

Every reply **must begin** with:

```
Yes, boss Yicheng!
```

## 2. Language & Output

- **Responses**: 清晰易懂的中文
- **Code/Comments**: Standard, formal English
- **Technical Terms**: Retain in English (e.g., token, prompt, prefill, decode, inference, reasoning)

## 3. Context & Quality

- Base answers on deep understanding of the entire codebase, including `README.md`, `task_memory/`, `AGENTS.md`, and all source files
- Use MCP tools (`serena`) for retrieval and search
- Provide direct, effective, professional solutions—no fluff
- When renaming variables or making changes, update **all** related files and references

## 4. Code Style

- Follow existing project conventions (indentation, naming, comments, file layout)
- **Reference snippets**: Copy verbatim—no edits
- **Long snippets**: Trim to core logic for clarity

## 5. Formatting

- Use Markdown headings, bullets, numbered lists for readability
- Keep explanations concise yet complete
- Remain courteous and solution-oriented; state assumptions if uncertain

## 6. Project Structure

- Place test/experimental scripts in `tests/` directory (with subdirectories: `unit/`, `integration/`, `performance/`, `e2e/`, etc.)
- Store temporary docs or drafts in `tests/` or appropriate subdirectories—**never** in project root

## 7. Error Handling

- **No Fallbacks**: Do not implement fallback logic—it obscures bugs
- **Fail Fast**: Raise errors explicitly for unexpected conditions

## 8. Testing and Validation

For any modification, addition, or deletion of code modules, you **must** perform comprehensive testing that fully covers the changes.

### What "Comprehensive Testing" Means

- **Coverage of modified code paths**: All changed functions, methods, and branches must be exercised
- **Edge cases**: Include boundary conditions, empty inputs, error conditions
- **Regression prevention**: Ensure existing functionality remains unaffected

### Test Type Selection

| Test Type | When to Use |
|-----------|-------------|
| **Unit tests** | Testing individual functions/methods in isolation; logic-heavy code; utility functions |
| **Integration tests** | Testing interactions between modules; API contracts; data flow between components |
| **End-to-end (e2e) tests** | Testing complete workflows; user-facing scenarios; system-level behavior |

**Rule**: Include unit tests for all logic changes. Add e2e tests when changes affect system behavior or cross-module interactions.

### Test Environment Prerequisites

Before running tests, ensure:

1. Correct environment is activated (conda, venv, etc.)
2. Required dependencies are installed
3. Environment variables are set (e.g., `PYTHONPATH`)
4. Test data/fixtures are available

### Test Report Format

After running tests, generate a **Markdown-formatted** report containing:

**1. Test Script Information**

- Full path to test script(s)
- Exact command to run (must be reproducible)
- Environment used (conda env name, Python version)

**2. Validation Criteria**

- Which metrics/outputs to monitor
- Expected results or acceptance criteria
- Specific assertions or conditions that must pass

**3. Test Results and Evidence**

- Summary of test outcomes (PASS/FAIL)
- Key outputs or log excerpts demonstrating correctness
- For failures: error messages, stack traces, relevant context

### Handling Test Failures

When tests fail:

1. **Do NOT proceed** with the change until tests pass
2. **Diagnose** the root cause (code bug vs. test bug vs. environment issue)
3. **Fix and re-run** all affected tests
4. **Document** the failure and resolution in the test report

### Test Report Storage

- Store test reports in `task_memory/<task_dir>/` alongside other task documentation
- Name format: `test_report_[YYYY-MM-DD]_[brief_description].md`
- For quick iterations, append results to existing `progress.md`

### Example Test Report

```markdown
## Test Report: User Authentication Module

**Date**: 2026-01-25  
**Environment**: conda activate myproject_env (Python 3.10)

### Test Script Information
- Script: `tests/unit/test_auth.py`, `tests/e2e/test_login_flow.sh`
- Commands:
  ```bash
  pytest tests/unit/test_auth.py -v
  bash tests/e2e/test_login_flow.sh
  ```

### Validation Criteria

- All unit tests pass (15 test cases)
- Login flow completes within 2 seconds
- Invalid credentials return 401 status

### Test Results

| Test Suite | Result | Details |
|------------|--------|---------|
| Unit tests | PASS | 15/15 passed |
| E2E login  | PASS | Completed in 1.2s |

### Evidence

- Exit code: 0
- Log excerpt: "Authentication successful for user_id=123"

```

## 9. Documentation

- Update related content within the **same file**—do not create duplicate files
- Maintain modification history at the top of each doc:

```markdown
## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-01-30 | Renamed task_memory to task_memory for MCP tool compatibility |
| YYYY-MM-DD | Brief description of what was modified      |
```

## 10. Communication Protocol

### Critical Modifications

For code logic changes not covered in instructions, **obtain explicit agreement** before implementing—such changes may impact subsequent development.

### Blocking Issues

If requirements CANNOT be met (missing dependencies, incompatible environments, etc.):

1. **STOP** all task execution immediately
2. **Report** to me with:
   - Which requirement cannot be met
   - Root cause analysis
   - Impact on timeline and deliverables
   - Proposed alternatives or request for guidance
3. **Wait** for explicit approval before proceeding

### Stop Response Rule

At every conversational stop point (including "continue" checkpoints), you must include:

1. A concise list of all pending tasks
2. Newly discovered issues from the current execution
3. Numbered recommended next steps (actionable, ordered)

## 11. Task Management

For complex tasks involving:

- Extensive cross-file context analysis
- Complex engineering implementation
- Multi-step coordinated workflows

Use the **Planning with Files** skill:

### Organization

- Store all tasks under `task_memory/`
- Create subdirectory per task: `task_[YYYY-MM-DD]_[brief_description]`
- For each task, maintain the following docs:
  - plan.md (scope, steps, acceptance criteria)
  - notes.md (requirements, constraints, special configuration)
  - progress.md (ongoing updates and decisions)
  - issues.md (open questions, blockers, and resolutions)
  - If the task grows complex, add additional focused docs as needed (e.g., design.md, experiments.md, results.md).
- If you’re still working within the same task, do not create a new task directory—keep all updates in the existing task folder and revise the docs in place

### Continuation Protocol

When resuming an existing task:

1. Locate the existing task directory
2. Read progress files to understand current state
3. Identify the interruption point
4. Resume from that position—avoid redundant work

### Progress Tracking

- Maintain progress files with clear status indicators (completed/in-progress/pending)
- Update documentation as work proceeds

## 12. Skills Playbook (When to Invoke Which Skill)

The goal of this section is to enable the agent to automatically select and invoke the correct skill at the "right time", avoiding haphazard actions based on intuition (especially in debugging, TDD, and multi-step implementations).

### 12.1 Invocation Rule

1. **Choose the method before acting**: Before any substantive action (reading code, changing code, running commands, writing plans, or concluding), decide whether a skill is triggered.
2. **If triggered, use it**: If a trigger condition matches (even with ~1% probability), invoke the skill first and then follow its workflow.
3. **Priority order**: Prefer "process/methodology" skills (decide HOW) before "execution" skills (guide DO).

### 12.2 Skill Map (Trigger → Action)

#### Meta / Guardrails

1. **using-superpowers**
   - Trigger: At the start of any new conversation or new task (even for clarifying questions).
   - Purpose: Force a "skill applicability check" before replying, to avoid skipping workflow-driven skills.
   - Path: `skills/using-superpowers/SKILL.md`

2. **karpathy-guidelines**
   - Trigger: Any time you write code, do code review, refactor, or change docs in a way that affects workflow/behavior.
   - Purpose: Keep changes surgical, simplicity-first, and goal-driven with verifiable success criteria; avoid over-engineering and hidden assumptions.
   - Path: `skills/karpathy-guidelines/SKILL.md`

3. **find-skills**
   - Trigger: When the user asks "is there a skill for X", wants to discover/install skills, or you suspect a relevant skill may exist.
   - Purpose: Help discover the best-matching skill(s) and the next step to install/use them.
   - Path: `skills/find-skills/SKILL.md`

#### Design / Planning

1. **brainstorming**
   - Trigger: Before any creative work: new feature, behavior change, new functionality, or architecture/component design.
   - Purpose / Output: Clarify intent, constraints, and success criteria; compare 2-3 options; produce an executable design/spec.
   - Path: `skills/brainstorming/SKILL.md`

2. **writing-plans**
   - Trigger: You already have a spec/requirements and the task is multi-step; before touching code.
   - Purpose / Output: Write an implementation plan with task breakdown, exact file paths, test steps, and acceptance criteria.
   - Path: `skills/writing-plans/SKILL.md`

3. **planning-with-files** (introduced in Section 11)
   - Trigger: Complex tasks (cross-file/cross-module, multi-phase verification, needs logging/traceability).
   - Purpose / Output: Manage work under `task_memory/` with plan/notes/progress/issues and follow the continuation protocol.
   - Path: `skills/planning-with-files/SKILL.md`

4. **using-git-worktrees**
   - Trigger: Starting feature work that needs isolation from the current workspace, or before executing a large implementation plan.
   - Purpose / Output: Create an isolated git worktree and verify a clean baseline (tests pass) before implementation.
   - Path: `skills/using-git-worktrees/SKILL.md`

#### Execution (Single vs Parallel)

1. **subagent-driven-development**
   - Trigger: Executing an implementation plan in the current session where tasks are mostly independent and separable.
   - Purpose / Output: Assign each task to a dedicated subagent with review checkpoints to reduce context pollution and rework.
   - Path: `skills/subagent-driven-development/SKILL.md`

2. **executing-plans**
   - Trigger: You have a written implementation plan and want to execute it in "separate session / batch + checkpoint review" mode.
   - Purpose / Output: Execute the plan in batches (default 3 tasks per batch) with a review/report checkpoint after each batch.
   - Path: `skills/executing-plans/SKILL.md`

3. **dispatching-parallel-agents**

- Trigger: There are 2+ independent problem domains (e.g., multiple failing test files with unrelated root causes) that can be investigated/fixed in parallel.
- Purpose / Output: Dispatch parallel agents by problem domain, then integrate and run full regression at the end.
- Path: `skills/dispatching-parallel-agents/SKILL.md`

#### Debug / Quality

1. **systematic-debugging**

- Trigger: Any bug, test failure, unexpected behavior, build failure, or performance anomaly; before proposing a fix.
- Purpose / Output: Establish root cause with evidence, then apply the minimal fix; avoid "symptom patching".
- Path: `skills/systematic-debugging/SKILL.md`

1. **test-driven-development**

- Trigger: Any feature/bugfix/refactor/behavior change; before writing implementation code.
- Purpose / Output: Strict RED → GREEN → REFACTOR, and you must personally observe tests fail first and then pass.
- Path: `skills/test-driven-development/SKILL.md`

1. **verification-before-completion**

- Trigger: Before claiming "fixed/done/passing/ready to merge/ready to deliver".
- Purpose / Output: Provide verification evidence from a real run (command + key output/exit code) before concluding.
- Path: `skills/verification-before-completion/SKILL.md`

#### Code Review / Integration

1. **code-review**

- Trigger: When you have a git commit (or a set of commits) and want a rigorous review of changes and side effects.
- Purpose / Output: Perform a structured review, categorize findings by severity, and highlight risks/regressions and test gaps.
- Path: `skills/code-review/SKILL.md`

1. **requesting-code-review**

- Trigger: Before merging/delivering (task completion, major feature completion), or when you need a "fresh perspective".
- Purpose / Output: Proactively request a review (optionally via a subagent reviewer) to catch issues before merge.
- Path: `skills/requesting-code-review/SKILL.md`

1. **receiving-code-review**

- Trigger: When you receive code review feedback, especially if unclear or technically questionable; before applying suggestions.
- Purpose / Output: Understand and verify each point, implement changes with tests, and push back technically when warranted.
- Path: `skills/receiving-code-review/SKILL.md`

1. **finishing-a-development-branch**

- Trigger: Implementation is complete and tests pass; you need to decide how to integrate (merge/PR/cleanup).
- Purpose / Output: Verify tests, present structured integration options, execute the chosen flow, then clean up.
- Path: `skills/finishing-a-development-branch/SKILL.md`

### 12.3 Recommended Skill Sequences (Common Workflows)

1. **New feature / behavior change (complex)**
   - using-superpowers → brainstorming → writing-plans → planning-with-files → using-git-worktrees → test-driven-development → (subagent-driven-development | executing-plans) → requesting-code-review → verification-before-completion → finishing-a-development-branch

2. **Bugfix / test failure**
   - using-superpowers → systematic-debugging → test-driven-development → verification-before-completion → requesting-code-review (optional)

3. **Multiple independent failures (parallel investigation)**
   - using-superpowers → dispatching-parallel-agents → systematic-debugging (within each domain) → verification-before-completion

## 13. Environment Configuration Rule

If you encounter environment configuration issues (e.g., dependency version incompatibilities, GLIBCXX errors), consult `task_memory/env_handbook.md` for solutions first.
If it's a new issue, after fix it, you should add the solution to the `task_memory/env_handbook.md` brefily and clearly.

## 14. Repo Safety (rm/mv, large changes)

- **Permission Required**: Any `rm` or `mv` operation requires your explicit permission; otherwise it is strictly forbidden.
- **No Bulk Delete/Replace**: Bulk deletion or bulk replacement in active development repos (e.g., `frontier`) is strictly forbidden unless you explicitly allow it.
- **Large Change Gate**: Before any large-scale change, require all existing local modifications to be stashed or committed (including submodules).