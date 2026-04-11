# Checkpoint/Resume & Weighted Focus Rotation — Requirements Spec

> **For Claude:** This spec was produced by /interview. Use /writing-plans to expand into an implementation plan, or /tdd to implement directly.

**Goal:** Add checkpoint/resume support and tech-debt-weighted focus rotation so evolution runs survive crashes and prioritize the worst files.

**Date:** 2026-04-11

---

## Requirements

### Checkpointing

- Save checkpoint every 10 iterations (configurable via `checkpoint_interval`)
- Checkpoint includes: OpenEvolve population database, best program per file, current rotation schedule state (which file is being focused, iteration within that file's allocation)
- Checkpoints written to `.codeevolve/output/checkpoints/checkpoint_{N}/`
- On KeyboardInterrupt, attempt to save a checkpoint before exiting

### Auto-Resume

- On `codeevolve run`, auto-detect the latest valid checkpoint in `.codeevolve/output/checkpoints/`
- Resume from that checkpoint: restore iteration count, focus file position in rotation schedule, and start a new OpenEvolve run for the current focus file from that checkpoint's population
- Add `--fresh` CLI flag to force a clean start, ignoring existing checkpoints
- If checkpoint is missing or corrupt, log a warning and start fresh (no crash)

### Tech-Debt-Weighted File Selection

- New config option `tech_debt_ledger: "path/to/TECH_DEBT_LEDGER.md"` (empty string = disabled)
- Parse the ledger's markdown table to extract file paths, type (prod/test), and combined scores
- Filter to `prod` files only (`prod_only: true` config, default true)
- Take top N files by combined score (`top_n_files: 20` config, default 20)
- Allocate iterations proportional to each file's debt score
  - Example: 500 iterations, file A (score 57.7) and file B (score 15.0) → A gets ~4x more iterations than B
- Files in ledger that don't exist on disk or lack EVOLVE-BLOCK markers → skip, redistribute their iterations

### Focus Rotation

- Rotation happens every 10 iterations (aligned with checkpoint interval)
- The rotation schedule is a list of (file, num_iterations) pairs computed upfront from the weighted allocation
- Each focus rotation starts a **fresh OpenEvolve population** for that file (no shared state between files)
- Best result from each file's evolution saved to `.codeevolve/output/best/<filename>`
- Schedule is deterministic: same config + ledger = same schedule

### Fallback Behavior

- If `tech_debt_ledger` is empty or file not found → fall back to round-robin across all EVOLVE-BLOCK-marked files
- Fallback round-robin distributes iterations equally, rotating every 10 iterations

### Controller Change

- Must use `openevolve.controller.OpenEvolve` directly instead of `openevolve.api.run_evolution()` to access the `checkpoint_path` parameter on `controller.run()`
- This gives full control over checkpoint save/load and iteration management

## What This Does NOT Do

- No evolving test files (prod only by default)
- No shared population across file rotations (fresh run per file)
- No smart/adaptive rotation (purely score-proportional schedule, computed once upfront)
- No interactive resume prompt (fully automatic)
- No changes to the 4-layer evaluation pipeline

## Config Additions (evolution.yaml)

```yaml
evolution:
  checkpoint_interval: 10    # iterations between checkpoints (default: 10)
  tech_debt_ledger: ""        # path to TECH_DEBT_LEDGER.md relative to project root (empty = round-robin)
  top_n_files: 20             # how many worst prod files to evolve (default: 20)
  prod_only: true             # filter ledger to prod files only (default: true)
```

## CLI Additions

```
codeevolve run --config .codeevolve/evolution.yaml --fresh   # ignore checkpoints, start clean
```

## Subtask Checklist

- [ ] Subtask 1: **Parse tech debt ledger** — Add `codeevolve/ledger.py` with `parse_ledger(path: Path) -> list[LedgerEntry]` that reads TECH_DEBT_LEDGER.md, parses the markdown table, returns list of (file_path, type, combined_score) entries. Filter by `prod_only`. Handle missing/malformed file gracefully.

- [ ] Subtask 2: **Compute rotation schedule** — Add `codeevolve/scheduler.py` with `build_schedule(entries: list[LedgerEntry], total_iterations: int, chunk_size: int = 10) -> list[ScheduleSlot]` that allocates iterations proportional to debt scores. Each `ScheduleSlot` is `(file: Path, start_iter: int, end_iter: int)`. Ensure minimum 10 iterations per file. Round to chunk_size boundaries.

- [ ] Subtask 3: **Add config fields** — Extend `CodeEvolveConfig` in `config.py` with `checkpoint_interval`, `tech_debt_ledger`, `top_n_files`, `prod_only`. Update `defaults/evolution.yaml`. Ensure backward compatibility (missing keys get defaults).

- [ ] Subtask 4: **Add `--fresh` CLI flag** — Add `--fresh` option to `codeevolve run` in `cli.py`. When set, delete existing checkpoints before starting.

- [ ] Subtask 5: **Refactor runner to use OpenEvolve controller directly** — Replace `openevolve.api.run_evolution` calls in `runner.py` with direct `OpenEvolve` controller instantiation. This is needed to pass `checkpoint_path` to `controller.run()`. Preserve existing single-file and multi-file behavior.

- [ ] Subtask 6: **Implement checkpoint save on interrupt** — In `cli.py`'s KeyboardInterrupt handler, trigger a checkpoint save before stopping the backend. Save current rotation position alongside OpenEvolve's checkpoint data.

- [ ] Subtask 7: **Implement auto-resume logic** — In `runner.py` or `cli.py`, before starting evolution: check for latest checkpoint in `.codeevolve/output/checkpoints/`, load rotation schedule state, pass `checkpoint_path` to controller. Skip if `--fresh` is set.

- [ ] Subtask 8: **Implement rotation loop** — Replace single `run_evolution` call with a loop that iterates through the schedule. Each slot: set up the focus file, create a fresh OpenEvolve controller, run for that slot's iteration count, save best result. Checkpoint at each rotation boundary.

- [ ] Subtask 9: **Wire fallback round-robin** — When no ledger is configured or ledger is missing, build a simple round-robin schedule across all marked files with equal iteration allocation.

- [ ] Subtask 10: **Integration tests** — Test checkpoint save/load round-trip, schedule computation from a mock ledger, resume from checkpoint with correct focus file, --fresh flag clears checkpoints, graceful fallback when ledger is missing.

## Research Notes

### OpenEvolve Checkpoint Infrastructure (already exists)

- `controller.py:_save_checkpoint(iteration)` → writes to `{output_dir}/checkpoints/checkpoint_{N}/`
- `controller.py:_load_checkpoint(path)` → calls `database.load(path)`
- `controller.run(checkpoint_path=...)` → resumes from checkpoint, sets `start_iteration = database.last_iteration + 1`
- `database.save(path, iteration)` → saves all programs + metadata (islands, best_program_id, last_iteration)
- `database.load(path)` → restores full state including island topology
- `config.checkpoint_interval` defaults to 100
- Checkpoints are triggered in `process_parallel.py` line 676: `completed_iteration % config.checkpoint_interval == 0`

### Current Runner Flow

- `runner.py:run_evolution()` dispatches to `_run_single_file()` or `_run_multi_file()`
- Both call `openevolve.api.run_evolution()` which creates `OpenEvolve` controller internally
- API does not expose `checkpoint_path` — must use controller directly
- Both paths backup/restore source files in try/finally blocks

### Tech Debt Ledger Format

Located at project root (e.g., `D:\rust-target\axiom2d\TECH_DEBT_LEDGER.md`). Markdown table with columns:
```
| File Path | Type | Structural | Semantic | Combined | Top Issue | Last Reviewed | Trend |
```
- Type is `prod` or `test`
- Combined is a float (higher = worse)
- 390 files total (234 prod, 156 test)
- Top prod file: splash/render.rs at 57.70

### Key Files

| File | Role |
|------|------|
| `codeevolve/runner.py` | Evolution orchestrator — main refactor target |
| `codeevolve/cli.py` | CLI entry point — add --fresh, auto-resume |
| `codeevolve/config.py` | Config dataclass — add new fields |
| `codeevolve/defaults/evolution.yaml` | Default config — add new defaults |
| `openevolve/controller.py` (vendored) | Has checkpoint save/load — use directly |
| `openevolve/database.py` (vendored) | Population persistence — already works |

## Open Questions

- None — all requirements confirmed by user.
