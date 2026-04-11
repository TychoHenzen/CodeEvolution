# Algorithm Audit Fixes + Impact-Weighted Selection -- Requirements Spec

> **For Claude:** This spec was produced by /interview. Use /writing-plans to expand into an implementation plan, or /tdd to implement directly.

**Goal:** Fix findings 1-5 from the algorithm audit to align CodeEvolution with the spirit of AlphaEvolve/OpenEvolve/science-codeevolve, and add impact-weighted file selection for intelligent prioritization of a 300-file workspace.

**Date:** 2026-04-11

---

## Requirements

### Fix 1: Fitness Metric Normalization (Critical)

**Problem:** The evaluator returns 10 metrics to OpenEvolve including raw counts (`tests_passed=42`, `clippy_warnings=3`). OpenEvolve's `get_fitness_score()` averages ALL numeric metrics. Raw counts dominate normalized scores, and `clippy_warnings` is an inverted metric that increases fitness when warnings increase.

**Solution:** Convert all evaluator metrics to consistent ratio-to-baseline representation:
- `>1` means improvement over original, `<1` means regression, `1.0` is baseline
- Metrics naturally bounded [0,1] (static_score, llm_score) are halved to normalize around 0.5
- No raw counts in the metrics dict returned to OpenEvolve
- Raw counts continue to be logged to CSV for human review

**Files affected:**
- `codeevolve/evaluator/pipeline.py` — change what `_evaluate_candidate()` returns in `EvaluationResult`
- `codeevolve/templates/evaluator.py.j2` — change the metrics dict passed to OpenEvolve

**Specifics:**
- `static_score`: currently `1/(1+penalty)` bounded [0,1]. Halve it → baseline ~0.5
- `perf_score`: already a ratio (baseline=1.0). Keep as-is
- `llm_score`: currently [0,1]. Halve it → baseline ~0.5
- `compile_time`: already a ratio. Keep as-is
- `binary_size`: already a ratio. Keep as-is
- `loc`: already a ratio. Keep as-is
- `tests_passed`, `clippy_warnings`, `passed_gates`: log to CSV only, do NOT include in metrics dict
- `combined_score`: recompute as weighted average of the above ratio-form values

### Fix 2: Code-Fitness Mismatch from LLM Fixer (Major)

**Problem:** When the fixer repairs broken code, OpenEvolve stores the original broken code but the fixed code's score. Selection uses broken code as parent → confused LLM.

**Solution:** After a successful fix, write the repaired EVOLVE-BLOCK content back to OpenEvolve's `program_path` temp file. This way `iteration.py` reads the corrected version when creating the `Program` object.

**Files affected:**
- `codeevolve/evaluator/pipeline.py` — `evaluate()` method needs to write fixed code back to `program_path`

**Specifics:**
- After all fix attempts succeed and Layer 1 passes, read the current (fixed) focus file content
- If the content differs from what was originally written (i.e., fixer changed it), write the fixed content back to `program_path` in the same format OpenEvolve expects (raw code for single-file, bundle for workspace mode)
- The `program_path` parameter is already available in `evaluate()`

### Fix 3: Artifact Feedback Channel (Major)

**Problem:** OpenEvolve has a full artifact API (evaluator stores artifacts → prompt sampler renders them into `{artifacts}` placeholder). CodeEvolution's evaluator returns only numeric metrics, leaving the artifact channel empty.

**Solution:** Return clippy diagnostics and test failure output as artifacts via OpenEvolve's evaluator artifact mechanism.

**Files affected:**
- `codeevolve/templates/evaluator.py.j2` — return artifacts alongside metrics
- `codeevolve/evaluator/pipeline.py` — collect artifact data during evaluation

**Specifics:**
- After Layer 1 (clippy + test), collect:
  - Clippy warning messages (the human-readable diagnostic text, not just counts)
  - Test failure output (stdout/stderr from failed tests)
  - Failing test source code (already collected via `_get_failing_test_context()`)
- Return these as a dict of artifacts alongside the metrics dict
- The evaluator.py.j2 template must return them in a format OpenEvolve's `Evaluator` can store
- OpenEvolve's evaluator expects `EvaluationResult.artifacts` — check if the evaluate function can return this or if we need to use a different mechanism
- Artifacts should be truncated to stay under OpenEvolve's 20KB limit
- Include artifacts even for passing candidates (clippy suggestions are valuable even when code compiles)

### Fix 4: Enable Meta-Prompting via programs_as_changes_description (Moderate)

**Problem:** Static system prompt limits evolutionary search to one framing. AlphaEvolve and science-codeevolve co-evolve prompts.

**Solution:** Enable OpenEvolve's `programs_as_changes_description` mode, which requires `diff_based_evolution=True`.

**Files affected:**
- `codeevolve/config.py` — add config fields and pass through to `to_openevolve_dict()`

**Specifics:**
- Add `changes_description: bool = False` to `EvolutionConfig` (opt-in, since it requires diff mode)
- When enabled, set in `to_openevolve_dict()`:
  - `diff_based_evolution: True` (forced)
  - `prompt.programs_as_changes_description: True`
  - `prompt.initial_changes_description: "<sensible default describing the initial code>"`
- The initial_changes_description should describe what the EVOLVE-BLOCK code does (could be auto-generated from the summary or a static default like "Initial implementation of the evolvable section")
- Validation: if `changes_description=True` but `diff_based_evolution=False`, warn and force diff mode

### Fix 5: Expose Exploration/Exploitation Ratios (Moderate)

**Problem:** OpenEvolve's parent selection ratios and temperature are not configurable from CodeEvolution's YAML.

**Solution:** Add config fields and pass through to `to_openevolve_dict()`.

**Files affected:**
- `codeevolve/config.py` — add fields to `EvolutionConfig`, pass through to OpenEvolve config dict
- `codeevolve/defaults/evolution.yaml` — add default values

**Specifics:**
- Add to `EvolutionConfig`:
  - `exploration_ratio: float = 0.2`
  - `exploitation_ratio: float = 0.7`
  - `temperature: float = 0.7`
- Pass through in `to_openevolve_dict()`:
  - `database.exploration_ratio`
  - `database.exploitation_ratio`
  - `llm.temperature`
- Document in defaults/evolution.yaml with comments explaining what each does

### Feature 6: Impact-Weighted File Selection

**Problem:** Current file selection uses tech debt score alone (`combined_score` from TECH_DEBT_LEDGER.md). A high-debt file with no dependents is less valuable to evolve than a moderately-messy core file imported by 50 others.

**Solution:** Weight priority by `priority = debt_score * (1 + reverse_dep_count)`. Build an intra-crate import graph via regex scanning.

**Files affected:**
- New: `codeevolve/import_graph.py` — scan .rs files for `use crate::`, `use super::`, `mod` statements to build reverse dependency counts
- `codeevolve/cli.py` — integrate import graph into file selection
- `codeevolve/scheduler.py` — accept weighted entries (already works since it uses LedgerEntry.combined_score)
- `codeevolve/ledger.py` — potentially adjust LedgerEntry or create weighted entries

**Specifics:**

**Import graph construction (`import_graph.py`):**
- Scan all .rs files in the project for:
  - `use crate::module::item` → file `src/module.rs` or `src/module/mod.rs` is imported
  - `use super::item` → parent module file is imported
  - `mod module_name;` → child module file is imported
- Map each import to the target .rs file path
- Build a reverse dependency count: for each file, how many other files import from it
- Cross-crate impact: use `crate_graph.deps` to count how many crates depend on the crate containing each file. Add this to the file's reverse dep count.
- Return `Dict[str, int]` mapping file path → total reverse dependency count

**Priority formula:**
```python
priority = ledger_entry.combined_score * (1 + reverse_dep_count)
```
- File with debt=30, imported by 20 files → priority = 30 * 21 = 630
- File with debt=80, imported by 0 files → priority = 80 * 1 = 80
- Core library files naturally float to the top

**Integration:**
- After parsing the ledger, compute import graph for the project
- Create new weighted LedgerEntry objects with adjusted scores
- Feed these to `build_schedule()` which already allocates proportionally

**Performance:**
- Regex scanning 300 files should complete in <1 second
- Cache result for the duration of the run (import graph doesn't change during evolution)

## Subtask Checklist

- [ ] Subtask 1: Fix evaluator metrics normalization — change `_evaluate_candidate()` return values to ratio-to-baseline form, update `evaluator.py.j2` metrics dict to exclude raw counts
- [ ] Subtask 2: Fix code-fitness mismatch — after successful LLM fix in `evaluate()`, write fixed code back to `program_path` in the format OpenEvolve expects
- [ ] Subtask 3: Add artifact feedback — collect clippy diagnostics and test failure output during evaluation, return as artifacts that OpenEvolve renders into prompts
- [ ] Subtask 4: Enable `programs_as_changes_description` — add config field, pass through to OpenEvolve config, force diff mode when enabled
- [ ] Subtask 5: Expose exploration/exploitation ratios — add `exploration_ratio`, `exploitation_ratio`, `temperature` to config and pass through
- [ ] Subtask 6: Build import graph module — scan .rs files for use/mod statements, compute reverse dependency counts per file
- [ ] Subtask 7: Integrate impact-weighted scoring into file selection — multiply debt score by `(1 + reverse_dep_count)`, feed weighted entries to scheduler
- [ ] Subtask 8: Update tests — add/update tests for all changed modules

## Research Notes

**OpenEvolve's `get_fitness_score()` (`utils/metrics_utils.py`):**
- Averages ALL numeric metrics excluding feature dimensions
- No special handling of `combined_score` for selection (only for best-program tracking)
- Feature dimensions `["complexity", "diversity"]` are computed by the database from code properties, NOT from evaluator metrics

**OpenEvolve artifact API (`evaluator.py`):**
- `_pending_artifacts: Dict[str, Dict[str, Union[str, bytes]]]`
- Stored via `EvaluationResult.artifacts` from evaluate function return
- Retrieved via `evaluator.get_pending_artifacts(program_id)` (one-time pop)
- Rendered into prompt via `{artifacts}` placeholder, truncated to 20KB
- The evaluate function in evaluator.py.j2 currently returns `dict` — OpenEvolve wraps this in EvaluationResult internally. To return artifacts, need to check if the evaluate function can return a tuple or if artifacts need to go through a different channel.

**OpenEvolve `programs_as_changes_description` (`config.py`):**
- Requires `diff_based_evolution=True`
- Config keys: `prompt.programs_as_changes_description`, `prompt.initial_changes_description`
- Validation in OpenEvolve: raises ValueError if enabled without diff mode

**OpenEvolve exploration/exploitation (`database.py`):**
- `database.exploration_ratio: float = 0.2`
- `database.exploitation_ratio: float = 0.7`
- Remaining 0.1 is random sampling
- Used in `_sample_parent()` with `random.random()` threshold

**Current evaluator return format:**
- `evaluate(program_path: str) -> dict` — returns metrics dict
- OpenEvolve's Evaluator wraps this in EvaluationResult internally
- Artifacts must be returned as a separate key or via EvaluationResult

**Crate graph (`crate_graph.py`):**
- `CrateGraph.deps: dict[str, list[str]]` — crate_name → local dependency names
- `CrateGraph.crate_roots: dict[str, Path]` — crate_name → root path
- `relevant_crates(name)` returns crate + direct deps (one hop)
- `crate_for_file(path)` determines which crate a file belongs to

## Open Questions

- How exactly does OpenEvolve's evaluator handle artifacts returned from the evaluate function? Need to verify whether the evaluate function can return `{"metrics": {...}, "artifacts": {...}}` or if it must be a flat metrics dict. (Research during implementation)
- Should `programs_as_changes_description` be enabled by default or opt-in? (Decided: opt-in via config flag, since it forces diff mode)
