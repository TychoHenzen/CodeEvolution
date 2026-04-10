from __future__ import annotations

import hashlib
import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set, Tuple

from codeevolve.config import CodeEvolveConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EVOLVE-BLOCK marker handling
# ---------------------------------------------------------------------------
_MARKER_START = "// EVOLVE-BLOCK-START"
_MARKER_END = "// EVOLVE-BLOCK-END"


def parse_evolve_block(code: str) -> Tuple[str, str, str] | None:
    """Parse code into (prefix, evolve_content, suffix).

    Returns None if markers are not found or malformed.
    """
    start_idx = code.find(_MARKER_START)
    if start_idx == -1:
        return None

    # Find the newline after the start marker
    content_start = code.find("\n", start_idx)
    if content_start == -1:
        return None
    content_start += 1  # skip the newline

    end_idx = code.find(_MARKER_END, content_start)
    if end_idx == -1:
        return None

    # Find the start of the line containing the end marker
    content_end = code.rfind("\n", content_start, end_idx)
    if content_end == -1:
        content_end = end_idx
    else:
        content_end += 1  # include the newline

    prefix = code[:start_idx + len(_MARKER_START)] + "\n"
    evolve_content = code[content_start:content_end]
    suffix = code[end_idx:]

    return prefix, evolve_content, suffix


def splice_evolve_block(prefix: str, new_content: str, suffix: str) -> str:
    """Splice new content between the preserved prefix and suffix."""
    # Ensure content ends with newline before suffix
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"
    return prefix + new_content + suffix


# ---------------------------------------------------------------------------
# Deduplication constants
# ---------------------------------------------------------------------------
# After this many consecutive duplicates, log a warning.  The evaluator
# returns score 0 for each duplicate, so OpenEvolve's early-stopping
# (patience configured in the config) will eventually halt the run.
_MAX_CONSECUTIVE_DUPLICATES_WARN = 10
from codeevolve.evaluator.cargo import (
    compute_clippy_score,
    run_cargo_build,
    run_cargo_clippy,
    run_cargo_test,
)
from codeevolve.evaluator.benchmark import (
    measure_binary_size,
    measure_compile_time,
    measure_loc,
    run_user_benchmark,
)
from codeevolve.evaluator.llm_judge import judge_code
from codeevolve.evaluator.llm_fixer import attempt_fix


@dataclass
class EvaluationResult:
    passed_gates: bool
    combined_score: float
    static_score: float = 0.0
    perf_score: float = 0.0
    llm_score: float = 0.0
    build_time: float = 0.0
    tests_passed: int = 0
    tests_failed: int = 0
    clippy_warnings: int = 0
    binary_size: float = 0.0
    compile_time: float = 0.0
    loc: float = 0.0
    error: str = ""


class EvaluationPipeline:
    """4-layer gated evaluation pipeline for Rust code."""

    def __init__(self, config: CodeEvolveConfig, project_path: Path, source_file: Path):
        self.config = config
        self.project_path = project_path
        self.source_file = source_file
        self._score_history: list[float] = []
        self._baseline_loc: Optional[int] = None
        self._baseline_compile_time: Optional[float] = None
        self._baseline_binary_size: Optional[int] = None
        self._baseline_bench: Optional[float] = None
        # Deduplication state
        self._seen_hashes: Set[str] = set()
        self._original_hash: Optional[str] = None
        self._consecutive_duplicates: int = 0
        self._total_duplicates: int = 0
        # EVOLVE-BLOCK structure (prefix, suffix) from original file
        self._evolve_prefix: Optional[str] = None
        self._evolve_suffix: Optional[str] = None

    def _is_top_quartile(self, pre_llm_score: float) -> bool:
        if len(self._score_history) < 4:
            return False
        threshold = statistics.quantiles(self._score_history, n=4)[2]
        return pre_llm_score >= threshold

    @staticmethod
    def _code_hash(code: str) -> str:
        """Hash normalised code (ignoring trailing whitespace differences)."""
        normalised = code.strip()
        return hashlib.sha256(normalised.encode()).hexdigest()

    def _try_llm_fix(
        self,
        error_type: str,
        error_output: str,
        cfg: CodeEvolveConfig,
        previous_attempts: list[str] | None = None,
        attempt_number: int = 0,
    ) -> bool:
        """Ask the LLM to fix the current code. Returns True if a fix was applied.

        Only sends the EVOLVE-BLOCK content to the LLM, then splices the
        fix back into the original file structure.
        """
        current_code = self.source_file.read_text()

        # Extract only the evolve-block content to send to the LLM
        if self._evolve_prefix is not None:
            parsed = parse_evolve_block(current_code)
            code_to_fix = parsed[1] if parsed else current_code
        else:
            code_to_fix = current_code

        fixed = attempt_fix(
            code_to_fix, error_type, error_output,
            cfg.ollama.api_base, cfg.ollama.evaluator_model,
            previous_attempts=previous_attempts or [],
            attempt_number=attempt_number,
        )
        if not fixed or fixed == code_to_fix:
            return False

        # Extract evolve content from the fix (strip markers if LLM added them)
        fixed_content = self._extract_evolve_content(fixed)

        # Splice back into original structure
        if self._evolve_prefix is not None:
            spliced = splice_evolve_block(self._evolve_prefix, fixed_content, self._evolve_suffix)
        else:
            spliced = fixed_content

        self.source_file.write_text(spliced)
        return True

    def _extract_evolve_content(self, candidate_code: str) -> str:
        """Extract content to use from candidate.

        If the candidate has EVOLVE-BLOCK markers, extract just that content.
        Otherwise, treat the entire candidate as the evolve content.
        This ensures the LLM cannot modify code outside the markers.
        """
        parsed = parse_evolve_block(candidate_code)
        if parsed:
            _, content, _ = parsed
            return content
        # No markers found - use entire candidate as evolve content
        return candidate_code

    def evaluate(self, program_path: str) -> EvaluationResult:
        """Run the full 4-layer evaluation pipeline on a candidate program.

        OpenEvolve passes a temp file path containing the candidate code.
        We extract only the EVOLVE-BLOCK content and splice it into the
        original file structure, preserving everything outside the markers.

        Before running the expensive cargo pipeline, we check whether the
        candidate is identical to the original source or a previously-seen
        candidate.  Duplicates are rejected immediately with score 0.
        """
        # Save the original source so we can restore it after evaluation
        original_code = self.source_file.read_text()

        # Lazily capture the original file structure (prefix/suffix around
        # EVOLVE-BLOCK content) so we can enforce marker boundaries.
        if self._evolve_prefix is None:
            parsed = parse_evolve_block(original_code)
            if parsed:
                self._evolve_prefix, original_content, self._evolve_suffix = parsed
                logger.info(
                    "EVOLVE-BLOCK parsed: prefix=%d chars, content=%d chars, suffix=%d chars",
                    len(self._evolve_prefix), len(original_content), len(self._evolve_suffix),
                )
            else:
                # No markers - treat entire file as evolvable (legacy mode)
                logger.warning(
                    "No EVOLVE-BLOCK markers found in %s - entire file is evolvable",
                    self.source_file.name,
                )

        # Lazily capture the hash of the very first (original) program so we
        # can detect "LLM returned the input verbatim" across all iterations.
        if self._original_hash is None:
            self._original_hash = self._code_hash(original_code)
            self._seen_hashes.add(self._original_hash)

        # Read candidate and extract the evolve content
        raw_candidate = Path(program_path).read_text()
        evolve_content = self._extract_evolve_content(raw_candidate)

        # Splice into original structure if we have markers
        if self._evolve_prefix is not None:
            candidate_code = splice_evolve_block(
                self._evolve_prefix, evolve_content, self._evolve_suffix
            )
        else:
            candidate_code = raw_candidate

        candidate_hash = self._code_hash(candidate_code)

        if candidate_hash in self._seen_hashes:
            self._consecutive_duplicates += 1
            self._total_duplicates += 1
            is_original = candidate_hash == self._original_hash
            reason = "identical to original" if is_original else "duplicate of previous candidate"
            logger.warning(
                "Duplicate candidate rejected (%s) — "
                "%d consecutive, %d total duplicates",
                reason,
                self._consecutive_duplicates,
                self._total_duplicates,
            )
            if self._consecutive_duplicates >= _MAX_CONSECUTIVE_DUPLICATES_WARN:
                logger.error(
                    "WARNING: %d consecutive duplicate candidates. "
                    "The LLM is stuck in a loop. Consider increasing "
                    "temperature or changing the prompt.",
                    self._consecutive_duplicates,
                )
            return EvaluationResult(
                passed_gates=False,
                combined_score=0.0,
                error=f"duplicate: {reason}",
            )

        # Novel candidate — reset consecutive counter, record hash
        self._consecutive_duplicates = 0
        self._seen_hashes.add(candidate_hash)

        # Write spliced code into the actual source file
        self.source_file.write_text(candidate_code)

        try:
            return self._evaluate_candidate()
        finally:
            # Always restore the original source file
            self.source_file.write_text(original_code)

    def _evaluate_candidate(self) -> EvaluationResult:
        """Run the 4-layer pipeline on the candidate already written to disk."""
        project_path = self.project_path
        cargo = self.config.rust.cargo_path
        cfg = self.config

        # --- Layer 1: Hard gates (with LLM fix retries) ---
        max_attempts = cfg.evolution.max_fix_attempts + 1  # initial + retries
        build = None
        test = None
        previous_fix_attempts: list[str] = []

        for attempt in range(max_attempts):
            is_last_attempt = attempt == max_attempts - 1
            attempt_label = f"[attempt {attempt + 1}/{max_attempts}]" if max_attempts > 1 else ""

            build = run_cargo_build(project_path, cargo, cfg.rust.target_dir)
            if not build.success:
                if not is_last_attempt:
                    logger.info(
                        "Layer 1 %s: cargo build failed, asking LLM to fix...",
                        attempt_label,
                    )
                    # Snapshot current code before fix for history
                    pre_fix = self.source_file.read_text()
                    if self._try_llm_fix(
                        "compile", build.error_output, cfg,
                        previous_attempts=previous_fix_attempts,
                        attempt_number=attempt,
                    ):
                        previous_fix_attempts.append(
                            self._extract_evolve_content(pre_fix)
                        )
                        continue
                logger.info("Layer 1 FAIL: cargo build failed (%.1fs)", build.elapsed_seconds)
                return EvaluationResult(
                    passed_gates=False, combined_score=0.0,
                    build_time=build.elapsed_seconds, error=build.error_output,
                )

            test = run_cargo_test(project_path, cargo, cfg.rust.test_args or None)
            if not test.success:
                if not is_last_attempt:
                    logger.info(
                        "Layer 1 %s: cargo test failed (%d passed, %d failed), asking LLM to fix...",
                        attempt_label, test.tests_passed, test.tests_failed,
                    )
                    pre_fix = self.source_file.read_text()
                    if self._try_llm_fix(
                        "pass tests", test.error_output, cfg,
                        previous_attempts=previous_fix_attempts,
                        attempt_number=attempt,
                    ):
                        previous_fix_attempts.append(
                            self._extract_evolve_content(pre_fix)
                        )
                        continue
                logger.info(
                    "Layer 1 FAIL: cargo test failed (%d passed, %d failed)",
                    test.tests_passed, test.tests_failed,
                )
                return EvaluationResult(
                    passed_gates=False, combined_score=0.0,
                    build_time=build.elapsed_seconds,
                    tests_passed=test.tests_passed, tests_failed=test.tests_failed,
                    error=test.error_output,
                )

            # Both build and test passed
            break

        logger.info(
            "Layer 1 PASS: build %.1fs, %d/%d tests passed",
            build.elapsed_seconds, test.tests_passed,
            test.tests_passed + test.tests_failed,
        )

        # --- Layer 2: Static analysis ---
        clippy = run_cargo_clippy(project_path, cargo, cfg.rust.clippy_args or None)
        raw_static = compute_clippy_score(clippy.warning_counts, cfg.fitness.clippy_weights)
        penalty = abs(raw_static)
        norm_static = 1.0 / (1.0 + penalty)
        logger.info(
            "Layer 2: %d clippy warnings %s, penalty=%d, static_score=%.3f",
            len(clippy.warnings),
            dict(clippy.warning_counts) if clippy.warning_counts else "{}",
            penalty, norm_static,
        )

        # --- Layer 3: Performance ---
        compile_time = 0.0
        binary_size = 0
        loc = measure_loc(self.source_file)
        bench_score: Optional[float] = None

        if cfg.benchmarks.measure_compile_time:
            compile_time = measure_compile_time(project_path, cargo)

        if cfg.benchmarks.measure_binary_size:
            binary_size = measure_binary_size(project_path, cfg.rust.target_dir)

        if cfg.benchmarks.custom_command:
            bench_result = run_user_benchmark(
                cfg.benchmarks.custom_command,
                project_path,
                cfg.benchmarks.custom_command_score_regex,
            )
            bench_score = bench_result.score

        # Set baseline on first evaluation
        if self._baseline_loc is None:
            self._baseline_loc = loc
            self._baseline_compile_time = compile_time
            self._baseline_binary_size = binary_size
            self._baseline_bench = bench_score
            logger.info(
                "Layer 3: baseline set - loc=%d, compile=%.2fs, binary=%d bytes",
                loc, compile_time, binary_size,
            )

        # Compute perf as ratio to baseline (>1 = improvement, 1 = same)
        loc_ratio = self._baseline_loc / loc if self._baseline_loc and loc else 1.0
        perf_ratios = [loc_ratio]

        compile_ratio = 1.0
        if cfg.benchmarks.measure_compile_time:
            if self._baseline_compile_time and compile_time:
                compile_ratio = self._baseline_compile_time / compile_time
            perf_ratios.append(compile_ratio)

        binary_ratio = 1.0
        if cfg.benchmarks.measure_binary_size:
            if self._baseline_binary_size and binary_size:
                binary_ratio = self._baseline_binary_size / binary_size
            perf_ratios.append(binary_ratio)

        if bench_score is not None:
            if self._baseline_bench and self._baseline_bench > 0:
                perf_ratios.append(bench_score / self._baseline_bench)
            else:
                perf_ratios.append(1.0)

        perf_ratio = statistics.mean(perf_ratios) if perf_ratios else 1.0
        # Map to [0, 1]: baseline (ratio=1.0) -> 0.5, 2x improvement -> 1.0
        norm_perf = max(0.0, min(1.0, perf_ratio / 2.0))
        logger.info(
            "Layer 3: loc=%d (ratio=%.3f), compile=%.2fs (ratio=%.3f), "
            "binary=%d (ratio=%.3f) -> perf_ratio=%.3f, norm_perf=%.3f",
            loc, loc_ratio, compile_time, compile_ratio,
            binary_size, binary_ratio, perf_ratio, norm_perf,
        )

        # --- Pre-LLM combined score ---
        w_static = cfg.fitness.static_analysis_weight
        w_perf = cfg.fitness.performance_weight
        pre_llm = (w_static * norm_static + w_perf * norm_perf) / (w_static + w_perf)
        self._score_history.append(pre_llm)

        # --- Layer 4: LLM judgment (top quartile only) ---
        norm_llm = 0.0
        top_q = self._is_top_quartile(pre_llm)
        if cfg.llm_judgment.enabled and (
            not cfg.llm_judgment.top_quartile_only or top_q
        ):
            code = self.source_file.read_text()
            judgment = judge_code(
                code=code,
                api_base=cfg.ollama.api_base,
                model=cfg.ollama.evaluator_model,
                dimensions=cfg.llm_judgment.dimensions,
                num_runs=cfg.llm_judgment.num_runs,
            )
            norm_llm = judgment.combined_score
            logger.info(
                "Layer 4: LLM judge raw=%.2f, norm=%.3f",
                judgment.combined_score, norm_llm,
            )
        else:
            reason = "disabled" if not cfg.llm_judgment.enabled else (
                "not top quartile (%.3f, need %d history, have %d)"
                % (pre_llm, 4, len(self._score_history))
            )
            logger.info("Layer 4: skipped - %s", reason)

        # --- Combined score ---
        combined = (
            cfg.fitness.static_analysis_weight * norm_static
            + cfg.fitness.performance_weight * norm_perf
            + cfg.fitness.llm_judgment_weight * norm_llm
        )
        logger.info(
            "Score: %.3f*%.3f + %.3f*%.3f + %.3f*%.3f = %.4f",
            cfg.fitness.static_analysis_weight, norm_static,
            cfg.fitness.performance_weight, norm_perf,
            cfg.fitness.llm_judgment_weight, norm_llm,
            combined,
        )

        return EvaluationResult(
            passed_gates=True,
            combined_score=combined,
            static_score=norm_static,
            perf_score=perf_ratio,
            llm_score=norm_llm,
            build_time=build.elapsed_seconds,
            tests_passed=test.tests_passed,
            tests_failed=test.tests_failed,
            clippy_warnings=len(clippy.warnings),
            binary_size=binary_ratio,
            compile_time=compile_ratio,
            loc=loc_ratio,
        )
