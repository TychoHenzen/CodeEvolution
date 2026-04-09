from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from codeevolve.config import CodeEvolveConfig
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
    binary_size: int = 0
    compile_time: float = 0.0
    loc: int = 0
    error: str = ""


class EvaluationPipeline:
    """4-layer gated evaluation pipeline for Rust code."""

    def __init__(self, config: CodeEvolveConfig):
        self.config = config
        self._score_history: list[float] = []
        self._static_min: Optional[float] = None
        self._static_max: Optional[float] = None
        self._perf_min: Optional[float] = None
        self._perf_max: Optional[float] = None

    def _update_range(self, value: float, current_min: Optional[float], current_max: Optional[float]):
        new_min = value if current_min is None else min(current_min, value)
        new_max = value if current_max is None else max(current_max, value)
        return new_min, new_max

    def _normalize(self, value: float, min_val: Optional[float], max_val: Optional[float]) -> float:
        if min_val is None or max_val is None or min_val == max_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)

    def _is_top_quartile(self, pre_llm_score: float) -> bool:
        if len(self._score_history) < 4:
            return True
        threshold = statistics.quantiles(self._score_history, n=4)[2]
        return pre_llm_score >= threshold

    def evaluate(self, program_path: str) -> EvaluationResult:
        """Run the full 4-layer evaluation pipeline on a candidate program."""
        project_path = Path(program_path).parent
        cargo = self.config.rust.cargo_path
        cfg = self.config

        # --- Layer 1: Hard gates ---
        build = run_cargo_build(project_path, cargo, cfg.rust.target_dir)
        if not build.success:
            return EvaluationResult(
                passed_gates=False, combined_score=0.0,
                build_time=build.elapsed_seconds, error=build.error_output,
            )

        test = run_cargo_test(project_path, cargo, cfg.rust.test_args or None)
        if not test.success:
            return EvaluationResult(
                passed_gates=False, combined_score=0.0,
                build_time=build.elapsed_seconds,
                tests_passed=test.tests_passed, tests_failed=test.tests_failed,
                error=test.error_output,
            )

        # --- Layer 2: Static analysis ---
        clippy = run_cargo_clippy(project_path, cargo, cfg.rust.clippy_args or None)
        raw_static = compute_clippy_score(clippy.warning_counts, cfg.fitness.clippy_weights)
        self._static_min, self._static_max = self._update_range(
            raw_static, self._static_min, self._static_max
        )
        norm_static = self._normalize(raw_static, self._static_min, self._static_max)

        # --- Layer 3: Performance ---
        compile_time = 0.0
        binary_size = 0
        loc = measure_loc(Path(program_path))
        perf_components = [-loc]  # fewer lines is better

        if cfg.benchmarks.measure_compile_time:
            compile_time = measure_compile_time(project_path, cargo)
            perf_components.append(-compile_time)

        if cfg.benchmarks.measure_binary_size:
            binary_size = measure_binary_size(project_path, cfg.rust.target_dir)
            perf_components.append(-binary_size)

        if cfg.benchmarks.custom_command:
            bench_result = run_user_benchmark(
                cfg.benchmarks.custom_command,
                project_path,
                cfg.benchmarks.custom_command_score_regex,
            )
            perf_components.append(bench_result.score)

        raw_perf = sum(perf_components) if perf_components else 0.0
        self._perf_min, self._perf_max = self._update_range(
            raw_perf, self._perf_min, self._perf_max
        )
        norm_perf = self._normalize(raw_perf, self._perf_min, self._perf_max)

        # --- Pre-LLM combined score ---
        w_static = cfg.fitness.static_analysis_weight
        w_perf = cfg.fitness.performance_weight
        pre_llm = (w_static * norm_static + w_perf * norm_perf) / (w_static + w_perf)
        self._score_history.append(pre_llm)

        # --- Layer 4: LLM judgment (top quartile only) ---
        norm_llm = 0.0
        if cfg.llm_judgment.enabled and (
            not cfg.llm_judgment.top_quartile_only or self._is_top_quartile(pre_llm)
        ):
            code = Path(program_path).read_text()
            judgment = judge_code(
                code=code,
                api_base=cfg.ollama.api_base,
                model=cfg.ollama.evaluator_model,
                dimensions=cfg.llm_judgment.dimensions,
                num_runs=cfg.llm_judgment.num_runs,
            )
            norm_llm = (judgment.combined_score - 1.0) / 4.0

        # --- Combined score ---
        combined = (
            cfg.fitness.static_analysis_weight * norm_static
            + cfg.fitness.performance_weight * norm_perf
            + cfg.fitness.llm_judgment_weight * norm_llm
        )

        return EvaluationResult(
            passed_gates=True,
            combined_score=combined,
            static_score=raw_static,
            perf_score=raw_perf,
            llm_score=norm_llm,
            build_time=build.elapsed_seconds,
            tests_passed=test.tests_passed,
            tests_failed=test.tests_failed,
            clippy_warnings=len(clippy.warnings),
            binary_size=binary_size,
            compile_time=compile_time,
            loc=loc,
        )
