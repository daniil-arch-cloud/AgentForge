"""Reproducible benchmark with varied starting budgets and multi-step changes."""

from __future__ import annotations

import random

from evaluator import TestCase
from tools import DEFAULT_PROJECT_BUDGETS


BENCHMARK_SEED = 20261020

TRAIN_SCENARIOS: list[list[tuple[str, float]]] = [
    [("alpha", 10), ("alpha", -10)],
    [("beta", -20), ("beta", 25)],
    [("alpha", 20), ("beta", 50)],
    [("beta", 50)],
    [("alpha", -30), ("beta", 10)],
    [("alpha", 15), ("alpha", 10)],
    [("alpha", -20), ("beta", -5)],
    [("alpha", 15), ("beta", 30)],
]

HOLDOUT_SCENARIOS: list[list[tuple[str, float]]] = [
    [("beta", 25), ("beta", -20)],
    [("alpha", 5), ("alpha", 5)],
    [("beta", 5), ("beta", 20)],
    [("alpha", 10), ("beta", 20), ("alpha", -15)],
]


def _change_phrase(project: str, percentage: float) -> str:
    title = project.title()
    amount = f"{abs(percentage):g}%"
    if percentage > 0:
        return f"бюджет {title} увеличили на {amount}"
    return f"бюджет {title} уменьшили на {amount}"


def _build_cases(
    scenarios: list[list[tuple[str, float]]],
    seed: int,
) -> list[TestCase]:
    rng = random.Random(seed)
    cases: list[TestCase] = []

    for scenario_index, changes in enumerate(scenarios, start=1):
        for _attempt in range(500):
            alpha = float(rng.randrange(650_000, 2_800_001, 25_000))
            beta = float(rng.randrange(650_000, 2_800_001, 25_000))
            budgets = {"alpha": alpha, "beta": beta}
            final_budgets = dict(budgets)

            for project, percentage in changes:
                final_budgets[project] = round(
                    final_budgets[project] * (1 + percentage / 100), 2
                )

            relative_gap = abs(final_budgets["alpha"] - final_budgets["beta"]) / final_budgets["beta"]
            if relative_gap >= 0.08:
                break
        else:
            raise RuntimeError("Не удалось сгенерировать непограничный тест")

        relation = "more" if final_budgets["alpha"] > final_budgets["beta"] else "less"
        expected = round(
            abs((final_budgets["alpha"] - final_budgets["beta"]) / final_budgets["beta"] * 100),
            2,
        )
        operations = "; затем ".join(
            _change_phrase(project, percentage)
            for project, percentage in changes
        )
        unchanged = [
            project.title()
            for project in ("alpha", "beta")
            if all(changed_project != project for changed_project, _ in changes)
        ]
        if unchanged:
            operations += "; бюджет " + " и ".join(unchanged) + " не менялся"

        question = (
            "Исходные бюджеты Alpha и Beta доступны через инструменты. "
            f"Изменения по порядку: {operations}. "
            "Каков процентный размер отличия итогового бюджета Alpha от итогового бюджета Beta? "
            "Укажи процент и обязательно назови, какой бюджет больше."
        )

        cases.append(
            TestCase(
                question=question,
                expected=expected,
                expected_changes=list(changes),
                comparison=("alpha", "beta"),
                budgets=budgets,
                expected_relation=relation,
            )
        )

    return cases


HARD_TRAIN_CASES = _build_cases(TRAIN_SCENARIOS, BENCHMARK_SEED)
HARD_HOLDOUT_CASES = _build_cases(HOLDOUT_SCENARIOS, BENCHMARK_SEED + 1)
