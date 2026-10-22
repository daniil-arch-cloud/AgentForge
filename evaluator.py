"""Evaluate final numeric answers, comparison direction, and tool traces."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tools import DEFAULT_PROJECT_BUDGETS, PROJECT_BUDGETS

if TYPE_CHECKING:
    from agent import Agent


@dataclass
class TestCase:
    question: str
    expected: float
    expected_changes: list[tuple[str, float]]
    comparison: tuple[str, str]
    budgets: dict[str, float] | None = None
    # Relation of comparison[0] to comparison[1]: "more", "less", or "equal".
    expected_relation: str | None = None


@dataclass
class TestResult:
    question: str
    expected: float
    predicted: float | None
    answer: str
    success: bool
    numeric_correct: bool
    relation_correct: bool
    tool_compliant: bool
    tool_issues: list[str]
    tool_calls: list[dict[str, object]]
    explanation_consistent: bool = True


@dataclass
class EvaluationReport:
    accuracy: float
    results: list[TestResult]

    @property
    def failures(self) -> list[TestResult]:
        return [result for result in self.results if not result.success]

    @property
    def numeric_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.numeric_correct for result in self.results) / len(self.results)

    @property
    def relation_accuracy(self) -> float:
        relevant = [result for result in self.results if result.relation_correct is not None]
        if not relevant:
            return 1.0
        return sum(result.relation_correct for result in relevant) / len(relevant)

    @property
    def tool_compliance(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.tool_compliant for result in self.results) / len(self.results)


def extract_percentage(answer: str, case: TestCase | None = None) -> float | None:
    """Extract the percentage attached to the Alpha/Beta comparison.

    Prefer a percentage that appears in an explicit comparative statement such as
    "Alpha меньше Beta на 23,71%". This avoids accidentally reading a second
    percentage from a parenthetical explanation such as "Alpha составляет 56,64%
    от Beta". If no comparative statement is present, retain a last-percentage
    fallback for concise answers like "Разница составляет 23,71%".
    """
    normalized = answer.replace("−", "-").replace("﹣", "-").replace("ё", "е")
    number = r"(?P<value>[+-]?\d+(?:[.,]\d+)?)\s*%"
    subject_pattern = r"(?P<subject>alpha|beta)"
    reference_pattern = r"(?P<reference>alpha|beta)"
    relation = r"(?:больше|меньше|выше|ниже|превышает|уступает)"

    # Form 1: "Alpha больше Beta на 25%".
    after_relation = re.compile(
        rf"\b{subject_pattern}\b[^.!?\n]{{0,180}}?\b{relation}\b"
        rf"[^.!?\n]{{0,120}}?\b{reference_pattern}\b"
        rf"[^.!?\n]{{0,100}}?(?:на\s*)?{number}",
        flags=re.IGNORECASE,
    )

    # Form 2: "Alpha на 25% больше Beta".
    before_relation = re.compile(
        rf"\b{subject_pattern}\b[^.!?\n]{{0,100}}?{number}"
        rf"[^.!?\n]{{0,100}}?\b{relation}\b"
        rf"[^.!?\n]{{0,100}}?\b{reference_pattern}\b",
        flags=re.IGNORECASE,
    )

    candidates: list[tuple[int, int, float]] = []
    for pattern in (after_relation, before_relation):
        for match in pattern.finditer(normalized):
            groups = match.groupdict()
            subject = str(groups["subject"]).lower()
            reference = str(groups["reference"]).lower()
            raw_value = groups.get("value")
            if raw_value is None:
                continue
            value = float(raw_value.replace(",", "."))

            # Prefer the benchmark's canonical Alpha-vs-Beta comparison. If the
            # answer only gives the reverse comparison, still extract it so the
            # denominator-consistency check can reject a misleading percentage.
            priority = 2
            if case is not None:
                target, base = (name.lower() for name in case.comparison)
                if subject == target and reference == base:
                    priority = 0
                elif subject == base and reference == target:
                    priority = 1
            candidates.append((priority, match.start(), value))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    # Fallback for answers that state the direction separately from the number.
    matches = re.findall(r"([+-]?\d+(?:[.,]\d+)?)\s*%", normalized)
    if not matches:
        return None
    return float(matches[-1].replace(",", "."))


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _numbers_match(actual: object, expected: float, *, abs_tol: float = 0.02) -> bool:
    number = _as_float(actual)
    return number is not None and math.isclose(
        number,
        expected,
        rel_tol=1e-9,
        abs_tol=abs_tol,
    )


def relation_matches(
    answer: str,
    target_project: str,
    reference_project: str,
    expected_relation: str | None,
) -> bool:
    """Check whether the final prose states the expected direction of comparison."""
    if expected_relation is None:
        return True

    text = answer.lower().replace("ё", "е")
    target = re.escape(target_project.lower())
    reference = re.escape(reference_project.lower())

    def has_relation(subject: str, words: str) -> bool:
        pattern = rf"\b{subject}\b.{{0,100}}?\b(?:{words})\b"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            fragment = match.group(0)
            if re.search(r"\bне\s+(?:больше|меньше|выше|ниже|превышает|равен|равна|равны)\b", fragment):
                continue
            return True
        return False

    if expected_relation == "more":
        return has_relation(target, "больше|выше|превышает") or has_relation(
            reference, "меньше|ниже"
        )
    if expected_relation == "less":
        return has_relation(target, "меньше|ниже") or has_relation(
            reference, "больше|выше|превышает"
        )
    if expected_relation == "equal":
        return bool(
            re.search(r"\b(?:равен|равна|равны|равно|одинаков\w*|не отличается)\b", text)
        )
    raise ValueError(f"Неизвестное expected_relation: {expected_relation!r}")


def explanation_consistent(answer: str, case: TestCase) -> bool:
    """Reject explicit percentage claims with a wrong base.

    The benchmark defines the percentage as (Alpha - Beta) / Beta * 100.
    Therefore "Alpha is less than Beta by X%" is valid for negative gaps,
    whereas "Beta is larger than Alpha by X%" uses a different denominator.
    This rule only handles explicit Alpha/Beta clauses; it does not claim
    to understand arbitrary natural-language explanations.
    """
    text = answer.lower().replace("ё", "е").replace("−", "-")
    budgets = _case_budgets(case)
    for project, percentage in case.expected_changes:
        budgets[project] = round(budgets[project] * (1 + percentage / 100), 2)
    alpha, beta = budgets["alpha"], budgets["beta"]
    # Match local comparative claims (both names and one percentage nearby).
    pattern = re.compile(
        r"\b(alpha|beta)\b[^.!?\n]{0,85}?"
        r"\b(больше|меньше)\b[^.!?\n]{0,55}?"
        r"\b(alpha|beta)\b[^.!?\n]{0,50}?"
        r"(?:на\s*)?([+-]?\d+(?:[.,]\d+)?)\s*%",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        subject, relation, reference, number = match.groups()
        if subject == reference:
            return False
        actual = float(number.replace(",", "."))
        values = {"alpha": alpha, "beta": beta}
        expected = (values[subject] - values[reference]) / values[reference] * 100
        if relation == "меньше":
            expected = -expected
        if expected < -0.06 or abs(abs(actual) - expected) > 0.06:
            return False
    return True


def _case_budgets(case: TestCase) -> dict[str, float]:
    return dict(case.budgets or DEFAULT_PROJECT_BUDGETS)


def validate_tool_trace(
    case: TestCase,
    tool_calls: list[dict[str, object]],
) -> list[str]:
    """Validate tool names, project identity, ordered changes, intermediate values and comparison."""
    issues: list[str] = []
    allowed_tools = {
        "get_project_budget",
        "apply_percentage_change",
        "calculate_percentage_increase",
    }
    budgets = _case_budgets(case)
    names = [str(call.get("tool", "")) for call in tool_calls]

    for name in names:
        if name not in allowed_tools:
            issues.append(f"Неизвестный инструмент: {name}")

    get_calls = [call for call in tool_calls if call.get("tool") == "get_project_budget"]
    change_calls = [call for call in tool_calls if call.get("tool") == "apply_percentage_change"]
    calc_calls = [call for call in tool_calls if call.get("tool") == "calculate_percentage_increase"]

    if len(get_calls) != 2:
        issues.append(f"Нужно ровно 2 вызова get_project_budget, получено {len(get_calls)}")

    looked_up: list[str] = []
    for call in get_calls:
        args = call.get("args")
        args = args if isinstance(args, dict) else {}
        project = str(args.get("project_name", "")).strip().lower()
        looked_up.append(project)
        if project not in budgets:
            issues.append(f"get_project_budget вызван для неизвестного проекта: {project!r}")
            continue
        if not _numbers_match(call.get("result"), budgets[project]):
            issues.append(f"Неверный результат получения бюджета {project}")

    if Counter(looked_up) != Counter({"alpha": 1, "beta": 1}):
        issues.append("Нужно получить исходные бюджеты Alpha и Beta ровно по одному разу")

    if len(change_calls) != len(case.expected_changes):
        issues.append(
            "Неверное количество изменений бюджета: "
            f"ожидалось {len(case.expected_changes)}, получено {len(change_calls)}"
        )

    intended_budgets = dict(budgets)
    for index, (expected_project, expected_percentage) in enumerate(case.expected_changes):
        if expected_project not in intended_budgets:
            issues.append(f"Некорректный проект в эталонном тесте: {expected_project}")
            continue

        if index >= len(change_calls):
            intended_budgets[expected_project] = round(
                intended_budgets[expected_project] * (1 + expected_percentage / 100), 2
            )
            continue

        call = change_calls[index]
        args = call.get("args")
        args = args if isinstance(args, dict) else {}
        actual_project = str(args.get("project_name", "")).strip().lower()

        if actual_project != expected_project:
            issues.append(
                f"Изменение #{index + 1}: ожидался проект {expected_project}, "
                f"получен {actual_project or 'не указан'}"
            )

        actual_percentage = _as_float(args.get("percentage"))
        if actual_percentage is None or not math.isclose(
            actual_percentage, expected_percentage, rel_tol=0.0, abs_tol=1e-6
        ):
            issues.append(
                f"Изменение #{index + 1}: ожидалось {expected_percentage}%, "
                f"получено {args.get('percentage')!r}"
            )

        expected_input = intended_budgets[expected_project]
        if not _numbers_match(args.get("value"), expected_input):
            issues.append(
                f"Изменение #{index + 1}: неверная промежуточная сумма для {expected_project}; "
                f"ожидалось {expected_input}, получено {args.get('value')!r}"
            )

        expected_result = round(expected_input * (1 + expected_percentage / 100), 2)
        if not _numbers_match(call.get("result"), expected_result):
            issues.append(
                f"Изменение #{index + 1}: неверный результат инструмента; "
                f"ожидалось {expected_result}, получено {call.get('result')!r}"
            )

        intended_budgets[expected_project] = expected_result

    if len(calc_calls) != 1:
        issues.append(
            "Нужно ровно 1 вызов calculate_percentage_increase, "
            f"получено {len(calc_calls)}"
        )
    else:
        target_project, reference_project = case.comparison
        if target_project not in intended_budgets or reference_project not in intended_budgets:
            issues.append("Некорректная пара проектов в эталонном сравнении")
        else:
            calc_call = calc_calls[0]
            args = calc_call.get("args")
            args = args if isinstance(args, dict) else {}
            target_value = intended_budgets[target_project]
            reference_value = intended_budgets[reference_project]

            if not _numbers_match(args.get("new_value"), target_value):
                issues.append(
                    "Неверный new_value в итоговом сравнении: "
                    f"ожидалось {target_value}, получено {args.get('new_value')!r}"
                )
            if not _numbers_match(args.get("old_value"), reference_value):
                issues.append(
                    "Неверная база процента (old_value): "
                    f"ожидалось {reference_value}, получено {args.get('old_value')!r}"
                )

            if reference_value != 0:
                expected_calculation = round(
                    (target_value - reference_value) / reference_value * 100, 2
                )
                if not _numbers_match(calc_call.get("result"), expected_calculation):
                    issues.append(
                        "Неверный результат calculate_percentage_increase: "
                        f"ожидалось {expected_calculation}, получено {calc_call.get('result')!r}"
                    )

    if get_calls and (change_calls or calc_calls):
        get_indices = [i for i, name in enumerate(names) if name == "get_project_budget"]
        later_indices = [
            i for i, name in enumerate(names)
            if name in {"apply_percentage_change", "calculate_percentage_increase"}
        ]
        if later_indices and max(get_indices) > min(later_indices):
            issues.append("Оба исходных бюджета должны быть получены до изменений")

    if len(calc_calls) == 1 and names and names[-1] != "calculate_percentage_increase":
        issues.append("Итоговое процентное сравнение должно быть последним вызовом инструмента")

    return issues


def evaluate(
    agent: Agent,
    cases: list[TestCase],
    pause_seconds: float = 25.0,
) -> EvaluationReport:
    if not cases:
        raise ValueError("Для оценки нужен хотя бы один тест")

    results: list[TestResult] = []
    for index, case in enumerate(cases, start=1):
        print(f"\nТест {index}: {case.question}", flush=True)
        answer = ""
        predicted: float | None = None
        case_budgets = _case_budgets(case)

        PROJECT_BUDGETS.clear()
        PROJECT_BUDGETS.update(case_budgets)
        try:
            try:
                answer = agent.run(case.question)
                predicted = extract_percentage(answer, case)
            except Exception as error:
                message = str(error)
                if (
                    "RESOURCE_EXHAUSTED" in message
                    or "429" in message
                    or "ReadTimeout" in type(error).__name__
                    or "DEADLINE_EXCEEDED" in message
                    or "timeout" in type(error).__name__.lower()
                    or "500 INTERNAL" in message
                    or "ServerError" in type(error).__name__
                    or "503 UNAVAILABLE" in message
                    or "504 DEADLINE_EXCEEDED" in message
                    or (
                        "FAILED_PRECONDITION" in message
                        and "User location is not supported" in message
                    )
                ):
                    print("Ошибка инфраструктуры API. Эксперимент прерван без оценки.")
                    raise
                answer = f"{type(error).__name__}: {error}"
            tool_calls = list(agent.last_tool_calls)
        finally:
            PROJECT_BUDGETS.clear()
            PROJECT_BUDGETS.update(DEFAULT_PROJECT_BUDGETS)

        tool_issues = validate_tool_trace(case, tool_calls)
        tool_compliant = not tool_issues
        numeric_correct = predicted is not None and abs(predicted - case.expected) < 0.06
        target_project, reference_project = case.comparison
        relation_correct = relation_matches(
            answer,
            target_project,
            reference_project,
            case.expected_relation,
        )
        consistent = explanation_consistent(answer, case)
        success = numeric_correct and relation_correct and consistent

        result = TestResult(
            question=case.question,
            expected=case.expected,
            predicted=predicted,
            answer=answer,
            success=success,
            numeric_correct=numeric_correct,
            relation_correct=relation_correct,
            tool_compliant=tool_compliant,
            tool_issues=tool_issues,
            tool_calls=tool_calls,
            explanation_consistent=consistent,
        )
        results.append(result)

        print("Полный ответ:", repr(answer))
        print("Получено:", predicted)
        print("Ожидалось (модуль процента):", case.expected)
        print("Численный ответ:", "PASS" if numeric_correct else "FAIL")
        if case.expected_relation is not None:
            print("Направление сравнения:", "PASS" if relation_correct else "FAIL")
        print("Согласованность объяснения:", "PASS" if consistent else "FAIL")
        print("Итоговый ответ:", "PASS" if success else "FAIL")
        print("Вызовы:", dict(Counter(str(call.get("tool")) for call in tool_calls)))
        print("Проверка трассы:", "PASS" if tool_compliant else "FAIL")
        if tool_issues:
            for issue in tool_issues:
                print("  -", issue)

        if index < len(cases) and pause_seconds > 0:
            time.sleep(pause_seconds)

    accuracy = sum(result.success for result in results) / len(results)
    report = EvaluationReport(accuracy=accuracy, results=results)
    print(f"\nAccuracy (число + направление): {report.accuracy:.2%}")
    print(f"Numeric accuracy: {report.numeric_accuracy:.2%}")
    print(f"Relation accuracy: {report.relation_accuracy:.2%}")
    print(f"Tool trace compliance: {report.tool_compliance:.2%}")
    print(f"Ошибок итогового ответа: {len(report.failures)}")
    print(
        "Ошибок трассы инструментов:",
        sum(not result.tool_compliant for result in results),
    )
    return report
