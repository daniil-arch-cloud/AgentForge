"""Offline positive and negative tests for semantic tool-trace validation."""

from __future__ import annotations

import copy
import unittest

from evaluator import TestCase, relation_matches, validate_tool_trace
from hard_cases import HARD_TRAIN_CASES, HARD_HOLDOUT_CASES


class ValidateToolTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = TestCase(
            question="Alpha вырос на 10%, затем сократился на 10%.",
            expected=23.75,
            expected_changes=[("alpha", 10), ("alpha", -10)],
            comparison=("alpha", "beta"),
        )
        self.valid_trace = [
            {
                "tool": "get_project_budget",
                "args": {"project_name": "alpha"},
                "result": 1_500_000,
            },
            {
                "tool": "get_project_budget",
                "args": {"project_name": "beta"},
                "result": 1_200_000,
            },
            {
                "tool": "apply_percentage_change",
                "args": {
                    "project_name": "alpha",
                    "value": 1_500_000,
                    "percentage": 10,
                },
                "result": 1_650_000,
            },
            {
                "tool": "apply_percentage_change",
                "args": {
                    "project_name": "alpha",
                    "value": 1_650_000,
                    "percentage": -10,
                },
                "result": 1_485_000,
            },
            {
                "tool": "calculate_percentage_increase",
                "args": {"new_value": 1_485_000, "old_value": 1_200_000},
                "result": 23.75,
            },
        ]

    def make_trace(self) -> list[dict[str, object]]:
        return copy.deepcopy(self.valid_trace)

    def test_accepts_correct_trace(self) -> None:
        self.assertEqual(validate_tool_trace(self.case, self.make_trace()), [])

    def test_rejects_wrong_percentage_or_sign(self) -> None:
        trace = self.make_trace()
        trace[2]["args"]["percentage"] = -10  # type: ignore[index]
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(any("ожидалось 10%" in issue for issue in issues))

    def test_rejects_wrong_intermediate_value(self) -> None:
        trace = self.make_trace()
        trace[3]["args"]["value"] = 1_600_000  # type: ignore[index]
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(any("неверная промежуточная сумма" in issue for issue in issues))

    def test_rejects_wrong_comparison_base(self) -> None:
        trace = self.make_trace()
        trace[-1]["args"]["old_value"] = 1_500_000  # type: ignore[index]
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(any("база процента" in issue for issue in issues))

    def test_rejects_wrong_change_order(self) -> None:
        trace = self.make_trace()
        trace[2], trace[3] = trace[3], trace[2]
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(issues)

    def test_rejects_change_applied_to_wrong_project(self) -> None:
        trace = self.make_trace()
        trace[2]["args"]["project_name"] = "beta"  # type: ignore[index]
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(any("ожидался проект alpha" in issue for issue in issues))

    def test_rejects_missing_change(self) -> None:
        trace = self.make_trace()
        del trace[3]
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(any("Неверное количество изменений бюджета" in issue for issue in issues))

    def test_rejects_extra_change(self) -> None:
        trace = self.make_trace()
        extra_change = copy.deepcopy(trace[3])
        trace.insert(4, extra_change)
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(any("Неверное количество изменений бюджета" in issue for issue in issues))

    def test_rejects_wrong_comparison_target_value(self) -> None:
        trace = self.make_trace()
        trace[-1]["args"]["new_value"] = 1_500_000  # type: ignore[index]
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(any("Неверный new_value" in issue for issue in issues))

    def test_rejects_wrong_change_result(self) -> None:
        trace = self.make_trace()
        trace[2]["result"] = 1_600_000
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(any("неверный результат инструмента" in issue for issue in issues))

    def test_rejects_budget_lookup_after_changes_begin(self) -> None:
        trace = self.make_trace()
        trace[1], trace[2] = trace[2], trace[1]
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(any("до изменений" in issue for issue in issues))

    def test_rejects_missing_final_comparison(self) -> None:
        trace = self.make_trace()[:-1]
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(any("Нужно ровно 1 вызов calculate_percentage_increase" in issue for issue in issues))

    def test_rejects_unknown_tool(self) -> None:
        trace = self.make_trace()
        trace[0]["tool"] = "unknown_tool"
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(any("Неизвестный инструмент" in issue for issue in issues))

    def test_rejects_wrong_final_calculation_result(self) -> None:
        trace = self.make_trace()
        trace[-1]["result"] = 22.5
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(any("Неверный результат calculate_percentage_increase" in issue for issue in issues))

    def test_rejects_duplicate_budget_lookup(self) -> None:
        trace = self.make_trace()
        trace[1]["args"]["project_name"] = "alpha"  # type: ignore[index]
        issues = validate_tool_trace(self.case, trace)
        self.assertTrue(any("ровно по одному разу" in issue for issue in issues))


class RelationValidationTests(unittest.TestCase):
    def test_accepts_target_project_as_larger(self) -> None:
        self.assertTrue(relation_matches("Alpha больше Beta на 12.5%", "alpha", "beta", "more"))

    def test_accepts_reference_project_as_smaller(self) -> None:
        self.assertTrue(relation_matches("Beta меньше Alpha на 12.5%", "alpha", "beta", "more"))

    def test_accepts_target_project_as_smaller(self) -> None:
        self.assertTrue(relation_matches("Alpha меньше Beta на 12.5%", "alpha", "beta", "less"))

    def test_rejects_wrong_direction(self) -> None:
        self.assertFalse(relation_matches("Alpha меньше Beta на 12.5%", "alpha", "beta", "more"))

    def test_rejects_answer_without_direction(self) -> None:
        self.assertFalse(relation_matches("Разница составляет 12.5%", "alpha", "beta", "more"))

    def test_accepts_unicode_minus_percentage(self) -> None:
        from evaluator import extract_percentage
        self.assertEqual(extract_percentage("Результат −12,5%"), -12.5)

    def test_extracts_comparison_percentage_not_later_percentage_of_base(self) -> None:
        from evaluator import extract_percentage
        answer = (
            "Бюджет Beta больше итогового бюджета Alpha на 43.36% "
            "(или итоговый бюджет Alpha составляет 56.64% от бюджета Beta)."
        )
        self.assertEqual(extract_percentage(answer), 43.36)

    def test_prefers_alpha_vs_beta_comparison_when_both_directions_are_mentioned(self) -> None:
        from evaluator import extract_percentage
        case = TestCase(
            question="Alpha больше Beta",
            expected=25.0,
            expected_changes=[],
            comparison=("alpha", "beta"),
        )
        answer = "Alpha больше Beta на 25%, а Beta меньше Alpha на 20%."
        self.assertEqual(extract_percentage(answer, case), 25.0)

    def test_extracts_generic_difference_when_direction_is_separate(self) -> None:
        from evaluator import extract_percentage
        self.assertEqual(
            extract_percentage("Бюджет Beta больше бюджета Alpha. Разница составляет 23,73%."),
            23.73,
        )


class GeneratedBenchmarkTests(unittest.TestCase):
    def test_train_and_holdout_use_varied_budgets(self) -> None:
        all_cases = HARD_TRAIN_CASES + HARD_HOLDOUT_CASES
        self.assertEqual(len(HARD_TRAIN_CASES), 8)
        self.assertEqual(len(HARD_HOLDOUT_CASES), 4)
        self.assertGreater(len({(case.budgets["alpha"], case.budgets["beta"]) for case in all_cases}), 8)

    def test_every_case_has_explicit_relation_and_budget_values(self) -> None:
        for case in HARD_TRAIN_CASES + HARD_HOLDOUT_CASES:
            self.assertIsNotNone(case.budgets)
            self.assertIn(case.expected_relation, {"more", "less"})
            self.assertGreater(case.budgets["alpha"], 0)
            self.assertGreater(case.budgets["beta"], 0)

    def test_generated_cases_have_valid_reference_traces(self) -> None:
        for case in HARD_TRAIN_CASES + HARD_HOLDOUT_CASES:
            budgets = dict(case.budgets)
            trace = [
                {"tool": "get_project_budget", "args": {"project_name": "alpha"}, "result": budgets["alpha"]},
                {"tool": "get_project_budget", "args": {"project_name": "beta"}, "result": budgets["beta"]},
            ]
            for project, percentage in case.expected_changes:
                value = budgets[project]
                result = round(value * (1 + percentage / 100), 2)
                trace.append({
                    "tool": "apply_percentage_change",
                    "args": {"project_name": project, "value": value, "percentage": percentage},
                    "result": result,
                })
                budgets[project] = result
            target, reference = case.comparison
            result = round((budgets[target] - budgets[reference]) / budgets[reference] * 100, 2)
            trace.append({
                "tool": "calculate_percentage_increase",
                "args": {"new_value": budgets[target], "old_value": budgets[reference]},
                "result": result,
            })
            self.assertEqual(validate_tool_trace(case, trace), [], msg=case.question)



if __name__ == "__main__":
    unittest.main(verbosity=2)


class ExplanationConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = TestCase(
            question="Alpha изменился, Beta изменился",
            expected=47.93,
            expected_changes=[("alpha", -30), ("beta", 10)],
            comparison=("alpha", "beta"),
            budgets={"alpha": 900000, "beta": 1100000},
            expected_relation="less",
        )

    def test_rejects_percentage_with_wrong_denominator(self) -> None:
        from evaluator import explanation_consistent
        self.assertFalse(explanation_consistent(
            "Бюджет Beta больше бюджета Alpha на 47.93%.", self.case
        ))

    def test_accepts_correct_smaller_statement(self) -> None:
        from evaluator import explanation_consistent
        self.assertTrue(explanation_consistent(
            "Бюджет Alpha меньше бюджета Beta на 47.93%.", self.case
        ))

    def test_rejects_mixed_correct_and_incorrect_clauses(self) -> None:
        from evaluator import explanation_consistent
        self.assertFalse(explanation_consistent(
            "Бюджет Alpha меньше бюджета Beta на 47.93%. "
            "Бюджет Beta больше бюджета Alpha на 47.93%.", self.case
        ))
