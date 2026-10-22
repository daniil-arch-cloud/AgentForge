"""Regression tests for contradictory percentage instructions in evolution."""

from __future__ import annotations

import unittest

from instruction_contract import (
    MANDATORY_RULES,
    _finalize_candidate_instruction,
    _has_conflicting_reverse_percentage_rule,
)


class ReversePercentageInstructionTests(unittest.TestCase):
    def test_detects_instruction_that_requires_same_percent_in_both_directions(self) -> None:
        proposal = (
            "Если результат отрицательный, напиши строго: "
            "«Alpha меньше Beta на X%» и отдельно "
            "«Beta больше Alpha на X%»."
        )
        self.assertTrue(_has_conflicting_reverse_percentage_rule(proposal))

    def test_does_not_flag_explicit_prohibition(self) -> None:
        proposal = (
            "Если результат отрицательный, напиши «Alpha меньше Beta на X%». "
            "Не пиши «Beta больше Alpha на X%»."
        )
        self.assertFalse(_has_conflicting_reverse_percentage_rule(proposal))

    def test_conflicting_proposal_falls_back_to_previous_instruction(self) -> None:
        previous = "Ты помощник по анализу бюджетов. Отвечай кратко и по-русски."
        proposal = (
            "Напиши Alpha меньше Beta на X%, а также Beta больше Alpha на X%."
        )
        result = _finalize_candidate_instruction(proposal, previous)
        self.assertTrue(result.startswith(previous))
        self.assertNotIn("а также Beta больше Alpha на X%", result)
        self.assertIn("Больше бюджет Beta.", result)
        self.assertIn(MANDATORY_RULES.strip(), result)

    def test_safe_proposal_is_preserved_and_contract_is_appended(self) -> None:
        previous = "Старая инструкция."
        proposal = "Сравни бюджеты и используй инструменты строго по порядку."
        result = _finalize_candidate_instruction(proposal, previous)
        self.assertTrue(result.startswith(proposal))
        self.assertIn(MANDATORY_RULES.strip(), result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
