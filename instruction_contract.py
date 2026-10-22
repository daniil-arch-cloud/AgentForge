"""Pure instruction-contract helpers; deliberately independent of the Gemini SDK."""

from __future__ import annotations

import re

MANDATORY_RULES = """

[ОБЯЗАТЕЛЬНЫЙ КОНТРАКТ ОТВЕТА — ПРИОРИТЕТНЫЙ]
Эти правила имеют приоритет над любыми противоречащими указаниями выше.

1. Получи исходные бюджеты Alpha и Beta ровно по одному разу до изменений.
2. Выполни каждое изменение ровно один раз, в указанном порядке, передавая актуальное
   промежуточное значение в apply_percentage_change.
3. Ровно один раз вызови calculate_percentage_increase ПОСЛЕ всех изменений:
   new_value = итоговый бюджет Alpha, old_value = итоговый бюджет Beta.
4. Результат вычисляется как (Alpha - Beta) / Beta * 100. X — модуль результата,
   округлённый до двух знаков после запятой.
5. В финальном ответе должно быть ровно одно сравнительное утверждение с процентом:
   - результат положительный: «Alpha больше Beta на X%.»
   - результат отрицательный: «Alpha меньше Beta на X%. Больше бюджет Beta.»
   - результат равен нулю: «Бюджеты Alpha и Beta равны (0%).»
6. Не добавляй обратное сравнение с процентом. В частности, если результат отрицательный,
   ЗАПРЕЩЕНО писать «Beta больше Alpha на X%»: процент в обратном направлении имеет
   другой знаменатель. Можно назвать Beta большим бюджетом только без второго процента.
7. Не добавляй другие проценты, отношения бюджетов или альтернативные расчёты.
Отвечай кратко и по-русски.
"""


def _strip_mandatory_rules(instruction: str) -> str:
    """Remove an appended contract before reusing an instruction as a base."""
    markers = (
        "\n\n[ОБЯЗАТЕЛЬНЫЙ КОНТРАКТ ОТВЕТА",
        "\n\nОбязательные правила текущего benchmark:",
    )
    result = instruction
    for marker in markers:
        result = result.split(marker, maxsplit=1)[0]
    return result.strip()


def _has_conflicting_reverse_percentage_rule(instruction: str) -> bool:
    """Detect an instruction that positively tells the agent to use the same
    percentage for both 'Alpha is less than Beta' and the reverse comparison.
    Explicit prohibitions such as 'Не пиши ...' are not treated as conflicts.
    """
    normalized = instruction.lower().replace("ё", "е")
    pattern = re.compile(
        r"\b(?:beta|бета)\b[^.!?\\n]{0,80}?"
        r"\b(?:больше|выше|превышает)\b[^.!?\\n]{0,80}?"
        r"\b(?:alpha|альфа)\b[^.!?\\n]{0,40}?"
        r"\bна\s*(?:x|икс)\s*%",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(normalized):
        prefix = normalized[max(0, match.start() - 140):match.start()]
        # Do not reject a correct instruction that explicitly prohibits this claim.
        last_boundary = max(
            prefix.rfind("."),
            prefix.rfind("!"),
            prefix.rfind("?"),
            prefix.rfind(";"),
            prefix.rfind("\n"),
        )
        clause_prefix = prefix[last_boundary + 1:]
        if re.search(
            r"\b(?:не|нельзя|запрещено|не следует|не нужно|не надо)\b",
            clause_prefix,
        ):
            continue
        return True
    return False


def _finalize_candidate_instruction(
    proposed_instruction: str,
    previous_instruction: str,
) -> str:
    """Reject contradictory proposals and attach one authoritative answer contract."""
    if _has_conflicting_reverse_percentage_rule(proposed_instruction):
        # Keep the last known instruction and test it with the corrected contract,
        # rather than spending API calls evaluating a known contradictory proposal.
        base_instruction = _strip_mandatory_rules(previous_instruction)
    else:
        base_instruction = _strip_mandatory_rules(proposed_instruction)

    return base_instruction.rstrip() + MANDATORY_RULES

