"""Automatic prompt evolution with TRAIN-only selection and HOLDOUT validation."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types

from agent import Agent
from evaluator import EvaluationReport, TestCase, evaluate
from hard_cases import HARD_HOLDOUT_CASES, HARD_TRAIN_CASES


from instruction_contract import (
    _finalize_candidate_instruction,
)


class EvolutionEngine:
    def __init__(
        self,
        client: genai.Client,
        initial_instruction: str,
        generations: int = 1,
        pause_seconds: float = 25.0,
    ) -> None:
        if generations < 1:
            raise ValueError("Число поколений должно быть не меньше 1")
        self.client = client
        self.initial_instruction = initial_instruction
        self.generations = generations
        self.pause_seconds = pause_seconds

    @staticmethod
    def _metrics(report: EvaluationReport) -> dict[str, float]:
        return {
            "accuracy": report.accuracy,
            "numeric_accuracy": report.numeric_accuracy,
            "relation_accuracy": report.relation_accuracy,
            "tool_trace_compliance": report.tool_compliance,
        }

    @staticmethod
    def _should_accept(candidate: EvaluationReport, current: EvaluationReport) -> bool:
        """Accept only a non-regressing candidate that improves at least one metric.

        A higher headline accuracy must not hide regressions in numeric accuracy or
        tool-trace compliance. All four metrics are evaluated on TRAIN only.
        """
        candidate_metrics = {
            "accuracy": candidate.accuracy,
            "numeric_accuracy": candidate.numeric_accuracy,
            "relation_accuracy": candidate.relation_accuracy,
            "tool_trace_compliance": candidate.tool_compliance,
        }
        current_metrics = {
            "accuracy": current.accuracy,
            "numeric_accuracy": current.numeric_accuracy,
            "relation_accuracy": current.relation_accuracy,
            "tool_trace_compliance": current.tool_compliance,
        }
        no_regression = all(
            candidate_metrics[name] >= current_metrics[name]
            for name in current_metrics
        )
        strict_improvement = any(
            candidate_metrics[name] > current_metrics[name]
            for name in current_metrics
        )
        return no_regression and strict_improvement

    @staticmethod
    def _details(report: EvaluationReport) -> list[dict[str, object]]:
        return [
            {
                "question": result.question,
                "expected": result.expected,
                "predicted": result.predicted,
                "answer": result.answer,
                "correct": result.success,
                "numeric_correct": result.numeric_correct,
                "relation_correct": result.relation_correct,
                "explanation_consistent": result.explanation_consistent,
                "tool_trace_compliant": result.tool_compliant,
                "tool_issues": result.tool_issues,
                "tool_calls": result.tool_calls,
            }
            for result in report.results
        ]

    def propose_instruction(
        self,
        instruction: str,
        report: EvaluationReport,
    ) -> str:
        observations = [
            {
                "question": result.question,
                "expected": result.expected,
                "predicted": result.predicted,
                "answer": result.answer,
                "numeric_correct": result.numeric_correct,
                "relation_correct": result.relation_correct,
                "explanation_consistent": result.explanation_consistent,
                "tool_trace_compliant": result.tool_compliant,
                "tool_issues": result.tool_issues,
                "tool_calls": result.tool_calls,
            }
            for result in report.results
            if not result.success or not result.tool_compliant
        ]

        if not observations:
            observations = [{
                "observation": (
                    "На текущих обучающих задачах ошибок ответа и ошибок трассы "
                    "инструментов не обнаружено. Не заявляй об улучшении без данных. "
                    "Предложи конкретную инструкцию, которая помогает сохранять "
                    "точность при правильном выборе проектов, процентов, промежуточных "
                    "значений и базы итогового сравнения."
                )
            }]

        prompt = f"""
Ты исследователь, который предлагает новую системную инструкцию для LLM-агента.
Код инструментов менять нельзя; можно менять только инструкцию агента.

Инструменты:
- get_project_budget(project_name): исходный бюджет Alpha или Beta;
- apply_percentage_change(project_name, value, percentage): применить одно
  изменение к указанному проекту. percentage — знаковое число; value — текущее
  значение до изменения. Результат предыдущего изменения должен стать value
  следующего изменения того же проекта;
- calculate_percentage_increase(new_value, old_value): вычислить
  (new_value - old_value) / old_value * 100.

Ожидаемое поведение:
- получить исходные бюджеты Alpha и Beta по одному разу до изменений;
- выполнить каждое изменение в условии ровно один раз, в указанном порядке;
- передавать верный проект, процент и промежуточную сумму в каждый инструмент;
- В ЭТОМ BENCHMARK итоговое сравнение всегда выполняется в фиксированном порядке:
  new_value = итоговый бюджет Alpha, old_value = итоговый бюджет Beta.
  Не меняй их местами, даже когда Beta больше Alpha.
- calculate_percentage_increase возвращает (Alpha - Beta) / Beta * 100.
  Положительный результат означает, что Alpha больше Beta; отрицательный — что
  Alpha меньше Beta. Для ответа используй модуль результата как положительный процент.
- В финальном ответе используй ровно одно сравнительное утверждение с процентом.
- Если результат отрицательный, ответь: «Alpha меньше Beta на X%. Больше бюджет Beta.»
  Не добавляй фразу «Beta больше Alpha на X%»: это обратный процент с другим знаменателем.
- Если результат положительный, ответь: «Alpha больше Beta на X%.»
- Не добавляй другие проценты, отношения бюджетов или альтернативные расчёты.
- Отвечай кратко по-русски и обязательно называй, какой бюджет больше.

Текущая инструкция:
{instruction}

Наблюдения только с TRAIN:
{json.dumps(observations, ensure_ascii=False, indent=2)}

Предложи ОДНУ улучшенную системную инструкцию. Не включай эталонные числовые
ответы или конкретные ответы на тесты. Не утверждай, что улучшение гарантировано.
Верни только текст новой инструкции.
"""
        response = self.client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=700,
            ),
        )
        candidate = (response.text or "").strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()

        if len(candidate) < 40:
            raise RuntimeError("LLM не вернула полноценную инструкцию")

        # A generated instruction can contradict the percentage convention even when
        # the proposal prompt warns against it. Reject such proposals before evaluation.
        return _finalize_candidate_instruction(candidate, instruction)

    def _evaluate(
        self,
        agent: Agent,
        cases: list[TestCase],
        title: str,
    ) -> EvaluationReport:
        print(f"\n=== {title} ===", flush=True)
        return evaluate(agent, cases, pause_seconds=self.pause_seconds)

    def run(self) -> dict[str, object]:
        history: list[dict[str, object]] = []
        baseline_agent = Agent(self.client, self.initial_instruction)
        baseline_train = self._evaluate(
            baseline_agent, HARD_TRAIN_CASES, "BASELINE / TRAIN"
        )

        selected_instruction = self.initial_instruction
        selected_train = baseline_train
        history.append({
            "generation": 0,
            "instruction": self.initial_instruction,
            "train": self._metrics(baseline_train),
            "accepted": True,
        })

        for generation in range(1, self.generations + 1):
            print(f"\n=== ЭВОЛЮЦИЯ: поколение {generation} ===", flush=True)
            if self.pause_seconds > 0:
                time.sleep(self.pause_seconds)

            candidate_instruction = self.propose_instruction(
                selected_instruction,
                selected_train,
            )
            print("\nСгенерирована новая инструкция:")
            print(candidate_instruction)

            if self.pause_seconds > 0:
                time.sleep(self.pause_seconds)

            candidate_agent = Agent(self.client, candidate_instruction)
            candidate_train = self._evaluate(
                candidate_agent,
                HARD_TRAIN_CASES,
                "CANDIDATE / TRAIN",
            )

            accepted = self._should_accept(candidate_train, selected_train)
            history.append({
                "generation": generation,
                "instruction": candidate_instruction,
                "train": self._metrics(candidate_train),
                "accepted": accepted,
            })

            print("\nСРАВНЕНИЕ НА TRAIN")
            print(f"Лучшая до изменения: {self._metrics(selected_train)}")
            print(f"Кандидат:           {self._metrics(candidate_train)}")
            print("Решение:", "ПРИНЯТ" if accepted else "ОТКЛОНЁН")

            if accepted:
                selected_instruction = candidate_instruction
                selected_train = candidate_train

        # HOLDOUT is never used to decide whether a candidate is accepted.
        if self.pause_seconds > 0:
            time.sleep(self.pause_seconds)
        baseline_holdout = self._evaluate(
            baseline_agent,
            HARD_HOLDOUT_CASES,
            "BASELINE / HOLDOUT",
        )

        if selected_instruction == self.initial_instruction:
            selected_holdout = baseline_holdout
        else:
            if self.pause_seconds > 0:
                time.sleep(self.pause_seconds)
            selected_agent = Agent(self.client, selected_instruction)
            selected_holdout = self._evaluate(
                selected_agent,
                HARD_HOLDOUT_CASES,
                "SELECTED / HOLDOUT",
            )

        output: dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": "gemini-3.5-flash-lite",
            "selection_rule": "TRAIN-only: accept a candidate only if accuracy, numeric accuracy, relation accuracy, and tool-trace compliance do not regress and at least one metric improves",
            "train_case_count": len(HARD_TRAIN_CASES),
            "holdout_case_count": len(HARD_HOLDOUT_CASES),
            "initial_instruction": self.initial_instruction,
            "selected_instruction": selected_instruction,
            "history": history,
            "baseline_train": {
                **self._metrics(baseline_train),
                "results": self._details(baseline_train),
            },
            "selected_train": {
                **self._metrics(selected_train),
                "results": self._details(selected_train),
            },
            "baseline_holdout": {
                **self._metrics(baseline_holdout),
                "results": self._details(baseline_holdout),
            },
            "selected_holdout": {
                **self._metrics(selected_holdout),
                "results": self._details(selected_holdout),
            },
        }

        results_path = Path(__file__).resolve().parent / "results"
        results_path.mkdir(exist_ok=True)
        filename = datetime.now().strftime("experiment_%Y%m%d_%H%M%S.json")
        result_file = results_path / filename
        result_file.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n=== ИТОГИ ===")
        print("Исходная версия / TRAIN:", self._metrics(baseline_train))
        print("Выбранная версия / TRAIN:", self._metrics(selected_train))
        print("Исходная версия / HOLDOUT:", self._metrics(baseline_holdout))
        print("Выбранная версия / HOLDOUT:", self._metrics(selected_holdout))
        print(
            "Выбрана:",
            "BASELINE" if selected_instruction == self.initial_instruction else "НОВАЯ ИНСТРУКЦИЯ",
        )
        print("История сохранена:", result_file)
        return output
