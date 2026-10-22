"""LLM agent that chooses among local budget tools."""

from __future__ import annotations

from google import genai
from google.genai import types

from tools import (
    TOOL_CALL_LOG,
    apply_percentage_change,
    calculate_percentage_increase,
    get_project_budget,
)


class Agent:
    def __init__(
        self,
        client: genai.Client,
        system_instruction: str,
    ) -> None:
        self.client = client
        self.system_instruction = system_instruction
        self.last_tool_calls: list[dict[str, object]] = []

    def run(self, task: str) -> str:
        # The tool trace is task-local, so evaluator checks cannot leak across cases.
        TOOL_CALL_LOG.clear()
        self.last_tool_calls = []

        chat = self.client.chats.create(
            model="gemini-3.5-flash-lite",
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                tools=[
                    get_project_budget,
                    apply_percentage_change,
                    calculate_percentage_increase,
                ],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=12,
                ),
            ),
        )

        try:
            response = chat.send_message(task)
            return response.text or ""
        finally:
            self.last_tool_calls = [dict(call) for call in TOOL_CALL_LOG]
