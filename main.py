"""Run one generation of prompt evolution on the multi-step budget benchmark."""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from evolution import EvolutionEngine


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Не найден GEMINI_API_KEY. Проверь, что файл .env лежит в корне проекта "
        "и содержит GEMINI_API_KEY=твой_ключ"
    )

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=60000,
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

# Intentionally minimal baseline. Evolution can propose a better instruction,
# but the evaluator—not the proposal text—decides whether it is better.
initial_instruction = (
    "Ты помощник по анализу бюджетов проектов. "
    "Отвечай кратко и по-русски."
)

engine = EvolutionEngine(
    client=client,
    initial_instruction=initial_instruction,
    generations=1,
    pause_seconds=25.0,
)

try:
    engine.run()
except Exception as error:
    message = str(error)
    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        print("\nДостигнут лимит Gemini API. Этот запуск не считается завершённым экспериментом.")
        print("Подожди восстановления квоты и запусти main.py снова.")
    else:
        raise
