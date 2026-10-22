"""Local tools for the experimental budget-analysis agent."""

DEFAULT_PROJECT_BUDGETS: dict[str, float] = {
    "alpha": 1_500_000,
    "beta": 1_200_000,
}

# Mutable values are set per test case by the evaluator.
PROJECT_BUDGETS: dict[str, float] = dict(DEFAULT_PROJECT_BUDGETS)

# Reset before each task. This is a single-process experimental prototype.
TOOL_CALL_LOG: list[dict[str, object]] = []


def get_project_budget(project_name: str) -> float:
    """Return a project's original budget in rubles.

    Args:
        project_name: Project name, either Alpha or Beta.
    """
    name = project_name.strip().lower()
    if name not in PROJECT_BUDGETS:
        raise ValueError(f"Неизвестный проект: {project_name}")

    result = PROJECT_BUDGETS[name]
    print(f"[TOOL] Получение бюджета: {name.title()}", flush=True)
    TOOL_CALL_LOG.append({
        "tool": "get_project_budget",
        "args": {"project_name": name},
        "result": result,
    })
    return result


def apply_percentage_change(
    project_name: str,
    value: float,
    percentage: float,
) -> float:
    """Apply one percentage change to the current value of a named project.

    Args:
        project_name: Project being changed, either Alpha or Beta.
        value: Current value BEFORE this change. Use the previous result for
            subsequent changes to the same project.
        percentage: Signed percentage; e.g. 10 means +10%, -10 means -10%.
    """
    name = project_name.strip().lower()
    if name not in PROJECT_BUDGETS:
        raise ValueError(f"Неизвестный проект: {project_name}")

    result = round(value * (1 + percentage / 100), 2)
    print(
        f"[TOOL] Изменение бюджета {name.title()} на {percentage}%",
        flush=True,
    )
    TOOL_CALL_LOG.append({
        "tool": "apply_percentage_change",
        "args": {
            "project_name": name,
            "value": value,
            "percentage": percentage,
        },
        "result": result,
    })
    return result


def calculate_percentage_increase(new_value: float, old_value: float) -> float:
    """Calculate signed percentage change relative to a reference value.

    Args:
        new_value: Final value being compared.
        old_value: Final reference value (must not be zero).
    """
    if old_value == 0:
        raise ValueError("Нельзя вычислить процент относительно нуля")

    result = round((new_value - old_value) / old_value * 100, 2)
    print("[TOOL] Вычисление процентной разницы", flush=True)
    TOOL_CALL_LOG.append({
        "tool": "calculate_percentage_increase",
        "args": {"new_value": new_value, "old_value": old_value},
        "result": result,
    })
    return result
