from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def choice(instructions: str, criteria: Mapping[str, str | None]) -> dict[str, Any]:
    """Pick one of a known set. Criteria keys are the option ids Jev returns."""
    question = {
        "type": "choice",
        "instructions": instructions,
        "criteria": {str(key): value for key, value in criteria.items()},
    }
    _check_question("choice", question)
    return question


def score(instructions: str, criteria: Sequence[str]) -> dict[str, Any]:
    """Place the state on an ordered scale. Levels run from low to high."""
    question = {
        "type": "score",
        "instructions": instructions,
        "criteria": [str(level) for level in criteria],
    }
    _check_question("score", question)
    return question


def noul(instructions: str, criteria: str | None = None) -> dict[str, Any]:
    """Yes-or-no probability. Optional criteria is a short clarification."""
    question: dict[str, Any] = {
        "type": "noul",
        "instructions": instructions,
    }
    if criteria is not None:
        question["criteria"] = criteria
    _check_question("noul", question)
    return question


def check_questions(questions: Mapping[str, Mapping[str, Any]]) -> None:
    if not questions:
        raise ValueError("at least one question is required")
    for name, question in questions.items():
        if not isinstance(question, Mapping):
            raise ValueError(f"{name}: question must be a mapping")
        _check_question(str(name), question)


def _check_question(name: str, question: Mapping[str, Any]) -> None:
    kind = question.get("type")
    if kind not in {"choice", "noul", "score"}:
        raise ValueError(f"{name}: type must be choice, noul, or score")
    if not str(question.get("instructions") or "").strip():
        raise ValueError(f"{name}: instructions must be non-empty")

    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, Mapping) or not 2 <= len(criteria) <= 255:
            raise ValueError(f"{name}: choice needs 2 to 255 options")
    elif kind == "score":
        if (
            not isinstance(criteria, Sequence)
            or isinstance(criteria, (str, bytes))
            or not 2 <= len(criteria) <= 10
        ):
            raise ValueError(f"{name}: score needs 2 to 10 levels")
        if any(not str(level).strip() for level in criteria):
            raise ValueError(f"{name}: score levels must be non-empty")
    elif criteria is not None and not isinstance(criteria, str):
        raise ValueError(f"{name}: noul criteria must be a string when set")
