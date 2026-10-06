from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from jev import Jev, JevError, choice, noul, score


def _payload() -> dict:
    return {
        "id": "req_1",
        "model": "typesafe/jev-1.13",
        "provider": "typesafe",
        "usage": {"input_tokens": 120, "output_tokens": 8, "cost": 0.00001},
        "answers": {
            "topic": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.91,
                "probabilities": {"billing": 0.91, "bug": 0.09},
            },
            "severity": {
                "type": "score",
                "score": 2.4,
                "confidence": 0.8,
                "legend": {"0": "routine", "3": "critical"},
                "probabilities": {"2": 0.6, "3": 0.4},
            },
            "escalate": {"type": "noul", "noul": 0.2},
        },
    }


def _client(handler) -> Jev:
    return Jev(api_key="sk-test", model="~typesafe/jev-latest", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_question_builders() -> None:
    picked = choice("Which team?", {"billing": "money", "bug": "broken", "other": None})
    ranked = score("How urgent?", ["routine", "today", "critical"])
    flag = noul("Escalate now?")
    assert picked["type"] == "choice"
    assert picked["criteria"]["other"] is None
    assert ranked["criteria"] == ["routine", "today", "critical"]
    assert flag == {"type": "noul", "instructions": "Escalate now?"}
    with pytest.raises(ValueError):
        choice("Which team?", {"only": "one"})
    with pytest.raises(ValueError):
        score("How urgent?", ["only one level"])


def test_decide_posts_all_three_primitives() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json=_payload())

    decision = _client(handler).decide(
        {"ticket": "I was charged twice."},
        {
            "topic": choice("What is the issue?", {"billing": "money", "bug": "broken"}),
            "severity": score("How urgent?", ["routine", "today", "urgent", "critical"]),
            "escalate": noul("Escalate to a human now?"),
        },
    )

    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "~typesafe/jev-latest"
    assert set(seen["body"]["questions"]) == {"topic", "severity", "escalate"}
    assert decision.answers["topic"].choice == "billing"
    assert decision.answers["topic"].confidence == 0.91
    assert decision.answers["severity"].score == 2.4
    assert decision.answers["severity"].legend["3"] == "critical"
    assert decision.answers["escalate"].noul == 0.2
    assert decision.answers["escalate"].confidence is None
    assert decision.cost == 0.00001
    assert decision.request_id == "req_1"


def test_decide_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": "insufficient credits"})

    with pytest.raises(JevError, match="402") as exc:
        _client(handler).decide("hello", {"flag": noul("Is this urgent?")})
    assert exc.value.status_code == 402


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    with pytest.raises(JevError, match="OPENROUTER_API_KEY"):
        Jev()


def test_decide_many_resumes_successes_and_retries_errors(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["state"]["id"])
        return httpx.Response(200, json=_payload())

    out = tmp_path / "results.jsonl"
    out.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "answers": {"escalate": {"type": "noul", "noul": 0.1}}}),
                json.dumps({"id": "b", "error": "timeout"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    batch = _client(handler).decide_many(
        items,
        state_of=lambda item: item,
        questions={"escalate": noul("Escalate now?")},
        id_of=lambda item: item["id"],
        out_path=out,
        concurrency=2,
    )

    assert calls == ["b", "c"] or calls == ["c", "b"]
    assert batch.ok == 2
    assert batch.errors == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["a", "b", "b", "c"]
    assert "error" not in rows[-1]
