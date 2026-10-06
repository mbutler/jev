from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from jev.questions import check_questions

DEFAULT_MODEL = "~typesafe/jev-latest"
SYSTEMONE_URL = "https://openrouter.ai/api/v1/systemone"

Questions = Mapping[str, Mapping[str, Any]]
QuestionsOf = Questions | Callable[[Any], Questions]


class JevError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class Answer:
    """One typed answer. Fields that do not apply to this type are None."""

    id: str
    type: str
    raw: dict[str, Any]

    @property
    def choice(self) -> str | None:
        value = self.raw.get("choice")
        return None if value is None else str(value)

    @property
    def noul(self) -> float | None:
        value = self.raw.get("noul")
        return None if value is None else float(value)

    @property
    def score(self) -> float | None:
        value = self.raw.get("score")
        return None if value is None else float(value)

    @property
    def probabilities(self) -> dict[str, float]:
        probs = self.raw.get("probabilities") or {}
        if not isinstance(probs, dict):
            return {}
        return {str(key): float(value) for key, value in probs.items()}

    @property
    def confidence(self) -> float | None:
        """Peakedness of a choice or score distribution. Noul answers omit this."""
        value = self.raw.get("confidence")
        return None if value is None else float(value)

    @property
    def legend(self) -> dict[str, str]:
        legend = self.raw.get("legend") or {}
        if not isinstance(legend, dict):
            return {}
        return {str(key): str(value) for key, value in legend.items()}


@dataclass(frozen=True)
class Decision:
    model: str
    answers: dict[str, Answer]
    usage: dict[str, Any]
    cost: float | None
    latency_ms: float
    request_id: str | None
    provider: Any
    raw: dict[str, Any]

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        latency_ms: float,
        fallback_model: str,
    ) -> Decision:
        raw_answers = payload.get("answers") or {}
        if not isinstance(raw_answers, dict):
            raw_answers = {}
        answers = {
            str(name): Answer(
                id=str(name),
                type=str(answer.get("type") or ""),
                raw=answer if isinstance(answer, dict) else {"value": answer},
            )
            for name, answer in raw_answers.items()
        }
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        cost = usage.get("cost")
        return cls(
            model=str(payload.get("model") or fallback_model),
            answers=answers,
            usage=usage,
            cost=None if cost is None else float(cost),
            latency_ms=round(latency_ms, 2),
            request_id=None if payload.get("id") is None else str(payload.get("id")),
            provider=payload.get("provider"),
            raw=payload,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "answers": {name: answer.raw for name, answer in self.answers.items()},
            "usage": self.usage,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "request_id": self.request_id,
            "provider": self.provider,
        }


@dataclass(frozen=True)
class BatchResult:
    results: list[dict[str, Any]]
    ok: int
    errors: int
    out_path: Path


class Jev:
    """Client for Jev's System One API on OpenRouter."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        url: str = SYSTEMONE_URL,
        timeout: float = 120.0,
        referer: str | None = None,
        title: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        load_dotenv()
        resolved_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not resolved_key:
            raise JevError("Missing OPENROUTER_API_KEY")
        self.api_key = resolved_key
        self.model = (model or os.getenv("OPENROUTER_MODEL") or "").strip() or DEFAULT_MODEL
        self.url = url
        self.timeout = timeout
        self.referer = referer
        self.title = title
        self._client = client

    def decide(
        self,
        state: Any,
        questions: Questions,
        *,
        model: str | None = None,
    ) -> Decision:
        """Ask every question about one state. Questions run in a single call."""
        if state is None:
            raise ValueError("state is required")
        check_questions(questions)
        body = {
            "model": model or self.model,
            "state": state,
            "questions": {name: dict(question) for name, question in questions.items()},
        }
        started = time.perf_counter()
        response = self._post(body)
        latency_ms = (time.perf_counter() - started) * 1000
        if response.status_code >= 400:
            snippet = response.text[:500]
            raise JevError(
                f"Jev request failed ({response.status_code}): {snippet}",
                status_code=response.status_code,
                body=response.text[:2000],
            )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise JevError(
                "Jev returned non-JSON",
                status_code=response.status_code,
                body=response.text[:2000],
            ) from exc
        if not isinstance(payload, dict):
            raise JevError("Jev returned an unexpected payload", body=response.text[:2000])
        return Decision.from_payload(payload, latency_ms=latency_ms, fallback_model=body["model"])

    def decide_many(
        self,
        items: Sequence[Any],
        *,
        state_of: Callable[[Any], Any],
        questions: QuestionsOf,
        id_of: Callable[[Any], str],
        out_path: Path,
        concurrency: int = 4,
        resume: bool = True,
        limit: int | None = None,
        model: str | None = None,
    ) -> BatchResult:
        """Score many items. Successful ids already in out_path are skipped."""
        selected = list(items[:limit] if limit is not None else items)
        done = _successful_ids(out_path) if resume and out_path.exists() else set()
        pending = [item for item in selected if str(id_of(item)) not in done]

        results: list[dict[str, Any]] = []
        errors = 0
        lock = threading.Lock()

        def _run(item: Any) -> dict[str, Any]:
            item_id = str(id_of(item))
            item_questions = questions(item) if callable(questions) else questions
            try:
                decision = self.decide(state_of(item), item_questions, model=model)
            except Exception as exc:  # noqa: BLE001 — record each item failure
                return {"id": item_id, "model": model or self.model, "error": str(exc)}
            return {"id": item_id, **decision.to_dict()}

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {pool.submit(_run, item): item for item in pending}
            for future in as_completed(futures):
                row = future.result()
                with lock:
                    _append_jsonl(out_path, row)
                    results.append(row)
                if "error" in row:
                    errors += 1

        return BatchResult(
            results=results,
            ok=len(results) - errors,
            errors=errors,
            out_path=out_path,
        )

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        headers = self._headers()
        if self._client is not None:
            return self._client.post(self.url, json=body, headers=headers)
        with httpx.Client(timeout=self.timeout) as client:
            return client.post(self.url, json=body, headers=headers)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-OpenRouter-Title"] = self.title
        return headers


def _successful_ids(path: Path) -> set[str]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        item_id = row.get("id")
        if item_id:
            latest[str(item_id)] = row
    return {item_id for item_id, row in latest.items() if "error" not in row}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
