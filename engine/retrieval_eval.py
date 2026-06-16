"""
Retrieval evaluation utilities for Lab 14.

This module is intentionally independent from the benchmark orchestration
so retrieval metrics can be reused, tested, and replaced without touching
the top-level pipeline.

Architecture note: the interface shape was adapted from the team/reference
implementation pattern, but the logic here is rewritten to fit this repo's
current mock-agent data contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


STOP_WORDS = {
    "la",
    "cua",
    "va",
    "trong",
    "den",
    "cho",
    "voi",
    "mot",
    "cac",
    "co",
    "duoc",
    "nay",
    "nhung",
    "toi",
    "ban",
    "o",
    "de",
    "hay",
    "hoac",
    "ma",
    "nhu",
    "thi",
    "tu",
    "ra",
    "gi",
    "the",
    "nao",
    "khong",
    "sao",
    "khi",
    "noi",
    "voi",
}


def _tokenize(text: str) -> List[str]:
    return [token.strip(".,:;!?\"'()[]{}").lower() for token in text.split() if token.strip()]


def compute_hit_rate(expected_ids: Sequence[str], retrieved_ids: Sequence[str], top_k: int = 3) -> float:
    """
    Return 1.0 if at least one expected_id appears in the top-k retrieved_ids.
    """
    top_retrieved = list(retrieved_ids)[:top_k]
    return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0


def compute_mrr(expected_ids: Sequence[str], retrieved_ids: Sequence[str]) -> float:
    """
    Compute Mean Reciprocal Rank for the first expected id appearing in retrieved_ids.
    """
    for index, doc_id in enumerate(retrieved_ids):
        if doc_id in expected_ids:
            return 1.0 / (index + 1)
    return 0.0


def _infer_retrieved_doc_id(text: str) -> str:
    tokens = _tokenize(text)
    for token in tokens:
        if token.startswith("doc_"):
            return token
    return ""


def _extract_expected_ids(case: Dict[str, Any]) -> List[str]:
    metadata = case.get("metadata", {})
    ground_truth_id = metadata.get("ground_truth_id") or metadata.get("doc_id")
    if ground_truth_id:
        return [str(ground_truth_id)]
    return []


def _extract_retrieved_ids(retrieved_contexts: Iterable[Any]) -> List[str]:
    retrieved_ids: List[str] = []
    for context in retrieved_contexts:
        if isinstance(context, dict):
            doc_id = str(context.get("doc_id", ""))
        else:
            doc_id = _infer_retrieved_doc_id(str(context))
        if doc_id:
            retrieved_ids.append(doc_id)
    return retrieved_ids


@dataclass
class RetrievalResult:
    hit_rate: float
    mrr: float
    retrieved_count: int
    ground_truth_ids: List[str]
    retrieved_ids: List[str]
    matched: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hit_rate": round(self.hit_rate, 3),
            "mrr": round(self.mrr, 3),
            "retrieved_count": self.retrieved_count,
            "ground_truth_ids": self.ground_truth_ids,
            "retrieved_ids": self.retrieved_ids,
            "matched": self.matched,
        }


class RetrievalEvaluator:
    """
    Evaluate retrieval quality using either explicit doc ids or inferred ids.

    The current mock agent does not expose a real retrieval backend, so the
    metrics reflect the present repository state: if no document ids are
    surfaced in retrieved_contexts, hit rate stays at 0.
    """

    def evaluate_retrieval(
        self,
        case: Dict[str, Any],
        prediction: Dict[str, Any],
        retrieved_contexts: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        retrieved_contexts = list(
            retrieved_contexts
            or prediction.get("retrieved_contexts", [])
            or prediction.get("contexts", [])
            or []
        )
        expected_ids = _extract_expected_ids(case)
        retrieved_ids = _extract_retrieved_ids(retrieved_contexts)

        hit_rate = compute_hit_rate(expected_ids, retrieved_ids)
        mrr = compute_mrr(expected_ids, retrieved_ids)
        matched = bool(hit_rate)

        result = RetrievalResult(
            hit_rate=hit_rate,
            mrr=mrr,
            retrieved_count=len(retrieved_contexts),
            ground_truth_ids=expected_ids,
            retrieved_ids=retrieved_ids,
            matched=matched,
        )
        return result.as_dict()

    def compute_retrieval_metrics(
        self,
        case: Dict[str, Any],
        prediction: Dict[str, Any],
        retrieved_contexts: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        return self.evaluate_retrieval(case, prediction, retrieved_contexts)

    def aggregate_retrieval_results(
        self,
        cases: Sequence[Dict[str, Any]],
        predictions: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        per_case = [
            self.evaluate_retrieval(case, prediction)
            for case, prediction in zip(cases, predictions)
        ]

        total = len(per_case)
        if total == 0:
            return {
                "avg_hit_rate": 0.0,
                "avg_mrr": 0.0,
                "total_cases": 0,
                "passed_cases": 0,
                "failed_cases": 0,
                "results": [],
            }

        avg_hit_rate = sum(item["hit_rate"] for item in per_case) / total
        avg_mrr = sum(item["mrr"] for item in per_case) / total
        passed_cases = sum(1 for item in per_case if item["matched"])
        failed_cases = total - passed_cases

        return {
            "avg_hit_rate": round(avg_hit_rate, 3),
            "avg_mrr": round(avg_mrr, 3),
            "total_cases": total,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "results": per_case,
        }
