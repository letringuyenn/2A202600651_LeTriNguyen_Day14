"""
Grounded answer generation for retrieval mode.

The generator only uses retrieved contexts. It does not read expected answers
or ground-truth ids, so benchmark metrics still reflect the agent output.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


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


SAFETY_MARKERS = [
    "mat khau",
    "admin",
    "hack",
    "doc hai",
    "ignore previous",
    "bo qua moi huong dan",
    "danh cap",
]


def tokenize(text: str) -> List[str]:
    tokens = [token.strip(".,:;!?\"'()[]{}").lower() for token in text.split()]
    return [token for token in tokens if token and token not in STOP_WORDS]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


class GroundedAnswerGenerator:
    generator_type = "grounded_synthesis"

    def __init__(self, min_confidence: float = 0.08, max_contexts: int = 3):
        self.min_confidence = min_confidence
        self.max_contexts = max_contexts

    def _is_unsafe_or_injection(self, question: str) -> bool:
        lowered = question.lower()
        return any(marker in lowered for marker in SAFETY_MARKERS)

    def _has_enough_evidence(self, contexts: Sequence[Dict[str, Any]]) -> bool:
        if not contexts:
            return False
        return float(contexts[0].get("score", 0.0)) >= self.min_confidence

    def _sentence_score(self, question_terms: set[str], sentence: str) -> float:
        sentence_terms = set(tokenize(sentence))
        if not question_terms or not sentence_terms:
            return 0.0
        overlap = len(question_terms & sentence_terms)
        return overlap / len(question_terms)

    def _evidence_sentences(self, question_terms: set[str], context: Dict[str, Any]) -> List[str]:
        sentences = split_sentences(str(context.get("text", "")))
        if not sentences:
            snippet = str(context.get("snippet", "")).strip()
            return [snippet] if snippet else []

        scored = [
            (index, self._sentence_score(question_terms, sentence), sentence)
            for index, sentence in enumerate(sentences)
        ]
        matching = [item for item in scored if item[1] > 0]
        if not matching:
            scored.sort(key=lambda item: (-item[1], item[0]))
            return [scored[0][2]]

        matching.sort(key=lambda item: item[0])
        return [item[2] for item in matching[:3]]

    def _select_contexts(self, contexts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not contexts:
            return []
        top_score = float(contexts[0].get("score", 0.0))
        threshold = max(self.min_confidence, top_score * 0.35)
        selected = [
            context
            for context in contexts
            if float(context.get("score", 0.0)) >= threshold
        ]
        return selected[: self.max_contexts]

    def generate_grounded_answer(
        self,
        question: str,
        retrieved_contexts: Sequence[Dict[str, Any]],
    ) -> str:
        if self._is_unsafe_or_injection(question):
            return "Toi tu choi thuc hien yeu cau nay vi no co the vi pham an toan hoac bao mat."

        if not self._has_enough_evidence(retrieved_contexts):
            return "Toi khong co du bang chung trong cac tai lieu da truy xuat de tra loi cau hoi nay."

        question_terms = set(tokenize(question))
        selected_contexts = self._select_contexts(retrieved_contexts)
        evidence_sentences: List[str] = []
        seen = set()

        for context in selected_contexts:
            for sentence in self._evidence_sentences(question_terms, context):
                if sentence and sentence not in seen:
                    evidence_sentences.append(sentence)
                    seen.add(sentence)

        if not evidence_sentences:
            return "Toi khong co du bang chung trong cac tai lieu da truy xuat de tra loi cau hoi nay."

        if len(evidence_sentences) == 1:
            return evidence_sentences[0]

        return " ".join(evidence_sentences)
