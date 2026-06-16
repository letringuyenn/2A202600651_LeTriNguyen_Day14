"""
Deterministic retrieval agent adapter for the benchmark system-under-test.

This adapter does not use a vector database yet. It builds a local corpus from
the generated golden dataset, scores documents with keyword overlap, and returns
the same benchmark contract as MainAgent plus structured retrieved_contexts.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from engine.answer_generator import GroundedAnswerGenerator


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
    "hay",
}


def _tokenize(text: str) -> List[str]:
    tokens = [token.strip(".,:;!?\"'()[]{}").lower() for token in text.split()]
    return [token for token in tokens if token and token not in STOP_WORDS]


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    text: str
    source: str


class RetrievalAgentAdapter:
    def __init__(self, corpus_path: str = "data/golden_set.jsonl", top_k: int = 3):
        self.name = "RetrievalAgentAdapter-v1"
        self.corpus_path = corpus_path
        self.top_k = top_k
        self.corpus = self._load_corpus(corpus_path)
        self.answer_generator = GroundedAnswerGenerator()

    def _load_corpus(self, corpus_path: str) -> List[CorpusDocument]:
        if not os.path.exists(corpus_path):
            return []

        by_doc_id: Dict[str, CorpusDocument] = {}
        with open(corpus_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue
                item = json.loads(line)
                metadata = item.get("metadata", {})
                doc_id = metadata.get("doc_id") or metadata.get("ground_truth_id")
                context = item.get("context", "")
                if not doc_id or not context or doc_id in by_doc_id:
                    continue
                by_doc_id[doc_id] = CorpusDocument(
                    doc_id=str(doc_id),
                    text=context,
                    source=f"{corpus_path}#{doc_id}",
                )
        return list(by_doc_id.values())

    def _score(self, query: str, document: CorpusDocument) -> float:
        query_terms = set(_tokenize(query))
        doc_terms = set(_tokenize(document.text))
        if not query_terms or not doc_terms:
            return 0.0

        overlap = len(query_terms & doc_terms)
        coverage = overlap / len(query_terms)
        density = overlap / len(doc_terms)
        return round((coverage * 0.8) + (density * 0.2), 4)

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        scored = [
            {
                "doc_id": document.doc_id,
                "text": document.text,
                "snippet": document.text[:240],
                "score": self._score(question, document),
                "source": document.source,
            }
            for document in self.corpus
        ]
        scored.sort(key=lambda item: (-item["score"], item["doc_id"]))
        return scored[: self.top_k]

    def generate_grounded_answer(self, question: str, contexts: Sequence[Dict[str, Any]]) -> str:
        return self.answer_generator.generate_grounded_answer(question, contexts)

    async def query(self, question: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        await asyncio.sleep(0)

        retrieved_contexts = self.retrieve(question)
        answer = self.generate_grounded_answer(question, retrieved_contexts)
        latency = time.perf_counter() - start_time

        return {
            "answer": answer,
            "response": answer,
            "contexts": [item["text"] for item in retrieved_contexts],
            "retrieved_contexts": retrieved_contexts,
            "latency": latency,
            "metadata": {
                "agent_mode": "retrieval",
                "retriever": "keyword_overlap",
                "generator_type": self.answer_generator.generator_type,
                "top_k": self.top_k,
                "tokens_used": len(_tokenize(question)) + len(_tokenize(answer)),
                "sources": [item["source"] for item in retrieved_contexts],
            },
        }
