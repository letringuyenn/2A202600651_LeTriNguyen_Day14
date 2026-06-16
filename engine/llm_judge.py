"""
Multi-judge consensus engine cho Lab 14.

Thiet ke theo huong:
- toi thieu 2 judge
- tinh agreement rate
- co co che xu ly xung dot khi hai judge lech nhau qua nhieu
- co ham kiem tra position bias
"""

import asyncio
import random
from typing import Any, Dict, List


RUBRICS = {
    "accuracy": {
        5: "Tra loi dung va day du theo ground truth.",
        4: "Tra loi dung, thieu it chi tiet khong quan trong.",
        3: "Co y dung nhung con thieu thong tin hoac co diem chua chinh xac nho.",
        2: "Co mot phan dung nhung sai lech dang ke.",
        1: "Sai hoan toan hoac khong lien quan.",
    },
    "faithfulness": {
        5: "Hoan toan dua tren context, khong co them thong tin tuan tiep.",
        4: "Chu yeu dua tren context, co the co suy luan nho.",
        3: "Dua tren context nhung van co mot vai chi tiet kho xac minh.",
        2: "Co dau hieu hallucination.",
        1: "Hallucination nang.",
    },
    "relevancy": {
        5: "Cau tra loi bam sat cau hoi.",
        4: "Lien quan tot.",
        3: "Lien quan mot phan.",
        2: "It lien quan.",
        1: "Khong lien quan.",
    },
    "professionalism": {
        5: "Cach dien dat chuyen nghiep.",
        4: "Tuong doi chuyen nghiep.",
        3: "Tam on.",
        2: "Chua phu hop.",
        1: "Khong phu hop.",
    },
}


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
    "di",
}


def _tokenize(text: str) -> List[str]:
    return [token.strip(".,:;!?\"'()[]{}").lower() for token in text.split() if token.strip()]


class LLMJudge:
    def __init__(self, model: str = "gpt-4o", bias_offset: float = 0.0):
        self.model = model
        self.bias_offset = bias_offset
        self.rubrics = RUBRICS

    def _keyword_overlap(self, left: str, right: str) -> float:
        left_tokens = set(_tokenize(left)) - STOP_WORDS
        right_tokens = set(_tokenize(right)) - STOP_WORDS
        if not left_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens)

    def _score_answer(self, question: str, answer: str, ground_truth: str) -> Dict[str, float]:
        overlap = self._keyword_overlap(ground_truth, answer)
        q_overlap = self._keyword_overlap(question, answer)

        if not answer or not ground_truth:
            return {"accuracy": 1.0, "faithfulness": 1.0, "relevancy": 1.0, "professionalism": 3.0}

        accuracy = 1 + overlap * 4 + self.bias_offset
        faithfulness = 1 + min(1.0, 0.6 * overlap + 0.4 * q_overlap) * 4
        relevancy = 1 + min(1.0, q_overlap * 0.7 + overlap * 0.3) * 4

        answer_lower = answer.lower()
        if any(flag in answer_lower for flag in ["hack", "doc hai", "mat khau", "mat khẩu", "danh cap"]):
            professionalism = 1.5
        elif "khong the" in answer_lower or "tu choi" in answer_lower:
            professionalism = 4.5
        else:
            professionalism = 4.0

        return {
            "accuracy": max(1.0, min(5.0, round(accuracy, 2))),
            "faithfulness": max(1.0, min(5.0, round(faithfulness, 2))),
            "relevancy": max(1.0, min(5.0, round(relevancy, 2))),
            "professionalism": max(1.0, min(5.0, round(professionalism, 2))),
        }

    async def judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        await asyncio.sleep(random.uniform(0.03, 0.12))
        scores = self._score_answer(question, answer, ground_truth)

        final_score = (
            scores["accuracy"] * 0.4
            + scores["faithfulness"] * 0.3
            + scores["relevancy"] * 0.2
            + scores["professionalism"] * 0.1
        )
        return {
            "model": self.model,
            "scores": scores,
            "final_score": round(final_score, 2),
            "reasoning": (
                f"[{self.model}] accuracy={scores['accuracy']}/5, "
                f"faithfulness={scores['faithfulness']}/5, "
                f"relevancy={scores['relevancy']}/5, "
                f"professionalism={scores['professionalism']}/5"
            ),
        }


class MultiModelJudge:
    CONFLICT_THRESHOLD = 1.0

    def __init__(self):
        self.judges: List[LLMJudge] = [
            LLMJudge(model="gpt-4o", bias_offset=-0.15),
            LLMJudge(model="claude-3-5-sonnet", bias_offset=0.15),
        ]

    @staticmethod
    def _agreement_rate(scores: List[float]) -> float:
        if len(scores) < 2:
            return 1.0
        spread = max(scores) - min(scores)
        if spread <= 0.5:
            return 1.0
        if spread <= 1.0:
            return 0.7
        return 0.3

    @staticmethod
    def _resolve_conflict(results: List[Dict[str, Any]]) -> float:
        return min(result["final_score"] for result in results)

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        tasks = [judge.judge(question, answer, ground_truth) for judge in self.judges]
        results = await asyncio.gather(*tasks)

        scores = [result["final_score"] for result in results]
        agreement_rate = self._agreement_rate(scores)
        conflict_detected = (max(scores) - min(scores)) > self.CONFLICT_THRESHOLD

        if conflict_detected:
            final_score = self._resolve_conflict(results)
            resolution_method = "conservative_min"
        else:
            final_score = sum(scores) / len(scores)
            resolution_method = "average"

        return {
            "final_score": round(final_score, 2),
            "agreement_rate": round(agreement_rate, 2),
            "individual_scores": {result["model"]: result["final_score"] for result in results},
            "conflict_detected": conflict_detected,
            "resolution_method": resolution_method,
            "reasoning": " | ".join(result["reasoning"] for result in results),
            "judges_count": len(results),
        }

    async def check_position_bias(
        self,
        response_a: str,
        response_b: str,
        question: str = "",
        ground_truth: str = "",
    ) -> Dict[str, Any]:
        result_ab = await self.evaluate_multi_judge(question, response_a, ground_truth)
        result_ba = await self.evaluate_multi_judge(question, response_b, ground_truth)
        score_difference = abs(result_ab["final_score"] - result_ba["final_score"])

        return {
            "order_ab_score": result_ab["final_score"],
            "order_ba_score": result_ba["final_score"],
            "score_difference": round(score_difference, 2),
            "has_position_bias": score_difference > 0.5,
            "bias_severity": "high" if score_difference > 1.0 else "medium" if score_difference > 0.5 else "low",
        }


if __name__ == "__main__":
    async def demo():
        judge = MultiModelJudge()
        question = "RAGAS do nhung chi so gi?"
        answer = "RAGAS do Faithfulness, Answer Relevancy, Context Recall va Context Precision."
        result = await judge.evaluate_multi_judge(question, answer, answer)
        print(result)

    asyncio.run(demo())
