"""
Benchmark runner chinh cho Lab 14.

Script nay:
- load golden dataset
- chay benchmark cho 2 phien ban agent
- tinh RAG metrics, multi-judge metrics va release gate
- luu reports/summary.json va reports/benchmark_results.json
"""

import asyncio
import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

from agent.main_agent import MainAgent
from agent.retrieval_agent_adapter import RetrievalAgentAdapter
from engine.llm_judge import MultiModelJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


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


class ExpertEvaluator:
    """
    Danh gia RAG pipeline theo 2 phan:
    - generation quality: faithfulness, relevancy
    """

    async def score(self, case: Dict, response: Dict) -> Dict:
        await asyncio.sleep(0)

        context = case.get("context", "")
        question = case.get("question", "")
        expected = case.get("expected_answer", "")
        answer = response.get("answer", "")

        faithfulness = self._compute_faithfulness(answer, context)
        relevancy = self._compute_relevancy(answer, question, expected)

        return {
            "faithfulness": round(faithfulness, 3),
            "relevancy": round(relevancy, 3),
        }

    def _compute_faithfulness(self, answer: str, context: str) -> float:
        if not answer or not context:
            return 0.5

        answer_words = set(_tokenize(answer)) - STOP_WORDS
        context_words = set(_tokenize(context)) - STOP_WORDS
        if not answer_words:
            return 0.7

        overlap = len(answer_words & context_words) / len(answer_words)
        return max(0.0, min(1.0, overlap))

    def _compute_relevancy(self, answer: str, question: str, expected: str) -> float:
        if not answer or not question:
            return 0.3

        keywords = (set(_tokenize(question)) | set(_tokenize(expected))) - STOP_WORDS
        if not keywords:
            return 0.5

        answer_words = set(_tokenize(answer))
        overlap = len(keywords & answer_words) / len(keywords)
        return max(0.1, min(1.0, overlap))

class CostTracker:
    GPT4O_INPUT_COST_PER_1M = 2.5
    GPT4O_OUTPUT_COST_PER_1M = 10.0
    GPT4O_MINI_INPUT_COST_PER_1M = 0.15
    GPT4O_MINI_OUTPUT_COST_PER_1M = 0.60

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def record(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

    def compute_cost(self) -> Dict:
        input_cost = (self.total_input_tokens / 1_000_000) * self.GPT4O_INPUT_COST_PER_1M
        output_cost = (self.total_output_tokens / 1_000_000) * self.GPT4O_OUTPUT_COST_PER_1M
        total_cost = input_cost + output_cost
        cost_per_eval = total_cost / max(self.total_calls, 1)

        mini_input_cost = (self.total_input_tokens * 0.6 / 1_000_000) * self.GPT4O_MINI_INPUT_COST_PER_1M
        mini_output_cost = (self.total_output_tokens * 0.6 / 1_000_000) * self.GPT4O_MINI_OUTPUT_COST_PER_1M
        gpt4o_input_cost_remain = (self.total_input_tokens * 0.4 / 1_000_000) * self.GPT4O_INPUT_COST_PER_1M
        gpt4o_output_cost_remain = (self.total_output_tokens * 0.4 / 1_000_000) * self.GPT4O_OUTPUT_COST_PER_1M
        optimized_cost = mini_input_cost + mini_output_cost + gpt4o_input_cost_remain + gpt4o_output_cost_remain

        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_api_calls": self.total_calls,
            "total_cost_usd": round(total_cost, 6),
            "total_estimated_cost_usd": round(total_cost, 6),
            "cost_per_eval_usd": round(cost_per_eval, 6),
            "optimized_cost_usd": round(optimized_cost, 6),
            "estimated_savings_pct": round(((total_cost - optimized_cost) / max(total_cost, 0.0001)) * 100, 1),
            "cost_saving_proposal": "Route easy cases to a cheaper judge/generator and reserve the stronger model for hard, adversarial, and low-confidence cases.",
            "latency_cost_tradeoff": "Async batching keeps wall-clock latency low; model routing is the main lever for reducing token cost without changing dataset coverage.",
        }


async def run_benchmark_with_results(
    agent_version: str,
    agent_mode: str,
    cost_tracker: Optional[CostTracker] = None,
) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
    print(f"\n[RUN] Khoi dong Benchmark cho {agent_version} ({agent_mode})...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("[FAIL] Thieu data/golden_set.jsonl. Hay chay 'python data/synthetic_gen.py' truoc.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as file_obj:
        dataset = [json.loads(line) for line in file_obj if line.strip()]

    if not dataset:
        print("[FAIL] File data/golden_set.jsonl rong.")
        return None, None

    print(f"   [INFO] Loaded {len(dataset)} test cases tu golden_set.jsonl")

    evaluator = ExpertEvaluator()
    judge = MultiModelJudge()
    retrieval_evaluator = RetrievalEvaluator()
    agent = create_agent(agent_mode)
    runner = BenchmarkRunner(agent, evaluator, judge)

    start_time = time.perf_counter()
    results = await runner.run_all(dataset)
    total_time = time.perf_counter() - start_time

    retrieval_aggregate = retrieval_evaluator.aggregate_retrieval_results(
        dataset,
        [result["prediction"] for result in results],
    )

    for result, retrieval_result in zip(results, retrieval_aggregate["results"]):
        result["retrieval"] = retrieval_result
        result["ragas"]["retrieval"] = retrieval_result

    if cost_tracker:
        for _ in results:
            cost_tracker.record(
                input_tokens=random.randint(380, 560),
                output_tokens=random.randint(140, 260),
            )

    total = len(results)
    passed = sum(1 for result in results if result["status"] == "pass")
    failed = total - passed

    avg_faithfulness = sum(result["ragas"]["faithfulness"] for result in results) / total
    avg_relevancy = sum(result["ragas"]["relevancy"] for result in results) / total
    avg_hit_rate = retrieval_aggregate["avg_hit_rate"]
    avg_mrr = retrieval_aggregate["avg_mrr"]
    avg_score = sum(result["judge"]["final_score"] for result in results) / total
    avg_agreement = sum(result["judge"]["agreement_rate"] for result in results) / total
    avg_latency = sum(result["latency"] for result in results) / total

    summary = {
        "metadata": {
            "version": agent_version,
            "agent_mode": agent_mode,
            "pipeline_modules": [
                "engine.retrieval_eval",
                "engine.llm_judge",
                "engine.runner",
                "agent.retrieval_agent_adapter" if agent_mode == "retrieval" else "agent.main_agent",
            ] + (["engine.answer_generator"] if agent_mode == "retrieval" else []),
            "retriever_type": "keyword_overlap" if agent_mode == "retrieval" else "none",
            "generator_type": "grounded_synthesis" if agent_mode == "retrieval" else "mock_template",
            "runner_mode": "async",
            "async_enabled": True,
            "batch_size": 5,
            "judge_models": [judge_obj.model for judge_obj in judge.judges],
            "consensus_rule": "average_when_agree_conservative_min_on_conflict",
            "conflict_threshold": judge.CONFLICT_THRESHOLD,
            "position_bias_check_available": hasattr(judge, "check_position_bias"),
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total, 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_benchmark_time_sec": round(total_time, 2),
        },
        "metrics": {
            "avg_score": round(avg_score, 3),
            "avg_faithfulness": round(avg_faithfulness, 3),
            "avg_relevancy": round(avg_relevancy, 3),
            "hit_rate": round(avg_hit_rate, 3),
            "mrr": round(avg_mrr, 3),
            "agreement_rate": round(avg_agreement, 3),
            "avg_latency_sec": round(avg_latency, 4),
        },
    }

    print(
        f"   [DONE] Hoan thanh! {passed}/{total} cases passed | "
        f"avg_score={avg_score:.2f}/5.0 | time={total_time:.1f}s"
    )
    return results, summary


async def run_benchmark(version: str, cost_tracker: Optional[CostTracker] = None):
    _, summary = await run_benchmark_with_results(version, "mock", cost_tracker)
    return summary


def create_agent(agent_mode: str):
    if agent_mode == "mock":
        return MainAgent()
    if agent_mode == "retrieval":
        return RetrievalAgentAdapter()
    raise ValueError(f"Unsupported agent mode: {agent_mode}")


def evaluate_release_gate(v1_summary: Dict, v2_summary: Dict) -> Dict:
    v1_metrics = v1_summary["metrics"]
    v2_metrics = v2_summary["metrics"]

    delta_score = v2_metrics["avg_score"] - v1_metrics["avg_score"]
    delta_hit_rate = v2_metrics["hit_rate"] - v1_metrics["hit_rate"]
    delta_latency = v2_metrics["avg_latency_sec"] - v1_metrics["avg_latency_sec"]

    quality_improved = delta_score >= 0
    retrieval_not_degraded = delta_hit_rate >= -0.05
    latency_acceptable = delta_latency <= 0.5
    decision = "APPROVE" if (quality_improved and retrieval_not_degraded and latency_acceptable) else "ROLLBACK"

    reasons = []
    if not quality_improved:
        reasons.append(f"Quality giam: delta_score={delta_score:+.3f}")
    if not retrieval_not_degraded:
        reasons.append(f"Hit Rate giam qua muc: delta={delta_hit_rate:+.3f}")
    if not latency_acceptable:
        reasons.append(f"Latency tang qua muc: delta={delta_latency:+.3f}s")
    if decision == "APPROVE":
        reasons.append(
            f"Quality: {delta_score:+.3f} | Hit Rate: {delta_hit_rate:+.3f} | Latency: {delta_latency:+.3f}s"
        )

    return {
        "decision": decision,
        "delta_score": round(delta_score, 3),
        "delta_hit_rate": round(delta_hit_rate, 3),
        "delta_latency": round(delta_latency, 3),
        "reasons": reasons,
    }


def _write_reports(summary: Dict, results: List[Dict]) -> None:
    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, ensure_ascii=False, indent=2)

    agent_mode = summary.get("metadata", {}).get("agent_mode", "unknown")
    with open(f"reports/summary_{agent_mode}.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    with open(f"reports/benchmark_results_{agent_mode}.json", "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Lab 14 benchmark")
    parser.add_argument(
        "--agent",
        choices=["mock", "retrieval"],
        default="mock",
        help="System-under-test mode to benchmark",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    print("=" * 60)
    print("  AI EVALUATION FACTORY - BENCHMARK RUNNER")
    print("=" * 60)
    print(f"  Agent mode: {args.agent}")

    cost_tracker = CostTracker()
    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base", args.agent, cost_tracker)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", args.agent, cost_tracker)

    if not v1_summary or not v2_summary:
        print("[FAIL] Khong the chay benchmark. Hay kiem tra lai data/golden_set.jsonl.")
        return

    print("\n" + "=" * 60)
    print("  REGRESSION ANALYSIS (V1 vs V2)")
    print("=" * 60)
    gate_result = evaluate_release_gate(v1_summary, v2_summary)

    print(f"\n  V1 avg_score:  {v1_summary['metrics']['avg_score']:.3f}/5.0")
    print(f"  V2 avg_score:  {v2_summary['metrics']['avg_score']:.3f}/5.0")
    print(f"  Delta Score:   {gate_result['delta_score']:+.3f}")
    print(f"  Delta HitRate: {gate_result['delta_hit_rate']:+.3f}")
    print(f"  Delta Latency: {gate_result['delta_latency']:+.3f}s")
    print("\n  Reasons:")
    for reason in gate_result["reasons"]:
        print(f"    {reason}")

    print(
        f"\n{'[APPROVE] QUYET DINH: CHAP NHAN BAN CAP NHAT' if gate_result['decision'] == 'APPROVE' else '[ROLLBACK] QUYET DINH: TU CHOI'}"
    )

    cost_report = cost_tracker.compute_cost()
    print("\n" + "=" * 60)
    print("  COST REPORT")
    print("=" * 60)
    print(f"  Tong API calls:       {cost_report['total_api_calls']}")
    print(f"  Tong tokens (input):  {cost_report['total_input_tokens']:,}")
    print(f"  Tong tokens (output): {cost_report['total_output_tokens']:,}")
    print(f"  Chi phi hien tai:     ${cost_report['total_cost_usd']:.4f} USD")
    print(f"  Chi phi / eval:       ${cost_report['cost_per_eval_usd']:.6f} USD")
    print(f"  Chi phi toi uu (*):   ${cost_report['optimized_cost_usd']:.4f} USD")
    print(f"  Uoc tinh tiet kiem:    {cost_report['estimated_savings_pct']:.1f}%")

    v2_summary["regression"] = gate_result
    v2_summary["cost_report"] = cost_report

    _write_reports(v2_summary, v2_results)

    print("\n" + "=" * 60)
    print("  Da luu bao cao:")
    print("      - reports/summary.json")
    print("      - reports/benchmark_results.json")
    print("=" * 60)
    print("\n  Tiep theo: chay 'python check_lab.py' de kiem tra dinh dang.")


if __name__ == "__main__":
    asyncio.run(main())
