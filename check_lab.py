import json
import os
import glob
import sys


def validate_lab():
    print("[CHECK] Dang kiem tra dinh dang bai nop...")
    failed = False

    required_files = [
        "reports/summary.json",
        "reports/benchmark_results.json",
        "reports/summary_mock.json",
        "reports/summary_retrieval.json",
        "analysis/failure_analysis.md",
    ]

    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] Tim thay: {file_path}")
        else:
            print(f"[MISS] Thieu file: {file_path}")
            missing.append(file_path)

    if missing:
        print(f"\n[FAIL] Thieu {len(missing)} file. Hay bo sung truoc khi nop bai.")
        sys.exit(1)

    reflection_files = [
        path
        for path in glob.glob("analysis/reflections/reflection_*.md")
        if not path.endswith("reflection_template.md")
    ]
    if reflection_files:
        print(f"[OK] Tim thay reflection ca nhan: {len(reflection_files)} file")
    else:
        print("[FAIL] Thieu reflection ca nhan trong analysis/reflections/")
        failed = True

    if os.path.exists(".env"):
        print("[FAIL] Phat hien .env trong repo. Khong nen nop API key.")
        failed = True
    else:
        print("[OK] Khong thay file .env")

    try:
        with open("reports/summary.json", "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except json.JSONDecodeError as exc:
        print(f"[FAIL] File reports/summary.json khong phai JSON hop le: {exc}")
        return

    try:
        with open("reports/benchmark_results.json", "r", encoding="utf-8") as file_obj:
            benchmark_results = json.load(file_obj)
    except json.JSONDecodeError as exc:
        print(f"[FAIL] File reports/benchmark_results.json khong phai JSON hop le: {exc}")
        return

    if "metrics" not in data or "metadata" not in data:
        print("[FAIL] File summary.json thieu truong 'metrics' hoac 'metadata'.")
        sys.exit(1)

    metrics = data["metrics"]
    metadata = data["metadata"]
    cost_report = data.get("cost_report", {})
    regression = data.get("regression", {})

    print("\n--- Thong ke nhanh ---")
    print(f"Tong so cases: {metadata.get('total', 'N/A')}")
    print(f"Agent mode: {metadata.get('agent_mode', 'N/A')}")
    print(f"Diem trung binh: {metrics.get('avg_score', 0):.2f}")

    if "hit_rate" in metrics:
        print(f"[OK] Da tim thay Retrieval Metrics (Hit Rate: {metrics['hit_rate']*100:.1f}%)")
    else:
        print("[FAIL] Thieu Retrieval Metrics (hit_rate).")
        failed = True

    if "mrr" in metrics:
        print(f"[OK] Da tim thay MRR ({metrics['mrr']:.3f})")
    else:
        print("[FAIL] Thieu Retrieval Metrics (mrr).")
        failed = True

    if "agreement_rate" in metrics:
        print(f"[OK] Da tim thay Multi-Judge Metrics (Agreement Rate: {metrics['agreement_rate']*100:.1f}%)")
    else:
        print("[FAIL] Thieu Multi-Judge Metrics (agreement_rate).")
        failed = True

    pipeline_modules = metadata.get("pipeline_modules", [])
    required_modules = ["engine.retrieval_eval", "engine.llm_judge", "engine.runner"]
    missing_modules = [module for module in required_modules if module not in pipeline_modules]
    if not missing_modules:
        print("[OK] Pipeline modules day du: retrieval_eval, llm_judge, runner")
    else:
        print(f"[FAIL] Thieu pipeline modules: {missing_modules}")
        failed = True

    if metadata.get("agent_mode"):
        print("[OK] Da tim thay agent_mode trong summary")
    else:
        print("[FAIL] Thieu agent_mode trong summary")
        failed = True

    if metadata.get("async_enabled") is True or metadata.get("runner_mode") == "async":
        print("[OK] Async runner metadata hop le")
    else:
        print("[FAIL] Thieu async runner metadata")
        failed = True

    if regression.get("decision") and {"delta_score", "delta_hit_rate", "delta_latency"}.issubset(regression):
        print(f"[OK] Regression gate: {regression['decision']}")
    else:
        print("[FAIL] Thieu regression gate/delta analysis")
        failed = True

    required_cost_fields = {"total_estimated_cost_usd", "cost_per_eval_usd", "estimated_savings_pct"}
    if required_cost_fields.issubset(cost_report):
        print(
            f"[OK] Cost report day du (cost/eval=${cost_report['cost_per_eval_usd']}, savings={cost_report['estimated_savings_pct']}%)"
        )
        if cost_report["estimated_savings_pct"] < 30:
            print("[FAIL] Cost saving proposal < 30%")
            failed = True
    else:
        print("[FAIL] Thieu cost metrics quan trong trong summary")
        failed = True

    if metadata.get("agent_mode") == "retrieval":
        if metadata.get("generator_type"):
            print(f"[OK] Generator type: {metadata['generator_type']}")
        else:
            print("[WARN] Thieu generator_type trong retrieval summary")

        has_doc_id = any(
            isinstance(context, dict) and context.get("doc_id")
            for result in benchmark_results
            for context in result.get("retrieved_contexts", [])
        )
        if has_doc_id:
            print("[OK] retrieved_contexts co doc_id that")
        else:
            print("[WARN] retrieval mode nhung chua thay doc_id trong retrieved_contexts")

    if metadata.get("version"):
        print("[OK] Da tim thay thong tin phien ban Agent (Regression Mode)")

    with open("analysis/failure_analysis.md", "r", encoding="utf-8") as file_obj:
        failure_report = file_obj.read().lower()
    if "failure clustering" in failure_report and "5 whys" in failure_report:
        print("[OK] failure_analysis co Failure Clustering va 5 Whys")
    else:
        print("[FAIL] failure_analysis thieu Failure Clustering hoac 5 Whys")
        failed = True

    if failed:
        print("\n[FAIL] Bai lab con thieu requirement quan trong.")
        sys.exit(1)

    print("\n[READY] Bai lab da san sang de cham diem!")


if __name__ == "__main__":
    validate_lab()
