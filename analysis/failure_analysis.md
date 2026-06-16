# Bao cao phan tich benchmark

## 1. Tong quan

Benchmark hien co 2 agent mode:
- `mock`: baseline giu nguyen `MainAgent`, dung de so sanh hoi quy.
- `retrieval`: `RetrievalAgentAdapter` moi, doc corpus tu `data/golden_set.jsonl`, cham diem tai lieu bang keyword overlap va tra ve `retrieved_contexts` co `doc_id`, `text`, `score`, `source`.

Retrieval mode khong dung vector database va khong hardcode `expected_doc_id`. Hit rate duoc tinh tu `doc_id` ma retriever thuc su tra ve.

## 2. So sanh ket qua that

| Metric | mock mode | retrieval mode |
|---|---:|---:|
| Total cases | 51 | 51 |
| Pass / Fail | 10 / 41 | 34 / 17 |
| Pass rate | 19.6% | 66.7% |
| Avg score | 2.798 / 5.0 | 3.464 / 5.0 |
| Avg faithfulness | 0.135 | 0.487 |
| Avg relevancy | 0.426 | 0.562 |
| Hit rate | 0.000 | 0.843 |
| MRR | 0.000 | 0.729 |
| Agreement rate | 1.000 | 1.000 |
| Avg latency | 0.5114 s | 0.0019 s |
| Benchmark time | 6.92 s | 1.26 s |
| Regression gate | APPROVE | APPROVE |

Ket qua moi cho thay benchmark da bat dau do duoc system-under-test co retrieval that hon. `hit_rate` tang tu 0.000 len 0.843 vi retrieval mode tra ve `doc_id` that trong `retrieved_contexts`. Sau khi them `GroundedAnswerGenerator`, pass count tang tu 28 len 34 va avg_score tang tu 3.222 len 3.464.

## 3. Vi sao mock mode thap

`MainAgent` baseline van tra ve template chung va context string khong co `doc_id`. Vi vay:
- retrieval evaluator khong co document id de doi chieu voi golden set
- answer khong du grounded vao context cua test case
- faithfulness va hit rate thap la ket qua dung, khong phai loi cham diem

Mock mode van duoc giu lai de lam baseline regression, khong bi xoa.

## 4. Vi sao retrieval mode tot hon

`RetrievalAgentAdapter` tao corpus local tu golden dataset, gom moi `doc_id` va context duy nhat. Khi nhan cau hoi, adapter:
- tokenize cau hoi va document
- tinh keyword overlap score
- tra top-k context object co `doc_id`, `text`, `score`, `source`
- sinh cau tra loi bang `GroundedAnswerGenerator`, lay nhieu cau lien quan tu top-k contexts hoac fallback/refusal neu cau hoi khong an toan

Nho contract moi nay, `engine.retrieval_eval` co the tinh Hit Rate va MRR tren retrieved ids that.

## 5. Multi-Judge va cost report

### Multi-Judge Consensus

| Requirement | Trang thai |
|---|---|
| Judge 1 | `gpt-4o` |
| Judge 2 | `claude-3-5-sonnet` |
| Agreement Rate | 1.000 |
| Consensus rule | Lay trung binh khi 2 judge dong thuan; neu conflict thi dung conservative min |
| Conflict threshold | > 1.0 diem |
| Position-bias check | Co ham `check_position_bias()` trong `engine.llm_judge.MultiModelJudge` |

### Cost va hieu nang

| Metric | retrieval mode |
|---|---:|
| total_estimated_cost_usd | 0.331755 |
| cost_per_eval_usd | 0.003253 |
| optimized_cost_usd | 0.144645 |
| estimated_savings_pct | 56.4% |
| avg_latency_sec | 0.0019 |

De xuat giam chi phi: route easy cases sang model/generator re hon, chi dung judge/generator manh cho hard, adversarial va low-confidence cases. Uoc tinh tiet kiem hien tai la 56.4%, vuot nguong 30% trong README.

## 6. Failure Clustering

Retrieval mode van fail 17/51 cases. Phan bo fail theo loai case:

| Loai case | Pass | Fail |
|---|---:|---:|
| fact-check | 20 | 0 |
| reasoning | 6 | 7 |
| edge-case | 2 | 6 |
| adversarial | 2 | 3 |
| multi-hop | 1 | 1 |
| multi-turn | 3 | 0 |

Nguyen nhan chinh:

| Nhom loi | Nguyen nhan |
|---|---|
| Keyword retrieval chua du semantic | Cac cau hoi reasoning/multi-hop co the dung y nhung khong lap lai tu khoa trong document |
| Generation con extractive | Answer da tong hop top-k nhung van la trich cau, chua co suy luan sau retrieval |
| Out-of-scope threshold con don gian | Keyword overlap = 0 moi fallback, nen mot so edge case co keyword chung van lay context |
| Safety layer moi la rule-based | Adversarial handling tot hon mock nhung chua phai moderation/classifier that |

## 7. 5 Whys Root Cause

### Cluster 1: Retrieval miss

1. Symptom: Mot so case co `hit_rate = 0`, vi doc dung khong nam trong top-k retrieved docs.
2. Why 1: Retriever hien tai chi dung keyword overlap, khong hieu semantic intent.
3. Why 2: Reasoning/multi-hop questions co the dung y nhung khong lap lai keyword cua document.
4. Why 3: Chua co BM25, embedding search, reranker hoac synonym expansion.
5. Why 4: Lab hien tai uu tien deterministic local retriever de benchmark chay duoc khong can API.
6. Root cause: Retrieval backend moi la keyword-overlap baseline, chua phai semantic retrieval pipeline.

### Cluster 2: Generation mismatch

1. Symptom: Co case retrieval dung doc nhung judge score van duoi nguong pass.
2. Why 1: Generator `grounded_synthesis` chi tong hop cau tu retrieved contexts, chua reasoning sau retrieval.
3. Why 2: Expected answer doi hoi giai thich ngan gon dung trong tam, con answer extractive co the thua/lech chi tiet.
4. Why 3: Chua co LLM generator hoac template rieng cho fact-check, reasoning, multi-hop.
5. Why 4: De tranh fake metric, generator khong doc `expected_answer`, nen khong the toi uu theo dap an vang.
6. Root cause: Generation layer da grounded nhung con extractive, chua du nang luc synthesis/reasoning.

### Cluster 3: Safety / out-of-scope

1. Symptom: Edge-case va adversarial van fail nhieu hon fact-check.
2. Why 1: Safety/out-of-scope handling hien la rule-based keyword list.
3. Why 2: Mot so cau hoi ngoai pham vi van co keyword chung voi corpus, nen retriever van tra context co diem.
4. Why 3: Confidence threshold con don gian, chua ket hop score margin/top-k agreement.
5. Why 4: Chua co intent classifier hoac moderation layer doc lap truoc retrieval/generation.
6. Root cause: Guardrail layer moi o muc baseline, chua du de xu ly prompt injection va ambiguity o muc production.

## 8. Ket luan

Benchmark framework hien da san sang hon de demo:
- `mock` mode chung minh baseline thap khi agent khong co retrieval that
- `retrieval` mode chung minh pipeline do duoc retrieval that voi `doc_id`, Hit Rate va MRR
- reports co `agent_mode`, `pipeline_modules`, `runner_mode=async`, retrieval metrics, benchmark metrics, regression gate va cost report

Huong tiep theo:
- thay keyword overlap bang BM25/FAISS/ChromaDB neu co thoi gian
- them reranking va confidence threshold tot hon
- thay answer template bang generator co instruction grounding
- them unit tests nho cho `RetrievalAgentAdapter.retrieve()` va `RetrievalEvaluator.evaluate_retrieval()`
