# Individual Reflection - letringuyenn

## Thong tin

- Ho va ten: Le Tri Nguyen
- Ma sinh vien: 2A202600651
- Lab: Lab 14 - AI Evaluation Factory
- Ngay: 2026-06-16

## 1. Nhiem vu cua toi

Toi phu trach phan baseline evaluation infrastructure trong repo nay, cu the la:

- viet lai `data/synthetic_gen.py` de sinh golden dataset day du hon
- nang cap `engine/llm_judge.py` thanh multi-judge consensus engine
- cai tien `main.py` de chay benchmark, regression gate va cost report
- sua loi run/check do console encoding tren Windows
- cap nhat cac file bao cao de phu hop voi ket qua benchmark that su

## 2. Phan da lam

- Toi thay the bo sinh du lieu placeholder bang script tao 51 test cases, bao gom fact-check, adversarial, edge-case, multi-turn, reasoning va multi-hop.
- Toi viet lai multi-judge de co 2 judge, co agreement rate, co xu ly xung dot va co kiem tra position bias.
- Toi cai tien benchmark runner de luu `reports/summary.json` va `reports/benchmark_results.json`.
- Toi tach retrieval evaluation ra thanh module rieng va noi lai vao flow benchmark.
- Toi them `RetrievalAgentAdapter` deterministic va CLI `--agent mock|retrieval` de benchmark do duoc system-under-test that hon.
- Toi chay benchmark that su tren repo hien tai va cap nhat group report theo so lieu moi sinh ra.
- Toi them `analysis/reflections/reflection_template.md` de giu dung cau truc nop bai.

## 3. Loi gap

- Lan dau chay `python main.py`, app bi loi ma hoa do console Windows khong in duoc emoji.
- `python check_lab.py` cung gap loi tuong tu vi output mac dinh co emoji.
- Dataset ban dau khong du 50 cases, nen phai bo sung logic sinh golden set.

## 4. Cach verify

- Toi kiem tra cu phap bang:

```bash
python -m py_compile main.py engine/runner.py engine/retrieval_eval.py engine/llm_judge.py check_lab.py agent/retrieval_agent_adapter.py engine/answer_generator.py
```

- Toi chay tao dataset:

```bash
python data/synthetic_gen.py
```

- Toi chay benchmark:

```bash
python main.py --agent mock
python main.py --agent retrieval
```

- Toi chay kiem tra bai nop:

```bash
python check_lab.py
```

## 5. Test result

- `py_compile`: pass
- `data/synthetic_gen.py`: tao thanh cong 51 test cases
- `main.py`: chay thanh cong, tao report va regression gate
- `check_lab.py`: pass
- retrieval_eval: da duoc module hoa va noi vao pipeline
- retrieval agent mode: chay thanh cong va tra `retrieved_contexts` co `doc_id`, `text`, `score`, `source`
- Benchmark result:
  - 10/51 cases pass
  - avg_score = 2.798/5.0
  - hit_rate = 0.000
  - agreement_rate = 1.000
  - avg_latency = 0.5072 s
- Retrieval benchmark result:
  - 34/51 cases pass
  - avg_score = 3.464/5.0
  - hit_rate = 0.843
  - MRR = 0.729
  - avg_latency = 0.0018 s

## 6. Demo evidence

- File report: `reports/summary.json`
- File result: `reports/benchmark_results.json`
- Group report da cap nhat: `analysis/failure_analysis.md`
- Output run that su tu `python main.py`
- Output kiem tra that su tu `python check_lab.py`
- Phan interface retrieval module duoc adapt tu baseline cua team/reference, nhung logic da duoc viet lai de khop data contract hien tai.

## 7. Bai hoc

- Toi hoc duoc rang benchmark infrastructure co the on dinh ngay ca khi agent con la mock, nhung phan score se phan anh trung thuc hon neu agent co retrieval that su.
- Toi cung thay rang report khong nen viet chung chung: phai dua tren output that, so that, va phan tich ro bottleneck.
- Mot loi nho ve encoding co the lam dut pipeline, nen khi chay tren Windows phai chu y output ASCII neu khong can emoji.
