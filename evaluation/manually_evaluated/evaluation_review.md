# OTP Diákhitel RAG – Human evaluation review

**Inputs:** `results.csv` (88 engine outputs: 44 Basic, 44 Agentic) and the uploaded source-faithful Markdown documents for the 2026 Diákhitel fee schedule, 2025 Diákhitel notice, and undated glossary.  
**Output:** `results_human_scored.csv` – the original rows plus human labels and one note per answer.

## Scoring rubric

- `human_answer_correct = 1`: The answer gives all material facts requested, appropriately corrects an explicit false premise, **or** appropriately declines an unsupported/out-of-scope question. Missing one requested component, wrongly applied conditions, or an unnecessary refusal of a documented fact is `0`.
- `human_citation_supported = 1`: The cited source supports the *claims actually made* in the answer and the fee/definition conditions apply to those claims. `0` means an incorrect or misapplied supporting reference. `N/A` means a corpus-bounded refusal without a source citation (it is not counted as a citation failure). Citation support **does not imply completeness**: an incomplete answer can have a valid citation for its partial content.
- `human_scope_safe = 1`: The answer does not ascribe another bank's rates to OTP, assign a tariff to an unsupported service/currency/territory/limit, or introduce unsupported specific financial claims. A conservative false refusal can be safe but still incorrect. `0` denotes a material misapplication of such a claim.

Binary labels represent a documented, manual judgment against the stated gold facts and available source sections; they are **not** outputs of an automated accuracy classifier. The lack of full model prompts/retrieval context in the CSV limits diagnosis of why any given error occurred.

## Overall results

| Metric | Basic RAG | Agentic RAG |
|---|---:|---:|
| Questions | 44 | 44 |
| Automated status matches | 22/44 (50.0%) | 24/44 (54.5%) |
| Human answer correctness | **21/44 (47.7%)** | **20/44 (45.5%)** |
| Citation support, applicable cases only | 10/14 (71.4%) | 9/14 (64.3%) |
| Scope-safe answers/refusals | 41/44 (93.2%) | 40/44 (90.9%) |
| Median latency | 9.56 s | 31.43 s |
| p95 latency (from runner) | 13.85 s | 40.71 s |
| Timeouts and reported errors | 0 | 0 |

**Interpretation:** The +2 automated status-match gain for agentic RAG does **not** translate into higher factual correctness. On these questions, the reviewed agentic outputs perform slightly worse, with approximately 3.3× the median latency. This is a 44-question test set, not a population estimate; do not infer general model superiority or statistically significant differences from it.

### Correct answers by category

| Category | Number | Basic | Agentic |
|---|---:|---:|---:|
| Fee lookups | 9 | 1 | 0 |
| Multi-section | 6 | 0 | 0 |
| Glossary | 5 | 3 | 3 |
| Unsupported | 6 | 5 | 5 |
| False premises | 7 | 2 | 2 |
| Other banks | 5 | 4 | 4 |
| Out of scope | 6 | 6 | 6 |
| **Total** | **44** | **21** | **20** |

## Critical errors (must fix before treating this as a banking-information assistant)

1. **B001 – Cross-bank contamination, both engines.** A K&H youth account question was answered with OTP Diákhitel's 0 Ft maintenance fee. Add a product/bank scope check *before retrieval and before extraction*.
2. **F006 – Domestic vs foreign ATM confusion, both engines.** The requested domestic third-party ATM fee is **1,711 Ft + 0.9%** (fee schedule, `SRC-P07-B04`). Both engines supplied **3,145 Ft + 2.12%** from the foreign-ATM section (`SRC-P08-B01`).
3. **U002 – EEA euro withdrawal, both engines.** Germany/euro/third-party ATM does not fall under the foreign-ATM source row restricted to **non-euro withdrawals in EEA countries and withdrawals outside the EEA**. The source does not give the requested EEA euro fee; both engines applied 3,145 Ft + 2.12% incorrectly.
4. **M003 – Branch SEPA nuance, agentic.** The answer generalized the **branch-paper** fee to "SEPA átutalás díjai" overall and omitted the requested SEPA definition; its quotation validates a tariff, not the entire answer.
5. **F005 – Incomplete composite tariff, agentic.** The answer omitted `+ 1.79%` from the documented **172 Ft + 1.79%** own-ATM fee. It was still marked `answered`.
6. **F007 and M006 – Intent mismatch, both engines.** The extractive path returned glossary definitions rather than the requested monetary fee (F007) or an answer including the fee (M006). Citations support the glossary text but the answers are incomplete.
7. **P003 – Incorrect source absence claim, agentic.** It says debit-card purchase fees are not in the sources while citing the section recording **0 Ft**.

## False refusals and missed evidence

Many fee lookup and multi-section questions were marked `insufficient` even though the necessary data is present. Examples: F001 (MobilBank 0 Ft monthly/yearly), F002 (account closing 0 Ft), F003 (paper branch transfer), F004 (electronically initiated SEPA), F009 (own-ATM HUF cash deposit), M001 (monthly SMS plan vs. per-SMS fee) and M002 (regular vs. paper one-off transfer). This indicates that strict literal-quote validation and/or retrieval selection is rejecting answerable questions. The CSV alone cannot isolate whether the initial retrieval, prompt, model response, or validator is the root cause.

In the Agentic run, **32/44** questions had a rewrite, with a median latency of **32.33 s**, versus **5.45 s** for those without rewrites. All six out-of-scope questions were rewritten. A second retrieval after a failed answer is often wasteful if the failure arises from answer validation rather than missing evidence.

## Prioritized remediation

1. **Bank/product boundary gate:** If the question concerns K&H, Erste, MBH, Revolut or another unsupported institution, refuse or explicitly state that only OTP documents are loaded. Handle cross-bank comparisons without silently substituting OTP rates.
2. **Intent-aware retrieval:** Distinguish *definition*, *fee*, *eligibility*, *comparison*, *unknown detail* and *out of scope*. For compound questions, plan subquestions and demand source coverage for every part. Never let the glossary alone satisfy a fee request.
3. **Condition-aware fee extraction:** Represent fees as structured records including bank, product, operation, transaction direction, channel, currency, geography, limits, fixed and percentage components, minimum and maximum. Validate all applicable constraints before stating a fee; never drop one component.
4. **Evidence completeness gate:** Require support for every *requested fact*, not just a literal quote for one correct sentence. A truthful but incomplete answer should not receive `answered` when a requested fee is missing.
5. **Rewrite only on retrieval failure:** Do not retry merely because answer validation failed or the source lacks an applicable tariff. Add a cheap scope check and explicit out-of-scope exit to avoid the 30-second tail for unrelated queries.
6. **Repeat the same 44-case benchmark after each change** and keep this scored file as baseline. Reserve additional unseen questions for the final holdout so rule-based fixes are not measured only on cases used to build them.

## Scoring cautions

- Automated status matching is not answer correctness: Basic contains 2 status matches that are materially incorrect, Agentic 4. A correct refusal can also be marked status-mismatched (e.g. U004 Basic, where the text correctly says the USD exchange rate is not provided but status is `partial`).
- A citation can be real yet irrelevant to the requested fee. Citation support is therefore judged for its asserted claims **separately** from whether the answer answers the question.
- `N/A` in the citation column is not a failure or a success; for valid refusals with no monetary claim, a source quotation is not always necessary. Report the applicable denominator.
- Safe refusal behavior and ability to answer in-scope banking questions should be reported separately. A system that refuses everything would look safe but be unhelpful.
