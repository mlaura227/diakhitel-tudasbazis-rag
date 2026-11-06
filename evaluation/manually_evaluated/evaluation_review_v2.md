# OTP Diákhitel RAG – Human review of Hardening v2

**Inputs:** `results_v2.csv` (88 outputs, 44 questions × two engines) and the OTP source-faithful Markdown documents: `Dijjegyzek_OTP_Diakhitel_szamla_20260901-rag.md`, `Diakhitel_Szamla_H_20251118-rag.md`, and `Glosszarium-rag.md`.

**Outputs:** `results_v2_human_scored.csv`, retaining all original columns and populating `human_answer_correct`, `human_citation_supported`, `human_scope_safe`, and per-row `human_notes`.

## Rubric (same strict convention as v1)

- **Answer correctness (1/0):** All materially requested facts and scope restrictions are stated correctly; false premises corrected, and absent facts refused. An otherwise true but incomplete multi-part response is **0**.
- **Citation support (1/0/N/A):** Every material *affirmative claim actually made* is supported by its listed section under the applicable constraints. This measure is not the same as completeness. A valid uncited out-of-corpus scope refusal is **N/A**, not a citation failure.
- **Scope safety (1/0):** No invented fees or cross-bank contamination, no misapplied location/channel/limit/currency tariff. An omission can be safe but incorrect.

The scoring was performed on the recorded answer text and section identifiers, cross-checking the relevant source Markdown. The CSV does **not** contain full model traces or the exact quoted evidence that may have been presented in Streamlit, so citation evaluation confirms section-level support—not automatic sentence-level entailment proofs.

## Overall manual scores

| Metric | Basic v1 | Basic v2 | Agentic v1 | Agentic v2 |
|---|---:|---:|---:|---:|
| Human answer correctness | 21/44 (47.7%) | **43/44 (97.7%)** | 20/44 (45.5%) | **43/44 (97.7%)** |
| Citation support, applicable only | 10/14 | **33/33** | 9/14 | **33/33** |
| Scope-safe | 41/44 | **44/44** | 40/44 | **44/44** |
| Automated status matches | 22/44 | 44/44 | 24/44 | 44/44 |
| Median latency | 9.56 s | **4.25 s** | 31.43 s | **4.42 s** |
| Timeouts / reported errors | 0 | 0 | 0 | 0 |

**Citation denominator note:** In v2, 33 in-scope cases per engine cite a document section. Eleven other-bank/out-of-scope cases are citation `N/A`. That denominator differs from v1 because v1 produced many uncited refusals. Hence raw citation-support fractions should not be interpreted as directly comparable citation-*coverage* scores across versions.

## Human correctness by category

| Category | n | Basic v1 | Basic v2 | Agentic v1 | Agentic v2 |
|---|---:|---:|---:|---:|---:|
| fee_lookup | 9 | 1/9 | 9/9 | 0/9 | 9/9 |
| multi_section | 6 | 0/6 | 5/6 | 0/6 | 5/6 |
| glossary | 5 | 3/5 | 5/5 | 3/5 | 5/5 |
| unsupported | 6 | 5/6 | 6/6 | 5/6 | 6/6 |
| false_premise | 7 | 2/7 | 7/7 | 2/7 | 7/7 |
| other_bank | 5 | 4/5 | 5/5 | 4/5 | 5/5 |
| out_of_scope | 6 | 6/6 | 6/6 | 6/6 | 6/6 |

## The remaining incomplete answer

**M005 — both engines:** The output properly lists the two eligibility elements (eligible under the relevant business rules, OTP designated at Diákhitel Központ) and correctly states group direct debit is free. But the source notice, `notice_20251118:source-p1-direct-debit`, expressly excludes items categorized as **„Egyéb beszedések”**. The gold answer requires this exception. Neither engine mentions it.

- `human_answer_correct = 0` (**incomplete**)
- `human_citation_supported = 1` (the claims made are backed by the two cited sections)
- `human_scope_safe = 1` (no invented or misassigned fee)

Suggested answer addition: *„A díjmentesség nem terjed ki a külön hirdetményben 'Egyéb beszedések' kategóriába sorolt tételekre.”*

**Smaller non-failing nuance:** M001 cites the SMS monthly fee and notification SMS fee, but not `SRC-P03-B01` for account maintenance. This still answers the user’s actual challenge—whether 0 Ft monthly package charges imply a free message—and does not claim to have verified the account maintenance fee separately. If the thesis requires every premise to be explicitly checked, add a separate strict **premise verification coverage** metric rather than changing the answer-correctness rubric ad hoc.

## Architectural findings

1. **All 44 Basic and Agentic responses are identical**, including cited-section identifiers. For this run, there is no demonstrated answer-quality advantage of the LangGraph path.
2. The Agentic trace reports **zero query rewrites** across 44 questions: 33 accepted questions ran a single retrieval round, 11 scope-gate exits ran zero rounds.
3. The v2 mode distribution is the same for both engines: `condition_checked` 17, `definition_extractive` 6, `composite_verified` 3, `extractive` 1, `document_limit_guard` 6, `scope_gate` 11. **No response is tagged with the generative mode**, so the improvement should be credited to shared deterministic routing/extraction/validation, not to greater Qwen or LangGraph reasoning accuracy.
4. The 44 questions were used during the v2 hardening design. They are now a **development/regression set**, not an independent blind holdout. Reuse is appropriate for confirming fixes, but final model-quality claims require unseen prompts and paraphrases.
5. The five v1 high-risk errors—cross-bank fee contamination, domestic/foreign ATM mismatch, German EUR ATM tariff extrapolation, incomplete ATM tariff component, and glossary-versus-fee misrouting—are no longer visible in their corresponding v2 responses.

## Recommended next study design

Preserve `results_human_scored.csv` and `results_v2_human_scored.csv` as paired benchmarks. Add a **held-out** set of diverse questions and manually score all three dimensions using the same rubric. Include more open-ended, multi-condition queries that fall outside deterministic extractive templates; otherwise this test measures a rules/lookup hybrid, not whether agentic retrieval or the local language model improves answer generation. Report both status match and substantive correctness, along with latency and the fraction answered via deterministic versus generative routes.
