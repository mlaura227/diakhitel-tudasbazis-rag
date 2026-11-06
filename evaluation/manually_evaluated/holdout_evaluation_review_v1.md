# OTP Diákhitel RAG — Independent 40-question holdout, human review

**Evidence reviewed:** `holdout_results_v1.csv` (80 recorded answers), `holdout_questions_v1.jsonl` answer expectations, and the processed OTP bank regulation (2026-10-15), product notice (2025-11-18), 2026 fee schedule and glossary Markdown documents. The review is restricted to these provided corpus sources; it is not a check against live OTP prices.

## Scoring method

- **`human_answer_correct` = 1/0:** 1 only for a fully responsive and factually supported answer, a properly corrected false premise, or a justified out-of-scope/unsupported refusal. Missing a requested material component = 0, even when supplied facts are true.
- **`human_citation_supported` = 1/0/N/A:** 1 if cited sections substantiate all material claims **under the applicable transaction, product, date, and channel conditions**; 0 for ungrounded or misapplied citations. `N/A` for uncited refusals, whether appropriate or false; these are excluded from the citation denominator. A citation can be supported even when the answer is incomplete.
- **`human_scope_safe` = 1/0:** 1 if no bank/product, transaction type, channel, geographical, time, or limit tariff is improperly carried over. Unnecessary abstention is generally safe but incorrect. 0 for the four materially misapplied answers HC008, HF005, HF006, and HU003.

## Human results

| Metric | Basic | Agentic |
|---|---:|---:|
| Automated status match | 21/40 (52.5%) | 22/40 (55.0%) |
| **Human answer correctness** | **11/40 (27.5%)** | **11/40 (27.5%)** |
| Citation support among cited answers | 10/14 | 11/15 |
| Scope safe | 36/40 (90.0%) | 36/40 (90.0%) |
| Fully correct answered outputs | 3/30 answerable questions | 3/30 answerable questions |
| Correct justified abstentions | 8/10 abstention cases | 8/10 abstention cases |
| Unnecessary abstentions on answerable questions | 18 | 17 |
| Answered but incorrect/incomplete | 10 | 11 |
| Median latency | 6.67 s | 6.86 s |
| Recorded timeouts / errors | 0 / 0 | 0 / 0 |

### Human correctness by category

| Category | Questions | Basic | Agentic |
|---|---:|---:|---:|
| `legal_single` | 8 | 1/8 | 1/8 |
| `cross_document` | 12 | 0/12 | 0/12 |
| `fee_condition` | 6 | 0/6 | 0/6 |
| `unsupported` | 6 | 4/6 | 4/6 |
| `false_premise` | 4 | 2/4 | 2/4 |
| `out_of_scope` | 4 | 4/4 | 4/4 |

## Critical findings

1. **Generality gap:** Earlier 44-case development regression reached 43/44 correct per engine, but this independent holdout records **11/40** for each. The two sets have different difficulty and composition; this is not a same-population before/after estimate.
2. **False refusals:** Basic unnecessarily refuses **18/30** answerable prompts and Agentic **17/30**. Many answerable legal paraphrases (HL001–HL006, HL008) fall through strict generative evidence checks, while some legal questions are accidentally routed to the fee-condition matcher.
3. **Compound-query incompleteness:** Neither engine fully answers any of the 12 cross-document questions or six conditional fee questions under the strict all-parts rubric. Examples: HC004 omits blocking fee; HC005 omits direct debit and exception; HF002 omits electronic SEPA 0 Ft; HF003 omits postal POS charge.
4. **Material condition errors:** HF005 uses the domestic purchase tariff for a foreign purchase; HF006 uses an ATM tariff for a branch withdrawal. HC008 and HU003 answer questions about future indexation with today’s 0 Ft / free charges, without explaining future applicability. These four cases receive `human_scope_safe=0` per engine.
5. **No demonstrated holdout advantage from Agentic RAG:** Both engines produce identical answer text/status/citations for **39 of 40** prompts. In HC006 the agent performs a targeted second retrieval and changes status from insufficient to answered, but still omits the Digital Services contract and the annual InternetBank fee; human correctness stays 0 for both.
6. **Safety versus usefulness:** Both engines refuse all four cross-bank/unrelated questions correctly, but the fourth (future Bitcoin price) bypasses the early scope gate and is instead refused after generative checking. Low false-positive fee rates do not compensate for poor coverage of in-scope source information.

## Specific examples and necessary distinctions

| ID | Failure or success | Gold fact / condition |
|---|---|---|
| HL001 | False refusal | OTP Diákhitel account management may not be suspended. |
| HC002 | Incomplete answer | 0 Ft maintenance is correct, but Diákhitel cannot use Bizalom offers. |
| HC004 | Incomplete answer | Both replacement card and card blocking are free. |
| HC006 | Agentic retrieval but incomplete final answer | Digital Services agreement on VideoBank opening, plus InternetBank Alap annual 0 Ft. |
| HF001 | False refusal | Yettel login SMS 35 Ft vs notification SMS 64 Ft. |
| HF005 | Wrong geography | Foreign debit-card purchases are distinct from domestic purchases. |
| HF006 | Wrong channel | Own-branch debit cash withdrawal is labelled fee type not applied, not own-ATM 172 Ft + 1.79%. |
| HU003 | Wrong future applicability | Current free replacement card does not establish numerical 2027 indexed tariff. |
| HP002 | Correctly defeats false premise | Existing OTP relationship requires branch for Diákhitel opening, not MobileBank. |
| HP003 | Correctly defeats false premise | Free card issuance does not imply free own-ATM withdrawal; tariff applies up to 150,000 Ft. |

## Limitations and next steps

This review scores the **stored answers**, not the full Ollama prompts or every retrieval candidate. Citation judgments are section-level checks against the processed Markdown; for critical legal applications, recheck original PDF provisions. Interpreted `knowledge.md` sections may contain editorial paraphrases. Because this holdout has now been examined and scored, **freeze the dataset as holdout v1**; use it for error analysis and subsequent regression, not for claiming independent validation after tailoring rules to these 40 prompts.

Prioritize an **intent decomposition and slot-coverage gate** for compound questions, a **legal routing fallback** for unseen paraphrases, and **fee applicability validation** for time/channel/geography. Evaluate these improvements using a newly reserved holdout set. For an Agentic-vs-Basic study, ensure the same answerer/validators but compare acquisition completeness under equal evidence budgets and record which missing clauses were recovered.
