# RAG Evaluation Metrics — A Practical Tutorial

Evaluating a Retrieval-Augmented Generation (RAG) system requires measuring two distinct stages:

1. **Retrieval** — Did the system find the right documents?
2. **Answer generation** — Did the LLM produce a good answer from those documents?

This tutorial explains the metrics used in our evaluation pipeline and walks through worked examples so you can build intuition for what the numbers actually mean.

---

## Part 1: Retrieval Evaluation

When a user asks a question, the retriever returns a ranked list of document chunks. We need to know:

- *How high up* in the list are the relevant chunks? (ranking quality)
- *Did we find them at all?* (coverage)

We use three complementary metrics: **MRR**, **nDCG**, and **keyword coverage**.

---

### 1.1 Mean Reciprocal Rank (MRR)

#### The idea

MRR answers: **"How far down the ranked list did the user have to look before finding a relevant result?"**

For a single query and keyword, the *reciprocal rank* is `1 / position` of the first relevant document. If the keyword never appears, the reciprocal rank is 0.

When a test question has multiple keywords, we compute the reciprocal rank for each keyword independently and average them.

#### Formula

For a single keyword:

```
RR(keyword) = 1 / rank_of_first_relevant_doc    (or 0 if not found)
```

Averaged over *K* keywords for one question:

```
MRR = (1/K) * Σ RR(keyword_i)
```

#### Worked example

Suppose we ask **"Who founded Insurellm?"** and our retriever returns 5 chunks:

| Rank | Chunk content (abbreviated) |
|------|-----------------------------|
| 1 | "Insurellm currently operates with 32 employees…" |
| 2 | "Avery Lancaster **founded** Insurellm in 2015…" |
| 3 | "The company was **founded** in San Francisco…" |
| 4 | "Insurellm's product **Markellm**…" |
| 5 | "**Avery** serves as CEO…" |

The test keywords are `["Avery", "Lancaster"]`.

- **"Avery"**: first appears at rank 2 → RR = 1/2 = **0.500**
- **"Lancaster"**: first appears at rank 2 → RR = 1/2 = **0.500**

```
MRR = (0.500 + 0.500) / 2 = 0.500
```

If the retriever had placed the "Avery Lancaster founded…" chunk at rank 1:

```
MRR = (1/1 + 1/1) / 2 = 1.000   (perfect)
```

#### Interpretation

| MRR | Meaning |
|-----|---------|
| 1.0 | Every keyword is found in the top-ranked document |
| 0.5 | On average, keywords first appear at rank 2 |
| 0.2 | On average, keywords first appear at rank 5 |
| 0.0 | No keywords found in any retrieved document |

#### Strengths and limitations

- **Strength**: simple, interpretable, focuses on the *first* hit.
- **Limitation**: ignores additional relevant documents further down the list. A system that returns 10 relevant docs at ranks 2-11 scores the same as one that returns a single hit at rank 2.

---

### 1.2 Normalized Discounted Cumulative Gain (nDCG)

#### The idea

nDCG answers: **"Across the full ranked list, how well are relevant documents concentrated near the top?"**

Unlike MRR, nDCG considers *all* relevant documents, not just the first one. It rewards having relevant results early (via a logarithmic discount) and penalizes relevant results that appear late.

#### Step-by-step formula

**Step 1 — Assign relevance scores.** In our implementation we use *binary* relevance: a chunk gets `rel = 1` if it contains the keyword, `rel = 0` otherwise.

**Step 2 — Compute Discounted Cumulative Gain (DCG).** Sum the relevance scores, each divided by a logarithmic discount based on position:

```
DCG@k = Σ (rel_i / log₂(i + 1))    for i = 1 … k
```

The `log₂(i+1)` denominator grows with position, so documents at rank 1 contribute more than documents at rank 5.

**Step 3 — Compute Ideal DCG (IDCG).** Sort the relevance scores in descending order (all 1s first) and compute DCG on that ideal ranking. This is the best possible DCG for this set of results.

**Step 4 — Normalize.**

```
nDCG@k = DCG@k / IDCG@k
```

If IDCG is 0 (no relevant documents at all), nDCG is defined as 0.

#### Worked example

Same question, same 5 retrieved chunks. Keyword: **"Avery"**.

| Rank (i) | Contains "Avery"? | rel_i | Discount log₂(i+1) | rel_i / discount |
|-----------|-------------------|-------|---------------------|------------------|
| 1 | No | 0 | log₂(2) = 1.000 | 0.000 |
| 2 | Yes | 1 | log₂(3) = 1.585 | 0.631 |
| 3 | No | 0 | log₂(4) = 2.000 | 0.000 |
| 4 | No | 0 | log₂(5) = 2.322 | 0.000 |
| 5 | Yes | 1 | log₂(6) = 2.585 | 0.387 |

```
DCG@5 = 0.000 + 0.631 + 0.000 + 0.000 + 0.387 = 1.018
```

Ideal ranking (push both relevant docs to positions 1 and 2):

| Rank (i) | rel_i | Discount | rel_i / discount |
|-----------|-------|----------|------------------|
| 1 | 1 | 1.000 | 1.000 |
| 2 | 1 | 1.585 | 0.631 |
| 3 | 0 | 2.000 | 0.000 |
| 4 | 0 | 2.322 | 0.000 |
| 5 | 0 | 2.585 | 0.000 |

```
IDCG@5 = 1.000 + 0.631 = 1.631
```

```
nDCG@5 = 1.018 / 1.631 = 0.624
```

If both relevant docs had been at ranks 1 and 2:

```
nDCG@5 = 1.631 / 1.631 = 1.000   (perfect)
```

For multiple keywords, we compute nDCG per keyword and average — the same pattern as MRR.

#### Interpretation

| nDCG | Meaning |
|------|---------|
| 1.0 | All relevant documents are ranked as high as possible |
| 0.6–0.8 | Good ranking — relevant docs are near the top but not perfectly ordered |
| 0.3–0.5 | Moderate — relevant docs exist but are mixed in with irrelevant ones |
| 0.0 | No relevant documents found at all |

#### Strengths and limitations

- **Strength**: considers all relevant documents and rewards better rankings with a smooth, principled discount.
- **Limitation**: with binary relevance, it cannot distinguish between a "somewhat relevant" and a "highly relevant" chunk (graded relevance would solve this but requires richer annotations).

---

### 1.3 Precision@k

#### The idea

Precision@k answers: **"Of the top-k documents the retriever returned, how many are actually relevant?"**

It measures the *purity* of the result set — a high precision means the user is not wading through noise.

#### Formula

```
Precision@k = (number of relevant documents in top-k) / k
```

#### Worked example

Question: **"What is Insurellm's vision statement?"**
Keywords: `["revolutionize", "insurance", "technology"]`

Suppose k = 5 and the retriever returns:

| Rank | Contains any keyword? | Relevant? |
|------|-----------------------|-----------|
| 1 | Yes ("revolutionize", "insurance", "technology") | ✓ |
| 2 | Yes ("insurance") | ✓ |
| 3 | No | ✗ |
| 4 | No | ✗ |
| 5 | Yes ("technology") | ✓ |

3 of the 5 returned chunks are relevant:

```
Precision@5 = 3 / 5 = 0.60
```

If all 5 chunks contained at least one keyword:

```
Precision@5 = 5 / 5 = 1.00   (perfect — no noise)
```

If only 1 chunk was relevant:

```
Precision@5 = 1 / 5 = 0.20   (mostly noise)
```

#### Interpretation

| Precision@k | Meaning |
|-------------|---------|
| 1.0 | Every returned document is relevant — no wasted slots |
| 0.6–0.8 | Most results are relevant with some noise |
| 0.2–0.4 | Mostly irrelevant results; retriever is too broad |
| 0.0 | None of the top-k documents are relevant |

#### Strengths and limitations

- **Strength**: directly measures how much noise the user (or the LLM) has to filter through. Especially important for RAG because irrelevant context can confuse the generator.
- **Limitation**: does not account for *how many* relevant documents exist in the corpus. If there are 20 relevant chunks but you only retrieve 5, a precision of 1.0 still misses 15.

---

### 1.4 Recall@k

#### The idea

Recall@k answers: **"Of all the relevant documents in the corpus, how many did the retriever actually find in its top-k?"**

It measures *completeness* — a high recall means the retriever is not leaving important information behind.

#### Formula

```
Recall@k = (number of relevant documents in top-k) / (total relevant documents in corpus)
```

In practice, the "total relevant documents" is often approximated. In a keyword-based evaluation like ours, we can define it per keyword: how many chunks in the entire collection contain that keyword? Or, more pragmatically, we treat the set of expected keywords as the ground truth and measure what fraction of them appear in the top-k results (which is closely related to keyword coverage).

#### Worked example

Question: **"How many office locations does Insurellm maintain?"**
Keywords: `["offices in", "5"]`

Suppose the full knowledge base has 4 chunks that mention "offices in" and 6 chunks that mention "5". The retriever returns k = 5 chunks:

| Rank | Contains "offices in"? | Contains "5"? |
|------|------------------------|---------------|
| 1 | Yes | Yes |
| 2 | No | Yes |
| 3 | No | No |
| 4 | Yes | No |
| 5 | No | No |

For keyword **"offices in"**: 2 retrieved out of 4 total in corpus:

```
Recall@5("offices in") = 2 / 4 = 0.50
```

For keyword **"5"**: 2 retrieved out of 6 total in corpus:

```
Recall@5("5") = 2 / 6 = 0.33
```

Average recall:

```
Recall@5 = (0.50 + 0.33) / 2 = 0.415
```

If the retriever had found all 4 "offices in" chunks and all 6 "5" chunks (impossible with k = 5, but illustrative):

```
Recall = (4/4 + 6/6) / 2 = 1.00   (perfect)
```

#### Interpretation

| Recall@k | Meaning |
|----------|---------|
| 1.0 | Every relevant document in the corpus was retrieved |
| 0.6–0.8 | Most relevant information is captured; some gaps remain |
| 0.2–0.4 | The retriever is missing a majority of relevant content |
| 0.0 | No relevant documents retrieved at all |

#### The precision–recall trade-off

Precision and recall pull in opposite directions:

- **Increasing k** (retrieving more documents) tends to **increase recall** (you find more relevant docs) but can **decrease precision** (more noise creeps in).
- **Decreasing k** tends to **increase precision** (only the most confident results survive) but can **decrease recall** (you miss relevant docs).

```
k=3:  Precision ↑  Recall ↓   (tight, focused results)
k=20: Precision ↓  Recall ↑   (broad net, more noise)
```

In RAG this trade-off is critical: too few chunks and the LLM lacks context; too many and the LLM gets distracted by irrelevant text. Tuning k is one of the most impactful knobs.

#### Strengths and limitations

- **Strength**: directly answers whether the retriever is finding *enough* of the relevant information, which is essential for answer completeness.
- **Limitation**: requires knowing (or estimating) the total number of relevant documents in the corpus, which is not always available. Using keyword presence as a proxy is practical but imperfect.

---

### 1.5 Keyword Coverage

#### The idea

The simplest metric: **"What fraction of the expected keywords appeared somewhere in the retrieved documents?"**

#### Formula

```
coverage = keywords_found / total_keywords × 100%
```

A keyword counts as *found* if its MRR > 0 (i.e., it appears in at least one retrieved chunk).

#### Worked example

Question: **"Where is Insurellm's headquarters located?"**
Keywords: `["San Francisco", "headquarters"]`

If both keywords appear in at least one of the top-k chunks:

```
coverage = 2/2 × 100% = 100%
```

If only `"headquarters"` is found:

```
coverage = 1/2 × 100% = 50%
```

#### Why it matters alongside MRR and nDCG

MRR and nDCG tell you about *ranking quality*, but if a keyword is missing entirely, those metrics just silently return 0 for that keyword. Keyword coverage gives you a direct, human-readable count: "we found 8 out of 10 expected keywords." This makes it easy to spot systematic retrieval gaps.

---

### 1.6 How the five metrics work together

| Scenario | MRR | nDCG | Precision@k | Recall@k | Coverage |
|----------|-----|------|-------------|----------|----------|
| All keywords in rank-1 chunk | 1.0 | 1.0 | High | Depends on corpus | 100% |
| All keywords found, but buried at rank 5+ | Low | Low-Med | Low-Med | Depends on corpus | 100% |
| Top-k is pure but misses many relevant docs | High | High | 1.0 | Low | May be < 100% |
| Half the keywords missing entirely | ≤ 0.5 | ≤ 0.5 | ≤ 0.5 | Low | 50% |

Each metric answers a different question:

| Metric | Question it answers |
|--------|---------------------|
| **Coverage** | "Did we find each expected keyword at all?" |
| **MRR** | "How quickly did we find the first relevant chunk?" |
| **nDCG** | "How well-ordered is the full ranked list?" |
| **Precision@k** | "How much noise is in the top-k results?" |
| **Recall@k** | "How much relevant content did we miss?" |

---

## Part 2: Answer Evaluation — LLM as a Judge

### 2.1 The idea

Once the retriever fetches relevant chunks and the LLM generates an answer, we need to evaluate the *quality of that answer*. Traditional automated metrics like BLEU or ROUGE compare surface-level word overlap, which misses semantic equivalence (e.g., "founded in 2015" vs. "established in 2015" would score poorly despite being correct).

**LLM-as-a-judge** uses a separate LLM call to evaluate the generated answer against a reference answer. The judge LLM receives the question, the generated answer, and a human-written reference answer, then scores the response on multiple dimensions.

### 2.2 The evaluation dimensions

Our judge evaluates three aspects:

#### Accuracy (1–5)

> "How factually correct is the answer compared to the reference?"

- **5** — Perfectly accurate; every fact matches the reference.
- **3** — Acceptable; core facts are correct but some details are wrong or imprecise.
- **1** — Wrong; the answer contains factual errors.

**Example:**

- Question: *"When was Insurellm founded?"*
- Reference: *"Insurellm was founded in 2015."*
- Generated: *"Insurellm was founded in 2016."* → **Accuracy: 1** (wrong year)
- Generated: *"Insurellm was founded in 2015."* → **Accuracy: 5** (exact match)

#### Completeness (1–5)

> "Does the answer cover all the information from the reference?"

- **5** — All information from the reference is included.
- **3** — The main point is addressed but supporting details are missing.
- **1** — Major information is absent.

**Example:**

- Question: *"How many office locations does Insurellm maintain?"*
- Reference: *"Insurellm maintains 5 office locations: San Francisco, New York, Austin, Chicago, and Denver."*
- Generated: *"Insurellm has 5 offices."* → **Completeness: 3** (correct count, but cities missing)
- Generated: *"Insurellm maintains 5 office locations: San Francisco, New York, Austin, Chicago, and Denver."* → **Completeness: 5**

#### Relevance (1–5)

> "Does the answer directly address the question without unnecessary extras?"

- **5** — Directly answers the question with no extraneous information.
- **3** — Answers the question but includes tangential details.
- **1** — Off-topic or largely irrelevant.

**Example:**

- Question: *"Who founded Insurellm?"*
- Reference: *"Avery Lancaster founded Insurellm in 2015."*
- Generated: *"Avery Lancaster founded Insurellm in 2015. The company now has 32 employees, 5 offices, and 8 product lines."* → **Relevance: 3** (correct, but loaded with extras)
- Generated: *"Avery Lancaster founded Insurellm in 2015."* → **Relevance: 5**

### 2.3 How it works in practice

```
┌──────────────┐
│  Test Case   │──→ question, reference_answer, keywords
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  RAG System  │──→ generated_answer (retriever + LLM)
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌────────────────────────────────────┐
│  Judge LLM   │◄────│ System: "You are an expert          │
│  (gpt-4.1-   │     │   evaluator…"                       │
│   nano)      │     │ User: question + generated_answer   │
└──────┬───────┘     │   + reference_answer                │
       │             └────────────────────────────────────┘
       ▼
┌──────────────┐
│  Structured  │──→ { feedback, accuracy, completeness, relevance }
│  Output      │
└──────────────┘
```

1. The test case provides the question and a human-written reference answer.
2. The RAG system generates an answer (retriever fetches context, LLM produces text).
3. A separate judge LLM call receives all three (question, generated answer, reference answer) and returns structured scores via a Pydantic model.
4. The structured output ensures we always get numeric scores and textual feedback that can be aggregated programmatically.

### 2.4 Why use structured output?

The judge's response is parsed into an `AnswerEval` Pydantic model. This guarantees:

- Scores are always numeric and within the expected range.
- Feedback is always present.
- Results can be aggregated across hundreds of test cases automatically (e.g., average accuracy across all tests).

### 2.5 Strengths and limitations of LLM-as-a-judge

| Strengths | Limitations |
|-----------|-------------|
| Captures semantic equivalence that BLEU/ROUGE miss | The judge itself can make mistakes — it is not infallible |
| Multi-dimensional scoring (accuracy, completeness, relevance) | Scores may vary slightly across runs (non-deterministic) |
| Provides human-readable feedback alongside numeric scores | Relies on the quality of the reference answer |
| Scales to hundreds of test cases without human reviewers | Using the same model family for generation and judging can introduce bias |

### 2.6 Tips for reliable LLM-as-a-judge evaluation

1. **Use clear rubrics in the system prompt.** Explicit definitions for each score level reduce ambiguity.
2. **Anchor the low end.** The prompt states "any wrong answer must score 1" for accuracy — this prevents the model from being overly generous.
3. **Separate judge from generator.** Ideally the judge model is different from (or more capable than) the generator to avoid self-serving bias.
4. **Write good reference answers.** The judge can only be as good as the reference it compares against.
5. **Aggregate over many test cases.** Individual scores can be noisy; averages over dozens of questions give a stable signal.

---

## Part 3: Putting It All Together

A single test case produces both a retrieval evaluation and an answer evaluation:

| Metric | Type | What it tells you |
|--------|------|-------------------|
| MRR | Retrieval | How quickly the first relevant chunk is found |
| nDCG | Retrieval | How well-ordered the full ranked list is |
| Keyword Coverage | Retrieval | Whether expected keywords are present at all |
| Accuracy | Answer | Factual correctness vs. reference |
| Completeness | Answer | Information coverage vs. reference |
| Relevance | Answer | Signal-to-noise ratio of the answer |

When diagnosing issues:

- **Low retrieval scores + low answer scores** → the retriever is the bottleneck. Improve chunking, embeddings, or the knowledge base.
- **High retrieval scores + low answer scores** → the LLM is struggling. Improve the prompt, switch models, or add few-shot examples.
- **High retrieval scores + high answer scores** → the system is working well for this question type.

By tracking these metrics across all test cases, you can identify systematic weaknesses and measure the impact of each change you make to the pipeline.
