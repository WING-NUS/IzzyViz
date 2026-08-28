# RAG External Contexts — **Answer-Bearing Subset**

A filtered copy of the Notre Dame / Cavanaugh / Lobund RAG context set that keeps
**only contexts whose text contains the gold answer** — fully or partially — and
annotates each with exactly *what* overlaps the answer.

> **Gold answer:** *Lobund Institute for Animal Studies*
> **Question:** *Which institute involving animal life did Cavanaugh create at Notre Dame?*
> **SQuAD id:** `5733926d4776f41900660d8f` (SQuAD v1.1 train, `University_of_Notre_Dame`, para 23)

This is the set to visualize when you want to study **how the model attends to the
answer span inside a retrieved context**. Contexts with no answer overlap were
dropped.

> 🆕 **Also in this package:** `synthetic_context.json` — a *parallel synthetic
> corpus* (repetition / entailment / contradiction / neutral probes) that uses the
> same record schema. See the **🆕 Synthetic parallel corpus** section below.

---

## What's different from the full corpus

1. **Filtered** to answer-bearing contexts only (58 of the original 106+).
2. Each record has a new **`answer_overlap`** field (see below).
3. Each record has a **`slice`** field recording how the window was cut:
   - `answer_centered` — a window cut *around* the answer span so the answer is
     guaranteed to be inside (added because the answer sits deep in long articles,
     past a plain first-N-token slice).
   - `article_start` — an original slice from the article's beginning that happened
     to already overlap the answer.

Everything else (`text`, `length`, `relatedness`, `source`, `license`,
`comments`, `attribution`) is unchanged from the parent corpus. `length` is still
the exact `bert-base-uncased` WordPiece count (no `[CLS]`/`[SEP]`).

---

## 📐 Data structure of each instance

`external_contexts.json` is a JSON object mapping a **context id → record**. Every
record has the structure below (🆕 = fields added in this answer-bearing subset;
all other fields are inherited unchanged from the parent corpus):

```text
"<context-id>"                             key, e.g. "ctx_ac_001"
├─ text            str        the retrieved context passage
├─ length          int        # bert-base-uncased WordPiece tokens (excl. [CLS]/[SEP])
├─ relatedness     str        high | medium | low   (vs. the target question)
├─ source          str        canonical Wikipedia URL the text was taken from
├─ license         str        "CC BY-SA 4.0"
├─ comments        str        curator note on the context
├─ attribution     str        ready-to-use attribution string
├─ slice           str        answer_centered | article_start                🆕
└─ answer_overlap  obj        what overlaps the gold answer                   🆕
   ├─ answer                              str        "Lobund Institute for Animal Studies"
   ├─ overlap_strength                    str        full | strong | partial
   ├─ full_answer_present                 bool       whole phrase appears verbatim
   ├─ distinctive_term_present            bool       the unique term "Lobund" appears
   ├─ matched_answer_words                list[str]  ⊆ [lobund, institute, animal, studies]
   ├─ overlap_ratio                       float      matched content words / 4
   ├─ longest_contiguous_match            str        e.g. "Animal Studies"
   └─ longest_contiguous_match_char_span  int[2]     [start, end] into this record's `text`
```

A complete, real example record:

```json
"ctx_ac_001": {
  "text": "mission and an expanded student body. He stressed advanced studies and research while quadrupling the university's student population, with undergraduate enrollment increasing by more than half and graduate student enrollment growing fivefold. Cavanaugh established the Lobund Institute for Animal Studies and Notre Dame's Medieval Institute, presided over the construction of Nieuwland Science Hall, Fisher Hall, and the Morris Inn, and the Hall of Liberal Arts (now O'Shaughnessy Hall), made",
  "length": 96,
  "relatedness": "high",
  "source": "https://en.wikipedia.org/wiki/University_of_Notre_Dame",
  "license": "CC BY-SA 4.0",
  "comments": "answer-centered window from 'University of Notre Dame'",
  "attribution": "Wikipedia contributors, \"University of Notre Dame\", Wikipedia, The Free Encyclopedia (text under CC BY-SA 4.0).",
  "slice": "answer_centered",
  "answer_overlap": {
    "answer": "Lobund Institute for Animal Studies",
    "overlap_strength": "full",
    "full_answer_present": true,
    "distinctive_term_present": true,
    "matched_answer_words": ["lobund", "institute", "animal", "studies"],
    "overlap_ratio": 1.0,
    "longest_contiguous_match": "Lobund Institute for Animal Studies",
    "longest_contiguous_match_char_span": [270, 305]
  }
}
```

In this example `text[270:305]` is exactly `"Lobund Institute for Animal Studies"` —
`longest_contiguous_match_char_span` indexes directly into this record's `text`, so
you can highlight the overlap in a heatmap without re-searching.

**Overlap-strength grades** (the stop-word "for" never counts as overlap):
- **full** — the entire phrase *Lobund Institute for Animal Studies* appears verbatim.
- **strong** — the distinctive term **Lobund** appears, or a contiguous ≥2-word
  span of the answer does (e.g. *Animal Studies*, *Lobund Institute*).
- **partial** — only isolated common answer word(s) appear (e.g. just *animal* or
  *studies*), without *Lobund* or a multi-word span.

---

## Distribution (58 contexts)

**By overlap strength**

| strength | count | note |
|---|---|---|
| full | 15 | whole answer phrase present |
| strong | 18 | contains "Lobund" or a ≥2-word answer span |
| partial | 25 | isolated common answer word only |

- **29** contexts contain the distinctive term **Lobund**; **15** contain the full phrase.

**By provenance** — `answer_centered` 25 · `article_start` 33
**By relatedness** — high 35 · medium 14 · low 9
**Length** (BERT tokens) — min **96** · max **~2048** (buckets: 64–255 → 11, 256–1023 → 20, 1024+ → 27)

> The truly answer-carrying contexts (`full` / `strong`, especially the
> `answer_centered` windows) are high-relatedness by construction. The `partial`
> rows are medium/low contexts that merely share a common word like *animal* — kept
> because "at least partial overlap" was requested, and clearly flagged so you can
> filter them out with `overlap_strength != "partial"` if you only want genuine
> answer hits.

---

## 🆕 Synthetic parallel corpus (`synthetic_context.json`)

A second, **synthetic** corpus that runs parallel to `external_contexts.json`. Instead
of retrieved Wikipedia text, these are purpose-built probes around the gold answer, so
you can compare *real retrieval* against controlled synthetic conditions. **33 records**
in **four groups**:

| group | count | what it is | contains verbatim answer? |
|---|---|---|---|
| `repetition` | 3 | the gold SQuAD context re-used **1× / 2× / 3×** — does simply repeating the context help? | yes |
| `entailment` | 10 | short passages (12–~55 tok) that contain the verbatim answer and **reinforce** it | yes (all 10) |
| `contradiction` | 10 | short passages that **contradict** the fact (re-attribute *Lobund*, or assert a false institute) | 4 of 10 (re-attributed) |
| `neutral` | 10 | on-topic passages that **neither** entail nor contradict the fact | none |

It uses the **same standard data structure** as the external corpus, plus
synthetic-specific fields:

```text
"<id>"   e.g. "syn_ent_06" · "syn_rep_2x" · "syn_con_01"
├─ text, length, relatedness, comments, attribution, answer_overlap   (same schema as external)
├─ source            str   "synthetic"
├─ license           str   "CC0-1.0"   (original AI-generated text, no third-party rights)
├─ slice             str   "synthetic"
├─ synthetic_group   str   repetition | entailment | contradiction | neutral      🆕
└─ repetition_count  int   1 | 2 | 3     (repetition group only)                   🆕
```

> ⚠️ `answer_overlap` is **lexical, not semantic**: a `contradiction` record can still be
> `overlap_strength: "full"` because it contains the answer *string* while denying the
> *fact*. The semantic relation lives in **`synthetic_group`** — use that field for the
> entailment / contradiction / neutral condition, and `answer_overlap` for *where* the
> answer tokens sit.

Example (the shortest entailment record):

```json
"syn_ent_06": {
  "text": "Cavanaugh created the Lobund Institute for Animal Studies.",
  "length": 12,
  "relatedness": "high",
  "source": "synthetic",
  "license": "CC0-1.0",
  "comments": "synthesized: contains verbatim answer and entails it",
  "attribution": "Synthetic text generated for research; original content, no third-party rights (CC0-1.0).",
  "slice": "synthetic",
  "synthetic_group": "entailment",
  "answer_overlap": { "overlap_strength": "full", "full_answer_present": true, "...": "..." }
}
```

---

## Files

```text
rag_contexts_answer_bearing/           🆕  answer-bearing subset (this package)
├─ external_contexts.json       the corpus — 58 answer-bearing contexts (id → record)
├─ synthetic_context.json       🆕 parallel synthetic corpus — 33 probes (id → record)
├─ squad_instance_original.txt  the original SQuAD instance (gold answer + context)
├─ sources_log.json             per-source counts for this subset
├─ build_summary.json           machine-readable distribution summary
├─ LICENSING.md                 licensing — Wikipedia CC BY-SA 4.0 · synthetic CC0-1.0
└─ README.md                    this file (data structure + distribution)
```

---

## Suggested filters

```python
import json
d = json.load(open("external_contexts.json"))

# only contexts with the full answer phrase
full = {k:v for k,v in d.items() if v["answer_overlap"]["full_answer_present"]}

# genuine answer hits (drop weak single-word partials)
strong = {k:v for k,v in d.items() if v["answer_overlap"]["overlap_strength"] != "partial"}

# answer-centered windows at a given length budget
short = {k:v for k,v in d.items() if v["slice"]=="answer_centered" and v["length"]<=384}

# 🆕 the synthetic corpus, grouped by NLI-style relation
syn = json.load(open("synthetic_context.json"))
entail = {k:v for k,v in syn.items() if v["synthetic_group"]=="entailment"}
```

Licensing: Wikipedia context text (`external_contexts.json`) is **CC BY-SA 4.0** —
attribution + share-alike on redistribution. Synthetic contexts
(`synthetic_context.json`) are **CC0-1.0** (public-domain dedication, no restrictions).
See `LICENSING.md`.
