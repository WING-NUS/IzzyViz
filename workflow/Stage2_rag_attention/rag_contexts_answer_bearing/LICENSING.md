# Licensing — read this before redistributing

> Short answer to "do we need an MIT / CC BY license?":
> **"MIT CC BY" is not one license.** MIT is for *code*; CC BY is for *content/data*.
> Use them for different parts of this artifact, and note the ShareAlike catch below.

## What each license is for

| Part of this deliverable | File(s) | Recommended license | Why |
|---|---|---|---|
| Any **generation/processing code** | (not included in this package) | **MIT** | Permissive software license; ideal for research code you want others to reuse freely. |
| The **context corpus** | `external_contexts.json` | **CC BY-SA 4.0** | Every record is Wikipedia text, which is CC BY-SA 4.0 — see below. |
| 🆕 The **synthetic corpus** | `synthetic_context.json` | **CC0-1.0** | Original AI-generated text (no third-party rights); dedicated to the public domain. |
| The **original SQuAD instance** | `squad_instance_original.txt` | **CC BY-SA 4.0** | SQuAD v1.1 is distributed under CC BY-SA 4.0. |

## The ShareAlike catch (important)

The corpus is now sourced **entirely from English Wikipedia**, recorded per-record
in the `license` and `attribution` fields:

- **English Wikipedia → CC BY-SA 4.0.** This is a *copyleft / ShareAlike*
  license, **not** a strictly "permissive" one. It requires (a) **attribution**
  and (b) that any redistribution of that text — including a derived corpus that
  contains it — also be licensed **CC BY-SA 4.0**. You therefore **cannot**
  relabel these passages as MIT or plain CC BY.

Because **all 106 records are CC BY-SA 4.0**, the corpus as a whole, *if
redistributed*, must be released under **CC BY-SA 4.0** with attribution. (For
internal/private research use, no redistribution license is triggered.) This is
the same license SQuAD itself uses, so it is the conventional, low-friction
choice for a SQuAD-derived artifact.

## If you need a strictly *permissive* (non-copyleft) corpus

The corpus is now 100% Wikipedia, so it is **entirely copyleft (CC BY-SA 4.0)**.
If you specifically need CC BY or CC0 only (no ShareAlike), rebuild from
permissive sources instead — e.g. Wikinews (CC BY 2.5), open-access CC BY journal
articles, U.S. government public-domain documents, or public-domain texts
(Project Gutenberg). Trade-off: **high-relatedness** coverage (Notre Dame /
Cavanaugh / Lobund) is hard without Wikipedia, since the university's own pages
are All-Rights-Reserved. Ask and I'll regenerate a `permissive-only` variant.

## 🆕 Synthetic contexts (`synthetic_context.json`)

The synthetic corpus is **original, AI-generated** text (repetition / entailment /
contradiction / neutral probes) and carries **no third-party copyright**. It is
released under **CC0-1.0** (public-domain dedication) — no attribution or
share-alike obligations — so you may mix, relabel, or redistribute it freely. The
`entailment`, `contradiction`, and `neutral` records (30 of 33) carry `CC0-1.0` in
their `license` field. The three `repetition` records are the exception: they
reproduce the gold SQuAD context paragraph (SQuAD / Wikipedia), so they are marked
**CC BY-SA 4.0** in their `license` field — attribute that upstream source if you
redistribute them.

## Attribution

Per-record attribution strings are in `external_contexts.json` (`attribution`
field) and the full resolved source list (titles + URLs + license) is in
`sources_log.json`. Keep those when redistributing.
