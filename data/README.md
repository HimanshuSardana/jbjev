# data/

3059 resume ↔ JD pairs. One JSON object per line.

| field     | type   | meaning |
|-----------|--------|---------|
| `text`    | string | `"resume: <...>\n[SEP]\njob: <...>"` — the model input |
| `verdict` | int    | `0` no match · `1` partial match · `2` strong match |
| `apply`   | int    | `1` worth applying · `0` don't bother |
| `score`   | float  | match quality in `[0, 1]`, rounded to 2dp |

## label contract

`score` is the source of truth; the other two labels are a pure function of it.

```
score < 0.40          → verdict 0
0.40 ≤ score < 0.70   → verdict 1
score ≥ 0.70          → verdict 2
score ≥ 0.50          → apply 1
```

A screener reading only the text can reproduce `score` from:

- **required-skill coverage** — how many of the JD's named skills the resume lists (heaviest term, ~0.42)
- **domain fit** — same discipline (~0.16), adjacent (mobile↔frontend, data↔backend, devops↔backend, systems↔embedded, qa↔either), or disjoint
- **seniority fit** — resume years vs the JD's stated minimum (~0.15)
- **evidence** — does the project line name the work the JD describes (~0.08)
- **degree fit** (~0.05) and **location fit** (~0.04)

Hard caps, applied after the sum:

- 3+ years required, under 1 held → capped ~0.34
- zero overlap with the named requirements → capped ~0.30
- disjoint domain → never scores as "partial" (capped ~0.38)
- internship JD, candidate with 3+ years → capped 0.62

## files

| file | rows | resumes | source |
|---|---|---|---|
| `train.jsonl` | 1147 | — | authored |
| `valid.jsonl` | 212 | — | authored |
| `test.jsonl` | 212 | — | authored |
| `real.jsonl` | 1488 | 40 | real scraped listings |

Authored total **1571** across **1156 distinct resumes**. Grand total **3059**.

## integrity

Verified on every merge:

- **0** contract violations — `verdict` and `apply` recomputed from `score` on every row
- **0** duplicate `text` values
- **0** malformed rows (every line has `resume: `, `[SEP]`, `job: `)
- **0** text overlap between any two files
- **0** *resume* overlap between any two files — splits are candidate-disjoint, grouped by
  resume string before splitting so a candidate's contrast set never straddles a boundary

## known gaps

- **Verdict imbalance**: 355 / 366 / 426 (v0 / v1 / v2) in train, and `real.jsonl` is
  962 / 507 / 19. Verdict 2 is overweight in the authored splits and badly under-represented
  in the real one. Use class weights rather than dropping rows.
- **Score histogram gap at 0.70–0.80**: only 33 authored rows land there, against 437 in
  0.90–1.00. Clean matches were written as clean matches. If the regression head is weak
  near the top of the strong band, thicken 0.70–0.85.
- **Decision band is healthy**: 382 authored rows sit in 0.40–0.70, which is where the two
  thresholds live.

## real.jsonl

Job side is **real scraped listings**: 62 unique postings, 40 candidate profiles, from `matchmyresume-scraper`
(`naukri_jobs.csv`, `internships.csv`, `Web Development_internshala.csv`). Skill lists come
from the scraped `skills` column where the source had one, otherwise keyword extraction over
the description. Descriptions trimmed to ~430 chars. Resume side is a 26-profile bank, 0–6 yrs.

Use it as a **domain-shift eval**, not a training set:

- score distribution is skewed low (mean 0.40, only 19 strong matches) because real Indian
  postings skew senior and generic
- phrasing is real — corporate boilerplate, internshala bullet lists, truncated scrapes with
  no description at all
- a model that scores 92% on `test.jsonl` and falls apart here is memorizing authored
  templates, not matching resumes

Train on the authored splits, evaluate on `real.jsonl`. If the gap is large, fold a portion
of `real.jsonl` into training and hold back the rest.

## provenance

No generator scripts. `train/valid/test` are authored line by line, including a 129-row block
that arrived as `_p1b.jsonl` (43 resumes × 3 JDs, terse JD style, experience as ranges like
`2-4 yrs`) and was merged by resume group. `real.jsonl` is a mechanical transform of scraped
listings — parsing, trimming, dedup, rubric scoring — with no synthetic content added.
