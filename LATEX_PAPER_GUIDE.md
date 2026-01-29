# LaTeX Paper Writing Guide (for Codex CLI)

This document guides an automated coding/writing agent (e.g., Codex CLI)
through **iterative research paper writing in LaTeX**, while staying consistent
with the repo’s engineering rules (see `KEY_PRINCIPLES.md`).

Goals:

- Fast iteration without turning the paper into a mess.
- Camera-ready quality (consistent style, correct references, clean build).
- No stale claims: when results change, the paper updates everywhere.
- Reproducibility: figures/tables/numbers can be traced to a run + commit hash.

## Policy keywords

- **MUST** = non-negotiable.
- **SHOULD** = strong default.
- **NICE** = optional polish.

> **Non‑negotiable rule:** the paper must compile cleanly, and every numeric claim
> in the paper must have a clear source of truth.

---

## 0) How this fits with the repo’s coding rules

This guide assumes you follow the repo conventions in `KEY_PRINCIPLES.md`, in particular:

- Makefile-driven execution (no long manual commands).
- Reproducibility via git commit hashes / tree hashes.
- Flat, explicit code and minimal “magic”.

When in doubt: prefer **simple, explicit** paper organization and regeneration workflows.

**IMPORTANT**: When writing the paper, focus only on the actual research content. This guide is a generic guide for writing style and policies. You **MUST NEVER** refer to this guide or its content in the actual paper. The paper should read as a standalone scientific document, **without any meta-references to the writing process or tooling.**

---

## 1) Canonical repository layout for a paper

Prefer a dedicated `paper/` directory with a *single* entrypoint and small modular files:

```
paper/
  main.tex                 # only wiring + structure; minimal content
  preamble.tex             # packages + formatting settings
  macros.tex               # all \newcommand, \DeclareMathOperator, etc.
  results.tex              # AUTO-GENERATED numbers/macros (see §9)
  sections/
    00_abstract.tex
    01_intro.tex
    02_related.tex
    03_method.tex
    04_experiments.tex
    05_results.tex
    06_discussion.tex
    07_limitations.tex
    08_conclusion.tex
  figures/                 # final exported figure files (pdf/png)
  tables/                  # optional: LaTeX table fragments
  refs.bib                 # bibliography database
  latexmkrc                # optional: build configuration
```

Rules:

- **MUST** keep `main.tex` mostly “wiring” (~50–150 lines).
- **MUST** put packages/formatting in `preamble.tex`.
- **MUST** put macros in `macros.tex`.
- **MUST** avoid giant monolithic `.tex` files.
- **SHOULD** keep section files self-contained and readable.

---

## 2) Build system (Makefile-first)

- **MUST** compile via a short target: `make paper`
- **MUST** use `pdflatex` (not latexmk).
- **MUST** treat “paper edit done” as: build succeeds with **zero** unresolved refs.

Conceptual build sequence:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory build main.tex
bibtex build/main
pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory build main.tex
```

NICE:
- `make paper.clean` (wipe `paper/build`)
- `make check` (paper build + staleness checks)

---

## 3) Diff-friendly LaTeX writing (prevents merge pain)

- **MUST** write prose with **one sentence per line**.
  - This makes diffs clean and makes LLM edits more localized (less accidental reflow).
- **MUST** avoid manual paragraph reflow (don’t wrap at 80 chars arbitrarily).

Example:

```tex
We propose a new method for X.
It improves Y by leveraging Z.
We validate the approach on A, B, and C.
```

(LaTeX ignores line breaks in paragraphs, so this is purely for diffs.)

---

## 4) Preamble policy (packages + formatting)

- **MUST** keep the preamble boring and stable.
- **SHOULD** avoid obscure packages unless necessary.

Common, safe defaults:

- `amsmath`, `amssymb`, `amsthm` – math
- `graphicx` – figures
- `booktabs` – tables
- `microtype` – spacing/kerning
- `hyperref` + `cleveref` – hyperlinks and consistent references
- `xcolor` – minimal color
- `siunitx` – aligned numbers/units (esp. for tables)
- `mathtools` – extra math helpers

---

## 5) Macros policy (write fewer, better macros)

Put all macros in `paper/macros.tex`.

Rules:

- **MUST** keep macros simple (no hidden logic).
- **SHOULD** create a macro only if it’s used ≥2 times.
- **MUST NOT** redefine standard commands unless absolutely required.

Naming conventions:

- Math objects: `\vecx`, `\matA`, `\distD`, `\Loss`, etc.
- Datasets/models: `\CIFAR`, `\ImageNet`, etc.
- Results macros (auto-generated): `\BestAcc`, `\BestAccStd`, etc. (see §9)

---

## 6) Labels and cross-references (never write “Figure 3” manually)

- **MUST** label everything you might reference.
- **MUST** reference via `\cref{...}` / `\Cref{...}` (from `cleveref`).
- **MUST NOT** hardcode “Figure 2”, “Table 1”, etc.

Label prefixes:

- Sections: `sec:...`
- Figures: `fig:...`
- Tables: `tab:...`
- Equations: `eq:...`
- Algorithms: `alg:...`
- Theorems/Lemmas: `thm:...`, `lem:...`, `prop:...`

Placement rules:

- Figures/tables: put `\label{...}` immediately after `\caption{...}`.
- Equations: label inside the equation environment.

---

## 7) Figures and tables (paper-facing rules)

### 7.1 Figures

- **MUST** store final figure files in `paper/figures/`.
- **SHOULD** export vector plots as **PDF** (line art/text).
- **SHOULD** export raster images as **PNG** at ≥300 dpi.

Canonical snippet:

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_main_results.pdf}
  \caption{...}
  \label{fig:main_results}
\end{figure}
```

### 7.2 Captions (required schema)

Captions are part of scientific honesty. They must stand alone.

- **MUST** define what is plotted (variables, units).
- **MUST** define what each line/marker/bar represents.
- **MUST** define uncertainty (std/SEM/CI) and **N** when applicable.
- **SHOULD** include a single-clause takeaway (do not oversell).

A good caption answers: “What am I looking at, and how should I interpret it?”

### 7.3 Tables

- **MUST** use `booktabs` (no vertical rules).
- **SHOULD** use `siunitx` for number alignment.
- **SHOULD** avoid cramped tables; if it doesn’t fit, reconsider presentation.

---

## 8) Citations and bibliography

- **MUST** use a single consistent citation style (natbib or biblatex).
- **MUST** avoid missing keys and unresolved citations.
- **MUST NEVER** manually edit `paper/references.bib`.
    - Assume it is auto-generated from a central source, along with summaries of papers for you to read and use as needed. You do not need to add new entries manually, and should not.
- **SHOULD** cite claims that aren’t clearly common knowledge.

## 9) Anti-stale policy: numbers and claims must have a single source of truth

**This is the most important part.**

### 9.1 Never type experimental numbers directly into prose

- **MUST NOT** manually embed experimental numbers in sentences (accuracy, loss, AUC, runtimes, etc.).
- **MUST** define paper-facing numbers in an auto-generated file:

- `paper/results.tex` — the **only** place that defines result numbers.

Example `results.tex`:

```tex
% AUTO-GENERATED. DO NOT EDIT BY HAND.
% Provenance: commit=abc1234 dirty=false run=file_storage/runs/042
\newcommand{\BestAcc}{92.3}
\newcommand{\BestAccStd}{0.4}
\newcommand{\MainRuntimeHours}{7.2}
```

Then in text:

```tex
Our method achieves \BestAcc$\pm$\BestAccStd on ...
```

This should be automated via scripts that parse experiment logs and regenerate `results.tex`. Remember to use commit/subtree hashes for provenance.

### 9.2 Regeneration workflow (when results change)

When results change, **MUST** regenerate:

1) figures (in `paper/figures/`)
2) tables (if any)
3) `paper/results.tex`

Then **MUST** update the narrative that depends on them (abstract + intro + conclusion + discussion).

### 9.3 Narrative consistency rule (no orphaned claims)

If a top-line result changes, you **MUST** update:

- abstract
- intro (contributions / summary)
- results section (details)
- discussion/limitations (interpretation)
- conclusion

### 9.4 Claim → evidence mapping

Create a small claim→evidence map so the story doesn’t drift during iteration.

- **MUST** maintain `paper/claims.md` (or similar) as:
  - claim id,
  - claim text (exact/excerpt),
  - evidence (figure/table + run id),
  - where in paper,
  - last verified commit.
  - potential subtree hashes if needed.


### 9.5 Use TODO markers to flag places needing updates

If some claim cannot be updated immediately, due to waiting on results, review, or ideation, add a TODO in red text (inline is ok). Use a simple macro for this. Use this also to flag general places needing attention, review, updates. For example if the writing might need improvement, if some figure or caption or section needs my review, etc. Particularly flag areas of the paper when you feel *less confident* about the wording or claims.

---

## 10) Editing protocol for Codex CLI (safe paper changes)

When asked to modify the paper:

1. **Locate all impacted locations**
   - Search for the concept/claim across `paper/`.
   - Identify figures/tables/captions that describe it.

2. **Make the edit**
   - Prefer local changes in the relevant section file.
   - Keep diffs small and readable (especially with one-sentence-per-line style).

3. **Update references**
   - If you add a figure/table/equation: add `\label{...}` and reference via `\cref`.

4. **Run the anti-stale pass** -- **ESSENTIAL**
   - If any numbers changed: update `results.tex` (or the generator that produces it).
   - Ensure text uses macros, not hardcoded numbers.

5. **Sanity check wording**
   - Ensure interpretation matches updated results.
   - Remove overclaiming; add qualifiers if needed.
   - Add TODO markers for uncertain areas.

6. **Definition-of-done checklist**
   - `make paper` succeeds.
   - No `??` references.
   - No missing citations.

7. Make incremental commits with clear messages.

---

## 11) Camera-ready checklist

Before a camera-ready build:

- [ ] Paper compiles cleanly (`make paper`)
- [ ] No `??` refs, no missing bib keys
- [ ] All figures are vector where appropriate; raster images ≥300 dpi
- [ ] Captions are standalone and honest (uncertainty + N when relevant)
- [ ] All experimental numbers come from `results.tex`
- [ ] Abstract/intro/conclusion match the final results
- [ ] Appendix (if any) is consistent with the main paper
- [ ] No accidental “we will” / “in future work” unless intended
- [ ] PDF metadata is reasonable (title/authors)

---
