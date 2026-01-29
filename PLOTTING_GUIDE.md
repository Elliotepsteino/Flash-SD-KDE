# Plotting Guide (camera-ready, informative, aesthetic, honest)

This document guides an automated agent (e.g., Codex CLI) in producing plots that are:

- **Camera-ready** (publication quality)
- **Informative** (communicate the point quickly)
- **Aesthetically pleasing** (clean, consistent, readable)
- **Honest** (no misleading framing; uncertainty and sample sizes surfaced)
- **Reproducible** (plots regenerate from code + data with clear provenance)

It assumes the repo’s conventions in `KEY_PRINCIPLES.md` (Makefile-first, config-driven,
absolute imports, git provenance).

## Policy keywords

- **MUST** = non-negotiable.
- **SHOULD** = strong default.
- **NICE** = optional polish.

---

## 1) Non-negotiable principles (scientific honesty)

1. **Show the data honestly**
   - **MUST NOT** crop axes to exaggerate effects unless clearly annotated and justified.
   - **MUST NOT** hide outliers or failed runs unless explicitly stated.
   - **MUST NOT** use visual tricks (3D bars, heavy smoothing without disclosure, etc.).

2. **Make comparisons fair**
   - **MUST** use comparable axes/limits when comparing methods.
   - **MUST** use the same preprocessing and evaluation protocol.
   - **MUST** state missing data or unmatched settings explicitly.

3. **Make it readable at final paper size**
   - **MUST** be legible when embedded in the paper at the intended column width.
   - **MUST** have readable labels/ticks/legend at 100% zoom in the compiled PDF.

4. **One source of truth**
   - **MUST** generate every plot from a committed script.
   - **MUST** make the exact data + config discoverable.

---

## 2) File organization and naming

Suggested structure:

```
analysis/
  figures_src/
    fig_main_results/
      config.py
      make_figure.py
    fig_ablation_sweep/
      config.py
      make_figure.py

paper/  (optional if working on a paper)
  figures/
    fig_main_results.pdf
    fig_main_results.png
    fig_ablation_sweep.pdf
    fig_ablation_sweep.png
```

Rules:

- **MUST** keep `paper/figures/` for **final exported** files used by LaTeX.
- **MUST** keep generators in `analysis/figures_src/**/make_figure.py`.
- **SHOULD** use stable, descriptive filenames: `fig_<topic>.pdf`.
- If no paper, **MUST** keep final figures in `analysis/figures/`.

---

## 3) Config-driven, not CLI-driven

- **MUST NOT** use argparse/CLI flags to parameterize plots.
- **MUST** define a `config.py` (ideally a dataclass) per figure or per figure family.

Config should include:

- run IDs / artifact paths
- which methods to plot
- metric definitions
- smoothing/window choices (if any)
- export filename(s)

Then `make_figure.py` imports the config and generates the figure.

---

## 4) Reproducibility & provenance (anti-stale for figures)

Every exported figure must be traceable to:

- generating script path
- git commit hash or subtree hash (+ dirty flag)
- data source(s) (files / run IDs)
- config used

### 4.1 Minimal provenance (required)

At minimum, record provenance in one of these ways:

- **MUST** include a sidecar manifest:
  - `paper/figures/manifest.json` mapping figure → script + commit/subtreehash + inputs + make target.

### 4.2 Staleness rule (required)

If any of the following changes, the figure is stale and **MUST** be regenerated:

- underlying results data
- preprocessing logic
- plotting script
- evaluation protocol

If figures change, **MUST** also update:

- the caption (what is shown, metric definition, uncertainty),
- any paper text that interprets the figure if working on a paper.

---

## 5) Sizing and typography (camera-ready defaults)

### 5.1 Target sizes

Typical widths for two-column venues:

- **Single column**: ~3.3 in
- **Double column**: ~6.8–7.0 in

Rule:

- **MUST** decide target embed width first, then choose `figsize` accordingly.

### 5.2 Font sizes

- **MUST** ensure text is readable at the final embed size (usually 8–10 pt effective).
- **MUST** keep legend text at least as large as tick labels.
- **MUST** ensure lines are nice and thick (enough to see when shrunk).

### 5.3 Consistency with LaTeX

- **SHOULD** use mathtext for symbols (e.g., `$\\alpha$`, `$\\mathcal{L}$`).
- **SHOULD** avoid `text.usetex=True` unless truly necessary (fragile builds).

---

## 6) Export formats and quality

Preferred outputs:

- **SHOULD** export vector plots as **PDF** (best for lines/text).
- **SHOULD** also export as **PNG** for sending to slack.
- **SHOULD** export photos/raster images as **PNG** at ≥300 dpi.

Export rules:

- **MUST** use tight bounding boxes (no huge whitespace).
- **MUST** verify the exported PDF keeps text as vector when possible.
- **SHOULD** avoid transparent backgrounds unless necessary.

---

## 7) Visual design rules (simple, clean, readable)

### 7.1 Axes and units

- **MUST** label axes.
- **MUST** include units where applicable (ms, s, %, etc.).
- **SHOULD** use log scales only when justified; state clearly.

### 7.2 Legends

- **SHOULD** prefer direct labeling when the plot is simple.
- Otherwise:
  - **MUST** keep legend readable and not covering key features,
  - **SHOULD** keep legend order consistent with line order.

### 7.3 Color and accessibility

- **SHOULD** use color-blind friendly palettes.
- **MUST NOT** rely on color alone; combine color with markers and/or line styles.
- **SHOULD** ensure interpretability in grayscale.

### 7.4 Reduce chartjunk

Avoid:

- 3D effects
- heavy gridlines
- unnecessary shading
- excessive ticks
- unreadable marker clouds

---

## 8) Uncertainty, variability, and honesty

### 8.1 Always report variability when it matters

If results vary across seeds/runs:

- **MUST** show uncertainty with error bars/bands or distribution plots, and
- **MUST** say what uncertainty means (std/SEM/CI) and **N**.

### 8.2 Smoothing disclosure

If you smooth curves (moving average, EMA, spline):

- **MUST** disclose smoothing in the caption/legend.
- **SHOULD** show unsmoothed curves faintly or include them in appendix.

### 8.3 Axis cropping

- **SHOULD** include zero for bar charts by default.
- If cropping:
  - **MUST** annotate the crop and justify it.

---

## 9) Multi-panel figures (common in camera-ready)

Multi-panel figures are a frequent failure mode (illegible mosaics). Do them deliberately.

- **MUST** label panels in-figure: (a), (b), (c), ...
- **MUST** ensure each panel is readable at final size (no microscopic fonts).
- **SHOULD** share axes/limits when panels are meant to be compared.
- **SHOULD** use a shared legend when possible (avoid repeating legends in every panel).
- **SHOULD** keep consistent styling (fonts, line widths, marker sizes) across panels.
- **MUST** avoid “too many panels” that force unreadable text; split into multiple figures if needed.

Caption rules for multi-panel:

- **MUST** describe each panel explicitly: “(a) … (b) …”.
- **MUST** keep descriptions short and factual.

---

### 10) Makefile targets (recommended)

- **MUST** add:
  - `make figs` (generate all paper figures)
  - `make fig-<name>` (generate one figure)

---

## 11) Definition of done (for any figure change)

A figure update is complete only when:

- [ ] axes labels, units, ticks are correct
- [ ] uncertainty / N is shown or explicitly stated
- [ ] exported in correct formats (PDF+PNG)
- [ ] provenance is recorded (commit hash / subtree hash / script / inputs)

If working on a paper:

- [ ] readable at final paper size
- [ ] caption is updated and standalone (honest, self-contained)
- [ ] any text interpreting the figure is updated accordingly

---

## 12) Quick “honesty audit” checklist

Before finalizing a plot, ask:

- Does the plot show the full story or could it mislead?
- Are axes, scales, and units clear?
- Are comparisons fair and on equal footing?
- Is uncertainty shown if variability exists?
- Would a skeptical reader accept the framing?

If not, revise.
