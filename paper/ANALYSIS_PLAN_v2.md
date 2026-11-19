# Analysis plan for the second protocol (v2)

Written 2026-09-25, after the v2 runs were launched and before any v2 result was
looked at. Every analysis the paper reports as confirmatory is listed here; anything
else is labelled exploratory in the paper.

## Why a second protocol

The council review (paper/review/COUNCIL_FINDINGS.md) found defects in the first
protocol that change measured values, not just their description:

1. Reasoning text from truncated qwen3 outputs was compiled as code (88 generations).
2. Empty outputs were scored as compiled (27B: 8 in teach, 1 in scaffold).
3. Three of the ten rule explanations in the intervention were wrong against the
   compiler: arkts-as-casts forbids `<T>v` and requires `v as T`, but the prompt
   said "avoid 'as' conversions"; the return-type and array-literal items
   described constructs that compile.
4. The knowledge-probe parser matched substrings and raw answers were not stored.
5. The only contrast for specificity was unnamed rules, whose baselines sit at the
   floor; there was no control for prompt length, format or preamble.
6. Ollama ran with its default 4,096-token context and 4,000-token budget.

The first protocol's data are kept unchanged (collection `icaart_compliance`,
probe/knowledge4_*.json) and serve one purpose in v2: the ten taught rules were
selected on them, so v2 tests that fixed list on independent data.

## Design

- Models: the seven 3B–8B models and qwen3.8-27b, as before.
- 100 ArkUI task prompts (probe/data/prompts100.csv), as before.
- Single-shot, temperature 0.2, seed fixed per replicate, reasoning disabled for
  every model that has a switch, 3,072-token budget, 8,192-token context (Ollama).
- Five conditions, interleaved within model: scaffold, generic, placebo, ids, teach
  (definitions in probe/compliance_v2.py). teach and placebo differ in the content
  of the ten listed rules only (1,760 vs 1,765 characters, same format and preamble).
- Replicates: seed 1 for all conditions first; seeds 2 and 3 if time allows.
- Knowledge probe: 66 compiler-verified items (44 forbidden, 22 allowed), six
  wordings (three per polarity), temperature 0, whole-word parser, raw answers kept.

## Outcomes (per generation)

- T: number of distinct taught rules reported by the compiler (0–10).
- U: number of distinct other arkts-* rules.
- Pl: number of distinct placebo rules (expected near zero everywhere).
- C: compiled, counted only when code was extracted (`extract_status == ok`).
- no_code rate per condition, reported separately; primary violation analyses use
  generations with extracted code, and a sensitivity analysis counts no_code as zero.
- Output length in characters, reported per condition. Length is affected by the
  condition, so the primary models do NOT adjust for it (total effect);
  length-adjusted estimates are secondary and labelled as direct effects.

## Confirmatory hypotheses and tests

Small-model panel (seven models) unless stated. Per-model estimates are Poisson
rate ratios with prompt-clustered (robust) standard errors; pooled estimates add
model fixed effects. Multiplicity: Holm across the seven models within each
hypothesis. Paired permutation tests (within-prompt label swaps, 10,000 draws,
clustered by prompt) as the model-free check. p-values from Monte Carlo reported as
(k+1)/(N+1).

- H1 teaching: T is lower under teach than scaffold.
- H2 specificity (primary specificity test): T is lower under teach than placebo.
  Also reported: the condition × rule-set interaction (teach vs placebo, T vs U).
- H3 preamble: generic vs scaffold on T and on U.
- H4 explanation: teach vs ids on T.
- H5 compilation: C under teach vs scaffold, per model exact McNemar, pooled
  stratified by model.
- H6 knowledge (probe): per model, majority-of-six scoring (ties dropped) gives
  balanced accuracy; test Fisher exact one-sided on the 2x2 of says-forbidden vs
  truly forbidden; Holm across the seven small models. Stable-only (all six parsed
  and agreeing) is secondary. Unparseable answers are reported per model and
  never scored as wrong.

## Rule-level analysis (pre-specified, directional prediction from v1 exploratory data)

For each model and taught rule: knows = majority-of-six says forbidden on every
forbidden item targeting that rule, and not on the allowed items adjacent to it.
Benefit = rate ratio of that rule's violations, teach vs placebo. Prediction from
the first protocol (exploratory there): benefit is larger where the model knows the
rule (activation), not where it does not (information supply). Test: Poisson model
with rule and model fixed effects and a knows × condition interaction, clustered by
prompt. Reported whatever its direction.

## Compilation analysis

- Rule-clean share: generations with T + U = 0.
- Compile rate given rule-clean, with Wilson intervals, per model.
- Failure classes of rule-clean failures from the stored full compiler output:
  unlabelled dialect diagnostics, ArkUI DSL, hallucinated API, resources, syntax.
- JSX markup in generated code (a `<Tag>...</Tag>` pattern) reported per model and
  condition, because the compiler reports JSX as arkts-as-casts.

## 27B

Reported per condition with the same statistics; n = 1 model, so it is a model
comparison, never a scale estimate.

## What would change the paper's claims

- H2 not supported (teach ≈ placebo): the paper claims a generic priming effect of
  a rule list, not rule-specific teaching.
- H1 supported only through length: stated as such.
- Rule-level interaction in the predicted direction: activation framing; opposite
  direction: information-supply framing; null: neither, stated as underpowered.
