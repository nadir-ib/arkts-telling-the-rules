# Telling Small Language Models the Rules — artifact

Code and data for the paper. Everything needed to reproduce every number in it
from the released data, and to rerun the experiment with a HarmonyOS toolchain.

## Reproduce the numbers (no GPU, no database)

    pip install -r requirements.txt
    python3 probe/analyze_v2.py --jsonl data/compliance_v2.jsonl.gz
    python3 probe/figures_v2.py

`paper/numbers_v2.json` is rewritten by the first command; `paper/numbers_v2.reference.json`
is the copy the paper was written from, for comparison.

## Contents

- `paper/ANALYSIS_PLAN_v2.md` — the analysis plan, fixed before outcomes were inspected.
- `probe/` — experiment and analysis code:
  `build_cases_v2.py` (compiler-verified probe items -> `cases_v2.json`),
  `knowledge_probe_v2.py` (six-wording probe), `compliance_v2.py` (five prompt conditions),
  `harness.py` (code extraction, compilation, diagnostic classes), `analyze_v2.py`,
  `figures_v2.py`, `export_v2.py`, `prompts_shared.py`, `backends.py`.
- `probe/data/prompts100.csv` — the 100 task descriptions.
- `probe/knowledge_v2/`, `probe/knowledge_v2_think/` — every probe answer with its raw text
  (with reasoning enabled, the last 2,000 characters; the 27B server returned its reasoning
  separately, so only its final answers are kept).
- `data/compliance_v2.jsonl.gz` — all 12,000 generations: code, compiler verdict, rules,
  diagnostic classes and the compiler output of failed builds (last 60,000 characters;
  builds ran at the error log level, so successful builds store no output).
- `harness/ArkProject/` — the minimal HarmonyOS project every page was compiled in.

## Rerun the experiment

Requires the HarmonyOS command-line tools (SDK 5.0.0.71, API 12; hvigor 5.8.9) under
`~/deveco/command-line-tools`, Ollama for the small models, and optionally an
OpenAI-compatible server (llama.cpp) for the large one. Copy
`probe/backends.example.json` to `probe/backends.json` and edit it; API keys are read
from the environment variable it names, never from the file. Results go to MongoDB
(`ICAART_MONGO`).

    python3 probe/build_cases_v2.py
    python3 probe/knowledge_probe_v2.py --backend local
    python3 probe/compliance_v2.py --backend local --seed 1

## Citation

N. İbrahimoğlu and M. U. Demirezen. Telling Small Language Models the Rules: Knowledge,
Compliance and Compilation in ArkTS. In Proceedings of the 19th International Conference on
Agents and Artificial Intelligence (ICAART 2027), SCITEPRESS, 2027.

## Licences

Code: MIT (`LICENSE`). Data and generated outputs: CC BY 4.0. Model outputs are subject
to the licences of the models that produced them (Qwen, Llama 3.2 Community License,
Gemma Terms of Use, MIT for Phi-4-mini).
