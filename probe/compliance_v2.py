"""Compliance and in-context statement arms, second protocol (v2).

Single-shot generation, no repair loop. Differences from compliance_run.py (the
first protocol, kept unchanged so its numbers stay reproducible):

  * Reasoning is disabled for every model that has a switch, so the code a model
    emits is never cut off inside a reasoning block and truncation cannot depend
    on condition (it did: qwen3:8b 8 -> 39 truncated outputs in the first run).
  * Extraction and compilation go through harness.py: an unclosed reasoning block
    is removed, empty or code-free outputs are `no_code` and never compiled, and
    the full compiler output is stored and classified.
  * The Ollama context window is set explicitly (8192) and every condition gets
    the same token budget, so no part of a longer prompt can be evicted.
  * Five conditions, interleaved within each model so that drift in server state
    falls on all of them equally:

        scaffold  the page skeleton only                        (baseline)
        generic   + "ArkTS is restricted; violations are errors" (preamble only)
        placebo   + preamble + ten rarely violated rules, explained
        ids       + preamble + the ten taught rule ids, unexplained
        teach     + preamble + the ten taught rules, explained

    teach vs placebo isolates the content of the named rules from list length,
    format and the preamble; generic vs scaffold measures the preamble alone.
  * The taught list is the one fixed in the first protocol. Selection therefore
    happened on independent data, which removes regression to the mean by design.

    python3 probe/compliance_v2.py --backend local --seed 1
    python3 probe/compliance_v2.py --backend bigbox --models qwen3.8-27b --seed 1 \\
        --project harness/ArkProject_b

Resumable: (model, prompt_id, condition, seed) already recorded are skipped.
"""
import argparse
import os
import signal
import sys
import time

import pandas as pd
import pymongo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backends import load_backends, generate as backend_generate  # noqa: E402
from prompts_shared import TEACH, TAUGHT_RULES, SCAFFOLD  # noqa: E402
from harness import DEFAULT_PROJECT, IndexGuard, compile_code, extract_code  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROMPTS = os.path.join(HERE, "data", "prompts100.csv")

MONGO = os.environ.get("ICAART_MONGO", "mongodb://127.0.0.1:27017")
DB = "arkts_workflow_db"
COLLECTION = os.environ.get("ICAART_COLLECTION", "icaart_v2")

SMALL = ["qwen2.5-coder:3b", "llama3.2:3b", "phi4-mini:latest", "qwen3:4b",
         "gemma3:4b", "qwen2.5-coder:7b", "qwen3:8b"]

MAX_TOKENS = 3072
NUM_CTX = 8192
TEMPERATURE = 0.2

# The controls below are built to match teach's framing line by line.
HEADER = "Write ArkTS for HarmonyOS (ArkUI).\n\n"
_SKELETON_TAIL = TEACH[TEACH.index("\nComplete the ArkTS page below"):]

# The first protocol's rule list, with three explanations corrected against the
# compiler (each checked by compiling a minimal snippet):
#   3. the rule fires only where inference fails (a recursive function without an
#      annotation is rejected; `function two() { return 2; }` compiles);
#   4. `const xs = []` compiles; the rule fires on arrays of untyped object literals;
#   5. arkts-as-casts forbids the angle-bracket form `<T>v`; `v as T` compiles. The
#      first protocol told models to "avoid 'as' conversions", i.e. the opposite.
TEACH_CONSTRAINTS = """ArkTS is a restricted dialect of TypeScript. The following are
COMPILER ERRORS in ArkTS. You must not use them:
1. 'any' or 'unknown' types (arkts-no-any-unknown) - always give a concrete type.
2. Object literals without a declared class or interface type
   (arkts-no-untyped-obj-literals) - declare an interface first, then annotate.
3. Functions whose return type cannot be inferred, such as recursive ones
   (arkts-no-implicit-return-types) - write ': void', ': number', etc. on every
   function and method.
4. Array literals with elements whose type cannot be inferred, such as untyped
   object literals (arkts-no-noninferrable-arr-literals) - declare an interface
   and annotate the array, e.g. 'const xs: Item[] = [...]'.
5. Angle-bracket type casts such as '<number>value' (arkts-as-casts) - only
   'value as number' is supported.
6. The comma operator outside a for loop (arkts-no-comma-outside-loops).
7. Structural type literals used as types (arkts-no-obj-literals-as-types)
   - use a named interface instead of '{ a: number }' in a type position.
8. The 'in' operator (arkts-no-in).
9. 'var' declarations (arkts-no-var) - use 'let' or 'const'.
10. 'this' outside a class or struct method (arkts-no-standalone-this).
"""
TEACH_V2 = HEADER + TEACH_CONSTRAINTS + _SKELETON_TAIL

GENERIC_PREAMBLE = """ArkTS is a restricted dialect of TypeScript. Many TypeScript
constructs are COMPILER ERRORS in ArkTS. You must not use them.
"""

# Ten rules the compiler verifiably enforces (each fired on a probe case) that the
# seven small models violated at most once in 700 clean scaffold generations.
PLACEBO_RULES = [
    "arkts-no-ctor-prop-decls", "arkts-no-decl-merging", "arkts-no-func-apply-call",
    "arkts-no-ns-as-obj", "arkts-no-props-by-index", "arkts-no-structural-typing",
    "arkts-no-type-query", "arkts-no-types-in-catch", "arkts-no-delete",
    "arkts-no-enum-mixed-types",
]
PLACEBO = """ArkTS is a restricted dialect of TypeScript. The following are
COMPILER ERRORS in ArkTS. You must not use them:
1. Parameter properties in constructors (arkts-no-ctor-prop-decls) - declare
   fields in the class body instead of 'constructor(private x: number)'.
2. Declaring the same interface or class twice to merge them
   (arkts-no-decl-merging) - declare each type once.
3. Function.apply and Function.call (arkts-no-func-apply-call) - call the
   function directly.
4. Namespaces used as objects (arkts-no-ns-as-obj) - refer to exported
   members directly, as 'NS.member'.
5. Accessing fields by string index (arkts-no-props-by-index) - write
   'obj.field', not "obj['field']".
6. Structural typing between unrelated classes (arkts-no-structural-typing)
   - a class does not match another class's type by shape alone.
7. 'typeof' in a type position (arkts-no-type-query) - write the type name.
8. Type annotations on catch clauses (arkts-no-types-in-catch) - write
   'catch (e)', not 'catch (e: Error)'.
9. The 'delete' operator (arkts-no-delete) - declare the field optional and
   assign undefined instead.
10. Enums mixing number and string members (arkts-no-enum-mixed-types)
   - give every member of one enum the same type.
"""

IDS = ("ArkTS is a restricted dialect of TypeScript. The following are\n"
       "COMPILER ERRORS in ArkTS. You must not use them:\n"
       + "".join(f"{i}. {r}\n" for i, r in enumerate(TAUGHT_RULES, 1)))

CONDITIONS = {
    "scaffold": SCAFFOLD,
    "generic": HEADER + GENERIC_PREAMBLE + _SKELETON_TAIL,
    "placebo": HEADER + PLACEBO + _SKELETON_TAIL,
    "ids": HEADER + IDS + _SKELETON_TAIL,
    "teach": TEACH_V2,
}
ORDER = ["scaffold", "generic", "placebo", "ids", "teach"]


def _stop(signum, frame):
    # turn SIGTERM into an exception so IndexGuard restores Index.ets
    raise SystemExit(f"stopped by signal {signum}")


def main():
    signal.signal(signal.SIGTERM, _stop)
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local")
    ap.add_argument("--models", default=",".join(SMALL))
    ap.add_argument("--conditions", default=",".join(ORDER))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    args = ap.parse_args()

    backend = load_backends()[args.backend]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    project = os.path.abspath(args.project)
    prompts = pd.read_csv(PROMPTS).head(args.n)["prompt"].tolist()
    coll = pymongo.MongoClient(MONGO)[DB][COLLECTION]
    done = {(d["model"], d["prompt_id"], d["condition"])
            for d in coll.find({"seed": args.seed},
                               {"model": 1, "prompt_id": 1, "condition": 1})}
    total = len(models) * len(prompts) * len(conditions)
    todo = total - sum(1 for m in models for i in range(1, len(prompts) + 1)
                       for c in conditions if (m, f"prompt_{i}", c) in done)
    print(f"v2  backend={args.backend}  seed={args.seed}  project={project}")
    print(f"{len(models)} models x {len(prompts)} prompts x {len(conditions)} "
          f"conditions = {total} ({todo} to run)\n", flush=True)

    t_start, seen = time.time(), 0
    with IndexGuard(project):
        for model in models:
            for i, prompt in enumerate(prompts, 1):
                pid = f"prompt_{i}"
                for cond in conditions:
                    if (model, pid, cond) in done:
                        continue
                    text = CONDITIONS[cond].replace("{prompt}", prompt)
                    meta, t0 = {}, time.time()
                    rec = dict(model=model, prompt_id=pid, prompt=prompt,
                               condition=cond, seed=args.seed, protocol="v2",
                               backend=args.backend, prompt_chars=len(text))
                    try:
                        _, tokens, stop = backend_generate(
                            backend, model, text, max_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE, seed=args.seed,
                            num_ctx=NUM_CTX, no_think=True, meta=meta)
                        rec["gen_seconds"] = time.time() - t0
                        code, status = extract_code(meta.get("raw", ""))
                        rec.update(tokens=tokens, stop_reason=stop,
                                   prompt_tokens=meta.get("prompt_tokens", 0),
                                   raw=meta.get("raw", "")[:40_000], code=code,
                                   extract_status=status, error="")
                        if status == "ok":
                            rec.update(compile_code(code, project))
                        else:
                            rec.update(compiled=False, rules=[], diag_classes={},
                                       output="")
                    except Exception as exc:                  # noqa: BLE001
                        rec.update(compiled=False, rules=[], diag_classes={},
                                   output="", code="", extract_status="error",
                                   error=f"{type(exc).__name__}: {exc}")
                    rec["n_rules"] = len(rec["rules"])
                    rec["seconds"] = time.time() - t0
                    rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    coll.insert_one(rec)
                    seen += 1
                    eta = (time.time() - t_start) / seen * (todo - seen) / 3600
                    print(f"[{model}] {pid} {cond:<8} "
                          f"{'PASS' if rec['compiled'] else rec['extract_status'].upper() if rec['extract_status'] != 'ok' else 'FAIL'} "
                          f"rules={','.join(rec['rules']) or '-'} stop={rec.get('stop_reason', '')} "
                          f"{rec['seconds']:.0f}s | {seen}/{todo} ETA {eta:.1f}h",
                          flush=True)
    print(f"\ndone: {coll.count_documents({'seed': args.seed})} records for seed {args.seed}")


if __name__ == "__main__":
    main()
