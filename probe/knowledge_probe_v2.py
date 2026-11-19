"""Knowledge probe, second protocol: six wordings, a strict parser, raw answers kept.

Changes from knowledge_probe4.py, each fixing an audit finding:

  * The answer parser matched substrings, so "NO" inside "ANNOTATION" or "NOT" was
    read as an answer. It now matches whole words only, and an answer containing
    both words (or neither) is unparseable.
  * Raw responses were not stored, so no parse could be audited. They are now.
  * Reasoning is disabled, as in the compliance arm, so an answer cannot be lost
    inside a reasoning block cut off by the token limit.
  * Six wordings instead of four, three per polarity, and the item set of
    build_cases_v2.py (every rule named in the intervention has its own items).

Scoring is left to probe/analyze_v2.py; this script only collects.

    python3 probe/knowledge_probe_v2.py --backend local
    python3 probe/knowledge_probe_v2.py --backend bigbox --models qwen3.8-27b

Output: probe/knowledge_v2/<model>.jsonl, one line per (case, wording); resumable.
With --think (exploratory, added after the no-reasoning run showed a strong "no"
response bias in two models): reasoning on, 6,144-token budget, output in
probe/knowledge_v2_think/.
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from backends import load_backends, generate as backend_generate  # noqa: E402
from harness import strip_reasoning  # noqa: E402

OUT_DIR = os.path.join(HERE, "knowledge_v2")
SMALL = ["qwen2.5-coder:3b", "llama3.2:3b", "phi4-mini:latest", "qwen3:4b",
         "gemma3:4b", "qwen2.5-coder:7b", "qwen3:8b"]

PREFIX = ("In ArkTS, the restricted TypeScript dialect used by HarmonyOS, "
          "consider this construct:\n\n{snippet}\n\n")

# (id, question, word meaning "forbidden", word meaning "allowed").
# P* ask about permission (a "yes" means allowed), N* about prohibition.
WORDINGS = [
    ("P1", "Is it permitted? Reply with exactly one word: ALLOWED or FORBIDDEN.",
     "FORBIDDEN", "ALLOWED"),
    ("P2", "Does ArkTS allow it? Reply with exactly one word: YES or NO.", "NO", "YES"),
    ("P3", "Would it compile without errors under the ArkTS compiler? Reply with "
           "exactly one word: YES or NO.", "NO", "YES"),
    ("N1", "Is it forbidden? Reply with exactly one word: YES or NO.", "YES", "NO"),
    ("N2", "Would the ArkTS compiler reject it? Reply with exactly one word: YES or NO.",
     "YES", "NO"),
    ("N3", "Does it violate an ArkTS language rule? Reply with exactly one word: "
           "YES or NO.", "YES", "NO"),
]


def read(text, forbid_word, allow_word):
    """-> True if the answer says forbidden, False if allowed, None if unparseable.
    The first word decides when it is one of the two; otherwise exactly one of the
    two must occur as a whole word."""
    words = re.findall(r"[A-Z]+", strip_reasoning(text).upper())
    if not words:
        return None
    if words[0] in (forbid_word, allow_word):
        return words[0] == forbid_word
    hits = {w for w in words if w in (forbid_word, allow_word)}
    if len(hits) == 1:
        return hits.pop() == forbid_word
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local")
    ap.add_argument("--models", default=",".join(SMALL))
    ap.add_argument("--think", action="store_true",
                    help="exploratory arm: reasoning on, larger budget, separate output")
    args = ap.parse_args()
    out_dir = OUT_DIR + ("_think" if args.think else "")

    backend = load_backends()[args.backend]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = [c for c in json.load(open(os.path.join(HERE, "cases_v2.json")))
             if c["usable"]]
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(cases)} cases x {len(WORDINGS)} wordings x {len(models)} models",
          flush=True)

    for model in models:
        path = os.path.join(out_dir, model.replace(":", "_").replace("/", "_") + ".jsonl")
        done = set()
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                done.add((r["case"], r["wording"]))
        t0, n = time.time(), 0
        with open(path, "a", encoding="utf-8") as fh:
            for case in cases:
                for wid, question, forbid_w, allow_w in WORDINGS:
                    if (case["name"], wid) in done:
                        continue
                    meta = {}
                    try:
                        backend_generate(backend, model,
                                         PREFIX.format(snippet=case["snippet"]) + question,
                                         max_tokens=6144 if args.think else 256,
                                         temperature=0.0, seed=1,
                                         num_ctx=8192 if args.think else 4096,
                                         no_think=not args.think, meta=meta)
                        raw, err = meta.get("raw", ""), ""
                    except Exception as exc:                  # noqa: BLE001
                        raw, err = "", f"{type(exc).__name__}: {exc}"
                    fh.write(json.dumps(dict(
                        model=model, case=case["name"], wording=wid,
                        truth_forbidden=case["compiler_verdict"] == "forbidden",
                        target=case.get("target"),
                        says_forbidden=read(raw, forbid_w, allow_w),
                        raw=raw[-2000:], think=args.think, error=err)) + "\n")
                    fh.flush()
                    n += 1
        print(f"{model:<22} {n} answers in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
