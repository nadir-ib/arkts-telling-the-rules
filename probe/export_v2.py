"""Export the second protocol's records to a self-contained JSONL file for release.

The released artifact must reproduce every number without database access and must
carry nothing that identifies the machine or its users. This script writes one
record per generation and refuses to finish if any identifying pattern survives.

    ICAART_MONGO=mongodb://<host>:27017 python3 probe/export_v2.py artifact/data/compliance_v2.jsonl
"""
import json
import os
import re
import sys

import pymongo

FIELDS = ["model", "prompt_id", "condition", "seed", "protocol", "backend",
          "prompt_chars", "tokens", "prompt_tokens", "stop_reason", "extract_status",
          "code", "compiled", "rules", "n_rules", "diag_classes", "output", "error",
          "gen_seconds", "seconds"]
# anything that would identify a machine or a network; names of people to screen for are
# passed in ICAART_EXTRA_LEAKS (a regex) so that they never appear in released code
LEAKS = re.compile(re.escape(os.path.expanduser("~")) +
                   r"|\b192\.168\.\d+\.\d+|\b10\.\d+\.\d+\.\d+\b|mongodb://(?!127\.0\.0\.1|localhost)[a-z0-9]"
                   + (("|" + os.environ["ICAART_EXTRA_LEAKS"])
                      if os.environ.get("ICAART_EXTRA_LEAKS") else ""), re.I)


HOME = os.path.expanduser("~")


def clean(value):
    """Mask this machine's home directory and private addresses. Only exact matches
    are replaced, so model-written URLs such as https://example.com/home/... survive."""
    if isinstance(value, str):
        value = value.replace(HOME, "<home>")
        value = re.sub(r"\b192\.168\.\d+\.\d+\b", "<host>", value)
    return value


def main():
    out = sys.argv[1]
    coll = pymongo.MongoClient(os.environ["ICAART_MONGO"])["arkts_workflow_db"]["icaart_v2"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    backend_names = {"local": "ollama", "bigbox": "llamacpp"}
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for d in coll.find({}, {"_id": 0}).sort([("model", 1), ("prompt_id", 1),
                                                  ("condition", 1), ("seed", 1)]):
            rec = {k: clean(d.get(k)) for k in FIELDS}
            rec["backend"] = backend_names.get(rec["backend"], rec["backend"])
            line = json.dumps(rec, ensure_ascii=False)
            if LEAKS.search(line):
                sys.exit(f"identifying pattern survived in {d['model']} {d['prompt_id']} "
                         f"{d['condition']}: {LEAKS.search(line).group(0)!r}")
            fh.write(line + "\n")
            n += 1
    print(f"wrote {n} records to {out}")


if __name__ == "__main__":
    main()
