"""Shared generation-to-verdict pipeline: extract code, compile it, classify output.

Everything that turns a model response into a measurement lives here, so the three
arms cannot drift apart. Two defects in the first protocol are fixed here and are
the reason this module exists:

  1. A reasoning block cut off by the token limit has no closing </think>. The old
     regex only removed closed blocks, so the reasoning prose itself was written to
     Index.ets and compiled; English words tripped `arkts-no-in`, `arkts-no-with`
     and the comma rule. An unclosed block is now removed to the end of the text.
  2. An empty Index.ets compiles. Empty or code-free outputs are now recorded as
     `no_code` and never reach the compiler, so they cannot count as successes.

The compiler output is stored in full (the old record kept the last 600 characters,
which made most failures unclassifiable) with the project path replaced by a
placeholder, so stored records carry no machine-specific paths.
"""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_PROJECT = os.path.join(ROOT, "harness", "ArkProject")
DEVECO_BIN = os.path.expanduser("~/deveco/command-line-tools/bin")

HVIGOR = ("hvigorw clean --mode module assembleHap --analyze=advanced --parallel "
          "--no-daemon --no-incrimental --no-type-check --generate-build-profile "
          "--analyze --error")

RULE = re.compile(r"arkts-[a-z0-9\-]+")
THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)
FENCE = re.compile(r"```[a-zA-Z]*[ \t]*\n(.*?)(?:```|\Z)", re.S)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# one diagnostic: "N ERROR: ArkTS:ERROR File: <path>:L:C\n <message>"
DIAG = re.compile(r"ERROR: ArkTS:ERROR File: [^\n]*?\.ets:\d+:\d+\n(.+?)"
                  r"(?=\n\s*\n|\n\d+ ERROR|\nCOMPILE RESULT|\Z)", re.S)
OUTPUT_CAP = 60_000

# Diagnostic classes, first match wins. `rule` is a diagnostic carrying an arkts-*
# id; the others are what the id-only scoring of the first protocol could not see.
CLASSES = [
    ("rule", RULE.pattern),
    ("dialect_unlabelled", r"must specify a type|Use explicit types|explicitly declared "
                           r"class or interface|cannot be inferred|is not allowed in ArkTS|"
                           r"not supported in ArkTS"),
    ("arkui_dsl", r"UI component syntax|cannot be used in this place|root container|"
                  r"can not have any child|can only have the child|can only be nested|"
                  r"struct component is not allowed|build method|Decorator .* is missing"),
    ("resource", r"Unknown resource name"),
    ("module", r"Cannot find module|has no exported member|Module declaration names"),
    ("syntax", r"expected\.|Unterminated|Declaration or statement expected|"
               r"Invalid character|Unexpected token"),
    ("api", r"Cannot find name|does not exist on type|not assignable|not callable|"
            r"Expected \d|No overload|Type '|cannot be used as|Object is possibly"),
]


def strip_reasoning(text):
    """Remove reasoning blocks, closed or cut off. Some chat templates emit only the
    closing tag (the opening one is part of the prompt); keep what follows it."""
    text = THINK.sub("", text or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def extract_code(text):
    """-> (code, status). status is 'ok', or 'no_code' when nothing compilable is
    left after the reasoning is removed. The longest fenced block wins; unfenced
    text is accepted only if it contains a brace, i.e. is plausibly code."""
    text = strip_reasoning(text)
    blocks = [b.strip() for b in FENCE.findall(text) if b.strip()]
    if blocks:
        return max(blocks, key=len), "ok"
    if text and "{" in text:
        return text, "ok"
    return "", "no_code"


def classify(output):
    """-> {class: count} over the compiler's diagnostics."""
    counts = {}
    for msg in DIAG.findall(output):
        name = "other"
        for cls, pattern in CLASSES:
            if re.search(pattern, msg):
                name = cls
                break
        counts[name] = counts.get(name, 0) + 1
    return counts


def compile_code(code, project=DEFAULT_PROJECT, timeout=900):
    """-> dict(compiled, rules, diag_classes, output). Never called on empty code."""
    if not code.strip():
        raise ValueError("refusing to compile empty code; record it as no_code")
    index = os.path.join(project, "entry", "src", "main", "ets", "pages", "Index.ets")
    with open(index, "w", encoding="utf-8") as fh:
        fh.write(code)
    env = dict(os.environ)
    env["PATH"] = DEVECO_BIN + os.pathsep + env.get("PATH", "")
    try:
        proc = subprocess.run(HVIGOR, shell=True, cwd=project, env=env,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(compiled=False, rules=[], diag_classes={"timeout": 1},
                    output="TIMEOUT")
    out = ANSI.sub("", (proc.stdout or "") + (proc.stderr or ""))
    out = out.replace(os.path.abspath(project), "<project>")
    out = out.replace(os.path.expanduser("~"), "<home>")
    return dict(compiled=proc.returncode == 0,
                rules=sorted(set(RULE.findall(out))),
                diag_classes=classify(out),
                output=out[-OUTPUT_CAP:])


class IndexGuard:
    """Restore the project's Index.ets on exit, whatever happens in between."""

    def __init__(self, project=DEFAULT_PROJECT):
        self.index = os.path.join(project, "entry", "src", "main", "ets", "pages",
                                  "Index.ets")

    def __enter__(self):
        with open(self.index, encoding="utf-8") as fh:
            self.saved = fh.read()
        return self

    def __exit__(self, *exc):
        with open(self.index, "w", encoding="utf-8") as fh:
            fh.write(self.saved)
        return False
