"""Every confirmatory number of the second protocol, as fixed in
paper/ANALYSIS_PLAN_v2.md. Prints a report and writes paper/numbers_v2.json, which is
the only source the manuscript's v2 numbers may be copied from.

    ICAART_MONGO=mongodb://<host>:27017 python3 probe/analyze_v2.py [--jsonl FILE]

With --jsonl the records are read from an exported file instead of MongoDB (see
probe/export_v2.py), which is how the released artifact reproduces the numbers.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest, fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from prompts_shared import TAUGHT_RULES  # noqa: E402
from compliance_v2 import PLACEBO_RULES, SMALL  # noqa: E402

warnings.filterwarnings("ignore")
TAUGHT, PLACEBO = set(TAUGHT_RULES), set(PLACEBO_RULES)
BIG = "qwen3.8-27b"
CONDS = ["scaffold", "generic", "placebo", "ids", "teach"]
# markup such as <Column> ... </Column>: the compiler reads JSX as angle-bracket casts
JSX = re.compile(r"<([A-Z][A-Za-z]*)[\s>/][\s\S]*?</\1\s*>")
N_PERM = 10_000
NOT_RULES = {"arkts-entry"}  # not a rule id: part of a module path in one compiler message
OUT = {}


def put(key, value):
    OUT[key] = value
    return value


# ---------------------------------------------------------------- loading
def load(jsonl=None):
    if jsonl:
        import gzip
        opener = gzip.open if jsonl.endswith(".gz") else open
        with opener(jsonl, "rt", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
    else:
        import pymongo
        coll = pymongo.MongoClient(os.environ["ICAART_MONGO"])["arkts_workflow_db"]["icaart_v2"]
        rows = list(coll.find({}, {"_id": 0, "raw": 0}))
    df = pd.DataFrame(rows)
    # the rule regex also matched a module path: "Cannot find module '@entry/arkts-entry'" (one page)
    df["rules"] = df["rules"].apply(lambda r: set(r or []) - NOT_RULES)
    df["T"] = df["rules"].apply(lambda r: len(r & TAUGHT))
    df["U"] = df["rules"].apply(lambda r: len(r - TAUGHT))
    df["Pl"] = df["rules"].apply(lambda r: len(r & PLACEBO))
    df["has_code"] = df["extract_status"] == "ok"
    df["chars"] = df["code"].fillna("").str.len()
    df["jsx"] = df["code"].fillna("").apply(lambda c: bool(JSX.search(c)))
    df["truncated"] = df["stop_reason"].isin(["length"])
    df["clean"] = df["has_code"] & (df["T"] + df["U"] == 0)
    df["unit"] = df["prompt_id"] + "|" + df["seed"].astype(str)
    return df


# ---------------------------------------------------------------- statistics
def poisson_contrast(d, outcome, a, b):
    """Rate ratio of condition a vs b for one model: Poisson GLM, SEs clustered by
    prompt. -> (rr, lo, hi, p) or None when either arm has no events."""
    sub = d[d["condition"].isin([a, b])].copy()
    if sub.groupby("condition")[outcome].sum().min() == 0:
        return None
    sub["x"] = (sub["condition"] == a).astype(float)
    groups = pd.factorize(sub["prompt_id"])[0]
    fit = smf.glm(f"{outcome} ~ x", sub, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    c, se = fit.params["x"], fit.bse["x"]
    return math.exp(c), math.exp(c - 1.96 * se), math.exp(c + 1.96 * se), fit.pvalues["x"]


def pooled_contrast(d, outcome, a, b):
    sub = d[d["condition"].isin([a, b])].copy()
    # models with no events in either arm carry no information about the ratio
    keep = sub.groupby("model")[outcome].transform("sum") > 0
    sub = sub[keep]
    if sub.empty or sub.groupby("condition")[outcome].sum().min() == 0:
        return None
    sub["x"] = (sub["condition"] == a).astype(float)
    groups = pd.factorize(sub["prompt_id"])[0]
    formula = f"{outcome} ~ x + C(model)" if sub["model"].nunique() > 1 else f"{outcome} ~ x"
    fit = smf.glm(formula, sub, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    c, se = fit.params["x"], fit.bse["x"]
    return math.exp(c), math.exp(c - 1.96 * se), math.exp(c + 1.96 * se), fit.pvalues["x"]


def perm_test(d, outcome, a, b, seed=0):
    """Paired sign-flip test, clustered by prompt: all models' and seeds' differences
    for one prompt are flipped together. Statistic: total difference a - b."""
    wide = d[d["condition"].isin([a, b])].pivot_table(
        index=["model", "prompt_id", "seed"], columns="condition", values=outcome)
    wide = wide.dropna()
    diff = (wide[a] - wide[b]).groupby(level="prompt_id").sum()
    obs = diff.sum()
    vals = diff.values
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(N_PERM, len(vals)))
    null = (signs * vals).sum(axis=1)
    k = int((null <= obs).sum()) if obs < 0 else int((null >= obs).sum())
    return float(obs), (k + 1) / (N_PERM + 1)


def interaction(d, a, b):
    """condition (a vs b) x rule set (taught vs other): ratio of rate ratios."""
    sub = d[d["condition"].isin([a, b])]
    if min(sub.groupby("condition")["U"].sum().min(),
           sub.groupby("condition")["T"].sum().min()) == 0:
        return None
    long = pd.concat([
        sub.assign(y=sub["T"], taught=1.0), sub.assign(y=sub["U"], taught=0.0)])
    long["x"] = (long["condition"] == a).astype(float)
    groups = pd.factorize(long["prompt_id"])[0]
    fit = smf.glm("y ~ C(model):taught + C(model) + x + x:taught", long,
                  family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    c, se = fit.params["x:taught"], fit.bse["x:taught"]
    return math.exp(c), math.exp(c - 1.96 * se), math.exp(c + 1.96 * se), fit.pvalues["x:taught"]


def fmt_rr(r):
    return "not estimable (no events)" if r is None else \
        f"RR {r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}] p={r[3]:.2g}"


def contrast_table(df, models, outcome, a, b, label):
    print(f"\n--- {label}: {outcome}, {a} vs {b}")
    res = {m: poisson_contrast(df[df.model == m], outcome, a, b) for m in models}
    ps = [(m, r[3]) for m, r in res.items() if r is not None]
    holm = dict(zip([m for m, _ in ps],
                    multipletests([p for _, p in ps], method="holm")[1])) if ps else {}
    rows = {}
    for m in models:
        r = res[m]
        h = holm.get(m)
        print(f"  {m:<20} {fmt_rr(r)}" + (f"  Holm p={h:.2g}" if h is not None else ""))
        rows[m] = None if r is None else dict(rr=r[0], lo=r[1], hi=r[2], p=r[3], holm=h)
    small = df[df.model.isin(models)]
    pooled = pooled_contrast(small, outcome, a, b)
    obs, pp = perm_test(small, outcome, a, b)
    below = sum(1 for r in res.values() if r is not None and r[0] < 1)
    sig = sum(1 for m, h in holm.items() if h < 0.05)
    print(f"  pooled {fmt_rr(pooled)} | {below}/{len(models)} below 1, "
          f"{sig} Holm-significant | permutation total {obs:+.0f}, p={pp:.2g}")
    put(f"{label}", dict(per_model=rows,
                         pooled=None if pooled is None else dict(zip(["rr", "lo", "hi", "p"], pooled)),
                         below_one=below, holm_significant=sig,
                         perm_total=obs, perm_p=pp))


# ---------------------------------------------------------------- sections
def integrity(df):
    print("=" * 88 + "\n0. INTEGRITY\n" + "=" * 88)
    g = df.groupby(["model", "condition"])
    t = pd.DataFrame({"n": g.size(), "no_code": g["has_code"].apply(lambda s: (~s).sum()),
                      "trunc": g["truncated"].sum(), "errors": g["error"].apply(lambda s: (s != "").sum()),
                      "chars": g["chars"].mean().round(0), "jsx": g["jsx"].mean().round(3),
                      "compiled": g["compiled"].sum()})
    print(t.to_string())
    put("integrity", {f"{m}|{c}": r for (m, c), r in t.to_dict("index").items()})


def compliance(df):
    small = [m for m in SMALL if m in set(df.model)]
    coded = df[df["has_code"]]
    print("\n" + "=" * 88 + "\n1. CONFIRMATORY CONTRASTS (generations with code)\n" + "=" * 88)
    contrast_table(coded, small, "T", "teach", "scaffold", "H1_teach_vs_scaffold")
    contrast_table(coded, small, "T", "teach", "placebo", "H2_teach_vs_placebo")
    contrast_table(coded, small, "T", "generic", "scaffold", "H3_generic_vs_scaffold_T")
    contrast_table(coded, small, "U", "generic", "scaffold", "H3_generic_vs_scaffold_U")
    contrast_table(coded, small, "T", "teach", "ids", "H4_teach_vs_ids")
    contrast_table(coded, small, "U", "teach", "placebo", "untaught_teach_vs_placebo")
    contrast_table(coded, small, "T", "placebo", "scaffold", "placebo_vs_scaffold_T")
    contrast_table(coded, small, "T", "teach", "generic", "teach_vs_generic_T")
    contrast_table(coded, small, "T", "ids", "scaffold", "ids_vs_scaffold_T")
    r = interaction(coded[coded.model.isin(small)], "teach", "placebo")
    print(f"\n  specificity interaction (teach vs placebo x taught vs other): {fmt_rr(r)}")
    put("H2_interaction", None if r is None else dict(zip(["rr", "lo", "hi", "p"], r)))

    print("\n--- sensitivity: no_code counted as zero violations")
    for a, b, lab in [("teach", "scaffold", "H1"), ("teach", "placebo", "H2")]:
        p = pooled_contrast(df[df.model.isin(small)], "T", a, b)
        print(f"  {lab} pooled incl. no_code: {fmt_rr(p)}")
        put(f"{lab}_incl_no_code", None if p is None else dict(zip(["rr", "lo", "hi", "p"], p)))

    print("\n--- secondary: length-adjusted (direct effect at fixed length)")
    for a, b, lab in [("teach", "scaffold", "H1"), ("teach", "placebo", "H2")]:
        sub = coded[coded.model.isin(small) & coded.condition.isin([a, b])].copy()
        sub["x"] = (sub.condition == a).astype(float)
        sub["loglen"] = np.log(sub["chars"].clip(lower=1))
        fit = smf.glm("T ~ x + loglen + C(model)", sub, family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.prompt_id)[0]})
        c, se = fit.params["x"], fit.bse["x"]
        print(f"  {lab}: RR {math.exp(c):.2f} [{math.exp(c-1.96*se):.2f}, "
              f"{math.exp(c+1.96*se):.2f}] p={fit.pvalues['x']:.2g}, length exponent "
              f"{fit.params['loglen']:.2f}")
        put(f"{lab}_length_adjusted", dict(rr=math.exp(c), p=fit.pvalues["x"],
                                           b=fit.params["loglen"]))

    print("\n--- H5 compilation, teach vs scaffold (paired by prompt and seed)")
    res, disc_a, disc_b = {}, 0, 0
    for m in small + ([BIG] if BIG in set(df.model) else []):
        w = df[df.model == m].pivot_table(index=["prompt_id", "seed"], columns="condition",
                                          values="compiled", aggfunc="first")
        w = w.dropna(subset=["teach", "scaffold"]).astype(bool)
        a = int((w.teach & ~w.scaffold).sum())
        b = int((~w.teach & w.scaffold).sum())
        p = binomtest(a, a + b, 0.5).pvalue if a + b else float("nan")
        res[m] = dict(scaffold=int(w.scaffold.sum()), teach=int(w.teach.sum()),
                      only_teach=a, only_scaffold=b, p=p)
        if m in small:
            disc_a, disc_b = disc_a + a, disc_b + b
        print(f"  {m:<20} {w.scaffold.sum():>3} -> {w.teach.sum():>3}   discordant "
              f"{a}/{b}  exact p={p:.2g}")
    pp = binomtest(disc_a, disc_a + disc_b, 0.5).pvalue if disc_a + disc_b else float("nan")
    print(f"  small panel pooled discordant {disc_a}/{disc_b}, exact p={pp:.2g}")
    put("H5_compile", dict(per_model=res, pooled_only_teach=disc_a,
                           pooled_only_scaffold=disc_b, pooled_p=pp))


def compilation(df):
    print("\n" + "=" * 88 + "\n2. WHAT BLOCKS COMPILATION (scaffold)\n" + "=" * 88)
    s = df[(df.condition == "scaffold") & df.has_code]
    rows = {}
    for m in [m for m in SMALL + [BIG] if m in set(s.model)]:
        d = s[s.model == m]
        clean = d[d.clean]
        k, n = int(clean.compiled.sum()), len(clean)
        lo, hi = proportion_confint(k, n, method="wilson") if n else (float("nan"),) * 2
        fails = clean[~clean.compiled]
        cls = defaultdict(int)
        for dc in fails["diag_classes"]:
            for c in (dc or {}):
                cls[c] += 1
        rows[m] = dict(n=len(d), clean_share=n / len(d), compiled=int(d.compiled.sum()),
                       compile_given_clean=(k / n) if n else None, wilson=(lo, hi),
                       clean_fail_classes=dict(cls), jsx_share=float(d.jsx.mean()))
        print(f"  {m:<20} clean {n:>3}/{len(d):<3} compile|clean "
              f"{(k/n if n else float('nan')):.2f} [{lo:.2f},{hi:.2f}]  jsx {d.jsx.mean():.2f}"
              f"  clean-failure classes {dict(cls)}")
    put("compilation", rows)


def knowledge(subdir="knowledge_v2", key="knowledge", title="six wordings, reasoning off"):
    print("\n" + "=" * 88 + f"\n3. KNOWLEDGE PROBE ({title})\n" + "=" * 88)
    files = sorted(glob.glob(os.path.join(HERE, subdir, "*.jsonl")))
    if not files:
        print("  no probe results yet")
        return None
    recs = [json.loads(l) for f in files for l in open(f, encoding="utf-8")]
    kd = pd.DataFrame(recs)
    per_case = kd.groupby(["model", "case"]).agg(
        truth=("truth_forbidden", "first"), target=("target", "first"),
        answers=("says_forbidden", list)).reset_index()

    def majority(a):
        v = [x for x in a if x is not None]
        f = sum(v)
        return None if not v or 2 * f == len(v) else 2 * f > len(v)

    per_case["majority"] = per_case.answers.apply(majority)
    per_case["stable"] = per_case.answers.apply(
        lambda a: a[0] if all(x is not None for x in a) and len(set(a)) == 1 else None)
    rows, ps = {}, []
    for m, d in per_case.groupby("model"):
        out = {"unparsed": int(kd[(kd.model == m)].says_forbidden.isna().sum()),
               "answers": int((kd.model == m).sum())}
        for scheme in ("majority", "stable"):
            e = d.dropna(subset=[scheme])
            tp = int(((e.truth) & (e[scheme] == True)).sum())   # noqa: E712
            fn = int(((e.truth) & (e[scheme] == False)).sum())  # noqa: E712
            fp = int(((~e.truth) & (e[scheme] == True)).sum())  # noqa: E712
            tn = int(((~e.truth) & (e[scheme] == False)).sum())  # noqa: E712
            sens = tp / (tp + fn) if tp + fn else None
            spec = tn / (tn + fp) if tn + fp else None
            bal = (sens + spec) / 2 if sens is not None and spec is not None else None
            p = fisher_exact([[tp, fn], [fp, tn]], alternative="greater")[1] \
                if (tp + fn) and (fp + tn) else None
            out[scheme] = dict(n=len(e), tp=tp, fn=fn, fp=fp, tn=tn, sens=sens,
                               spec=spec, bal=bal, fisher_p=p)
        rows[m] = out
        if m in SMALL and out["majority"]["fisher_p"] is not None:
            ps.append((m, out["majority"]["fisher_p"]))
    holm = dict(zip([m for m, _ in ps], multipletests([p for _, p in ps], method="holm")[1]))
    cases = {c["name"]: c for c in json.load(open(os.path.join(HERE, "cases_v2.json")))}
    for m, d in per_case.groupby("model"):
        fps = sorted(d[(~d.truth) & (d.majority == True)].case)          # noqa: E712
        rows[m]["majority_false_positives"] = fps
        rows[m]["fp_where_compiler_overruled_prior"] = [
            c for c in fps if cases[c]["expected"] != cases[c]["compiler_verdict"]]
    rows["_prior_overruled_cases"] = sorted(
        n for n, c in cases.items() if c["usable"] and c["expected"] != c["compiler_verdict"])
    for m, out in [(k, v) for k, v in rows.items() if not k.startswith("_")]:
        mj, st = out["majority"], out["stable"]
        f = lambda v: "  -  " if v is None else f"{v:.2f}"   # noqa: E731
        print(f"  {m:<20} majority n={mj['n']:>2} bal {f(mj['bal'])} sens {f(mj['sens'])} "
              f"spec {f(mj['spec'])} Fisher p={mj['fisher_p'] if mj['fisher_p'] is None else round(mj['fisher_p'], 4)}"
              f" Holm {round(holm[m], 3) if m in holm else '-'} | stable n={st['n']:>2} "
              f"bal {f(st['bal'])} | unparsed {out['unparsed']}/{out['answers']}")
        out["holm_majority"] = holm.get(m)
    for m in [k for k in rows if not k.startswith("_")]:
        sub = kd[(kd.model == m) & (kd.wording != "P1") & kd.says_forbidden.notna()]
        # "YES" means forbidden in the N wordings and allowed in P2/P3; derived from
        # the parsed answer, not the raw text, which may start with reasoning
        yes = np.where(sub.wording.str.startswith("N"), sub.says_forbidden,
                       ~sub.says_forbidden.astype(bool))
        rows[m]["yes_share"] = float(np.mean(yes.astype(bool)))
        print(f"  {m:<20} share of YES among yes/no answers {rows[m]['yes_share']:.2f}")
    put(key, rows)
    return per_case


def rule_level(df, per_case):
    print("\n" + "=" * 88 + "\n4. RULE LEVEL: does telling help more where the rule is known?\n" + "=" * 88)
    if per_case is None:
        print("  needs probe results")
        return
    knows = {}
    for (m, rule), d in per_case[per_case.target.isin(TAUGHT)].groupby(["model", "target"]):
        forb = d[d.truth]
        knows[(m, rule)] = bool(len(forb)) and all(v is True for v in forb.majority)
    coded = df[df.has_code & df.model.isin(SMALL) & df.condition.isin(["teach", "placebo"])]
    long = []
    for rule in sorted(TAUGHT):
        sub = coded.assign(y=coded.rules.apply(lambda r, rule=rule: float(rule in r)),
                           rule=rule)
        sub["knows"] = [float(knows.get((m, rule), False)) for m in sub.model]
        long.append(sub)
    long = pd.concat(long)
    long["x"] = (long.condition == "teach").astype(float)
    counts = long.groupby(["knows", "condition"]).y.sum()
    print("  violations by known / not known:\n" + counts.to_string())
    try:
        fit = smf.glm("y ~ C(model) + C(rule) + x * knows", long,
                      family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(long.prompt_id)[0]})
        c, se = fit.params["x:knows"], fit.bse["x:knows"]
        r = (math.exp(c), math.exp(c - 1.96 * se), math.exp(c + 1.96 * se),
             fit.pvalues["x:knows"])
        print(f"  knows x teach interaction: {fmt_rr(r)}  (<1: telling helps more where known)")
        put("rule_level", dict(interaction=dict(zip(["rr", "lo", "hi", "p"], r)),
                               known_cells=int(sum(knows.values())), cells=len(knows),
                               violations_known=int(long[long.knows == 1].y.sum()),
                               violations_unknown=int(long[long.knows == 0].y.sum())))
    except Exception as exc:                                    # noqa: BLE001
        print(f"  model failed: {exc}")


def hksj(df, models, outcome, a, b):
    """Random-effects meta-analysis over per-model log rate ratios with the
    Hartung-Knapp-Sidik-Jonkman small-k interval: the estimate for a new model."""
    from scipy.stats import t as tdist
    ys, vs = [], []
    for m in models:
        r = poisson_contrast(df[df.model == m], outcome, a, b)
        if r is None:
            continue
        ys.append(math.log(r[0]))
        vs.append(((math.log(r[2]) - math.log(r[1])) / 3.92) ** 2)
    ys, vs = np.array(ys), np.array(vs)
    k, w = len(ys), 1 / vs
    fe = (w * ys).sum() / w.sum()
    q = (w * (ys - fe) ** 2).sum()
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (vs + tau2)
    mu = (ws * ys).sum() / ws.sum()
    se = math.sqrt((ws * (ys - mu) ** 2).sum() / (k - 1) / ws.sum())
    crit = tdist.ppf(0.975, k - 1)
    p = 2 * tdist.sf(abs(mu / se), k - 1)
    pi = tdist.ppf(0.975, k - 2) * math.sqrt(tau2 + se ** 2) if k > 2 else float("nan")
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return dict(rr=math.exp(mu), lo=math.exp(mu - crit * se), hi=math.exp(mu + crit * se),
                p=p, i2=i2, tau2=tau2, pred_lo=math.exp(mu - pi), pred_hi=math.exp(mu + pi), k=k)


def robustness(df):
    print("\n" + "=" * 88 + "\n5. ROBUSTNESS (reported beside the confirmatory tests)\n" + "=" * 88)
    small = [m for m in SMALL if m in set(df.model)]
    coded = df[df.has_code & df.model.isin(small)].copy()
    res = {}
    for lab, a, b in [("H1", "teach", "scaffold"), ("H2", "teach", "placebo"),
                      ("H4", "teach", "ids"), ("placebo_vs_scaffold", "placebo", "scaffold"),
                      ("generic_vs_scaffold", "generic", "scaffold"),
                      ("teach_vs_generic", "teach", "generic")]:
        r = hksj(coded, small, "T", a, b)
        res[f"{lab}_hksj"] = r
        print(f"  {lab:<20} random effects RR {r['rr']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] "
              f"p={r['p']:.2g} I2={r['i2']:.2f} prediction [{r['pred_lo']:.2f}, {r['pred_hi']:.2f}]")
    two = ["gemma3:4b", "qwen3:8b"]
    rest = coded[~coded.model.isin(two)]
    for lab, a, b in [("H1", "teach", "scaffold"), ("H2", "teach", "placebo"), ("H4", "teach", "ids")]:
        r = pooled_contrast(rest, "T", a, b)
        res[f"{lab}_without_two"] = dict(zip(["rr", "lo", "hi", "p"], r))
        print(f"  {lab} without gemma3/qwen3:8b: {fmt_rr(r)}")
    for sd in sorted(coded.seed.unique()):
        for lab, a, b in [("H1", "teach", "scaffold"), ("H2", "teach", "placebo")]:
            r = pooled_contrast(coded[coded.seed == sd], "T", a, b)
            res[f"{lab}_seed{sd}"] = dict(zip(["rr", "lo", "hi", "p"], r))
            print(f"  {lab} seed {sd}: {fmt_rr(r)}")
    for m in two:
        per = [poisson_contrast(coded[(coded.model == m) & (coded.seed == sd)], "T", "teach",
                                "placebo")[0] for sd in sorted(coded.seed.unique())]
        res[f"H2_per_seed_{m}"] = per
        print(f"  H2 per seed {m}: {', '.join(f'{x:.2f}' for x in per)}")
    r = pooled_contrast(coded[~coded.jsx], "T", "teach", "scaffold")
    res["H1_no_jsx"] = dict(zip(["rr", "lo", "hi", "p"], r))
    print(f"  H1 excluding generations containing JSX: {fmt_rr(r)}")
    # is the mismatched-rule list inert? other rules, placebo rules excluded
    coded["U2"] = coded.rules.apply(lambda r: len(r - TAUGHT - PLACEBO))
    for lab, out, a, b in [("placebo_vs_generic_other", "U2", "placebo", "generic"),
                           ("placebo_vs_generic_T", "T", "placebo", "generic"),
                           ("teach_vs_generic_other", "U2", "teach", "generic")]:
        r = pooled_contrast(coded, out, a, b)
        res[lab] = dict(zip(["rr", "lo", "hi", "p"], r))
        print(f"  {lab}: {fmt_rr(r)}")
    for a, b in [("teach", "generic"), ("teach", "ids")]:
        r = interaction(coded, a, b)
        res[f"interaction_{a}_vs_{b}"] = dict(zip(["rr", "lo", "hi", "p"], r))
        print(f"  interaction {a} vs {b} x named/other: {fmt_rr(r)}")
    comp = df[df.model.isin(small)].groupby("condition").compiled.sum().astype(int).to_dict()
    res["compiled_by_condition"] = comp
    n_cond = int((df.model.isin(small) & (df.condition == "scaffold")).sum())
    print(f"  compiled of {n_cond} per condition: {comp}")
    for a, b in [("placebo", "scaffold"), ("generic", "scaffold"), ("ids", "scaffold"), ("teach", "placebo")]:
        w = df[df.model.isin(small)].pivot_table(index=["model", "prompt_id", "seed"],
                                                 columns="condition", values="compiled",
                                                 aggfunc="first").astype(bool)
        x, y = int((w[a] & ~w[b]).sum()), int((~w[a] & w[b]).sum())
        p = binomtest(x, x + y, 0.5).pvalue
        res[f"compile_{a}_vs_{b}"] = dict(only_a=x, only_b=y, p=p)
        print(f"  compile {a} vs {b}: discordant {x}/{y}, exact p={p:.2g}")
    g = coded.groupby("condition")
    res["mean_T"] = g["T"].mean().round(3).to_dict()
    res["any_T_share"] = g["T"].apply(lambda s: float((s > 0).mean())).round(3).to_dict()
    print(f"  mean named violations per generation: {res['mean_T']}")
    print(f"  share with at least one named violation: {res['any_T_share']}")
    sets = df.groupby(["model", "prompt_id", "condition"]).rules.apply(
        lambda s: len({frozenset(x) for x in s}) == 1)
    code_same = df.groupby(["model", "prompt_id", "condition"]).code.apply(lambda s: s.nunique() == 1)
    res["identical_rule_set_across_seeds"] = float(sets.mean())
    sm = df[df.model.isin(small)].groupby(["model", "prompt_id", "condition"])
    res["small_identical_rule_set_across_seeds"] = float(
        sm.rules.apply(lambda s: len({frozenset(x) for x in s}) == 1).mean())
    res["small_code_changed_across_seeds"] = float(1 - sm.code.apply(lambda s: s.nunique() == 1).mean())
    print(f"  small models: rule set identical {res['small_identical_rule_set_across_seeds']:.3f}, "
          f"code changed {res['small_code_changed_across_seeds']:.3f}")
    res["identical_code_across_seeds"] = float(code_same.mean())
    print(f"  cells with identical rule set across seeds {sets.mean():.2f}; identical code {code_same.mean():.3f}")
    big = df[(df.model == BIG) & df.has_code]
    res["27B_T_events"] = big.groupby("condition")["T"].sum().astype(int).to_dict()
    res["27B_compiled"] = df[df.model == BIG].groupby("condition").compiled.sum().astype(int).to_dict()
    print(f"  27B named-rule events by condition: {res['27B_T_events']}; compiled {res['27B_compiled']}")
    sc = df[(df.condition == "scaffold") & df.has_code]
    for lab, sub in [("small", sc[sc.model.isin(small)]), ("27B", sc[sc.model == BIG])]:
        clean = sub[sub.clean]
        res[f"decomp_{lab}"] = dict(clean_share=float(sub.clean.mean()),
                                    compile_given_clean=float(clean.compiled.mean()),
                                    compiled=float(sub.compiled.mean()))
        fails = clean[~clean.compiled]
        share = {c: float(fails.diag_classes.apply(lambda d, c=c: c in (d or {})).mean())
                 for c in ("api", "arkui_dsl", "syntax", "resource", "module",
                           "dialect_unlabelled")}
        names, props = defaultdict(int), defaultdict(int)
        has_name = has_prop = has_component_prop = has_either = 0
        for o in fails.output:
            n = re.findall(r"Cannot find name '([^']+)'", o)
            pr = re.findall(r"Property '([^']+)' does not exist on type '([^']+)'", o)
            has_name += bool(n)
            has_prop += bool(pr)
            has_component_prop += any(b.endswith("Attribute") for _, b in pr)
            has_either += bool(n) or bool(pr)
            for x in n:
                names[x] += 1
            for a, b in pr:
                props[f"{a} on {b}"] += 1
        top = lambda d: sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:10]   # noqa: E731
        res[f"decomp_{lab}"].update(
            n_clean_failures=len(fails), failure_class_share=share,
            share_unknown_name=has_name / max(len(fails), 1),
            share_unknown_property=has_prop / max(len(fails), 1),
            share_unknown_component_property=has_component_prop / max(len(fails), 1),
            share_unknown_name_or_property=has_either / max(len(fails), 1),
            top_unknown_names=top(names), top_unknown_properties=top(props))
        print(f"  {lab}: unknown name in {has_name}/{len(fails)}, unknown property in "
              f"{has_prop}/{len(fails)}; names {top(names)[:6]}; properties {top(props)[:4]}")
        print(f"  scaffold {lab}: rule-clean {sub.clean.mean():.3f}, compile|clean "
              f"{clean.compiled.mean():.3f}, compiled {sub.compiled.mean():.3f}; "
              f"{len(fails)} rule-clean failures, share with class {share}")
    put("robustness", res)


WARNING_ITEMS = {"globalthis_use", "class_as_obj"}   # compile, but the compiler warns


def dprime_table(subdir, exclude=()):
    """Answer-level signal detection: every parsed answer is a trial. Log-linear correction
    (+0.5/+1). Bootstrap CI resamples items within the forbidden and allowed sets."""
    from scipy.stats import norm
    rows = {}
    rng = np.random.default_rng(0)
    for f in sorted(glob.glob(os.path.join(HERE, subdir, "*.jsonl"))):
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        recs = [r for r in recs if r["says_forbidden"] is not None and r["case"] not in exclude]
        model = recs[0]["model"]
        items = defaultdict(list)
        for r in recs:
            items[(r["case"], r["truth_forbidden"])].append(r["says_forbidden"])
        forb = [v for (c, t), v in items.items() if t]
        allo = [v for (c, t), v in items.items() if not t]

        def dp(fs, als):
            h = (sum(map(sum, fs)) + 0.5) / (sum(map(len, fs)) + 1)
            fa = (sum(map(sum, als)) + 0.5) / (sum(map(len, als)) + 1)
            return norm.ppf(h) - norm.ppf(fa), -(norm.ppf(h) + norm.ppf(fa)) / 2, h, fa
        d, c, h, fa = dp(forb, allo)
        boots = []
        for _ in range(2000):
            fs = [forb[i] for i in rng.integers(0, len(forb), len(forb))]
            als = [allo[i] for i in rng.integers(0, len(allo), len(allo))]
            boots.append(dp(fs, als)[0])
        rows[model] = dict(dprime=d, criterion=c, hit=h, fa=fa,
                           lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                           boots=boots)
    return rows


def dprime_paired_diff(file_a, file_b, n_boot=2000):
    """d'(a) - d'(b) with a paired bootstrap: both models answered the same items, so items
    are resampled once, within the forbidden and allowed sets, and both d' are recomputed on
    that resample. Items one model left unparsed contribute no answers for that model."""
    from scipy.stats import norm

    def items(f):
        out = defaultdict(list)
        for line in open(os.path.join(HERE, f), encoding="utf-8"):
            r = json.loads(line)
            if r["says_forbidden"] is not None:
                out[(r["case"], r["truth_forbidden"])].append(r["says_forbidden"])
        return out

    ia, ib = items(file_a), items(file_b)
    keys = sorted(set(ia) | set(ib))
    forb, allo = [k for k in keys if k[1]], [k for k in keys if not k[1]]

    def dp(it, fk, ak):
        hs = [v for k in fk for v in it.get(k, [])]
        fs = [v for k in ak for v in it.get(k, [])]
        h, fa = (sum(hs) + 0.5) / (len(hs) + 1), (sum(fs) + 0.5) / (len(fs) + 1)
        return norm.ppf(h) - norm.ppf(fa)

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        fk = [forb[i] for i in rng.integers(0, len(forb), len(forb))]
        ak = [allo[i] for i in rng.integers(0, len(allo), len(allo))]
        boots.append(dp(ia, fk, ak) - dp(ib, fk, ak))
    return dict(diff=dp(ia, forb, allo) - dp(ib, forb, allo),
                lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)))


def revision_analyses(df):
    print("\n" + "=" * 88 + "\n6. ANALYSES ADDED IN REVISION (exploratory unless stated)\n" + "=" * 88)
    small = [m for m in SMALL if m in set(df.model)]
    coded = df[df.has_code & df.model.isin(small)].copy()
    res = {}

    # (a) signal detection on the probe
    off, on = dprime_table("knowledge_v2"), dprime_table("knowledge_v2_think")
    off_nw = dprime_table("knowledge_v2", WARNING_ITEMS)
    for lab, tab in (("off", off), ("on", on), ("off_without_warning_items", off_nw)):
        res[f"dprime_{lab}"] = {m: {k: v for k, v in r.items() if k != "boots"} for m, r in tab.items()}
    for m in sorted(off):
        r = off[m]
        extra = ""
        if m in on:
            extra = f" | reasoning on d' {on[m]['dprime']:.2f} c {on[m]['criterion']:.2f}"
        print(f"  {m:<20} d' {r['dprime']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]  c {r['criterion']:.2f}"
              f"  hit {r['hit']:.2f} fa {r['fa']:.2f}{extra}")
    diff = np.array(on[BIG]["boots"]) - np.array(off["qwen3:8b"]["boots"])
    res["dprime_27B_on_minus_qwen3_8b_off"] = dict(
        diff=on[BIG]["dprime"] - off["qwen3:8b"]["dprime"],
        lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)))
    print(f"  27B (reasoning on) minus qwen3:8b (off): {res['dprime_27B_on_minus_qwen3_8b_off']}")
    res["dprime_27B_on_minus_qwen3_8b_off_paired"] = dprime_paired_diff(
        "knowledge_v2_think/qwen3.8-27b.jsonl", "knowledge_v2/qwen3_8b.jsonl")
    print(f"  same, paired item bootstrap: {res['dprime_27B_on_minus_qwen3_8b_off_paired']}")
    above = [m for m in small if off[m]["lo"] > 0]
    res["dprime_small_above_chance"] = above
    print(f"  small models with d' CI above 0: {len(above)}/{len(small)} {above}")

    # (b) the unnamed restriction the outcome does not count
    coded["unnamed"] = coded.output.str.contains("must specify a type").astype(int)
    res["unnamed_share"] = coded.groupby("condition").unnamed.mean().round(4).to_dict()
    res["unnamed_pages"] = int(coded.unnamed.sum())
    res["unnamed_compiled"] = int(coded[coded.unnamed == 1].compiled.sum())
    for a, b in (("generic", "scaffold"), ("placebo", "scaffold"), ("teach", "scaffold"), ("teach", "placebo")):
        r = pooled_contrast(coded, "unnamed", a, b)
        res[f"unnamed_{a}_vs_{b}"] = dict(zip(["rr", "lo", "hi", "p"], r))
        print(f"  unnamed restriction {a} vs {b}: {fmt_rr(r)}")
    print(f"  pages with the unnamed restriction: {res['unnamed_pages']} of {len(coded)}, compiled {res['unnamed_compiled']}")

    # (c) compilation with task-clustered permutation and Holm over compile contrasts
    w = df[df.model.isin(small)].pivot_table(index=["model", "prompt_id", "seed"], columns="condition",
                                             values="compiled", aggfunc="first").astype(int)
    rng = np.random.default_rng(1)
    comp, ps = {}, []
    for a, b in (("teach", "scaffold"), ("generic", "scaffold"), ("placebo", "scaffold"),
                 ("ids", "scaffold"), ("teach", "placebo")):
        d = (w[a] - w[b]).groupby(level="prompt_id").sum().values
        obs = d.sum()
        null = (rng.choice([-1, 1], size=(N_PERM, len(d))) * d).sum(axis=1)
        pp = (int((np.abs(null) >= abs(obs)).sum()) + 1) / (N_PERM + 1)
        comp[f"{a}_vs_{b}"] = dict(diff=int(obs), p_cluster=pp)
        ps.append(pp)
    holm = multipletests(ps, method="holm")[1]
    for (k, v), h in zip(comp.items(), holm):
        v["holm"] = float(h)
        print(f"  compile {k}: net {v['diff']:+d} pages, task-clustered p={v['p_cluster']:.3f}, Holm {h:.3f}")
    res["compile_clustered"] = comp
    sc = df[(df.condition == "scaffold") & df.model.isin(small) & ~df.compiled]
    only_rules = sc.diag_classes.apply(lambda d: bool(d) and set(d) <= {"rule"})
    res["skeleton_fail_rules_only"] = int(only_rules.sum())
    print(f"  skeleton pages failing on named rules alone: {res['skeleton_fail_rules_only']} of "
          f"{int((df.model.isin(small) & (df.condition == 'scaffold')).sum())}")

    # (d) target rules without arkts-as-casts (the JSX detector)
    coded["T_noas"] = coded.rules.apply(lambda r: len((r & TAUGHT) - {"arkts-as-casts"}))
    for lab, a, b in (("H1", "teach", "scaffold"), ("H2", "teach", "placebo")):
        r = pooled_contrast(coded, "T_noas", a, b)
        res[f"{lab}_without_as_casts"] = dict(zip(["rr", "lo", "hi", "p"], r))
        print(f"  {lab} without arkts-as-casts: {fmt_rr(r)}")

    # (e) per-rule breakdown (pooled over models)
    per_rule = {}
    for rule in TAUGHT_RULES:
        coded["y"] = coded.rules.apply(lambda r, rule=rule: int(rule in r))
        base = coded.groupby("condition").y.sum().to_dict()
        e = {"counts": {k: int(v) for k, v in base.items()}}
        for a, b in (("teach", "scaffold"), ("teach", "placebo")):
            r = pooled_contrast(coded, "y", a, b)
            e[f"{a}_vs_{b}"] = None if r is None else dict(zip(["rr", "lo", "hi", "p"], r))
        per_rule[rule] = e
        t1 = e["teach_vs_placebo"]
        print(f"  {rule:<38} skel {base.get('scaffold',0):>4} plac {base.get('placebo',0):>4} "
              f"teach {base.get('teach',0):>4}  teach/plac {fmt_rr(None if t1 is None else tuple(t1.values()))}")
    res["per_rule"] = per_rule

    # (f) the placebo rules themselves (planned outcome Pl)
    res["placebo_rule_mean"] = coded.groupby("condition").Pl.mean().round(4).to_dict()
    print(f"  mismatched-rule violations per page: {res['placebo_rule_mean']}")

    # (g) other rules, target vs mismatched, random effects
    r = hksj(coded, small, "U", "teach", "placebo")
    res["U_teach_vs_placebo_hksj"] = r
    print(f"  other rules, target vs mismatched, random effects RR {r['rr']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]")

    # (h) 27B probe with the two warning items set aside
    kd = []
    for f in glob.glob(os.path.join(HERE, "knowledge_v2_think", "*.jsonl")):
        kd += [json.loads(l) for l in open(f, encoding="utf-8")]
    kd = pd.DataFrame(kd)
    big = kd[(kd.model == BIG) & ~kd.case.isin(WARNING_ITEMS)]
    maj = big.groupby("case").agg(t=("truth_forbidden", "first"),
                                  a=("says_forbidden", lambda s: [x for x in s if x is not None]))
    maj["v"] = maj.a.apply(lambda v: None if not v or 2 * sum(v) == len(v) else 2 * sum(v) > len(v))
    e = maj.dropna(subset=["v"])
    spec = ((~e.t) & (e.v == False)).sum() / (~e.t).sum()   # noqa: E712
    res["27B_on_spec_without_warning_items"] = float(spec)
    res["27B_on_fp_without_warning_items"] = int(((~e.t) & (e.v == True)).sum())   # noqa: E712
    print(f"  27B reasoning on, warning items set aside: specificity {spec:.2f}, "
          f"false positives {res['27B_on_fp_without_warning_items']}")
    put("revision", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl")
    args = ap.parse_args()
    df = load(args.jsonl)
    print(f"{len(df)} records, seeds {sorted(df.seed.unique())}, models {df.model.nunique()}")
    integrity(df)
    compliance(df)
    if BIG in set(df.model):
        print("\n--- 27B (one model; a model comparison, not a scale estimate)")
        big = df[(df.model == BIG) & df.has_code]
        for a, b in [("teach", "scaffold"), ("teach", "placebo")]:
            r = poisson_contrast(big, "T", a, b)
            print(f"  {a} vs {b}: {fmt_rr(r)}")
            put(f"27B_{a}_vs_{b}", None if r is None else dict(zip(["rr", "lo", "hi", "p"], r)))
    compilation(df)
    per_case = knowledge()
    knowledge("knowledge_v2_think", "knowledge_think",
              "exploratory: reasoning on, models with a reasoning mode")
    rule_level(df, per_case)
    robustness(df)
    revision_analyses(df)
    out = os.path.join(ROOT, "paper", "numbers_v2.json")
    json.dump(OUT, open(out, "w"), indent=1, default=lambda o: None if o != o else str(o))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
