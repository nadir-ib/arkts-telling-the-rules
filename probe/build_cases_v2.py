"""Probe items, second protocol: the first protocol's 50 candidates plus items that
make every rule named in the intervention measurable on its own.

Why the extension: the rule-level question -- does being told a rule help more
where the model can already state it? -- needs a knowledge score per taught rule.
The first item set had no item for arkts-as-casts or
arkts-no-noninferrable-arr-literals, and its only return-type item compiles
(inference succeeds), so it measured a construct the rule does not target.

The compiler still decides every label. `target` names the rule an item is meant to
measure; an item counts for that rule only if the compiler reports it.

    python3 probe/build_cases_v2.py [--project harness/ArkProject_c]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts_shared import CANDIDATES, PAGE  # noqa: E402
from harness import DEFAULT_PROJECT, IndexGuard, compile_code  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# the first protocol's items, with the rule each forbidden one is meant to measure
TARGETS_V1 = {
    "untyped_obj_literal": "arkts-no-untyped-obj-literals",
    "obj_literal_as_type": "arkts-no-obj-literals-as-types",
    "any_type": "arkts-no-any-unknown", "unknown_type": "arkts-no-any-unknown",
    "typed_catch": "arkts-no-types-in-catch",
    "destructuring_decl": "arkts-no-destruct-decls",
    "obj_destructuring": "arkts-no-destruct-decls",
    "ctor_prop_decl": "arkts-no-ctor-prop-decls", "object_spread": "arkts-no-spread",
    "func_bind": "arkts-no-func-bind", "func_apply": "arkts-no-func-apply-call",
    "in_operator": "arkts-no-in", "with_statement": "arkts-no-with",
    "standalone_this": "arkts-no-standalone-this",
    "comma_outside_loop": "arkts-no-comma-outside-loops",
    "props_by_index": "arkts-no-props-by-index",
    "namespace_as_obj": "arkts-no-ns-as-obj",
    "inferred_generic": "arkts-no-inferred-generic-params",
    "polymorphic_unop": "arkts-no-polymorphic-unops",
    "delete_operator": "arkts-no-delete", "typeof_as_type": "arkts-no-type-query",
    "index_signature": "arkts-no-indexed-signatures",
    "function_expression": "arkts-no-func-expressions",
    "enum_string_mixed": "arkts-no-enum-mixed-types",
    "decl_merging": "arkts-no-decl-merging", "var_declaration": "arkts-no-var",
    "structural_typing": "arkts-no-structural-typing",
}

# (name, snippet, prior, target rule or None)
NEW = [
    ("angle_bracket_cast",   "const v1: Object = 1; const w1 = <number>v1;", "forbidden", "arkts-as-casts"),
    ("angle_bracket_cast2",  "class Q1 { a: number = 1 } const o1: Object = new Q1(); const q1 = <Q1>o1;", "forbidden", "arkts-as-casts"),
    ("jsx_markup",           "const view = <Column><Text>hi</Text></Column>;", "forbidden", None),
    ("recursive_no_return",  "function fact(n: number) { return n <= 1 ? 1 : n * fact(n - 1); }", "forbidden", "arkts-no-implicit-return-types"),
    ("recursive_method_no_return", "class T2 { f(n: number) { return n > 0 ? this.f(n - 1) : 0; } }", "forbidden", "arkts-no-implicit-return-types"),
    ("array_of_untyped_objects", "interface I1 { a: number } const arr2 = [{ a: 1 }, { a: 2 }];", "forbidden", "arkts-no-noninferrable-arr-literals"),
    ("array_of_untyped_objects2", "const rows = [{ id: 1, label: 'a' }];", "forbidden", "arkts-no-noninferrable-arr-literals"),
    ("var_in_for_loop",      "for (var i = 0; i < 3; i++) { }", "forbidden", "arkts-no-var"),
    ("in_operator_class",    "class H { a: number = 1 } const h = new H(); const hasA = 'a' in h;", "forbidden", "arkts-no-in"),
    ("comma_statement",      "let p3: number = 0; let q3: number = 0; p3 = 1, q3 = 2;", "forbidden", "arkts-no-comma-outside-loops"),
    ("untyped_obj_in_function", "function mk(): void { const cfg = { debug: true }; }", "forbidden", "arkts-no-untyped-obj-literals"),
    ("obj_type_parameter",   "function area(r: { w: number, h: number }): number { return r.w * r.h; }", "forbidden", "arkts-no-obj-literals-as-types"),
    ("any_parameter",        "function logv(v: any): void { }", "forbidden", "arkts-no-any-unknown"),
    ("this_in_function",     "function helper(): number { return this.count; }", "forbidden", "arkts-no-standalone-this"),
    # allowed counterparts for the same rules, so a model that says "forbidden" to
    # everything near a rule does not score as knowing it
    ("as_cast_supported",    "class A1 { a: number = 1 } class B1 extends A1 { } const a1: A1 = new B1(); const b1 = a1 as B1;", "allowed", None),
    ("simple_no_return",     "function two() { return 2; }", "allowed", None),
    ("empty_array_const",    "const empty = [];", "allowed", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    args = ap.parse_args()
    project = os.path.abspath(args.project)

    items = [(n, s, e, TARGETS_V1.get(n)) for n, s, e in CANDIDATES] + NEW
    results = []
    with IndexGuard(project):
        for name, snippet, prior, target in items:
            r = compile_code(PAGE % snippet, project)
            verdict = "allowed" if r["compiled"] else "forbidden"
            usable = r["compiled"] or bool(r["rules"])
            on_target = target in r["rules"] if target else None
            results.append(dict(name=name, snippet=snippet, expected=prior,
                                compiler_verdict=verdict, rules=r["rules"],
                                usable=usable, target=target,
                                target_confirmed=on_target, protocol="v2" if
                                (name, snippet, prior, target) in NEW else "v1"))
            flag = "" if verdict == prior else "  <-- DISAGREES WITH PRIOR"
            miss = "  <-- TARGET RULE NOT REPORTED" if target and not on_target \
                and verdict == "forbidden" else ""
            print(f"  {name:28} {verdict:9} {','.join(r['rules']) or '-':<60}"
                  f"{'' if usable else ' UNUSABLE'}{flag}{miss}", flush=True)

    out = os.path.join(HERE, "cases_v2.json")
    json.dump(results, open(out, "w", encoding="utf-8"), indent=2)
    usable = [r for r in results if r["usable"]]
    print(f"\n{len(usable)}/{len(results)} usable "
          f"({sum(r['compiler_verdict'] == 'forbidden' for r in usable)} forbidden, "
          f"{sum(r['compiler_verdict'] == 'allowed' for r in usable)} allowed); wrote {out}")


if __name__ == "__main__":
    main()
