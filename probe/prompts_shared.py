"""Prompt texts and probe candidates shared by both protocols.

Moved out of compliance_run.py and build_cases.py unchanged, so that the second-protocol
scripts (and the released artifact) do not depend on the first-protocol scripts.
"""

FREE = (
    "Write a single-file ArkTS page for HarmonyOS (ArkUI) that satisfies the request "
    "below. Output ONLY the code, no explanation and no markdown prose.\n\n"
    "Request: {prompt}"
)

# Providing the page skeleton fixes the ArkUI-boilerplate variable so that whatever
# the compiler then reports is about ArkTS constraints, not about whether the model
# knows that a page is an @Entry struct with a build() method.
SCAFFOLD = (
    "Complete the ArkTS page below so that it satisfies the request. Keep the "
    "@Entry/@Component/struct/build() skeleton exactly as given: add state fields, "
    "helper methods and UI inside it. Output ONLY the complete file, no explanation "
    "and no markdown prose.\n\n"
    "Request: {prompt}\n\n"
    "Skeleton to complete:\n"
    "@Entry\n"
    "@Component\n"
    "struct Index {\n"
    "  // add @State fields here\n"
    "\n"
    "  build() {\n"
    "    Column() {\n"
    "      // add UI here\n"
    "    }\n"
    "    .width('100%')\n"
    "    .height('100%')\n"
    "  }\n"
    "}\n"
)

# Third arm: the scaffold plus an explicit statement of the constraints the models
# actually violate most (measured in scaffold mode). If declarative knowledge in
# context were the missing ingredient, violation of these named rules should drop.
# The listed rules are exactly the ten most-violated ones, so the arm is a direct
# A/B against scaffold on the same prompts and models.
CONSTRAINTS = """ArkTS is a restricted dialect of TypeScript. The following are
COMPILER ERRORS in ArkTS. You must not use them:
1. 'any' or 'unknown' types (arkts-no-any-unknown) - always give a concrete type.
2. Object literals without a declared class or interface type
   (arkts-no-untyped-obj-literals) - declare an interface first, then annotate.
3. Functions without an explicit return type (arkts-no-implicit-return-types)
   - write ': void', ': number', etc. on every function and method.
4. Array literals whose element type cannot be inferred
   (arkts-no-noninferrable-arr-literals) - annotate as 'const xs: number[] = []'.
5. Casts between unrelated types (arkts-as-casts) - avoid 'as' conversions.
6. The comma operator outside a for loop (arkts-no-comma-outside-loops).
7. Structural type literals used as types (arkts-no-obj-literals-as-types)
   - use a named interface instead of '{ a: number }' in a type position.
8. The 'in' operator (arkts-no-in).
9. 'var' declarations (arkts-no-var) - use 'let' or 'const'.
10. 'this' outside a class or struct method (arkts-no-standalone-this).
"""

TEACH = (
    "Write ArkTS for HarmonyOS (ArkUI).\n\n"
    + CONSTRAINTS +
    "\nComplete the ArkTS page below so that it satisfies the request, obeying every "
    "constraint above. Keep the @Entry/@Component/struct/build() skeleton exactly as "
    "given: add state fields, helper methods and UI inside it. Output ONLY the "
    "complete file, no explanation and no markdown prose.\n\n"
    "Request: {prompt}\n\n"
    "Skeleton to complete:\n"
    "@Entry\n"
    "@Component\n"
    "struct Index {\n"
    "  // add @State fields here\n"
    "\n"
    "  build() {\n"
    "    Column() {\n"
    "      // add UI here\n"
    "    }\n"
    "    .width('100%')\n"
    "    .height('100%')\n"
    "  }\n"
    "}\n"
)

# the ten rules named in CONSTRAINTS, for the primary (named-rule) analysis
TAUGHT_RULES = [
    "arkts-no-any-unknown", "arkts-no-untyped-obj-literals",
    "arkts-no-implicit-return-types", "arkts-no-noninferrable-arr-literals",
    "arkts-as-casts", "arkts-no-comma-outside-loops",
    "arkts-no-obj-literals-as-types", "arkts-no-in", "arkts-no-var",
    "arkts-no-standalone-this",
]


# Candidate constructs. `expect` is our prior, recorded only so the report can flag
# where the compiler disagreed with us — the compiler's verdict is what is used.
CANDIDATES = [
    # --- expected FORBIDDEN: targets the rules the corpus actually shows ---
    ("untyped_obj_literal",   "const p = { name: 'a', age: 3 };",                 "forbidden"),
    ("obj_literal_as_type",   "let cfg: { w: number } = { w: 1 };",               "forbidden"),
    ("any_type",              "let x: any = 5;",                                  "forbidden"),
    ("unknown_type",          "let u: unknown = 5;",                              "forbidden"),
    ("typed_catch",           "try { this.work(); } catch (e: Error) { }",        "forbidden"),
    ("destructuring_decl",    "const [a, b] = [1, 2];",                           "forbidden"),
    ("obj_destructuring",     "const o: Point = new Point(); const { x } = o;",   "forbidden"),
    ("empty_array_inferred",  "let bag = []; bag.push(1);",                       "forbidden"),
    ("mixed_array_literal",   "const mixed = [1, 'two', true];",                  "forbidden"),
    ("require_call",          "const m = require('mod');",                        "forbidden"),
    ("ctor_prop_decl",        "class Holder { constructor(private v: number) {} }", "forbidden"),
    ("object_spread",         "const a: Point = new Point(); const b = { ...a };", "forbidden"),
    ("func_bind",             "class B { go(): void {} } const b = new B(); const f = b.go.bind(b);", "forbidden"),
    ("func_apply",            "class C { go(): void {} } const c = new C(); c.go.apply(c, []);", "forbidden"),
    ("in_operator",           "const o: Record<string, number> = {}; const has = 'k' in o;", "forbidden"),
    ("with_statement",        "with (Math) { const r = PI; }",                    "forbidden"),
    ("standalone_this",       "function loose(): void { const t = this; }",       "forbidden"),
    ("globalthis_use",        "const g = globalThis;",                            "forbidden"),
    ("comma_outside_loop",    "let z: number = 0; z = (1, 2);",                   "forbidden"),
    ("props_by_index",        "class K { a: number = 1; } const k = new K(); const v = k['a'];", "forbidden"),
    ("class_as_obj",          "class D { static s: number = 1; } const ref = D;", "forbidden"),
    ("namespace_as_obj",      "namespace NS { export const v: number = 1; } const n = NS;", "forbidden"),
    ("implicit_return_type",  "function pick(flag: boolean) { return flag ? 1 : 'x'; }", "forbidden"),
    ("inferred_generic",      "class Box<T> { v?: T } const bx = new Box();",     "forbidden"),
    ("as_cast_unrelated",     "const s: string = 'a'; const n2 = s as unknown as number;", "forbidden"),
    ("polymorphic_unop",      "const s3: string = '5'; const neg = -s3;",         "forbidden"),
    ("delete_operator",       "class E { a?: number } const e = new E(); delete e.a;", "forbidden"),
    ("typeof_as_type",        "const base: number = 1; let same: typeof base = 2;", "forbidden"),
    ("index_signature",       "interface Bag { [key: string]: number }",          "forbidden"),
    ("function_expression",   "const fe = function (x: number): number { return x; };", "forbidden"),
    ("generic_arrow_untyped", "const id = (v) => v;",                             "forbidden"),
    ("enum_string_mixed",     "enum M { A = 1, B = 'b' }",                        "forbidden"),
    ("decl_merging",          "interface Dup { a: number } interface Dup { b: number }", "forbidden"),
    ("var_declaration",       "var legacy: number = 1;",                          "forbidden"),
    ("structural_typing",     "class P1 { x: number = 0 } class P2 { x: number = 0 } const conv: P1 = new P2();", "forbidden"),

    # --- expected ALLOWED: valid ArkTS, to measure false alarms ---
    ("arrow_function",        "const f: (x: number) => number = (x: number): number => x * 2;", "allowed"),
    ("async_await",           "async function g(): Promise<void> { await Promise.resolve(); }", "allowed"),
    ("template_literal",      "const n: string = 'x'; const s: string = `hi ${n}`;", "allowed"),
    ("typed_obj_literal",     "interface P { name: string } const p: P = { name: 'a' };", "allowed"),
    ("for_of_loop",           "const xs: number[] = [1, 2]; let t: number = 0; for (const x of xs) { t += x; }", "allowed"),
    ("explicit_class",        "class Point { x: number = 0; y: number = 0; }",     "allowed"),
    ("typed_array_literal",   "const nums: number[] = [1, 2, 3];",                 "allowed"),
    # 'Shape' and 'Sq' collide with ArkUI built-ins, which makes the compiler report
    # arkts-unique-names for a construct that is otherwise fine -- renamed so the case
    # tests interface implementation rather than a name clash.
    ("interface_impl",        "interface ZzShape { zarea(): number } class ZzSq implements ZzShape { zarea(): number { return 1; } }", "allowed"),
    ("generic_explicit",      "class Box2<T> { v?: T } const b2 = new Box2<number>();", "allowed"),
    ("optional_chaining",     "class N { next?: N } const nn = new N(); const q = nn.next?.next;", "allowed"),
    ("string_methods",        "const s2: string = 'abc'; const up: string = s2.toUpperCase();", "allowed"),
    ("ternary_typed",         "const flag: boolean = true; const r: number = flag ? 1 : 2;", "allowed"),
    ("map_usage",             "const m2: Map<string, number> = new Map<string, number>(); m2.set('a', 1);", "allowed"),
    ("union_type",            "let u2: number | string = 1; u2 = 'a';",            "allowed"),
    ("readonly_field",        "class R { readonly id: number = 1; }",              "allowed"),
]


PAGE = """%s

@Entry
@Component
struct Index {
  @State message: string = 'probe';

  work(): void { }

  build() {
    Column() {
      Text(this.message).fontSize(16)
    }
    .width('100%%')
    .height('100%%')
  }
}
"""
