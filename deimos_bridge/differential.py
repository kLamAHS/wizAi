"""Run wizAi and Deimos over the same fights and diff the numbers.

Deimos's combat math was written against the live client -- its author
could watch a real Fireblade land on a real mob and check the number. So
where the two engines disagree, Deimos is the better witness to what the
game does, and the disagreement is a wizAi bug report.

That is not the same as Deimos being right by definition. Some of what it
does is visibly a display heuristic (it is drawing a number in a HUD, not
resolving a duel), and this module labels those separately instead of
counting them against wizAi. See `VERDICTS`.

    python -m deimos_bridge.differential            # table
    python -m deimos_bridge.differential --json out.json
"""
import argparse
import json

from .scenarios import suite

#: Scenarios where a disagreement is *expected* and does not indicate a
#: wizAi bug, with the reason. Anything not listed here is a real finding.
EXPECTED = {}

#: Divergences that used to be listed above and are now fixed on one side
#: or the other. Kept as a record of what the harness was for, and as a
#: reminder that the pierce rows are load-bearing tests rather than
#: decoration -- if a future edit reintroduces the unit confusion, those
#: three rows are what catches it.
RESOLVED = {
    "shield vs pierce": "Deimos's pierce/ward unit bug, fixed in "
                        "combat_math.py and effect_simulation.py",
    "pierce exceeds shield": "same fix",
    "full stack": "same fix",
    "flat damage": "wizAi adopted Deimos's flat-damage placement",
    "flat resist": "wizAi adopted Deimos's flat-resist placement",
    "full stack (no pierce)": "both of the above",
    "duplicate blade":
        "wizAi adopted Deimos's rule. This sat in EXPECTED for a while on "
        "the argument that both engines refuse to count one effect twice "
        "and merely do it at different stages -- Deimos during damage "
        "resolution, wizAi by refusing the cast -- so the two could only "
        "differ if something other than a player cast placed a duplicate. "
        "Both halves of that were wrong. There is no such cast "
        "restriction in the game: three Ice Traps go on one mob, and each "
        "hit consumes one. And wizAi's guard was inert in the only place "
        "it mattered, because live-read hangings are named "
        "`live:<template id>` and never match a card in hand's stack key. "
        "So in a real fight wizAi laid duplicate after duplicate and then "
        "multiplied all of them into a single strike -- 2.744x for three "
        "traps against the true 1.4x -- which is what made stacking look "
        "worth spending rounds on.",
    "duplicate trap": "same fix, on the ward side",
}

TOL = 0.5   # absolute damage; both engines work in floats


def legacy_ruleset():
    """The pre-0.4 flat-stat placement, for reproducing the old numbers."""
    from w101_sim import Rules
    return Rules(ruleset_id="w101-pve-classic-0.3-legacy-flat",
                 flat_damage_before_multipliers=False,
                 flat_resist_before_resist=False)


#: Kept under the old name so anything that imported it still works; the
#: flat placement it used to switch on is now the default.
def deimos_ruleset():
    from w101_sim import Rules
    return Rules()


def compare(scenarios=None, rules=None):
    rows = []
    scenarios = scenarios if scenarios is not None else suite()
    for sc in scenarios:
        sc.rules_override = rules
        w = sc.wizai_damage()
        d = sc.deimos_damage()
        delta = w - d
        pct = (delta / d * 100.0) if abs(d) > 1e-9 else (0.0 if abs(w) < 1e-9 else float("inf"))
        rows.append({
            "scenario": sc.name,
            "note": sc.note,
            "wizai": round(w, 2),
            "deimos": round(d, 2),
            "delta": round(delta, 2),
            "pct": round(pct, 2),
            "agree": abs(delta) <= TOL,
            "expected_divergence": EXPECTED.get(sc.name),
        })
    return rows


def render(rows) -> str:
    w = max(len(r["scenario"]) for r in rows)
    out = [f"{'scenario':<{w}}  {'wizAi':>10}  {'Deimos':>10}  {'delta':>10}  {'%':>8}  ok",
           "-" * (w + 48)]
    for r in rows:
        mark = "ok" if r["agree"] else ("~" if r["expected_divergence"] else "DIFF")
        pct = "inf" if r["pct"] == float("inf") else f"{r['pct']:.1f}"
        out.append(f"{r['scenario']:<{w}}  {r['wizai']:>10.2f}  {r['deimos']:>10.2f}"
                   f"  {r['delta']:>10.2f}  {pct:>8}  {mark}")
    agree = sum(1 for r in rows if r["agree"])
    out.append("-" * (w + 48))
    out.append(f"{agree}/{len(rows)} agree within {TOL}")
    diffs = [r for r in rows if not r["agree"] and not r["expected_divergence"]]
    if diffs:
        out.append("")
        out.append("divergences:")
        for r in diffs:
            out.append(f"  * {r['scenario']}: wizAi {r['wizai']:.1f} vs "
                       f"Deimos {r['deimos']:.1f} ({r['pct']:+.1f}%)")
            if r["note"]:
                out.append(f"      {r['note']}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="also write the rows here")
    ap.add_argument("--legacy", action="store_true",
                    help="re-run with the pre-0.4 flat-stat placement, to "
                         "see the divergence the current default fixes")
    ap.add_argument("--both", action="store_true",
                    help="run the suite under current and legacy rules")
    a = ap.parse_args()

    if a.both:
        runs = [("current rules", None), ("legacy flat placement", legacy_ruleset())]
    else:
        runs = [("legacy flat placement", legacy_ruleset())] if a.legacy \
            else [("current rules", None)]

    payload = {}
    for label, rules in runs:
        rows = compare(rules=rules)
        print(f"\n### {label}\n")
        print(render(rows))
        payload[label] = rows

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"tolerance": TOL, "runs": payload}, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
