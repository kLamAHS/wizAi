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
EXPECTED = {
    "duplicate blade":
        "Both engines refuse to count one effect twice; they just do it at "
        "different stages. Deimos dedupes during damage resolution (by "
        "`spell_effect_stacking_id`), wizAi refuses the *cast* "
        "(`Sim.can_cast`, w101_sim.py:1007-1029, keyed on "
        "(name, source, sub)), so a duplicate never reaches the board. This "
        "scenario builds the board directly and so walks past wizAi's "
        "guard. The two only observably differ if something other than a "
        "player cast places a duplicate.",

    "shield vs pierce":
        "wizAi is right and Deimos is wrong here. Deimos keeps pierce as a "
        "fraction (0.15) but shield params in points (-50), then adds them: "
        "`ward_param += caster_pierce` (combat_math.py:196, and the same in "
        "effect_simulation.py:271). 0.15 against -50 is nearly a no-op, so "
        "pierce in Deimos barely dents a shield. wizAi converts first "
        "(w101_sim.py:1099-1103) and shaves the shield properly.",

    "pierce exceeds shield":
        "Same Deimos unit bug as 'shield vs pierce'.",

    "full stack":
        "Same Deimos unit bug again, now inside a busy board. The matched "
        "'full stack (no pierce)' row is this exact scenario with pierce "
        "held out; it agrees to the cent, which localises the whole gap to "
        "pierce-vs-shield.",
}

TOL = 0.5   # absolute damage; both engines work in floats


def deimos_ruleset():
    """wizAi's rules with Deimos's flat-stat placement adopted."""
    from w101_sim import Rules
    return Rules(ruleset_id="w101-pve-classic-0.3+deimos-flat",
                 flat_damage_before_multipliers=True,
                 flat_resist_before_resist=True)


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
    ap.add_argument("--adopt-deimos", action="store_true",
                    help="re-run with Deimos's flat-stat placement enabled "
                         "in wizAi (Rules.flat_damage_before_multipliers / "
                         "flat_resist_before_resist)")
    ap.add_argument("--both", action="store_true",
                    help="run the suite under both rulesets")
    a = ap.parse_args()

    runs = []
    if a.both:
        runs = [("v0.3 defaults", None),
                ("Deimos flat placement", deimos_ruleset())]
    else:
        runs = [("Deimos flat placement" if a.adopt_deimos else "v0.3 defaults",
                 deimos_ruleset() if a.adopt_deimos else None)]

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
