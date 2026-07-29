"""How much do the two flat-stat findings actually cost?

`differential.py` establishes *that* wizAi puts flat damage and flat
resist in the wrong place. This asks the question that decides whether
anyone should care: by how much, and when.

The answer has a sharp edge to it. wizAi's gear layer supplies no flat
stats at all -- `gear.loadout()` returns no `flat_damage` or
`flat_resist` key at any level, and `Actor` defaults both to 0.0. So on
every table the project has published, the two placements are
*indistinguishable*: nothing multiplies, nothing subtracts, both branches
compute the same number. The findings are real and latent.

They stop being latent the moment a real wizard's stats enter, which is
exactly what the live bridge does -- wizwalker reads `dmg_bonus_flat` and
`dmg_reduce_flat` straight off the participant, and in the live game
those are not zero. This module sweeps realistic values so the size of
the error is a number rather than an argument.

    python -m deimos_bridge.flat_stat_probe
    python -m deimos_bridge.flat_stat_probe --json results_flat_stat.json
"""
import argparse
import json

from w101_sim import Rules

from .scenarios import Buff, Scenario, Side

CLASSIC = Rules()
DEIMOS = Rules(ruleset_id="w101-pve-classic-0.3+deimos-flat",
               flat_damage_before_multipliers=True,
               flat_resist_before_resist=True)


def sweep_flat_damage(base=500, blades=(0, 1, 2, 3, 4), flats=(0, 25, 50, 100, 200)):
    """Flat damage under a growing blade stack.

    The error scales with the stack, which is the part worth seeing: the
    blade-stack meta is the whole subject of this project, and the
    placement bug is worst exactly where the project spends its time.
    """
    rows = []
    for n_blades in blades:
        stack = [Buff(f"Blade{i}", 35, "fire", stack=100 + i)
                 for i in range(n_blades)]
        for flat in flats:
            sc = Scenario(f"{n_blades} blades, +{flat} flat", base, "fire",
                          caster=Side(flat_damage=flat, is_player=True),
                          blades=stack)
            a = sc.wizai_damage(CLASSIC)
            b = sc.wizai_damage(DEIMOS)
            rows.append({
                "blades": n_blades, "flat_damage": flat,
                "current": round(a, 1), "deimos": round(b, 1),
                "delta": round(b - a, 1),
                "pct": round((b - a) / a * 100, 1) if a else 0.0,
            })
    return rows


def sweep_flat_resist(base=500, resists=(0.0, 0.2, 0.4, 0.6),
                      flats=(0, 25, 50, 100)):
    rows = []
    for resist in resists:
        for flat in flats:
            sc = Scenario(f"{int(resist*100)}% resist, {flat} flat", base, "fire",
                          target=Side(school="ice", resist_pct=resist,
                                      flat_resist=flat))
            a = sc.wizai_damage(CLASSIC)
            b = sc.wizai_damage(DEIMOS)
            rows.append({
                "resist": resist, "flat_resist": flat,
                "current": round(a, 1), "deimos": round(b, 1),
                "delta": round(b - a, 1),
                "pct": round((b - a) / a * 100, 1) if a else 0.0,
            })
    return rows


def _grid(rows, row_key, col_key, label):
    rowvals = sorted({r[row_key] for r in rows})
    colvals = sorted({r[col_key] for r in rows})
    by = {(r[row_key], r[col_key]): r for r in rows}
    out = [f"{label} (% change when Deimos's placement is adopted)", ""]
    head = f"{row_key:>12} |" + "".join(f"{c:>10}" for c in colvals)
    out.append(head)
    out.append("-" * len(head))
    for rv in rowvals:
        cells = []
        for cv in colvals:
            r = by.get((rv, cv))
            cells.append(f"{r['pct']:>9.1f}%" if r else " " * 10)
        out.append(f"{rv:>12} |" + "".join(cells))
    out.append(f"{'':>12}   (columns: {col_key})")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json")
    args = ap.parse_args()

    dmg = sweep_flat_damage()
    res = sweep_flat_resist()

    print(_grid(dmg, "blades", "flat_damage", "FLAT DAMAGE placement"))
    print()
    print(_grid(res, "resist", "flat_resist", "FLAT RESIST placement"))
    print()
    zero = [r for r in dmg if r["flat_damage"] == 0]
    print(f"With flat damage at 0 -- which is every stat the gear layer "
          f"currently produces -- the two placements differ by "
          f"{max(abs(r['delta']) for r in zero)} damage across "
          f"{len(zero)} blade counts. The finding is real and dormant; it "
          f"wakes up the moment real gear numbers arrive, which is what "
          f"the live bridge feeds in.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"flat_damage": dmg, "flat_resist": res}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
