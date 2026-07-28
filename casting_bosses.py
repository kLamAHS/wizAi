"""
Do casting bosses remove the cliff?

Real-boss fights in this repo have been cliff-like: a scan of boss HP
bands wins ~100% below 9k and ~0% above 14k, with almost nothing
contested between. That is a signature of the boss model, not of the
game — a flat `40 + 20*rank` hit every round makes an encounter a
deterministic arithmetic race, so the outcome is decided before the
first card is drawn and variance has nowhere to enter.

The repo owner confirms bosses simply cast from their spell pool.
`load_bosses_full(spell_pools=True)` now fills `Boss.pool` for 1795
creatures and zeroes the flat hit, routing boss damage through the same
charm/ward/fizzle/pip machinery as the player's.

Prediction stated before the run: casting should (a) shift average
difficulty, because a boss that must draw pips and can fizzle is not
the same as one that hits for a constant every round, and more
importantly (b) WIDEN the contested band, because outcomes now depend
on what the boss draws.
"""
import copy
import json
import random
import statistics as st
import time

from data_full import LIVE_RULES, load_bosses_full, load_spells_full
from gear import loadout
from w101_sim import Sim, evaluate_paired, make_blade_stack, make_survival

cards = load_spells_full()
t0 = time.time()
POLS = {f"blade-stack({k})": make_blade_stack(k) for k in (1, 2, 3)}
POLS.update({f"triage({k})": make_survival(make_blade_stack(k))
             for k in (1, 2)})
DECK = ["Fireblade", "Fireblade", "Fire Trap", "Feint", "Tower Shield",
        "Tower Shield", "Fire Dragon", "Fire Dragon", "Sunbird",
        "Sunbird", "Pixie", "Fire Elf"]
LVL = 100
lo = loadout(LVL, "fire", rules=LIVE_RULES)

rep = {}
flat, _ = load_bosses_full()
casting, _ = load_bosses_full(report=rep, spell_pools=True, cards=cards)
print(f"[pools] {rep['spell_pools']['bosses_with_pool']} of "
      f"{len(casting)} creatures given a pool", flush=True)
for k, v in sorted(rep["spell_pools"]["sources"].items(),
                   key=lambda kv: -kv[1]):
    print(f"    {k:<26} {v}", flush=True)
report = {"pool_sources": rep["spell_pools"]}


def win(boss, n=300):
    sim = Sim(dict(cards), DECK, "fire", copy.copy(boss),
              player_hp=lo["player_hp"], rules=LIVE_RULES,
              player_stats=lo["player_stats"],
              power_pip=lo["power_pip"])
    r = evaluate_paired(sim, POLS, n=n)
    return max(r.values(), key=lambda v: (round(v["win_rate"], 2),
                                          -v["mean_ttk"]))["win_rate"]


# same creatures, same seeds, only the boss MODEL changes
names = [b.name for b in flat.values() if 3000 <= b.hp <= 16000 and b.rank]
names.sort()
random.Random(3).shuffle(names)
names = names[:24]
rows = []
print(f"\n[Q] {len(names)} real bosses, flat hit vs casting from a pool\n",
      flush=True)
for nm in names:
    a, b = win(flat[nm]), win(casting[nm])
    rows.append({"boss": nm, "hp": flat[nm].hp, "rank": flat[nm].rank,
                 "flat": a, "casting": b})
report["rows"] = rows


def contested(vals, lo_b=0.05, hi_b=0.95):
    return sum(1 for v in vals if lo_b < v < hi_b)


fa = [r["flat"] for r in rows]
ca = [r["casting"] for r in rows]
print(f"    {'':<26} flat      casting", flush=True)
print(f"    {'mean win rate':<26} {st.mean(fa)*100:5.1f}%    "
      f"{st.mean(ca)*100:5.1f}%", flush=True)
print(f"    {'pinned at 0% or 100%':<26} {len(fa)-contested(fa):5d}     "
      f"{len(ca)-contested(ca):5d}", flush=True)
print(f"    {'contested (5-95%)':<26} {contested(fa):5d}     "
      f"{contested(ca):5d}   of {len(rows)}", flush=True)
report["summary"] = {
    "mean_win_flat": st.mean(fa), "mean_win_casting": st.mean(ca),
    "contested_flat": contested(fa), "contested_casting": contested(ca),
    "n": len(rows)}

print(f"\n    the cliff, by HP band (contested / total):", flush=True)
bands = [(3000, 7000), (7000, 11000), (11000, 16000)]
band_rep = {}
for loh, hih in bands:
    sub = [r for r in rows if loh <= r["hp"] < hih]
    if not sub:
        continue
    f = contested([r["flat"] for r in sub])
    c = contested([r["casting"] for r in sub])
    band_rep[f"{loh}-{hih}"] = {"n": len(sub), "flat": f, "casting": c}
    print(f"      {loh:>5}-{hih:<6} n={len(sub):2d}   flat {f}/{len(sub)}"
          f"      casting {c}/{len(sub)}", flush=True)
report["by_band"] = band_rep

json.dump(report, open("results_casting_bosses.json", "w",
                       encoding="utf-8"), indent=1)
print(f"\nwrote results_casting_bosses.json ({time.time()-t0:.0f}s)",
      flush=True)
