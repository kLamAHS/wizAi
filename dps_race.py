"""
Do I need to extract every non-boss enemy? And does the DPS race work?

The repo owner asked whether it is worth pulling a JSON of ordinary
enemies and aligning them with their bosses, since real fights are a
boss plus one to three mobs. That is a question about where the
information VALUE is, so it is measurable rather than a matter of
opinion: if the outcome barely moves when a minion's health is halved
or doubled, then their health is not worth extracting; if it swings
when the minion can or cannot CAST, then their spell pools are.

He also described how he actually plays: build damage buffs, then
one-shot with an AoE. That is a policy claim, and this repo has a
policy sweep, so it is worth checking against the now-more-realistic
model rather than taken on trust or dismissed.

Q1  2x2: minion health share (0.2 / 0.5) x minions cast (off / on).
    Which term moves the result?
Q2  the race itself: how many blades before the swing, and does
    racing beat defensive play once bosses and minions both cast?
Q3  single-target vs AoE finisher, with the party present.

All under `spell_pools=True` — the whole point is that everything in
the encounter casts.
"""
import copy
import json
import random
import statistics as st
import time

from data_full import (LIVE_RULES, encounter, load_bosses_full,
                       load_spells_full)
from gear import loadout
from w101_sim import (Sim, evaluate_paired, make_blade_stack,
                      make_survival, register_enchants)

cards = dict(load_spells_full())
register_enchants(cards, ["Fireblade", "Balanceblade"], "Sharpen Blade")
register_enchants(cards, ["Fire Trap", "Feint"], "Potent Trap")
t0 = time.time()
LVL = 100
lo = loadout(LVL, "fire", rules=LIVE_RULES)
bosses, _ = load_bosses_full(spell_pools=True, cards=cards)

AOE = ["Fireblade", "Fireblade+sharp", "Balanceblade", "Fire Trap",
       "Fire Trap+potent", "Feint", "Feint+potent", "Fire Dragon",
       "Fire Dragon", "Rain of Fire", "Tower Shield", "Pixie"]
SINGLE = ["Fireblade", "Fireblade+sharp", "Balanceblade", "Fire Trap",
          "Fire Trap+potent", "Feint", "Feint+potent", "Efreet",
          "Efreet", "Sunbird", "Tower Shield", "Pixie"]
# swept past the peak on purpose: an optimum at the edge of a
# sweep is not an optimum, it is a boundary
RACE = {f"race({k})": make_blade_stack(k)
        for k in (1, 2, 3, 4, 5, 6, 7, 8)}
DEFENSIVE = {f"triage({k})": make_survival(make_blade_stack(k))
             for k in (2, 3, 4)}


def run(boss, comps, deck=AOE, pols=None, n=300):
    sim = Sim(dict(cards), deck, "fire", copy.copy(boss),
              player_hp=lo["player_hp"], rules=LIVE_RULES,
              player_stats=lo["player_stats"],
              power_pip=lo["power_pip"],
              enemies=[copy.copy(c) for c in comps])
    return evaluate_paired(sim, pols or {**RACE, **DEFENSIVE}, n=n)


def best(stats):
    return max(stats.items(), key=lambda kv: (round(kv[1]["win_rate"], 2),
                                              -kv[1]["mean_ttk"]))


def mute(comps):
    """Strip a companion down to an HP bag: no pool, no damage."""
    out = []
    for c in comps:
        c = copy.copy(c)
        c.pool, c.dmg = None, 0
        out.append(c)
    return out


# pick encounters that have SYNTHETIC companions (the ones an
# extraction effort would replace) and are contested under casting
cands = [b for b in bosses.values()
         if b.minions and b.rank and 4000 <= b.hp <= 14000
         and not any(m in bosses for m in b.minions)]
cands.sort(key=lambda b: b.name)
random.Random(5).shuffle(cands)
picked, scanned = [], 0
for b in cands:
    if len(picked) >= 6 or scanned > 40:
        break
    scanned += 1
    _b, comps = encounter(b.name, bosses)
    w = best(run(b, comps, n=200))[1]["win_rate"]
    if 0.10 < w < 0.95:
        picked.append(b)
print(f"[setup] {len(picked)} contested encounters with inferred "
      f"companions, from {scanned} scanned", flush=True)
for b in picked:
    print(f"    {b.name:<30} {b.hp:6d} HP + {len(b.minions)}", flush=True)

report = {"encounters": [b.name for b in picked]}

# ---- Q1: health or casting? ----------------------------------------
print(f"\n[Q1] which term actually moves the fight?\n", flush=True)
grid = {}
for share in (0.2, 0.5):
    for casts in (False, True):
        ws = []
        for b in picked:
            _b, comps = encounter(b.name, bosses, hp_share=share)
            ws.append(best(run(b, comps if casts else mute(comps)))[1]
                      ["win_rate"])
        grid[(share, casts)] = st.mean(ws)
        print(f"    minion HP {share:.1f} x boss HP, casting "
              f"{'ON ' if casts else 'OFF'}   mean win "
              f"{st.mean(ws)*100:5.1f}%", flush=True)
hp_effect = st.mean([grid[(0.2, c)] - grid[(0.5, c)] for c in (False, True)])
cast_effect = st.mean([grid[(s, False)] - grid[(s, True)]
                       for s in (0.2, 0.5)])
print(f"\n    halving minion HP is worth   {hp_effect*100:+5.1f} points",
      flush=True)
print(f"    silencing the minions is worth {cast_effect*100:+5.1f} points",
      flush=True)
report["q1"] = {"grid": {f"{s}/{c}": v for (s, c), v in grid.items()},
                "hp_effect_points": hp_effect * 100,
                "cast_effect_points": cast_effect * 100}

# ---- Q2: how many blades before the swing? -------------------------
print(f"\n[Q2] the race — buffs before the swing, vs defensive play\n",
      flush=True)
agg = {}
for b in picked:
    _b, comps = encounter(b.name, bosses)
    stats = run(b, comps, n=400)
    for k, v in stats.items():
        agg.setdefault(k, []).append(v["win_rate"])
means = {k: st.mean(v) for k, v in agg.items()}
for k in sorted(means, key=lambda k: -means[k]):
    print(f"    {k:<12} mean win {means[k]*100:5.1f}%", flush=True)
race_best = max((k for k in means if k.startswith("race")),
                key=lambda k: means[k])
def_best = max((k for k in means if k.startswith("triage")),
               key=lambda k: means[k])
print(f"\n    best race line {race_best} at {means[race_best]*100:.1f}% "
      f"vs best defensive {def_best} at {means[def_best]*100:.1f}%   "
      f"({(means[race_best]-means[def_best])*100:+.1f} points)",
      flush=True)
report["q2"] = {"policy_means": means, "best_race": race_best,
                "best_defensive": def_best}

# ---- Q3: AoE vs single target, with the party there ----------------
print(f"\n[Q3] AoE finisher vs single-target, party present\n", flush=True)
q3 = {}
for label, deck in (("AoE", AOE), ("single-target", SINGLE)):
    ws, ts = [], []
    for b in picked:
        _b, comps = encounter(b.name, bosses)
        _p, v = best(run(b, comps, deck=deck, n=400))
        ws.append(v["win_rate"])
        ts.append(v["mean_ttk"])
    ok = [t for t in ts if t == t]
    q3[label] = {"win": st.mean(ws),
                 "ttk": st.mean(ok) if ok else float("nan")}
    print(f"    {label:<14} mean win {st.mean(ws)*100:5.1f}%   "
          f"mean ttk {q3[label]['ttk']:5.2f}", flush=True)
report["q3"] = q3

json.dump(report, open("results_dps_race.json", "w", encoding="utf-8"),
          indent=1)
print(f"\nwrote results_dps_race.json ({time.time()-t0:.0f}s)", flush=True)
