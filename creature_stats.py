"""
The creature stats that were scraped and then never used (roadmap 2).

Roadmap item 2 asked to "scrape real creature pages (stats, stunable
flags, actual cheat scripts)". No scraping is possible or wanted — the
repo owner's IP is banned from the wiki and the standing rule is that
no further scraping happens. It turns out none is needed: the pre-ban
scrape in bosses_clean.json already carries per-creature PIERCE,
STARTING PIPS, CRITICAL and BLOCK ratings for 1912 creatures, and the
loader was reading only health, school, resist and boost. The data gap
was never in the file; it was in `load_bosses_full`.

Coverage is partial and uneven, so nothing here pretends otherwise: an
absent field means the wiki page did not list it, NOT that the creature
has none.

Q1 — HOW MUCH EASIER HAS THIS SIMULATOR BEEN? Two of the unused fields
are systematically pro-player. Bosses opened every fight with an empty
pip rack (real median: 4 pips), and boss pierce was always zero (real
median where listed: 19%), so player shields and resist have been
working at full strength against opponents that should be cutting
through them. Measured by re-running real encounters with the field
zeroed — which is exactly what the old loader did — against the value
that was sitting in the file all along.

Q2 — A CONCLUSION OF MINE, RE-CHECKED AGAINST REAL DATA. gear_probe.py
concluded that a quad-critical pet is worth nothing against a
high-block boss, using a block rating of 400 that I made up. The real
distribution is now readable: median 57, p90 570, max 1039. So 400 was
neither typical nor extreme, and the honest question is which bucket a
player is actually in. Re-run per real block bucket.
"""
import copy
import json
import random
import statistics as st
import time

from data_full import load_spells_full, load_bosses_full, LIVE_RULES
from gear import loadout
from rulesets import CRIT_ERA
from w101_sim import Sim, evaluate_paired, make_blade_stack, make_survival

cards = load_spells_full()
t0 = time.time()
POLS = {f"blade-stack({k})": make_blade_stack(k) for k in (1, 2, 3)}
POLS.update({f"triage({k})": make_survival(make_blade_stack(k))
             for k in (1, 2)})
DECK = ["Fireblade", "Fireblade", "Fire Trap", "Feint", "Tower Shield",
        "Tower Shield", "Fire Dragon", "Fire Dragon", "Sunbird",
        "Sunbird", "Pixie", "Fire Elf"]
# Endgame content outruns a 12-card deck: this sim makes decks RUN OUT
# (the graveyard only returns via Reshuffle, as in the real game), so
# the high-block bucket needs a deck built for a long fight.
ENDGAME_DECK = (["Fireblade"] * 3 + ["Fire Trap"] * 3 + ["Feint"] * 2
                + ["Fire Dragon"] * 3 + ["Sunbird"] * 3
                + ["Fire Elf"] * 2 + ["Reshuffle"] * 2)


def best_of(sim, n=400):
    r = evaluate_paired(sim, POLS, n=n)
    return max(r.items(), key=lambda kv: (round(kv[1]["win_rate"], 2),
                                          -kv[1]["mean_ttk"]))[1]


report = {}
cov = {}
bosses, registry = load_bosses_full(report=cov)
crit_bosses, _ = load_bosses_full(rules=CRIT_ERA)
print(f"[coverage] {len(bosses)} creatures loaded from the pre-ban "
      f"scrape; per-stat coverage:", flush=True)
for k, v in cov["coverage"].items():
    print(f"    {k:<15} {v['found']:5d}/{v['of']} ({v['frac']*100:4.1f}%)",
          flush=True)
report["coverage"] = cov["coverage"]

pierce_vals = [b.pierce for b in bosses.values() if b.pierce]
pip_vals = [b.start_pips for b in bosses.values() if b.start_pips]
blocks = sorted(b.block_chance for b in crit_bosses.values()
                if b.block_chance)
crits = sorted(b.crit_chance for b in crit_bosses.values()
               if b.crit_chance)
print(f"    pierce      median {st.median(pierce_vals)*100:.0f}%  "
      f"max {max(pierce_vals)*100:.0f}%", flush=True)
print(f"    start pips  median {st.median(pip_vals)}  "
      f"max {max(pip_vals)}", flush=True)
print(f"    block       median {st.median(blocks):.0f}  "
      f"p90 {blocks[int(len(blocks)*.9)]:.0f}  max {max(blocks):.0f}",
      flush=True)
print(f"    critical    median {st.median(crits):.0f}  "
      f"max {max(crits):.0f}", flush=True)
report["distributions"] = {
    "pierce_median": st.median(pierce_vals),
    "pierce_max": max(pierce_vals),
    "start_pips_median": st.median(pip_vals),
    "block_median": st.median(blocks),
    "block_p90": blocks[int(len(blocks) * .9)],
    "block_max": max(blocks),
    "crit_median": st.median(crits)}

# ---- Q1: how much has zero pierce been worth to the player? --------
print("\n[Q1] boss pierce, zeroed (the old loader) vs the value that "
      "was in the file\n", flush=True)
LVL = 100
lo = loadout(LVL, "fire", rules=LIVE_RULES)


def run(boss, pierce=None, rules=LIVE_RULES, php=None, glo=None):
    bb = copy.copy(boss)
    if pierce is not None:
        bb.pierce = pierce
    g = glo or lo
    sim = Sim(dict(cards), DECK, "fire", bb,
              player_hp=g["player_hp"] if php is None else php,
              rules=rules, player_stats=g["player_stats"],
              power_pip=g["power_pip"])
    return best_of(sim)


# Only bosses where the CONTROL arm is contested can show the effect at
# all: a fight the wizard already wins 100% of the time cannot get
# easier, and one it never wins cannot get harder. Screening on the
# control arm (never on the treatment) is what makes the measurement
# sensitive; the first cut skipped this and 9 of 12 rows were pinned at
# a ceiling or a floor, so the mean was two bosses wearing a trenchcoat.
cands = [b for b in bosses.values()
         if b.pierce >= 0.10 and 3000 <= b.hp <= 12000]
cands.sort(key=lambda b: b.name)
rng = random.Random(11)
rng.shuffle(cands)
rows, deltas, scanned = [], [], 0
for b in cands:
    if len(rows) >= 10:
        break
    scanned += 1
    off = run(b, pierce=0.0)
    if not 0.15 <= off["win_rate"] <= 0.98:
        continue                      # pinned: no room to move
    on = run(b)
    d = (off["win_rate"] - on["win_rate"]) * 100
    deltas.append(d)
    rows.append({"boss": b.name, "hp": b.hp, "pierce": b.pierce,
                 "win_pierce_off": off["win_rate"],
                 "win_pierce_real": on["win_rate"], "drop_points": d})
    print(f"    {b.name:<28} hp {b.hp:5d}  pierce {b.pierce*100:3.0f}%   "
          f"{off['win_rate']*100:5.1f}% -> {on['win_rate']*100:5.1f}%   "
          f"({-d:+5.1f} pts)", flush=True)
print(f"\n    {len(rows)} contested encounters out of {scanned} scanned "
      f"(the rest were pinned at 0% or 100% with pierce off)", flush=True)
print(f"    mean win-rate drop from honouring boss pierce: "
      f"{sum(deltas)/len(deltas):+.1f} points "
      f"(median {st.median(deltas):+.1f})", flush=True)
report["q1_pierce"] = {"rows": rows, "scanned": scanned,
                       "screen": "control arm win rate in [0.15, 0.98]",
                       "mean_drop_points": sum(deltas) / len(deltas),
                       "median_drop_points": st.median(deltas)}

# ---- Q2: the crit-pet call against the REAL block distribution -----
print("\n[Q2] quad-critical vs triple-double per REAL block bucket "
      "(gear_probe used a made-up 400)\n", flush=True)
# Block rating tracks TIER in the real data: every creature with a
# block rating >= 570 has at least 13675 HP. So the buckets get their
# own HP windows rather than a shared one — forcing a common window
# empties the high bucket entirely, which is what the first cut did.
# TTKs are therefore not comparable ACROSS buckets; the pet gains
# within a bucket are, and those are what is reported.
BUCKETS = [("low   (<100)", lambda v: v < 100, (4000, 12000)),
           ("mid   (100-570)", lambda v: 100 <= v < 570, (4000, 12000)),
           ("high  (>=570)", lambda v: v >= 570, (13000, 30000))]
pet_rows = {}


def race_ttk(boss, pet, lvl=None):
    """TTK on a pure damage race, endgame deck, 80-turn budget."""
    glo = loadout(lvl or LVL, "fire", rules=CRIT_ERA, pet=pet,
                  pet_age="mega")
    bb = copy.copy(boss)
    bb.dmg = 0
    sim = Sim(dict(cards), ENDGAME_DECK, "fire", bb, player_hp=10**9,
              rules=CRIT_ERA, player_stats=glo["player_stats"],
              power_pip=glo["power_pip"])
    r = evaluate_paired(sim, POLS, n=250, max_turns=80)
    return max(r.values(), key=lambda v: (round(v["win_rate"], 2),
                                          -v["mean_ttk"]))


for bname, pred, (hp_lo, hp_hi) in BUCKETS:
    pool = [b for b in crit_bosses.values()
            if b.block_chance and pred(b.block_chance)
            and hp_lo <= b.hp <= hp_hi]
    if not pool:
        print(f"    {bname:<16} no creature in range", flush=True)
        continue
    pool.sort(key=lambda b: b.name)
    sampled = random.Random(5).sample(pool, min(6, len(pool)))
    # Reachability is screened PER BOSS on the no-pet control arm: TTK
    # is undefined where the wizard cannot win, and one unkillable
    # boss must not turn the whole bucket into a NaN.
    picks, base_ttk, unreached = [], {}, []
    for b in sampled:
        r = race_ttk(b, None)
        if r["win_rate"] >= 0.5:
            picks.append(b)
            base_ttk[b.name] = r["mean_ttk"]
        else:
            unreached.append((b.name, b.hp, r["win_rate"]))
        if len(picks) >= 4:
            break
    mean_block = st.mean(b.block_chance for b in sampled)
    if not picks:
        worst = unreached[0]
        pet_rows[bname] = {"n_sampled": len(sampled), "reachable": 0,
                           "hp_window": [hp_lo, hp_hi],
                           "mean_block": mean_block,
                           "example": {"boss": worst[0], "hp": worst[1],
                                       "win_rate": worst[2]}}
        print(f"    {bname:<16} mean block {mean_block:6.0f}   "
              f"UNREACHABLE: 0 of {len(sampled)} killable by a solo "
              f"level-{LVL} single-target fire deck in 80 turns "
              f"(e.g. {worst[0]}, {worst[1]} HP)", flush=True)
        continue
    base = sum(base_ttk.values()) / len(base_ttk)
    gains = {}
    for pet in ("triple-double", "quad-critical"):
        ttks = [race_ttk(b, pet)["mean_ttk"] for b in picks]
        gains[pet] = base - sum(ttks) / len(ttks)
    ratio = (gains["quad-critical"] / gains["triple-double"]
             if gains["triple-double"] else None)
    pet_rows[bname] = {"n_sampled": len(sampled), "reachable": len(picks),
                       "hp_window": [hp_lo, hp_hi],
                       "mean_block": st.mean(b.block_chance
                                             for b in picks),
                       "no_pet_ttk": base,
                       "triple_double_gain": gains["triple-double"],
                       "quad_critical_gain": gains["quad-critical"],
                       "quad_over_triple": ratio,
                       "bosses": [b.name for b in picks],
                       "unreachable": [u[0] for u in unreached]}
    print(f"    {bname:<16} n={len(picks)}/{len(sampled)} reachable  "
          f"mean block {st.mean(b.block_chance for b in picks):6.0f}   "
          f"triple-double {gains['triple-double']:+.2f}   "
          f"quad-critical {gains['quad-critical']:+.2f}   ratio "
          f"{'n/a' if ratio is None else format(ratio, '.2f')}",
          flush=True)

report["q2_pet_by_real_block"] = pet_rows

json.dump(report, open("results_creature_stats.json", "w",
                       encoding="utf-8"), indent=1)
print(f"\nwrote results_creature_stats.json ({time.time()-t0:.0f}s)",
      flush=True)
