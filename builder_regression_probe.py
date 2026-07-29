"""
The builder thought every living boss was harmless. Found, then fixed.

HOW IT SURFACED. Regenerating `results_living.json` moved the
living-Krokopatra ladder around, and one of the moves was not the
blade/trap fix everything else was being blamed on: the DECK underneath
the ladder had changed. The builder used to hand level 20 ten cards
with a Pixie in them; it started returning eight with no heal and no
shield. That cost two things — the deck was weaker, and `make_survival`
on a deck with neither a heal nor a shield is the identity wrapper, so
the "triage" pilot in that ladder stopped being a distinct policy at
all.

Two candidate causes were separated first:

  SCORING     the screen sees both decks and prefers the worse one
  GENERATION  the screen never sees the better one

It answered GENERATION. Offer the screen both decks and it ranks the
old one first by eleven points on its own n=250 protocol, and
quadrupling the candidate budget recovered almost nothing.

THE CAUSE. `sample_deck` gated its whole defensive branch — shields,
and heals at 0.8 instead of 0.3 — on `boss.dmg > 0`. That is the flat
per-round damage number a LEGACY boss carries. A living boss deals its
damage out of a spell pool and carries `dmg = 0`. That is not an edge
case: of the 1,909 bosses in the live data, 1,795 are pool casters and
every one of them reads `dmg = 0`, so the test was False for 94% of the
game and for every boss `main.py` fights. The builder was offering
defense to legacy bosses only, and treating the entire living-boss
layer — the thing the repo added specifically because flat damage
understates a real fight — as if it could not hurt you.

THE FIX. Lethality is `can_hurt(boss)`: flat damage OR a spell pool.
And it is a property of the ENCOUNTER, not the boss, so `build_deck`
ORs in the minions — a harmless boss escorted by a hitter still kills
you.

    python builder_regression_probe.py
"""
import copy
import json
import random

from data_full import load_spells_full, load_bosses_full, LIVE_RULES
from player_curves import school_hp, base_power_pip
from deck_builder import (build_deck, screen, sample_deck, legal_pool,
                          can_hurt)
from w101_sim import Sim, evaluate_paired, make_blade_stack, make_survival

LEVEL = 20
N_REPORT = 3000            # the ladder's own budget
CAPACITY = 14

cards = load_spells_full()
bosses, _ = load_bosses_full()

living = copy.copy(bosses["Krokopatra"])
living.dmg = 0
living.pool = ["Thunder Snake", "Lightning Bats", "Storm Shark",
               "Stormblade", "Storm Shield"]
living.archetype = "hitter"
living.discipline = 0.7

HP, PP = school_hp("death", LEVEL), base_power_pip(LEVEL)

# the deck the builder returned before the template changes, taken from
# the results file as it stood at commit 653e8d9, and the one it
# returned while the lethality test was broken
OLD_DECK = ["Banshee", "Banshee", "Sacrifice", "Ghoul", "Deathblade",
            "Deathblade", "Balanceblade", "Hex", "Curse", "Pixie"]
BROKEN_DECK = ["Banshee", "Banshee", "Banshee", "Sacrifice",
               "Deathblade", "Deathblade", "Hex", "Hex"]

# TWO triage pilots on purpose. `build_deck`'s screen judges candidates
# with blade-stack(3) and its triage wrap, so bs3 is the pilot the deck
# was SELECTED for; `living_bosses.py` then reports the same deck under
# blade-stack(2). That gap is worth seeing rather than picking one side
# of — on the deck the fixed builder returns it is 22 points wide.
triage3 = make_survival(make_blade_stack(3))
triage2 = make_survival(make_blade_stack(2))


def report(deck, n=N_REPORT):
    sim = Sim(dict(cards), deck, "death", living, player_hp=HP,
              power_pip=PP, rules=LIVE_RULES)
    st = evaluate_paired(sim, {"triage(bs3)": triage3,
                               "triage(bs2)": triage2,
                               "bs3": make_blade_stack(3)}, n=n)
    return {k: v["win_rate"] for k, v in st.items()}


print("== how many bosses did `boss.dmg > 0` call harmless? ==",
      flush=True)
full, _ = load_bosses_full(spell_pools=True, cards=cards)
pooled = [b for b in full.values() if getattr(b, "pool", None)]
flat = [b for b in full.values() if not getattr(b, "pool", None)
        and b.dmg > 0]
print(f"   {len(full)} bosses: {len(pooled)} pool casters, "
      f"{len(flat)} flat-damage", flush=True)
print(f"   pool casters reading dmg == 0: "
      f"{sum(1 for b in pooled if b.dmg == 0)} "
      f"({sum(1 for b in pooled if b.dmg == 0) / len(full) * 100:.0f}% "
      f"of every boss in the data)", flush=True)

print("\n== does the template offer defense now? same seeds, "
      "lethal on and off ==", flush=True)
pool20 = legal_pool(cards, "death", level=LEVEL)
counts = {}
for flagname, flag in (("lethal=False", False), ("lethal=True", True)):
    n_def = 0
    for i in range(200):
        dl = sample_deck(pool20, "death", living, random.Random(i),
                         CAPACITY, 3, lethal=flag)
        if any(cards[c].kind in ("heal", "shield") for c in dl):
            n_def += 1
    counts[flagname] = n_def
    print(f"   {flagname:<14} {n_def:>3}/200 candidates carry a heal "
          f"or a shield", flush=True)
print(f"   can_hurt(living Krokopatra) = {can_hurt(living)}   "
      f"(its dmg is {living.dmg})", flush=True)

print(f"\n== the decks, at the ladder's own budget (n={N_REPORT}) ==",
      flush=True)
decks = [("old (10, Pixie)", OLD_DECK),
         ("broken ( 8, no heal)", BROKEN_DECK)]
built, _, _, _ = build_deck(cards, "death", living, LIVE_RULES,
                            n_candidates=60, top_k=3, seed=12,
                            level=LEVEL, player_hp=HP, power_pip=PP,
                            capacity=CAPACITY, log=None)
decks.append((f"built now ({len(built):>2})", built))
scores = {}
for name, deck in decks:
    r = report(deck)
    scores[name] = r
    same = abs(r["triage(bs3)"] - r["bs3"]) < 1e-9
    print(f"   {name:<21} triage(bs3) {r['triage(bs3)']*100:5.1f}%   "
          f"triage(bs2) {r['triage(bs2)']*100:5.1f}%   "
          f"bs3 {r['bs3']*100:5.1f}%"
          + ("   <- triage == bs3: no heal, no shield" if same else ""),
          flush=True)
print(f"   built deck: {sorted(set(built))}", flush=True)

print("\n== with defense back on the table, the candidate budget "
      "starts to bind ==", flush=True)
budget = {}
for nc in (60, 240):
    dl, _, _, _ = build_deck(cards, "death", living, LIVE_RULES,
                             n_candidates=nc, top_k=3, seed=12,
                             level=LEVEL, player_hp=HP, power_pip=PP,
                             capacity=CAPACITY, log=None)
    r = report(dl)
    defense = sorted({c for c in dl
                      if cards[c].kind in ("heal", "shield")})
    budget[nc] = {"triage(bs3)": r["triage(bs3)"],
                  "triage(bs2)": r["triage(bs2)"],
                  "size": len(dl), "defense": defense}
    print(f"   n_candidates={nc:>4}  size {len(dl):>2}  "
          f"triage(bs3) {r['triage(bs3)']*100:5.1f}%  "
          f"triage(bs2) {r['triage(bs2)']*100:5.1f}%  "
          f"defense {defense}", flush=True)
print("   -> 4x the budget is worth another ~15 points here. Before "
      "the fix it bought 0.4, because\n      the extra candidates were "
      "all drawn from the same defense-free template.", flush=True)

print("\n== was it ever a SCORING failure? offer the screen both ==",
      flush=True)
table = screen(cards, [OLD_DECK, BROKEN_DECK], "death", living,
               LIVE_RULES, player_hp=HP, power_pip=PP)
for name, row in zip(("old", "broken"), table):
    print(f"   {name:<21} screen {row[0]*100:5.1f}%", flush=True)
print("   -> no. the screen preferred the old deck whenever it could "
      "see it; it was never offered one.", flush=True)

json.dump({"level": LEVEL, "player_hp": HP, "report_n": N_REPORT,
           "census": {"bosses": len(full), "pool_casters": len(pooled),
                      "flat_damage": len(flat),
                      "pool_casters_dmg_zero":
                          sum(1 for b in pooled if b.dmg == 0)},
           "defensive_candidates_per_200": counts,
           "decks": {"old": OLD_DECK, "broken": BROKEN_DECK,
                     "built_now": built},
           "budget_sweep": {str(k): v for k, v in budget.items()},
           "report": {k: v for k, v in scores.items()},
           "screen_n250": {"old": table[0][0], "broken": table[1][0]}},
          open("results_builder_regression.json", "w", encoding="utf-8"),
          indent=1)
print("\nwrote results_builder_regression.json")
