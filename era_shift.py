"""
Does the meta shift when the era does? (roadmap item 8)

The repo has spent its whole life in one era: criticals off, no
mastery amulet. `rulesets.py` freezes the post-classic ones, so the
question the sim can now answer is whether the strategy conclusions
drawn under classic rules are era-specific or era-invariant.

Two probes, both on paired seeds so eras are compared on identical
draws:

  CRITICALS — does the optimal blade count move? Criticals multiply
  the FINAL hit, so they compose multiplicatively with blades and in
  principle shouldn't change the stacking optimum; they do add
  variance, which helps a deck that needs one big hit. Measured
  rather than argued.

  MASTERY — a fire wizard facing a 60%-fire-resist wall carries a
  storm splash. Without the amulet the off-school nuke costs full
  pips (power pips pay 1); with it they pay 2, which is the whole
  point of the item. Does the splash go from unaffordable to the
  correct line?

Confidence: mastery's rule is documented mechanics; the crit CURVES
are modeled (see make_rating_crit) and the gear ratings are
illustrative. So read the crit arm as "how the meta responds to a
crit regime of roughly this shape", not as a live-game claim.
"""
import copy
import json
import time

from data_full import load_spells_full, load_bosses_full
from rulesets import BOSS_STATS, ERAS, stats_for
from search_policy import make_search_policy
from w101_sim import Sim, Boss, evaluate_paired, make_blade_stack

cards = load_spells_full()
bosses, _ = load_bosses_full()
t0 = time.time()
report = {}

# ---------------------------------------------------------------- crits
print("[1/2] criticals: does the optimal blade count move?", flush=True)
# the blade count only BINDS if the nukes are cheap enough to cast
# immediately — with 6-7 pip finishers the policy stacks buffs during
# the idle turns anyway and every k collapses to the same line (the
# first cut of this probe scored k=0..3 identically, which measured
# the deck, not the era)
DECK = ["Banshee", "Banshee", "Banshee", "Ghoul", "Ghoul",
        "Deathblade", "Deathblade", "Death Trap", "Curse", "Feint"]
base = Boss("crit-probe", 1400, "death", 0)
base.resist_map = {}
base.boost_map = {}
crit_rows = {}
for era in ("live", "crit-era", "modern"):
    rules = ERAS[era]
    boss = copy.copy(base)
    if rules.crit_resolver:                 # give the boss block to
        boss.block_chance = BOSS_STATS["block"]   # contest player crits
    sim = Sim(dict(cards), DECK, "death", boss, player_hp=10**9,
              rules=rules, player_stats=stats_for(rules))
    pols = {f"blade-stack({k})": make_blade_stack(k) for k in range(4)}
    st = evaluate_paired(sim, pols, n=800)
    best = max(st.items(), key=lambda kv: (round(kv[1]["win_rate"], 2),
                                           -kv[1]["mean_ttk"]))
    crit_rows[era] = {k: {"win": v["win_rate"], "ttk": v["mean_ttk"]}
                      for k, v in st.items()}
    crit_rows[era]["best"] = best[0]
    line = "  ".join(f"k={k[-2]} {v['win_rate']*100:4.0f}%/"
                     f"{v['mean_ttk']:5.2f}" for k, v in st.items())
    print(f"    {era:<9} {line}   best: {best[0]}", flush=True)
report["criticals"] = crit_rows

# --------------------------------------------------------------- mastery
print("\n[2/2] mastery amulet: a fire wizard vs a 60% fire wall, "
      "carrying a storm splash", flush=True)
wall = Boss("fire-wall", 2600, "fire", 0)
wall.resist_map = {"fire": 0.60}
wall.boost_map = {}
SPLASH = ["Fireblade", "Fireblade", "Fire Trap", "Fire Trap",
          "Triton", "Triton", "Triton", "Fire Dragon", "Sunbird",
          "Fire Elf"]
mast_rows = {}
for era in ("live", "mastery"):
    rules = ERAS[era]
    sim = Sim(dict(cards), SPLASH, "fire", copy.copy(wall),
              player_hp=10**9, rules=rules,
              player_stats=stats_for(rules))
    pols = {f"blade-stack({k})": make_blade_stack(k) for k in (1, 2, 3)}
    pols["search(k=5)"] = make_search_policy(k=5)
    st = evaluate_paired(sim, pols, n=250)
    mast_rows[era] = {k: {"win": v["win_rate"], "ttk": v["mean_ttk"]}
                      for k, v in st.items()}
    for k, v in st.items():
        ttk = v["mean_ttk"]
        print(f"    {era:<8} {k:<16} win {v['win_rate']*100:5.1f}%"
              + (f"  ttk {ttk:5.2f}" if ttk == ttk else ""), flush=True)
report["mastery"] = mast_rows


def _no_nan(o):
    if isinstance(o, dict):
        return {k: _no_nan(v) for k, v in o.items()}
    if isinstance(o, float) and o != o:
        return None
    return o


json.dump(_no_nan({"eras": {k: v.ruleset_id for k, v in ERAS.items()},
                   "results": report}),
          open("results_eras.json", "w", encoding="utf-8"), indent=1,
          allow_nan=False)
print(f"\nwrote results_eras.json ({time.time()-t0:.0f}s)", flush=True)
