"""
Archmastery: does the splash die when pips get school-locked?

The user's prediction, stated before the run: "the AI just learns to
use their school pips, since main-class damage is buffed by gear and
off-school isn't usually." Two separable claims:

  MECHANICAL — school pips pay 2 for the wizard's own school and
  NOTHING elsewhere, so any policy that plays what's legal drifts
  toward own-school casts as archmastery rises. Measured as the
  own-school share of damage casts.

  ECONOMIC — gear buffs the main school only, so an off-school splash
  loses twice (no gear bonus, no school pips). If both hold, the
  mastery amulet's big win from era_shift.py (TTK 20.9 -> 14.0 on a
  60% fire wall) should shrink or invert once archmastery and
  school-specific gear are on.

Setup holds the deck and the wall fixed and varies only the era, on
paired seeds: a fire wizard with a heavy storm splash against a boss
that resists fire 60% — deliberately the WORST case for the
prediction, since the splash is what the wall is punishing.

Confidence: mastery's rule is documented; the archmastery RATE and
the gear magnitude are MODELED (see rulesets.py).
"""
import copy
import json
import time
from collections import Counter

from data_full import load_spells_full
from rulesets import ERAS, stats_for
from w101_sim import Sim, Boss, evaluate_paired, make_blade_stack

cards = load_spells_full()
t0 = time.time()

wall = Boss("fire-wall", 2600, "fire", 0)
wall.resist_map = {"fire": 0.60}
wall.boost_map = {}
SPLASH = ["Fireblade", "Fireblade", "Fire Trap", "Fire Trap",
          "Triton", "Triton", "Triton", "Fire Dragon", "Sunbird",
          "Fire Elf"]
MONO = ["Fireblade", "Fireblade", "Fire Trap", "Fire Trap",
        "Fire Dragon", "Fire Dragon", "Sunbird", "Sunbird",
        "Fire Elf", "Fire Elf"]


def cast_mix(rules, deck, school_gear, n=300):
    """Own-school share of DAMAGE casts — the behavioral claim."""
    sim = Sim(dict(cards), deck, "fire", copy.copy(wall),
              player_hp=10**9, rules=rules, log_events=True,
              player_stats=stats_for(rules, "fire", school_gear))
    pol = make_blade_stack(2)
    tally = Counter()
    for i in range(n):
        sim.rng = __import__("random").Random(9000 + i)
        sim.run(pol)
        for e in sim.last_state.events:
            if e.get("type") == "cast_declared":
                c = cards.get(e["card"])
                if c is not None and c.kind in ("damage", "drain"):
                    tally[c.school] += 1
    total = sum(tally.values())
    return (tally.get("fire", 0) / total if total else float("nan"),
            dict(tally))


ARMS = [("mastery", "mastery", False),
        ("mastery+gear", "mastery", True),
        ("am+mastery", "am+mastery", False),
        ("am+mastery+gear", "am+mastery", True)]

print("fire wizard vs a 60% fire wall, carrying a heavy storm splash",
      flush=True)
print("(the worst case for the prediction: the wall punishes the very "
      "school the splash exists to dodge)\n", flush=True)
report = {}
for label, era, gear in ARMS:
    rules = ERAS[era]
    sim = Sim(dict(cards), SPLASH, "fire", copy.copy(wall),
              player_hp=10**9, rules=rules,
              player_stats=stats_for(rules, "fire", gear))
    st = evaluate_paired(sim, {f"blade-stack({k})": make_blade_stack(k)
                               for k in (1, 2, 3)}, n=400)
    best = max(st.items(), key=lambda kv: (round(kv[1]["win_rate"], 2),
                                           -kv[1]["mean_ttk"]))
    share, mix = cast_mix(rules, SPLASH, gear)
    report[label] = {"best_policy": best[0],
                     "win": best[1]["win_rate"],
                     "ttk": best[1]["mean_ttk"],
                     "own_school_damage_share": share,
                     "cast_mix": mix}
    print(f"  {label:<16} win {best[1]['win_rate']*100:5.1f}%  "
          f"ttk {best[1]['mean_ttk']:5.2f}   own-school damage casts "
          f"{share*100:5.1f}%   {mix}", flush=True)

print("\ncontrol — the same wizard on a MONO-fire deck (no splash at "
      "all), to separate 'splash got worse' from 'fight got easier':",
      flush=True)
for label, era, gear in ARMS:
    rules = ERAS[era]
    sim = Sim(dict(cards), MONO, "fire", copy.copy(wall),
              player_hp=10**9, rules=rules,
              player_stats=stats_for(rules, "fire", gear))
    st = evaluate_paired(sim, {f"blade-stack({k})": make_blade_stack(k)
                               for k in (1, 2, 3)}, n=400)
    best = max(st.items(), key=lambda kv: (round(kv[1]["win_rate"], 2),
                                           -kv[1]["mean_ttk"]))
    report[f"mono/{label}"] = {"best_policy": best[0],
                               "win": best[1]["win_rate"],
                               "ttk": best[1]["mean_ttk"]}
    print(f"  {label:<16} win {best[1]['win_rate']*100:5.1f}%  "
          f"ttk {best[1]['mean_ttk']:5.2f}", flush=True)

# the verdict the prediction actually rides on: splash vs mono, per era
print("\nsplash advantage (mono TTK - splash TTK; positive = the "
      "splash is still worth carrying):", flush=True)
verdict = {}
for label, _, _ in ARMS:
    sp, mo = report[label]["ttk"], report[f"mono/{label}"]["ttk"]
    if mo != mo:                      # mono can't win this fight at all
        verdict[label] = None
        print(f"  {label:<16}  n/a  (mono deck unwinnable: the splash "
              f"is the ONLY line)", flush=True)
    else:
        verdict[label] = mo - sp
        print(f"  {label:<16} {mo - sp:+6.2f} turns", flush=True)
report["splash_advantage_turns"] = verdict


def _no_nan(o):
    if isinstance(o, dict):
        return {k: _no_nan(v) for k, v in o.items()}
    if isinstance(o, float) and o != o:
        return None
    return o


json.dump(_no_nan({"setup": {"boss": "fire-wall 2600hp 60% fire resist",
                             "splash": SPLASH, "mono": MONO},
                   "arms": report}),
          open("results_archmastery.json", "w", encoding="utf-8"),
          indent=1, allow_nan=False)
print(f"\nwrote results_archmastery.json ({time.time()-t0:.0f}s)",
      flush=True)
