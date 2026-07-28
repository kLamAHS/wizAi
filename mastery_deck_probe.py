"""
Does the DECK BUILDER rediscover the splash economics on its own?

archmastery_probe.py showed that with school-locked pips and
own-school gear, a hand-built storm splash stops paying for a fire
wizard — but it also showed the tax lands on the DECK rather than the
play: a policy cannot spend school pips it has no own-school cards
for. So the adaptation belongs to deck CONSTRUCTION, and that is
testable: `legal_pool(..., mastery=...)` now widens the buildable
pool with the amulet school's trained line, so the builder is FREE to
splash — and nothing tells it whether to.

Two OPPONENTS x two eras, because the first cut of this probe used
only a 60%-fire-resist wall and found no era effect at all — a wall
that punishes your school makes splashing correct for a reason that
has nothing to do with pips or gear, and it is a bigger force than
both combined. The neutral boss is where the prediction actually
applies; the wall is kept as the informative exception.

  mastery          amulet only: off-school storm pays full power pips,
                   no school-locked pips, no own-school gear bonus
  am+mastery+gear  the modern economics: 35% of pips arrive
                   fire-locked, and gear buffs fire only

Prediction under test (the repo owner's): the builder should splash
heavily in the first arm and go mono-fire in the second, because the
splash loses its gear bonus AND strands the school pips. Reported as
the storm share of the built deck's damage cards.
"""
import copy
import json
import time

from data_full import load_spells_full
from deck_builder import build_deck, legal_pool
from rulesets import ERAS, stats_for
from w101_sim import Boss

cards = load_spells_full()
t0 = time.time()

wall = Boss("fire-wall", 2600, "fire", 0)
wall.resist_map = {"fire": 0.60}
wall.boost_map = {}
neutral = Boss("neutral", 2600, "death", 0)
neutral.resist_map = {}
neutral.boost_map = {}
BOSSES = [("neutral", neutral), ("fire-wall", wall)]

pool = legal_pool(cards, "fire", mastery="storm")
storm_avail = sum(1 for n, c in pool.items() if c.school == "storm")
print(f"buildable pool with a storm mastery: {len(pool)} cards "
      f"({storm_avail} storm) — the builder CAN splash, but nothing "
      f"tells it to", flush=True)

ARMS = [("mastery", "mastery", False),
        ("am+mastery+gear", "am+mastery", True)]
report = {}
for bname, boss in BOSSES:
  for label, era, gear in ARMS:
    key = f"{bname}/{label}"
    rules = ERAS[era]
    stats = stats_for(rules, "fire", gear)
    print(f"\n== {bname}: building under {label} "
          f"(archmastery {rules.archmastery:.2f}, gear {stats or '{}'})",
          flush=True)
    dl, w, m, _ = build_deck(cards, "fire", copy.copy(boss), rules,
                             n_candidates=60, top_k=3, seed=31,
                             mastery="storm", player_stats=stats,
                             log=None)
    dmg = [cards[n] for n in dl if cards[n].kind in ("damage", "drain")]
    storm = sum(1 for c in dmg if c.school == "storm")
    share = storm / len(dmg) if dmg else float("nan")
    report[key] = {"deck": dl, "size": len(dl), "win": w, "ttk": m,
                   "damage_cards": len(dmg), "storm_damage": storm,
                   "storm_share_of_damage": share}
    print(f"   built (size {len(dl)}): {sorted(set(dl))}", flush=True)
    print(f"   win {w*100:5.1f}%  ttk {m:5.2f}   storm share of damage "
          f"cards: {share*100:5.1f}% ({storm}/{len(dmg)})", flush=True)

print("\nsplash share, mastery -> am+mastery+gear:", flush=True)
drops = {}
for bname, _ in BOSSES:
    a = report[f"{bname}/mastery"]["storm_share_of_damage"]
    b = report[f"{bname}/am+mastery+gear"]["storm_share_of_damage"]
    drops[bname] = a - b
    print(f"  {bname:<10} {a*100:5.1f}% -> {b*100:5.1f}%  "
          f"({(a-b)*100:+.1f} points)", flush=True)
report["splash_share_drop"] = drops

json.dump({"setup": {"bosses": {"neutral": "2600hp, no resists",
                                "fire-wall": "2600hp, 60% fire resist"},
                     "wizard": "fire, storm mastery amulet",
                     "arms": {a[0]: {"era": a[1], "school_gear": a[2]}
                              for a in ARMS},
                     "pool_size": len(pool),
                     "storm_cards_available": storm_avail},
           "arms": report},
          open("results_mastery_deck.json", "w", encoding="utf-8"),
          indent=1)
print(f"\nwrote results_mastery_deck.json ({time.time()-t0:.0f}s)",
      flush=True)
