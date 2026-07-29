"""
Given the choice, does the BUILDER spend slots on enchantments?

enchant_probe.py showed hand-built enchanted decks beating hand-built
plain ones by ~6 turns at matched deck slots. That is a statement about
decks I wrote. The sharper question is whether the deck search
discovers the trade on its own, because the trade is genuinely
non-obvious: an enchanted card costs TWO physical slots and competes
for its base spell's copy limit, so every enchant buys power by
shrinking the deck.

`legal_pool(..., enchants=True)` now offers the enchanted variants a
wizard of that level owns, `check_legal` counts real slots and
per-SPELL copies, and `sample_deck` respects both. Nothing tells the
builder whether to use them.

Swept one level per world, each at the END of its world so the wizard
owns everything that world offers. Level 50 (Dragonspyre) is a clean
CONTROL: Sun enchants open at Celestia 51, so the two arms cannot
differ there and any difference would be a bug.

Confidence: the enchant magnitudes and rules are as supplied by the
repo owner, and the unlock levels now resolve through `worlds.py`
against the level bands he supplied — so the floors are sourced to a
world boundary rather than guessed.
"""
import copy
import json
import time
from collections import Counter

from data_full import load_spells_full, LIVE_RULES
from deck_builder import (ENCHANT_UNLOCK, build_deck, legal_pool,
                          _best_damage_enchant)
from gear import loadout
from w101_sim import Boss, enchant_base, enchanted_deck_size, is_enchanted
from worlds import world_for

cards = load_spells_full()
t0 = time.time()
# one level per enchant tier, each the END of its world so the
# wizard owns everything that world offers: 50 Dragonspyre (a
# clean control — Sun enchants open at Celestia 51), 55 mid
# Celestia, 70 Zafaria, 100 Khrysalis, 120 Mirage.
LEVELS = [50, 55, 70, 100, 120]
CAPACITY = 30


def boss_for(level):
    """A rank-appropriate wall: enough HP that deck efficiency matters."""
    hp = int(1200 + 110 * level)
    b = Boss(f"L{level}-wall", hp, "death", 0)
    b.resist_map = {"*": 0.35}
    b.boost_map = {}
    return b


print(f"deck capacity {CAPACITY} real slots; an enchanted entry costs 2 "
      f"and shares its base spell's copy limit\n", flush=True)
report = {}
for lvl in LEVELS:
    best = _best_damage_enchant(lvl)
    pool = legal_pool(cards, "fire", level=lvl, enchants=True)
    offered = sum(1 for n in pool if is_enchanted(n))
    stats = loadout(lvl, "fire", rules=LIVE_RULES)
    row = {"offered": offered, "best_damage_enchant": best}
    for arm, ench in (("plain", False), ("free to enchant", True)):
        dl, w, m, _ = build_deck(
            cards, "fire", copy.copy(boss_for(lvl)), LIVE_RULES,
            n_candidates=60, top_k=3, capacity=CAPACITY, copy_limit=3,
            seed=17, level=lvl, enchants=ench,
            player_stats=stats["player_stats"],
            power_pip=stats["power_pip"], log=None)
        n_ench = sum(1 for n in dl if is_enchanted(n))
        row[arm] = {"deck": dl, "castables": len(dl),
                    "slots": enchanted_deck_size(dl),
                    "enchanted_cards": n_ench,
                    "enchanted_share": n_ench / len(dl) if dl else 0.0,
                    "win": w, "ttk": m}
    a, b = row["plain"], row["free to enchant"]
    row["ttk_gain"] = a["ttk"] - b["ttk"]
    row["win_gain"] = (b["win"] - a["win"]) * 100
    report[lvl] = row
    row["world"] = world_for(lvl)
    print(f"L{lvl:<4} {str(world_for(lvl)):<12} best flat enchant "
          f"{str(best):<11} {offered:2d} offered", flush=True)
    print(f"      plain           {a['castables']:2d} cards / "
          f"{a['slots']:2d} slots   win {a['win']*100:5.1f}%  "
          f"ttk {a['ttk']:5.2f}", flush=True)
    print(f"      free to enchant {b['castables']:2d} cards / "
          f"{b['slots']:2d} slots   win {b['win']*100:5.1f}%  "
          f"ttk {b['ttk']:5.2f}   "
          f"{b['enchanted_cards']} enchanted "
          f"({b['enchanted_share']*100:.0f}% of the deck)", flush=True)

print("\nwhat the builder chose, by level:", flush=True)
for lvl in LEVELS:
    r = report[lvl]
    b = r["free to enchant"]
    picked = sorted({n for n in b["deck"] if is_enchanted(n)})
    print(f"  L{lvl:<4} {str(r['world']):<12} "
          f"{b['enchanted_share']*100:4.0f}% enchanted   "
          f"ttk {r['ttk_gain']:+5.2f}  win {r['win_gain']:+5.1f} pts   "
          f"{picked or 'none'}", flush=True)

json.dump({"setup": {"capacity_real_slots": CAPACITY, "copy_limit": 3,
                     "levels": LEVELS,
                     "unlocks": ENCHANT_UNLOCK,
                     "confidence": "enchant magnitudes and rules as "
                                   "supplied by the repo owner; level "
                                   "floors INFERRED from the worlds he "
                                   "named"},
           "by_level": report},
          open("results_enchant_decks.json", "w", encoding="utf-8"),
          indent=1)
print(f"\nwrote results_enchant_decks.json ({time.time()-t0:.0f}s)",
      flush=True)
