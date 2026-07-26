"""
Progression sweep: the level-1-upward benchmark. At each level milestone
the deck builder searches the level-gated pool against a level-scaled
opponent; what should emerge is the strategy ladder (attacks -> blade+
attack -> full stacks) and deck size tracking fight length. Writes
progression.json for plotting.
"""
import json
import random

from data_full import load_spells_full, LIVE_RULES
from deck_builder import build_deck
from w101_sim import Boss

cards = load_spells_full()
LEVELS = [1, 5, 10, 15, 20, 25, 30, 40, 50]
rows = []
for lvl in LEVELS:
    boss = Boss(f"lvl{lvl}-training-dummy", 400 + 95 * lvl, "death", 0)
    boss.resist_map = {"death": 0.5}
    boss.boost_map = {}
    cap = min(10 + (lvl // 10) * 2, 16)
    print(f"[lvl {lvl}] boss {boss.hp} HP, capacity {cap} — "
          f"searching...", flush=True)
    dl, w, m, table = build_deck(
        cards, "fire", boss, LIVE_RULES, n_candidates=60, top_k=3,
        capacity=cap, copy_limit=3, seed=lvl, level=lvl,
        log=lambda msg: print("   ", msg, flush=True))
    from deck_builder import legal_pool
    pool = legal_pool(cards, "fire", level=lvl)
    row = dict(level=lvl, boss_hp=boss.hp, pool=len(pool), capacity=cap,
               deck_size=len(dl), win=w, ttk=m, deck=sorted(set(dl)))
    rows.append(row)
    print(f"lvl {lvl:>2}: pool {len(pool):>3}  cap {cap}  "
          f"deck {len(dl):>2}  win {w*100:5.1f}%  ttk {m:5.2f}  "
          f"{row['deck']}", flush=True)
json.dump({"ruleset_id": LIVE_RULES.ruleset_id, "school": "fire",
           "levels": rows}, open("progression.json", "w",
                                 encoding="utf-8"), indent=1)
print("wrote progression.json")
