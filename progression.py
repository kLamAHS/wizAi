"""
Progression sweep: the level-1-upward benchmark. At each level milestone
the deck builder searches the level-gated pool against a level-scaled
opponent; what should emerge is the strategy ladder (attacks -> blade+
attack -> full stacks) and deck size tracking fight length. Writes
progression.json for plotting.

Boss HP tracks real early-game scale (Lost Soul ~100 HP at level 1,
Wizard City bosses ~600 by 10, Mooshu ~1800 by 30): with the buildable
pool now restricted to genuinely trained spells, a level-1 wizard tops
out at 3x Fire Cat = 300 damage, so the old 495-HP level-1 dummy was
only ever beatable with pet-card pollution.
"""
import json
import random

from data_full import load_spells_full, LIVE_RULES
from deck_builder import build_deck
from player_curves import base_power_pip, school_hp
from w101_sim import Boss

cards = load_spells_full()
LEVELS = [1, 5, 10, 15, 20, 25, 30, 40, 50]
rows = []
for lvl in LEVELS:
    boss = Boss(f"lvl{lvl}-training-dummy", 45 + 55 * lvl, "death", 0)
    boss.resist_map = {"death": 0.5}
    boss.boost_map = {}
    cap = min(10 + (lvl // 10) * 2, 16)
    pp = base_power_pip(lvl)               # 0% until 10, 40% cap at 50
    print(f"[lvl {lvl}] boss {boss.hp} HP, capacity {cap}, base power "
          f"pips {pp*100:.0f}% — searching...", flush=True)
    dl, w, m, table = build_deck(
        cards, "fire", boss, LIVE_RULES, n_candidates=60, top_k=3,
        capacity=cap, copy_limit=3, seed=lvl, level=lvl, power_pip=pp,
        log=lambda msg: print("   ", msg, flush=True))
    from deck_builder import legal_pool
    pool = legal_pool(cards, "fire", level=lvl)
    row = dict(level=lvl, boss_hp=boss.hp, pool=len(pool), capacity=cap,
               power_pip=pp,
               player_hp_base=school_hp("fire", lvl),   # reference curve value;
               # dummy fights are immortal — HP is NOT simulated here
               deck_size=len(dl), win=w, ttk=m, deck=sorted(set(dl)))
    rows.append(row)
    print(f"lvl {lvl:>2}: pool {len(pool):>3}  cap {cap}  "
          f"deck {len(dl):>2}  win {w*100:5.1f}%  ttk {m:5.2f}  "
          f"{row['deck']}", flush=True)
json.dump({"ruleset_id": LIVE_RULES.ruleset_id, "school": "fire",
           "levels": rows}, open("progression.json", "w",
                                 encoding="utf-8"), indent=1)
print("wrote progression.json")
