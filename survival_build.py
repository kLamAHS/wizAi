"""
Survival deck building: what changes when the boss hits back — and
when it hits back in BURSTS.

Two damage regimes against live Jade Oni (6000 HP, 80% life resist,
240 dmg/round rank-inferred):

  chip  — steady 240/round, player_hp = 9 rounds of it. Survival play
          slows the kill (~9 turns vs the immortal ~6.5), so 9 rounds
          is the genuine edge — at 6 rounds NOTHING survives (best
          build 0.3%): under a chip-only model the fight has a hard
          HP floor and the smart money is all offense.
  burst — same boss plus a cheat: every 4th round an out-of-turn 650
          hit ("Oni's Fury"). Finding: burst pressure makes the fight
          MORE of a race, not less — the winning deck is the race
          chassis plus exactly one Pixie, flown aggressively (83.4%);
          shield-heavy builds lose (0.9-35%), and the same winning
          deck under the heal-when-low triage script wins 0.6%. The
          pipeline lesson: a triage-only survival screen went BLIND
          here (every candidate 0-6%, ranking noise) and mis-built;
          the screen now scores each candidate under BOTH the race
          proxy and the triage proxy and keeps the better one.

Three builds per regime: speed (built immortal, then thrown into the
lethal fight), survival (built with player_hp set), surv-p90 (ranked
by the slow tail). All final decks get the same per-deck RL fine-tune
UNDER the lethal sim (no strawman) plus a scripted-triage reference
pilot, cross-evaluated on one paired seed stream.
"""
import copy
import json

from data_full import load_spells_full, load_bosses_full, LIVE_RULES
from deck_builder import build_deck, fine_tune
from w101_sim import (Sim, evaluate_paired, make_blade_stack,
                      make_survival, CheatScript, CheatRule)

cards = load_spells_full()
bosses, _ = load_bosses_full()
jade = bosses["Jade Oni"]

burst = copy.copy(jade)
burst.name = "Jade Oni (burst)"
burst.cheat = CheatScript([
    CheatRule(event="round_start", when={"every": 4},
              ops=[{"op": "hit", "amount": 650, "tgt": "enemy"}],
              message="Oni's Fury"),
])

MATCHUPS = [("chip", jade, 9 * jade.dmg),
            ("burst", burst, 2800)]
ARMS = [("speed", None, "mean"),
        ("survival", "php", "mean"),
        ("surv-p90", "php", "p90")]

results = {}
for label, boss, PLAYER_HP in MATCHUPS:
    print(f"\n#### {label}: death vs {boss.name} ({boss.hp} HP, "
          f"{boss.dmg} dmg/round"
          + (", +650 every 4th round" if boss.cheat else "")
          + f") at {PLAYER_HP} player HP", flush=True)
    speed_boss = copy.copy(boss)
    speed_boss.dmg = 0
    speed_boss.cheat = None
    builds = {}
    for name, php_flag, obj in ARMS:
        php = PLAYER_HP if php_flag else None
        b = boss if php_flag else speed_boss
        print(f"== build: {name} (player_hp={php}, objective={obj}) ==",
              flush=True)
        dl, w, m, _ = build_deck(cards, "death", b, LIVE_RULES,
                                 n_candidates=60, top_k=3, seed=5,
                                 player_hp=php, objective=obj,
                                 log=lambda s: print("  ", s, flush=True))
        builds[name] = dl
        kinds = [cards[n].kind for n in dl]
        print(f"   pick: size {len(dl)}  defense "
              f"{sum(k in ('shield', 'heal') for k in kinds)}  "
              f"{sorted(set(dl))}", flush=True)

    print(f"== cross-evaluation under fire: RL(8k) + scripted triage, "
          f"{PLAYER_HP} HP, paired seeds n=3000 ==", flush=True)
    report = {}
    for name, dl in builds.items():
        w, m, agent = fine_tune(cards, dl, "death", boss, LIVE_RULES,
                                seed=5, player_hp=PLAYER_HP)
        sim = Sim(dict(cards), dl, "death", boss, player_hp=PLAYER_HP,
                  rules=LIVE_RULES)
        st2 = evaluate_paired(sim, {
            "RL(8k)": agent.policy(),
            "triage": make_survival(make_blade_stack(3))}, n=3000)
        report[name] = dict(deck=dl, **{
            pilot: {k: s[k] for k in
                    ("win_rate", "mean_ttk", "median_ttk", "p90_ttk",
                     "p99_ttk", "mean_hp_left")}
            for pilot, s in st2.items()})
        for pilot, s in st2.items():
            print(f"   {name:<9} [{pilot:<7}] win "
                  f"{s['win_rate']*100:5.1f}%  mean {s['mean_ttk']:5.2f}"
                  f"  p90 {s['p90_ttk']:3.0f}  "
                  f"hp left {s['mean_hp_left']:5.0f}", flush=True)
        print(f"             {sorted(set(dl))}", flush=True)
    results[label] = dict(boss=boss.name, boss_hp=boss.hp,
                          boss_dmg=boss.dmg, player_hp=PLAYER_HP,
                          picks=report)

json.dump({"school": "death", "matchups": results},
          open("results_survival_build.json", "w", encoding="utf-8"),
          indent=1)
print("\nwrote results_survival_build.json")
