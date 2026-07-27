"""
Survival deck building: what changes when the boss hits back.

The knife-edge matchup: death vs live Jade Oni (6000 HP, 80% life
resist, 240 dmg/round rank-inferred) with player_hp = 9 rounds of
boss damage. Survival play itself slows the kill (~9 turns vs the
immortal ~6.5), so 9 rounds is the genuine edge — at 6 rounds NOTHING
survives (first run: best build 0.3%), which is itself informative:
under this cheatless damage model the fight has a hard HP floor.
Three builds:

  speed     — built immortal (the old objective), then thrown into
              the lethal fight it wasn't built for,
  survival  — built with player_hp set: boss damage counts, the
              screen proxy gains triage, the template offers
              shields/heals,
  surv-p90  — survival build ranked by the slow tail (win, p90, size)
              — where the reliability objective should finally bite.

All three final decks get the same per-deck RL fine-tune UNDER the
lethal sim (no strawman: the speed deck's pilot also trains with
death on the line) and are cross-evaluated on one paired seed stream.
"""
import copy
import json

from data_full import load_spells_full, load_bosses_full, LIVE_RULES
from deck_builder import build_deck, fine_tune
from w101_sim import Sim, evaluate_paired

cards = load_spells_full()
bosses, _ = load_bosses_full()
boss = bosses["Jade Oni"]
PLAYER_HP = 9 * boss.dmg
print(f"matchup: death vs {boss.name} ({boss.hp} HP, resist "
      f"{boss.resist_map}, {boss.dmg} dmg/round) at {PLAYER_HP} player HP",
      flush=True)

builds = {}
speed_boss = copy.copy(boss)
speed_boss.dmg = 0
ARMS = [("speed", speed_boss, None, "mean"),
        ("survival", boss, PLAYER_HP, "mean"),
        ("surv-p90", boss, PLAYER_HP, "p90")]
for name, b, php, obj in ARMS:
    print(f"\n== build: {name} (player_hp={php}, objective={obj}) ==",
          flush=True)
    dl, w, m, _ = build_deck(cards, "death", b, LIVE_RULES,
                             n_candidates=60, top_k=4, seed=5,
                             player_hp=php, objective=obj,
                             log=lambda s: print("  ", s, flush=True))
    builds[name] = dl
    kinds = [cards[n].kind for n in dl]
    print(f"   pick: size {len(dl)}  "
          f"defense {sum(k in ('shield', 'heal') for k in kinds)}  "
          f"{sorted(set(dl))}", flush=True)

print(f"\n== cross-evaluation under fire: RL(8k) + scripted triage "
      f"per deck at {PLAYER_HP} HP, paired seeds n=3000 ==", flush=True)
from w101_sim import make_blade_stack, make_survival
report = {}
for name, dl in builds.items():
    w, m, agent = fine_tune(cards, dl, "death", boss, LIVE_RULES,
                            seed=5, player_hp=PLAYER_HP)
    sim = Sim(dict(cards), dl, "death", boss, player_hp=PLAYER_HP,
              rules=LIVE_RULES)
    # the tabular agent cannot see its own HP, so also report the
    # triage-wrapped scripted pilot — whichever flies the deck better
    st2 = evaluate_paired(sim, {
        "RL(8k)": agent.policy(),
        "triage": make_survival(make_blade_stack(3))}, n=3000)
    report[name] = dict(deck=dl, **{
        pilot: {k: s[k] for k in ("win_rate", "mean_ttk", "median_ttk",
                                  "p90_ttk", "p99_ttk", "mean_hp_left")}
        for pilot, s in st2.items()})
    for pilot, s in st2.items():
        print(f"   {name:<9} [{pilot:<7}] win {s['win_rate']*100:5.1f}%  "
              f"mean {s['mean_ttk']:5.2f}  p90 {s['p90_ttk']:3.0f}  "
              f"hp left {s['mean_hp_left']:5.0f}", flush=True)
    print(f"             {sorted(set(dl))}", flush=True)

json.dump({"boss": boss.name, "boss_hp": boss.hp, "boss_dmg": boss.dmg,
           "player_hp": PLAYER_HP, "school": "death", "picks": report},
          open("results_survival_build.json", "w", encoding="utf-8"),
          indent=1)
print("\nwrote results_survival_build.json")
