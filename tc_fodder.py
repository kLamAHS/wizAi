"""
The fodder tax: do treasure cards pay for their own logistics?

The playstyle claim under test: using TCs forces you to carry a
bigger deck (discard fodder — the hand refills from the deck every
round, so drawing a TC means throwing a real card away), and the
bigger deck dilutes your draws; worse, the fodder you need may not be
in hand when you want to draw. The sim models every part of this:
discards are conserved (the card is gone), the TC reflex must rank
what to throw, and deck size shapes the draw distribution.

Four arms on one fight (death vs live Jade Oni, immortal speed
objective — pure tempo economics, no survival confound). The lean
deck is the searched speed build from results_survival_build.json;
the padded deck adds 4x Dark Sprite fodder; the sideboard is 3x
Wraith TC (extra gas — offense TCs, the best case for TCs per every
result so far). RL(8k) pilots per arm, paired seeds, TC casts
audited.

  lean          — the optimized build, no TCs (baseline)
  lean+tc       — TC draws must eat the lean kill line (fodder tax
                  on a deck with no fodder)
  padded        — fodder carried, no TCs (the dilution tax alone)
  padded+tc     — the full TC playstyle (does access repay dilution?)
"""
import json

from data_full import load_spells_full, load_bosses_full, LIVE_RULES
from deck_builder import fine_tune
from w101_sim import Sim, evaluate_paired

cards = load_spells_full()
bosses, _ = load_bosses_full()
import copy
boss = copy.copy(bosses["Jade Oni"])
boss.dmg = 0                                # immortal: tempo only

lean = json.load(open("results_survival_build.json", encoding="utf-8")
                 )["matchups"]["chip"]["picks"]["speed"]["deck"]
FODDER = ["Dark Sprite"] * 4
SIDE = ["Wraith@tc"] * 3

ARMS = {
    "lean":      (lean,          None),
    "lean+tc":   (lean,          SIDE),
    "padded":    (lean + FODDER, None),
    "padded+tc": (lean + FODDER, SIDE),
}
print(f"death vs {boss.name} ({boss.hp} HP, immortal player) — lean "
      f"deck size {len(lean)}, fodder +{len(FODDER)}, sideboard "
      f"{len(SIDE)}x Wraith TC", flush=True)

report = {}
for name, (dl, side) in ARMS.items():
    w, m, agent = fine_tune(cards, dl, "death", boss, LIVE_RULES, seed=9,
                            sideboard=side)
    sim = Sim(dict(cards), dl, "death", boss, player_hp=10**9,
              rules=LIVE_RULES, sideboard=side)
    st = evaluate_paired(sim, {"pol": agent.policy()}, n=3000)["pol"]
    report[name] = dict(deck_size=len(dl), sideboard=side or [],
                        **{k: st[k] for k in
                           ("win_rate", "mean_ttk", "p90_ttk",
                            "mean_tc_casts")})
    print(f"   {name:<10} deck {len(dl):>2}  win {st['win_rate']*100:5.1f}%"
          f"  mean {st['mean_ttk']:5.2f}  p90 {st['p90_ttk']:3.0f}  "
          f"tc/fight {st['mean_tc_casts']:.2f}", flush=True)

json.dump({"boss": boss.name, "lean_deck": lean, "arms": report},
          open("results_tc_fodder.json", "w", encoding="utf-8"), indent=1)
print("\nwrote results_tc_fodder.json")
