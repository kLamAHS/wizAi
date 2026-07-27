"""
Risk-sensitive deck building: the reliability build vs the average
build (roadmap item 5).

Storm under live rules is the variance stress test: exact roll tables
(Wild Bolt's 10/70/1000) and 70%-accuracy nukes make the mean and the
tail of the TTK distribution disagree. Same screen, same candidates,
same per-deck RL fine-tunes — only the FINAL PICK differs:
objective='mean' ranks by (win, mean TTK, size); objective='p90' by
(win, p90 TTK, size), pricing the slow tail instead of the average
fight. Both picks are then cross-evaluated on the same seed stream so
the trade is visible in one table.
"""
import json

from data_full import load_spells_full, LIVE_RULES
from deck_builder import build_deck, fine_tune
from w101_sim import Sim, Boss, evaluate_paired

cards = load_spells_full()
# shallow enough that several candidates saturate win rate — the tie
# the tail metric exists to break; deep fights are usually decided by
# win rate alone and the two objectives agree
boss = Boss("risk-dummy", 1200, "death", 0)
boss.resist_map = {"death": 0.5}
boss.boost_map = {}

decks = {}
for obj in ("mean", "p90"):
    print(f"== build_deck objective={obj} (storm vs {boss.hp} HP) ==",
          flush=True)
    dl, w, m, _ = build_deck(cards, "storm", boss, LIVE_RULES,
                             n_candidates=60, top_k=4, seed=3,
                             objective=obj,
                             log=lambda s: print("  ", s, flush=True))
    decks[obj] = dl
    print(f"   pick: size {len(dl)}  {sorted(set(dl))}\n", flush=True)

print("== cross-evaluation: same seed stream, RL(8k) per deck, "
      "n=3000 ==", flush=True)
report = {}
for obj, dl in decks.items():
    w, m, agent = fine_tune(cards, dl, "storm", boss, LIVE_RULES, seed=3)
    sim = Sim(dict(cards), dl, "storm", boss, player_hp=10**9,
              rules=LIVE_RULES)
    st = evaluate_paired(sim, {"pol": agent.policy()}, n=3000)["pol"]
    report[obj] = dict(deck=dl, **{k: st[k] for k in
                       ("win_rate", "mean_ttk", "median_ttk",
                        "p90_ttk", "p99_ttk")})
    print(f"   {obj:>4}-built: win {st['win_rate']*100:5.1f}%  "
          f"mean {st['mean_ttk']:5.2f}  p90 {st['p90_ttk']:3.0f}  "
          f"p99 {st['p99_ttk']:3.0f}  {sorted(set(dl))}", flush=True)

json.dump({"boss_hp": boss.hp, "school": "storm", "picks": report},
          open("results_risk.json", "w", encoding="utf-8"), indent=1)
print("\nwrote results_risk.json")
