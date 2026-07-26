"""
Live-data experiment: real scraped spells (spells_full.json) against real
scraped bosses (bosses_clean.json), under the 'w101-pve-live-scrape'
ruleset. Same ladder as the classic table — DP lower bound, heuristics,
DP-transfer, RL — plus the determinized search baseline via paired seeds.

Numbers here are NOT comparable to the classic tables: card values are
current-era (Fuel +40x3, Sprite 280/4), boss health/resists are scraped,
and boss per-round damage is rank-inferred (dmg_confidence='inferred').
"""
import json
import time

from w101_sim import (Sim, evaluate, evaluate_paired, strat_nuke_asap,
                      make_blade_stack)
from data_full import (load_spells_full, load_bosses_full, LIVE_DECKS,
                       LIVE_RULES)
from dp_solver import solve, dp_policy
from rl_agent import train_agent
from search_policy import make_search_policy

report = {}
cards = load_spells_full(report=report)
bosses, registry = load_bosses_full()
print(f"live data: {len(cards)} cards ({len(report['skipped'])} skipped "
      f"honestly), {len(bosses)} bosses")

MATCHES = [
    ("fire", "Lord Nightshade", "oneshot"),   # 690 HP death
    ("fire", "Krokopatra", "oneshot"),        # 960 HP storm
    ("death", "Jade Oni", "oneshot"),         # 6000 HP life, 80% life resist
    ("storm", "Jade Oni", "oneshot"),
    ("balance", "Krokopatra", "oneshot"),
    ("ice", "Krokopatra", "prism"),           # storm boss; prism arm anyway
]

results = []
for school, bname, dname in MATCHES:
    boss = bosses[bname]
    import copy as _copy
    speed_boss = _copy.copy(boss)          # immortal-player view
    speed_boss.dmg = 0
    dl = LIVE_DECKS[school][dname]
    print(f"\n=== {school} [{dname}] vs {bname} "
          f"({boss.hp} HP, {boss.school}, resist {boss.resist_map}) ===")
    t0 = time.time()
    V, pol, meta = solve(dict(cards), dl, speed_boss, school)
    lb = V[meta["H"], 1, 0]
    star = "*" if meta["unmodeled"] else ""
    sim = Sim(dict(cards), dl, school, speed_boss, player_hp=10**9,
              rules=LIVE_RULES)
    stats = evaluate_paired(sim, {
        "heuristic": max((make_blade_stack(k) for k in (1, 2, 3, 4)),
                         key=lambda p: evaluate(sim, p, n=800)[0]),
        "dp-transfer": dp_policy(V, pol, meta, school),
        "search(k=5)": make_search_policy(k=5),
    }, n=400)
    agent, rsim = train_agent(dict(cards), dl, school, speed_boss,
                              episodes=20000, warm=True, seed=0)
    w_rl, m_rl = evaluate(rsim, agent.policy(), n=4000)
    print(f"  DP-LB {lb:6.2f}{star}   ({time.time()-t0:.0f}s)")
    for name, st in stats.items():
        print(f"  {name:<12} win {st['win_rate']*100:5.1f}%  "
              f"mean {st['mean_ttk']:6.2f}  p90 {st['p90_ttk']:3.0f}")
    print(f"  RL(20k)      win {w_rl*100:5.1f}%  mean {m_rl:6.2f}")
    results.append(dict(school=school, deck=dname, boss=bname,
                        boss_hp=boss.hp, dp_lb=lb, dp_lb_partial=bool(star),
                        paired=stats, rl=[w_rl, m_rl]))

json.dump({"ruleset_id": LIVE_RULES.ruleset_id, "matches": results},
          open("results_live.json", "w"), indent=1)
print("\nwrote results_live.json")
