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
    hp_buckets = -(-speed_boss.hp // 25)
    print(f"  [1/4] DP value iteration ({hp_buckets} hp buckets"
          f"{', X-pip actions — the slow one' if any(cards[n].x_pips for n in dl) else ''})...",
          flush=True)
    V, pol, meta = solve(dict(cards), dl, speed_boss, school)
    lb = V[meta["H"], 1, 0]
    star = "*" if meta["unmodeled"] else ""
    print(f"        DP-LB {lb:.2f}{star}  ({time.time()-t0:.0f}s)"
          + (f"  unmodeled: {meta['unmodeled']}" if star else ""), flush=True)
    sim = Sim(dict(cards), dl, school, speed_boss, player_hp=10**9,
              rules=LIVE_RULES)
    print("  [2/4] picking best blade-stack heuristic (4 x 800 fights)...",
          flush=True)
    heur = max((make_blade_stack(k) for k in (1, 2, 3, 4)),
               key=lambda p: evaluate(sim, p, n=800)[0])
    policies = {"heuristic": heur,
                "dp-transfer": dp_policy(V, pol, meta, school),
                "search(k=5)": make_search_policy(k=5)}
    stats = {}
    print("  [3/4] paired evaluation, 400 seeded fights per policy:",
          flush=True)
    for name, p in policies.items():
        tp = time.time()
        stats[name] = evaluate_paired(sim, {name: p}, n=400)[name]
        st = stats[name]
        print(f"        {name:<12} win {st['win_rate']*100:5.1f}%  "
              f"mean {st['mean_ttk']:6.2f}  p90 {st['p90_ttk']:3.0f}  "
              f"({time.time()-tp:.0f}s)", flush=True)
    print("  [4/4] RL: 20k training episodes (snapshot every 4k):",
          flush=True)
    agent, rsim = train_agent(dict(cards), dl, school, speed_boss,
                              episodes=20000, warm=True, seed=0,
                              log=4000, snap_every=4000)
    w_rl, m_rl = evaluate(rsim, agent.policy(), n=4000)
    print(f"        RL(20k)      win {w_rl*100:5.1f}%  mean {m_rl:6.2f}  "
          f"(match total {time.time()-t0:.0f}s)", flush=True)
    results.append(dict(school=school, deck=dname, boss=bname,
                        boss_hp=boss.hp, dp_lb=lb, dp_lb_partial=bool(star),
                        paired=stats, rl=[w_rl, m_rl]))

json.dump({"ruleset_id": LIVE_RULES.ruleset_id, "matches": results},
          open("results_live.json", "w", encoding="utf-8"), indent=1)
print("\nwrote results_live.json")
