"""
Headline experiment: for each (wizard school, preset boss), the agent
  1. picks a deck via UCB bandit (context = the boss is known in advance),
  2. learns the in-fight policy via Q-learning warm-started from the DP,
and is compared against:
  - DP-LB: value-iteration optimum of the deck-free abstraction (lower bound)
  - DP->sim: that policy transferred into the real deck sim (scarcity-blind)
  - heuristic: best "stack k distinct buffs then nuke" strategy, best deck
Immortal player (boss dmg zeroed) => pure turns-to-kill objective, matching
the DP bound. Survival objective is a one-line change (player_hp/boss.dmg).
"""
import time, json
from w101_sim import load_cards, Boss, Sim, evaluate, strat_nuke_asap, make_blade_stack
from bosses import BOSSES, DECKS
from dp_solver import solve, dp_policy
from rl_agent import DeckBandit, FAIL_PENALTY

cards = load_cards("cards_clean.json")

MATCHES = [
    ("fire", "Rattlebones"),        # 500 HP — speed should beat greed
    ("fire", "Krokopatra"),         # 2200, storm (neutral to fire)
    ("fire", "Jade Oni"),           # 4000, life (neutral)
    ("fire", "Ervin Flamerender"),  # 3600, FIRE — 40% resist vs own school
    ("fire", "Malistaire"),         # 6000 — full stacks mandatory
    ("ice",  "Ervin Flamerender"),  # fire boss — ice boosted +25%
    ("ice",  "Prince Gobblestone"), # ice boss  — ice resisted 40%
    ("myth", "Krokopatra"),         # storm boss — myth boosted +25%
]

EPISODES = 36000
results = []

for school, bname in MATCHES:
    b = BOSSES[bname]
    boss = Boss(b.name, b.hp, b.school, 0)          # immortal-player view
    decks = DECKS[school]
    print(f"\n=== {school} wizard vs {bname} ({b.hp} HP, {b.school}) ===")

    # baselines + DP bounds per deck
    lb, transfer, heur = {}, {}, {}
    dp_pols = {}
    for dname, dl in decks.items():
        V, pol, meta = solve(cards, dl, boss, school)
        lb[dname] = V[meta["H"], 1, 0]
        dp_pols[dname] = dp_policy(V, pol, meta, school)
        sim = Sim(cards, dl, school, boss, player_hp=10**9)
        transfer[dname] = evaluate(sim, dp_pols[dname], n=4000)
        heur[dname] = max(
            [evaluate(sim, strat_nuke_asap, n=2500)] +
            [evaluate(sim, make_blade_stack(k), n=2500) for k in (1, 2, 3, 4, 5)],
            key=lambda r: (round(r[0], 2), -r[1]))

    # bandit + RL (reuse solves via prebuilt dp policies)
    t0 = time.time()
    bandit = DeckBandit.__new__(DeckBandit)
    from rl_agent import QAgent
    bandit.arms = {n: (QAgent(cards, dl, school, dp_pol=dp_pols[n]),
                       Sim(cards, dl, school, boss, player_hp=10**9))
                   for n, dl in decks.items()}
    bandit.stats = {n: [0, 0.0] for n in decks}
    bandit.t = 0
    bandit.train(EPISODES)
    pick, agent, sim = bandit.best()
    w_rl, m_rl = evaluate(sim, agent.policy(), n=8000)
    dt = time.time() - t0

    pulls = {n: bandit.stats[n][0] for n in decks}
    print(f"  bandit pulls: {pulls}  -> picked '{pick}'  ({dt:.0f}s)")
    print(f"  {'deck':<9}{'DP-LB':>7}{'DP->sim':>16}{'heuristic':>16}")
    for dname in decks:
        tw, tm = transfer[dname]
        hw, hm = heur[dname]
        print(f"  {dname:<9}{lb[dname]:>7.2f}"
              f"{tm:>10.2f} ({tw*100:3.0f}%){hm:>10.2f} ({hw*100:3.0f}%)")
    print(f"  RL [{pick}]: kill {w_rl*100:.1f}%  TTK {m_rl:.2f}")

    best_h = min((heur[d] for d in decks if heur[d][0] > 0.95),
                 key=lambda r: r[1], default=(0, float("nan")))
    results.append(dict(school=school, boss=bname, pick=pick, pulls=pulls,
                        dp_lb=min(lb.values()), rl=[w_rl, m_rl],
                        transfer={d: list(transfer[d]) for d in decks},
                        heuristic_best=list(best_h),
                        lb={d: lb[d] for d in decks}))

json.dump(results, open("results.json", "w"), indent=1)
print("\nwrote results.json")

print(f"\n{'matchup':<28}{'DP-LB':>7}{'RL':>14}{'best heur':>12}")
for r in results:
    w, m = r["rl"]
    hw, hm = r["heuristic_best"]
    print(f"{r['school']+' vs '+r['boss']:<28}{r['dp_lb']:>7.2f}"
          f"{m:>8.2f} ({w*100:3.0f}%){hm:>12.2f}")
