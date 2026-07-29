"""
Hybrid search on the hard matchup: determinized rollout search whose BASE
policy is the trained RL agent instead of the blade-stack heuristic.

Storm vs Jade Oni is the motivating case: scripted search scores 0%
because its heuristic rollouts can never win, so every candidate action
looks equally lost. Rolling out with the learned policy restores the
gradient — search then adds draw-averaged lookahead on top of learning.

Writes results_hybrid.json and prints the ladder. Also records the RL
learning curve for plotting (rl_curve_storm.json).
"""
import json
import random

from w101_sim import Sim, evaluate, evaluate_paired, make_blade_stack
from data_full import (load_spells_full, load_bosses_full, LIVE_DECKS,
                       LIVE_RULES)
from rl_agent import QAgent
from search_policy import make_search_policy

cards = load_spells_full()
bosses, _ = load_bosses_full()

import copy
boss = copy.copy(bosses["Jade Oni"])
boss.dmg = 0
dl = LIVE_DECKS["storm"]["oneshot"]

# train RL, keeping a learning curve (train_agent doesn't expose history)
rng = random.Random(0)
sim = Sim(dict(cards), dl, "storm", boss, player_hp=10**9,
          rules=LIVE_RULES, rng=rng)
agent = QAgent(dict(cards), dl, "storm", rng=random.Random(1))
curve = []
EPISODES, SNAP = 20000, 1000
best = (-1.0, float("inf"), None)
for ep in range(EPISODES):
    frac = ep / EPISODES
    agent.alpha = 0.30 * (1 - 0.9 * frac)
    agent.train_episode(sim, eps=max(0.02, 0.3 * (1 - frac)), dp_w=0.0)
    if (ep + 1) % SNAP == 0:
        w, m = evaluate(sim, agent.policy(), n=1500)
        curve.append(dict(episode=ep + 1, win=w, ttk=m if w else None))
        print(f"  ep {ep+1:>6}: win {w*100:5.1f}%  ttk {m:6.2f}")
        if (w, -m) > (best[0], -best[1]):
            best = (w, m, dict(agent.Q))
if best[2] is not None:
    from collections import defaultdict
    agent.Q = defaultdict(float, best[2])
json.dump(curve, open("rl_curve_storm.json", "w", encoding="utf-8"), indent=1)

rl_pol_agent = agent.policy()

def rl_base(sim_, s):
    return rl_pol_agent(sim_, s)

stats = evaluate_paired(sim, {
    "heuristic": make_blade_stack(3),
    "search(heur-base)": make_search_policy(k=5),
    "RL(20k)": rl_pol_agent,
    "search(RL-base)": make_search_policy(base=rl_base, k=5),
}, n=250)

print(f"\nstorm vs Jade Oni ({boss.hp} HP) — paired seeds, n=250")
for name, st in stats.items():
    print(f"  {name:<18} win {st['win_rate']*100:5.1f}%  "
          f"mean {st['mean_ttk']:6.2f}  p90 {st['p90_ttk']:3.0f}")

json.dump({"ruleset_id": LIVE_RULES.ruleset_id,
           "matchup": "storm vs Jade Oni", "paired": stats},
          open("results_hybrid.json", "w", encoding="utf-8"), indent=1)
print("wrote results_hybrid.json")
