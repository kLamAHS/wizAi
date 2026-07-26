"""
Headline experiment: for each (wizard school, preset boss), the agent
  1. picks a deck via UCB bandit (context = the boss is known in advance),
  2. learns the in-fight policy via Q-learning warm-started from the DP,
and is compared against:
  - DP-LB: value-iteration optimum of the deck-free abstraction (lower bound
    for decks fully inside the blade/trap/nuke abstraction; decks that lean
    on prisms/heals/shields report meta['unmodeled'] and the bound weakens)
  - DP->sim: that policy transferred into the real deck sim (scarcity-blind)
  - heuristic: best "stack k distinct buffs then nuke" strategy, best deck

Two tables:
  - speed (immortal player, boss dmg zeroed): pure turns-to-kill, matches
    the DP bound's objective
  - survival (real boss damage, era-appropriate player HP): drains, shields
    and heals become live trade-offs
"""
import time, json
from w101_sim import (load_cards, Boss, Sim, evaluate, strat_nuke_asap,
                      make_blade_stack, make_survival)
from bosses import BOSSES, DECKS, PLAYER_HP
from dp_solver import solve, dp_policy
from rl_agent import DeckBandit, QAgent, train_agent, FAIL_PENALTY

cards = load_cards("cards_clean.json")

MATCHES = [
    ("fire", "Rattlebones"),        # 500 HP — speed should beat greed
    ("fire", "Krokopatra"),         # 2200, storm (neutral to fire)
    ("fire", "Jade Oni"),           # 4000, life (neutral)
    ("fire", "Ervin Flamerender"),  # 3600, FIRE — 40% resist vs own school
    ("fire", "Malistaire"),         # 6000 — full stacks mandatory
    ("ice",  "Ervin Flamerender"),  # fire boss — ice boosted +25%
    ("ice",  "Prince Gobblestone"), # ice boss — resist wall; prism deck arm
    ("myth", "Krokopatra"),         # storm boss — myth boosted +25%
    ("storm", "Jade Oni"),          # Kraken + X-pip Tempest
    ("death", "Jade Oni"),          # drains; death boosted +25% vs life
    ("balance", "Krokopatra"),      # Judgement: X-pip patience problem
    ("life", "Lord Nightshade"),    # life boosted +25% vs death
]

EPISODES = 36000
speed_results = []

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
        star = "*" if meta["unmodeled"] else ""
        lb[dname] = (V[meta["H"], 1, 0], star)
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
    print(f"  {'deck':<9}{'DP-LB':>8}{'DP->sim':>16}{'heuristic':>16}")
    for dname in decks:
        tw, tm = transfer[dname]
        hw, hm = heur[dname]
        v, star = lb[dname]
        print(f"  {dname:<9}{v:>7.2f}{star:<1}"
              f"{tm:>10.2f} ({tw*100:3.0f}%){hm:>10.2f} ({hw*100:3.0f}%)")
    print(f"  RL [{pick}]: kill {w_rl*100:.1f}%  TTK {m_rl:.2f}")

    best_h = min((heur[d] for d in decks if heur[d][0] > 0.95),
                 key=lambda r: r[1], default=(0, float("nan")))
    speed_results.append(dict(
        school=school, boss=bname, pick=pick, pulls=pulls,
        dp_lb=min(v for v, _ in lb.values()), rl=[w_rl, m_rl],
        transfer={d: list(transfer[d]) for d in decks},
        heuristic_best=list(best_h),
        lb={d: lb[d][0] for d in decks},
        lb_partial={d: bool(lb[d][1]) for d in decks}))

# ---------------------------------------------------------------- survival
# Real boss damage + era-appropriate player HP: kills stop being free.
SURVIVAL = [
    ("death", "Jade Oni", "oneshot"),     # drains double as sustain
    ("life", "Krokopatra", "sustain"),    # heals + Tower Shields
    ("fire", "Malistaire", "oneshot"),    # no heals: pure race
]
survival_results = []
print("\n=== survival objective (real boss damage) ===")
for school, bname, dname in SURVIVAL:
    b = BOSSES[bname]
    hp = PLAYER_HP[school]
    dl = DECKS[school][dname]
    sim = Sim(cards, dl, school, b, player_hp=hp)
    plain = max([evaluate(sim, make_blade_stack(k), n=4000)
                 for k in (0, 2, 4)], key=lambda r: (round(r[0], 2), -r[1]))
    wrapped = max([evaluate(sim, make_survival(make_blade_stack(k)), n=4000)
                   for k in (0, 2, 4)], key=lambda r: (round(r[0], 2), -r[1]))
    t0 = time.time()
    agent, rsim = train_agent(cards, dl, school, b, episodes=24000,
                              warm=True, seed=0, player_hp=hp)
    w_rl, m_rl = evaluate(rsim, agent.policy(), n=8000)
    print(f"  {school} vs {bname} [{dname}] (HP {hp} vs {b.dmg}/rd): "
          f"heur {plain[0]*100:.0f}%/{plain[1]:.2f}  "
          f"survival-heur {wrapped[0]*100:.0f}%/{wrapped[1]:.2f}  "
          f"RL {w_rl*100:.0f}%/{m_rl:.2f}  ({time.time()-t0:.0f}s)")
    survival_results.append(dict(
        school=school, boss=bname, deck=dname, player_hp=hp, boss_dmg=b.dmg,
        heuristic=list(plain), survival_heuristic=list(wrapped),
        rl=[w_rl, m_rl]))

from w101_sim import Rules
json.dump({"ruleset_id": Rules().ruleset_id,
           "speed_immortal": speed_results, "survival": survival_results},
          open("results.json", "w", encoding="utf-8"), indent=1)
print("\nwrote results.json")

print(f"\n{'matchup':<28}{'DP-LB':>7}{'RL':>14}{'best heur':>12}")
for r in speed_results:
    w, m = r["rl"]
    hw, hm = r["heuristic_best"]
    print(f"{r['school']+' vs '+r['boss']:<28}{r['dp_lb']:>7.2f}"
          f"{m:>8.2f} ({w*100:3.0f}%){hm:>12.2f}")
