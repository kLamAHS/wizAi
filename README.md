# W101 combat lab: heuristics → exact DP → RL

Turns-to-kill (TTK) study of Wizard101 PvE combat: how much of the blade-stack
meta falls out of the raw card math, and what a learner adds on top.

## Pipeline

| file | role |
|---|---|
| `scrape_central_wiki.py` | (run locally) refresh `cards_clean.json` from the wiki |
| `w101_sim.py` | v0.2 simulator: no-replacement draw, real normal/power pips, **same-name buffs don't stack**, Fuel charges, Feint backlash, self-damage, boss resist/boost |
| `bosses.py` | preset enemy registry (all W101 enemies are scripted → boss stats are known context) + candidate decks per school |
| `dp_solver.py` | value iteration on the deck-free abstraction (distinct-buff configs, ward charge states) = perfect-information lower bound; + policy transfer into the deck sim |
| `rl_agent.py` | tabular Q-learning (backward MC returns, DP warm-start, scarcity-aware state) + UCB deck-selection bandit per boss |
| `experiment.py` | the headline table (`results.json`) |

## Headline results (immortal player = pure speed objective)

```
matchup                      DP-LB       RL (kill%)   best heuristic
fire vs Rattlebones           2.57     2.75 (100%)      2.77
fire vs Krokopatra            5.33     5.57 (100%)      6.89
fire vs Jade Oni              7.33     7.73 (100%)     10.02
fire vs Ervin Flamerender    11.00    12.20 (100%)     13.35
fire vs Malistaire           11.00    11.98 (100%)     13.38
ice  vs Ervin Flamerender     7.50     9.51 (100%)     10.83
ice  vs Prince Gobblestone   10.50    11.47 (100%)     11.58
myth vs Krokopatra            5.25     5.52 (100%)      5.79
```

Ordering everywhere: **DP-LB < RL < DP-transfer / heuristic**, i.e. the agent
learns draw-aware play (dig timing, partial-stack fires, nuke conservation)
that the perfect-information abstraction can't represent — on Jade Oni it
recovers ~75% of the 8.5→7.33 randomness gap.

Notable structure the pipeline surfaced:
- **Non-stacking changes everything.** Buff depth now comes from *diversity*
  (Fireblade + Elemental Blade + Fire Trap + Elemental Trap + Fuel + Feint),
  which is exactly the real one-shot meta. Duplicate copies are only draw
  redundancy.
- **The DP transfer's blind spot is scarcity.** With infinite abstract copies
  it happily wastes buffed hits; on tight decks it kills only 60–70% of runs
  before the deck runs dry. The RL state carries a `nukes_left` feature and
  fixes this to ~100% while also being faster.
- **School matchup is effective HP.** Ervin (3600 fire, 40% self-resist) is
  exactly as hard for a fire wizard as 6000-HP Malistaire (3600/0.6 = 6000);
  the DP bound comes out 11.00 for both.
- **Same-school walls are real.** At Gobblestone's original 3200 HP, no ice
  deck here can kill him at all (total buffed damage < HP through 40% resist)
  — the in-game answer is prisms, which are the next mechanic to model.
- **Deck choice is learnable context.** The UCB bandit (round-robin warmup +
  EMA rewards, because arms improve as they train) converges to the right
  loadout per boss, including switching ice→oneshot only when boosted.

## RL details that mattered

- One-step Q-learning failed outright (2% kill): ~13-step horizon, sparse
  tabular visits. Backward Monte-Carlo returns fixed it in one change.
- Optimistic-init exploration only works after shrinking the state: hand bits
  are kept for damage cards only (buff availability is already encoded in the
  legal-action set).
- α-decay + best-checkpoint selection prevents late-training drift.

## Next

- Fresh wiki scrape → storm/death decks (their mid-tier nukes are missing),
  real boss stats from creature pages, prisms + shields + X-pip spells.
- Survival objective: set `player_hp`/`boss.dmg` real values — Feint's +30%
  backlash and Immolate's self-damage are already modeled and become live
  trade-offs.
- Mob fights: `aoe` flag is preserved in the data.
