# W101 combat lab: heuristics → exact DP → RL

Turns-to-kill (TTK) and survival study of Wizard101 PvE combat: how much of
the blade-stack meta falls out of the raw card math, and what a learner adds
on top. The v0.3 simulator follows the design notes in
`docs/RESEARCH.md`: a **rules engine plus structured symbolic state**, with
effect provenance as a first-class citizen.

## Pipeline

| file | role |
|---|---|
| `scrape_central_wiki.py` | (run locally) refresh `cards_clean.json` from the wiki |
| `scrape_creatures.py` | (run locally) creature pages → `creatures_clean.json` via the MediaWiki API: descriptive UA, ≥1 s throttle, `maxlag`, resumable, `cloudscraper` fallback, and a `--from-xml` mode for browser-downloaded `Special:Export` dumps when Cloudflare blocks scripts. 403 = block, never retried |
| `spells_full.json` / `bosses_clean.json` | full scraped datasets: 18k spell records with game-data effect primitives (typed effects, params, targets, provenance variants) and 1.9k bosses with real stats (health, school, resist/boost, stunable, cheat notes) |
| `data_full.py` | loaders for the full datasets under the **`w101-pve-live-scrape`** ruleset: effect-id map derived by cross-referencing documented spells, provenance from the variant field (`core`/`treasure`/`wand`/`amulet`/`pet` → stack sources), classic-average backfill for range-damage spells the dump doesn't carry (tagged `backfilled-avg`), undecoded effects skipped with reasons — 4.6k usable cards, `LIVE_DECKS` built from game names (the wiki scrape's "Fire Shark"/"Ice Snake" turn out not to exist; "Elemental Blade" is game-named **Tri Blade**) |
| `experiment_full.py` | the live-data table (`results_live.json`): DP-LB, heuristics, DP-transfer, search, RL on real boss stats |
| `content.py` | curated effect primitives (the `spell_effects` layer): DoT splits, multi-hit components, drains, prisms, absorbs, per-pip spells, dispels, summons — each entry confidence-tagged (`community`/`approx`/`inferred`), unsupported cards excluded instead of mis-modeled |
| `w101_sim.py` | v0.3 engine: structured hanging effects with **(name, source) stack keys**, FIFO ward pass with prism school conversion, shields/weaknesses/mantles/dispels/absorbs, scheduled DoTs/HoTs (snapshot at cast), drains, X-pip spells, multi-hit link groups, AoE with per-target resolution, stuns + stun blocks, threat-driven enemy targeting, minions, boss **cheat-script hooks**, treasure-card sideboard, crit/block/pierce machinery, version-tagged `Rules` |
| `bosses.py` | preset enemy registry + illustrative cheat scripts + candidate decks for **all seven schools** |
| `dp_solver.py` | value iteration on the deck-free abstraction (distinct-buff configs, ward charge states, X-pip actions) = perfect-information lower bound; decks that lean on unmodeled mechanics (prisms/heals/shields) are flagged via `meta['unmodeled']` |
| `rl_agent.py` | tabular Q-learning (backward MC returns, DP warm-start, scarcity-aware state incl. drains and prisms) + UCB deck-selection bandit per boss |
| `search_policy.py` | belief-state baseline: determinized rollout search over the hidden draw order (deck composition is known, order isn't — the POMDP move). No training, interpretable, sits between heuristics and RL |
| `experiment.py` | the headline tables (`results.json`): speed (immortal player) + survival (real boss damage) |
| `tests/test_sim.py` | 70 mechanics tests, one per documented combat rule |
| `tests/test_properties.py` | invariants that guard the ML results: blade/pierce monotonicity, deck + ward-charge conservation, seed → identical event log, damage-bound dominance, no-impossible-kill |

## Engine rules worth knowing (v0.3)

- **Provenance stacking.** A deck Fireblade, a treasure-card `Fireblade@tc`
  and a `Sharpened Fireblade` all stack with each other; a second copy of
  the *same* one is an illegal cast, like the in-game X. This is the
  research doc's "if you only run one ablation, run no-effect-provenance"
  representation, implemented at the engine level.
- **FIFO ward processing with a running school.** Traps placed *before* a
  prism boost the hit, then the prism converts it and the *converted*
  school meets resist; traps placed after the prism strand. One strike
  consumes **all** matching charms and wards at once.
- **Link groups.** Fire Elf's 50 + DoT is one strike (a single Tower Shield
  halves both portions); Minotaur's 50-then-445 are separate groups, so the
  50-hit strips shields and blades before the payload lands — the classic
  shield-breaker play.
- **DoTs/HoTs snapshot at cast**; drains ignore heal modifiers; absorbs
  soak ticks; fizzles now **discard the card** but keep pips
  (community-confirmed; the v0.2 retry-forever rule inflated kill rates and
  is available as `Rules(fizzle_discards_card=False)`).
- **Cheats** are per-boss scripted interrupts (`after_player_cast` /
  `round_start` / `hp_below`) executing free effect ops — the "rules engine
  plus cheat scripts" pattern. The bundled scripts are illustrative shapes,
  not scraped encounters.
- **Out of scope, declared:** criticals default off (classic era; the
  machinery exists, is tested, and is pluggable via `Rules.crit_resolver`),
  no archmastery/shadow/school pips, no gear/pet stat layer (fields exist,
  default 0, including flat damage/flat resist), enemy decks are flat
  scripted hits + cheats. Beguile is the one card excluded as unsupported.
- **Auditable + versioned.** `Sim(log_events=True)` records a structured
  event log per duel (`cast_declared`, `charm_consumed`, `ward_consumed`,
  `prism_converted`, `damage_applied`, `dot_created`, `cheat_fired`, ...):
  same seed ⇒ identical log, and every evaluation is stamped with
  `Rules.ruleset_id` (`w101-pve-classic-0.3`) so numbers from different
  rule versions never mix. `max_remaining_damage()` gives an optimistic
  bound on what the remaining deck can still deliver — the scarcity
  feature and the `provably_unwinnable()` classifier in one.

## Headline results (v0.3 rules)

Speed objective (immortal player = pure TTK; RL kill% in parens; heuristic
column is the best "stack k buffs → nuke" that clears 95% kill):

```
matchup                      DP-LB       RL (kill%)   best heuristic
fire  vs Rattlebones          2.57     3.86 ( 99%)      2.78
fire  vs Krokopatra           5.33     6.74 ( 98%)      6.75
fire  vs Jade Oni             7.33     9.03 ( 94%)        —
fire  vs Ervin Flamerender   11.00    12.77 ( 84%)        —
fire  vs Malistaire          11.00    12.70 ( 84%)        —
ice   vs Ervin Flamerender    7.50    11.48 ( 88%)     11.36
ice   vs Prince Gobblestone  10.50*    7.64 ( 94%)      7.01
myth  vs Krokopatra           5.25*   12.55 ( 88%)        —
storm vs Jade Oni             7.03    10.89 ( 81%)        —
death vs Jade Oni             8.35    10.48 ( 93%)        —
balance vs Krokopatra         5.19     6.29 ( 97%)      8.75
life  vs Lord Nightshade      4.17     5.91 (100%)      5.05
```

`*` = the DP abstraction doesn't cover the deck's key mechanic, and it
shows — in both directions:

- **Ice vs Gobblestone: RL beats the "lower bound".** The bound only holds
  inside the blade/trap/nuke abstraction; the prism deck escapes it
  (ice → fire conversion turns 40% resist into a +25% boost). The bandit
  put 29k of its 36k pulls on the prism deck, whose *DP transfer* scores
  0% kills — the learner exploits exactly the mechanic the abstraction
  can't see.
- **Myth vs Krokopatra: the bound is now very loose.** The DP still treats
  Minotaur as one buffed 495 hit; the engine makes its 50-point first hit
  consume all blades and traps (real behavior — the shield-breaker play).
  Myth's classic decks are genuinely weak under the true rule, and v0.2's
  5.52 was an artifact of aggregating multi-hits.

Kill rates below 100% on 70–80%-accuracy schools are the new
**fizzle-discards-card** rule pricing accuracy for real: thin 3-nuke decks
stop being free wins (v0.2 let you retry a fizzled nuke forever).

Survival objective (real boss damage, era-appropriate player HP):

```
matchup (deck)                 HP vs dmg/rd   plain heur   +triage wrap        RL
death vs Jade Oni (oneshot)    3100 vs 300    94% / 11.4    94% / 11.4    93% /  9.6
life  vs Krokopatra (sustain)  3200 vs 220    43% / 11.3    65% / 13.7    63% / 14.8
fire  vs Malistaire (oneshot)  2900 vs 400     0% /   —      0% /   —      0% /   —
```

- **Drains are free sustain**: death's survival kill% matches its immortal
  kill% and the RL agent is *faster* than the immortal-player heuristic
  line because Wraith heals while it nukes.
- **Life buys kill% with tempo**: shield/heal triage lifts 43% → 65% at
  +2.4 turns.
- **Fire vs Malistaire is a lost race without heals** (400/rd kills in ~8
  rounds; the kill needs ~13): the right fix is Fairy treasure cards, and
  the sideboard mechanic is implemented — wiring it into the action space
  is the next experiment.

Other structure the v0.3 pipeline surfaced:

- **The FIFO ward order is learnable structure.** Traps must be laid
  *before* the prism to count pre-conversion; the stack heuristic and the
  RL agent both handle it, and the `prism-before-trap strands the trap`
  case is a regression test.
- **Deck choice is learnable context.** The UCB bandit converges to the
  right loadout on hard bosses (prism into the same-school wall, oneshot
  into big HP pools). On trivial bosses (Rattlebones) arm rewards are
  within noise of each other and the pick wobbles — a known limitation,
  reported as-is.

Search baseline (determinized rollouts, paired seeds, n=400, no training):

```
                              blade-stack(3)      search(k=6)
fire vs Jade Oni [oneshot]    87.2% / 9.72        91.2% / 9.30
ice vs Gobblestone [prism]    73.0% / 9.01        87.0% / 9.68
```

Search closes most of the heuristic→RL gap without any training (RL:
94.3% / 7.64 on the prism matchup after 36k episodes) — the remaining RL
edge is draw-distribution knowledge, not mechanics. Search also trades
mean speed for reliability on the prism line, which is what a
risk-sensitive objective would ask for.

## Charts

Regenerate with `python plots.py` (reads the results JSONs, writes
`plots/*.png`).

![Live-data baseline ladder](plots/live_ladder.png)

![Classic bound vs learned play](plots/classic_gap.png)

![Survival trade-off](plots/survival.png)

![Storm learning curve](plots/storm_curve.png)

## Live-data results (`w101-pve-live-scrape`)

Real scraped spells vs real scraped bosses (`results_live.json`; paired
seeds n=400 for the scripted policies, RL at 20k episodes; not comparable
to the classic tables — different rules, values, and boss stats):

```
matchup                                DP-LB   heuristic   dp-transfer   search(k=5)    RL(20k)
fire vs Lord Nightshade (690)           3.36   98%/ 5.5     82%/ 3.4      100%/ 4.1     98%/ 5.5
fire vs Krokopatra (960, 70% storm-res) 3.36   98%/ 5.5     82%/ 3.4      100%/ 4.2     98%/ 5.2
death vs Jade Oni (6000, 80% life-res)  8.35   77%/11.1     83%/ 9.2       89%/10.4     94%/10.4
storm vs Jade Oni (6000)                8.73    0%/  —      43%/10.0        0%/  —      59%/12.5
balance vs Krokopatra (960)             3.45   97%/ 9.0     98%/ 4.3       96%/ 5.2     99%/ 6.3
ice [prism] vs Krokopatra (960)         4.31*  99%/ 5.5     97%/ 4.7       97%/ 5.0     97%/ 5.9
```

The storm row is the headline: with two Krakens and an X-pip Tempest
against 6,000 HP, **no scripted policy wins at all** — the blade-stack
heuristic can't sequence it, and the determinized search inherits that
blindness because its rollouts use the heuristic as the base policy (all
candidates look equally lost). The DP transfer wins 43% because the
abstraction actually knows the Tempest pip math, and the RL agent reaches
59% by learning X-pip patience on top of draw adaptation. Debugging this
table also caught two real defects (a drain-only-hand stall in the DP
transfer and multi-school buffs invisible to the abstraction) — the
baseline ladder keeps earning its keep.

## Scope of the current claims

This is an **Arc-1-style, single-enemy PvE optimization laboratory**, not
a general Wizard101 combat model. The supported conclusion from the tables
is: *in this simulator and these candidate decks, a scarcity-aware
Monte-Carlo learner improves over transferred perfect-information policies
by adapting to realized draws and conserving finite damage lines* — with
the prism matchup as a live demonstration that representation gaps
(mechanics the abstraction can't see) dominate learner choice. Whatever
"meta" is learned here is the strategy family induced by these decks,
these bosses, the classic ruleset, and the immortal/survival objectives.

## RL details that mattered

- One-step Q-learning failed outright (2% kill): ~13-step horizon, sparse
  tabular visits. Backward Monte-Carlo returns fixed it in one change.
- Optimistic-init exploration only works after shrinking the state: hand
  bits are kept for damage cards only (buff availability is already encoded
  in the legal-action set).
- α-decay + best-checkpoint selection prevents late-training drift.
- The DP warm-start still transfers cleanly to v0.3 because the abstraction
  (blades/traps/nukes, now + X-pip) is a strict subspace of the engine.

## Data honesty

Per the research notes: unknowns are never silently zeroed. Every curated
mechanic in `content.py` is confidence-tagged; minion stat blocks are
`inferred` placeholders; `stunable=None` means *unspecified*, not false;
cards the engine can't express are excluded at load and listed in the
loader report.

## Roadmap (in dependency order)

1. **Done in v0.3.x**: ruleset versioning, structured event log, exact
   stacking identities, accuracy/shields/pierce/flat stats, ordered prisms,
   DoT/multi-hit/drains/X-pip, cheat scripts with cooldowns and one-shot
   thresholds, property-test suite, determinized search baseline.
2. Scrape real creature pages (stats, stunable flags, actual cheat
   scripts) to replace the ballpark boss registry; damage *ranges* instead
   of averages.
3. Sideboard/discard policy learning: the TC mechanic (random draw, no
   same-round discard) is implemented; wire it into the action space —
   fire vs Malistaire survival is winnable only through it.
4. Mob fights: the engine is multi-enemy (AoE, per-target wards, threat)
   but the experiment table is still 1v1.
5. Risk-sensitive objectives (reliability / percentile-TTK scalarizations
   — `evaluate_paired` already reports the distribution) and
   `max_remaining_damage` as an RL feature.
6. Search-generated expert data → filtered behavior cloning → conservative
   offline RL (CQL/IQL) → sequence models, benchmarked against each other
   on held-out bosses/cards/rulesets.
7. Deck × policy bilevel optimization (the bandit is the placeholder).
8. Post-classic rulesets behind `Rules`: criticals-on eras, mastery
   amulets, school pips/archmastery — each as a frozen named ruleset.

## Boundary

Fully offline simulator and research benchmark, built from public wiki
data and community documentation. No live-client control, memory reading,
traffic interception, or gameplay automation — see `docs/RESEARCH.md`.
