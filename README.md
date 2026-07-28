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

![Progression sweep](plots/progression.png)

![Deck scorer validation](plots/scorer.png)

![Deck-conditioned generalist](plots/generalist.png)

![Survival builds under fire](plots/survival_builds.png)

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

*(Table shows the first live run; `results_live.json` + `plots/` are
regenerated per data drop and are canonical. Under exact roll tables the
notable shift: damage variance costs the scarcity-blind DP transfer ~10
points on the fire matchups — 82% -> 70-74% — while heuristic/search/RL
barely move.)*

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

**Hybrid search — an honest negative result** (`hybrid_search.py`,
`results_hybrid.json`, paired seeds n=250 on storm vs Jade Oni): swapping
the search's rollout base from the heuristic to the trained RL policy
restores the gradient exactly as predicted (0% → 56%), but the hybrid
**does not beat plain RL** (67.2%). One-ply argmax over k=5 noisy rollouts
adds enough variance on a ~13-turn horizon to override good learned
decisions. The fix directions are classic: more rollouts per candidate,
or search over the RL agent's value estimates instead of raw returns —
logged in the roadmap rather than pretended away. Under live rules,
damage RANGES are now sampled too (`Rules.damage_ranges`; parsed from the
classic descriptions), so win rates price in damage variance, not just
fizzle variance.

## Living bosses (2026 boss-AI report)

The July 2026 research report grounds two new layers, both
confidence-tagged in `docs/RESEARCH.md`:

**Player base curves** (`player_curves.py`): school HP from the
official-forum L1/L120 anchors (linear interpolation, documented as
approximation, clamped past 120 — nothing fabricated beyond the
anchors) and the universal base power-pip ramp (0% before 10, 40% cap
at 50). The progression sweep now runs on the era pip curve — a
level-1 wizard gets zero power pips, and the level-15 trough deepens
honestly (TTK 9.7 vs the geared-pip 6.4).

**The living-boss caster** (`Boss(pool=..., archetype=...,
discipline=...)`): the report's three-layer model — configured
reusable spell pool + state-aware-but-imperfect legal-action AI +
deterministic scripts (our existing `CheatRule` layer IS layer 3) —
with enemy pips (7-slot rack, white→power upgrade when full), role
archetypes (hitter/healer/buffer/debuffer/tank), duplicate-hanging
checks, saving/passing as real actions, and fizzles. Boss casts route
through the same charm/ward/crit engine as player casts, so mantles
and dispels bite the boss. Legacy flat-damage bosses are untouched
(`pool=None`, byte-identical). Everything beyond the report's
evidence (exact weights, enemy pip odds) is tagged `modeled`.

**Solo-feasibility frontier** (`living_bosses.py`,
`results_living.json`; base-stat death wizard, no gear/wand, vs
Krokopatra under each model): at level 12 both models say no (719 HP
soloing 4-person content should fail); at level 20 the flat model
calls the solo trivial (**98.6%**) while the living boss still wins
most fights (**33.4%** best pilot — heal-aware triage on a
Pixie-carrying deck) — chip damage understates a hitter that blades
into Storm Shark spikes and shields your kill turns.
Two riders. First, the stochastic opponent INVERTS the baseline
ladder (`pilot_ladder_at_20` in the results): triage 33.4% >
search(k=5) 24.1% > blade-stack(3) 20.1% > per-deck RL 3.8–7.9%.
Diagnosed, not assumed — the blade-blindness hypothesis was tested
and falsified (enemy blade/shield state was ADDED to the tabular
featurizer and made 8k-episode RL worse by fragmenting visits; 24k +
a scripted-advisor warm start still trails every scripted pilot).
The real cause is sample starvation: opponent stochasticity
(discipline, enemy fizzles, spell choice) widens the visited-state
distribution and drowns sparse win credit, exactly where tabular MC
and shallow determinized rollouts are weakest — the concrete
motivation for the sequence-model rung. The enemy-state features and
the `fine_tune(advisor=...)` warm start both ship (correct
representation and a 2x improvement at 24k, honestly short of the
prior). Second, one 300-HP healer acolyte (which really heals its
boss — enemy heals route to the neediest teammate) drops the fight
to **0%** without target switching: the report's role-segmentation
pattern, quantified.

This diff was adversarially reviewed by a 30-agent workflow before
merge; it caught a boss pip-livelock, self-only enemy healing, X-pip
cost inversion, and boss immunity to player mantles — all fixed and
regression-tested (156 tests).

## Mob fights: target switching (roadmap 4, opening move)

`with_focus(policy)` adds the report's team-fight rule — support
enemies (healer/buffer/debuffer archetypes) die first, then the
lowest-HP attacker — on top of the engine's per-target casting, and
`build_deck(enemies=...)` builds against a full encounter with
focus-wrapped screen proxies. The healer-cliff fight decomposes into
three separable constraints (`mob_fights.py`, `results_mob.json`;
level-20 base-stat death wizard vs living Krokopatra + 300-HP healer
acolyte):

```
                              mortal      immortal (tempo view)
boss alone, triage             38.1%        72.1%
+healer, target-blind           0.0%         0.0%   <- the cliff
+healer, focus, solo deck       0.0%         0.0%   <- out of ammo
+healer, focus, ammo deck       0.1%        46.4%   <- both needed
+healer, BLIND, ammo deck        —           0.0%   <- focus necessary
+2 healers, focus, ammo         0.0%         2.1%   <- sustain scales
```

Three lessons. TARGETING is necessary but not sufficient: with
identical ammunition, blind play stays at 0% while focus reaches
46.4%. AMMUNITION binds next: the solo-built 10-card deck (4 hits)
runs dry at boss=388 even after a perfect healer kill — mob fights
re-price deck size, and the builder's size penalty is exactly wrong
for them. And the MORTAL verdict is game-accurate: at base stats a
boss+minion encounter is multi-player content (the report:
enemy count = players + 1) — no targeting rule rescues a solo
level-20 at 915 HP. A pipeline honesty fix rode along: when every
screen candidate scores 0% (infeasible encounter), the ranking is
noise and the size tiebreak silently favors SMALL decks — build_deck
now warns instead of pretending.
**Learned targeting: a clean negative result**
(`mob_generalist.py`, `results_mob_generalist.json`). The generalist
grew a target dimension — (card, target) actions with per-target
overkill/kill-now/support-flag features, mob episodes mixed into
training, 1v1 behavior bit-compatible, old policy files zero-padded
on load — and it FAILS the scripted bar: 0% vs focus(bs2)'s 46.4%,
opening on the boss instead of the healer. A four-arm controlled
study then separated exploration failure from representation failure
(self-play, advisor-guided exploration that follows the focus script
early, BC on random-distribution demonstrations, and BC on
demonstrations of THE BAR FIGHT ITSELF — no distribution-shift
excuse): every learned arm scores 0%. Along the way: two credit
fixes tried (raw backward-MC lets mob-only features absorb the
mob episodes' difficulty as negative weight — the feature-level
cousin of "losers teach losing habits"; an advantage baseline with a
mob-aware state head absorbs it properly, still 0%), and one
matcher bug caught (focus wraps every card in a target tuple, so
teacher blade casts logged as PASSes until normalized — demo data
had deleted the very line it demonstrated). The decisive number came
from the on-bar cell: the TEACHER itself collapses from 46.4% to 1%
under 10% action noise — across a ~25-decision grind the winning
line tolerates almost no deviation, so a memoryless linear ranking
that wobbles anywhere loses everywhere. Three independent negatives
(X-pip two-hit planning, healer-first commitment, wobble-free
sequence execution) now isolate the same missing capability:
sequence-level planning and consistency — the sequence-model rung,
with reproducible bars at 46.4% (this fight) and 85% (balance
X-pip).
**The first bar falls to decision-time search**
(`sequence_search.py`, `results_sequence.json`). Target-aware
determinized rollout search with the focus script as rollout base
scores **63.0%** on the healer fight — clearing the 46.4% teacher
bar by 16 points AND faster (mean 17.5 vs 20.4) — because search
PLANS the remaining sequence per draw instead of imitating one: it
deviates from the script exactly where rollouts prove it profitable,
so it is not capped at its teacher the way BC is. On the X-pip bar,
search(k=16) reaches **72.0%** vs blade-stack's 60.0% — half the gap
to per-deck RL's 85.1% closed with zero training; the residual is
deck-specific draw memorization a one-ply rollout can't capture,
which keeps the learned-sequence-model rung motivated for what
remains. Cost profile is the mirror of RL's: no training, ~1s of
inference per fight.

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
3. Sideboard/discard policy: DONE — every policy now has the TC
   reflex (`tc_reflex`: make room honoring the fresh-TC rule, draw
   one, castable same round), the tabular agent carries TC names AND
   own HP in its state, and the generalist values drawn TCs by
   properties zero-shot (`tc_experiment.py`, `results_tc.json`).
   The motivating claim — "fire vs Malistaire survival is winnable
   only through TCs" — is REFUTED by arithmetic: the immortal DP
   bound is 11 turns vs death during round 8; off-school Satyr costs
   ~4 rounds of pip income (sustain eats the kill budget); zero-pip
   Death Shields trade 320 damage per TEMPO turn and
   400L−320S ≤ 2900 has no solution alongside the ~7 damage-line
   turns 6000 HP requires. Even on a marginal WINNABLE control fight
   (2800 HP, 500/round), the RL pilot's best use of a shield
   sideboard is to mostly ignore it (70.9% vs 71.0% bare) and TC
   heals cost 8 points — offense dominance again, and another pilot-
   alignment lesson: scripted triage + shields = shield-lock (0%).
   The TC payoff regime (unraceable burst one-shots, mob fights,
   bigger HP pools) needs items 4+ below.
   **Design rule — TCs are never a crutch**: treasure cards cost real
   gold, and most fights are beatable without them, so TC use is
   OPT-IN everywhere: no default training run, deck search, or
   benchmark attaches a sideboard, and every `evaluate_paired` row
   now reports `mean_tc_casts` so any policy leaning on TCs shows it
   in the table. If a sideboard search is ever added, TC casts must
   carry an explicit cost term in the objective — an unpriced TC is
   a free lunch the optimizer would hack.
   **The fodder tax, measured** (`tc_fodder.py`,
   `results_tc_fodder.json` — death vs Jade Oni, immortal tempo, RL
   pilots trained per arm): lean 11-card deck 99.4%/9.72; the same
   deck actually USING a 3x Wraith TC sideboard drops to 92.0%/12.55
   (drawing a TC forcibly discards a real card — on a lean deck the
   fodder IS the kill line); padding the deck with 4 fodder cards
   costs 18 points by dilution alone (81.3%/12.64); and the full TC
   playstyle — fodder carried AND sideboard used — is the worst of
   all four arms at 61.6%/18.46. TC access never repays its own
   logistics here; the optimal use of a sideboard on a healthy deck
   is to not touch it, which is why usage is audited rather than
   assumed.
4. Mob fights: the engine is multi-enemy (AoE, per-target wards, threat)
   but the experiment table is still 1v1.
5. Risk-sensitive objectives: `build_deck(objective='p90')` ranks the
   final pick by (win, p90 TTK, size) instead of the mean — the
   reliability build (`risk_experiment.py`), and
   `build_deck(player_hp=...)` is the SURVIVAL arm — boss damage
   counts, the screen proxy gains triage, and the template offers
   shields/heals only against a boss that hits back
   (`survival_build.py`, `results_survival_build.json`, chart below).
   Two damage regimes vs live Jade Oni. CHIP (240/round, 9 rounds of
   player HP): the survival-built deck beats the speed-built deck
   **98.6% vs 89.0%** under fire and is FASTER (mean 8.28 vs 9.36) —
   one Pixie + an extra blade keeps the kill line alive — and the
   deck itself carries the reliability (scripted-triage pilot: 94.5%
   vs 50.1%). BURST (+650 cheat hit every 4th round): burst pressure
   makes the fight MORE of a race — the winner is the race chassis
   plus exactly one Pixie flown aggressively (**83.4% vs 58.5%** for
   the pure-speed build); shield-heavy candidates lose (1–35%), and
   the winning deck under heal-when-low triage wins 0.6% — the heal
   must be TIMED between bursts, which RL learns and the script
   can't. Pipeline lesson from the first burst run: a triage-only
   survival screen went blind (every candidate 0–6%, ranking =
   noise) and mis-built at 37%; the screen now scores every
   candidate under BOTH the race proxy and the triage proxy and
   keeps the better — the screen must not presuppose the fighting
   style. At 6 rounds of chip HP nothing survives at all (best build
   0.3%): the cheatless damage model gives fights a hard HP floor.
   Honest
   finding on the tail objective, twice now: mean and p90 pick the
   SAME deck (storm dummies, and this fight) — within the
   plausibility-capped pool the winning deck dominates the whole TTK
   distribution rather than trading mean for tail; the p90 column
   discriminates candidates (10 vs 14 vs 18) but win rate decides.
   Still open: `max_remaining_damage` as an RL feature, and an
   HP-aware state for the per-deck agent (the tabular pilot cannot
   see its own HP yet still beats triage by killing faster — less
   exposure beats more healing at this damage level).
6. Offline RL ladder — first three rungs SHIPPED (`offline_rl.py`,
   `results_offline.json`; the dataset regenerates from seeds and is
   not committed). Design: policy class held fixed (the generalist's
   linear features), so the comparison isolates the DATA SOURCE —
   per-deck tabular experts' demonstrations (38k logged decisions,
   16 pairs, ε=0.1 behavior noise) vs online RL's own exploration.
   Zero-shot means on 8 feasibility-filtered held-out pairs:
   BC-all **48.0%**, BC-filtered **74.8%**, CQL-lite **71.3%**,
   online generalist **76.7%**, per-pair experts 78.9%, scripted
   heuristic 83.6%. Findings, in order of surprise: (1) filtering
   demonstrations to winning episodes is the single biggest lever
   (+27 points — losers teach losing habits); (2) offline learning
   from demonstrations nearly matches online exploration in the same
   policy class (74.8 vs 76.7); (3) nobody learned beats the
   scripted blade-stack prior on raceable random pairs — and the 8k-
   episode experts themselves average below it, which caps what any
   clone can learn (garbage-ceiling, not garbage-in). Remaining
   rungs: IQL-style expectile variants and sequence models for the
   multi-hit planning the linear class provably can't express.
   SEARCH-GENERATED TEACHERS, done (`offline_search.py`,
   `results_offline_search.json`) with a finding that sharpened the
   ladder's theory: stronger teachers made WORSE clones. Search
   teachers (76–100% per pair, demos at the brittleness-lesson
   ε=0.02) cloned to 67.6% vs 74.8% from the weaker RL teachers —
   because clone quality is teacher quality × in-class
   REPRESENTABILITY × coverage, and search's edge lives in rollout
   information the linear student cannot observe, while the cleaner
   demos also covered fewer states (21k vs 38k decisions). The union
   cell separated the two: combining both datasets recovers the
   coverage loss (storm row 4.7% → 37.9%, mean back to 73.1%) but
   plateaus AT the old ceiling, not above it. The binding constraint
   has formally moved from teacher quality to STUDENT CAPACITY —
   every data source converges at ~75% for the linear class, which
   is the third independent line of evidence pointing at the
   sequence-model rung.
7. Deck × policy bilevel optimization. First results (death vs live
   Jade Oni): the searched 9-card deck reaches **100% win / TTK 6.45**
   vs the hand-built 12-card oneshot's 92.8% / 9.95 — smaller AND
   stronger, exactly the redundancy-vs-consistency trade the objective
   prices. Against randomized held-out bosses, deck size scales with
   fight length (5 cards vs 800 HP, 8–11 vs 1.9–7.4k) and prisms appear
   only when the hit school is resisted. The buildable pool is now the
   curated `TRAINED` whitelist in `deck_builder.py` (trained quest lines
   under their dev names, with approximate unlock floors): the dump has
   NO reliable trainability marker — pet cards (Firezilla), mutations
   (Ice Cat) and cross-school reskins (Skeletal Dragon Fire) are all
   `variant='core'` with null `level_restriction`, `training_cost` is 0
   for school quest spells and pet cards alike, and `pve_flag` marks
   PvE-only restrictions, not trainability — so after three rounds of
   heuristic whack-a-mole the whitelist is the honest fix (validated
   against the dump by a regression test). Note the search's FIRST run
   found a genuine reward hack — boss-only spells mislabeled `core` in
   the dump gave 1-turn kills — now screened + regression-tested.
   Status: all four rungs are
   implemented. `deck_builder.py` covers rungs 1–2 (legal deck space
   with capacity/copy limits, template-sampled candidate search,
   two-stage screen → RL fine-tune, size-aware scoring so extra cards
   must buy reliability) plus `random_boss()` and a held-out
   generalization harness. Level gating is max(curated unlock floor,
   the dump's `level_restriction`) — the restriction field alone is
   null below ~30.
   Rung 3 shipped as `deck_scorer.py`: a closed-form ridge surrogate of
   the simulation screen over deck-vs-boss features (damage/blade/trap
   sums, prism gain vs the boss's resists, overkill ratio, plus the
   interactions a linear model can't invent — buffs×damage, X-pip×HP).
   Trained on 1,280 logged screen rows from 32 random bosses and
   validated on 8 HELD-OUT bosses (split by boss — a row split would
   leak): mean Spearman 0.67, mean top-1 regret 0.3 points (worst 1.3).
   `build_deck(scorer=...)` uses it to simulate only the predicted-best
   third of candidates and finds the identical final deck on a fresh
   boss. The scorer never replaces simulation — it only picks which
   candidates get simulated, and the pruning is logged, never silent.
   Screens append training rows via `build_deck(screen_log=...)`
   (`deck_screen_log.jsonl`), so the dataset grows with normal use.
   Rung 4 shipped as `generalist.py`: a deck-conditioned combat policy
   — linear Q over card-vs-state features (how hard THIS card hits
   THIS boss through the exact blades/traps hanging, kill-now,
   duplicate-blade, X-pip-waiting...), trained with the same backward
   Monte-Carlo returns as the tabular agent but on a fresh random
   (school, deck, boss) every episode, so one policy plays any legal
   deck zero-shot. On six held-out (deck, boss) pairs it averages
   49.2% vs the scripted heuristic's 50.7% and per-deck RL(8k)'s 53.5%
   — within 1.5 points of the heuristic and 4.3 of the per-deck
   ceiling at ZERO marginal training per deck. The honest gap: X-pip
   pip-timing (balance 53% vs RL's 85%) — a linear feature can't
   express "wait exactly until pips × per-pip ≥ HP". A hand-coded
   wait-until-lethal threshold feature was probed and made that row
   WORSE (53% → 44%): no single Judgement can be lethal there (~16
   pips needed through blades, 14 is the cap), so the winning line is
   two chunked hits — the real ceiling is multi-hit sequencing, which
   a memoryless linear policy cannot plan. Kept as a negative result;
   the fix belongs to the sequence-model rung of the offline-RL
   roadmap, not to more features. As build_deck's
   stage-2 evaluator (`build_deck(generalist=...)`) it picks an
   equally good final deck 3.3x faster (14s vs 46s). Exact hanging-
   effect percents in the features mattered: with blade/trap COUNTS
   the death-grind row scored 7%; with percents, 48% — parity.
8. Post-classic rulesets behind `Rules`: criticals-on eras, mastery
   amulets, school pips/archmastery — each as a frozen named ruleset.

## Boundary

Fully offline simulator and research benchmark. The classic ruleset is
built from community wiki data; the live ruleset from user-extracted
local game data files. Strictly offline research: no live-client
control, no traffic interception, no gameplay automation, and no
further wiki scraping (the scraper is retained for reference only) —
see `docs/RESEARCH.md`.

## Extraction pipeline (game data)

`extract_spells_phase1/2.py`, `add_variant.py`, `add_effect_names.py`
(run locally against a wiztype type-list dump, e.g.
`r803238_Wizard_1_610.json`). Three upstream improvements unlock the
remaining data gaps — all in `parse_effects` / post-passes:

1. **Effect names**: run `add_effect_names.py` and re-export — the
   loader's inferred id map becomes verifiable ground truth, and the
   ~3.5k cards skipped as "undecoded effect id" become decodable by
   NAME (`kDamage`, `kHealOverTime`, ...).
2. **Random damage ranges**: `parse_effects` drops list/dict members
   from `_raw`, and range hits are wrapper effects (Random/Variable
   SpellEffect) whose `m_effectList` holds the real sub-effects —
   recurse into `m_effectList` instead of dropping it and param=-1
   spells (Sunbird, Triton, Kraken...) get exact min/max, ending the
   classic-average backfill.
3. **Locale names**: resolve `display_name`/`description` keys against
   the game's locale string table so dev names align with display
   names.
