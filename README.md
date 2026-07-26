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
| `content.py` | curated effect primitives (the `spell_effects` layer): DoT splits, multi-hit components, drains, prisms, absorbs, per-pip spells, dispels, summons — each entry confidence-tagged (`community`/`approx`/`inferred`), unsupported cards excluded instead of mis-modeled |
| `w101_sim.py` | v0.3 engine: structured hanging effects with **(name, source) stack keys**, FIFO ward pass with prism school conversion, shields/weaknesses/mantles/dispels/absorbs, scheduled DoTs/HoTs (snapshot at cast), drains, X-pip spells, multi-hit link groups, AoE with per-target resolution, stuns + stun blocks, threat-driven enemy targeting, minions, boss **cheat-script hooks**, treasure-card sideboard, crit/block/pierce machinery, version-tagged `Rules` |
| `bosses.py` | preset enemy registry + illustrative cheat scripts + candidate decks for **all seven schools** |
| `dp_solver.py` | value iteration on the deck-free abstraction (distinct-buff configs, ward charge states, X-pip actions) = perfect-information lower bound; decks that lean on unmodeled mechanics (prisms/heals/shields) are flagged via `meta['unmodeled']` |
| `rl_agent.py` | tabular Q-learning (backward MC returns, DP warm-start, scarcity-aware state incl. drains and prisms) + UCB deck-selection bandit per boss |
| `experiment.py` | the headline tables (`results.json`): speed (immortal player) + survival (real boss damage) |
| `tests/test_sim.py` | 70 mechanics tests, one per documented combat rule |

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
  machinery exists and is tested), no archmastery/shadow/school pips, no
  gear/pet stat layer (fields exist, default 0), enemy decks are flat
  scripted hits + cheats. Beguile is the one card excluded as unsupported.

## Headline results (v0.3 rules)

RESULTS_TABLE_PLACEHOLDER

Ordering everywhere: **DP-LB < RL ≲ DP-transfer / heuristic**. The agent
learns draw-aware play (dig timing, partial-stack fires, nuke conservation,
fizzle-risk management) that the perfect-information abstraction can't
represent.

Structure the v0.3 pipeline surfaced:

- **Fizzle-discard changes deck building.** With cards lost on fizzle,
  thin 3-nuke decks stop being free wins: kill% now prices in accuracy.
  The RL agent learns to hold buffs until the nuke is actually in hand.
- **Prisms crack the same-school wall.** Ice vs Prince Gobblestone (40%
  ice resist): the prism deck converts ice → fire and picks up the +25%
  boost instead. The FIFO ward order (traps *then* prism) is learnable
  structure, not a hand-coded rule.
- **Drains turn damage into sustain.** Death's kit reads as mid-power on a
  TTK table but dominates survival matchups — exactly the school-identity
  prior the research doc predicts.
- **Multi-hit spells price shields correctly.** Minotaur under-performs on
  clean boards but is the cheapest answer to shield-cycling cheats.
- **Deck choice is learnable context.** The UCB bandit converges to the
  right loadout per boss, including switching ice → prism only into
  same-school walls.

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

## Next

- Scrape real creature pages (stats, stunable flags, actual cheat scripts)
  to replace the ballpark boss registry.
- Sideboard policy learning: the treasure-card mechanic (random draw,
  no same-round discard) is implemented but the RL action space doesn't
  use it yet.
- Mob fights: the engine is multi-enemy (AoE, per-target wards, threat)
  but the experiment table is still 1v1.
- Post-classic layers behind `Rules`: criticals-on eras, mastery amulets,
  school pips.
