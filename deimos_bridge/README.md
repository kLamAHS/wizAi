# deimos_bridge

Connects the wizAi combat research code to [Deimos](../Deimos), a
Wizard101 automation bot built on wizwalker.

Deimos matters here for one reason: it reads the **live game client's
memory**. Its combat math was not derived from a wiki, it was written by
someone who could watch a Fireblade land on a real mob and check the
number. That makes it two things wizAi did not previously have — an
independent oracle for the damage model, and a way to put a learned
policy in front of real enemies.

## What runs where

Only one file needs Windows. This is worth stating plainly because it is
the main constraint on the whole exercise:

| module | needs the game? | what it does |
|---|---|---|
| `deimos_damage.py` | no | Deimos's `src/combat_math.py`, ported to plain Python |
| `scenarios.py` | no | one fight, rendered into both engines' terms |
| `differential.py` | no | runs both engines and diffs them |
| `flat_stat_probe.py` | no | how much the divergences actually cost |
| `effect_audit.py` | no | wizAi's effect coverage vs the client's own enum |
| `live_state.py` | no (duck-typed) | live combat → a wizAi `State` |
| `live_backend.py` | no (duck-typed) | the policy → a cast |
| `telemetry.py` | no | what a run records, and the damage-model residuals |
| `gui/` | no (`--demo`) | the ML-facing window |
| `mock_client.py` | no | fakes of the wizwalker objects, for testing the above |
| `combat_api_shim.py` | no | wizsprinter's types, or equivalents off Windows |
| **`run_live.py`** | **yes** | actually fights |

The live leg is hard-blocked off Windows, and it is worth knowing that
none of these are soft blocks:

- `wizwalker/constants.py` binds `ctypes.windll.user32` at module scope,
  so `import wizwalker` raises on Linux; `wizwalker/__main__.py:40`
  refuses to start on anything but win32.
- `pymem==1.13.1` is declared `sys_platform == 'win32'` — the memory
  layer is not merely broken elsewhere, it is absent.
- Wine is not an escape hatch: wizwalker pattern-scans the client's PE
  for byte signatures, patches live instructions, and runs hand-assembled
  x86-64 shellcode via `CreateRemoteThread`.
- Deimos and wizsprinter both declare `requires-python = ">=3.13"`
  (wizwalker itself is content with 3.11).
- There is no headless mode — `Deimos.py`'s entry point starts the Qt GUI
  unconditionally — and nothing in the tree offers a replay or offline
  duel.

Hence `mock_client.py`: the live read and the backend are duck-typed
against the wizwalker async API rather than importing it, so the
identical code path is exercised and tested on any machine before it ever
meets a real fight.

## The cross-check

```
python -m deimos_bridge.differential --both
```

21 scenarios, each declared once and rendered into both engines. wizAi is
driven through the same call sequence a real cast uses
(`_consume_damage_charms` → `_crit_mult` → `_strike`), so shields, traps
and prisms resolve exactly as they do in a duel.

**20 of 21 now agree to the cent.** The one that doesn't is the
duplicate-effect guard firing at a different stage: Deimos dedupes during
damage resolution via `spell_effect_stacking_id`, wizAi refuses the
*cast* (`Sim.can_cast`), so a duplicate never reaches the board. Both are
right; the scenario builds a board directly and walks past wizAi's guard.

Getting there took a fix on each side.

**wizAi had flat damage and flat resist in the wrong place.** Flat damage
was added after charms, wards and crit; Deimos adds it right after the
school damage %, so charms multiply it and shields reduce it. Flat resist
was subtracted after the percent-resist multiply; Deimos subtracts it
before. Deimos does both in *two independently written code paths*
(`combat_math.py:157,253` and `effect_simulation.py:397,441`), which is
what makes it the better witness. wizAi now matches, via
`Rules.flat_damage_before_multipliers` / `flat_resist_before_resist`
(default `True`; set either `False` to recover the old arithmetic, and
`--legacy` re-runs the suite that way to show the gap the fix closes).

Adopting it cost nothing historically — see the flat-stat probe below.

**Deimos had pierce in the wrong units.** It keeps pierce as a fraction
(`0.15`) but ward params in points (`-50`), and added them directly:
`ward_param += caster_pierce` at `combat_math.py:196`, the same slip at
`effect_simulation.py:271`, and `caster_pierce += effect_param` on a
pierce *blade*, which made a `+10` blade worth 1000% pierce. So pierce
did essentially nothing to shields and everything to nothing else. Fixed
in both files; the file's own `pierce += param / 100` two cases down was
already doing it correctly and settled which unit was intended.

### How much this costs

```
python -m deimos_bridge.flat_stat_probe
```

Historically: nothing, which is why the fix could be adopted as the
default. `gear.loadout()` returns no `flat_damage` or `flat_resist` at
any level and `Actor` defaults both to `0.0`, so the two placements
compute identical numbers on every table this project has published — no
committed result moves.

They wake up with real gear numbers — which is exactly what a live run
supplies, since wizwalker reads `dmg_bonus_flat` and `dmg_reduce_flat`
straight off the participant:

```
flat damage error, by blade count and flat damage
  1 blade,  +50 flat →  2.4%      4 blades, +50 flat →  6.8%
  1 blade, +200 flat →  8.0%      4 blades, +200 flat → 25.0%
```

The error scales with the blade stack, which is the uncomfortable part:
the blade-stack meta is what this project studies.

## Effect coverage

```
python -m deimos_bridge.effect_audit
```

wizwalker's `SpellEffects` enum is the client's own, so it settles what
effects exist and what they are called.

- wizAi's `effect_type_enum.json` agrees with it on **151 of 152**
  entries. The one difference is where a capital letter falls
  (`kModifyCardDamagebyRank` vs `modify_card_damage_by_rank`).
- The decoder builds **8,148 of 18,162** records. 4,971 are lost to an
  undecoded effect, and **3,227 of those to `kSummonCreature` and
  `kKillCreature` alone** — so most of the gap is one coherent feature
  (summons), not scattered rot.

## Fighting real enemies

Needs Windows, Python 3.11+, and the game running and logged in.
**[RUNNING_LIVE.md](RUNNING_LIVE.md) is the step-by-step guide**, including
the install, what to look at afterwards, and troubleshooting.

The short version, from the repository root. Only wizwalker is required —
the live path goes through `WizAiCombatHandler`, not wizsprinter, so
neither wizsprinter's 3.13 floor nor wizlaunch's Rust extension applies,
and `uv` is not needed:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e Deimos\libs\wizwalker numpy
.venv\Scripts\python.exe -m deimos_bridge.run_live --school fire
```

then walk into a fight.

`--policy trained` trains a `QAgent` against the simulator first and then
plays the live fight with the resulting table, rather than exploring in
real duels. It requires `--deck`, because the agent's state key is keyed
on the deck's own blade and nuke positions (`Featurizer.__init__`) — a
table trained for one decklist means nothing for another.

Each planning phase the board is read into a wizAi `State`,
`policy(sim, state)` is called — wizAi's existing in-fight contract — and
the answer is turned back into a cast. The action strings already *are*
card names and `"name@i"` already carries a target index, so the
translation is nearly direct.

Every decision is logged with the state that produced it. On a live run
the first thing to check is not whether the policy won but whether it was
ever shown the right board.

### Two ways in, and only one of them measures your policy

`WizAiCombatHandler` (what `run_live.py` uses) subclasses wizwalker's
`CombatHandler` and issues **exactly one cast per decision**.

`WizAiBackend` also works as a `wizsprinter` combat backend, which is
convenient if you want wizAi picking moves inside an existing Deimos
setup — but do not measure a policy through it. `SprintyCombat`'s cast
loop re-queries the same spec after every cast and only stops when
`needs_post_filter` is set (`sprinty_combat.py:1554,1758-1763`), and that
flag requires a `TemplateSpell`. A `NamedSpell` can never set it, so the
loop keeps going while another copy of the card is in hand: **three
Fireblades in hand become three Fireblades cast** from one decision. For
a config-driven bot that is a feature; for an experiment it means the
thing that played the fight is not the thing you loaded.

The handler also fixes a quieter divergence. wizAi's `"name@i"` indexes
its own enemy list — alive, hostile, in read order — while
`SprintyCombat.get_enemies()` partitions by team and keeps the dead, so
after the first kill the two disagree and the policy hits the wrong mob
without ever raising. The handler resolves the index against
`LiveRead.enemy_members`, which is built index-aligned with the very list
the policy was shown.

### One more trap, in wizwalker itself

`CombatMember.is_monster()` is defined as *not a player and not a minion*
(`combat/member.py:84-88`). An **enemy minion** is neither, so a reader
that partitions on it files enemy summons as allies — and Wizard101 PvE
is minion-dense. The policy would then plan against a board missing
targets that are actively hitting it, which for this project is the worst
possible place to be wrong: multi-enemy targeting is the deficit its own
tables already flag. `live_state` partitions on `team_id` instead, with
`is_monster()` only as a fallback when the participant will not read.

### The part most likely to break

Naming, and the failure is silent: wizAi's card table is keyed on exact
name, so an unresolved card is not an error — it is simply a card the
policy never had. Three things guard it.

**A langcode layer.** `CombatCard.display_name_code()` returns the game's
own stable identifier (`Spells_Fireblade`), which does not move when a
spell is renamed and is identical on a non-English client. It is not
unique, though: that code is shared by `Fireblade`, `Fireblade - EM`,
`Fireblade - SIT`, `Fireblade - Tear`, `FirebladeBOSS01`,
`FirebladeBOSS02` and a raid sigil. `base_spell` settles it without
guessing — the canonical record is the one whose `name` *is* its
`base_spell`, which is the player-facing spell by definition. Groups
where that does not single one out are dropped rather than picked from.

**No fuzzy matching, anywhere.** Deimos ships `thefuzz` and uses it for
UI convenience; casting the wrong spell in a real fight is worse than
passing.

**Misses are classified, not just counted**, because the two kinds need
opposite responses. Most names that fail are not misspellings at all —
they are internal engine templates (`Summon589244`, `Kill1223126`,
`Hydra - T04 - C`) that never reach a hand, or real cards the decoder
skipped for a named reason. `build_catalog()` pays one pass over
`spells_full.json` to tell them apart, so a miss reads as *"undecoded
effect kSummonCreature — close the gap in `data_full._map_effect`"* or
*"not in the game data under this name — check spelling, add to
`ALIASES`"*.

**And an unresolvable card is recorded, not silently dropped.** It still
cannot enter the policy's hand — there is no wizAi `Card` to reason about
— but `LiveRead.hidden` and `hand_visibility` say so. This is the failure
that quietly voids a run: the policy plans a five-card hand while holding
seven, and its scarcity feature counts the wrong number of nukes left.
The GUI leads the Naming tab with hand visibility for exactly that
reason, and says outright that a run below 90% is not measuring the
policy you trained.

## The GUI

```
python -m deimos_bridge.gui           # live
python -m deimos_bridge.gui --demo    # canned fight, runs anywhere
```

Deimos's own window answers an operator's questions — is it questing, is
it stuck, how long has it run. Training and evaluating a policy needs a
different set, and getting them wrong is expensive in a way that is easy
to miss: **a run where the policy never saw half its hand looks exactly
like a run where the policy played badly.** Five tabs, in the order you
need them:

- **Board** — the board *as the policy saw it*: HP, pips, hand, and every
  hanging effect rendered as arithmetic (`+35% fire`, `prism -> ice`)
  rather than as the template ids the client actually hands over. Cards
  that failed to resolve are called out in red here, because that failure
  is otherwise completely silent.
- **Decisions** — every planning phase: what was cast, at whom, why, and
  what it *passed over*. A policy repeatedly declining a nuke it could
  afford is the shape of a state-featurisation bug.
- **Damage model** — the one number simulation alone cannot produce.
  Before each cast wizAi predicts the damage; the next round's real HP
  says what happened. Shows bias, mean absolute error, RMSE, and a
  predicted-vs-actual scatter. Rounds where a DoT, an AoE or a kill could
  have muddied the HP delta are marked and excluded from the headline
  statistics, and blade rounds — which predict 0 and deliver 0 — are not
  counted as observations at all, since including them would make a
  buff-heavy deck look accurate purely for nuking less often.
- **Naming** — unresolved card names with counts, and what to do about
  each.
- **Runs** — per-fight rounds, outcome, damage, passes; `Export run`
  writes the whole thing to JSON.

Training runs on a worker thread, so the window stays responsive through
a Q-learning run, and the deck is validated against the card table before
training starts — a decklist naming a card that does not resolve would
otherwise train a policy whose action space does not exist.

`telemetry.py` holds all of it and imports no Qt, so every judgement the
GUI makes is testable headless. `--demo` drives the real window from
`mock_client`, which is how it is tested off Windows.

## Tests

`../tests/test_deimos_bridge.py`, 27 tests. The oracle tests pin the port
to the arithmetic in `combat_math.py` — if someone edits the port to make
a divergence disappear, they fail, which is the point. The plumbing tests
drive the live read and the backend against `mock_client`, including a
policy that raises (must cost one round, not the fight) and a policy that
names a card not in hand (must pass, not guess).

## Playing live: the school assumption

wizAi's built-in heuristics were written against decks the builder
produced, and every one of those is **school-coherent** — a fire deck
holds fire blades and fire nukes. That assumption is invisible in the
simulator and false the moment a real wizard opens a real hand: starter
wands hand out Thunder Snake (storm), Imp (fire), Scarab (myth) and Dark
Sprite (death) regardless of school.

`make_blade_stack` picks the biggest buff and, separately, the biggest
nuke, with nothing tying them together. On a live starter hand it will
stack a **Mythblade** and then fire a **Thunder Snake** — and the blade
does nothing, because `_consume_damage_charms` only applies charms where
`h.matches(school)`.

`policies.school_aware_blade_stack` decides the nuke first and only
stacks buffs that can multiply it. It is the `run_live` default;
`--policy blade-stack` still selects the original.

It also fixes a counting bug that has nothing to do with mixed hands. A
**Tri Trap** places three ward legs — fire, ice and storm. An ice
wizard's hit consumes the ice leg and the other two stay on the enemy
for the rest of the duel, because nothing in an ice deck will ever
trigger them. `State.traps` counts all three, so `make_blade_stack`
believes it has a full stack while holding one live multiplier and two
corpses, and fires early. Measured over the project's own live decks,
paired seeds, n=500:

```
deck                  blade-stack       school-aware    delta
                     kill%    TTK      kill%    TTK
fire/speed            0.0      nan      0.0      nan     +0.0
fire/oneshot         88.8     8.78     88.8     8.78     +0.0
ice/stack            23.6    15.73     40.8    15.74    +17.2
ice/prism            71.8     9.07     72.2     9.12     +0.4
death/oneshot        93.6     9.06     94.0     9.09     +0.4
storm/oneshot        96.8     5.54     96.8     5.54     +0.0
balance/oneshot      96.2     8.52     96.2     8.52     +0.0
```

Never worse, and +17 points where the stranded ward legs bite. A test
pins that, since it is the live default.

Prisms are deliberately *not* school-filtered: charms are consumed
against the card's own school before the ward pass converts it, so an ice
blade still multiplies an ice hit that a prism is about to turn into
myth. Filtering them cost 24 points on `ice/prism` before that was fixed.
