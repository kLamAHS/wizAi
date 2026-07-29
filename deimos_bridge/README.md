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
| `live_backend.py` | no (duck-typed) | a wizsprinter backend driven by a wizAi policy |
| `mock_client.py` | no | fakes of the wizwalker objects, for testing the two above |
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

On the current rules, **14 of 21 agree to the cent**. The seven that
don't split three ways.

**Three are Deimos being wrong.** It keeps pierce as a fraction (`0.15`)
but shield params in points (`-50`), then adds them —
`ward_param += caster_pierce`, `combat_math.py:196`, and the same slip at
`effect_simulation.py:271`. So 15% pierce moves a Tower Shield from −50
to −49.85. wizAi converts first and is correct.

**One is the duplicate-effect guard firing at a different stage.** Deimos
dedupes during damage resolution via `spell_effect_stacking_id`; wizAi
refuses the *cast* (`Sim.can_cast`), so a duplicate never reaches the
board. Both are right; the scenario builds a board directly and walks
past wizAi's guard.

**Two are real wizAi findings**, and one is those two compounded:

- **Flat damage is in the wrong place.** wizAi adds it after charms,
  wards and crit. Deimos adds it right after the school damage %, so
  charms multiply it and shields reduce it.
- **Flat resist is in the wrong place.** wizAi subtracts it after the
  percent-resist multiply; Deimos subtracts it before.

Deimos does both of these in *two independently written code paths*
(`combat_math.py:157,253` and `effect_simulation.py:397,441`), which is
what makes it the better witness.

Both are available as opt-in `Rules` flags, the pattern wizAi already
uses for `fizzle_discards_card`:

```python
Rules(flat_damage_before_multipliers=True, flat_resist_before_resist=True)
```

Defaults preserve current behaviour, so no published table moves. With
them on the suite goes to 17/21, and the matched pair pins the residue:
`full stack (no pierce)` lands on **505.30 from both engines**, and the
only thing separating it from the row that still differs is 10% pierce.

### How much this costs

```
python -m deimos_bridge.flat_stat_probe
```

Currently: nothing. `gear.loadout()` returns no `flat_damage` or
`flat_resist` at any level and `Actor` defaults both to `0.0`, so the two
placements compute identical numbers on every table this project has
published. The findings are real and dormant.

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

Needs Windows, Python 3.13+, and the game running and logged in.

```
cd Deimos && uv sync          # once
python -m deimos_bridge.run_live --school fire --policy blade-stack
python -m deimos_bridge.run_live --school fire --policy trained \
    --deck "Fireblade,Fireblade,Sunbird,Sunbird,Sunbird,Tri Blade"
```

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

Naming. wizAi's card table is keyed on exact name and a miss is silent —
the policy simply never sees the card. Resolution is layered (exact →
alias → normalised) with **no fuzzy fallback**: Deimos ships `thefuzz`
and uses it for UI convenience, but casting the wrong spell in a real
fight is worse than passing. Misses are counted and
`resolver.report()` prints them at the end of a run.

## Tests

`../tests/test_deimos_bridge.py`, 27 tests. The oracle tests pin the port
to the arithmetic in `combat_math.py` — if someone edits the port to make
a divergence disappear, they fail, which is the point. The plumbing tests
drive the live read and the backend against `mock_client`, including a
policy that raises (must cost one round, not the fight) and a policy that
names a card not in hand (must pass, not guess).
