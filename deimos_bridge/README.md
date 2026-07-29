# deimos_bridge

Testing the wizAi learner against Deimos.

wizAi trains a policy inside `w101_sim`, a simulator built from a wiki
scrape. Deimos is a bot that reads the running Wizard101 client's memory,
so its combat numbers are the game's numbers. Putting one against the
other asks a question the simulator cannot ask itself: **how much of the
learner's advantage is real, and how much did it learn from a modelling
gap?**

There are two ways to ask, and they cost very different amounts.

| | what it needs | what it tells you |
|---|---|---|
| `engine.py` | nothing — runs anywhere | the policy scored under the client's real stat curves |
| `live.py` | Windows, game open, Deimos | the policy actually fighting actual enemies |

---

## What had to be fixed first

Deimos ships `src/effect_simulation.py`, a simulation of the client's own
effect resolution. It had never executed. Two import-time errors sit
ahead of any of its arithmetic:

```python
class MagicSchoolID(MagicSchool):   # TypeError: cannot extend an enum
    universal = 80289               #   that already has members

class MagicSchoolIndex(Enum):
    MagicSchool.fire = 0            # AttributeError: cannot reassign
                                    #   member 'fire'
```

Neither is platform-specific, so the module has never run on Windows
either. Behind them were seven more defects that only running it could
surface — `ids` read one line above its assignment, `result_cache = Cache`
returning the type alias, a function with a declared return type and no
`return`, `get_stat.` for `get_stats.`, `MagicSchoolIndex[damage]`
indexing by the damage amount, `pips` used but never defined, and
`calc_crit` dividing by zero for any wizard without crit gear. Two were
wrong rather than fatal: absorb wards ran `damage += param`, so a shield
*added* the attack to itself, and the stacking-dedup key included the
list index, which made every effect unique and disabled the rule it
exists to enforce.

All of it is fixed in place and annotated with the defect repaired. **The
damage formula itself is untouched.** `tests/test_deimos_bridge.py` pins
each fix to the behaviour it restores.

## Running Deimos without Windows

`wizwalker/__init__.py` calls `ctypes.windll` at import, so the package
cannot be imported off Windows at all. But the module we need from it is
pure `enum`, and Deimos' simulation is pure functions over plain dicts.

`headless.py` stubs the package, loads `memory_objects/enums.py` from its
real file — the effect ids have to be the client's, not something we
invented — and then imports everything under `Deimos/src` unmodified.

```python
>>> from deimos_bridge.headless import install
>>> es = install().effect_simulation
>>> es.calc_crit(300, 100, 50, 50)
(1.5, 0.9, 0.04)
```

The stubs raise if anything constructs one, so leaving the offline path
fails loudly instead of returning a mystery object.

## The disagreement that matters

The two engines differ in several places, but only one is large.

wizAi treats gear damage as linear, without limit:

```python
return 1 + caster.damage_bonus.get(school, 0.0) + ...   # w101_sim
```

The client bends it. Past a soft cap each further point buys less than
the last; past a hard cap, nothing. Deimos implements that as
`curve_stat`.

The gap this opens scales with level, which makes it testable:

| level | gear damage | after the curve |
|------:|------------:|----------------:|
| 20 | 5% | 5% — below the bend |
| 50 | 36% | 36% — below the bend |
| 70 | 66% | 66% — below the bend |
| 100 | 114% | curved |
| 120 | 147% | 109% |

A policy trained under the linear model learned a game where the last
blade is worth as much as the first.

`DeimosSim` is `w101_sim.Sim` with `curve_stat` — Deimos' own function,
imported, not reimplemented — spliced into the two places it belongs, and
Deimos' crit formula behind wizAi's existing crit hook. Cards, pips,
draws, wards and enemy AI stay wizAi's, because those are not in dispute
and changing them would confound the measurement.

```bash
python -m deimos_bridge.evaluate --levels 20,50,70,100,120 --sweep 120
```

Read the **edge** column, not the win rate. Win rate falls under the
curve for everyone, learner and scripted lines alike; that is the curve
working. Edge is trained minus best scripted under each engine, and it is
what says whether the policy is real.

### One caveat, stated plainly

Deimos reads the curve constants from the live client (`Duel` offsets
612–632) and ships no static copy. The defaults in `DuelCurve` are
**assumed, not measured**. `--sweep` varies the cap instead of trusting
one setting; a conclusion that holds across the sweep does not depend on
getting the constant right. The sweep includes an effectively-uncapped
setting, which must reproduce the plain `Sim` numbers exactly — if it
ever stops doing so, the splice is measuring itself and every other
number here is void.

## What it found

Running the curve experiment turned up something bigger than the curve.

At levels 20, 50, 70 and 120 the two engines agree — at the low levels
because the gear sheet sits below the bend, at 120 because the fight is
won either way (100% both, TTK 5 → 6). At **level 100** the trained
policy went from 92.7% to 0.2%.

A 92-point collapse is a large claim, so the next step was to check it
rather than report it. The curve at level 100 cuts damage by about 7%.
Feeding a *flat* 7% cut into plain `w101_sim` — no Deimos, no curve,
nothing from this bridge in the loop — reproduces the collapse exactly.
So the cliff is not the curve's, and not the bridge's. It is the
policy's.

That is what `--robustness` measures, and it needs none of the assumed
constants:

```bash
python -m deimos_bridge.evaluate --robustness 20,50,70,100,120
```

| level | fight | trained @1.00 | @0.99 | @0.97 | @0.95 | @0.80 | margin |
|------:|---|---:|---:|---:|---:|---:|---:|
| 20 | 510 HP, 4 turns | 92.4% | 92.4% | 84.0% | 84.0% | 46.8% | x0.95 |
| 50 | 2.4k HP, 6 turns | 95.2% | 95.2% | 95.2% | 95.2% | 45.4% | x0.95 |
| 70 | 1.3k HP, 4 turns | 71.8% | 71.8% | 69.4% | 69.4% | 66.6% | **x0.80** |
| 100 | 10.6k HP + 2 adds, 17 turns | 95.2% | 80.8% | **2.0%** | 1.2% | 0.0% | **x1.00** |
| 120 | 4.6k HP + 2 adds, 5 turns | 100% | 100% | 100% | 100% | 100% | **x0.80** |

The scripted blade-stack lines are flat across every row — `race(3)`
holds 100% at level 100 through the entire sweep. So the sensitivity is
not the fight's. It is specific to what was learned.

**The level-100 policy loses 93 points of win rate to a 3% change in
damage.** It is a knife-edge kill line fitted to the simulator's exact
arithmetic, in the one fight long enough (17 turns) for the boss to win
the attrition race if the kill slips by a turn. It is also, under wizAi's
own numbers, already *worse* than simply stacking three blades — 95.2%
against 100%. On that board the learner is both fragile and unnecessary.

The short fights are fine. Levels 70 and 120 shrug off a 20% cut.

The practical reading: no simulator matches a live client to within 3%,
so the level-100 policy will not transfer, and the margin column — not
the win rate — is what says which policies will. Nothing here required
knowing the client's constants, which is why this result is the solid
one and the curve numbers above it are the provisional ones.

## Fighting real enemies

```python
from deimos_bridge.live import WizAiFighter

fighter = WizAiFighter(client, policy=agent.policy(),
                       cards=cards, decklist=decklist, school="fire")
await fighter.handle_combat()
```

Each round it snapshots the client, builds a real `w101_sim.State` from
what it sees, runs the policy, and casts what comes back. It reuses
wizAi's own `State` and `Sim` rather than duck-typing them, so legality
is decided by wizAi's rules and not a second copy of them.

Two limits are deliberate and visible in the code:

- **The undrawn deck is estimated.** The client shows a hand, not a
  remainder. The policy's `nukes_left` feature counts cards still in the
  deck, so the adapter tracks what has been seen and subtracts it from
  the decklist you pass in. Wrong if that decklist is not what is
  actually equipped.
- **Names must match.** wizAi knows cards by wiki name, the client by its
  own. Unmatched names are logged and dropped rather than guessed at — a
  silent mismatch would leave the policy reasoning about a card it does
  not hold.

Everything between reading the client and casting is a plain function
over plain data, so it is tested without a game in
`tests/test_live_adapter.py`.

## Layout

| file | role |
|---|---|
| `headless.py` | stub wizwalker, load the real enums, import Deimos' combat modules |
| `caches.py` | wizAi objects → the dict shapes Deimos' simulation expects; `DuelCurve` |
| `engine.py` | `DeimosSim` — wizAi's loop with the client's stat curves |
| `evaluate.py` | the experiment, and the cap sensitivity sweep |
| `live.py` | `WizAiFighter` — a Deimos `CombatHandler` driven by a wizAi policy |
