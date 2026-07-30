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

### Moves carry a target

A policy returns `(card, enemy_index)` — the tuple `Sim._normalize_action`
already unpacks, so aiming works identically in the simulator, in
`evaluate`, and live.

That is not cosmetic. Before it, no policy chose a target at all: every
cast went to `enemies[0]`, whichever mob the participant list happened to
put first, and when that one died the rest of the plan silently moved to
a different mob with the traps left behind on a corpse. A trap only pays
off if the hit it is buying lands on the same enemy, so the two have to
agree.

`greedy_ttk` scores every (card, enemy) pair — on a boss-and-minion board
those genuinely differ, and that difference *is* the decision.
`school_aware_blade_stack` aims everything at one `focus_target`: the
lowest-health living enemy, chosen because hitting it only lowers its
health further, so the focus re-derives to the same mob until it dies
rather than wandering and splitting a buff stack across two.

### Duplicate effects: the rule wizAi had backwards

wizAi used to model "you may not place the same effect twice" as a
**cast restriction** — `Sim.can_cast` refused a pure hanging-effect card
whose effect was already on the target, and `execute_ops` silently
dropped it if it got through anyway. There is no such restriction in the
game. Three Ice Traps go on one mob perfectly happily; what they do not
do is all fire on the same strike.

Deimos is the corroborating witness. It reads every hanging effect off
the live participant and dedupes by `spell_effect_stacking_id` *during
damage resolution* (`combat_math.py:161-194`) — a design that only makes
sense if duplicates can sit on a target, because otherwise there would be
nothing to dedupe.

Getting this backwards was not a wash, because the guard was **inert
exactly where it mattered**. Live-read hangings are named
`live:<template id>` with source `"live"`, so their stack keys never
matched a card in hand and `has_stack` always said no. In a real fight
wizAi therefore laid duplicate after duplicate *and* multiplied every one
of them into a single strike:

```
3 x 40% Ice Trap, one Snow Serpent
  wizAi (before)   480 damage   2.744x     <- 1.4^3
  Deimos / game    245 damage   1.400x     <- one fires, two are banked
```

A 96% overvaluation of the third trap is why stacking looked worth
spending rounds on. `_ward_pass` and `_consume_damage_charms` now apply
one hanging per stacking identity per strike and leave the rest standing,
which is both the game's rule and Deimos's; the placement guards are
gone. This is what closed the last row in the differential suite — the
two engines now agree on all 22 scenarios.

Nothing about legality depends on the target any more, so `can_cast`'s
`target` argument is vestigial (kept because callers legitimately know
what they are aiming at, and `Sim.run` passes it).

### Knowing when to stop setting up

Three separate things made the policy over-invest in setup, all of them
in how a line gets scored rather than in the policy itself.

**Enemies dealt no damage.** `read_state` builds enemies with no
`flat_hit`, so `_enemy_turn` did nothing and every rollout modelled the
mobs as harmless. Setting up cost turns and nothing else, and
`_rollout`'s `player.alive` check could never fire — the policy had no
way to know it was about to die to the minion. Nothing in the client
reports a mob's spell damage, so `WizAiBackend._estimate_incoming`
measures it: the player's health drop across a round, split over the mobs
alive to cause it. That folds in DoTs and minion hits, which is right —
what matters is how long the fight can afford to take, not who is
responsible. Before the first measurement it uses the trainer's own
dummy-boss prior, so turn one is not priced differently from the fight
the Q table learned on.

**A losing board erased every distinction.** `_rollout` returned one flat
constant whenever the line died, so on a board where nothing survives the
horizon *every* candidate scored identically, the comparison collapsed,
and the choice fell through to the tiebreak — which takes the cheapest
card. An Ice Trap costs zero pips. That is the whole mechanism behind
"it spams every trap card": not a preference for traps, an absence of any
preference at all. Dying now ranks below stalling and above nothing, and
carries the damage the line actually banked.

**Overkill counted as progress.** `Sim._strike` does `target.hp -= dmg`
uncapped and `cast` returns the raw number, so a 300-damage nuke into a
mob with 50 left banked 300 — and the damage tiebreak rewards banking
more. Two pushes toward waste at once, since the third key was
`-card.damage` ("take the biggest hit"). Damage is now floored at each
mob's health, and ties break toward the *cheapest* card.

On top of those, `policies.cheapest_lethal` asks the question that was
missing entirely: *is it already dead?* A buff round against a mob the
plain nuke already finishes is a round given away, and stacked three deep
it is the fight given away. It runs the engine's own cast path, so a
shielded mob is correctly not lethal, and it picks the smallest card that
still kills rather than the biggest available.

### The wizard is not naked

`Sim` has always taken `player_stats` — damage, accuracy, pierce, crit,
resist — and nothing was filling it in. `live_state.read_player_stats`
reads them off `GameStats` on connect, adding the "all schools" scalar to
the by-school vector the way `combat_math.real_stat` does, and both the
live `Sim` and `train_agent` now get them.

This changes decisions rather than decorating them: on a 2000hp mob with
an ice deck, `greedy_ttk` opens with a trap 100% of the time given 9%
damage and 4% pierce and 0% of the time given nothing. The direction
depends on the board — more damage shortens the fight, which can make a
trap pay off *or* stop being worth the round — so there is no safe
direction to guess in, which is the argument for reading it.

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
python -m deimos_bridge.gui           # press "Play live" to fight
python -m deimos_bridge.gui --demo    # canned fight, runs anywhere
```

**Play live** connects to the client, installs the hooks and takes over
combat from the window — same engine as `run_live`, with the panels
filling in as the fight happens. School, policy, deck and how many fights
come from the controls at the top; `fights = 0` means keep going until
you press Stop. A trained policy has to be trained first (the Train
button) and needs its deck, since the Q table is keyed on that deck's own
blade and nuke positions.

The fight runs on its own thread with its own asyncio loop, so a slow
memory read cannot freeze the window, and nothing on that thread touches
a widget — the worker emits signals and the GUI thread draws.

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

## Deck entry, and questing

**Deck.** Press **Choose…** next to the deck field. Three ways in: *Build
one for me* runs `deck_builder.build_deck` for the school and level;
search-and-click filters the card table as you type; *From the last
fight* seeds from the cards actually seen in hand during the last run.

That last one is as close to "read my deck off the game" as is honest.
The client exposes the deck as **template ids** (`deck_behavior.spell_list`),
and wizAi's card table carries no template ids to match them against —
`spells_full.json` records have no id field at all. So a real deck read
cannot be turned into names. Cards *in combat* can, because
`CombatCard.name()` returns one.

The search hides boss and event variants by default. The table holds
`Iceblade`, `Iceblade - EM`, `Iceblade - SIT`, `IcebladeBOSS01` and more —
a mob can cast those, you cannot, and they bury the real card. The filter
uses the same `base_spell` rule as the langcode index: the canonical
record is the one whose `name` *is* its `base_spell`. 1,301 of the 8,148
table entries. Switch to "every variant" to see them all.

**Questing.** `deimos_bridge/questing.py`, all on plain wizwalker:

- **Teleport to quest** — `client.quest_position`, the same hook
  `wizwalker/examples/quest_teleporter.py` uses.
- **Auto-dialogue** — watches for dialogue and clicks through it for the
  whole run, paused during combat so it cannot fight the card clicks.
  Bounded per conversation, so one that reopens forever cannot hang a run.
- **Auto-quest between fights** — wait out loading, read the marker,
  teleport, clear dialogue, press X, look for combat, repeat.

Three things that version one got wrong, all of which showed up on the
first real run:

- **The buttons did nothing.** Requests were drained at the top of the
  fight loop, which spends nearly all its time blocked inside
  `wait_for_combat` — so a press queued while waiting sat there until a
  fight had started *and* finished. They now run on a concurrent service
  task and act within a second.
- **The hunt died at the first zone change.** It returned on the first
  failed read, and a zone change makes several fail in a row. It now
  waits out loading screens, retries, and only gives up after `max_hops`
  or on a cause retrying cannot fix.
- **It never interacted.** Arriving at the marker is not enough for
  sigils, dungeon doors or quest NPCs, so it presses X.

If a teleport does nothing, the usual cause is the **in-game quest arrow
being switched off**: `activate_all_hooks` notes that "the quest hook is
not written if the quest arrow is off" (`memory/handler.py:187`), which
leaves the position reading as the origin. That is reported as a reason
now rather than a silent failure.

### Deimos's questing does the navigating

The light version above has no navigation, so a quest whose marker is
across a zone boundary stalls. Deimos already solved that properly, and
`deimos_questing.py` uses it rather than reimplementing it — navmap
teleports, spiral doors, dungeon entry, NPC talking, zone correction,
chest rerolls and potions.

It composes instead of conflicting, for one specific reason:
`Quester.auto_quest_solo` opens with `if await is_free(self.client):`,
and `is_free` is False during combat, loading or dialogue
(`src/questing.py:1414-1416`). So it advances the quest while the wizard
is idle and does nothing once a duel starts. Deimos gets you to the
fight; wizAi's policy fights it; neither reaches for the mouse at the
same time.

Deimos drives it as `while questing_status: sleep(1);
auto_quest_solo(...)`. Handing that loop control would take the fight
loop's ownership away, so `DeimosQuester.step()` runs a single iteration
from the service task instead.

`Quester` also reads a dozen attributes that Deimos sets on the Client
(`Deimos.py:_init_client_attrs`) and wizwalker does not have —
`questing_status`, `use_potions`, `entity_detect_combat_status` and so
on. `init_client()` supplies them, defaulting to no pet training and no
potion buying.

### Between-fights upkeep

An unattended run dies by attrition long before it runs out of quests,
and a policy that lost because the wizard was at 12% health has told you
nothing about the policy. Two toggles, both on by default:

- **Collect wisps** — after each fight, teleport over the health and mana
  wisps it dropped. Skips any sitting next to a mob (Deimos's
  `find_safe_entities_from`), so topping up does not start a second
  fight, and is bounded so a zone full of pickups cannot stall the loop.
- **Use potions** — drinks one below Deimos's threshold (under 55%
  health, or low mana). It never *buys*: refilling means a vendor trip,
  real gold, and a navigation detour that can strand the run.

`upkeep.py` builds both on `SprintyClient`, which is **pure wizwalker**.
Deimos's own `collect_wisps` lives in `src/utils.py` and would drag in
wizsprinter with it, so the three calls are rebuilt directly — upkeep
works on the light install, with no extra dependency. Deimos's questing
already does this inside `auto_quest_solo`, but only while questing;
here it is its own toggle so it also runs when auto-quest is off, which
is exactly the case when farming one fixed mob.

Cost: `src.utils` imports `wizwalker.extensions.wizsprinter`, so this
needs wizsprinter (Python 3.13+) plus `thefuzz`, `loguru`, `pyyaml` and
`requests`. `setup-windows.bat` installs them and treats failure as
non-fatal — without them the bridge falls back to the light questing and
says so in the status bar. Nothing else in the package is affected.
