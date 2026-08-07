"""Planning one round for a whole party, not one wizard at a time.

Four wizards farming the same battle circle with four independent copies
of the policy is not four times one wizard. It is four wizards making the
same decision, because they are shown the same board and run the same
arithmetic on it — so they focus the same mob, kill it three times over,
and leave the mob beside it at full health for another round of free
hits. Every wasted cast is also a wasted *round*, and rounds are what the
whole `greedy_ttk` lookahead is trying to save.

The fix is not a better single-wizard policy. It is to let the four
decisions see each other before any of them is clicked:

  **A barrier.** Every seat's planning phase blocks until the whole party
  has arrived (or `timeout` passes — one client hanging must never stall
  the others). W101 gives ~30 s of planning, so a couple of seconds of
  agreement is free.

  **A shared ledger.** What one seat commits to this round is applied to
  the board the next seat plans against: expected damage comes off the
  mob's health, and traps, shields and DoTs the cast places land on the
  enemy the others are looking at. So a trap laid by the death wizard is
  *in* the storm wizard's rollout, and it cashes it — which is the actual
  hivemind play, and it falls straight out of planning against the same
  board instead of four copies of it.

  **Coordinate descent.** One pass is sequential-greedy and order
  dependent: whoever plans first never sees the others. So every seat
  re-plans against the other three's current commitments, twice by
  default, which is enough to settle the case that matters — two wizards
  who each open on the same mob, one of whom then finds the second mob
  cheaper. `passes=0` reproduces uncoordinated play exactly, which is
  what the A/B in `tests/test_hivemind.py` measures against.

  **An overkill guard.** A seat whose board is already lethal this round
  passes instead of firing into a corpse. Guarded by *expected* damage —
  each commitment is priced at accuracy × damage, so a 75%-accurate nuke
  never persuades three other wizards that a mob is dead.

  **Pressure through the horizon.** The ledger tells a seat what the
  party is doing *this* round. `_decide_one` also tells its rollout how
  fast the rest of the circle keeps clearing the board for the other
  eleven, via `policies.set_ally_rate`.

**What it buys.** Measured over 120 sampled planning phases per party
size -- boards of one to four mobs between 40 and 1,100 HP, mixed
schools, wizards drawn from four different school decks at 2-7 pips --
scoring both arms the same way by replaying every chosen move against
one shared board and counting the damage that lands past a mob's
health:

    wizards   past lethal, alone   as a party   casts the party changed
    2                        517            0                       34%
    3                      3,038           72                       49%
    4                      5,520          240                       52%

...plus 10, 42 and 73 casts held back entirely, out of 240, 360 and 480.
Half of a four-wizard party's casts are not the cast that wizard would
have made alone, and the damage thrown at corpses drops by 96-100%. The
residue at three and four wizards is real and is the accuracy discount
doing its job: a cast believed at 75% still leaves a sliver, and three
wizards can each add a legitimate share to a mob that then dies to the
first of them.

The last of those was measured and rejected once, and shipping it now is
a correction rather than a change of mind. `party_pressure_probe.py`
scored 18 cells on kill rate and turns-to-kill and found +0.74 points
for a party of two and +0.00 for a party of four -- but its boards were
"scaled for a party" and every four-wizard cell saturated at 100%, so
the only thing ally-awareness could move there was how *fast* an already
won fight was won. It never recorded the quantity that actually breaks:
whether `turns` carries any signal at all.

The first real two-wizard run against three 435hp Sand Stalkers made
that regime unmissable. Straight off its own exported candidate tables,
no reconstruction involved: in 28 of the 58 scored rounds not one
candidate killed inside the horizon, and 17 of the 58 played a card the
rollout scored bit-identical to passing.

The arithmetic behind it is in those tables too. Banked damage over the
horizon is the line's own throughput, so board health divided by it is
turns-to-clear alone. Of the 21 rounds where both wizards were scored
against the same board, 10 read "the party clears inside twelve turns
and neither wizard alone does":

    fight round   board   alone (ice / fire)   together
    1     3        1305   15.0 / 21.9          8.9
    1     4        1305   13.6 / 36.0          9.9
    2     4        1103   16.8 / 12.9          7.3
    4     2        1265   21.7 / 13.8          8.4

Fights 1 and 2 were both lost. Raising the horizon is not the
alternative: reconstructed at 40 seeded decks a cell, the sentinel rate
goes 82.6% at horizon 12 to 36.7% at 16 and stays at 36.7% at 20, 24, 32
and 40, because a wizard cannot out-damage the board alone however long
it looks.

The rate has to be honest in both directions. Too low and the sentinel
stays; too high and it returns for the opposite reason, since a party
modelled as overwhelming clears the board on turn one of every line and
`turns` ties again at 1. So it is measured (`Telemetry.damage_rate`, 51
and 57 a round for those two wizards) and not taken from the
coordinator's own priced plan, which reads 172 and 201 because it prices
a single cast and no wizard casts its best card every round. `minion_power` is still 0.0 and the allies still stand there doing
nothing: `_minion_turn` funnels every ally's damage into `enemies[0]`
whatever the ledger assigned, so the pressure is applied by
`policies._allies_hit` instead, weakest mob first.

Everything here is plain Python: no Qt, no wizwalker, no game. The
coordinator is driven by `WizAiBackend.coordinator`, and the policies it
calls are the ordinary `policy(sim, state)` callables, so a party plan is
testable headless and a party of one behaves exactly like no party at
all.
"""
import asyncio
import copy
import threading
import time

from dataclasses import dataclass, field

from . import policies


# --------------------------------------------------------------------------
# aligning four reads of one board
# --------------------------------------------------------------------------
def _same_enemy(a, b) -> bool:
    return a.name == b.name and abs(float(a.max_hp) - float(b.max_hp)) <= 1.0


def align_enemies(enemies, reference):
    """`[reference index for each of `enemies`]`, or None if they disagree.

    Four clients read the same duel, so the participant lists normally
    line up index for index and this is the identity. Normally is not
    always: a client that read a round later has already dropped a
    corpse, and one that failed a read has a shorter list. Index `i` then
    means a different mob to each seat, and a ledger keyed on it would
    credit one seat's damage against another seat's *other* mob — which
    is worse than not coordinating at all.

    So the positional answer is checked rather than assumed, with a
    name-and-health match as the fallback and `None` — "do not
    coordinate this seat" — as the honest last resort.
    """
    if not reference:
        return None
    if len(enemies) == len(reference) and \
            all(_same_enemy(e, r) for e, r in zip(enemies, reference)):
        return list(range(len(enemies)))

    out, used = [], set()
    for e in enemies:
        found = None
        for tier in (0, 1):
            for j, r in enumerate(reference):
                if j in used:
                    continue
                if (_same_enemy(e, r) if tier == 0 else r.name == e.name):
                    found = j
                    break
            if found is not None:
                break
        if found is None:
            return None
        used.add(found)
        out.append(found)
    return out


# --------------------------------------------------------------------------
# what one seat's move does to the shared board
# --------------------------------------------------------------------------
@dataclass
class CastEffect:
    """Everything one cast does to the *enemy* side, keyed by index.

    Enemy side only, deliberately. A blade goes on the caster and no
    other wizard can use it; a trap goes on the mob and every wizard in
    the circle cashes it. Sharing the first would be a lie, and sharing
    the second is the whole point.
    """

    card: str = ""
    target: int = None
    #: expected damage per enemy index: what the cast does, priced at its
    #: accuracy. A cast the party plans around has to be discounted by
    #: the odds it lands, or one fizzle strands three other wizards.
    damage: dict = field(default_factory=dict)
    #: hangings the cast leaves on an enemy (traps, prisms, shields,
    #: Feint's backlash) — the currency of party setup
    wards: dict = field(default_factory=dict)
    #: scheduled damage left ticking on an enemy
    over_time: dict = field(default_factory=dict)
    accuracy: float = 1.0
    #: raw damage before the accuracy discount, for the readout
    raw_damage: float = 0.0

    @property
    def total(self):
        return sum(self.damage.values())

    def empty(self):
        return not self.damage and not self.wards and not self.over_time


def _split(action):
    """(card, target) out of whatever a policy returned."""
    if isinstance(action, tuple):
        return action[0], (action[1] if len(action) > 1 else 0)
    return action, 0


def cast_accuracy(state, card):
    """The odds this cast lands, gear included.

    Buffs do not fizzle, and pricing a Fireblade at 75% would make the
    party plan around a certainty as though it were a coin flip.
    """
    if getattr(card, "kind", "") not in ("damage", "drain"):
        return 1.0
    base = float(getattr(card, "accuracy", 1.0) or 1.0)
    bonus = float(getattr(state.player, "accuracy_bonus", 0.0) or 0.0)
    return max(0.0, min(1.0, base + bonus))


def measure_cast(sim, state, action):
    """What `action` would do to the board, on a copy, without fizzling.

    Runs the engine's own cast path — the same thing `predict_damage`
    does for one number — and diffs the whole enemy side, so charms,
    wards, prisms, absorbs, AoE splash and any DoT the card schedules are
    all accounted for by the engine rather than guessed at here.
    """
    from .telemetry import _NoFizzle

    card, target = _split(action)
    if card is None:
        return CastEffect()
    name = getattr(card, "name", None)
    if not isinstance(name, str):
        return CastEffect()

    try:
        s = copy.deepcopy(state)
        probe = copy.copy(sim)
        probe.rng = _NoFizzle()
        target = max(0, min(int(target or 0), len(s.enemies) - 1))
        played = None
        for c in s.hand:
            if c.name == name and probe.can_cast(s, c, target):
                played = c
                break
        if played is None:
            return CastEffect(card=name, target=target)

        before_hp = [float(e.hp) for e in s.enemies]
        before_wards = [len(e.wards) for e in s.enemies]
        before_ot = [len(e.over_time) for e in s.enemies]
        probe.cast(s, played, target)

        acc = cast_accuracy(state, played)
        effect = CastEffect(card=name, target=target, accuracy=acc)
        for i, enemy in enumerate(s.enemies):
            # Floored at what the mob had: damage past lethal is not
            # damage the party can plan around, and crediting it would
            # let one oversized nuke read as though it covered the board.
            lost = min(before_hp[i] - float(enemy.hp), max(before_hp[i], 0.0))
            if lost > 0:
                effect.raw_damage += lost
                effect.damage[i] = lost * acc
            new_wards = enemy.wards[before_wards[i]:]
            if new_wards:
                effect.wards[i] = list(new_wards)
            new_ot = enemy.over_time[before_ot[i]:]
            if new_ot:
                effect.over_time[i] = list(new_ot)
        return effect
    except Exception:
        # A commitment that cannot be priced is simply not shared. The
        # seat still plays its move; the others just do not plan around
        # it, which is the uncoordinated behaviour and never worse.
        return CastEffect(card=name, target=target)


#: How sure the party has to be that a cast lands before it abandons the
#: mob that cast is aimed at. Below this the mob keeps a sliver of health
#: in the shared board and the next wizard finishes it.
#:
#: A judgement, and the arithmetic behind it: believing a kill costs one
#: round of that mob's damage when the cast fizzles, at probability
#: (1 - accuracy). Not believing it costs a whole wizard's turn *every*
#: round, spent finishing something already dead. At 75% -- the accuracy
#: of a level-5 fire hit, the worst commonly cast -- the first is a
#: quarter of a round against the second's whole one, so the party is
#: right to believe it. At 60% the two are close enough that the safe
#: answer wins.
KILL_CONFIDENCE = 0.7


class Ledger:
    """The party's commitments this round, in reference-board indices."""

    def __init__(self, kill_confidence=KILL_CONFIDENCE):
        #: expected damage, i.e. discounted by the odds the cast lands
        self.damage = {}
        #: what it does if it lands, which is what decides lethality
        self.raw = {}
        #: the least confident cast aimed at each mob
        self.confidence = {}
        self.wards = {}
        self.over_time = {}
        self.casts = 0
        self.kill_confidence = kill_confidence

    def add(self, effect, mapping):
        """Fold one seat's `CastEffect` in, translated to reference indices."""
        if effect is None or mapping is None or effect.empty():
            return
        self.casts += 1
        for i, dmg in effect.damage.items():
            if i >= len(mapping):
                continue
            j = mapping[i]
            self.damage[j] = self.damage.get(j, 0.0) + dmg
            self.raw[j] = self.raw.get(j, 0.0) + (
                dmg / effect.accuracy if effect.accuracy else dmg)
            self.confidence[j] = min(self.confidence.get(j, 1.0),
                                     effect.accuracy)
        for src, dest in ((effect.wards, self.wards),
                          (effect.over_time, self.over_time)):
            for i, items in src.items():
                if i < len(mapping):
                    dest.setdefault(mapping[i], []).extend(items)

    def apply(self, state, mapping, margin=1.0):
        """A copy of `state` with the rest of the party's round in it.

        Health first. A mob the party's committed casts kill *if they
        land* is written down to zero rather than merely reduced, so
        `Actor.alive` goes false and the next wizard re-aims instead of
        spending its turn finishing something already finished. That is
        gated on `KILL_CONFIDENCE`, because "if they land" is doing real
        work: a 51-damage hit into a 50 HP mob is lethal three times in
        four, and pretending otherwise is how four wizards end up putting
        four Fire Cats into one Lost Soul.

        Anything short of lethal comes off as *expected* damage --
        discounted by accuracy -- so the next wizard plans against the
        health the mob will probably have, not the health it would have
        in the luckiest case.

        Then the hangings, deep-copied per seat: a `Hanging` carries
        charges and is consumed in place, so handing the same object to
        four rollouts would let one seat's hit spend a trap the other
        three are still counting on.
        """
        if mapping is None or (not self.damage and not self.wards
                               and not self.over_time):
            return state
        out = copy.deepcopy(state)
        for i, enemy in enumerate(out.enemies):
            if i >= len(mapping):
                continue
            j = mapping[i]
            hp = float(enemy.hp)
            raw = self.raw.get(j, 0.0) * margin
            if (raw >= hp > 0
                    and self.confidence.get(j, 1.0) >= self.kill_confidence):
                enemy.hp = 0.0
            else:
                dealt = self.damage.get(j, 0.0) * margin
                if dealt:
                    enemy.hp = max(0.0, hp - dealt)
            for h in self.wards.get(j, ()):
                enemy.wards.append(copy.deepcopy(h))
            for o in self.over_time.get(j, ()):
                enemy.over_time.append(copy.deepcopy(o))
        return out


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------
@dataclass
class SeatMove:
    """One seat's share of the round, and why it is not what it would
    have played alone."""

    seat: int
    name: str = ""
    card: str = ""
    target: int = None
    target_name: str = ""
    #: what this seat would have cast with no party around it
    solo_card: str = ""
    solo_target: int = None
    damage: float = 0.0
    #: {enemy name: expected damage} — every mob this cast takes health
    #: off, not just the one it was aimed at. `target_name` cannot
    #: answer that: an AoE aims at nothing in particular, so
    #: `_aims_at_an_enemy` gives it no index and no name, and the
    #: telemetry that keys off the name then cannot see it at all. The
    #: run at rev 8666bda7 is what that costs -- every one of the five
    #: rounds where the fire wizard cast Meteor Strike reported an
    #: empty `party_hits` on the ice wizard's side, so his residual
    #: silently absorbed the whole AoE. One round recorded 987 damage
    #: against a 115 prediction and was filed as clean.
    hit: dict = field(default_factory=dict)
    note: str = ""

    @property
    def retargeted(self):
        return (self.card, self.target) != (self.solo_card, self.solo_target)


@dataclass
class PartyPlan:
    """The whole round, as one object the GUI can render."""

    round_number: int = 0
    moves: list = field(default_factory=list)
    #: reference board, as (name, hp, max_hp) — the mobs the plan is about
    board: list = field(default_factory=list)
    passes: int = 0
    seconds: float = 0.0
    #: seats told to hold their card because the board was already dead
    saved: int = 0
    #: seats whose move changed once they could see the others
    retargets: int = 0

    def summary(self):
        if not self.moves:
            return "no party round planned yet"
        bits = [f"{len(self.moves)} wizard(s)"]
        if self.retargets:
            bits.append(f"{self.retargets} re-aimed")
        if self.saved:
            bits.append(f"{self.saved} held (board already lethal)")
        bits.append(f"{self.seconds * 1000:.0f} ms")
        return " · ".join(bits)


@dataclass
class _Submission:
    seat: int
    name: str
    sim: object
    state: object
    policy: object
    read: object = None
    mapping: list = None
    #: enemy health a round this seat removes, measured. What every
    #: OTHER seat's rollout needs in order to model a party fight rather
    #: than a solo one -- see `policies.set_ally_rate`.
    rate: float = 0.0


class _Gather:
    """One round's barrier: who is expected, who has arrived, the plan."""

    def __init__(self, expected):
        self.expected = set(expected)
        self.subs = {}
        self.waiters = {}          # seat -> (loop, asyncio.Event)
        self.plan = {}             # seat -> action
        self.party = None          # PartyPlan
        self.closing = False
        self.done = False

    def full(self):
        return self.expected and set(self.subs) >= self.expected


class Hivemind:
    """Joint planning for 1-4 wizards in one battle circle.

    A seat joins when its worker connects and enters combat, and leaves
    when it stops or its duel ends. Only seats that are *in a fight* are
    waited for, so a wizard still walking to the circle never stalls the
    three that are already swinging.

    Thread-safe on purpose: each `LiveWorker` owns a client on its own
    thread with its own event loop, so the barrier cannot be an
    `asyncio` primitive — those bind to one loop. The state is guarded by
    a plain lock held only for bookkeeping (never across the planning
    itself, which would block another seat's whole event loop), and a
    waiting seat is woken through `call_soon_threadsafe` onto its own
    loop.
    """

    #: how long a seat waits for the rest of the party before planning
    #: with whoever turned up. A planning phase is ~30 s, so this is
    #: bought cheaply; it only has to outlast the jitter between four
    #: clients reaching the same round.
    TIMEOUT = 2.5

    def __init__(self, timeout=None, passes=2, overkill_guard=True,
                 margin=1.0, budget=6.0, on_status=None, on_plan=None,
                 kill_confidence=KILL_CONFIDENCE):
        #: coordination passes after the solo baseline. 0 reproduces
        #: uncoordinated play exactly, which is what the A/B measures
        #: against; 2 settles the two-wizards-one-mob case.
        self.passes = max(0, int(passes))
        self.timeout = self.TIMEOUT if timeout is None else float(timeout)
        self.overkill_guard = bool(overkill_guard)
        #: scales committed damage before it is believed. Below 1 the
        #: party under-commits (more overlap, fewer survivors); above 1
        #: it over-commits. 1.0 is "believe the expected damage".
        self.margin = float(margin)
        #: see `KILL_CONFIDENCE`
        self.kill_confidence = float(kill_confidence)
        #: seconds of planning per round before the remaining passes are
        #: abandoned. A party plan that outlasts the game's own timer
        #: would lose the round it was trying to save.
        self.budget = float(budget)
        self.on_status = on_status
        self.on_plan = on_plan

        self._lock = threading.Lock()
        self._seats = {}           # seat -> name
        self._fighting = set()
        self._gather = None
        self.last_plan = None
        #: seat -> its share of the most recent plan. Per seat rather
        #: than read off `last_plan`, because a fast client can open the
        #: next round's gather before a slow one has read the last one.
        self._moves = {}
        #: lifetime counters, for the Party panel
        self.rounds = 0
        self.saved = 0
        self.retargets = 0
        self.solo_rounds = 0

    # -- membership -------------------------------------------------------
    def join(self, seat, name=""):
        with self._lock:
            self._seats[seat] = name or f"wizard {seat + 1}"
        return seat

    def leave(self, seat):
        with self._lock:
            self._seats.pop(seat, None)
            self._fighting.discard(seat)
            gather = self._gather
        # Whoever is left must not keep waiting for a seat that has gone.
        if gather is not None and not gather.done:
            self._maybe_close(gather)

    def enter_combat(self, seat):
        with self._lock:
            self._fighting.add(seat)

    def leave_combat(self, seat):
        with self._lock:
            self._fighting.discard(seat)
            gather = self._gather
        if gather is not None and not gather.done:
            self._maybe_close(gather)

    @property
    def size(self):
        with self._lock:
            return len(self._seats)

    def fighting(self):
        with self._lock:
            return sorted(self._fighting)

    def last_move(self, seat):
        """This seat's share of the most recent plan, or None."""
        with self._lock:
            return self._moves.get(seat)

    def _say(self, message):
        if self.on_status:
            try:
                self.on_status(message)
            except Exception:
                pass

    # -- the barrier ------------------------------------------------------
    async def decide(self, seat, sim, state, policy, read=None, rate=0.0):
        """This seat's move for this round, agreed with the rest of the party.

        Returns whatever a policy returns — a `(card, target)` pair, a
        card, or None for a pass — so the caller cannot tell a
        coordinated decision from a solo one and needs no new branch.

        `rate` is enemy health this seat removes per round, measured.
        It is not used for this seat's own decision at all; it is what
        the OTHER seats' rollouts are told about this one. Zero -- the
        default, and what a caller with nothing to report should send --
        leaves the party modelled exactly as it was before there was a
        rate at all.
        """
        gather, wait = self._submit(seat, sim, state, policy, read, rate)
        if wait is not None:
            _loop, event = wait
            deadline = time.monotonic() + self.timeout
            extended = False
            while not gather.done:
                left = deadline - time.monotonic()
                if left > 0:
                    try:
                        await asyncio.wait_for(event.wait(), left)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break
                    continue
                if gather.closing and not extended:
                    # Somebody is already planning. The timeout is for
                    # the party turning up, not for the arithmetic, so
                    # it is extended by the planning budget -- once,
                    # because a policy that hangs inside the plan still
                    # has to give this seat its round back.
                    extended = True
                    deadline = time.monotonic() + self.budget + 1.0
                    continue
                if not gather.closing:
                    self._close(gather)
                break
        if not gather.done:
            self._close(gather)
        if seat not in gather.plan:
            # Only reachable when another seat claimed the plan and never
            # finished it. Passing is the safe move, but it is said out
            # loud: a wizard silently passing every round looks like a
            # broken policy, and the cause is somewhere else entirely.
            self._say(f"wizard {seat + 1} passed — the party's plan for this "
                      f"round never came back")
        return gather.plan.get(seat)

    def _submit(self, seat, sim, state, policy, read, rate=0.0):
        """Put this seat in the open gather. Returns (gather, wait-or-None)."""
        stale = None
        with self._lock:
            gather = self._gather
            if gather is None or gather.done or gather.closing \
                    or seat in gather.subs:
                # A seat coming round again means the previous round is
                # over for it — plan what is sitting there rather than
                # merging two rounds into one board.
                if gather is not None and not gather.done \
                        and not gather.closing:
                    stale = gather
                    gather.closing = True
                expected = set(self._fighting) or {seat}
                gather = self._gather = _Gather(expected)
            gather.expected.add(seat)
            gather.subs[seat] = _Submission(
                seat, self._seats.get(seat, f"wizard {seat + 1}"),
                sim, state, policy, read, rate=max(0.0, float(rate or 0.0)))
            if gather.full():
                gather.closing = True
                close_now = True
                wait = None
            else:
                close_now = False
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                event = asyncio.Event()
                gather.waiters[seat] = (loop, event)
                wait = (loop, event)

        if stale is not None:
            self._plan_gather(stale)
        if close_now:
            self._plan_gather(gather)
        return gather, (None if gather.done else wait)

    def _maybe_close(self, gather):
        """Close a gather whose remaining seats have left the fight."""
        with self._lock:
            if gather.done or gather.closing or not gather.subs:
                return
            if not gather.full() and \
                    not (set(gather.subs) >= (gather.expected
                                              & set(self._fighting))):
                return
            gather.closing = True
        self._plan_gather(gather)

    def _close(self, gather):
        with self._lock:
            if gather.done or gather.closing:
                return
            gather.closing = True
        self._plan_gather(gather)

    def _plan_gather(self, gather):
        """Run the joint plan, then wake everyone waiting on it.

        Called with no lock held: planning is the expensive part, and
        holding the lock through it would block another seat's event
        loop rather than just its planning phase.
        """
        try:
            plan, party = self.plan(
                [gather.subs[s] for s in sorted(gather.subs)])
        except Exception as exc:
            # A party plan is an optimisation. If it raises, every seat
            # falls back to the move it would have made alone — the
            # fight is never dropped over coordination.
            self._say(f"party planning failed — {type(exc).__name__}: {exc}; "
                      f"each wizard is deciding for itself")
            plan, party = self._solo_fallback(gather)
        gather.plan = plan
        gather.party = party
        with self._lock:
            gather.done = True
            waiters = list(gather.waiters.values())
            gather.waiters.clear()
            if self._gather is gather:
                self._gather = None
            self.last_plan = party
            if party is not None:
                for move in party.moves:
                    self._moves[move.seat] = move
                self.rounds += 1
                self.saved += party.saved
                self.retargets += party.retargets
                if len(party.moves) <= 1:
                    self.solo_rounds += 1
        if party is not None and self.on_plan:
            try:
                self.on_plan(party)
            except Exception:
                pass
        for loop, event in waiters:
            try:
                if loop is not None and loop.is_running():
                    loop.call_soon_threadsafe(event.set)
                else:
                    event.set()
            except Exception:
                pass

    def _solo_fallback(self, gather):
        plan = {}
        for seat, sub in gather.subs.items():
            try:
                plan[seat] = sub.policy(sub.sim, sub.state)
            except Exception:
                plan[seat] = None
        return plan, None

    # -- the joint plan ---------------------------------------------------
    def plan(self, subs):
        """({seat: action}, PartyPlan) for one round of the whole party.

        Pass 0 is every seat deciding alone — the uncoordinated answer,
        kept as the baseline the readout compares against. Each pass
        after it re-plans every seat against the *other* seats' current
        commitments (leave-one-out, so it is coordinate descent and not
        an order-dependent sweep), and stops early the moment nothing
        moves.
        """
        started = time.monotonic()
        if not subs:
            return {}, PartyPlan()

        reference = max(subs, key=lambda s: len(s.state.enemies))
        ref_enemies = list(reference.state.enemies)
        for sub in subs:
            sub.mapping = (list(range(len(sub.state.enemies)))
                           if sub is reference
                           else align_enemies(sub.state.enemies, ref_enemies))

        actions, effects, solo = {}, {}, {}
        #: enemy health a round, per seat: what everyone ELSE's rollout
        #: is told about that seat. Measured where the seat could report
        #: one, and its own pass-0 rollout's throughput where it could
        #: not -- see `policies.rollout_throughput`.
        rates = {}
        notes = {}
        saved = 0
        passes_run = 0

        for step in range(self.passes + 1):
            changed = False
            for sub in subs:
                ledger = Ledger(self.kill_confidence)
                if step > 0:
                    for other in subs:
                        if other.seat != sub.seat:
                            ledger.add(effects.get(other.seat),
                                       other.mapping)
                board = ledger.apply(sub.state, sub.mapping, self.margin)
                # Pass 0 is left genuinely solo: it is the baseline the
                # readout reports as "alone it would have played X", and
                # a baseline that already knows about the party is not a
                # baseline. It is also where each seat's cold-start
                # throughput estimate comes from.
                pressure = 0.0 if step == 0 else sum(
                    rates.get(o.seat, 0.0) for o in subs if o.seat != sub.seat)
                action, note = self._decide_one(sub, board, ledger, step,
                                                pressure)
                if step == 0:
                    solo[sub.seat] = action
                    rates[sub.seat] = sub.rate or policies.rollout_throughput(
                        getattr(sub.policy, "last_candidates", ()))
                if step and _move_id(action) != _move_id(actions.get(sub.seat)):
                    # `or`, not `=`: seat 3 settling must not erase the
                    # fact that seat 1 moved, or the descent stops one
                    # pass after the first seat that agrees with itself.
                    changed = True
                actions[sub.seat] = action
                notes[sub.seat] = note
                effects[sub.seat] = measure_cast(sub.sim, board, action)
            passes_run = step
            if step and not changed:
                break
            if time.monotonic() - started > self.budget:
                if step < self.passes:
                    self._say(
                        f"party planning stopped after {step + 1} pass(es) — "
                        f"over the {self.budget:.0f}s budget")
                break

        moves, retargets = [], 0
        for sub in subs:
            action = actions.get(sub.seat)
            card, target = _split(action)
            base_card, base_target = _split(solo.get(sub.seat))
            effect = effects.get(sub.seat) or CastEffect()
            # An index only means something for a card that aims at one
            # enemy. `_split` gives a bare action target 0, so without
            # this every self-cast was labelled with `enemies[0]`.
            aims = card is not None and _aims_at_an_enemy(card)
            base_aims = base_card is not None and _aims_at_an_enemy(base_card)
            # By name rather than by index, because the readers are other
            # seats and an index only means anything on the board it was
            # read from. Summed per name, which conflates same-named mobs
            # -- three O'Leary Nappers on one board are one key -- but
            # that is what `_party_hits` and `_party_expected` already do
            # with `target_name`, so this changes nothing about it.
            hit = {}
            for i, dmg in (effect.damage or {}).items():
                if not dmg:
                    continue
                nm = _enemy_name(sub.state, i)
                if nm:
                    hit[nm] = hit.get(nm, 0.0) + float(dmg)
            move = SeatMove(
                seat=sub.seat, name=sub.name,
                card=getattr(card, "name", "") or "",
                target=(target if aims else None),
                target_name=_enemy_name(sub.state, target if aims else None),
                solo_card=getattr(base_card, "name", "") or "",
                solo_target=(base_target if base_aims else None),
                damage=effect.total, hit=hit, note=notes.get(sub.seat, ""))
            if move.retargeted:
                retargets += 1
            if notes.get(sub.seat) == "held":
                saved += 1
            moves.append(move)

        party = PartyPlan(
            round_number=int(getattr(
                getattr(reference, "read", None), "round_number", 0) or 0),
            moves=moves,
            board=[(e.name, float(e.hp), float(e.max_hp))
                   for e in ref_enemies],
            passes=passes_run, seconds=time.monotonic() - started,
            saved=saved, retargets=retargets)
        return actions, party

    def _decide_one(self, sub, board, ledger, step, pressure=0.0):
        """One seat's move against the board the party has left it.

        Two different questions get answered here, and only the first
        used to be. `Ledger.apply` has already settled "what is left for
        me *this* round"; `pressure` settles "and how fast is the rest
        of the circle clearing it", which is what the rollout needs in
        order to see past the next twelve turns.

        Without the second a seat plans every fight as though it were
        alone in it. Against the three 435hp Sand Stalkers of the first
        live two-wizard run that meant no candidate could kill inside
        the horizon in 28 of 58 scored rounds -- the sentinel for every
        one of them, `turns` tied, and the whole decision handed to a
        tiebreak that rates a Tower Shield exactly as highly as doing
        nothing. It played one six times.

        `pressure` is deliberately NOT the total of `ledger.damage`,
        which is sitting right here and is the obvious thing to reach
        for. The ledger prices one cast this round; a rate has to hold
        for twelve.
        Over the day's run the ledger's number averages 172 and 201 a
        round against realised rates of 51 and 57 -- 3.2x high, which
        would tell the rollout the 1305hp board dies in five turns when
        it really takes twelve, and stop it buying either setup or
        survival on precisely the boards where both wizards died.
        """
        with policies.party_rate(pressure):
            return self._decide_alone(sub, board, ledger, step)

    def _decide_alone(self, sub, board, ledger, step):
        if step and self.overkill_guard and ledger.casts:
            alive_now = any(e.alive for e in board.enemies)
            alive_before = any(e.alive for e in sub.state.enemies)
            if alive_before and not alive_now:
                # Every mob is spoken for. Firing into it buys nothing
                # and spends a card the next fight will want.
                policy = sub.policy
                if hasattr(policy, "last_candidates"):
                    # ...and the decision panel must not show the
                    # comparison from the pass that was thrown away.
                    policy.last_candidates = []
                return None, "held"
        try:
            return sub.policy(sub.sim, board), ("party" if step else "solo")
        except Exception as exc:
            return None, f"policy raised {type(exc).__name__}: {exc}"


def _move_id(action):
    card, target = _split(action)
    return (getattr(card, "name", None), target if card is not None else None)


def _aims_at_an_enemy(card) -> bool:
    """Does an enemy index mean anything for this card?

    A heal, a blade and an aura do not aim at a mob, and `_split`
    defaults a bare action's target to 0 -- so labelling every move with
    `enemies[0]` put a mob's name on a self-cast. Live that read
    "Pixie @Sokkwi Ripper" on the party plan, which is what a heal aimed
    at a monster looks like from the outside. The cast itself went to
    the caster (`WizAiCombatHandler._resolve_target` reads the card, not
    the plan), so this was a lie about a correct action -- but it also
    fed `_party_expected`, which sums a round's expected damage BY
    TARGET NAME and was therefore crediting a heal to a mob.
    """
    try:
        from .policies import aimed_at_one_enemy

        return bool(aimed_at_one_enemy(card))
    except Exception:
        return False


def _enemy_name(state, index):
    if index is None:
        return ""
    try:
        return state.enemies[index].name
    except Exception:
        return ""
