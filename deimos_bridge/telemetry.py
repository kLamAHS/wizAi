"""What a live run has to record for the ML work to be worth anything.

Deimos's own GUI is built for a bot operator: is it questing, is it
stuck, how many hours has it run. None of that is what you need while
training or evaluating a policy. This module is the data behind a GUI
built for the other job, and it is deliberately Qt-free so the
interesting parts are testable without a display.

Four things get recorded, in rough order of how often they are the reason
a run was worthless:

  1. **What the policy was shown.** A policy that never saw a card cannot
     play it, and a card that failed to resolve is invisible rather than
     loud. `RoundRecord.unresolved` and the board snapshot make that
     checkable per round instead of after the fact.

  2. **What it decided, and why.** The chosen card, the target, the
     stated reason, and the legal alternatives it passed over.

  3. **Whether the simulator was right.** Before each cast, wizAi
     predicts the damage; the following round the target's real HP says
     what actually happened. The residual between them is the honest
     answer to "how does my model do against real enemies", and it is
     the one number that cannot be obtained from simulation alone.

  4. **How the fight went.** Rounds, damage dealt and taken, outcome.

On (3), a caveat that belongs in the data rather than in a footnote: an
HP delta measured across a round boundary also contains DoT ticks, minion
hits, and anything the enemy healed. `DamageObservation.clean` marks the
rounds where no such confound was detectable, and the error statistics
are reported over clean observations by default.

Also on (3): *which mob* the delta is measured on is not obvious, and
getting it wrong is silent. See `match_enemies` -- a name lookup was
differencing one mob's health against its identically-named twin's, and
a prediction that was right to within half a point was being reported as
a 634% error.
"""
import copy
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field


# --------------------------------------------------------------------------
# damage prediction
# --------------------------------------------------------------------------
class _NoFizzle(random.Random):
    """An RNG that always lands the cast and never crits.

    `Sim.cast` rolls accuracy with `rng.random() > acc` and crit the same
    way, so a stream of zeros means "this spell hits, without a critical".
    That is the right thing to predict: the residual against a real cast
    should measure the *damage model*, not whether this particular cast
    happened to fizzle. Fizzles are recorded separately.
    """

    def random(self):
        return 0.0

    def uniform(self, a, b):
        return (a + b) / 2.0

    def choice(self, seq):
        seq = list(seq)
        return seq[len(seq) // 2]


def predict_damage(sim, state, card, target_index=0):
    """What wizAi expects this cast to do to that target, if it lands.

    Runs the real cast path on a deep copy, so every rule the engine has
    -- charms, wards, prisms, absorbs, link groups -- is applied exactly
    as it would be in a duel, and nothing touches the caller's state.

    Returns None if the cast cannot be predicted (not castable, no such
    target), rather than a misleading zero.
    """
    try:
        s = copy.deepcopy(state)
        probe = copy.copy(sim)
        probe.rng = _NoFizzle()
        # the copied card is the one in the copied hand
        target_index = max(0, min(target_index, len(s.enemies) - 1))
        for c in s.hand:
            if c.name == card.name and probe.can_cast(s, c, target_index):
                before = s.enemies[target_index].hp
                probe.cast(s, c, target_index)
                after = s.enemies[target_index].hp
                return max(before - after, 0.0)
        return None
    except Exception:
        return None


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
def describe_threat(e) -> str:
    """How the rollouts modelled a mob's offence, for the Board panel.

    Two models exist since `_apply_pool`: a catalog caster (spell pool
    plus the pips read off the client) and the measured flat hit. Which
    one priced the fight decides whether the policy shields before the
    spike or races through the drizzle -- and neither the model nor the
    enemy's pip rack was visible anywhere in the window.
    """
    norm = int(getattr(e, "norm_pips", 0) or 0)
    pow_ = int(getattr(e, "pow_pips", 0) or 0)
    rack = f"{norm}+{pow_}p" if pow_ else f"{norm}p"
    pool = getattr(e, "spell_pool", None)
    if pool:
        head = ", ".join(pool[:3]) + ("…" if len(pool) > 3 else "")
        return f"{rack} · casts {head}"
    hit = float(getattr(e, "flat_hit", 0) or 0)
    if hit:
        return f"{rack} · ~{hit:,.0f}/round"
    return rack


def describe_hanging(h) -> str:
    """A hanging effect as something a human can act on.

    Effects read off the live client have no name -- the participant
    carries a spell template id, not a string -- so `read_hangings` names
    them `live:<id>`. That is the right identity for stacking but is
    useless on screen: "live:1004" says nothing about whether the policy
    is looking at a Fireblade or a Tower Shield. What matters for
    debugging is the arithmetic, so show that.
    """
    schools = ("all" if not h.schools else "/".join(sorted(h.schools)))
    if h.kind == "prism":
        return f"prism -> {h.convert_to or '?'}"
    if h.kind == "absorb":
        return f"absorb {h.amount:,.0f}"
    if h.kind == "damage":
        return f"{h.percent * 100:+.0f}% {schools}"
    if h.kind == "accuracy":
        return f"{h.percent * 100:+.0f}% acc {schools}"
    return f"{h.name} ({h.kind})"


def _round_id(rec) -> str:
    """"f6r8" -- which fight, which round."""
    return f"f{getattr(rec, 'fight', 0)}r{getattr(rec, 'round', 0)}"


def revision() -> str:
    """The git revision this ran on, or "".

    Stamped into every export. Two runs uploaded twenty minutes before
    a fix landed were read as evidence about the fixed code, and there
    was nothing in the file to say otherwise -- a measurement whose
    provenance is a guess is not a measurement.
    """
    import os
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(["git", "-C", root, "describe", "--always",
                              "--dirty", "--abbrev=8"],
                             capture_output=True, text=True, timeout=5)
    except Exception:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _party_expected(party, target_name):
    """What the WHOLE party expects to take off `target_name` this round.

    The answer to a measurement problem that a party creates and cannot
    escape. Settling one wizard's cast means differencing one mob's
    health across a round -- and when two wizards fire into the same
    mob, that delta is both of theirs and neither's. The honest thing
    is to refuse the observation, which is what happens; the cost is
    that in a real party most rounds are shared, so refusing them
    starves the damage model to nothing. Over one live run it left
    wizard 2 with two usable rounds out of seventy.

    So the shared rounds are measured as what they actually are: a
    claim by the party about a mob, checked against what the mob lost.
    Every seat's expected damage on that mob, summed. It answers a
    slightly different question from the solo residual -- it cannot say
    which wizard was wrong -- but it is the same arithmetic under test,
    it is available on nearly every round of a party run, and a number
    from thirty rounds beats a number from two.

    Returns None when the coordinator did not price the round, so a
    missing plan is never confused with a plan that expects nothing.

    Read off `SeatMove.hit`, which names every mob a cast takes health
    off, rather than off `target_name`, which names only the one it was
    aimed at. An AoE is aimed at nothing in particular and so carries no
    name at all, and it was contributing zero here -- the party claim on
    a round somebody cast Meteor Strike was the other seats' damage
    only.
    """
    moves = getattr(party, "moves", None)
    if not moves or not target_name:
        return None
    total = 0.0
    for move in moves:
        hit = getattr(move, "hit", None)
        if hit:
            total += float(hit.get(target_name, 0.0) or 0.0)
        elif move.target_name == target_name:
            total += float(getattr(move, "damage", 0) or 0)
    return total


def _party_hits(party, seat):
    """{enemy name: "wizard 2", ...} for the OTHER wizards' DAMAGING casts.

    Only enemies, and only other seats: this wizard's own cast is the
    thing being measured, and a teammate's self-buff moves no health.

    And only casts that move health, which is the whole point and was
    the bug. Any card *aimed at* an enemy used to count -- and a trap,
    a shield and a debuff are all aimed at an enemy while taking
    nothing off it. Over one live party run that filed a false "the
    other wizard also hit this mob" on 18 of Konstantin's 64 rounds
    (Jeffrey had cast Ice Trap or Tower Shield) and 24 of Jeffrey's
    (Konstantin had cast Fireblade, Fire Trap or Pixie). Each one marks
    the round unclean, so it is not a cosmetic mislabel: it was the
    single biggest reason the damage model came back with two usable
    observations out of seventy.

    `SeatMove.damage` is the coordinator's own estimate for that move,
    which is exactly the question -- a card it scored at zero moves no
    health and cannot be the reason this wizard's residual is off.

    And EVERY mob it moves health on, via `SeatMove.hit`. Keying off
    `target_name` missed an AoE completely: `_aims_at_an_enemy` gives a
    card that hits the whole board no index and no name, so a teammate's
    Meteor Strike registered as nobody hitting anything. In the run at
    rev 8666bda7 that is all five Meteor Strike rounds, and on one of
    them the ice wizard's Snow Serpent was credited with 987 damage
    against a 115 prediction -- the fire wizard's AoE, banked as a clean
    solo observation. It is the reverse of the trap-and-shield bug above
    and it is worse: that one refused good rounds, this one accepts bad
    ones.
    """
    out = {}
    for move in getattr(party, "moves", ()) or ():
        if move.seat == seat or not move.card:
            continue
        hit = getattr(move, "hit", None)
        if not hit:
            # a plan from before `hit` existed, or a cast that moved no
            # health at all
            if not move.target_name or not (getattr(move, "damage", 0) or 0) > 0:
                continue
            hit = {move.target_name: move.damage}
        for name, dmg in hit.items():
            if not name or not (dmg or 0) > 0:
                continue
            who = out.get(name)
            out[name] = f"{who} and {move.name}" if who else move.name
    return out


def _attr(obj, name, default=None):
    """Field off a dataclass or a dict, whichever the caller has.

    Candidates arrive as `policies.Candidate` live and as plain dicts
    when a run is reloaded from JSON, and both have to render.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def outcome_of(cand) -> str:
    """"" for a real turn count, or what the sentinel actually means.

    `_rollout` encodes three non-outcomes above its horizon: `+1` ran out
    of horizon alive, `+2` the wizard died, `+3` the move could not be
    made. The panel rendered all three as turn counts, so a board where
    no line survives showed every candidate -- including `pass` -- at a
    flat "14 turns", which reads as a tie between equally good options
    when it means the opposite. A sentinel drawn as a number is a lie,
    and this is the string that stops it being one.
    """
    horizon = int(_attr(cand, "horizon", 12) or 12)
    turns = float(_attr(cand, "turns", 0.0) or 0.0)
    if turns >= horizon + 3:
        return "unplayable"
    if turns >= horizon + 2:
        return "dies"
    if turns >= horizon + 1:
        return "no clear"
    return ""


def turns_label(cand, unit=" turns") -> str:
    """What to print in a cell for one candidate's score."""
    why = outcome_of(cand)
    if why:
        return why
    return f"{float(_attr(cand, 'turns', 0.0)):g}{unit}"


def _move_label(cand) -> str:
    """'Ice Trap → 1', the identity a decision matrix column needs.

    The target is part of the move, not decoration: on a two-mob board
    the same card aimed at each enemy scores differently, and collapsing
    them into one column would average away the entire targeting
    decision.
    """
    card = _attr(cand, "card", "?")
    target = _attr(cand, "target")
    return card if target is None else f"{card} → {target}"


def match_enemies(before, after):
    """`{index in before: the same mob in after, or None}`.

    Settling a round means differencing one mob's health across two
    reads, and until now that was done by NAME. Wizard101 boards are
    mostly *two of the same mob*: "Nirini Warrior" twice, "Lost Soul"
    three times. A name lookup built as `{e.name: e for e in enemies}`
    keeps the LAST of each name and `next(e for e in ... if e.name ==
    target)` takes the FIRST, so every such fight differenced one mob's
    before-health against a different mob's after-health.

    It is not a small error. Live, wizard 2 hit a full-health Nirini
    Warrior for a predicted 18.53 while a second Nirini sat at 259/395;
    the mob went 395 -> 376, so the prediction was right to within half
    a point -- and the round was recorded as 136.0 actual, a 634% model
    error, because 395 - 259 = 136. Three consecutive rounds recorded
    136, 117, 117 against real deltas of 19, 19, 19. Every damage-model
    number this app has ever reported for a duel with two same-named
    mobs was measured against the wrong mob.

    **Position is the primary key**, and health is only the tie-breaker.
    A first attempt at this matched on health instead -- exact health
    first, on the reasoning that a mob nobody touched is the easiest
    thing to recognise -- and that is wrong in precisely the case that
    matters. Two Ice Weavers at 395 each, hit one for 224: the board
    reads `[395, 395, 395]` then `[171, 395, 395]`, and matching the
    hit mob's before-health of 395 to an *untouched twin's* 395 records
    the cast as doing nothing. Over the second live party run that was
    nine rounds recorded as 0 damage against predictions of 93 to 412,
    and two more measured against the wrong twin. Re-matched by
    position, every one of the nine lands within 1-62% of its
    prediction and not one round gets worse.

    wizwalker reads participants in a stable order, so as long as the
    board has the same shape it had last round, the i-th mob IS the i-th
    mob. Health then only has to sanity-check it: health falls within a
    fight, so a positional pairing that has a mob GAINING health is not
    a pairing, and the fallback takes over.

    When the shape changed -- something died, or something joined
    mid-duel, which Wizard101 allows -- position cannot be read off
    directly, but it is not gone: a death REMOVES an entry and leaves
    the rest in order, and an arrival INSERTS one. So the true pairing
    is still order-preserving, and the only question is which entries
    were skipped. `_monotone` searches for that exactly.

    A greedy nearest-health match was tried first and gets an ordinary
    board wrong. Three Sand Stalkers at `[211, 435, 201]`; the 211 is
    hit for a predicted 294 and dies, so the board reads `[435, 201]`.
    Nearest-health pairs the 211 with the 201 -- a drop of 10 -- and
    declares the *survivor* dead, so the round records 10 damage
    against a prediction of 294 and calls it clean. Order-preserving
    alignment has one legal reading of that board: skip the 211, pair
    435 with 435 and 201 with 201. Which is what happened.
    """
    out = {i: None for i in range(len(before))}
    if not after:
        return out

    def key(e):
        return (getattr(e, "name", ""), float(getattr(e, "max_hp", 0) or 0))

    def lines_up():
        """Same board, same order, nobody healed -- so i is i."""
        if len(before) != len(after):
            return False
        return all(key(x) == key(y) and float(y.hp) <= float(x.hp)
                   for x, y in zip(before, after))

    if lines_up():
        return {i: after[i] for i in range(len(before))}

    return _monotone(before, after, key)


#: what one skipped entry costs the alignment, in health points. A death
#: is ordinary -- fights kill things -- so it is cheap. A mob appearing
#: from nowhere at LESS than full health is not something that happens,
#: so reading a damaged after-entry as an arrival is the expensive
#: mistake and is priced to be chosen only when nothing else fits.
_DIED_COST = 500.0
_JOINED_COST = 0.0
_IMPOSSIBLE_JOIN_COST = 1000.0


def _monotone(before, after, key):
    """The best order-preserving pairing of `before` onto `after`.

    Health may never go up, and a pairing costs the health it claims was
    lost -- so among the readings a board admits, the one claiming the
    least damage wins. Skipping a `before` entry means that mob died;
    skipping an `after` entry means it joined.

    An exact search rather than a heuristic: Wizard101 seats at most
    eight participants, so the table is tiny and there is nothing to be
    gained by guessing.
    """
    n, m = len(before), len(after)
    INF = float("inf")

    def arrived(j):
        """Could `after[j]` be a mob that just joined? Only if unhurt."""
        try:
            return float(after[j].hp) >= float(after[j].max_hp) > 0
        except (AttributeError, TypeError, ValueError):
            return False

    def join_cost(j):
        return _JOINED_COST if arrived(j) else _IMPOSSIBLE_JOIN_COST

    def pairing(i, j):
        """What it costs to say before[i] and after[j] are the same mob."""
        if key(before[i]) != key(after[j]):
            return INF
        drop = float(before[i].hp) - float(after[j].hp)
        return drop if drop >= 0 else INF        # health went up: not it

    # cost[i][j]: the best cost of aligning before[i:] onto after[j:]
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    how = [[None] * (m + 1) for _ in range(n + 1)]
    cost[n][m] = 0.0
    for j in range(m - 1, -1, -1):               # only arrivals left
        cost[n][j] = cost[n][j + 1] + join_cost(j)
    for i in range(n - 1, -1, -1):               # only deaths left
        cost[i][m] = cost[i + 1][m] + _DIED_COST
        how[i][m] = ("died", None)

    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best, choice = cost[i + 1][j] + _DIED_COST, ("died", None)
            pair = pairing(i, j)
            if pair < INF and cost[i + 1][j + 1] + pair < best:
                best, choice = cost[i + 1][j + 1] + pair, ("pair", j)
            joined = cost[i][j + 1] + join_cost(j)
            if joined < best:
                best, choice = joined, ("joined", None)
            cost[i][j], how[i][j] = best, choice

    out = {i: None for i in range(n)}
    i = j = 0
    while i < n:
        step = how[i][j] if j <= m else None
        if step is None or step[0] == "died":
            out[i] = None
            i += 1
        elif step[0] == "pair":
            out[i] = after[step[1]]
            i += 1
            j = step[1] + 1
        else:
            j += 1
    return out


@dataclass
class EnemyView:
    name: str
    hp: float
    max_hp: float
    charms: list = field(default_factory=list)
    wards: list = field(default_factory=list)
    #: the mob's school, read off the client. Recorded because training
    #: has to know it: `Boss.resist_own` is 0.40, so a training mob given
    #: the wrong school -- and it was given the literal "ice" for every
    #: mob in every fight -- can resist 40% of the whole deck.
    school: str = ""
    #: how the rollouts modelled this mob's offence, preformatted:
    #: "4+1p · casts Banshee, Ghoul, Dark Sprite" for a catalog caster,
    #: "2p · ~136/round" for the measured flat model. The decision the
    #: policy makes hangs on which of the two priced the fight, and
    #: neither was visible anywhere in the window.
    threat: str = ""


@dataclass
class RoundRecord:
    fight: int
    round: int
    chosen: str = None
    target_index: int = None
    target_name: str = None
    passing: bool = False
    reason: str = ""
    #: which policy played this round, and by which path -- e.g.
    #: "trained (Q) — fallback (state not in Q table)". Per round rather
    #: than per run because the policy can be swapped without
    #: disconnecting, so a single run is not a single policy.
    policy: str = ""
    #: every move weighed this round and what the lookahead said, so a
    #: decision can be read as a comparison rather than an assertion --
    #: "it played the trap" and "the trap beat the nuke by half a turn"
    #: are different facts, and only the second is diagnosable.
    candidates: list = field(default_factory=list)
    hand: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    player_hp: float = 0.0
    player_max_hp: float = 0.0
    norm_pips: int = 0
    pow_pips: int = 0
    #: archmastery pips of the wizard's own school. Zero for most of a
    #: levelling wizard's career and the whole rack for a max-level one
    #: -- which is why a round record without this column once showed a
    #: booster "at 0 pips" for 22 straight rounds of passing.
    school_pips: int = 0
    player_charms: list = field(default_factory=list)
    enemies: list = field(default_factory=list)
    #: castable cards this round the policy could not see at all
    hidden: list = field(default_factory=list)
    hand_visibility: float = 1.0
    predicted_damage: float = None
    actual_damage: float = None
    clean: bool = True
    #: why an observation was marked unclean, if it was
    confounds: list = field(default_factory=list)
    #: measured damage per living enemy per round, off the live fight.
    #: Recorded so training can be given the real number instead of
    #: deriving one from the wizard's own health.
    incoming: float = 0.0
    #: {enemy name: which other wizards aimed at it} for this round, when
    #: a party planned it. The damage model settles by differencing the
    #: board, so a mob another wizard also hit hands this wizard's cast a
    #: residual that is the party's total -- which is not a measurement
    #: of anything. See `Telemetry._settle`.
    party_hits: dict = field(default_factory=dict)
    #: what the WHOLE party expected to take off this round's target,
    #: this wizard's own share included. The measurement that survives a
    #: shared mob -- see `_party_expected`.
    party_predicted: float = None
    # -- pacing. Seconds, on the same clock as the questing log ---------
    #: when the round's planning phase opened, run-relative
    at: float = None
    #: from the previous round finishing to this one opening. The
    #: GAME's time: cast animations, the other wizards' turns, the
    #: planning timer running down when somebody did not click.
    waited: float = None
    #: seconds inside `decide()`. OURS: the memory read, the policy,
    #: and the wait at the hivemind barrier for the rest of the party.
    planned: float = None
    #: seconds spent clicking the chosen card out. Ours as well, and
    #: `cast_time` is the knob on it.
    acted: float = None

    @property
    def error(self):
        if self.predicted_damage is None or self.actual_damage is None:
            return None
        return self.actual_damage - self.predicted_damage

    @property
    def party_error(self):
        """Residual of the PARTY's claim about this mob, or None.

        Only where the party's claim is a different claim: a round this
        wizard had to itself is already reported as a solo observation
        and counting it twice would weight it twice.
        """
        if (self.party_predicted is None or self.actual_damage is None
                or not self.party_hits):
            return None
        return self.actual_damage - self.party_predicted

    @property
    def pct_error(self):
        """Error as a share of what was PREDICTED, not of what landed.

        Dividing by the actual is the wrong denominator for this
        question. The prediction is the claim under test, so it is the
        thing the error is a percentage of -- and normalising by the
        outcome inflates every over-prediction without bound (a cast
        that lands for 1 against a prediction of 100 is -9,900%) while
        squashing every under-prediction toward -100%. Live, that
        reported wizard 2's model bias as -20.8% when it was -13.7%,
        and the asymmetry made the direction itself hard to read.
        """
        e = self.error
        if e is None or not self.predicted_damage:
            return None
        return 100.0 * e / self.predicted_damage


@dataclass
class FightRecord:
    index: int
    #: the duel's opening board, as "Ice Weaver@395+Ice Weaver@395".
    #:
    #: The join key between two wizards' exports. `index` cannot be one:
    #: it counts this seat's own fights, and seats do not see the same
    #: fights -- one live party run had wizard 1's fight 1 be wizard 2's
    #: fight 3, because wizard 2 had already fought two duels alone. So
    #: the two files could not be lined up at all, and the obvious
    #: fallback of matching on (board, round number) is ambiguous for a
    #: quarter of the rounds: three Ice Weavers at 395 is the same key
    #: in every fight against three Ice Weavers.
    opening: str = ""
    rounds: int = 0
    won: bool = None
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    passes: int = 0
    unresolved: int = 0


class Telemetry:
    """Everything a live run produced. The GUI is a view over this."""

    def __init__(self, policy_name="", school="", deck=None, resolver=None,
                 wizard="", seat=0):
        self.policy_name = policy_name
        #: which wizard this record belongs to, and what it is called
        #: in the game. "wizard 1" and "wizard 2" are the window's own
        #: numbering, and they mean nothing once three exports are
        #: sitting in a folder -- the first live party run had to be
        #: identified by reading the hands and guessing. The name is
        #: learned from the first duel; the seat is always known.
        self.wizard = wizard
        self.seat = int(seat or 0)
        self.school = school
        self.deck = list(deck or [])
        #: the run's NameResolver, if there is one. Only used so the GUI
        #: can classify a miss as "the decoder skipped it" vs "no such
        #: card"; everything else here works without it.
        self.resolver = resolver
        self.rounds = []
        self.fights = []
        #: what happened between the fights -- see `note_questing`
        self.questing = []
        self.script_log = []
        self.started_at = None
        self._fight = 0
        self._pending = None       # RoundRecord awaiting its actual damage
        #: the last round recorded, damage or no damage. `_pending` is
        #: cleared the moment the next board settles against it and is
        #: never set for a round the board could not be read for, so it
        #: cannot carry pacing. See `time_round`.
        self._pending_pace = None
        self._listeners = []
        self._curve = []           # [(episode, kill rate %)]
        self._ttk = []             # [(episode, mean turns to kill)]

    # -- wiring -----------------------------------------------------------
    def subscribe(self, fn):
        """fn(event: str, payload) -- the GUI's only coupling to this."""
        self._listeners.append(fn)

    def _emit(self, event, payload=None):
        for fn in list(self._listeners):
            try:
                fn(event, payload)
            except Exception:
                pass       # a broken view must never stop a live fight

    def clear(self):
        """Throw the run away and start an empty one.

        For when the record turns out to belong to somebody else. The
        window keeps one `Telemetry` per SEAT and reuses it across Play
        live presses, but `get_new_clients()` returns windows in whatever
        order it finds them -- so seat 1 can be one wizard on one run and
        a different one on the next, and the file then holds two wizards'
        fights under one name. The first named party run's export did:
        two rounds of a 1,053 HP ice wizard followed by six of a 713 HP
        fire wizard, exported as one wizard called Konstantin.

        Not a new object: the panels hold a reference to this one.
        """
        self.rounds.clear()
        self.fights.clear()
        self._fight = 0
        self._pending = None
        self._pending_pace = None
        self.clear_curve()

    def start_fight(self):
        self._fight += 1
        self.fights.append(FightRecord(index=self._fight))
        self._pending = None
        self._emit("fight_started", self.fights[-1])
        return self.fights[-1]

    def fought(self):
        """The fights that actually had a round in them.

        `start_fight` is called before the wait for a duel, so a run
        that is stopped while waiting leaves an empty record. It is not
        dropped -- the GUI counts fights as they are claimed and the
        indices have to keep meaning what they meant -- but it is not a
        fight either, and counting it understated the win rate of every
        run this app has exported.
        """
        return [f for f in self.fights if f.rounds]

    # -- the per-round hook ----------------------------------------------
    def observe(self, decision, read, sim=None, cards=None, party=None,
                seat=0):
        """Record one planning phase. Call from the backend's on_decision.

        `party` is the round's `hivemind.PartyPlan`, when one was made.
        It is not decoration: the damage model settles by differencing
        the board between this wizard's rounds, and in a party that delta
        carries every wizard's damage. Without knowing who else aimed
        where, a round in which two wizards hit the same mob is recorded
        as one cast landing for both -- silently, and with no confound.
        """
        if not self.fights:
            self.start_fight()
        s = read.state
        if not self.fights[-1].opening:
            # The first board this duel showed. Written once, here, so
            # it is the board as fought rather than whatever is left of
            # it -- see `FightRecord.opening`.
            self.fights[-1].opening = "+".join(
                f"{e.name}@{e.max_hp:.0f}" for e in s.enemies)

        self._settle(read)

        # Only name a mob the cast actually went at. A self-buff or an
        # AoE has no enemy target, and reporting `enemies[0]` for one --
        # which this used to do for every decision that carried no index
        # -- puts a mob in the log that was never clicked, and makes the
        # damage residual settle against the wrong actor.
        kind = getattr(decision, "target_kind", None) or "enemy"
        target_name = None
        if decision.passing:
            pass
        elif kind in ("self", "ally", "allies"):
            target_name = "self" if kind == "self" else "ally"
        elif kind in ("enemies", "global"):
            target_name = "all enemies"
        elif decision.target_index is not None and \
                decision.target_index < len(s.enemies):
            target_name = s.enemies[decision.target_index].name
        elif s.enemies:
            target_name = s.enemies[0].name

        rec = RoundRecord(
            fight=self._fight,
            round=read.round_number,
            chosen=decision.card_name,
            target_index=decision.target_index,
            target_name=target_name,
            passing=decision.passing,
            reason=decision.reason,
            policy=getattr(decision, "policy", "") or self.policy_name,
            candidates=list(getattr(decision, "candidates", ()) or ()),
            hand=sorted(read.hand_cards),
            alternatives=sorted(set(read.hand_cards) - {decision.card_name}),
            unresolved=sorted(read.resolver.misses),
            hidden=sorted(getattr(read, "hidden", []) or []),
            hand_visibility=getattr(read, "hand_visibility", 1.0),
            player_hp=s.player_hp,
            player_max_hp=s.player.max_hp,
            norm_pips=s.norm_pips,
            pow_pips=s.pow_pips,
            school_pips=int(getattr(s.player, "school_pips", 0) or 0),
            player_charms=[describe_hanging(h) for h in s.player.charms],
            enemies=[EnemyView(e.name, e.hp, e.max_hp,
                               [describe_hanging(h) for h in e.charms],
                               [describe_hanging(h) for h in e.wards]
                               # Scheduled damage rides in the ward
                               # column: the Board panel shows the burn,
                               # and `_settle`'s DoT-confound check --
                               # which greps this list -- starts firing
                               # for live DoTs instead of never.
                               + [f"{o.per_tick:,.0f}/tick x{o.rounds_left} "
                                  f"{o.kind}"
                                  for o in getattr(e, "over_time", [])],
                               getattr(e, "school", "") or "",
                               describe_threat(e))
                     for e in s.enemies],
        )

        rec.at = round(self.clock(), 1)
        rec.party_hits = _party_hits(party, seat)
        rec.party_predicted = _party_expected(party, rec.target_name)

        unreadable = list(getattr(read, "unreadable", ()) or ())
        if unreadable:
            # The board the policy was shown was missing effects it could
            # not read -- charms, wards, shields, prisms. Its damage
            # prediction was computed against that board, so the residual
            # is not evidence about the damage model.
            rec.clean = False
            rec.confounds.append(
                "could not read " + "; ".join(unreadable))

        if sim is not None and not decision.passing and cards:
            card = cards.get(decision.chosen if hasattr(decision, "chosen")
                             else decision.card_name)
            if card is not None:
                rec.predicted_damage = predict_damage(
                    sim, s, card, decision.target_index or 0)

        self.rounds.append(rec)
        f = self.fights[-1]
        f.rounds = max(f.rounds, read.round_number)
        if decision.passing:
            f.passes += 1
        f.unresolved = len(rec.unresolved)
        self._pending = rec
        self._pending_pace = rec
        self._emit("round", rec)
        return rec

    def observe_lost_round(self, round_number, reason):
        """Record a round the run could not read the board for.

        Not merely a counter. A round dropped inside the combat handler
        used to leave no record anywhere: the Decisions table went round
        3, round 5, `FightRecord.passes` was never incremented while
        `rounds` picked up 5 -- so the fight read as "5 rounds, 1 pass"
        when it was five rounds and two passes, one of them unexplained.

        `_pending` is cleared without settling, because the next board
        this run sees is two rounds away from the last one it recorded.
        Differencing across that gap charges a round nobody watched to
        the previous round's residual, which then scores against the
        damage model as though the model had missed.
        """
        if not self.fights:
            self.start_fight()
        self._pending = None
        rec = RoundRecord(fight=self._fight, round=int(round_number or 0),
                          passing=True, reason=reason,
                          policy="board read failed",
                          at=round(self.clock(), 1))
        self.rounds.append(rec)
        f = self.fights[-1]
        f.rounds = max(f.rounds, rec.round)
        f.passes += 1
        self._pending_pace = rec
        self._emit("round", rec)
        return rec

    def time_round(self, waited=None, planned=None, acted=None):
        """Stamp the round that has just been played with what it cost.

        Called from the combat handler AFTER the cast, because two of
        the three numbers are not known when `observe` runs. It writes
        to `_pending_pace` rather than `_pending`: `_settle` clears
        `_pending` as soon as the next board arrives, and a lost round
        never sets it at all, but a lost round still took time and that
        time is exactly the kind this is meant to find.

        `waited` is the game's, `planned` and `acted` are ours. The
        split is the whole point -- "a round takes 46 seconds" is not
        actionable and "40 of them are the game's animation" and "20 of
        them are our barrier" call for opposite fixes.
        """
        rec = self._pending_pace
        if rec is None:
            return None
        if waited is not None:
            rec.waited = round(float(waited), 2)
        if planned is not None:
            rec.planned = round(float(planned), 2)
        if acted is not None:
            rec.acted = round(float(acted), 2)
        return rec

    def pacing(self):
        """Where a round's seconds went, over the run.

        Medians rather than means: one round spent in a stalled
        `waitfor` drags a mean far enough to hide the ordinary round,
        and the ordinary round is what "it is very slow" is about.
        """
        def mid(values):
            values = sorted(v for v in values if v is not None)
            if not values:
                return None
            n = len(values)
            return round(values[n // 2] if n % 2
                         else (values[n // 2 - 1] + values[n // 2]) / 2, 1)

        rounds = self.rounds
        out = {
            "rounds": len(rounds),
            "waited": mid(r.waited for r in rounds),
            "planned": mid(r.planned for r in rounds),
            "acted": mid(r.acted for r in rounds),
        }
        # The round-to-round wall clock, which is the number the
        # operator actually feels, and which is NOT waited+planned+acted
        # -- anything outside the handler falls in the difference.
        gaps = []
        for prev, rec in zip(rounds, rounds[1:]):
            if (prev.at is None or rec.at is None
                    or rec.fight != prev.fight or rec.at < prev.at):
                continue
            gaps.append(rec.at - prev.at)
        out["round_to_round"] = mid(gaps)
        return out

    #: How many questing events a run keeps. Days of running would
    #: otherwise put a megabyte of "teleported" in every export; the
    #: interesting ones are the recent ones, because the operator uploads
    #: after noticing something.
    #: Raised from 400 when the log went from "a few notable events" to
    #: a timeline: a heartbeat a minute per wizard, every zone change,
    #: every defeat, every stall report. An hour of three wizards is
    #: roughly 250 entries, and the point of the file is to answer "what
    #: was it doing for those twenty minutes" afterwards -- a cap that
    #: throws the beginning away cannot.
    QUESTING_LOG = 2000
    #: ...and the script's own commentary, kept WHOLE and kept apart.
    #:
    #: Apart, because it would otherwise evict the log it is meant to
    #: explain: a quester prints on every pass of its dispatch, which is
    #: thousands of lines an hour against `QUESTING_LOG`'s 2000, so
    #: within minutes there would be no heartbeat, no stuck-detail and
    #: no realm note left to read it against. Whole, because the
    #: operator asked for all of it -- and because the SEQUENCE is the
    #: diagnosis. A thinned sample says a leg ran often; the full stream
    #: says which legs it alternates between, which is the difference
    #: between "stuck in one place" and "cycling and matching nothing".
    SCRIPT_LOG = 20000

    def clock(self):
        """Seconds since this run's first timed event.

        One zero for the whole export, so a round's `at` and a questing
        note's `at` can be read against each other -- "the party was
        forty seconds a round while the desync was open" is a question
        the export could not answer while the two used different clocks.
        """
        import time

        if self.started_at is None:
            self.started_at = time.monotonic()
        return time.monotonic() - self.started_at

    def note_questing(self, kind, detail=""):
        """Record something that happened OUTSIDE a duel.

        The exports had nothing at all between fights -- not a teleport,
        not a zone change, not a stall -- so a run whose combat is
        healthy and whose questing wedges every ten minutes exports as a
        clean run. Three wizards uploaded at rev 3c8b8087 won every one
        of their seventeen fights and said nothing whatever about the
        thing the operator was actually fighting with.

        `kind` is a short slug ("stranded", "rejoined", "stuck",
        "desync") so a reader can count them; `detail` is the sentence a
        human needs. Timestamps are relative to the run so two wizards'
        logs line up without a clock.
        """
        self.questing.append({
            "at": round(self.clock(), 1),
            "fight": len(self.fights),
            "kind": kind,
            "detail": detail,
        })
        del self.questing[:-self.QUESTING_LOG]

    def note_script(self, text):
        """One line the running script printed, verbatim and in order.

        Its own stream rather than a `questing` entry -- see
        `SCRIPT_LOG`. Timestamped the same way, so the two line up
        against each other and against the other wizards' exports.
        """
        self.script_log.append({
            "at": round(self.clock(), 1),
            "fight": len(self.fights),
            "detail": text,
        })
        del self.script_log[:-self.SCRIPT_LOG]

    def questing_counts(self):
        """{kind: n} over the whole log, for the summary.

        A count is what says "this run needed a human": eleven
        `stranded` in an hour is the report, and it is one line rather
        than four hundred.
        """
        out = {}
        for event in self.questing:
            kind = event.get("kind") or "?"
            out[kind] = out.get(kind, 0) + 1
        return out

    def note_failed_cast(self, reason):
        """The round in hand claimed a cast that never went out.

        The record is written before the click, so this arrives after the
        fact. Dropping the prediction is the substance of it: with the
        prediction left in place, the next board shows the target at
        unchanged HP, the residual settles at minus the whole prediction,
        and a misclick is counted as the damage model's worst miss of the
        run -- with `clean` reading "yes".
        """
        rec = self._pending
        if rec is None:
            return None
        rec.predicted_damage = None
        rec.passing = True
        rec.reason = reason
        rec.clean = False
        rec.confounds.append("the cast failed; nothing was played")
        if self.fights:
            self.fights[-1].passes += 1
        return rec

    def note_recovery_failed(self, why):
        """Why the runner-up did not go out either.

        `note_failed_cast` has already marked the round a pass and that
        is right; what was missing is the second half of the sentence.
        `_try_the_next_best` gives up for four different reasons -- no
        alternatives, the card is not in hand, the cast threw, the card
        stayed in hand -- and all four returned `False` silently, so a
        passed round read as though nothing had been attempted. Rev
        bb8f2b3c has exactly one failed cast in it, with two castable
        alternatives sitting in the same hand, and no way to tell which
        of the four happened.
        """
        rec = self._pending
        if rec is None:
            return None
        rec.reason = f"{rec.reason} — {why}" if rec.reason else why
        rec.clean = False
        return rec

    def note_recovered_cast(self, card, target, first_choice):
        """The runner-up went out after the chosen card would not.

        Undoes exactly what `note_failed_cast` did a moment earlier, and
        no more. That call is right at the instant it runs -- the chosen
        card really did not go out -- but it marks the round as a pass,
        and a round in which the second-choice card was cast is not a
        pass. Left alone, the next board's health drop would be
        differenced against a round claiming nothing was played, and the
        real cast's damage would be folded into the round after it.

        The prediction is NOT restored. It was priced for a different
        card, and this round's measurement belongs to a cast nobody
        predicted; `damage_observations` excludes it rather than scoring
        the damage model against a number it never produced.
        """
        rec = self._pending
        if rec is None:
            return None
        rec.passing = False
        rec.chosen = card
        rec.target_index = target
        rec.reason = (f"{first_choice} would not go out; played {card} "
                      f"instead")
        rec.clean = False
        try:
            rec.confounds.remove("the cast failed; nothing was played")
        except ValueError:
            pass
        rec.confounds.append(
            f"{first_choice} did not go out and {card} was played in its "
            f"place -- there is no prediction for what was actually cast")
        if self.fights and self.fights[-1].passes:
            self.fights[-1].passes -= 1
        return rec

    def _settle(self, read):
        """Fill in the previous round's actual damage from the new board."""
        prev = self._pending
        if prev is None or prev.predicted_damage is None:
            self._pending = None
            return
        if not prev.predicted_damage:
            # A trap, a shield, a debuff: aimed at an enemy and
            # predicted at exactly zero. Settling it anyway hands the
            # cast whatever the board lost that round, which is the
            # PARTY's damage -- so a Fire Trap and the Frost Beetle
            # that fired into the same mob both recorded 56, and the
            # fight's `damage_dealt` counted those 56 twice. Nothing
            # about the damage model is learnable from a card that was
            # never going to move health.
            self._pending = None
            return
        board = list(read.state.enemies)
        pairs = match_enemies(prev.enemies, board)
        target = prev.target_name
        i = prev.target_index
        if i is None or not (0 <= i < len(prev.enemies)):
            i = next((j for j, e in enumerate(prev.enemies)
                      if e.name == target), None)
        before = prev.enemies[i] if i is not None else None
        after = pairs.get(i) if i is not None else None
        if before is None:
            self._pending = None
            return
        joined = len(board) - len(prev.enemies)
        if joined > 0:
            # `match_enemies` books arrivals out by the fact that they
            # come in at full health, which is right often enough to be
            # worth doing and is still a guess. Wizard101 seats a
            # newcomer in whichever circle position is free, so two
            # identical mobs and one arrival is genuinely ambiguous --
            # and a guess that reads as a measurement is worse than no
            # measurement.
            prev.clean = False
            prev.confounds.append(
                f"{joined} mob(s) joined the duel this round, so which "
                f"reading is the target is inferred rather than known")
        if after is None:
            # It died. The hit landed for at least its remaining HP; that
            # is a floor, not a measurement, so it is recorded unclean.
            prev.actual_damage = before.hp
            prev.clean = False
            prev.confounds.append("target died: actual damage is a lower bound")
        else:
            prev.actual_damage = max(before.hp - after.hp, 0.0)
            # A mob another wizard also aimed at hands this cast a
            # residual that is the PARTY's damage, not its own. Silent
            # before parties existed, because there was nobody else to
            # confuse it with; in the first live party run three of
            # wizard 1's rounds carried the collateral note below with
            # "AoE, DoT or a minion" as the suggested cause, when the
            # cause was wizard 2.
            sharing = prev.party_hits.get(target)
            if sharing:
                prev.clean = False
                prev.confounds.append(
                    f"{sharing} also hit {target} this round -- the board "
                    f"delta is the party's damage, not this cast's, so it "
                    f"is measured against what the party expected "
                    f"({prev.party_predicted:,.0f}) instead"
                    if prev.party_predicted is not None else
                    f"{sharing} also hit {target} this round -- the board "
                    f"delta is the party's damage, not this cast's")
            for j, e in enumerate(prev.enemies):
                other = pairs.get(j)
                if other is not None and j != i and other.hp < e.hp:
                    prev.clean = False
                    who = prev.party_hits.get(e.name)
                    prev.confounds.append(
                        f"{e.name} also lost HP -- {who} hit it" if who
                        else f"{e.name} also lost HP -- AoE, DoT or a minion")
                    break
            if (prev.party_predicted
                    and prev.party_predicted > before.hp
                    and prev.actual_damage < before.hp):
                # The party expected to take more off this mob than it
                # had, and the mob is still standing. The delta cannot
                # exceed the health that was there, so the residual is
                # bounded by the board rather than by the arithmetic --
                # it measures a fizzle or a ward the read missed, not
                # the model. Marked so the party series does not read a
                # 90% over-prediction off a mob with 78 HP left.
                prev.confounds.append(
                    f"the party expected {prev.party_predicted:,.0f} off a "
                    f"mob with {before.hp:,.0f} health and it survived — "
                    f"something did not land, so the shortfall is not the "
                    f"damage model's")
                prev.clean = False
            if any("dot" in w.lower() or "over time" in w.lower()
                   for w in (before.wards or [])):
                prev.clean = False
                prev.confounds.append("a DoT was ticking on the target")
            if prev.actual_damage <= 0.0 and prev.predicted_damage > 0.0:
                # Nothing landed at all. The prediction is computed
                # through `_NoFizzle`, so it is the damage the cast does
                # WHEN IT LANDS -- a real fizzle then reads as a 100%
                # model error, and the damage model exists to check the
                # simulator's arithmetic rather than its luck. Live
                # example: a Snow Serpent predicted at 294 into a 249 HP
                # mob that was still at 249 the next round, recorded
                # clean, and it was the only observation in the run --
                # so the whole reported model quality was one coin flip.
                #
                # Marked rather than dropped: a genuine zero can also be
                # a shield or an absorb the reader missed, and that IS
                # worth looking at. Either way it is not a measurement of
                # the arithmetic.
                prev.clean = False
                prev.confounds.append(
                    "nothing landed on a target that survived -- a fizzle, "
                    "or a ward the read missed. The prediction assumes the "
                    "cast lands, so this is not a measurement of the "
                    "damage model")
        if prev.actual_damage is not None:
            self.fights[-1].damage_dealt += prev.actual_damage
        self._pending = None

    def end_fight(self, won=None):
        if self.fights:
            self.fights[-1].won = won
            self._emit("fight_ended", self.fights[-1])

    # -- what the GUI asks for -------------------------------------------
    def damage_observations(self, clean_only=True):
        """Rounds that actually say something about the damage model.

        A round where a blade was cast has predicted 0 and actual 0, and
        an error of exactly 0. Counting those would be flattering
        nonsense -- they would drag the mean error toward zero in
        proportion to how buff-heavy the deck is, so a deck that stacks
        five blades before firing would look five times more accurate
        than one that nukes every round. Only rounds where damage was
        expected or observed count.
        """
        return [r for r in self.rounds
                if r.error is not None
                and (r.predicted_damage or r.actual_damage)
                and (r.clean or not clean_only)]

    def error_stats(self, clean_only=True):
        obs = self.damage_observations(clean_only)
        if not obs:
            return {"n": 0}
        errs = [r.error for r in obs]
        pcts = [r.pct_error for r in obs if r.pct_error is not None]
        return {
            "n": len(obs),
            "mean_error": statistics.fmean(errs),
            "mean_abs_error": statistics.fmean(abs(e) for e in errs),
            "median_abs_error": statistics.median(abs(e) for e in errs),
            "rmse": math.sqrt(statistics.fmean(e * e for e in errs)),
            "mean_pct_error": statistics.fmean(pcts) if pcts else None,
            # "f6r8", not "8". A bare round number does not identify a
            # round -- a run reporting `worst: 8` had two candidates in
            # different fights and the interesting one was neither the
            # first nor the last.
            "worst": _round_id(max(obs, key=lambda r: abs(r.error))),
        }

    #: a shared round is unclean for the SOLO series by construction --
    #: that is what makes it a party observation -- so this confound is
    #: not disqualifying here. Anything else still is.
    _SHARED = "also hit"

    def party_observations(self):
        """Rounds where the PARTY's claim about a mob can be checked.

        Exactly the rounds the solo series has to throw away: two
        wizards into one mob, one board delta. What cannot be split can
        still be added up, and the sum is a claim the same arithmetic
        made. See `_party_expected`.

        Every other confound still disqualifies. A DoT ticking on the
        target or a third mob losing health corrupts the party's number
        exactly as much as it corrupts one wizard's.
        """
        out = []
        for r in self.rounds:
            if r.party_error is None or not (r.party_predicted
                                             or r.actual_damage):
                continue
            if any(self._SHARED not in c for c in r.confounds):
                continue
            out.append(r)
        return out

    def party_error_stats(self):
        obs = self.party_observations()
        if not obs:
            return {"n": 0}
        errs = [r.party_error for r in obs]
        pcts = [100.0 * r.party_error / r.party_predicted
                for r in obs if r.party_predicted]
        return {
            "n": len(obs),
            "mean_error": statistics.fmean(errs),
            "mean_abs_error": statistics.fmean(abs(e) for e in errs),
            "median_abs_error": statistics.median(abs(e) for e in errs),
            "rmse": math.sqrt(statistics.fmean(e * e for e in errs)),
            "mean_pct_error": statistics.fmean(pcts) if pcts else None,
            "worst": _round_id(max(obs, key=lambda r: abs(r.party_error))),
        }

    def unresolved_names(self):
        counts = {}
        for r in self.rounds:
            for n in r.unresolved:
                counts[n] = counts.get(n, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def hand_visibility(self):
        """How much of the hand the policy could see, averaged over rounds.

        The single number that says whether a run is worth analysing. A
        policy shown 70% of its hand is not the policy you trained, and
        no amount of care in the other panels recovers that.
        """
        if not self.rounds:
            return 1.0
        return statistics.fmean(r.hand_visibility for r in self.rounds)

    def hidden_cards(self):
        counts = {}
        for r in self.rounds:
            for n in r.hidden:
                counts[n] = counts.get(n, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def observed_mob_hps(self):
        """Each mob's max health, from the opening round of the last fight.

        The individual numbers matter, not just the count. `Featurizer.key`
        carries `(living, weakest_index, ...)`, so a training board whose
        mobs all start at the same health has `weakest_index == 0` in
        every opening state -- the entire `weakest == 1` half is never
        visited, and a real board with a 515hp mob beside a 390hp one
        lands squarely in it. That one field was the whole of a measured
        0% coverage.
        """
        for rec in reversed(self.rounds):
            if not rec.enemies:
                continue
            if rec.round != 1 and any(r.fight == rec.fight and r.round == 1
                                      for r in self.rounds):
                continue
            return [e.max_hp for e in rec.enemies]
        return []

    def observed_mob_schools(self):
        """Each mob's school, from the opening round of the last fight.

        Training needs this and never had it: `TrainWorker` built every
        mob as the literal `"ice"`, and `Boss.resist_own` is 0.40, so an
        ice wizard trained against a board that resisted 40% of every
        card in the deck. On a real board that was the difference
        between a kill rate that climbs and one that is flat at zero for
        40,000 episodes.
        """
        for rec in reversed(self.rounds):
            if not rec.enemies:
                continue
            if rec.round != 1 and any(r.fight == rec.fight and r.round == 1
                                      for r in self.rounds):
                continue
            return [getattr(e, "school", "") or "" for e in rec.enemies]
        return []

    def observed_mob_names(self):
        """Each mob's name, from the opening round of the last fight.

        The deck search needs them: the bestiary's per-boss resist and
        boost tables are keyed by name, and resist is the most decisive
        stat a deck choice has -- it decides which school of damage to
        slot at all.
        """
        for rec in reversed(self.rounds):
            if not rec.enemies:
                continue
            if rec.round != 1 and any(r.fight == rec.fight and r.round == 1
                                      for r in self.rounds):
                continue
            return [e.name for e in rec.enemies]
        return []

    def observed_incoming(self):
        """Measured damage per enemy per round, over the whole run.

        The live backend computes this every round and it was thrown
        away, while the trainer guessed the same quantity from the
        *wizard's* own health -- a number that makes the fight equally
        hard however the wizard is geared, so a table trained on it
        cannot learn that more health buys more turns.

        Averaged over rounds rather than taken from the last: a single
        round's drop has a standard deviation about 40% of its own mean,
        because it depends on what the mobs happened to cast.
        """
        seen = [r.incoming for r in self.rounds
                if getattr(r, "incoming", 0.0) > 0]
        return sum(seen) / len(seen) if seen else 0.0

    def damage_rate(self):
        """Enemy health this wizard actually removes per round of fight.

        What a teammate's rollout needs to know about this seat, and the
        one number nothing published. `policies.set_ally_rate` explains
        why it matters; this explains why it is *this* denominator.

        Total damage over total fight rounds -- zeros and all -- not a
        mean over the rounds that happened to yield a measurement. Only
        about half of rounds do (a confounded board, an unmatched read,
        a round the settle could not attribute), and those are exactly
        the rounds where nothing landed. Averaging over the measurable
        half of the day's two-wizard run gives 99/round for the ice
        wizard and 151 for the fire one; over every round of every fight
        it gives 51 and 57, and 51 and 57 are what the boards actually
        lost. Feeding a rollout the first pair would tell it a fight
        ends in half the rounds it really takes.

        0.0 until a fight has finished, which leaves the first duel of a
        session modelled solo. That is the shipped behaviour, so the
        cold start costs nothing it was not already costing; the caller
        supplies its own estimate if it has a better one.
        """
        rounds = sum(f.rounds for f in self.fights)
        if not rounds:
            return 0.0
        return sum(f.damage_dealt for f in self.fights) / rounds

    def observed_board(self):
        """(enemy count, biggest mob's max health) from the last fight.

        Training has to match this or the Q table is unusable, and not
        approximately -- `Featurizer.key` appends a `foes` tuple only when
        the board holds more than one enemy, so a 1v1-trained table and a
        two-mob fight produce keys of *different length* and cannot share
        a single state. The health matters too, though less brutally: the
        key buckets it as `max_health // 250`.

        Taken from the opening round of the most recent fight, because
        that is the board the fight was actually fought against -- later
        rounds have mobs already dead.

        Returns None when nothing has been fought yet.
        """
        for rec in reversed(self.rounds):
            if not rec.enemies:
                continue
            if rec.round != 1 and any(r.fight == rec.fight and r.round == 1
                                      for r in self.rounds):
                continue
            return len(rec.enemies), max(e.max_hp for e in rec.enemies)
        return None

    #: how many moves the decision matrix shows before it stops adding
    #: columns. Past roughly this many the cells are too narrow to carry
    #: a number and the grid stops being readable; the count that was
    #: dropped is reported rather than silently truncated.
    MATRIX_COLUMNS = 9

    def decision_matrix(self, rounds=14):
        """(row list, column list, dropped) for the decision heatmap.

        Rows are the most recent planning phases, columns the moves that
        were weighed, cell values turns-to-clear on the lookahead's own
        rollout. Only rounds that recorded a candidate set appear -- a
        Q-table lookup produces no comparison, and inventing one would
        misrepresent how that decision was made.

        Columns are ordered by how often a move came up, so the ones that
        actually recur sit together instead of the grid being shuffled by
        whichever card was drawn first.
        """
        recs = [r for r in self.rounds if r.candidates][-rounds:]
        if not recs:
            return [], [], 0
        multi_fight = len({r.fight for r in recs}) > 1

        counts = {}
        for rec in recs:
            for cand in rec.candidates:
                key = _move_label(cand)
                counts[key] = counts.get(key, 0) + 1
        order = sorted(counts, key=lambda k: (-counts[k], k))
        cols = order[:self.MATRIX_COLUMNS]
        dropped = len(order) - len(cols)

        rows = []
        for rec in recs:
            cells = {}
            for cand in rec.candidates:
                key = _move_label(cand)
                if key not in cols:
                    continue
                why = outcome_of(cand)
                cells[key] = (
                    round(float(_attr(cand, "turns", 0.0)), 1),
                    bool(_attr(cand, "chosen", False)),
                    (f"{why} — no clear inside the horizon; "
                     if why else
                     f"kills in {_attr(cand, 'turns', 0):g} turn(s), ")
                    + f"{_attr(cand, 'damage', 0):,.0f} damage banked")
            # Fight and round. Labelling by round alone repeats "r1"
            # once per fight, and the matrix stacks fights, so three
            # fights showed r1/r2/r3/r1/r2/r4/r7/r1... with no way to
            # tell which fight a row belonged to.
            rows.append((f"f{rec.fight} r{rec.round}"
                         if multi_fight else f"r{rec.round}", cells))
        return rows, cols, dropped

    def candidate_bars(self, index, rounds=14):
        """[(label, turns, chosen, note)] for one row of the matrix.

        Ranked best-first, because the question a person asks of a single
        decision is "what else was close" and a ranked list answers it
        directly.

        `index` is a **row of the matrix**, so it is windowed the same
        way `decision_matrix` windows its rows. It was not: the matrix
        showed the last 14 candidate rounds while this indexed the whole
        list, so past round 14 every row was off by N-14 and clicking a
        row broke out a different round's candidates than the one
        labelled beside it.
        """
        recs = [r for r in self.rounds if r.candidates][-rounds:]
        if not recs or not (0 <= index < len(recs)):
            return [], ""
        rec = recs[index]
        bars = sorted(
            ((_move_label(c), round(float(_attr(c, "turns", 0.0)), 1),
              bool(_attr(c, "chosen", False)),
              (f"{outcome_of(c)} — " if outcome_of(c) else "")
              + f"{_attr(c, 'damage', 0):,.0f} damage banked, "
              f"{_attr(c, 'pips', 0):g} pip(s)",
              outcome_of(c) or None)
             for c in rec.candidates),
            key=lambda b: (b[1], -b[2]))
        lost = [c for c in rec.candidates if outcome_of(c)]
        title = f"fight {rec.fight}, round {rec.round}"
        if lost and len(lost) == len(rec.candidates):
            # Every option is a sentinel: the lookahead did not rank
            # these, it gave up on all of them. Saying so is the
            # difference between "a close call" and "no line survives".
            title += " — no line survives the horizon"
        return bars, title

    def training_curve(self):
        """[(episode, kill rate %)] recorded during the last training run."""
        return list(self._curve)

    def record_snapshot(self, episode, kill_rate, ttk):
        self._curve.append((episode, kill_rate * 100.0))
        self._ttk.append((episode, ttk))

    def ttk_curve(self):
        """[(episode, mean turns to kill)].

        Kept apart from the kill-rate curve rather than sharing an axis
        with it: two measures on two y-scales invent a correlation out of
        the arbitrary alignment of the scales.
        """
        return list(self._ttk)

    def clear_curve(self):
        self._curve, self._ttk = [], []

    def policy_mix(self):
        """Rounds played, per policy and per path within it.

        Read back off the round records rather than off a counter living
        on the policy object, because the policy can be swapped without
        disconnecting: a counter on the policy only knows about the
        rounds it was installed for, and says nothing about the ones
        before or after. This is the number that answers "is the model I
        picked actually driving?" -- a run where `trained (Q)` was
        selected but every row reads `fallback` is the failure this
        exists to make visible.
        """
        counts = {}
        for r in self.rounds:
            key = r.policy or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def trained_coverage(self):
        """(decided, missed, {reason: count}) for the trained policy.

        One source for a number the window was reporting twice. The
        config line took it off `TrainedPolicy`'s own counters and the
        Learning tab took it off the round records, and the two
        disagreed on screen -- "decided 0% (1 fell back)" beside
        "14%, 1 of 7 rounds" -- because the policy object's counters
        reset every time the policy is swapped, which `set_policy` does
        without disconnecting. Two numbers for one fact is worse than
        either number alone.

        The reasons come along because they are the actionable part: a
        run that missed 6 boards for one stated reason has a fix, and
        the round records already carry it.
        """
        decided = missed = 0
        reasons = {}
        for r in self.rounds:
            name = r.policy or ""
            if "trained" not in name:
                continue
            if "fallback" in name:
                missed += 1
                why = name.split("—", 1)[-1].strip() if "—" in name else name
                why = why.removeprefix("fallback").strip(" —()")
                if why and why != "state not in Q table":
                    reasons[why] = reasons.get(why, 0) + 1
            else:
                decided += 1
        return decided, missed, dict(
            sorted(reasons.items(), key=lambda kv: -kv[1]))

    def summary(self):
        st = self.error_stats()
        return {
            "wizard": self.wizard or f"wizard {self.seat + 1}",
            "seat": self.seat + 1,
            #: which build produced these numbers. See `revision`.
            "revision": revision(),
            "policy": self.policy_name,
            "school": self.school,
            # Fights that actually happened. The live worker claims a
            # record at the top of its loop and then waits for a duel,
            # so stopping the run leaves a 0-round record behind -- and
            # counting it read as "9 wins from 13 fights" for a wizard
            # that fought twelve and won nine.
            "fights": len(self.fought()),
            "rounds": len(self.rounds),
            "passes": sum(f.passes for f in self.fights),
            "wins": sum(1 for f in self.fought() if f.won),
            "damage_model": st,
            # The same arithmetic, checked on the rounds the solo series
            # cannot use. In a real party those are most of them.
            "party_damage_model": self.party_error_stats(),
            "policy_mix": self.policy_mix(),
            "unresolved": self.unresolved_names(),
            "hand_visibility": self.hand_visibility(),
            "hidden_cards": self.hidden_cards(),
            "questing": self.questing_counts(),
            "script_log_lines": len(self.script_log),
            # Where a round's seconds went. "It is very slow" was a
            # report the exports could not answer at all: rounds carried
            # no time in them, so a 46-second round and a 12-second one
            # looked identical in the file.
            "pacing": self.pacing(),
        }

    def to_json(self, path):
        payload = {
            "summary": self.summary(),
            "fights": [asdict(f) for f in self.fights],
            "rounds": [asdict(r) for r in self.rounds],
            # Everything that happened between the fights. A run can be
            # perfect in every duel and still need a human every ten
            # minutes, and until this existed the export could not tell
            # the two apart.
            "questing": list(self.questing),
            # The script's own `print` output, every line of it. Kept
            # out of `questing` so it cannot evict the log it explains.
            "script_log": list(self.script_log),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return path
