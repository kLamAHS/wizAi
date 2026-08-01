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

    @property
    def error(self):
        if self.predicted_damage is None or self.actual_damage is None:
            return None
        return self.actual_damage - self.predicted_damage

    @property
    def pct_error(self):
        e = self.error
        if e is None or not self.actual_damage:
            return None
        return 100.0 * e / self.actual_damage


@dataclass
class FightRecord:
    index: int
    rounds: int = 0
    won: bool = None
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    passes: int = 0
    unresolved: int = 0


class Telemetry:
    """Everything a live run produced. The GUI is a view over this."""

    def __init__(self, policy_name="", school="", deck=None, resolver=None):
        self.policy_name = policy_name
        self.school = school
        self.deck = list(deck or [])
        #: the run's NameResolver, if there is one. Only used so the GUI
        #: can classify a miss as "the decoder skipped it" vs "no such
        #: card"; everything else here works without it.
        self.resolver = resolver
        self.rounds = []
        self.fights = []
        self._fight = 0
        self._pending = None       # RoundRecord awaiting its actual damage
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

    def start_fight(self):
        self._fight += 1
        self.fights.append(FightRecord(index=self._fight))
        self._pending = None
        self._emit("fight_started", self.fights[-1])
        return self.fights[-1]

    # -- the per-round hook ----------------------------------------------
    def observe(self, decision, read, sim=None, cards=None):
        """Record one planning phase. Call from the backend's on_decision."""
        if not self.fights:
            self.start_fight()
        s = read.state

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
                               getattr(e, "school", "") or "")
                     for e in s.enemies],
        )

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
                          policy="board read failed")
        self.rounds.append(rec)
        f = self.fights[-1]
        f.rounds = max(f.rounds, rec.round)
        f.passes += 1
        self._emit("round", rec)
        return rec

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

    def _settle(self, read):
        """Fill in the previous round's actual damage from the new board."""
        prev = self._pending
        if prev is None or prev.predicted_damage is None:
            self._pending = None
            return
        by_name = {e.name: e for e in read.state.enemies}
        target = prev.target_name
        before = next((e for e in prev.enemies if e.name == target), None)
        after = by_name.get(target)
        if before is None:
            self._pending = None
            return
        if after is None:
            # It died. The hit landed for at least its remaining HP; that
            # is a floor, not a measurement, so it is recorded unclean.
            prev.actual_damage = before.hp
            prev.clean = False
            prev.confounds.append("target died: actual damage is a lower bound")
        else:
            prev.actual_damage = max(before.hp - after.hp, 0.0)
            for e in prev.enemies:
                other = by_name.get(e.name)
                if other is not None and e.name != target and other.hp < e.hp:
                    prev.clean = False
                    prev.confounds.append(
                        f"{e.name} also lost HP -- AoE, DoT or a minion")
                    break
            if any("dot" in w.lower() or "over time" in w.lower()
                   for w in (before.wards or [])):
                prev.clean = False
                prev.confounds.append("a DoT was ticking on the target")
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
            "worst": max(obs, key=lambda r: abs(r.error)).round,
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
            "policy": self.policy_name,
            "school": self.school,
            "fights": len(self.fights),
            "rounds": len(self.rounds),
            "passes": sum(f.passes for f in self.fights),
            "wins": sum(1 for f in self.fights if f.won),
            "damage_model": st,
            "policy_mix": self.policy_mix(),
            "unresolved": self.unresolved_names(),
            "hand_visibility": self.hand_visibility(),
            "hidden_cards": self.hidden_cards(),
        }

    def to_json(self, path):
        payload = {
            "summary": self.summary(),
            "fights": [asdict(f) for f in self.fights],
            "rounds": [asdict(r) for r in self.rounds],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return path
