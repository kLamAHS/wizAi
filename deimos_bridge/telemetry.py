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


@dataclass
class EnemyView:
    name: str
    hp: float
    max_hp: float
    charms: list = field(default_factory=list)
    wards: list = field(default_factory=list)


@dataclass
class RoundRecord:
    fight: int
    round: int
    chosen: str = None
    target_index: int = None
    target_name: str = None
    passing: bool = False
    reason: str = ""
    hand: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    player_hp: float = 0.0
    player_max_hp: float = 0.0
    norm_pips: int = 0
    pow_pips: int = 0
    player_charms: list = field(default_factory=list)
    enemies: list = field(default_factory=list)
    predicted_damage: float = None
    actual_damage: float = None
    clean: bool = True
    #: why an observation was marked unclean, if it was
    confounds: list = field(default_factory=list)

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

    def __init__(self, policy_name="", school="", deck=None):
        self.policy_name = policy_name
        self.school = school
        self.deck = list(deck or [])
        self.rounds = []
        self.fights = []
        self._fight = 0
        self._pending = None       # RoundRecord awaiting its actual damage
        self._listeners = []

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

        target_name = None
        if decision.target_index is not None and \
                decision.target_index < len(s.enemies):
            target_name = s.enemies[decision.target_index].name
        elif s.enemies and not decision.passing:
            target_name = s.enemies[0].name

        rec = RoundRecord(
            fight=self._fight,
            round=read.round_number,
            chosen=decision.card_name,
            target_index=decision.target_index,
            target_name=target_name,
            passing=decision.passing,
            reason=decision.reason,
            hand=sorted(read.hand_cards),
            alternatives=sorted(set(read.hand_cards) - {decision.card_name}),
            unresolved=sorted(read.resolver.misses),
            player_hp=s.player_hp,
            player_max_hp=s.player.max_hp,
            norm_pips=s.norm_pips,
            pow_pips=s.pow_pips,
            player_charms=[describe_hanging(h) for h in s.player.charms],
            enemies=[EnemyView(e.name, e.hp, e.max_hp,
                               [describe_hanging(h) for h in e.charms],
                               [describe_hanging(h) for h in e.wards])
                     for e in s.enemies],
        )

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
            "unresolved": self.unresolved_names(),
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
