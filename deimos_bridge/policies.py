"""Policies for live play.

wizAi's built-in heuristics were written against decks the builder
produced, and every one of those is **school-coherent** — a fire deck
holds fire blades and fire nukes. That assumption is invisible in the
simulator and false the moment a real wizard opens a real hand: starter
wands hand out Thunder Snake (storm), Imp (fire), Scarab (myth) and Dark
Sprite (death) regardless of school, and pets add trained cards of their
own.

`make_blade_stack` picks the biggest available buff by percentage and,
separately, the biggest available nuke by damage, with nothing tying the
two together. On a coherent deck they always match. On a real starter
hand it will happily stack a **Mythblade** and then fire a **Thunder
Snake**, and the blade does nothing at all — `_consume_damage_charms`
only applies charms where `h.matches(school)`.

`school_aware_blade_stack` closes that loop: decide the nuke first, then
only stack buffs that can actually multiply it.

On a school-coherent deck the two are **identical** — every blade matches
the only nuke school available, so "best matching buff" and "best buff"
select the same card. `tests/test_deimos_bridge.py` asserts that over the
project's own live decks, which is what makes this safe as the live
default without invalidating any published table.
"""


import contextlib
import re

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """One move the policy weighed, and what the lookahead said about it.

    Recorded for every option rather than only the winner, so a decision
    can be read as a comparison instead of an assertion. `turns` is
    turns-to-clear-the-board on a deterministic rollout that opens with
    this move (lower is better); `damage` is the effective damage banked
    inside the horizon, floored at each mob's health so overkill earns
    nothing.
    """

    card: str
    target: int = None
    turns: float = 0.0
    damage: float = 0.0
    pips: int = 0
    chosen: bool = False
    #: the rollout's horizon, so a reader can tell a turn count from a
    #: sentinel. `turns` above `horizon` is not a turn count at all --
    #: it encodes dying, stalling, or being unplayable -- and the panel
    #: was rendering all three as numbers. A whole board of candidates
    #: reading "14 turns", including "pass", is `died()` at a horizon of
    #: 12, i.e. "no line survives", and it looked like a flat tie.
    horizon: int = 12


#: op kinds that carry the cast's real target, in the order they decide
#: it. A card whose first op is a self-charm is a self-cast even if a
#: later op mentions an enemy (Feint places a ward on both sides).
TARGET_OPS = ("hit", "dot", "drain", "charm", "ward", "prism", "heal",
              "absorb", "dispel", "stun", "aura")


def _is_inert(card, state) -> bool:
    """Would this card provably do nothing at all right now?

    One case: **a heal on a wizard already at full health.**
    `Sim._heal` caps a heal at `max_hp - hp`, so at full health it banks
    exactly nothing -- and a card worth nothing should not be able to
    win a round. It reaches the top anyway because candidates are ranked
    by rollout: when no line kills inside the horizon every candidate
    ties on turns, and the tie-break is then free to pick the one move
    that is certainly worthless.

    This is a *waste* rule, not a legality one. It was written on the
    theory that Wizard101 refuses a heal at full health, which the next
    live run refuted -- a Pixie failed to cast at 526/897, with 371
    health missing. So it does not fix that, and does not claim to; it
    only stops a round being spent on a heal that would restore zero.

    X-pip cards used to be the other case, and that was my error, not
    the engine's. The claim here was that `Sim` "gives them 0 damage at
    every pip count, with the pips not even spent". It does not. The
    engine resolves them correctly and always has: the ops carry
    `per_pip`, `Sim.spend` returns the effective rack and `_resolve_ops`
    multiplies by it. Re-measured against a 4000hp mob with the
    rollout's own fixed RNG, Heck Hound ticks 43.3 / 86.7 / 130.0 /
    173.3 / 216.7 / 260.0 / 303.3 per round at one through seven pips --
    130 a pip, exactly as printed, and the rack goes to zero every time.
    The original measurement had a live RNG in it and I read two fizzles
    as a broken card.

    What was really broken was the PRICE, in the ranking key rather than
    in the engine: `card.pips` reads 0 and `card.damage` reads per-pip,
    so an X-pip card was both free and oversized in every tiebreak.
    `cast_price` and `cast_reach` fix that where it belongs, and the
    rollout then does what it always could -- it rejects Heck Hound at
    one and two pips (13.0, a sentinel) and takes it at four (5 turns
    against Fire Cat's 9) and at seven (3 against 6).

    Skipping the card instead cost a hand slot for a whole session: Heck
    Hound sat in the fire wizard's hand for 74 of 75 rounds of the run
    at rev e523684f and was never once played, out of a median hand of
    four.

    Deliberately narrow otherwise. A blade on a wizard that already has
    three is not inert -- it stacks -- and a trap on a mob that is about
    to die is a judgement call the rollout is better placed to make than
    a rule here.
    """
    if getattr(card, "kind", "") != "heal":
        return False
    me = getattr(state, "player", None)
    if me is None:
        return False
    try:
        if float(me.hp) < float(me.max_hp):
            return False
    except (TypeError, ValueError):
        return False
    # A heal that also does something else -- a HoT, a charm -- still has
    # a reason to be cast at full health.
    for op in getattr(card, "ops", None) or []:
        if op.get("op") not in ("heal",):
            return False
    return True


def primary_target(card):
    """'self' | 'enemy' | 'enemies' | 'ally' | 'allies' | 'global' | None.

    Reads it off the card's own ops rather than guessing from its kind,
    because the data already carries it and the kinds do not map cleanly
    -- a 'trap' goes on an enemy, a 'blade' on the caster, and both are
    charms.
    """
    ops = getattr(card, "ops", None) or []
    if not ops:
        return None
    for want in TARGET_OPS:
        for op in ops:
            if op.get("op") == want:
                return op.get("tgt")
    return ops[0].get("tgt")


def aimed_at_one_enemy(card) -> bool:
    """Does choosing *which* enemy mean anything for this card?

    False for self-buffs, AoEs and globals -- picking a target index for
    those is meaningless, and pretending otherwise puts a fabricated
    enemy name in the decision log.
    """
    return primary_target(card) == "enemy"


def focus_target(state) -> int:
    """Which enemy to build the turn around.

    The lowest-health living enemy. Two properties make this the right
    default rather than an arbitrary one:

      * It is **coherent**. A trap only pays off if the hit lands on the
        same enemy, so the buff and the hit have to agree on a target.
        Before this, nothing chose at all -- every cast went to
        `enemies[0]`, whichever mob the participant list happened to put
        first, and when that one died the rest of the plan silently moved
        to a different mob with the traps left behind on a corpse.
      * It is **stable**. Hitting the focus only lowers its health, so
        the choice re-derives to the same enemy next round until it dies.
        A rule like "highest health" would thrash between two mobs as
        their totals crossed, splitting a buff stack across both.

    Focus fire is also just correct play: a dead enemy stops attacking,
    and every round a mob lives is another round of incoming damage.
    """
    live = [(i, e) for i, e in enumerate(state.enemies) if e.alive]
    if not live:
        return 0
    return min(live, key=lambda ie: (ie[1].hp, ie[0]))[0]


def _card_schools(card):
    """Damage schools a buff applies to, or None for universal.

    Read off the ops rather than `card.school`: a Tri Blade is a *balance*
    card whose three charms boost ice, storm and fire, and Balanceblade is
    a balance card that boosts everything.
    """
    out = set()
    for op in card.ops or []:
        if op.get("op") not in ("charm", "ward"):
            continue
        schools = op.get("schools")
        if schools is None:
            return None                  # universal
        out.update(schools)
    return out or None


def buff_matches(card, school):
    """Would this buff multiply a hit of `school`?"""
    schools = _card_schools(card)
    return schools is None or school in schools


def pending_for(state, school, target=0):
    """Buffs already on the board that apply to `school`, against `target`.

    Counts the same things `make_blade_stack` counts — charms on the
    caster, wards on the target enemy, the aura — but only those that
    can actually act on this school, which it does not check.

    `target` matters on any board with more than one mob. Counting wards
    on `enemies[0]` while the hit is aimed at `enemies[1]` credits the
    turn with traps that will never fire, so the stack stops early and
    the nuke goes out unbuffed.

    That distinction is worth more than it sounds, and not only on mixed
    hands. A **Tri Trap** places three ward legs: fire, ice and storm. An
    ice wizard's hit consumes the ice leg; the fire and storm legs stay
    on the enemy for the rest of the duel, because nothing in an ice deck
    will ever trigger them. `State.traps` counts all three, so
    `make_blade_stack` believes it has a full buff stack while holding
    one live multiplier and two corpses — and fires early. On the
    project's own ice/stack deck that costs 17 points of kill rate.
    """
    n = 0
    for h in state.player.charms:
        if h.kind == "damage" and h.percent > 0 and h.matches(school):
            n += 1
    enemy = (state.enemies[target]
             if 0 <= target < len(state.enemies) else None)
    if enemy is not None:
        for h in enemy.wards:
            if h.kind == "prism" or (h.kind == "damage" and h.percent > 0
                                     and h.matches(school)):
                n += 1
    if state.player.aura is not None:
        n += 1
    return n


def choose_nuke(sim, state):
    """The hit this turn is being built around.

    Prefers the caster's own school on a tie-ish call, because power pips
    are worth two there and one elsewhere (`effective_pips`) — an ice
    wizard's Frost Beetle is cheaper to fire than a storm wand card of
    similar size, and only the former can be blade-stacked with the ice
    blades an ice deck carries.
    """
    nukes = [c for c in state.hand if c.kind in ("damage", "drain")]
    if not nukes:
        return None
    own = [c for c in nukes if c.school == sim.school]
    pool = own or nukes
    return max(pool, key=lambda c: c.damage)


# --------------------------------------------------------------------------
# trained policies, and the hole they fall into
# --------------------------------------------------------------------------
class TrainedPolicy:
    """A `QAgent` that declines to guess where it was never trained.

    A tabular Q-learner has no opinion about a state it has not visited:
    every action scores 0, `QAgent.greedy` takes `max` over the legal
    list, and `Featurizer.legal` puts `PASS` first -- so an unseen state
    silently becomes "pass". `rl_agent.py:66-77` records this exact
    failure ("Q stayed all-zero, and greedy fell through to the first
    legal action -- PASS. The agent scored a clean 0.0%").

    Live play walks into it immediately, and for a reason that is
    invisible unless you go looking. `train_agent` defaults to
    `player_hp=10**9`, and `Featurizer.key` puts `-1` in the health slot
    for an immortal fight and a real bucket otherwise:

        php = min(int(s.player_hp // 300), 9) if sim.player_hp0 < 10**9 else -1

    So a policy trained with the default is keyed on `php == -1` in every
    state it ever saw, and a live wizard with 800 health is keyed on
    `php == 2`. The two state spaces do not intersect **at all**, the Q
    table is uniformly zero for everything live, and the agent passes
    every single turn. Which is what it did.

    Training mortal fixes the overlap. It does not fix the general
    problem: real bosses, wand item cards and unfamiliar health buckets
    will always produce states no amount of training visited. So this
    wraps the agent rather than trusting it -- where the table has an
    opinion it is used, and where it does not the fallback plays. The
    miss rate is counted, because "the trained policy had never seen 94%
    of these boards" is the single most useful thing to know about a live
    run of a learned policy.
    """

    #: How many times the winning action must actually have been updated
    #: before the table is allowed to play it. "Is this entry non-zero"
    #: cannot tell one lucky episode from ten thousand, and a single
    #: visit is not an estimate -- it is a sample. Measured on a
    #: 12,000-episode table over a board ladder: gating at 20 lifts the
    #: easy board from 84% to 93%, the 900 HP board from 66% to 84%
    #: (past the heuristic's 79%), and finds 17% on a 480x2 board where
    #: both the ungated table and the heuristic score 0%.
    MIN_VISITS = 20

    def __init__(self, agent, fallback=None, min_visits=None):
        self.agent = agent
        self.fallback = fallback or greedy_ttk()
        self.min_visits = (self.MIN_VISITS if min_visits is None
                           else int(min_visits))
        self.seen = 0
        self.missed = 0
        #: visits behind the last decision, for the miss reason
        self.last_support = 0
        #: which path the LAST decision took. Surfaced per round, because
        #: "is the model I picked actually driving?" is otherwise
        #: unanswerable from the outside -- a trained policy and its
        #: fallback look identical in a decision log.
        self.last_source = ""
        #: why the last miss missed, in the operator's own terms. "it
        #: always goes to fallback" is not actionable; "you set mob HP
        #: 690 and walked into a 1,500 HP boss" is, and it is computable
        #: from what training was told.
        self.last_reason = ""

    @property
    def last_candidates(self):
        """Whatever the fallback weighed, when the fallback decided.

        Empty when the Q table drove: a tabular lookup does not produce a
        comparison to show, and inventing one would misrepresent how the
        decision was made.
        """
        if self.last_source.startswith("fallback"):
            return getattr(self.fallback, "last_candidates", [])
        return []

    @property
    def coverage(self) -> float:
        total = self.seen + self.missed
        return 1.0 if not total else self.seen / total

    def __call__(self, sim, s):
        try:
            key = self.agent.feat.key(sim, s)
            legal = self.agent.feat.legal(sim, s)
        except Exception:
            self.missed += 1
            self.last_source = "fallback (unreadable state)"
            self.last_reason = ""
            return self.fallback(sim, s)

        # Evidence, not merely a non-zero float. `.get`, not `[]`:
        # QAgent.Q is a defaultdict and indexing it here would insert a
        # zero for every state we merely asked about.
        support = getattr(self.agent, "support", None)
        if support is not None:
            self.last_support, _total = support(key, legal)
            known = self.last_support >= self.min_visits
        else:
            # A table trained before visit counts existed. Fall back to
            # the old test rather than refusing to use it at all.
            self.last_support = -1
            known = any(self.agent.Q.get((key, a), 0.0) for a in legal)
        if not known:
            self.missed += 1
            self.last_reason = self.why_missed(s)
            self.last_source = ("fallback — " + self.last_reason
                                if self.last_reason
                                else "fallback (state not in Q table)")
            return self.fallback(sim, s)

        self.seen += 1
        self.last_source = "Q table"
        self.last_reason = ""
        return self.agent.policy()(sim, s)

    def why_missed(self, s) -> str:
        """Which fact about this board the table was never trained on.

        `Featurizer.key` buckets enemy health absolutely (`hp // 250`)
        and carries its targeting tuple only above one enemy, so a board
        outside the trained band or above the trained mob count is not a
        near miss -- it is a key the table has no entry of any kind for.
        Measured: crossing the bucket edge at 1,250 HP takes the agent
        from 63% to 0% for a 1.6% change in enemy health.

        Empty when nothing can be said. `trained_on` is stamped by
        whoever trained the agent; without it there is no band to
        compare against and inventing one would be worse than silence.
        """
        if 0 <= self.last_support < self.min_visits:
            if self.last_support == 0:
                pass          # never seen at all -- the band may explain it
            else:
                return (f"the table played this board {self.last_support} "
                        f"time(s) in training, under the {self.min_visits} "
                        f"it needs to be an estimate rather than a sample")
        band = getattr(self.agent, "trained_on", None)
        if not band:
            return ""
        alive = [e for e in s.enemies if e.alive]
        mobs = band.get("mobs")
        if mobs and len(alive) > mobs:
            return (f"{len(alive)} mobs, trained for up to {mobs} — the "
                    f"state key changes shape above the trained count, so "
                    f"nothing matches at all")
        # Per-count first: the winnable span differs sharply by mob
        # count, so "above the band" means a different number depending
        # on how many are on the board.
        per_count = (band.get("bands") or {}).get(len(alive))
        lo, hi = per_count or (band.get("hp") or (None, None))
        if lo is not None and alive:
            # Buckets, not raw health. `Featurizer.key` stores
            # `hp // HP_BUCKET`, so a 480 HP mob and a 365 HP band edge
            # are the same symbol and the band cannot be what the table
            # failed to recognise. Comparing the raw numbers blamed the
            # band for every miss on a board whose biggest mob happened
            # to sit past the edge, including the ones it explains
            # nothing about.
            from rl_agent import HP_BUCKET

            def bucket(hp):
                return min(int(hp // HP_BUCKET), 24)

            biggest = max(e.max_hp for e in alive)
            b, blo, bhi = bucket(biggest), bucket(lo), bucket(hi)
            if not (blo <= b <= bhi):
                side = "above" if b > bhi else "below"
                return (f"a {biggest:,.0f} HP mob is {side} the "
                        f"{lo:,.0f}–{hi:,.0f} band this table was trained "
                        f"on")
        schools = band.get("schools") or []
        here = {getattr(e, "school", "") for e in alive} - {""}
        if schools and here and not (here & set(schools)):
            return (f"trained against {'/'.join(sorted(schools))}, fighting "
                    f"{'/'.join(sorted(here))}")
        return ""


def trained_policy(agent, fallback=None, min_visits=None):
    """`policy(sim, state)` for a QAgent, with a fallback for unseen states."""
    return TrainedPolicy(agent, fallback, min_visits)


# --------------------------------------------------------------------------
# lookahead
# --------------------------------------------------------------------------
class _Fixed:
    """A deterministic RNG for rollouts.

    Rollouts are used to *compare* candidate moves, so the comparison has
    to be free of the engine's noise -- a nuke that happened to fizzle in
    one rollout and not another would be ranked on luck. Zeros mean every
    cast lands and nothing crits; ranged damage takes its midpoint.
    """

    def random(self):
        return 0.0

    def uniform(self, a, b):
        return (a + b) / 2.0

    def choice(self, seq):
        seq = list(seq)
        return seq[len(seq) // 2]

    def shuffle(self, seq):
        return None

    def randrange(self, *a, **k):
        return 0

    def randint(self, a, b):
        return a

    def sample(self, population, k):
        return list(population)[:k]


#: The candidates for the policy a rollout plays after its first move.
#: This is the one place learning fits an engine this size: the
#: continuation is a single small policy reused on every board and every
#: rollout, so improving it improves every decision without needing to
#: have *seen* the board. Measured across 30 real-creature boards,
#: swapping only the continuation and holding `greedy_ttk` fixed moves
#: kill rate 40.0% -> 55.3%, and the shipped choice sits 6.9 points below
#: the best -- it over-buffs inside the rollout, spending horizon on
#: traps the line never cashes.
#:
#: Fitting one bought nothing: a 17-weight CEM-trained continuation tied
#: the five-line heuristic exactly (55.300% vs 55.300%), and a depth-2
#: inner search beat it by 0.3 points for 11x the cost. The headroom is
#: in WHICH, not in fitting -- so this was a five-way choice, not a
#: model, until the sixth candidate EARNED a seat: "neural" is the
#: search distilled into an entity net (deimos_bridge/neural_net.py;
#: neural_bc.py behaviour-clones plus a DAgger round, then
#: neural_es.py optimizes the seat's own objective by evolution
#: strategies from that seed). Held out at n=800 paired: first on the
#: storm deck's hard board on two fresh streams (29.9/31.5 against
#: 15.8 for the best hand-coded) and on the starter deck's hard board
#: (22.4/22.1 against 17.1), tied-first at 65.5 on the starter's mid
#: board -- and LAST on the buff-heavy deck's hard board (9.1 against
#: 24.1). Deck-specific like every other candidate, which is why the
#: per-deck paired probes decide where it actually drives.
#: "neural-b" is the SECOND seat -- the campaign-4 charm specialist,
#: trained where the spread survey measured the biggest unclaimed
#: lever. Held out on the survey points: first on the buff-heavy
#: deck's board on both fresh streams (38.2/37.8 against 32.9/31.9
#: for school-aware(0), the previous leader there), and second-close
#: on the myth board (40.0 vs school-aware(3)'s 41.1), which stays
#: the hand-coded candidate's ground. Separate weights file; the
#: sweep prices both nets like any other candidate.
#: "neural-c" is the THIRD seat -- the campaign-5 sustain specialist
#: (death/myth/life decks: scheduled damage and lifesteal). Held out
#: on the survey points: first on L30 death on both fresh streams
#: (34.9/38.1 vs school-aware(3)'s 28.5/31.8), a dead tie with the
#: leader on L30 life, and behind the leaders on L50 death and L30
#: myth, which stay hand-coded ground.
CONTINUATIONS = ("nuke-asap", "school-aware(3)", "school-aware(0)",
                 "blade-stack(2)", "blade-stack(3)", "neural",
                 "neural-b", "neural-c")

#: The shipped default, kept because the right answer is deck-specific
#: and the alternative is right only on some decks: the same swap that
#: is +5.2 on one deck is -7.6 on another. See `choose_continuation`.
DEFAULT_CONTINUATION = "school-aware(3)"

_CONTINUATION = None
_CONTINUATION_NAME = None


def build_continuation(name):
    """One of `CONTINUATIONS`, as a policy."""
    from w101_sim import make_blade_stack, strat_nuke_asap

    if name == "nuke-asap":
        return strat_nuke_asap
    if name == "school-aware(0)":
        return school_aware_blade_stack(0)
    if name == "blade-stack(2)":
        return make_blade_stack(2)
    if name == "blade-stack(3)":
        return make_blade_stack(3)
    if name.startswith("neural"):
        # Falls back to the default heuristic if the weights cannot
        # load: a missing or stale file must never break a sweep or a
        # live fight -- it just costs this candidate its chance.
        try:
            from . import neural_net
            path = {"neural": None,
                    "neural-b": neural_net.DEFAULT_WEIGHTS_B,
                    "neural-c": neural_net.DEFAULT_WEIGHTS_C}[name]
            return neural_net.net_policy(path)
        except Exception:
            return school_aware_blade_stack(3)
    return school_aware_blade_stack(3)


def set_continuation(name):
    """Choose the rollout's continuation for this deck. Returns the name.

    Deck-scoped rather than global because the measurement says so: the
    continuation that is +5.2 points on one deck is -7.6 on another, so
    there is no universal answer to hardcode.
    """
    global _CONTINUATION, _CONTINUATION_NAME
    name = name if name in CONTINUATIONS else DEFAULT_CONTINUATION
    _CONTINUATION_NAME = name
    _CONTINUATION = build_continuation(name)
    return name


def continuation_name():
    return _CONTINUATION_NAME or DEFAULT_CONTINUATION


#: Horizons the per-deck pick sweeps. Two, not a range: 12 is the
#: shipped default and 6 is the one with a measured, reproduced win --
#: +10.3 points on a two-mob attrition board, three independent
#: reproductions -- because a shorter horizon denies the rollout the
#: patience that long setup lines (the buff-spam) are made of.
#:
#: A finer grid was measured and rejected (n=300 paired truth tables
#: over the full continuation x (4,6,8,12) grid, four contested boards,
#: two deck profiles): h8 never beat max(h6, h12), and h4 was +2.7 on
#: one board while 9-11 points WORSE on its neighbours -- a horizon
#: that cannot see the second mob's worth of fight ranks lines by
#: their opening. Two candidates keep the sweep half the cost of four
#: and lose nothing real.
HORIZONS = (6, 12)

#: Determinization widths the driver probe sweeps. More sampled draw
#: orders answer the hidden-hand question better and cost linearly;
#: which width pays is deck-dependent like everything else here, and
#: k=6 was the only one ever measured before this sweep existed.
SEARCH_WIDTHS = (4, 6, 10)


def _probe_mobs(board, dmg):
    """(boss, extras) for one probe board spec.

    A spec is `(hp, n_mobs, school)` -- flat mobs hitting for `dmg` --
    or `(hp, n_mobs, school, names)` carrying observed enemy names, in
    which case every named mob that resolves in the catalog becomes the
    casting boss it is in the live fight (spell pool, opening pips,
    exact defences) at the probe's health.

    MEASURED NULL, recorded so the next session does not redo it: since
    live rollouts price named enemies as casters, tuning on caster
    probes instead of flat ones looks like it should matter -- and on
    three boards (weak caster, spike caster, 6-pip opener) it changed
    the pick twice (nuke-asap -> school-aware(0)) without ever changing
    the outcome. Ketil@1300: 51.5% flat-tuned vs 50.1% caster-tuned at
    n=800. Jack of Knaves@1100 (opens with 6 pips, the regime flat
    probes cannot express at all): 35.6/37.9/39.4 vs 40.6/37.8/38.1
    across three n=800 evaluations. Two single-run deltas (+8.7 at
    n=300, +5.0 at n=800) both collapsed on replication -- the
    selection-noise trap, again. So the GUI keeps tuning on flat probes
    (2x faster, measured equivalent); the named form exists because a
    future per-boss tuner needs exactly this hook and the experiment
    should not have to be rebuilt to rerun.
    """
    from w101_sim import Boss

    hp, mobs, mob_school = board[:3]
    names = list(board[3]) if len(board) > 3 and board[3] else []
    out = []
    for i in range(mobs):
        b = None
        if i < len(names) and names[i]:
            try:
                from .bestiary import full_boss
                b = full_boss(names[i], hp)
            except Exception:
                b = None
        if b is None or not b.pool:
            b = Boss(name=f"probe {i}" if i else "probe", hp=hp,
                     school=mob_school, dmg=dmg)
        out.append(b)
    return out[0], out[1:]


def choose_search(cards, deck, school, boards, n=60, on_probe=None,
                  dmg=0):
    """The best (continuation, horizon) for this deck, measured.

    `boards` is [(hp, n_mobs, school)] -- see `_probe_mobs` for the
    named-caster form. Ten probes of a few seconds each -- against
    12,000 episodes for a table that then only works on one deck at one
    health band. That ratio is the whole argument for this shape of
    learning: the search is the model, and the per-deck learning is a
    handful of measured choices about how to run it.

    `dmg` puts incoming pressure on the probe boards. At zero the wizard
    is never punished for patience, so every horizon ties and the sweep
    cannot see the one thing it exists to decide.

    Every candidate is measured on the SAME seed stream
    (`evaluate_paired` -- common random numbers), because the pick is an
    argmax over differences and shared seeds cancel the shared luck: a
    stream of bad draws hits every candidate equally instead of sinking
    whichever one happened to be under measurement. The probes stay the
    same size; only the comparison gets sharper.

    Returns (continuation, horizon, {"name@h6": kill rate, ...}).
    """
    from w101_sim import Sim, evaluate_paired

    # Explicit, never installed: the GUI runs this sweep while the live
    # fight keeps playing, and a sweep that set the global per candidate
    # had live decisions rolling out with whatever probe setting
    # happened to be under measurement at that moment. Only the winner
    # is installed, once, below.
    candidates = {}
    for name in CONTINUATIONS:
        cont = build_continuation(name)
        for horizon in HORIZONS:
            candidates[f"{name}@h{horizon}"] = greedy_ttk(
                horizon, continuation=cont)

    totals = {k: 0.0 for k in candidates}
    for board in boards:
        boss, extra = _probe_mobs(board, dmg)
        sim = Sim(cards, deck, school, boss, enemies=extra)
        for k, st in evaluate_paired(sim, candidates, n=n).items():
            totals[k] += st["win_rate"]
    scores = {}
    for key in candidates:
        scores[key] = totals[key] / max(1, len(boards))
        if on_probe:
            on_probe(key, scores[key])
    best = max(scores, key=lambda k: scores[k])
    best_name, best_h = best.rsplit("@h", 1)
    set_continuation(best_name)
    set_search_horizon(int(best_h))

    # Third choice: the driver itself. Determinized search (k=6) beat
    # the plain lookahead by +2.8 and +3.3 points on richer decks and
    # lost by 3.6 on a starter deck -- deck-dependent, exactly like the
    # continuation and the horizon, so it is picked the same way: by
    # measurement on the same probe boards. It is ~12x slower per
    # decision, which is nothing against a ~30 s live planning phase.
    from search_policy import make_search_policy

    # The tuned continuation, so the probe measures the driver that
    # would actually play -- passed explicitly rather than read from
    # the global, same hygiene as the sweep above. The tuned lookahead
    # rides in the SAME paired call: the old comparison put the ttk
    # number from one seed stream at full n against search numbers from
    # another at n/3, and the +0.02 threshold was carrying that
    # mismatch as well as the noise it was meant for.
    tuned_cont = build_continuation(best_name)
    rivals = {"ttk": greedy_ttk(int(best_h), continuation=tuned_cont)}
    for k in SEARCH_WIDTHS:
        rivals[f"search(k={k})"] = make_search_policy(base=tuned_cont, k=k)
    totals = {k: 0.0 for k in rivals}
    for board in boards:
        boss, extra = _probe_mobs(board, dmg)
        sim = Sim(cards, deck, school, boss, enemies=extra)
        for k, st in evaluate_paired(sim, rivals,
                                     n=max(20, n // 3)).items():
            totals[k] += st["win_rate"]
    ttk_paired = totals["ttk"] / max(1, len(boards))
    for k in SEARCH_WIDTHS:
        scores[f"search(k={k})"] = totals[f"search(k={k})"] \
            / max(1, len(boards))
    global _DRIVER
    top_k = max(SEARCH_WIDTHS, key=lambda k: scores[f"search(k={k})"])
    _DRIVER = (f"search(k={top_k})"
               if scores[f"search(k={top_k})"] > ttk_paired + 0.02
               else "ttk")
    return best_name, int(best_h), scores


#: Which policy actually drives when the GUI says "ttk-lookahead":
#: the plain lookahead, or determinized search when the per-deck probes
#: measured it ahead by more than noise (+2 points on the probe mean).
_DRIVER = "ttk"


def tuned_driver(continuation=None, horizon=None, driver=None):
    """The measured-best driver for the tuned deck.

    The search drives with the TUNED continuation as its rollout policy,
    not `make_search_policy`'s hardcoded blade-stack default -- the
    continuation choice is worth ~14 points inside a rollout, and a
    driver that ignored it would play a different game from the one the
    probes measured.

    Called with no arguments it reads the deck-scoped globals, and keeps
    reading them: `_rollout` looks up `_CONTINUATION` and `_HORIZON` at
    DECISION time, not here, so a later `set_continuation` reaches a
    policy already built. That is what makes "tune, then keep playing"
    work.

    It is also exactly wrong for a party. Four wizards hold four decks
    and the quartet is deck-scoped, so four `set_continuation` calls
    leave all four playing whichever was written last. Passing the pick
    in explicitly binds it into the closure instead, which is why these
    arguments exist -- see `gui/live.LiveWorker._seat_search`.
    """
    name = _DRIVER if driver is None else driver
    m = re.match(r"search\(k=(\d+)\)$", str(name or ""))
    if m:
        from search_policy import make_search_policy
        base = _continuation() if continuation is None else continuation
        return make_search_policy(base=base, k=int(m.group(1)))
    if continuation is None and horizon is None:
        return greedy_ttk()
    return greedy_ttk(max_turns=horizon, continuation=continuation)


def driver_name():
    return _DRIVER


def set_driver(name):
    """Install the tuned driver by name; anything unknown means ttk."""
    global _DRIVER
    _DRIVER = (name if re.match(r"search\(k=\d+\)$", str(name or ""))
               else "ttk")
    return _DRIVER


def choose_continuation(cards, deck, school, boards, n=60, on_probe=None):
    """Back-compatible wrapper over `choose_search`, horizon fixed at the
    default. Prefer `choose_search`."""
    from w101_sim import Boss, Sim, evaluate_paired

    # explicit continuations, never installed; paired seeds, same as
    # choose_search
    candidates = {name: greedy_ttk(continuation=build_continuation(name))
                  for name in CONTINUATIONS}
    totals = {k: 0.0 for k in candidates}
    for hp, mobs, mob_school in boards:
        boss = Boss(name="probe", hp=hp, school=mob_school, dmg=0)
        extra = [Boss(name=f"probe {i}", hp=hp, school=mob_school, dmg=0)
                 for i in range(1, mobs)]
        sim = Sim(cards, deck, school, boss, enemies=extra)
        for k, st in evaluate_paired(sim, candidates, n=n).items():
            totals[k] += st["win_rate"]
    scores = {}
    for name in candidates:
        scores[name] = totals[name] / max(1, len(boards))
        if on_probe:
            on_probe(name, scores[name])
    best = max(scores, key=lambda k: scores[k])
    set_continuation(best)
    return best, scores


def _continuation():
    """The policy a rollout plays after its first move. Built once."""
    global _CONTINUATION
    if _CONTINUATION is None:
        set_continuation(DEFAULT_CONTINUATION)
    return _CONTINUATION


def _split(action):
    """(card, target) from whatever a policy returned.

    `Sim._normalize_action` already accepts a `(card, target)` tuple, so
    that is the contract policies use to aim -- no string encoding, and
    it flows through `Sim.run`, `evaluate` and `evaluate_paired`
    untouched.
    """
    if isinstance(action, tuple):
        return action[0], (action[1] if len(action) > 1 else 0)
    return action, 0


#: How to rank candidates on a board the rollout thinks it loses. This
#: branch is not an edge case -- it fires on 17% of candidates on a
#: level-5 board and on 37-100% of them on a hard one -- so it was worth
#: asking whether "bank the most damage" is the right bet.
#:
#: Measured twice, and the second measurement had what the first
#: lacked: a board where the rankings actually disagree. The original
#: run (3 decks x 4 boards, 400 fights a cell) was a null -- damage
#: 67.5%, kills +0.75, survive -0.06, all inside the noise floor --
#: because its boards rarely posed the choice the rankings differ on.
#: A live trace then posed it exactly: a dying board with a 75 HP
#: minion dealing ~50/round beside a full-health boss, where "bank the
#: most damage" hit the boss for 81 banked instead of removing the
#: attacker for 75. On that board shape "kills" wins +2.4/+2.4/+1.2
#: across three paired seed streams (n=500 each), and re-measured on
#: two canonical boards without the pattern it makes IDENTICAL
#: decisions -- the credit only fires where a kill is actually on
#: offer. So "kills" ships: upside where threat removal is real,
#: provably zero cost where it is not.
#:
#: The same runs also killed a hypothesis: buff rate is flat at 23-47%
#: across ALL THREE rankings, so whatever drives over-buffing, it is
#: not the losing-board objective.
LOST_RANKING = "kills"

#: The learned leaf value, if one is installed. `None` keeps the shipped
#: behaviour bit-for-bit. When set, a rollout that runs out of horizon
#: still alive is ranked by what the LEAF is worth -- P(win) from
#: scale-free features -- instead of by damage banked, which cannot tell
#: a nearly-won board from a nearly-lost one at the same total.
_LEAF = None


def set_leaf_value(model):
    """Install (or clear, with None) the leaf evaluator."""
    global _LEAF
    _LEAF = model
    return model


def load_leaf_value():
    """The committed weights, installed. Returns the model or None."""
    try:
        from .leaf_value import LeafValue
        return set_leaf_value(LeafValue.load())
    except Exception:
        return None


#: How the rollout prices a cast that can miss. "optimistic" is what
#: shipped: the deterministic rollout lands every cast, so a 75% Fire
#: Cat is planned exactly like a 100% card -- at level five nothing in a
#: fire deck is above 75%, so the whole plan is built on damage that
#: only arrives three times in four. "expected" prices each uncertain
#: hit at accuracy x damage instead: still deterministic, no random
#: fizzles polluting the comparison, but a blade's certain cost is
#: finally weighed against a hit's uncertain payoff.
#:
#: Measured alone, "expected" LOSES 4.3 points: the honest answer to a
#: discounted board is "I lose this", the sentinel rate triples, and the
#: sentinel's damage ranking is a bad judge. It is only safe to turn on
#: together with the leaf value, which gives the rollout something
#: better to say at exactly the states honesty produces more of.
ROLLOUT_ACCURACY = "optimistic"

_EV_CARDS = {}


def ev_card(card):
    """`card` priced at expected value: lands always, worth acc x damage.

    Buffs, shields and traps are left untouched -- a Fireblade really is
    100% accurate -- which is the point of the whole exercise.
    """
    acc = max(0.0, min(1.0, float(getattr(card, "accuracy", 1.0) or 1.0)))
    if acc >= 1.0 or card.kind not in ("damage", "drain"):
        return card
    priced = _EV_CARDS.get(card.name)
    if priced is None:
        import dataclasses

        def scale(op):
            out = dict(op)
            for field in ("amount", "total"):
                if isinstance(out.get(field), (int, float)):
                    out[field] = out[field] * acc
            if isinstance(out.get("outcomes"), (list, tuple)):
                out["outcomes"] = [v * acc for v in out["outcomes"]]
            return out

        priced = dataclasses.replace(
            card, accuracy=1.0, damage=(card.damage or 0) * acc,
            ops=[scale(o) if o.get("op") in ("hit", "dot") else dict(o)
                 for o in (card.ops or [])])
        _EV_CARDS[card.name] = priced
    return priced


#: Enemy health the REST of the circle takes off the board each round.
#: Zero is a wizard fighting alone, and is the shipped behaviour bit for
#: bit -- every solo path, every test and every tuning sweep leaves this
#: at zero and gets the same numbers it always got.
_ALLY_RATE = 0.0


def set_ally_rate(dps):
    """Tell the rollout how hard the rest of the party is hitting.

    A rollout that models only its own caster is not merely pessimistic,
    it is *blind in a way that destroys the ranking*. In 28 of the 58
    scored rounds of the first live two-wizard run, no candidate killed
    inside the horizon -- so `turns`, the objective this policy is named
    for, was the same sentinel for every option and the whole decision
    fell through to the damage tiebreak. That tiebreak cannot tell a
    Tower Shield from doing nothing: 17 of those 58 rounds played a card
    the rollout scored bit-identical to passing.

    The reason nothing kills is arithmetic, and the exports state it
    themselves -- no reconstruction involved. Each candidate table
    carries the banked damage of the best line over the horizon, so
    board health divided by that rate is turns-to-clear alone. Of the 21
    rounds where both wizards were scored against the same board, 10
    read "the party clears inside twelve turns and neither wizard alone
    does". Fight 1 round 3, three 435hp Sand Stalkers: 15.0 turns for
    the ice wizard alone, 21.9 for the fire one, 8.9 together. Both of
    them lost that fight.

    So the rate has to be honest in both directions. Too low and the
    sentinel stays; too high and it returns for the opposite reason,
    since a party modelled as overwhelming clears the board on turn one
    of every line and `turns` ties again at 1. Feed this what the party
    is measured to do (`Telemetry.damage_rate`, 51 and 57 a round for
    those two wizards), never a hopeful number, and never the
    coordinator's priced plan for the round -- that reads 172 and 201,
    because it prices a single cast and no wizard casts its best card
    every round.
    """
    global _ALLY_RATE
    _ALLY_RATE = max(0.0, float(dps or 0.0))
    return _ALLY_RATE


def ally_rate():
    return _ALLY_RATE


@contextlib.contextmanager
def party_rate(dps):
    """`set_ally_rate` for the duration of one decision, then back.

    The hivemind plans seats one at a time and each seat faces a
    different set of teammates, so the rate is per *call*, not per run.
    Restoring on the way out is what keeps a seat that raises from
    leaving its neighbours' rate installed on the next round's solo
    fallback.
    """
    before = _ALLY_RATE
    set_ally_rate(dps)
    try:
        yield
    finally:
        set_ally_rate(before)


def rank_candidate(cand):
    """`greedy_ttk`'s ranking, as far as a recorded `Candidate` can say it.

    For a caller holding the decision *after* the fact -- the live
    handler picking a runner-up when the chosen card would not go out --
    rather than the policy, which ranks live card objects and has more
    to go on.

    An approximation, and the gap is worth naming: the policy's key ends
    `..., card.pips, card.damage, target`, and `card.damage` is the
    card's nominal damage, which a `Candidate` does not record.
    `Candidate.damage` is a different quantity entirely -- damage banked
    over the whole rollout. So this reproduces the first three keys
    exactly and stops, which is enough to order a hand and is honest
    about not being the same function.

    Kept here rather than in the handler so the ranking has one home; a
    copy over there would drift the first time this key changes, and it
    has changed before.
    """
    return (cand.turns, -cand.damage, cand.pips)


def rollout_throughput(candidates):
    """Damage a round the best line in `candidates` expects to deal.

    The cold start for `set_ally_rate`: before a seat has finished a
    fight it has no measured rate to report, and the first duel of a
    session is exactly where the party-blind rollout hurt most -- both
    wizards of the day's run died in fight 1.

    The rollout's own answer is free and already computed. Checked
    against what the two wizards went on to deal per round (51 and 57),
    their best fight-1 lines banked 1045 and 679 over a horizon of 12,
    i.e. 87 and 57 a round: one exact and one high by 1.7x. That beats
    the other number lying around -- the coordinator's priced plan for
    the round, 172 and 201 a round, high by 3.2x -- because a plan
    prices a single cast and a wizard cannot cast its best card every
    round.

    Returns 0.0 when there is nothing to read, which models no ally.
    """
    best = None
    for cand in candidates or ():
        if cand.card == "pass" or not cand.horizon:
            continue
        rate = float(cand.damage) / float(cand.horizon)
        if best is None or rate > best:
            best = rate
    return best or 0.0


def _allies_hit(state, rate):
    """Take `rate` health off the board, weakest mob first.

    Weakest first because that is what the party demonstrably does: the
    ledger writes a mob the committed casts kill down to zero and the
    overkill guard stops a second wizard firing into it, so the circle
    finishes what is nearly dead before it starts something new.
    Spreading the rate evenly instead would model a party that never
    kills anything, which is the failure this exists to fix.
    """
    left = float(rate)
    if left <= 0:
        return
    for enemy in sorted((e for e in state.enemies if e.alive),
                        key=lambda e: e.hp):
        if left <= 0:
            break
        bite = min(left, float(enemy.hp))
        enemy.hp -= bite
        left -= bite


def is_sentinel(turns, max_turns) -> bool:
    """Did this rollout fail to clear the board, rather than take `turns`?

    `turns > max_turns` is the obvious test and it is not correct.
    `_lost_score` credits a losing line 0.4 per kill, so a `stalled`
    rollout that killed three of four mobs scores `max_turns + 1 - 1.2`
    -- BELOW the horizon, and indistinguishable from a real turn count
    by size alone. (That the credit can also let a losing line outrank a
    winning one is a separate, older problem; this only has to tell them
    apart.)

    So it tests the shape instead. A cleared board returns the loop
    variable, always a whole number in 1..max_turns. Every sentinel is
    either above the horizon or carries a fractional kill credit: with
    `LOST_RANKING = "kills"` that is 0.4, 0.8 or 1.2, since a stalled or
    dead rollout has at least one mob still standing and Wizard101 seats
    at most four. "survive" is fractional too, and "none" is integral but
    always above the horizon.

    The one arrangement that would defeat it -- five kills, an exact 2.0
    -- needs six enemies in a duel, which the game does not have.
    """
    return turns > max_turns or float(turns) != int(turns)


def _lost_score(rank, dealt, kills, turn):
    """A 2-tuple, always: the caller unpacks (turns, neg_damage).

    The extra ordering rides in the RANK rather than as a third element,
    so the second stays real banked damage and the decision panel keeps
    showing a number that means something. The offsets are small enough
    that a lost line can never outrank a won one: at most 4 kills x 0.4
    is 1.6, against the +1 that separates `stalled` from a clean win.
    """
    if LOST_RANKING == "kills":
        return (rank - 0.4 * kills, -dealt)
    if LOST_RANKING == "survive":
        return (rank - 0.05 * turn, -dealt)
    return (rank, -dealt)


def _rollout(sim, state, first_action, max_turns=12, target=0,
             continuation=None, allies=None):
    """Score playing `first_action` now: (turns_to_kill, -damage_dealt).

    Lower is better on both, so `min` ranks them.

    Two numbers because turn counts tie constantly at low level -- a
    170hp mob dies on turn two whether you open with a trap or a beetle,
    and against a 400hp one every opening kills on turn four. The tie is
    where the interesting choice lives, so it breaks on cumulative damage
    dealt: more damage banked by the same turn means more margin against
    a fizzle, a heal or an incoming shield. That is what separates
    Snow Serpent from Frost Beetle when both reach the same turn count,
    and what stops a third Ice Trap from looking free.

    The continuation is `school_aware_blade_stack`, not "biggest
    affordable hit". That matters more than it sounds: a hit-only
    continuation never casts a second buff, so any line that *opens* with
    a buff is scored as though the buff were never used, and buff-heavy
    decks get systematically undervalued. Measured on the project's own
    decks, that alone was worth 21 points of kill rate on ice/stack and
    11 on balance/oneshot.

    `allies` is enemy health the rest of the circle removes each round
    (`set_ally_rate` for why, and for the numbers). It lands AFTER the
    caster's own cast, so a card that kills a mob outright is still
    credited with the kill rather than having it stolen by a teammate
    the engine happened to resolve first -- the real turn order is
    unknowable, and this is the reading that keeps each candidate's own
    contribution visible.
    """
    import copy

    # Explicit beats global: the tuning sweeps evaluate candidate
    # continuations while a live fight may be deciding, and a sweep that
    # installed each candidate globally had the live rollout playing
    # whatever the probe happened to be measuring.
    continuation = continuation or _continuation()
    allies = _ALLY_RATE if allies is None else max(0.0, float(allies))
    probe = copy.copy(sim)
    probe.rng = _Fixed()
    s = copy.deepcopy(state)
    if ROLLOUT_ACCURACY == "expected":
        s.player.hand[:] = [ev_card(c) for c in s.player.hand]
        s.player.deck[:] = [ev_card(c) for c in s.player.deck]

    def enemy_alive():
        return any(e.alive for e in s.enemies)

    def board_hp():
        """Enemy health left, floored at zero per mob.

        Flooring is the point. `Sim._strike` does `target.hp -= dmg` and
        `cast` returns the raw number, so a 300-damage nuke into a mob
        with 50 left used to bank 300 -- and the tiebreak below *rewards*
        banking more. That is a scoring function that prefers overkill,
        which is the opposite of the "what is the smallest thing that
        kills this?" question. Damage past lethal buys nothing, so it
        counts for nothing.
        """
        return sum(max(e.hp, 0.0) for e in s.enemies)

    start_board = board_hp() or 1.0

    def dealt_now():
        """Everything the board has lost since the rollout began.

        The board DELTA, not a sum over casts -- the old per-cast
        accounting silently excluded DoT ticks (they land in
        `end_round`, not in a cast), so a Fire Elf line that killed
        with its burn banked only its initial hit. On a live fire
        wizard that bias decided ties the wrong way every time: a
        blade line and a DoT line both killing on the same turn
        compared 182 banked against 135, and the blade won three
        rounds running while the burn was doing the actual work.
        """
        return start_board - board_hp()

    #: Ranks worse than any line that clears the board, and worse than
    #: one that merely runs out of horizon while alive. `unplayable` is
    #: for a move that cannot be made at all.
    unplayable = (max_turns + 3, 0.0)

    def kills():
        return sum(1 for e in s.enemies if not e.alive)

    def died(turn=0):
        """Dying still banked whatever it banked.

        This used to return a flat constant, and that was the bug behind
        "it spams every trap card": on a board where no line survives the
        horizon, *every* candidate scored identically, the comparison
        collapsed, and the decision fell through to the tiebreak -- which
        picks the cheapest card, and an Ice Trap costs zero pips. A
        policy in trouble should still play the line that gets furthest,
        not the one that costs least.

        Which "furthest" is the right one is `LOST_RANKING`, and it
        matters more than it looks: this branch fires on 17% of
        candidates on a level-5 board and on 37-100% of them on a hard
        one, so it is not an edge case -- it is a large fraction of every
        decision the lookahead makes.
        """
        return _lost_score(max_turns + 2, dealt_now(), kills(), turn)

    def stalled(turn=0):
        if _LEAF is not None:
            # Ranked by what the surviving position is worth, not by
            # damage banked. The value rides in the damage slot scaled
            # to board units, so the tuple shape and the panel's display
            # stay intact; within the stalled bucket only these compare
            # against each other, so the units only have to be
            # consistent, and they are.
            try:
                worth = _LEAF(probe, s)
            except Exception:
                worth = 0.0
            return (max_turns + 1, -(dealt_now() + worth * start_board))
        return _lost_score(max_turns + 1, dealt_now(), kills(), turn)

    # find the copied card matching the chosen one
    action = None
    if first_action is not None:
        for c in s.hand:
            if c.name == first_action.name and probe.can_cast(s, c, target):
                action = c
                break
        if action is None:
            return unplayable

    for turn in range(1, max_turns + 1):
        if action is not None:
            try:
                # Aimed. Casting the whole rollout at enemy 0 made every
                # line look identical on a multi-mob board, so no target
                # could ever be *chosen* -- and a rollout that traps one
                # mob and hits another scores a buff that never fires.
                if not probe.can_cast(s, action, target):
                    return unplayable
                probe.cast(s, action, target)
            except Exception:
                return unplayable
        if turn > 1:
            # Turn 1 is already on the board. `Hivemind.plan` hands the
            # policy a state `Ledger.apply` has *already* taken this
            # round's committed party damage off, so hitting again here
            # would count the same casts twice -- and the only producer
            # of a non-zero rate is that same coordinator. Under-counts
            # by one round out of the horizon when a teammate is held or
            # casts nothing, which is the conservative direction.
            _allies_hit(s, allies)
        if not enemy_alive():
            return turn, -dealt_now()
        try:
            probe.end_round(s)
        except Exception:
            return stalled(turn)
        if not enemy_alive():
            return turn, -dealt_now()
        if not s.player.alive:
            return died(turn)

        action, target = _split(continuation(probe, s))
        if not (0 <= target < len(s.enemies)) or not s.enemies[target].alive:
            # The focus died to the move just played; let the
            # continuation re-aim rather than casting into a corpse.
            target = focus_target(s)

    return stalled(max_turns)


#: The rollout horizon `greedy_ttk` uses when the caller does not say.
#: Deck-scoped for the same reason the continuation is: measured on a
#: 12-cell grid at n=400, horizon 6 is +10.3 points on a two-mob
#: attrition board (reproduced three times: +17, +9, +10.3), never worse
#: than 2 anywhere, and a null on the mean -- a long horizon buys the
#: rollout patience for long setup lines, and on a board that punishes
#: patience that patience is the buff-spam. `choose_search` picks it per
#: deck on envelope probes; this is only the fallback.
DEFAULT_HORIZON = 12
_HORIZON = None


def set_search_horizon(h):
    """Choose the rollout horizon for this deck. None restores default."""
    global _HORIZON
    _HORIZON = int(h) if h else None
    return _HORIZON or DEFAULT_HORIZON


def search_horizon():
    return _HORIZON or DEFAULT_HORIZON


def greedy_ttk(max_turns: int = None, continuation=None):
    """Pick the move that kills soonest, by simulating each candidate.

    `max_turns=None` takes the deck-scoped horizon -- see
    `set_search_horizon`. `continuation=None` takes the deck-scoped
    continuation; the tuning sweeps pass one explicitly so measuring a
    candidate never means installing it globally under a live fight.

    `school_aware_blade_stack` stacks a *fixed* number of buffs and only
    then looks for a hit, which is wrong in both directions and both were
    visible on a real level-6 wizard:

      * It fires the biggest nuke it can afford rather than the one that
        kills fastest, so a 1-pip Ice Beetle goes out on turn one when
        waiting a turn for a 2-pip Ice Serpent would have been quicker.
      * It stacks its quota regardless of whether the buffs pay off. Three
        Ice Traps in the deck means three Ice Traps on the boss before a
        single hit, which against a low-health mob is three wasted turns.

    Both fall out of the same missing question: *does this move shorten
    the fight?* This asks it directly -- one ply of real lookahead over
    every castable card plus passing, scored by the engine's own damage
    model on a deterministic rollout. The buff/hit tradeoff, pip banking
    and diminishing returns on a fourth blade all come out of the
    arithmetic rather than a hand-tuned count.

    **Every candidate carries a target.** A move is a (card, enemy) pair,
    not a card, because on a board with a boss and a minion those score
    differently and the difference is the whole decision: a trap is worth
    nothing unless the hit it is buying lands on the same mob. Before
    this the rollout cast everything at `enemies[0]`, so every target
    scored identically and none could be chosen -- the live handler then
    clicked whichever mob the participant list put first, which is how
    traps ended up spread across two enemies.

    Slower than the heuristics -- roughly (castable × live enemies + 1)
    rollouts per decision -- which is still nothing next to a live
    planning phase.
    """
    fixed = max_turns
    fixed_continuation = continuation

    def strat(sim, s):
        # Cleared FIRST: a decision that ends early (nothing castable
        # this round) used to leave the PREVIOUS round's comparison on
        # the attribute, and the round record then showed a phantom --
        # "policy chose to pass" beside a candidate table claiming a
        # Pixie was chosen, copied verbatim from the round before.
        strat.last_candidates = []
        strat.last_party_blind = False
        max_turns = fixed if fixed is not None else search_horizon()
        from w101_sim import cast_price, cast_reach, castable

        foes = [i for i, e in enumerate(s.enemies) if e.alive]
        if not foes:
            foes = [0]

        candidates = []
        seen = set()
        for card in s.hand:
            if card.name in seen:
                continue
            # A self-buff or an AoE has one version of itself; only a
            # single-enemy card is worth rolling out once per mob.
            if _is_inert(card, s):
                continue
            aims = foes if aimed_at_one_enemy(card) else foes[:1]
            playable = [t for t in aims if sim.can_cast(s, card, t)]
            if not playable:
                continue
            seen.add(card.name)
            candidates.extend((card, t) for t in playable)

        if not candidates:
            return None

        # Third key: spend the least that still gets there -- but only
        # while the earlier keys mean something.
        #
        # Damage is counted floored at each mob's health, so two lines
        # that kill on the same turn and bank the same *effective*
        # damage are genuinely equivalent in outcome, and between
        # equivalent outcomes the cheaper card is strictly better: it
        # leaves pips banked and the bigger card still in hand. Two
        # rounds from the second live party run are exactly that and the
        # cheap card is right in both -- a 190hp board where Frost
        # Beetle and Evil Snowman both clear on turn 1, and a 348hp
        # board where both clear on turn 2. Spending three pips for a
        # one-pip result is the mistake there, not the fix.
        #
        # Above the horizon it means nothing at all. `turns` is a
        # sentinel, `neg_damage` is the whole rollout line's board delta,
        # and a tie says only that the lookahead could not tell the
        # candidates apart -- so "cheapest" is not thrift, it is a coin
        # flip that always lands on the free card. Measured over that
        # run's 197 scored rounds: 107 have no candidate killing inside
        # the horizon, and in 38 of those the chosen move was a weaker
        # card that tied a stronger one EXACTLY. Three consecutive rounds
        # of its fight 14 are the shape the operator reported --
        #
        #     r5  Frost Beetle (85 dmg) over Evil Snowman (270), on 4 pips
        #     r6  Frost Beetle (85 dmg) over Evil Snowman (270), on 5 pips
        #     r7  Ice Trap     (0 dmg)  over Evil Snowman (270), on 4n+1p
        #
        # -- a nine-round fight that dealt 180 while the pips piled up.
        #
        # So the key is CONDITIONAL, and the condition is not a knob:
        # it is whether the keys ahead of it carried information.
        #
        # Making it unconditional was tried, shipped and reverted
        # (d3b7962, bd99c57), and that is worth knowing before reaching
        # for it again. It measures WORSE solo and in horizon -- three of
        # six myth cells lose five points of kill rate on the project's
        # own decks (97.0 -> 91.5, 97.0 -> 92.5, 82.5 -> 77.5) and an
        # independent 66-cell sweep at 13,200 seeds gives 572 shipped
        # wins to 357 -- and BETTER in a party on live-shaped boards, 16
        # shipped-only wins against 205. Both reproduce, because it is
        # not a better tiebreak but a more aggressive one, and aggression
        # wins a race and loses an attrition fight. Conditioning on the
        # sentinel keeps the aggression where the alternative is a coin
        # flip and keeps the thrift where the outcome is real: on those
        # same six myth cells the conditional form is identical to this
        # one, cell for cell.
        #
        # A warning for whoever measures it next: **the test suite
        # cannot see this key at all.** 855 tests passed either way
        # while roughly a third of their `greedy_ttk` decisions resolved
        # differently, because 44% of decisions tie at (turns, damage)
        # and every test that asserts a specific card makes one or two
        # decisions, none of which tie. A green suite is not evidence
        # about this line. See the forced-tie tests in
        # `tests/test_hivemind.py`, which exist so that stops being true.
        # Priced once, off the rack as it stands before any of this is
        # cast. Both halves of the key are printed values that an X-pip
        # card does not honour -- it prints 0 pips and spends everything,
        # and prints damage PER PIP -- so `card.pips` made Heck Hound
        # unbeatable on the thrift key and `card.damage` compared its
        # 130-a-pip against a Fire Cat's 100 flat. See `cast_price`.
        #
        # The shape of the fix, on seven pips against a 300hp mob: Heck
        # Hound and Sunbird both clear on turn one banking the same 300,
        # so the thrift key decides. At a printed 0 the hound wins and
        # the whole rack goes into a mob a 3-pip card already kills; at
        # its real 7 it loses and four pips stay banked.
        #
        # Measured paired, same seed stream, against the shipped
        # behaviour (X-pip cards skipped outright by `_is_inert`):
        #
        #   live-shaped fire deck, 2 hounds of 18 cards
        #     mob    98.8 -> 100.0   ttk 3.18 -> 3.06
        #     elite  98.5 -> 100.0   ttk 4.51 -> 4.11
        #     boss   94.5 ->  99.0   ttk 7.15 -> 6.14
        #   live-shaped ice deck, no X-pip card
        #     identical on all three cells, as it has to be
        #
        # On a deck BUILT around the card the shipped behaviour was
        # removing the deck's only nuke: storm/Tempest, fire/Heck Hound
        # and balance/Judgement all score 0.0% skipped against
        # 87.5-100% priced.
        price = {c.name: cast_price(sim, s, c) for c, _ in candidates}
        reach = {c.name: cast_reach(sim, s, c) for c, _ in candidates}

        def score_all(allies):
            out = []
            for card, target in candidates:
                turns, neg_damage = _rollout(sim, s, card, max_turns, target,
                                             continuation=fixed_continuation,
                                             allies=allies)
                # Mixed semantics in one tuple is safe here and only here:
                # element 0 is `turns`, and two candidates can only reach
                # element 2 by tying on it -- which puts them on the same
                # side of the horizon and so in the same form.
                thrift = ((-reach[card.name], price[card.name])
                          if is_sentinel(turns, max_turns)
                          else (price[card.name], reach[card.name]))
                out.append(((turns, neg_damage) + thrift + (target,),
                            (card, target)))
            return out

        scored = score_all(None)
        best_score, best_action = min(scored, key=lambda sc: sc[0])
        pass_score = _rollout(sim, s, None, max_turns,
                              continuation=fixed_continuation)

        # When the party makes the move irrelevant, decide as if alone.
        #
        # `_allies_hit` takes a flat rate off the board every round, so on
        # a board the rest of the circle will clear anyway *every*
        # candidate reaches the same turn with the same banked damage --
        # and so does passing. The comparison has then said nothing at
        # all, and the thrift key below decides it: cheapest pips wins,
        # and the cheapest card in any hand is a trap, a blade or a
        # shield. Every seat runs the same reasoning about every other
        # seat, so nobody attacks and the traps pile up. That is the
        # free-rider, and it is what the operator saw as buff spam.
        #
        # It is measurable in the two live exports (rev e523684f): 27 of
        # the 79 in-horizon rounds chose a move that tied passing
        # EXACTLY, and 24 of those 27 played a buff, a trap or a shield.
        # The three that did not had no other candidate. Fight 13 is the
        # shape end to end -- Willie Marks sat on 420 health for five
        # rounds under seven traps while both wizards played 0-pip cards
        # and waited for each other.
        #
        # The re-score drops the *extrapolated* ally damage and keeps the
        # *committed* kind: `Ledger.apply` has already taken the other
        # seats' actual casts off the board this policy was handed, and
        # nothing here puts them back. So this trusts what the party has
        # committed to and distrusts the average it was extrapolated
        # from, which is the right way round -- the rate is a single
        # number standing in for a whole wizard's next twelve rounds.
        #
        # Gated on there being a party at all, so no solo decision, test
        # or tuning sweep can reach it, and it costs nothing when the
        # rate is zero.
        strat.last_party_blind = False
        if _ALLY_RATE > 0 and (best_score[0], best_score[1]) == pass_score:
            scored = score_all(0.0)
            best_score, best_action = min(scored, key=lambda sc: sc[0])
            pass_score = _rollout(sim, s, None, max_turns,
                                  continuation=fixed_continuation, allies=0.0)
            strat.last_party_blind = True

        # Keep the whole comparison, not just its winner. A decision log
        # that records only the chosen card cannot answer the question
        # actually worth asking -- *what else was on the table, and by how
        # much did it lose?* -- and that is the difference between "the
        # policy played a trap" and "the trap and the nuke were half a
        # turn apart". Everything here was already computed; only the
        # discarding was deliberate.
        # Marked chosen only if passing does not beat them, which is
        # decided below. A live export showed round 4 with Frost Beetle
        # @0 AND "pass" both flagged chosen, because this list was
        # written before the pass was scored -- so the decision matrix
        # rendered two winners for one round.
        # `price`, not `c.pips`: what the panel and `rank_candidate` want
        # is what the cast costs, and for an X-pip card the printed 0 is
        # the one number that is certainly wrong. A decision matrix
        # showing "Heck Hound, 0 pips" beside a 1-pip Fire Cat reads as a
        # free card losing to a paid one.
        def mark(scored, passing):
            return [
                Candidate(card=c.name, target=t, turns=score[0],
                          damage=-score[1], pips=price[c.name],
                          chosen=(not passing) and (c, t) == best_action,
                          horizon=max_turns)
                for score, (c, t) in scored]

        strat.last_candidates = mark(scored, False)

        # Passing is a real move -- banking a pip for a bigger hit next
        # turn is exactly the call the heuristic could not make -- but it
        # has to earn it by killing *sooner*, not merely by ending the
        # horizon with more damage on the board. Comparing pass against
        # the damage tiebreak instead made it pass almost every turn: a
        # line that skips a turn accumulates pips and so does more total
        # damage later, which is not a reason to do nothing now.
        pass_turns, pass_damage = pass_score
        passing = pass_turns < best_score[0]
        strat.last_candidates = mark(scored, passing)
        strat.last_candidates.append(
            Candidate(card="pass", target=None, turns=pass_turns,
                      damage=-pass_damage, pips=0, chosen=passing,
                      horizon=max_turns))
        if passing:
            return None

        if best_score[0] > max_turns and best_score[1] == 0.0:
            # Nothing on offer even connects inside the horizon; fall
            # back to the heuristic rather than flailing.
            strat.last_candidates = []      # none of it decided anything
            return school_aware_blade_stack(3)(sim, s)
        return best_action

    #: the last decision's whole candidate set, newest call wins. Read by
    #: `WizAiBackend` straight after the call, the same way `last_source`
    #: is -- a policy cannot reach the telemetry and should not try.
    strat.last_candidates = []
    #: did the last decision fall back to a party-blind comparison? Set
    #: whenever the ally rate made every candidate tie passing, so the
    #: round record can say why the numbers on it are not the ones the
    #: party plan was built from.
    strat.last_party_blind = False
    return strat


def cheapest_lethal(sim, s, target):
    """The smallest thing in hand that finishes `target` this turn.

    "Is it already dead?" has to be asked before "would another blade
    help?", and nothing was asking it. Buff-then-hit is only worth a
    round if the hit still needs the buff -- against a mob the plain nuke
    already kills, every buff round is a free round for the mob, and a
    stack spent on overkill is a stack that is not there for the next
    one.

    Cheapest, not biggest: among cards that all kill, the one costing
    fewest pips leaves the most banked and keeps the big card in hand.
    That is the minimum-damage threshold -- the answer to a 400hp mob
    holding a 300 hit and a 100 hit is both, in that order, not the 500
    hit twice.

    Uses the engine's own cast path (`predict_damage`), so resist,
    shields, prisms and absorbs are all accounted for rather than
    estimated -- a shielded mob is correctly *not* lethal.

    X-pip cards are priced rather than skipped. Skipping them was a
    workaround for reading `card.pips` and `card.damage` off the printed
    face, where an X-pip card is free and tiny; `cast_price` and
    `cast_reach` give the rack-relative numbers, and Tempest or
    Judgement on a full rack is a real finisher that this could not
    previously see. A DoT like Heck Hound still drops out on its own --
    `predict_damage` reports what lands NOW, and a burn lands nothing
    now, which is the right answer to "does this kill it this turn?".
    """
    from w101_sim import cast_price, cast_reach

    from .telemetry import predict_damage

    foe = s.enemies[target] if 0 <= target < len(s.enemies) else None
    if foe is None or not foe.alive:
        return None

    best = None
    seen = set()
    for card in s.hand:
        if card.name in seen or card.kind not in ("damage", "drain"):
            continue
        seen.add(card.name)
        if not sim.can_cast(s, card, target):
            continue
        # Cheap gate before the expensive one: `predict_damage` runs a
        # real cast on a deep copy, and a card that could not reach even
        # tripled is not worth that.
        if cast_reach(sim, s, card) * 3 < foe.hp:
            continue
        got = predict_damage(sim, s, card, target)
        if got is not None and got >= foe.hp:
            rank = (cast_price(sim, s, card), cast_reach(sim, s, card))
            if best is None or rank < best[0]:
                best = (rank, card)
    return best[1] if best else None


def _buff_options(sim, s, school, target=0):
    """The buffs worth casting toward a hit of `school`.

    Mirrors `make_blade_stack.buffs` exactly -- blades and traps compete
    on value, an aura counts when none is up, prisms are the fallback
    when nothing else is available -- with one change: anything that
    cannot act on `school` is filtered out.

    Prisms are deliberately *not* filtered. A prism converts the hit on
    the way in, and charms are consumed against the card's own school
    before the ward pass converts it (`_strike` computes `charm_mult`
    from `spec_school`, then `_ward_pass` changes it). So an ice blade
    still multiplies an ice hit that a prism is about to turn into myth,
    and a prism is never the wrong school for the hit it converts.
    """
    from w101_sim import castable

    options = [c for c in castable(sim, s, "blade") + castable(sim, s, "trap")
               if buff_matches(c, school)]
    if s.player.aura is None:
        options = options + castable(sim, s, "aura")
    return options or castable(sim, s, "prism")


def school_aware_blade_stack(n_buffs=3):
    """Stack up to `n_buffs` buffs that apply to the nuke, then fire it.

    Everything it plays is aimed at one focus enemy (`focus_target`), and
    that is not cosmetic: a trap on one mob followed by a nuke on another
    is two wasted turns, and it is what the un-aimed version produced on
    every board with a minion on it.
    """

    def strat(sim, s):
        from w101_sim import castable, effective_pips

        focus = focus_target(s)

        def aimed(card):
            """The move, carrying its target if the target means anything."""
            if card is None:
                return None
            return (card, focus) if aimed_at_one_enemy(card) else (card, 0)

        # Lethal beats another buff, always. Checked first because a
        # buff round against a mob that is already dead to the plain hit
        # is a round given away -- and stacked three deep it is the whole
        # fight given away.
        finisher = cheapest_lethal(sim, s, focus)
        if finisher is not None:
            return (finisher, focus)

        nuke = choose_nuke(sim, s)
        if nuke is None:
            # No hit in hand: bank the best buff on offer, else pass.
            options = (castable(sim, s, "blade")
                       + castable(sim, s, "trap"))
            return aimed(max(options, key=lambda c: c.percent, default=None))

        school = nuke.school
        if pending_for(s, school, focus) < n_buffs:
            options = _buff_options(sim, s, school, focus)
            if options:
                return aimed(max(options, key=lambda c: c.percent))

        if sim.can_cast(s, nuke, focus) and not nuke.x_pips:
            return aimed(nuke)
        if nuke.x_pips and effective_pips(sim, s, nuke) >= 7:
            return aimed(nuke)

        # Cannot fire the one we were building toward -- keep building.
        options = _buff_options(sim, s, school, focus)
        if options:
            return aimed(max(options, key=lambda c: c.percent))

        # Nothing to build with either. Fire the biggest hit we CAN
        # afford rather than passing: a level-6 wizard holding a 1-pip
        # Frost Beetle and a 2-pip Snow Serpent, with one pip and no
        # blades, was passing the turn away entirely.
        affordable = [c for c in castable(sim, s, "damage")
                      + castable(sim, s, "drain") if not c.x_pips]
        return aimed(max(affordable, key=lambda c: c.damage, default=None))

    return strat
