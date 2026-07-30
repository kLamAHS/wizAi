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


def pending_for(state, school):
    """Buffs already on the board that apply to `school`.

    Counts the same things `make_blade_stack` counts — charms on the
    caster, wards on the primary enemy, the aura — but only those that
    can actually act on this school, which it does not check.

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
    enemy = state.enemies[0] if state.enemies else None
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

    def __init__(self, agent, fallback=None):
        self.agent = agent
        self.fallback = fallback or greedy_ttk()
        self.seen = 0
        self.missed = 0
        #: which path the LAST decision took. Surfaced per round, because
        #: "is the model I picked actually driving?" is otherwise
        #: unanswerable from the outside -- a trained policy and its
        #: fallback look identical in a decision log.
        self.last_source = ""

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
            return self.fallback(sim, s)

        # `.get`, not `[]`: QAgent.Q is a defaultdict, and indexing it
        # here would insert a zero for every state we merely asked about.
        known = any(self.agent.Q.get((key, a), 0.0) for a in legal)
        if not known:
            self.missed += 1
            self.last_source = "fallback (state not in Q table)"
            return self.fallback(sim, s)

        self.seen += 1
        self.last_source = "Q table"
        return self.agent.policy()(sim, s)


def trained_policy(agent, fallback=None):
    """`policy(sim, state)` for a QAgent, with a fallback for unseen states."""
    return TrainedPolicy(agent, fallback)


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


_CONTINUATION = None


def _continuation():
    """The policy a rollout plays after its first move. Built once."""
    global _CONTINUATION
    if _CONTINUATION is None:
        _CONTINUATION = school_aware_blade_stack(3)
    return _CONTINUATION


def _rollout(sim, state, first_action, max_turns=12):
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
    """
    import copy

    continuation = _continuation()
    probe = copy.copy(sim)
    probe.rng = _Fixed()
    s = copy.deepcopy(state)

    def enemy_alive():
        return any(e.alive for e in s.enemies)

    dealt = 0.0
    lost = (max_turns + 1, 0.0)

    # find the copied card matching the chosen one
    action = None
    if first_action is not None:
        for c in s.hand:
            if c.name == first_action.name and probe.can_cast(s, c):
                action = c
                break
        if action is None:
            return lost

    for turn in range(1, max_turns + 1):
        if action is not None:
            try:
                dealt += probe.cast(s, action) or 0.0
            except Exception:
                return lost
        if not enemy_alive():
            return turn, -dealt
        try:
            probe.end_round(s)
        except Exception:
            return lost
        if not enemy_alive():
            return turn, -dealt
        if not s.player.alive:
            return lost

        action = continuation(probe, s)

    return max_turns + 1, -dealt


def greedy_ttk(max_turns: int = 12):
    """Pick the move that kills soonest, by simulating each candidate.

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

    Slower than the heuristics -- roughly (hand size + 1) rollouts per
    decision -- which is nothing next to a live planning phase.
    """

    def strat(sim, s):
        from w101_sim import castable

        candidates = []
        seen = set()
        for card in s.hand:
            if card.name in seen or not sim.can_cast(s, card):
                continue
            seen.add(card.name)
            candidates.append(card)

        if not candidates:
            return None

        # Third key: front-load. Among lines that kill on the same turn
        # having banked the same damage, take the bigger hit now --
        # damage already dealt cannot be undone by a heal, a shield or a
        # fizzle on a later turn.
        scored = []
        for card in candidates:
            turns, neg_damage = _rollout(sim, s, card, max_turns)
            scored.append(((turns, neg_damage, -card.damage), card))
        best_score, best_action = min(scored, key=lambda sc: sc[0])

        # Passing is a real move -- banking a pip for a bigger hit next
        # turn is exactly the call the heuristic could not make -- but it
        # has to earn it by killing *sooner*, not merely by ending the
        # horizon with more damage on the board. Comparing pass against
        # the damage tiebreak instead made it pass almost every turn: a
        # line that skips a turn accumulates pips and so does more total
        # damage later, which is not a reason to do nothing now.
        pass_turns, _ = _rollout(sim, s, None, max_turns)
        if pass_turns < best_score[0]:
            return None

        if best_score[0] > max_turns and best_score[1] == 0.0:
            # Nothing on offer even connects inside the horizon; fall
            # back to the heuristic rather than flailing.
            return school_aware_blade_stack(3)(sim, s)
        return best_action

    return strat


def _buff_options(sim, s, school):
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
    """Stack up to `n_buffs` buffs that apply to the nuke, then fire it."""

    def strat(sim, s):
        from w101_sim import castable, effective_pips

        nuke = choose_nuke(sim, s)
        if nuke is None:
            # No hit in hand: bank the best buff on offer, else pass.
            options = castable(sim, s, "blade") + castable(sim, s, "trap")
            return max(options, key=lambda c: c.percent, default=None)

        school = nuke.school
        if pending_for(s, school) < n_buffs:
            options = _buff_options(sim, s, school)
            if options:
                return max(options, key=lambda c: c.percent)

        if sim.afford(s, nuke) and not nuke.x_pips:
            return nuke
        if nuke.x_pips and effective_pips(sim, s, nuke) >= 7:
            return nuke

        # Cannot fire the one we were building toward -- keep building.
        options = _buff_options(sim, s, school)
        if options:
            return max(options, key=lambda c: c.percent)

        # Nothing to build with either. Fire the biggest hit we CAN
        # afford rather than passing: a level-6 wizard holding a 1-pip
        # Frost Beetle and a 2-pip Snow Serpent, with one pip and no
        # blades, was passing the turn away entirely.
        affordable = [c for c in castable(sim, s, "damage")
                      + castable(sim, s, "drain") if not c.x_pips]
        return max(affordable, key=lambda c: c.damage, default=None)

    return strat
