"""A wizsprinter combat backend that asks a wizAi policy what to cast.

This is the piece that puts the learned policy in front of real enemies.
`SprintyCombat` drives a real duel and, each planning phase, asks its
backend for a `PriorityLine`; the backend here answers by reading the
live board into a wizAi `State`, handing that to a wizAi policy, and
translating the policy's chosen card back into a move the handler can
click.

    from deimos_bridge.live_backend import WizAiBackend
    from wizwalker.extensions.wizsprinter import SprintyCombat

    backend = WizAiBackend.from_trained(school="fire", deck=my_decklist)
    await SprintyCombat(client, backend).handle_combat()

Policies are just callables, which is the whole reason this is short:
wizAi's in-fight contract is
`policy(sim, state) -> Card | (Card, target) | str | None`
(`w101_sim.py:1994`, `:2128`), and its action strings are already card
names -- optionally `"name@i"` for a target index (`rl_agent.py:97-107`).
That maps onto `NamedSpell(name)` plus a `TargetData` almost directly.

The wizwalker/wizsprinter imports are deferred into the methods that need
them, so this module imports cleanly on Linux and can be unit-tested
against `mock_client`.
"""
import random

from .live_state import NameResolver, WIKI_TO_GAME, read_state
#: Where a card wants to be clicked. Lives in `policies` because it is
#: card reasoning that both sides need -- the policies use it to decide
#: whether choosing an enemy means anything at all, this module uses it
#: to decide what the mouse clicks after the card.
from .policies import TARGET_OPS as _TARGET_OPS       # noqa: F401 re-export
from .policies import primary_target as _primary_target


#: what a card's `kind` implies about where it is clicked, for cards
#: whose ops carry no `tgt` at all.
_KIND_TARGETS = {"aura": "global", "global": "global", "blade": "self",
                 "shield": "self", "heal": "self", "util": "global",
                 "damage": "enemy", "drain": "enemy"}


def _target_from_kind(card):
    """Where to click a card whose ops do not say. None if even the kind
    does not say.

    349 of the 8,148 cards in the table have no `tgt` on any op --
    almost all of them auras and globals, including plain Amplify. They
    used to fall through to the enemy branch and be clicked on a mob,
    and Wizard101 does not cast a self-buff at an enemy: the second
    click deselects the card, the round passes with nothing played, and
    the duel sits there until the round timer runs out. That is the
    stall, and from outside it looks exactly like the bot deciding to
    do nothing.

    `kind` is the wizAi card table's own classification and it is
    populated for every one of the 349, so it answers what the ops
    forgot to.
    """
    return _KIND_TARGETS.get(getattr(card, "kind", None))


def _missing_health(actor) -> float:
    """How much health this wizard is down. 0 when it will not read."""
    try:
        return max(float(actor.max_hp) - float(actor.hp), 0.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0


class PolicyDecision:
    """What the policy decided, before it becomes a wizsprinter move.

    Kept as its own type so the decision can be tested, logged, and
    replayed without any wizsprinter object in sight.
    """

    def __init__(self, card_name=None, target_index=None, passing=False,
                 reason="", policy="", target_kind=None):
        self.card_name = card_name
        self.target_index = target_index
        self.passing = passing
        self.reason = reason
        #: what the card is aimed at -- 'enemy', 'self', 'enemies', ... .
        #: Kept so a log can say "on myself" rather than naming a mob the
        #: cast never touched; `target_index` only means something when
        #: this is 'enemy'.
        self.target_kind = target_kind
        #: every move the policy weighed this round, with its score. The
        #: chosen card alone cannot say whether the decision was close.
        self.candidates = []
        #: which policy played this round, and by which path. Recorded
        #: per decision rather than per run because the policy can be
        #: swapped mid-run, and because a trained policy falling through
        #: to its fallback is indistinguishable from the fallback itself
        #: unless it is written down at the moment it happens.
        self.policy = policy

    def __repr__(self):
        if self.passing:
            return f"PolicyDecision(pass, {self.reason!r})"
        t = "" if self.target_index is None else f"@{self.target_index}"
        return f"PolicyDecision({self.card_name!r}{t}, {self.reason!r})"


class WizAiBackend:
    """The decision loop: live board -> wizAi policy -> a chosen card.

    Standalone by design -- `WizAiCombatHandler` drives it, and
    `make_sprinty_backend()` wraps it into a real `BaseCombatBackend`
    when wizsprinter is present. Neither the base class nor wizwalker is
    imported here, so `mock_client` can exercise this on any machine.
    """

    def __init__(self, policy, cards, school, decklist=None, cast_time=0.3,
                 on_decision=None, rules=None, catalog=None, policy_name="",
                 player_stats=None, on_lost_round=None, seat=0,
                 coordinator=None, party_size=1):
        """
        Args:
            policy:   `policy(sim, state) -> Card | str | None`, wizAi's
                      in-fight contract.
            cards:    the wizAi card table (`data_full.load_spells_full()`).
            school:   the wizard's school, lower case.
            decklist: card names in the deck, used only for the scarcity
                      feature -- the client cannot report undrawn cards.
            on_decision: optional callback(PolicyDecision, LiveRead) for
                      logging a live run.
            on_lost_round: optional callback(round number, reason) for a
                      round whose board could not be read at all. There
                      is no `LiveRead` to hand over in that case, which
                      is precisely why it needs its own channel rather
                      than being dropped.
            policy_name: what to call `policy` on screen. Purely a label;
                      the decision loop never reads it.
            player_stats: the wizard's gear, as `Sim(player_stats=...)`
                      takes it. See `live_state.read_player_stats`;
                      leaving it empty models a wizard wearing nothing.
            seat:     which wizard of the party this backend drives.
                      Only an identity -- the coordinator keys on it.
            coordinator: a `hivemind.Hivemind`, or None to decide alone.
                      With one installed the policy is still the same
                      callable and still gets `(sim, state)`; what
                      changes is that the state it is handed already
                      carries what the rest of the party committed to
                      this round. See `deimos_bridge/hivemind.py`.
            party_size: how many wizards are in the circle. Used to
                      apportion a casting boss's modelled output, which
                      a party splits between them.
        """
        #: The policy and its display name, held as one tuple so that
        #: swapping is a single atomic rebind. Two attributes would let a
        #: decision read the new name against the old callable and log a
        #: round under a policy that did not choose the card on it.
        #: `.policy` and `.policy_name` are properties over this, so
        #: assigning either still works.
        self._policy = (policy, policy_name)
        self.cards = cards
        self.school = school
        self.decklist = list(decklist or [])
        #: settable after construction -- the stats are read once the
        #: hooks are up, which is after the backend exists
        self.player_stats = dict(player_stats or {})
        #: seconds wizwalker pauses after each click while casting --
        #: twice per cast. Consumed by the combat handler; raise it if
        #: the status bar starts reporting casts that did not go through.
        self.cast_time = cast_time
        self.combat = None
        self.on_decision = on_decision
        self.on_lost_round = on_lost_round
        self.on_failed_cast = None
        #: optional callback(message) when a cast had to be retried more
        #: slowly. NOT `on_failed_cast` -- nothing has failed yet, and
        #: that one rewrites the round. See `report_slow_cast`.
        self.on_slow_cast = None
        #: optional callback(card, target, first choice) when the chosen
        #: card would not go out and the runner-up was played instead.
        #: NOT `on_failed_cast` -- that one has already run and recorded
        #: the round as a pass, which this has to undo.
        self.on_recovered_cast = None
        #: optional callback(actual school) for a wizard whose client
        #: disagrees with the school it was configured with. See
        #: `_check_school`.
        self.on_school_mismatch = None
        self._school_checked = False
        #: optional callback() the first time this wizard is found at 0
        #: health in a duel it is still nominally in. See `_check_defeated`.
        self.on_defeated = None
        #: optional callback() -> enemy health this wizard removes per
        #: round, for the coordinator to hand the OTHER seats' rollouts.
        #: A hook rather than a `Telemetry` reference because the backend
        #: does not own one -- the GUI does, and it wires this to
        #: `Telemetry.damage_rate`. Absent means "nothing measured",
        #: which is what a headless backend and a solo wizard both want.
        self.damage_rate = None
        #: whether the last read found it down, so the message is said
        #: once per knockdown rather than once per round
        self._down = False
        #: the wizard's power-pip odds, off the client. None means never
        #: read, and then the actor keeps whatever default it has rather
        #: than being given a made-up number.
        self.power_pip_chance = None
        self.rules = rules
        #: which wizard of the party this is. An identity and nothing
        #: else -- the coordinator keys its plan on it.
        self.seat = int(seat or 0)
        #: who agrees the round with the other seats. `None` is a wizard
        #: deciding alone, which is bit-for-bit the path that shipped
        #: before parties existed.
        self.coordinator = coordinator
        #: wizards in the circle, for splitting a boss's output
        self.party_size = max(1, int(party_size or 1))
        self.resolver = NameResolver(cards, catalog)
        self.history = []          # PolicyDecision, in order
        self._seen = []            # card names observed in hand so far
        #: the `LiveRead` behind the most recent decision. The handler
        #: casts from this, so the card it clicks and the target it picks
        #: come from the same snapshot the policy saw.
        self.last_read = None
        #: running measurement of how hard this board hits, per living
        #: enemy per round. See `_estimate_incoming`.
        self._incoming = []
        self._hp_seen = None
        self._round_seen = None
        self._enemies_seen = 1
        #: rollouts model named enemies as CASTERS (catalog spell pool +
        #: the pips read off the client) instead of a flat measured hit.
        #: See `_apply_pool` for the measurement behind the default.
        self.use_pool_model = True

    # -- the policy, swappable mid-fight ----------------------------------
    @property
    def policy(self):
        return self._policy[0]

    @policy.setter
    def policy(self, value):
        self._policy = (value, self._policy[1])

    @property
    def policy_name(self):
        return self._policy[1]

    @policy_name.setter
    def policy_name(self, value):
        self._policy = (self._policy[0], value)

    def set_policy(self, policy, name=""):
        """Swap the policy and its label together, mid-fight if need be.

        One rebind. Setting the two attributes in sequence would leave a
        window where a decision in flight reads the new name and the old
        callable, and mislabels the round -- which defeats the point of
        recording the name at all.
        """
        self._policy = (policy, name)

    # -- BaseCombatBackend ------------------------------------------------
    def attach_combat(self, combat):
        self.combat = combat

    async def handle_no_cards_given(self):
        pass

    async def get_relative_round(self, r):
        return await self.get_real_round(r)

    async def get_real_round(self, r):
        """The hook `SprintyCombat` calls once per planning phase."""
        decision = await self.decide()
        return self._to_priority_line(decision)

    # -- the actual decision ----------------------------------------------
    async def decide(self):
        """Read the board, ask the policy, return a `PolicyDecision`.

        Separated from `get_real_round` so it can be driven by a mock and
        asserted on without constructing any wizsprinter types.
        """
        read = await read_state(self.combat, self.resolver, self.school,
                                deck_remaining=self._deck_remaining(read_hand=None))
        for name in read.hand_cards:
            if name not in self._seen:
                self._seen.append(name)
        read.state.player.deck = self._deck_remaining(read.hand_cards)
        self._check_school(read)
        self._measured_incoming = self._estimate_incoming(read)
        self._apply_player_stats(read)
        self._apply_bestiary(read)
        self._apply_pool(read)
        self._apportion_incoming(read)
        self.last_read = read

        defeated = self._check_defeated(read)
        if defeated is not None:
            return defeated

        sim = self._sim_for(read)
        # Read once, both halves together. `_policy` is rebindable from
        # the GUI thread, and a round that asked one policy for a card
        # must be logged against that policy, not against whatever
        # replaced it midway.
        policy, label = self._policy
        try:
            if self.coordinator is not None:
                # The party decides the round together. The policy is
                # still this callable and still sees `(sim, state)` --
                # what the coordinator changes is the board it is shown,
                # which already carries the other wizards' commitments.
                # It also blocks until they have all arrived, which is
                # why `decide` is async and this branch is awaited.
                rate = 0.0
                if self.damage_rate is not None:
                    try:
                        rate = float(self.damage_rate() or 0.0)
                    except Exception:
                        rate = 0.0   # a broken hook must not cost a round
                choice = await self.coordinator.decide(
                    self.seat, sim, read.state, policy, read, rate=rate)
            else:
                choice = policy(sim, read.state)
        except Exception as exc:                      # a policy must never
            decision = PolicyDecision(                # break a live fight
                passing=True, policy=label or "policy",
                reason=f"policy raised {type(exc).__name__}: {exc}")
            self._record(decision, read)
            return decision

        decision = self._interpret(choice, read)
        # After the call, deliberately: a wrapped policy records which
        # path it took while deciding, and a lookahead policy records
        # what it weighed.
        decision.policy = self._why(policy, label)
        decision.candidates = list(getattr(policy, "last_candidates", ()) or ())
        self._note_party(decision)
        # After `_note_party`, which rewrites the reason outright when the
        # seat was held. The numbers in `candidates` are the party-blind
        # ones whenever this fired, and a reader comparing them against
        # the plan's own arithmetic deserves to be told which comparison
        # they are looking at.
        if getattr(policy, "last_party_blind", False):
            decision.reason += (" (every move tied doing nothing once the "
                                "party's damage was counted, so this was "
                                "decided as if alone)")
        self._record(decision, read)
        return decision

    def _check_defeated(self, read):
        """A `PolicyDecision` for a wizard at 0 health, or None.

        A defeated wizard is still in the duel as far as wizwalker is
        concerned: `handle_round` keeps firing, the read keeps coming
        back, and the board keeps advancing. What it has is no health,
        no pips and no hand, so every round from the moment it goes down
        is the same round -- and the first live party run recorded four
        consecutive ones for wizard 1, each "policy chose to pass"
        against an empty hand, sitting in the export beside the rounds
        it actually played.

        Two things go wrong beyond the noise, and both matter more:

          * **The barrier waits for it.** The party gathers every seat
            in combat before anyone casts. A corpse submits like anyone
            else, so the wizards still standing pay the full round-trip
            to be told it is passing -- every round, for the rest of the
            fight.
          * **Its board joins the ledger.** A dead wizard's read is a
            real read of the enemy side, and it is folded into the
            shared enemy ledger like a live one. Nothing it contributes
            is wrong, but nothing it contributes is a *cast* either, and
            the coordinate descent spends a pass on a seat that cannot
            act.

        So it leaves the circle -- `leave_combat`, not `leave`: it is
        still a member of this party and will be back as soon as the
        fight ends -- and the round is passed without being recorded.
        """
        hp = getattr(read.state, "player_hp", None)
        if hp is None:
            hp = getattr(getattr(read.state, "player", None), "hp", None)
        if hp is None or hp > 0:
            self._down = False
            return None

        hive = self.coordinator
        if hive is not None:
            hive.leave_combat(self.seat)
        first, self._down = not self._down, True
        if first and self.on_defeated:
            try:
                self.on_defeated()
            except Exception:
                pass
        # Not recorded. `_record` is what fills the export and drives
        # every panel, and a round with no hand, no pips and no choice
        # says nothing about the policy that was not already said by the
        # round it died in.
        return PolicyDecision(passing=True, policy=self._policy[1] or "policy",
                              reason="defeated — waiting for the fight to end")

    def _note_party(self, decision):
        """Record when the party is what changed this wizard's mind.

        Without it a coordinated run is unreadable: the decision row
        says "ttk-lookahead played Snow Serpent on the minion" whether
        that was this wizard's own call or the consequence of another
        wizard having already claimed the boss, and those are different
        events. The whole reason to run four instances is the second
        one, so it has to be visible.
        """
        hive = self.coordinator
        if hive is None:
            return
        # The label goes on whether or not the plan is still readable: a
        # round decided through the coordinator was coordinated, and a
        # log that only says so when the outcome happened to change is a
        # log that cannot answer "was the party on?".
        decision.policy += " · party"
        try:
            move = hive.last_move(self.seat)
        except Exception:
            return
        if move is None:
            return
        if move.note == "held":
            decision.reason = ("held this card — the rest of the party "
                               "already has this board dead this round")
        elif move.retargeted:
            was = move.solo_card or "pass"
            if move.solo_target is not None:
                was += f" @{move.solo_target}"
            decision.reason += (f" (alone it would have played {was}; "
                                f"the party's plan changed it)")

    def _interpret(self, choice, read):
        if choice is None:
            return PolicyDecision(passing=True, reason="policy chose to pass")

        # `(card, target)` is wizAi's own aimed-move form --
        # `Sim._normalize_action` unpacks exactly this tuple, so a policy
        # that aims works unchanged in the simulator, in `evaluate`, and
        # here. It is how a trap and the nuke it is buying agree on one
        # mob; without it every cast went to whichever enemy the
        # participant list happened to put first.
        aimed = None
        if isinstance(choice, tuple):
            choice, aimed = (choice + (None,))[:2]
            if choice is None:
                return PolicyDecision(passing=True,
                                      reason="policy chose to pass")

        card = choice if hasattr(choice, "ops") else None
        name = getattr(choice, "name", choice)
        if not isinstance(name, str):
            return PolicyDecision(passing=True,
                                  reason=f"policy returned {choice!r}")

        # Two different things put an "@" in a card name and they collide:
        #   rl_agent encodes a target as "Fireblade@1"   (index)
        #   data_full encodes provenance as "Imp@item"   (source tier)
        # Splitting on "@" unconditionally turns "Thunder Snake - Starter
        # Wand@item" into "Thunder Snake - Starter Wand", which is not a
        # card, so every wand and treasure card became unplayable.
        #
        # Only a trailing *integer* is a target index. rpartition takes the
        # last "@", so a targeted item card ("Imp@item@0") still splits
        # correctly into ("Imp@item", 0).
        target_index = aimed
        if "@" in name:
            head, _, idx = name.rpartition("@")
            if idx.isdigit():
                name, target_index = head, int(idx)

        if name in ("__pass__", "pass"):
            return PolicyDecision(passing=True, reason="policy chose to pass")

        if name not in read.hand_cards:
            return PolicyDecision(
                passing=True,
                reason=f"{name!r} is not a castable card in hand "
                       f"({sorted(read.hand_cards)})")

        # A target index on a self-buff or an AoE is meaningless, and
        # carrying one would put a fabricated enemy name in the log --
        # the decision row would claim the blade went "on Bastilla".
        card = card if card is not None else self.cards.get(name)
        kind = _primary_target(card) if card is not None else "enemy"
        if kind != "enemy":
            target_index = None
        elif target_index is not None:
            n = len(read.state.enemies)
            if not (0 <= target_index < n):
                target_index = None

        return PolicyDecision(card_name=name, target_index=target_index,
                              target_kind=kind, reason="policy choice")

    def _why(self, policy, label=""):
        """Which policy just decided, and by which path.

        The path half only exists for wrappers that keep one --
        `TrainedPolicy` sets `last_source` on every call. Without it a
        learned policy and the heuristic it silently fell back to produce
        identical decision logs, so "is the model I selected actually
        driving?" has no answer.
        """
        label = label or self.policy_name or "policy"
        source = getattr(policy, "last_source", "")
        return f"{label} — {source}" if source else label

    #: per-enemy incoming damage assumed before there is a round to
    #: measure. Matches the trainer's own dummy-boss rule
    #: (`gui/app.TrainWorker`), so a live fight and the fight the Q table
    #: was learned on are not priced differently on turn one.
    #:
    #: Divided by the party, because the quantity is "damage to THIS
    #: wizard", and an enemy choosing one of four wizards lands on this
    #: one a quarter of the time. Measured on the first live party run:
    #: the solo wizard's prior read 57/enemy against 78-84 actually
    #: measured (right, and slightly low), while the two party wizards'
    #: priors read 85 and 58 against 15-30 and 36-53 -- over by up to
    #: five times. Round one is exactly where that hurts most, because
    #: it is the only round with nothing measured yet: wizard 2's
    #: opening had EVERY candidate scored at the horizon sentinel, all
    #: 14.0 turns and all 235 damage, because a board dealing an
    #: imagined 115/round killed it inside the horizon on every line.
    #: The comparison collapses and the opening move falls through to
    #: the pip tiebreak.
    INCOMING_PRIOR_DIVISOR = 12.0

    #: how many rounds of observation the prior is worth. Three, so a
    #: single quiet round cannot collapse the threat model and three
    #: real ones already outweigh it. Zero reproduces the old behaviour
    #: -- the prior as a bare fallback -- and is what the exports were
    #: recorded under.
    INCOMING_PRIOR_WEIGHT = 3.0

    def _estimate_incoming(self, read):
        """How hard this board actually hits, measured round to round.

        Enemies built by `read_state` carry no `flat_hit`, so `_enemy_turn`
        did nothing and every rollout modelled the mobs dealing **zero**
        damage. The consequences are not subtle: setting up costs nothing
        but turns, `_rollout`'s `player.alive` check can never fire, and
        the policy has no reason to prefer killing something now over
        stacking one more buff. That is the exact shape of "it spends the
        fight setting up and then dies to the minion".

        Nothing in the client reports a mob's spell damage, so it is
        measured instead: the player's health drop across a round, split
        over the mobs that were alive to cause it. That folds in DoTs and
        minion hits, which is right -- what the policy needs to know is
        how long it can afford to take, not which mob is responsible.
        """
        player = read.state.player
        living = [e for e in read.state.enemies if e.alive]
        if self._hp_seen is not None and self._round_seen != read.round_number:
            lost = self._hp_seen - player.hp
            # Zero counts. A round where every mob fizzled or passed is
            # a real round of the damage distribution, and dropping it
            # biased the mean high -- measured on a live fight, the
            # per-enemy estimate read 117/round on a board dealing ~78,
            # because the two rounds that hurt were averaged and the
            # round that didn't was thrown away. Negative rounds (the
            # wizard healed more than was dealt) still stay out: that
            # is the heal's number, not the board's.
            if lost >= 0:
                self._incoming.append(lost / max(1, self._enemies_seen))
        self._hp_seen = player.hp
        self._round_seen = read.round_number
        self._enemies_seen = max(1, len(living))

        prior = max(30.0, player.max_hp / (self.INCOMING_PRIOR_DIVISOR
                                           * self.party_size))
        # The prior is WEIGHED IN, not merely a fallback for round one.
        # As a fallback the first real observation replaced it outright,
        # and a first observation of zero therefore set the whole threat
        # model to exactly zero -- which is what happened in the first
        # live party run and is visible in both exports. Jeffrey's
        # `incoming` reads 52.92 at fight 1 round 2 (the prior itself,
        # 1270/(12*2)), then 0.0 at rounds 3 and 4, because the rounds
        # between those reads happened to be quiet. Konstantin's does
        # the same at fight 1 round 2.
        #
        # Zero incoming is not a small error, it is a different game:
        # `_enemy_turn` deals nothing, `_rollout`'s `player.alive` check
        # can never fire, `died()` becomes unreachable, and mitigation is
        # worth literally nothing -- so setting up costs only turns and a
        # shield scores exactly like passing. The two rounds Jeffrey
        # spent on Tower Shield at full health were both planned against
        # a board the model believed was harmless, in the fight that
        # then killed him.
        #
        # A pseudo-count fixes it without slowing the estimate down
        # much: three rounds of real observation already outweigh the
        # prior, and with nothing measured this is still exactly the
        # prior, so round one is unchanged.
        weight = self.INCOMING_PRIOR_WEIGHT
        per_enemy = ((sum(self._incoming) + prior * weight)
                     / (len(self._incoming) + weight))
        for enemy in read.state.enemies:
            enemy.flat_hit = per_enemy
        return per_enemy

    def _check_school(self, read):
        """Say so when this client is not the wizard it was configured as.

        `ClientHandler.get_new_clients()` returns windows in whatever
        order it finds them, and nothing else can tell which one is
        "wizard 1" -- so a party's seats and clients can be crossed, and
        the first live party run's were: an ice wizard configured fire
        and a fire wizard configured ice.

        What it costs depends on the wizard, and it is worth being exact
        because the obvious answer is wrong. The gear read asks for the
        configured school's damage stat, so `player.damage_bonus` comes
        back keyed on a school this wizard never casts and
        `_damage_bonus` misses on every card -- but a low-level wizard
        has no damage or accuracy stat to lose, so on that run the
        crossing cost nothing there. It bites where the school itself is
        the input: `choose_nuke` prefers the caster's own school and
        finds none of it, `effective_pips` values a power pip at two in
        the wizard's school and one elsewhere, `TrainWorker.board_schools`
        excludes the wizard's own school from an evaluation board and
        excludes the wrong one, and the exported run names a school its
        own hand contradicts. It gets expensive the moment the wizard has
        gear worth reading.

        Checked once, from the read the fight already does -- the client
        member is a `CombatParticipant` and knows its own school. Only
        reported here; correcting it needs the gear re-read, which is the
        caller's business.
        """
        if self._school_checked:
            return
        actual = getattr(read, "client_school", "") or ""
        if not actual:
            actual = self._school_from_hand(read)
        if not actual:
            return          # no evidence is not a mismatch
        self._school_checked = True
        if actual == self.school:
            return
        if self.on_school_mismatch:
            self.on_school_mismatch(actual)

    #: how much of a hand's damage has to agree before the hand is taken
    #: as evidence of the wizard's school. Not a majority: starter wands
    #: hand out Thunder Snake, Imp, Scarab and Dark Sprite regardless of
    #: school, and pets add trained cards of their own, so a real hand is
    #: never pure. Four of five is well past what that noise produces and
    #: well short of demanding purity.
    HAND_SCHOOL_SHARE = 0.8
    #: and enough cards to mean anything. A two-card opening hand that
    #: happens to be one school is not evidence of anything.
    HAND_SCHOOL_MIN = 3

    def _school_from_hand(self, read):
        """The school this wizard looks like, off what it is holding.

        The fallback for when `primary_magic_school_id` will not read off
        the client's own combat member -- which is what happened on the
        first two live party runs, where it reported every enemy's school
        correctly and gave nothing for the wizard. The hand is not a
        guess in that situation: a wizard holding Frost Beetle, Ice Trap,
        Snow Serpent and Evil Snowman under a `fire` configuration is an
        ice wizard, and reading that off the cards is exactly how the
        crossing was spotted by eye in the first place.

        Damage cards only. Blades and traps carry a school too, but a
        Balanceblade is a balance card in an ice deck and would vote for
        a school nobody is playing.
        """
        hits = [c for c in read.state.hand
                if getattr(c, "kind", "") in ("damage", "drain")
                and getattr(c, "school", "")]
        if len(hits) < self.HAND_SCHOOL_MIN:
            return ""
        counts = {}
        for card in hits:
            counts[card.school] = counts.get(card.school, 0) + 1
        school, n = max(counts.items(), key=lambda kv: kv[1])
        return school if n / len(hits) >= self.HAND_SCHOOL_SHARE else ""

    def _apply_player_stats(self, read):
        """Put the wizard's real gear and power pips on the read player.

        `read_state._mk_actor` builds a bare `Actor`, and every stat the
        simulator prices a hit with lives on the actor: `damage_bonus`,
        `accuracy_bonus`, `power_pip_chance`. `Sim._build_player` is the
        only thing that applies `player_stats`, and the live path never
        calls it -- so `_sim_for` carefully carried the gear to a `Sim`
        that then handed the policy a naked wizard.

        It decides real moves. On a 258 HP mob with a 40% Ice Trap up,
        Snow Serpent's midpoint is 175: 175 x 1.4 = 245, which does not
        kill, so the lookahead scores the trap line at three turns and
        weighs a different opening. With 9% ice damage the same line is
        175 x 1.4 x 1.09 = 267, which does, and it is two turns. The
        gear was the difference between "this kills it" and "this does
        not", and the rollout never saw the gear.

        Power pips matter for the same reason in the other direction:
        at `power_pip_chance = 0` a two-pip Snow Serpent is a turn-two
        card and a three-pip Snowman a turn-three card, so every line
        that opens with a buff is scored as though the payoff were a
        turn later than it is.
        """
        player = read.state.player
        stats = self.player_stats or {}
        damage = stats.get("damage") or {}
        if damage:
            player.damage_bonus = dict(damage)
        if stats.get("accuracy"):
            player.accuracy_bonus = stats["accuracy"]
        if stats.get("pierce"):
            player.pierce = stats["pierce"]
        if stats.get("crit"):
            player.crit_chance = stats["crit"]
        if stats.get("resist"):
            player.resist = dict(stats["resist"])
        if self.power_pip_chance is not None:
            player.power_pip_chance = self.power_pip_chance

    def _apply_bestiary(self, read):
        """Exact per-boss facts, off the scraped catalog.

        The read infers a mob's school and nothing else about its
        defences; the catalog KNOWS them for named creatures -- "50% to
        Death, +20% to Life" is the difference between a death wizard's
        hit landing at half and a life wizard's landing at 1.2x, and the
        sim's `_resist_mult` consumes exactly these two dicts. Observed
        facts are never overwritten: only empty fields are filled, so a
        live-read shield or boost stays authoritative.

        SCHOOL is the exception, and the catalog wins it outright for
        an exact name match: `read_school`'s failure mode is a silent
        "balance" guess, and that guess poisons everything downstream
        -- a fire wizard's fight against the fire boss Alicane
        Swiftarrow was read as "480 balance + 235 balance", so training
        boards, the envelope and the trained table all priced fire
        damage landing at full when the real fight halves it. A
        catalog's school for a named creature is scraped fact; a read
        school disagreeing with it is far more likely the fallback
        than a genuine tier variant of a different school.
        """
        try:
            from .bestiary import lookup, stat_overrides
        except Exception:
            return
        for enemy in read.state.enemies:
            try:
                found = stat_overrides(enemy.name, enemy.max_hp)
                rec = lookup(enemy.name, enemy.max_hp)
            except Exception:
                continue
            if rec and rec.get("school") and \
                    enemy.school != rec["school"]:
                enemy.school = rec["school"]
            if not found:
                continue
            resist, boost, _stunable = found
            if resist and not enemy.resist:
                enemy.resist = dict(resist)
            if boost and not enemy.boost:
                enemy.boost = dict(boost)

    def _apply_pool(self, read):
        """Named enemies become CASTERS in the rollouts.

        The flat measured hit gets the average right and the SHAPE
        wrong: a boss saving six pips for a Wraith deals nothing for
        three rounds and then a third of the wizard's health at once,
        and shield-or-race is decided exactly by that shape. The client
        reports every member's pip rack (`read_state` now keeps it) and
        the catalog knows what 1,795 creatures cast, so the sim's
        living-boss layer -- `_enemy_turn`'s pool branch, pip economy
        and all -- can price "the Wraith is legal RIGHT NOW" instead of
        a constant drizzle.

        Measured (belief-vs-truth A/B, greedy_ttk vs casting catalog
        bosses, N=250 paired seeds, contested boards only): the pool
        belief wins +8 to +10 points of kill rate on hitter pools
        (storm Ketil band +10.0/+2.8/+8.0/+9.2, fire Usunoki +8.4) and
        is a wash inside the ~2.4-point noise floor on debuffer/DoT
        pools (myth -0.8, death -2.0). In-sim the belief's pool is
        exact, so those are upper bounds of the live gain -- but no
        board showed a significant loss, and the measured flat hit
        stays on every enemy as the fallback the sim uses the moment a
        pool is absent.

        A pool the card table cannot resolve to at least one damage
        spell is NOT stamped: `_enemy_choose` would find nothing to
        cast and the boss would deal zero for the whole rollout, which
        is strictly worse than the measured flat model.
        """
        if not self.use_pool_model:
            return
        try:
            from .bestiary import full_boss
        except Exception:
            return
        for enemy in read.state.enemies:
            if getattr(enemy, "spell_pool", None) is not None:
                continue
            try:
                b = full_boss(enemy.name, enemy.max_hp)
            except Exception:
                continue
            if b is None or not b.pool:
                continue
            known = [n for n in b.pool if n in self.cards]
            if not any(self.cards[n].kind in ("damage", "drain")
                       for n in known):
                continue
            enemy.spell_pool = list(b.pool)
            enemy.archetype = b.archetype
            enemy.discipline = b.discipline
            enemy.power_pip_chance = b.pip_chance
            # gear-like scraped extras, never over an observed fact
            if b.outgoing_bonus and not enemy.damage_bonus:
                enemy.damage_bonus = {"*": b.outgoing_bonus}
            if b.pierce and not enemy.pierce:
                enemy.pierce = b.pierce

    def _modeled_pool_output(self, enemy):
        """A casting enemy's expected damage per round, roughly.

        Mean damage-per-pip over the pool's hits, times pip income
        (one plus the power-pip chance), times a typical enemy
        accuracy. Sanity: the calibrated casting Alicane Swiftarrow
        measures 94/round in the sim; this estimate prices him at ~92.
        """
        hits = [self.cards[n] for n in (enemy.spell_pool or [])
                if n in self.cards
                and self.cards[n].kind in ("damage", "drain")]
        if not hits:
            return 0.0
        per_pip = sum(h.damage / max(1, h.pips) for h in hits) / len(hits)
        income = 1.0 + float(getattr(enemy, "power_pip_chance", 0.0) or 0)
        return per_pip * income * 0.75

    def _apportion_incoming(self, read):
        """Split the measured incoming between casters and flat mobs.

        `_estimate_incoming` divides the health lost each round evenly
        across the living enemies -- which hands half of a casting
        boss's Sunbird to the minion standing next to him, and then
        `_apply_pool` makes the boss cast his own damage ON TOP. On a
        live fight that double-count read Magma Man at 117/round (his
        real output is ~50), the rollouts saw ~230/round incoming on a
        ~150/round board, every line died inside the horizon, and the
        decision collapsed to the sentinel ranking. The flat mobs'
        share is the measured total minus what the casters are already
        modeled to deal -- only ever LOWERED, never raised, and
        floored so no living mob reads as harmless.
        """
        living = [e for e in read.state.enemies if e.alive]
        pooled = [e for e in living
                  if getattr(e, "spell_pool", None) is not None]
        flat = [e for e in living if e not in pooled]
        if not pooled or not flat:
            return
        total = self._measured_incoming * len(living)
        # Divided by the party, because the measurement it is subtracted
        # from already is. `_estimate_incoming` reads THIS wizard's health
        # drop, and a boss casting one Wraith a round in a circle of four
        # spreads that round's output across four health bars -- so
        # subtracting the caster's whole modelled output from one
        # wizard's share drives the flat mobs to the floor and reads a
        # dangerous board as a harmless one.
        modeled = sum(self._modeled_pool_output(e)
                      for e in pooled) / self.party_size
        share = max(10.0, (total - modeled) / len(flat))
        for e in flat:
            e.flat_hit = min(e.flat_hit, share)

    def _record(self, decision, read):
        self.history.append(decision)
        if self.on_decision:
            self.on_decision(decision, read)

    def report_lost_round(self, round_number, reason):
        """A round nobody could read. Recorded, rather than only counted.

        Separate from `_record` because there is no board to record: the
        read is what failed. `on_lost_round` is wired to the telemetry so
        the round appears in the Decisions table as the failure it was,
        instead of leaving a gap that reads as though the round never
        happened.
        """
        decision = PolicyDecision(passing=True, policy="board read failed",
                                  reason=reason)
        self.history.append(decision)
        if self.on_lost_round:
            self.on_lost_round(round_number, reason)

    def report_failed_cast(self, card_name, exc):
        """The chosen card never went out. Amend the round, do not hide it.

        The record was written before the click, so by the time this is
        known the round already claims the card was cast. The prediction
        has to go with it -- a round in which nothing was played says
        nothing about the damage model, and leaving it in counts the
        misclick as the model's error.
        """
        reason = (f"the cast of {card_name} did not go through "
                  f"({type(exc).__name__}: {exc}) — passed instead")
        if self.on_failed_cast:
            self.on_failed_cast(reason)

    def report_recovered_cast(self, pick, first_choice):
        """The runner-up went out after the chosen card would not.

        Its own channel again, and for the same reason `on_slow_cast` is
        separate: `on_failed_cast` has already run and marked the round
        a pass, which is now wrong. See
        `Telemetry.note_recovered_cast` for what has to be undone and
        what deliberately is not.
        """
        if self.on_recovered_cast:
            self.on_recovered_cast(pick.card, pick.target, first_choice)

    def report_slow_cast(self, card_name):
        """The fast cast did not take; trying again slowly.

        Its own channel, deliberately NOT `on_failed_cast`: that one
        amends the round to "nothing was played" and drops the
        prediction, which would be a lie told before the retry has even
        run. Nothing has failed yet.

        Said out loud because it is the measurement. A card that only
        goes out on the second, slower attempt is telling us the two
        clicks are too close together for it -- which is a knob
        (`cast_time`) rather than a mystery. A card that fails both ways
        is a different problem and reports itself separately.
        """
        if self.on_slow_cast:
            self.on_slow_cast(
                f"{card_name} did not go out on the first click — trying "
                f"again with a longer pause between selecting the card and "
                f"clicking the target")

    def _deck_remaining(self, read_hand):
        """Best-effort undrawn deck. Only its *size* and the kinds in it
        matter -- `Featurizer.key` uses it for a scarcity count and
        nothing else (rl_agent.py:52-54)."""
        remaining = []
        drawn = dict()
        for n in self._seen:
            drawn[n] = drawn.get(n, 0) + 1
        for n in self.decklist:
            if drawn.get(n):
                drawn[n] -= 1
                continue
            card = self.cards.get(n)
            if card is not None:
                remaining.append(card)
        return remaining

    def _sim_for(self, read):
        """A `Sim` the policy can call helpers on (`sim.can_cast`,
        `sim.rules`). It is never stepped -- the *game* is the engine
        here, this only supplies the policy's expected interface."""
        from w101_sim import Boss, Rules, Sim
        enemy = read.state.enemies[0]
        return Sim(
            cards=self.cards, decklist=self.decklist, school=self.school,
            boss=Boss(name=enemy.name, hp=int(enemy.max_hp),
                      school=enemy.school, dmg=0),
            rules=self.rules or Rules(),
            player_hp=int(read.state.player.max_hp) or 3000,
            # The wizard's real gear. Empty means "no gear at all", so
            # every hit is priced below what it lands for and the policy
            # is optimising a different fight than the one on screen.
            player_stats=self.player_stats,
            rng=random.Random(0),
        )

    # -- translation to wizsprinter ---------------------------------------
    #
    # WARNING, and the reason `WizAiCombatHandler` below exists.
    #
    # `SprintyCombat`'s cast loop re-queries the same spec after each cast
    # and only stops when `needs_post_filter` is set
    # (`sprinty_combat.py:1554,1758-1763`). That flag requires
    # `isinstance(move.card, TemplateSpell)`, so a `NamedSpell` can never
    # set it -- and the loop therefore keeps casting while another copy of
    # that card remains in hand. Three Fireblades in hand become three
    # Fireblades cast, from one policy decision.
    #
    # For a config-driven bot that is a feature. For measuring a policy it
    # is fatal: the thing that played the fight is not the thing you
    # loaded. Use `WizAiCombatHandler` for any run whose numbers matter.
    def _to_priority_line(self, decision):
        from .combat_api_shim import (Move, MoveConfig, NamedSpell,
                                      PriorityLine, TargetData, TargetType)
        if decision.passing:
            return PriorityLine(priorities=[
                MoveConfig(move=Move(card=NamedSpell("pass")))])

        game_name = WIKI_TO_GAME.get(decision.card_name, decision.card_name)
        target = TargetData(TargetType.type_enemy)
        if decision.target_index is not None:
            target = TargetData(TargetType.type_enemy,
                                extra_data=decision.target_index)
        return PriorityLine(priorities=[
            MoveConfig(move=Move(card=NamedSpell(game_name)), target=target),
            # If the named card cannot actually be played this round --
            # the client is the authority on that, not our read -- fall
            # through to a pass rather than stalling the fight.
            MoveConfig(move=Move(card=NamedSpell("pass"))),
        ])

    # -- construction helpers ---------------------------------------------
    @classmethod
    def from_trained(cls, school, deck=None, policy=None, cards=None,
                     **kw):
        """Wire up the default: wizAi's card table plus a policy.

        With no `policy`, uses `make_blade_stack(3)` -- the heuristic
        wizAi's own tables use as the baseline, so a live run is directly
        comparable to the simulated numbers.
        """
        if cards is None:
            from data_full import load_spells_full
            cards = load_spells_full()
        if policy is None:
            from w101_sim import make_blade_stack
            policy = make_blade_stack(3)
        return cls(policy=policy, cards=cards, school=school,
                   decklist=deck or [], **kw)


class WizAiCombatHandler:
    """One policy decision, one cast. The path to use when the numbers matter.

    Subclasses wizwalker's `CombatHandler` (resolved lazily, see
    `make_combat_handler`), whose documented extension point is a single
    `handle_round()` per planning phase. That is exactly the contract a
    wizAi policy is written against, and going straight to it avoids two
    problems that `SprintyCombat` introduces:

      * **Multi-casting.** `SprintyCombat` re-queries a `NamedSpell` after
        every cast and plays every duplicate in hand (see the note on
        `WizAiBackend._to_priority_line`). Here, one decision issues one
        `CombatCard.cast` and the round ends.

      * **Target indices meaning different things.** wizAi's `"name@i"`
        indexes *its own* enemy list -- alive, hostile, in read order --
        while `SprintyCombat.get_enemies()` partitions by team and keeps
        the dead, so after the first kill the two disagree and the policy
        quietly hits the wrong mob. Here the index is resolved against
        the very list the policy was shown, because this class built it.
    """

    def __init__(self, client, backend):
        self.client = client
        self.backend = backend
        self._last_read = None
        #: rounds lost to a failed board read, surfaced at the end of a run
        self._read_failures = 0
        backend.attach_combat(self)

    # `CombatHandler` supplies get_members/get_cards/round_number/
    # pass_button/in_combat; `read_state` only needs those, so `self` is a
    # valid `combat` for the backend.

    async def handle_round(self):
        # Announced before anything is read: the coordinator only waits
        # for seats it knows are in a duel, so a wizard still walking to
        # the circle never stalls the ones already swinging -- and one
        # that has just walked in has to be counted before the others
        # plan without it.
        hive = getattr(self.backend, "coordinator", None)
        if hive is not None:
            hive.enter_combat(self.backend.seat)
        # Every action here is a mouse click -- wizwalker has no memory
        # API for casting -- and `MouseHandler.__aenter__` is what
        # activates the mouseless cursor hook that makes those clicks land
        # without stealing the real cursor. `SprintyCombat.handle_round`
        # wraps its whole round the same way (sprinty_combat.py:1783).
        # Without this the first cast silently does nothing.
        async with self.client.mouse_handler:
            try:
                decision = await self.backend.decide()
            except Exception as exc:
                # A memory read can fail mid-round -- the duel ends, a
                # participant is freed, the client hiccups. `decide()`
                # already contains policy errors; this contains the read.
                # Losing a round is recoverable, crashing out of a live
                # fight is not.
                #
                # But it goes through the decision channel rather than
                # into a counter nothing reads. A round dropped here used
                # to leave *no* record at all: the Decisions table went
                # round 3, round 5, the pass was never counted, and round
                # 5's board was differenced against round 3's -- so round
                # 4's damage was folded into round 3's residual and
                # scored against the damage model. A round the run could
                # not read has to appear as exactly that.
                self._read_failures += 1
                await self._report_lost_round(exc)
                await self.pass_button()
                return
            read = self.backend.last_read
            self._last_read = read

            if decision.passing or read is None:
                await self.pass_button()
                return

            card = self._pick_card(read, decision.card_name)
            if card is None:
                await self.pass_button()
                return

            target = await self._resolve_target(
                read, decision.target_index,
                self.backend.cards.get(decision.card_name))
            try:
                # `sleep_time` is per pause and wizwalker pauses twice --
                # after clicking the card and after aiming -- so the
                # default 1.0 costs ~2 s of standing still per cast, ~12 s
                # of dead time in a six-round fight. `cast_time` is the
                # backend's own knob for exactly this and nothing ever
                # consumed it. 0.3 is not reckless because a miss is no
                # longer silent: a cast that does not go through is
                # reported and amended in telemetry, so if this machine
                # needs slower clicks the operator sees it immediately
                # instead of wondering.
                held = await self._cards_in_hand()
                await card.cast(target, sleep_time=self.backend.cast_time)
                if not await self._card_left_the_hand(held):
                    # Once more, slowly, before giving the round away.
                    #
                    # The two clicks are "select the card" and "click the
                    # target", `sleep_time` apart. A card the game can
                    # cast without asking anything goes out on the first
                    # click; one that has to put the board into target
                    # selection first needs the UI to be up before the
                    # second click lands, and 0.3s is not obviously long
                    # enough for that. Live, Pixie -- a heal, which is
                    # the one card in these decks that makes the game ask
                    # who -- failed twice this way while blades, traps
                    # and nukes from the same hands went out fine.
                    #
                    # So the retry is at wizwalker's own default rather
                    # than ours, and it SAYS which attempt worked. If the
                    # slow one lands, the cause is this timing and the
                    # cure is a bigger `cast_time`; if it does not, the
                    # card really cannot be cast where it was aimed, and
                    # the round is passed exactly as before.
                    self.backend.report_slow_cast(decision.card_name)
                    held = await self._cards_in_hand()
                    await card.cast(self._retry_target(read, decision, target),
                                    sleep_time=self.RETRY_CAST_TIME)
                    if not await self._card_left_the_hand(held):
                        raise RuntimeError(
                            f"the card was still in hand after casting, "
                            f"twice — once at {self.backend.cast_time:.1f}s "
                            f"between clicks and once at "
                            f"{self.RETRY_CAST_TIME:.1f}s"
                            + await self._why_not(read, decision, card))
            except Exception as exc:
                # A misclick or a board that moved under us costs this
                # round, not the fight -- but the round has *already*
                # been recorded, by `decide()`, as a cast that was made.
                # Left alone, next round's board shows the target at
                # unchanged HP, the residual settles at minus the whole
                # prediction, and a cast that never happened is charged
                # to the damage model as its worst miss of the run.
                self.backend.report_failed_cast(decision.card_name, exc)
                if not await self._try_the_next_best(read, decision):
                    await self.pass_button()

    async def _try_the_next_best(self, read, decision):
        """The runner-up, once, rather than surrendering the round.

        A card that will not go out used to cost the whole round: the
        handler passed, and against a board that is out-damaging the
        party a free round is the difference between winning and not.
        The operator's report of it is exact -- "it double clicks, which
        unselects the spell, so nothing happens and the fight stalls
        until the round counter hits 0".

        The policy already ranked every other move it weighed this round
        and the list is sitting on the decision. So the answer to "the
        best card did not go out" is the second best, not nothing.

        Once. If the card failed because the board moved under us, the
        alternative is likely to fail the same way, and a loop of them
        would spend the planning window clicking instead of playing. One
        attempt turns a lost round into a played one where the cause was
        the card; where the cause is the board, the round is passed
        exactly as it was before.

        Deliberately silent about `report_failed_cast`, which has
        already run: the round's record now says nothing was played, and
        if this succeeds it is amended by the cast that did.

        What it costs if `_card_left_the_hand` is ever wrong: that check
        is a hand-COUNT drop, so a false negative means the first cast
        committed and the count had not caught up. Clicking another card
        then does not cast twice -- Wizard101 allows one spell a round
        and a second selection replaces the queued one -- it downgrades
        the round from the best card to the second best. That is the
        bound on the damage, against a benefit of nothing-to-second-best
        in the case this exists for, and it only fires at all once the
        handler already believes nothing was played.
        """
        alternatives = [c for c in (decision.candidates or ())
                        if c.card and c.card != decision.card_name
                        and c.card != "pass"]
        if not alternatives:
            return False
        # The candidate list is not stored in rank order -- it is in the
        # order the rollout scored the hand -- so the runner-up has to be
        # taken by score, by the same key `greedy_ttk` chose the winner
        # by. `rank_candidate` is that key, and lives over there so it
        # has one home.
        from .policies import rank_candidate
        pick = min(alternatives, key=rank_candidate)

        card = self._pick_card(read, pick.card)
        if card is None:
            return False
        spec = self.backend.cards.get(pick.card)
        try:
            target = await self._resolve_target(read, pick.target, spec)
            held = await self._cards_in_hand()
            await card.cast(target, sleep_time=self.RETRY_CAST_TIME)
            if not await self._card_left_the_hand(held):
                return False
        except Exception:
            return False

        self.backend.report_recovered_cast(pick, decision.card_name)
        return True

    async def _report_lost_round(self, exc):
        """Put the dropped round in the record, not just in a counter.

        `_read_failures` is documented as surfaced at the end of a run
        and never was -- nothing in the package read it. Ten rounds lost
        across a fight were invisible to the operator, who saw a policy
        that appeared to pass with no logged reason.
        """
        try:
            number = int(await self.round_number())
        except Exception:
            last = getattr(self._last_read, "round_number", 0) or 0
            number = last + 1
        try:
            self.backend.report_lost_round(
                number,
                f"could not read the board ({type(exc).__name__}: {exc}) "
                f"— passed this round")
        except Exception:
            pass          # reporting a lost round must not lose the fight

    def _pick_card(self, read, name):
        cards = read.hand_cards.get(name) or []
        return cards[0] if cards else None

    #: how long to wait for the hand to shrink after a cast, and how
    #: often to look. A cast is two clicks and a server round-trip, so
    #: the hand does not update the instant `cast` returns.
    CAST_SETTLE = 2.0
    CAST_POLL = 0.15
    #: `sleep_time` for the second attempt -- wizwalker's own default,
    #: which is what the fast one is a deliberate reduction from.
    RETRY_CAST_TIME = 1.0

    async def _cards_in_hand(self):
        """How many cards are in hand, or None if it will not read."""
        try:
            return len(await self.get_cards())
        except Exception:
            return None

    async def _card_left_the_hand(self, before):
        """Did the cast actually go through?

        `CombatCard.cast` clicks and returns; it raises nothing when the
        click does not take. Wizard101 **deselects** a card clicked at
        something it cannot be cast on -- a heal aimed at a monster, a
        self-buff aimed at a mob -- so the card goes back to the hand,
        the round passes with nothing played, and the duel sits there
        until the round timer expires. From outside that is a bot that
        decided to do nothing, and it was invisible: the round had
        already been recorded as a cast that was made, so the next
        board's unchanged health settled as the damage model's worst
        miss of the run.

        wizwalker checks this itself for the enchant branch
        (`card.py:63-65`) and for no other. So it is checked here.

        Unreadable counts are treated as success -- refusing to believe
        a cast landed because the hand would not read would turn every
        bad read into a false failure, and the settle already reports
        those separately.
        """
        import asyncio

        if before is None:
            return True
        deadline = self.CAST_SETTLE
        while deadline > 0:
            now = await self._cards_in_hand()
            if now is None or now < before:
                return True
            await asyncio.sleep(self.CAST_POLL)
            deadline -= self.CAST_POLL
        return False

    def _retry_target(self, read, decision, first):
        """What to aim at on the SECOND attempt, which is not the first.

        Retrying identically but slower tests one hypothesis, timing, and
        the instrumented run answered it: not timing. Across that run
        every self-targeted cast that costs no pips went out -- eleven of
        eleven, Tower Shield four times and Fireblade seven -- and every
        self-targeted cast that costs pips failed, three of three, all of
        them Pixie. The diagnostic line rules out the rest: aimed at
        self, a card sharing that aim in the same hand went out, the pips
        were there, and the client still called it castable.

        What separates them is the second click. `CombatCard.cast` with a
        `CombatMember` clicks the card, waits, then clicks the TARGET'S
        HEALTH TEXT -- for a self-cast, the caster's own number in the
        HUD. With `None` it clicks the card once and stops
        (`card.py:72-76`). If the game has already aimed the spell at the
        only wizard it can go on, that second click is at best redundant
        and at worst deselects it, which is exactly what the operator
        described: "it double clicks, which unselects the spell after it
        tries to cast it so nothing happens".

        So the retry drops the second click for self-casts. Deliberately
        only the RETRY: the two-click path is working for eleven of
        eleven zero-pip self-casts, and there is no reason to disturb
        those to fix three. If a single click turns out to be right in
        general, the next run says so and it can move.
        """
        card = self.backend.cards.get(decision.card_name)
        tgt = (_primary_target(card) or _target_from_kind(card)) if card \
            else None
        if tgt == "self":
            return None
        return first

    def _who_needs_the_shield(self, read):
        """The wizard in most danger, which is not always the caster.

        A shield's ops read `tgt: 'self'`, and that is the right default
        -- cast without choosing, it lands on you -- but Wizard101 lets
        it go on any wizard in the circle, and the card data has no way
        to say so. So every shield went on the caster, and the second
        live party run shows what that costs: the ice wizard cast
        nineteen Tower Shields, four of them twice in a row, and in
        every one of those pairs it was at FULL health while the fire
        wizard beside it was not (1270/1270 against 938/1042, 922/1042,
        889/1064). The fire wizard was also dealing more of the damage,
        so the shields went on the wizard least likely to need them.

        Ranked by health FRACTION rather than health missing, which is
        the difference between the two and the right one here: a heal
        restores what is gone, so `_resolve_target`'s ally branch asks
        who has lost the most; a shield buys rounds against what is
        coming, so what matters is who is nearest to dying. A 1270-health
        wizard down 200 is in less trouble than a 900-health one down
        200.

        A wizard already carrying a damage ward drops behind everyone
        who is not, because that is the "two Tower Shields on the same
        wizard" case exactly -- they stack in game, but a second one on
        a full-health caster while a teammate has none is not a stack,
        it is a wasted card.

        The caster wins ties, so a solo wizard and a party that is level
        on both counts behave exactly as before.
        """
        me = read.state.player
        mates = list(getattr(read, "ally_members", ()) or ())
        allies = list(getattr(read.state, "allies", ()) or ())

        def shielded(actor):
            for ward in getattr(actor, "wards", ()) or ():
                # A damage ward is the one a shield duplicates. Traps are
                # wards too and sit on the enemy, so they cannot appear
                # here, but a prism or an absorb is not a shield.
                if getattr(ward, "kind", "") == "damage" and \
                        float(getattr(ward, "percent", 0.0) or 0.0) < 0:
                    return True
            return False

        def danger(actor):
            try:
                left = float(actor.hp) / float(actor.max_hp)
            except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                left = 1.0
            return (shielded(actor), left)

        best, who = danger(me), read.client_member
        for i, ally in enumerate(allies):
            if i >= len(mates) or mates[i] is None:
                continue
            if not getattr(ally, "alive", True):
                continue
            score = danger(ally)
            if score < best:
                best, who = score, mates[i]
        return who

    async def _why_not(self, read, decision, clicked):
        """Everything that could distinguish the causes, in one line.

        The message this replaced asserted a cause: "Wizard101 deselects
        a card clicked at something it cannot be cast on, so this is
        usually a card aimed at the wrong kind of target." That is a
        good explanation of *a* failure and the wrong one for the
        failure it kept being printed on. Pixie is `tgt: 'self'`, so
        `_resolve_target` hands it `read.client_member` -- the identical
        object, down the identical branch, that Fireblade and Tower
        Shield get. In the run that prompted this the fire wizard cast
        Fireblade in round 1 and Pixie failed in round 3 of the same
        duel, on the same client, aimed at the same member. Whatever
        stopped it, it was not the kind of target.

        So this states facts instead of a theory, and states the ones
        that separate the surviving explanations:

          * what it was aimed at, and what else in hand aims at the same
            thing -- if a self-cast card fails while another self-cast
            card in the same hand went out, aiming is exonerated;
          * the card's school and cost against the wizard's pips, since
            a power pip is worth 2 only for the caster's own school and
            an off-school card is the one that can be unaffordable while
            looking affordable;
          * whether the client still calls the card castable, which is
            the game's own answer and worth more than any of ours.

        Best effort throughout: this runs while a round is already being
        given away, so a read that fails here must not replace the
        failure being reported with its own.
        """
        bits = []
        card = self.backend.cards.get(decision.card_name)
        tgt = (_primary_target(card) or _target_from_kind(card)) if card \
            else None
        bits.append(f"aimed at {tgt or 'nothing'}")

        try:
            same = sorted({c.name for c in read.state.player.hand
                           if c.name != decision.card_name
                           and (_primary_target(c) or _target_from_kind(c))
                           == tgt})
            if same:
                bits.append("same aim in hand: " + ", ".join(same))
        except Exception:
            pass

        try:
            me = read.state.player
            own = (card.school or "") == (self.backend.school or "")
            worth = 2 if own else 1
            bits.append(
                f"{card.pips}-pip {card.school} card, {'own' if own else 'off'}"
                f" school, and the wizard holds {me.norm_pips} normal + "
                f"{me.pow_pips} power (worth {worth} each here) = "
                f"{me.norm_pips + me.pow_pips * worth}")
        except Exception:
            pass

        try:
            bits.append("the client "
                        + ("still calls it castable"
                           if await clicked.is_castable()
                           else "no longer calls it castable"))
        except Exception:
            pass

        return ". " + "; ".join(bits)

    async def _resolve_target(self, read, index, card=None):
        """What to click after clicking the card.

        Not always an enemy, which is what the first version assumed. A
        blade is `tgt: 'self'` and has to be clicked on the *caster* --
        wizsprinter resolves `type_self` to `get_client_member()`
        (sprinty_combat.py:746-747) -- and aiming it at a mob simply
        fails, silently, wasting the round. The wizAi card already
        records this: `Card.ops[i]['tgt']` is one of self / enemy /
        enemies / ally / allies / global.

        AoE and global spells take no target at all; `CombatCard.cast`
        treats `None` as "just click the card" (card.py:24-30).
        """
        tgt = _primary_target(card) or _target_from_kind(card)

        if tgt == "self":
            if getattr(card, "kind", "") == "shield":
                return self._who_needs_the_shield(read)
            return read.client_member
        if tgt in ("enemies", "allies", "global"):
            return None                      # no target click
        if tgt == "ally":
            # This wizard's own team, never `read.members` -- that is
            # every participant in the duel, both sides, and the enemy
            # team comes first. Picking "the first member that is not
            # me" out of it therefore lands on a MOB, and in a solo
            # fight there is nothing else in it to land on.
            #
            # Wizard101 does not cast a friend-only spell at an enemy;
            # the click deselects the card. So Pixie was selected, aimed
            # at a Sokkwi Ripper, unselected, and the round passed with
            # nothing cast -- the duel then sat there until the round
            # timer expired. It is the whole failure: not a misprediction
            # but a wasted round, every time a heal came up.
            #
            # The hurt one, because that is what a heal is for, and it
            # is the caster whenever it is alone.
            mates = [m for m in read.ally_members if m is not None]
            allies = list(getattr(read.state, "allies", ()) or ())
            if not mates or not allies:
                return read.client_member
            hurt = max(allies, key=_missing_health)
            if _missing_health(hurt) > _missing_health(read.state.player):
                i = allies.index(hurt)
                if 0 <= i < len(mates):
                    return mates[i]
            return read.client_member

        foes = read.enemy_members
        if not foes:
            return None
        if index is None or not (0 <= index < len(foes)):
            index = 0
        return foes[index]


# --------------------------------------------------------------------------
# Attaching the real base classes.
#
# Both classes above have to work without wizwalker installed -- that is
# what lets `mock_client` exercise them anywhere -- but must be genuine
# subclasses of wizwalker's types at runtime, because `SprintyCombat` and
# `CombatHandler.handle_combat` both rely on inheritance.
#
# The obvious trick, assigning `__bases__` after the fact, does not work:
#
#     >>> class Mine: pass
#     >>> Mine.__bases__ = (SomeBase,)
#     TypeError: __bases__ assignment: 'SomeBase' deallocator differs
#                from 'object'
#
# CPython refuses whenever the current base is exactly `object`, because
# `object` uses `object_dealloc` and a heap type uses `subtype_dealloc`.
# That is long-standing behaviour, not a 3.13 change.
#
# So the subclass is *built* at call time instead, and cached. The class
# construction is factored out so a test can drive it with a stub base
# and cover this path without wizwalker present -- the earlier version of
# these helpers was never executed by a test, which is exactly how the
# broken `__bases__` assignment survived.
# --------------------------------------------------------------------------
_handler_class = None
_backend_class = None


def _build_handler_class(combat_handler_cls):
    class LiveWizAiCombatHandler(WizAiCombatHandler, combat_handler_cls):
        """`WizAiCombatHandler`, but really a `CombatHandler`."""

        def __init__(self, client, backend):
            # wizwalker's first: it sets up the window cache that
            # get_members()/get_cards() read.
            combat_handler_cls.__init__(self, client)
            WizAiCombatHandler.__init__(self, client, backend)

    return LiveWizAiCombatHandler


def make_combat_handler(client, backend):
    """A handler that is both a wizAi decision loop and a `CombatHandler`.

    This is the path `run_live` uses, and the one to use for any run whose
    numbers matter -- see `WizAiCombatHandler`.
    """
    global _handler_class
    if _handler_class is None:
        from wizwalker.combat import CombatHandler
        _handler_class = _build_handler_class(CombatHandler)
    return _handler_class(client, backend)


def _build_backend_class(base_backend_cls):
    class LiveWizAiBackend(WizAiBackend, base_backend_cls):
        """`WizAiBackend`, but really a `BaseCombatBackend`."""

    return LiveWizAiBackend


def make_sprinty_backend(*args, **kwargs):
    """A `WizAiBackend` that `SprintyCombat` will accept.

    Only for driving wizAi inside an existing Deimos setup. Do not
    measure a policy through it: `SprintyCombat` re-queries a
    `NamedSpell` after every cast and plays every duplicate in hand.
    """
    global _backend_class
    if _backend_class is None:
        from wizwalker.extensions.wizsprinter.combat_backends.backend_base \
            import BaseCombatBackend
        _backend_class = _build_backend_class(BaseCombatBackend)
    return _backend_class(*args, **kwargs)
