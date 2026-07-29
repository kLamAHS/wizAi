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
wizAi's in-fight contract is `policy(sim, state) -> Card | str | None`
(`w101_sim.py:1978`, `:2128`), and its action strings are already card
names -- optionally `"name@i"` for a target index (`rl_agent.py:97-107`).
That maps onto `NamedSpell(name)` plus a `TargetData` almost directly.

The wizwalker/wizsprinter imports are deferred into the methods that need
them, so this module imports cleanly on Linux and can be unit-tested
against `mock_client`.
"""
import random

from .live_state import NameResolver, WIKI_TO_GAME, read_state


class PolicyDecision:
    """What the policy decided, before it becomes a wizsprinter move.

    Kept as its own type so the decision can be tested, logged, and
    replayed without any wizsprinter object in sight.
    """

    def __init__(self, card_name=None, target_index=None, passing=False,
                 reason=""):
        self.card_name = card_name
        self.target_index = target_index
        self.passing = passing
        self.reason = reason

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
                 on_decision=None, rules=None, catalog=None):
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
        """
        self.policy = policy
        self.cards = cards
        self.school = school
        self.decklist = list(decklist or [])
        self.cast_time = cast_time
        self.combat = None
        self.on_decision = on_decision
        self.rules = rules
        self.resolver = NameResolver(cards, catalog)
        self.history = []          # PolicyDecision, in order
        self._seen = []            # card names observed in hand so far
        #: the `LiveRead` behind the most recent decision. The handler
        #: casts from this, so the card it clicks and the target it picks
        #: come from the same snapshot the policy saw.
        self.last_read = None

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
        self.last_read = read

        sim = self._sim_for(read)
        try:
            choice = self.policy(sim, read.state)
        except Exception as exc:                      # a policy must never
            decision = PolicyDecision(                # break a live fight
                passing=True, reason=f"policy raised {type(exc).__name__}: {exc}")
            self._record(decision, read)
            return decision

        decision = self._interpret(choice, read)
        self._record(decision, read)
        return decision

    def _interpret(self, choice, read):
        if choice is None:
            return PolicyDecision(passing=True, reason="policy chose to pass")

        name = getattr(choice, "name", choice)
        if not isinstance(name, str):
            return PolicyDecision(passing=True,
                                  reason=f"policy returned {choice!r}")

        target_index = None
        if "@" in name:
            name, _, idx = name.rpartition("@")
            try:
                target_index = int(idx)
            except ValueError:
                target_index = None

        if name in ("__pass__", "pass"):
            return PolicyDecision(passing=True, reason="policy chose to pass")

        if name not in read.hand_cards:
            return PolicyDecision(
                passing=True,
                reason=f"{name!r} is not a castable card in hand "
                       f"({sorted(read.hand_cards)})")

        return PolicyDecision(card_name=name, target_index=target_index,
                              reason="policy choice")

    def _record(self, decision, read):
        self.history.append(decision)
        if self.on_decision:
            self.on_decision(decision, read)

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
        backend.attach_combat(self)

    # `CombatHandler` supplies get_members/get_cards/round_number/
    # pass_button/in_combat; `read_state` only needs those, so `self` is a
    # valid `combat` for the backend.

    async def handle_round(self):
        # Every action here is a mouse click -- wizwalker has no memory
        # API for casting -- and `MouseHandler.__aenter__` is what
        # activates the mouseless cursor hook that makes those clicks land
        # without stealing the real cursor. `SprintyCombat.handle_round`
        # wraps its whole round the same way (sprinty_combat.py:1783).
        # Without this the first cast silently does nothing.
        async with self.client.mouse_handler:
            decision = await self.backend.decide()
            read = self.backend.last_read
            self._last_read = read

            if decision.passing or read is None:
                await self.pass_button()
                return

            card = self._pick_card(read, decision.card_name)
            if card is None:
                await self.pass_button()
                return

            target = await self._resolve_target(read, decision.target_index)
            try:
                await card.cast(target)
            except Exception:
                # A misclick or a board that moved under us costs this
                # round, not the fight.
                await self.pass_button()

    def _pick_card(self, read, name):
        cards = read.hand_cards.get(name) or []
        return cards[0] if cards else None

    async def _resolve_target(self, read, index):
        """Map the policy's enemy index onto a live member.

        `read.enemy_members` is recorded in the same order as
        `read.state.enemies`, so index i here is the same mob the policy
        was looking at.
        """
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
