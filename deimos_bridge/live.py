"""Let a trained wizAi policy fight real enemies through Deimos.

This is the end of the road the rest of the bridge is building toward.
`engine.py` measures a policy against Deimos' *model* of the client;
this module hands the policy to Deimos and lets it play the actual game.

It only runs on Windows with Wizard101 open, because that is the only
place `wizwalker` can read the client's memory. Everything in here that
does not touch the client is a plain function over plain data, so the
translation can be tested anywhere -- and is, in `tests/test_live_adapter.py`.

    from deimos_bridge.live import WizAiFighter

    fighter = WizAiFighter(client, policy=agent.policy(),
                           cards=cards, decklist=decklist, school="fire")
    await fighter.handle_combat()

Two honest limits, both visible in the code:

  the deck is estimated
      The client shows a hand, not the undrawn remainder. The policy's
      `nukes_left` feature counts cards still in the deck, so we track
      what has been seen and subtract. Wrong only if the decklist passed
      in does not match the deck actually equipped.

  names must match
      wizAi knows cards by wiki name, the client by its own. Anything
      unmatched is dropped from the hand with a warning rather than
      guessed at, because a silent mismatch would have the policy
      reasoning about a card it does not hold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from w101_sim import Actor, Card, Hanging, Sim, State

log = logging.getLogger(__name__)

#: Effect types that read as a damage charm or ward on the wizAi side.
_CHARM_TYPES = ("modify_outgoing_damage",)
_WARD_TYPES = ("modify_incoming_damage",)


# --------------------------------------------------------------- reading


@dataclass
class LiveEffect:
    """One hanging effect, flattened out of the client."""

    effect_type: str
    param: float
    school: str | None = None      # None = every school
    spell_name: str = ""

    @property
    def percent(self) -> float:
        return self.param / 100.0


@dataclass
class LiveMember:
    """One combatant, flattened out of the client."""

    name: str
    school: str
    health: float
    max_health: float
    level: int = 1
    is_client: bool = False
    is_player: bool = False
    normal_pips: int = 0
    power_pips: int = 0
    shadow_pips: int = 0
    stunned: bool = False
    effects: list = field(default_factory=list)


@dataclass
class LiveRound:
    """Everything one planning phase needs, with no live objects left in it."""

    client: LiveMember
    enemies: list
    allies: list
    hand: list                     # card names, in hand order
    round_number: int = 0


# ------------------------------------------------------------ translation


def _hanging(effect: LiveEffect, slot: str, kind: str = "damage") -> Hanging:
    schools = None if effect.school is None else {effect.school}
    return Hanging(name=effect.spell_name or effect.effect_type, slot=slot,
                   kind=kind, percent=effect.percent, schools=schools,
                   source="deck")


def to_actor(member: LiveMember, team: int) -> Actor:
    """A wizAi `Actor` from a live combatant.

    Charms and wards are split by effect type rather than by sign: the
    client distinguishes outgoing from incoming, and a negative outgoing
    effect is a weakness, not a shield.
    """
    charms, wards = [], []
    for effect in member.effects:
        if effect.effect_type in _CHARM_TYPES:
            charms.append(_hanging(effect, slot="charm"))
        elif effect.effect_type in _WARD_TYPES:
            wards.append(_hanging(effect, slot="ward"))

    return Actor(
        name=member.name, school=member.school, hp=float(member.health),
        max_hp=float(member.max_health or member.health), team=team,
        norm_pips=member.normal_pips, pow_pips=member.power_pips,
        charms=charms, wards=wards, stunned=1 if member.stunned else 0,
    )


def resolve_hand(names, cards: dict) -> tuple:
    """Live card names -> wizAi `Card`s, plus whatever could not be matched.

    Returns `(hand, unmatched)`. Unmatched names are reported rather than
    dropped silently: the policy is about to choose among these, and a
    card it cannot see is a card it will never cast.
    """
    hand, unmatched = [], []
    for name in names:
        card = cards.get(name)
        if card is None:
            unmatched.append(name)
            continue
        hand.append(card)
    return hand, unmatched


def estimate_deck(decklist, cards: dict, seen) -> list:
    """What is plausibly left undrawn.

    The client does not expose the undrawn deck. The policy only uses it
    to count remaining damage cards, so an estimate -- the decklist minus
    everything observed so far -- is enough for that feature and is
    labelled as an estimate wherever it surfaces.
    """
    remaining = list(decklist)
    for name in seen:
        if name in remaining:
            remaining.remove(name)
    return [cards[n] for n in remaining if n in cards]


def build_state(live: LiveRound, cards: dict, decklist, seen=()) -> State:
    """A wizAi `State` describing the round in front of us."""
    player = to_actor(live.client, team=0)
    hand, unmatched = resolve_hand(live.hand, cards)
    if unmatched:
        log.warning("hand cards with no wizAi record, ignored: %s",
                    ", ".join(sorted(set(unmatched))))
    player.hand = hand
    player.deck = estimate_deck(decklist, cards, seen)

    enemies = [to_actor(m, team=1) for m in live.enemies]
    allies = [to_actor(m, team=0) for m in live.allies]
    state = State(player, enemies, allies)
    state.turn = live.round_number
    return state


def choose(policy, sim: Sim, state: State):
    """Run the policy and normalise whatever it hands back.

    Policies may return a `Card`, an `Action`, or `None` for pass.
    Returns `(card_name, target_index)`, or `(None, 0)` to pass.
    """
    action = policy(sim, state)
    if action is None:
        return None, 0
    card = getattr(action, "card", action)
    if card is None:
        return None, 0
    return card.name, getattr(action, "target", 0)


# ------------------------------------------------------------- the driver


def _combat_handler_base():
    """The real `CombatHandler` where one exists, else the headless stub.

    Order matters. On Windows with the game installed, `wizwalker`
    imports for real and must be left alone -- installing the shim there
    would replace the memory readers this class depends on with
    placeholders. Only when the real import fails (no Windows, no
    `ctypes.windll`) do we fall back, and then the stub is fine to
    subclass but refuses to be constructed, so the failure arrives with
    a message about needing the game rather than an AttributeError.
    """
    try:
        from wizwalker.combat import CombatHandler
        return CombatHandler
    except (ImportError, AttributeError):
        from deimos_bridge.headless import install
        install()
        from wizwalker.combat import CombatHandler
        return CombatHandler


class WizAiFighter(_combat_handler_base()):
    """A Deimos combat handler that picks cards with a wizAi policy.

    Args:
        client: the wizwalker `Client` in combat.
        policy: a wizAi policy -- `QAgent.policy()`, or any
            `callable(sim, state) -> Card | Action | None`.
        cards: the wizAi card table the policy was trained against.
        decklist: the deck actually equipped, by wizAi card name.
        school: the wizard's school.
        boss: a wizAi `Boss` used only to construct the `Sim` that
            answers `can_cast`; its stats never reach a decision.
        pass_on_unknown: pass the round rather than improvise when the
            policy names a card the client does not offer.
    """

    def __init__(self, client, policy, cards, decklist, school, boss=None,
                 pass_on_unknown: bool = True):
        super().__init__(client)
        self.policy = policy
        self.cards = dict(cards)
        self.decklist = list(decklist)
        self.school = school
        self.pass_on_unknown = pass_on_unknown
        self._seen: list = []
        self._sim = self._make_sim(boss)

    def _make_sim(self, boss):
        """A `Sim` used purely as a rules oracle for `can_cast`.

        Real health comes from the client every round; this instance
        exists so legality is decided by wizAi's own rules rather than a
        second implementation of them.
        """
        from w101_sim import Boss

        boss = boss or Boss(name="live", hp=10 ** 6, school="fire", dmg=0)
        sim = Sim(dict(self.cards), self.decklist, self.school, boss,
                  player_hp=10 ** 6)
        return sim

    # ---- reading the client ------------------------------------------

    async def snapshot(self) -> LiveRound:
        """Flatten the current round out of the client."""
        members = await self.get_members()
        client_member = await self.get_client_member()
        client_owner = await client_member.owner_id()

        me, enemies, allies = None, [], []
        for member in members:
            flat = await self._flatten(member)
            if await member.owner_id() == client_owner:
                me = flat
            elif await member.is_monster():
                enemies.append(flat)
            else:
                allies.append(flat)

        hand = []
        for card in await self.get_cards():
            hand.append(await card.name())

        return LiveRound(client=me, enemies=enemies, allies=allies, hand=hand,
                         round_number=await self.round_number())

    async def _flatten(self, member) -> LiveMember:
        participant = await member.get_participant()
        effects = []
        for effect in await participant.hanging_effects():
            effects.append(LiveEffect(
                effect_type=(await effect.effect_type()).name,
                param=await effect.effect_param()))

        return LiveMember(
            name=await member.name(),
            school=self.school,        # the client exposes ids, not names
            health=await member.health(),
            max_health=await member.max_health(),
            level=await member.level(),
            is_client=await member.is_client(),
            is_player=await member.is_player(),
            normal_pips=await member.normal_pips(),
            power_pips=await member.power_pips(),
            shadow_pips=await member.shadow_pips(),
            stunned=await member.is_stunned(),
            effects=effects)

    # ---- one round ---------------------------------------------------

    async def handle_round(self):
        live = await self.snapshot()
        state = build_state(live, self.cards, self.decklist, self._seen)
        self._seen.extend(c.name for c in state.hand)

        name, target = choose(self.policy, self._sim, state)
        if name is None:
            log.info("round %s: policy passes", live.round_number)
            await self.pass_button()
            return

        card = await self._card_named(name)
        if card is None:
            log.warning("round %s: policy chose %r, not in hand",
                        live.round_number, name)
            if self.pass_on_unknown:
                await self.pass_button()
            return

        member = await self._target_member(target)
        log.info("round %s: casting %s at %s", live.round_number, name,
                 await member.name() if member else "no target")
        await card.cast(member)

    async def _card_named(self, name):
        for card in await self.get_cards():
            if await card.name() == name and await card.is_castable():
                return card
        return None

    async def _target_member(self, index: int):
        monsters = [m for m in await self.get_all_monster_members()
                    if not await m.is_dead()]
        if not monsters:
            return None
        return monsters[index] if index < len(monsters) else monsters[0]
