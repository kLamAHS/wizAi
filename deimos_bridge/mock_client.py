"""Fakes for the wizwalker combat objects.

`live_state` and `live_backend` are the two modules that cannot be run
against the real thing here -- wizwalker's `constants.py` does
`ctypes.windll.user32` at import time, so it will not even load off
Windows, let alone without the game. Everything they touch is async
duck-typed, so these stand-ins let both modules be exercised and tested
for real on any machine.

The point is not to simulate Wizard101 -- wizAi already does that. It is
to check the *plumbing*: that a live read produces a sane wizAi `State`,
that names resolve, that a policy's answer maps back to a card the
handler can actually click.
"""
from dataclasses import dataclass, field


class _Awaitable:
    """Turns a plain value into `await obj.thing()`."""
    def __init__(self, value):
        self.value = value

    async def __call__(self):
        return self.value


def _attrs(obj, mapping):
    for name, value in mapping.items():
        setattr(obj, name, _Awaitable(value))


@dataclass
class MockEffect:
    effect_type_name: str
    effect_param: float = 0.0
    damage_type: int = 80289
    spell_template_id: int = 0
    num_rounds: int = 3

    def __post_init__(self):
        class _Enum:
            name = self.effect_type_name
        _attrs(self, {
            "effect_type": _Enum(),
            "effect_param": self.effect_param,
            "damage_type": self.damage_type,
            "spell_template_id": self.spell_template_id,
            "num_rounds": self.num_rounds,
        })


class MockParticipant:
    def __init__(self, hangings=None, auras=None, team_id=0):
        _attrs(self, {"hanging_effects": list(hangings or []),
                      "aura_effects": list(auras or []),
                      "team_id": team_id})


class MockMember:
    def __init__(self, name, health, max_health=None, *, monster=False,
                 boss=False, client=False, dead=False, level=1,
                 normal_pips=0, power_pips=0, shadow_pips=0, hangings=None,
                 minion=False, team_id=None, school_id=None,
                 school_pips=None):
        # Default the team from the side, so existing callers keep working;
        # pass team_id explicitly to build a minion on either side.
        if team_id is None:
            team_id = 1 if monster else 0
        self._participant = MockParticipant(hangings, team_id=team_id)
        _attrs(self, {
            "name": name,
            "health": health,
            "max_health": max_health if max_health is not None else health,
            # wizwalker derives this as "not player and not minion"
            # (combat/member.py:84-88) -- reproduced exactly, so a test can
            # catch code that trusts it for a minion.
            "is_monster": (not (not monster)) and not minion,
            "is_boss": boss,
            "is_client": client,
            "is_player": not monster,
            "is_minion": minion,
            "is_dead": dead,
            "is_stunned": False,
            "level": level,
            "normal_pips": normal_pips,
            "power_pips": power_pips,
            "shadow_pips": shadow_pips,
            "owner_id": abs(hash(name)) % 10 ** 8,
            "get_participant": self._participant,
        })
        # Only when asked for. `read_school` treats a missing accessor as
        # an unreadable school, which is a real state worth being able to
        # reproduce -- so it stays the default.
        if school_id is not None:
            _attrs(self, {"primary_magic_school_id": school_id})
        # Same rule for the archmastery rack: absent by default, because
        # a member without the accessor (an older build, a mock) is a
        # real state the reader has to survive.
        if school_pips is not None:
            _attrs(self, {"school_pips": dict(school_pips)})


class MockCard:
    """Records what it was asked to do, so a test can assert on the cast."""
    def __init__(self, name, *, castable=True, treasure=False, item=False,
                 enchanted=False, langcode=None):
        self.cast_log = []
        _attrs(self, {
            "name": name,
            "display_name": name,
            # the game's stable identifier, e.g. "Spells_Fireblade"
            "display_name_code": langcode or f"Spells_{name.replace(' ', '')}",
            "is_castable": castable,
            "is_treasure_card": treasure,
            "is_item_card": item,
            "is_enchanted": enchanted,
            "is_side_board": False,
        })

    #: the combat this card is in, so casting can leave the hand the way
    #: the game does. Set by `MockCombat`; None for a card on its own.
    combat = None
    #: set to make a cast go through the motions and NOT take -- which
    #: is what Wizard101 does when a card is clicked at something it
    #: cannot be cast on: it deselects, and the card comes back. It
    #: raises nothing, which is why it needs modelling rather than
    #: assuming every `cast` worked.
    misfires = False

    async def cast(self, target=None, **kw):
        self.cast_log.append(target)
        if not self.misfires and self.combat is not None:
            self.combat.discard(self)
        return True


class MockMouseHandler:
    """`client.mouse_handler`, which is an async context manager.

    Entering it is what activates wizwalker's mouseless cursor hook, and
    every cast is a mouse click, so a code path that forgets the `async
    with` does nothing at all in a real fight. `entered` records that it
    was used so a test can assert on it.
    """

    def __init__(self):
        self.entered = 0
        self.depth = 0

    async def __aenter__(self):
        self.entered += 1
        self.depth += 1
        return self

    async def __aexit__(self, *exc):
        self.depth -= 1
        return False


class MockClient:
    def __init__(self):
        self.mouse_handler = MockMouseHandler()


class MockCombat:
    """Stands in for `CombatHandler` / `SprintyCombat`."""
    def __init__(self, members, cards, round_number=1, client_index=0):
        self._members = list(members)
        #: every card the fight was built with, cast or not, so a test
        #: can still read `_cards[0].cast_log` after it has been played
        self._cards = list(cards)
        #: what is actually in hand right now. Separate from `_cards`
        #: because casting takes a card OUT of the hand, and that is the
        #: only signal a cast went through -- see `discard`.
        self._hand = list(cards)
        self._client = self._members[client_index]
        self.passed = 0
        for c in self._cards:
            c.combat = self
        _attrs(self, {
            "get_members": self._members,
            "get_cards": self._hand,
            "get_client_member": self._client,
            "round_number": round_number,
            "in_combat": True,
        })

    def discard(self, card):
        """Take a cast card out of the hand, as the game does.

        Not decoration: a card leaving the hand is the only signal that
        a cast actually went through. `CombatCard.cast` clicks and
        returns, raising nothing when the click does not take, so a mock
        whose hand never changed made every cast look successful and
        `WizAiCombatHandler._card_left_the_hand` untestable.

        Mutates `_hand`, which is the list `_attrs` closed over --
        `get_cards` is an instance attribute wrapping *that* object, so
        removing from it is what the handler's next read sees. `_cards`
        keeps every card so a test can still assert on what was played.
        """
        if card in self._hand:
            self._hand.remove(card)

    async def pass_button(self):
        self.passed += 1


def simple_fight(player_hp=3000, enemy_hp=2000, hand=("Fireblade", "Sunbird"),
                 pips=4, power_pips=0, enemy_hangings=None):
    """A one-enemy planning phase: the shape almost every test wants."""
    me = MockMember("Wizard", player_hp, client=True, normal_pips=pips,
                    power_pips=power_pips)
    foe = MockMember("Lost Soul", enemy_hp, monster=True,
                     hangings=enemy_hangings)
    cards = [MockCard(n) for n in hand]
    return MockCombat([me, foe], cards)
