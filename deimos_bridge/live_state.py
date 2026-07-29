"""Live wizwalker combat -> a wizAi `State` the policy can reason about.

Everything here is duck-typed against the wizwalker async API rather than
imported from it, so the identical code path runs against `mock_client`
on Linux and against the real client on Windows. The methods used are:

    CombatMember   is_client() is_player() is_monster() is_boss()
                   is_dead() name() health() max_health() level()
                   normal_pips() power_pips() shadow_pips()
                   get_participant() -> .hanging_effects(), .aura_effects()
    CombatCard     name() display_name() is_castable() is_treasure_card()
                   is_item_card() is_enchanted()
    CombatHandler  get_members() get_cards() round_number()

The hard part is not reading the state, it is **naming**. wizAi's card
table is keyed by exact wiki name (`data_full.load_spells_full`, which
does `cards[key]` with `key = s["name"]`), and the game does not always
agree with the wiki -- wizAi's own README records that the wiki's
"Elemental Blade" is the game's "Tri Blade". A miss is silent and
catastrophic: the policy simply never sees that card. So resolution is
explicit, layered, and *reports* what it could not resolve.
"""
import re
import unicodedata

from w101_sim import Actor, Hanging, State

#: Stale name -> the name wizAi's card table actually uses.
#:
#: `load_spells_full` keys on the *game data* dump, so it already holds
#: in-game names: "Tri Blade" and "Tri Trap" are present, and the wiki's
#: "Elemental Blade"/"Elemental Trap" are not (wizAi's README records
#: exactly this rename). A live client therefore resolves on the exact
#: path and never reaches this table. It exists for the other direction:
#: a hand-written decklist carrying the old wiki names still works.
#:
#: Verified against the built table rather than assumed -- see
#: `tests/test_deimos_bridge.py::test_aliases_point_at_real_cards`, which
#: fails if a key here is already a real card or a value is not.
ALIASES = {
    "Elemental Blade": "Tri Blade",
    "Elemental Trap": "Tri Trap",
}

#: wizAi card name -> the name to type in the game. The card table is
#: already game-named, so this is the identity for everything except a
#: decklist written against the old wiki names.
WIKI_TO_GAME = dict(ALIASES)


def _normal(name: str) -> str:
    """Fold the differences that are never meaningful: case, accents,
    punctuation, and runs of whitespace. 'Krokopatra's Curse' and
    'krokopatras curse' land on the same key."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    # Apostrophes are dropped, not turned into separators: the game and
    # the wiki disagree about them ("Krokopatra's Curse" / "Krokopatras
    # Curse"), and splitting on one would leave a stray "s" token that
    # matches neither spelling.
    s = re.sub(r"[’']", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


class NameResolver:
    """Game card name -> wizAi `Card`.

    Layers, in order, stopping at the first hit:
      1. exact key
      2. game-name alias table
      3. normalized key
      4. normalized alias

    Deliberately no fuzzy match. Deimos ships `thefuzz` and uses it for
    UI convenience, but a wrong card here is a wrong *cast* in a real
    fight -- silently playing Fire Elf because Fire Dragon was not in the
    table is worse than passing. Misses are recorded instead.
    """

    def __init__(self, cards: dict):
        self.cards = cards
        self._norm = {}
        for key, card in cards.items():
            self._norm.setdefault(_normal(key), card)
        self.misses = {}          # game name -> times seen

    def resolve(self, game_name: str, treasure: bool = False,
                item: bool = False):
        if not game_name:
            return None
        suffix = "@tc" if treasure else ("@item" if item else "")

        aliased = ALIASES.get(game_name)
        for candidate in (game_name + suffix,
                          (aliased + suffix) if aliased else None):
            if candidate and candidate in self.cards:
                return self.cards[candidate]

        for base in (game_name, aliased):
            if not base:
                continue
            n = _normal(base + suffix)
            if n in self._norm:
                return self._norm[n]

        # A treasure/item copy the table only carries as a deck card is
        # still the right spell -- fall back rather than lose the action.
        if suffix:
            base = self.resolve(game_name)
            if base is not None:
                return base

        self.misses[game_name] = self.misses.get(game_name, 0) + 1
        return None

    def report(self) -> str:
        if not self.misses:
            return "all card names resolved"
        lines = ["unresolved card names (the policy never saw these):"]
        for name, n in sorted(self.misses.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {n:>4}x  {name!r}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# hanging effects
# --------------------------------------------------------------------------
#: wizwalker `SpellEffects` member names that wizAi models as a damage
#: charm (outgoing) or ward (incoming).
OUTGOING_DAMAGE = {"modify_outgoing_damage"}
INCOMING_DAMAGE = {"modify_incoming_damage"}
INCOMING_PRISM = {"modify_incoming_damage_type"}
INCOMING_ABSORB = {"absorb_damage"}

_SCHOOL_BY_ID = {
    2343174: "fire", 72777: "ice", 83375795: "storm", 2448141: "myth",
    2330892: "life", 78318724: "death", 1027491821: "balance",
}
UNIVERSAL_DAMAGE_TYPE = 80289


async def _effect_name(effect) -> str:
    for attr in ("effect_type",):
        try:
            v = await getattr(effect, attr)()
            return getattr(v, "name", str(v))
        except Exception:
            continue
    return ""


async def read_hangings(member, slot: str) -> list:
    """Read a participant's hanging effects into wizAi `Hanging`s.

    `slot` is 'charm' for the caster's outgoing effects and 'ward' for a
    target's incoming ones -- the same split wizAi's `Actor.charms` /
    `Actor.wards` uses.
    """
    out = []
    try:
        participant = await member.get_participant()
        effects = list(await participant.hanging_effects())
        try:
            effects += list(await participant.aura_effects())
        except Exception:
            pass
    except Exception:
        return out

    for i, e in enumerate(effects):
        kind_name = await _effect_name(e)
        try:
            param = float(await e.effect_param())
            dtype = int(await e.damage_type())
        except Exception:
            continue
        schools = (None if dtype == UNIVERSAL_DAMAGE_TYPE
                   else ({_SCHOOL_BY_ID[dtype]} if dtype in _SCHOOL_BY_ID else None))
        try:
            tid = int(await e.spell_template_id())
        except Exception:
            tid = i

        if slot == "charm" and kind_name in OUTGOING_DAMAGE:
            out.append(Hanging(name=f"live:{tid}", slot="charm", kind="damage",
                               percent=param / 100.0, schools=schools,
                               source="live", sub=tid))
        elif slot == "ward" and kind_name in INCOMING_DAMAGE:
            out.append(Hanging(name=f"live:{tid}", slot="ward", kind="damage",
                               percent=param / 100.0, schools=schools,
                               source="live", sub=tid))
        elif slot == "ward" and kind_name in INCOMING_PRISM:
            out.append(Hanging(name=f"live:{tid}", slot="ward", kind="prism",
                               schools=schools,
                               convert_to=_SCHOOL_BY_ID.get(int(param), None),
                               source="live", sub=tid))
        elif slot == "ward" and kind_name in INCOMING_ABSORB:
            out.append(Hanging(name=f"live:{tid}", slot="ward", kind="absorb",
                               amount=param, source="live", sub=tid))
    return out


# --------------------------------------------------------------------------
# the read
# --------------------------------------------------------------------------
class LiveRead:
    """One planning-phase snapshot, plus the bookkeeping the policy needs
    that the client does not directly expose."""

    def __init__(self, state, hand_cards, resolver, members, client_member,
                 round_number):
        self.state = state
        #: wizAi Card -> the live CombatCard to click. Populated only for
        #: cards that both resolved *and* are castable.
        self.hand_cards = hand_cards
        self.resolver = resolver
        self.members = members
        self.client_member = client_member
        self.round_number = round_number


async def read_state(combat, resolver: NameResolver, school: str,
                     deck_remaining=None) -> LiveRead:
    """Build a wizAi `State` from a live combat.

    `deck_remaining` is the caller's running list of wizAi `Card`s still
    undrawn. The client cannot report the undrawn deck -- only the deck's
    *composition* is readable, and the order is hidden by design. wizAi's
    RL featurizer uses it solely for a scarcity count
    (`Featurizer.key`, rl_agent.py:52-54), so an approximate list from
    the configured decklist minus what has been seen is faithful enough
    for the feature and is what `live_backend` tracks.
    """
    members = list(await combat.get_members())
    me = await combat.get_client_member()

    async def _mk_actor(m, team):
        return Actor(
            name=await m.name(),
            school=school if team == 0 else "ice",
            hp=float(await m.health()),
            max_hp=float(await m.max_health()) or 1.0,
            team=team,
            norm_pips=int(await m.normal_pips()) if team == 0 else 0,
            pow_pips=int(await m.power_pips()) if team == 0 else 0,
        )

    player = await _mk_actor(me, 0)
    player.charms = await read_hangings(me, "charm")
    player.wards = await read_hangings(me, "ward")

    enemies, allies = [], []
    me_name = await me.name()
    for m in members:
        if await m.is_dead():
            continue
        if await m.name() == me_name and await m.is_client():
            continue
        actor = await _mk_actor(m, 1 if await m.is_monster() else 0)
        actor.charms = await read_hangings(m, "charm")
        actor.wards = await read_hangings(m, "ward")
        (enemies if await m.is_monster() else allies).append(actor)

    hand, hand_cards = [], {}
    for c in await combat.get_cards():
        try:
            if not await c.is_castable():
                continue
        except Exception:
            pass
        game_name = await c.name()
        card = resolver.resolve(
            game_name,
            treasure=bool(await c.is_treasure_card()),
            item=bool(await c.is_item_card()),
        )
        if card is None:
            continue
        hand.append(card)
        hand_cards.setdefault(card.name, []).append(c)

    player.hand = hand
    player.deck = list(deck_remaining or [])

    state = State(player, enemies or [Actor(name="none", school="ice", hp=0,
                                            max_hp=1, team=1)], allies)
    return LiveRead(state, hand_cards, resolver, members, me,
                    await combat.round_number())
