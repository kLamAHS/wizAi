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


#: Why a card name could not be turned into a wizAi `Card`. The
#: distinction is the whole point: one of these is a decoder gap with a
#: named cause and a known fix, the other is a card nobody has ever heard
#: of, and they need completely different responses.
MISS_DECODER = "decoder"     # in the game data; load_spells_full skipped it
MISS_UNKNOWN = "unknown"     # not in the game data under that name at all


def build_catalog(spells_path="spells_full.json", cards_path="cards_clean.json"):
    """Every name the game data knows, and why the loader dropped it.

    Costs one pass over `spells_full.json` (18k records), so it is opt-in
    -- but without it a miss is just a string, and with it a miss is
    "needs kSummonCreature" or "genuinely not a spell this build knows".
    """
    import json

    from data_full import load_spells_full

    report = {}
    cards = load_spells_full(spells_path, cards_path, report=report)
    reasons = dict(report.get("skipped") or [])

    raw = json.load(open(spells_path, encoding="utf-8"))
    known = {rec["name"] for rec in raw if rec.get("name")}

    # A langcode is *not* unique. "Spells_Fireblade" is shared by
    # Fireblade, Fireblade - EM, Fireblade - SIT, Fireblade - Tear,
    # FirebladeBOSS01, FirebladeBOSS02 and a raid sigil -- same display
    # text, different spells. Resolving on the code alone could hand the
    # policy a boss variant.
    #
    # `base_spell` settles it without guessing: every one of those points
    # at "Fireblade", and the canonical record is the one where
    # `name == base_spell`. That is the player-facing spell by
    # definition, so a group with exactly one canonical member has an
    # unambiguous answer. Groups that do not are dropped.
    groups = {}
    for rec in raw:
        if rec.get("variant") != "core":
            continue
        code = rec.get("display_name")
        if code:
            groups.setdefault(code, []).append(rec)

    langcodes, clashes = {}, set()
    for code, recs in groups.items():
        canonical = [r for r in recs if r["name"] == r.get("base_spell")]
        pick = (canonical[0] if len(canonical) == 1
                else recs[0] if len(recs) == 1 else None)
        card = cards.get(pick["name"]) if pick else None
        if card is not None:
            langcodes[code] = card
        else:
            clashes.add(code)

    # The player-facing spells. Same rule as the langcode groups: a
    # record whose `name` IS its `base_spell` is the real card, and
    # everything else pointing at it is an event/boss/test variant
    # (Iceblade - EM, Iceblade - SIT, IcebladeBOSS01, Wand Ice_Boss001).
    # The card table has to keep those -- a mob really can cast them --
    # but anything asking "what could I put in my deck" wants this set.
    canonical = {rec["name"] for rec in raw
                 if rec.get("name") and rec["name"] == rec.get("base_spell")}

    return {"cards": cards, "reasons": reasons, "known": known,
            "langcodes": langcodes, "ambiguous_langcodes": clashes,
            "canonical": canonical}


class NameResolver:
    """Game card name -> wizAi `Card`.

    Layers, in order, stopping at the first hit:
      1. exact key
      2. **langcode** (`Spells_Fireblade`), if the caller supplies one
      3. stale-name alias table
      4. normalized key, then normalized alias

    The langcode layer matters more than it looks. wizwalker's
    `CombatCard.name()` and this table's keys both come from the game
    data, so they usually agree -- but `display_name_code()` is the
    game's own stable identifier: it does not move when a spell is
    renamed and it is identical on a non-English client, where the
    localized display name is not. It is used only where a langcode maps
    to exactly one card; `build_catalog` drops the ambiguous ones rather
    than picking.

    Deliberately no fuzzy match, anywhere. Deimos ships `thefuzz` and
    uses it for UI convenience, but a wrong card here is a wrong *cast*
    in a real fight -- silently playing Fire Elf because Fire Dragon was
    not in the table is worse than passing.
    """

    def __init__(self, cards: dict, catalog: dict = None):
        self.cards = cards
        self._norm = {}
        for key, card in cards.items():
            self._norm.setdefault(_normal(key), card)
        self.misses = {}          # game name -> times seen
        catalog = catalog or {}
        self._reasons = catalog.get("reasons", {})
        self._known = catalog.get("known", set())
        self._langcodes = catalog.get("langcodes", {})

    # -- resolution -------------------------------------------------------
    def resolve(self, game_name: str, langcode: str = None,
                treasure: bool = False, item: bool = False):
        if not game_name and not langcode:
            return None
        suffix = "@tc" if treasure else ("@item" if item else "")

        aliased = ALIASES.get(game_name)
        for candidate in ((game_name + suffix) if game_name else None,
                          (aliased + suffix) if aliased else None):
            if candidate and candidate in self.cards:
                return self.cards[candidate]

        if langcode and langcode in self._langcodes:
            return self._langcodes[langcode]

        for base in (game_name, aliased):
            if not base:
                continue
            n = _normal(base + suffix)
            if n in self._norm:
                return self._norm[n]

        # A treasure/item copy the table only carries as a deck card is
        # still the right spell -- fall back rather than lose the action.
        if suffix:
            base = self.resolve(game_name, langcode)
            if base is not None:
                return base

        if game_name:
            self.misses[game_name] = self.misses.get(game_name, 0) + 1
        return None

    # -- triage -----------------------------------------------------------
    def classify(self, game_name: str):
        """(kind, detail) for a name that would not resolve.

        Without a catalog everything reads as unknown, which is honest --
        the resolver genuinely cannot tell the two apart without one.
        """
        reason = self._reasons.get(game_name)
        if reason:
            return MISS_DECODER, reason
        if game_name in self._known:
            return MISS_DECODER, "in the game data but not in the built table"
        return MISS_UNKNOWN, None

    def miss_report(self):
        """Misses, each with its classification, worst first."""
        out = []
        for name, n in sorted(self.misses.items(), key=lambda kv: -kv[1]):
            kind, detail = self.classify(name)
            out.append({"name": name, "count": n, "kind": kind,
                        "detail": detail or ""})
        return out

    def report(self) -> str:
        if not self.misses:
            return "all card names resolved"
        rows = self.miss_report()
        decoder = [r for r in rows if r["kind"] == MISS_DECODER]
        unknown = [r for r in rows if r["kind"] == MISS_UNKNOWN]
        lines = [f"{len(rows)} unresolved card names "
                 f"(the policy never saw these):"]
        if decoder:
            lines.append("  the decoder skipped these -- close the gap in "
                         "data_full._map_effect to get them:")
            for r in decoder:
                lines.append(f"    {r['count']:>4}x  {r['name']!r}  --  {r['detail']}")
        if unknown:
            lines.append("  not in the game data under this name -- check "
                         "spelling, or add to deimos_bridge.live_state.ALIASES:")
            for r in unknown:
                lines.append(f"    {r['count']:>4}x  {r['name']!r}")
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
#: Scheduled damage and healing, hanging on the target. `effect_param`
#: is the REMAINING total -- Deimos's own `reduce_over_time` subtracts
#: from it and `detonate` pops it whole -- so per-tick is param over the
#: rounds left. Dropping these was the reader's costliest gap: a mob
#: carrying a Fire Elf's 157 scheduled damage looked identical to a
#: healthy one, so the policy would spend a whole hit re-killing
#: something that was already dead in two ticks -- the exact "it does
#: not understand DoT damage" failure, caused not by the model but by
#: the model never being told.
OVER_TIME_DOT = {"damage_over_time", "deferred_damage"}
OVER_TIME_HOT = {"heal_over_time"}
#: Accuracy charms -- mantles when negative, aims when positive. The sim
#: has always modelled them (`Hanging(kind="accuracy")` feeds the fizzle
#: roll); the reader silently dropped them, so a Black Mantle's -45% on
#: the wizard's next cast was invisible to every prediction.
OUTGOING_ACCURACY = {"modify_accuracy", "modify_outgoing_accuracy"}

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

    A failed read leaves `read_hangings.last_error` set. An empty list is
    otherwise the same answer for "this mob has no shields" and "the
    effect list would not read", and the second is a materially different
    fight: the policy prices its hit against a bare mob, walks it into a
    50% Tower Shield, and the residual is filed as a clean observation
    the damage model got wrong by half.
    """
    read_hangings.last_error = ""
    out = []
    try:
        participant = await member.get_participant()
        effects = list(await participant.hanging_effects())
        try:
            effects += list(await participant.aura_effects())
        except Exception:
            pass
    except Exception as exc:
        read_hangings.last_error = f"{type(exc).__name__}: {exc}"
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
            tid = None

        # The STACKING identity is the effect's shape, not its list
        # position. The old fallback (`sub=i`, the loop index) gave two
        # copies of the same Fireblade DIFFERENT stack keys, so the
        # rollout consumed both on a single hit -- a live trace priced
        # a Fire Cat at 100 x 1.35 x 1.35 = 182, and three rounds of
        # blade-spam followed, because every extra blade looked
        # multiplicative when the game applies one per hit
        # (`_consume_damage_charms`, one per stack_key -- Deimos's own
        # dedupe rule). Same-shaped charms now share a key whether or
        # not the template id reads; two genuinely different blades
        # (different percent or school) still stack.
        ident = f"{kind_name}:{dtype}:{param:g}"
        shown = tid if tid is not None else ident

        if slot == "charm" and kind_name in OUTGOING_DAMAGE:
            out.append(Hanging(name=f"live:{shown}", slot="charm",
                               kind="damage", percent=param / 100.0,
                               schools=schools, source="live", sub=ident))
        elif slot == "ward" and kind_name in INCOMING_DAMAGE:
            out.append(Hanging(name=f"live:{shown}", slot="ward",
                               kind="damage", percent=param / 100.0,
                               schools=schools, source="live", sub=ident))
        elif slot == "ward" and kind_name in INCOMING_PRISM:
            out.append(Hanging(name=f"live:{shown}", slot="ward",
                               kind="prism", schools=schools,
                               convert_to=_SCHOOL_BY_ID.get(int(param), None),
                               source="live", sub=ident))
        elif slot == "ward" and kind_name in INCOMING_ABSORB:
            # absorbs are separate POOLS -- two absorbs both soak, so
            # they keep per-instance identity
            out.append(Hanging(name=f"live:{shown}", slot="ward",
                               kind="absorb", amount=param, source="live",
                               sub=tid if tid is not None else i))
        elif slot == "charm" and kind_name in OUTGOING_ACCURACY:
            out.append(Hanging(name=f"live:{shown}", slot="charm",
                               kind="accuracy", percent=param / 100.0,
                               schools=schools, source="live", sub=ident))
        elif slot == "ward" and kind_name in (OVER_TIME_DOT | OVER_TIME_HOT):
            from w101_sim import OverTime

            try:
                rounds = max(1, int(await e.num_rounds()))
            except Exception:
                rounds = 3
            school = _SCHOOL_BY_ID.get(dtype, "fire")
            out.append(OverTime(
                name=f"live:{shown}",
                kind="dot" if kind_name in OVER_TIME_DOT else "hot",
                school=school, per_tick=abs(param) / rounds,
                rounds_left=rounds, caster="live"))
    return out


# --------------------------------------------------------------------------
# the read
# --------------------------------------------------------------------------
async def _team_id(member):
    """`CombatParticipant.team_id` (offset 144), or None if unreadable."""
    try:
        return int(await (await member.get_participant()).team_id())
    except Exception:
        return None


async def _is_hostile(member, my_team) -> bool:
    """Is this member on the other side?

    Not `is_monster()`. wizwalker derives that as "not a player and not a
    minion" (`combat/member.py:84-88`), which is wrong for exactly the
    case that matters most here: an **enemy minion** is `is_player=False,
    is_minion=True`, so `is_monster()` returns False and the summon would
    be filed as an ally. wizAi's weak spot is multi-enemy target
    selection, so handing it a board with the summons missing would be a
    quiet way to flatter the policy.

    `team_id` is the real discriminator. It is only a fallback to
    `is_monster()` when the participant will not read.
    """
    team = await _team_id(member)
    if team is not None and my_team is not None:
        return team != my_team
    return bool(await member.is_monster())


class LiveRead:
    """One planning-phase snapshot, plus the bookkeeping the policy needs
    that the client does not directly expose."""

    def __init__(self, state, hand_cards, resolver, members, client_member,
                 round_number, enemy_members=(), hidden=(), ally_members=()):
        self.state = state
        #: wizAi card name -> the live CombatCards that produced it.
        #: Populated only for cards that both resolved *and* are castable.
        self.hand_cards = hand_cards
        self.resolver = resolver
        self.members = members
        self.client_member = client_member
        self.round_number = round_number
        #: the school the CLIENT says this wizard is, or "" if it would
        #: not read. Deliberately not written into `state.player.school`
        #: -- that stays whatever the run was configured with, so nothing
        #: downstream shifts under a policy mid-fight. It exists so the
        #: configuration can be checked against the game, because
        #: `ClientHandler.get_new_clients()` hands back windows in
        #: whatever order it finds them and nothing else can tell which
        #: wizard is which. See `WizAiBackend._check_school`.
        self.client_school = ""
        #: The live members behind `state.enemies`, **in the same order**.
        #: This is what makes a policy's "name@i" mean the same mob to the
        #: game as it did to the policy. Deriving the target from a
        #: separately-built enemy list would drift the moment something
        #: dies, and would fail silently as bad targeting.
        self.enemy_members = list(enemy_members)
        #: The live members behind `state.allies` -- this wizard's OWN
        #: team, minus itself. `members` is every participant in the
        #: duel, both sides, so picking "an ally" out of it lands on a
        #: mob: `get_members()` returns the enemy team first, and in a
        #: solo fight there is nothing else in it at all. A friend-only
        #: heal aimed at a mob does not cast -- the second click
        #: deselects the card -- so the round silently does nothing and
        #: the duel stalls until the timer runs out. See
        #: `WizAiBackend._resolve_target`.
        self.ally_members = list(ally_members)
        #: Castable cards in hand this round that could NOT be turned into
        #: a wizAi Card. The policy did not see them. A run with a high
        #: `hand_visibility` deficit is not measuring the policy.
        self.hidden = list(hidden)
        #: Effect lists that would not read, as human-readable strings.
        #: An empty ward list is the same value for "no shields" and "the
        #: read failed", and the second hands the policy a materially
        #: different fight -- so the difference is carried here rather
        #: than being lost in an empty list.
        self.unreadable = []

    @property
    def hand_visibility(self) -> float:
        """Fraction of the castable hand the policy could actually see."""
        total = sum(len(v) for v in self.hand_cards.values()) + len(self.hidden)
        return 1.0 if not total else \
            (total - len(self.hidden)) / total


async def read_school(member, default="balance") -> str:
    """A participant's school, off the client.

    This used to be the literal `"ice"` for every enemy on the board, and
    that literal is expensive in a way that is invisible until you look
    at the arithmetic. `Boss.resist_own` is 0.40, so a mob modelled as
    the wizard's own school resists 40% of every hit -- the worst matchup
    in the game. An ice wizard was therefore planning every fight against
    mobs that were, as far as the simulator knew, ice.

    Falls back to balance rather than to anything school-shaped: balance
    is neutral in both directions, so an unreadable school costs
    accuracy, not a fabricated resistance.
    """
    from .deimos_damage import SCHOOL_TO_STR

    try:
        sid = int(await member.primary_magic_school_id())
    except Exception:
        return default
    name = SCHOOL_TO_STR.get(sid)
    if name is None:
        return default
    name = name.lower()
    # Star/Sun/Moon/Shadow are real school ids that no mob fights as.
    return name if name in _SCHOOL_SLOT else default


#: `GameStats`'s by-school vectors are indexed by Deimos's
#: `school_list_ids` -- the same 0..15 ordering `deimos_damage` already
#: ports. Only the seven playable schools matter here.
_SCHOOL_SLOT = {"fire": 0, "ice": 1, "storm": 2, "myth": 3, "life": 4,
                "death": 5, "balance": 6}


def _stat_for(vector, school, universal=0.0):
    """One school's entry out of a by-school stat vector, plus the
    'all schools' scalar the game keeps separately.

    Deimos does the same addition in `combat_math.real_stat` -- a stat
    lives in two places and reading only the vector silently drops
    everything granted as universal, which on low-level gear is most of
    it.
    """
    slot = _SCHOOL_SLOT.get(school)
    base = 0.0
    if slot is not None and vector and slot < len(vector):
        try:
            base = float(vector[slot])
        except (TypeError, ValueError):
            base = 0.0
    return base + float(universal or 0.0)


async def read_player_stats(client, school: str) -> dict:
    """The wizard's real gear, as `Sim(player_stats=...)` wants it.

    Without this the simulator models a wizard with **no gear at all**:
    zero damage bonus, zero pierce, base accuracy. Every hit is then
    priced below what it really lands for, so the policy is solving a
    different fight than the one being played -- and it changes real
    decisions. Measured on a 2000hp mob with an ice deck, `greedy_ttk`
    opens with a trap 100% of the time given 9% damage and 4% pierce, and
    0% of the time given nothing; on a 400hp mob both open with the hit.
    Which way the error goes depends on the board, so there is no safe
    direction to guess in.

    Returns `{}` if the stats will not read; the caller then gets the old
    no-gear behaviour rather than a crash, and can say so.

    A stat that fails to read is recorded under `"unread"` rather than
    folded into 0.0. They were the same value, so a wizard in full gear
    whose `dmg_bonus_percent` raised was told "read your gear: 0% ice
    damage" -- a confident number, indistinguishable from a true zero,
    that then priced every hit in training below what it lands for.
    """
    stats = getattr(client, "stats", None)
    if stats is None:
        return {}

    unread = []

    async def vec(name):
        try:
            return await getattr(stats, name)()
        except Exception:
            unread.append(name)
            return []

    async def one(name):
        try:
            return float(await getattr(stats, name)())
        except Exception:
            unread.append(name)
            return 0.0

    school = (school or "").lower()
    dmg = _stat_for(await vec("dmg_bonus_percent"), school,
                    await one("dmg_bonus_percent_all"))
    acc = _stat_for(await vec("acc_bonus_percent"), school,
                    await one("acc_bonus_percent_all"))
    pierce = _stat_for(await vec("ap_bonus_percent"), school,
                       await one("ap_bonus_percent_all"))
    resist = _stat_for(await vec("dmg_reduce_percent"), school,
                       await one("dmg_reduce_percent_all"))
    crit = await one("critical_hit_percent_all")
    block = await one("block_percent_all")
    # Power-pip odds. Not gear in the damage sense, but it decides WHEN
    # a card becomes affordable, and the live path was handing the
    # policy a wizard with none -- so a two-pip Snow Serpent was scored
    # as a turn-two card and a three-pip Snowman as turn three.
    pip = await one("power_pip_base") + await one("power_pip_bonus_percent_all")

    out = {}
    if dmg:
        out["damage"] = {school: dmg}
    if acc:
        out["accuracy"] = acc
    if pierce:
        out["pierce"] = pierce
    if resist:
        out["resist"] = {"*": resist}
    # The game reports crit and block as ratings on some builds and
    # percentages on others; anything above 1.0 is a rating, and feeding
    # a rating in as a probability would make every cast a critical.
    if 0.0 < crit <= 1.0:
        out["crit"] = crit
    if 0.0 < block <= 1.0:
        out["block"] = block
    if 0.0 < pip <= 1.0:
        out["power_pip_chance"] = pip
    if unread and out:
        # Only when something else read: an entirely failed read already
        # returns {} and is reported by the caller as such. This is the
        # partial case, which looked like a complete one.
        out["unread"] = sorted(set(unread))
    return out


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

    async def _school_rack(m):
        """Archmastery pips, {school: count}, or {} when unreadable.

        Read defensively on purpose: mocks and older builds have no
        `school_pips`, and a rack that cannot be read must degrade to
        the pre-archmastery arithmetic, not to an exception that costs
        the whole board read.
        """
        reader = getattr(m, "school_pips", None)
        if reader is None:
            return {}
        try:
            rack = await reader()
        except Exception:
            return {}
        return {k: int(v) for k, v in dict(rack or {}).items() if int(v) > 0}

    async def _mk_actor(m, team):
        who = school if team == 0 else await read_school(m)
        # Archmastery pips are the third rack, and the one this read
        # spent a fight ignoring: a max-level storm wizard whose rack
        # had converted to storm pips scored 0 normal + 0 power for 22
        # consecutive rounds while the game showed every card castable,
        # so the policy could only "afford" its zero-pip buffs and then
        # passed the rest of the fight. Own-school pips go to the
        # sim's `school_pips` (worth 2 there, nothing elsewhere --
        # `Ruleset.afford`); pips of any OTHER school spend like a
        # white pip for what this wizard actually casts, so they fold
        # into norm_pips.
        rack = await _school_rack(m)
        own_school_pips = int(rack.pop(who, 0))
        return Actor(
            name=await m.name(),
            # Read, not assumed. `"ice"` was hardcoded here for every
            # enemy, and with `Boss.resist_own = 0.40` that told the
            # simulator each mob resisted 40% of an ice wizard's damage
            # -- so the rollout planned every fight against the worst
            # matchup in the game and concluded nothing could be killed.
            school=who,
            hp=float(await m.health()),
            max_hp=float(await m.max_health()) or 1.0,
            team=team,
            # Enemy pips used to be zeroed here -- a real observation
            # thrown away. The client reports every member's rack, and
            # "the boss has six pips" is the difference between shielding
            # this round and shielding after the Wraith lands: with the
            # catalog spell pool on the enemy (live_backend._apply_pool)
            # the rollout knows exactly which casts are legal RIGHT NOW.
            norm_pips=int(await m.normal_pips()) + sum(rack.values()),
            pow_pips=int(await m.power_pips()),
            school_pips=own_school_pips,
        )

    unreadable = []

    async def _hangings(m, slot, who, actor=None):
        """Hangings for `slot`; DoTs/HoTs land on `actor.over_time`.

        Split by type because they live on different fields of the
        actor: a trap is consulted when a hit resolves, a DoT ticks in
        `end_round` whether anyone acts or not.
        """
        from w101_sim import OverTime

        effects = await read_hangings(m, slot)
        if read_hangings.last_error:
            unreadable.append(f"{who}'s {slot}s "
                              f"({read_hangings.last_error})")
        scheduled = [e for e in effects if isinstance(e, OverTime)]
        if actor is not None and scheduled:
            actor.over_time.extend(scheduled)
        return [e for e in effects if not isinstance(e, OverTime)]

    player = await _mk_actor(me, 0)
    player.charms = await _hangings(me, "charm", "you", actor=player)
    player.wards = await _hangings(me, "ward", "you", actor=player)

    enemies, allies, enemy_members, ally_members = [], [], [], []
    me_name = await me.name()
    my_team = await _team_id(me)
    for m in members:
        if await m.is_dead():
            continue
        if await m.name() == me_name and await m.is_client():
            continue
        hostile = await _is_hostile(m, my_team)
        actor = await _mk_actor(m, 1 if hostile else 0)
        actor.charms = await _hangings(m, "charm", actor.name, actor=actor)
        actor.wards = await _hangings(m, "ward", actor.name, actor=actor)
        actor.is_minion = bool(await m.is_minion())
        if hostile:
            enemies.append(actor)
            enemy_members.append(m)      # kept index-aligned with `enemies`
        else:
            allies.append(actor)
            ally_members.append(m)       # kept index-aligned with `allies`

    hand, hand_cards, hidden = [], {}, []
    for c in await combat.get_cards():
        try:
            if not await c.is_castable():
                continue
        except Exception:
            pass
        game_name = await c.name()
        try:
            langcode = await c.display_name_code()
        except Exception:
            langcode = None       # older wizwalker, or a card mid-read
        card = resolver.resolve(
            game_name, langcode,
            treasure=bool(await c.is_treasure_card()),
            item=bool(await c.is_item_card()),
        )
        if card is None:
            # The card stays out of the policy's hand -- there is no wizAi
            # Card to reason about -- but it is *recorded*. Dropping it
            # silently is how a run quietly becomes meaningless: the
            # policy plans a five-card hand while holding seven, and its
            # scarcity feature counts the wrong number of nukes left.
            hidden.append(game_name)
            continue
        hand.append(card)
        hand_cards.setdefault(card.name, []).append(c)

    player.hand = hand
    player.deck = list(deck_remaining or [])

    state = State(player, enemies or [Actor(name="none", school="ice", hp=0,
                                            max_hp=1, team=1)], allies)
    read = LiveRead(state, hand_cards, resolver, members, me,
                    await combat.round_number(), enemy_members, hidden,
                    ally_members)
    read.unreadable = unreadable
    # The client's own answer, for checking the configuration against.
    # Empty rather than "balance" on a failed read: an unreadable school
    # must not be reported as a mismatch with the configured one.
    read.client_school = await read_school(me, default="")
    return read
