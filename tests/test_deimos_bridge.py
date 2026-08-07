"""Tests for the wizAi <-> Deimos bridge.

Two things are being guarded here.

The **oracle** tests pin the Deimos damage port to the arithmetic in
`Deimos/src/combat_math.py`. If someone edits the port to make a
divergence go away, these fail -- which is the point, because a port that
drifts toward wizAi stops being an independent check.

The **plumbing** tests drive `live_state` and `live_backend` against
`mock_client`. Neither can be exercised against the real wizwalker off
Windows, so the mocks are the only way these code paths are covered at
all before they are pointed at a live fight.
"""
import asyncio

import pytest

from deimos_bridge import deimos_damage as dd
from deimos_bridge.differential import (EXPECTED, TOL, compare,
                                         legacy_ruleset)
from deimos_bridge.live_backend import WizAiBackend
from deimos_bridge.live_state import ALIASES, NameResolver, _normal, read_state
from deimos_bridge.mock_client import (MockCard, MockCombat, MockEffect,
                                       MockMember, simple_fight)
from deimos_bridge.scenarios import Buff, Scenario, Side, suite


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- the port
def test_port_reproduces_hand_computed_values():
    """Worked by hand from combat_math.py, so the port cannot drift."""
    fire = dd.school_id("fire")
    caster = dd.Stats(is_player=True)
    target = dd.Stats()

    # bare hit
    assert dd.deimos_damage(500, fire, caster, target, [], []) == 500

    # one 35 blade: 500 * 1.35
    blade = dd.Effect(dd.SpellEffects.modify_outgoing_damage, 35, fire, 1)
    assert dd.deimos_damage(500, fire, caster, target, [blade], []) == \
        pytest.approx(675.0)

    # blades compound rather than adding: 500 * 1.35 * 1.25
    b2 = dd.Effect(dd.SpellEffects.modify_outgoing_damage, 25, fire, 2)
    assert dd.deimos_damage(500, fire, caster, target, [blade, b2], []) == \
        pytest.approx(843.75)


def test_port_dedupes_by_stacking_id():
    """combat_math.py:79-93 -- one stacking identity counts once."""
    fire = dd.school_id("fire")
    caster, target = dd.Stats(is_player=True), dd.Stats()
    same = [dd.Effect(dd.SpellEffects.modify_outgoing_damage, 35, fire, 7),
            dd.Effect(dd.SpellEffects.modify_outgoing_damage, 35, fire, 7)]
    assert dd.deimos_damage(500, fire, caster, target, same, []) == \
        pytest.approx(675.0)


def test_port_puts_flat_damage_before_the_multipliers():
    """The first of the two findings. combat_math.py:157 adds flat damage
    straight after the school damage %, so a blade multiplies it."""
    fire = dd.school_id("fire")
    caster = dd.Stats(flat_damage={fire: 100}, is_player=True)
    blade = dd.Effect(dd.SpellEffects.modify_outgoing_damage, 35, fire, 1)
    # (500 + 100) * 1.35, not 500 * 1.35 + 100
    assert dd.deimos_damage(500, fire, caster, dd.Stats(), [blade], []) == \
        pytest.approx(810.0)


def test_port_puts_flat_resist_before_percent_resist():
    """The second finding. combat_math.py:253-254 subtracts flat resist
    before the percent multiply."""
    fire = dd.school_id("fire")
    target = dd.Stats(resist={fire: 0.30}, flat_resist={fire: 75})
    # (500 - 75) * 0.7, not 500 * 0.7 - 75
    assert dd.deimos_damage(500, fire, dd.Stats(is_player=True), target,
                            [], []) == pytest.approx(297.5)


def test_curve_stat_is_off_when_the_duel_carries_no_limits():
    """A zero damage_limit would divide by zero in the original."""
    caster = dd.Stats(damage={dd.school_id("fire"): 0.5}, is_player=True)
    got = dd.deimos_damage(500, dd.school_id("fire"), caster, dd.Stats(), [], [])
    assert got == pytest.approx(750.0)


def test_curve_stat_bends_a_large_stat_down():
    """With real duel limits, a big damage stat is curved (combat_math.py:39)."""
    curved = dd.curve_stat(1.5, l=1.2, k0=50.0, n0=25.0)
    assert curved < 1.5


# ------------------------------------------------------------ the diff run
def test_plain_scenarios_agree_exactly():
    """Everything with no flat stat and no pierce must match to the cent.
    This is what makes the divergences credible: the two engines are not
    merely close, they are identical wherever nothing contested is in
    play."""
    rows = {r["scenario"]: r for r in compare()}
    for name in ("bare hit", "one blade", "three stacked blades", "one trap",
                 "blade and trap", "tower shield", "mob resist", "mob boost",
                 "gear damage", "gear damage and blade", "prism",
                 "trap before prism", "trap after prism",
                 "mob resist vs pierce"):
        assert rows[name]["agree"], (name, rows[name])


def test_the_two_engines_now_agree_everywhere():
    """After three fixes -- wizAi adopting Deimos's flat-stat placement,
    Deimos's pierce units being corrected, and wizAi adopting Deimos's
    effect-stacking rule -- two independently written engines agree on
    every scenario in the suite.

    The last of those sat in EXPECTED for a while, written off as the
    same rule enforced at different stages. It was not: wizAi refused the
    duplicate *cast*, which the game permits, and then multiplied every
    duplicate into one strike, which the game does not."""
    diverging = {r["scenario"] for r in compare() if not r["agree"]}
    assert diverging == set(), diverging


def test_legacy_rules_still_show_the_flat_divergence():
    """The finding has to stay reproducible, or the fix is unfalsifiable."""
    rows = {r["scenario"]: r for r in compare(rules=legacy_ruleset())}
    for name in ("flat damage", "flat resist"):
        assert not rows[name]["agree"], (name, rows[name])


def test_every_divergence_is_explained():
    """No silent disagreements: a row that differs must be listed in
    EXPECTED with a reason."""
    for r in compare():
        if not r["agree"]:
            assert r["scenario"] in EXPECTED, f"unexplained divergence: {r}"


def test_pierce_shaves_a_shield_in_both_engines():
    """The pierce unit fix, pinned in numbers. 20% pierce must move a -50
    shield to -30 -- in wizAi, which always did, and now in Deimos, which
    used to move it to -49.8 and leave the shield almost intact."""
    sc = Scenario("probe", 500, "fire",
                  caster=Side(pierce=0.20, is_player=True),
                  wards=[Buff("Tower Shield", -50, None)])
    assert sc.wizai_damage() == pytest.approx(350.0)    # 500 * 0.70
    assert sc.deimos_damage() == pytest.approx(350.0)


def test_a_pierce_blade_is_worth_its_face_value():
    """kModifyOutgoingArmorPiercing arrives in points. Folded into the
    fraction-valued accumulator without conversion, a +10 pierce blade was
    worth 1000% pierce and erased any shield outright."""
    fire = dd.school_id("fire")
    blade = dd.Effect(dd.SpellEffects.modify_outgoing_armor_piercing, 10,
                      fire, 1)
    shield = dd.Effect(dd.SpellEffects.modify_incoming_damage, -50, 80289, 2)
    got = dd.deimos_damage(500, fire, dd.Stats(is_player=True), dd.Stats(),
                           [blade], [shield])
    # -50 shield, 10 points of pierce -> -40
    assert got == pytest.approx(300.0)


# ------------------------------------------------------------------ naming
def test_normalisation_folds_case_and_punctuation():
    assert _normal("Krokopatra's Curse") == _normal("krokopatras curse")
    assert _normal("Tri  Blade") == "tri blade"


def test_resolver_layers(capsys):
    from data_full import load_spells_full
    cards = load_spells_full()
    r = NameResolver(cards)
    assert r.resolve("Fireblade").name == "Fireblade"
    assert r.resolve("fireblade").name == "Fireblade"      # normalised
    assert r.resolve("FIREBLADE").name == "Fireblade"
    assert r.resolve("Tri Blade").name == "Tri Blade"      # exact, game name
    assert r.resolve("No Such Spell") is None
    assert "No Such Spell" in r.misses


def test_aliases_point_at_real_cards():
    """An alias is only useful if its key is *not* already a card and its
    value *is*. Without this the table silently rots into decoration --
    which is exactly what a first pass at it did."""
    from data_full import load_spells_full
    cards = load_spells_full()
    for stale, real in ALIASES.items():
        assert stale not in cards, \
            f"{stale!r} is a real card; aliasing it would rewrite a good name"
        assert real in cards, f"alias target {real!r} is not in the card table"


def test_langcode_resolves_a_card_whose_name_does_not():
    """`display_name_code()` is the game's own stable identifier. It does
    not move when a spell is renamed and it is identical on a non-English
    client, so it is a better key than the display string -- and it
    rescues a name lookup that misses."""
    from deimos_bridge.live_state import build_catalog
    cat = build_catalog()
    r = NameResolver(cat["cards"], cat)
    assert r.resolve("Renamed In A Later Patch",
                     langcode="Spells_Fireblade") is cat["cards"]["Fireblade"]


def test_a_shared_langcode_resolves_to_the_canonical_spell():
    """"Spells_Fireblade" is shared by Fireblade, Fireblade - EM,
    Fireblade - SIT, Fireblade - Tear, FirebladeBOSS01, FirebladeBOSS02
    and a raid sigil. Resolving on the code alone could hand the policy a
    boss variant; `base_spell` settles it, because the canonical record
    is the one whose name IS its base spell."""
    from deimos_bridge.live_state import build_catalog
    cat = build_catalog()
    assert cat["langcodes"]["Spells_Fireblade"].name == "Fireblade"


def test_langcodes_with_no_single_canonical_are_dropped():
    """Where `base_spell` cannot single one out, there is no answer, and
    the catalog must leave the code out instead of picking."""
    from deimos_bridge.live_state import build_catalog
    cat = build_catalog()
    assert cat["ambiguous_langcodes"], "expected some undecidable groups"
    for code in cat["ambiguous_langcodes"]:
        assert code not in cat["langcodes"]


def test_misses_are_classified_by_cause():
    """The two kinds need different fixes, so they must not look alike.
    A name the game data knows but the decoder skipped is a gap in
    _map_effect with a named cause; a name the data has never heard of is
    a spelling problem."""
    from deimos_bridge.live_state import (MISS_DECODER, MISS_UNKNOWN,
                                          build_catalog)
    cat = build_catalog()
    r = NameResolver(cat["cards"], cat)

    assert r.resolve("Summon589244") is None
    kind, detail = r.classify("Summon589244")
    assert kind == MISS_DECODER
    assert "kSummonCreature" in detail

    assert r.resolve("Not A Real Spell") is None
    assert r.classify("Not A Real Spell")[0] == MISS_UNKNOWN


def test_classification_degrades_honestly_without_a_catalog():
    """A resolver built without a catalog genuinely cannot tell the two
    apart, and must say unknown rather than invent a cause."""
    from data_full import load_spells_full
    from deimos_bridge.live_state import MISS_UNKNOWN
    r = NameResolver(load_spells_full())
    assert r.classify("Summon589244")[0] == MISS_UNKNOWN


def test_hidden_cards_are_recorded_not_silently_dropped():
    """An unresolvable card cannot enter the policy's hand -- there is no
    wizAi Card for it -- but dropping it silently is how a run quietly
    stops measuring anything: the policy plans a 2-card hand while
    holding 4, and its scarcity feature counts the wrong nukes."""
    from deimos_bridge.live_state import build_catalog
    cat = build_catalog()
    r = NameResolver(cat["cards"], cat)
    combat = MockCombat(
        [MockMember("Wizard", 2000, client=True, team_id=0),
         MockMember("Lost Soul", 900, monster=True, team_id=1)],
        [MockCard("Fireblade"), MockCard("Sunbird"),
         MockCard("Not A Real Spell"), MockCard("Summon589244")])
    read = run(read_state(combat, r, "fire"))
    assert sorted(read.hidden) == ["Not A Real Spell", "Summon589244"]
    assert sorted(c.name for c in read.state.hand) == ["Fireblade", "Sunbird"]
    assert read.hand_visibility == pytest.approx(0.5)


def test_full_visibility_when_everything_resolves():
    from data_full import load_spells_full
    r = NameResolver(load_spells_full())
    read = run(read_state(simple_fight(hand=("Fireblade", "Sunbird")),
                          r, "fire"))
    assert read.hidden == []
    assert read.hand_visibility == 1.0


def test_report_separates_the_two_causes():
    from deimos_bridge.live_state import build_catalog
    cat = build_catalog()
    r = NameResolver(cat["cards"], cat)
    r.resolve("Summon589244")
    r.resolve("Not A Real Spell")
    text = r.report()
    assert "decoder skipped these" in text
    assert "kSummonCreature" in text
    assert "not in the game data" in text


def test_resolver_never_guesses():
    """A near-miss must fail rather than resolve to a neighbour. Casting
    the wrong spell in a real fight is worse than passing."""
    from data_full import load_spells_full
    r = NameResolver(load_spells_full())
    assert r.resolve("Fireblad") is None
    assert r.resolve("Fire Blades") is None


# ---------------------------------------------------------------- live read
def test_read_state_builds_a_usable_wizai_state():
    from data_full import load_spells_full
    r = NameResolver(load_spells_full())
    combat = simple_fight(player_hp=2500, enemy_hp=1800,
                          hand=("Fireblade", "Sunbird"), pips=3)
    read = run(read_state(combat, r, "fire"))
    s = read.state
    assert s.player_hp == 2500
    assert s.norm_pips == 3
    assert sorted(c.name for c in s.hand) == ["Fireblade", "Sunbird"]
    assert len(s.enemies) == 1
    assert s.enemies[0].hp == 1800


def test_read_state_skips_dead_and_uncastable():
    from data_full import load_spells_full
    r = NameResolver(load_spells_full())
    me = MockMember("Wizard", 2000, client=True, normal_pips=2)
    alive = MockMember("Lost Soul", 500, monster=True)
    dead = MockMember("Corpse", 0, monster=True, dead=True)
    cards = [MockCard("Fireblade"), MockCard("Sunbird", castable=False)]
    read = run(read_state(MockCombat([me, alive, dead], cards), r, "fire"))
    assert [e.name for e in read.state.enemies] == ["Lost Soul"]
    assert [c.name for c in read.state.hand] == ["Fireblade"]


def test_read_state_reads_enemy_hanging_effects():
    """A shield on the mob has to reach the policy, or it plans into a
    wall."""
    from data_full import load_spells_full
    r = NameResolver(load_spells_full())
    shield = MockEffect("modify_incoming_damage", -50, 80289, 4242)
    combat = simple_fight(enemy_hangings=[shield])
    read = run(read_state(combat, r, "fire"))
    wards = read.state.enemies[0].wards
    assert len(wards) == 1
    assert wards[0].kind == "damage"
    assert wards[0].percent == pytest.approx(-0.5)


def test_enemy_minion_counts_as_an_enemy():
    """The trap in wizwalker's API. `is_monster()` is "not a player and
    not a minion", so an enemy summon answers False and a reader that
    trusts it files the summon as an ally -- leaving the policy planning
    against a board that is missing a target it will actually be hit by.
    `team_id` is what settles it."""
    from data_full import load_spells_full
    r = NameResolver(load_spells_full())
    me = MockMember("Wizard", 2000, client=True, team_id=0)
    boss = MockMember("Krokopatra", 3000, monster=True, team_id=1)
    summon = MockMember("Scarab", 400, monster=True, minion=True, team_id=1)
    assert run(summon.is_monster()) is False        # the trap, in the mock

    read = run(read_state(MockCombat([me, boss, summon], [MockCard("Sunbird")]),
                          r, "fire"))
    assert sorted(e.name for e in read.state.enemies) == ["Krokopatra", "Scarab"]
    assert read.state.allies == []


def test_friendly_minion_counts_as_an_ally():
    from data_full import load_spells_full
    r = NameResolver(load_spells_full())
    me = MockMember("Wizard", 2000, client=True, team_id=0)
    pet = MockMember("Golem", 500, minion=True, team_id=0)
    foe = MockMember("Lost Soul", 600, monster=True, team_id=1)
    read = run(read_state(MockCombat([me, pet, foe], [MockCard("Sunbird")]),
                          r, "fire"))
    assert [e.name for e in read.state.enemies] == ["Lost Soul"]
    assert [a.name for a in read.state.allies] == ["Golem"]


def test_read_state_survives_a_participant_that_will_not_read():
    """Memory reads fail mid-fight; a bad read must not end the duel."""
    from data_full import load_spells_full
    r = NameResolver(load_spells_full())

    class Broken(MockMember):
        async def get_participant(self):
            raise RuntimeError("MemoryReadError")

    me = MockMember("Wizard", 2000, client=True)
    foe = Broken("Lost Soul", 900, monster=True)
    read = run(read_state(MockCombat([me, foe], [MockCard("Fireblade")]),
                          r, "fire"))
    assert read.state.enemies[0].wards == []


# ------------------------------------------------------------- the backend
def _backend(policy=None, deck=None):
    from data_full import load_spells_full
    from w101_sim import make_blade_stack
    return WizAiBackend.from_trained(
        school="fire", deck=deck or (["Fireblade"] * 3 + ["Sunbird"] * 4),
        cards=load_spells_full(), policy=policy or make_blade_stack(2))


def test_backend_turns_a_policy_choice_into_a_named_move():
    be = _backend()
    be.attach_combat(simple_fight(hand=("Fireblade", "Sunbird"), pips=4))
    decision = run(be.decide())
    assert decision.card_name in ("Fireblade", "Sunbird")
    line = be._to_priority_line(decision)
    assert line.priorities[0].move.card.name == decision.card_name
    # always a pass fallback, so an unplayable pick cannot stall the fight
    assert line.priorities[-1].move.card.name == "pass"


def test_backend_passes_when_the_policy_returns_nothing():
    be = _backend(policy=lambda sim, s: None)
    be.attach_combat(simple_fight())
    d = run(be.decide())
    assert d.passing
    assert be._to_priority_line(d).priorities[0].move.card.name == "pass"


def test_backend_survives_a_policy_that_raises():
    """A learned policy that throws must cost one round, not the fight."""
    def boom(sim, s):
        raise ValueError("bad state key")

    be = _backend(policy=boom)
    be.attach_combat(simple_fight())
    d = run(be.decide())
    assert d.passing
    assert "ValueError" in d.reason


def test_backend_refuses_a_card_that_is_not_in_hand():
    be = _backend(policy=lambda sim, s: "Efreet")
    be.attach_combat(simple_fight(hand=("Fireblade",)))
    d = run(be.decide())
    assert d.passing
    assert "not a castable card in hand" in d.reason


def test_provenance_suffix_is_not_mistaken_for_a_target_index():
    """Two encodings collide on "@": rl_agent writes a target as
    "Fireblade@1", data_full writes provenance as "Imp@item". Splitting
    unconditionally turned "Thunder Snake - Starter Wand@item" into a
    name that is not a card, so every wand, treasure and pet card became
    unplayable and the policy passed every round."""
    be = _backend(policy=lambda sim, s: "Thunder Snake - Starter Wand@item")
    combat = MockCombat(
        [MockMember("Wizard", 1000, client=True, normal_pips=2, team_id=0),
         MockMember("Lost Soul", 450, monster=True, team_id=1)],
        [MockCard("Thunder Snake - Starter Wand", item=True)])
    be.attach_combat(combat)
    d = run(be.decide())
    assert not d.passing, d.reason
    assert d.card_name == "Thunder Snake - Starter Wand@item"
    assert d.target_index is None


def test_a_targeted_item_card_splits_on_the_index_only():
    """The awkward case both encodings produce together: rl_agent appends
    its index to a name that already carries a provenance suffix."""
    be = _backend(policy=lambda sim, s: "Imp - Starter Wand@item@1")
    me = MockMember("Wizard", 1000, client=True, normal_pips=2, team_id=0)
    foes = [MockMember("A", 300, monster=True, team_id=1),
            MockMember("B", 400, monster=True, team_id=1)]
    be.attach_combat(MockCombat(
        [me] + foes, [MockCard("Imp - Starter Wand", item=True)]))
    d = run(be.decide())
    assert d.card_name == "Imp - Starter Wand@item"
    assert d.target_index == 1


def test_a_real_starter_hand_produces_a_cast():
    """End to end over the hand that exposed this: an ice wizard holding
    one deck card and four starter-wand item cards must play something."""
    from deimos_bridge.live_state import build_catalog
    from w101_sim import make_blade_stack

    cat = build_catalog()
    be = WizAiBackend.from_trained(
        school="ice", deck=["Frost Beetle"] * 4, cards=cat["cards"],
        policy=make_blade_stack(3), catalog=cat)
    hand = [MockCard("Frost Beetle")] + [
        MockCard(n, item=True) for n in
        ("Dark Sprite - Starter Wand", "Imp - Starter Wand",
         "Scarab - Starter Wand", "Thunder Snake - Starter Wand")]
    be.attach_combat(MockCombat(
        [MockMember("Wizard", 1000, client=True, normal_pips=1, team_id=0),
         MockMember("Lost Soul", 450, monster=True, team_id=1)], hand))
    d = run(be.decide())
    assert not d.passing, d.reason
    assert d.card_name in be.last_read.hand_cards


def test_backend_understands_a_targeted_action():
    """`rl_agent` emits 'name@i' on a multi-enemy board.

    On a *hit*. This used to ask for "Fireblade@1" and assert the index
    survived -- but a Fireblade goes on the caster, so the index was
    never clicked, only written into the log. The parsing is the thing
    worth pinning, so it is pinned on a card the index can mean
    something for; the self-buff case has its own test.
    """
    be = _backend(policy=lambda sim, s: "Sunbird@1")
    me = MockMember("Wizard", 2000, client=True, normal_pips=6)
    foes = [MockMember("A", 500, monster=True), MockMember("B", 700, monster=True)]
    be.attach_combat(MockCombat([me] + foes, [MockCard("Sunbird")]))
    d = run(be.decide())
    assert d.card_name == "Sunbird"
    assert d.target_index == 1


def test_backend_records_history():
    be = _backend()
    be.attach_combat(simple_fight())
    run(be.decide())
    run(be.decide())
    assert len(be.history) == 2


# ------------------------------------------------------------- the handler
class _StubCombatHandler:
    """Stands in for `wizwalker.combat.CombatHandler`.

    Same shape as the real one: constructed with a client, and supplies
    the member/card/round accessors the handler reads. Going through
    `_build_handler_class` with this means the tests exercise the actual
    class-construction path rather than side-stepping it -- the previous
    helper built the handler with `object.__new__`, which is exactly how
    a broken `__bases__` assignment in `make_combat_handler` survived
    every test and only failed on a real machine.
    """

    def __init__(self, client):
        self.client = client
        self._spell_check_boxes = None

    def _bind(self, combat):
        self.pass_button = combat.pass_button
        self.get_members = combat.get_members
        self.get_cards = combat.get_cards
        self.get_client_member = combat.get_client_member
        self.round_number = combat.round_number


def _Handler(combat, backend):
    """A live handler built the real way, over the stub base."""
    from deimos_bridge.live_backend import _build_handler_class
    from deimos_bridge.mock_client import MockClient

    cls = _build_handler_class(_StubCombatHandler)
    h = cls(MockClient(), backend)
    h._bind(combat)
    backend.attach_combat(combat)
    return h


def test_the_live_handler_class_is_a_real_subclass():
    """The bug this guards: `WizAiCombatHandler.__bases__ = (CombatHandler,)`
    raises `TypeError: deallocator differs from 'object'` on every CPython,
    because the current base is exactly `object`. Building the subclass
    instead is the fix, and this asserts the result really inherits from
    both sides."""
    from deimos_bridge.live_backend import (WizAiCombatHandler,
                                            _build_handler_class)
    cls = _build_handler_class(_StubCombatHandler)
    assert issubclass(cls, _StubCombatHandler)
    assert issubclass(cls, WizAiCombatHandler)

    from deimos_bridge.mock_client import MockClient
    be = _backend()
    h = cls(MockClient(), be)
    # both __init__s ran
    assert h._spell_check_boxes is None      # the stub base's
    assert h.backend is be                   # wizAi's
    assert be.combat is h                    # and it attached itself


def test_the_backend_class_is_a_real_subclass():
    """Same fix, same reason, for the wizsprinter backend path."""
    from deimos_bridge.live_backend import WizAiBackend, _build_backend_class

    class StubBase:
        def __init__(self, cast_time=0.3):
            self.cast_time = cast_time

    cls = _build_backend_class(StubBase)
    assert issubclass(cls, StubBase) and issubclass(cls, WizAiBackend)


def test_handler_casts_exactly_once_per_round():
    """The whole reason this class exists. SprintyCombat would re-query a
    NamedSpell and play every duplicate in hand; three Fireblades must be
    one cast, not three."""
    be = _backend(policy=lambda sim, s: "Fireblade")
    combat = MockCombat(
        [MockMember("Wizard", 2000, client=True, normal_pips=4, team_id=0),
         MockMember("Lost Soul", 900, monster=True, team_id=1)],
        [MockCard("Fireblade"), MockCard("Fireblade"), MockCard("Fireblade")])
    h = _Handler(combat, be)
    run(h.handle_round())
    casts = sum(len(c.cast_log) for c in combat._cards)
    assert casts == 1, f"one decision produced {casts} casts"
    assert combat.passed == 0


def test_handler_enters_the_mouse_handler():
    """Casting is mouse clicks -- wizwalker has no memory API for it --
    and entering client.mouse_handler is what activates the mouseless
    cursor hook. A round that skips it clicks nothing, silently."""
    be = _backend(policy=lambda sim, s: "Fireblade")
    combat = MockCombat(
        [MockMember("Wizard", 2000, client=True, normal_pips=4, team_id=0),
         MockMember("Lost Soul", 900, monster=True, team_id=1)],
        [MockCard("Fireblade")])
    h = _Handler(combat, be)
    run(h.handle_round())
    assert h.client.mouse_handler.entered == 1
    assert h.client.mouse_handler.depth == 0     # and exited cleanly


def test_handler_passes_when_the_policy_passes():
    be = _backend(policy=lambda sim, s: None)
    combat = simple_fight()
    h = _Handler(combat, be)
    run(h.handle_round())
    assert combat.passed == 1
    assert sum(len(c.cast_log) for c in combat._cards) == 0


def test_handler_targets_the_mob_the_policy_meant():
    """'name@1' has to hit the second enemy *the policy was shown*, not
    the second entry of some other list."""
    be = _backend(policy=lambda sim, s: "Sunbird@1")
    boss = MockMember("Krokopatra", 3000, monster=True, team_id=1)
    summon = MockMember("Scarab", 400, monster=True, minion=True, team_id=1)
    combat = MockCombat(
        [MockMember("Wizard", 2000, client=True, normal_pips=6, team_id=0),
         boss, summon],
        [MockCard("Sunbird")])
    h = _Handler(combat, be)
    run(h.handle_round())
    card = combat._cards[0]
    assert card.cast_log == [summon], \
        f"aimed at {card.cast_log} instead of the summon"


def test_handler_survives_a_board_read_that_fails():
    """Memory reads fail mid-round when a duel ends under them. That
    costs a round; it must not end the fight."""
    be = _backend()

    async def boom():
        raise RuntimeError("MemoryReadError")

    combat = simple_fight()
    h = _Handler(combat, be)
    be.decide = boom
    run(h.handle_round())
    assert combat.passed == 1
    assert h._read_failures == 1


def test_handler_falls_back_to_pass_when_the_cast_throws():
    be = _backend(policy=lambda sim, s: "Fireblade")

    class Angry(MockCard):
        async def cast(self, target=None, **kw):
            raise RuntimeError("window vanished")

    combat = MockCombat(
        [MockMember("Wizard", 2000, client=True, normal_pips=4, team_id=0),
         MockMember("Lost Soul", 900, monster=True, team_id=1)],
        [Angry("Fireblade")])
    h = _Handler(combat, be)
    run(h.handle_round())
    assert combat.passed == 1


# ------------------------------------------------------------- diagnostics
def test_the_two_known_signatures_are_told_apart(tmp_path):
    """The whole point of knowing both: a binary carrying the fork's
    signature but not the vendored one means "install the fork", not
    "restart the client". Confusing the two sends someone restarting a
    client that will never work."""
    import random

    from deimos_bridge.diagnose_hooks import (AUTOBOT_PATTERN, FORK_PATTERN,
                                              scan_file)

    random.seed(2)
    noise = bytes(random.randrange(256) for _ in range(60_000))
    old_sig = (b"\x48\x8B\xC4\x55\x41\x54\x41\x55\x41\x56\x41\x57"
               + b"\xAA" * 7 + b"\x48" + b"\xAA" * 6 + b"\x48" + b"\xAA" * 7
               + b"\x48\x89\x58\x10\x48\x89\x70\x18\x48\x89\x78\x20"
               + b"\xAA" * 7 + b"\x48\x33\xC4" + b"\xAA" * 7
               + b"\x4C\x8B\xE9" + b"\xAA" * 7 + b"\x80" + b"\xAA" * 6
               + b"\x0F")
    new_sig = (b"\x48\x89\x5C\x24\xAA\x48\x89\x74\x24\xAA"
               b"\x48\x89\x7C\x24\xAA"
               b"\x55\x41\x54\x41\x55\x41\x56\x41\x57"
               b"\x48\x8D\xAC\x24" + b"\xAA" * 4 + b"\x48\x81\xEC"
               + b"\xAA" * 4 + b"\x48\x8B\x05" + b"\xAA" * 4
               + b"\x48\x33\xC4\x48\x89\x85" + b"\xAA" * 4
               + b"\x4C\x8B\xF1" + b"\xAA" * 7 + b"\x80" + b"\xAA" * 6
               + b"\x0F\x84" + b"\xAA" * 4)

    old_exe = tmp_path / "old.exe"
    old_exe.write_bytes(noise[:3000] + old_sig + noise[3000:])
    new_exe = tmp_path / "patched.exe"
    new_exe.write_bytes(noise[:3000] + new_sig + noise[3000:])

    assert scan_file(str(old_exe), AUTOBOT_PATTERN)
    assert not scan_file(str(old_exe), FORK_PATTERN)
    assert scan_file(str(new_exe), FORK_PATTERN)
    assert not scan_file(str(new_exe), AUTOBOT_PATTERN)


def test_autobot_pattern_matches_the_one_wizwalker_ships():
    """The diagnostic duplicates the signature rather than importing it,
    because the import is exactly what may be broken when you need it. So
    the copy has to be checked against the original."""
    import re as _stdre

    src = open("Deimos/libs/wizwalker/wizwalker/memory/handler.py",
               encoding="utf-8").read()
    body = _stdre.search(r"AUTOBOT_PATTERN = \((.*?)\n    \)", src, _stdre.S)
    assert body, "AUTOBOT_PATTERN not found in wizwalker"
    theirs = "".join(_stdre.findall(r'rb"([^"]*)"', body.group(1)))

    from deimos_bridge.diagnose_hooks import AUTOBOT_PATTERN
    ours = AUTOBOT_PATTERN.decode("latin-1")
    assert ours == theirs, "the diagnostic's copy has drifted from wizwalker"


def test_scanner_finds_the_signature_and_does_not_invent_one(tmp_path):
    """Both directions matter: a false negative would send someone off to
    update wizwalker for nothing, and a false positive would send them
    restarting a client that will never work."""
    import random

    from deimos_bridge.diagnose_hooks import scan_file

    random.seed(1)
    noise = bytes(random.randrange(256) for _ in range(120_000))
    signature = (
        b"\x48\x8B\xC4\x55\x41\x54\x41\x55\x41\x56\x41\x57" + b"\xAA" * 7 +
        b"\x48" + b"\xAA" * 6 + b"\x48" + b"\xAA" * 7 +
        b"\x48\x89\x58\x10\x48\x89\x70\x18\x48\x89\x78\x20" + b"\xAA" * 7 +
        b"\x48\x33\xC4" + b"\xAA" * 7 + b"\x4C\x8B\xE9" + b"\xAA" * 7 +
        b"\x80" + b"\xAA" * 6 + b"\x0F")

    has = tmp_path / "has.bin"
    has.write_bytes(noise[:4000] + signature + noise[4000:])
    lacks = tmp_path / "lacks.bin"
    lacks.write_bytes(noise)

    assert scan_file(str(has)) == [4000]
    assert scan_file(str(lacks)) == []


def test_scanner_sees_a_zeroed_autobot_region_as_absent():
    """The failure mode the diagnostic exists to identify: a previous
    attach zeroed the region, so the bytes are gone from memory while the
    file on disk still has them."""
    import tempfile

    from deimos_bridge.diagnose_hooks import scan_file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(b"\x00" * 8000)
        path = f.name
    assert scan_file(path) == []


# --------------------------------------------------------------- enum audit
def test_wizai_effect_table_matches_the_client_enum():
    """wizAi's effect_type_enum.json against the enum lifted out of the
    game client. One cosmetic capitalisation difference is tolerated; a
    real id/name disagreement is not."""
    from deimos_bridge.effect_audit import compare_enums

    def squash(s):
        return s.lower().lstrip("k").replace("_", "")

    r = compare_enums()
    real = [d for d in r["disagree"] if squash(d[1]) != squash(d[2])]
    assert not real, f"wizAi disagrees with the client enum: {real}"
    assert r["agree"] >= 150


def test_shim_matches_the_real_combat_api_shape():
    """The stand-in types must carry the attributes SprintyCombat reads,
    or a live run breaks on Windows in a way no test here would catch."""
    from deimos_bridge import combat_api_shim as shim
    line = shim.PriorityLine(priorities=[
        shim.MoveConfig(move=shim.Move(card=shim.NamedSpell("Sunbird")),
                        target=shim.TargetData(shim.TargetType.type_enemy))])
    assert line.round is None
    mc = line.priorities[0]
    assert mc.move.card.name == "Sunbird"
    assert mc.move.enchant is None and mc.move.second_enchant is None
    assert mc.condition is None
    assert mc.target.target_type is shim.TargetType.type_enemy
    assert mc.target.extra_data is None and mc.target.is_literal is False


# ------------------------------------------------------- live-play policies
def test_self_targeted_cards_are_cast_at_the_caster():
    """A blade is `tgt: 'self'`. Aiming it at a mob fails silently and
    wastes the round -- which is what happened on the first live fight:
    the policy chose Mythblade, the click went to the enemy, nothing was
    placed. wizsprinter resolves `type_self` to `get_client_member()`."""
    from deimos_bridge.live_backend import _primary_target
    from data_full import load_spells_full

    cards = load_spells_full()
    assert _primary_target(cards["Mythblade"]) == "self"
    assert _primary_target(cards["Iceblade"]) == "self"
    assert _primary_target(cards["Tower Shield"]) == "self"
    assert _primary_target(cards["Frost Beetle"]) == "enemy"
    assert _primary_target(cards["Ice Trap"]) == "enemy"
    assert _primary_target(cards["Blizzard"]) == "enemies"


def test_handler_clicks_the_caster_for_a_blade():
    be = _backend(policy=lambda sim, s: "Iceblade",
                  deck=["Iceblade"] * 3 + ["Frost Beetle"] * 3)
    me = MockMember("Wizard", 2000, client=True, normal_pips=4, team_id=0)
    foe = MockMember("Lost Soul", 900, monster=True, team_id=1)
    combat = MockCombat([me, foe], [MockCard("Iceblade")])
    h = _Handler(combat, be)
    run(h.handle_round())
    assert combat._cards[0].cast_log == [me], \
        f"blade was aimed at {combat._cards[0].cast_log}, not the caster"


def test_handler_clicks_nothing_for_an_aoe():
    be = _backend(policy=lambda sim, s: "Blizzard",
                  deck=["Blizzard"] * 4)
    me = MockMember("Wizard", 2000, client=True, normal_pips=7, team_id=0)
    combat = MockCombat(
        [me, MockMember("A", 500, monster=True, team_id=1),
         MockMember("B", 500, monster=True, team_id=1)],
        [MockCard("Blizzard")])
    h = _Handler(combat, be)
    run(h.handle_round())
    assert combat._cards[0].cast_log == [None]


def test_school_aware_policy_ignores_a_buff_it_cannot_use():
    """The live observation: an ice wizard holding a Mythblade and a
    storm wand card stacked the blade, which can never multiply the hit."""
    from deimos_bridge.policies import buff_matches
    from data_full import load_spells_full

    cards = load_spells_full()
    assert not buff_matches(cards["Mythblade"], "ice")
    assert buff_matches(cards["Iceblade"], "ice")
    assert buff_matches(cards["Tri Blade"], "ice")       # ice/storm/fire
    assert buff_matches(cards["Balanceblade"], "ice")    # universal
    assert not buff_matches(cards["Tri Blade"], "death")


def test_dead_ward_legs_are_not_counted_as_pending_buffs():
    """A Tri Trap places fire, ice and storm legs. An ice hit consumes
    only the ice one; the other two sit on the enemy forever. `State.traps`
    counts all three, so make_blade_stack fires on a stack of one live
    multiplier and two corpses."""
    from deimos_bridge.policies import pending_for
    from w101_sim import Actor, Hanging, State

    player = Actor(name="W", school="ice", hp=1000, max_hp=1000, team=0)
    enemy = Actor(name="M", school="ice", hp=1000, max_hp=1000, team=1)
    for school in ("fire", "storm"):          # the stranded legs
        enemy.wards.append(Hanging(name="Tri Trap", slot="ward", kind="damage",
                                   percent=0.3, schools={school}, sub=hash(school)))
    s = State(player, [enemy])

    assert len(s.traps) >= 1                  # the legacy view sees them
    assert pending_for(s, "ice") == 0         # none of them can act on ice


def test_school_aware_matches_or_beats_blade_stack_on_the_live_decks():
    """It is the live default, so it must not be a regression on the
    school-coherent decks the published tables use."""
    import random

    from data_full import LIVE_DECKS, load_spells_full
    from w101_sim import Boss, Sim, evaluate_paired, make_blade_stack

    from deimos_bridge.policies import school_aware_blade_stack

    cards = load_spells_full()
    pols = {"old": make_blade_stack(3), "new": school_aware_blade_stack(3)}
    for school, variants in LIVE_DECKS.items():
        for label, deck in variants.items():
            sim = Sim(cards, deck, school,
                      Boss(name="d", hp=2500, school="ice", dmg=140),
                      rng=random.Random(1), player_hp=2500)
            r = evaluate_paired(sim, pols, n=200)
            assert r["new"]["win_rate"] >= r["old"]["win_rate"] - 0.03, \
                f"{school}/{label}: {r['new']['win_rate']} < {r['old']['win_rate']}"


# ------------------------------------------------ which policy actually played
def test_a_decision_names_the_policy_that_made_it():
    """Without this the Decisions log cannot answer 'is the model I
    selected driving?' -- every policy produces the same shape of row."""
    be = _backend()
    be.policy_name = "blade-stack(2)"
    be.attach_combat(simple_fight(hand=("Fireblade", "Sunbird"), pips=4))
    d = run(be.decide())
    assert d.policy == "blade-stack(2)"


def test_a_trained_policy_says_when_it_fell_back():
    """The failure this exists to surface: a trained policy that never
    recognises a board plays the fallback heuristic, and the fight looks
    completely normal while doing it."""
    from deimos_bridge.policies import TrainedPolicy

    class _Agent:
        Q = {}

        class feat:
            @staticmethod
            def key(sim, s):
                return ("k",)

            @staticmethod
            def legal(sim, s):
                return ["__pass__"]

    wrapped = TrainedPolicy(_Agent(), fallback=lambda sim, s: "Sunbird")
    be = _backend(policy=wrapped)
    be.policy_name = "trained (Q)"
    be.attach_combat(simple_fight(hand=("Fireblade", "Sunbird"), pips=4))
    d = run(be.decide())
    assert d.card_name == "Sunbird"
    assert d.policy == "trained (Q) — fallback (state not in Q table)"


def test_a_raising_policy_does_not_report_a_stale_source():
    """`last_source` is left over from the previous call when the policy
    blows up, and reporting it would credit the Q table for a round it
    never decided."""
    class _Boom:
        last_source = "Q table"

        def __call__(self, sim, s):
            raise RuntimeError("nope")

    be = _backend(policy=_Boom())
    be.policy_name = "trained (Q)"
    be.attach_combat(simple_fight())
    d = run(be.decide())
    assert d.passing
    assert d.policy == "trained (Q)"
    assert "RuntimeError" in d.reason


def test_the_policy_can_be_swapped_between_rounds():
    """The point of the hot swap: reconnecting to change models throws
    away the deck and the health reading the next decision depends on."""
    be = _backend(policy=lambda sim, s: "Fireblade")
    be.policy_name = "a"
    be.attach_combat(simple_fight(hand=("Fireblade", "Sunbird"), pips=4))
    first = run(be.decide())

    be.set_policy(lambda sim, s: "Sunbird", "b")
    second = run(be.decide())

    assert (first.card_name, first.policy) == ("Fireblade", "a")
    assert (second.card_name, second.policy) == ("Sunbird", "b")


def test_a_swap_does_not_land_halfway_through_a_round():
    """`decide` reads the policy and its label together, once, before
    calling. A swap arriving mid-decision must not rename the round it
    interrupted -- the row would then name a policy that did not choose
    the card on it."""
    be = _backend()

    def _swapper(sim, s):
        be.set_policy(lambda *a: "Sunbird", "after")   # arrives mid-round
        return "Fireblade"

    be.set_policy(_swapper, "before")
    be.attach_combat(simple_fight(hand=("Fireblade", "Sunbird"), pips=4))
    d = run(be.decide())
    assert d.card_name == "Fireblade"         # the in-flight policy's answer
    assert d.policy == "before"               # ...credited to it, not "after"

    nxt = run(be.decide())                    # the next round is swapped
    assert (nxt.card_name, nxt.policy) == ("Sunbird", "after")


# --------------------------------------------------------------- aiming
def _two_mobs(hand, pips=6, boss_hp=800, minion_hp=250, deck=None):
    """A boss and a minion -- the board where targeting starts to matter."""
    import random

    from data_full import load_spells_full
    from w101_sim import Actor, Boss, Sim, State

    cards = load_spells_full()
    deck = deck or (["Frost Beetle"] * 3 + ["Ice Trap"] * 3
                    + ["Snow Serpent"] * 3)
    sim = Sim(cards, deck, "ice",
              Boss(name="boss", hp=boss_hp, school="fire", dmg=40),
              rng=random.Random(4), player_hp=800)
    p = Actor(name="W", school="ice", hp=800, max_hp=800, team=0,
              norm_pips=pips)
    p.hand = [cards[n] for n in hand]
    p.deck = [cards[n] for n in deck]
    foes = [Actor(name="boss", school="fire", hp=boss_hp, max_hp=boss_hp,
                  team=1),
            Actor(name="minion", school="fire", hp=minion_hp,
                  max_hp=minion_hp, team=1)]
    return sim, State(p, foes), cards


def _play(sim, s, policy, turns=8):
    """(card name, target name) per turn, playing the moves out."""
    from deimos_bridge.policies import _split

    out = []
    for _ in range(turns):
        if not any(e.alive for e in s.enemies):
            break
        card, tgt = _split(policy(sim, s))
        if card is None:
            out.append(("pass", None))
        else:
            who = s.enemies[tgt].name if 0 <= tgt < len(s.enemies) else None
            out.append((card.name, who))
            if sim.can_cast(s, card, tgt):
                sim.cast(s, card, tgt)
        sim.end_round(s)
    return out


def test_policies_aim_at_a_specific_enemy():
    """They used to return a bare card. Nothing chose a target, so the
    live handler clicked `enemy_members[0]` -- whichever mob the
    participant list happened to put first -- and when that one died
    everything silently moved to a different mob."""
    from deimos_bridge.policies import (greedy_ttk, school_aware_blade_stack,
                                        _split)

    for policy in (greedy_ttk(), school_aware_blade_stack(3)):
        sim, s, _ = _two_mobs(["Ice Trap", "Snow Serpent", "Frost Beetle"])
        card, target = _split(policy(sim, s))
        assert card is not None
        assert isinstance(target, int) and 0 <= target < len(s.enemies)


def test_a_trap_and_the_hit_it_buys_land_on_the_same_mob():
    """The reported symptom: traps spread across two enemies. A trap on
    one mob followed by a nuke on another is two wasted turns -- the
    charm never fires."""
    from deimos_bridge.policies import school_aware_blade_stack

    sim, s, _ = _two_mobs(["Ice Trap", "Ice Trap", "Snow Serpent",
                           "Frost Beetle"])
    played = _play(sim, s, school_aware_blade_stack(3), turns=4)
    traps = {who for name, who in played if name == "Ice Trap"}
    hits = {who for name, who in played if name in ("Snow Serpent",
                                                    "Frost Beetle")}
    assert len(traps) == 1, played          # all traps on ONE mob
    assert traps == hits, played            # and that is the mob we hit


def test_the_focus_does_not_wander_while_the_target_lives():
    """`focus_target` picks the lowest-health living enemy, which is
    self-reinforcing: hitting it only lowers its health further. A rule
    that wandered would split a buff stack across two mobs."""
    from deimos_bridge.policies import school_aware_blade_stack

    sim, s, _ = _two_mobs(["Ice Trap", "Snow Serpent", "Frost Beetle",
                           "Snow Serpent"], minion_hp=600, boss_hp=900)
    played = _play(sim, s, school_aware_blade_stack(3), turns=3)
    assert len({who for _, who in played}) == 1, played


def test_focus_moves_on_once_the_target_dies():
    from deimos_bridge.policies import focus_target
    from w101_sim import Actor, State

    p = Actor(name="W", school="ice", hp=800, max_hp=800, team=0)
    a = Actor(name="a", school="fire", hp=100, max_hp=500, team=1)
    b = Actor(name="b", school="fire", hp=300, max_hp=500, team=1)
    s = State(p, [a, b])
    assert focus_target(s) == 0
    a.hp = 0
    assert focus_target(s) == 1


def test_the_rollout_scores_targets_apart():
    """It cast every candidate at enemy 0, so on a multi-mob board every
    target scored identically -- and a scoring function that cannot tell
    two targets apart can never choose between them.

    Made unambiguous by putting a trap on one mob: the same nuke is worth
    more against it, and only an aimed rollout can see that."""
    from deimos_bridge.policies import _rollout

    sim, s, _ = _two_mobs(["Ice Trap", "Snow Serpent"], boss_hp=900,
                          minion_hp=900)
    sim.cast(s, s.hand[0], 1)                    # trap the second mob
    nuke = next(c for c in s.hand if c.name == "Snow Serpent")

    # One ply, so the first cast is the whole score and the trap is the
    # only thing separating the two lines.
    _, dmg_untrapped = _rollout(sim, s, nuke, 1, 0)
    _, dmg_trapped = _rollout(sim, s, nuke, 1, 1)
    # `_rollout` returns damage negated, so lower is more damage.
    assert dmg_trapped < dmg_untrapped, (dmg_trapped, dmg_untrapped)
    assert dmg_trapped == pytest.approx(dmg_untrapped * 1.4)


def test_a_second_identical_trap_is_legal_and_is_banked():
    """Three Ice Traps go on one mob. Each hit consumes one; the rest
    stay standing. wizAi used to refuse the second cast outright *and*
    multiply every trap into a single strike, which valued three at
    2.744x instead of 1.4x -- the arithmetic that made stacking look
    worth spending rounds on."""
    from deimos_bridge.telemetry import predict_damage

    sim, s, cards = _two_mobs(["Ice Trap", "Ice Trap", "Snow Serpent"])
    sim.cast(s, s.hand[0], 1)
    trap2 = next(c for c in s.hand if c.name == "Ice Trap")
    assert sim.can_cast(s, trap2, 1)             # legal, like the game

    nuke = next(c for c in s.hand if c.name == "Snow Serpent")
    one = predict_damage(sim, s, nuke, 1)
    sim.cast(s, trap2, 1)
    assert len(s.enemies[1].wards) == 2          # both standing
    two = predict_damage(sim, s, nuke, 1)
    assert two == pytest.approx(one)             # the second does not add


def test_backend_accepts_an_aimed_move():
    be = _backend(policy=lambda sim, s: (s.hand[0], 1))
    me = MockMember("Wizard", 2000, client=True, normal_pips=4, team_id=0)
    foes = [MockMember("A", 500, monster=True, team_id=1),
            MockMember("B", 700, monster=True, team_id=1)]
    be.attach_combat(MockCombat([me] + foes, [MockCard("Sunbird")]))
    d = run(be.decide())
    assert d.card_name == "Sunbird"
    assert d.target_index == 1


def test_a_self_buff_carries_no_enemy_target():
    """A blade goes on the caster. Carrying an index for it would put a
    mob in the decision log that the cast never touched."""
    be = _backend(policy=lambda sim, s: (s.hand[0], 1))
    me = MockMember("Wizard", 2000, client=True, normal_pips=4, team_id=0)
    foes = [MockMember("A", 500, monster=True, team_id=1),
            MockMember("B", 700, monster=True, team_id=1)]
    be.attach_combat(MockCombat([me] + foes, [MockCard("Fireblade")]))
    d = run(be.decide())
    assert d.card_name == "Fireblade"
    assert d.target_kind == "self"
    assert d.target_index is None


def test_an_out_of_range_target_is_dropped_not_clicked():
    """A stale index -- the mob it named died between the read and the
    decision -- must fall back to the handler's default rather than
    indexing off the end of the board."""
    be = _backend(policy=lambda sim, s: (s.hand[0], 7))
    be.attach_combat(simple_fight(hand=("Sunbird",), pips=6))
    d = run(be.decide())
    assert d.card_name == "Sunbird"
    assert d.target_index is None


# ----------------------------------------------- the trained policy's aiming
def _q_board(hand, deck=None, foes=((900, "boss"), (400, "minion"))):
    import random

    from data_full import load_spells_full
    from w101_sim import Actor, Boss, Sim, State

    cards = load_spells_full()
    deck = deck or (["Ice Trap"] * 3 + ["Snow Serpent"] * 3)
    sim = Sim(cards, deck, "ice",
              Boss(name="boss", hp=foes[0][0], school="fire", dmg=40),
              rng=random.Random(0), player_hp=900)
    p = sim.new_state().player
    p.hp = p.max_hp = 900
    p.norm_pips = 6
    p.hand = [cards[n] for n in hand]
    p.deck = [cards[n] for n in deck]
    enemies = [Actor(name=n, school="fire", hp=hp, max_hp=hp, team=1)
               for hp, n in foes]
    return sim, State(p, enemies), cards


def test_the_action_set_offers_a_trap_on_every_mob():
    """A trap on the boss does not remove Ice Trap from the action set --
    the game lets you put one on each mob, and on a board with a minion
    that is often the right play."""
    from rl_agent import Featurizer, apply_action

    sim, s, cards = _q_board(["Ice Trap", "Ice Trap", "Snow Serpent"])
    feat = Featurizer(cards, ["Ice Trap"] * 3 + ["Snow Serpent"] * 3)
    assert "Ice Trap@0" in feat.legal(sim, s)

    apply_action(sim, s, "Ice Trap@0")
    now = feat.legal(sim, s)
    assert "Ice Trap@1" in now
    assert "Ice Trap@0" in now              # a second on the boss is legal too
    assert len(s.enemies[0].wards) == 1


def test_a_second_trap_on_one_mob_does_not_double_the_hit():
    """It is banked for the next strike instead. This is the arithmetic
    that decides whether stacking is worth the round, and wizAi had it
    at 1.4^n."""
    from rl_agent import Featurizer, apply_action

    from deimos_bridge.telemetry import predict_damage

    sim, s, cards = _q_board(["Ice Trap", "Ice Trap", "Snow Serpent"])
    apply_action(sim, s, "Ice Trap@0")
    nuke = next(c for c in s.hand if c.name == "Snow Serpent")
    one = predict_damage(sim, s, nuke, 0)
    apply_action(sim, s, "Ice Trap@0")
    assert len(s.enemies[0].wards) == 2
    assert predict_damage(sim, s, nuke, 0) == pytest.approx(one)


def test_single_enemy_action_sets_are_unchanged():
    """Target 0 is the only enemy in a 1v1, so every published table
    predating the aiming work has to stay bit-identical."""
    from rl_agent import Featurizer

    sim, s, cards = _q_board(["Ice Trap", "Snow Serpent"],
                             foes=((900, "boss"),))
    feat = Featurizer(cards, ["Ice Trap"] * 3 + ["Snow Serpent"] * 3)
    assert feat.legal(sim, s) == ["__pass__", "Ice Trap", "Snow Serpent"]


# ------------------------------------- knowing when to stop setting up
def test_it_fires_when_the_hit_already_kills():
    """The reported gap: "it doesn't seem to understand minimum damage
    thresholds to kill". A buff round against a mob the plain nuke
    already finishes is a round given to the mob for nothing."""
    from deimos_bridge.policies import school_aware_blade_stack, _split

    # Snow Serpent comfortably kills a 90hp mob unbuffed.
    sim, s, _ = _two_mobs(["Ice Trap", "Snow Serpent"], boss_hp=900,
                          minion_hp=90)
    card, target = _split(school_aware_blade_stack(3)(sim, s))
    assert card.name == "Snow Serpent"
    assert target == 1                         # the one it kills


def test_it_picks_the_cheapest_card_that_still_kills():
    """Between two lethal options, the small one -- it banks pips and
    keeps the big card for the mob that needs it."""
    from deimos_bridge.policies import cheapest_lethal

    sim, s, _ = _two_mobs(["Frost Beetle", "Snow Serpent"], boss_hp=900,
                          minion_hp=60)
    got = cheapest_lethal(sim, s, 1)
    assert got is not None and got.name == "Frost Beetle"


def test_a_shielded_mob_is_not_lethal():
    """Lethality goes through the engine's own cast path, so a shield,
    a resist or an absorb correctly makes the hit not enough."""
    from w101_sim import Hanging

    from deimos_bridge.policies import cheapest_lethal

    sim, s, _ = _two_mobs(["Snow Serpent"], boss_hp=900, minion_hp=170)
    assert cheapest_lethal(sim, s, 1) is not None
    s.enemies[1].wards.append(
        Hanging(name="Tower Shield", slot="ward", kind="damage",
                percent=-0.5, schools=None, source="deck"))
    assert cheapest_lethal(sim, s, 1) is None


def test_a_losing_board_still_ranks_its_moves():
    """The trap-spam mechanism, pinned. When no line survives the
    horizon, `_rollout` used to return one flat constant for every
    candidate -- so the comparison collapsed and the decision fell
    through to the tiebreak, which takes the cheapest card. An Ice Trap
    costs zero pips, so the answer to a hopeless board was three traps."""
    from deimos_bridge.policies import _rollout

    sim, s, _ = _two_mobs(["Ice Trap", "Snow Serpent", "Frost Beetle"],
                          pips=2, boss_hp=9000, minion_hp=9000)
    for e in s.enemies:
        e.flat_hit = 900               # kills the wizard well inside the horizon

    scores = {c.name: _rollout(sim, s, c, 6, 0) for c in s.hand}
    assert len(set(scores.values())) > 1, scores
    # and the nuke outranks the trap, rather than losing on pip cost
    assert scores["Snow Serpent"] < scores["Ice Trap"], scores


def test_overkill_earns_no_credit():
    """`Sim._strike` does `target.hp -= dmg` uncapped, so a 300 hit into
    a mob with 50 left used to bank 300 -- and the tiebreak rewards
    banking more, which is a scoring function that prefers waste."""
    from deimos_bridge.policies import _rollout

    sim, s, _ = _two_mobs(["Snow Serpent"], boss_hp=9000, minion_hp=20)
    nuke = s.hand[0]
    _, dealt = _rollout(sim, s, nuke, 1, 1)
    assert -dealt == pytest.approx(20.0)      # the mob's health, not the hit's


def test_incoming_damage_is_measured_rather_than_assumed_zero():
    """Enemies built by `read_state` carry no `flat_hit`, so every
    rollout modelled the mobs dealing zero damage: setting up cost
    nothing but turns and the player could never die inside the horizon.
    That is the shape of "it spends the fight setting up and then dies to
    the minion"."""
    be = _backend(policy=lambda sim, s: None)
    hand = [MockCard("Fireblade")]

    def board(hp):
        return MockCombat(
            [MockMember("Wizard", hp, client=True, team_id=0, normal_pips=4,
                        max_health=600),
             MockMember("A", 400, monster=True, team_id=1),
             MockMember("B", 400, monster=True, team_id=1)], hand,
            round_number=board.rnd)
    board.rnd = 1

    be.attach_combat(board(600))
    run(be.decide())
    prior = be.last_read.state.enemies[0].flat_hit
    assert prior > 0                       # never zero, even before a reading

    board.rnd = 2
    be.attach_combat(board(400))           # lost 200 across two mobs
    run(be.decide())
    measured = be.last_read.state.enemies[0].flat_hit

    # It MOVES toward the reading -- 100 per enemy -- rather than staying
    # on the prior...
    assert prior < measured < 100.0
    # ...but does not become it outright. The prior is weighed in for
    # `INCOMING_PRIOR_WEIGHT` rounds, because as a bare fallback the
    # first reading replaced it and a first reading of ZERO therefore
    # set the whole threat model to zero. See
    # `test_one_quiet_round_does_not_make_the_board_harmless`.
    weight = be.INCOMING_PRIOR_WEIGHT
    assert measured == pytest.approx((100.0 + prior * weight) / (1 + weight))


def test_a_cast_that_does_not_take_is_reported_and_the_round_passed():
    """Wizard101 deselects a card clicked at something it cannot be cast
    on. `CombatCard.cast` clicks and returns — it raises nothing — so
    the card comes back to the hand, the round passes with nothing
    played, and the duel sits there until the round timer expires. The
    round had already been recorded as a cast that was made, so next
    round's unchanged health settled as the damage model's worst miss.

    wizwalker checks this for its enchant branch and no other.
    """
    be = _backend(policy=lambda sim, s: "Fireblade")
    told = []
    be.on_failed_cast = told.append

    card = MockCard("Fireblade")
    card.misfires = True                 # clicks, does not take
    combat = MockCombat(
        [MockMember("Wizard", 2000, client=True, normal_pips=4, team_id=0),
         MockMember("Lost Soul", 900, monster=True, team_id=1)],
        [card])
    h = _Handler(combat, be)
    h.CAST_SETTLE, h.CAST_POLL = 0.2, 0.05
    run(h.handle_round())

    assert card.cast_log, "it never even tried"
    assert combat.passed == 1, "the round was left to time out"
    assert told, "the operator was never told the cast did not go through"
    assert "still in hand" in told[0], told


def test_a_cast_that_takes_is_not_reported_as_a_failure():
    """The check must not turn every working cast into a false alarm."""
    be = _backend(policy=lambda sim, s: "Fireblade")
    told = []
    be.on_failed_cast = told.append

    combat = MockCombat(
        [MockMember("Wizard", 2000, client=True, normal_pips=4, team_id=0),
         MockMember("Lost Soul", 900, monster=True, team_id=1)],
        [MockCard("Fireblade")])
    h = _Handler(combat, be)
    h.CAST_SETTLE, h.CAST_POLL = 0.2, 0.05
    run(h.handle_round())

    assert combat._cards[0].cast_log
    assert combat.passed == 0 and told == []


def test_a_cast_that_needs_a_slower_click_gets_one():
    """The two clicks are "select the card" and "click the target",
    `sleep_time` apart. A card the game casts without asking anything
    goes out on the first click; one that has to put the board into
    target selection first needs the UI up before the second click
    lands, and 0.3s may not be long enough. Live, Pixie — a heal, the
    one card in these decks that makes the game ask who — failed twice
    this way while blades, traps and nukes from the same hands went out
    fine."""
    be = _backend(policy=lambda sim, s: "Fireblade")
    slow, failed = [], []
    be.on_slow_cast = slow.append
    be.on_failed_cast = failed.append

    class _SlowCard(MockCard):
        """Only takes when given wizwalker's full pause."""

        async def cast(self, target=None, sleep_time=None, **kw):
            self.cast_log.append((target, sleep_time))
            if sleep_time is not None and sleep_time >= 1.0:
                self.combat.discard(self)
            return True

    card = _SlowCard("Fireblade")
    combat = MockCombat(
        [MockMember("Wizard", 2000, client=True, normal_pips=4, team_id=0),
         MockMember("Lost Soul", 900, monster=True, team_id=1)],
        [card])
    h = _Handler(combat, be)
    h.CAST_SETTLE, h.CAST_POLL = 0.2, 0.05
    run(h.handle_round())

    assert len(card.cast_log) == 2, "it never tried again"
    assert card.cast_log[0][1] == be.cast_time      # the fast one first
    assert card.cast_log[1][1] == h.RETRY_CAST_TIME
    assert slow, "the retry was silent"
    assert "longer pause" in slow[0]
    assert failed == [], "a cast that worked was reported as a failure"
    assert combat.passed == 0, "the round was given away after it worked"


def test_a_card_that_fails_both_ways_says_it_tried_twice():
    """A card that cannot be cast where it was aimed is a different
    problem from one that needed a slower click, and has to read as one."""
    be = _backend(policy=lambda sim, s: "Fireblade")
    slow, failed = [], []
    be.on_slow_cast = slow.append
    be.on_failed_cast = failed.append

    card = MockCard("Fireblade")
    card.misfires = True
    combat = MockCombat(
        [MockMember("Wizard", 2000, client=True, normal_pips=4, team_id=0),
         MockMember("Lost Soul", 900, monster=True, team_id=1)],
        [card])
    h = _Handler(combat, be)
    h.CAST_SETTLE, h.CAST_POLL = 0.2, 0.05
    run(h.handle_round())

    assert len(card.cast_log) == 2
    assert slow, "the retry was silent"
    assert failed and "twice" in failed[0], failed
    assert combat.passed == 1
