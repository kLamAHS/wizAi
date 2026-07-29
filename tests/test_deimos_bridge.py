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
from deimos_bridge.differential import EXPECTED, TOL, compare, deimos_ruleset
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


def test_adopting_deimos_flat_placement_closes_the_flat_divergences():
    rows = {r["scenario"]: r for r in compare(rules=deimos_ruleset())}
    for name in ("flat damage", "flat resist", "full stack (no pierce)"):
        assert rows[name]["agree"], (name, rows[name])


def test_the_residual_gap_is_exactly_pierce():
    """The matched pair. Same board twice, pierce the only difference: the
    row without it agrees, the row with it does not. That localises the
    entire remaining divergence to Deimos's pierce unit bug rather than
    leaving it as an unexplained delta."""
    rows = {r["scenario"]: r for r in compare(rules=deimos_ruleset())}
    assert rows["full stack (no pierce)"]["agree"]
    assert not rows["full stack"]["agree"]


def test_every_divergence_is_either_a_finding_or_explained():
    """No silent disagreements: a row that differs must be listed in
    EXPECTED with a reason, or be one of the two known findings."""
    findings = {"flat damage", "flat resist", "full stack (no pierce)"}
    for r in compare():
        if not r["agree"]:
            assert r["scenario"] in EXPECTED or r["scenario"] in findings, \
                f"unexplained divergence: {r}"


def test_wizai_pierce_beats_deimos_on_shields():
    """wizAi is the correct one here, and the test says so in numbers: a
    20% pierce should move a -50 shield to -30, not to -49.8."""
    sc = Scenario("probe", 500, "fire",
                  caster=Side(pierce=0.20, is_player=True),
                  wards=[Buff("Tower Shield", -50, None)])
    assert sc.wizai_damage() == pytest.approx(350.0)   # 500 * 0.70
    assert sc.deimos_damage() == pytest.approx(251.0)  # 500 * 0.502


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


def test_backend_understands_a_targeted_action():
    """`rl_agent` emits 'name@i' on a multi-enemy board."""
    be = _backend(policy=lambda sim, s: "Fireblade@1")
    me = MockMember("Wizard", 2000, client=True, normal_pips=4)
    foes = [MockMember("A", 500, monster=True), MockMember("B", 700, monster=True)]
    be.attach_combat(MockCombat([me] + foes, [MockCard("Fireblade")]))
    d = run(be.decide())
    assert d.card_name == "Fireblade"
    assert d.target_index == 1


def test_backend_records_history():
    be = _backend()
    be.attach_combat(simple_fight())
    run(be.decide())
    run(be.decide())
    assert len(be.history) == 2


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
