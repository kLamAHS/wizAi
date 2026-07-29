"""Tests for the live-combat adapter.

The client cannot be reached from here, so what is tested is everything
between reading it and casting: the translation from a flattened round
into a wizAi `State`, and the choice that comes back out. Those are the
parts that can be wrong silently. A misread hand does not crash -- it
just makes the policy play a different game than the one on screen.
"""

import pytest

from w101_sim import Actor, Card, State
from deimos_bridge.live import (LiveEffect, LiveMember, LiveRound, build_state,
                                choose, estimate_deck, resolve_hand, to_actor)


def _card(name, kind="damage", pips=3, damage=100.0, percent=0.0):
    return Card(name=name, school="fire", pips=pips, accuracy=0.85, kind=kind,
                damage=damage, percent=percent,
                ops=[{"op": "hit", "amount": damage, "tgt": "enemy", "group": 0}]
                if kind == "damage" else
                [{"op": "charm", "percent": percent, "schools": "own",
                  "tgt": "self"}])


CARDS = {"Fire Cat": _card("Fire Cat"),
         "Fire Elf": _card("Fire Elf", damage=170.0),
         "Fireblade": _card("Fireblade", kind="blade", pips=0, damage=0.0,
                            percent=0.35)}


def _member(name="Wizard", hp=2000, max_hp=2500, **kw):
    return LiveMember(name=name, school="fire", health=hp, max_health=max_hp, **kw)


# ------------------------------------------------------------- to_actor

def test_actor_carries_health_and_pips():
    actor = to_actor(_member(normal_pips=2, power_pips=3), team=0)
    assert actor.hp == 2000 and actor.max_hp == 2500
    assert actor.norm_pips == 2 and actor.pow_pips == 3
    assert actor.team == 0


def test_outgoing_effects_become_charms_incoming_become_wards():
    member = _member(effects=[
        LiveEffect("modify_outgoing_damage", 35, spell_name="Fireblade"),
        LiveEffect("modify_incoming_damage", -50, spell_name="Fire Shield"),
    ])
    actor = to_actor(member, team=0)
    assert [h.name for h in actor.charms] == ["Fireblade"]
    assert [h.name for h in actor.wards] == ["Fire Shield"]
    assert actor.charms[0].percent == pytest.approx(0.35)
    assert actor.wards[0].percent == pytest.approx(-0.50)


def test_a_negative_outgoing_effect_is_a_weakness_not_a_shield():
    """Split by direction, not by sign -- the client distinguishes them."""
    member = _member(effects=[
        LiveEffect("modify_outgoing_damage", -25, spell_name="Weakness")])
    actor = to_actor(member, team=0)
    assert [h.name for h in actor.charms] == ["Weakness"]
    assert actor.wards == []


def test_unknown_effect_types_are_ignored_not_guessed():
    member = _member(effects=[LiveEffect("stun_block", 1, spell_name="?")])
    actor = to_actor(member, team=0)
    assert actor.charms == [] and actor.wards == []


def test_stun_carries_over():
    assert to_actor(_member(stunned=True), team=1).stunned == 1
    assert to_actor(_member(stunned=False), team=1).stunned == 0


# ---------------------------------------------------------- resolve_hand

def test_hand_resolves_known_cards():
    hand, unmatched = resolve_hand(["Fire Cat", "Fireblade"], CARDS)
    assert [c.name for c in hand] == ["Fire Cat", "Fireblade"]
    assert unmatched == []


def test_unknown_card_names_are_reported_not_dropped_silently():
    hand, unmatched = resolve_hand(["Fire Cat", "Frostbite"], CARDS)
    assert [c.name for c in hand] == ["Fire Cat"]
    assert unmatched == ["Frostbite"]


def test_duplicate_cards_are_kept_as_separate_slots():
    hand, _ = resolve_hand(["Fire Cat", "Fire Cat"], CARDS)
    assert len(hand) == 2


# ---------------------------------------------------------- estimate_deck

def test_deck_estimate_subtracts_what_has_been_seen():
    decklist = ["Fire Cat", "Fire Cat", "Fire Elf", "Fireblade"]
    remaining = estimate_deck(decklist, CARDS, seen=["Fire Cat", "Fireblade"])
    assert sorted(c.name for c in remaining) == ["Fire Cat", "Fire Elf"]


def test_deck_estimate_never_goes_negative():
    remaining = estimate_deck(["Fire Cat"], CARDS,
                              seen=["Fire Cat", "Fire Cat", "Fire Cat"])
    assert remaining == []


def test_deck_estimate_ignores_cards_with_no_record():
    remaining = estimate_deck(["Fire Cat", "Mystery"], CARDS, seen=[])
    assert [c.name for c in remaining] == ["Fire Cat"]


# ----------------------------------------------------------- build_state

def _round(**kw):
    defaults = dict(
        client=_member(normal_pips=4),
        enemies=[_member(name="Rattlebones", hp=800, max_hp=800)],
        allies=[],
        hand=["Fire Cat", "Fireblade"],
        round_number=3)
    defaults.update(kw)
    return LiveRound(**defaults)


def test_state_is_a_real_wizai_state():
    """Reusing wizAi's own class means its own rules apply unchanged."""
    state = build_state(_round(), CARDS, ["Fire Cat", "Fireblade"])
    assert isinstance(state, State)
    assert state.player_hp == 2000
    assert state.boss_hp == 800
    assert state.norm_pips == 4
    assert state.turn == 3


def test_state_exposes_the_features_the_policy_reads():
    """The featurizer reads blades, traps, hand and deck off the state."""
    live = _round(client=_member(normal_pips=4, effects=[
        LiveEffect("modify_outgoing_damage", 35, spell_name="Fireblade")]))
    state = build_state(live, CARDS, ["Fire Cat", "Fire Elf", "Fireblade"])
    assert "Fireblade" in state.blades
    assert [c.name for c in state.hand] == ["Fire Cat", "Fireblade"]
    assert state.living_enemies()


def test_multiple_enemies_are_all_present_and_ordered():
    live = _round(enemies=[_member(name="A", hp=100),
                           _member(name="B", hp=200),
                           _member(name="C", hp=0)])
    state = build_state(live, CARDS, [])
    assert [e.name for e in state.enemies] == ["A", "B", "C"]
    assert [e.name for e in state.living_enemies()] == ["A", "B"]


def test_dead_enemies_are_excluded_from_living():
    live = _round(enemies=[_member(name="Dead", hp=0)])
    assert build_state(live, CARDS, []).living_enemies() == []


def test_allies_land_on_the_player_team():
    live = _round(allies=[_member(name="Friend")])
    state = build_state(live, CARDS, [])
    assert [a.name for a in state.allies] == ["Friend"]
    assert state.allies[0].team == 0


# ---------------------------------------------------------------- choose

class _Sim:
    player_hp0 = 10 ** 6


def test_choose_unwraps_a_bare_card():
    name, target = choose(lambda sim, s: CARDS["Fire Cat"], _Sim(), None)
    assert name == "Fire Cat" and target == 0


def test_choose_unwraps_an_action_with_a_target():
    from w101_sim import Action
    action = Action(card=CARDS["Fire Elf"], target=2)
    name, target = choose(lambda sim, s: action, _Sim(), None)
    assert name == "Fire Elf" and target == 2


def test_choose_reads_none_as_pass():
    assert choose(lambda sim, s: None, _Sim(), None) == (None, 0)


def test_choose_reads_an_empty_action_as_pass():
    from w101_sim import Action
    assert choose(lambda sim, s: Action(card=None), _Sim(), None) == (None, 0)


# ------------------------------------------------- end-to-end, no client

def test_a_trained_policy_can_drive_a_live_round():
    """The whole translation path, with a real policy and no game.

    Reads a synthetic round, builds the state, runs a wizAi policy over
    it, and gets back a card the hand actually contains.
    """
    from w101_sim import Boss, Sim, make_blade_stack

    decklist = ["Fire Cat"] * 4 + ["Fireblade"] * 3
    boss = Boss(name="live", hp=10 ** 6, school="fire", dmg=0)
    sim = Sim(dict(CARDS), decklist, "fire", boss, player_hp=10 ** 6)

    state = build_state(_round(), CARDS, decklist)
    name, target = choose(make_blade_stack(1), sim, state)

    assert name in {c.name for c in state.hand}
    assert target == 0
