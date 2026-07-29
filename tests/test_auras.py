"""
Star-school auras: the buff a board wipe cannot take.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import LIVE_RULES, load_spells_full
from deck_builder import legal_pool
from rulesets import CLASSIC, CRIT_ERA
from w101_sim import (Aura, Boss, Hanging, Sim, castable, make_blade_stack)

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))


def _sim(deck, school="myth", rules=LIVE_RULES, **kw):
    b = Boss("dummy", 10**6, "death", 0)
    b.resist_map = {}
    b.boost_map = {}
    return Sim(dict(CARDS), deck, school, b, player_hp=10**6,
               rules=rules, rng=random.Random(0), **kw)


# --- the extraction ---------------------------------------------------

def test_auras_are_their_own_kind_not_blades():
    """Amplify uses kModifyOutgoingDamage — the same effect id a blade
    uses. Only the spell's type separates them, and filing Amplify as a
    blade would make a board wipe strip it."""
    assert CARDS["Amplify"].kind == "aura"
    assert CARDS["Amplify"].ops == [
        {"op": "aura", "field": "outgoing", "percent": 0.15,
         "rounds": 4, "schools": None}]


def test_real_values_come_from_the_dump():
    def op(name, field):
        return next(o for o in CARDS[name].ops if o["field"] == field)
    assert op("Fortify", "incoming")["percent"] == -0.15
    assert op("Berserk", "outgoing")["percent"] == 0.30
    assert op("Berserk", "incoming")["percent"] == 0.40      # take MORE
    assert op("Vengeance", "crit_rating")["percent"] == 0.20
    assert op("Infallible", "accuracy")["percent"] == 0.15
    assert op("Infallible", "pierce")["percent"] == 0.15
    assert CARDS["Amplify"].level == 60 and CARDS["Berserk"].level == 70
    assert CARDS["Brace"].level == 110
    assert all(CARDS[n].pips == 0 for n in
               ("Amplify", "Fortify", "Vengeance", "Berserk"))


def test_unmodelled_auras_are_skipped_not_approximated():
    rep = {}
    load_spells_full(str(ROOT / "spells_full.json"),
                     str(ROOT / "cards_clean.json"), report=rep)
    why = dict((k, v) for k, v in rep["skipped"])
    # Conviction is stun resist + crit block; Empowerment is pip
    # conversion. Neither has an engine field, so neither ships.
    assert "not modeled" in why["Conviction"]
    assert "not modeled" in why["Empowerment"]
    # Punishment's five legs span Balance damage and Death/Life/Myth
    # resist; one Aura carries one school scope
    assert "disagree on school" in why["Punishment"]
    for n in ("Conviction", "Empowerment", "Punishment"):
        assert n not in CARDS


# --- the mechanic -----------------------------------------------------

def test_amplify_is_additive_with_the_gear_damage_stat():
    """150% damage + Amplify is 165%, not 150% x 1.15. The damage stat
    is additive; this is the repo owner's rule and the reason the aura
    does not go through a second multiplier."""
    hits = {}
    for aura in (False, True):
        sim = _sim(["Sunbird"], school="fire",
                   player_stats={"damage": {"*": 0.50}})
        s = sim.new_state()
        if aura:
            s.player.aura = Aura("Amplify", 4, outgoing=0.15)
        foe = s.enemies[0]
        before = foe.hp
        sim.execute_ops(s, s.player, "Sunbird", "deck",
                        CARDS["Sunbird"].ops, "fire")
        hits[aura] = before - foe.hp
    assert abs(hits[True] / hits[False] - 1.65 / 1.50) < 1e-9, hits


def test_fortify_is_additive_with_resist_and_berserk_subtracts():
    taken = {}
    for name, inc in (("none", None), ("Fortify", -0.15),
                      ("Berserk", 0.40)):
        sim = _sim(["Sunbird"], school="fire")
        s = sim.new_state()
        s.player.resist["*"] = 0.20
        if inc is not None:
            s.player.aura = Aura(name, 4, incoming=inc)
        before = s.player.hp
        sim._strike(s, s.enemies[0], s.player, "death", 1000.0, 1.0, 1.0)
        taken[name] = before - s.player.hp
    assert abs(taken["none"] - 800) < 1e-6
    assert abs(taken["Fortify"] - 650) < 1e-6        # 20% + 15% resist
    assert abs(taken["Berserk"] - 1200) < 1e-6       # 20% - 40% = -20%


def test_only_one_aura_at_a_time():
    sim = _sim(["Amplify", "Fortify"], school="fire")
    s = sim.new_state()
    sim.execute_ops(s, s.player, "Amplify", "deck",
                    CARDS["Amplify"].ops, "star")
    assert s.player.aura.name == "Amplify"
    sim.execute_ops(s, s.player, "Fortify", "deck",
                    CARDS["Fortify"].ops, "star")
    assert s.player.aura.name == "Fortify"
    assert s.player.aura.outgoing == 0.0             # replaced, not merged


def test_a_multi_effect_aura_merges_into_one_slot():
    """Berserk is two ops from ONE cast; the second must not replace the
    aura the first just created."""
    sim = _sim(["Berserk"], school="fire")
    s = sim.new_state()
    sim.execute_ops(s, s.player, "Berserk", "deck",
                    CARDS["Berserk"].ops, "star")
    a = s.player.aura
    assert (a.name, a.outgoing, a.incoming) == ("Berserk", 0.30, 0.40)


def test_the_aura_expires():
    sim = _sim(["Amplify"], school="fire")
    s = sim.new_state()
    s.player.aura = Aura("Amplify", 2, outgoing=0.15)
    sim._tick_aura(s, s.player)
    assert s.player.aura is not None and s.player.aura.rounds == 1
    sim._tick_aura(s, s.player)
    assert s.player.aura is None


# --- the reason it exists ---------------------------------------------

def test_earthquake_wipes_the_board_and_leaves_the_aura():
    """The counterplay the repo owner asked for: a myth mob's Earthquake
    strips every charm and ward on the team and cannot touch an aura."""
    sim = _sim(["Earthquake"])
    s = sim.new_state()
    # put the wipe on the PLAYER's side of the board by aiming a
    # player-team Earthquake at a player-team actor
    victim = s.player
    victim.charms.append(Hanging("Mythblade", "charm", "damage",
                                 percent=0.35, source="deck"))
    victim.wards.append(Hanging("Tower Shield", "ward", "damage",
                                percent=-0.5, source="deck"))
    victim.aura = Aura("Amplify", 4, outgoing=0.15)
    sim.execute_ops(s, s.player, "Earthquake", "deck",
                    [{"op": "remove", "what": "charm_all", "tgt": "self"},
                     {"op": "remove", "what": "ward_all", "tgt": "self"}],
                    "myth")
    assert victim.charms == [] and victim.wards == []
    assert victim.aura is not None and victim.aura.outgoing == 0.15


# --- the crit-rating gate ---------------------------------------------

def test_vengeance_rating_is_dropped_under_classic_rules():
    """`crit_chance` holds a PROBABILITY unless a rating resolver is
    installed. Adding Vengeance's rating of 20 to a probability would
    read as +2000%, so it is applied only where a rating means
    something — the same gate the gear layer uses."""
    classic = CLASSIC
    assert classic.crit_resolver is None
    sim = _sim(["Sunbird"], school="fire", rules=classic)
    s = sim.new_state()
    s.player.crit_chance = 0.0
    s.player.aura = Aura("Vengeance", 4, crit_rating=0.20)
    assert all(sim._crit_mult(s.player, s.enemies[0]) == 1.0
               for _ in range(50))


def test_vengeance_rating_lands_under_a_rating_resolver():
    seen = []

    def resolver(caster, target, rng, rules):
        seen.append(caster.crit_chance)
        return 1.0

    import dataclasses
    rules = dataclasses.replace(CRIT_ERA, crit_resolver=resolver)
    sim = _sim(["Sunbird"], school="fire", rules=rules)
    s = sim.new_state()
    s.player.crit_chance = 120.0            # a RATING here
    s.player.aura = Aura("Vengeance", 4, crit_rating=20.0)
    sim._crit_mult(s.player, s.enemies[0])
    assert seen == [140.0]
    assert s.player.crit_chance == 120.0    # restored, not mutated


# --- the pipeline -----------------------------------------------------

def test_the_scripted_pilot_casts_auras():
    """A buff the screen's own policy never casts makes every deck
    holding it score worse and gets it filtered out of the builder —
    that is how Feint went missing. Auras must not repeat it."""
    sim = _sim(["Amplify", "Sunbird", "Sunbird"], school="fire")
    s = sim.new_state()
    s.player.hand = [CARDS["Amplify"], CARDS["Sunbird"]]
    pick = make_blade_stack(3)(sim, s)
    assert pick is not None and pick.name == "Amplify"
    # and it does not re-cast over a live aura: with one up and a
    # castable nuke in hand, the pilot swings instead
    s.player.aura = Aura("Amplify", 4, outgoing=0.15)
    s.player.norm_pips = 7
    assert castable(sim, s, "aura")          # still in hand, still legal
    assert make_blade_stack(3)(sim, s).name == "Sunbird"


def test_auras_unlock_at_the_data_s_levels():
    pool = {lvl: {n for n, c in legal_pool(CARDS, "fire", level=lvl).items()
                  if c.kind == "aura"} for lvl in (50, 60, 70, 110)}
    assert pool[50] == set()
    assert {"Amplify", "Fortify", "Vengeance", "Infallible"} == pool[60]
    assert "Berserk" in pool[70] and "Berserk" not in pool[60]
    assert {"Brace", "Magnify", "Frenzy", "Flawless"} <= pool[110]
