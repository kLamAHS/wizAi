"""Tests for the Deimos bridge.

Three things are worth pinning here, in order of how quietly they would
break:

  1. The bug fixes in `Deimos/src/effect_simulation.py`. That module had
     never executed -- it raised at import -- so nothing downstream of it
     had ever been observed. Each test below names the defect it guards.
  2. The headless shim. It works by lying to the import system, so it
     should fail loudly rather than hand back a half-built module.
  3. The curve splice, which is the whole experiment. If `DeimosSim`
     stopped differing from `Sim`, every result would still compute and
     would silently read "no transfer gap".
"""

import copy
import math

import pytest

from deimos_bridge.caches import (DuelCurve, SCHOOL_ID, effect_cache,
                                  member_cache, stats_cache)
from deimos_bridge.headless import install

MODS = install()
ES = MODS.effect_simulation
CM = MODS.combat_math
SE = MODS.enums.SpellEffects
FIRE = MODS.enums.MagicSchool.fire.value
ICE = MODS.enums.MagicSchool.ice.value


# ------------------------------------------------------------- fixtures

def _member(owner, hp, is_player, effects=(), level=50, **stat_kwargs):
    participant = {f"{p}_effects": [] for p in ES.hanging_effect_prefixes}
    participant["hanging_effects"] = list(effects)
    return {"owner_id": owner, "health": float(hp), "level": level,
            "is_player": is_player, "get_stats": stats_cache(**stat_kwargs),
            "get_participant": participant}


def _effect(kind, param, school=FIRE, tid=1, ench=0):
    return {"effect_type": kind, "effect_param": param, "damage_type": school,
            "spell_template_id": tid, "enchantment_spell_template_id": ench}


@pytest.fixture
def duel():
    return DuelCurve().as_cache()


# --------------------------------------------- effect_simulation defects

def test_module_imports_at_all():
    """`MagicSchoolID(MagicSchool)` raised TypeError; the module was dead."""
    assert ES.MagicSchoolID.universal.value == 80289
    assert ES.MagicSchoolID.fire.value == FIRE


def test_magic_school_id_compares_to_raw_client_ints():
    """The `damage_type != universal` guards see ints, not enum members."""
    assert 80289 == ES.MagicSchoolID.universal
    assert FIRE == ES.MagicSchoolID.fire


def test_school_index_defined_and_ordered_like_the_stat_lists():
    """The members were attribute assignments, so the enum had none.

    The order is load-bearing: it indexes the per-school stat lists, and
    must match `combat_objects.school_ids`.
    """
    expected = ["fire", "ice", "storm", "myth", "life", "death", "balance"]
    assert [m.name for m in ES.MagicSchoolIndex][:7] == expected
    for index, school_id in list(MODS.combat_objects.school_ids.items())[:7]:
        assert ES.MagicSchoolIndex[school_id].value == index


def test_school_index_accepts_id_member_and_name():
    assert ES.MagicSchoolIndex[FIRE].value == 0
    assert ES.MagicSchoolIndex[MODS.enums.MagicSchool.storm].value == 2
    assert ES.MagicSchoolIndex["ice"].value == 1
    with pytest.raises(KeyError):
        ES.MagicSchoolIndex[123456]


def test_outgoing_effects_do_not_raise_on_the_first_charm():
    """`ids` was read by the guard one line before it was assigned."""
    caster = _member(1, 1000, True, [_effect(SE.modify_outgoing_damage, 35)])
    _cache, school, damage, pierce = ES.sim_outgoing_dmg_effects(
        caster, FIRE, 100.0, 0.0)
    assert damage == pytest.approx(135.0)
    assert school == FIRE and pierce == 0.0


def test_outgoing_effects_return_the_cache_not_the_type_alias():
    """`result_cache = Cache` returned `Dict[str, Any]`, the annotation."""
    caster = _member(1, 1000, True, [_effect(SE.modify_outgoing_damage, 35)])
    cache, *_ = ES.sim_outgoing_dmg_effects(caster, FIRE, 100.0, 0.0)
    assert isinstance(cache, dict)
    assert "get_participant" in cache


def test_outgoing_heal_effects_return_a_value():
    """The function declared a return type and had no return statement."""
    caster = _member(1, 1000, True, [_effect(SE.modify_outgoing_heal, 50)])
    result = ES.sim_outgoing_heal_effects(caster, FIRE, 100.0)
    assert result is not None
    _cache, healed = result
    assert healed == pytest.approx(150.0)


def test_identical_spells_do_not_stack():
    """The dedup key included the list index, so it never matched itself.

    Two copies of one blade are one blade; the game refuses the second.
    """
    same = [_effect(SE.modify_outgoing_damage, 35, tid=7),
            _effect(SE.modify_outgoing_damage, 35, tid=7)]
    caster = _member(1, 1000, True, same)
    _c, _s, damage, _p = ES.sim_outgoing_dmg_effects(caster, FIRE, 100.0, 0.0)
    assert damage == pytest.approx(135.0)


def test_distinct_spells_do_stack():
    both = [_effect(SE.modify_outgoing_damage, 35, tid=7),
            _effect(SE.modify_outgoing_damage, 20, tid=9)]
    caster = _member(1, 1000, True, both)
    _c, _s, damage, _p = ES.sim_outgoing_dmg_effects(caster, FIRE, 100.0, 0.0)
    assert damage == pytest.approx(100.0 * 1.35 * 1.20)


def test_absorb_soaks_damage_instead_of_adding_it(duel):
    """`damage += param` made a shield heal the incoming attack."""
    caster = _member(1, 2000, True)
    target = _member(2, 1000, False,
                     [_effect(SE.absorb_damage, 40, school=ES.MagicSchoolID.universal)])
    _c, _t, dealt = ES.sim_damage(duel, caster, target,
                                  _effect(SE.damage, 85), crit_threshold=1.1)
    assert dealt == pytest.approx(45.0)


def test_partial_absorb_leaves_the_remainder_on_the_ward(duel):
    caster = _member(1, 2000, True)
    target = _member(2, 1000, False,
                     [_effect(SE.absorb_damage, 200, school=ES.MagicSchoolID.universal)])
    _c, result, dealt = ES.sim_damage(duel, caster, target,
                                      _effect(SE.damage, 85), crit_threshold=1.1)
    assert dealt == 0.0
    remaining = result["get_participant"]["hanging_effects"][0]["effect_param"]
    assert remaining == pytest.approx(115.0)


def test_crit_with_no_rating_does_not_divide_by_zero():
    """The ordinary low-level case: no crit gear, no enemy block."""
    assert ES.calc_crit(0.0, 0.0, 20, 20) == (1.0, 0.0, 0.0)


def test_crit_scales_with_rating():
    _mult, chance, _block = ES.calc_crit(300.0, 100.0, 50, 50)
    assert 0.0 < chance <= 0.95


def test_block_rating_is_read_from_the_right_key(duel):
    """`get_stat.` (singular) returned None and the subscript raised."""
    caster = _member(1, 2000, True, crit_rating=300.0)
    target = _member(2, 1000, False, block_rating=100.0)
    _c, _t, dealt = ES.sim_damage(duel, caster, target, _effect(SE.damage, 85))
    assert dealt > 0


def test_variable_effect_takes_pips_as_an_argument():
    """`pips` was read without being defined; every X-pip spell raised."""
    subeffects = [{"pip_num": n, "effect_param": n * 100} for n in (1, 2, 3)]
    chosen = ES.collapse_effect(subeffects, "VariableSpellEffect", {}, {}, pips=2)
    assert chosen["pip_num"] == 2


# ------------------------------------------------------- damage pipeline

def test_damage_pipeline_matches_hand_arithmetic(duel):
    """85 base, +30% gear, one +35% blade."""
    caster = _member(1, 2000, True, [_effect(SE.modify_outgoing_damage, 35)],
                     damage={"fire": 0.30})
    target = _member(2, 1000, False)
    _c, result, dealt = ES.sim_damage(duel, caster, target,
                                      _effect(SE.damage, 85), crit_threshold=1.1)
    assert dealt == pytest.approx(85 * 1.30 * 1.35, rel=1e-6)
    assert result["health"] == pytest.approx(1000 - dealt)


def test_shield_halves_the_hit(duel):
    caster = _member(1, 2000, True, damage={"fire": 0.30})
    target = _member(2, 1000, False, [_effect(SE.modify_incoming_damage, -50)])
    _c, _t, dealt = ES.sim_damage(duel, caster, target,
                                  _effect(SE.damage, 85), crit_threshold=1.1)
    assert dealt == pytest.approx(85 * 1.30 * 0.50, rel=1e-6)


def test_resist_reduces_and_pierce_restores(duel):
    plain = _member(1, 2000, True)
    piercing = _member(1, 2000, True, pierce=0.40)
    target = _member(2, 1000, False, resist={"fire": 0.40})
    spell = _effect(SE.damage, 100)
    _c, _t, without = ES.sim_damage(duel, plain, copy.deepcopy(target), spell,
                                    crit_threshold=1.1)
    _c, _t, with_pierce = ES.sim_damage(duel, piercing, copy.deepcopy(target),
                                        spell, crit_threshold=1.1)
    assert without == pytest.approx(60.0)
    assert with_pierce > without


def test_health_never_goes_negative(duel):
    caster = _member(1, 2000, True)
    target = _member(2, 10, False)
    _c, result, _d = ES.sim_damage(duel, caster, target,
                                   _effect(SE.damage, 5000), crit_threshold=1.1)
    assert result["health"] == 0


# ---------------------------------------------------------- stat curves

def test_curve_is_a_no_op_below_the_bend():
    curve = DuelCurve()
    bend = (curve.d_k0 + curve.d_n0) / 100
    assert CM.curve_stat(bend - 0.1, curve.damage_limit, curve.d_k0,
                         curve.d_n0) == pytest.approx(bend - 0.1)


def test_curve_bites_above_the_bend():
    curve = DuelCurve()
    raw = 1.4715                     # what gear.loadout gives a level-120 fire
    curved = CM.curve_stat(raw, curve.damage_limit, curve.d_k0, curve.d_n0)
    assert curved < raw
    assert curved < curve.damage_limit


def test_curve_is_monotone_and_bounded():
    curve = DuelCurve()
    values = [CM.curve_stat(x / 20, curve.damage_limit, curve.d_k0, curve.d_n0)
              for x in range(0, 80)]
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))
    assert all(v <= curve.damage_limit + 1e-9 for v in values)


def test_curve_constants_must_be_self_consistent():
    """A soft cap above the hard cap makes `curve_stat` take log of <= 0."""
    with pytest.raises(ValueError):
        DuelCurve(damage_limit=0.5, d_k0=60.0, d_n0=25.0)
    with pytest.raises(ValueError):
        DuelCurve(resist_limit=0.3, r_k0=40.0, r_n0=15.0)


# ------------------------------------------------------- cache building

def test_stat_lists_are_indexed_by_school():
    stats = stats_cache(damage={"fire": 0.5, "ice": 0.1})
    lst = stats["dmg_bonus_percent"]
    assert lst[ES.MagicSchoolIndex[SCHOOL_ID["fire"]].value] == pytest.approx(0.5)
    assert lst[ES.MagicSchoolIndex[SCHOOL_ID["ice"]].value] == pytest.approx(0.1)
    assert lst[ES.MagicSchoolIndex[SCHOOL_ID["storm"]].value] == 0.0


def test_universal_damage_applies_to_every_school():
    stats = stats_cache(damage={"*": 0.2, "fire": 0.5})
    lst = stats["dmg_bonus_percent"]
    assert lst[ES.MagicSchoolIndex[SCHOOL_ID["storm"]].value] == pytest.approx(0.2)
    assert lst[ES.MagicSchoolIndex[SCHOOL_ID["fire"]].value] == pytest.approx(0.7)


def test_boost_is_folded_in_as_negative_resist():
    """Deimos has no boost field; it reads boost as resist below zero."""
    stats = stats_cache(boost={"fire": 0.25})
    idx = ES.MagicSchoolIndex[SCHOOL_ID["fire"]].value
    assert stats["dmg_reduce_percent"][idx] == pytest.approx(-0.25)


def test_effect_cache_marks_any_school_as_universal():
    assert effect_cache(SE.modify_outgoing_damage, 35,
                        school="*")["damage_type"] == ES.MagicSchoolID.universal
    assert effect_cache(SE.modify_outgoing_damage, 35,
                        school="fire")["damage_type"] == SCHOOL_ID["fire"]


# ------------------------------------------------------------ the shim

def test_shim_is_idempotent():
    """Re-executing would mint enum members unequal to the ones in use."""
    assert install() is install()
    assert install().effect_simulation is ES


def test_stub_classes_refuse_to_be_constructed():
    """A stub reached at runtime means we left the offline code path."""
    import wizwalker
    with pytest.raises(RuntimeError, match="headless stub"):
        wizwalker.Client()


def test_real_enums_not_stubs():
    """The effect ids must be the client's, not something we invented."""
    assert ES.SpellEffects.damage.value == 1
    assert ES.SpellEffects.modify_outgoing_damage.value == 28
    assert ES.SpellEffects.absorb_damage.value == 38


# --------------------------------------------------- the curve splice

@pytest.fixture(scope="module")
def fixture_fight():
    from data_full import LIVE_RULES, load_bosses_full, load_spells_full
    from deck_builder import legal_pool
    from gear import loadout

    cards = load_spells_full()
    bosses, _registry = load_bosses_full(rules=LIVE_RULES, spell_pools=True,
                                         cards=cards, fill_boost=True)
    boss = next(b for b in bosses.values() if b.rank and 3000 < b.hp < 9000)
    pool = legal_pool(cards, "fire", level=120)
    hits = sorted((c for c in pool.values()
                   if c.kind == "damage" and not c.x_pips),
                  key=lambda c: -c.damage)[:2]
    blades = sorted((c for c in pool.values() if c.kind == "blade"),
                    key=lambda c: -c.percent)[:1]
    decklist = [c.name for c in hits] * 4 + [c.name for c in blades] * 4
    stats = loadout(120, "fire", rules=LIVE_RULES)
    return cards, decklist, boss, stats, LIVE_RULES


def _run(sim_cls, fixture, n=120, **kwargs):
    from w101_sim import evaluate_paired, make_blade_stack
    cards, decklist, boss, stats, rules = fixture
    sim = sim_cls(dict(cards), decklist, "fire", copy.copy(boss),
                  player_hp=stats["player_hp"], rules=rules,
                  player_stats=stats["player_stats"],
                  power_pip=stats["power_pip"], **kwargs)
    return evaluate_paired(sim, {"race": make_blade_stack(2)}, n=n)["race"]


def test_uncapped_curve_reproduces_wizai(fixture_fight):
    """The control. A cap far above any real stat must change nothing.

    If this drifts, the splice is wrong and every other number here is
    measuring a bug rather than the curve.
    """
    from w101_sim import Sim
    from deimos_bridge.engine import DeimosSim

    base = _run(Sim, fixture_fight)
    linear = _run(DeimosSim, fixture_fight,
                  curve=DuelCurve(damage_limit=999.0, resist_limit=999.0))
    assert linear["mean_ttk"] == pytest.approx(base["mean_ttk"], rel=1e-9)
    assert linear["win_rate"] == pytest.approx(base["win_rate"])


def test_curve_costs_the_high_level_wizard_damage(fixture_fight):
    """At level 120 the gear sheet is far above the bend, so it must bite."""
    from w101_sim import Sim
    from deimos_bridge.engine import DeimosSim

    base = _run(Sim, fixture_fight)
    curved = _run(DeimosSim, fixture_fight, curve=DuelCurve())
    assert curved["mean_ttk"] > base["mean_ttk"]


def test_enemies_are_not_curved_by_default(fixture_fight):
    """Deimos curves players only; the asymmetry is deliberate.

    An enemy with a damage bonus far above the bend must still hit for
    its full, uncurved value.
    """
    from w101_sim import Actor
    from deimos_bridge.engine import DeimosSim

    cards, decklist, boss, stats, rules = fixture_fight
    sim = DeimosSim(dict(cards), decklist, "fire", copy.copy(boss),
                    player_hp=stats["player_hp"], rules=rules,
                    player_stats=stats["player_stats"],
                    power_pip=stats["power_pip"], curve=DuelCurve())

    loaded = {"fire": 1.50}
    enemy = Actor(name="e", school="fire", hp=100, max_hp=100, team=1,
                  damage_bonus=loaded)
    player = Actor(name="p", school="fire", hp=100, max_hp=100, team=0,
                   damage_bonus=loaded)

    assert not sim._curved(enemy)
    assert sim._curved(player)
    assert sim._damage_bonus(enemy, "fire") == pytest.approx(2.50)
    assert sim._damage_bonus(player, "fire") < 2.50


def test_curve_enemies_flag_makes_the_model_symmetric(fixture_fight):
    from w101_sim import Actor
    from deimos_bridge.engine import DeimosSim

    cards, decklist, boss, stats, rules = fixture_fight
    sim = DeimosSim(dict(cards), decklist, "fire", copy.copy(boss),
                    player_hp=stats["player_hp"], rules=rules,
                    player_stats=stats["player_stats"],
                    power_pip=stats["power_pip"], curve=DuelCurve(),
                    curve_enemies=True)
    enemy = Actor(name="e", school="fire", hp=100, max_hp=100, team=1,
                  damage_bonus={"fire": 1.50})
    assert sim._curved(enemy)
    assert sim._damage_bonus(enemy, "fire") < 2.50
