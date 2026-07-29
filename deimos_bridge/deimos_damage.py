"""Deimos's damage formula, ported to run without the game.

A line-for-line transcription of `Deimos/src/combat_math.py`:

    base_damage_calculation_from_id()   -> deimos_damage()
    curve_stat()                        -> curve_stat()
    spell_effect_stacking_id()          -> stacking_id()

The original is `async` and reads every operand out of the live client
(`await caster_stats.dmg_bonus_percent()`, `await effect.effect_param()`,
...). Nothing about the *arithmetic* needs the client, so the port keeps
the arithmetic and takes the operands as plain values. Where the original
reads a stat that the game curves, the curve parameters come in through
`DuelCurve` instead of `await client.duel.damage_limit()`.

Deviations from the original are recorded in DEVIATIONS below rather than
silently smoothed over: the point of this module is to be an *oracle*, so
anywhere it stops being a faithful transcription has to be visible.
"""
from dataclasses import dataclass, field
from enum import Enum
import math

# --------------------------------------------------------------------------
# Ported verbatim from Deimos/src/combat_objects.py:7-11
# --------------------------------------------------------------------------
SCHOOL_IDS = {
    0: 2343174, 1: 72777, 2: 83375795, 3: 2448141, 4: 2330892, 5: 78318724,
    6: 1027491821, 7: 2625203, 8: 78483, 9: 2504141, 10: 663550619,
    11: 1429009101, 12: 1488274711, 13: 1760873841, 14: 806477568,
    15: 931528087,
}
SCHOOL_ID_TO_NAMES = {
    'Fire': 2343174, 'Ice': 72777, 'Storm': 83375795, 'Myth': 2448141,
    'Life': 2330892, 'Death': 78318724, 'Balance': 1027491821,
    'Star': 2625203, 'Sun': 78483, 'Moon': 2504141, 'Gardening': 663550619,
    'Shadow': 1429009101, 'Fishing': 1488274711, 'Cantrips': 1760873841,
    'CastleMagic': 806477568, 'WhirlyBurly': 931528087,
}
SCHOOL_LIST_IDS = {v: k for k, v in SCHOOL_IDS.items()}
SCHOOL_TO_STR = {v: k for k, v in SCHOOL_ID_TO_NAMES.items()}

#: The wildcard damage type Deimos treats as "applies to every school".
#: `combat_math.py:166` and `:190` compare against this literal.
UNIVERSAL_DAMAGE_TYPE = 80289

#: wizAi names schools in lower case; Deimos keys them by game id.
def school_id(name: str) -> int:
    return SCHOOL_ID_TO_NAMES[name.capitalize()]


def school_name(sid: int) -> str:
    return SCHOOL_TO_STR[sid].lower()


class SpellEffects(str, Enum):
    """The subset of the real enum that `base_damage_calculation_from_id`
    branches on. Values are the wizwalker member *names* so a scenario
    reads the same here as it does against the live enum."""
    modify_outgoing_damage = "modify_outgoing_damage"
    modify_outgoing_damage_flat = "modify_outgoing_damage_flat"
    modify_outgoing_armor_piercing = "modify_outgoing_armor_piercing"
    modify_outgoing_damage_type = "modify_outgoing_damage_type"
    modify_incoming_damage = "modify_incoming_damage"
    modify_incoming_damage_flat = "modify_incoming_damage_flat"
    modify_incoming_armor_piercing = "modify_incoming_armor_piercing"
    modify_incoming_damage_type = "modify_incoming_damage_type"
    absorb_damage = "absorb_damage"
    intercept = "intercept"


@dataclass
class Effect:
    """Stands in for `EffectAttributes` (combat_math.py:13-28), which is
    itself Deimos's non-async snapshot of a `DynamicSpellEffect`."""
    effect_type: SpellEffects
    effect_param: float
    damage_type: int
    spell_template_id: int = 0
    enchantment_spell_template_id: int = 0


@dataclass
class Stats:
    """The operands `base_damage_calculation_from_id` pulls off a
    participant's `GameStats`, already universal-adjusted the way
    `real_stat()` (combat_math.py:31-36) does it. Indexed by school id."""
    damage: dict = field(default_factory=dict)        # dmg_bonus_percent
    flat_damage: dict = field(default_factory=dict)   # dmg_bonus_flat
    crit: dict = field(default_factory=dict)          # critical_hit_rating
    pierce: dict = field(default_factory=dict)        # ap_bonus_percent
    resist: dict = field(default_factory=dict)        # dmg_reduce_percent
    flat_resist: dict = field(default_factory=dict)   # dmg_reduce_flat
    block: dict = field(default_factory=dict)         # block_rating
    level: int = 1
    is_player: bool = False

    def get(self, which: str, sid: int) -> float:
        return getattr(self, which).get(sid, 0.0)


@dataclass
class DuelCurve:
    """`client.duel.damage_limit()` / `d_k0()` / `d_n0()` and the resist
    triplet. Defaults disable curving, which is what a duel with all-zero
    curve params does."""
    damage_limit: float = 0.0
    d_k0: float = 0.0
    d_n0: float = 0.0
    resist_limit: float = 0.0
    r_k0: float = 0.0
    r_n0: float = 0.0

    @property
    def curving(self) -> bool:
        return self.damage_limit > 0 or self.resist_limit > 0


#: Places this port knowingly differs from `Deimos/src/combat_math.py`.
DEVIATIONS = [
    "Crit is applied deterministically when `force_crit` is set, matching "
    "the original's `force_crit` branch. The original's *other* trigger -- "
    "firing crit whenever the computed chance is >= 0.85 -- is preserved "
    "but is a display heuristic in Deimos (it is showing the player a "
    "number, not simulating a duel), so callers comparing against a "
    "stochastic engine should pass force_crit explicitly.",
    "`get_total_effects` also folds in aura and shadow effects. The port "
    "takes whatever effect list the caller supplies; scenarios that want "
    "auras pass them as ordinary effects, which is how Deimos sees them "
    "by the time the formula runs.",
]


def curve_stat(stat: float, l: float, k0: float, n0: float) -> float:
    """combat_math.py:39-55, verbatim."""
    if stat > (k0 + n0) / 100:
        limit = l * 100

        if k0 != 0:
            k = math.log(limit / (limit - k0)) / k0
        else:
            k = 1 / limit

        n = math.log(1 - (k0 + n0) / limit) + k * (k0 + n0)

        stat = l - l * math.e ** (-1 * k * (stat * 100) + n)

    return stat


def stacking_id(spell_template_id: int, enchantment_spell_template_id: int) -> tuple:
    """combat_math.py:79-93, verbatim."""
    enchantments_not_stackable_with_unenchanted = {
        655113637,  # Indemity
        85300353,   # Aegis
    }
    if enchantment_spell_template_id in enchantments_not_stackable_with_unenchanted:
        enchantment_spell_template_id = 0
    return (spell_template_id, enchantment_spell_template_id)


def deimos_damage(
    damage: float,
    damage_type: int,
    caster: Stats,
    target: Stats,
    caster_effects: list,
    target_effects: list,
    curve: DuelCurve = None,
    force_crit: bool = False,
) -> float:
    """`base_damage_calculation_from_id` (combat_math.py:96-266).

    `caster_effects` must already be in the order Deimos hands to the
    loop: it reverses the raw list because "Charms use FIFO (queue
    behavior) in game, but the first applied blades show up at the bottom
    of this list" (combat_math.py:106-107). `target_effects` is *not*
    reversed -- "Traps/Shields use LIFO (stack behavior) in game"
    (:111). See `scenarios.py`, which does that ordering for you.
    """
    curve = curve or DuelCurve()

    initial_damage_type = damage_type

    caster_damage = caster.get("damage", initial_damage_type)
    caster_flat_damage = caster.get("flat_damage", initial_damage_type)
    caster_crit = caster.get("crit", initial_damage_type)
    caster_pierce = caster.get("pierce", initial_damage_type)
    caster_level = caster.level

    # Curve damage stats, then apply. (:151-158)
    # `curve_damage` (combat_math.py:58-66) curves only for a player. A
    # zero `damage_limit` means the duel carried no curve data, and the
    # original's `k = 1 / limit` would divide by zero -- so no limit, no
    # curve. That is also the case that makes wizAi and Deimos comparable:
    # wizAi has no curve at all, so an uncurved run isolates every *other*
    # difference. `DuelCurve` with real numbers turns the curve back on.
    if caster.is_player and curve.damage_limit > 0:
        curved_caster_damage = curve_stat(
            caster_damage, curve.damage_limit, curve.d_k0, curve.d_n0)
    else:
        curved_caster_damage = caster_damage
    curved_caster_damage += 1

    damage *= curved_caster_damage
    damage += caster_flat_damage

    # ---- outgoing hanging effects (caster) ------------------------- (:160)
    seen = set()
    for e in caster_effects:
        sid = stacking_id(e.spell_template_id, e.enchantment_spell_template_id)
        if sid in seen:
            continue
        if not (e.damage_type == damage_type or e.damage_type == UNIVERSAL_DAMAGE_TYPE):
            continue
        seen.add(sid)
        if e.effect_type == SpellEffects.modify_outgoing_damage:
            damage *= (e.effect_param / 100) + 1
        elif e.effect_type == SpellEffects.modify_outgoing_damage_flat:
            damage += e.effect_param
        elif e.effect_type == SpellEffects.modify_outgoing_armor_piercing:
            caster_pierce += e.effect_param / 100      # points -> fraction
        elif e.effect_type == SpellEffects.modify_outgoing_damage_type:
            damage_type = e.effect_param

    # ---- incoming hanging effects (target) ------------------------- (:185)
    seen = set()
    for e in target_effects:
        sid = stacking_id(e.spell_template_id, e.enchantment_spell_template_id)
        if sid in seen:
            continue
        if not (e.damage_type == damage_type or e.damage_type == UNIVERSAL_DAMAGE_TYPE):
            continue
        seen.add(sid)
        if e.effect_type == SpellEffects.modify_incoming_damage:
            ward_param = e.effect_param
            if ward_param < 0:
                # ward params are POINTS, pierce is a FRACTION
                ward_param += caster_pierce * 100
                caster_pierce += e.effect_param / 100
                if ward_param > 0:
                    ward_param = 0
                if caster_pierce < 0:
                    caster_pierce = 0
            damage *= (ward_param / 100) + 1
        elif e.effect_type == SpellEffects.intercept:
            damage *= (e.effect_param / 100) + 1
        elif e.effect_type == SpellEffects.modify_incoming_damage_flat:
            damage += e.effect_param
        elif e.effect_type == SpellEffects.absorb_damage:
            damage += e.effect_param
        elif e.effect_type == SpellEffects.modify_incoming_armor_piercing:
            caster_pierce += e.effect_param / 100      # points -> fraction
        elif e.effect_type == SpellEffects.modify_incoming_damage_type:
            damage_type = e.effect_param

    final_damage_type = damage_type

    target_resist = target.get("resist", final_damage_type)
    target_flat_resist = target.get("flat_resist", final_damage_type)
    target_block = target.get("block", final_damage_type)

    if target.is_player and curve.resist_limit > 0:
        curved_target_resist = curve_stat(
            target_resist, curve.resist_limit, curve.r_k0, curve.r_n0)
    else:
        curved_target_resist = target_resist

    # ---- crit ------------------------------------------------------ (:236)
    if caster_crit > 0:
        if caster_level > 100:
            caster_level = 100

        crit_damage_multiplier = (
            2 - (target_block / ((caster_crit / 3) + target_block))
        ) if ((caster_crit / 3) + target_block) else 2.0
        client_school_critical = (0.03 * caster_level * caster_crit)
        mob_block = (3 * caster_crit + target_block)
        crit_chance = client_school_critical / mob_block if mob_block else 0.0

        if (crit_chance >= 0.85 and force_crit is None) or force_crit:
            damage *= crit_damage_multiplier

    # ---- flat resist, then resist ---------------------------------- (:253)
    damage -= target_flat_resist
    damage = abs(damage)

    if curved_target_resist > 0:
        curved_target_resist -= caster_pierce
        if curved_target_resist <= 0:
            curved_target_resist = 1
        else:
            curved_target_resist = 1 - curved_target_resist
    else:
        curved_target_resist = abs(curved_target_resist) + 1

    damage *= curved_target_resist

    return damage
