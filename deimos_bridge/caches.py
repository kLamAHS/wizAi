"""Translate wizAi combat objects into the dict shapes Deimos expects.

Deimos never passes objects around its combat simulation. It passes
`Cache` dicts -- snapshots produced by `class_snapshot`, which calls every
zero-argument method on a memory object and files the result under the
method's name. So a participant's stats live at `get_stats.dmg_bonus_percent`
and its blades at `get_participant.hanging_effects`.

That shape is what we reproduce here, from wizAi `Actor`s. Getting it
right matters more than it looks: Deimos indexes every per-school stat
list by `MagicSchoolIndex`, so a list in the wrong order silently reads
another school's resist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deimos_bridge.headless import install

_MODS = install()
_ES = _MODS.effect_simulation
_ENUMS = _MODS.enums

SpellEffects = _ENUMS.SpellEffects
MagicSchool = _ENUMS.MagicSchool
MagicSchoolID = _ES.MagicSchoolID
MagicSchoolIndex = _ES.MagicSchoolIndex

#: wizAi names schools in lowercase; the client identifies them by id.
SCHOOL_ID = {name: MagicSchool[name].value
             for name in ("fire", "ice", "storm", "myth", "life", "death",
                          "balance", "star", "sun", "moon", "shadow")}

#: Length of every per-school stat list Deimos indexes into.
N_SCHOOLS = len(MagicSchoolIndex)


@dataclass(frozen=True)
class DuelCurve:
    """The client's diminishing-returns constants for damage and resist.

    Deimos reads these off the live `Duel` object (offsets 612-632); it
    ships no static copy, so running headless means supplying them. They
    are the one part of this bridge that is **assumed rather than
    measured** -- treat any single-number result that depends on them as
    provisional, and prefer `evaluate.sweep_curve`, which varies the cap
    instead of trusting one setting.

    `curve_stat` needs `k0 + n0 < limit * 100` or it takes the log of a
    non-positive number, which is the shape of the real curve: the soft
    cap has to sit below the hard one.

    Only players are curved. Enemy damage and resist are used raw, in
    Deimos and therefore here too.
    """

    damage_limit: float = 1.40   # hard cap on outgoing damage %, as a fraction
    d_k0: float = 60.0           # where damage starts to bend, in points
    d_n0: float = 25.0
    resist_limit: float = 0.85   # hard cap on incoming resist %
    r_k0: float = 40.0
    r_n0: float = 15.0
    pvp: bool = False
    raid: bool = False

    def __post_init__(self):
        if self.d_k0 + self.d_n0 >= self.damage_limit * 100:
            raise ValueError(
                f"damage soft cap ({self.d_k0} + {self.d_n0}) must sit below "
                f"the hard cap ({self.damage_limit * 100})"
            )
        if self.r_k0 + self.r_n0 >= self.resist_limit * 100:
            raise ValueError(
                f"resist soft cap ({self.r_k0} + {self.r_n0}) must sit below "
                f"the hard cap ({self.resist_limit * 100})"
            )

    def as_cache(self) -> dict:
        """The subset of a duel snapshot that `sim_damage` reads."""
        return {"damage_limit": self.damage_limit, "d_k0": self.d_k0,
                "d_n0": self.d_n0, "resist_limit": self.resist_limit,
                "r_k0": self.r_k0, "r_n0": self.r_n0,
                "pvp": self.pvp, "raid": self.raid}


def _by_school(mapping, default_key="*") -> list:
    """A per-school stat list from a wizAi {school|'*': value} dict."""
    universal = mapping.get(default_key, 0.0) if mapping else 0.0
    out = [float(universal)] * N_SCHOOLS
    for school, value in (mapping or {}).items():
        if school == default_key:
            continue
        idx = MagicSchoolIndex[SCHOOL_ID[school]].value
        out[idx] = float(universal) + float(value)
    return out


def stats_cache(*, damage=None, resist=None, boost=None, pierce=0.0,
                flat_damage=0.0, flat_resist=0.0, crit_rating=0.0,
                block_rating=0.0) -> dict:
    """A `get_stats` snapshot.

    wizAi carries boost as a separate positive multiplier; Deimos has no
    such field and expresses the same thing as negative resist, which its
    `resist <= 0` branch turns back into a damage multiplier. Folding it
    in here keeps the two engines describing the same wizard.
    """
    combined = dict(resist or {})
    for school, value in (boost or {}).items():
        combined[school] = combined.get(school, 0.0) - float(value)

    return {
        "dmg_bonus_percent": _by_school(damage),
        "dmg_bonus_percent_all": 0.0,      # already folded into the list
        "dmg_bonus_flat": [float(flat_damage)] * N_SCHOOLS,
        "dmg_bonus_flat_all": 0.0,
        "dmg_reduce_percent": _by_school(combined),
        "dmg_reduce_percent_all": 0.0,
        "dmg_reduce_flat": [float(flat_resist)] * N_SCHOOLS,
        "dmg_reduce_flat_all": 0.0,
        "ap_bonus_percent": [float(pierce)] * N_SCHOOLS,
        "ap_bonus_percent_all": 0.0,
        "critical_hit_rating_by_school": [float(crit_rating)] * N_SCHOOLS,
        "critical_hit_rating_all": 0.0,
        "block_rating_by_school": [float(block_rating)] * N_SCHOOLS,
        "block_rating_all": 0.0,
    }


def effect_cache(effect_type, param, school="*", spell_template_id=0,
                 enchantment_template_id=0) -> dict:
    """One hanging effect -- a blade, shield, trap, or absorb.

    `spell_template_id` is what Deimos stacks on: two effects sharing one
    are the same spell and the second is ignored. wizAi expresses the
    same rule as a `(name, source)` stack key, so callers should hash
    that key into the id to preserve which effects stack.
    """
    return {
        "effect_type": effect_type,
        "effect_param": param,
        "damage_type": (MagicSchoolID.universal if school in (None, "*")
                        else SCHOOL_ID[school]),
        "spell_template_id": spell_template_id,
        "enchantment_spell_template_id": enchantment_template_id,
    }


def _stack_id(hanging) -> int:
    """Fold a wizAi `(name, source)` stack key into a template id."""
    return abs(hash(hanging.stack_key)) % (2 ** 31)


def member_cache(actor, *, owner_id: int, is_player: bool, level: int = 50,
                 charms=(), wards=()) -> dict:
    """A `CombatMember` snapshot built from a wizAi `Actor`."""
    participant = {f"{prefix}_effects": []
                   for prefix in _ES.hanging_effect_prefixes}
    participant["hanging_effects"] = list(charms) if charms else []
    # Deimos reads the caster's charms and the target's wards from the
    # same key; a member is only ever one of the two in a given call.
    if wards:
        participant["hanging_effects"] = list(wards)

    return {
        "owner_id": owner_id,
        "health": float(actor.hp),
        "max_health": float(actor.max_hp),
        "level": level,
        "is_player": is_player,
        "get_stats": stats_cache(
            damage=actor.damage_bonus, resist=actor.resist, boost=actor.boost,
            pierce=actor.pierce, flat_damage=actor.flat_damage,
            flat_resist=actor.flat_resist),
        "get_participant": participant,
    }


def charms_from_actor(actor, damage_school: str) -> list:
    """wizAi outgoing charms as Deimos effect caches.

    Only the charms that apply to `damage_school` come across; wizAi
    decides applicability from a per-charm school set, Deimos from the
    effect's `damage_type`, so the filter has to happen on this side.
    """
    from w101_sim import buff_applies

    out = []
    for hanging in actor.charms:
        if hanging.schools is not None and damage_school not in hanging.schools:
            continue
        out.append(effect_cache(
            SpellEffects.modify_outgoing_damage,
            round(hanging.percent * 100),
            school="*",
            spell_template_id=_stack_id(hanging)))
    return out


def wards_from_actor(actor, damage_school: str) -> list:
    """wizAi incoming wards as Deimos effect caches.

    A wizAi ward is signed: negative is a shield, positive a trap. Deimos
    uses the same sign convention on `modify_incoming_damage`, so the
    percent carries over directly.
    """
    out = []
    for hanging in actor.wards:
        if hanging.schools is not None and damage_school not in hanging.schools:
            continue
        out.append(effect_cache(
            SpellEffects.modify_incoming_damage,
            round(hanging.percent * 100),
            school="*",
            spell_template_id=_stack_id(hanging)))
    return out
