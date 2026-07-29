"""wizsprinter's `combat_api` types, or local equivalents off Windows.

`live_backend` has to build `PriorityLine(...)` objects for
`SprintyCombat` to consume. On Windows those types come from wizsprinter
and must be the genuine article -- `SprintyCombat` does `isinstance`
checks against them. Off Windows wizsprinter will not import at all
(wizwalker's `constants.py` runs `ctypes.windll.user32` at module level),
which would make the translation step untestable on any dev machine.

So: use the real types when they are importable, and otherwise define
structurally identical stand-ins. `USING_REAL` says which happened, and
the tests assert the shim's shape matches what wizsprinter expects.
"""
from enum import Enum, auto

USING_REAL = False

try:  # pragma: no cover - only reachable on a Windows box with the game
    from wizwalker.extensions.wizsprinter.combat_backends.combat_api import (  # noqa: F401
        AggregationMode, ComparisonOp, Condition, ConditionTarget, DrawSpell,
        Move, MoveConfig, NamedSpell, Spell, SpellType, TargetData,
        TargetType, TemplateSpell,
    )
    from wizwalker.extensions.wizsprinter.combat_backends.combat_config_parser \
        import PriorityLine  # noqa: F401
    USING_REAL = True

except Exception:
    class TargetType(Enum):
        type_self = auto()
        type_boss = auto()
        type_enemy = auto()
        type_ally = auto()
        type_aoe = auto()
        type_named = auto()
        type_spell = auto()
        type_select = auto()
        type_enemies = auto()
        type_allies = auto()

    class SpellType(Enum):
        type_damage = auto()
        type_aoe = auto()
        type_heal = auto()
        type_heal_self = auto()
        type_blade = auto()
        type_charm = auto()
        type_ward = auto()
        type_trap = auto()
        type_enchant = auto()

    class Spell:
        pass

    class NamedSpell(Spell):
        def __init__(self, name: str, is_literal: bool = False):
            self.name = name
            self.is_literal = is_literal

        def __repr__(self):
            return f'NamedSpell(name="{self.name}", is_literal={self.is_literal})'

    class TemplateSpell(Spell):
        def __init__(self, requirements, optional=False):
            self.requirements = requirements
            self.optional = optional

        def __repr__(self):
            return (f"TemplateSpell(requirements={self.requirements}, "
                    f"optional={self.optional})")

    class Move:
        def __init__(self, card, enchant=None, second_enchant=None):
            self.card = card
            self.enchant = enchant
            self.second_enchant = second_enchant

        def __repr__(self):
            return (f"Move(card={self.card}, enchant={self.enchant}, "
                    f"second_enchant={self.second_enchant})")

    class TargetData:
        def __init__(self, target_type, extra_data=None, is_literal=False):
            self.target_type = target_type
            self.extra_data = extra_data
            self.is_literal = is_literal

        def __repr__(self):
            return (f"TargetData(target_type={self.target_type}, "
                    f"extra_data={self.extra_data}, is_literal={self.is_literal})")

    class MoveConfig:
        def __init__(self, move, target=None, condition=None):
            self.move = move
            self.target = target
            self.condition = condition

        def __repr__(self):
            return f"MoveConfig(move={self.move}, target={self.target})"

    class PriorityLine:
        def __init__(self, priorities, _round=None):
            self.priorities = priorities
            self.round = _round

        def __repr__(self):
            return (f"PriorityLine(priorities={self.priorities}, "
                    f"round={self.round})")
