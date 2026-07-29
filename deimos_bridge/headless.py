"""Import Deimos' combat code on a machine with no Wizard101 and no Windows.

`Deimos/src/effect_simulation.py` is the interesting part of Deimos for
our purposes: a simulation of the client's own effect resolution. It is
written against plain dicts -- `Cache = Dict[str, Any]` -- so nothing in
its arithmetic needs a live game. What stops it importing is scenery:

  * `wizwalker/__init__.py` reaches for `ctypes.windll` at import time,
    which does not exist off Windows, so the whole package is
    unimportable here even though the module we want from it
    (`memory_objects/enums.py`) imports nothing but `enum`.
  * `src/combat_math.py` and friends take `Client` and `CombatMember`
    from that package, but only ever as type annotations on the async
    functions. The sync functions we call never touch them.

So we stand up a stub `wizwalker` whose classes are inert placeholders,
with one exception: `memory_objects.enums` is loaded from Deimos' real
file, because `SpellEffects` members are the actual effect ids the game
uses and getting one wrong would quietly change what a spell does.

Everything under `Deimos/src` is then imported unmodified. The damage
formula that results is Deimos', not a re-implementation of it.

    >>> from deimos_bridge.headless import install
    >>> mods = install()
    >>> mods.effect_simulation.calc_crit(300, 100, 50, 50)[1]
    0.9
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

DEIMOS_ROOT = Path(__file__).resolve().parent.parent / "Deimos"
WIZWALKER_ROOT = DEIMOS_ROOT / "libs" / "wizwalker"
ENUMS_PATH = WIZWALKER_ROOT / "wizwalker" / "memory" / "memory_objects" / "enums.py"


class _Stub:
    """Stand-in for a memory-backed wizwalker class.

    Only ever used as an annotation in the code paths we execute. If one
    is actually constructed we are off the intended path, so say so
    loudly rather than returning a mystery object.
    """

    def __init__(self, *_args, **_kwargs):
        raise RuntimeError(
            f"{type(self).__name__} is a headless stub: it exists so Deimos' "
            "sync combat code can import without Windows. Constructing one "
            "means you reached code that wants a live game client."
        )


def _stub_names(*names):
    return {n: type(n, (_Stub,), {}) for n in names}


def _load_real_module(name: str, path: Path) -> types.ModuleType:
    """Import a single file as `name`, bypassing its package's __init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _extract_functions(path: Path, names) -> dict:
    """Pull named top-level functions out of a file, verbatim, and exec them.

    `src/utils.py` is 58k of Windows-only imports guarding a handful of
    pure helpers. `src/combat_utils.py` wants exactly one of them. Rather
    than re-type it here -- where it could drift from Deimos' version --
    we lift that function's own source and run it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = set(names)
    picked = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    missing = wanted - {node.name for node in picked}
    if missing:
        raise ImportError(f"{path.name} no longer defines {sorted(missing)}")

    namespace: dict = {}
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in wanted}


def _make_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


@dataclass
class DeimosModules:
    """Deimos' real combat modules, imported and ready to call."""

    enums: types.ModuleType
    combat_cache: types.ModuleType
    combat_math: types.ModuleType
    combat_objects: types.ModuleType
    effect_simulation: types.ModuleType


_installed: DeimosModules | None = None


def install(force: bool = False) -> DeimosModules:
    """Make Deimos' combat modules importable, and import them.

    Idempotent: repeated calls hand back the same modules, because
    re-executing them would mint fresh `SpellEffects` enum members that
    compare unequal to the ones already in flight.
    """
    global _installed
    if _installed is not None and not force:
        return _installed

    if not ENUMS_PATH.exists():
        raise ImportError(
            f"Deimos not found at {DEIMOS_ROOT}. This bridge expects the "
            "Deimos subtree to be present in the repository."
        )

    enums = _load_real_module("wizwalker.memory.memory_objects.enums", ENUMS_PATH)

    # The stub package. Attribute wiring matters as much as sys.modules:
    # `from wizwalker.combat import CombatMember` consults sys.modules,
    # but `wizwalker.combat.CombatMember` consults the parent module.
    memory_objects = _make_module(
        "wizwalker.memory.memory_objects",
        enums=enums,
        **_stub_names("MemoryObject", "DynamicMemoryObject"),
    )
    memory = _make_module(
        "wizwalker.memory",
        memory_objects=memory_objects,
        **_stub_names("DynamicWindow", "Window", "HookHandler"),
    )
    combat = _make_module(
        "wizwalker.combat",
        **_stub_names("CombatHandler", "CombatMember", "CombatCard"),
    )
    utils = _make_module(
        "wizwalker.utils",
        **_stub_names("XYZ",),
    )

    async def _never(*_args, **_kwargs):  # pragma: no cover - not on our path
        raise RuntimeError("wizwalker async helpers need a live game client")

    utils.maybe_wait_for_any_value_with_timeout = _never
    utils.maybe_wait_for_value_with_timeout = _never

    wizwalker = _make_module(
        "wizwalker",
        memory=memory,
        combat=combat,
        utils=utils,
        **_stub_names("Client", "ClientHandler", "Keycode", "XYZ"),
    )
    wizwalker.__path__ = []  # marks it a package so submodule imports resolve

    # `spell_effect` and `game_stats` are memory objects; the enums they
    # re-export are the part Deimos' sync code actually reads.
    _make_module(
        "wizwalker.memory.memory_objects.spell_effect",
        SpellEffects=enums.SpellEffects,
        HangingDisposition=enums.HangingDisposition,
        **_stub_names("SpellEffect", "DynamicSpellEffect"),
    )
    _make_module(
        "wizwalker.memory.memory_objects.game_stats",
        **_stub_names("GameStats", "DynamicGameStats"),
    )
    _make_module(
        "wizwalker.memory.memory_objects.combat_participant",
        **_stub_names("CombatParticipant", "DynamicCombatParticipant", "DynamicGameStats"),
    )
    memory_objects.spell_effect = sys.modules["wizwalker.memory.memory_objects.spell_effect"]
    memory_objects.game_stats = sys.modules["wizwalker.memory.memory_objects.game_stats"]
    memory_objects.combat_participant = sys.modules[
        "wizwalker.memory.memory_objects.combat_participant"
    ]

    # `src.utils` drags in win32 and a GUI; `src.combat_utils` wants one
    # pure helper out of it. Lift that helper's own source.
    _make_module("src", __path__=[str(DEIMOS_ROOT / "src")])
    _make_module(
        "src.utils",
        **_extract_functions(DEIMOS_ROOT / "src" / "utils.py", ["index_with_str"]),
    )

    if str(DEIMOS_ROOT) not in sys.path:
        sys.path.insert(0, str(DEIMOS_ROOT))

    # Deimos' own modules, unmodified, from here down.
    import src.combat_cache as combat_cache
    import src.combat_math as combat_math
    import src.combat_objects as combat_objects
    import src.effect_simulation as effect_simulation

    _installed = DeimosModules(
        enums=enums,
        combat_cache=combat_cache,
        combat_math=combat_math,
        combat_objects=combat_objects,
        effect_simulation=effect_simulation,
    )
    return _installed


def deimos_modules() -> DeimosModules:
    """`install()` under a name that reads better at a call site."""
    return install()
