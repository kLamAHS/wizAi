"""What the catalog already knows about the mob you just walked into.

`bosses_clean.json` carries 1,912 scraped creatures -- school, rank,
health, and for the ones that cheat, the cheat list and human-written
notes. The live run reads names off the client every round and never
once looked them up, so a wizard could walk into Lord Nightshade with
the answer sitting in a file. The simulator cannot *model* an arbitrary
cheat, but the operator can be told, which is the difference between a
surprise interrupt and a known one.

Matching is exact-name first. The catalog keeps tier variants
("Lord Nightshade (Tier 3)") beside the base name; when several share a
base, the observed max health picks the closest tier, because the tiers
differ by health more reliably than by anything else.
"""
import copy
import json
import os
import re
import unicodedata

_INDEX = None
_FULL = None


def _normal(name):
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _base(name):
    """'Lord Nightshade (Tier 3)' -> 'lord nightshade'."""
    return _normal(re.sub(r"\s*\((tier [^)]*|standard|rematch|challenge)\)\s*$",
                          "", str(name or ""), flags=re.I))


def _load():
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "bosses_clean.json")
    by_exact, by_base = {}, {}
    try:
        for rec in json.load(open(path, encoding="utf-8")):
            if not rec.get("name"):
                continue
            by_exact.setdefault(_normal(rec["name"]), rec)
            by_base.setdefault(_base(rec["name"]), []).append(rec)
    except Exception:
        pass          # no catalog is a quiet no-op, never a crash
    _INDEX = (by_exact, by_base)
    return _INDEX


def lookup(name, max_hp=None):
    """The catalog record for a live enemy name, or None.

    `max_hp` disambiguates tier variants: the entry whose scraped health
    is nearest the observed maximum is the fight actually in progress.
    """
    by_exact, by_base = _load()
    rec = by_exact.get(_normal(name))
    variants = by_base.get(_base(name)) or []
    if max_hp and len(variants) > 1:
        # The observed health outranks the exact name: the client says
        # "Lord Nightshade" for every tier, and the 13,200 HP one is not
        # the 690 HP one in any way that matters.
        return min(variants,
                   key=lambda r: abs((r.get("health") or 0) - max_hp))
    if rec is not None:
        return rec
    return variants[0] if variants else None


def cheat_warning(name, max_hp=None):
    """One line worth interrupting the status bar for, or ''."""
    rec = lookup(name, max_hp)
    if not rec or not rec.get("has_cheats"):
        return ""
    notes = [n for n in (rec.get("cheat_notes") or []) if n]
    detail = notes[0] if notes else ", ".join(
        (rec.get("cheats") or [])[:3]) or "cheats (no notes scraped)"
    return f"⚠ {rec['name']} cheats: {detail}"


_SCHOOLS = ("fire", "ice", "storm", "myth", "life", "death", "balance")


def _note_schools(note):
    """'to [Myth][Life][Death' -> ['myth', 'life', 'death']."""
    return [sc for sc in _SCHOOLS if sc in str(note or "").lower()]


def _stat_value(v):
    """(number, schools-text) from the scrape's three shapes.

    `{"value": 50, "note": "to [Death"}` is the common one, but 44
    resists are a bare `60` (no note -- universal), and 34 more are
    strings like `"+25 to [Ice][Myth]"` that carry both the number and
    the schools. Strings with no number at all ("?? to [Storm]",
    "Immune to [Global]") stay dropped -- an unknown value is not a
    fact to stamp.
    """
    if isinstance(v, dict):
        try:
            return float(v.get("value") or 0), str(v.get("note") or "")
        except (TypeError, ValueError):
            return 0.0, ""
    if isinstance(v, (int, float)):
        return float(v), ""
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        return (float(m.group()) if m else 0.0), v
    return 0.0, ""


def stat_overrides(name, max_hp=None):
    """(resist dict, boost dict, stunable) for a named creature, or None.

    The catalog's per-boss tables are exact where the live read can only
    infer: `resist {"value": 50, "note": "to [Death"}` is the scraped
    wiki fact that this boss halves death damage, and `incoming_boost
    {"value": 20, "note": "to [Life"}` that life hits land 20% harder.
    The sim consumes both natively -- `_resist_mult` reads
    `target.resist` and `target.boost` by school -- so stamping them
    onto the read actor makes every prediction and rollout price this
    exact boss instead of a school-typical one.
    """
    rec = lookup(name, max_hp)
    if not rec:
        return None
    stats = rec.get("stats") or {}
    resist, boost = {}, {}
    val, note = _stat_value(stats.get("resist"))
    if val:
        schools = _note_schools(note)
        if schools and "all school" not in note.lower():
            for sc in schools:
                resist[sc] = val / 100.0
        else:
            # Universal resist. It arrives in two shapes -- a note
            # saying "to all schools", or (44 creatures, the Nightshade
            # tiers among them) a BARE NUMBER with no note at all --
            # and both were dropped on the floor, so a 60%-resist boss
            # was priced at zero. The sim reads resist["*"] on every
            # hit (`_resist_mult`); named schools win unless the note
            # says "all school", the same rule data_full applies.
            resist["*"] = val / 100.0
    val, note = _stat_value(stats.get("incoming_boost"))
    if val:
        # no universal boost: the sim's boost read is per-school only
        for sc in _note_schools(note):
            boost[sc] = val / 100.0
    if not resist and not boost and stats.get("stunable") is None:
        return None
    return resist, boost, stats.get("stunable")


def _full_registry():
    """name -> casting Boss for the whole catalog, loaded once.

    `load_bosses_full(spell_pools=True)` is the repo's real boss model:
    spell pools for 1,795 creatures, starting pips at 98.8% coverage,
    pierce, outgoing bonus, and the resist/boost tables with the
    universal "*" entries `stat_overrides` cannot always express. It
    costs about two seconds once (it was thirty before the boss_pools
    pattern cache), which is why this is lazy and cached rather than an
    import-time load the GUI would pay on startup.
    """
    global _FULL
    if _FULL is not None:
        return _FULL
    try:
        from data_full import load_bosses_full, load_spells_full
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cards = load_spells_full(os.path.join(root, "spells_full.json"),
                                 os.path.join(root, "cards_clean.json"))
        bosses, _ = load_bosses_full(
            os.path.join(root, "bosses_clean.json"),
            spell_pools=True, cards=cards)
    except Exception:
        bosses = {}  # no catalog is a quiet no-op, never a crash
    _FULL = bosses
    return _FULL


def full_boss(name, max_hp=None):
    """The catalog's CASTING Boss for a live enemy name, or None.

    Everything `stat_overrides` returns and then the rest of the model:
    a spell pool (bosses cast, they do not auto-attack), the pips the
    boss opens the fight with (397 of them start at 6 -- a legal
    heavy hit on round one, which a from-zero pip model prices as six
    rounds away), pierce, and outgoing bonus. Tier disambiguation is
    `lookup`'s: the observed max health picks the variant actually in
    progress, and then OVERRIDES the scraped health, because the client
    read is the ground truth for this fight.

    Returns a copy -- callers stamp observed school and health onto it,
    and the cached registry must not inherit one fight's observations.
    """
    rec = lookup(name, max_hp)
    if not rec:
        return None
    b = _full_registry().get(rec.get("name"))
    if b is None:
        return None
    b = copy.copy(b)
    b.pool = list(b.pool) if b.pool else None
    b.resist_map = dict(b.resist_map) if b.resist_map else None
    b.boost_map = dict(b.boost_map) if b.boost_map else None
    if max_hp:
        b.hp = int(max_hp)
    return b


def full_encounter(name, max_hp=None):
    """(casting boss, [companions]) for a named fight, or None.

    The pre-fight complement to `full_boss`: 419 catalog bosses carry
    the scraped names of the creatures that fight beside them, and
    `data_full.encounter` resolves those into real stats when the
    companion has its own page and a rank-derived stand-in when it is a
    generic mob. The observed-board path knows more once a fight has
    happened; this is what knows the whole encounter FIRST, so a deck
    can be built for a boss before ever walking in. `max_hp` picks the
    tier, as everywhere in this module; the catalog's health is kept --
    before the fight, the catalog is the best information there is.
    """
    rec = lookup(name, max_hp)
    if not rec:
        return None
    reg = _full_registry()
    key = rec.get("name")
    if key not in reg:
        return None
    try:
        from data_full import encounter
        boss, rest = encounter(key, reg)
    except Exception:
        return None
    boss = copy.copy(boss)
    boss.pool = list(boss.pool) if boss.pool else None
    boss.resist_map = dict(boss.resist_map) if boss.resist_map else None
    boss.boost_map = dict(boss.boost_map) if boss.boost_map else None
    return boss, rest
