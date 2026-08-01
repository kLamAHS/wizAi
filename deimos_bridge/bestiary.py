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
import json
import os
import re
import unicodedata

_INDEX = None


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
    r = stats.get("resist")
    if isinstance(r, dict) and r.get("value"):
        for sc in _note_schools(r.get("note")):
            resist[sc] = float(r["value"]) / 100.0
    b = stats.get("incoming_boost")
    if isinstance(b, dict) and b.get("value"):
        for sc in _note_schools(b.get("note")):
            boost[sc] = float(b["value"]) / 100.0
    if not resist and not boost and stats.get("stunable") is None:
        return None
    return resist, boost, stats.get("stunable")
