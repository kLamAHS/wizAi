"""
Spell pools for real bosses: bosses cast, they do not auto-attack.

The boss registry has carried `dmg = 40 + 20 * rank` since the start —
a flat per-round hit, tagged `inferred`, and the last fully invented
number in the boss model. The repo owner confirms bosses simply cast
from their spell pool. That flat hit is also what made every real-boss
fight CLIFF-like: identical damage every round turns an encounter into
an arithmetic race with no variance and almost no contested middle
(a band scan wins ~100% below 9k boss HP and ~0% above 14k).

The engine already has the living-boss layer — `Boss.pool`,
`_enemy_choose`, archetype, discipline, pip economy — from the 2026
boss-AI report. It has only ever been driven by hand, one boss at a
time. This module fills it for all 1912 creatures.

Two routes, and which one a boss got is recorded per boss:

  SOURCED (33% of creatures) — `spell_notes` is free text but it names
    real spells: "Casts a version of Storm Lord that doesn't Stun",
    "Casts an Aegis-Protected Tower Shield". Any name in that text that
    resolves in the card registry goes in the pool.

  INFERRED (the rest) — built from the boss's SCHOOL and an estimated
    level, using the same curated TRAINED line the player's deck
    builder draws on. The level estimate chains through `worlds.py`:
    the record's `world` field maps to a band and the band's last level
    is the content level. For the ~30% of creatures in worlds that
    table does not cover (Grizzleheim, Darkmoor, Karamelle, housing
    instances), the level is fitted from RANK against the bosses whose
    world IS known — a regression on real pairs rather than a guess.

Confidence: the pools are 'modeled' in the same sense as the rest of
the living-boss layer. What is sourced is that bosses cast from a pool
(owner) and, for a third of them, which spells that pool contains
(scrape). The archetype and discipline knobs remain modeled.
"""
import re
import statistics as st

from worlds import WORLDS, band

# archetype from the shape of what a school does, not from the scrape
SCHOOL_ARCHETYPE = {"life": "healer", "balance": "buffer",
                    "death": "debuffer", "ice": "tank",
                    "fire": "hitter", "storm": "hitter",
                    "myth": "hitter", "shadow": "hitter",
                    "star": "buffer", "sun": "buffer", "moon": "hitter"}


def _known_world_level(record):
    w = record.get("world")
    return band(w)[1] if w in WORLDS else None


MIN_RANK_SAMPLE = 20


def fit_rank_to_level(records, min_n=MIN_RANK_SAMPLE):
    """Median content level per rank, learned from the creatures whose
    world IS in the table, then made MONOTONE.

    Both guards are load-bearing. The raw medians are clean and rising
    from rank 1 to 17 (L10 -> L130) and then fall apart: rank 18 reads
    L10 off 26 samples and rank 24 reads L20 off 11, because the
    high-rank creatures mostly live in worlds this table does not cover
    (Darkmoor, Arcanum), leaving an unrepresentative rump behind. Taking
    those medians literally put Annoushka — rank 24, 17,240 HP — at
    level 20 and handed it a Dark Sprite. So: drop ranks with thin
    support, force the curve non-decreasing, and hold the top value
    above the last trusted rank rather than extrapolating.
    """
    by_rank = {}
    for r in records:
        lvl = _known_world_level(r)
        rank = r.get("rank")
        if lvl and isinstance(rank, int) and 0 < rank < 100:
            by_rank.setdefault(rank, []).append(lvl)
    trusted = {k: int(st.median(v)) for k, v in by_rank.items()
               if len(v) >= min_n}
    fit, running = {}, 0
    for rank in sorted(trusted):
        running = max(running, trusted[rank])
        fit[rank] = running
    return fit


def estimate_level(record, rank_fit):
    """(level, how) — 'world' when the band is known, 'rank-fit' when
    it had to be learned, 'floor' when neither is available."""
    lvl = _known_world_level(record)
    if lvl:
        return lvl, "world"
    rank = record.get("rank")
    if isinstance(rank, int) and rank > 0 and rank_fit:
        if rank in rank_fit:
            return rank_fit[rank], "rank-fit"
        top = max(rank_fit)
        if rank > top:                     # above the trusted range
            return rank_fit[top], "rank-fit-held"
        near = min(rank_fit, key=lambda k: abs(k - rank))
        return rank_fit[near], "rank-fit"
    return 10, "floor"


_STOP = {"Damage", "Round", "Rounds", "Aura", "Shield", "Blade", "Trap",
         "Casts", "Deals", "Spell", "Natural", "Attack", "Always",
         "Random", "All", "The", "This", "That"}


def _note_patterns(cards):
    """[(name, compiled pattern)] in the match order pool_from_notes uses.

    Compiled ONCE per registry because the alternative was the whole
    load: ~400 candidate names tried against 1,911 creatures is 765k
    `re.search` calls, and with more distinct patterns than the re
    module's cache holds, every single one recompiled. That recompiling
    was 93% of a 29-second `build_pools` — the pools themselves cost
    about a second.
    """
    out = []
    # longest names first so "Storm Lord" wins over "Storm"
    for name in sorted(cards, key=len, reverse=True):
        if " - " in name or name.startswith("NA ") or "@" in name:
            continue
        if len(name) < 5 or name in _STOP:
            continue
        out.append((name, re.compile(r"\b" + re.escape(name) + r"\b")))
    return out


def pool_from_notes(record, cards, _patterns=None):
    """Spell names the scrape's free text actually mentions."""
    text = " ".join(record.get("spell_notes") or [])
    if not text:
        return []
    found = []
    for name, pat in (_patterns if _patterns is not None
                      else _note_patterns(cards)):
        if pat.search(text):
            if name not in found and not any(name in f for f in found):
                found.append(name)
    return found[:6]


def pool_from_school(school, level, cards, trained, size=5):
    """A rank-appropriate pool from the school's curated trained line:
    the strongest unlocked nukes plus a blade, which is what the
    boss-AI report describes bosses doing."""
    line = trained.get(school) or {}
    owned = [n for n, floor in line.items()
             if floor <= level and n in cards]
    hits = sorted((cards[n] for n in owned
                   if cards[n].kind in ("damage", "drain")
                   and not cards[n].x_pips),
                  key=lambda c: -c.damage)[:size - 1]
    blades = sorted((cards[n] for n in owned if cards[n].kind == "blade"),
                    key=lambda c: -c.percent)[:1]
    return [c.name for c in hits + blades]


def build_pools(records, cards, trained, level_mode="world"):
    """name -> {'pool', 'archetype', 'level', 'source'} for every
    creature that can be given one.

    `level_mode` picks how a creature's SPELL TIER is decided:

      'world'  the world its page lists, resolved to that band's last
               level. My own inference, and generous — it assumes a
               Zafaria boss casts the best spells Zafaria offers.
      'rank'   the community heuristic the 2026 enemy-rank report
               quotes: boss ~ rank * 5, normal ~ rank * 6. SOURCED, and
               consistently lower than the world rule.

    Both land inside the report's documented 85-115 damage-per-pip band
    because that band is flat in rank — higher-tier spells cost
    proportionally more pips — so the choice moves PIP COST and tempo,
    not damage density.
    """
    from enemy_ranks import spell_level
    rank_fit = fit_rank_to_level(records)
    patterns = _note_patterns(cards)
    out = {}
    for r in records:
        name, school = r.get("name"), r.get("school")
        if not name or not school:
            continue
        if level_mode == "rank" and r.get("rank"):
            level, how = spell_level(r["rank"], "boss"), "rank-heuristic"
        else:
            level, how = estimate_level(r, rank_fit)
        pool = pool_from_notes(r, cards, _patterns=patterns)
        source = "spell_notes"
        if len(pool) < 3:
            # a note like "Casts a version of Storm Lord" names ONE
            # spell; a boss with a one-card pool passes most rounds.
            # Top it up from the school line rather than crippling it.
            extra = pool_from_school(school, level, cards, trained)
            source = "spell_notes+school" if pool else f"school+{how}"
            pool = pool + [n for n in extra if n not in pool]
        if not pool:
            continue
        out[name] = {"pool": pool, "level": level, "source": source,
                     "archetype": SCHOOL_ARCHETYPE.get(school, "hitter")}
    return out
