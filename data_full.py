"""
Loaders for the full scraped datasets (spells_full.json, bosses_clean.json).

spells_full.json carries game-data-style effect records: numeric effect
types with param/target/damage_type/rounds. The mapping below was derived
by cross-referencing spells whose behavior is documented (Fireblade,
Tower Shield, Fire Elf, Vampire, Judgement, Fuel, Feint, ...). Values are
CURRENT-game-era (post spell audit), so cards built here run under their
own named ruleset — never mix results with the classic tables.

Honesty rules, same as everywhere else in the repo:
  - an effect id we haven't decoded => the card is skipped and reported,
    never approximated silently
  - random-range damage (effect 0 with param -1) has no range in this
    dump; we backfill the classic average from cards_clean.json when the
    name matches, tagged 'backfilled-avg', else skip
  - boss stats: numeric fields are used as scraped; ratings-based stats
    (critical/block) are carried raw in the registry but NOT converted to
    percentages (the conversion is level-dependent); per-round damage is
    NOT in the data and is estimated from rank, tagged 'inferred'

Effect map (id, target) -> op:
  0   damage, random range (param -1) or per-pip (x_pip spells)
  1   damage, fixed
  3   heal
  5   drain (steal-damage, half converts)
  23  ward: trap (+, on enemy) / shield (-, on self); repeats merge into
      charges (Fuel = 3 identical effects)
  26  prism (converts to the opposing school)
  28  damage modifier: blade (target 9/11) / weakness (target 8) /
      damage bubble (target 3)
  29  heal modifier: heal charm, or heal bubble at target 3
  38  absorb
  40  accuracy modifier (Precision +, Black Mantle -)
  41  dispel (school from damage_type)
  68  stun
  70  reshuffle
  72  gain pips
  76  DoT (param = total over `rounds`)
  77  HoT

Targets: 8 selected enemy, 5/4 all enemies, 9 friendly (default self),
11 self, 6/7 all allies, 3 the global field, 1 a card (enchant — not yet
castable in the engine, skipped). Spell type 'aoe' forces multi-target.
"""
import copy
import json

from w101_sim import (Boss, Rules, load_cards, Card, _kind_from_ops,
                      assign_subkeys)

LIVE_RULES = Rules(ruleset_id="w101-pve-live-scrape", era="live",
                   damage_ranges=True)

VARIANT_SOURCE = {"core": "deck", "treasure": "tc", "wand": "item",
                  "amulet": "item", "pet": "pet"}
OPPOSING = {"fire": "ice", "ice": "fire", "storm": "myth", "myth": "storm",
            "life": "death", "death": "life"}

ENEMY_ALL = (4, 5)
ALLY_ALL = (6, 7)


def _tgt(effect, spell, offensive):
    t = effect["target"]
    if t == 3:
        return "global"
    if t in ENEMY_ALL or (spell.get("type") == "aoe" and offensive):
        return "enemies"
    if t in ALLY_ALL or (spell.get("type") == "aoe" and not offensive):
        return "allies"
    if t == 11:
        return "self"
    if t == 8:
        return "enemy"
    if t in (9, 0):
        return "enemy" if offensive else "self"
    return None


def _schools(damage_type):
    if not damage_type or damage_type == "All":
        return None
    return [damage_type.lower()]


def _map_effect(e, spell, avg_backfill):
    """One scraped effect -> (op dict or None, error or None, note)."""
    i, p = e["effect_type_id"], e.get("param")
    dt = (e.get("damage_type") or "").strip() or None
    school = dt.lower() if dt and dt != "All" else None
    per_pip = bool(spell.get("x_pip_spell"))
    note = None
    if i in (1, 3, 5, 23, 28, 29, 38, 40, 76, 77) and p is None:
        return None, f"effect {i} missing param", None

    # name-first dispatch (names stamped from the wiztype enum are ground
    # truth); these have no id in the numeric table but map onto existing
    # engine ops
    nm = e.get("effect_type")
    self_tgt = e["target"] in (9, 11)
    if nm == "kStunBlock":
        return {"op": "stun_block", "n": max(int(p or 1), 1),
                "tgt": "self" if self_tgt else
                _tgt(e, spell, False)}, None, None
    if nm == "kRemoveOverTime":
        return {"op": "remove", "what": "dot",
                "tgt": "self" if self_tgt else "ally"}, None, None
    # An ALL-ENEMIES removal is a board WIPE, not a single strip:
    # Earthquake "completely wipes out all positive and negative charms
    # and wards active on the enemy team". Without this case target 4
    # fell through to the self/negative default, so the extracted
    # Earthquake removed the CASTER's own weaknesses and shields — the
    # opposite of the spell.
    if nm == "kRemoveWard":
        if e["target"] in ENEMY_ALL:
            return {"op": "remove", "what": "ward_all",
                    "tgt": "enemies"}, None, None
        return {"op": "remove",
                "what": "ward_pos" if e["target"] == 8 else "ward_neg",
                "tgt": "enemy" if e["target"] == 8 else "self"}, None, None
    if nm == "kRemoveCharm":
        if e["target"] in ENEMY_ALL:
            return {"op": "remove", "what": "charm_all",
                    "tgt": "enemies"}, None, None
        return {"op": "remove",
                "what": "charm_pos" if e["target"] == 8 else "charm_neg",
                "tgt": "enemy" if e["target"] == 8 else "self"}, None, None

    if i in (0, 1):
        amount = p
        if i == 0 and (p is None or p < 0):
            if per_pip:
                return None, "x-pip spell with no per-pip param", None
            amount = avg_backfill
            if amount is None:
                return None, ("wrapper effect (kInvalidSpellEffect): "
                              "sub-effects not extracted"), None
            note = "backfilled-avg"
        op = {"op": "hit", "amount": float(amount),
              "tgt": _tgt(e, spell, True), "group": 0}
        if school:
            op["school"] = school
        if per_pip:
            op["per_pip"] = True
        return op, None, note
    if i == 76:
        op = {"op": "dot", "total": float(p), "rounds": e.get("rounds") or 3,
              "tgt": _tgt(e, spell, True), "group": 0}
        if school:
            op["school"] = school
        if per_pip:
            op["per_pip"] = True
        return op, None, None
    if i == 5:
        return {"op": "drain", "amount": float(p), "fraction": 0.5,
                "tgt": _tgt(e, spell, True)}, None, None
    if i == 3:
        return {"op": "heal", "amount": float(p),
                "tgt": _tgt(e, spell, False),
                **({"per_pip": True} if per_pip else {})}, None, None
    if i == 77:
        return {"op": "hot", "total": float(p),
                "rounds": e.get("rounds") or 3,
                "tgt": _tgt(e, spell, False)}, None, None
    if i == 23:
        return {"op": "ward", "percent": p / 100.0,
                "schools": _schools(dt),
                "tgt": "self" if e["target"] in (9, 11) else
                _tgt(e, spell, True)}, None, None
    if i == 28:
        if e["target"] == 3:
            return {"op": "global", "field": "damage",
                    "schools": _schools(dt), "percent": p / 100.0}, None, None
        tgt = "self" if e["target"] in (9, 11) else _tgt(e, spell, p < 0)
        return {"op": "charm", "percent": p / 100.0,
                "schools": _schools(dt), "tgt": tgt}, None, None
    if i == 29:
        if e["target"] == 3:
            return {"op": "global", "field": "heal", "schools": None,
                    "percent": p / 100.0}, None, None
        return {"op": "charm", "percent": p / 100.0, "kind": "heal",
                "schools": None,
                "tgt": "self" if e["target"] in (9, 11) else "enemy"}, \
            None, None
    if i == 40:
        return {"op": "charm", "percent": p / 100.0, "kind": "accuracy",
                "schools": None,
                "tgt": "self" if e["target"] in (9, 11) else "enemy"}, \
            None, None
    if i == 38:
        return {"op": "absorb", "amount": float(p),
                "tgt": "self" if e["target"] in (9, 11) else "ally"}, \
            None, None
    if i == 26:
        if not school or school not in OPPOSING:
            return None, "prism with unknown source school", None
        return {"op": "prism", "from": school, "to": OPPOSING[school],
                "tgt": _tgt(e, spell, True)}, None, None
    if i == 41:
        if not school:
            return None, "dispel without school", None
        return {"op": "dispel", "school": school, "tgt": "enemy"}, None, None
    if i == 68:
        return {"op": "stun", "rounds": max(int(p or 1), 1),
                "tgt": _tgt(e, spell, True)}, None, None
    if i == 70:
        return {"op": "reshuffle"}, None, None
    if i == 72:
        if p is None:
            return None, "pip gain without amount", None
        return {"op": "gain_pips", "n": int(p),
                "tgt": "self" if e["target"] in (9, 11) else "ally"}, \
            None, None
    return None, f"undecoded effect {nm or i}", None


def _merge_wrapped(ops):
    """Random-outcome wrappers arrive as N sibling sub-effects (Fire Cat:
    kDamage 80/90/100/110/120, tagged _wrapped). Merge consecutive matching
    siblings into ONE op carrying the roll table in 'outcomes'; amount
    becomes the mean (used by DP/heuristics and when damage_ranges is off)."""
    MERGE = ("hit", "dot", "drain", "heal", "hot")
    out = []
    for op in ops:
        w = op.pop("_wrapped", False)
        key = "amount" if "amount" in op else "total"
        same = (w and out and out[-1].get("outcomes") is not None and
                out[-1]["op"] == op["op"] and
                out[-1].get("tgt") == op.get("tgt") and
                out[-1].get("school") == op.get("school") and
                out[-1].get("rounds") == op.get("rounds") and
                op["op"] in MERGE)
        if same:
            prev = out[-1]
            pkey = "amount" if "amount" in prev else "total"
            prev["outcomes"].append(op[key])
            prev[pkey] = sum(prev["outcomes"]) / len(prev["outcomes"])
        elif w and op["op"] in MERGE:
            op["outcomes"] = [op[key]]
            out.append(op)
        else:
            out.append(op)
    for o in out:
        if o.get("outcomes") is not None and o.get("per_pip"):
            # X-pip wrappers list PIP TIERS (80/160/.../1120 = base*pips),
            # not a roll table: keep the per-pip base, never rng.choice it
            key = "amount" if "amount" in o else "total"
            o[key] = o["outcomes"][0]
            o.pop("outcomes")
        elif o.get("outcomes") is not None and len(o["outcomes"]) == 1:
            o.pop("outcomes")
    return out


def _merge_ward_charges(ops):
    """Fuel-style repeats: N identical ward ops -> one op with N charges."""
    out = []
    for op in ops:
        if (out and op["op"] == "ward" and out[-1]["op"] == "ward" and
                {k: v for k, v in op.items() if k != "charges"} ==
                {k: v for k, v in out[-1].items() if k != "charges"}):
            out[-1]["charges"] = out[-1].get("charges", 1) + 1
        else:
            out.append(dict(op))
    return out


def _regroup(ops):
    """Damage hits get their own link group; a DoT joins the group of the
    hit before it (Fire Elf / Frostbite are one strike)."""
    g = -1
    for op in ops:
        if op["op"] == "hit":
            g += 1
            op["group"] = g
        elif op["op"] == "dot":
            op["group"] = max(g, 0)
    return ops


def load_spells_full(path="spells_full.json",
                     cards_clean_path="cards_clean.json", report=None):
    """Build a Card table from the full dump. Non-core variants get
    provenance suffixes/sources (@tc, @item, @pet) and stack accordingly."""
    raw = json.load(open(path, encoding="utf-8"))
    classic = load_cards(cards_clean_path) if cards_clean_path else {}
    # damage RANGES parsed from the classic descriptions ("deal 80 - 120")
    # — the dump stores random-range params as -1 with no range fields
    import re
    ranges = {}
    if cards_clean_path:
        pat = re.compile(r"(\d[\d,]*)\s*(?:-|to)\s*(\d[\d,]*)")
        for r in json.load(open(cards_clean_path, encoding="utf-8")):
            if r.get("type") in ("damage", "steal") and r.get("description"):
                m = pat.search(r["description"])
                if m:
                    lo, hi = (int(x.replace(",", "")) for x in m.groups())
                    if 0 < lo < hi:
                        ranges[r["name"]] = (float(lo), float(hi))
    cards, skipped = {}, []
    for s in raw:
        variant = s.get("variant")
        src = VARIANT_SOURCE.get(variant)
        if src is None:
            continue                      # mob/battleground/polymorph/...
        name = s["name"]
        key = name if src == "deck" else \
            name + {"tc": "@tc", "item": "@item", "pet": "@pet"}[src]
        if key in cards or not s.get("school"):
            continue
        if not s.get("effects"):
            skipped.append((key, "no effects"))
            continue
        backfill = None
        if name in classic and classic[name].kind in ("damage", "drain"):
            # avg of the classic RANGE HIT only (never the aggregate
            # headline: Fire Elf's DoT portion must not double-count)
            c = classic[name]
            backfill = next((o.get("amount") for o in c.ops
                             if o["op"] in ("hit", "drain")), c.damage)
        ops, why, notes = [], None, set()
        for e in s["effects"]:
            op, err, note = _map_effect(e, s, backfill)
            if err:
                why = err
                break
            if note:
                notes.add(note)
                if name in ranges:            # backfilled hit gets its range
                    op["spread"] = ranges[name]
                    notes.add("range")
            if e.get("_wrapped"):
                op["_wrapped"] = True
            ops.append(op)
        if why:
            skipped.append((key, why))
            continue
        ops = assign_subkeys(_regroup(_merge_ward_charges(
            _merge_wrapped(ops))))
        try:
            pips = int(s.get("pips") or 0)
        except (TypeError, ValueError):
            skipped.append((key, "bad pip cost"))
            continue
        headline = sum(o.get("amount", o.get("total", 0.0)) for o in ops
                       if o["op"] in ("hit", "dot", "drain"))
        # buff coverage for the DP abstraction / dig heuristics: union of
        # the card's damage-modifier schools (None = any school)
        applies = set()
        any_school = False
        for o in ops:
            if o["op"] in ("charm", "ward") and \
                    o.get("kind", "damage") == "damage":
                if o.get("schools") is None:
                    any_school = True
                else:
                    applies.update(o["schools"])
        applies_to = None if any_school else (applies or "own")
        card = Card(
            name=key, school=s["school"], pips=pips,
            accuracy=(s.get("accuracy") or 100) / 100,
            kind=_kind_from_ops(ops), damage=headline,
            percent=max((o.get("percent", 0.0) for o in ops), default=0.0),
            charges=max((o.get("charges", 1) for o in ops), default=1),
            ops=ops, x_pips=bool(s.get("x_pip_spell")), source=src,
            applies_to=applies_to,
            aoe=any(o.get("tgt") in ("enemies", "allies") for o in ops),
            confidence="scrape-full" if not notes else
            "scrape-full+" + ",".join(sorted(notes)),
        )
        card.level = int(s["level_restriction"]) \
            if s.get("level_restriction") else 1   # null = no gate
        cards[key] = card
    if report is not None:
        report["skipped"] = skipped
    return cards


# ---------------------------------------------------------------- decks
# Live-data decks use GAME names ("Tri Blade" is the wiki's "Elemental
# Blade", "Bloodbat" not "Blood Bat"; "Fire Shark"/"Ice Snake"/"Jolted
# Snowman" from the old wiki scrape don't exist in the game data at all).
LIVE_DECKS = {
    "fire": {
        "speed": ["Fire Elf"] * 4 + ["Fire Cat"] * 3 + ["Fireblade"] * 2,
        "oneshot": (["Helephant"] * 2 + ["Immolate"] +
                    ["Fireblade"] * 2 + ["Tri Blade"] * 2 +
                    ["Fire Trap"] + ["Tri Trap"] + ["Fuel"] + ["Feint"] * 2),
    },
    "ice": {
        "stack": (["Colossus"] * 2 + ["Frostbite"] * 2 +
                  ["Iceblade"] * 2 + ["Tri Blade"] * 2 +
                  ["Ice Trap"] * 2 + ["Tri Trap"] * 2),
        "prism": (["Colossus"] * 2 + ["Evil Snowman"] +
                  ["Iceblade"] * 2 + ["Tri Blade"] * 2 +
                  ["Ice Trap"] + ["Ice Prism"] * 2 + ["Feint"] * 2),
    },
    "death": {
        "oneshot": (["Wraith"] * 2 + ["Vampire"] +
                    ["Deathblade"] * 2 + ["Spirit Blade"] * 2 +
                    ["Death Trap"] + ["Spirit Trap"] + ["Curse"] +
                    ["Feint"] * 2),
    },
    "storm": {
        "oneshot": (["Triton"] * 2 + ["Tempest"] +
                    ["Stormblade"] * 2 + ["Tri Blade"] * 2 +
                    ["Storm Trap"] + ["Tri Trap"] + ["Feint"] * 2),
    },
    "balance": {
        "oneshot": (["Judgement"] * 2 + ["Locust Swarm"] +
                    ["Balanceblade"] * 2 + ["Bladestorm"] +
                    ["Hex"] * 2 + ["Curse"] + ["Feint"] * 2),
    },
}


# ---------------------------------------------------------------- bosses

def _num(v):
    """First number in a scraped stat value (int/float/str/dict), else 0."""
    if isinstance(v, dict):
        v = v.get("value")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        import re
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        return float(m.group()) if m else 0.0
    return 0.0


def _parse_boost(stats):
    inc = stats.get("incoming_boost")
    if not isinstance(inc, dict):
        return {}
    note = (inc.get("note") or "").lower()
    for school in ("fire", "ice", "storm", "myth", "life", "death",
                   "balance"):
        if school in note:
            return {school: _num(inc) / 100.0}
    return {}


#      scraped stat  ->  (Boss attribute, coverage over 1912 creatures)
CREATURE_STATS = {"pierce": ("pierce", 0.339),
                  "starting_pips": ("start_pips", 0.988),
                  "critical": ("crit_chance", 0.321),
                  "critical_block": ("block_chance", 0.310),
                  "stunable": ("stunable", 0.902)}


def load_bosses_full(path="bosses_clean.json", report=None, rules=None,
                     spell_pools=False, cards=None,
                     level_mode="world", fill_boost=False):
    """Registry of real bosses. Returns (bosses, registry): `bosses` maps
    name -> Boss ready for Sim; `registry` keeps every scraped field raw
    (ratings, cheats text, locations) for encounter metadata.

    Wires the per-creature combat stats that were scraped alongside
    health and school and then sat unused: PIERCE (median 19% where
    present — it eats through player shields and resist on every hit,
    including the flat-`dmg` path), STARTING PIPS (median 4 — real
    bosses do not open a fight with an empty rack), and the CRITICAL
    and BLOCK ratings.

    CRIT GATE, the same load-bearing rule as gear.py: Actor.crit_chance
    holds a PROBABILITY in the classic era and a RATING only when the
    ruleset installs a rating resolver, so writing a scraped 1085 into
    a classic-era sim would mean "always crit". Ratings are therefore
    emitted only when `rules.crit_resolver` is set; `rules=None` leaves
    them at zero, which is why every pre-existing number in this repo
    is unmoved by this wiring.

    Coverage is partial and uneven (see CREATURE_STATS) — an absent
    field means the wiki page did not list it, NOT that the creature
    has none. `report["coverage"]` records what was actually found so
    a result can say how much of its boss table was real.
    """
    raw = json.load(open(path, encoding="utf-8"))
    pools = {}
    if spell_pools:
        # bosses CAST — they do not auto-attack. Opt-in for now so that
        # every result predating this keeps reproducing; see boss_pools
        # for why the flat `40 + 20*rank` it replaces was the last
        # invented number in the boss model.
        from boss_pools import build_pools
        from deck_builder import TRAINED
        if cards is None:
            cards = load_spells_full()
        pools = build_pools(raw, cards, TRAINED, level_mode)
        _set_pool_cards(cards)
    bosses, registry, skipped = {}, {}, []
    found = {k: 0 for k in CREATURE_STATS}
    crit_era = rules is not None and getattr(rules, "crit_resolver", None)
    for r in raw:
        name, school, hp = r.get("name"), r.get("school"), r.get("health")
        if not name or not school or not hp:
            skipped.append((name or "?", "missing name/school/health"))
            continue
        stats = r.get("stats") or {}
        resist = {}
        res = stats.get("resist")
        res_v = _num(res)
        if res_v:
            # a school named in the note means school-specific resist, not
            # universal (the scrape flattens "70% to Storm" and "46% to
            # all" into the same shape)
            note = (res.get("note") or "").lower() \
                if isinstance(res, dict) else ""
            school_hits = [sc for sc in ("fire", "ice", "storm", "myth",
                                         "life", "death", "balance")
                           if sc in note]
            if school_hits and "all school" not in note:
                for sc in school_hits:
                    resist[sc] = res_v / 100.0
            else:
                resist["*"] = res_v / 100.0
        try:
            hp = int(_num(hp))
            rank = int(_num(r.get("rank")) or 1)
        except (TypeError, ValueError):
            skipped.append((name, "unparseable health/rank"))
            continue
        if hp <= 0:
            skipped.append((name, "unparseable health"))
            continue
        # per-round damage is not scraped; rank-scaled estimate (inferred)
        dmg = int(40 + 20 * rank)
        b = Boss(name, hp, school, dmg,
                 stunable=stats.get("stunable"))
        b.resist_map = resist or None
        b.boost_map = _parse_boost(stats) or None
        b.outgoing_bonus = _num(stats.get("outgoing_boost")) / 100.0
        if fill_boost and not b.outgoing_bonus and rank:
            # only 36% of creatures publish this; the rest were fighting
            # with a flat 0% damage bonus, which no real enemy has
            from enemy_ranks import outgoing_boost as _ob
            b.outgoing_bonus = _ob(rank)
        for key, (attr, _cov) in CREATURE_STATS.items():
            if stats.get(key) is None:
                continue
            found[key] += 1
            if attr in ("crit_chance", "block_chance"):
                if crit_era:                       # ratings, not chances
                    setattr(b, attr, float(_num(stats[key])))
            elif attr == "pierce":
                b.pierce = _num(stats[key]) / 100.0
            elif attr == "start_pips":
                b.start_pips = int(_num(stats[key]))
        b.rank = rank
        b.minions = list(r.get("minions") or []) or None
        if name in pools:
            spec = pools[name]
            b.pool = list(spec["pool"])
            b.archetype = spec["archetype"]
            b.dmg = 0            # the pool replaces the flat estimate
        bosses[name] = b
        registry[name] = dict(r, dmg_estimate=dmg,
                              dmg_confidence="inferred")
    if report is not None:
        report["skipped"] = skipped
        report["coverage"] = {k: {"found": v, "of": len(raw),
                                  "frac": v / max(len(raw), 1)}
                              for k, v in found.items()}
        report["crit_ratings_emitted"] = bool(crit_era)
        report["spell_pools"] = {
            "enabled": bool(spell_pools),
            "bosses_with_pool": sum(1 for b in bosses.values() if b.pool),
            "sources": {k: sum(1 for n in bosses
                               if n in pools and pools[n]["source"] == k)
                        for k in {v["source"] for v in pools.values()}}
            if pools else {}}
    return bosses, registry


# Legacy fraction-of-boss health for an unresolved minion. SUPERSEDED
# by enemy_ranks.normal_hp, which derives a mob's health from its RANK
# against the report's normalized sample instead of from an invented
# share of whatever it happens to be standing next to. Kept because
# results predating the change quote it and because hp_share= is still
# the knob the sensitivity sweep uses.
MINION_HP_SHARE = 0.35
MINION_RANK_DROP = 2
MINION_POOL_KEEP = 3   # weakest N of the boss's pool
_POOL_CARDS = {}


def _set_pool_cards(cards):
    """Card registry used to rank a synthetic minion's pool."""
    _POOL_CARDS.clear()
    _POOL_CARDS.update(cards or {})


def encounter(name, bosses, hp_share=None, synth=True):
    """Resolve a boss into the full fight: (boss, [companions]).

    Real encounters are not 1v1. `Boss.minions` holds the scraped names
    of everything that fights alongside it, and they come in two kinds:

      RESOLVED (25% of references) — the name matches another creature
        in the file, so the companion arrives with its real scraped
        stats. These turn out to be PEERS, not underlings: same rank as
        the boss and ~0.95x its health, i.e. genuine multi-boss fights
        like Othin Stormfather with two Coven Bosses.

      UNRESOLVED (75%) — the name is a generic mob with no page. The
        fight is still not 1v1, so refusing to model it would be a
        worse error than modeling it roughly; `synth=True` builds a
        stand-in a couple of ranks below the boss, with health from
        `enemy_ranks.normal_hp` — the report's rank->HP sample for
        ORDINARY enemies — and spells from the rank spell-tier
        heuristic. Pass `hp_share=` to override that with the legacy
        fraction-of-boss rule, which is what the sensitivity sweep
        varies. `synth=False` gives only the companions whose stats are
        real.
    """
    from enemy_ranks import normal_hp, outgoing_boost
    boss = bosses[name]
    out = []
    for m in (boss.minions or []):
        if m in bosses:
            out.append(copy.copy(bosses[m]))
        elif synth:
            rank = max(1, (boss.rank or 1) - MINION_RANK_DROP)
            hp = (int(boss.hp * hp_share) if hp_share is not None
                  else normal_hp(rank))
            mb = Boss(f"{m} (inferred)", max(1, hp),
                      boss.school, int(40 + 20 * rank))
            mb.rank = rank
            # a mob with no damage buff is not a mob, it is a prop:
            # enemies carry an outgoing damage % exactly as players do
            # from gear, and synthesized ones were getting zero
            mb.outgoing_bonus = (boss.outgoing_bonus
                                 or outgoing_boost(rank))
            mb.resist_map = dict(boss.resist_map or {})
            mb.boost_map = dict(boss.boost_map or {})
            if boss.pool:
                # a mute minion is an HP bag, and HP is what an AoE deck
                # destroys incidentally — so a silent stand-in
                # understates the encounter in the dimension that
                # matters. But it must not cast the BOSS's spells
                # either: handing two rank-9 adds a Helephant each
                # deletes a level-50 wizard in two rounds and made the
                # first end-to-end run 0% at every level. It gets the
                # WEAK end of the pool — same school, a real tier down.
                # MODELED.
                weaker = sorted(
                    boss.pool,
                    key=lambda n: getattr(_POOL_CARDS.get(n), "damage", 0))
                mb.pool = weaker[:MINION_POOL_KEEP] or list(boss.pool)
                mb.archetype = boss.archetype
                mb.dmg = 0
            out.append(mb)
    return boss, out
