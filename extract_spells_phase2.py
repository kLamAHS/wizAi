#!/usr/bin/env python3
"""
PHASE 2 of 2 — deserialize the BINd spell files into spells_full.json.

Your extracted files are KingsIsle "ObjectProperty" binary (magic: BINd).
Deserializing them needs katsuba (a Rust-backed reader) plus a "type list"
JSON that wiztype generates from your running game client.

=========================  ONE-TIME SETUP  =========================

1) Install the tools:
       pip install katsuba wiztype

2) Generate the type list FROM YOUR RUNNING GAME:
   - Launch Wizard101 and get to the login screen (client must be running).
   - In a terminal:
         wiztype
     This writes a file named like  r<revision>.json  in the current dir.
   - Note that filename; you'll pass it below as --types.

======================  RUN THE PARSER  ======================

   # inspect first (recommended): dumps the property names of 3 spells so
   # you can sanity-check the field mapping
   python extract_spells_phase2.py --types r*.json --inspect

   # full run over everything extracted in phase 1 (./wad_spells/Spells)
   python extract_spells_phase2.py --types r*.json

   # options:
   #   --spells-dir DIR   where phase 1 put files (default ./wad_spells)
   #   --out FILE         output (default spells_full.json)
   #   --raw              also write spells_raw.json with every property kept

If a property you want isn't being captured, run --inspect, paste me the
output, and I'll extend the field map — the raw game names differ from the
wiki's and I've mapped the common ones, but there are many.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

try:
    from katsuba.op import (
        TypeList, Serializer, SerializerOptions,
        STATEFUL_FLAGS, HUMAN_READABLE_ENUMS,
    )
except ImportError:
    sys.exit("Missing katsuba. Run:  pip install katsuba\n"
             "(and: pip install wiztype, then run `wiztype` with the game open)")


# --------------------------------------------------------------- helpers

def obj_to_plain(o, depth=0):
    """Recursively turn katsuba LazyObject/LazyList/scalars into plain Python.

    Confirmed API for the installed build:
      - LazyObject: has .items() and .get(); iterating/dict() do NOT work.
      - LazyList:   supports len() and integer indexing; list() works.
      - string values arrive as bytes and must be decoded.
    """
    if depth > 25:
        return None
    tn = type(o).__name__

    if isinstance(o, (bytes, bytearray)):
        try:
            return bytes(o).decode("utf-8")
        except UnicodeDecodeError:
            return bytes(o).decode("latin-1")
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if tn == "LazyObject":
        return {str(k): obj_to_plain(v, depth + 1) for k, v in o.items()}
    if tn == "LazyList" or isinstance(o, (list, tuple)):
        return [obj_to_plain(v, depth + 1) for v in o]
    if isinstance(o, dict):
        return {str(k): obj_to_plain(v, depth + 1) for k, v in o.items()}
    # katsuba value types (Color, Vec3, PointInt, ...) — try items(), else stringify
    if hasattr(o, "items"):
        try:
            return {str(k): obj_to_plain(v, depth + 1) for k, v in o.items()}
        except Exception:
            pass
    return str(o)


def g(d, *keys, default=None):
    """First present key among aliases."""
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in (None, ""):
            return d[k]
    return default


def parse_effects(raw_effects):
    """Flatten m_effects into dicts, preserving ALL fields (KI's SpellEffect
    schema) so nothing is lost, plus a few friendly aliases for common members.
    Effect type arrives as an enum int in this build; kept as-is under
    'effect_type_id' with the raw fields alongside."""
    out = []
    if not raw_effects:
        return out
    for e in raw_effects:
        ed = e if isinstance(e, dict) else obj_to_plain(e)
        if not isinstance(ed, dict):
            continue
        # Wrapper effects (Random/Variable SpellEffect: m_effectType == 0 /
        # kInvalidSpellEffect) carry their real payload in m_effectList —
        # recurse so range hits and per-pip tiers emit as typed sub-effects
        # instead of being dropped with the list members.
        sub = ed.get("m_effectList")
        if isinstance(sub, list) and sub:
            inner = parse_effects(sub)
            for s_e in inner:
                s_e["_wrapped"] = True
            out.extend(inner)
            if not ed.get("m_effectType"):
                continue                 # pure wrapper: sub-effects suffice
        friendly = {
            "effect_type_id": ed.get("m_effectType"),
            "param": ed.get("m_effectParam"),
            "target": ed.get("m_effectTarget"),
            "disposition": ed.get("m_disposition"),
            "damage_type": ed.get("m_sDamageType"),
            "rounds": ed.get("m_numRounds"),
            "school": ed.get("m_sEffectSchool") or ed.get("m_sMagicSchoolName"),
        }
        friendly = {k: v for k, v in friendly.items() if v not in (None, "", -1)}
        # keep the full raw effect too, minus noisy/huge members
        raw = {k: v for k, v in ed.items()
               if k not in ("m_animationSequence",) and not isinstance(v, (list, dict))}
        friendly["_raw"] = raw
        out.append(friendly)
    return out


# m_sTypeName is KI's own category string ("Damage", "Charm", "Ward", "Heal"...).
# Map it to a lowercase coarse type resembling your card file's `type`.
TYPE_MAP = {
    "damage": "damage", "heal": "heal", "charm": "charm", "ward": "ward",
    "aura": "aura", "global": "global", "manipulation": "manipulation",
    "enchantment": "enchantment", "polymorph": "polymorph",
    "shadow": "shadow", "curse": "curse",
}


def derive_type(type_name, effects):
    if type_name:
        key = str(type_name).strip().lower()
        return TYPE_MAP.get(key, key)
    return None


def pip_info(obj):
    """Pip cost + variants live in the m_spellRank sub-object."""
    r = g(obj, "m_spellRank")
    if not isinstance(r, dict):
        return {}
    per_school = {
        "fire": r.get("m_firePips"), "ice": r.get("m_icePips"),
        "storm": r.get("m_stormPips"), "myth": r.get("m_mythPips"),
        "life": r.get("m_lifePips"), "death": r.get("m_deathPips"),
        "balance": r.get("m_balancePips"),
    }
    per_school = {k: v for k, v in per_school.items() if v}
    return {
        "pips": r.get("m_spellRank"),
        "shadow_pips": r.get("m_shadowPips") or None,
        "x_pip_spell": bool(r.get("m_xPipSpell")),
        "school_pips": per_school or None,
    }


def normalize(name, obj):
    """obj is a plain dict from a deserialized spell template."""
    effects = parse_effects(g(obj, "m_effects", default=[]))
    school = g(obj, "m_sMagicSchoolName")
    if isinstance(school, str):
        school = school.lower()
    type_name = g(obj, "m_sTypeName")
    pips = pip_info(obj)

    return {
        "name": g(obj, "m_name", default=name),
        "display_name": g(obj, "m_displayName"),
        "school": school,
        "type": derive_type(type_name, effects),
        "type_name": type_name,
        "pips": pips.get("pips"),
        "shadow_pips": pips.get("shadow_pips"),
        "school_pips": pips.get("school_pips"),
        "x_pip_spell": pips.get("x_pip_spell", False),
        "accuracy": g(obj, "m_accuracy"),
        "base_spell": g(obj, "m_spellBase"),   # base spell name (string)
        "training_cost": g(obj, "m_trainingCost"),
        "level_restriction": g(obj, "m_levelRestriction"),
        "pvp_flag": g(obj, "m_PvP"),
        "pve_flag": g(obj, "m_PvE"),
        "battlegrounds_only": g(obj, "m_battlegroundsOnly"),
        "treasure_card": g(obj, "m_Treasure"),
        "no_discard": g(obj, "m_noDiscard"),
        "cloaked": g(obj, "m_cloaked"),
        "no_pvp_enchant": g(obj, "m_noPvPEnchant"),
        "no_pve_enchant": g(obj, "m_noPvEEnchant"),
        "description": g(obj, "m_description"),
        "effects": effects,
        "image": g(obj, "m_imageName"),
        "source_type": g(obj, "m_spellSourceType"),
    }


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", required=True,
                    help="wiztype type list JSON (e.g. r764987.json); globs ok")
    ap.add_argument("--spells-dir", default="wad_spells")
    ap.add_argument("--out", default="spells_full.json")
    ap.add_argument("--raw", action="store_true",
                    help="also write spells_raw.json with all deserialized props")
    ap.add_argument("--inspect", action="store_true",
                    help="print property names of the first few spells and exit")
    ap.add_argument("--deep", nargs="?", const=True, default=False,
                    help="fully dump one spell's nested structure. Optionally pass a "
                         "filename, e.g. --deep \"Fire Cat.xml\"; otherwise auto-picks "
                         "a real combat damage spell (skips CastleMagic/MOB variants)")
    args = ap.parse_args()

    # resolve type list (allow glob like r*.json)
    matches = sorted(glob.glob(args.types))
    if not matches:
        sys.exit(f"No type list matched {args.types!r}. Run `wiztype` with the game open.")
    type_path = matches[-1]
    print(f"Loading type list: {type_path}")
    types = TypeList.open(type_path)

    opts = SerializerOptions()
    opts.flags |= STATEFUL_FLAGS | HUMAN_READABLE_ENUMS  # readable enum strings
    opts.shallow = False                                 # required for full deserialize
    opts.skip_unknown_types = True
    ser = Serializer(opts, types)

    spells_root = Path(args.spells_dir)
    files = sorted(spells_root.rglob("*.xml"))
    if not files:
        sys.exit(f"No .xml spell files under {spells_root}/. Did phase 1 run?")
    print(f"Found {len(files)} spell files.")

    if args.deep:
        target = None
        if isinstance(args.deep, str):
            # explicit filename: exact, then substring
            for f in files:
                if f.name.lower() == args.deep.lower():
                    target = f
                    break
            if not target:
                cands = [f for f in files if args.deep.lower() in f.name.lower()]
                if len(cands) == 1:
                    target = cands[0]
                elif len(cands) > 1:
                    print(f"{len(cands)} files match {args.deep!r}; using first. Others:")
                    for c in cands[:10]:
                        print("   ", c.name)
                    target = cands[0]
            if not target:
                sys.exit(f"No spell file matching {args.deep!r} under {spells_root}/.")
            data = target.read_bytes()
            plain = obj_to_plain(ser.deserialize(data[4:]))
            print("DEEP DUMP of:", target.name)
            print(json.dumps(plain, indent=2, ensure_ascii=False, default=str)[:6500])
            print("\n... (truncated) — paste this back.")
            return

        # auto mode: scan and DIAGNOSE
        deser_ok = 0
        type_names = {}
        with_effects = None
        with_effects_file = None
        checked = 0
        for f in files:
            n = f.name.lower()
            if any(bad in n for bad in ("- mob", "amulet", "bg_", "castlemagic")):
                continue
            data = f.read_bytes()
            if data[:4] != b"BINd":
                continue
            checked += 1
            try:
                o = obj_to_plain(ser.deserialize(data[4:]))
            except Exception:
                continue
            deser_ok += 1
            tn = o.get("m_sTypeName")
            tn = tn if isinstance(tn, str) else f"<{type(tn).__name__}>"
            type_names[tn] = type_names.get(tn, 0) + 1
            eff = o.get("m_effects")
            if eff and with_effects is None:
                with_effects = o
                with_effects_file = f.name
            if checked >= 800:  # enough to characterize
                break

        print(f"Scanned {checked} candidate files: {deser_ok} deserialized OK.")
        print("m_sTypeName values seen:", dict(sorted(type_names.items(),
                                                       key=lambda x: -x[1])))
        if with_effects is not None:
            print(f"\nDumping first spell WITH non-empty effects: {with_effects_file}")
            print(json.dumps(with_effects, indent=2, ensure_ascii=False, default=str)[:6500])
            print("\n... (truncated) — paste this back.")
        else:
            print("\nNo spell had a non-empty m_effects list in the sample.")
            print("That means the effects LazyList still isn't expanding. Dumping one "
                  "spell raw so we can see the m_effects field's actual repr:")
            for f in files:
                n = f.name.lower()
                if any(b in n for b in ("- mob", "amulet", "bg_", "castlemagic")):
                    continue
                data = f.read_bytes()
                if data[:4] != b"BINd":
                    continue
                obj = ser.deserialize(data[4:])
                raw_eff = obj["m_effects"] if "m_effects" in [k for k in iter(obj)] else None
                print("file:", f.name)
                print("m_effects python type:", type(raw_eff).__name__)
                print("m_effects dir():", [a for a in dir(raw_eff) if not a.startswith('_')])
                try:
                    print("len(m_effects):", len(raw_eff))
                except Exception as e:
                    print("len() failed:", e)
                try:
                    print("list(m_effects)[:1] types:",
                          [type(x).__name__ for x in list(raw_eff)[:1]])
                except Exception as e:
                    print("iter failed:", e)
                break
        return

    if args.inspect:
        shown = 0
        for f in files:
            data = f.read_bytes()
            if data[:4] != b"BINd":
                continue
            try:
                obj = ser.deserialize(data[4:])
            except Exception as e:
                print(f"  (skip {f.name}: {e})")
                continue
            plain = obj_to_plain(obj)
            print("\n" + "=" * 60)
            print("FILE:", f.name)
            print("type_hash:", getattr(obj, "type_hash", "?"))
            print("python type:", type(obj).__module__ + "." + type(obj).__name__)
            if isinstance(plain, dict) and plain:
                print("top-level property names:")
                for k in plain.keys():
                    v = plain[k]
                    vshow = (str(v)[:60] + "…") if isinstance(v, (str, int, float, bool)) \
                        else f"<{type(v).__name__}>"
                    print(f"    {k} = {vshow}")
            else:
                # flattening failed — dump the real API so we can adapt
                print("could NOT flatten to a dict. Object's public API:")
                pub = [a for a in dir(obj) if not a.startswith("_")]
                print("   attrs/methods:", pub)
                for probe in ("keys", "items", "values", "properties", "fields"):
                    fn = getattr(obj, probe, None)
                    if callable(fn):
                        try:
                            print(f"   {probe}() ->", list(fn())[:20])
                        except Exception as e:
                            print(f"   {probe}() raised {e}")
                # try iterating
                try:
                    print("   list(iter(obj))[:20] ->", list(iter(obj))[:20])
                except Exception as e:
                    print("   iter(obj) raised", e)
            shown += 1
            if shown >= 3:
                break
        print("\nPaste this back if the field map needs adjusting.")
        return

    clean, raw, skipped = [], [], 0
    for f in files:
        data = f.read_bytes()
        if data[:4] != b"BINd":
            skipped += 1
            continue
        try:
            obj = ser.deserialize(data[4:])
        except Exception:
            skipped += 1
            continue
        plain = obj_to_plain(obj)
        internal_name = f.stem
        clean.append(normalize(internal_name, plain))
        if args.raw:
            raw.append({"_file": str(f.relative_to(spells_root)), **plain})

    # de-dup by (name, template_id) keeping richest
    seen = {}
    for s in clean:
        key = (s.get("name"), s.get("template_id"))
        if key not in seen or len(json.dumps(s)) > len(json.dumps(seen[key])):
            seen[key] = s
    deduped = sorted(seen.values(), key=lambda s: (s.get("school") or "~", str(s.get("name"))))

    Path(args.out).write_text(json.dumps(deduped, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"Wrote {len(deduped)} spells to {args.out} "
          f"({len(clean)} before dedup, {skipped} skipped).")
    if args.raw:
        Path("spells_raw.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
        print("Wrote spells_raw.json")


if __name__ == "__main__":
    main()