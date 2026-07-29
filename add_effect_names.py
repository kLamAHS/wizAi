#!/usr/bin/env python3
"""
Add human-readable effect-type names to spells_full.json, using the EXACT
enum from your wiztype type list (ground truth for your game version) — no
guessing.

The type list stores enum values per-property:
    classes[hash].properties["m_effectType"].enum_options = { "kDamage": 1, ... }
We find that enum, invert it (int -> name), and stamp every effect with an
`effect_type` name alongside its existing `effect_type_id`.

Usage:
    python add_effect_names.py --types r803238_Wizard_1_610.json
    # options:
    #   --in spells_full.json   (default)
    #   --out spells_full.json  (default: overwrite in place)
"""

import argparse
import json
import sys
from pathlib import Path


def find_effect_enum(type_list):
    """Return {int: name} for the spell-effect type enum.

    Strategy: scan every property's enum_options for the one that looks like
    the spell-effect enum — i.e. its option names include the telltale set
    (Damage + Heal + at least one of Steal/DOT/Jinx/Charm...). We pick the
    richest matching enum. This is robust to the exact property/class name,
    which has changed across game versions.
    """
    TELLTALE_STRONG = {"kDamage", "kHeal"}
    TELLTALE_ANY = {"kSteal", "kDamageOverTime", "kDamageNoCrit", "kJinx",
                    "kStealOverTime", "kHealOverTime", "kModifyIncomingDamage",
                    "kAbsorbDamage", "kReduceOverTime"}
    best = None
    best_name = None
    classes = type_list.get("classes", {})
    for chash, cdef in classes.items():
        for pname, pdef in (cdef.get("properties") or {}).items():
            opts = pdef.get("enum_options")
            if not isinstance(opts, dict):
                continue
            names = {k for k in opts.keys() if not k.startswith("__")}
            if not (TELLTALE_STRONG <= names):
                continue
            if not (names & TELLTALE_ANY):
                continue
            # candidate — keep the richest one
            if best is None or len(names) > len(best):
                best = names
                best_name = f"{cdef.get('name')}::{pname}"
                best_opts = opts

    if best is None:
        return None, None
    # invert (name -> int) to (int -> name), dropping __DEFAULT/__BASECLASS
    mapping = {}
    for k, v in best_opts.items():
        if k.startswith("__"):
            continue
        if isinstance(v, int):
            mapping[v] = k
    return mapping, best_name


def pretty(name):
    """kDamageOverTime -> 'damage over time' style, and a short form."""
    if not name:
        return None
    s = name[1:] if name.startswith("k") else name
    # split camelCase
    out = []
    for i, ch in enumerate(s):
        if ch.isupper() and i and not s[i - 1].isupper():
            out.append(" ")
        out.append(ch)
    return "".join(out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", required=True, help="wiztype type list JSON")
    ap.add_argument("--in", dest="infile", default="spells_full.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.infile

    import glob
    matches = sorted(glob.glob(args.types))
    if not matches:
        sys.exit(
            f"No file matches --types {args.types!r} from this directory.\n"
            "The wiztype dump is wherever wiztype wrote it (usually next to "
            "the wiztype exe or its working directory), not necessarily "
            "here. Either copy it into this folder or pass the full path, "
            "quoted:\n"
            '  python add_effect_names.py --types '
            '"C:\\path\\to\\r803238_Wizard_1_610.json"\n'
            "(A glob like r*_Wizard_*.json also works; the newest match "
            "is used.)")
    tl_path = matches[-1]
    print(f"Reading type list: {tl_path}")
    type_list = json.loads(Path(tl_path).read_text(encoding="utf-8"))

    mapping, src = find_effect_enum(type_list)
    if not mapping:
        sys.exit("Could not locate the spell-effect enum in the type list. "
                 "Paste me a few lines of the type list around 'kDamage' and "
                 "I'll point the finder at it.")
    print(f"Found effect enum from {src}: {len(mapping)} values.")
    print("Sample:", {k: mapping[k] for k in sorted(mapping)[:8]})

    spells = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    stamped = 0
    unknown_ids = set()
    for s in spells:
        for e in s.get("effects", []):
            eid = e.get("effect_type_id")
            if eid is None:
                continue
            name = mapping.get(eid)
            if name is None:
                unknown_ids.add(eid)
                continue
            e["effect_type"] = name              # raw enum name, e.g. kDamage
            e["effect_type_label"] = pretty(name)  # friendly, e.g. Damage
            stamped += 1

    Path(out).write_text(json.dumps(spells, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"Stamped {stamped} effects with names. Wrote {out}.")
    if unknown_ids:
        print(f"{len(unknown_ids)} effect ids had no enum entry "
              f"(rare/removed effects): {sorted(unknown_ids)[:15]}...")
    # also dump the full mapping for reference
    ref = {mapping[k]: k for k in sorted(mapping)}
    Path("effect_type_enum.json").write_text(
        json.dumps({str(k): mapping[k] for k in sorted(mapping)}, indent=2),
        encoding="utf-8")
    print("Wrote effect_type_enum.json (full int -> name table).")


if __name__ == "__main__":
    main()
