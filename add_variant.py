#!/usr/bin/env python3
"""
Add a `variant` field to an existing spells_full.json (no re-extraction).

Classifies each spell by markers found in its internal name / data. Priority
order matters: a spell can carry several markers (e.g. "MOB Fire Cat - Pet"),
so we assign the single most specific/derived one.

Variants used (chosen to match what's actually in the data):
  castle_magic, battleground, treasure, pet, amulet, wand, scion, fusion,
  aberrant, minion, mob, polymorph, maycast, core

`core` is the residual: real player spells PLUS some unmarked system/enemy
entries. There is no single field that cleanly isolates player-castable
spells, so `core` should be read as "not a recognized derived variant",
not "guaranteed player spell". Filtering OUT the derived variants is
reliable; treating `core` as pristine is not.

Usage:
    python add_variant.py                      # in/out spells_full.json
    python add_variant.py --in x.json --out y.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# (variant, compiled regex) in priority order — first match wins.
RULES = [
    ("battleground", re.compile(r"\bBG_|Battleground", re.I)),
    ("treasure",     re.compile(r"\bTC\b|Treasure", re.I)),
    ("pet",          re.compile(r"(^|\W)Pet\s*-|\-\s*Pet\b", re.I)),
    ("amulet",       re.compile(r"-\s*Amulet\b", re.I)),
    ("wand",         re.compile(r"Starter Wand|Wand\s*-|-\s*Wand\b", re.I)),
    ("scion",        re.compile(r"\bScion\b", re.I)),
    ("fusion",       re.compile(r"\bFUSE\b|Fusion", re.I)),
    ("aberrant",     re.compile(r"\bAberrant\b", re.I)),
    ("minion",       re.compile(r"\bMinion\b", re.I)),
    ("mob",          re.compile(r"\bMOB\b|MOB ONLY", re.I)),
    ("polymorph",    re.compile(r"Polymorph|_Poly\d|\bPoly\d", re.I)),
]


def classify(spell):
    name = spell.get("name") or ""
    school = (spell.get("school") or "")
    school = school.lower() if isinstance(school, str) else ""

    # data-driven first
    if school == "castlemagic" or re.search(r"CastleMagic|Castle Magic", name, re.I):
        return "castle_magic"
    if spell.get("source_type") == 3 or re.search(r"Maycast", name, re.I):
        return "maycast"

    for variant, rx in RULES:
        if rx.search(name):
            return variant
    return "core"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="spells_full.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.infile

    spells = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    counts = Counter()
    for s in spells:
        v = classify(s)
        s["variant"] = v
        counts[v] += 1

    # place `variant` right after `name` for readability
    reordered = []
    for s in spells:
        first = {}
        if "name" in s:
            first["name"] = s["name"]
        first["variant"] = s["variant"]
        for k, val in s.items():
            if k not in first:
                first[k] = val
        reordered.append(first)

    Path(out).write_text(json.dumps(reordered, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"Tagged {len(spells)} spells -> {out}")
    print("Variant breakdown:")
    for v, n in counts.most_common():
        print(f"  {n:6d}  {v}")
    print("\nFilter examples:")
    print("  player-ish spells:   [s for s in spells if s['variant'] == 'core']")
    print("  drop derived noise:  [s for s in spells if s['variant'] "
          "not in ('mob','minion','battleground','aberrant')]")


if __name__ == "__main__":
    main()
