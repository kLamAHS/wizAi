"""Print one spell's record from spells_full.json (debug helper).

Usage:  python inspect_spell.py "Fire Cat" [--variant core]
"""
import json
import sys

name = sys.argv[1] if len(sys.argv) > 1 else "Fire Cat"
sp = json.load(open("spells_full.json", encoding="utf-8"))
hits = [s for s in sp if s["name"] == name]
if not hits:
    close = sorted({s["name"] for s in sp if name.lower() in
                    s["name"].lower()})[:10]
    sys.exit(f"no spell named {name!r}; close: {close}")
for s in hits:
    print(f"== {s['name']} [{s.get('variant')}] school={s.get('school')} "
          f"pips={s.get('pips')} acc={s.get('accuracy')}")
    print(json.dumps(s["effects"], indent=1)[:4000])
