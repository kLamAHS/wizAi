#!/usr/bin/env python3
"""
PHASE 1 of 2 — extract Wizard101 spell files and report what they look like.

This does NOT produce the final JSON yet. KingsIsle stores game objects in a
binary-serialized format, and the exact layout varies by game version, so we
inspect real files first, then write a parser that fits (phase 2).

Requirements:
    pip install wizwad

Usage (Windows, game installed at the default path):
    python extract_spells_phase1.py

If your game is elsewhere, pass the Root.wad path:
    python extract_spells_phase1.py "D:\\Games\\Wizard101\\Data\\GameData\\Root.wad"

What it does:
    1. Locates Root.wad
    2. Lists everything under Spells/ inside it
    3. Extracts those files to ./wad_spells/
    4. Prints a format report: how many files, and a hexdump + text preview
       of a few, so we can tell whether they're plain XML, BINd-serialized
       binary, or something else.

Then paste me the report and I'll write extract_spells_phase2.py to parse them.
"""

import sys
from pathlib import Path

try:
    import wizwad
except ImportError:
    sys.exit("Missing dependency. Run:  pip install wizwad")

DEFAULT_ROOT = Path(
    r"C:\ProgramData\KingsIsle Entertainment\Wizard101\Data\GameData\Root.wad"
)
OUT = Path("wad_spells")


def find_root(argv):
    if len(argv) > 1:
        p = Path(argv[1])
        if p.exists():
            return p
        sys.exit(f"Not found: {p}")
    if DEFAULT_ROOT.exists():
        return DEFAULT_ROOT
    # try a couple of common alternatives
    for alt in [
        Path(r"C:\Program Files (x86)\Wizard101\Data\GameData\Root.wad"),
        Path(r"C:\Program Files\Wizard101\Data\GameData\Root.wad"),
    ]:
        if alt.exists():
            return alt
    sys.exit(
        "Could not find Root.wad automatically.\n"
        "Find it under your Wizard101 install (…/Data/GameData/Root.wad) "
        "and pass the path:\n"
        '    python extract_spells_phase1.py "PATH\\TO\\Root.wad"'
    )


def main():
    root = find_root(sys.argv)
    print(f"Using: {root}")
    wad = wizwad.Wad(str(root))

    # list every file; find the spell-related ones
    all_names = list(wad.name_list())
    print(f"Root.wad contains {len(all_names)} files total.")

    spell_names = [n for n in all_names if n.lower().startswith("spells/")
                   or "/spells/" in n.lower()]
    print(f"Found {len(spell_names)} files under a Spells/ path.")
    if not spell_names:
        # fall back: show the top-level folders so we can locate spells
        tops = sorted({n.split("/")[0] for n in all_names})
        print("No Spells/ folder. Top-level entries in Root.wad:")
        for t in tops:
            print("  ", t)
        print("\nTell me which of these looks spell-related and I'll adjust.")
        return

    # extract just the spell files
    OUT.mkdir(exist_ok=True)
    print(f"Extracting {len(spell_names)} spell files to {OUT}/ ...")
    for name in spell_names:
        data = wad.read(name)
        dest = OUT / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    print("Done extracting.")

    # ---- format report ----
    print("\n" + "=" * 70)
    print("FORMAT REPORT — paste everything below this line back to me")
    print("=" * 70)

    # extension breakdown
    exts = {}
    for n in spell_names:
        ext = Path(n).suffix.lower() or "(none)"
        exts[ext] = exts.get(ext, 0) + 1
    print("File extensions:", exts)

    # sample a few files
    sample = spell_names[:3]
    # also try to grab one that clearly names a known spell
    for n in spell_names:
        low = n.lower()
        if any(s in low for s in ("fireball", "firecat", "frostbeetle", "imp")):
            sample.append(n)
            break

    for name in sample:
        data = wad.read(name)
        print("\n" + "-" * 70)
        print("FILE:", name, f"({len(data)} bytes)")
        print("first 16 bytes hex:", data[:16].hex(" "))
        magic = data[:4]
        print("magic:", magic, "->", "BINd (binary serialized)"
              if magic == b"BINd" else "looks like text/XML"
              if data[:5].lstrip().startswith(b"<") else "unknown")
        # printable preview
        preview = bytes(b if 32 <= b < 127 or b in (9, 10, 13) else 0x2e
                        for b in data[:400]).decode("latin-1")
        print("text preview:\n", preview)

    print("\n" + "=" * 70)
    print("End of report. Paste it back and I'll write the parser.")


if __name__ == "__main__":
    main()
