"""What fraction of the real game's spell effects does wizAi model?

wizAi's `data_full.load_spells_full` refuses to guess: an effect it
cannot decode fails the whole card, with a reason, rather than being
flattened into something that would read as a different spell. That is
the right call, but it means the simulator's reach is bounded by the
decoder and the bound was only visible as a 7k-line skip list.

Deimos supplies the missing yardstick twice over:

  * wizwalker's `SpellEffects` enum (`memory_objects/enums.py`) is the
    *client's* enum, so it is the authority on which effects exist and
    what they are called.
  * That makes wizAi's own `effect_type_enum.json` checkable -- if the
    two disagree about an id, one of them is wrong about the game.

Three questions, in increasing order of how much they matter:

  1. Does wizAi's effect table agree with the client's enum?
  2. Of the effects that appear on a real card, how many can wizAi
     decode?
  3. Weighted by real cards, how much of the game does that cost?

    python -m deimos_bridge.effect_audit
    python -m deimos_bridge.effect_audit --json results_effect_audit.json
"""
import argparse
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ENUMS = os.path.join(HERE, "..", "Deimos", "libs", "wizwalker", "wizwalker",
                     "memory", "memory_objects", "enums.py")


def real_enum(path=ENUMS) -> dict:
    """`SpellEffects` from wizwalker, parsed rather than imported.

    Importing it would pull in `wizwalker.constants`, which runs
    `ctypes.windll.user32` at module scope and cannot load off Windows.
    The enum body is plain `name = int` lines, so parsing is exact.
    """
    with open(path, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"class SpellEffects\(Enum\):(.*?)(?=\nclass |\Z)", src, re.S)
    if not m:
        raise RuntimeError(f"SpellEffects not found in {path}")
    out = {}
    for line in m.group(1).splitlines():
        hit = re.match(r"\s+([a-z_0-9]+)\s*=\s*(\d+)\s*$", line)
        if hit:
            out[hit.group(1)] = int(hit.group(2))
    return out


def _snake(k_name: str) -> str:
    """wizAi's 'kModifyOutgoingDamage' -> the client's
    'modify_outgoing_damage'."""
    s = k_name[1:] if k_name.startswith("k") else k_name
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s)
    return s.lower()


def compare_enums(spells_enum_path="effect_type_enum.json"):
    """Question 1: does wizAi's effect table match the client's enum?"""
    game = real_enum()                      # snake_name -> id
    game_by_id = {}
    for name, i in game.items():
        game_by_id.setdefault(i, name)

    wiz = {int(k): v for k, v in
           json.load(open(spells_enum_path, encoding="utf-8")).items()}

    agree, disagree, wiz_only, game_only = [], [], [], []
    for i, kname in sorted(wiz.items()):
        want = _snake(kname)
        if i not in game_by_id:
            wiz_only.append((i, kname))
        elif game_by_id[i] == want:
            agree.append(i)
        else:
            disagree.append((i, kname, game_by_id[i]))
    for i, name in sorted(game_by_id.items()):
        if i not in wiz:
            game_only.append((i, name))
    return {"agree": len(agree), "disagree": disagree,
            "wizai_only": wiz_only, "game_only": game_only,
            "wizai_entries": len(wiz), "game_entries": len(game_by_id)}


def loader_coverage(spells_path="spells_full.json",
                    cards_path="cards_clean.json"):
    """Questions 2 and 3, straight from the loader's own skip report --
    the authority on what wizAi can actually build."""
    from data_full import load_spells_full

    report = {}
    cards = load_spells_full(spells_path, cards_path, report=report)
    skipped = report.get("skipped") or []

    raw = json.load(open(spells_path, encoding="utf-8"))
    ids_on_cards = Counter()
    for spell in raw:
        for e in spell.get("effects") or []:
            i = e.get("effect_type_id")
            if i is not None:
                ids_on_cards[i] += 1

    # Skip reasons name the effect, e.g. "undecoded effect kSummonCreature".
    by_effect = Counter()
    other = Counter()
    for entry in skipped:
        reason = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 \
            else str(entry)
        m = re.search(r"undecoded effect (\w+)", str(reason))
        if m:
            by_effect[m.group(1)] += 1
        else:
            other[re.sub(r"\d+", "N", str(reason))[:70]] += 1

    return {
        "records": len(raw),
        "usable_cards": len(cards),
        "skipped": len(skipped),
        "distinct_effect_ids_on_cards": len(ids_on_cards),
        "by_effect": by_effect,
        "other_reasons": other,
    }


def audit(spells_path="spells_full.json", cards_path="cards_clean.json"):
    enums = compare_enums()
    cov = loader_coverage(spells_path, cards_path)
    game = real_enum()

    lost = []
    for kname, n in cov["by_effect"].most_common():
        snake = _snake(kname)
        lost.append({
            "wizai_name": kname,
            "game_name": snake if snake in game else "<not in SpellEffects>",
            "cards_lost": n,
        })

    total = cov["records"]
    return {
        "enum_check": enums,
        "records": total,
        "usable_cards": cov["usable_cards"],
        "skipped": cov["skipped"],
        "pct_usable": round(100.0 * cov["usable_cards"] / total, 1) if total else 0,
        "cards_lost_to_undecoded_effects": sum(cov["by_effect"].values()),
        "lost_by_effect": lost,
        "other_skip_reasons": cov["other_reasons"].most_common(10),
    }


def render(a) -> str:
    e = a["enum_check"]
    out = ["=== 1. wizAi's effect table vs the client's SpellEffects enum ==="]
    out.append(f"  wizAi entries {e['wizai_entries']}, client entries "
               f"{e['game_entries']}, agreeing on id+name: {e['agree']}")
    if e["disagree"]:
        out.append(f"  NAME MISMATCHES ({len(e['disagree'])}):")
        for i, kname, real in e["disagree"][:15]:
            out.append(f"    id {i:>4}: wizAi {kname!r} vs client {real!r}")
    else:
        out.append("  no id/name disagreements -- wizAi's table is faithful "
                   "to the client")
    if e["wizai_only"]:
        out.append(f"  ids wizAi lists that the client's enum does not "
                   f"({len(e['wizai_only'])}): "
                   f"{', '.join(k for _, k in e['wizai_only'][:8])}"
                   f"{' ...' if len(e['wizai_only']) > 8 else ''}")
    if e["game_only"]:
        out.append(f"  ids the client has that wizAi's table lacks "
                   f"({len(e['game_only'])}): "
                   f"{', '.join(n for _, n in e['game_only'][:8])}"
                   f"{' ...' if len(e['game_only']) > 8 else ''}")

    out.append("")
    out.append("=== 2. what the decoder can build ===")
    out.append(f"  {a['usable_cards']} usable cards from {a['records']} "
               f"records ({a['pct_usable']}%)")
    out.append(f"  {a['skipped']} skipped, of which "
               f"{a['cards_lost_to_undecoded_effects']} to an undecoded effect")

    out.append("")
    out.append("=== 3. what each missing effect costs, in cards ===")
    rows = a["lost_by_effect"][:20]
    if rows:
        w = max(len(r["wizai_name"]) for r in rows)
        out.append(f"  {'effect':<{w}}  {'cards':>7}   in client enum?")
        for r in rows:
            known = "yes" if r["game_name"] != "<not in SpellEffects>" else "NO"
            out.append(f"  {r['wizai_name']:<{w}}  {r['cards_lost']:>7}   {known}")
    if a["other_skip_reasons"]:
        out.append("")
        out.append("  skips that are not an undecoded effect:")
        for reason, n in a["other_skip_reasons"]:
            out.append(f"    {n:>6}  {reason}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json")
    ap.add_argument("--spells", default="spells_full.json")
    ap.add_argument("--cards", default="cards_clean.json")
    args = ap.parse_args()
    a = audit(args.spells, args.cards)
    print(render(a))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(a, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
