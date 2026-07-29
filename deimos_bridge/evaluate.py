"""Score a wizAi-trained policy under Deimos' stat model.

    python -m deimos_bridge.evaluate --levels 20,50,70,100,120

The learner trains where it always has: `w101_sim`, linear gear stats,
the pipeline `main.py` runs. Then the *same trained policy*, on the same
deck against the same encounter, is scored a second time with Deimos'
diminishing-returns curves in place. Nothing else changes, so the
difference between the two columns is attributable to the stat model.

Two numbers matter, and they answer different questions:

  win rate     how the wizard does. Expected to fall once damage stops
               being linear -- that is the curve doing its job, and it
               costs the scripted lines as much as the learner.
  edge         trained minus best scripted, under each engine. This is
               the one that speaks to the ML. An edge that survives the
               curve was a real policy; an edge that only exists under
               linear stats was the learner exploiting a modelling gap.

`--sweep` varies the damage cap instead of fixing it, because the cap is
assumed rather than measured (see `caches.DuelCurve`). A conclusion that
holds across the sweep does not depend on getting the constant right.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time

from data_full import LIVE_RULES, encounter, load_bosses_full, load_spells_full
from deck_builder import build_deck, fine_tune, legal_pool
from gear import loadout
from w101_sim import Sim, evaluate_paired, make_blade_stack, with_focus
from worlds import world_for

from deimos_bridge.caches import DuelCurve
from deimos_bridge.engine import DeimosSim

SCRIPTED = {f"race({k})": make_blade_stack(k) for k in (1, 2, 3, 4, 5, 6)}
SCRIPTED |= {f"focus({k})": with_focus(make_blade_stack(k)) for k in (1, 2, 3)}


def _best_scripted(arms):
    """The strongest non-learned line, by win rate then speed."""
    return max((k for k in arms if k != "trained"),
               key=lambda k: (round(arms[k]["win_rate"], 2),
                              -arms[k]["mean_ttk"]))


def _edge(arms):
    """Trained minus best scripted, in win-rate points."""
    scripted = _best_scripted(arms)
    return (arms["trained"]["win_rate"] - arms[scripted]["win_rate"]) * 100, scripted


def _fight_kwargs(stats, comps):
    """Shared by `Sim`, `build_deck` and `fine_tune`.

    Deliberately excludes `rules`: the two pipeline functions take it as
    their fourth positional argument, so passing it here as well is a
    duplicate-argument TypeError.
    """
    return dict(player_hp=stats["player_hp"],
                player_stats=stats["player_stats"],
                power_pip=stats["power_pip"],
                enemies=[copy.copy(c) for c in comps] or None)


def _sim_kwargs(stats, comps):
    return dict(_fight_kwargs(stats, comps), rules=LIVE_RULES)


def run_level(level, school, cards, bosses, registry, args, log):
    """Train once under wizAi, score under both engines."""
    from main import pick_encounter

    rng = random.Random(1000 + level)
    stats = loadout(level, school, rules=LIVE_RULES)
    boss = pick_encounter(level, school, cards, bosses, registry, stats, rng)
    _b, comps = encounter(boss.name, bosses)
    pool = legal_pool(cards, school, level=level, enchants=True)
    full = {**cards, **pool}

    log(f"\n{'=' * 70}")
    log(f"LEVEL {level}  ({world_for(level)})  — {school} wizard "
        f"vs {boss.name} ({boss.hp:,} HP, rank {boss.rank})")
    gear_damage = stats["player_stats"].get("damage", {})
    log(f"  gear damage {gear_damage}  "
        f"(the curve bends above {(args.d_k0 + args.d_n0) / 100:.2f})")

    t = time.time()
    decklist, *_ = build_deck(
        cards, school, copy.copy(boss), LIVE_RULES,
        n_candidates=args.candidates, top_k=args.top_k, seed=7, level=level,
        enchants=True, log=None, **_fight_kwargs(stats, comps))
    log(f"  deck        {len(decklist)} cards ({time.time() - t:.0f}s)")

    probe = Sim(dict(full), decklist, school, copy.copy(boss),
                **_sim_kwargs(stats, comps))
    scored = evaluate_paired(probe, SCRIPTED, n=200)
    advisor = max(scored, key=lambda k: (round(scored[k]["win_rate"], 2),
                                         -scored[k]["mean_ttk"]))

    t = time.time()
    log(f"  training    {args.episodes:,} episodes (advisor {advisor})...")
    _w, _ttk, agent = fine_tune(
        full, decklist, school, copy.copy(boss), LIVE_RULES,
        episodes=args.episodes, seed=11, advisor=SCRIPTED[advisor],
        **_fight_kwargs(stats, comps))
    log(f"  trained     {time.time() - t:.0f}s")

    policies = {**SCRIPTED, "trained": agent.policy()}
    curve = DuelCurve(damage_limit=args.damage_limit, d_k0=args.d_k0,
                      d_n0=args.d_n0, resist_limit=args.resist_limit,
                      r_k0=args.r_k0, r_n0=args.r_n0)

    out = {"level": level, "world": world_for(level), "school": school,
           "boss": boss.name, "boss_hp": boss.hp, "boss_rank": boss.rank,
           "companions": [c.name for c in comps], "deck": decklist,
           "gear_damage": gear_damage, "advisor": advisor, "engines": {}}

    for label, factory in (("wizai", Sim),
                           ("deimos", lambda *a, **k: DeimosSim(*a, curve=curve, **k))):
        sim = factory(dict(full), decklist, school, copy.copy(boss),
                      **_sim_kwargs(stats, comps))
        arms = evaluate_paired(sim, policies, n=args.eval_n)
        edge, scripted = _edge(arms)
        trained = arms["trained"]
        out["engines"][label] = {
            "trained_win": trained["win_rate"], "trained_ttk": trained["mean_ttk"],
            "best_scripted": scripted,
            "scripted_win": arms[scripted]["win_rate"],
            "scripted_ttk": arms[scripted]["mean_ttk"],
            "edge_points": edge,
        }
        log(f"  {label:8s}    trained {trained['win_rate'] * 100:5.1f}% "
            f"/ ttk {trained['mean_ttk']:5.2f}   "
            f"best scripted {scripted} {arms[scripted]['win_rate'] * 100:5.1f}%   "
            f"edge {edge:+.1f}")

    wiz, dei = out["engines"]["wizai"], out["engines"]["deimos"]
    out["win_delta"] = (dei["trained_win"] - wiz["trained_win"]) * 100
    out["edge_delta"] = dei["edge_points"] - wiz["edge_points"]
    log(f"  DELTA       win {out['win_delta']:+.1f} points, "
        f"edge {out['edge_delta']:+.1f} points")
    return out


def sweep_curve(level, school, cards, bosses, registry, args, log,
                caps=(1.0, 1.4, 1.8, 2.4, 999.0)):
    """Re-score one level across damage caps, including no cap at all.

    `999.0` is effectively linear, so it should reproduce the wizAi
    column and acts as a control: if it does not, the splice is wrong
    rather than the curve interesting.
    """
    from main import pick_encounter

    rng = random.Random(1000 + level)
    stats = loadout(level, school, rules=LIVE_RULES)
    boss = pick_encounter(level, school, cards, bosses, registry, stats, rng)
    _b, comps = encounter(boss.name, bosses)
    pool = legal_pool(cards, school, level=level, enchants=True)
    full = {**cards, **pool}

    decklist, *_ = build_deck(
        cards, school, copy.copy(boss), LIVE_RULES,
        n_candidates=args.candidates, top_k=args.top_k, seed=7, level=level,
        enchants=True, log=None, **_fight_kwargs(stats, comps))
    _w, _ttk, agent = fine_tune(
        full, decklist, school, copy.copy(boss), LIVE_RULES,
        episodes=args.episodes, seed=11, **_fight_kwargs(stats, comps))
    policies = {**SCRIPTED, "trained": agent.policy()}

    rows = []
    log(f"\n  curve sweep at level {level} vs {boss.name}")
    for cap in caps:
        curve = DuelCurve(damage_limit=cap, d_k0=args.d_k0, d_n0=args.d_n0,
                          resist_limit=args.resist_limit, r_k0=args.r_k0,
                          r_n0=args.r_n0)
        sim = DeimosSim(dict(full), decklist, school, copy.copy(boss),
                        curve=curve, **_sim_kwargs(stats, comps))
        arms = evaluate_paired(sim, policies, n=args.eval_n)
        edge, scripted = _edge(arms)
        rows.append({"damage_limit": cap,
                     "trained_win": arms["trained"]["win_rate"],
                     "trained_ttk": arms["trained"]["mean_ttk"],
                     "edge_points": edge, "best_scripted": scripted})
        log(f"    cap {cap:6.1f}  trained {arms['trained']['win_rate'] * 100:5.1f}% "
            f"/ ttk {arms['trained']['mean_ttk']:5.2f}   edge {edge:+.1f}")
    return {"level": level, "boss": boss.name, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--school", default="fire")
    p.add_argument("--levels", default="20,50,70,100,120")
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--eval-n", type=int, default=600)
    p.add_argument("--candidates", type=int, default=60)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--damage-limit", type=float, default=1.40)
    p.add_argument("--d-k0", type=float, default=60.0)
    p.add_argument("--d-n0", type=float, default=25.0)
    p.add_argument("--resist-limit", type=float, default=0.85)
    p.add_argument("--r-k0", type=float, default=40.0)
    p.add_argument("--r-n0", type=float, default=15.0)
    p.add_argument("--sweep", type=int, default=0,
                   help="also sweep the damage cap at this level")
    p.add_argument("--out", default="results_deimos.json")
    args = p.parse_args()

    def log(msg):
        print(msg, flush=True)

    cards = load_spells_full()
    bosses, registry = load_bosses_full(rules=LIVE_RULES, spell_pools=True,
                                        cards=cards, fill_boost=True)
    log(f"loaded {len(cards):,} cards, {len(bosses):,} creatures")
    log(f"curve: damage cap {args.damage_limit}, bends above "
        f"{(args.d_k0 + args.d_n0) / 100:.2f}; resist cap {args.resist_limit}")
    log("NOTE: the curve constants are assumed, not measured -- Deimos "
        "reads them from the live client. Use --sweep to test sensitivity.")

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    results = [run_level(lv, args.school, cards, bosses, registry, args, log)
               for lv in levels]

    payload = {"school": args.school, "curve": vars(args), "levels": results}
    if args.sweep:
        payload["sweep"] = sweep_curve(args.sweep, args.school, cards, bosses,
                                       registry, args, log)

    log(f"\n{'=' * 70}\nSUMMARY  (win% and edge, wizAi -> Deimos curves)")
    log(f"{'level':>6} {'boss':<28} {'wizAi win':>10} {'deimos win':>11} "
        f"{'wizAi edge':>11} {'deimos edge':>12}")
    for r in results:
        w, d = r["engines"]["wizai"], r["engines"]["deimos"]
        log(f"{r['level']:>6} {r['boss'][:28]:<28} "
            f"{w['trained_win'] * 100:>9.1f}% {d['trained_win'] * 100:>10.1f}% "
            f"{w['edge_points']:>+10.1f} {d['edge_points']:>+11.1f}")

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
