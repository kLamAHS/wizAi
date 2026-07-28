"""
End-to-end training run: one command, every layer that has been built.

    python main.py                 # ~5 minutes, 5 levels
    python main.py --full          # every world boundary, bigger search
    python main.py --school storm --levels 30,60,90

What it puts together, per level, in the order a wizard actually
experiences them:

  1. GEAR + PET     gear.loadout(level, school) — the report's damage /
                    resist / crit envelope, school-tilted, plus a
                    mega triple-double pet
  2. ENCOUNTER      a real creature from the world that level is
                    questing, with the companions the scrape names for
                    it, everything CASTING from a spell pool rather
                    than auto-attacking
  3. DECK           deck_builder.build_deck over the level-gated pool,
                    with Sun enchantments offered once they unlock and
                    charged their real two-slot cost
  4. POLICY         a tabular Q-learner fine-tuned on that exact
                    (deck, encounter) pair
  5. SCORE          the trained policy against the best scripted race
                    line, paired seeds

then writes results_main.json and renders the charts.

Nothing here is new science — it is the assembly. Every component
carries its own confidence tag; see the README for which numbers are
scraped, which are modeled, and which are inferred.
"""
import argparse
import copy
import json
import random
import time

from data_full import LIVE_RULES, encounter, load_bosses_full, load_spells_full
from deck_builder import build_deck, fine_tune, legal_pool
from gear import loadout
from w101_sim import (Sim, evaluate_paired, is_enchanted, make_blade_stack,
                      enchanted_deck_size)
from worlds import world_for

SCHOOLS = ("fire", "ice", "storm", "myth", "life", "death", "balance")
DEFAULT_LEVELS = (20, 50, 70, 100, 120)
FULL_LEVELS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120)
RACE = {f"race({k})": make_blade_stack(k) for k in (1, 2, 3, 4, 5, 6)}


def pick_encounter(level, school, cards, bosses, registry, stats, rng,
                   log=None):
    """A creature from the world this level is questing, preferring one
    that brings company — and SCREENED for solo feasibility.

    The screen is not cosmetic. Much of this registry is four-person
    dungeon content: the first end-to-end run picked by size alone,
    drew Gurtok Firebender with two adds, and returned 0% at every
    level, which trains nothing and plots nothing. Candidates are
    probed with a cheap scripted line and the first one a solo wizard
    can actually contest is used."""
    world = world_for(level)
    pool = [b for b in bosses.values()
            if registry[b.name].get("world") == world and b.rank]
    if not pool:
        pool = [b for b in bosses.values() if b.rank]
    pool.sort(key=lambda b: (0 if b.minions else 1, b.hp, b.name))
    probe_deck = _probe_deck(school, cards, level)
    fallback = None
    for b in pool[:40]:
        _x, comps = encounter(b.name, bosses)
        sim = Sim(dict(cards), probe_deck, school, copy.copy(b),
                  player_hp=stats["player_hp"], rules=LIVE_RULES,
                  player_stats=stats["player_stats"],
                  power_pip=stats["power_pip"],
                  enemies=[copy.copy(c) for c in comps] or None)
        w = max(v["win_rate"] for v in
                evaluate_paired(sim, RACE, n=120).values())
        if fallback is None or w > fallback[1]:
            fallback = (b, w)
        if 0.15 <= w <= 0.90:
            if log:
                log(f"  screened  {b.name} — scripted probe "
                    f"{w*100:.0f}%, contested")
            return b
    if log:
        log(f"  screened  no contested encounter in {world}; using the "
            f"most winnable ({fallback[0].name}, {fallback[1]*100:.0f}%)")
    return fallback[0]


def _probe_deck(school, cards, level):
    """A plain level-appropriate deck for the feasibility screen."""
    pool = legal_pool(cards, school, level=level)
    hits = sorted((c for c in pool.values()
                   if c.kind in ("damage", "drain") and not c.x_pips),
                  key=lambda c: -c.damage)[:2]
    blades = sorted((c for c in pool.values() if c.kind == "blade"),
                    key=lambda c: -c.percent)[:1]
    traps = sorted((c for c in pool.values() if c.kind == "trap"),
                   key=lambda c: -c.percent)[:1]
    deck = [c.name for c in hits] * 3 + [c.name for c in blades] * 3 \
        + [c.name for c in traps] * 2
    return deck or [next(iter(pool))]


def run_level(level, school, cards, bosses, registry, args, log):
    rng = random.Random(1000 + level)
    stats = loadout(level, school, rules=LIVE_RULES, pet=args.pet)
    boss = pick_encounter(level, school, cards, bosses, registry, stats,
                          rng, log)
    _b, comps = encounter(boss.name, bosses)
    pool = legal_pool(cards, school, level=level, enchants=True)
    log(f"\n{'='*66}")
    log(f"LEVEL {level}  ({world_for(level)})  — {school} wizard")
    log(f"{'='*66}")
    log(f"  gear      {stats['player_hp']:,} HP, "
        f"{stats['power_pip']*100:.0f}% power pips, "
        f"damage {stats['player_stats'].get('damage', {})}")
    log(f"  pet       {args.pet or 'none'}")
    log(f"  encounter {boss.name} — {boss.hp:,} HP, rank {boss.rank}, "
        f"{boss.school}")
    log(f"            casts {boss.pool}")
    for c in comps:
        log(f"            + {c.name} ({c.hp:,} HP)")
    log(f"  pool      {len(pool)} cards, "
        f"{sum(1 for n in pool if is_enchanted(n))} enchanted")

    t = time.time()
    log(f"  building deck ({args.candidates} candidates)...")
    dl, w_deck, ttk_deck, _ = build_deck(
        cards, school, copy.copy(boss), LIVE_RULES,
        n_candidates=args.candidates, top_k=args.top_k,
        capacity=args.capacity, seed=7, level=level, enchants=True,
        enemies=[copy.copy(c) for c in comps] or None,
        player_hp=stats["player_hp"], player_stats=stats["player_stats"],
        power_pip=stats["power_pip"], log=None)
    log(f"  deck      {len(dl)} cards / {enchanted_deck_size(dl)} slots "
        f"({time.time()-t:.0f}s)")
    for name in sorted(set(dl)):
        log(f"              {dl.count(name)}x {name}")

    full = {**cards, **pool}
    t = time.time()
    log(f"  training policy ({args.episodes:,} episodes)...")
    w_rl, ttk_rl, agent = fine_tune(
        full, dl, school, copy.copy(boss), LIVE_RULES,
        episodes=args.episodes, seed=11, player_hp=stats["player_hp"],
        power_pip=stats["power_pip"],
        enemies=[copy.copy(c) for c in comps] or None,
        player_stats=stats["player_stats"])
    log(f"  trained   {time.time()-t:.0f}s")

    sim = Sim(dict(full), dl, school, copy.copy(boss),
              player_hp=stats["player_hp"], rules=LIVE_RULES,
              player_stats=stats["player_stats"],
              power_pip=stats["power_pip"],
              enemies=[copy.copy(c) for c in comps] or None)
    arms = evaluate_paired(sim, {**RACE, "trained": agent.policy()},
                           n=args.eval_n)
    scripted = max((k for k in arms if k != "trained"),
                   key=lambda k: (round(arms[k]["win_rate"], 2),
                                  -arms[k]["mean_ttk"]))
    tr, sc = arms["trained"], arms[scripted]
    log(f"  RESULT    trained {tr['win_rate']*100:5.1f}% / "
        f"ttk {tr['mean_ttk']:5.2f}      "
        f"best scripted {scripted} {sc['win_rate']*100:5.1f}% / "
        f"ttk {sc['mean_ttk']:5.2f}")
    edge = (tr["win_rate"] - sc["win_rate"]) * 100
    log(f"            edge {edge:+.1f} points")
    if edge < -5 and comps:
        # not a training bug — QAgent picks CARDS, not targets, so on a
        # multi-enemy board it cannot commit to killing one thing. This
        # is the representation deficit mob_generalist.py isolated
        # (advisor-guided training moves it 4.7% -> 6.2%, i.e. barely).
        log(f"            NOTE: the learner cannot choose targets, so "
            f"it loses on multi-enemy boards; shipping the scripted "
            f"line for this level")
    return {
        "level": level, "world": world_for(level), "school": school,
        "boss": boss.name, "boss_hp": boss.hp, "boss_rank": boss.rank,
        "boss_pool": boss.pool, "companions": [c.name for c in comps],
        "player_hp": stats["player_hp"],
        "power_pip": stats["power_pip"],
        "deck": dl, "deck_cards": len(dl),
        "deck_slots": enchanted_deck_size(dl),
        "enchanted": sum(1 for n in dl if is_enchanted(n)),
        "n_companions": len(comps),
        "best": ("trained" if tr["win_rate"] >= sc["win_rate"]
                 else scripted),
        "trained": {"win": tr["win_rate"], "ttk": tr["mean_ttk"],
                    "p90": tr["p90_ttk"]},
        "scripted": {"policy": scripted, "win": sc["win_rate"],
                     "ttk": sc["mean_ttk"], "p90": sc["p90_ttk"]},
        "all_policies": {k: {"win": v["win_rate"], "ttk": v["mean_ttk"]}
                         for k, v in arms.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--school", default="fire", choices=SCHOOLS)
    ap.add_argument("--levels", default=None,
                    help="comma-separated, e.g. 30,60,90")
    ap.add_argument("--full", action="store_true",
                    help="every world boundary, bigger search")
    ap.add_argument("--pet", default="triple-double",
                    help="triple-double | quad-critical | none")
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--capacity", type=int, default=24)
    ap.add_argument("--episodes", type=int, default=12000)
    ap.add_argument("--eval-n", type=int, default=1500)
    ap.add_argument("--out", default="results_main.json")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()
    if args.pet == "none":
        args.pet = None
    if args.full:
        args.candidates = max(args.candidates, 60)
        args.top_k = max(args.top_k, 3)
        args.episodes = max(args.episodes, 12000)
    levels = ([int(x) for x in args.levels.split(",")] if args.levels
              else list(FULL_LEVELS if args.full else DEFAULT_LEVELS))

    t0 = time.time()
    print(f"wizAi — end-to-end run: {args.school} wizard, levels "
          f"{levels}", flush=True)
    print(f"loading spells and CASTING bosses...", flush=True)
    cards = load_spells_full()
    bosses, registry = load_bosses_full(spell_pools=True, cards=cards)
    print(f"  {len(cards):,} cards, {len(bosses):,} creatures "
          f"({sum(1 for b in bosses.values() if b.pool):,} with spell "
          f"pools)", flush=True)

    def log(msg):
        print(msg, flush=True)

    rows = []
    for lvl in levels:
        rows.append(run_level(lvl, args.school, cards, bosses, registry,
                              args, log))

    out = {"setup": {"school": args.school, "levels": levels,
                     "pet": args.pet, "candidates": args.candidates,
                     "episodes": args.episodes, "eval_n": args.eval_n,
                     "ruleset": LIVE_RULES.ruleset_id,
                     "boss_model": "spell pools (casting)",
                     "seconds": round(time.time() - t0)},
           "levels": rows}
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\n{'='*66}\nSUMMARY\n{'='*66}", flush=True)
    print(f"{'lvl':<5}{'world':<13}{'encounter':<24}{'+':<3}{'deck':<6}"
          f"{'trained':<10}{'scripted':<10}{'edge':<8}{'ship'}",
          flush=True)
    for r in rows:
        print(f"{r['level']:<5}{r['world']:<13}"
              f"{r['boss'][:22]:<24}{r['n_companions']:<3}"
              f"{r['deck_cards']:<6}"
              f"{r['trained']['win']*100:>6.1f}%   "
              f"{r['scripted']['win']*100:>6.1f}%   "
              f"{(r['trained']['win']-r['scripted']['win'])*100:>+6.1f}  "
              f"  {r['best']}", flush=True)
    solo = [r for r in rows if not r["n_companions"]]
    mob = [r for r in rows if r["n_companions"]]
    if solo and mob:
        ds = sum(r["trained"]["win"] - r["scripted"]["win"]
                 for r in solo) / len(solo) * 100
        dm = sum(r["trained"]["win"] - r["scripted"]["win"]
                 for r in mob) / len(mob) * 100
        print(f"\nlearner vs scripted: {ds:+.1f} points on solo bosses, "
              f"{dm:+.1f} on multi-enemy boards", flush=True)
        print(f"  the learner ranks CARDS, not targets — see "
              f"mob_generalist.py for the isolation of that deficit",
              flush=True)
    print(f"\nwrote {args.out} ({time.time()-t0:.0f}s)", flush=True)

    if not args.no_charts:
        import plots
        made = plots.main_run(args.out)
        print(f"charts: {', '.join(str(p) for p in made)}", flush=True)


if __name__ == "__main__":
    main()
