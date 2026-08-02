"""Where does the continuation seat matter? The campaign-4 method.

Campaign 3 failed by specialising where the seat turned out to be
flat; campaign 4 shipped on the first try by measuring first. This is
the measuring: for each deck configuration, auto-locate a contested
health (win rate steered toward the discriminating middle under the
default), then run EVERY continuation candidate paired and record the
spread, the leader, and where the neural seats land.

Reading the output:
  - spread < ~5: the seat is not a lever on this board; nothing to
    win here (campaign 3's lesson).
  - leader near ceiling: covered ground; matching it is redundancy
    (campaign 2's lesson).
  - spread large AND a hand-coded leader: campaign material. The bar
    for any specialist trained on it is BEATING that leader on this
    exact held-out point.

PASS 1 RESULTS (L30 tier, 2-mob flat boards, 2026-08): the shipped
first net already led low(ice) 40.0, mid(ice) 42.0 and L30 life 64.4;
L30 ice was flat (spread 1.6) and L30 death near-ceiling (96.8). The
two real levers -- buffy(ice), spread 34.4, and L30 myth, spread
15.2, both charm-heavy, both hand-coded-led -- became campaign 4,
which took the buffy lever (38.2/37.8 vs 32.9/31.9 held out) and now
holds it as neural-b.

    python3 continuation_survey.py [LEVEL] [MOBS]
"""
import random
import sys

sys.path.insert(0, ".")

from data_full import LIVE_RULES, load_spells_full
from deck_builder import legal_pool
from deimos_bridge import policies as P
from w101_sim import Boss, Sim, evaluate_paired

CARDS = load_spells_full()
SCHOOLS = ("fire", "ice", "storm", "myth", "life", "death", "balance")


def survey_deck(school, level):
    """Representative deck off the school's legal pool: top hits plus
    a blade, a trap and a shield -- the shape real decks have."""
    pool = legal_pool(CARDS, school, level=level)
    hits = sorted((c for c in pool.values()
                   if c.kind in ("damage", "drain") and not c.x_pips),
                  key=lambda c: -c.damage)
    picks = []
    for c in hits[:3]:
        picks += [c.name] * 4
    for kind, n in (("blade", 2), ("trap", 2), ("shield", 2)):
        of = [c for c in pool.values() if c.kind == kind]
        if of:
            picks += [of[0].name] * n
    return picks


def _win(deck, school, php, dmgb, hp, mobs, dmg, n, cont=None):
    boss = Boss(name="p", hp=hp, school="death", dmg=dmg)
    extra = [Boss(name=f"p{i}", hp=hp, school="death", dmg=dmg)
             for i in range(1, mobs)]
    sim = Sim(CARDS, list(deck), school, boss, enemies=extra,
              player_hp=php, player_stats={"damage": {"*": dmgb}},
              rules=LIVE_RULES)
    pol = P.greedy_ttk(6, continuation=cont)
    wins = 0
    for i in range(n):
        sim.rng = random.Random(210_000 + i)
        _, won, _ = sim.run(pol, max_turns=25)
        wins += won
    return wins / n


def survey(configs, mobs=2):
    for label, deck, school, php, dmgb in configs:
        dmg = max(40, php // (9 + 2 * mobs))
        contested = None
        hp = 400
        # geometric ladder, extended until the deck actually struggles
        # -- pass 1 declared the fire deck "saturated" because the
        # ladder simply stopped too low
        while hp < 12000:
            w = _win(deck, school, php, dmgb, hp, mobs, dmg, 80)
            if w < 0.2:
                break
            if w <= 0.7:
                contested = (hp, w)
                break
            hp = int(hp * 1.4)
        if contested is None:
            print(f"{label:14s} no contested point below 12k", flush=True)
            continue
        hp, _w0 = contested

        boss = Boss(name="p", hp=hp, school="death", dmg=dmg)
        extra = [Boss(name=f"p{i}", hp=hp, school="death", dmg=dmg)
                 for i in range(1, mobs)]
        sim = Sim(CARDS, list(deck), school, boss, enemies=extra,
                  player_hp=php, player_stats={"damage": {"*": dmgb}},
                  rules=LIVE_RULES)
        pols = {nm: P.greedy_ttk(6, continuation=P.build_continuation(nm))
                for nm in P.CONTINUATIONS}
        st = evaluate_paired(sim, pols, n=250, base_seed=210_000)
        ranked = sorted(((v["win_rate"], k) for k, v in st.items()),
                        reverse=True)
        spread = (ranked[0][0] - ranked[-1][0]) * 100
        nets = " ".join(
            f"{k}#{i}={st[k]['win_rate']*100:.1f}"
            for i, (_, k) in enumerate(ranked, 1) if k.startswith("neural"))
        print(f"{label:14s} hp {hp:5d}x{mobs} d{dmg:3d}: "
              f"spread {spread:5.1f}  "
              f"leader {ranked[0][1]}={ranked[0][0]*100:.1f}  {nets}",
              flush=True)


if __name__ == "__main__":
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    mobs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    from player_curves import school_hp

    configs = [(f"L{level} {sc}", survey_deck(sc, level), sc,
                int(school_hp(sc, level) * 1.2), 0.15 + level * 0.002)
               for sc in SCHOOLS]
    survey(configs, mobs)
