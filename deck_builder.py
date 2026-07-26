"""
Deck-construction search: the outer half of the bilevel problem.

Replaces "pick one of a few hand-built decks" with a LEGAL deck space
(equipped-deck capacity, per-card copy limits, unlocked pool) searched in
two stages:

  1. SCREEN: sample legal candidates from role templates (hits, blades,
     traps, prism, heals, redundancy) and simulate each cheaply with the
     scripted blade-stack policy on paired seeds — a fast proxy scorer.
  2. FINE-TUNE: train the tabular combat policy on the shortlist and
     re-evaluate; score lexicographically by (win rate, speed, SMALLER
     deck) so extra cards must pay for themselves in reliability.

Also provides random_boss() and a held-out generalization check: build
against random opponents, evaluate on unseen seeds, so what transfers is
the RULE ("compact prism deck into same-school walls, redundancy priced
by fizzle risk"), not a memorized answer per boss.

Still roadmap (bilevel steps 3-4): a learned deck scorer replacing the
simulation screen, and joint deck+combat training with a single
deck-conditioned policy. Level gating comes straight from the game
data's level_restriction (null = no gate).
"""
import random
import re

from w101_sim import Sim, evaluate, make_blade_stack, Boss, OPPOSING
from rl_agent import QAgent


UNIVERSAL_BUFFS = {"Tri Blade", "Tri Trap", "Spirit Blade", "Spirit Trap",
                   "Hex", "Curse", "Feint", "Balanceblade", "Bladestorm"}

# the dump tags boss-only / encounter-scripted spells as variant='core'
# too ("Scald - KRBoss Death", "NA PL-Bear-Tweedle-B"): the first deck
# search promptly reward-hacked them into one-turn kills. Player-trainable
# spells carry none of these markers and obey era damage efficiency.
_INTERNAL = re.compile(
    r"(-|_|\bNA\b|BOSS|Tutorial|Mutate|Mashup|FUSE|Loremaster|Token|"
    r"Polymorph|Test|\d|\bAdv\b|\bMass\b|"
    r"\s(Sun|Moon|Star|Shadow)$)", re.IGNORECASE)   # enchant variants


def _player_plausible(name, c):
    if _INTERNAL.search(name):
        return False
    if c.kind in ("damage", "drain"):
        per_pip = c.damage if c.x_pips else c.damage / max(c.pips, 1)
        if per_pip > 200:
            return False
    if c.kind == "blade" and c.percent > 0.45:
        return False
    if c.kind in ("trap", "weakness") and abs(c.percent) > 0.75:
        return False
    return True


def legal_pool(cards, school, level=None):
    """Unlocked, deck-buildable cards for a school: own-school trained
    spells plus the cross-trained universal buffs, with boss-only and
    internal spells screened out. `level` gates by the game data's
    level_restriction (null = available from level 1) — the progression
    knob: legal_pool(cards, 'fire', level=12) is a level-12 wizard."""
    pool = {}
    for name, c in cards.items():
        if c.source != "deck" or not _player_plausible(name, c):
            continue
        if level is not None and getattr(c, "level", 1) > level:
            continue
        if c.school == school or name in UNIVERSAL_BUFFS:
            if c.kind in ("damage", "drain", "blade", "trap", "prism",
                          "shield", "heal", "weakness"):
                pool[name] = c
    return pool


def sample_deck(pool, school, boss, rng, capacity=16, copy_limit=3):
    """One legal candidate from a role template with sampled counts."""
    hits = sorted((c for c in pool.values()
                   if c.kind in ("damage", "drain") and not c.x_pips),
                  key=lambda c: -c.damage)[:4]
    xpips = [c for c in pool.values() if c.x_pips and
             c.kind in ("damage", "drain")]
    blades = sorted((c for c in pool.values() if c.kind == "blade"),
                    key=lambda c: -c.percent)[:4]
    traps = sorted((c for c in pool.values() if c.kind == "trap"),
                   key=lambda c: -c.percent)[:4]
    prisms = [c for c in pool.values() if c.kind == "prism"
              and boss.incoming_mult(school) < 1.0]
    if not hits and not xpips:
        return None
    deck = []

    def add(card, n):
        room = capacity - len(deck)
        deck.extend([card.name] * min(n, copy_limit, room))

    n_hits = rng.randint(2, 5)
    top = hits[0] if hits else xpips[0]
    add(top, rng.randint(1, min(n_hits, copy_limit)))
    backups = [c for c in hits[1:]] + [c for c in xpips if c is not top]
    rng.shuffle(backups)
    for c in backups:
        if sum(1 for n in deck if pool[n].kind in ("damage", "drain")) \
                >= n_hits:
            break
        add(c, 1)
    for c in rng.sample(blades, k=min(len(blades), rng.randint(0, 3))):
        add(c, rng.randint(1, 2))
    for c in rng.sample(traps, k=min(len(traps), rng.randint(0, 3))):
        add(c, rng.randint(1, 2))
    if prisms and rng.random() < 0.8:
        add(prisms[0], rng.randint(1, 2))
    heals = [c for c in pool.values() if c.kind == "heal"]
    if heals and rng.random() < 0.3:
        add(rng.choice(heals), 1)
    return deck if deck else None


def check_legal(deck, capacity, copy_limit):
    from collections import Counter
    return len(deck) <= capacity and \
        all(v <= copy_limit for v in Counter(deck).values())


def screen(cards, decks, school, boss, rules=None, n=250, base_seed=7000):
    """Cheap proxy scores: scripted policy on paired seeds."""
    out = []
    for dl in decks:
        sim = Sim(dict(cards), dl, school, boss, player_hp=10**9,
                  rules=rules)
        wins, ttk = 0, []
        for i in range(n):
            sim.rng = random.Random(base_seed + i)
            t, won, _ = sim.run(make_blade_stack(3))
            if won:
                wins += 1
                ttk.append(t)
        out.append((wins / n, sum(ttk) / len(ttk) if ttk else 99.0, dl))
    return sorted(out, key=lambda r: (-round(r[0], 2), r[1], len(r[2])))


def fine_tune(cards, dl, school, boss, rules=None, episodes=8000, seed=0):
    """Train the combat policy on one candidate; return (win, ttk, pol)."""
    sim = Sim(dict(cards), dl, school, boss, player_hp=10**9, rules=rules,
              rng=random.Random(seed))
    agent = QAgent(dict(cards), dl, school, rng=random.Random(seed + 1))
    for ep in range(episodes):
        frac = ep / episodes
        agent.alpha = 0.30 * (1 - 0.9 * frac)
        agent.train_episode(sim, eps=max(0.02, 0.3 * (1 - frac)), dp_w=0.0)
    w, m = evaluate(sim, agent.policy(), n=2000)
    return w, m, agent


def build_deck(cards, school, boss, rules=None, n_candidates=150,
               top_k=5, capacity=16, copy_limit=3, seed=0, log=print,
               level=None):
    """Two-stage search over the legal deck space. Returns
    (deck, win, ttk, screen_table). `level` gates the unlocked pool."""
    rng = random.Random(seed)
    pool = legal_pool(cards, school, level=level)
    seen, cands = set(), []
    while len(cands) < n_candidates:
        dl = sample_deck(pool, school, boss, rng, capacity, copy_limit)
        if dl is None:
            break
        key = tuple(sorted(dl))
        if key in seen or not check_legal(dl, capacity, copy_limit):
            continue
        seen.add(key)
        cands.append(dl)
    if log:
        log(f"screening {len(cands)} legal candidates "
            f"(capacity {capacity}, copy limit {copy_limit})")
    table = screen(cards, cands, school, boss, rules)
    best = None
    for w0, m0, dl in table[:top_k]:
        w, m, _ = fine_tune(cards, dl, school, boss, rules, seed=seed)
        if log:
            log(f"  size {len(dl):>2}  screen {w0*100:3.0f}%/{m0:5.2f}  "
                f"RL {w*100:5.1f}%/{m:5.2f}  {sorted(set(dl))}")
        score = (round(w, 2), -m, -len(dl))
        if best is None or score > best[0]:
            best = (score, dl, w, m)
    return best[1], best[2], best[3], table


# ---------------------------------------------------------------- opponents

def random_boss(rng, name="rand"):
    """Randomized opponent for generalization training/testing."""
    school = rng.choice(("fire", "ice", "storm", "myth", "life", "death"))
    hp = rng.randrange(500, 8001, 100)
    b = Boss(f"{name}-{school}-{hp}", hp, school, 0)
    b.resist_map = {school: round(rng.uniform(0.2, 0.8), 2)}
    if rng.random() < 0.5:
        b.resist_map["*"] = round(rng.uniform(0.0, 0.3), 2)
    opp = OPPOSING.get(school)
    b.boost_map = {opp: round(rng.uniform(0.0, 0.5), 2)} if opp else {}
    return b


if __name__ == "__main__":
    from data_full import load_spells_full, load_bosses_full, LIVE_DECKS, \
        LIVE_RULES
    import copy as _copy
    cards = load_spells_full()
    bosses, _ = load_bosses_full()

    print("== deck search: death vs Jade Oni (live rules) ==")
    boss = _copy.copy(bosses["Jade Oni"])
    boss.dmg = 0
    dl, w, m, _ = build_deck(cards, "death", boss, LIVE_RULES,
                             n_candidates=120, top_k=4, seed=0)
    print(f"built (size {len(dl)}): win {w*100:.1f}%  ttk {m:.2f}")
    hand = LIVE_DECKS["death"]["oneshot"]
    w0, m0, _ = fine_tune(cards, hand, "death", boss, LIVE_RULES, seed=0)
    print(f"hand-built oneshot (size {len(hand)}): win {w0*100:.1f}%  "
          f"ttk {m0:.2f}")

    print("\n== generalization: build vs 3 random bosses, held-out seeds ==")
    rng = random.Random(42)
    for i in range(3):
        rb = random_boss(rng)
        dl, w, m, _ = build_deck(cards, "ice", rb, LIVE_RULES,
                                 n_candidates=60, top_k=3, seed=i, log=None)
        prisms = sum(1 for n in dl if cards[n].kind == "prism")
        print(f"{rb.name:<18} resist={rb.resist_map} -> size {len(dl)}, "
              f"prisms {prisms}, win {w*100:.1f}%, ttk {m:.2f}")
