"""
Learned deck scorer: bilevel rung 3.

build_deck's screen stage is the cost center — 250 paired-seed fights
per candidate deck. This module learns a SURROGATE of that screen from
logged (deck, boss, screen result) pairs: closed-form ridge regression
over hand-crafted deck-vs-boss features. build_deck(scorer=...) then
pre-ranks candidates with the surrogate and simulates only the top
slice, so the same search visits a third of the fights.

Honesty rules, in order:
  - validation is split BY BOSS, never by row (rows from one screen
    share a boss; a row split would leak the answer),
  - we report Spearman rank correlation and top-1 regret (screen win
    the surrogate's #1 pick gives up vs the simulated #1), not R^2
    alone,
  - the pruned screen logs exactly how many candidates it skipped,
  - the surrogate only picks WHICH candidates get simulated — the
    final ranking still comes from real simulation + RL fine-tune.
"""
import json
import math

import numpy as np

from w101_sim import Boss, OPPOSING


# ------------------------------------------------------------- features

FEATURES = [
    "n_cards", "n_hit_cards", "n_hit_names", "n_xpip", "max_copies",
    "exp_damage", "top_hit_per_pip", "blade_pct", "trap_pct",
    "prism_gain", "n_heals", "n_shields", "overkill", "hp_k",
    # interactions a linear model can't invent: buffs MULTIPLY damage,
    # X-pip value scales with fight length, overkill saturates
    "blade_x_dmg", "trap_x_dmg", "xpip_x_hp", "log_overkill",
]


def features(deck, cards, school, boss):
    """One row of deck-vs-boss features (same order as FEATURES)."""
    mult_own = boss.incoming_mult(school)
    opp = OPPOSING.get(school)
    mult_opp = boss.incoming_mult(opp) if opp else mult_own
    n_hit = n_xpip = n_heal = n_shield = n_prism = 0
    exp_dmg = top_pp = blade = trap = 0.0
    hit_names, copies = set(), {}
    for name in deck:
        c = cards[name]
        copies[name] = copies.get(name, 0) + 1
        if c.kind in ("damage", "drain"):
            hit_names.add(name)
            if c.x_pips:
                n_xpip += 1
                per_pip = c.damage
            else:
                n_hit += 1
                per_pip = c.damage / max(c.pips, 1)
                exp_dmg += c.accuracy * c.damage * mult_own
            top_pp = max(top_pp, per_pip * mult_own)
        elif c.kind == "blade":
            blade += c.accuracy * c.percent
        elif c.kind in ("trap", "weakness"):
            trap += c.accuracy * abs(c.percent)
        elif c.kind == "prism":
            n_prism += 1
        elif c.kind == "heal":
            n_heal += 1
        elif c.kind == "shield":
            n_shield += 1
    overkill = min(exp_dmg / max(boss.hp, 1), 5.0)
    hp_k = boss.hp / 1000.0
    return [
        len(deck), n_hit, len(hit_names), n_xpip,
        max(copies.values()) if copies else 0,
        exp_dmg / 1000.0, top_pp / 100.0, blade, trap,
        n_prism * (mult_opp - mult_own), n_heal, n_shield,
        overkill, hp_k,
        blade * exp_dmg / 1000.0, trap * exp_dmg / 1000.0,
        n_xpip * hp_k, math.log1p(overkill),
    ]


# ------------------------------------------------------------- ridge fit

def fit_ridge(X, y, lam=1.0):
    """Closed-form ridge on standardized features; returns (w, mu, sd).
    w[0] is the intercept (not penalized)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    reg = lam * np.eye(Z.shape[1])
    reg[0, 0] = 0.0
    w = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
    return w, mu, sd


def spearman(a, b):
    """Rank correlation with average ranks for ties."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(list(a)), ranks(list(b))
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else 0.0


# ------------------------------------------------------------- the scorer

class DeckScorer:
    """Ridge surrogate of the simulation screen: predicts (win, ttk)."""

    def __init__(self, w_win, w_ttk, mu, sd):
        self.w_win, self.w_ttk = np.asarray(w_win), np.asarray(w_ttk)
        self.mu, self.sd = np.asarray(mu), np.asarray(sd)

    def _predict(self, w, deck, cards, school, boss):
        x = (np.asarray(features(deck, cards, school, boss)) -
             self.mu) / self.sd
        return float(w[0] + w[1:] @ x)

    def predict(self, deck, cards, school, boss):
        return (self._predict(self.w_win, deck, cards, school, boss),
                self._predict(self.w_ttk, deck, cards, school, boss))

    def rank_key(self, deck, cards, school, boss):
        """Sort key: best predicted deck first."""
        pw, pt = self.predict(deck, cards, school, boss)
        return (-pw, pt)

    def save(self, path):
        json.dump({"features": FEATURES, "w_win": self.w_win.tolist(),
                   "w_ttk": self.w_ttk.tolist(), "mu": self.mu.tolist(),
                   "sd": self.sd.tolist()},
                  open(path, "w", encoding="utf-8"), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path, encoding="utf-8"))
        return cls(d["w_win"], d["w_ttk"], d["mu"], d["sd"])


TTK_CAP = 20.0    # screen writes ttk=99 for 0-win decks; cap for the fit


def fit_scorer(rows, cards, lam=1.0):
    """Fit from logged screen rows (see boss_to_dict for the format)."""
    X = [features(r["deck"], cards, r["school"], boss_from_dict(r["boss"]))
         for r in rows]
    w_win, mu, sd = fit_ridge(X, [r["win"] for r in rows], lam)
    w_ttk, _, _ = fit_ridge(X, [min(r["ttk"], TTK_CAP) for r in rows],
                            lam)   # same X -> same standardization basis
    return DeckScorer(w_win, w_ttk, mu, sd)


# ------------------------------------------------------- (de)serialization

def boss_to_dict(b):
    return {"name": b.name, "hp": b.hp, "school": b.school,
            "resist_map": b.resist_map, "boost_map": b.boost_map}


def boss_from_dict(d):
    b = Boss(d["name"], d["hp"], d["school"], 0)
    b.resist_map = d.get("resist_map") or {}
    b.boost_map = d.get("boost_map") or {}
    return b


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------- experiment

if __name__ == "__main__":
    import random
    import time

    from data_full import load_spells_full, LIVE_RULES
    from deck_builder import (legal_pool, sample_deck, screen, build_deck,
                              random_boss)

    cards = load_spells_full()
    rng = random.Random(0)
    SCHOOLS = ("fire", "ice", "storm", "death", "life", "myth", "balance")

    LOG = "deck_screen_log.jsonl"
    N_BOSS, N_TEST, N_DECKS, N_SIM = 40, 8, 40, 150
    print(f"[1/3] logging screen data: {N_BOSS} random bosses x "
          f"{N_DECKS} decks x {N_SIM} seeded fights -> {LOG}", flush=True)
    rows, boss_ids = [], []
    with open(LOG, "w", encoding="utf-8") as f:
        for i in range(N_BOSS):
            school = SCHOOLS[i % len(SCHOOLS)]
            boss = random_boss(rng, f"scorer{i}")
            pool = legal_pool(cards, school)
            seen, decks = set(), []
            tries = 0
            while len(decks) < N_DECKS and tries < N_DECKS * 40:
                tries += 1
                dl = sample_deck(pool, school, boss, rng)
                if dl is None:
                    break
                key = tuple(sorted(dl))
                if key not in seen:
                    seen.add(key)
                    decks.append(dl)
            t0 = time.time()
            table = screen(cards, decks, school, boss, LIVE_RULES,
                           n=N_SIM)
            for w, m, dl in table:
                row = dict(school=school, boss=boss_to_dict(boss),
                           deck=dl, win=w, ttk=m)
                rows.append(row)
                boss_ids.append(i)
                f.write(json.dumps(row) + "\n")
            print(f"   boss {i:>2} {boss.name:<20} {school:<8} "
                  f"{len(decks)} decks  best {table[0][0]*100:3.0f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    train = [r for r, b in zip(rows, boss_ids) if b < N_BOSS - N_TEST]
    print(f"\n[2/3] fit on {len(train)} rows "
          f"({N_BOSS - N_TEST} bosses), test on {N_TEST} held-out bosses:",
          flush=True)
    scorer = fit_scorer(train, cards)
    scorer.save("deck_scorer.json")
    test_report = []
    for i in range(N_BOSS - N_TEST, N_BOSS):
        rs = [r for r, b in zip(rows, boss_ids) if b == i]
        sim = [r["win"] for r in rs]
        pred = [scorer.predict(r["deck"], cards, r["school"],
                               boss_from_dict(r["boss"]))[0] for r in rs]
        # a boss every deck beats (or none does) carries no ranking
        # signal — report it as degenerate instead of rho=0
        rho = spearman(pred, sim) if max(sim) > min(sim) else None
        top1 = sim[max(range(len(rs)), key=lambda j: pred[j])]
        regret = max(sim) - top1
        test_report.append(dict(
            boss=rs[0]["boss"]["name"], school=rs[0]["school"],
            spearman=rho, top1_regret=regret, sim=sim, pred=pred))
        rho_s = f"{rho:5.2f}" if rho is not None else "  n/a"
        print(f"   {rs[0]['boss']['name']:<20} {rs[0]['school']:<8} "
              f"spearman {rho_s}  top-1 regret "
              f"{regret*100:4.1f} pts", flush=True)
    rhos = [t["spearman"] for t in test_report
            if t["spearman"] is not None]
    regs = [t["top1_regret"] for t in test_report]
    print(f"   mean spearman {sum(rhos)/len(rhos):.2f} "
          f"({len(rhos)}/{len(test_report)} bosses with signal), "
          f"mean top-1 regret {sum(regs)/len(regs)*100:.1f} pts",
          flush=True)
    json.dump({"features": FEATURES, "test": test_report},
              open("scorer_results.json", "w", encoding="utf-8"),
              indent=1)

    print("\n[3/3] end-to-end: full screen vs surrogate-pruned screen "
          "(fresh boss, same seeds):", flush=True)
    fresh = random_boss(random.Random(99), "fresh")
    for name, sc in (("full", None), ("pruned", scorer)):
        t0 = time.time()
        dl, w, m, _ = build_deck(cards, "death", fresh, LIVE_RULES,
                                 n_candidates=60, top_k=3, seed=7,
                                 scorer=sc, log=None)
        print(f"   {name:<7} win {w*100:5.1f}%  ttk {m:5.2f}  "
              f"size {len(dl):>2}  ({time.time()-t0:.0f}s)", flush=True)
    print("\nwrote deck_screen_log.jsonl, deck_scorer.json, "
          "scorer_results.json")
