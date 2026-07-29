"""
Learned deck scorer: features, ridge fit, ranking, and the pruned
screen path through build_deck.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import load_spells_full, LIVE_RULES
from deck_builder import build_deck, legal_pool, sample_deck, random_boss
from deck_scorer import (FEATURES, features, fit_ridge, fit_scorer,
                         spearman, DeckScorer, boss_to_dict,
                         boss_from_dict)
from w101_sim import Boss

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))


def _boss(hp=2000, school="life", resist=0.8):
    b = Boss("t", hp, school, 0)
    b.resist_map = {school: resist}
    b.boost_map = {}
    return b


def test_features_shape_and_signal():
    deck = ["Wraith", "Wraith", "Deathblade", "Feint", "Dark Sprite"]
    x = features(deck, CARDS, "death", _boss())
    assert len(x) == len(FEATURES)
    row = dict(zip(FEATURES, x))
    assert row["n_cards"] == 5
    assert row["n_hit_cards"] == 3 and row["n_hit_names"] == 2
    assert row["max_copies"] == 2
    assert row["exp_damage"] > 0 and row["blade_pct"] > 0
    assert row["trap_pct"] > 0
    # prism gain vs a same-school wall is positive (death -> life boss
    # resists nothing extra; use a death boss for the real check)
    db = Boss("d", 2000, "death", 0)
    db.resist_map = {"death": 0.8}
    db.boost_map = {}
    with_prism = dict(zip(FEATURES, features(
        ["Death Prism", "Wraith"], CARDS, "death", db)))
    assert with_prism["prism_gain"] > 0


def test_ridge_recovers_linear_signal():
    rng = random.Random(0)
    w_true = [2.0, -1.0, 0.5]
    X = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(400)]
    y = [sum(wi * xi for wi, xi in zip(w_true, x)) + 3.0 for x in X]
    w, mu, sd = fit_ridge(X, y, lam=1e-6)
    sc = DeckScorer(w, w, mu, sd)
    for x, yi in zip(X[:20], y[:20]):
        import numpy as np
        pred = float(w[0] + w[1:] @ ((np.asarray(x) - mu) / sd))
        assert abs(pred - yi) < 1e-3


def test_spearman_endpoints():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert abs(spearman([1, 1, 2, 2], [1, 2, 1, 2])) < 1e-9


def test_scorer_roundtrip(tmp_path):
    rng = random.Random(1)
    boss = random_boss(rng)
    pool = legal_pool(CARDS, "death")
    rows = []
    for i in range(12):
        dl = sample_deck(pool, "death", boss, rng)
        rows.append(dict(school="death", boss=boss_to_dict(boss),
                         deck=dl, win=rng.random(),
                         ttk=rng.uniform(3, 15)))
    sc = fit_scorer(rows, CARDS)
    p = tmp_path / "scorer.json"
    sc.save(str(p))
    sc2 = DeckScorer.load(str(p))
    for r in rows[:3]:
        a = sc.predict(r["deck"], CARDS, "death",
                       boss_from_dict(r["boss"]))
        b = sc2.predict(r["deck"], CARDS, "death",
                        boss_from_dict(r["boss"]))
        assert all(abs(x - y) < 1e-12 for x, y in zip(a, b))


def test_scorer_learns_overkill_matters():
    """Synthetic labels that reward expected damage: the fitted scorer
    must rank a redundant hit deck above a hitless-ish one."""
    rng = random.Random(2)
    boss = _boss(hp=3000, school="death", resist=0.5)
    pool = legal_pool(CARDS, "death")
    rows = []
    for _ in range(60):
        dl = sample_deck(pool, "death", boss, rng)
        row = dict(zip(FEATURES, features(dl, CARDS, "death", boss)))
        rows.append(dict(school="death", boss=boss_to_dict(boss),
                         deck=dl, win=min(1.0, row["overkill"] / 2),
                         ttk=8.0))
    sc = fit_scorer(rows, CARDS)
    big = ["Wraith", "Wraith", "Wraith", "Scarecrow", "Deathblade"]
    small = ["Dark Sprite", "Deathblade"]
    pw_big = sc.predict(big, CARDS, "death", boss)[0]
    pw_small = sc.predict(small, CARDS, "death", boss)[0]
    assert pw_big > pw_small


def test_build_deck_pruned_screen_and_log(tmp_path):
    """scorer= prunes the screen; screen_log= writes JSONL rows."""
    rng = random.Random(3)
    boss = random_boss(rng)
    rows = []
    pool = legal_pool(CARDS, "death")
    for _ in range(30):
        dl = sample_deck(pool, "death", boss, rng)
        rows.append(dict(school="death", boss=boss_to_dict(boss),
                         deck=dl, win=rng.random(), ttk=rng.uniform(3, 15)))
    sc = fit_scorer(rows, CARDS)
    logp = tmp_path / "screen.jsonl"
    msgs = []
    dl, w, m, table = build_deck(
        CARDS, "death", boss, LIVE_RULES, n_candidates=18, top_k=2,
        seed=5, scorer=sc, screen_log=str(logp), log=msgs.append)
    assert any("surrogate pruned" in s for s in msgs)
    assert len(table) <= max(2 * 2, math.ceil(18 / 3))
    lines = logp.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(table)
    import json
    row = json.loads(lines[0])
    assert row["school"] == "death" and "win" in row and "deck" in row
