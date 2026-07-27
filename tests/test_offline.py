"""
Offline RL ladder: cloning recovers experts, filtering beats mixed
demonstrations, conservatism stays sane, and the pipeline runs
end-to-end on real sims.
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import load_spells_full, LIVE_RULES
from deck_builder import legal_pool, sample_deck, random_boss
from generalist import FEATS, GeneralistQ
from offline_rl import MAX_A, gen_dataset, train_bc, train_cql
from w101_sim import Sim, evaluate

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))
F = len(FEATS)


def _synth(n=3000, n_legal=6, seed=0, sabotage_losers=True):
    """Synthetic decisions: winners follow argmax w*, losers argmin."""
    rng = np.random.default_rng(seed)
    w_star = rng.normal(size=F)
    P = np.zeros((n, MAX_A, F), np.float32)
    M = np.zeros((n, MAX_A), bool)
    C = np.zeros(n, np.int16)
    W = np.zeros(n, bool)
    for i in range(n):
        P[i, :n_legal] = rng.normal(size=(n_legal, F))
        M[i, :n_legal] = True
        q = P[i, :n_legal] @ w_star
        win = i % 2 == 0
        W[i] = win
        C[i] = int(np.argmax(q)) if (win or not sabotage_losers) \
            else int(np.argmin(q))
    G = (P[np.arange(n), C] @ w_star).astype(np.float32)
    return dict(phis=P, mask=M, chosen=C, G=G, won=W), w_star


def _agreement(w, data):
    P, M, C = data["phis"], data["mask"], data["chosen"]
    q = P @ w
    q[~M] = -1e9
    return float((q.argmax(axis=1) == C).mean())


def test_filtered_bc_beats_mixed_demos():
    data, w_star = _synth()
    w_all = train_bc(data)
    w_filt = train_bc(data, winners_only=True)
    winners = {k: v[data["won"]] for k, v in data.items()}
    agree_all = _agreement(w_all, winners)
    agree_filt = _agreement(w_filt, winners)
    assert agree_filt > 0.9, agree_filt
    assert agree_filt > agree_all + 0.1, (agree_filt, agree_all)


def test_cql_fits_returns_and_stays_finite():
    data, w_star = _synth(sabotage_losers=False)
    w = train_cql(data, alpha=0.5)
    assert np.all(np.isfinite(w))
    rows = np.arange(len(data["G"]))
    pred = (data["phis"][rows, data["chosen"]] @ w)
    r = np.corrcoef(pred, data["G"])[0, 1]
    assert r > 0.5, r


def test_pipeline_end_to_end_on_real_sims():
    data, pairs = gen_dataset(CARDS, LIVE_RULES, n_pairs=2,
                              eps_per_pair=15, seed=3)
    assert data["phis"].shape[1:] == (MAX_A, F)
    assert data["mask"][np.arange(len(data["chosen"])),
                        data["chosen"]].all()
    pol = GeneralistQ(w=train_bc(data, winners_only=True))
    rng = random.Random(5)
    boss = random_boss(rng, "offl")
    dl = sample_deck(legal_pool(CARDS, "death"), "death", boss, rng)
    sim = Sim(dict(CARDS), dl, "death", boss, player_hp=10**9,
              rules=LIVE_RULES)
    evaluate(sim, pol.policy(), n=50)       # plays without error
