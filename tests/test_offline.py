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
from offline_rl import (MAX_A, gen_dataset, train_bc, train_cql,
                        train_iql, train_expectile_v, state_phi,
                        advantages)
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


# ---- IQL-flavored expectile rung ------------------------------------

def _adv_synth(n=4000, n_legal=6, seed=1):
    """Decisions whose quality varies WITHIN an episode outcome: half
    the winning episodes contain a deliberately bad move. That is the
    regime per-decision credit is supposed to handle and per-episode
    filtering cannot."""
    rng = np.random.default_rng(seed)
    w_star = rng.normal(size=F)
    P = np.zeros((n, MAX_A, F), np.float32)
    M = np.zeros((n, MAX_A), bool)
    C = np.zeros(n, np.int16)
    W = np.ones(n, bool)                    # every episode "won"
    for i in range(n):
        P[i, :n_legal] = rng.normal(size=(n_legal, F))
        M[i, :n_legal] = True
        q = P[i, :n_legal] @ w_star
        C[i] = int(np.argmax(q)) if i % 2 == 0 else int(np.argmin(q))
    G = (P[np.arange(n), C] @ w_star).astype(np.float32)
    return dict(phis=P, mask=M, chosen=C, G=G, won=W), w_star


def test_state_phi_is_the_masked_mean():
    data, _ = _adv_synth(n=50)
    S = state_phi(data)
    assert S.shape == (50, F)
    row = data["phis"][0][data["mask"][0]]
    assert np.allclose(S[0], row.mean(axis=0), atol=1e-5)


def test_expectile_tau_orders_the_baseline():
    """tau=0.5 is least squares; raising tau must not lower the fitted
    value — that monotonicity IS the expectile."""
    data, _ = _adv_synth()
    S = state_phi(data)
    vals = [float((S @ train_expectile_v(data, tau=t, S=S)).mean())
            for t in (0.3, 0.5, 0.7, 0.9)]
    assert vals == sorted(vals), vals


def test_expectile_half_is_least_squares():
    data, _ = _adv_synth(n=1500)
    S = state_phi(data)
    w = train_expectile_v(data, tau=0.5, S=S, epochs=40, lr0=0.2)
    ols = np.linalg.lstsq(S, data["G"], rcond=None)[0]
    assert np.corrcoef(S @ w, S @ ols)[0, 1] > 0.95


def test_advantage_weighting_beats_episode_filtering():
    """Every episode here is a 'win', so BC-filtered has nothing to
    filter and clones the bad moves too; AWR can still tell them
    apart because the advantage is per decision."""
    data, _ = _adv_synth()
    good = {k: v[::2] for k, v in data.items()}     # the argmax half
    a_bc = _agreement(train_bc(data, winners_only=True), good)
    a_iql = _agreement(train_iql(data, tau=0.7, beta=2.0), good)
    assert a_iql > a_bc + 0.1, (a_iql, a_bc)


def test_iql_weights_are_clipped_and_finite():
    """Unclipped exp(beta*A) lets one lucky decision own the gradient."""
    data, _ = _adv_synth(n=800)
    w = train_iql(data, tau=0.9, beta=50.0, clip=20.0)
    assert np.all(np.isfinite(w))


def test_iql_never_queries_unplayed_actions():
    """IQL's whole point vs CQL: the value target uses only logged
    (state, action) pairs. Corrupting the UNCHOSEN feature rows must
    leave the fitted baseline bit-identical."""
    data, _ = _adv_synth(n=600)
    S = state_phi(data)
    v0 = train_expectile_v(data, tau=0.7, S=S)
    bad = {k: (v.copy() if hasattr(v, "copy") else v)
           for k, v in data.items()}
    rows = np.arange(len(bad["chosen"]))
    keep = bad["phis"][rows, bad["chosen"]].copy()
    bad["phis"][:] = 999.0
    bad["phis"][rows, bad["chosen"]] = keep
    v1 = train_expectile_v(bad, tau=0.7, S=S)
    assert np.array_equal(v0, v1)


def test_winners_only_composes_the_two_credit_schemes():
    """The episode filter picks WHICH decisions to clone; the
    advantage weights say how much each counts. The baseline stays fit
    on all episodes — an advantage measured only against winners is
    not an advantage."""
    data, _ = _adv_synth(n=1200)
    data["won"][::2] = False                 # only the bad half "wins"
    w_all = train_iql(data, tau=0.7, beta=2.0)
    w_filt = train_iql(data, tau=0.7, beta=2.0, winners_only=True)
    assert not np.allclose(w_all, w_filt)
    losers = {k: v[1::2] for k, v in data.items()}   # the argmin half
    assert _agreement(w_filt, losers) > _agreement(w_all, losers)
