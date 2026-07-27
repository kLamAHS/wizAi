"""
Deck-conditioned generalist: features, legality, learning, and the
zero-shot path through build_deck.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from data_full import load_spells_full, LIVE_RULES
from deck_builder import build_deck, random_boss
from generalist import (FEATS, phi, legal_actions, GeneralistQ,
                        train_generalist)
from rl_agent import PASS
from w101_sim import Sim, Boss, evaluate

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))


def _dummy(hp=100):
    b = Boss("dummy", hp, "death", 0)
    b.resist_map = {"death": 0.5}
    b.boost_map = {}
    return b


def test_phi_shape_and_flags():
    sim = Sim(dict(CARDS), ["Wraith", "Deathblade"], "death", _dummy(200),
              player_hp=10**9, rules=LIVE_RULES, rng=random.Random(0))
    s = sim.new_state()
    s.norm_pips, s.pow_pips = 6, 4          # afford anything
    wraith = next(c for c in s.hand if c.name == "Wraith")
    blade = next(c for c in s.hand if c.name == "Deathblade")
    f = dict(zip(FEATS, phi(sim, s, wraith)))
    assert f["k_hit"] == 1.0 and f["dmg_est"] > 0
    assert f["kill_now"] == 1.0             # 500 avg x 0.5 resist vs 200
    assert f["dup"] == 0.0
    fp = dict(zip(FEATS, phi(sim, s, PASS)))
    assert fp["k_pass"] == 1.0 and fp["k_hit"] == 0.0
    sim.cast(s, blade)                      # now the blade would be a dup
    b2 = next((c for c in s.hand if c.name == "Deathblade"), None)
    if b2 is not None:
        f2 = dict(zip(FEATS, phi(sim, s, b2)))
        assert f2["dup"] == 1.0


def test_legal_actions_unique_and_castable():
    sim = Sim(dict(CARDS), ["Dark Sprite"] * 3 + ["Wraith"], "death",
              _dummy(1000), player_hp=10**9, rules=LIVE_RULES,
              rng=random.Random(1))
    s = sim.new_state()
    s.norm_pips, s.pow_pips = 0, 0          # can afford nothing
    acts = legal_actions(sim, s)
    assert acts == [PASS]
    s.norm_pips = 1                          # 1 pip: Dark Sprite only
    names = [a if a == PASS else a.name for a in legal_actions(sim, s)]
    assert names.count("Dark Sprite") <= 1
    assert "Wraith" not in names


def test_learns_to_hit_on_trivial_matchup():
    """One fixed easy sim: the linear Q must learn hit > pass fast."""
    sim = Sim(dict(CARDS), ["Fire Cat"] * 3, "fire", _dummy(100),
              player_hp=10**9, rules=LIVE_RULES, rng=random.Random(2))
    agent = GeneralistQ(rng=random.Random(3))
    for ep in range(800):
        frac = ep / 800
        agent.alpha = 0.03 * (1 - 0.9 * frac)
        agent.train_episode(sim, eps=max(0.05, 0.3 * (1 - frac)))
    w, m = evaluate(sim, agent.policy(), n=400)
    assert w > 0.8, (w, m)


def test_train_generalist_roundtrip(tmp_path):
    gen = train_generalist(CARDS, episodes=600, seed=0, rules=LIVE_RULES,
                           snap_every=300)
    assert np.all(np.isfinite(gen.w)) and np.any(gen.w != 0)
    p = tmp_path / "gen.json"
    gen.save(str(p))
    gen2 = GeneralistQ.load(str(p))
    assert np.allclose(gen.w, gen2.w)
    # the loaded policy must play a full game without error
    boss = random_boss(random.Random(9), "rt")
    sim = Sim(dict(CARDS), ["Wraith", "Wraith", "Deathblade", "Feint"],
              "death", boss, player_hp=10**9, rules=LIVE_RULES,
              rng=random.Random(4))
    evaluate(sim, gen2.policy(), n=50)


def test_build_deck_generalist_stage2():
    """generalist= replaces the per-deck fine-tune in stage 2."""
    gen = GeneralistQ()
    gen.w[FEATS.index("overkill")] = 1.0    # 'cast the biggest hit'
    boss = _dummy(600)
    msgs = []
    dl, w, m, table = build_deck(
        CARDS, "death", boss, LIVE_RULES, n_candidates=10, top_k=2,
        seed=11, generalist=gen, log=msgs.append)
    assert any("generalist" in s for s in msgs)
    assert 0.0 <= w <= 1.0 and len(dl) > 0
