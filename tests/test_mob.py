"""
Target switching: focus selection, per-target casting, and the
kill-the-healer-first mechanic end to end.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import load_spells_full, LIVE_RULES
from w101_sim import (Sim, Boss, evaluate_paired, make_blade_stack,
                      pick_focus, with_focus)

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))


def _boss(name="boss", hp=2000, pool=None, archetype="hitter"):
    b = Boss(name, hp, "storm", 0, pool=pool, archetype=archetype)
    b.resist_map = {}
    b.boost_map = {}
    return b


def _sim(extra, seed=0):
    return Sim(dict(CARDS), ["Wraith", "Wraith", "Deathblade"], "death",
               _boss(pool=["Thunder Snake"]), player_hp=10**6,
               rules=LIVE_RULES, rng=random.Random(seed), enemies=extra,
               log_events=True)


def test_pick_focus_prefers_support_then_lowest_hp():
    healer = _boss("healer", 300, pool=["Sprite"], archetype="healer")
    sim = _sim([healer])
    s = sim.new_state()
    assert pick_focus(s) == 1                   # support first
    s.enemies[1].hp = 0                         # alive is derived
    assert pick_focus(s) == 0                   # then the boss
    sim2 = _sim([_boss("add", 150, pool=["Thunder Snake"])])
    s2 = sim2.new_state()
    assert pick_focus(s2) == 1                  # lowest-HP attacker


def test_focus_wrapper_hits_the_chosen_target():
    healer = _boss("healer", 300, pool=["Sprite"], archetype="healer")
    sim = _sim([healer], seed=3)
    pol = with_focus(make_blade_stack(0))
    t, won, hp = sim.run(pol, max_turns=20)
    ev = sim.last_state.events
    # the healer died before the boss did (or the fight was won)
    healer_deaths = [e for e in ev if e.get("type") == "death"
                     and e.get("actor") == "healer"]
    assert won or not sim.last_state.enemies[1].alive or healer_deaths


def test_generalist_actions_expand_targets_only_in_mobs():
    import numpy as np
    from generalist import legal_actions, phi
    healer = _boss("healer", 300, pool=["Sprite"], archetype="healer")
    sim = _sim([healer])
    s = sim.new_state()
    s.norm_pips, s.pow_pips = 6, 3
    acts = legal_actions(sim, s)
    tuples = [a for a in acts if isinstance(a, tuple)]
    assert tuples and {t for _, t in tuples} == {0, 1}
    # primary-target tuple vector == bare-card vector (compat)
    card = tuples[0][0]
    assert np.allclose(phi(sim, s, card), phi(sim, s, (card, 0)))
    # 1v1: no tuples at all
    solo = Sim(dict(CARDS), ["Wraith", "Deathblade"], "death",
               _boss(pool=["Thunder Snake"]), player_hp=10**6,
               rules=LIVE_RULES, rng=random.Random(1))
    s1 = solo.new_state()
    s1.norm_pips = 8
    assert not any(isinstance(a, tuple) for a in legal_actions(solo, s1))


def test_generalist_old_policies_load_padded():
    import json as _json
    import tempfile, os
    from generalist import FEATS, GeneralistQ
    old = {"features": FEATS[:-2], "w": [0.1] * (len(FEATS) - 2)}
    fd, p = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        _json.dump(old, f)
    gen = GeneralistQ.load(p)
    os.unlink(p)
    assert len(gen.w) == len(FEATS)
    assert gen.w[-1] == 0.0 and gen.w[-2] == 0.0


def test_handcrafted_generalist_kills_the_healer_first():
    """Value overkill + support-flag: the greedy pick must aim the
    first hit at the 300-HP healer, not the 2000-HP boss."""
    from generalist import FEATS, GeneralistQ
    gen = GeneralistQ()
    gen.w[FEATS.index("overkill")] = 1.0
    gen.w[FEATS.index("kill_now")] = 1.0
    gen.w[FEATS.index("tgt_is_support")] = 1.0
    gen.w[FEATS.index("k_hit")] = 0.5
    healer = _boss("healer", 300, pool=["Sprite"], archetype="healer")
    sim = _sim([healer], seed=4)
    s = sim.new_state()
    s.norm_pips, s.pow_pips = 6, 3
    a = gen.policy()(sim, s)
    assert isinstance(a, tuple) and a[1] == 1
    assert a[0].kind in ("damage", "drain")


def test_focus_beats_blind_against_a_healer():
    """The report's team-fight rule, measured: focus-fire on the
    healer must strictly beat target-blind play."""
    healer = _boss("healer", 400, pool=["Sprite", "Pixie"],
                   archetype="healer")
    boss = _boss(hp=1500, pool=["Thunder Snake", "Stormblade"])
    deck = ["Wraith"] * 3 + ["Banshee"] * 3 + \
           ["Deathblade", "Death Trap", "Curse"]
    sim = Sim(dict(CARDS), deck, "death", boss, player_hp=10**6,
              rules=LIVE_RULES, enemies=[healer])
    st = evaluate_paired(sim, {
        "blind": make_blade_stack(2),
        "focus": with_focus(make_blade_stack(2))}, n=300)
    assert st["focus"]["win_rate"] > st["blind"]["win_rate"]
