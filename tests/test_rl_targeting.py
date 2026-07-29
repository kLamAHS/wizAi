"""
Target selection: the agent picks (card, target), not just card.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_full import LIVE_RULES, load_spells_full
from rl_agent import Featurizer, PASS, apply_action, split_target
from w101_sim import Boss, Sim

ROOT = Path(__file__).resolve().parent.parent
CARDS = load_spells_full(str(ROOT / "spells_full.json"),
                         str(ROOT / "cards_clean.json"))
DECK = ["Kraken", "Kraken", "Stormblade", "Storm Trap"]


def _sim(n_enemies):
    def mk(i):
        b = Boss(f"e{i}", 900, "death", 40)
        b.resist_map = {}
        b.boost_map = {}
        return b
    return Sim(dict(CARDS), DECK, "storm", mk(0), player_hp=10**6,
               rules=LIVE_RULES, rng=random.Random(0),
               enemies=[mk(i) for i in range(1, n_enemies)])


def test_split_target_ignores_treasure_card_suffixes():
    assert split_target("Kraken@2") == ("Kraken", 2)
    assert split_target("Kraken") == ("Kraken", 0)
    assert split_target("Catalan TC@tc") == ("Catalan TC@tc", 0)


def test_single_enemy_actions_are_bare_names():
    """Every 1v1 result predating targeting must be bit-identical."""
    sim = _sim(1)
    s = sim.new_state()
    acts = Featurizer(dict(CARDS), DECK).legal(sim, s)
    assert all("@" not in a or not a.rsplit("@", 1)[-1].isdigit()
               for a in acts)


def test_multi_enemy_expands_single_target_cards_only():
    sim = _sim(3)
    s = sim.new_state()
    s.player.norm_pips = 10
    acts = Featurizer(dict(CARDS), DECK).legal(sim, s)
    aimed = [a for a in acts if a.startswith("Kraken@")]
    assert sorted(aimed) == ["Kraken@0", "Kraken@1", "Kraken@2"]
    # a self-targeted blade is not aimed at anything
    assert "Stormblade" in acts
    assert not any(a.startswith("Stormblade@") for a in acts)


def test_the_target_index_actually_lands():
    sim = _sim(3)
    s = sim.new_state()
    s.player.norm_pips = 10
    before = [e.hp for e in s.enemies]
    apply_action(sim, s, "Kraken@2")
    after = [e.hp for e in s.enemies]
    assert after[2] < before[2], (before, after)
    assert after[0] == before[0] and after[1] == before[1]


def test_state_key_grows_only_on_a_multi_enemy_board():
    f = Featurizer(dict(CARDS), DECK)
    solo, mob = _sim(1), _sim(3)
    k1 = f.key(solo, solo.new_state())
    k3 = f.key(mob, mob.new_state())
    assert len(k3) == len(k1) + 1
    assert isinstance(k3[-1], tuple) and len(k3[-1]) == 3


def test_targeting_state_stays_coarse():
    """A health bucket per enemy made every state unique, Q stayed
    all-zero, greedy fell through to PASS and the agent scored 0.0%.
    The signal is kept to (living count, weakest index, its bucket)."""
    f = Featurizer(dict(CARDS), DECK)
    sim = _sim(4)
    s = sim.new_state()
    seen = set()
    for hp in range(200, 900, 50):
        for e in s.enemies:
            e.hp = hp
        s.enemies[1].hp = hp - 100          # a clear weakest
        seen.add(f.key(sim, s)[-1])
    assert all(k[1] == 1 for k in seen)     # weakest is always found
    assert len(seen) <= 8                   # and the space stays small
