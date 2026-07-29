"""Tests for the live-run telemetry and the GUI over it.

The telemetry tests are the substantive ones -- every judgement the GUI
makes about a run lives in `telemetry.py`, which is Qt-free precisely so
those judgements can be checked without a display.

The GUI tests are shallower by design: they build the real window under
Qt's offscreen platform and drive every panel, which catches the failures
that actually happen (a panel that raises on empty data, a signal wired
to a missing slot) without asserting on pixels.
"""
import os

import pytest

from deimos_bridge.telemetry import (Telemetry, describe_hanging,
                                     predict_damage)
from w101_sim import Hanging

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --------------------------------------------------------------- prediction
def _sim_and_state(hand=("Sunbird",), blades=(), enemy_hp=2000,
                   enemy_resist=None):
    from data_full import load_spells_full
    from w101_sim import Actor, Boss, Rules, Sim, State

    cards = load_spells_full()
    player = Actor(name="Wizard", school="fire", hp=3000, max_hp=3000, team=0,
                   norm_pips=7)
    player.hand = [cards[n] for n in hand]
    for i, pct in enumerate(blades):
        player.charms.append(Hanging(name=f"b{i}", slot="charm", kind="damage",
                                     percent=pct, schools={"fire"},
                                     source="live", sub=i))
    enemy = Actor(name="Mob", school="ice", hp=enemy_hp, max_hp=enemy_hp,
                  team=1, resist={"*": enemy_resist} if enemy_resist else {})
    sim = Sim(cards=cards, decklist=list(hand), school="fire",
              boss=Boss(name="Mob", hp=enemy_hp, school="ice", dmg=0),
              rules=Rules())
    return sim, State(player, [enemy]), cards


def test_predict_damage_returns_a_number_for_a_nuke():
    sim, state, cards = _sim_and_state()
    got = predict_damage(sim, state, cards["Sunbird"], 0)
    assert got is not None and got > 0


def test_predict_damage_does_not_mutate_the_caller_state():
    """It runs a real cast; if that leaked, the live board would drift out
    from under the policy every round."""
    sim, state, cards = _sim_and_state()
    hp_before = state.enemies[0].hp
    hand_before = [c.name for c in state.hand]
    pips_before = state.norm_pips
    predict_damage(sim, state, cards["Sunbird"], 0)
    assert state.enemies[0].hp == hp_before
    assert [c.name for c in state.hand] == hand_before
    assert state.norm_pips == pips_before


def test_prediction_scales_with_blades_read_off_the_live_board():
    bare, _, cards = _sim_and_state()
    sim0, st0, _ = _sim_and_state()
    sim1, st1, _ = _sim_and_state(blades=(0.35,))
    a = predict_damage(sim0, st0, cards["Sunbird"], 0)
    b = predict_damage(sim1, st1, cards["Sunbird"], 0)
    assert b == pytest.approx(a * 1.35, rel=1e-6)


def test_prediction_accounts_for_enemy_resist():
    sim0, st0, cards = _sim_and_state()
    sim1, st1, _ = _sim_and_state(enemy_resist=0.5)
    assert predict_damage(sim1, st1, cards["Sunbird"], 0) == \
        pytest.approx(predict_damage(sim0, st0, cards["Sunbird"], 0) * 0.5)


def test_prediction_is_none_for_a_card_not_in_hand():
    sim, state, cards = _sim_and_state()
    assert predict_damage(sim, state, cards["Efreet"], 0) is None


def test_prediction_never_fizzles():
    """Two calls must agree. Sim.cast rolls accuracy, so a live prediction
    that sometimes returned 0 would poison the residual statistics with
    the engine's RNG rather than its arithmetic."""
    sim, state, cards = _sim_and_state()
    vals = {predict_damage(sim, state, cards["Sunbird"], 0) for _ in range(12)}
    assert len(vals) == 1 and vals.pop() > 0


# -------------------------------------------------------------- hanging text
def test_describe_hanging_reads_as_arithmetic():
    blade = Hanging(name="live:1", slot="charm", kind="damage", percent=0.35,
                    schools={"fire"})
    shield = Hanging(name="live:2", slot="ward", kind="damage", percent=-0.5)
    prism = Hanging(name="live:3", slot="ward", kind="prism", convert_to="ice")
    absorb = Hanging(name="live:4", slot="ward", kind="absorb", amount=400)
    assert describe_hanging(blade) == "+35% fire"
    assert describe_hanging(shield) == "-50% all"
    assert describe_hanging(prism) == "prism -> ice"
    assert describe_hanging(absorb) == "absorb 400"


# ---------------------------------------------------------------- telemetry
class _Decision:
    def __init__(self, card_name=None, target_index=None, passing=False,
                 reason=""):
        self.card_name = card_name
        self.target_index = target_index
        self.passing = passing
        self.reason = reason


class _Resolver:
    def __init__(self, misses=None):
        self.misses = dict(misses or {})


class _Read:
    def __init__(self, state, round_number=1, hand=(), misses=None):
        self.state = state
        self.round_number = round_number
        self.hand_cards = {n: [] for n in hand}
        self.resolver = _Resolver(misses)
        self.enemy_members = []


def _read(enemy_hp, round_number, hand=("Sunbird",), misses=None):
    from w101_sim import Actor, State
    player = Actor(name="Wizard", school="fire", hp=3000, max_hp=3000, team=0)
    enemy = Actor(name="Mob", school="ice", hp=enemy_hp, max_hp=2000, team=1)
    return _Read(State(player, [enemy]), round_number, hand, misses)


def test_actual_damage_is_settled_from_the_next_round():
    tel = Telemetry()
    tel.start_fight()
    r1 = tel.observe(_Decision("Sunbird"), _read(2000, 1))
    r1.predicted_damage = 500.0
    tel.observe(_Decision("Sunbird"), _read(1550, 2))
    assert r1.actual_damage == pytest.approx(450.0)
    assert r1.error == pytest.approx(-50.0)
    assert r1.clean


def test_blade_rounds_are_not_damage_observations():
    """A blade predicts 0 and delivers 0. Counting it as a perfect
    prediction would make a buff-heavy deck look more accurate purely for
    casting fewer nukes."""
    tel = Telemetry()
    tel.start_fight()
    blade = tel.observe(_Decision("Fireblade"), _read(2000, 1))
    blade.predicted_damage = 0.0
    nuke = tel.observe(_Decision("Sunbird"), _read(2000, 2))
    nuke.predicted_damage = 500.0
    tel.observe(_Decision("Sunbird"), _read(1600, 3))

    assert blade.actual_damage == 0.0
    obs = tel.damage_observations()
    assert [o.round for o in obs] == [2]
    assert tel.error_stats()["n"] == 1
    assert tel.error_stats()["mean_abs_error"] == pytest.approx(100.0)


def test_a_dead_target_is_marked_unclean():
    """The hit landed for at least the target's remaining HP, which is a
    floor and not a measurement."""
    from w101_sim import Actor, State
    tel = Telemetry()
    tel.start_fight()
    r1 = tel.observe(_Decision("Sunbird"), _read(300, 1))
    r1.predicted_damage = 900.0

    player = Actor(name="Wizard", school="fire", hp=3000, max_hp=3000, team=0)
    empty = _Read(State(player, [Actor(name="Other", school="ice", hp=500,
                                       max_hp=500, team=1)]), 2)
    tel.observe(_Decision(passing=True, reason="won"), empty)
    assert r1.actual_damage == 300.0
    assert not r1.clean
    assert "lower bound" in r1.confounds[0]


def test_collateral_damage_is_marked_unclean():
    """A second mob losing HP means an AoE, a DoT or a minion also fired,
    so the target's delta is not attributable to this cast alone."""
    from w101_sim import Actor, State

    def board(a, b, rnd):
        player = Actor(name="Wizard", school="fire", hp=3000, max_hp=3000,
                       team=0)
        return _Read(State(player, [
            Actor(name="Mob", school="ice", hp=a, max_hp=2000, team=1),
            Actor(name="Mob2", school="ice", hp=b, max_hp=2000, team=1)]), rnd)

    tel = Telemetry()
    tel.start_fight()
    r1 = tel.observe(_Decision("Sunbird", 0), board(2000, 2000, 1))
    r1.predicted_damage = 400.0
    tel.observe(_Decision("Sunbird", 0), board(1600, 1750, 2))
    assert not r1.clean
    assert "Mob2" in r1.confounds[0]
    assert tel.error_stats()["n"] == 0                # excluded by default
    assert tel.error_stats(clean_only=False)["n"] == 1


def test_unresolved_names_are_counted_across_rounds():
    tel = Telemetry()
    tel.start_fight()
    tel.observe(_Decision("Sunbird"), _read(2000, 1, misses={"Ghost Spell": 1}))
    tel.observe(_Decision("Sunbird"), _read(1500, 2, misses={"Ghost Spell": 2}))
    assert tel.unresolved_names() == {"Ghost Spell": 2}


def test_a_broken_listener_cannot_stop_a_fight():
    tel = Telemetry()
    tel.subscribe(lambda event, payload: 1 / 0)
    tel.start_fight()
    tel.observe(_Decision("Sunbird"), _read(2000, 1))     # must not raise
    assert len(tel.rounds) == 1


def test_export_round_trips(tmp_path):
    import json
    tel = Telemetry(policy_name="p", school="fire")
    tel.start_fight()
    tel.observe(_Decision("Sunbird"), _read(2000, 1))
    tel.end_fight(won=True)
    path = tel.to_json(tmp_path / "run.json")
    payload = json.load(open(path))
    assert payload["summary"]["policy"] == "p"
    assert payload["fights"][0]["won"] is True
    assert len(payload["rounds"]) == 1


# ---------------------------------------------------------------------- gui
@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_window_builds_and_every_panel_survives_empty_data(qapp):
    from deimos_bridge.gui.app import MainWindow
    win = MainWindow(Telemetry())
    for panel in (win.model, win.naming, win.runs):
        panel.refresh()
    win.model.plot.repaint()          # the empty-data path of the painter
    assert win.windowTitle().startswith("wizAi")


def test_demo_run_populates_every_panel(qapp):
    from deimos_bridge.gui.app import MainWindow, demo_telemetry
    tel = demo_telemetry()
    assert tel.rounds, "the demo produced no rounds"

    win = MainWindow(tel)
    for panel in (win.model, win.naming, win.runs):
        panel.refresh()
    win.board.render(tel.rounds[-1])
    for rec in tel.rounds:
        win.decisions.append(rec)

    assert win.decisions.table.rowCount() == len(tel.rounds)
    assert win.model.table.rowCount() == len(
        tel.damage_observations(clean_only=False))
    # the demo deliberately deals a mob an unmodelled resist, so the panel
    # must be showing a non-zero error rather than a flat pass
    assert tel.error_stats(clean_only=False)["n"] >= 1
    assert abs(tel.error_stats(clean_only=False)["mean_error"]) > 1


def test_naming_panel_surfaces_the_unresolvable_card(qapp):
    """The demo hand contains a card that is not in the table; it has to
    show up, because in a real run that is a silent failure."""
    from deimos_bridge.gui.app import MainWindow, demo_telemetry
    tel = demo_telemetry()
    win = MainWindow(tel)
    win.naming.refresh()
    assert "Not A Real Spell" in tel.unresolved_names()
    assert win.naming.table.rowCount() >= 1


def test_board_panel_shows_hangings_as_arithmetic(qapp):
    from deimos_bridge.gui.app import MainWindow, demo_telemetry
    tel = demo_telemetry()
    win = MainWindow(tel)
    with_charms = [r for r in tel.rounds if r.player_charms]
    assert with_charms, "the demo never stacked a blade"
    win.board.render(with_charms[-1])
    assert "%" in win.board.charms.text()


def test_train_button_rejects_a_deck_the_table_cannot_build(qapp, monkeypatch):
    """A decklist naming a card that does not resolve would train a policy
    whose action space does not exist. Better to refuse than to train."""
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    warned = {}
    monkeypatch.setattr(app_mod.QMessageBox, "warning",
                        lambda *a, **k: warned.setdefault("hit", a))
    win = MainWindow(Telemetry())
    win.deck.setText("Fireblade,Definitely Not A Card")
    win.on_train()
    assert "hit" in warned
    assert win.worker is None
