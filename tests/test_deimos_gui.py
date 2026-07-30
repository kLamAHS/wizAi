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
                 reason="", policy=""):
        self.card_name = card_name
        self.target_index = target_index
        self.passing = passing
        self.reason = reason
        self.policy = policy


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

    win = MainWindow(tel)          # populates itself; see refresh_all
    assert win.decisions.table.rowCount() == len(tel.rounds)
    assert win.model.table.rowCount() == len(
        tel.damage_observations(clean_only=False))
    # the demo deliberately deals a mob an unmodelled resist, so the panel
    # must be showing a non-zero error rather than a flat pass
    assert tel.error_stats(clean_only=False)["n"] >= 1
    assert abs(tel.error_stats(clean_only=False)["mean_error"]) > 1


def test_naming_panel_surfaces_the_unresolvable_card(qapp):
    """The demo hand contains cards that are not in the table; they have
    to show up, because in a real run that is a silent failure."""
    from deimos_bridge.gui.app import MainWindow, demo_telemetry
    tel = demo_telemetry()
    win = MainWindow(tel)
    win.naming.refresh()
    assert "Not A Real Spell" in tel.hidden_cards()
    assert win.naming.table.rowCount() >= 2


def test_naming_panel_separates_the_two_miss_causes(qapp):
    """A decoder gap and a nonexistent card look identical in a plain
    miss list and need opposite responses, so the panel must not merge
    them."""
    from deimos_bridge.gui.app import MainWindow, demo_telemetry
    tel = demo_telemetry()
    win = MainWindow(tel)
    win.naming.refresh()
    causes = {win.naming.table.item(i, 0).text():
              win.naming.table.item(i, 2).text()
              for i in range(win.naming.table.rowCount())}
    assert "kSummonCreature" in causes["Summon589244"]
    assert "not in the game data" in causes["Not A Real Spell"]


def test_naming_panel_leads_with_hand_visibility(qapp):
    """The miss count is not the number that matters -- how much of the
    hand the policy actually saw is, because a low figure invalidates
    every other panel."""
    from deimos_bridge.gui.app import MainWindow, demo_telemetry
    tel = demo_telemetry()
    assert tel.hand_visibility() < 0.9
    win = MainWindow(tel)
    win.naming.refresh()
    assert "% of its hand" in win.naming.headline.text()
    assert "not measuring the policy you trained" in win.naming.detail.text()


def test_hand_visibility_is_perfect_when_nothing_is_hidden(qapp):
    tel = Telemetry()
    assert tel.hand_visibility() == 1.0
    from deimos_bridge.gui.app import MainWindow
    win = MainWindow(tel)
    win.naming.refresh()
    assert "whole hand" in win.naming.headline.text()


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


# ----------------------------------------------------------------- live tab
def test_window_populates_itself_without_manual_wiring(qapp):
    """The blank-window bug: the panels only ever drew because `main()`
    hand-called them for `--demo`. Constructed with data, the window has
    to show it."""
    from deimos_bridge.gui.app import MainWindow, demo_telemetry
    tel = demo_telemetry()
    win = MainWindow(tel)
    assert win.decisions.table.rowCount() == len(tel.rounds)
    assert win.runs.table.rowCount() == len(tel.fights)
    assert win.naming.table.rowCount() >= 1
    assert "round" in win.board.round_lab.text()


def test_panels_do_not_subscribe_to_the_telemetry(qapp):
    """Thread safety, structurally. A live run fills the telemetry from a
    worker thread; a panel that updated widgets from its callback would be
    touching Qt off the GUI thread. Updates must arrive only through
    MainWindow.refresh_all, which the worker reaches by queued signal."""
    from deimos_bridge.gui.app import MainWindow
    tel = Telemetry()
    MainWindow(tel)
    assert tel._listeners == [], \
        "a panel subscribed directly; live updates would cross threads"


def test_live_worker_builds_each_policy(qapp):
    from deimos_bridge.gui.live import LiveWorker

    def worker(name, agent=None):
        return LiveWorker(Telemetry(), "ice", ["Frost Beetle"], name, 1,
                          agent=agent)

    assert callable(worker("school-aware")._build_policy())
    assert callable(worker("blade-stack(3)")._build_policy())
    assert callable(worker("blade-stack(2)")._build_policy())
    assert callable(worker("nuke-asap")._build_policy())


def test_live_worker_refuses_a_trained_policy_with_no_agent(qapp):
    """Silently falling back to a heuristic would report a heuristic's
    numbers under the trained policy's name."""
    from deimos_bridge.gui.live import LiveWorker
    w = LiveWorker(Telemetry(), "ice", ["Frost Beetle"], "trained (Q)", 1)
    with pytest.raises(RuntimeError, match="No trained policy"):
        w._build_policy()


def test_live_worker_reports_a_missing_wizwalker_clearly(qapp):
    """Off Windows this is the expected path, and it must arrive as a
    readable message on the `failed` signal rather than a crash."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker
    w = LiveWorker(Telemetry(), "ice", ["Frost Beetle"], "school-aware", 1)
    try:
        import wizwalker  # noqa: F401
        pytest.skip("wizwalker present; this is the off-Windows path")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="wizwalker did not import"):
        asyncio.run(w._go())


def test_start_live_blocks_a_trained_run_with_no_agent(qapp, monkeypatch):
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    warned = {}
    monkeypatch.setattr(app_mod.QMessageBox, "warning",
                        lambda *a, **k: warned.setdefault("hit", a))
    win = MainWindow(Telemetry())
    win.policy.setCurrentText("trained (Q)")
    win.on_start_live()
    assert "hit" in warned
    assert win.live is None
    assert win.start_btn.isEnabled()


# ------------------------------------------------------------- questing
class _Win:
    """A wizwalker window: a name, children, visibility, maybe text."""

    def __init__(self, name, children=(), visible=True, text=""):
        self._name = name
        self._children = list(children)
        self._visible = visible
        self._text = text

    async def name(self):
        return self._name

    async def children(self):
        return list(self._children)

    async def is_visible(self):
        return self._visible

    async def maybe_text(self):
        return self._text


class _Mouse:
    def __init__(self):
        self.clicks = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def click_window(self, window):
        self.clicks.append(window)
        # dialogue closes after two clicks, like a short conversation
        if len(self.clicks) >= 2:
            window._visible = False


class _XYZ:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __repr__(self):
        return f"XYZ({self.x}, {self.y}, {self.z})"


class _QuestClient:
    def __init__(self, root, position=_XYZ(1.0, 2.0, 3.0), in_battle=False,
                 loading=0, zone="Wizard City"):
        self.root_window = root
        self.mouse_handler = _Mouse()
        self.teleported = []
        self.keys = []
        self._position = position
        self._in_battle = in_battle
        self._loading = loading          # rounds of loading left
        self._zone = zone

        client = self

        class _QP:
            async def position(self):
                if client._position is None:
                    raise RuntimeError("no quest")
                return client._position

        self.quest_position = _QP()

    async def teleport(self, position):
        self.teleported.append(position)

    async def in_battle(self):
        return self._in_battle

    async def is_loading(self):
        if self._loading > 0:
            self._loading -= 1
            return True
        return False

    async def zone_name(self):
        return self._zone

    async def send_key(self, key, seconds=0):
        self.keys.append(key)


def _dialogue_root(visible=True):
    button = _Win("btnRight", visible=visible)
    text = _Win("txtMessage", text="Greetings, young wizard.")
    return _Win("root", [
        _Win("WorldView", [
            _Win("wndDialogMain", [button, _Win("txtArea", [text])])])]), button


def test_window_from_path_walks_the_tree():
    import asyncio

    from deimos_bridge.questing import ADVANCE_DIALOG_PATH, window_from_path
    root, button = _dialogue_root()
    found = asyncio.run(window_from_path(root, ADVANCE_DIALOG_PATH))
    assert found is button
    assert asyncio.run(window_from_path(root, ["WorldView", "nope"])) is None


def test_in_dialogue_tracks_visibility():
    import asyncio

    from deimos_bridge.questing import in_dialogue
    root, _ = _dialogue_root(visible=True)
    assert asyncio.run(in_dialogue(_QuestClient(root)))
    root, _ = _dialogue_root(visible=False)
    assert not asyncio.run(in_dialogue(_QuestClient(root)))


def test_advance_dialogue_stops_when_the_window_closes():
    import asyncio

    from deimos_bridge.questing import advance_dialogue
    root, _ = _dialogue_root()
    client = _QuestClient(root)
    clicks = asyncio.run(advance_dialogue(client, settle=0))
    assert clicks == 2
    assert len(client.mouse_handler.clicks) == 2


def test_advance_dialogue_is_bounded():
    """A dialogue that reopens forever must not hang the run."""
    import asyncio

    from deimos_bridge.questing import advance_dialogue

    class _Sticky(_Mouse):
        async def click_window(self, window):
            self.clicks.append(window)      # never closes

    root, _ = _dialogue_root()
    client = _QuestClient(root)
    client.mouse_handler = _Sticky()
    assert asyncio.run(advance_dialogue(client, max_clicks=5, settle=0)) == 5


def test_teleport_to_quest_reports_failure_rather_than_raising():
    import asyncio

    from deimos_bridge.questing import teleport_to_quest
    root, _ = _dialogue_root()

    ok_client = _QuestClient(root)
    ok, reason = asyncio.run(teleport_to_quest(ok_client))
    assert ok is True and reason == ""
    assert ok_client.teleported == [_XYZ(1.0, 2.0, 3.0)]

    no_quest = _QuestClient(root, position=None)
    ok, reason = asyncio.run(teleport_to_quest(no_quest))
    assert ok is False
    assert "quest arrow" in reason      # the actual cause, not a bare False
    assert no_quest.teleported == []


def test_hop_to_next_fight_stops_once_combat_starts():
    import asyncio

    from deimos_bridge.questing import hop_to_next_fight
    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root, in_battle=True)
    said = []
    assert asyncio.run(hop_to_next_fight(client, settle=0,
                                         on_status=said.append)) is True
    assert client.teleported == []          # already fighting, no hop needed


def test_hop_to_next_fight_gives_up_rather_than_spinning():
    import asyncio

    from deimos_bridge.questing import hop_to_next_fight
    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root, in_battle=False)
    said = []
    assert asyncio.run(hop_to_next_fight(client, max_hops=3, settle=0,
                                         on_status=said.append)) is False
    assert len(client.teleported) == 3
    assert "no fight after" in said[-1]


def test_hop_keeps_going_through_a_zone_change():
    """The bug behind "it teleported to another zone and then stopped".
    A zone change makes several reads fail in a row, and the first
    version returned on the first failure -- ending the hunt and leaving
    the run waiting for a fight that would never start."""
    import asyncio

    from deimos_bridge.questing import hop_to_next_fight
    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root, in_battle=False, loading=3)

    fights = {"after": 3}
    real_in_battle = client.in_battle

    async def in_battle():
        fights["after"] -= 1
        return fights["after"] <= 0

    client.in_battle = in_battle
    said = []
    assert asyncio.run(hop_to_next_fight(client, max_hops=8, settle=0,
                                         on_status=said.append)) is True


def test_hop_presses_x_to_interact(monkeypatch):
    """Arriving at the marker is often not enough -- sigils, dungeon
    doors and quest NPCs need an interact."""
    import asyncio

    from deimos_bridge import questing

    monkeypatch.setattr(questing, "keycode_x", lambda: "X")
    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root, in_battle=False)
    asyncio.run(questing.hop_to_next_fight(client, max_hops=2, settle=0))
    assert client.keys == ["X", "X"], "never pressed X"


def test_press_x_is_a_no_op_without_wizwalker(monkeypatch):
    """Off Windows there is no Keycode to send; that must not raise."""
    import asyncio

    from deimos_bridge import questing

    monkeypatch.setattr(questing, "keycode_x", lambda: None)
    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root)
    assert asyncio.run(questing.press_x(client)) is False
    assert client.keys == []


def test_hop_stops_early_when_asked():
    import asyncio

    from deimos_bridge.questing import hop_to_next_fight
    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root, in_battle=False)
    assert asyncio.run(hop_to_next_fight(
        client, max_hops=20, settle=0, should_stop=lambda: True)) is False
    assert client.teleported == []


def test_a_zeroed_quest_position_is_reported_as_the_arrow_being_off():
    """activate_all_hooks warns the quest hook is not written when the
    in-game quest arrow is off (memory/handler.py:187). That reads as
    the origin, and is the commonest cause of a teleport doing nothing."""
    import asyncio

    from deimos_bridge.questing import read_quest_position
    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root, position=_XYZ(0.0, 0.0, 0.0))
    position, reason = asyncio.run(read_quest_position(client))
    assert position is None
    assert "quest arrow" in reason


def test_wait_until_ready_rides_out_a_loading_screen():
    import asyncio

    from deimos_bridge.questing import wait_until_ready
    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root, loading=3)
    assert asyncio.run(wait_until_ready(client, timeout=5, poll=0)) is True
    assert asyncio.run(client.is_loading()) is False


# ----------------------------------------------------------- deck picker
def test_deck_picker_round_trips_a_decklist(qapp):
    from data_full import load_spells_full
    from deimos_bridge.gui.deckpicker import DeckPicker

    cards = load_spells_full()
    deck = ["Frost Beetle", "Frost Beetle", "Iceblade"]
    d = DeckPicker(cards, "ice", deck)
    assert d.decklist() == sorted(deck)


def test_deck_picker_search_filters(qapp):
    from data_full import load_spells_full
    from deimos_bridge.gui.deckpicker import DeckPicker

    d = DeckPicker(load_spells_full(), "ice")
    d.search.setText("frostbite")
    d.refilter()
    names = [d.results.item(i).data(0x0100) for i in range(d.results.count())]
    assert names and all("frostbite" in n.lower() for n in names)


def test_deck_picker_can_seed_from_the_last_fight(qapp):
    """The honest 'read it off the game': the client exposes the deck as
    template ids and wizAi's table has none to match them to, but cards
    seen in hand during a fight do have names."""
    from data_full import load_spells_full
    from deimos_bridge.gui.deckpicker import DeckPicker

    d = DeckPicker(load_spells_full(), "ice", seen=["Frost Beetle", "Iceblade"])
    assert d.seen_btn.isEnabled()
    d.on_from_seen()
    assert sorted(set(d.decklist())) == ["Frost Beetle", "Iceblade"]


def test_deck_picker_warns_about_a_deck_that_cannot_kill(qapp):
    from data_full import load_spells_full
    from deimos_bridge.gui.deckpicker import DeckPicker

    d = DeckPicker(load_spells_full(), "ice", ["Iceblade", "Iceblade"])
    assert "cannot kill" in d.warn.text()
    d.chosen["Frost Beetle"] = 3
    d.redraw_deck()
    assert d.warn.text() == ""


def test_picker_seed_button_is_off_with_no_run(qapp):
    from data_full import load_spells_full
    from deimos_bridge.gui.deckpicker import DeckPicker

    d = DeckPicker(load_spells_full(), "ice")
    assert not d.seen_btn.isEnabled()


def test_questing_buttons_need_a_live_run(qapp, monkeypatch):
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    told = {}
    monkeypatch.setattr(app_mod.QMessageBox, "information",
                        lambda *a, **k: told.setdefault("hit", a))
    win = MainWindow(Telemetry())
    win.on_teleport()
    assert "hit" in told


def test_live_worker_queues_a_questing_request(qapp):
    from deimos_bridge.gui.live import LiveWorker
    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    w.request("teleport")
    w.request("dialogue")
    assert w._requests == ["teleport", "dialogue"]


def test_picker_hides_boss_and_event_variants(qapp):
    """The table holds Iceblade, Iceblade - EM, Iceblade - SIT,
    IcebladeBOSS01 and more. A mob can cast those; a player cannot, and
    they bury the real card in the search results."""
    from deimos_bridge.gui.deckpicker import DeckPicker
    from deimos_bridge.live_state import build_catalog

    cat = build_catalog()
    assert "IcebladeBOSS01" in cat["cards"]          # still in the table
    assert "IcebladeBOSS01" not in cat["canonical"]  # but not player-facing
    assert "Iceblade" in cat["canonical"]

    d = DeckPicker(cat["cards"], "ice", canonical=cat["canonical"])
    d.search.setText("iceblade")
    d.refilter()
    names = [d.results.item(i).data(0x0100) for i in range(d.results.count())]
    assert "Iceblade" in names
    assert not any("BOSS" in n for n in names)

    d.player_only.setCurrentIndex(1)                 # every variant
    d.refilter()
    names = [d.results.item(i).data(0x0100) for i in range(d.results.count())]
    assert any("BOSS" in n for n in names)


def test_service_loop_runs_requests_while_the_fight_loop_waits(qapp):
    """The bug behind "TP to quest says teleporting and nothing happens".

    Requests used to drain at the top of the fight loop, which spends
    nearly all its time blocked inside wait_for_combat -- so a request
    queued while waiting sat there until a fight had started AND
    finished. The service loop is concurrent, so it acts in about a
    second.
    """
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root, in_battle=False)

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    worker.auto_dialogue = False
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()

    async def drive():
        task = asyncio.ensure_future(worker._service_loop(client))
        worker.request("teleport")
        for _ in range(40):                 # ~2s of 50ms ticks
            await asyncio.sleep(0.05)
            if client.teleported:
                break
        worker._stop = True
        task.cancel()
        try:
            await task
        except BaseException:
            pass

    asyncio.run(drive())
    assert client.teleported, "the request never ran"
    assert any("teleported" in m for m in said)


def test_service_loop_leaves_the_mouse_alone_during_combat(qapp):
    """Two coroutines clicking at once produce misclicks, so anything
    that clicks is skipped while a duel is on."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    root, button = _dialogue_root(visible=True)
    client = _QuestClient(root, in_battle=True)

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                        auto_dialogue=True)
    worker.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()

    async def drive():
        task = asyncio.ensure_future(worker._service_loop(client))
        worker.request("teleport")
        await asyncio.sleep(0.3)
        worker._stop = True
        task.cancel()
        try:
            await task
        except BaseException:
            pass

    asyncio.run(drive())
    assert client.teleported == []
    assert client.mouse_handler.clicks == []


def test_auto_dialogue_clicks_without_being_asked(qapp):
    """The feature: watch and click as dialogue appears, for the whole
    run, not only when the button is pressed."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    root, _ = _dialogue_root(visible=True)
    client = _QuestClient(root, in_battle=False)

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                        auto_dialogue=True)
    worker.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()

    async def drive():
        task = asyncio.ensure_future(worker._service_loop(client))
        for _ in range(40):
            await asyncio.sleep(0.05)
            if client.mouse_handler.clicks:
                break
        worker._stop = True
        task.cancel()
        try:
            await task
        except BaseException:
            pass

    asyncio.run(drive())
    assert client.mouse_handler.clicks, "auto-dialogue never fired"


# ------------------------------------------------- Deimos's own questing
def test_deimos_questing_reports_why_it_is_unavailable():
    """It has to name the module that actually failed, and must never
    tell anyone to pip install wizsprinter -- that is not a PyPI package.
    It is vendored at Deimos/libs/wizsprinter and overlaid onto the
    wizwalker namespace, so the one instruction people would copy from
    the old message could not have worked."""
    from deimos_bridge import deimos_questing

    ok, reason = deimos_questing.available()
    if ok:
        pytest.skip("Deimos's questing is importable here")
    assert "pip install wizsprinter" not in reason
    assert "not importable" in reason
    assert "light questing" in reason


def test_init_client_supplies_every_attribute_quester_reads():
    """Quester reads attributes Deimos sets in _init_client_attrs and
    wizwalker's Client has none of them, so the first read would raise
    AttributeError. This pins the ones the questing path touches."""
    import asyncio

    from deimos_bridge import deimos_questing

    class _Stats:
        async def reference_level(self):
            return 42

    class _Client:
        stats = _Stats()

    client = asyncio.run(deimos_questing.init_client(_Client()))
    for attr in ("questing_status", "use_potions", "buy_potions",
                 "auto_pet_status", "entity_detect_combat_status",
                 "character_level", "title", "helper_clients",
                 "in_solo_zone", "duel_circle_joinable"):
        assert hasattr(client, attr), attr
    assert client.character_level == 42
    assert client.questing_status is True     # Deimos's loop flag
    assert client.auto_pet_status is False    # no pet training on a data run
    assert client.helper_clients == []


def test_init_client_takes_overrides():
    import asyncio

    from deimos_bridge import deimos_questing

    class _Client:
        class stats:
            @staticmethod
            async def reference_level():
                return 1

    client = asyncio.run(
        deimos_questing.init_client(_Client(), buy_potions=True))
    assert client.buy_potions is True


def test_quester_step_survives_a_failed_read():
    """Deimos's questing reads a lot of memory and a failed read during a
    zone change is routine. A raise is a skipped tick, not a dead run --
    but it is counted, so a permanently broken setup is visible."""
    import asyncio

    from deimos_bridge.deimos_questing import DeimosQuester

    class _Boom:
        async def auto_quest_solo(self, **kw):
            raise RuntimeError("MemoryReadError")

    q = DeimosQuester(object(), _Boom())
    assert asyncio.run(q.step()) is False
    assert q.failures == 1
    assert "MemoryReadError" in q.last_error

    class _Fine:
        def __init__(self):
            self.calls = 0

        async def auto_quest_solo(self, **kw):
            self.calls += 1

    fine = _Fine()
    q = DeimosQuester(object(), fine)
    assert asyncio.run(q.step()) is True
    assert q.failures == 0
    assert fine.calls == 1


def test_worker_prefers_deimos_questing_then_falls_back(qapp):
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                        auto_quest=True)
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()

    asyncio.run(worker._setup_questing(object()))
    # Either it wired Deimos's in, or it said why it did not.
    if worker.quester is None:
        assert any("light questing" in m for m in said)
    else:
        assert any("Deimos" in m for m in said)


def test_only_one_dialogue_clicker_runs(qapp):
    """Deimos's questing does its own dialogue handling; a second clicker
    would race it for the same button."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    root, _ = _dialogue_root(visible=True)
    client = _QuestClient(root, in_battle=False)

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                        auto_quest=True, auto_dialogue=True)
    worker.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()

    class _Quester:
        failures = 0
        last_error = ""

        async def step(self, **kw):
            return True

    worker.quester = _Quester()

    async def drive():
        task = asyncio.ensure_future(worker._service_loop(client))
        await asyncio.sleep(0.3)
        worker._stop = True
        task.cancel()
        try:
            await task
        except BaseException:
            pass

    asyncio.run(drive())
    assert client.mouse_handler.clicks == [], \
        "our dialogue clicker ran while Deimos's questing was driving"


# ---------------------------------------------------------------- upkeep
class _Entity:
    def __init__(self, xyz):
        self._xyz = xyz

    async def location(self):
        return self._xyz


class _Stats:
    def __init__(self, hp=1000, max_hp=1000, mana=100, max_mana=100,
                 level=50, charges=2.0):
        self._v = dict(hp=hp, max_hp=max_hp, mana=mana, max_mana=max_mana,
                       level=level, charges=charges)

    async def current_hitpoints(self):
        return self._v["hp"]

    async def max_hitpoints(self):
        return self._v["max_hp"]

    async def current_mana(self):
        return self._v["mana"]

    async def max_mana(self):
        return self._v["max_mana"]

    async def reference_level(self):
        return self._v["level"]

    async def potion_charge(self):
        return self._v["charges"]


class _UpkeepClient:
    def __init__(self, stats=None):
        self.stats = stats or _Stats()
        self.teleported = []

    async def teleport(self, xyz):
        self.teleported.append(xyz)


def test_needs_potion_matches_deimos_threshold():
    """Low mana, or under 55% health (src/utils.py:527-544)."""
    import asyncio

    from deimos_bridge.upkeep import needs_potion

    healthy = _UpkeepClient(_Stats(hp=1000, max_hp=1000, mana=100))
    assert asyncio.run(needs_potion(healthy)) is False

    hurt = _UpkeepClient(_Stats(hp=500, max_hp=1000, mana=100))
    assert asyncio.run(needs_potion(hurt)) is True        # 50% < 55%

    drained = _UpkeepClient(_Stats(hp=1000, max_hp=1000, mana=1,
                                   max_mana=100))
    assert asyncio.run(needs_potion(drained)) is True


def test_needs_potion_is_false_on_an_unreadable_client():
    """A failed stat read must not trigger potion spam."""
    import asyncio

    from deimos_bridge.upkeep import needs_potion

    class _Broken:
        class stats:
            @staticmethod
            async def current_mana():
                raise RuntimeError("MemoryReadError")

    assert asyncio.run(needs_potion(_Broken())) is False


def test_drink_potion_does_nothing_without_a_charge():
    import asyncio

    from deimos_bridge.upkeep import drink_potion

    empty = _UpkeepClient(_Stats(charges=0.0))
    assert asyncio.run(drink_potion(empty)) is False


def test_collect_wisps_teleports_to_each(monkeypatch):
    import asyncio

    from deimos_bridge import upkeep

    picked = [_Entity("wisp1"), _Entity("wisp2")]

    class _Sprinty:
        def __init__(self, client):
            pass

        async def get_base_entities_with_vague_name(self, name):
            return picked if name == "WispHealth" else []

        async def find_safe_entities_from(self, entities):
            return entities

    monkeypatch.setattr(upkeep, "_sprinty", lambda client: _Sprinty(client))
    client = _UpkeepClient()
    assert asyncio.run(upkeep.collect_wisps(client)) == 2
    assert client.teleported == ["wisp1", "wisp2"]


def test_collect_wisps_skips_the_ones_next_to_a_mob(monkeypatch):
    """Topping up should not start a second fight."""
    import asyncio

    from deimos_bridge import upkeep

    class _Sprinty:
        def __init__(self, client):
            pass

        async def get_base_entities_with_vague_name(self, name):
            return [_Entity("safe"), _Entity("guarded")] if name == "WispHealth" else []

        async def find_safe_entities_from(self, entities):
            return [e for e in entities if e._xyz == "safe"]

    monkeypatch.setattr(upkeep, "_sprinty", lambda client: _Sprinty(client))
    client = _UpkeepClient()
    assert asyncio.run(upkeep.collect_wisps(client)) == 1
    assert client.teleported == ["safe"]


def test_collect_wisps_is_bounded(monkeypatch):
    """A zone strewn with pickups must not stall the loop."""
    import asyncio

    from deimos_bridge import upkeep

    many = [_Entity(f"w{i}") for i in range(50)]

    class _Sprinty:
        def __init__(self, client):
            pass

        async def get_base_entities_with_vague_name(self, name):
            return many if name == "WispHealth" else []

        async def find_safe_entities_from(self, entities):
            return entities

    monkeypatch.setattr(upkeep, "_sprinty", lambda client: _Sprinty(client))
    client = _UpkeepClient()
    assert asyncio.run(upkeep.collect_wisps(client, limit=5)) == 5


def test_collect_wisps_is_a_no_op_when_sprinty_is_missing(monkeypatch):
    """On a machine without Deimos this returns 0, not an exception."""
    import asyncio

    from deimos_bridge import upkeep

    def _boom(client):
        raise ImportError("no src.sprinty_client")

    monkeypatch.setattr(upkeep, "_sprinty", _boom)
    assert asyncio.run(upkeep.collect_wisps(_UpkeepClient())) == 0


def test_after_fight_takes_wisps_before_potions(monkeypatch):
    """Wisps are free and can lift the wizard back over the potion
    threshold, so spending a charge first would waste it."""
    import asyncio

    from deimos_bridge import upkeep

    order = []

    async def _wisps(client, **kw):
        order.append("wisps")
        return 1

    async def _needs(client, **kw):
        order.append("check")
        return False

    monkeypatch.setattr(upkeep, "collect_wisps", _wisps)
    monkeypatch.setattr(upkeep, "needs_potion", _needs)
    asyncio.run(upkeep.after_fight(_UpkeepClient()))
    assert order == ["wisps", "check"]


def test_upkeep_toggles_reach_the_worker(qapp):
    from deimos_bridge.gui.app import MainWindow
    from deimos_bridge.gui.live import LiveWorker

    win = MainWindow(Telemetry())
    assert win.collect_wisps.isChecked()
    assert win.use_potions.isChecked()

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                        collect_wisps=False, use_potions=False)
    assert worker.collect_wisps is False
    assert worker.use_potions is False


# ------------------------------------------------------------ ttk policy
def _ice_board(deck, hand, pips, hp):
    import random

    from data_full import load_spells_full
    from w101_sim import Actor, Boss, Sim, State

    cards = load_spells_full()
    sim = Sim(cards, deck, "ice",
              Boss(name="mob", hp=hp, school="fire", dmg=40),
              rng=random.Random(0), player_hp=800)
    p = Actor(name="W", school="ice", hp=800, max_hp=800, team=0,
              norm_pips=pips)
    p.hand = [cards[n] for n in hand]
    p.deck = [cards[n] for n in deck]
    return sim, State(p, [Actor(name="mob", school="fire", hp=hp,
                                max_hp=hp, team=1)])


def _name(action):
    """Card name from a policy return, which is now an aimed (card, target)."""
    from deimos_bridge.policies import _split

    card, _ = _split(action)
    return getattr(card, "name", card) if card is not None else "PASS"


def test_heuristic_no_longer_passes_on_an_affordable_hit():
    """A level-6 ice wizard with one pip, a 1-pip Frost Beetle and a
    2-pip Snow Serpent used to pass the turn away: it picked Serpent as
    the nuke to build toward, could not afford it, found no buffs, and
    returned None."""
    from deimos_bridge.policies import school_aware_blade_stack

    sim, s = _ice_board(["Frost Beetle"] * 3 + ["Snow Serpent"] * 3,
                        ["Frost Beetle", "Snow Serpent"], 1, 400)
    assert _name(school_aware_blade_stack(3)(sim, s)) == "Frost Beetle"


def test_ttk_banks_a_pip_when_that_kills_sooner():
    """The calculation the heuristic could not make: with one pip against
    a 400hp mob, waiting a turn for Snow Serpent beats firing Frost
    Beetle now."""
    from deimos_bridge.policies import greedy_ttk

    sim, s = _ice_board(["Frost Beetle"] * 3 + ["Snow Serpent"] * 3,
                        ["Frost Beetle", "Snow Serpent"], 1, 400)
    assert _name(greedy_ttk()(sim, s)) == "PASS"


def test_ttk_hits_when_the_mob_dies_this_turn_anyway():
    """Banking a pip must not become a reflex -- against 170hp the beetle
    line kills just as fast, and ties go to acting."""
    from deimos_bridge.policies import greedy_ttk

    sim, s = _ice_board(["Frost Beetle"] * 3 + ["Snow Serpent"] * 3,
                        ["Frost Beetle", "Snow Serpent"], 1, 170)
    assert _name(greedy_ttk()(sim, s)) == "Frost Beetle"


def test_ttk_skips_traps_that_do_not_pay_off():
    """Three Ice Traps in the deck meant three Ice Traps on the boss
    before a single hit, which against a weak mob is three wasted turns.
    Traps should go down only when the fight is long enough to use them."""
    from deimos_bridge.policies import greedy_ttk

    deck = ["Ice Trap"] * 3 + ["Frost Beetle"] * 3
    sim, s = _ice_board(deck, ["Ice Trap", "Frost Beetle"], 2, 200)
    assert _name(greedy_ttk()(sim, s)) == "Frost Beetle"

    sim, s = _ice_board(deck, ["Ice Trap", "Frost Beetle"], 2, 1500)
    assert _name(greedy_ttk()(sim, s)) == "Ice Trap"


def test_rollout_is_deterministic():
    """Rollouts compare candidate moves, so the comparison must not turn
    on whether a cast happened to fizzle."""
    from deimos_bridge.policies import _rollout

    scores = set()
    for _ in range(5):
        sim, s = _ice_board(["Frost Beetle"] * 4,
                            ["Frost Beetle"], 2, 300)
        card = s.hand[0]
        scores.add(_rollout(sim, s, card, 12))
    assert len(scores) == 1


def test_rollout_does_not_mutate_the_live_state():
    from deimos_bridge.policies import _rollout

    sim, s = _ice_board(["Frost Beetle"] * 4, ["Frost Beetle"], 2, 300)
    hp, pips, hand = s.enemies[0].hp, s.norm_pips, [c.name for c in s.hand]
    _rollout(sim, s, s.hand[0], 12)
    assert s.enemies[0].hp == hp
    assert s.norm_pips == pips
    assert [c.name for c in s.hand] == hand


# ---------------------------------------------------------------- scripts
def test_script_check_reports_why_it_cannot_compile():
    from deimos_bridge.scripts import available, check

    ok, _ = available()
    if not ok:
        good, reason = check("anything")
        assert "pip install" in reason or "not importable" in reason
        return
    good, reason = check("this is not valid deimoslang @@@")
    assert good is False and reason


def test_script_runner_stops_cleanly():
    import asyncio

    from deimos_bridge.scripts import ScriptRunner

    class _VM:
        running = True
        killed = False

        async def step(self):
            self.running = False

        def kill(self):
            self.killed = True

    vm = _VM()
    runner = ScriptRunner(vm, "src")
    assert asyncio.run(runner.step()) is True
    assert asyncio.run(runner.step()) is False
    assert runner.finished
    runner.stop()
    assert vm.killed


def test_script_runner_counts_failures():
    import asyncio

    from deimos_bridge.scripts import ScriptRunner

    class _Boom:
        running = True

        async def step(self):
            raise RuntimeError("bad instruction")

        def kill(self):
            pass

    runner = ScriptRunner(_Boom(), "src")
    assert asyncio.run(runner.step()) is False
    assert runner.failures == 1
    assert "bad instruction" in runner.last_error


def test_script_dialog_round_trips(qapp):
    from deimos_bridge.gui.scriptdialog import ScriptDialog

    d = ScriptDialog("waitfor combat\n")
    assert d.source() == "waitfor combat\n"
    d.on_check()
    assert d.result.text()          # says something either way


# --------------------------------------------------------- trained policy
def _tiny_agent(player_hp, episodes=1200):
    from data_full import load_spells_full
    from rl_agent import train_agent
    from w101_sim import Boss

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 4 + ["Snow Serpent"] * 4 + ["Iceblade"] * 2
    agent, _ = train_agent(
        cards, deck, "ice",
        Boss(name="d", hp=1200, school="fire", dmg=60),
        episodes=episodes, player_hp=player_hp)
    return agent, cards, deck


def _live_sim(cards, deck, player_hp=800, mob_hp=1200):
    import random

    from w101_sim import Boss, Sim
    return Sim(cards, deck, "ice",
               Boss(name="mob", hp=mob_hp, school="fire", dmg=60),
               rng=random.Random(3), player_hp=player_hp)


def test_an_immortally_trained_agent_knows_nothing_about_a_live_board():
    """The reported bug, reproduced. train_agent defaults to
    player_hp=10**9 and Featurizer.key writes -1 into the health slot for
    an immortal fight, so a policy trained on the default shares almost
    no state with a mortal wizard -- the Q table reads zero, and
    QAgent.greedy falls through to the first legal action, which is
    PASS."""
    from deimos_bridge.policies import trained_policy

    agent, cards, deck = _tiny_agent(player_hp=10 ** 9)
    wrapped = trained_policy(agent)
    sim = _live_sim(cards, deck)
    for _ in range(6):
        s = sim.new_state()
        wrapped(sim, s)
    assert wrapped.coverage < 0.5, wrapped.coverage


def test_training_mortal_makes_the_state_spaces_overlap():
    from deimos_bridge.policies import trained_policy

    agent, cards, deck = _tiny_agent(player_hp=800)
    wrapped = trained_policy(agent)
    sim = _live_sim(cards, deck, player_hp=800)
    for _ in range(6):
        s = sim.new_state()
        wrapped(sim, s)
    assert wrapped.coverage > 0.5, wrapped.coverage


def test_an_unseen_state_falls_back_instead_of_passing():
    """The safety net. Even trained mortal, real bosses and wand item
    cards produce states no training visited, and silently passing is
    the worst possible answer."""
    from deimos_bridge.policies import trained_policy

    agent, cards, deck = _tiny_agent(player_hp=10 ** 9)
    sim = _live_sim(cards, deck)
    s = sim.new_state()

    assert agent.policy()(sim, s) is None          # raw agent: pass
    wrapped = trained_policy(agent)
    assert wrapped(sim, s) is not None             # wrapped: plays
    assert wrapped.missed == 1


def test_the_wrapper_does_not_grow_the_q_table():
    """QAgent.Q is a defaultdict; indexing it to ask whether a state is
    known would insert a zero for every board ever seen."""
    from deimos_bridge.policies import trained_policy

    agent, cards, deck = _tiny_agent(player_hp=800)
    before = len(agent.Q)
    wrapped = trained_policy(agent)
    sim = _live_sim(cards, deck)
    for _ in range(4):
        wrapped(sim, sim.new_state())
    assert len(agent.Q) == before


def test_coverage_starts_at_one_and_tracks_misses():
    from deimos_bridge.policies import TrainedPolicy

    class _Agent:
        Q = {}

        class feat:
            @staticmethod
            def key(sim, s):
                return ("k",)

            @staticmethod
            def legal(sim, s):
                return ["__pass__"]

    p = TrainedPolicy(_Agent(), fallback=lambda sim, s: "fallback")
    assert p.coverage == 1.0
    assert p(None, None) == "fallback"
    assert p.coverage == 0.0


def test_gui_trains_mortal(qapp):
    """The default that caused the bug was immortal; the window must not
    reintroduce it."""
    from deimos_bridge.gui.app import MainWindow, TrainWorker

    win = MainWindow(Telemetry())
    assert win.player_hp.value() < 10 ** 9
    worker = TrainWorker({}, [], "ice", 500, player_hp=win.player_hp.value())
    assert worker.player_hp == win.player_hp.value()


# --------------------------------------------- swapping models while connected
def _real_backend():
    """The genuine `WizAiBackend`, not a stand-in.

    A stub with the same attribute names would pass these tests while the
    real swap contract rotted underneath -- that failure mode has already
    cost this project once. The backend needs no client to construct.
    """
    from data_full import load_spells_full

    from deimos_bridge.live_backend import WizAiBackend
    return WizAiBackend.from_trained(
        school="ice", deck=["Frost Beetle"] * 4, cards=load_spells_full(),
        policy=lambda sim, s: None, policy_name="school-aware")


def _ice_combat():
    """A board a swapped-in policy can actually decide on."""
    from deimos_bridge.mock_client import MockCard, MockCombat, MockMember
    return MockCombat(
        [MockMember("Wizard", 800, client=True, team_id=0, normal_pips=2),
         MockMember("Lost Soul", 450, monster=True, team_id=1)],
        [MockCard("Frost Beetle"), MockCard("Iceblade")])


def test_set_policy_swaps_without_touching_the_connection(qapp):
    """The whole point. Reconnecting to change models throws away what
    the run observed -- the cards the deck picker learned, the health the
    client reported -- which are the inputs to the next decision."""
    from deimos_bridge.gui.live import LiveWorker

    tel = Telemetry()
    w = LiveWorker(tel, "ice", ["Frost Beetle"] * 4, "school-aware", 1)
    w._backend = be = _real_backend()
    was = be.policy

    assert w.set_policy("ttk-lookahead") is True
    assert be.policy_name == "ttk-lookahead"
    assert be.policy is not was                   # actually replaced
    assert w.policy_name == "ttk-lookahead"
    assert tel.policy_name == "ttk-lookahead"

    # And the swapped-in policy plays, rather than merely being installed.
    be.attach_combat(_ice_combat())
    import asyncio
    d = asyncio.run(be.decide())
    assert not d.passing, d.reason
    assert d.policy == "ttk-lookahead"


def test_selecting_trained_with_nothing_trained_keeps_the_old_policy(qapp):
    """A backend left with no policy cannot play, and the fight is still
    running -- so a failed swap has to be a no-op, not a teardown."""
    from deimos_bridge.gui.live import LiveWorker

    w = LiveWorker(Telemetry(), "ice", ["Frost Beetle"] * 4, "school-aware", 1)
    be = w._backend = _real_backend()
    was = be.policy

    assert w.set_policy("trained (Q)") is False
    assert be.policy is was
    assert be.policy_name == "school-aware"
    assert w.policy_name == "school-aware"


def test_swapping_away_from_trained_drops_the_coverage_readout(qapp):
    """`trained` drives the 'Q table decided N%' line. Left set, it would
    report a learned policy's numbers for the heuristic that replaced
    it."""
    from deimos_bridge.gui.live import LiveWorker

    agent, cards, deck = _tiny_agent(player_hp=800)
    w = LiveWorker(Telemetry(), "ice", deck, "trained (Q)", 1, agent=agent)
    w._backend = _real_backend()

    assert w.set_policy("trained (Q)") is True
    assert w.trained is not None
    assert w.set_policy("ttk-lookahead") is True
    assert w.trained is None


def test_the_dropdown_swaps_a_running_fight(qapp):
    """Wiring check: changing the combo has to reach the worker, not
    just sit there until the next Play live."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    swaps = []

    class _Live:
        trained = None

        def isRunning(self):
            return True

        def set_policy(self, name, agent=None):
            swaps.append(name)
            return True

    win.live = _Live()
    # Something other than the default, or the combo emits nothing.
    assert win.policy.currentText() != "nuke-asap"
    win.policy.setCurrentText("nuke-asap")
    assert swaps == ["nuke-asap"]


def test_training_stays_available_during_a_live_run(qapp):
    """Requiring a disconnect to train meant training on guesses: the
    deck the picker learned and the health the client reported both come
    from a connected run."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.on_start_live = lambda: None       # not connecting to a real game
    assert win.train_btn.isEnabled()


def test_retraining_hands_the_new_table_to_the_running_fight(qapp):
    """Otherwise the fight keeps playing the table that was current when
    Play live was pressed, and a retrain looks like it did nothing."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    handed = []

    class _Live:
        trained = None

        def isRunning(self):
            return True

        def set_policy(self, name, agent=None):
            handed.append((name, agent))
            return True

    win.live = _Live()
    win.policy.blockSignals(True)
    win.policy.setCurrentText("trained (Q)")
    win.policy.blockSignals(False)

    sentinel = object()
    win.on_trained(sentinel)
    assert handed == [("trained (Q)", sentinel)]
    assert win.agent is sentinel


def test_max_health_is_read_off_the_client(qapp):
    """Training buckets health as a fraction of the maximum, so a table
    trained against a typed-in 800 and played on a 1,300 HP wizard
    indexes different states for the same board."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.on_hp_read(1337)
    assert win.player_hp.value() == 1337


def test_reading_max_health_never_fails_the_connect(qapp):
    """A stat read is a nicety; the hooks are already up by then and
    losing the run over it would be absurd."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    class _Stats:
        async def max_hitpoints(self):
            raise RuntimeError("bad read")

    class _Client:
        stats = _Stats()

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    asyncio.run(w._read_max_hp(_Client()))       # must not raise


def test_the_run_records_which_policy_played_each_round():
    """Read back off the rounds, not off a counter on the policy: after
    a mid-run swap the policy object only knows about the rounds it was
    installed for."""
    tel = Telemetry(policy_name="school-aware")
    read = _read(2000, 1, hand=("Fireblade",))
    tel.observe(_Decision(card_name="Fireblade", policy="school-aware"), read)
    tel.observe(_Decision(card_name="Fireblade",
                          policy="trained (Q) — Q table"), read)
    tel.observe(_Decision(card_name="Fireblade",
                          policy="trained (Q) — Q table"), read)
    assert tel.policy_mix() == {"trained (Q) — Q table": 2, "school-aware": 1}


def test_a_round_with_no_policy_label_falls_back_to_the_run_name():
    """Older records and the CLI path carry no per-decision label."""
    tel = Telemetry(policy_name="ttk-lookahead")
    tel.observe(_Decision(card_name="Fireblade"),
                _read(2000, 1, hand=("Fireblade",)))
    assert tel.rounds[-1].policy == "ttk-lookahead"


def test_the_window_says_whether_the_selected_model_is_driving(qapp):
    """The reported symptom was that a trained policy 'just passed every
    turn' with nothing on screen to say why. Coverage has to be visible
    without exporting the run."""
    from deimos_bridge.gui.app import MainWindow
    from deimos_bridge.policies import TrainedPolicy

    tel = Telemetry(policy_name="trained (Q)")
    read = _read(2000, 1, hand=("Fireblade",))
    for _ in range(3):
        tel.observe(_Decision(card_name="Fireblade",
                              policy="trained (Q) — fallback (state not in "
                                     "Q table)"), read)
    win = MainWindow(tel)

    class _Agent:
        Q = {}

        class feat:
            @staticmethod
            def key(sim, s):
                return ("k",)

            @staticmethod
            def legal(sim, s):
                return ["__pass__"]

    # Built through the real constructor, then driven -- a hand-stuffed
    # instance would not prove the counters move.
    trained = TrainedPolicy(_Agent(), fallback=lambda sim, s: None)
    for _ in range(3):
        trained(None, None)

    class _Live:
        def isRunning(self):
            return True

    live = _Live()
    live.trained = trained
    win.live = live

    win._update_policy_state()
    text = win.policy_state.text()
    assert "3 round(s)" in text
    assert "fallback" in text
    assert "0%" in text


# ---------------------------------------------------------------- hotkeys
def test_hotkeys_report_why_they_are_unavailable():
    """Off Windows there is no RegisterHotKey. It has to say so rather
    than raise, or the checkbox takes down the run."""
    from deimos_bridge import hotkeys

    ok, reason = hotkeys.available()
    if not ok:
        assert "Windows" in reason or "import" in reason


def test_a_keypress_lands_in_the_same_queue_as_the_button(qapp):
    """A hotkey must not drive the client itself. The service task owns
    the mouse between fights; a second clicker firing mid-cast misclicks.
    So a keypress does exactly what the button does -- queues a request."""
    import asyncio

    from deimos_bridge import hotkeys
    from deimos_bridge.gui.live import LiveWorker

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                   hotkeys={"teleport": "F1"})
    hk = hotkeys.Hotkeys({"teleport": "F1"}, w.request)
    asyncio.run(hk._make("teleport")())
    assert w._requests == ["teleport"]


def test_an_unavailable_key_is_skipped_not_fatal(qapp):
    """The usual cause is another program already holding the key --
    which only the person at the keyboard can fix, and which is no reason
    to lose the fight."""
    import asyncio

    from deimos_bridge import hotkeys

    class _Listener:
        started = False

        async def add_hotkey(self, key, cb, **kw):
            if key.name == "F1":
                raise ValueError("already registered")

        def start(self):
            self.__class__.started = True

    said = []
    hk = hotkeys.Hotkeys({"teleport": "F1", "dialogue": "F2"},
                         lambda a: None, on_status=said.append)

    class _WW:
        HotkeyListener = _Listener

        class Keycode:
            class F1:
                name = "F1"

            class F2:
                name = "F2"

    import sys
    real = sys.modules.get("wizwalker")
    sys.modules["wizwalker"] = _WW
    try:
        installed = asyncio.run(hk.start())
    finally:
        if real is None:
            del sys.modules["wizwalker"]
        else:
            sys.modules["wizwalker"] = real

    assert installed == {"dialogue": "F2"}          # F2 still took
    assert any("another program" in m for m in said)


def test_the_window_offers_a_hotkey_for_the_teleport_button(qapp):
    """The ask: teleport without leaving a full-screen game."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    assert "teleport" in win.hotkey_boxes
    assert win.hotkey_bindings()["teleport"] == "F1"

    win.hotkey_boxes["teleport"].setCurrentText("F5")
    assert win.hotkey_bindings()["teleport"] == "F5"

    win.use_hotkeys.setChecked(False)
    assert win.hotkey_bindings() == {}


def test_two_actions_cannot_share_one_key(qapp):
    """RegisterHotKey takes the first and refuses the second, so the
    second action would look bound and silently do nothing."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.hotkey_boxes["teleport"].setCurrentText("F4")
    win.hotkey_boxes["dialogue"].setCurrentText("F4")
    bindings = win.hotkey_bindings()
    assert list(bindings.values()) == ["F4"]
    assert "bound twice" in win.status.text()


def test_hotkey_choices_are_all_real_keycodes():
    """Every key the dropdown offers has to resolve, or picking it fails
    at run time with the game already open."""
    from deimos_bridge import hotkeys

    ok, _ = hotkeys.available()
    if not ok:
        pytest.skip("wizwalker (Windows) not importable")
    for name in hotkeys.KEY_CHOICES:
        assert hotkeys.resolve(name) is not None, name


# ------------------------------------------------------- gear, and over-buffing
class _GearStats:
    """`GameStats`, as much of it as `read_player_stats` touches.

    The by-school vectors are indexed by Deimos's `school_list_ids`
    ordering: fire 0, ice 1, storm 2, myth 3, life 4, death 5, balance 6.
    """

    def __init__(self, dmg=None, dmg_all=0.0, acc=None, pierce=None,
                 resist=None, crit=0.0, block=0.0):
        self._dmg = dmg or [0.0] * 7
        self._dmg_all = dmg_all
        self._acc = acc or [0.0] * 7
        self._pierce = pierce or [0.0] * 7
        self._resist = resist or [0.0] * 7
        self._crit, self._block = crit, block

    async def dmg_bonus_percent(self):
        return list(self._dmg)

    async def dmg_bonus_percent_all(self):
        return self._dmg_all

    async def acc_bonus_percent(self):
        return list(self._acc)

    async def acc_bonus_percent_all(self):
        return 0.0

    async def ap_bonus_percent(self):
        return list(self._pierce)

    async def ap_bonus_percent_all(self):
        return 0.0

    async def dmg_reduce_percent(self):
        return list(self._resist)

    async def dmg_reduce_percent_all(self):
        return 0.0

    async def critical_hit_percent_all(self):
        return self._crit

    async def block_percent_all(self):
        return self._block


class _GearClient:
    def __init__(self, stats):
        self.stats = stats


def test_gear_is_read_off_the_client_per_school():
    """The simulator was pricing every hit as though the wizard wore
    nothing, and then optimising that fight instead of the real one."""
    import asyncio

    from deimos_bridge.live_state import read_player_stats

    dmg = [0.0] * 7
    dmg[1] = 0.09                                # ice
    stats = _GearStats(dmg=dmg, pierce=[0.0, 0.04, 0, 0, 0, 0, 0])
    got = asyncio.run(read_player_stats(_GearClient(stats), "ice"))
    assert got["damage"] == {"ice": pytest.approx(0.09)}
    assert got["pierce"] == pytest.approx(0.04)


def test_the_universal_stat_is_added_to_the_school_one():
    """A stat lives in two places -- the by-school vector and an 'all
    schools' scalar -- and Deimos adds them (`combat_math.real_stat`).
    Reading only the vector drops everything granted universally, which
    on low-level gear is most of it."""
    import asyncio

    from deimos_bridge.live_state import read_player_stats

    dmg = [0.0] * 7
    dmg[1] = 0.09
    got = asyncio.run(read_player_stats(
        _GearClient(_GearStats(dmg=dmg, dmg_all=0.06)), "ice"))
    assert got["damage"]["ice"] == pytest.approx(0.15)


def test_a_crit_rating_is_not_mistaken_for_a_probability():
    """Some builds report crit as a rating rather than a percentage, and
    feeding a rating in as a probability makes every cast a critical."""
    import asyncio

    from deimos_bridge.live_state import read_player_stats

    got = asyncio.run(read_player_stats(
        _GearClient(_GearStats(crit=140.0)), "ice"))
    assert "crit" not in got
    got = asyncio.run(read_player_stats(
        _GearClient(_GearStats(crit=0.12)), "ice"))
    assert got["crit"] == pytest.approx(0.12)


def test_unreadable_gear_is_empty_not_an_exception():
    """It is read right after the hooks come up; losing the run over a
    stat would be absurd."""
    import asyncio

    from deimos_bridge.live_state import read_player_stats

    class _Bad:
        def __getattr__(self, _name):
            raise RuntimeError("no")

    assert asyncio.run(read_player_stats(_GearClient(_Bad()), "ice")) == {}
    assert asyncio.run(read_player_stats(object(), "ice")) == {}


def test_gear_reaches_the_backend_and_the_trainer(qapp):
    """Both, or the Q table is learned for a different wizard than the
    one it plays."""
    from deimos_bridge.gui.app import MainWindow, TrainWorker
    from deimos_bridge.live_backend import WizAiBackend

    stats = {"damage": {"ice": 0.09}, "pierce": 0.04}
    win = MainWindow(Telemetry())
    win.on_gear_read(stats)
    assert win.player_stats == stats
    worker = TrainWorker({}, [], "ice", 500, player_stats=win.player_stats)
    assert worker.player_stats == stats

    be = WizAiBackend.from_trained(school="ice", deck=["Frost Beetle"],
                                   policy=lambda sim, s: None,
                                   player_stats=stats)
    assert be.player_stats == stats


def test_gear_changes_what_the_simulator_predicts():
    """If it did not, threading it through would be decoration."""
    import random

    from data_full import load_spells_full
    from w101_sim import Actor, Boss, Sim, State

    from deimos_bridge.telemetry import predict_damage

    cards = load_spells_full()

    def probe(stats):
        sim = Sim(cards, ["Snow Serpent"], "ice",
                  Boss(name="m", hp=400, school="fire", dmg=0),
                  rng=random.Random(0), player_hp=900, player_stats=stats)
        p = Actor(name="W", school="ice", hp=900, max_hp=900, team=0,
                  norm_pips=6, damage_bonus=dict((stats or {})
                                                 .get("damage", {})))
        p.hand = [cards["Snow Serpent"]]
        s = State(p, [Actor(name="m", school="fire", hp=400, max_hp=400,
                            team=1)])
        return predict_damage(sim, s, cards["Snow Serpent"], 0)

    bare, geared = probe({}), probe({"damage": {"ice": 0.09}})
    assert geared > bare
    assert geared == pytest.approx(bare * 1.09, rel=1e-6)


def test_the_window_says_when_gear_was_never_read(qapp):
    """Silence here reads as 'fine'. It is not -- it is the state that
    makes the policy over-buff."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    assert "as if you wore none" in win.policy_state.text()
    win.on_gear_read({"damage": {"ice": 0.09}})
    assert "9% ice damage" in win.policy_state.text()


# ------------------------------- training the board you are actually fighting
def _key_for(mob_hps, deck=None, player_hp=784):
    """The opening state key for a board with these mobs."""
    import random

    from data_full import load_spells_full
    from rl_agent import Featurizer
    from w101_sim import Boss, Sim

    cards = load_spells_full()
    deck = deck or (["Frost Beetle"] * 3 + ["Ice Trap"] * 3
                    + ["Snow Serpent"] * 3)
    sim = Sim(cards, deck, "ice",
              Boss(name="a", hp=mob_hps[0], school="fire", dmg=65),
              enemies=[Boss(name=f"b{i}", hp=h, school="fire", dmg=65)
                       for i, h in enumerate(mob_hps[1:], 1)],
              rng=random.Random(0), player_hp=player_hp)
    return Featurizer(cards, deck).key(sim, sim.new_state())


def test_a_table_trained_1v1_cannot_match_a_two_mob_board():
    """Not "matches badly" -- cannot match. `Featurizer.key` appends its
    targeting tuple only when the board holds more than one enemy, so the
    keys are different LENGTHS. Measured coverage: 0%, at any number of
    episodes."""
    solo = _key_for([1200])
    pair = _key_for([515, 390])
    assert len(solo) != len(pair)
    assert solo != pair


def test_mobs_of_equal_health_never_produce_a_real_opening_state():
    """The subtler half, and the one that survived matching the count.
    The key carries (living, weakest_index, ...); with every mob on the
    same health the weakest is index 0 in every opening state, so the
    whole weakest-is-not-first half of the space goes unvisited -- and a
    real board of 515 beside 390 opens squarely in it. Also measured at
    0% coverage."""
    equal = _key_for([500, 500])
    spread = _key_for([515, 390])
    assert len(equal) == len(spread)
    assert equal[-1] != spread[-1]              # only the foes tuple differs
    assert equal[:-1] == spread[:-1]
    assert equal[-1][1] == 0 and spread[-1][1] == 1


def test_the_trainer_spreads_mobs_when_no_board_was_observed(qapp):
    """Falling back to one number repeated is the degenerate board."""
    from deimos_bridge.gui.app import TrainWorker

    w = TrainWorker({}, [], "ice", 500, boss_hp=1000, n_enemies=3)
    hps = w.board_hps()
    assert len(hps) == 3
    assert len(set(hps)) == 3, hps


def test_the_trainer_prefers_the_healths_actually_observed(qapp):
    from deimos_bridge.gui.app import TrainWorker

    w = TrainWorker({}, [], "ice", 500, boss_hp=1000, n_enemies=2,
                    mob_hps=[515, 390])
    assert w.board_hps() == [515, 390]


def test_the_window_adopts_the_board_it_just_fought(qapp):
    """Centres the training range on real fights, so nobody has to know
    that mob count and health touch the state key at all. It only ever
    widens -- a model that already covers three mobs must not forget them
    because the last fight had one."""
    from deimos_bridge.gui.app import MainWindow

    tel = Telemetry()
    read = _read(2000, 1, hand=("Fireblade",))
    from deimos_bridge.telemetry import EnemyView
    read.state.enemies = []
    rec = tel.observe(_Decision(card_name="Fireblade"), read)
    rec.enemies = [EnemyView("Alicane", 515, 515),
                   EnemyView("Magma Man", 390, 390)]

    win = MainWindow(tel)
    win.n_enemies.setValue(1)
    assert win.adopt_observed_board()
    assert win.n_enemies.value() == 2
    assert win.boss_hp.value() == 515
    assert win.observed_hps == [515, 390]

    win.n_enemies.setValue(4)          # already broader
    win.adopt_observed_board()
    assert win.n_enemies.value() == 4  # not narrowed back to 2


def test_the_coverage_warning_names_the_mismatch_not_more_episodes(qapp):
    """"Train more episodes" is wrong advice for every cause here, and
    expensive advice to follow before finding that out."""
    from deimos_bridge.gui.app import MainWindow
    from deimos_bridge.telemetry import EnemyView

    tel = Telemetry()
    rec = tel.observe(_Decision(card_name="Fireblade"),
                      _read(2000, 1, hand=("Fireblade",)))
    rec.enemies = [EnemyView("A", 515, 515), EnemyView("B", 390, 390)]

    win = MainWindow(tel)
    win.n_enemies.setValue(1)
    assert "Raise 'up to mobs' to 2" in win._why_coverage_is_low()

    win.n_enemies.setValue(2)
    win.boss_hp.setValue(4000)         # 515 falls outside 1600-7200
    assert "HP//250" in win._why_coverage_is_low()

    win.generalize.setChecked(False)
    assert "any board" in win._why_coverage_is_low()


# ------------------------------------------ one model, more than one fight
def _covers(agent, mob_hps, n=12):
    """Coverage of a trained agent on a board of these mobs."""
    import random

    from data_full import load_spells_full
    from w101_sim import Boss, Sim

    from deimos_bridge.policies import trained_policy

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3
    wrapped = trained_policy(agent)
    for seed in range(n):
        sim = Sim(cards, deck, "ice",
                  Boss(name="m0", hp=mob_hps[0], school="fire", dmg=65),
                  enemies=[Boss(name=f"m{i}", hp=h, school="fire", dmg=65)
                           for i, h in enumerate(mob_hps[1:], 1)],
                  rng=random.Random(seed), player_hp=784)
        wrapped(sim, sim.new_state())
    return wrapped.coverage


def test_a_randomised_board_covers_fights_it_never_saw():
    """The reason 'any board' is on by default. Trained on one board, a
    table covers that board and nothing else -- and retraining before
    every fight needs you to know the board before you can learn to fight
    it, which is not a workflow."""
    from data_full import load_spells_full
    from rl_agent import make_board_sampler, train_agent
    from w101_sim import Boss

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3
    boards = [[515], [515, 390], [400, 600, 350]]

    fixed, _ = train_agent(
        cards, deck, "ice", Boss(name="d", hp=800, school="ice", dmg=65),
        episodes=1500, player_hp=784)
    roaming, _ = train_agent(
        cards, deck, "ice", Boss(name="d", hp=800, school="ice", dmg=65),
        episodes=1500, player_hp=784,
        board_sampler=make_board_sampler("ice", (300, 1400), max_mobs=3,
                                         dmg=65))

    for hps in boards:
        assert _covers(fixed, hps) == 0.0, hps
        assert _covers(roaming, hps) > 0.5, hps


def test_the_sampler_varies_both_the_count_and_which_mob_is_weakest():
    """Equal healths, or healths sorted, would make the weakest index
    constant -- the same degeneracy that measured 0% coverage, wearing a
    different hat."""
    import random

    from rl_agent import make_board_sampler

    sample = make_board_sampler("ice", (200, 1200), max_mobs=3, dmg=60)
    rng = random.Random(0)
    counts, weakest = set(), set()
    for _ in range(200):
        boss, extra = sample(rng)
        hps = [boss.hp] + [b.hp for b in extra]
        counts.add(len(hps))
        weakest.add(min(range(len(hps)), key=lambda i: hps[i]))
    assert counts == {1, 2, 3}
    assert weakest == {0, 1, 2}


def test_training_scores_clearing_the_board_not_killing_the_first_mob():
    """`State.boss_hp` is `enemies[0].hp`, so the episode used to declare
    victory the moment the first mob fell and hand out the full reward
    with the rest still swinging."""
    import random

    from data_full import load_spells_full
    from rl_agent import QAgent
    from w101_sim import Boss, Sim

    cards = load_spells_full()
    deck = ["Snow Serpent"] * 9
    sim = Sim(cards, deck, "ice",
              Boss(name="a", hp=1, school="fire", dmg=0),
              enemies=[Boss(name="b", hp=10 ** 6, school="fire", dmg=0)],
              rng=random.Random(0), player_hp=784)
    agent = QAgent(cards, deck, "ice", rng=random.Random(1))
    turns, won = agent.train_episode(sim, eps=1.0, dp_w=0.0)
    # mob b cannot be killed inside the horizon, so this is not a win
    assert not won


# ------------------------------------------------ importing Deimos's own code
def test_the_wizsprinter_overlay_never_touches_sys_path():
    """`libs/wizsprinter/wizwalker/` has no `__init__.py`, so putting that
    root on sys.path makes `wizwalker` resolvable as a NAMESPACE package
    -- which shadows the real one and turns "wizsprinter is absent" into
    "wizwalker is broken". Extending the already-imported package's
    `__path__` can only ever add a submodule."""
    import sys

    from deimos_bridge import deimos_path

    before = list(sys.path)
    deimos_path.ensure_path()
    added = [p for p in sys.path if p not in before]
    assert all("wizsprinter" not in p for p in added), added


def test_deimos_root_is_put_on_the_path():
    """Deimos's modules import each other as `src.*`."""
    import sys

    from deimos_bridge import deimos_path

    deimos_path.ensure_path()
    assert deimos_path.DEIMOS_ROOT in sys.path


def test_a_missing_requirement_is_named_individually():
    from deimos_bridge import deimos_path

    assert deimos_path.missing_requirement(
        ModuleNotFoundError("no", name="yaml")) == "pyyaml"
    assert deimos_path.missing_requirement(
        ModuleNotFoundError("no", name="thefuzz")) == "thefuzz"
    assert deimos_path.missing_requirement(
        ModuleNotFoundError("no", name="lark.lexer")) == "lark"
    # wizwalker is not something to pip install your way out of here
    assert deimos_path.missing_requirement(
        ModuleNotFoundError("no", name="wizwalker.extensions")) is None


def test_no_advice_anywhere_tells_you_to_pip_install_wizsprinter():
    """It is a workspace member, not a PyPI package. The old message
    headed its install line with it, so the one command anybody would
    copy could not succeed."""
    from deimos_bridge import deimos_path, deimos_questing, scripts

    hint = deimos_path.install_hint(
        ModuleNotFoundError("no", name="wizwalker.extensions.wizsprinter"))
    assert "pip install wizsprinter" not in hint
    assert "vendored" in hint

    for mod in (deimos_questing, scripts):
        ok, reason = mod.available()
        assert "pip install wizsprinter" not in reason


# -------------------------------------------------------- quest-only dialogue
class _Pos:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Body:
    def __init__(self, pos):
        self._pos = pos

    async def position(self):
        return self._pos


class _QuestPos:
    def __init__(self, pos):
        self._pos = pos

    async def position(self):
        if self._pos is None:
            raise RuntimeError("quest hook not written")
        return self._pos


class _DialogueClient:
    def __init__(self, here, quest):
        self.body = _Body(here)
        self.quest_position = _QuestPos(quest)


def test_at_quest_marker_is_true_only_near_the_objective():
    import asyncio

    from deimos_bridge import questing

    near = asyncio.run(questing.at_quest_marker(
        _DialogueClient(_Pos(100, 100), _Pos(150, 130))))
    assert near[0] is True

    far = asyncio.run(questing.at_quest_marker(
        _DialogueClient(_Pos(100, 100), _Pos(9000, 9000))))
    assert far[0] is False
    assert "not at the quest marker" in far[1]


def test_height_alone_does_not_disqualify_a_quest_npc():
    """Z is height. A quest NPC one storey up a ramp is still the quest
    NPC, and counting it would refuse the unambiguous cases."""
    import asyncio

    from deimos_bridge import questing

    ok, _ = asyncio.run(questing.at_quest_marker(
        _DialogueClient(_Pos(100, 100, 0), _Pos(120, 120, 4000))))
    assert ok is True


def test_an_unreadable_quest_position_says_why():
    """With the in-game quest arrow off the position never reads, so
    nothing is ever at the marker and auto-dialogue would silently do
    nothing. It has to be reportable."""
    import asyncio

    from deimos_bridge import questing

    ok, why = asyncio.run(questing.at_quest_marker(
        _DialogueClient(_Pos(0, 0), None)))
    assert ok is False
    assert "quest arrow" in why


def test_auto_dialogue_ignores_an_npc_that_is_not_the_quest(monkeypatch):
    """The reported annoyance: it talked to everyone walked past. The
    game shows its press-X prompt for every interactable in range."""
    import asyncio

    from deimos_bridge import questing

    pressed = []

    async def _no(_c):
        return False

    async def _yes(_c):
        return True

    monkeypatch.setattr(questing, "in_dialogue", _no)
    monkeypatch.setattr(questing, "near_interactable", _yes)
    monkeypatch.setattr(questing, "press_x",
                        lambda c: pressed.append(1) or _yes(c))

    far = _DialogueClient(_Pos(0, 0), _Pos(9000, 9000))
    assert asyncio.run(questing.open_dialogue_if_near(far)) is False
    assert pressed == []

    near = _DialogueClient(_Pos(0, 0), _Pos(50, 50))
    assert asyncio.run(questing.open_dialogue_if_near(near)) is True
    assert pressed == [1]


# ------------------------------------------------------------- the visuals
def test_a_lookahead_decision_records_what_it_weighed():
    """The decision matrix needs the losers, not just the winner. Those
    scores were being computed and thrown away, so a log could say what
    the policy played but never how close the call was."""
    import random

    from data_full import load_spells_full
    from w101_sim import Actor, Boss, Sim, State

    from deimos_bridge.policies import greedy_ttk

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3
    sim = Sim(cards, deck, "ice", Boss(name="b", hp=600, school="fire", dmg=40),
              rng=random.Random(2), player_hp=800)
    p = Actor(name="W", school="ice", hp=800, max_hp=800, team=0, norm_pips=4)
    p.hand = [cards[n] for n in ("Ice Trap", "Snow Serpent", "Frost Beetle")]
    p.deck = [cards[n] for n in deck]
    s = State(p, [Actor(name="b", school="fire", hp=600, max_hp=600, team=1)])

    policy = greedy_ttk()
    policy(sim, s)
    cands = policy.last_candidates
    assert len(cands) >= 4                       # three cards plus pass
    assert sum(1 for c in cands if c.chosen) == 1
    assert any(c.card == "pass" for c in cands)
    assert all(c.turns > 0 for c in cands)


def test_candidates_reach_the_telemetry_through_the_backend():
    from deimos_bridge.telemetry import Telemetry as T

    tel = T()
    from deimos_bridge.policies import Candidate
    d = _Decision(card_name="Sunbird")
    d.candidates = [Candidate(card="Sunbird", target=0, turns=3.0, chosen=True),
                    Candidate(card="pass", turns=5.0)]
    tel.observe(d, _read(2000, 1, hand=("Sunbird",)))
    assert len(tel.rounds[-1].candidates) == 2


def test_the_matrix_keeps_the_target_in_the_move_identity():
    """On a two-mob board the same card aimed at each enemy scores
    differently. Collapsing them into one column would average away the
    entire targeting decision."""
    from deimos_bridge.policies import Candidate
    from deimos_bridge.telemetry import Telemetry as T

    tel = T()
    d = _Decision(card_name="Ice Trap")
    d.candidates = [Candidate(card="Ice Trap", target=0, turns=6.0),
                    Candidate(card="Ice Trap", target=1, turns=4.0,
                              chosen=True)]
    tel.observe(d, _read(2000, 1, hand=("Ice Trap",)))
    _rows, cols, _dropped = tel.decision_matrix()
    assert set(cols) == {"Ice Trap → 0", "Ice Trap → 1"}


def test_the_matrix_reports_what_it_left_out():
    """A grid that quietly dropped six moves reads as 'these were all the
    options', which is a lie."""
    from deimos_bridge.policies import Candidate
    from deimos_bridge.telemetry import Telemetry as T

    tel = T()
    d = _Decision(card_name="c0")
    d.candidates = [Candidate(card=f"c{i}", turns=float(i))
                    for i in range(T.MATRIX_COLUMNS + 4)]
    tel.observe(d, _read(2000, 1, hand=("c0",)))
    _rows, cols, dropped = tel.decision_matrix()
    assert len(cols) == T.MATRIX_COLUMNS
    assert dropped == 4


def test_a_q_table_decision_contributes_no_candidate_row():
    """A tabular lookup produces no comparison. Inventing one would
    misrepresent how that decision was made."""
    from deimos_bridge.policies import TrainedPolicy

    class _Agent:
        Q = {("k", "a"): 1.0}

        class feat:
            @staticmethod
            def key(sim, s):
                return "k"

            @staticmethod
            def legal(sim, s):
                return ["a"]

        @staticmethod
        def policy():
            return lambda sim, s: None

    fallback = lambda sim, s: None
    fallback.last_candidates = ["something"]
    wrapped = TrainedPolicy(_Agent(), fallback=fallback)
    wrapped(None, None)
    assert wrapped.last_source == "Q table"
    assert wrapped.last_candidates == []


def test_axis_ticks_land_on_clean_numbers():
    """Axis labels carry the values that are not directly labelled, so
    0 / 50 / 100 rather than 0 / 47.3 / 94.6."""
    from deimos_bridge.gui.charts import nice_ticks

    assert nice_ticks(0, 95) == [0, 50, 100]
    assert nice_ticks(0, 9.5) == [0, 5, 10]
    assert nice_ticks(0, 640)[0] == 0
    assert all(t == round(t, 6) for t in nice_ticks(0, 1234))


def test_ranked_bars_plot_the_gap_to_the_best_when_lower_wins(qapp):
    """Turns-to-clear all land within a turn or two, so zero-based bars
    are the same length and show nothing. The gap to the winner is the
    quantity the panel exists to display."""
    from deimos_bridge.gui.charts import RankedBars

    bars = RankedBars(lower_is_better=True)
    bars.set_bars([("a", 4.6, True, ""), ("b", 5.2, False, ""),
                   ("c", 7.1, False, "")])
    drawn, printed = bars._plotted()
    assert drawn == [0.0, pytest.approx(0.6), pytest.approx(2.5)]
    assert printed == [4.6, 5.2, 7.1]


def test_ranked_bars_plot_raw_values_when_more_is_better(qapp):
    from deimos_bridge.gui.charts import RankedBars

    bars = RankedBars()
    bars.set_bars([("a", 4, True, ""), ("b", 2, False, "")])
    drawn, printed = bars._plotted()
    assert drawn == printed == [4, 2]


def test_the_sequential_ramp_is_one_hue_and_monotone(qapp):
    """A hue rotation would be a rainbow ramp, which is the classic way
    to make magnitude unreadable."""
    from deimos_bridge.gui.charts import ramp_color

    shades = [ramp_color(t / 10) for t in range(11)]
    hues = {s.hue() for s in shades}
    assert max(hues) - min(hues) <= 8            # one hue
    lightness = [s.lightness() for s in shades]
    assert lightness == sorted(lightness, reverse=True)


def test_every_chart_renders_empty_and_populated(qapp):
    """The empty state is the most common thing a person sees, and a
    blank rectangle reads as broken."""
    from deimos_bridge.gui.charts import (Heatmap, LineChart, Meter,
                                          RankedBars, Scatter)

    line = LineChart("t")
    line.resize(360, 190)
    line.grab()                                   # empty
    line.set_points([(0, 1), (100, 4)])
    line.grab()

    sc = Scatter("t")
    sc.resize(360, 230)
    sc.grab()
    sc.set_points([(100.0, 90.0, True, "a"), (200.0, 260.0, False, "b")])
    sc.grab()
    assert "error" in sc.hit_text(0)

    hm = Heatmap("t")
    hm.resize(360, 200)
    hm.grab()
    hm.set_matrix([("r1", {"a": (3.0, True, "note")})], ["a"])
    hm.grab()
    assert "chosen" in hm.hit_text(0)

    rb = RankedBars("t")
    rb.resize(360, 200)
    rb.grab()
    rb.set_bars([("a", 2.0, True, "note")])
    rb.grab()

    m = Meter("cover")
    m.resize(300, 74)
    m.set_value(0.5, "3 of 6", "good")
    m.grab()


def test_the_learning_tab_survives_an_empty_run(qapp):
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.learning.refresh()
    assert win.learning.coverage.value == 0.0


def test_a_training_snapshot_lands_on_the_curve(qapp):
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.on_snapshot(2000, 0.42, 7.5)
    win.on_snapshot(4000, 0.61, 6.1)
    assert win.tel.training_curve() == [(2000, pytest.approx(42.0)),
                                        (4000, pytest.approx(61.0))]
    assert win.tel.ttk_curve() == [(2000, 7.5), (4000, 6.1)]


# --------------------------------------------------------------- it has to fit
def test_every_tab_scrolls(qapp):
    """A panel that stacks a chart, a second chart and a table is taller
    than a laptop window, and Qt's answer to 'does not fit' is to squeeze
    every child until none is readable."""
    from PyQt6.QtWidgets import QScrollArea

    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    assert win.tabs.count() == 6
    for i in range(win.tabs.count()):
        assert isinstance(win.tabs.widget(i), QScrollArea), i
        assert win.tabs.widget(i).widgetResizable()


def test_the_window_fits_a_laptop(qapp):
    """It could not go narrower than 1577px, which does not fit a
    1366-wide screen at all. Ten controls in one non-wrapping row were
    setting the floor."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.show()
    hint = win.minimumSizeHint()
    assert hint.width() <= 1280, hint.width()
    assert hint.height() <= 700, hint.height()


def test_a_short_window_folds_the_options_away(qapp):
    """Five rows of set-once controls is ~250px; on a 520px-tall window
    that is half the screen spent on things nobody is looking at."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.show()
    win.resize(1180, 800)
    qapp.processEvents()
    assert win.more_btn.isChecked()

    win.resize(1000, 560)
    qapp.processEvents()
    assert not win.more_btn.isChecked()
    assert not win.more.isVisible()


def test_folding_by_hand_is_not_undone_by_a_resize(qapp):
    """Adaptive layout that reverses what someone just did is worse than
    no adaptive layout."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.show()
    win.resize(1000, 560)
    qapp.processEvents()
    win.more_btn.setChecked(True)          # a deliberate choice
    qapp.processEvents()

    win.resize(1000, 500)                  # still short
    qapp.processEvents()
    assert win.more_btn.isChecked()


def test_the_readout_stays_visible_when_the_options_fold(qapp):
    """Coverage and gear answer 'is this working right now', so they are
    not part of what folds away."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.show()
    win.more_btn.setChecked(False)
    qapp.processEvents()
    assert win.policy_state.isVisible()
    assert win.start_btn.isVisible()        # and so is Play live


def test_description_labels_wrap(qapp):
    """A non-wrapping QLabel reports its whole sentence as its minimum
    width, so one paragraph sets a floor on the entire panel -- and in a
    scroll area that floor becomes a horizontal scrollbar under charts
    that would otherwise have fitted."""
    from deimos_bridge.gui.panels import _label

    assert _label("some long explanatory sentence").wordWrap()


def test_a_panel_can_shrink_below_its_content(qapp):
    """Otherwise the scroll area has nothing to scroll and clips
    instead."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.show()
    win.resize(900, 500)
    qapp.processEvents()
    area = win.tabs.widget(1)               # Decisions
    assert area.height() < win.decisions.sizeHint().height()
