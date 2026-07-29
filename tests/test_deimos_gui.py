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

    assert callable(worker("school-aware")._build_policy({}))
    assert callable(worker("blade-stack(3)")._build_policy({}))
    assert callable(worker("blade-stack(2)")._build_policy({}))
    assert callable(worker("nuke-asap")._build_policy({}))


def test_live_worker_refuses_a_trained_policy_with_no_agent(qapp):
    """Silently falling back to a heuristic would report a heuristic's
    numbers under the trained policy's name."""
    from deimos_bridge.gui.live import LiveWorker
    w = LiveWorker(Telemetry(), "ice", ["Frost Beetle"], "trained (Q)", 1)
    with pytest.raises(RuntimeError, match="No trained policy"):
        w._build_policy({})


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
    """Off Windows wizsprinter cannot import, and the fallback has to say
    so with the install line rather than failing silently."""
    from deimos_bridge import deimos_questing

    ok, reason = deimos_questing.available()
    if ok:
        pytest.skip("Deimos's questing is importable here")
    assert "pip install" in reason
    assert "wizsprinter" in reason
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
    return getattr(action, "name", action) if action is not None else "PASS"


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
