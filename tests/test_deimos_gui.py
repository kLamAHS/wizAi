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
    casting fewer nukes.

    It is not settled at all now, rather than settled to zero. A card
    predicted at exactly zero moves no health, so whatever the board
    lost that round belongs to somebody else — live, a Fire Trap and
    the Frost Beetle that fired into the same mob both recorded 56 and
    the fight counted those 56 twice."""
    tel = Telemetry()
    tel.start_fight()
    blade = tel.observe(_Decision("Fireblade"), _read(2000, 1))
    blade.predicted_damage = 0.0
    nuke = tel.observe(_Decision("Sunbird"), _read(2000, 2))
    nuke.predicted_damage = 500.0
    tel.observe(_Decision("Sunbird"), _read(1600, 3))

    assert blade.actual_damage is None, "a buff was credited with damage"
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
    clicks, why = asyncio.run(advance_dialogue(client, settle=0))
    assert clicks == 2
    assert why == ""
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
    assert asyncio.run(advance_dialogue(client, max_clicks=5, settle=0)) \
        == (5, "")


def test_a_failed_dialogue_click_is_not_reported_as_no_dialogue():
    """Zero clicks had two causes and one story.

    A click that failed -- the window moved, another program is over the
    game -- returned the same 0 as "there was no dialogue", so the status
    bar said "no dialogue open" at a wizard looking at an open one.
    """
    import asyncio

    from deimos_bridge.questing import advance_dialogue

    class _Broken(_Mouse):
        async def click_window(self, window):
            raise RuntimeError("window moved")

    root, _ = _dialogue_root()
    client = _QuestClient(root)
    client.mouse_handler = _Broken()
    clicks, why = asyncio.run(advance_dialogue(client, settle=0))
    assert clicks == 0
    assert "click failed" in why and "RuntimeError" in why

    # ...and the genuinely-empty case still says nothing, so the caller
    # can keep its own wording for it.
    quiet, _ = _dialogue_root(visible=False)
    assert asyncio.run(advance_dialogue(_QuestClient(quiet), settle=0)) \
        == (0, "")


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
    ok, why = asyncio.run(questing.press_x(client))
    assert ok is False
    assert "keycode for X" in why      # and it says so rather than shrugging
    assert client.keys == []


def test_a_failed_press_x_says_so(monkeypatch):
    """Auto-quest that teleports correctly and never interacts looks
    exactly like auto-quest that is working, unless this speaks up."""
    import asyncio

    from deimos_bridge import questing

    class _NoKeys(_QuestClient):
        async def send_key(self, key, seconds):
            raise RuntimeError("send_key is gone")

    monkeypatch.setattr(questing, "keycode_x", lambda: "X")
    root, _ = _dialogue_root(visible=False)
    ok, why = asyncio.run(questing.press_x(_NoKeys(root)))
    assert ok is False
    assert "RuntimeError" in why and "sigils" in why


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
    finished. The service tasks are concurrent, so it acts in about a
    second.

    Both tasks are started because they are what a run starts: the queue
    has its own task and the service tick no longer drains it. Two
    drainers meant both could pop an action, and the second pop
    overwrote `seat.busy` -- so the dedupe stopped covering whichever
    action was actually running.
    """
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root, in_battle=False)

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    worker.auto_dialogue = False
    worker.seats[0].client = client
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()

    async def drive():
        tasks = [asyncio.ensure_future(worker._service_loop(client)),
                 asyncio.ensure_future(worker._request_loop(client))]
        worker.request("teleport")
        for _ in range(40):                 # ~2s of 50ms ticks
            await asyncio.sleep(0.05)
            if client.teleported:
                break
        worker._stop = True
        for task in tasks:
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


def test_a_broken_stage_does_not_take_the_others_off_the_air(qapp):
    """One `except` around the whole service tick meant a broken mouse
    hook killed auto-dialogue, the script runner and auto-quest at once,
    forever, without a word on screen."""
    import asyncio

    from deimos_bridge import questing
    from deimos_bridge.gui.live import LiveWorker

    root, _ = _dialogue_root(visible=True)
    client = _QuestClient(root, in_battle=False)

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                        auto_dialogue=True, auto_quest=True)
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()

    stepped = []
    worker._quest_step = lambda c: _tick(stepped)

    async def _boom(_client, **kw):
        raise RuntimeError("the mouse hook is not active")

    async def drive():
        task = asyncio.ensure_future(worker._service_loop(client))
        for _ in range(40):
            await asyncio.sleep(0.05)
            if stepped:
                break
        worker._stop = True
        task.cancel()
        try:
            await task
        except BaseException:
            pass

    # auto-dialogue runs above the quest step in the tick, so breaking it
    # is the case that used to take everything below it off the air.
    real = questing.advance_dialogue
    questing.advance_dialogue = _boom
    try:
        asyncio.run(drive())
    finally:
        questing.advance_dialogue = real

    assert stepped, "the quest step never ran below the broken stage"
    assert any("auto-dialogue failed" in m for m in said), said


async def _tick(seen):
    seen.append(1)


def test_a_dropped_request_is_not_reported_as_happening(qapp):
    """A hotkey held down, or a second press during a multi-second wisp
    sweep, must not read as another sweep that never happened."""
    from deimos_bridge.gui.live import LiveWorker

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    assert w.request("wisps") is True
    assert w.request("wisps") is False        # already queued

    w._requests.clear()
    w._busy = "wisps"
    assert w.request("wisps") is False        # already running
    assert w.request("potion") is True


def test_the_request_drain_stops_when_a_duel_starts(qapp):
    """A wisp sweep runs for seconds. A duel that starts partway through
    would leave this teleporting while the handler clicks cards."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root, in_battle=False)

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    worker.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    done = []

    async def _slow(_client, action):
        done.append(action)
        client._in_battle = True              # a duel starts mid-sweep

    worker._upkeep_now = _slow
    worker._requests[:] = ["wisps", "potion"]
    asyncio.run(worker._drain_requests(client))
    assert done == ["wisps"]
    assert worker._requests == ["potion"], "the rest must wait, not be lost"


def test_upkeep_and_questing_do_not_drive_the_client_at_once(qapp):
    """A quest-marker teleport landing between two wisp teleports moves
    the wizard off the field while the sweep keeps counting."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root, in_battle=False)

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                        auto_quest=True)
    worker.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    hopped = []
    worker._quest_step = lambda c: _tick(hopped)
    worker._in_upkeep = True

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
    assert hopped == [], "questing ran during the between-fights chores"


def test_the_manual_wisp_sweep_relays_the_reason_it_found_none(qapp):
    """Five diagnosed reasons collapsed into one invented line, "no safe
    wisps in range", printed at a wizard standing on a pile of them."""
    import asyncio

    from deimos_bridge import upkeep
    from deimos_bridge.gui.live import LiveWorker

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()

    async def _available():
        return True, ""

    async def _wisps(client, on_status=None, **kw):
        on_status("could not read the wisp entities: WispHealth "
                  "(MemoryReadError)")
        return 0

    real_available, real_wisps = upkeep.available, upkeep.collect_wisps
    upkeep.available, upkeep.collect_wisps = _available, _wisps
    try:
        asyncio.run(worker._upkeep_now(object(), "wisps"))
    finally:
        upkeep.available, upkeep.collect_wisps = real_available, real_wisps

    assert any("MemoryReadError" in m for m in said), said
    assert not any("no safe wisps in range" in m for m in said)


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


def test_training_reports_a_count_not_just_a_moving_bar(qapp):
    """Checkpoints are 5,000 episodes apart and each costs a 2,000-fight
    evaluation, so between them there was nothing on screen but an
    indeterminate bar -- which reads the same whether the run is working
    or hung."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.train_progress.setVisible(True)

    # The warm-start solve runs before episode 1 and has nothing to
    # count, so it stays indeterminate and says what it is doing.
    win.on_stage("solving the warm start")
    assert win.train_progress.maximum() == 0
    assert "warm start" in win.train_progress.format()

    win.on_stage("training")
    win.on_tick(2500, 100_000, 1240.0)
    text = win.train_progress.format()
    assert win.train_progress.value() == 2500
    assert win.train_progress.maximum() == 100_000
    assert "2,500 / 100,000 episodes" in text
    assert "2%" in text
    assert "21 min left" in text

    # A checkpoint's numbers ride along, so a run left on another tab
    # still says whether it is going anywhere.
    win.on_snapshot(5000, 0.42, 6.13)
    win.on_tick(5000, 100_000, 1180.0)
    assert "kill 42%" in win.train_progress.format()

    # An undefined turns-to-kill is a checkpoint that won nothing, not a
    # missing number, and must never reach a format that wants an int.
    win.on_snapshot(10_000, 0.0, float("nan"))
    win.on_tick(10_500, 100_000, 900.0)
    assert "won nothing" in win.train_progress.format()

    # No estimate once it is finished; "0s left" is noise.
    win.on_tick(100_000, 100_000, 0.0)
    assert "left" not in win.train_progress.format()
    assert "100%" in win.train_progress.format()


def test_the_tick_reaches_the_end_on_any_episode_count():
    """The loop's ticks land on multiples of the interval, so a count
    that is not one would leave the bar short of the end forever."""
    from rl_agent import train_agent

    ticks = []
    calls = {"episodes": 0}

    class _Agent:
        Q = {}

        def train_episode(self, *a, **kw):
            calls["episodes"] += 1

        def policy(self):
            return lambda sim, s: None

        alpha = 0.0

    import rl_agent
    real_agent, real_eval = rl_agent.QAgent, rl_agent.evaluate
    real_sim = rl_agent.Sim
    rl_agent.QAgent = lambda *a, **kw: _Agent()
    rl_agent.Sim = lambda *a, **kw: type("S", (), {"boss": None,
                                                   "extra_bosses": []})()
    rl_agent.evaluate = lambda *a, **kw: (0.0, float("nan"))
    try:
        train_agent({}, [], "ice", None, episodes=1_003, warm=False,
                    snap_every=10_000, on_tick=lambda d, t: ticks.append(d))
    finally:
        rl_agent.QAgent, rl_agent.evaluate = real_agent, real_eval
        rl_agent.Sim = real_sim

    assert calls["episodes"] == 1_003
    assert ticks[-1] == 1_003, ticks[-3:]
    assert 100 <= len(ticks) <= 300, len(ticks)   # ~200 over any run


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
    said = []
    w.status = type("S", (), {"emit": staticmethod(said.append)})()
    asyncio.run(w._read_max_hp(_Client()))       # must not raise

    # ...but it is not silent. The symptom of an unread maximum is "the Q
    # table decided 0% of the boards it was shown", whose obvious fix --
    # train for longer -- cannot help, so the real cause has to be said.
    assert any("max health" in m for m in said), said
    assert w.hp_known is False


def test_an_unread_max_health_is_offered_as_the_coverage_cause(qapp):
    """The mob count and mob HP can both be perfectly in range while the
    wizard's own health bucket matches nothing, and then the advice was
    'raise episodes and retrain', which is the one fix that cannot help."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.generalize.setChecked(True)

    class _Live:
        hp_known = False

    win.live = _Live()
    assert "max health" in win._why_coverage_is_low()

    _Live.hp_known = True
    assert "max health" not in win._why_coverage_is_low()


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
    keys = list(bindings.values())
    assert len(keys) == len(set(keys)), keys      # no key bound twice
    assert bindings["teleport"] == "F4"           # first claim wins
    assert "dialogue" not in bindings             # second is dropped
    assert "bound twice" in win.status.text()


def test_hotkey_choices_are_all_real_keycodes():
    """Every key the dropdown offers has to resolve, or picking it fails
    at run time with the game already open.

    Read out of wizwalker's source rather than by importing it. The
    import needs Windows, so this test skipped everywhere it was run --
    which is how `NUMPAD0`..`NUMPAD3` sat in the dropdown for four
    choices that could never bind (`Keycode` spells them
    `Numeric_pad_0`).
    """
    import ast
    import pathlib

    from deimos_bridge import hotkeys

    source = pathlib.Path(__file__).resolve().parents[1] / (
        "Deimos/libs/wizwalker/wizwalker/constants.py")
    if not source.exists():
        pytest.skip("the vendored wizwalker is not checked out")
    tree = ast.parse(source.read_text(encoding="utf-8", errors="replace"))
    members = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Keycode":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    members.update(t.id for t in stmt.targets
                                   if isinstance(t, ast.Name))
    assert members, "could not read Keycode out of wizwalker"
    for name in hotkeys.KEY_CHOICES:
        assert name in members or name.upper() in members, name


def test_a_hotkey_that_will_not_install_says_what_went_wrong(qapp):
    """Every failure was blamed on "another program has it", so a
    wizwalker API mismatch sent the user cycling through all fourteen
    keys, none of which was ever the problem."""
    import asyncio
    import sys

    from deimos_bridge import hotkeys

    class _Listener:
        async def add_hotkey(self, key, cb, **kw):
            raise TypeError("object NoneType can't be used in 'await'")

        def start(self):
            pass

    class _WW:
        HotkeyListener = _Listener

        class Keycode:
            F1 = 1

    said = []
    hk = hotkeys.Hotkeys({"teleport": "F1"}, lambda a: None,
                         on_status=said.append)
    real = sys.modules.get("wizwalker")
    sys.modules["wizwalker"] = _WW
    try:
        assert asyncio.run(hk.start()) == {}
    finally:
        if real is None:
            del sys.modules["wizwalker"]
        else:
            sys.modules["wizwalker"] = real

    assert any("TypeError" in m for m in said), said
    assert not any("another program" in m for m in said)


def test_a_collision_names_every_action_it_unbound(qapp):
    """With four actions on four default keys, retargeting one onto
    another's default drops an action the user never touched."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.hotkey_boxes["teleport"].setCurrentText("F3")     # wisps' default
    bindings = win.hotkey_bindings()
    assert bindings["teleport"] == "F3"
    assert "wisps" not in bindings
    assert "wisps" in win.status.text(), win.status.text()


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
def _covers(agent, mob_hps, n=12, min_visits=1):
    """Coverage of a trained agent on a board of these mobs.

    `min_visits=1` because this measures whether the table has ANY
    experience of a board, which is what domain randomisation is for.
    The live default is `TrainedPolicy.MIN_VISITS` (20), a much stricter
    bar that a 1,500-episode run in a test would never clear -- and
    which is about whether an entry is an estimate rather than whether
    it exists.
    """
    import random

    from data_full import load_spells_full
    from w101_sim import Boss, Sim

    from deimos_bridge.policies import trained_policy

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3
    wrapped = trained_policy(agent, min_visits=min_visits)
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

    async def _pressed(_c):
        return True, ""

    monkeypatch.setattr(questing, "in_dialogue", _no)
    monkeypatch.setattr(questing, "near_interactable", _yes)
    monkeypatch.setattr(questing, "press_x",
                        lambda c: pressed.append(1) or _pressed(c))

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
    # Board, Decisions, Damage model, Learning, Naming, Runs, Party.
    assert win.tabs.count() == 7
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


# ------------------------------------------------- crashes must not be silent
def test_a_chart_that_cannot_draw_says_so_instead_of_aborting(qapp):
    """PyQt6 does not print an unhandled exception raised inside a Qt
    virtual -- it calls qFatal and the process aborts. So one chart
    meeting an unanticipated data shape would kill a live fight mid-duel,
    which is the one thing this window must not do."""
    from deimos_bridge.gui.charts import Chart

    class _Broken(Chart):
        def has_data(self):
            return True

        def paint_data(self, p, r):
            raise ValueError("boom")

    seen = []
    original = Chart._paint_failed
    Chart._paint_failed = lambda self, p, exc: (seen.append(exc),
                                                original(self, p, exc))
    try:
        c = _Broken("t")
        c.resize(300, 150)
        c.grab()                                  # must not abort
    finally:
        Chart._paint_failed = original
    assert seen and isinstance(seen[0], ValueError)


def test_a_heatmap_row_with_no_cells_is_not_data(qapp):
    """`min()` over an empty sequence, inside paintEvent -- the exact
    shape that aborted. A round whose every candidate fell outside the
    shown columns produces one."""
    from deimos_bridge.gui.charts import Heatmap

    h = Heatmap("t")
    h.set_matrix([("r1", {})], ["a"])
    assert not h.has_data()
    h.resize(300, 150)
    h.grab()


def test_a_painted_chart_closes_its_painter(qapp):
    """`paintEvent` left its QPainter for the garbage collector, and a
    QPainter still open on a widget that is then destroyed segfaults the
    interpreter -- no traceback, no qFatal line, nothing to read
    afterwards. Same outcome the guarded paintEvent exists to prevent,
    arriving by the one route the `except` cannot cover.

    Asserted on the source rather than by crashing: the failure mode is
    a segfault during a later garbage collection, which no assertion
    survives to report.
    """
    import inspect

    from deimos_bridge.gui.charts import Chart

    body = inspect.getsource(Chart.paintEvent)
    code = "\n".join(line.split("#")[0] for line in body.splitlines())
    assert "finally:" in code and "p.end()" in code


def test_hovering_a_broken_chart_does_not_abort(qapp):
    """A hover is the last thing that should be able to end a fight."""
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    from deimos_bridge.gui.charts import RankedBars

    bars = RankedBars("t")
    bars.set_bars([("a", 1.0, True, "n")])
    bars.resize(300, 150)
    bars.hits = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    event = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(10, 10),
                        QPointF(10, 10), Qt.MouseButton.NoButton,
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    bars.mouseMoveEvent(event)                    # must not raise


def test_the_resize_hook_survives_running_before_its_widgets_exist(qapp):
    """Qt can deliver a resize before `_build_config` has run, and an
    AttributeError in a virtual aborts rather than prints."""
    from PyQt6.QtCore import QSize
    from PyQt6.QtGui import QResizeEvent

    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    event = QResizeEvent(QSize(900, 500), QSize(800, 400))

    # exactly the half-built state Qt can deliver into
    button, win.more_btn = win.more_btn, None
    del win.more_btn
    win.resizeEvent(event)                        # must not raise
    win.more_btn = button

    # and a failure inside the hook is survived too
    win.more_btn = object()                       # has no isChecked()
    win.resizeEvent(event)
    win.more_btn = button


def test_the_crash_log_captures_an_unhandled_error(tmp_path, monkeypatch):
    """Qt loses errors well: from a desktop shortcut a fatal is a window
    vanishing with nothing written down, which makes 'it crashes' a
    report nobody can act on."""
    import sys

    from deimos_bridge.gui import crashlog

    path = tmp_path / "wizAi-crash.log"
    monkeypatch.setattr(crashlog, "log_path", lambda: str(path))
    shown = []
    previous = sys.excepthook
    try:
        crashlog.install(show=shown.append)
        try:
            raise RuntimeError("probe")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())
    finally:
        sys.excepthook = previous

    assert shown and "probe" in shown[0]
    assert path.exists() and "RuntimeError: probe" in path.read_text()


def test_logging_a_crash_never_becomes_the_crash(tmp_path, monkeypatch):
    from deimos_bridge.gui import crashlog

    monkeypatch.setattr(crashlog, "log_path",
                        lambda: str(tmp_path / "nope" / "deep" / "x.log"))
    crashlog._write("anything")                   # must not raise


# ------------------------------------------------ non-finite numbers in charts
def test_an_undefined_turns_to_kill_does_not_break_the_chart(qapp):
    """The reported crash. `w101_sim.evaluate` returns float('nan') for
    mean turns-to-kill when a checkpoint won no fights, Qt's coordinate
    calls take ints, and `int(nan)` raises ValueError inside paintEvent
    -- which in PyQt6 aborts the process."""
    from deimos_bridge.gui.charts import Chart, LineChart

    seen = []
    original = Chart._paint_failed
    Chart._paint_failed = lambda self, p, exc: (seen.append(exc),
                                                original(self, p, exc))
    try:
        c = LineChart("turns to kill")
        c.set_points([(1000, float("nan")), (2000, 6.1),
                      (3000, float("inf"))])
        c.resize(360, 190)
        c.grab()
    finally:
        Chart._paint_failed = original

    assert not seen, seen
    assert c.points == [(2000.0, 6.1)]
    assert c.dropped == 2


def test_a_dropped_checkpoint_is_reported_not_hidden(qapp):
    """A curve with invisible holes in it is worse than one that says
    which samples had no value."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.on_snapshot(1000, 0.0, float("nan"))       # won nothing
    win.on_snapshot(2000, 0.4, 7.0)
    assert win.learning.ttk.dropped == 1
    assert "won no fights" in win.learning.ttk.dropped_note
    assert win.learning.kill.dropped == 0          # kill rate is still real


def test_an_all_nan_curve_reports_the_drops_rather_than_looking_empty(qapp):
    """The exact shape of a run that wins nothing: every checkpoint has
    an undefined turns-to-kill, so every sample drops and the note that
    reports them lives inside paint_data, which the empty state never
    reaches. The Learning tab showed a populated kill-rate chart reading
    0% above a turns-to-kill chart reading "no training run yet"."""
    from deimos_bridge.gui.charts import LineChart

    c = LineChart("turns to kill")
    c.dropped_note = "{n} checkpoint(s) won no fights"
    c.set_points([(2000, float("nan")), (4000, float("nan")),
                  (6000, float("nan"))])
    assert c.points == [] and c.dropped == 3
    assert c.has_data() is False
    assert c.empty_message() == "3 checkpoint(s) won no fights"

    # ...and a chart that genuinely has nothing yet still says so.
    fresh = LineChart("turns to kill")
    assert fresh.empty_message() == fresh.empty_text

    c.resize(360, 190)
    c.grab()                    # and it still paints without aborting


def test_a_round_whose_board_would_not_read_is_recorded_as_that():
    """It used to vanish entirely: the Decisions table went round 3,
    round 5, the pass was never counted, and the next board differenced
    against round 3 -- so the missing round's damage was folded into its
    predecessor's residual and charged to the damage model."""
    from deimos_bridge.telemetry import Telemetry

    tel = Telemetry()
    tel.start_fight()
    rec = tel.observe_lost_round(4, "could not read the board "
                                    "(MemoryReadError: ...) — passed")
    assert rec.round == 4 and rec.passing
    assert rec.policy == "board read failed"
    assert tel.rounds[-1] is rec
    assert tel.fights[-1].passes == 1
    assert tel.fights[-1].rounds >= 4
    assert tel._pending is None, "the next board must not settle across it"


def test_a_cast_that_never_went_out_is_not_charged_to_the_damage_model():
    """The record is written before the click. A failed cast left the
    round claiming the card was played, so the next board showed the
    target unchanged and the residual settled at minus the whole
    prediction -- the model's worst miss of the run, marked clean."""
    from deimos_bridge.telemetry import RoundRecord, Telemetry

    tel = Telemetry()
    tel.start_fight()
    rec = RoundRecord(fight=1, round=1, chosen="Sunbird",
                      target_name="Krokopatra")
    rec.predicted_damage = 325.0
    tel.rounds.append(rec)
    tel._pending = rec

    amended = tel.note_failed_cast("the cast of Sunbird did not go "
                                   "through (ClickError: x) — passed instead")
    assert amended is rec
    assert rec.predicted_damage is None
    assert rec.passing is True
    assert rec.clean is False
    assert any("cast failed" in c for c in rec.confounds)
    assert tel.fights[-1].passes == 1


class _Mouseless:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _stub_handler(decide, read, card=None):
    """A `WizAiCombatHandler` with just enough around it to run a round."""
    from deimos_bridge.live_backend import WizAiCombatHandler

    passes = []

    class _Backend:
        cards = {}
        cast_time = 0.3
        last_read = read
        failed = []
        lost = []

        async def decide(self):
            return await decide()

        def report_failed_cast(self, name, exc):
            self.failed.append((name, type(exc).__name__))

        def report_lost_round(self, number, reason):
            self.lost.append((number, reason))

    class _Client:
        mouse_handler = _Mouseless()

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    handler.client = _Client()
    handler.backend = _Backend()
    handler._last_read = None
    handler._read_failures = 0
    handler.pass_button = lambda: _tick(passes)
    handler.round_number = lambda: _value(4)
    handler._pick_card = lambda _read, _name: card
    handler._resolve_target = lambda *a, **kw: _value(None)
    return handler, passes


async def _value(v):
    return v


def test_the_handler_reports_a_failed_cast_rather_than_swallowing_it():
    """The round was recorded as a cast before the click. A swallowed
    failure leaves the record claiming a card was played that was not."""
    import asyncio

    from deimos_bridge.live_backend import PolicyDecision

    class _Card:
        async def cast(self, target, **kw):
            raise RuntimeError("the board moved")

    async def _decide():
        return PolicyDecision(card_name="Sunbird", target_index=0)

    handler, passes = _stub_handler(_decide, read=object(), card=_Card())
    asyncio.run(handler.handle_round())

    assert handler.backend.failed == [("Sunbird", "RuntimeError")]
    assert passes, "it must still pass the round rather than hang"


def test_a_round_lost_to_a_failed_read_goes_through_the_record():
    """`_read_failures` was documented as surfaced at the end of a run
    and never was -- nothing in the package read it, so ten rounds lost
    across a fight were invisible."""
    import asyncio

    async def _decide():
        raise RuntimeError("MemoryReadError")

    handler, passes = _stub_handler(_decide, read=None)
    asyncio.run(handler.handle_round())

    assert handler._read_failures == 1
    assert len(handler.backend.lost) == 1
    number, reason = handler.backend.lost[0]
    assert number == 4                          # off the live round number
    assert "could not read the board" in reason and "MemoryReadError" in reason
    assert passes


def test_a_board_with_unreadable_wards_is_not_a_clean_observation():
    """An empty ward list is the same value for "no shields" and "the
    read failed". The second hands the policy a bare mob, it prices its
    hit against no Tower Shield, and the ~50%-off residual is filed as
    evidence the damage model is wrong."""
    from deimos_bridge.live_backend import PolicyDecision
    from deimos_bridge.telemetry import Telemetry

    tel = Telemetry()
    tel.start_fight()

    class _Read:
        def __init__(self, state):
            self.state = state
            self.round_number = 1
            self.hand_cards = {}
            self.resolver = type("R", (), {"misses": set()})()
            self.hidden = []
            self.hand_visibility = 1.0
            self.unreadable = ["Lost Soul's wards (MemoryReadError: x)"]

    from w101_sim import Actor, State
    me = Actor(name="Wizard", school="ice", hp=3000, max_hp=3000, team=0)
    foe = Actor(name="Lost Soul", school="death", hp=2000, max_hp=2000,
                team=1)
    read = _Read(State(me, [foe]))
    rec = tel.observe(PolicyDecision(card_name="Sunbird", target_index=0),
                      read)
    assert rec.clean is False
    assert any("could not read" in c for c in rec.confounds)


def test_clicking_a_heatmap_row_breaks_out_that_round(qapp):
    """The matrix windows to the last 14 candidate rounds; the detail
    bars indexed the unwindowed list, so past 14 rounds every row was off
    by N-14 and the bars belonged to a different round than the label."""
    from deimos_bridge.telemetry import RoundRecord, Telemetry

    class _C:
        def __init__(self, name):
            self.card, self.turns, self.chosen = name, 2.0, name == "fire"
            self.damage, self.pips = 100, 2

    tel = Telemetry()
    tel.rounds = [RoundRecord(fight=1, round=i,
                              candidates=[_C("fire"), _C("ice")])
                  for i in range(1, 21)]

    rows, _cols, _dropped = tel.decision_matrix()
    assert len(rows) == 14 and rows[-1][0] == "r20"
    assert tel.candidate_bars(len(rows) - 1)[1] == "fight 1, round 20"
    assert tel.candidate_bars(0)[1] == "fight 1, round 7"


def test_every_chart_entry_point_rejects_non_finite(qapp):
    from deimos_bridge.gui.charts import (Heatmap, LineChart, Meter,
                                          RankedBars, Scatter, finite,
                                          nice_ticks)

    nan, inf = float("nan"), float("inf")
    assert finite(nan) == 0.0 and finite(inf) == 0.0
    assert finite(nan, None) is None
    assert finite("x", 3.0) == 3.0
    assert all(t == t for t in nice_ticks(nan, inf))   # no NaN ticks

    s = Scatter("t")
    s.set_points([(nan, 1.0, True, "a"), (5.0, 6.0, True, "b")])
    assert len(s.points) == 1

    b = RankedBars("t")
    b.set_bars([("a", nan, True, ""), ("b", 2.0, False, "")])
    assert len(b.bars) == 1

    h = Heatmap("t")
    h.set_matrix([("r1", {"a": (nan, True, "n"), "b": (2.0, False, "n")})],
                 ["a", "b"])
    assert list(h.rows[0][1]) == ["b"]

    m = Meter("t")
    m.set_value(nan)
    assert m.value == 0.0

    line = LineChart("t")
    line.set_points([(0, nan)])
    assert not line.has_data()


# ------------------------------------------------------- wisps and potions
def test_wisps_and_potions_have_hotkeys_and_buttons(qapp):
    """The automatic versions run only between fights, so there was no
    way to top up during a long questing stretch -- or when the automatic
    path was silently failing."""
    from deimos_bridge.gui.app import MainWindow
    from deimos_bridge.hotkeys import DEFAULTS

    assert "wisps" in DEFAULTS and "potion" in DEFAULTS

    win = MainWindow(Telemetry())
    assert set(win.hotkey_boxes) == {"teleport", "dialogue", "wisps", "potion"}
    bindings = win.hotkey_bindings()
    assert bindings["wisps"] and bindings["potion"]
    assert len(set(bindings.values())) == len(bindings)     # all distinct
    assert win.wisps_btn is not None and win.potion_btn is not None


def test_the_buttons_queue_the_worker_actions(qapp):
    from deimos_bridge.gui.app import MainWindow
    from deimos_bridge.gui.live import LiveWorker

    win = MainWindow(Telemetry())
    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    worker.isRunning = lambda: True
    win.live = worker
    win.on_wisps()
    win.on_potion()
    assert worker._requests == ["wisps", "potion"]


def test_a_held_hotkey_does_not_queue_a_burst(qapp):
    """RegisterHotKey repeats while the key is down, and eight queued
    wisp sweeps would take a minute to work through with the fight
    waiting on them."""
    from deimos_bridge.gui.live import LiveWorker

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    assert w.request("wisps") is True
    assert w.request("wisps") is False
    assert w._requests == ["wisps"]
    assert w.request("nonsense") is False


def test_collect_wisps_says_why_it_found_nothing(monkeypatch):
    """Five separate except blocks each returned without a word, so a
    wizard whose wisps were never collected got no message and no way to
    tell 'there were none' from 'the import failed'."""
    import asyncio

    from deimos_bridge import upkeep

    said = []

    def boom(_client):
        raise ImportError("no wizsprinter")

    monkeypatch.setattr(upkeep, "_sprinty", boom)
    n = asyncio.run(upkeep.collect_wisps(object(), on_status=said.append))
    assert n == 0
    assert said and "unavailable" in said[0] and "wizsprinter" in said[0]


def test_collect_wisps_distinguishes_none_from_broken(monkeypatch):
    import asyncio

    from deimos_bridge import upkeep

    class _Sprinty:
        async def get_base_entities_with_vague_name(self, _name):
            return []

    said = []
    monkeypatch.setattr(upkeep, "_sprinty", lambda c: _Sprinty())
    asyncio.run(upkeep.collect_wisps(object(), on_status=said.append))
    assert said == ["no wisps in this zone"]


def test_a_failed_potion_reports_the_reason(monkeypatch):
    """It returned False for both 'no charges' and 'the helper could not
    import', and the message said 'no charge left' either way."""
    import asyncio

    from deimos_bridge import upkeep

    class _Stats:
        async def potion_charge(self):
            return 3.0

    class _Client:
        stats = _Stats()

    monkeypatch.setattr(upkeep, "_sprinty", lambda c: None)
    ok = asyncio.run(upkeep.drink_potion(_Client()))
    assert ok is False
    assert getattr(upkeep.drink_potion, "last_error", "")


def test_upkeep_resolves_wizsprinter_through_the_shared_path_helper():
    """`src.utils` -- which the potion helper imports -- needs
    wizsprinter, and upkeep used to do its own sys.path insert that
    skipped the overlay entirely. That is why potions did nothing."""
    import inspect

    from deimos_bridge import upkeep

    for fn in (upkeep._sprinty, upkeep.drink_potion):
        # Code only -- the comments explain why the hand-rolled insert is
        # wrong, so matching raw source would match the explanation.
        code = "\n".join(line.split("#")[0]
                         for line in inspect.getsource(fn).splitlines())
        assert "ensure_path" in code, fn.__name__
        assert "sys.path" not in code, fn.__name__


# ------------------------------- upkeep has to wait for the wizard to be free
class _FreeClient:
    """A client that is busy for the first `busy` reads, then free."""

    def __init__(self, busy=0, hud=True, loading=False):
        self.busy = busy
        self._hud = hud
        self._loading = loading
        self.teleported = []
        self.root_window = _Win("root", [
            _Win("WorldView", [
                _Win("windowHUD", [_Win("btnPotions", visible=hud)])])])

    async def is_loading(self):
        return self._loading and self.busy > 0

    async def in_battle(self):
        if self.busy > 0:
            self.busy -= 1
            return not self._loading
        return False

    async def teleport(self, xyz):
        self.teleported.append(xyz)


def test_upkeep_waits_for_the_duel_to_actually_end():
    """`wait_for_combat` returns when duel_phase is `ended`, which is
    before the results screen clears and before the body is released --
    so the chores ran against a wizard the game still owned."""
    import asyncio

    from deimos_bridge import upkeep

    client = _FreeClient(busy=3)
    ok, why = asyncio.run(upkeep.wait_until_free(client, timeout=5.0,
                                                 poll=0.01, settle=0))
    assert ok is True and why == ""
    assert client.busy == 0, "it returned before the duel had ended"


def test_upkeep_skips_the_chores_when_the_duel_never_ends():
    import asyncio

    from deimos_bridge import upkeep

    client = _FreeClient(busy=10_000)
    ok, why = asyncio.run(upkeep.wait_until_free(client, timeout=0.05,
                                                 poll=0.01, settle=0))
    assert ok is False
    assert "still in the duel" in why and "skipped" in why


def test_upkeep_goes_ahead_when_the_hud_cannot_be_read():
    """The light-install case. An unanswerable gate must not become a
    timeout that stops upkeep from ever running."""
    import asyncio

    from deimos_bridge import upkeep

    class _NoWindows:
        root_window = None

    ok, why = asyncio.run(upkeep.wait_until_free(_NoWindows(), timeout=9.0,
                                                 poll=0.01, settle=0))
    assert ok is True and why == ""


def test_upkeep_says_so_when_the_hud_never_returns():
    import asyncio

    from deimos_bridge import upkeep

    client = _FreeClient(hud=False)
    ok, why = asyncio.run(upkeep.wait_until_free(client, timeout=0.05,
                                                 poll=0.01, settle=0))
    assert ok is True, "a missing HUD must not stop the chores outright"
    assert "HUD never came back" in why


def test_after_fight_will_not_touch_a_client_that_is_still_in_the_duel(
        monkeypatch):
    import asyncio

    from deimos_bridge import upkeep

    ran = []

    async def _wisps(client, **kw):
        ran.append("wisps")
        return 1

    monkeypatch.setattr(upkeep, "collect_wisps", _wisps)
    said = []
    asyncio.run(upkeep.after_fight(_FreeClient(busy=10_000), potions=False,
                                   on_status=said.append))
    assert ran == []
    assert said and "skipped" in said[0]


def test_a_wisp_teleport_that_snaps_back_is_not_counted(monkeypatch):
    """`client.teleport` raises nothing when the game undoes it, so
    counting the call reported "collected 3 wisp(s)" for a wizard that
    never moved."""
    import asyncio

    from deimos_bridge import upkeep

    class _Pos3:
        def __init__(self, x, y, z=0.0):
            self.x, self.y, self.z = x, y, z

    class _Wisp:
        def __init__(self, xyz):
            self._xyz = xyz

        async def location(self):
            return self._xyz

    class _Sprinty:
        async def get_base_entities_with_vague_name(self, name):
            return ([_Wisp(_Pos3(500, 500)), _Wisp(_Pos3(600, 600))]
                    if name == "WispHealth" else [])

        async def find_safe_entities_from(self, entities):
            return entities

    class _Body:
        async def position(self):
            return _Pos3(0, 0)          # never moved: the duel circle

    class _Stuck:
        body = _Body()

        async def teleport(self, xyz):
            pass                        # accepted, then undone

    said = []
    monkeypatch.setattr(upkeep, "_sprinty", lambda c: _Sprinty())
    n = asyncio.run(upkeep.collect_wisps(_Stuck(), on_status=said.append))
    assert n == 0
    assert said and "snapped back" in said[0]


def test_an_unsafe_wisp_sweep_says_it_is_unsafe(monkeypatch):
    """`safe_only=True` degrading to False in silence walks the wizard
    into the second fight the checkbox promises to avoid."""
    import asyncio

    from deimos_bridge import upkeep

    class _Sprinty:
        async def get_base_entities_with_vague_name(self, name):
            return ([_Entity("safe"), _Entity("guarded")]
                    if name == "WispHealth" else [])

        async def find_safe_entities_from(self, entities):
            raise RuntimeError("MemoryReadError")

    said = []
    monkeypatch.setattr(upkeep, "_sprinty", lambda c: _Sprinty())
    client = _UpkeepClient()
    n = asyncio.run(upkeep.collect_wisps(client, on_status=said.append))
    assert n == 2 and client.teleported == ["safe", "guarded"]
    assert any("may pull a fight" in m for m in said)


def test_a_failed_health_read_is_not_reported_as_healthy(monkeypatch):
    """A wizard at 12% with three charges sat there all run because one
    of five stat reads raised and False came back either way."""
    import asyncio

    from deimos_bridge import upkeep

    class _Stats:
        async def current_mana(self):
            raise RuntimeError("MemoryReadError")

    class _Client:
        stats = _Stats()

    assert asyncio.run(upkeep.needs_potion(_Client())) is False
    assert "could not check" in upkeep.needs_potion.last_error

    said = []
    asyncio.run(upkeep.after_fight(_Client(), wisps=False, potions=True,
                                   on_status=said.append, wait=False))
    assert said and "could not check" in said[0]


def test_a_failed_charge_read_is_not_reported_as_no_charges():
    """"No charges left" is an assertion about the game. Saying it
    because the read failed sends you to a vendor to buy potions you
    already have."""
    import asyncio

    from deimos_bridge import upkeep

    class _Stats:
        async def potion_charge(self):
            raise RuntimeError("MemoryReadError")

    class _Client:
        stats = _Stats()

    assert asyncio.run(upkeep.drink_potion(_Client())) is False
    why = upkeep.drink_potion.last_error
    assert "could not read your potion charges" in why
    assert "no charges" not in why


def test_the_potion_error_does_not_outlive_the_call_that_made_it():
    """It was a module-level function attribute that nothing cleared, so
    fight 1's ModuleNotFoundError was still being reported at fight 8 as
    the reason an empty bottle had not been drunk."""
    import asyncio

    from deimos_bridge import upkeep

    upkeep.drink_potion.last_error = "ModuleNotFoundError: no thefuzz"

    class _Stats:
        async def potion_charge(self):
            return 0.0

    class _Client:
        stats = _Stats()

    assert asyncio.run(upkeep.drink_potion(_Client())) is False
    assert "thefuzz" not in upkeep.drink_potion.last_error
    assert "empty" in upkeep.drink_potion.last_error


# ------------------------------------ the agent has to be able to use its table
def test_the_greedy_policy_can_cast_on_a_multi_mob_board():
    """The 0%-kill-rate bug. `Featurizer.legal` emits "Snow Serpent@1" on
    a multi-enemy board, but `QAgent.policy` matched `c.name == a` --
    which never matches -- so it returned None and the agent passed every
    single turn. `train_episode` was unaffected because it goes through
    `apply_action`, which has always split the target: the agent learned
    a table it could then not use, and the only symptom was a kill rate
    that never moved."""
    import random

    from data_full import load_spells_full
    from rl_agent import PASS, QAgent
    from w101_sim import Boss, Sim

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3

    for n_mobs in (1, 3):
        sim = Sim(cards, deck, "ice",
                  Boss(name="m0", hp=200, school="ice", dmg=40),
                  enemies=[Boss(name=f"m{i}", hp=150, school="ice", dmg=40)
                           for i in range(1, n_mobs)],
                  rng=random.Random(1), player_hp=800)
        agent = QAgent(cards, deck, "ice", rng=random.Random(2))
        s = sim.new_state()
        want = next(a for a in agent.feat.legal(sim, s) if a != PASS)
        agent.Q[(agent.feat.key(sim, s), want)] = 99.0

        got = agent.policy()(sim, s)
        assert got is not None, f"{n_mobs} mob(s): policy passed"
        card, target = got
        assert card.name == want.split("@")[0]
        assert 0 <= target < max(1, n_mobs)


def test_the_greedy_policy_keeps_the_target_it_chose():
    """Returning the bare card would throw away the target, which is
    most of what the multi-enemy action space exists for."""
    import random

    from data_full import load_spells_full
    from rl_agent import QAgent
    from w101_sim import Boss, Sim

    cards = load_spells_full()
    deck = ["Snow Serpent"] * 6
    sim = Sim(cards, deck, "ice", Boss(name="a", hp=400, school="ice", dmg=0),
              enemies=[Boss(name="b", hp=400, school="ice", dmg=0),
                       Boss(name="c", hp=400, school="ice", dmg=0)],
              rng=random.Random(0), player_hp=800)
    agent = QAgent(cards, deck, "ice", rng=random.Random(1))
    s = sim.new_state()
    s.norm_pips = 6
    key = agent.feat.key(sim, s)
    aimed = [a for a in agent.feat.legal(sim, s) if a.endswith("@2")]
    if not aimed:
        pytest.skip("no aimed action available on this draw")
    agent.Q[(key, aimed[0])] = 99.0
    card, target = agent.policy()(sim, s)
    assert target == 2


def test_a_trained_agent_wins_on_a_multi_mob_board():
    """End to end: with the policy able to cast, training on a winnable
    three-mob board must move off zero. It reported a flat 0% across
    100,000 episodes before."""
    import random

    from data_full import load_spells_full
    from rl_agent import train_agent
    from w101_sim import Boss, Sim, evaluate

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3
    hps, dmg = [135, 108, 81], 68
    boss = Boss(name="m0", hp=hps[0], school="ice", dmg=dmg)
    extra = [Boss(name=f"m{i}", hp=h, school="ice", dmg=dmg)
             for i, h in enumerate(hps[1:], 1)]

    agent, sim = train_agent(cards, deck, "ice", boss, enemies=extra,
                             episodes=1500, player_hp=826,
                             player_stats={"damage": {"ice": 0.09}})
    kill, _ttk = evaluate(sim, agent.policy(), n=300)
    assert kill > 0.15, kill


def test_the_board_the_user_trained_on_is_winnable():
    """Before blaming the agent, the board has to be beatable at all --
    otherwise a 0% kill rate is the honest answer rather than a bug."""
    import random

    from data_full import load_spells_full
    from w101_sim import Boss, Sim, evaluate

    from deimos_bridge.policies import school_aware_blade_stack

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3
    hps, dmg = [135, 108, 81], 68
    sim = Sim(cards, deck, "ice",
              Boss(name="m0", hp=hps[0], school="ice", dmg=dmg),
              enemies=[Boss(name=f"m{i}", hp=h, school="ice", dmg=dmg)
                       for i, h in enumerate(hps[1:], 1)],
              rng=random.Random(7), player_hp=826,
              player_stats={"damage": {"ice": 0.09}, "accuracy": 0.05})
    kill, _ = evaluate(sim, school_aware_blade_stack(3), n=150)
    assert kill > 0.5, kill


def test_a_flat_zero_kill_rate_is_named_as_the_cause(qapp):
    """Every other coverage explanation is a distraction from "training
    never won a fight" -- that table has nothing to apply."""
    from deimos_bridge.gui.app import MainWindow

    tel = Telemetry()
    for ep in (5000, 10000, 100000):
        tel.record_snapshot(ep, 0.0, float("nan"))
    win = MainWindow(tel)
    why = win._why_coverage_is_low()
    assert "never won a fight" in why

    tel.record_snapshot(200000, 0.4, 6.0)          # it did learn something
    assert "never won a fight" not in win._why_coverage_is_low()


# ------------------------------- the training board has to be the real fight
def test_enemy_school_is_read_not_assumed():
    """`"ice"` was hardcoded for every live enemy, and that is not a
    cosmetic default: `Boss.resist_own` is 0.40, so an ice wizard's whole
    deck was being planned against mobs that resisted 40% of it."""
    import asyncio

    from deimos_bridge.deimos_damage import SCHOOL_ID_TO_NAMES
    from deimos_bridge.live_state import read_school

    class _M:
        def __init__(self, sid):
            self._sid = sid

        async def primary_magic_school_id(self):
            return self._sid

    assert asyncio.run(read_school(_M(SCHOOL_ID_TO_NAMES["Death"]))) == "death"
    assert asyncio.run(read_school(_M(SCHOOL_ID_TO_NAMES["Fire"]))) == "fire"
    # Star/Sun/Moon are real ids no mob fights as, and an unreadable
    # school falls back to balance -- neutral both ways, so it costs
    # accuracy rather than inventing a resistance.
    assert asyncio.run(read_school(_M(SCHOOL_ID_TO_NAMES["Sun"]))) == "balance"

    class _Broken:
        async def primary_magic_school_id(self):
            raise RuntimeError("MemoryReadError")

    assert asyncio.run(read_school(_Broken())) == "balance"


def test_training_never_pits_a_wizard_against_its_own_school(qapp):
    """The shipped default. An ice wizard trained against ice mobs is
    the worst matchup in the game, and it made the board unwinnable by
    every policy in the repo at every enemy damage down to zero."""
    from deimos_bridge.gui.app import TrainWorker
    from rl_agent import MOB_SCHOOLS

    w = TrainWorker({}, [], "ice", 500, boss_hp=690, n_enemies=2)
    assert w.school_pool() == list(MOB_SCHOOLS)     # unknown: all seven
    assert w.school_pool() != ["ice"]

    w = TrainWorker({}, [], "ice", 500, boss_hp=690, n_enemies=2,
                    mob_schools=["death", "death"])
    assert w.school_pool() == ["death"]             # observed: the real one
    assert w.board_schools() == ["death", "death"]


def test_the_board_sampler_varies_the_school():
    import random

    from rl_agent import MOB_SCHOOLS, make_board_sampler

    sample = make_board_sampler("balance", (400, 900), max_mobs=2, dmg=60,
                                schools=list(MOB_SCHOOLS))
    rng = random.Random(0)
    seen = set()
    for _ in range(200):
        boss, extra = sample(rng)
        seen.update(b.school for b in [boss] + list(extra))
    assert len(seen) >= 5, seen


def test_training_refuses_a_board_nothing_can_win(qapp):
    """40,000 episodes on an unwinnable board is not a failed run, it is
    an impossible one -- and it draws exactly the same flat 0% line."""
    from deimos_bridge.gui.app import TrainWorker
    from data_full import load_spells_full
    from w101_sim import Boss

    cards = load_spells_full()
    deck = (["Evil Snowman"] * 3 + ["Frost Beetle"] * 3 + ["Ice Trap"] * 2
            + ["Scarab - Starter Wand@item"] + ["Snow Serpent"] * 3)
    w = TrainWorker(cards, deck, "ice", 500, player_hp=857, boss_hp=690,
                    n_enemies=2)

    ice = Boss(name="d", hp=690, school="ice", dmg=71)
    icex = [Boss(name="m", hp=552, school="ice", dmg=71)]
    ok, note = w.preflight(ice, icex, n=80)
    assert ok is False
    assert "cannot be won" in note
    # It names a cause rather than listing knobs. This deck HAS the
    # damage for the board, so the diagnosis is the race, not the deck.
    assert "loses on time" in note

    death = Boss(name="d", hp=690, school="death", dmg=71)
    deathx = [Boss(name="m", hp=552, school="death", dmg=71)]
    assert w.preflight(death, deathx, n=80)[0] is True


def test_enemy_damage_comes_off_the_enemy_when_it_has_been_measured(qapp):
    """`player_hp // 12` is the *wizard's* health over a constant, so the
    death clock is the same at every level and a table trained on it
    cannot learn that more health buys more turns."""
    from deimos_bridge.gui.app import TrainWorker

    guessed = TrainWorker({}, [], "ice", 500, player_hp=857)
    assert guessed.enemy_damage() == 71             # the old behaviour

    measured = TrainWorker({}, [], "ice", 500, player_hp=857, mob_damage=87)
    assert measured.enemy_damage() == 87

    # and the guess still scales with the wizard, which is the defect --
    # pinned here so a change to it is deliberate
    assert TrainWorker({}, [], "ice", 500, player_hp=3000).enemy_damage() == 250


def test_the_run_records_what_training_needs_to_know():
    from deimos_bridge.telemetry import EnemyView, RoundRecord, Telemetry

    tel = Telemetry()
    tel.start_fight()
    tel.rounds.append(RoundRecord(
        fight=1, round=1, incoming=84.0,
        enemies=[EnemyView("Lord Nightshade", 690, 690, school="death"),
                 EnemyView("Field Guard", 255, 255, school="death")]))
    tel.rounds.append(RoundRecord(fight=1, round=2, incoming=90.0))
    assert tel.observed_mob_schools() == ["death", "death"]
    assert tel.observed_incoming() == 87.0          # averaged, not last


def test_a_sentinel_is_not_rendered_as_a_turn_count():
    """Every candidate reading "14 turns", including "pass", is
    `died()` at a horizon of 12 -- "no line survives", which is the
    opposite of the tie it looks like."""
    from deimos_bridge.policies import Candidate
    from deimos_bridge.telemetry import (RoundRecord, Telemetry, outcome_of,
                                         turns_label)

    assert outcome_of(Candidate(card="x", turns=4, horizon=12)) == ""
    assert outcome_of(Candidate(card="x", turns=13, horizon=12)) == "no clear"
    assert outcome_of(Candidate(card="x", turns=14, horizon=12)) == "dies"
    assert outcome_of(Candidate(card="x", turns=15, horizon=12)) == "unplayable"
    assert turns_label(Candidate(card="x", turns=4, horizon=12)) == "4 turns"
    assert turns_label(Candidate(card="x", turns=14, horizon=12)) == "dies"

    tel = Telemetry()
    tel.rounds = [RoundRecord(fight=1, round=1, candidates=[
        Candidate(card="Ice Trap", target=1, turns=14, damage=1126,
                  chosen=True, horizon=12),
        Candidate(card="pass", turns=14, damage=778, horizon=12)])]
    bars, title = tel.candidate_bars(0)
    assert "no line survives" in title
    assert all(b[4] == "dies" for b in bars), bars


def test_a_gridline_outside_the_plot_is_not_drawn(qapp):
    """A flat 0% curve was labelled -2% / 0% / 2%: the 2% printed over
    the subtitle and the -2% fell off the bottom edge. Measured at 23
    pixels of literal overprint at the shipped window size."""
    from deimos_bridge.gui.charts import LineChart

    c = LineChart("kill rate", "% of simulated fights won",
                  fmt=lambda v: f"{v:.0f}%", height=170)
    c.set_points([(e, 0.0) for e in range(5000, 45000, 5000)])
    c.resize(654, 170)

    drawn = []
    real = LineChart.grid_y

    def spy(self, p, r, ticks, fmt=lambda v: f"{v:g}"):
        drawn.extend(v for v, y in ticks if r.top() - 1 <= y <= r.bottom() + 1)
        return real(self, p, r, ticks, fmt)

    LineChart.grid_y = spy
    try:
        c.grab()
    finally:
        LineChart.grid_y = real

    assert drawn, "no gridlines at all"
    assert min(drawn) >= 0.0, f"a kill rate axis went negative: {drawn}"


def test_bar_and_cell_text_gets_a_line_of_height(qapp):
    """Qt clips drawText to its rect, and the rect was the bar's own
    thickness -- 4.55px against a 15px font on the 11-candidate board,
    so every row rendered as a horizontal slice."""
    from PyQt6.QtGui import QFontMetrics

    from deimos_bridge.gui.charts import Heatmap, RankedBars

    bars = [(f"card {i} → 0", 14, i == 3, "n") for i in range(11)]
    b = RankedBars("round detail", "", height=180)
    b.set_bars(bars, unit=" turns")
    fm = QFontMetrics(b.font())
    assert b.minimumHeight() >= fm.height() * 11, b.minimumHeight()

    h = Heatmap("matrix", "", height=250)
    h.set_matrix([(f"r{i}", {"a": (2.0, False, "n")}) for i in range(14)], ["a"])
    assert h.minimumHeight() >= fm.height() * 14, h.minimumHeight()

    # ...and it still paints
    b.resize(700, b.minimumHeight()); b.grab()
    h.resize(700, h.minimumHeight()); h.grab()


def test_the_decisions_table_gives_its_width_to_what_needs_it(qapp):
    """Seven equal columns gave 'fight' 165px for a one-character value
    while 'policy' elided away the reason the column exists for."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.resize(1180, 800)
    table = win.decisions.table
    assert getattr(table, "_weights", None), "no column weights"
    win.decisions.refresh()
    widths = [table.columnWidth(i) for i in range(table.columnCount())]
    assert widths[5] > widths[0] * 2, widths      # 'why' beats 'fight'
    assert widths[2] > widths[1] * 2, widths      # 'policy' beats 'round'


def test_the_window_says_whether_the_table_beats_its_own_fallback(qapp):
    """Coverage is not competence: a table can key 95% of boards and
    play every one of them worse than the heuristic it is keeping out of
    the driver's seat, and every number the Learning tab showed would
    still look healthy."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.on_verdict(0.064, 0.824)
    assert "weaker player" in win.verdict_text
    assert "6%" in win.verdict_text and "82%" in win.verdict_text

    win.on_verdict(0.91, 0.82)
    assert "worth playing" in win.verdict_text


def test_a_miss_says_which_fact_it_did_not_recognise():
    """"It always goes to fallback" is not actionable. "A 1,500 HP mob is
    above the 276-1,242 band this table was trained on" points at a box
    already on screen."""
    from deimos_bridge.policies import trained_policy
    from rl_agent import QAgent
    from w101_sim import Actor, State

    agent = QAgent({}, [], "ice")
    agent.trained_on = {"hp": (276, 1242), "mobs": 2, "schools": ["death"],
                        "player_hp": 857}
    tp = trained_policy(agent)

    def board(hps, schools=("death",)):
        me = Actor(name="W", school="ice", hp=857, max_hp=857, team=0)
        foes = [Actor(name=f"m{i}", school=schools[i % len(schools)],
                      hp=h, max_hp=h, team=1) for i, h in enumerate(hps)]
        return State(me, foes)

    assert "above the 276–1,242 band" in tp.why_missed(board([1500]))
    assert "below the 276–1,242 band" in tp.why_missed(board([100]))
    assert "3 mobs, trained for up to 2" in tp.why_missed(board([500] * 3))
    assert "trained against death, fighting fire" in \
        tp.why_missed(board([690], schools=("fire",)))
    assert tp.why_missed(board([690])) == ""      # in band: nothing to say

    # No stamp, no claim -- inventing a band would be worse than silence.
    bare = trained_policy(QAgent({}, [], "ice"))
    assert bare.why_missed(board([9999])) == ""


# ------------------------------- the band is discovered, not typed in
def test_the_trainable_range_is_discovered_per_mob_count(qapp):
    """"Why do I have to train for specific healths" — you should not.
    The band was `mob HP` x0.4 to x1.8, so typing 235 bought 94–423 and
    a 480 HP mob fell off the end and keyed nothing at all."""
    from deimos_bridge.gui.app import TrainWorker
    from data_full import load_spells_full

    cards = load_spells_full()
    deck = (["Evil Snowman"] * 3 + ["Frost Beetle"] * 3 + ["Ice Trap"] * 3
            + ["Snow Serpent"] * 3)
    w = TrainWorker(cards, deck, "ice", 500, player_hp=1022, boss_hp=235,
                    n_enemies=3, mob_schools=["death"], mob_damage=85)
    bands = w.envelope(n=60)

    assert set(bands) == {1, 2, 3}
    # The whole point: the winnable span is sharply different per count,
    # which a single band cannot express in either direction.
    assert bands[1][1] > bands[2][1] > bands[3][1], bands
    # ...and it reaches far past the x1.8 the box would have given (423).
    assert bands[1][1] > 1000, bands
    assert "training over 1 mob to" in w.describe_envelope(bands)


def test_a_deck_that_can_clear_nothing_reports_an_empty_envelope(qapp):
    from deimos_bridge.gui.app import TrainWorker
    from data_full import load_spells_full

    cards = load_spells_full()
    # A deck with no damage card in it can never kill anything, at any
    # health, so the envelope is genuinely empty rather than small.
    w = TrainWorker(cards, ["Ice Trap"] * 4, "ice", 500, player_hp=1022,
                    n_enemies=1, mob_schools=["death"], mob_damage=85)
    bands = w.envelope(n=40)
    assert bands == {}
    assert "cannot clear" in w.describe_envelope(bands)


def test_the_sampler_honours_a_band_per_mob_count():
    import random

    from rl_agent import make_board_sampler

    sample = make_board_sampler("death", (50, 400), max_mobs=3, dmg=60,
                                bands={1: (50, 1400), 2: (50, 700),
                                       3: (50, 480)})
    rng = random.Random(0)
    seen = {}
    for _ in range(3000):
        boss, extra = sample(rng)
        board = [boss] + list(extra)
        lo, hi = seen.get(len(board), (10 ** 9, 0))
        seen[len(board)] = (min(lo, *[b.hp for b in board]),
                            max(hi, *[b.hp for b in board]))
    assert set(seen) == {1, 2, 3}
    assert seen[1][1] > 1300 and seen[2][1] <= 700 and seen[3][1] <= 480, seen


def test_a_miss_uses_the_band_for_the_count_on_the_board():
    """The winnable span differs by mob count, so "above the band" is a
    different number depending on how many are up."""
    from deimos_bridge.policies import trained_policy
    from rl_agent import QAgent
    from w101_sim import Actor, State

    agent = QAgent({}, [], "ice")
    agent.trained_on = {"hp": (40, 1400), "mobs": 3, "schools": ["death"],
                        "bands": {1: (40, 1400), 2: (40, 700), 3: (40, 480)}}
    tp = trained_policy(agent)

    def board(hps):
        me = Actor(name="W", school="ice", hp=1022, max_hp=1022, team=0)
        foes = [Actor(name=f"m{i}", school="death", hp=h, max_hp=h, team=1)
                for i, h in enumerate(hps)]
        return State(me, foes)

    assert tp.why_missed(board([900])) == ""              # fine at 1 mob
    assert "above the 40–700 band" in tp.why_missed(board([900, 900]))
    assert "above the 40–480 band" in tp.why_missed(board([600] * 3))


# ------------------------------ the panel must not contradict itself
def test_coverage_is_one_number_from_one_source():
    """The config line read it off `TrainedPolicy`'s counters and the
    Learning tab off the round records, so the window showed "decided 0%
    (1 fell back)" beside "14%, 1 of 7 rounds" — the policy object's
    counters reset on every swap and the records do not."""
    from deimos_bridge.telemetry import RoundRecord, Telemetry

    tel = Telemetry()
    tel.rounds = [
        RoundRecord(fight=1, round=1, policy="ttk-lookahead"),
        RoundRecord(fight=1, round=2, policy="trained (Q) — Q table"),
        RoundRecord(fight=1, round=3,
                    policy="trained (Q) — fallback (state not in Q table)"),
        RoundRecord(fight=1, round=4, policy="trained (Q) — fallback — a "
                    "480 HP mob is above the 94–423 band this table was "
                    "trained on"),
        RoundRecord(fight=1, round=5, policy="trained (Q) — fallback — a "
                    "480 HP mob is above the 94–423 band this table was "
                    "trained on"),
        RoundRecord(fight=1, round=6, policy="board read failed"),
    ]
    decided, missed, reasons = tel.trained_coverage()
    assert (decided, missed) == (1, 3)

    # Same denominator the Learning tab's meter uses.
    mix = tel.policy_mix()
    trained = sum(n for name, n in mix.items() if "trained" in name)
    assert decided + missed == trained

    # And the reason is aggregated, not merely counted.
    top, n = next(iter(reasons.items()))
    assert "480 HP mob is above" in top and n == 2


def test_the_stated_cause_uses_what_the_misses_recorded(qapp):
    """The run that prompted this printed the real cause on one line and
    "the states are mostly unvisited — raise episodes and retrain" on the
    next. The second contradicted the first and was the one fix that
    could not have helped."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    said = win._why_coverage_is_low(
        {"a 480 HP mob is above the 94–423 band this table was trained on": 3,
         "3 mobs, trained for up to 2": 1})
    assert "480 HP mob is above" in said
    assert "plus 1 for other reasons" in said
    assert "raise episodes" not in said

    # With nothing recorded it still falls back to the inferred causes.
    assert win._why_coverage_is_low({}) != ""
    assert win._why_coverage_is_low(None) != ""


def test_heatmap_rows_name_their_fight_when_fights_are_stacked():
    """Labelling by round alone repeats "r1" once per fight, and the
    matrix stacks fights: three fights showed r1/r2/r3/r1/r2/r4/r7/r1."""
    from deimos_bridge.policies import Candidate
    from deimos_bridge.telemetry import RoundRecord, Telemetry

    def rec(fight, rnd):
        return RoundRecord(fight=fight, round=rnd, candidates=[
            Candidate(card="Frost Beetle", target=0, turns=5, chosen=True)])

    tel = Telemetry()
    tel.rounds = [rec(1, 1), rec(1, 2), rec(2, 1), rec(3, 1), rec(3, 7)]
    rows, _cols, _dropped = tel.decision_matrix()
    labels = [label for label, _cells in rows]
    assert len(labels) == len(set(labels)), labels
    assert labels[0] == "f1 r1" and labels[-1] == "f3 r7"

    # One fight keeps the short label -- the fight number adds nothing.
    tel.rounds = [rec(1, 1), rec(1, 2)]
    assert [l for l, _ in tel.decision_matrix()[0]] == ["r1", "r2"]


def test_the_verdict_probes_where_the_board_can_discriminate(qapp):
    """Scoring at one point is why "98% against 100%" read as a tie: on
    an easy board every policy is at the ceiling and the comparison ranks
    nothing. The same two policies near the edge of the envelope are 30%
    against 76%."""
    from deimos_bridge.gui.app import TrainWorker

    probed = []

    class _Sim:
        def __init__(self, *a, **kw):
            probed.append((kw["enemies"], a))

    w = TrainWorker({}, [], "ice", 500, player_hp=1022, boss_hp=480,
                    n_enemies=2)

    seen = []

    import w101_sim
    real_sim, real_eval = w101_sim.Sim, w101_sim.evaluate_paired

    class _S:
        def __init__(self, cards, deck, school, boss, **kw):
            seen.append(boss.hp)

    def _ep(sim, policies, n=0):
        # trained is flat 50%; the heuristic is good on easy boards and
        # bad on hard ones, so the largest gap is at the hard end. One
        # call per probe now -- both policies ride the same seed stream.
        hp = seen[-1]
        rival = 1.0 if hp < 300 else 0.2
        return {"trained": {"win_rate": 0.5},
                "rival": {"win_rate": rival}}

    w101_sim.Sim, w101_sim.evaluate_paired = _S, _ep
    try:
        t, r = w.compare(object(), {1: (100, 1000), 2: (100, 500)}, 85,
                         ["death"], n=10)
    finally:
        w101_sim.Sim, w101_sim.evaluate_paired = real_sim, real_eval

    # It walked both counts at three depths each, not one board.
    assert len(seen) >= 6, seen
    assert min(seen) < 400 < max(seen), seen
    # And reported the pair that disagreed most, not the last or the mean.
    assert abs(t - r) >= 0.29, (t, r)


# ------------------------------ the rollout has to play the wizard you have
def test_the_live_rollout_gets_the_wizard_s_gear_and_power_pips():
    """`_mk_actor` builds a bare Actor and `Sim._build_player` — the only
    thing that applies player_stats — is never called on the live path,
    so `_sim_for` carried the gear to a Sim that handed the policy a
    naked wizard."""
    from deimos_bridge.live_backend import WizAiBackend
    from w101_sim import Actor, State

    be = WizAiBackend(policy=lambda sim, s: None, cards={}, school="ice",
                      player_stats={"damage": {"ice": 0.09},
                                    "accuracy": 0.05, "pierce": 0.04})
    be.power_pip_chance = 0.85

    class _Read:
        def __init__(self):
            me = Actor(name="W", school="ice", hp=1022, max_hp=1022, team=0)
            self.state = State(me, [Actor(name="m", school="death", hp=258,
                                          max_hp=258, team=1)])

    read = _Read()
    assert read.state.player.damage_bonus == {}      # as the live path builds it
    assert read.state.player.power_pip_chance == 0.0

    be._apply_player_stats(read)
    p = read.state.player
    assert p.damage_bonus == {"ice": 0.09}
    assert p.accuracy_bonus == 0.05
    assert p.pierce == 0.04
    assert p.power_pip_chance == 0.85


def test_gear_flips_a_two_turn_kill_the_lookahead_was_calling_three():
    """The operator's arithmetic, and the reason the sim disagreed. Snow
    Serpent's midpoint is 175 under a 40% Ice Trap: 175 x 1.4 = 245 does
    not kill a 258 HP mob, so the line scores three turns. With 9% ice
    damage the same line is 267 and kills, which is two."""
    from data_full import load_spells_full
    from deimos_bridge.live_backend import WizAiBackend
    from deimos_bridge.policies import greedy_ttk
    from w101_sim import Actor, Boss, Sim, State

    cards = load_spells_full()
    deck = (["Evil Snowman"] * 3 + ["Frost Beetle"] * 3 + ["Ice Trap"] * 3
            + ["Snow Serpent"] * 3)

    def best_turns(geared):
        me = Actor(name="W", school="ice", hp=1022, max_hp=1022, team=0,
                   norm_pips=1)
        foe = Actor(name="Mob", school="death", hp=258, max_hp=258, team=1)
        foe.flat_hit = 85.0
        me.hand = [cards[n] for n in ("Frost Beetle", "Ice Trap",
                                      "Snow Serpent", "Evil Snowman",
                                      "Frost Beetle")]
        me.deck = [cards[n] for n in deck]

        class _Read:
            pass

        read = _Read()
        read.state = State(me, [foe])
        if geared:
            be = WizAiBackend(policy=None, cards=cards, school="ice",
                              player_stats={"damage": {"ice": 0.09}})
            be.power_pip_chance = 0.85
            be._apply_player_stats(read)

        sim = Sim(cards, deck, "ice",
                  Boss(name="Mob", hp=258, school="death", dmg=0),
                  player_hp=1022, player_stats={"damage": {"ice": 0.09}})
        pol = greedy_ttk()
        pol(sim, read.state)
        return min(c.turns for c in pol.last_candidates)

    assert best_turns(geared=False) == 3
    assert best_turns(geared=True) == 2


def test_the_window_spells_out_the_board_it_will_train(qapp):
    """One spinbox cannot say "a 690 HP boss beside a 255 HP minion".
    It does not have to — after a fight the healths come from what was
    seen — but nothing said so, so a derived board looked like a typo in
    a field that was not being used."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.boss_hp.setValue(690)
    win.n_enemies.setValue(2)
    win.player_hp.setValue(1022)

    before = win._board_line()
    assert "690" in before and "552" in before        # 100% / 80%
    assert "spread around the biggest" in before
    assert "no fight measured yet" in before

    win.observed_hps = [690, 255]
    win.observed_schools = ["death", "death"]
    win.observed_incoming = 87.0
    win.mob_damage_measured = True
    after = win._board_line()
    assert "690 death + 255 death" in after           # the real board
    assert "87/round" in after and "measured live" in after
    assert "from your last fight" in after


def test_the_live_fight_count_is_not_labelled_like_an_episode_count(qapp):
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    assert "training" in win.fights.toolTip().lower()
    assert "episodes" in win.fights.toolTip()


# ---------------------------- the refusal has to name the actual cause
def test_the_eval_board_never_wears_the_wizards_own_school(qapp):
    """`Boss.resist_own` is 0.40, so a same-school mob is the worst
    matchup in the game — putting one on a *guessed* board makes the
    guess harder than any fight it stands in for. Cycling the seven
    schools gave an ice wizard a "fire + ice" eval board."""
    from deimos_bridge.gui.app import TrainWorker

    for school in ("ice", "fire", "death"):
        w = TrainWorker({}, [], school, 0, boss_hp=780, n_enemies=4)
        assert school not in w.board_schools(), (school, w.board_schools())

    # An observed board is respected even when it IS the wizard's school
    # — that is a real fight, not a guess.
    w = TrainWorker({}, [], "ice", 0, boss_hp=780, n_enemies=2,
                    mob_schools=["ice", "ice"])
    assert w.board_schools() == ["ice", "ice"]


def test_a_deck_that_cannot_deliver_the_health_is_told_so(qapp):
    """The run that prompted this suggested lowering mob HP, lowering the
    mob count, raising health, and checking the enemy school. None of
    them was the answer: the deck had lost its Evil Snowmen and could not
    deliver 1,404 health however it was played."""
    from data_full import load_spells_full
    from deimos_bridge.gui.app import TrainWorker
    from w101_sim import Boss

    cards = load_spells_full()
    nine = ["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3
    gear = {"damage": {"ice": 0.09}}

    w = TrainWorker(cards, nine, "ice", 0, player_hp=1022, boss_hp=780,
                    n_enemies=2, player_stats=gear)
    # Optimistic upper bound: every card lands, every buff on the biggest
    # hits. Nothing in a real fight beats it.
    assert 1000 < w.damage_ceiling() < 1200

    board = Boss(name="b", hp=780, school="fire", dmg=85)
    extra = [Boss(name="m", hp=624, school="storm", dmg=85)]
    ok, note = w.preflight(board, extra, n=60)
    assert ok is False
    assert "Your deck is the reason" in note
    assert "1,404 health" in note
    assert "would close" in note          # concrete cards, not a shrug
    # ...and it does not send the operator round the knobs that cannot help
    assert "check that the enemy school" not in note

    # The same board with the Snowmen back is winnable, so no refusal.
    w.deck = nine + ["Evil Snowman"] * 3
    assert w.damage_ceiling() > 2000
    assert w.preflight(board, extra, n=60)[0] is True


def test_a_board_lost_on_time_is_diagnosed_differently_from_one_lost_on_damage(qapp):
    """A deck with the damage but not the turns is a different problem
    with a different fix, and must not be told to add damage cards."""
    from data_full import load_spells_full
    from deimos_bridge.gui.app import TrainWorker
    from w101_sim import Boss

    cards = load_spells_full()
    deck = (["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3
            + ["Evil Snowman"] * 3)
    w = TrainWorker(cards, deck, "ice", 0, player_hp=400, boss_hp=900,
                    n_enemies=2, player_stats={"damage": {"ice": 0.09}})

    board = Boss(name="b", hp=900, school="fire", dmg=200)
    extra = [Boss(name="m", hp=720, school="storm", dmg=200)]
    ok, note = w.preflight(board, extra, n=60)
    assert ok is False
    assert "loses on time, not" in note
    assert "Your deck is the reason" not in note
    assert "rounds" in note


# --------------------------- the band has to mean what the key means
def test_a_miss_blames_the_band_only_when_the_bucket_changes():
    """`Featurizer.key` stores `hp // 250`, so a 480 HP mob and a 365 HP
    band edge are the SAME symbol — the band cannot be what the table
    failed to recognise. Comparing raw health blamed it anyway, on every
    board whose biggest mob happened to sit past the edge."""
    from deimos_bridge.policies import trained_policy
    from rl_agent import HP_BUCKET, QAgent
    from w101_sim import Actor, State

    agent = QAgent({}, [], "ice")
    agent.trained_on = {"hp": (40, 365), "mobs": 2, "schools": ["balance"],
                        "bands": {1: (40, 1900), 2: (40, 365)}}
    tp = trained_policy(agent)

    def board(hps):
        me = Actor(name="W", school="ice", hp=1022, max_hp=1022, team=0)
        foes = [Actor(name=f"m{i}", school="balance", hp=h, max_hp=h, team=1)
                for i, h in enumerate(hps)]
        return State(me, foes)

    assert 480 // HP_BUCKET == 365 // HP_BUCKET       # the premise
    assert tp.why_missed(board([480, 235])) == ""     # same bucket, no blame
    assert tp.why_missed(board([365, 235])) == ""
    # A genuinely different bucket still gets named.
    assert "above" in tp.why_missed(board([900, 235]))
    assert "900" in tp.why_missed(board([900, 235]))


def test_the_envelope_stops_on_a_bucket_edge(qapp):
    """Stopping at an arbitrary frontier trains part of a bucket and then
    reports the rest of that bucket as out of band, which is a
    distinction the model does not make."""
    from data_full import load_spells_full
    from deimos_bridge.gui.app import TrainWorker
    from rl_agent import HP_BUCKET

    cards = load_spells_full()
    deck = (["Evil Snowman"] + ["Frost Beetle"] * 4 + ["Ice Trap"] * 4
            + ["Snow Serpent"] * 4)
    w = TrainWorker(cards, deck, "ice", 0, player_hp=1022, boss_hp=780,
                    n_enemies=2, mob_schools=["balance"], mob_damage=147,
                    player_stats={"damage": {"ice": 0.09}})
    bands = w.envelope(n=60)
    assert bands, "the deck should clear something"
    for count, (_lo, hi) in bands.items():
        assert hi % HP_BUCKET == 0, (count, hi)


# ------------------ the table drives only where it has evidence
def test_the_table_needs_evidence_not_just_a_non_zero_entry():
    """"Is this entry non-zero" cannot tell one lucky episode from ten
    thousand, and a single visit is a sample, not an estimate."""
    from deimos_bridge.policies import trained_policy
    from rl_agent import QAgent
    from w101_sim import Actor, State

    agent = QAgent({}, [], "ice")
    tp = trained_policy(agent, min_visits=20)

    key, legal = ("k",), ["a", "b"]
    agent.Q[(key, "a")] = -3.0
    agent.N[(key, "a")] = 4
    # Every return is negative, so the untried "b" sits at 0.0 and would
    # outrank the measured "a". `support` reports the action that will
    # actually be played, which is the one with evidence.
    assert agent.support(key, legal) == (4, 4)

    # Thin evidence: the wrapper must not drive on it...
    agent.feat.key = lambda sim, s: key
    agent.feat.legal = lambda sim, s: legal
    fell_back = []
    tp.fallback = lambda sim, s: fell_back.append(1)

    me = Actor(name="W", school="ice", hp=100, max_hp=100, team=0)
    s = State(me, [Actor(name="m", school="death", hp=50, max_hp=50, team=1)])
    tp(None, s)
    assert fell_back == [1] and tp.missed == 1
    assert "4 time(s) in training" in tp.why_missed(s)

    # ...and must drive once the evidence is there.
    agent.N[(key, "a")] = 40
    agent.policy = lambda: (lambda sim, st: "played")
    assert tp(None, s) == "played"
    assert tp.seen == 1


def test_a_table_without_visit_counts_still_plays():
    """A table trained before visit counts existed must keep working —
    refusing to use it at all would be worse than the old test."""
    from deimos_bridge.policies import trained_policy
    from w101_sim import Actor, State

    class _Old:                       # no `support`, no `N`
        class feat:
            @staticmethod
            def key(sim, s):
                return ("k",)

            @staticmethod
            def legal(sim, s):
                return ["a"]

        Q = {(("k",), "a"): -2.0}

        @staticmethod
        def policy():
            return lambda sim, s: "played"

    tp = trained_policy(_Old())
    me = Actor(name="W", school="ice", hp=100, max_hp=100, team=0)
    s = State(me, [Actor(name="m", school="death", hp=50, max_hp=50, team=1)])
    assert tp(None, s) == "played"
    assert tp.seen == 1


def test_the_biggest_mob_hp_box_reaches_the_trained_band(qapp):
    """Typing 780 and being told a 480 HP mob is outside the band is the
    box doing nothing. Training past the winnable frontier is only safe
    because thin states now hand the round back on their own."""
    from data_full import load_spells_full
    from deimos_bridge.gui.app import TrainWorker
    from rl_agent import HP_BUCKET

    cards = load_spells_full()
    deck = (["Evil Snowman"] + ["Frost Beetle"] * 4 + ["Ice Trap"] * 4
            + ["Snow Serpent"] * 4)
    w = TrainWorker(cards, deck, "ice", 0, player_hp=1022, boss_hp=780,
                    n_enemies=2, mob_hps=[480, 235], mob_schools=["balance"],
                    mob_damage=136, player_stats={"damage": {"ice": 0.09}})
    bands = w.envelope(n=60)
    assert bands
    for count, (_lo, hi) in bands.items():
        assert hi > 780, (count, hi)          # the box is covered
        assert hi % HP_BUCKET == 0, (count, hi)
    assert "training over" in w.describe_envelope(bands)


def test_the_played_policy_does_not_prefer_untried_actions():
    """Every return here is negative, so an entry the agent never updated
    sits at the defaultdict's 0.0 and outranks everything it measured.
    Useful while exploring; the opposite of what you want while playing.
    Measured on a real table: 14% of decisions on states that DID have
    data landed on an action never tried once."""
    from rl_agent import QAgent

    agent = QAgent({}, [], "ice")
    key = ("k",)
    agent.feat.key = lambda sim, s: key
    agent.Q[(key, "measured")] = -3.0
    agent.N[(key, "measured")] = 40

    legal = ["measured", "never_tried"]
    # Exploring still reaches for the unknown — that is the point of it.
    assert agent.greedy(None, None, legal) == "never_tried"
    # Playing does not.
    assert agent.greedy(None, None, legal, tried_only=True) == "measured"

    # A state with no evidence at all still returns something legal
    # rather than nothing.
    empty = ("nothing",)
    agent.feat.key = lambda sim, s: empty
    assert agent.greedy(None, None, legal, tried_only=True) in legal


# ------------------- learning that needs no coverage: the continuation
def test_the_rollout_continuation_is_a_deck_scoped_choice():
    """One small policy reused on every board and every rollout, so it
    needs coverage of nothing — which is what fits a game with 1,912
    creatures in it. Measured worth ~14 points between best and worst,
    and deck-specific: the choice that is +5.2 on one deck is -7.6 on
    another, so there is no global answer to hardcode."""
    from deimos_bridge import policies

    original = policies.continuation_name()
    try:
        for name in policies.CONTINUATIONS:
            assert policies.set_continuation(name) == name
            assert policies.continuation_name() == name
            assert callable(policies.build_continuation(name))
        # An unknown name falls back rather than breaking the rollout.
        assert policies.set_continuation("nonsense") == \
            policies.DEFAULT_CONTINUATION
        # And `_continuation()` reflects the choice.
        policies.set_continuation("nuke-asap")
        from w101_sim import strat_nuke_asap
        assert policies._continuation() is strat_nuke_asap
    finally:
        policies.set_continuation(original)


def test_choosing_a_continuation_ranks_all_the_candidates():
    from data_full import load_spells_full
    from deimos_bridge import policies

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3
    original = policies.continuation_name()
    try:
        best, scores = policies.choose_continuation(
            cards, deck, "ice", [(500, 1, "death")], n=12)
        assert set(scores) == set(policies.CONTINUATIONS)
        assert best in policies.CONTINUATIONS
        assert scores[best] == max(scores.values())
        # The winner is installed, not merely reported.
        assert policies.continuation_name() == best
    finally:
        policies.set_continuation(original)


def test_probe_boards_come_from_the_envelope_not_the_ceiling(qapp):
    """A board every candidate clears ranks nothing. Measured: near the
    ceiling the five continuations scored 97.5-99.0%, a spread that is
    noise; near the edge of the same deck's envelope, 60.0-68.0%."""
    from deimos_bridge.gui.app import TrainWorker

    w = TrainWorker({}, [], "ice", 0, boss_hp=780, n_enemies=2)
    boards = w.probe_boards({1: (40, 1000), 2: (40, 500)}, ["death"])
    assert len(boards) == 4                       # two depths per count
    hps = [hp for hp, _n, _s in boards]
    assert max(hps) < 1000 and min(hps) > 40      # inside, not at the edges
    assert {n for _hp, n, _s in boards} == {1, 2}
    # With no envelope it still returns something usable.
    assert w.probe_boards({}, ["death"]) == [(780, 2, "death")]


def test_search_does_not_stand_still_on_a_board_it_cannot_win():
    """Every candidate returned the identical -(turn + fail), the argmax
    collapsed, and `None` sat at the head of the candidate list — so it
    passed. Measured at a 90% pass rate on unwinnable boards, removing
    3.9% of enemy health where greedy_ttk removes 42%."""
    from data_full import load_spells_full
    from search_policy import make_search_policy
    from w101_sim import Boss, Sim

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3
    sim = Sim(cards, deck, "ice",
              Boss(name="b", hp=1400, school="death", dmg=140), player_hp=1022,
              enemies=[Boss(name="m", hp=1400, school="death", dmg=140)])
    policy = make_search_policy(k=4)

    s = sim.new_state()
    passes = casts = 0
    for _ in range(6):
        move = policy(sim, s)
        if move is None:
            passes += 1
        else:
            card, target = (move if isinstance(move, tuple) else (move, 0))
            sim.cast(s, card, target)
            casts += 1
        sim.end_round(s)
        if s.player_hp <= 0 or not any(e.alive for e in s.enemies):
            break
    assert casts > 0, "it stood still through a losing fight"
    assert passes <= casts, (passes, casts)


def test_the_losing_board_ranking_is_pluggable_and_defaults_to_shipped():
    """This branch fires on 17% of candidates on a level-5 board and on
    37-100% of them on a hard one, so it is worth keeping testable. The
    default is "kills" since the live trace that posed the choice the
    first measurement's boards never did: threat removal beat six
    points of banked damage +2.4/+2.4/+1.2 across three paired
    streams, and on boards without a kill on offer the credit is
    provably decision-identical."""
    import deimos_bridge.policies as P

    assert P.LOST_RANKING == "kills"

    original = P.LOST_RANKING
    try:
        P.LOST_RANKING = "damage"
        # the previous shipped form: rank untouched, second element
        # real banked damage
        assert P._lost_score(14, 250.0, 2, 5) == (14, -250.0)

        P.LOST_RANKING = "kills"
        rank, dealt = P._lost_score(14, 250.0, 2, 5)
        assert dealt == -250.0, "damage must stay real for the panel"
        assert rank < 14, "more kills must rank better"
        # A lost line can never outrank a won one: 4 kills is the most
        # the game offers and the win/stall gap is a whole point.
        assert P._lost_score(14, 0.0, 4, 0)[0] > 12

        P.LOST_RANKING = "survive"
        rank, dealt = P._lost_score(14, 250.0, 0, 9)
        assert dealt == -250.0 and rank < 14
        assert P._lost_score(14, 0.0, 0, 12)[0] > 12
    finally:
        P.LOST_RANKING = original


# ---------------- the leaf value, the horizon, and the tuned search
def test_leaf_features_are_scale_free():
    """The property the tabular key lacks and the one levelling applies
    constantly: multiply every health and damage by k and the state is
    the same fight. The key is 0% invariant to it; phi must be 100%."""
    import dataclasses

    import numpy as np

    from data_full import load_spells_full
    from deimos_bridge.leaf_value import phi
    from w101_sim import Actor, State

    cards = load_spells_full()

    def board(k):
        me = Actor(name="W", school="ice", hp=800 * k, max_hp=1000 * k,
                   team=0, norm_pips=3)
        me.hand = [dataclasses.replace(cards["Frost Beetle"],
                                       damage=85.0 * k),
                   dataclasses.replace(cards["Ice Trap"])]
        me.deck = [dataclasses.replace(cards["Snow Serpent"],
                                       damage=175.0 * k)]
        foes = [Actor(name="m", school="death", hp=400.0 * k,
                      max_hp=500.0 * k, team=1)]
        foes[0].flat_hit = 60.0 * k
        return State(me, foes)

    a, b = phi(None, board(1)), phi(None, board(20))
    assert np.allclose(a, b), (a, b)


def test_the_committed_leaf_weights_load_and_predict():
    from deimos_bridge.leaf_value import FEATURES, LeafValue

    model = LeafValue.load()
    assert len(model.w) == len(FEATURES)

    class _E:
        def __init__(self, hp, mx):
            self.hp, self.max_hp, self.alive = hp, mx, hp > 0
            self.wards = []
            self.flat_hit = 50.0

    class _P:
        hp, max_hp = 900.0, 1000.0
        charms = []

    class _S:
        player = _P()
        enemies = [_E(50.0, 500.0)]
        hand, deck = [], []
        norm_pips, pow_pips = 6, 2

    nearly_won = model(None, _S())
    _S.enemies = [_E(500.0, 500.0), _E(500.0, 500.0), _E(500.0, 500.0)]
    _S.player.hp = 100.0
    nearly_lost = model(None, _S())
    assert 0.0 <= nearly_lost < nearly_won <= 1.0, (nearly_lost, nearly_won)


def test_the_leaf_reranks_only_the_stalled_bucket():
    """When installed, a rollout that runs out of horizon alive is ranked
    by what the surviving position is worth; wins and deaths untouched."""
    from data_full import load_spells_full
    from deimos_bridge import policies as P
    from w101_sim import Boss, Sim

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 4 + ["Ice Trap"] * 4
    sim = Sim(cards, deck, "ice",
              Boss(name="wall", hp=50000, school="death", dmg=0),
              player_hp=1000)
    s = sim.new_state()

    base = P._rollout(sim, s, None, max_turns=2)
    try:
        assert P.load_leaf_value() is not None
        with_leaf = P._rollout(sim, s, None, max_turns=2)
    finally:
        P.set_leaf_value(None)
    assert base[0] == with_leaf[0] == 3          # both stalled (2 + 1)
    assert with_leaf[1] != base[1]               # ranked differently


def test_ev_pricing_exists_but_is_off_by_default():
    """Measured: EV accuracy alone is -4.0 points, EV + leaf is -5.0.
    The knob stays so the numbers are re-checkable; the default must
    remain the shipped optimistic rollout."""
    from data_full import load_spells_full
    from deimos_bridge import policies as P

    assert P.ROLLOUT_ACCURACY == "optimistic"
    cards = load_spells_full()
    cat = P.ev_card(cards["Fire Cat"])
    assert cat.accuracy == 1.0 and cat.damage == 75.0       # 100 x 0.75
    elf = P.ev_card(cards["Fire Elf"])
    dot = next(o for o in elf.ops if o["op"] == "dot")
    assert dot["total"] == 157.5                            # 210 x 0.75
    blade = P.ev_card(cards["Fireblade"])
    assert blade.percent == 0.35 and blade is cards["Fireblade"]


def test_the_search_horizon_is_deck_scoped():
    from deimos_bridge import policies as P

    assert P.DEFAULT_HORIZON == 12
    assert P.search_horizon() == 12
    try:
        assert P.set_search_horizon(6) == 6
        assert P.search_horizon() == 6
    finally:
        P.set_search_horizon(None)
    assert P.search_horizon() == 12


def test_choose_search_sweeps_continuations_and_horizons():
    from data_full import load_spells_full
    from deimos_bridge import policies as P

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3
    try:
        name, horizon, scores = P.choose_search(
            cards, deck, "ice", [(350, 2, "death")], n=8, dmg=55)
        assert name in P.CONTINUATIONS
        assert horizon in P.HORIZONS
        # continuations x horizons, plus one entry per search width
        assert len(scores) == (len(P.CONTINUATIONS) * len(P.HORIZONS)
                               + len(P.SEARCH_WIDTHS))
        assert P.search_horizon() == horizon      # installed, not reported
        assert P.continuation_name() == name
    finally:
        P.set_search_horizon(None)
        P.set_continuation(P.DEFAULT_CONTINUATION)


# ---------------- the live reader stops dropping scheduled damage
def test_a_live_dot_lands_on_the_actors_schedule():
    """The reader mapped four effect kinds and dropped the rest, so a
    mob carrying a Fire Elf's remaining 200 damage looked identical to a
    healthy one — "it does not understand DoTs", caused not by the model
    but by the model never being told."""
    import asyncio

    from deimos_bridge.live_state import NameResolver, read_state
    from deimos_bridge.mock_client import (MockCard, MockCombat, MockEffect,
                                           MockMember)
    from data_full import load_spells_full

    cards = load_spells_full()
    me = MockMember("W", 800, client=True, normal_pips=2)
    burning = MockMember("Burning", 180, monster=True, hangings=[
        MockEffect("damage_over_time", 200.0, 2343174, 999, num_rounds=2)])
    combat = MockCombat([me, burning], [MockCard("Frost Beetle")])
    read = asyncio.new_event_loop().run_until_complete(
        read_state(combat, NameResolver(cards), "ice"))

    e = read.state.enemies[0]
    assert [(o.kind, o.per_tick, o.rounds_left) for o in e.over_time] == \
        [("dot", 100.0, 2)]
    assert all(h.kind != "damage" or h.percent for h in e.wards)


def test_a_live_dot_changes_the_target():
    """Two identical mobs, one already dying from a DoT: the lookahead
    must spend its hit on the other one."""
    import asyncio

    from deimos_bridge.live_state import NameResolver, read_state
    from deimos_bridge.mock_client import (MockCard, MockCombat, MockEffect,
                                           MockMember)
    from deimos_bridge.policies import greedy_ttk
    from data_full import load_spells_full
    from w101_sim import Boss, Sim

    cards = load_spells_full()

    def board(dot):
        me = MockMember("W", 800, client=True, normal_pips=2)
        hang = ([MockEffect("damage_over_time", 200.0, 2343174, 999,
                            num_rounds=2)] if dot else [])
        return MockCombat(
            [me, MockMember("Burning", 180, monster=True, hangings=hang),
             MockMember("Healthy", 180, monster=True)],
            [MockCard("Frost Beetle"), MockCard("Snow Serpent")])

    run = asyncio.new_event_loop().run_until_complete
    r = NameResolver(cards)
    sim = Sim(cards, ["Frost Beetle"] * 3 + ["Snow Serpent"] * 3, "ice",
              Boss(name="B", hp=180, school="death", dmg=0), player_hp=800)

    targets = {}
    for dot in (False, True):
        move = greedy_ttk()(sim, run(read_state(board(dot), r, "ice")).state)
        targets[dot] = move[1] if isinstance(move, tuple) else 0
    assert targets[False] == 0          # nothing scheduled: hit the first
    assert targets[True] == 1           # it is already dying: hit the other


def test_a_live_mantle_reaches_the_accuracy_charms():
    import asyncio

    from deimos_bridge.live_state import NameResolver, read_state
    from deimos_bridge.mock_client import (MockCard, MockCombat, MockEffect,
                                           MockMember)
    from data_full import load_spells_full

    cards = load_spells_full()
    me = MockMember("W", 800, client=True, normal_pips=2,
                    hangings=[MockEffect("modify_accuracy", -45.0,
                                         80289, 555)])
    combat = MockCombat([me, MockMember("m", 180, monster=True)],
                        [MockCard("Frost Beetle")])
    read = asyncio.new_event_loop().run_until_complete(
        read_state(combat, NameResolver(cards), "ice"))
    acc = [h for h in read.state.player.charms if h.kind == "accuracy"]
    assert len(acc) == 1 and acc[0].percent == -0.45


def test_the_board_panel_shows_the_burn():
    from deimos_bridge.live_backend import PolicyDecision
    from deimos_bridge.telemetry import Telemetry
    from w101_sim import Actor, OverTime, State

    tel = Telemetry()
    tel.start_fight()
    me = Actor(name="W", school="ice", hp=800, max_hp=800, team=0)
    foe = Actor(name="m", school="death", hp=400, max_hp=400, team=1)
    foe.over_time.append(OverTime("live:999", "dot", "fire", 70.0, 3))

    class _Read:
        state = State(me, [foe])
        round_number = 1
        hand_cards = {}
        resolver = type("R", (), {"misses": set()})()
        hidden = []
        hand_visibility = 1.0

    rec = tel.observe(PolicyDecision(passing=True, reason="x"), _Read())
    assert any("70/tick x3 dot" in w for w in rec.enemies[0].wards)


def test_the_tuner_also_picks_the_driver():
    """search(k=6) beat the lookahead by +2.8/+3.3 on richer decks and
    lost by 3.6 on a starter — deck-dependent like the continuation and
    the horizon, so it is chosen the same way: measured on the probes."""
    from data_full import load_spells_full
    from deimos_bridge import policies as P

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3
    try:
        _n, _h, scores = P.choose_search(cards, deck, "ice",
                                         [(350, 2, "death")], n=9, dmg=55)
        assert all(f"search(k={k})" in scores for k in P.SEARCH_WIDTHS)
        assert (P.driver_name() == "ttk"
                or P.driver_name() in {f"search(k={k})"
                                       for k in P.SEARCH_WIDTHS})
        assert callable(P.tuned_driver())
    finally:
        P.set_search_horizon(None)
        P.set_continuation(P.DEFAULT_CONTINUATION)
        P._DRIVER = "ttk"


def test_the_tuned_trio_survives_the_wire():
    """Continuation, horizon AND driver ride the same string, so a
    restart between Train and Play live keeps all three choices."""
    from deimos_bridge import policies as P

    wire = "nuke-asap @ horizon 6 @ driver search(k=6)"
    name = wire
    if " @ driver " in name:
        name, drv = name.rsplit(" @ driver ", 1)
        P.set_driver(drv)
    if " @ horizon " in name:
        name, h = name.rsplit(" @ horizon ", 1)
        P.set_search_horizon(int(h))
    try:
        assert P.set_continuation(name) == "nuke-asap"
        assert P.search_horizon() == 6
        assert P.driver_name() == "search(k=6)"
        assert P.set_driver("nonsense") == "ttk"   # unknown -> safe default
    finally:
        P.set_search_horizon(None)
        P.set_continuation(P.DEFAULT_CONTINUATION)
        P.set_driver("ttk")


def test_deck_advice_names_castable_cards(qapp):
    """The boards no policy can win are lost in the deck box; the advice
    must name real, cheap-pip additions, not a max-level spell a level-5
    wizard cannot cast."""
    from data_full import load_spells_full
    from deimos_bridge.gui.app import TrainWorker

    cards = load_spells_full()
    nine = ["Frost Beetle"] * 3 + ["Ice Trap"] * 3 + ["Snow Serpent"] * 3
    w = TrainWorker(cards, nine, "ice", 0, player_hp=1022, boss_hp=780,
                    n_enemies=2, player_stats={"damage": {"ice": 0.09}})
    hint = w.deck_advice(325)
    assert "would close" in hint
    named = [part.split("x ", 1)[1] for part in
             hint.split("Adding ", 1)[1].split(" would")[0].split(", ")]
    for name in named:
        card = cards[name]
        assert card.pips <= 4 and card.school == "ice", (name, card.pips)
    assert w.deck_advice(0) == "" or "would close" in w.deck_advice(0)


def test_the_cast_uses_the_backends_click_pacing():
    """wizwalker pauses `sleep_time` twice per cast and defaults it to
    1.0 — ~2 s of standing still per cast, ~12 s per fight. The backend
    has always declared `cast_time` for this; nothing consumed it."""
    import inspect

    from deimos_bridge import live_backend

    src = inspect.getsource(live_backend.WizAiCombatHandler.handle_round)
    assert "sleep_time=self.backend.cast_time" in src


# ------------------- the deck search builds for the measured fight
def test_the_deck_worker_builds_for_the_observed_board(qapp, monkeypatch):
    """The repo has had a two-stage deck search all along; what it never
    had was the real fight to build for. The worker hands it the board
    the live run measured — healths, schools, incoming — not a guess."""
    import deck_builder
    from deimos_bridge.gui.app import DeckWorker

    seen = {}

    def fake_build_deck(cards, school, boss, enemies=None, **kw):
        seen.update(boss_hp=boss.hp, boss_school=boss.school,
                    dmg=boss.dmg, n_extra=len(enemies or []),
                    level=kw.get("level"), player_hp=kw.get("player_hp"))
        return ["Frost Beetle"] * 4, 0.87, 5.2, []

    monkeypatch.setattr(deck_builder, "build_deck", fake_build_deck)
    w = DeckWorker({}, "ice", 1022, {"damage": {"ice": 0.09}},
                   [480, 235], ["death", "death"], 136, 780, 2)
    got = {}
    w.finished_ok.connect(lambda d, win, ttk: got.update(deck=d, win=win))
    w.status.connect(lambda *_: None)
    w.run()                                       # synchronous: no thread

    assert seen["boss_hp"] == 480 and seen["boss_school"] == "death"
    assert seen["n_extra"] == 1 and seen["dmg"] == 136
    assert seen["player_hp"] == 1022
    assert got["deck"] == ["Frost Beetle"] * 4 and got["win"] == 0.87
    # The level came off the health curve, gated LOW so the search can
    # hide a trained card but never propose an untrained one.
    assert seen["level"] is None or 1 <= seen["level"] <= 120


def test_the_level_guess_gates_low():
    from deimos_bridge.gui.app import DeckWorker

    w = DeckWorker({}, "ice", 1022, {}, [], [], 0, 780, 2)
    level = w.level_guess()
    if level is not None:
        from player_curves import school_hp
        assert school_hp("ice", level) <= 1022


# ---------------------- the catalog's cheat notes finally get read
def test_the_bestiary_matches_names_and_tiers():
    from deimos_bridge.bestiary import cheat_warning, lookup

    r = lookup("Lord Nightshade", 690)
    assert r and r["school"] == "death" and r["health"] == 690
    # Tier disambiguation by observed health.
    high = lookup("Lord Nightshade", 13200)
    assert high and high["health"] != 690
    assert lookup("No Such Creature XYZ") is None
    assert cheat_warning("No Such Creature XYZ") == ""


def test_a_known_cheater_is_announced_once(qapp):
    """750 creatures in the catalog cheat, with scraped notes; the run
    reads enemy names every round and never looked them up. The operator
    is told once per boss per session — a known interrupt is survivable
    in a way a surprise one is not."""
    import json

    from deimos_bridge.gui.app import MainWindow
    from deimos_bridge.telemetry import EnemyView, RoundRecord

    cheater = next(x for x in json.load(open("bosses_clean.json"))
                   if x.get("has_cheats") and x.get("cheat_notes"))
    win = MainWindow(Telemetry())
    rec = RoundRecord(fight=1, round=1, enemies=[
        EnemyView(cheater["name"], cheater["health"], cheater["health"])])
    win.on_round(rec)
    assert "cheats" in win.status.text()
    first = win.status.text()

    win.status.setText("something else")
    win.on_round(rec)                     # same boss again: no re-announce
    assert win.status.text() == "something else"
    assert first in win._cheats_warned


def test_the_catalogs_boss_stats_reach_the_read_actor():
    """The read infers a mob's school and nothing else about its
    defences; the catalog KNOWS them for named creatures. Lord
    Nightshade halves death damage and takes +20% from life — the
    difference between a hit landing at 0.5x and 1.2x, and the sim's
    _resist_mult consumes exactly these dicts."""
    from deimos_bridge.bestiary import stat_overrides
    from deimos_bridge.live_backend import WizAiBackend
    from w101_sim import Actor, State

    assert stat_overrides("Lord Nightshade", 690) == \
        ({"death": 0.5}, {"life": 0.2}, False)

    be = WizAiBackend(policy=lambda sim, s: None, cards={}, school="ice")
    me = Actor(name="W", school="ice", hp=800, max_hp=800, team=0)
    boss = Actor(name="Lord Nightshade", school="death", hp=690,
                 max_hp=690, team=1)

    class _Read:
        state = State(me, [boss])

    be._apply_bestiary(_Read())
    assert boss.resist == {"death": 0.5}
    assert boss.boost == {"life": 0.2}

    # Observed facts stay authoritative: a live-read resist is kept.
    boss2 = Actor(name="Lord Nightshade", school="death", hp=690,
                  max_hp=690, team=1)
    boss2.resist = {"*": 0.1}
    _Read.state = State(me, [boss2])
    be._apply_bestiary(_Read())
    assert boss2.resist == {"*": 0.1}


def test_the_deck_search_prices_the_named_boss(qapp, monkeypatch):
    """Resist decides which school of damage a deck should slot at all;
    a search that priced Lord Nightshade as a generic death mob would
    happily fill the deck with the one school he halves."""
    import deck_builder
    from deimos_bridge.gui.app import DeckWorker

    seen = {}

    def fake_build_deck(cards, school, boss, enemies=None, **kw):
        seen.update(name=boss.name, resist=boss.resist_map,
                    boost=boss.boost_map)
        return ["Frost Beetle"] * 4, 0.9, 5.0, []

    monkeypatch.setattr(deck_builder, "build_deck", fake_build_deck)
    w = DeckWorker({}, "ice", 1022, {}, [690], ["death"], 136, 690, 1,
                   mob_names=["Lord Nightshade"])
    w.status.connect(lambda *_: None)
    w.finished_ok.connect(lambda *_: None)
    w.run()
    assert seen["name"] == "Lord Nightshade"
    assert seen["resist"] == {"death": 0.5}
    assert seen["boost"] == {"life": 0.2}


def test_the_catalog_corrects_a_misread_school():
    """`read_school`'s failure mode is a silent "balance" guess, and
    the guess poisons everything downstream: a fire wizard's fight
    against the FIRE boss Alicane Swiftarrow was read as "480 balance
    + 235 balance", so training, the envelope and the trained table
    all priced fire damage landing at full when the real fight halves
    it. For an exact catalog name, the scraped school wins."""
    from deimos_bridge.live_backend import WizAiBackend
    from w101_sim import Actor, State

    be = WizAiBackend(policy=lambda sim, s: None, cards={}, school="fire")
    me = Actor(name="W", school="fire", hp=589, max_hp=589, team=0)
    boss = Actor(name="Alicane Swiftarrow", school="balance", hp=480,
                 max_hp=480, team=1)

    class _Read:
        state = State(me, [boss])

    be._apply_bestiary(_Read())
    assert boss.school == "fire"                  # the catalog's fact
    assert boss.resist == {"fire": 0.4}           # and his real wall
    assert boss.boost == {"ice": 0.2}


def test_overkill_burns_die_with_their_target():
    """A live trace showed the lookahead spending a 2-pip Fire Elf on
    a 14 HP minion instead of the 1-pip Cat -- because the sim
    re-resolved the dot op after the killing hit, transferring the
    whole burn to the boss. The real game binds the spell to its
    target (live-confirmed: the burn never appeared on the boss). A
    target that dies mid-cast stays the target; the ops that follow
    no-op."""
    import random

    from data_full import LIVE_RULES, load_spells_full
    from deimos_bridge.policies import greedy_ttk
    from w101_sim import Boss, Sim

    cards = load_spells_full()
    sim = Sim(cards, ["Fire Elf"] * 4, "fire",
              Boss(name="boss", hp=550, school="death", dmg=0),
              enemies=[Boss(name="minion", hp=14, school="balance",
                            dmg=0)],
              player_hp=684, rules=LIVE_RULES, rng=random.Random(5),
              player_stats={"accuracy": 0.5})     # never fizzles
    s = sim.new_state()
    s.player.norm_pips = 2
    elf = next(c for c in s.hand if c.name == "Fire Elf")
    sim.cast(s, elf, target=1)
    assert not s.enemies[1].alive                 # the minion died
    assert s.enemies[1].hp <= 0
    assert s.enemies[0].over_time == []           # no transferred burn
    hp0 = s.enemies[0].hp
    sim.end_round(s)
    assert s.enemies[0].hp == hp0                 # and nothing ticks

    # The decision that exposed it: minion at 14, cat and elf both
    # kill it -- with no free transfer, the 1-pip Cat is not beaten
    # by the 2-pip Elf.
    deck = ["Fire Cat"]*3 + ["Fire Elf"]*3 + ["Fireblade"]*3 + ["Pixie"]*2
    sim2 = Sim(cards, deck, "fire",
               Boss(name="boss", hp=440, school="death", dmg=60),
               enemies=[Boss(name="minion", hp=14, school="balance",
                             dmg=79)],
               player_hp=684, rules=LIVE_RULES, rng=random.Random(5))
    s2 = sim2.new_state()
    s2.player.hand[:] = [cards["Fire Cat"], cards["Fire Elf"],
                         cards["Fireblade"], cards["Pixie"]]
    s2.player.norm_pips = 2
    pol = greedy_ttk(6)
    pol(sim2, s2)
    by = {(c.card, c.target): c for c in pol.last_candidates}
    cat = by[("Fire Cat", 1)]
    elf2 = by[("Fire Elf", 1)]
    assert (cat.turns, -cat.damage, 1) <= (elf2.turns, -elf2.damage, 2)


def test_duplicate_live_blades_share_a_stacking_identity():
    """A live trace priced a Fire Cat at 100 x 1.35 x 1.35 = 182: two
    copies of the same Fireblade had been given DIFFERENT stack keys
    (the old fallback used the effect's list position), so the rollout
    consumed both on a single hit -- every extra blade looked
    multiplicative, and three rounds of blade-spam followed on a fight
    the wizard nearly lost. Same shape now means same key, whether or
    not the template id reads; the sim then applies one per hit, which
    is the game's rule."""
    import asyncio

    from deimos_bridge.live_state import read_hangings
    from w101_sim import Boss, Sim

    class _Enum:
        name = "modify_outgoing_damage"

    class _NoTid:
        async def effect_type(self):
            return _Enum()

        async def effect_param(self):
            return 35.0

        async def damage_type(self):
            return 2343174                      # fire

        async def spell_template_id(self):
            raise RuntimeError("old wizwalker")

    class _Participant:
        async def hanging_effects(self):
            return [_NoTid(), _NoTid()]

        async def aura_effects(self):
            return []

    class _Member:
        async def get_participant(self):
            return _Participant()

    charms = asyncio.new_event_loop().run_until_complete(
        read_hangings(_Member(), "charm"))
    assert len(charms) == 2
    assert charms[0].stack_key == charms[1].stack_key

    from data_full import load_spells_full
    cards = load_spells_full()
    sim = Sim(cards, ["Fire Cat"], "fire",
              Boss(name="b", hp=500, school="balance", dmg=0),
              player_hp=600)
    s = sim.new_state()
    s.player.charms[:] = charms
    mult = sim._consume_damage_charms(s, s.player, "fire")
    assert abs(mult - 1.35) < 1e-9          # ONE applies, not both
    assert len(s.player.charms) == 1        # the duplicate stays banked


def test_the_rollout_banks_the_burn():
    """`dealt` used to sum cast damage only; DoT ticks land in
    end_round and were invisible, so a Fire Elf line that killed with
    its burn banked just the initial hit and lost every damage
    tiebreak to a blade line -- an anti-DoT bias, on the school built
    around DoTs. Damage is the board delta now: a kill banks the whole
    board no matter who delivered the last point."""
    import random

    from data_full import LIVE_RULES, load_spells_full
    from deimos_bridge.policies import greedy_ttk
    from w101_sim import Boss, Hanging, Sim

    cards = load_spells_full()
    deck = ["Fire Cat"]*3 + ["Fire Elf"]*3 + ["Fireblade"]*3 + ["Pixie"]*2
    sim = Sim(cards, deck, "fire",
              Boss(name="Warhorn", hp=285, school="balance", dmg=54),
              player_hp=666,
              player_stats={"damage": {"*": 0.0}, "accuracy": 0.05},
              rules=LIVE_RULES,
              # seeded: new_state shuffles the deck, and the rollout's
              # continuation draws from it -- near-tie picks must not
              # swing with the shuffle inside a test
              rng=random.Random(11))
    s = sim.new_state()
    s.player.hand[:] = [cards["Fire Cat"], cards["Fire Elf"],
                        cards["Fireblade"], cards["Pixie"]]
    s.player.norm_pips = 2
    pol = greedy_ttk(6)
    pol(sim, s)
    elf = next(c for c in pol.last_candidates if c.card == "Fire Elf")
    assert elf.turns <= 6                   # the line kills in-horizon
    assert elf.damage == 285.0              # and banks the WHOLE board

    # The behavioural regression from the live export: one blade
    # already up, the same hand -- the pick must not be a second blade.
    s2 = sim.new_state()
    s2.player.charms[:] = [Hanging(name="live:b", slot="charm",
                                   kind="damage", percent=0.35,
                                   schools={"fire"}, source="live",
                                   sub="b")]
    s2.player.hand[:] = [cards["Fire Cat"], cards["Fire Elf"],
                        cards["Fireblade"], cards["Pixie"]]
    s2.player.norm_pips = 2
    move = pol(sim, s2)
    assert move is not None and move[0].name != "Fireblade"


def test_the_fight_outcome_is_read_off_the_client(qapp):
    """Twelve live fights exported as "wins: 0" with won=null on every
    one, including a clear win. The combat handler does not report
    outcomes; the client does -- a defeated wizard leaves the duel at
    zero health. Zero-round fights (spurious boundaries) stay
    unknown."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    tel = Telemetry()
    w = LiveWorker(tel, "fire", ["Fire Cat"] * 4, "ttk-lookahead", 1)

    class _Stats:
        hp = 223

        async def current_hitpoints(self):
            return self.hp

    class _Client:
        stats = _Stats()

    run = asyncio.new_event_loop().run_until_complete
    tel.start_fight()
    tel.fights[-1].rounds = 13
    assert run(w._fight_outcome(_Client())) is True     # alive: won

    _Stats.hp = 0
    assert run(w._fight_outcome(_Client())) is False    # defeated

    tel.start_fight()                                   # 0 rounds
    _Stats.hp = 500
    assert run(w._fight_outcome(_Client())) is None     # unknown


def test_an_early_pass_does_not_wear_last_rounds_candidates():
    """Round 10 of a live export said "policy chose to pass" beside a
    candidate table claiming Pixie was chosen -- the previous round's
    comparison, left on the attribute when the decision ended early
    because nothing was castable."""
    from data_full import load_spells_full
    from deimos_bridge.policies import greedy_ttk
    from w101_sim import Boss, Sim

    cards = load_spells_full()
    sim = Sim(cards, ["Fire Cat"] * 3 + ["Fire Elf"] * 3, "fire",
              Boss(name="b", hp=480, school="fire", dmg=0),
              player_hp=589)
    pol = greedy_ttk(6)
    s = sim.new_state()
    pol(sim, s)
    assert pol.last_candidates            # a real comparison happened

    s.player.hand[:] = []                 # nothing castable this round
    assert pol(sim, s) is None
    assert pol.last_candidates == []      # and the record says so


def test_maxed_episodes_get_honest_advice_not_a_dead_end(qapp):
    """The window told an operator at the episode box's MAXIMUM to
    "raise episodes and retrain" — advice its own spinbox made
    impossible to follow, about a scaling law (coverage ~episodes^0.43)
    its own measurements say would barely help. At high episode counts
    the message now says what is true: the table cannot key this range
    and the ttk policy is the stronger driver. And the box itself goes
    to 2,000,000 now, so the low-count advice stays followable."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    assert win.episodes.maximum() >= 2_000_000

    win.generalize.setChecked(True)
    win.episodes.setValue(200_000)
    why = win._why_coverage_is_low()
    assert "ttk" in why and "raise episodes and retrain" not in why

    win.episodes.setValue(20_000)
    assert "raise episodes and retrain" in win._why_coverage_is_low()


def test_the_incoming_mean_counts_the_quiet_rounds():
    """A round where every mob fizzled or passed is a real round of
    the damage distribution. Dropping zero-loss rounds biased the
    estimate high: a live fight read 117/round per enemy on a board
    dealing ~78, because the two rounds that hurt were averaged and
    the round that didn't was thrown away."""
    from deimos_bridge.live_backend import WizAiBackend
    from w101_sim import Actor, State

    be = WizAiBackend(policy=lambda sim, s: None, cards={}, school="fire")
    me = Actor(name="W", school="fire", hp=589, max_hp=589, team=0)
    foes = [Actor(name="a", school="fire", hp=480, max_hp=480, team=1),
            Actor(name="b", school="fire", hp=235, max_hp=235, team=1)]

    def read_at(hp, rnd):
        class _R:
            state = State(me, foes)
            round_number = rnd
        me.hp = hp
        return _R()

    be._estimate_incoming(read_at(589, 1))     # baseline
    be._estimate_incoming(read_at(589, 2))     # quiet round: 0 lost
    be._estimate_incoming(read_at(475, 3))     # 114 lost
    per = be._estimate_incoming(read_at(121, 4))   # 354 lost
    assert abs(per - (0 + 57 + 177) / 3) < 1e-6    # the zero counts

    # A healing round (hp went UP) stays out: that is the heal's
    # number, not the board's.
    be._estimate_incoming(read_at(500, 5))
    assert len(be._incoming) == 3


def test_a_casters_damage_is_not_billed_to_the_minion():
    """`_estimate_incoming` splits the health lost evenly across the
    living enemies, which hands half the boss's Sunbird to the minion
    beside him -- and then the boss's casting model deals his own
    damage ON TOP. On a live fight that double-count read Magma Man at
    117/round (real: ~50), every rollout died inside the horizon, and
    the decisions collapsed to the sentinel ranking."""
    from data_full import load_spells_full
    from deimos_bridge.live_backend import WizAiBackend
    from w101_sim import Actor, State

    cards = load_spells_full()
    be = WizAiBackend(policy=lambda sim, s: None, cards=cards,
                      school="fire")
    me = Actor(name="W", school="fire", hp=589, max_hp=589, team=0)
    boss = Actor(name="Alicane Swiftarrow", school="fire", hp=480,
                 max_hp=480, team=1, flat_hit=117.0,
                 spell_pool=["Sunbird", "Fire Elf", "Fire Cat"],
                 power_pip_chance=0.0)
    magma = Actor(name="Magma Man", school="balance", hp=235,
                  max_hp=235, team=1, flat_hit=117.0)

    class _Read:
        state = State(me, [boss, magma])

    # At the (biased-high) 117 mean the remainder still exceeds the
    # plain split, and the cap holds: apportionment only ever LOWERS
    # a stamp, so a noisy total cannot inflate the minion further.
    be._measured_incoming = 117.0
    be._apportion_incoming(_Read())
    assert magma.flat_hit == 117.0
    assert boss.flat_hit == 117.0          # casters keep theirs (unused)

    # The fixes compose: zero-counting brings the mean to 78, and then
    # the caster's ~92/round comes out of the flat mob's bill -- total
    # 156 minus ~92 modeled lands the minion at ~64, near his true ~50.
    magma.flat_hit = 78.0
    be._measured_incoming = 78.0
    be._apportion_incoming(_Read())
    assert 40.0 <= magma.flat_hit <= 70.0


def test_a_kill_outranks_slightly_more_banked_damage():
    """The live trace that flipped LOST_RANKING: a dying board where
    "bank the most damage" hit the full-health boss for 81 banked
    instead of removing a 75 HP attacker dealing ~50/round. The kill
    credit prices threat removal; on boards without a kill on offer it
    provably changes nothing."""
    from deimos_bridge.policies import LOST_RANKING, _lost_score

    assert LOST_RANKING == "kills"
    kill_line = _lost_score(8, 75.0, 1, 4)      # kills the minion
    bank_line = _lost_score(8, 81.0, 0, 4)      # 6 more banked, no kill
    assert min(kill_line, bank_line) == kill_line


def test_the_preflight_asks_the_search_before_refusing(qapp):
    """The scripted canary is the weakest policy in the repo, and
    refusing on its word alone overclaims badly: on a live operator's
    board (480 + 235 balance at 77/round, a heal-less fire deck) the
    canary won 0.0% of 500 while greedy_ttk -- the policy that
    actually drives live fights -- won 60.4%. The GUI told the
    operator "this board cannot be won" about a fight the AI wins
    three times in five."""
    from data_full import load_spells_full
    from deimos_bridge.gui.app import TrainWorker
    from w101_sim import Boss

    cards = load_spells_full()
    deck = ["Fire Cat"] * 3 + ["Fire Elf"] * 3 + ["Fireblade"] * 3
    w = TrainWorker(cards, deck, "fire", 100, player_hp=589,
                    boss_hp=480, player_stats={"accuracy": 0.05},
                    n_enemies=2, mob_hps=[480, 235],
                    mob_schools=["balance", "balance"], mob_damage=77)
    board = Boss(name="b", hp=480, school="balance", dmg=77)
    extra = [Boss(name="m", hp=235, school="balance", dmg=77)]
    ok, note = w.preflight(board, extra, n=200)
    assert ok, f"refused a board the search wins: {note}"


def test_a_refusal_is_not_a_failure(qapp, monkeypatch):
    """The preflight's verdict is a finding about the BOARD; rendering
    it as "training failed" reads as the tool breaking, which is
    exactly how a live operator read it."""
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    warned = []
    monkeypatch.setattr(app_mod.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a))
    win = MainWindow(Telemetry())
    win.on_train_refused("this board cannot be won at these settings")
    assert "refused" in win.status.text()
    assert "failed" not in win.status.text()
    assert warned


def test_the_refusal_names_a_school_wall(qapp):
    """A fire wizard's all-fire deck against a fire board loses ~40%
    of every hit to own-school resist, and same-school deck advice
    cannot fix that -- the refusal must say so and point off-school.
    Measured on the fight that earned this: Alicane Swiftarrow + Magma
    Man vs a level-7 fire wizard, 0.2% for every policy in the repo."""
    from data_full import load_spells_full
    from deimos_bridge.gui.app import TrainWorker
    from w101_sim import Boss

    cards = load_spells_full()
    deck = (["Fire Cat"] * 3 + ["Fire Elf"] * 3 + ["Fireblade"] * 3
            + ["Pixie"] * 2)
    w = TrainWorker(cards, deck, "fire", 100, player_hp=589,
                    boss_hp=480, player_stats={"accuracy": 0.05},
                    n_enemies=2, mob_hps=[480, 235],
                    mob_schools=["fire", "fire"], mob_damage=150)
    board = Boss(name="b", hp=480, school="fire", dmg=150)
    extra = [Boss(name="m", hp=235, school="fire", dmg=150)]
    ok, note = w.preflight(board, extra, n=60)
    assert not ok
    assert "school wall" in note and "treasure cards" in note
    assert "fire damage" in note


def test_the_bestiary_reads_universal_resist():
    """The scrape stores universal resist in two shapes stat_overrides
    used to drop entirely: a bare number with no note (44 creatures,
    the Nightshade tiers among them) and a "to all schools" note (226
    more). The sim reads resist["*"] on every hit, so dropping them
    priced a 60%-resist boss at zero."""
    from deimos_bridge.bestiary import stat_overrides

    resist, boost, _ = stat_overrides("Lord Nightshade", 13200)
    assert resist == {"*": 0.6} and boost == {}
    # String-shaped stats parse too: "+25 to [Fire][Myth]" carries both
    # the value and the schools.
    _, boost, _ = stat_overrides("Cake Mimic")
    assert boost == {"fire": 0.25, "myth": 0.25}


def test_the_full_boss_is_a_casting_boss():
    """`full_boss` hands back the repo's real boss model — spell pool,
    opening pips, exact defences — with the observed health stamped on,
    because the client read is the ground truth for the fight actually
    in progress."""
    from deimos_bridge.bestiary import full_boss

    b = full_boss("Lord Nightshade", 690)
    assert b is not None and b.school == "death"
    assert b.pool and b.dmg == 0          # casts, does not auto-attack
    assert b.start_pips >= 1
    assert b.resist_map == {"death": 0.5} and b.boost_map == {"life": 0.2}
    assert b.hp == 690

    high = full_boss("Lord Nightshade", 13200)
    assert high.hp == 13200               # observed health is stamped
    assert high.resist_map == {"*": 0.6}  # ...and picked the tier
    assert high.start_pips == 5

    # The returned boss is a copy: one fight's edits stay its own.
    high.pool.append("XXX")
    high.resist_map["fire"] = 9.9
    again = full_boss("Lord Nightshade", 13200)
    assert "XXX" not in again.pool and "fire" not in again.resist_map

    assert full_boss("No Such Creature XYZ") is None


def test_the_deck_search_fights_the_casting_boss(qapp, monkeypatch):
    """A named catalog boss reaches build_deck with its spell pool and
    opening pips, so the search prices the heavy hit an opener makes
    legal — the exact tempo a shield-or-race deck choice hangs on —
    instead of a flat per-round hit."""
    import deck_builder
    from deimos_bridge.gui.app import DeckWorker

    seen = {}

    def fake_build_deck(cards, school, boss, enemies=None, **kw):
        seen.update(pool=boss.pool, dmg=boss.dmg, hp=boss.hp,
                    pips=boss.start_pips, school=boss.school)
        return ["Frost Beetle"] * 4, 0.9, 5.0, []

    monkeypatch.setattr(deck_builder, "build_deck", fake_build_deck)
    w = DeckWorker({}, "ice", 1022, {}, [690], ["death"], 136, 690, 1,
                   mob_names=["Lord Nightshade"])
    w.status.connect(lambda *_: None)
    w.finished_ok.connect(lambda *_: None)
    w.run()
    assert seen["pool"], "catalog boss lost its spell pool on the way in"
    assert seen["dmg"] == 0               # its damage IS the pool
    assert seen["pips"] >= 1
    assert seen["hp"] == 690 and seen["school"] == "death"


def test_enemy_pips_are_read_not_zeroed():
    """The client reports every member's pip rack; the read used to
    zero it for enemies -- a real observation thrown away. "The boss
    has six pips" is the difference between shielding this round and
    shielding after the Wraith lands."""
    import asyncio

    from data_full import load_spells_full
    from deimos_bridge.live_backend import NameResolver
    from deimos_bridge.live_state import read_state
    from deimos_bridge.mock_client import MockCard, MockCombat, MockMember

    cards = load_spells_full()
    me = MockMember("W", 800, client=True, normal_pips=2)
    foe = MockMember("Lord Nightshade", 690, monster=True,
                     normal_pips=4, power_pips=1)
    combat = MockCombat([me, foe], [MockCard("Frost Beetle")])
    read = asyncio.new_event_loop().run_until_complete(
        read_state(combat, NameResolver(cards), "ice"))
    e = read.state.enemies[0]
    assert (e.norm_pips, e.pow_pips) == (4, 1)


def test_named_enemies_cast_in_the_rollouts():
    """`_apply_pool` puts the catalog spell pool on a named read enemy
    so rollouts price its actual casts against its actual pips, with
    the measured flat hit kept underneath as the fallback. Measured in
    belief_probe.py: +8 to +10 points of kill rate on hitter pools,
    a wash inside noise on debuffer pools."""
    from data_full import load_spells_full
    from deimos_bridge.live_backend import WizAiBackend
    from w101_sim import Actor, State

    cards = load_spells_full()
    me = Actor(name="W", school="ice", hp=800, max_hp=800, team=0)

    def read_for(actor):
        class _Read:
            state = State(me, [actor])
        return _Read()

    be = WizAiBackend(policy=lambda sim, s: None, cards=cards,
                      school="ice")
    boss = Actor(name="Lord Nightshade", school="death", hp=690,
                 max_hp=690, team=1, flat_hit=136.0)
    be._apply_pool(read_for(boss))
    assert boss.spell_pool                # it casts in rollouts now
    assert boss.archetype == "debuffer"
    # rank-scaled pip economy: the 690 HP Nightshade is rank 3, and
    # rank 3 measured ~no power pips (the Alicane calibration)
    from deimos_bridge.bestiary import full_boss
    assert boss.power_pip_chance == full_boss("Lord Nightshade",
                                              690).pip_chance
    assert boss.flat_hit == 136.0         # the fallback stays measured

    # An unknown mob keeps the flat model untouched.
    nobody = Actor(name="No Such Creature XYZ", school="fire", hp=300,
                   max_hp=300, team=1, flat_hit=90.0)
    be._apply_pool(read_for(nobody))
    assert nobody.spell_pool is None

    # A pool the card table cannot resolve to a single damage spell is
    # NOT stamped: `_enemy_choose` would find nothing to cast and the
    # boss would deal zero all rollout -- worse than the flat model.
    be_bare = WizAiBackend(policy=lambda sim, s: None, cards={},
                           school="ice")
    boss2 = Actor(name="Lord Nightshade", school="death", hp=690,
                  max_hp=690, team=1, flat_hit=136.0)
    be_bare._apply_pool(read_for(boss2))
    assert boss2.spell_pool is None

    # The knob turns it off wholesale.
    be.use_pool_model = False
    boss3 = Actor(name="Lord Nightshade", school="death", hp=690,
                  max_hp=690, team=1)
    be._apply_pool(read_for(boss3))
    assert boss3.spell_pool is None


def test_the_board_panel_shows_the_threat_model(qapp):
    """Whether the policy shields before the spike or races through the
    drizzle hangs on which offence model priced the mob -- the catalog
    caster with its read pips, or the measured flat hit -- and neither
    was visible anywhere in the window."""
    from deimos_bridge.gui.panels import BoardPanel
    from deimos_bridge.live_backend import PolicyDecision
    from deimos_bridge.telemetry import Telemetry
    from w101_sim import Actor, State

    tel = Telemetry()
    tel.start_fight()
    me = Actor(name="W", school="ice", hp=800, max_hp=800, team=0)
    caster = Actor(name="Lord Nightshade", school="death", hp=690,
                   max_hp=690, team=1, norm_pips=4, pow_pips=1,
                   spell_pool=["Banshee", "Ghoul", "Dark Sprite",
                               "Death Trap"])
    drizzle = Actor(name="minion", school="fire", hp=300, max_hp=300,
                    team=1, norm_pips=2, flat_hit=136.0)

    class _Read:
        state = State(me, [caster, drizzle])
        round_number = 1
        hand_cards = {}
        resolver = type("R", (), {"misses": set()})()
        hidden = []
        hand_visibility = 1.0

    rec = tel.observe(PolicyDecision(passing=True, reason="x"), _Read())
    assert rec.enemies[0].threat == "4+1p · casts Banshee, Ghoul, " \
                                    "Dark Sprite…"
    assert rec.enemies[1].threat == "2p · ~136/round"

    panel = BoardPanel(tel)
    panel.render(rec)
    assert "casts Banshee" in panel.enemies.item(0, 2).text()
    assert "~136/round" in panel.enemies.item(1, 2).text()


def test_probe_boards_can_carry_named_casters():
    """The 4-tuple probe form builds the catalog casting boss at the
    probe's health; unnamed slots and unknown names stay the flat mob.
    (The GUI deliberately does NOT use this -- caster probes measured
    equivalent to flat ones, see the _probe_mobs docstring -- but the
    hook must keep working for the day a per-boss tuner needs it.)"""
    from deimos_bridge.policies import _probe_mobs

    boss, extra = _probe_mobs((800, 2, "death", ["Lord Nightshade"]), 90)
    assert boss.pool and boss.dmg == 0 and boss.hp == 800
    assert len(extra) == 1 and extra[0].pool is None and extra[0].dmg == 90

    flat, extra = _probe_mobs((500, 1, "storm"), 55)
    assert flat.pool is None and flat.dmg == 55

    unknown, _ = _probe_mobs((500, 1, "storm", ["No Such Creature XYZ"]),
                             55)
    assert unknown.pool is None and unknown.dmg == 55


def test_the_tune_worker_probes_the_observed_fight(qapp, monkeypatch):
    """The auto-tuner measures the quartet on the fight actually being
    farmed: the observed board and a 0.7x cousin (probes at a single
    ceiling rank nothing), under the measured incoming damage."""
    from deimos_bridge import policies
    from deimos_bridge.gui import app as app_mod

    seen = {}

    def fake_choose_search(cards, deck, school, boards, n=60, dmg=0,
                           **kw):
        seen.update(deck=list(deck), school=school,
                    boards=list(boards), dmg=dmg)
        return "nuke-asap", 6, {"nuke-asap@h6": 0.8}

    monkeypatch.setattr(policies, "choose_search", fake_choose_search)
    got = []
    w = app_mod.TuneWorker("ice", ["Frost Beetle"] * 4, [690, 300],
                           ["death", "fire"], 136)
    w.tuned.connect(lambda wire, scores: got.append((wire, scores)))
    w.failed.connect(lambda m: got.append(("FAILED", m)))
    w.run()
    assert got and got[0][0] == "nuke-asap @ horizon 6 @ driver ttk"
    assert seen["boards"] == [(690, 2, "death"), (482, 2, "death")]
    assert seen["dmg"] == 136 and seen["school"] == "ice"


def test_a_fight_on_an_untuned_deck_tunes_itself(qapp, monkeypatch):
    """A wizard who connects and just fights played the untuned
    defaults indefinitely -- the quartet was only ever picked during a
    train. The first finished fight on an untuned deck now starts the
    tuner in the background; a deck the train already tuned does not."""
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    started = []
    monkeypatch.setattr(app_mod.TuneWorker, "start",
                        lambda self, *a, **k: started.append(self))

    win = MainWindow(Telemetry())
    win.deck.setText("Frost Beetle,Snow Serpent")
    win.observed_hps = [690]
    win.observed_schools = ["death"]
    win.observed_incoming = 136

    class _Live:
        def isRunning(self):
            return True

        policy_name = "ttk-lookahead"
        swapped = []

        def set_policy(self, name, agent=None):
            self.swapped.append(name)
            return True

    # Not connected: nothing fires.
    win._maybe_autotune()
    assert not started

    win.live = _Live()
    win._maybe_autotune()
    assert len(started) == 1
    assert started[0].deck == ["Frost Beetle", "Snow Serpent"]
    assert started[0].mob_damage == 136

    # The tuned result installs, and the same deck never re-tunes.
    win.on_autotuned("nuke-asap @ horizon 6 @ driver ttk", {})
    assert win.continuation == "nuke-asap @ horizon 6 @ driver ttk"
    assert win.live.swapped == ["ttk-lookahead"]
    win._autotune = None                  # the thread object is done
    win._maybe_autotune()
    assert len(started) == 1              # tuned deck: no second run


def test_the_whole_caster_chain_survives_a_real_decide(qapp):
    """End to end through the genuine backend and the mock client: a
    named boss holding pips must reach the policy as a caster holding
    those pips. The pieces are unit-tested; this guards the CHAIN --
    read_state keeps the enemy rack, _estimate_incoming leaves the
    measured fallback, _apply_pool stamps the catalog pool -- in the
    order decide() actually runs them."""
    import asyncio

    from deimos_bridge.mock_client import MockCard, MockCombat, MockMember

    seen = {}

    def spy_policy(sim, s):
        e = s.enemies[0]
        seen.update(pool=e.spell_pool, pips=(e.norm_pips, e.pow_pips),
                    flat=e.flat_hit, arch=e.archetype)
        return None

    be = _real_backend()
    be.set_policy(spy_policy, "spy")
    be.attach_combat(MockCombat(
        [MockMember("Wizard", 800, client=True, team_id=0, normal_pips=2),
         MockMember("Lord Nightshade", 690, monster=True, team_id=1,
                    normal_pips=4, power_pips=1)],
        [MockCard("Frost Beetle")]))
    d = asyncio.run(be.decide())
    assert d.passing                       # the spy passes; that is fine
    assert seen["pool"], "the catalog pool never reached the policy"
    assert seen["pips"] == (4, 1)
    assert seen["flat"] > 0                # measured fallback intact
    assert seen["arch"] == "debuffer"


def test_the_catalog_knows_the_whole_encounter():
    """419 bosses carry the scraped names of the creatures that fight
    beside them; `full_encounter` resolves the fight a deck can be
    built for BEFORE ever walking in."""
    from deimos_bridge.bestiary import full_encounter

    found = full_encounter("Malificus Mangemort")
    assert found is not None
    boss, rest = found
    assert boss.name == "Malificus Mangemort" and boss.pool
    assert rest and all(r.hp > 0 for r in rest)

    assert full_encounter("No Such Creature XYZ") is None


def test_build_deck_by_boss_name(qapp, monkeypatch):
    """A typed boss name builds for the catalog encounter -- the
    casting boss plus its companions -- instead of the measured board;
    a name the catalog does not know fails with a message rather than
    silently building for the wrong fight."""
    import deck_builder
    from deimos_bridge.gui.app import DeckWorker

    seen = {}

    def fake_build_deck(cards, school, boss, enemies=None, **kw):
        seen.update(name=boss.name, pool=boss.pool,
                    n_extra=len(enemies or []))
        return ["Frost Beetle"] * 4, 0.9, 5.0, []

    monkeypatch.setattr(deck_builder, "build_deck", fake_build_deck)
    # Observed board present but IGNORED: the named fight wins.
    w = DeckWorker({}, "ice", 1022, {}, [690], ["death"], 136, 0, 1,
                   mob_names=["Lord Nightshade"],
                   encounter_name="Malificus Mangemort")
    w.status.connect(lambda *_: None)
    w.finished_ok.connect(lambda *_: None)
    w.run()
    assert seen["name"] == "Malificus Mangemort"
    assert seen["pool"] and seen["n_extra"] >= 1

    failures = []
    w2 = DeckWorker({}, "ice", 1022, {}, [], [], 0, 0, 1,
                    encounter_name="No Such Creature XYZ")
    w2.status.connect(lambda *_: None)
    w2.failed.connect(failures.append)
    w2.run()
    assert failures and "not in the catalog" in failures[0]


def test_the_boss_name_field_reaches_the_deck_worker(qapp, monkeypatch):
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    monkeypatch.setattr(app_mod.DeckWorker, "start",
                        lambda self, *a, **k: None)
    win = MainWindow(Telemetry())
    win.boss_name.setText("  Lord Nightshade  ")
    win.on_build_deck()
    assert win.deck_worker.encounter_name == "Lord Nightshade"


def test_the_envelope_is_the_same_band_every_run(qapp):
    """Band edges are stamped onto the trained table and quoted back at
    the operator ("above the 40-1,500 band this table was trained on");
    edges that wobbled with each run's evaluation luck made those
    messages disagree between sessions about what the same deck could
    clear. Probes are seeded per (count, hp): same deck, same settings,
    same bands."""
    from data_full import load_spells_full
    from deimos_bridge.gui.app import TrainWorker

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 4 + ["Snow Serpent"] * 4
    w = TrainWorker(cards, deck, "ice", 10, player_hp=1022, boss_hp=690,
                    n_enemies=1, mob_schools=["death"])
    a = w.envelope(dmg=55, n=60)
    b = w.envelope(dmg=55, n=60)
    assert a and a == b


def test_the_sweep_never_installs_what_it_is_measuring():
    """The GUI runs choose_search while the live fight keeps playing,
    and the sweep used to install each candidate continuation globally
    to measure it -- so a live decision landing mid-sweep rolled out
    with whatever probe setting happened to be under measurement.
    Candidates are passed explicitly now; only the winner is installed,
    once, at the end."""
    from data_full import load_spells_full
    from deimos_bridge import policies as P

    cards = load_spells_full()
    deck = ["Frost Beetle"] * 3 + ["Snow Serpent"] * 3
    P.set_continuation("nuke-asap")
    seen = []
    try:
        picked, _horizon, _scores = P.choose_search(
            cards, deck, "ice", [(300, 1, "death")], n=6, dmg=40,
            on_probe=lambda k, v: seen.append(P.continuation_name()))
        assert seen and all(nm == "nuke-asap" for nm in seen)
        assert P.continuation_name() == picked    # winner, installed once
    finally:
        P.set_continuation(P.DEFAULT_CONTINUATION)
        P.set_search_horizon(None)
        P.set_driver("ttk")


# --------------------------------------------------------------- four wizards
def test_the_window_offers_up_to_four_wizards(qapp):
    """Four is the game's own limit — a battle circle seats four — so it
    is the ceiling rather than a chosen one."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    assert win.wizards.minimum() == 1 and win.wizards.maximum() == 4
    assert win.wizards.value() == 1


def test_the_party_controls_stay_out_of_the_way_of_one_wizard(qapp):
    """A window driving one client must look exactly like the window that
    always drove one client — no selector, no Party tab."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    assert win.which.isHidden()
    assert not win.tabs.isTabVisible(win.party_tab)

    win.wizards.setValue(3)
    assert win.which.count() == 3
    assert not win.which.isHidden()
    assert win.tabs.isTabVisible(win.party_tab)

    win.wizards.setValue(1)
    assert win.which.count() == 1
    assert win.which.isHidden()
    assert not win.tabs.isTabVisible(win.party_tab)


def test_each_wizard_keeps_its_own_school_deck_and_policy(qapp):
    """Four identical wizards is the one party worth nothing. Switching
    the selector must not carry wizard 1's deck onto wizard 2."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.wizards.setValue(2)
    win.school.setCurrentText("death")
    win.deck.setText("Dark Sprite,Dark Sprite")

    win.which.setCurrentIndex(1)
    assert win.deck.text() == ""             # a fresh wizard, not a copy
    win.school.setCurrentText("storm")
    win.deck.setText("Thunder Snake")

    win.which.setCurrentIndex(0)
    assert win.school.currentText() == "death"
    assert win.deck.text() == "Dark Sprite,Dark Sprite"

    seats = win.seat_configs_now()
    assert [s["school"] for s in seats] == ["death", "storm"]
    assert seats[1]["deck"] == ["Thunder Snake"]


def test_the_tabs_follow_the_selected_wizard(qapp):
    """One selector governs the boxes AND the tabs. Two would let the
    window show wizard 2's decisions beside wizard 1's deck."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.wizards.setValue(2)
    assert win.decisions.tel is win.tels[0]
    win.which.setCurrentIndex(1)
    assert win.decisions.tel is win.tels[1]
    assert win.board.tel is win.tels[1]
    assert win.runs.tel is win.tels[1]
    assert win.current_tel() is win.tels[1]
    # ...and wizard 1's record is untouched, not retargeted away.
    assert win.tel is win.tels[0]


def test_each_wizard_has_its_own_trained_table_and_gear(qapp):
    """A Q table is keyed on its own decklist and gear is read per
    client, so neither can be shared even between two ice wizards."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.wizards.setValue(2)
    sentinel = object()
    win.agent = sentinel
    win.on_gear_read({"damage": {"ice": 0.09}}, seat=0)
    win.on_gear_read({"damage": {"storm": 0.31}}, seat=1)

    assert win.agent is sentinel
    assert win.player_stats == {"damage": {"ice": 0.09}}
    win.which.setCurrentIndex(1)
    assert win.agent is None
    assert win.player_stats == {"damage": {"storm": 0.31}}


def test_a_health_read_lands_only_in_its_own_wizards_box(qapp):
    """The box drives training. Wizard 3's 2,100 arriving while wizard 1
    is shown would train wizard 1's deck against wizard 3's health --
    the exact mismatch reading it off the client exists to prevent."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.wizards.setValue(3)
    win.on_hp_read(1337, seat=0)
    assert win.player_hp.value() == 1337
    win.on_hp_read(2100, seat=2)
    assert win.player_hp.value() == 1337
    win.which.setCurrentIndex(2)
    win.on_hp_read(2100, seat=2)
    assert win.player_hp.value() == 2100


def test_starting_a_party_hands_every_wizard_its_own_seat(qapp, monkeypatch):
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    monkeypatch.setattr(app_mod.LiveWorker, "start", lambda self, *a: None)
    win = MainWindow(Telemetry())
    win.wizards.setValue(3)
    win.school.setCurrentText("fire")
    win.deck.setText("Fire Cat")
    win.which.setCurrentIndex(1)
    win.school.setCurrentText("ice")
    win.deck.setText("Frost Beetle")
    win.which.setCurrentIndex(2)
    win.school.setCurrentText("storm")
    win.deck.setText("Thunder Snake")
    win.which.setCurrentIndex(0)

    win.on_start_live()
    seats = win.live.seats
    assert [s.school for s in seats] == ["fire", "ice", "storm"]
    assert [s.deck for s in seats] == [["Fire Cat"], ["Frost Beetle"],
                                       ["Thunder Snake"]]
    # Each wizard fills its own record, or a round settles its damage
    # against another wizard's board.
    assert [s.tel for s in seats] == [win.tels[0], win.tels[1], win.tels[2]]
    assert len({id(s.tel) for s in seats}) == 3


def test_a_party_refuses_to_start_for_the_wizard_that_has_no_table(
        qapp, monkeypatch):
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    said = {}
    monkeypatch.setattr(app_mod.QMessageBox, "warning",
                        lambda *a, **k: said.setdefault("text", a[2]))
    monkeypatch.setattr(app_mod.LiveWorker, "start", lambda self, *a: None)
    win = MainWindow(Telemetry())
    win.wizards.setValue(2)
    win.which.setCurrentIndex(1)
    win.policy.setCurrentText("trained (Q)")
    win.which.setCurrentIndex(0)

    win.on_start_live()
    assert win.live is None
    assert "Wizard 2" in said["text"], said


def test_the_worker_builds_a_hivemind_only_for_a_party(qapp):
    """A hive of one costs a barrier, a plan and a copy of the board
    every round and buys nothing a single wizard could not decide."""
    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    alone = LiveWorker(Telemetry(), "ice", [], "ttk-lookahead", 1)
    assert alone._make_hive() is None

    pair = LiveWorker(Telemetry(), "ice", [], "ttk-lookahead", 1,
                      seats=[SeatConfig(school="storm", deck=[],
                                        policy_name="ttk-lookahead")])
    hive = pair._make_hive()
    assert hive is not None and hive.size == 2


def test_seat_zero_is_still_reachable_straight_off_the_worker(qapp):
    """Everything written against the single-wizard worker keeps working:
    the seat-0 fields are the worker's own, not a second copy that can
    drift out of step."""
    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    tel = Telemetry()
    w = LiveWorker(tel, "ice", ["Frost Beetle"], "school-aware", 1,
                   seats=[SeatConfig(school="fire", deck=["Fire Cat"])])
    assert w.tel is tel and w.school == "ice" and w.deck == ["Frost Beetle"]
    w.school = "myth"
    assert w.seats[0].school == "myth"
    assert w.seats[1].school == "fire"
    assert w.party == 2


def test_a_button_press_reaches_every_wizard(qapp):
    """One 'collect wisps' is meant to sweep the whole party's wisps, not
    one quarter of them."""
    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                   seats=[SeatConfig(school="fire")])
    assert w.request("wisps") is True
    assert [s.requests for s in w.seats] == [["wisps"], ["wisps"]]
    assert w.request("wisps") is False          # still deduped, per seat


def test_a_seat_swaps_only_its_own_policy(qapp):
    """Four wizards in a circle are meant to play differently; that is
    most of what makes a party worth more than one wizard four times."""
    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    w = LiveWorker(Telemetry(), "ice", ["Frost Beetle"] * 4, "school-aware", 1,
                   seats=[SeatConfig(school="fire", deck=["Fire Cat"] * 4,
                                     policy_name="school-aware")])
    assert w.set_policy("ttk-lookahead", seat=1) is True
    assert w.seats[1].policy_name == "ttk-lookahead"
    assert w.seats[0].policy_name == "school-aware"


def test_a_party_exports_one_file_per_wizard(qapp, monkeypatch, tmp_path):
    """Interleaving four wizards' rounds into one file makes every
    residual in it meaningless: a round settles its damage against the
    board that wizard was shown."""
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    target = tmp_path / "run.json"
    monkeypatch.setattr(app_mod.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(target), ""))
    win = MainWindow(Telemetry())
    win.wizards.setValue(2)
    win.on_export()
    assert (tmp_path / "run-wizard1.json").exists()
    assert (tmp_path / "run-wizard2.json").exists()
    assert not target.exists()


def test_the_party_panel_reads_a_plan(qapp):
    from deimos_bridge.gui.app import MainWindow
    from deimos_bridge.hivemind import PartyPlan, SeatMove

    win = MainWindow(Telemetry())
    win.wizards.setValue(2)
    win.on_party_plan(PartyPlan(
        round_number=3,
        board=[("Lost Soul", 40.0, 450.0)],
        moves=[SeatMove(seat=0, name="wizard 1", card="Fire Cat", target=0,
                        target_name="Lost Soul", solo_card="Fire Cat",
                        solo_target=0, damage=51.0),
               SeatMove(seat=1, name="wizard 2", note="held",
                        solo_card="Fire Cat", solo_target=0)],
        saved=1, passes=2, seconds=0.12))
    assert win.party.table.rowCount() == 2
    assert "round 3" in win.party.headline.text()
    assert "held" in win.party.headline.text()
    assert "Lost Soul" in win.party.board_lab.text()


def test_two_wizards_tuned_differently_do_not_overwrite_each_other(qapp):
    """The quartet is deck-scoped and lives in module globals that
    `_rollout` reads at DECISION time. Four wizards holding four decks
    would therefore all play whichever pick was installed last — so a
    seat's quartet is bound into its own closure instead, and nothing is
    installed globally at all."""
    from data_full import load_spells_full
    from deimos_bridge import policies as P
    from deimos_bridge.gui.live import LiveWorker, SeatConfig
    from w101_sim import Actor, Boss, Rules, Sim, State

    cards = load_spells_full()
    worker = LiveWorker(
        Telemetry(), "fire", ["Fire Cat"], "ttk-lookahead", 1,
        continuation="nuke-asap @ horizon 6 @ driver ttk",
        seats=[SeatConfig(school="fire", deck=["Fire Cat"],
                          policy_name="ttk-lookahead",
                          continuation="school-aware(3) @ horizon 12 "
                                       "@ driver ttk")])
    before = (P.continuation_name(), P.search_horizon(), P.driver_name())
    policies = [worker._build_policy(seat) for seat in worker.seats]

    def horizon_seen(policy):
        player = Actor(name="W", school="fire", hp=900, max_hp=900, team=0,
                       norm_pips=6)
        player.hand = [cards["Fire Cat"]]
        foe = Actor(name="Mob", school="ice", hp=400, max_hp=400, team=1)
        sim = Sim(cards=cards, decklist=["Fire Cat"], school="fire",
                  boss=Boss(name="Mob", hp=400, school="ice", dmg=40),
                  rules=Rules(), player_hp=900)
        policy(sim, State(player, [foe]))
        return {c.horizon for c in policy.last_candidates}

    assert horizon_seen(policies[0]) == {6}
    assert horizon_seen(policies[1]) == {12}
    # ...and neither seat reached for the globals to get there.
    assert (P.continuation_name(), P.search_horizon(),
            P.driver_name()) == before


def test_only_the_followers_chase_the_leader(qapp):
    """The leader quests; a leader that also chased itself would stand
    still forever, and one wizard has nobody to follow."""
    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    w = LiveWorker(Telemetry(), "ice", [], "ttk-lookahead", 1,
                   seats=[SeatConfig(school="fire"),
                          SeatConfig(school="storm")])
    assert [w._follows(s) for s in w.seats] == [False, True, True]

    w.follow_leader = False
    assert not any(w._follows(s) for s in w.seats)

    alone = LiveWorker(Telemetry(), "ice", [], "ttk-lookahead", 1)
    assert alone._follows(alone.seats[0]) is False


def test_the_follow_checkbox_only_appears_with_a_party(qapp):
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    assert win.follow_leader.isHidden()
    win.wizards.setValue(2)
    assert not win.follow_leader.isHidden()
    assert win.follow_leader.isChecked()


def test_the_follow_choice_reaches_the_worker(qapp, monkeypatch):
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    monkeypatch.setattr(app_mod.LiveWorker, "start", lambda self, *a: None)
    win = MainWindow(Telemetry())
    win.wizards.setValue(2)
    win.follow_leader.setChecked(False)
    win.on_start_live()
    assert win.live.follow_leader is False


def test_the_leaders_name_is_learned_from_its_first_duel(qapp):
    """The client only offers it on the character-select screen, which a
    running wizard is not on — but every combat read already carries it,
    and the cross-zone follow cannot pick a leader out of the friends
    list without it."""
    from deimos_bridge.gui.live import LiveWorker

    w = LiveWorker(Telemetry(), "ice", [], "ttk-lookahead", 1)
    seat = w.seats[0]
    assert seat.wizard_name is None

    class _Player:
        name = "Wolf Deathblade"

    class _State:
        player = _Player()

    class _Read:
        state = _State()

    w._learn_name(seat, _Read())
    assert seat.wizard_name == "Wolf Deathblade"


def test_every_wizard_in_the_party_tunes_its_own_search(qapp, monkeypatch):
    """The quartet is deck-scoped and worth ~14 points of kill rate.
    Tuning only whichever wizard happens to be selected leaves the other
    three playing the untuned defaults for the whole run, on the exact
    boards the run is measuring for them."""
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    started = []
    monkeypatch.setattr(app_mod.TuneWorker, "start",
                        lambda self, *a, **k: started.append(self))

    win = MainWindow(Telemetry())
    win.wizards.setValue(3)
    win.school.setCurrentText("fire")
    win.deck.setText("Fire Cat")
    win.which.setCurrentIndex(1)
    win.school.setCurrentText("ice")
    win.deck.setText("Frost Beetle")
    win.which.setCurrentIndex(2)
    win.school.setCurrentText("storm")
    win.deck.setText("Thunder Snake")
    win.which.setCurrentIndex(0)

    win.observed_hps = [690]
    win.observed_schools = ["death"]
    win.observed_incoming = 136

    class _Live:
        seats = [object(), object(), object()]

        def isRunning(self):
            return True

    win.live = _Live()
    win._maybe_autotune()
    assert len(started) == 3
    assert [w.school for w in started] == ["fire", "ice", "storm"]
    assert [w.deck for w in started] == [["Fire Cat"], ["Frost Beetle"],
                                         ["Thunder Snake"]]


def test_a_tuned_result_lands_on_the_wizard_that_asked_for_it(qapp):
    """The sweep takes about a minute. Landing its answer on whichever
    wizard is selected when it finishes would hand wizard 3's pick to
    wizard 1 -- and the pick is deck-scoped, so that is worse than not
    tuning at all."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.wizards.setValue(2)
    win.on_autotuned("nuke-asap @ horizon 6 @ driver ttk", {}, 1)
    assert win.continuations[1] == "nuke-asap @ horizon 6 @ driver ttk"
    assert win.continuations[0] == ""
    assert win.continuation == ""            # wizard 1 is still showing


def test_a_party_member_is_not_trained_against_four_times_the_damage(qapp):
    """An enemy picking one of four wizards to hit lands on this one a
    quarter of the time. The measured number knows that -- it is read off
    this wizard's own health bar -- but the stand-in did not, so a party
    member with no fight measured yet trained against a board hitting
    four times as hard as the one it plays."""
    from deimos_bridge.gui.app import TrainWorker

    solo = TrainWorker({}, [], "ice", 500, player_hp=2000)
    party = TrainWorker({}, [], "ice", 500, player_hp=2000, party_size=4)
    assert party.enemy_damage() < solo.enemy_damage()
    assert party.enemy_damage() == max(30, 2000 // 48)

    # ...but a measured number is already this wizard's share, so it is
    # taken as it is rather than divided twice.
    measured = TrainWorker({}, [], "ice", 500, player_hp=2000,
                           mob_damage=77, party_size=4)
    assert measured.enemy_damage() == 77


def test_the_deck_ceiling_refusal_asks_for_this_wizards_share(qapp):
    """Three other wizards are hitting the same board. Refusing to train
    a deck because it cannot deliver all of a board's health alone is a
    refusal to the question the simulator can ask rather than the one
    being played."""
    from deimos_bridge.gui.app import TrainWorker
    from data_full import load_spells_full
    from w101_sim import Boss

    cards = load_spells_full()
    deck = ["Fire Cat"] * 4          # about 400 damage, all told

    def refusal(hp, party_size):
        worker = TrainWorker(cards, deck, "fire", 500, player_hp=900,
                             party_size=party_size)
        return worker.preflight(
            Boss(name="Wall", hp=hp, school="ice", dmg=40), [], n=20)[1]

    # Too big for the party's share as well: still the deck's fault, but
    # the number it is measured against — and the advice sized off it —
    # is this wizard's share rather than the whole board.
    assert "Your deck is the reason" in refusal(2400, 1)
    party = refusal(2400, 4)
    assert "Your deck is the reason" in party
    assert "share of it, across 4 wizards, is about 600" in party

    # Inside the share: no longer the deck's fault at all, where solo it
    # squarely was.
    assert "Your deck is the reason" in refusal(1200, 1)
    assert "Your deck is the reason" not in refusal(1200, 4)


def test_a_party_is_told_the_training_board_models_one_wizard(qapp):
    """The trained table is pessimistic about the fight -- the safe
    direction, but not a free one: it will not learn to leave a mob to
    somebody else, and nothing on screen said so."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    assert "models ONE wizard" not in win._board_line()
    win.wizards.setValue(4)
    line = win._board_line()
    assert "models ONE wizard" in line
    assert "other 3 wizards" in line


def test_a_follower_does_not_chase_twice_a_second(qapp):
    """The service tick runs at 2Hz and a follow is not a cheap read: it
    teleports, and against a leader mid-duel it also reaches for the
    nearest mob. A follower that cannot get in — the circle already
    seats four — would retry that for the length of the fight."""
    import asyncio

    from deimos_bridge import party as party_mod
    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    w = LiveWorker(Telemetry(), "ice", [], "ttk-lookahead", 1,
                   seats=[SeatConfig(school="fire")])
    w.seats[0].client = object()
    follower = w.seats[1]
    follower.client = object()

    tried = []

    async def _follow(f, leader, leader_name=None, radius=0.0):
        tried.append(f)
        return False, ""

    real, party_mod.follow = party_mod.follow, _follow
    try:
        async def drive():
            for _ in range(6):
                await w._follow_step(follower.client, follower)
        asyncio.run(drive())
    finally:
        party_mod.follow = real

    assert len(tried) == 1, "six ticks must not be six teleports"


def test_a_crossed_seat_is_caught_from_the_first_round(qapp):
    """From the first live party run: wizard 1 was configured `fire` and
    held a pure ice hand laying +40% ice traps, while wizard 2 was
    configured `ice` and held Fireblade. `get_new_clients()` returns
    windows in whatever order it finds them and nothing else can tell
    which is which, so the seats and the clients were crossed.

    It is expensive and silent: the gear read asks for the wrong
    school's damage stat, so `player.damage_bonus` comes back keyed on a
    school this wizard never casts and every hit is priced at no gear at
    all."""
    import asyncio

    from deimos_bridge.live_backend import WizAiBackend
    from deimos_bridge.mock_client import MockCard, MockCombat, MockMember

    seen = []
    backend = WizAiBackend.from_trained(
        school="fire", deck=["Frost Beetle"], cards=_cards_once(),
        policy=lambda sim, s: None, policy_name="ttk-lookahead")
    backend.on_school_mismatch = seen.append
    backend.attach_combat(MockCombat(
        # the game's own id for Ice, out of `deimos_damage.SCHOOL_TO_STR`
        [MockMember("Wizard", 1022, client=True, team_id=0, normal_pips=1,
                    school_id=72777),
         MockMember("Lord Nightshade", 690, monster=True, team_id=1)],
        [MockCard("Frost Beetle")]))

    asyncio.run(backend.decide())
    assert seen == ["ice"], seen
    # Once, not once a round.
    asyncio.run(backend.decide())
    assert seen == ["ice"]


def test_an_unreadable_school_is_not_reported_as_a_mismatch(qapp):
    """A school that will not read is not evidence of anything, and a
    false 'you configured the wrong wizard' would send the operator
    chasing a setup problem that does not exist."""
    import asyncio

    from deimos_bridge.live_backend import WizAiBackend
    from deimos_bridge.mock_client import MockCard, MockCombat, MockMember

    seen = []
    backend = WizAiBackend.from_trained(
        school="fire", deck=["Fire Cat"], cards=_cards_once(),
        policy=lambda sim, s: None)
    backend.on_school_mismatch = seen.append
    backend.attach_combat(MockCombat(
        [MockMember("Wizard", 691, client=True, team_id=0, normal_pips=1),
         MockMember("Lost Soul", 450, monster=True, team_id=1)],
        [MockCard("Fire Cat")]))
    asyncio.run(backend.decide())
    assert seen == []


def test_the_worker_switches_the_seat_to_what_the_client_says(qapp):
    """The client is the authority: it is the one that knows which wizard
    is logged into it."""
    from deimos_bridge.gui.live import LiveWorker

    w = LiveWorker(Telemetry(), "fire", ["Fire Cat"], "ttk-lookahead", 1)
    seat = w.seats[0]
    seat.backend = type("B", (), {"school": "fire"})()
    said = []
    w.status = type("S", (), {"emit": staticmethod(said.append)})()

    w._on_school_mismatch("ice", seat)
    assert seat.school == "ice"
    assert seat.tel.school == "ice"
    assert seat.backend.school == "ice"
    assert any("is a ice wizard, not the fire" in m for m in said), said
    assert any("seats were crossed" in m for m in said), said


_CARDS_ONCE = None


def _cards_once():
    global _CARDS_ONCE
    if _CARDS_ONCE is None:
        from data_full import load_spells_full
        _CARDS_ONCE = load_spells_full()
    return _CARDS_ONCE


def test_the_export_says_which_wizard_it_is(qapp):
    """Three files called -wizard1/2/3 have to be identified by reading
    the hands and guessing, which is what the first party run's exports
    needed. The game names the wizard; the record should carry it."""
    tel = Telemetry(policy_name="ttk-lookahead", school="ice",
                    wizard="Wolf Deathblade", seat=1)
    s = tel.summary()
    assert s["wizard"] == "Wolf Deathblade"
    assert s["seat"] == 2

    # ...and falls back to the seat rather than to nothing.
    assert Telemetry(seat=2).summary()["wizard"] == "wizard 3"


def test_a_named_party_exports_named_files(qapp, monkeypatch, tmp_path):
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    target = tmp_path / "run.json"
    monkeypatch.setattr(app_mod.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(target), ""))
    win = MainWindow(Telemetry())
    win.wizards.setValue(2)
    win.tels[0].wizard = "Wolf Deathblade"
    win.on_export()
    assert (tmp_path / "run-wizard1-WolfDeathblade.json").exists()
    # The unnamed one keeps the seat, so nothing is lost when a wizard
    # has not fought yet.
    assert (tmp_path / "run-wizard2.json").exists()


def test_learning_the_name_relabels_everything_that_shows_it(qapp):
    """The window's selector, the record, the hive's plan and the game
    window's own title bar all name the same wizard, or the operator has
    to hold the mapping in their head."""
    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    w = LiveWorker(Telemetry(), "ice", [], "ttk-lookahead", 1,
                   seats=[SeatConfig(school="fire")])
    w.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    w.hive = w._make_hive()
    seat = w.seats[1]

    class _Client:
        title = "Wizard101"

    seat.client = _Client()

    class _Player:
        name = "Wolf Deathblade"

    class _Read:
        state = type("S", (), {"player": _Player()})()

    named = []
    w.seat_named.connect(lambda i, n: named.append((i, n)))
    w._learn_name(seat, _Read())

    assert seat.wizard_name == "Wolf Deathblade"
    assert seat.name == "Wolf Deathblade"
    assert seat.tel.wizard == "Wolf Deathblade"
    assert named == [(1, "Wolf Deathblade")]
    assert "Wolf Deathblade" in seat.client.title
    assert "wizAi 2" in seat.client.title
    # The plan the Party tab renders names it too.
    assert w.hive._seats[1] == "Wolf Deathblade"


def test_the_window_title_is_stamped_before_any_fight(qapp):
    """Half an answer now beats a whole one after the first duel: the
    operator is working out which window is which before it starts."""
    from deimos_bridge.gui.live import LiveWorker

    w = LiveWorker(Telemetry(), "storm", [], "ttk-lookahead", 1)
    seat = w.seats[0]

    class _Client:
        title = "Wizard101"

    seat.client = _Client()
    w._stamp_title(seat)
    assert seat.client.title == "wizAi 1 · wizard 1 · storm"

    w.label_windows = False
    seat.client.title = "Wizard101"
    w._stamp_title(seat)
    assert seat.client.title == "Wizard101"


def test_the_selector_carries_a_name_it_already_learned(qapp):
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.wizards.setValue(2)
    win.on_seat_named(1, "Wolf Deathblade")
    assert win.which.itemText(1) == "wizard 2 — Wolf Deathblade"
    # Naming a seat is not the user picking a different wizard.
    assert win._seat_showing == 0

    # Resizing the party rebuilds the list; a learned name survives it.
    win.tels[2].wizard = "Autumn Frost"
    win.wizards.setValue(3)
    assert win.which.itemText(2) == "wizard 3 — Autumn Frost"


def test_hooks_that_return_are_not_hooks_that_answer(qapp):
    """`activate_hooks()` returning is not the same as the hooks being
    up. On a live party run one client hooked and then would not answer
    for its own wizard's name or school — it reported every enemy's
    school on the same read — and the run carried on with a wizard it
    could not identify. The operator's only recourse was to hook and
    unhook until it took."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    class _Stats:
        async def max_hitpoints(self):
            return 1053

    class _Body:
        async def position(self):
            raise RuntimeError("not hooked yet")

    class _Client:
        stats = _Stats()
        body = _Body()
        activations = 0

        async def zone_name(self):
            return "Unicorn Way"

        async def activate_hooks(self):
            type(self).activations += 1
            # the retry is what fixes it, as it did for the operator
            _Client.body = type("B", (), {
                "position": staticmethod(lambda: _value((1, 2, 3)))})()

    w = LiveWorker(Telemetry(), "ice", [], "ttk-lookahead", 1)
    said = []
    w.status = type("S", (), {"emit": staticmethod(said.append)})()
    w.seats[0].client = _Client()

    ok, missing = asyncio.run(w._verify_hooks(w.seats[0], settle=0.0))
    assert ok is True and missing == ""
    assert _Client.activations == 1
    assert any("not answering yet (position)" in m for m in said), said
    assert any("answering now" in m for m in said), said


def test_a_hook_that_never_answers_is_named_not_hidden(qapp):
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    class _Client:
        async def activate_hooks(self):
            pass

        async def zone_name(self):
            raise RuntimeError("no")

        class stats:
            @staticmethod
            async def max_hitpoints():
                return 1053

        class body:
            @staticmethod
            async def position():
                return (1, 2, 3)

    w = LiveWorker(Telemetry(), "ice", [], "ttk-lookahead", 1)
    said = []
    w.status = type("S", (), {"emit": staticmethod(said.append)})()
    w.seats[0].client = _Client()

    ok, missing = asyncio.run(w._verify_hooks(w.seats[0], tries=2,
                                              settle=0.0))
    assert ok is False and missing == "zone"
    # Named, with what it costs, and it still plays.
    assert any("will not read" in m and "different zone" in m
               for m in said), said


def test_an_unnamed_record_is_still_claimed_by_health(qapp):
    """`_learn_name` cannot catch the case that actually happened: the
    first run never got a name, an empty name matches everything, and two
    rounds of a 1,053 HP ice wizard were exported under a 713 HP fire
    wizard's name."""
    from deimos_bridge.gui.live import LiveWorker
    from deimos_bridge.telemetry import RoundRecord

    def seeded(max_hp, wizard=""):
        tel = Telemetry()
        tel.wizard = wizard
        tel.start_fight()
        tel.rounds.append(RoundRecord(fight=1, round=1,
                                      player_max_hp=max_hp))
        return tel

    # The Konstantin case: unnamed 1,053 record, 713 client.
    tel = seeded(1053.0)
    w = LiveWorker(tel, "fire", [], "ttk-lookahead", 1)
    said = []
    w.status = type("S", (), {"emit": staticmethod(said.append)})()
    w.seats[0].max_hp = 713
    w._claim_record(w.seats[0])
    assert tel.rounds == []
    assert any("a different wizard" in m for m in said), said

    # A level or two later is the same wizard, and its fights are kept.
    tel = seeded(1053.0, wizard="Jeffrey")
    w = LiveWorker(tel, "ice", [], "ttk-lookahead", 1)
    w.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    w.seats[0].max_hp = 1130
    w._claim_record(w.seats[0])
    assert len(tel.rounds) == 1


async def _value(v):
    return v


def _hooking_worker(clients, **kw):
    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    w = LiveWorker(Telemetry(), "ice", [], "ttk-lookahead", 1,
                   seats=[SeatConfig(school="fire")
                          for _ in range(len(clients) - 1)], **kw)
    said = []
    w.status = type("S", (), {"emit": staticmethod(said.append)})()
    for seat, client in zip(w.seats, clients):
        seat.client = client
    return w, said


class _HookClient:
    """A client whose hooks only finish while it is in the foreground.

    Which is the real behaviour: wizwalker waits for addresses the game
    writes from its UI and render paths, and a background Wizard101
    client barely renders.
    """

    def __init__(self, needs_focus=True):
        self.needs_focus = needs_focus
        self.is_foreground = False
        self.attempts = 0

    async def activate_hooks(self, timeout=None):
        self.attempts += 1
        if self.needs_focus and not self.is_foreground:
            raise TimeoutError("Hook value took too long")


def test_hooking_the_second_client_does_not_hang_the_run(qapp):
    """`activate_hooks()` defaults to no timeout at all, so a client that
    never writes its render hook parks the whole run — with nothing to do
    but kill it. This is the bug behind 'I have to kill the bot and
    re-hook every time'."""
    import asyncio

    a, b = _HookClient(), _HookClient()
    w, said = _hooking_worker([a, b])
    asyncio.run(w._activate_all_hooks())

    # Each client was brought to the front for its own turn.
    assert a.attempts == 1 and b.attempts == 1
    assert a.is_foreground and b.is_foreground
    assert sum("activating hooks…" in m for m in said) == 2


def test_a_client_that_will_not_hook_says_what_to_do(qapp):
    import asyncio

    import pytest

    from deimos_bridge.gui.live import LiveWorker

    stuck = _HookClient()
    stuck.is_foreground = False
    w, said = _hooking_worker([stuck])
    # focus does nothing for this one: it is minimised, on another
    # desktop, or sitting on a loading screen
    w._focus = lambda seat: False
    with pytest.raises(RuntimeError, match="never finished writing"):
        asyncio.run(w._activate_all_hooks())

    assert stuck.attempts == 2, "it must try again before giving up"
    assert any("render loop" in m for m in said), said


def test_hooks_already_up_from_a_previous_run_are_not_an_error(qapp):
    import asyncio

    class _Already:
        is_foreground = False

        async def activate_hooks(self, timeout=None):
            raise type("HookAlreadyActivated", (Exception,), {})()

    w, _said = _hooking_worker([_Already()])
    asyncio.run(w._activate_all_hooks())      # must not raise


def test_the_operators_own_window_is_given_back(qapp):
    """Four clients yanked to the front in turn and left that way is a
    rude way to start a run. Off Windows there is no window handle to
    read, and the restore has to be a no-op rather than a crash."""
    import asyncio

    w, _said = _hooking_worker([_HookClient()])
    assert w._foreground_window() is None      # no wizwalker here
    w._restore_foreground(None)                # must not raise
    asyncio.run(w._activate_all_hooks())


def test_focus_stealing_can_be_turned_off(qapp):
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    client = _HookClient(needs_focus=False)
    w, _said = _hooking_worker([client])
    w.FOCUS_TO_HOOK = False
    asyncio.run(w._activate_all_hooks())
    assert client.is_foreground is False
    assert client.attempts == 1


def test_a_press_does_not_wait_behind_a_quest_hop(qapp):
    """The drain shared a tick with auto-dialogue, the script runner and
    the quest step. `advance_dialogue` clicks up to forty times at half a
    second each and a quest hop settles for 1.2s, so a press could sit
    unserviced for many seconds — during which `enqueue` refused every
    further press of that key with 'already running' while nothing was
    running. That is the dead teleport hotkey."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    class _Client:
        async def in_battle(self):
            return False

    done = []
    slow_started = asyncio.Event()

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1, auto_quest=True)
    w.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    seat = w.seats[0]
    client = _Client()
    seat.client = client

    async def drive():
        seat.drive = asyncio.Lock()

        async def slow_quest(_c):
            slow_started.set()
            await asyncio.sleep(5)          # a hop that takes its time

        w._quest_step = slow_quest
        w._do_request = lambda c, a, s=None: _record(done, a)

        service = asyncio.ensure_future(w._service_loop(client, seat))
        requests = asyncio.ensure_future(w._request_loop(client, seat))
        await asyncio.wait_for(slow_started.wait(), 2)
        w.request("teleport")               # pressed mid-hop
        for _ in range(20):
            await asyncio.sleep(0.05)
            if done:
                break
        w._stop = True
        for task in (service, requests):
            task.cancel()
            try:
                await task
            except BaseException:
                pass

    asyncio.run(drive())
    assert done == ["teleport"], "the press waited for the quest hop"


async def _record(seen, action):
    seen.append(action)


def test_two_things_never_steer_one_wizard_at_once(qapp):
    """The queue getting its own task is only safe because they share a
    lock: a quest hop and a wisp sweep both teleport, and interleaving
    them walks the wizard somewhere nobody asked for."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    class _Client:
        async def in_battle(self):
            return False

    overlap = []
    inside = []

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1, auto_quest=True)
    w.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    seat = w.seats[0]
    seat.client = client = _Client()

    async def steer(_c=None, *a, **kw):
        inside.append(1)
        overlap.append(len(inside))
        await asyncio.sleep(0.15)
        inside.pop()

    async def drive():
        seat.drive = asyncio.Lock()
        w._quest_step = steer
        w._do_request = lambda c, a, s=None: steer()
        service = asyncio.ensure_future(w._service_loop(client, seat))
        requests = asyncio.ensure_future(w._request_loop(client, seat))
        for _ in range(6):
            w.request("wisps")
            await asyncio.sleep(0.1)
        w._stop = True
        for task in (service, requests):
            task.cancel()
            try:
                await task
            except BaseException:
                pass

    asyncio.run(drive())
    assert overlap, "nothing ran at all"
    assert max(overlap) == 1, f"two coroutines steered at once: {overlap}"


def test_a_request_that_can_never_run_expires(qapp):
    """A queue entry that cannot be serviced is worse than no entry: the
    dedupe refuses every further press, so the key goes dead and the only
    way back is a combat cycle."""
    from deimos_bridge.gui.live import LiveWorker

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    said = []
    w.status = type("S", (), {"emit": staticmethod(said.append)})()
    seat = w.seats[0]

    assert w.request("teleport") is True
    assert w.request("teleport") is False      # deduped, as it should be

    seat.queued_at["teleport"] -= w.REQUEST_TTL + 1
    w._expire_requests(seat)

    assert seat.requests == []
    assert any("dropped the queued teleport" in m for m in said), said
    assert w.request("teleport") is True        # the key answers again


def test_every_run_toggle_reaches_a_running_fight(qapp, monkeypatch):
    """They were read once at Play live and never again, so auto-quest,
    auto-dialogue, the upkeep chores and the follow could not be turned
    on or off during a run — only by stopping and reconnecting."""
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    monkeypatch.setattr(app_mod.LiveWorker, "start", lambda self, *a: None)
    win = MainWindow(Telemetry())
    win.wizards.setValue(2)
    win.on_start_live()
    live = win.live
    monkeypatch.setattr(type(live), "isRunning", lambda self: True)

    assert live.auto_quest is False
    win.auto_quest.setChecked(True)
    assert live.auto_quest is True
    win.auto_quest.setChecked(False)
    assert live.auto_quest is False

    for box, attr in (("auto_dialogue", "auto_dialogue"),
                      ("collect_wisps", "collect_wisps"),
                      ("use_potions", "use_potions"),
                      ("follow_leader", "follow_leader")):
        getattr(win, box).setChecked(False)
        assert getattr(live, attr) is False, attr
        getattr(win, box).setChecked(True)
        assert getattr(live, attr) is True, attr


def test_the_script_can_be_started_and_stopped_mid_run(qapp, monkeypatch):
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    monkeypatch.setattr(app_mod.LiveWorker, "start", lambda self, *a: None)
    win = MainWindow(Telemetry())
    win.script_source = "sendkey W 1"
    win.on_start_live()
    live = win.live
    monkeypatch.setattr(type(live), "isRunning", lambda self: True)

    assert live.script == ""                 # the box was unticked
    win.use_script.setChecked(True)
    assert live.script == "sendkey W 1"
    win.use_script.setChecked(False)
    assert live.script == ""


def test_a_worker_notices_the_script_being_switched_on(qapp):
    """The runner is built on the worker's own loop, because building one
    takes the client and the client belongs to that loop."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    w.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    seat = w.seats[0]
    seat.client = object()

    built = []

    async def _setup(client, s=None):
        s = s or seat
        s.runner = type("R", (), {"stop": lambda self: stopped.append(1)})()
        built.append(w.script)

    stopped = []
    w._setup_script = _setup

    # nothing configured: nothing built, and no churn on repeat ticks
    asyncio.run(w._sync_script(seat))
    asyncio.run(w._sync_script(seat))
    assert built == [] and seat.runner is None

    # switched on mid-run
    w.script = "sendkey W 1"
    asyncio.run(w._sync_script(seat))
    assert built == ["sendkey W 1"] and seat.runner is not None
    asyncio.run(w._sync_script(seat))
    assert built == ["sendkey W 1"], "an unchanged script must not rebuild"

    # ...and off again
    w.script = ""
    asyncio.run(w._sync_script(seat))
    assert seat.runner is None and stopped == [1]


def test_the_friends_list_holds_a_full_name_and_a_duel_gives_a_first(qapp):
    """wizwalker matches `friend_name == name`, exactly. A duel reports
    "Jeffrey"; the list holds "Jeffrey IslandBringer"; the teleport could
    never find a leader that was sitting right there on the list."""
    import asyncio

    from deimos_bridge import party

    party._FULL_NAMES.clear()
    tried = []

    async def _tp(follower, name=None, **kw):
        tried.append(name)
        if name != "Jeffrey IslandBringer":
            raise ValueError(
                f"Could not find friend with icon None icon list None "
                f"and/or name {name}")

    async def _lookup(follower, short):
        return "Jeffrey IslandBringer" if short == "Jeffrey" else ""

    class _Mouse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Follower:
        mouse_handler = _Mouse()

    import sys
    import types
    for name in ("wizwalker", "wizwalker.extensions"):
        stub = types.ModuleType(name)
        stub.__path__ = []            # a package, so ensure_path can extend it
        sys.modules[name] = stub
    mod = types.ModuleType("wizwalker.extensions.scripting")
    mod.teleport_to_friend_from_list = _tp
    sys.modules["wizwalker.extensions.scripting"] = mod
    real_lookup = party.friends_list_name
    party.friends_list_name = _lookup
    try:
        ok, why = asyncio.run(
            party.teleport_to_leader_across_zones(_Follower(), "Jeffrey"))
    finally:
        party.friends_list_name = real_lookup
        for name in ("wizwalker.extensions.scripting",
                     "wizwalker.extensions", "wizwalker"):
            sys.modules.pop(name, None)

    assert ok is True, why
    assert tried == ["Jeffrey", "Jeffrey IslandBringer"]
    # and the resolved name is remembered, so it is one lookup per run
    assert party._FULL_NAMES["Jeffrey"] == "Jeffrey IslandBringer"
    party._FULL_NAMES.clear()


def test_a_stage_that_never_returns_cannot_take_the_hotkeys_with_it(qapp):
    """The regression that killed every hotkey.

    wizwalker's friends-list teleport opens with `_cycle_to_online_friends`,
    which is `while (await text()) != "Online Friends": click(); wait(5)`
    — no bound at all. A wizard whose list never reads exactly that spins
    in it for the rest of the run, and the follow step holds the drive
    lock while it does. Every queued teleport, wisp sweep and potion then
    waits behind it forever, and every further press is refused as
    already queued: four dead keys from one unbounded loop.

    So no stage may hold the wheel without a deadline.
    """
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    class _Client:
        async def in_battle(self):
            return False

    done, said = [], []
    stuck = asyncio.Event()

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1, auto_quest=True)
    w.status = type("S", (), {"emit": staticmethod(said.append)})()
    seat = w.seats[0]
    seat.client = client = _Client()
    w.STAGE_LIMITS = dict(w.STAGE_LIMITS, **{"quest step": 0.3})

    async def never_returns(_c):
        stuck.set()
        await asyncio.Event().wait()        # the wizwalker loop, in spirit

    async def drive():
        seat.drive = asyncio.Lock()
        w._quest_step = never_returns
        w._do_request = lambda c, a, s=None: _record(done, a)

        tasks = [asyncio.ensure_future(w._service_loop(client, seat)),
                 asyncio.ensure_future(w._request_loop(client, seat))]
        await asyncio.wait_for(stuck.wait(), 2)
        w.request("teleport")
        for _ in range(40):
            await asyncio.sleep(0.05)
            if done:
                break
        w._stop = True
        for task in tasks:
            task.cancel()
            try:
                await task
            except BaseException:
                pass

    asyncio.run(drive())
    assert done == ["teleport"], "the hotkey never got the wheel back"
    assert any("was cut off" in m for m in said), said
    assert any("hotkeys included" in m for m in said), said


def test_a_press_that_has_to_wait_says_what_it_is_waiting_for(qapp):
    """A press that goes quiet for thirty seconds is indistinguishable
    from a key that is not bound, and that is what got reported."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    class _Client:
        async def in_battle(self):
            return False

    said, started = [], asyncio.Event()

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1, auto_quest=True)
    w.status = type("S", (), {"emit": staticmethod(said.append)})()
    seat = w.seats[0]
    seat.client = client = _Client()

    async def slow_quest(_c):
        started.set()
        await asyncio.sleep(0.6)

    async def drive():
        seat.drive = asyncio.Lock()
        w._quest_step = slow_quest
        w._do_request = lambda c, a, s=None: _record([], a)
        tasks = [asyncio.ensure_future(w._service_loop(client, seat)),
                 asyncio.ensure_future(w._request_loop(client, seat))]
        await asyncio.wait_for(started.wait(), 2)
        w.request("wisps")
        await asyncio.sleep(0.8)
        w._stop = True
        for task in tasks:
            task.cancel()
            try:
                await task
            except BaseException:
                pass

    asyncio.run(drive())
    # and it names the stage, not just "something": "waiting for quest
    # step" is actionable and "waiting" is not
    assert any("waiting for quest step to let go of the wheel" in m
               for m in said), said


def test_the_friends_list_teleport_is_bounded(qapp, monkeypatch):
    """`teleport_to_friend_from_list` can genuinely never return. A
    follower that calls it holds the party's wheel while it does not."""
    import asyncio
    import sys
    import types

    from deimos_bridge import party

    async def _never(*_a, **_kw):
        await asyncio.Event().wait()

    class _Mouse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

    class _Follower:
        mouse_handler = _Mouse()

    for name in ("wizwalker", "wizwalker.extensions"):
        stub = types.ModuleType(name)
        stub.__path__ = []
        sys.modules[name] = stub
    mod = types.ModuleType("wizwalker.extensions.scripting")
    mod.teleport_to_friend_from_list = _never
    sys.modules["wizwalker.extensions.scripting"] = mod
    monkeypatch.setattr(party, "TELEPORT_TIMEOUT", 0.2)
    try:
        ok, why = asyncio.run(
            party.teleport_to_leader_across_zones(_Follower(), "Jeffrey"))
    finally:
        for name in ("wizwalker.extensions.scripting",
                     "wizwalker.extensions", "wizwalker"):
            sys.modules.pop(name, None)
        party._FULL_NAMES.clear()

    assert ok is False
    assert "was cut off" in why, why
    assert "Online Friends" in why, why


def test_restoring_a_seats_boxes_does_not_retune_the_running_party(qapp,
                                                                  monkeypatch):
    """`setChecked` fires `toggled` exactly like a click. Switching the
    wizard dropdown mid-run would otherwise push whatever that seat's
    snapshot held onto the live worker — turning auto-quest off for the
    whole party because wizard 3 was configured without it."""
    from deimos_bridge.gui import app as app_mod
    from deimos_bridge.gui.app import MainWindow

    monkeypatch.setattr(app_mod.LiveWorker, "start", lambda self, *a: None)
    win = MainWindow(Telemetry())
    win.auto_quest.setChecked(True)
    win.wizards.setValue(2)
    win.on_start_live()
    win.live.isRunning = lambda: True
    assert win.live.auto_quest is True

    # wizard 2's saved configuration, restored into the boxes
    win._loading = True
    try:
        win.auto_quest.setChecked(False)
    finally:
        win._loading = False

    assert win.live.auto_quest is True, "a restore reconfigured the run"
    assert win.auto_quest.isChecked() is False, "the box still tracks the seat"

    # and a person moving it still reaches the run
    win.auto_quest.setChecked(True)
    win.auto_quest.setChecked(False)
    assert win.live.auto_quest is False


def _twins(hps, name="Nirini Warrior", max_hp=395.0, round_number=1):
    """A read of two identically-named mobs — the ordinary Wizard101 board."""
    from w101_sim import Actor, State
    player = Actor(name="Wizard", school="ice", hp=757, max_hp=757, team=0)
    mobs = [Actor(name=name, school="balance", hp=hp, max_hp=max_hp, team=1)
            for hp in hps]
    return _Read(State(player, mobs), round_number, ("Frost Beetle",))


def test_damage_is_measured_against_the_mob_that_was_hit(qapp):
    """The measurement bug behind a 634% "model error" that was not one.

    `_settle` looked the target up by NAME, and Wizard101 boards are
    mostly two of the same mob. The lookup dict kept the LAST of each
    name while the before-list scan took the FIRST, so a full-health
    Nirini Warrior hit for 19 was differenced against its twin sitting
    at 259 and recorded as 136.

    Numbers from the live party run: wizard 2 predicted 18.53, the mob
    went 395 -> 376, and three consecutive rounds were filed as 136,
    117, 117.
    """
    tel = Telemetry()
    tel.start_fight()
    r = tel.observe(_Decision("Frost Beetle", target_index=0),
                    _twins([395.0, 259.0], round_number=4))
    r.predicted_damage = 18.53
    tel.observe(_Decision("Frost Beetle", target_index=0),
                _twins([376.0, 259.0], round_number=5))

    assert r.actual_damage == pytest.approx(19.0), \
        "measured against the wrong twin again"
    assert abs(r.error) < 1.0
    assert r.clean, r.confounds


def test_the_twin_that_died_is_the_one_the_health_says(qapp):
    """A board that loses one of two same-named mobs. Matching in list
    order would pair the survivor with the corpse's slot and report the
    living mob as dead."""
    tel = Telemetry()
    tel.start_fight()
    r = tel.observe(_Decision("Frost Beetle", target_index=1),
                    _twins([19.0, 259.0], round_number=2))
    r.predicted_damage = 100.0
    tel.observe(_Decision("Frost Beetle", target_index=0),
                _twins([160.0], round_number=3))

    # the target (259) survived at 160; the 19 HP twin is the one gone
    assert r.actual_damage == pytest.approx(99.0)
    assert r.clean, r.confounds
    assert not any("died" in c for c in r.confounds)


def test_a_defeated_wizard_leaves_the_circle_and_stops_recording(qapp):
    """A knocked-out wizard is still in the duel as far as wizwalker is
    concerned: `handle_round` keeps firing against an empty hand. The
    live run filed four consecutive "policy chose to pass" rounds for
    it — and, worse, the rest of the party waited for it at the barrier
    every one of those rounds."""
    import asyncio

    from deimos_bridge.hivemind import Hivemind
    from deimos_bridge.live_backend import WizAiBackend
    from w101_sim import Actor, State

    hive = Hivemind(timeout=0.2)
    hive.join(0, "Konstantin")
    hive.join(1, "Jeffrey")
    hive.enter_combat(0)
    hive.enter_combat(1)

    said = []
    tel = Telemetry()
    backend = WizAiBackend(policy=lambda *_a: None, cards={}, school="fire",
                           decklist=[], catalog={"cards": {}},
                           seat=0, coordinator=hive, party_size=2)
    backend.on_defeated = lambda: said.append("down")
    backend.telemetry = tel

    player = Actor(name="Konstantin", school="fire", hp=0, max_hp=757, team=0)
    mob = Actor(name="Nirini Warrior", school="balance", hp=395,
                max_hp=395, team=1)
    read = _Read(State(player, [mob]), 4, ())
    read.state.player_hp = 0.0

    async def go():
        backend.read_state_for_test = read
        return backend._check_defeated(read)

    decision = asyncio.run(go())
    assert decision is not None and decision.passing
    assert "defeated" in decision.reason
    assert said == ["down"], "the operator was never told"
    assert hive.fighting() == [1], "the party is still waiting for a corpse"
    assert hive.size == 2, "it left the circle, not the party"

    # said once per knockdown, not once per round
    backend._check_defeated(read)
    assert said == ["down"]


def test_the_hivemind_tab_shows_every_wizard_at_once(qapp):
    """The tab exists because every other one shows a wizard at a time.
    Working out why a party has stalled by switching a dropdown four
    times is exactly what it replaces, so the roster has to carry the
    state that answers it: who is in the circle, on what health, and
    what each of them last said."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.wizards.setValue(3)
    assert win.tabs.isTabVisible(win.hivemind_tab)
    assert win.tabs.tabText(win.hivemind_tab) == "Hivemind"

    win.on_seat_named(0, "Konstantin")
    win.on_seat_named(1, "Jeffrey")
    win.on_seat_hp_read(0, 757)
    win.on_seat_status(1, "following the leader — could not read the "
                          "leader's position")
    win.refresh_hivemind()

    roster = win.hivemind.roster
    assert roster.rowCount() == 3
    assert roster.item(0, 0).text() == "Konstantin"
    assert roster.item(1, 0).text() == "Jeffrey"
    assert "wizard 3" in roster.item(2, 0).text()
    assert "757" in roster.item(0, 3).text()
    assert "leader's position" in roster.item(1, 8).text(), \
        "a stuck wizard's own line is the point of the roster"
    assert "3 wizard(s) connected" in win.hivemind.party_lab.text()


def test_the_roster_says_who_is_actually_in_the_circle(qapp):
    """A wizard connected but not in the duel is not being planned for —
    the party agrees a round among the wizards actually in the circle,
    and the rest are on their own. That is the difference between a
    hivemind and four bots, and nothing showed it."""
    from deimos_bridge.gui.app import MainWindow
    from deimos_bridge.hivemind import Hivemind

    win = MainWindow(Telemetry())
    win.wizards.setValue(2)
    hive = Hivemind(timeout=0.2)
    hive.join(0, "Konstantin")
    hive.join(1, "Jeffrey")
    hive.enter_combat(0)
    hive.enter_combat(1)
    win.live = type("L", (), {"hive": hive, "isRunning": lambda self: True})()
    win.refresh_hivemind()

    assert win.hivemind.roster.item(0, 4).text() == "in the circle"
    assert "2 in the circle" in win.hivemind.party_lab.text()

    hive.leave_combat(1)
    win.refresh_hivemind()
    # one wizard left swinging is fighting ALONE, which is a different
    # thing from fighting with the party and worth flagging
    assert win.hivemind.roster.item(0, 4).text() == "fighting alone"
    assert win.hivemind.roster.item(1, 4).text() != "in the circle"


def test_the_roster_flags_a_defeated_wizard(qapp):
    """0 health is the state that explains a party that has stopped
    making progress, and it was visible nowhere."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.wizards.setValue(2)
    win.on_seat_hp_read(0, 757)
    win.seat_live[0]["hp"] = 0.0
    win.refresh_hivemind()
    assert win.hivemind.roster.item(0, 4).text() == "defeated"


def test_the_wheel_is_given_back_between_stages(qapp):
    """The drive lock stops two coroutines steering one wizard at the
    same moment. Held for the whole tick it is held for the SUM of the
    stages instead, so a press waits out an auto-dialogue AND a follow
    AND a quest hop before its own turn."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    class _Client:
        async def in_battle(self):
            return False

    order, first = [], asyncio.Event()

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1, auto_quest=True,
                   auto_dialogue=True)
    w.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    seat = w.seats[0]
    seat.client = client = _Client()

    async def slow_dialogue(_c):
        first.set()
        order.append("dialogue")
        await asyncio.sleep(0.4)

    async def slow_quest(_c):
        order.append("quest")
        await asyncio.sleep(0.4)

    async def drive():
        seat.drive = asyncio.Lock()
        w._auto_dialogue = slow_dialogue
        w._quest_step = slow_quest
        w._do_request = lambda c, a, s=None: _record(order, a)
        tasks = [asyncio.ensure_future(w._service_loop(client, seat)),
                 asyncio.ensure_future(w._request_loop(client, seat))]
        await asyncio.wait_for(first.wait(), 2)
        w.request("teleport")
        for _ in range(30):
            await asyncio.sleep(0.05)
            if "teleport" in order:
                break
        w._stop = True
        for task in tasks:
            task.cancel()
            try:
                await task
            except BaseException:
                pass

    asyncio.run(drive())
    assert "teleport" in order, "the press never ran"
    assert order.index("teleport") < len(order) - 1 or "quest" not in order, \
        order
    # the press got in at the gap between two stages rather than after
    # every stage of the tick
    assert order[:2] == ["dialogue", "teleport"], order


# ------------------------------------------------- deimoslang scripts, for real
class _CountingVM:
    """A VM that runs `total` instructions and then stops, like a real one."""

    def __init__(self, total=5000):
        self.running = True
        self.killed = False
        self.total = total
        self.done = 0
        self.loads = 0

    async def step(self):
        self.done += 1
        if self.done >= self.total:
            self.running = False

    def load_from_text(self, _src):
        self.loads += 1
        self.done = 0

    def kill(self):
        self.killed = True
        self.running = False


def test_a_burst_runs_thousands_of_instructions_not_one():
    """The reason a real Deimos quester "does nothing" under wizAi.

    `VM.step()` runs ONE instruction. The TTS Arc 1 script people
    actually use compiles to 18,366 of them, so at one per half-second
    service tick it needs two and a half hours to reach the end of the
    program once — and its opening Close_Menus block alone takes
    seventeen seconds to do what Deimos does instantly. Deimos runs
    `while v.running: await v.step()`.
    """
    import asyncio

    from deimos_bridge.scripts import ScriptRunner

    vm = _CountingVM(total=10 ** 9)
    runner = ScriptRunner(vm, "src")
    runner.SLICE = 0.2

    done = asyncio.run(runner.run_for())
    assert done > 500, f"only {done} instructions in a 0.2s burst"
    assert runner.steps == done


def test_a_burst_stops_the_moment_the_policy_needs_the_wizard():
    """A duel starting mid-burst has to end the burst, or the script
    keeps clicking through the policy's planning phase."""
    import asyncio

    from deimos_bridge.scripts import ScriptRunner

    vm = _CountingVM(total=10 ** 9)
    runner = ScriptRunner(vm, "src")
    runner.SLICE = 5.0
    fighting = []

    def should_stop():
        fighting.append(1)
        return len(fighting) > 50

    done = asyncio.run(runner.run_for(should_stop=should_stop))
    assert done == 50, done
    assert vm.running, "the VM was stopped rather than parked"


def test_a_script_that_runs_off_the_end_starts_again(monkeypatch):
    """Deimos reloads and reruns it (Deimos.py:2144-2152) and questers
    are written expecting that — 'If the script ever restarts…'. Only
    kill ends a run."""
    import asyncio

    from deimos_bridge import scripts
    from deimos_bridge.scripts import ScriptRunner

    fresh = []
    monkeypatch.setattr(scripts, "build_vm",
                        lambda c, s: fresh.append(_CountingVM(total=3))
                        or fresh[-1])

    vm = _CountingVM(total=3)
    runner = ScriptRunner(vm, "src")
    asyncio.run(runner.run_for(seconds=1.0))
    assert not runner.running

    assert runner.restart() is True
    assert len(fresh) == 1 and runner.vm is fresh[0]
    assert runner.running and runner.restarts == 1

    runner.stop()
    assert runner.restart() is False, "a killed script must stay dead"


def test_a_non_expert_script_is_named_not_compiled(monkeypatch):
    """Deimos's older one-command-per-line format is a different
    language sharing a text box. Handing it to the deimoslang compiler
    produces a parse error about a token the author never thought of as
    one."""
    from deimos_bridge import scripts

    monkeypatch.setattr(scripts, "available", lambda: (True, ""))
    ok, why = scripts.check("sendkey W 1\nteleport 10, 10, 0\n")
    assert ok is False
    assert "not an expert-mode script" in why
    assert "###deimos_expertmode" in why

    with pytest.raises(RuntimeError, match="not an expert-mode script"):
        scripts.make_runner([object()], "sendkey W 1\n")


def test_the_clients_header_is_read_and_enforced(monkeypatch):
    """A four-wizard quester run with one client hooked does not fail —
    it walks one wizard into a dungeon and waits forever for three that
    are not there. p2..p4 resolve to None (vm.py:135), which surfaces as
    an AttributeError somewhere unrelated."""
    from deimos_bridge import scripts

    assert scripts.wants_clients("# @clients: > 1\n") == 2
    assert scripts.wants_clients("# @clients: 4\n") == 4
    assert scripts.wants_clients("# @clients: >= 3\n") == 3
    assert scripts.wants_clients("no header here") == 0

    monkeypatch.setattr(scripts, "available", lambda: (True, ""))
    src = "###deimos_expertmode\n# @clients: > 1\nsendkey W, 1\n"
    with pytest.raises(RuntimeError) as got:
        scripts.make_runner([object()], src)
    assert "needs 2 wizards" in str(got.value)
    assert "1 is hooked" in str(got.value)


def test_the_party_gets_one_vm_over_every_client(qapp, monkeypatch):
    """deimoslang addresses wizards as p1..p4 and there is ONE program.
    A runner per seat is four copies of the same quester each believing
    it is p1, walking four wizards to four places."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker, SeatConfig
    from deimos_bridge import scripts

    built = []

    def _fake(clients, source):
        built.append(list(clients))
        return type("R", (), {"running": True, "stop": lambda self: None,
                              "steps": 0, "failures": 0})()

    monkeypatch.setattr(scripts, "make_runner", _fake)

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                   seats=[SeatConfig(school="fire", deck=[])],
                   script="###deimos_expertmode\nsendkey W, 1\n")
    w.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    w.seats[0].client = "clientA"
    w.seats[1].client = "clientB"

    asyncio.run(w._sync_script(w.seats[0]))
    assert built == [["clientA", "clientB"]], built

    # and seat 1 does not build a second one
    asyncio.run(w._sync_script(w.seats[1]))
    assert len(built) == 1, "a second VM was built for the other seat"
    assert w.seats[1].runner is None


def test_a_running_script_stops_wizai_walking_the_same_wizard(qapp):
    """The script walks the wizard. wizAi's own quest hop or follow
    would walk it somewhere else between two of the script's
    instructions, which is how a scripted run ends up in a doorway."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    class _Client:
        async def in_battle(self):
            return False

    ran = []

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1, auto_quest=True)
    w.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    seat = w.seats[0]
    seat.client = client = _Client()

    async def quest(_c):
        ran.append("quest")

    async def script(_s=None):
        ran.append("script")

    async def drive():
        seat.drive = asyncio.Lock()
        w._quest_step = quest
        w._script_step = script
        seat.runner = type("R", (), {"running": True})()
        task = asyncio.ensure_future(w._service_loop(client, seat))
        await asyncio.sleep(0.6)
        w._stop = True
        task.cancel()
        try:
            await task
        except BaseException:
            pass

    asyncio.run(drive())
    assert "script" in ran, ran
    assert "quest" not in ran, "wizAi walked a wizard the script is driving"


def test_an_instruction_that_never_finishes_reloads_the_script(monkeypatch):
    """`waitforzonechange completion` blocks for a whole loading screen
    and the TTS Arc 1 script has 122 of them, so the burst cannot simply
    be cancelled at 30s. But there is no timeout inside the VM at all —
    a zone change that never comes waits for the rest of the run holding
    the wizard's wheel. Cancelling mid-instruction leaves the VM half
    way through one, so the only honest recovery is a reload."""
    import asyncio

    from deimos_bridge import scripts
    from deimos_bridge.scripts import ScriptRunner

    class _Hangs:
        running = True
        killed = False

        async def step(self):
            await asyncio.Event().wait()

        def kill(self):
            self.killed = True

    fresh = _CountingVM(total=10)
    monkeypatch.setattr(scripts, "build_vm", lambda c, s: fresh)

    vm = _Hangs()
    runner = ScriptRunner(vm, "src")
    runner.STEP_LIMIT = 0.2

    assert asyncio.run(runner.step()) is False
    assert runner.stale, "a cancelled instruction left the VM trusted"
    assert "without finishing" in runner.last_error

    assert runner.restart() is True
    assert runner.vm is fresh, "the half-executed VM was kept"
    assert runner.stale is False


def test_the_script_stage_limit_sits_above_the_runners_own(qapp):
    """Otherwise the stage cancel fires first and cancels an instruction
    the runner was going to bound and reload by itself."""
    from deimos_bridge.gui.live import LiveWorker
    from deimos_bridge.scripts import ScriptRunner

    assert (LiveWorker.STAGE_LIMITS["script step"]
            > ScriptRunner.STEP_LIMIT), "the backstop fires before the bound"


def test_a_repeating_script_error_keeps_being_reported(qapp):
    """The old rule said it at failure 1 and failure 10 and then went
    silent forever, while the runner sat retrying the same instruction
    every tick for the rest of the run."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    said = []
    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    w.status = type("S", (), {"emit": staticmethod(said.append)})()
    seat = w.seats[0]

    class _Runner:
        running = True
        stale = False
        failures = 0
        steps = 0
        last_error = "RuntimeError: no such window"

        async def run_for(self, **_kw):
            self.failures += 1
            return 0

    seat.runner = _Runner()
    for _ in range(60):
        asyncio.run(w._script_step(seat))

    errs = [m for m in said if "script error" in m]
    assert len(errs) >= 3, f"went silent after {len(errs)} reports"
    assert any("still failing" in m for m in errs)


def test_a_restart_actually_runs_the_program_again(monkeypatch):
    """`load_from_text` assigns `program` and nothing else (vm.py:127).
    `current_task.ip` is still past the end of the old program and
    `current_task.running` is still the False the epilogue set when it
    got there (vm.py:1833), so the next step() takes the
    `if not self.current_task.running` branch and returns without
    executing anything — for ever. Deimos builds a fresh VM each pass."""
    import asyncio

    from deimos_bridge import scripts

    built = []

    def _build(clients, source):
        built.append(list(clients))
        return _CountingVM(total=10)

    monkeypatch.setattr(scripts, "build_vm", _build)

    dead = _CountingVM(total=1)
    runner = scripts.ScriptRunner(dead, "src", clients=["a", "b"])
    asyncio.run(runner.run_for(seconds=0.5))
    assert not runner.running

    assert runner.restart() is True
    assert built == [["a", "b"]], "the old VM was reused"
    assert runner.vm is not dead, "the exhausted VM was kept"
    assert runner.running, "the fresh VM was not started"


def test_only_one_seat_builds_the_partys_script(qapp, monkeypatch):
    """`_setup_script` builds a VM over EVERY client, so calling it once
    per seat gives four VMs each driving all four wizards — four copies
    of one quester, worse than the single-client VM it replaced."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker, SeatConfig
    from deimos_bridge import scripts

    built = []
    monkeypatch.setattr(
        scripts, "make_runner",
        lambda clients, src: built.append(list(clients)) or
        type("R", (), {"running": True, "stop": lambda self: None})())

    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                   seats=[SeatConfig(school="fire", deck=[]),
                          SeatConfig(school="life", deck=[])],
                   script="###deimos_expertmode\nsendkey W, 1\n")
    w.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    for i, s in enumerate(w.seats):
        s.client = f"client{i}"

    for seat in w.seats:
        if w._scripted(seat):
            asyncio.run(w._setup_script(seat.client, seat))
    assert len(built) == 1, f"{len(built)} VMs for one script"
    assert built[0] == ["client0", "client1", "client2"]


def test_a_press_is_not_dropped_while_it_can_see_what_it_waits_for(qapp):
    """A script burst can hold the wheel for longer than the TTL.
    Dropping the press with 'press it again' while the thing it is
    waiting for is visibly working loses a keypress and lies."""
    from deimos_bridge.gui.live import LiveWorker

    said = []
    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    w.status = type("S", (), {"emit": staticmethod(said.append)})()
    seat = w.seats[0]

    assert w.request("teleport") is True
    seat.queued_at["teleport"] -= w.REQUEST_TTL + 1

    seat.driver = "script step"
    w._expire_requests(seat)
    assert seat.requests == ["teleport"], "dropped while something held the wheel"
    assert not said

    seat.driver = None
    w._expire_requests(seat)
    assert seat.requests == []
    assert any("dropped the queued teleport" in m for m in said)


class _Move:
    def __init__(self, seat, name, card="", target_name="", damage=0.0):
        self.seat, self.name, self.card = seat, name, card
        self.target_name, self.damage = target_name, damage
        self.target = 0
        self.solo_card, self.solo_target, self.note = card, 0, ""


class _Plan:
    def __init__(self, *moves):
        self.moves = list(moves)


def test_a_teammates_trap_is_not_a_teammate_hitting_the_mob():
    """Any card AIMED at an enemy used to count as the teammate having
    hit it — and a trap, a shield and a debuff are all aimed at an enemy
    while taking nothing off it. Live that filed a false 'also hit' on
    18 of one wizard's 64 rounds and 24 of the other's, each one marking
    the round unclean. It was the single biggest reason the damage model
    came back with two usable observations out of seventy."""
    from deimos_bridge.telemetry import _party_hits

    plan = _Plan(_Move(0, "Jeffrey", "Ice Trap", "Mob", damage=0.0),
                 _Move(1, "Konstantin", "Fire Cat", "Mob", damage=104.0))

    assert _party_hits(plan, seat=1) == {}, "a trap counted as a hit"
    assert _party_hits(plan, seat=0) == {"Mob": "Konstantin"}


def test_a_shared_mob_is_measured_as_the_partys_claim():
    """Two wizards into one mob is one board delta that is both of
    theirs and neither's, so the solo residual has to refuse it. In a
    real party that is most rounds — one live run left a wizard with two
    usable rounds out of seventy. What cannot be split can still be
    added up."""
    tel = Telemetry(seat=1)
    tel.start_fight()

    plan = _Plan(_Move(0, "Jeffrey", "Frost Beetle", "Mob", damage=90.0),
                 _Move(1, "Konstantin", "Fire Cat", "Mob", damage=110.0))
    r = tel.observe(_Decision("Fire Cat", target_index=0), _read(2000, 1),
                    party=plan, seat=1)
    r.predicted_damage = 110.0
    assert r.party_predicted == 200.0, "the party's claim was not recorded"

    tel.observe(_Decision("Fire Cat", target_index=0), _read(1810, 2),
                party=plan, seat=1)

    assert r.actual_damage == 190.0
    assert r.clean is False, "a shared delta is not a solo measurement"
    assert tel.error_stats()["n"] == 0

    # but the party's claim about that mob IS checkable, and was right
    assert r.party_error == pytest.approx(-10.0)
    party = tel.party_error_stats()
    assert party["n"] == 1
    assert party["mean_error"] == pytest.approx(-10.0)
    assert party["mean_pct_error"] == pytest.approx(-5.0)


def test_a_solo_round_is_not_counted_twice():
    """A round this wizard had to itself is already a solo observation;
    counting it in the party series as well would weight it twice."""
    tel = Telemetry(seat=0)
    tel.start_fight()
    plan = _Plan(_Move(0, "Jeffrey", "Frost Beetle", "Mob", damage=90.0))
    r = tel.observe(_Decision("Frost Beetle", target_index=0), _read(2000, 1),
                    party=plan, seat=0)
    r.predicted_damage = 90.0
    tel.observe(_Decision("Frost Beetle", target_index=0), _read(1910, 2),
                party=plan, seat=0)

    assert tel.error_stats()["n"] == 1
    assert tel.party_error_stats()["n"] == 0
    assert r.party_error is None


def test_a_dirty_shared_round_is_still_refused():
    """A DoT ticking on the target corrupts the party's number exactly
    as much as it corrupts one wizard's."""
    from w101_sim import Actor, State

    tel = Telemetry(seat=1)
    tel.start_fight()
    plan = _Plan(_Move(0, "Jeffrey", "Frost Beetle", "Mob", damage=90.0),
                 _Move(1, "Konstantin", "Fire Cat", "Mob", damage=110.0))

    burning = _read(2000, 1)
    r = tel.observe(_Decision("Fire Cat", target_index=0), burning,
                    party=plan, seat=1)
    r.predicted_damage = 110.0
    r.enemies[0].wards.append("50/tick x3 dot")

    tel.observe(_Decision("Fire Cat", target_index=0), _read(1810, 2),
                party=plan, seat=1)

    assert any("DoT" in c for c in r.confounds), r.confounds
    assert tel.party_error_stats()["n"] == 0


def test_the_export_says_which_build_made_it():
    """Two runs uploaded twenty minutes before a fix landed were read as
    evidence about the fixed code, and there was nothing in the file to
    say otherwise."""
    tel = Telemetry()
    rev = tel.summary()["revision"]
    assert isinstance(rev, str)
    if rev:                       # a git checkout; empty in a tarball
        assert len(rev) >= 7


def test_a_run_stopped_while_waiting_does_not_invent_a_fight():
    """`start_fight` is called before the wait for a duel, so stopping
    leaves a 0-round record — and counting it read as '9 wins from 13
    fights' for a wizard that fought twelve and won nine."""
    tel = Telemetry()
    tel.start_fight()
    tel.observe(_Decision("Sunbird", target_index=0), _read(2000, 1))
    tel.fights[-1].rounds = 1
    tel.end_fight(won=True)
    tel.start_fight()             # the next duel, which never started

    assert len(tel.fights) == 2, "the record is kept"
    assert tel.summary()["fights"] == 1, "but it is not a fight"
    assert tel.summary()["wins"] == 1


def test_the_party_series_refuses_a_mob_that_could_not_absorb_the_claim():
    """The delta cannot exceed the health that was there. A party that
    expects 586 off a mob with 78 left and finds it standing has learnt
    about a fizzle, not about the arithmetic — and would otherwise
    report a 90% over-prediction."""
    tel = Telemetry(seat=1)
    tel.start_fight()
    plan = _Plan(_Move(0, "Jeffrey", "Evil Snowman", "Mob", damage=400.0),
                 _Move(1, "Konstantin", "Sunbird", "Mob", damage=186.0))

    r = tel.observe(_Decision("Sunbird", target_index=0), _read(200, 1),
                    party=plan, seat=1)
    r.predicted_damage = 186.0
    tel.observe(_Decision("Sunbird", target_index=0), _read(122, 2),
                party=plan, seat=1)

    assert r.party_predicted == 586.0
    assert any("could not" in c or "did not land" in c for c in r.confounds), \
        r.confounds
    assert tel.party_error_stats()["n"] == 0


def test_two_wizards_exports_can_be_joined_on_the_opening_board():
    """`index` counts this seat's own fights and seats do not see the
    same fights — one live run had wizard 1's fight 1 be wizard 2's
    fight 3. Matching on (board, round) instead is ambiguous for a
    quarter of the rounds: three Ice Weavers at 395 is the same key in
    every fight against three Ice Weavers."""
    from w101_sim import Actor, State

    def board(*hps):
        player = Actor(name="W", school="ice", hp=100, max_hp=100, team=0)
        mobs = [Actor(name="Ice Weaver", school="balance", hp=h, max_hp=395,
                      team=1) for h in hps]
        return _Read(State(player, mobs), 1, ("Frost Beetle",))

    a, b = Telemetry(seat=0), Telemetry(seat=1)
    for tel in (a, b):
        tel.start_fight()
        tel.observe(_Decision("Frost Beetle", target_index=0),
                    board(395, 395, 395))

    assert a.fights[-1].opening == "Ice Weaver@395+Ice Weaver@395+Ice Weaver@395"
    assert a.fights[-1].opening == b.fights[-1].opening, \
        "the two seats cannot line their fights up"

    # and a different duel is a different key
    c = Telemetry(seat=0)
    c.start_fight()
    c.observe(_Decision("Frost Beetle", target_index=0), board(395, 395))
    assert c.fights[-1].opening != a.fights[-1].opening


def test_the_party_damage_number_counts_each_round_once(qapp):
    """Every seat that fired into the shared mob records the same claim
    about the same board delta, so adding the seats' series together
    would count one measurement twice for two wizards and four times for
    four."""
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    win.wizards.setValue(2)

    plan = _Plan(_Move(0, "Jeffrey", "Frost Beetle", "Mob", damage=90.0),
                 _Move(1, "Konstantin", "Fire Cat", "Mob", damage=110.0))
    for seat in (0, 1):
        tel = win.tels[seat]
        tel.start_fight()
        r = tel.observe(_Decision("Fire Cat", target_index=0), _read(2000, 1),
                        party=plan, seat=seat)
        r.predicted_damage = 90.0 if seat == 0 else 110.0
        tel.observe(_Decision("Fire Cat", target_index=0), _read(1810, 2),
                    party=plan, seat=seat)
        assert tel.party_error_stats()["n"] == 1

    pooled = win._party_model()
    assert pooled["n"] == 1, f"one round counted {pooled['n']} times"
    assert pooled["mean_error"] == pytest.approx(-10.0)

    win.party.show_model(pooled)
    assert "1 shared round" in win.party.model_lab.text()
    assert "over-predicting by 10 HP" in win.party.model_lab.text()


def test_a_script_wedged_on_one_instruction_reloads_itself():
    """deimoslang has instructions that raise WITHOUT advancing the
    instruction pointer, and the VM has no handler. `teleport client 3`
    with two wizards hooked is one: player_by_num answers None
    (vm.py:137) and `target_client.body` throws (vm.py:1414) before the
    arm's `ip += 1` (vm.py:1826). Every later burst re-enters the same
    instruction and throws the same way, for the rest of the run.

    The @clients header cannot catch it — TTS Arc 1 declares
    `@clients: > 1` and still has 18 reachable such sites with two
    wizards hooked."""
    import asyncio

    from deimos_bridge.scripts import ScriptRunner

    class _Task:
        ip = 8907                       # never advances

    class _Wedged:
        running = True
        killed = False
        current_task = _Task()

        async def step(self):
            raise AttributeError("'NoneType' object has no attribute 'body'")

        def kill(self):
            self.killed = True

    runner = ScriptRunner(_Wedged(), "src", clients=["a", "b"])
    for _ in range(ScriptRunner.STUCK_AT - 1):
        assert asyncio.run(runner.step()) is False
        assert not runner.stale, "gave up while the script might still move on"

    asyncio.run(runner.step())
    assert runner.stale, "spun on one instruction for the rest of the run"
    assert "stuck on one instruction" in runner.last_error
    assert "NoneType" in runner.last_error, "the real cause was lost"


def test_a_moving_script_is_never_called_stuck():
    """Errors at different instructions are a script having a bad time,
    not a wedge — reloading would throw away its progress."""
    import asyncio

    from deimos_bridge.scripts import ScriptRunner

    class _Task:
        ip = 0

    class _Unlucky:
        running = True
        killed = False
        current_task = _Task()

        async def step(self):
            self.current_task.ip += 1
            raise RuntimeError("no such window")

        def kill(self):
            pass

    runner = ScriptRunner(_Unlucky(), "src")
    for _ in range(ScriptRunner.STUCK_AT * 2):
        asyncio.run(runner.step())
    assert not runner.stale
    assert runner.failures == ScriptRunner.STUCK_AT * 2


def test_a_script_that_names_more_wizards_than_are_hooked_says_so(qapp,
                                                                  monkeypatch):
    """Not a refusal — the parts that name p3 and p4 are usually behind
    the script's own configuration flags, so refusing would refuse every
    party script written for four and run with two."""
    import asyncio

    from deimos_bridge import scripts
    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    assert scripts.mentions_clients("p1 sendkey W\np4 sendkey W\n") == 4
    assert scripts.mentions_clients("###deimos_expertmode\n") == 0

    monkeypatch.setattr(scripts, "make_runner",
                        lambda c, s: type("R", (), {"running": True})())
    said = []
    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                   seats=[SeatConfig(school="fire", deck=[])],
                   script="###deimos_expertmode\np1 sendkey W, 1\n"
                          "p4 sendkey W, 1\n")
    w.status = type("S", (), {"emit": staticmethod(said.append)})()
    w.seats[0].client, w.seats[1].client = "a", "b"

    asyncio.run(w._setup_script("a", w.seats[0]))
    assert any("names up to p4" in m and "2 wizard(s) are hooked" in m
               for m in said), said
