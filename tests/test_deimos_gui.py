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


def test_a_round_records_what_it_cost():
    """"It is very slow" was unanswerable from an export: rounds carried
    no time at all, so a 46-second round and a 12-second one were the
    same file. Three numbers because they have three owners — the game's
    animation, our planning, our clicking — and they call for opposite
    fixes."""
    tel = Telemetry()
    tel.start_fight()
    rec = tel.observe(_Decision("Sunbird"), _read(2000, 1))
    tel.time_round(waited=34.0, planned=4.25, acted=1.5)
    assert rec.waited == 34.0
    assert rec.planned == 4.25
    assert rec.acted == 1.5
    assert rec.at is not None, "a round with no clock cannot be lined up"


def test_a_round_the_board_could_not_be_read_for_is_still_timed():
    """It is the round most worth timing. `_pending` is never set for
    one — it is the damage-settling slot, and a lost round has no damage
    — so pacing rides its own pointer."""
    tel = Telemetry()
    tel.start_fight()
    tel.observe_lost_round(3, "read failed")
    tel.time_round(waited=50.0, planned=9.0, acted=None)
    assert tel.rounds[-1].waited == 50.0
    assert tel.rounds[-1].planned == 9.0


def test_pacing_reports_the_median_not_the_mean():
    """One round parked in a stalled waitfor drags a mean past every
    ordinary round, and the ordinary round is the complaint."""
    tel = Telemetry()
    tel.start_fight()
    for n, waited in enumerate([10.0, 12.0, 11.0, 600.0], start=1):
        tel.observe(_Decision("Sunbird"), _read(2000 - n, n))
        tel.time_round(waited=waited, planned=2.0, acted=1.0)
    pace = tel.pacing()
    assert pace["rounds"] == 4
    assert pace["waited"] == pytest.approx(11.5)
    assert pace["planned"] == pytest.approx(2.0)


def test_pacing_does_not_measure_across_a_fight_boundary():
    """The gap between the last round of one duel and the first of the
    next is questing, not a round — and it is minutes long."""
    tel = Telemetry()
    tel.start_fight()
    tel.observe(_Decision("Sunbird"), _read(2000, 1)).at = 10.0
    tel.observe(_Decision("Sunbird"), _read(1900, 2)).at = 25.0
    tel.start_fight()
    tel.observe(_Decision("Sunbird"), _read(2000, 1)).at = 900.0
    assert tel.pacing()["round_to_round"] == pytest.approx(15.0)


def test_pacing_says_nothing_rather_than_zero_when_nothing_was_timed():
    tel = Telemetry()
    tel.start_fight()
    tel.observe(_Decision("Sunbird"), _read(2000, 1))
    pace = tel.pacing()
    assert pace["waited"] is None and pace["planned"] is None


def test_the_questing_log_and_the_rounds_share_one_clock():
    """Otherwise "the party was forty seconds a round while the desync
    was open" cannot be asked of the export at all."""
    tel = Telemetry()
    tel.note_questing("zone", "somewhere -> elsewhere")
    tel.start_fight()
    rec = tel.observe(_Decision("Sunbird"), _read(2000, 1))
    assert rec.at >= tel.questing[0]["at"]


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
                 loading=0, zone="Wizard City", lands=True, quest_at=None):
        self.lands = lands
        #: where the marker is, when it is somewhere other than where
        #: the wizard already stands. Defaults to the wizard's own
        #: position, which is what every test written before arrival
        #: was checked assumes.
        self.quest_at = quest_at
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
                if client.quest_at is not None:
                    return client.quest_at
                if client._position is None:
                    raise RuntimeError("no quest")
                return client._position

        self.quest_position = _QP()

    async def teleport(self, position):
        self.teleported.append(position)
        # Arrival is part of the contract now: `teleport_to_quest`
        # checks where the wizard ended up, because a teleport that
        # silently did nothing used to return the same answer as one
        # that worked. `lands` False models exactly that failure.
        if self.lands:
            self._position = position

    @property
    def body(self):
        client = self

        class _Body:
            async def position(self):
                return client._position

        return _Body()

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


def test_advance_dialogue_does_not_sleep_out_the_settle_on_every_click():
    """It used to sleep a flat `settle` after every click. On a
    five-window NPC that is two and a half seconds of a wizard standing
    still, reported from a live run as auto-dialogue skipping "very
    slow".

    The click has landed when the page turns or the box closes, and both
    are readable, so it polls for that instead. Timed rather than
    asserted structurally: the claim is about wall clock.
    """
    import asyncio
    import time

    from deimos_bridge.questing import advance_dialogue

    pages = ["page one", "page two", "page three"]

    class _Text:
        async def name(self):
            return "txtMessage"

        async def children(self):
            return []

        async def maybe_text(self):
            return pages[min(len(clicked), len(pages) - 1)]

        async def is_visible(self):
            return True

    clicked = []

    class _Button:
        async def name(self):
            return "btnRight"

        async def children(self):
            return []

        async def is_visible(self):
            return len(clicked) < len(pages)

    button, text = _Button(), _Text()

    class _Mouse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def click_window(self, win):
            clicked.append(win)

    class _Client:
        mouse_handler = _Mouse()
        root_window = _Win("root", [
            _Win("WorldView", [
                _Win("wndDialogMain", [button, _Win("txtArea", [text])])])])

    started = time.monotonic()
    clicks, why = asyncio.run(
        advance_dialogue(_Client(), settle=0.5, poll=0.01))
    took = time.monotonic() - started

    assert clicks == 3 and why == ""
    # Three clicks at the old flat rate is 1.5s. The page turns
    # immediately here, so it should cost about three polls.
    assert took < 0.5, f"three clicks took {took:.2f}s; the settle is 0.5 each"


def test_a_dialogue_that_never_changes_still_waits_the_full_settle():
    """The fall-through, which is what keeps the worst case exactly what
    it was: if nothing observable changes -- including when the text
    cannot be read at all -- it waits `settle` as before."""
    import asyncio
    import time

    from deimos_bridge.questing import _dialogue_moved

    class _Stuck:
        root_window = _Win("root", [
            _Win("WorldView", [
                _Win("wndDialogMain", [_Win("btnRight"),
                                       _Win("txtArea",
                                            [_Win("txtMessage",
                                                  text="same")])])])])

    from deimos_bridge.questing import STALLED

    started = time.monotonic()
    moved = asyncio.run(_dialogue_moved(_Stuck(), "same", 0.2, 0.02))
    took = time.monotonic() - started
    # STALLED, not merely falsy: the box is up, its text read, and the
    # text did not change, so the click provably did not take. That is a
    # different fact from "could not tell", and `advance_dialogue` acts
    # on one and not the other.
    assert moved == STALLED
    assert took >= 0.2, f"gave up after {took:.2f}s of a 0.2s settle"


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


def test_a_quest_teleport_that_does_not_land_is_retried():
    """The operator's point: "the questing scripts ... add protections
    at the very least to the quest marker teleports which often fail".
    This was one bare `client.teleport(position)` with no check that it
    landed, while the quester wraps every `tp quest` in `times 10`."""
    import asyncio

    from deimos_bridge.questing import QUEST_TP_TRIES, teleport_to_quest
    root, _ = _dialogue_root(visible=False)

    client = _QuestClient(root, lands=False,     # teleports do nothing
                          quest_at=_XYZ(9000.0, 9000.0, 0.0))
    ok, reason = asyncio.run(teleport_to_quest(client))
    assert ok is False
    assert len(client.teleported) >= QUEST_TP_TRIES, \
        f"gave up after {len(client.teleported)} attempt(s)"
    assert "would not take after" in reason and "away and the cutoff" in reason


def test_a_quest_teleport_bumps_out_of_bounds_when_plain_attempts_fail():
    """`tp XYZ(0,100000,0)` appears 116 times in the quester, always just
    before a real teleport: the game clamps the wizard back to the
    nearest valid point, which forces the position update the next
    teleport needs. A teleport issued from a position the client already
    thinks is correct is the one that silently does nothing."""
    import asyncio

    from deimos_bridge.questing import NOWHERE, teleport_to_quest
    root, _ = _dialogue_root(visible=False)

    client = _QuestClient(root, lands=False,
                          quest_at=_XYZ(9000.0, 9000.0, 0.0))
    asyncio.run(teleport_to_quest(client))
    bumped = [p for p in client.teleported
              if abs(getattr(p, "y", 0) - NOWHERE[1]) < 1]
    assert bumped, "never tried the out-of-bounds bump"


def test_a_quest_teleport_that_works_does_not_pay_for_the_protections():
    """The common case is a teleport that simply works, and a party of
    four should not pay twenty-four window lookups a hop for it."""
    import asyncio

    from deimos_bridge.questing import teleport_to_quest
    root, _ = _dialogue_root(visible=False)

    client = _QuestClient(root)
    ok, reason = asyncio.run(teleport_to_quest(client))
    assert ok is True and reason == ""
    assert len(client.teleported) == 1, "retried a teleport that worked"
    assert not client.mouse_handler.clicks, "closed menus it did not need to"


def test_close_menus_shuts_every_window_it_can_reach():
    """wizAi knew about one of these paths (`NPCServicesWin`) and the
    quester closes twenty-four, 63 times across the program -- once
    after essentially every quest teleport. A window left open blocks
    movement AND reads to `is_loading()` as a loading screen, so the
    next teleport is refused too."""
    import asyncio

    from deimos_bridge.questing import close_menus

    shop = _Win("exit", visible=True)
    tournament = _Win("exit", visible=True)
    hidden = _Win("exit", visible=False)
    root = _Win("root", [
        _Win("WorldView", [
            _Win("shopGUI", [_Win("buyWindow", [shop])]),
            _Win("TournamentRanking", [tournament]),
            _Win("ClassPicture", [hidden]),
        ])])
    client = _QuestClient(root)
    closed = asyncio.run(close_menus(client))
    assert closed == 2, f"closed {closed}"
    assert shop in client.mouse_handler.clicks
    assert tournament in client.mouse_handler.clicks
    assert hidden not in client.mouse_handler.clicks


def test_close_menus_keeps_going_past_a_window_that_will_not_close():
    """One button whose path has moved must not stop the other
    twenty-three. The point is to leave the wizard able to move."""
    import asyncio

    from deimos_bridge.questing import close_menus

    class _Angry(_Win):
        async def is_visible(self):
            raise RuntimeError("this window is not answering")

    root = _Win("root", [
        _Win("WorldView", [
            _Win("TournamentRanking", [_Angry("exit", visible=True)]),
            _Win("shopGUI", [_Win("buyWindow", [_Win("exit", visible=True)])]),
        ])])
    client = _QuestClient(root)
    assert asyncio.run(close_menus(client)) == 1


def test_x_is_pressed_until_something_happens_not_once(monkeypatch):
    """X is a proximity interact and does nothing until the wizard has
    finished arriving, which after a teleport is a second or two. One
    press lands in the gap and the step never starts -- the quester
    presses up to twenty."""
    import asyncio

    from deimos_bridge import questing
    from deimos_bridge.questing import press_x_until
    monkeypatch.setattr(questing, "keycode_x", lambda: "X")
    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root)

    state = {"n": 0}

    async def done():
        state["n"] += 1
        return state["n"] > 4              # the interact takes a moment

    pressed, why = asyncio.run(press_x_until(client, done, tries=20))
    assert why == ""
    assert 1 <= pressed <= 5, f"pressed {pressed}"
    assert len(client.keys) == pressed


def test_pressing_x_gives_up_rather_than_hammering_forever(monkeypatch):
    import asyncio

    from deimos_bridge import questing
    from deimos_bridge.questing import press_x_until
    monkeypatch.setattr(questing, "keycode_x", lambda: "X")
    root, _ = _dialogue_root(visible=False)
    client = _QuestClient(root)

    async def never():
        return False

    pressed, _why = asyncio.run(press_x_until(client, never, tries=6))
    assert pressed == 6


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
        # Waits for the REPORT, not for the first teleport. A quest
        # teleport is a retry loop now -- it checks where the wizard
        # landed and tries again -- so the first `client.teleport` is
        # the middle of the action rather than the end of it.
        for _ in range(120):                # ~6s of 50ms ticks
            await asyncio.sleep(0.05)
            if any("teleported" in m for m in said):
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


def _party_worker(goals=("a", "b"), ahead_at=(10.0, 5.0)):
    """A two-seat worker with each wizard's goal and when it last moved."""
    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                        seats=[SeatConfig(school="fire", deck=[],
                                          policy_name="school-aware")])
    worker.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    worker.DESYNC_GRACE = 0.0
    for seat, goal, at in zip(worker.seats, goals, ahead_at):
        seat.goal, seat.goal_at = goal, at
        seat.client = object()
    return worker


def test_the_wizard_that_got_ahead_goes_back_for_the_other(qapp):
    """The reported failure: "sometimes one will get ahead of the other
    and it'll keep going despite being on a different quest".

    The script cannot see it. Every quest test in all three arc questers
    is `p<N> tracking_goal "<literal>"` -- 1,700 of them -- and
    deimoslang has no form that compares two clients to each other
    (`vm.py:766-796` asserts the operand is a string). So a wizard that
    falls behind lands in a state no branch covers, and the program
    marches on.
    """
    worker = _party_worker(goals=("Talk to Zaltanna", "Defeat Krokopatra"),
                           ahead_at=(50.0, 10.0))
    a, b = worker.seats
    worker._check_in_step(a)
    worker._check_in_step(a)

    # `a` advanced most recently, so `b` is the one still on the missed
    # step -- quest names have no order, but "who moved last" does.
    assert worker._behind is b
    assert worker._should_catch_up(a), "the wizard ahead should go back"
    assert not worker._should_catch_up(b), \
        "the wizard that is BEHIND must keep the script; only it knows " \
        "how to finish the step"


def test_nobody_goes_anywhere_while_the_party_is_in_step(qapp):
    worker = _party_worker(goals=("Talk to Zaltanna", "Talk to Zaltanna"))
    a, b = worker.seats
    worker._check_in_step(a)
    worker._check_in_step(a)
    assert worker._behind is None
    assert not worker._should_catch_up(a)
    assert not worker._should_catch_up(b)


def test_a_party_that_lines_up_again_stops_chasing(qapp):
    """The laggard has to be cleared, or the wizard ahead keeps being
    dragged back to a wizard that caught up ten minutes ago."""
    worker = _party_worker(goals=("Talk to Zaltanna", "Defeat Krokopatra"),
                           ahead_at=(50.0, 10.0))
    a, b = worker.seats
    worker._check_in_step(a)
    worker._check_in_step(a)
    assert worker._should_catch_up(a)

    b.goal = "Talk to Zaltanna"          # caught up
    worker._check_in_step(a)
    assert worker._behind is None
    assert not worker._should_catch_up(a)


def test_a_solo_wizard_never_goes_back_for_anyone(qapp):
    from deimos_bridge.gui.live import LiveWorker

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    worker.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    assert not worker._should_catch_up(worker.seats[0])


def test_the_catch_up_aims_at_the_laggard_not_the_configured_leader(qapp):
    """`follow_leader` points at a fixed seat. The wizard to go back for
    is whoever fell behind, which is usually not that one."""
    import asyncio

    from deimos_bridge import party as party_mod

    worker = _party_worker(goals=("Talk to Zaltanna", "Defeat Krokopatra"),
                           ahead_at=(50.0, 10.0))
    a, b = worker.seats
    worker.leader = 0                     # the CONFIGURED leader is `a`
    worker._check_in_step(a)
    worker._check_in_step(a)

    went = []

    async def _follow(client, leader, leader_name=""):
        went.append(leader)
        return True, "teleported"

    real, party_mod.follow = party_mod.follow, _follow
    try:
        asyncio.run(worker._catch_up(a))
    finally:
        party_mod.follow = real

    assert went == [b.client], "went to the configured leader, not the laggard"


def test_a_script_hammering_a_loop_that_never_works_says_so(qapp):
    """Across KamarJ's three arc questers there are 2,438 bounded retry
    loops and 528 unbounded ones, and `tp` alone appears in 331 of the
    unbounded ones -- because Deimos's `navmap_tp` returns nothing at
    all, so a script cannot ask whether a teleport landed and has to
    poll windows and try again.

    When that never succeeds the run still looks alive from outside:
    instructions execute, the counter climbs, and the wizard stands
    still. That is "one wizard might get through, while the other is
    still trying to", and nothing could see it.
    """
    import time

    from deimos_bridge.gui.live import LiveWorker

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()

    class _Runner:
        running = True
        steps = 41_233

    seat = worker.seats[0]
    seat.runner = _Runner()
    seat.progress = ("Krokotopia/KT_Interior", (3, 1, 0), "Defeat Krokopatra")
    seat.progress_at = time.monotonic() - (worker.STUCK_AFTER + 1)

    worker._check_progress(seat)
    assert any("retry loop that is not working" in m for m in said), said
    # Named: which zone and which goal, plus how far the script has got.
    assert any("Defeat Krokopatra" in m and "41,233" in m for m in said), said

    # Said once per distinct situation, not every half-second.
    said.clear()
    worker._check_progress(seat)
    assert said == [], said


def test_progress_is_not_stuck_while_something_is_changing(qapp):
    import time

    from deimos_bridge.gui.live import LiveWorker

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()

    class _Runner:
        running = True
        steps = 10

    seat = worker.seats[0]
    seat.runner = _Runner()
    seat.progress = ("Wizard City", (1, 1, 1), "Talk to Ambrose")
    seat.progress_at = time.monotonic()        # just moved
    worker._check_progress(seat)
    assert said == [], said


def test_a_wizard_no_script_is_driving_is_not_called_stuck(qapp):
    """Standing still between fights is not a stuck script, and the run
    must not narrate it as one."""
    import time

    from deimos_bridge.gui.live import LiveWorker

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()

    seat = worker.seats[0]
    seat.runner = None                          # nothing is driving
    seat.progress = ("Wizard City", (1, 1, 1), "Talk to Ambrose")
    seat.progress_at = time.monotonic() - 10_000
    worker._check_progress(seat)
    assert said == [], said


def test_breathing_does_not_count_as_progress(qapp):
    """A wizard's idle animation moves it by a fraction constantly. A
    progress check that counts that as movement never fires."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    class _XYZ:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    drift = [0.0]

    class _Body:
        async def position(self):
            drift[0] += 0.4                    # a fraction, every read
            return _XYZ(100.0 + drift[0], 200.0, 0.0)

    class _Client:
        body = _Body()
        root_window = _Win("root", [])

        async def zone_name(self):
            return "Wizard City"

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    worker.status = type("S", (), {"emit": staticmethod(lambda *_: None)})()
    worker.GOAL_POLL = 0.0
    seat = worker.seats[0]
    seat.client = _Client()

    asyncio.run(worker._read_goal(seat))
    first = seat.progress_at
    for _ in range(5):
        asyncio.run(worker._read_goal(seat))
    assert seat.progress_at == first, "idle drift reset the progress clock"


def _goal_client(goal):
    """A client whose quest tracker reads `goal`."""
    text = _Win("txtGoalName", text=goal)
    root = _Win("root", [
        _Win("WorldView", [
            _Win("windowHUD", [
                _Win("QuestHelperHud", [
                    _Win("ElementWindow", [_Win("", [text])])])])])])
    return _QuestClient(root)


def test_the_quest_goal_reads_off_the_tracker():
    import asyncio

    from deimos_bridge.questing import read_quest_goal

    assert asyncio.run(read_quest_goal(_goal_client("Defeat Krokopatra"))) \
        == "Defeat Krokopatra"
    # A tracker that is not there reads empty rather than raising -- this
    # feeds a comparison, and "could not read" must never look like "on a
    # different quest".
    assert asyncio.run(read_quest_goal(_QuestClient(_Win("root", [])))) == ""


def test_a_party_on_two_different_quests_is_reported(qapp):
    """The reported failure: "it's not uncommon that one wizard will get
    ahead a quest or 2 from the other because one misses a dialogue or
    fails a teleport". With a script driving both clients from one
    program, the program cannot notice -- its own instruction pointer is
    fine. What diverged is the game's quest state."""
    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                        seats=[SeatConfig(school="fire", deck=[],
                                          policy_name="school-aware")])
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()
    worker.DESYNC_GRACE = 0.0

    a, b = worker.seats
    a.goal, b.goal = "Talk to Zaltanna", "Defeat Krokopatra"
    worker._check_in_step(a)          # first call only starts the clock
    worker._check_in_step(a)
    assert any("different quests" in m for m in said), said
    # Named, not counted: which wizard is on what.
    assert any("Talk to Zaltanna" in m and "Defeat Krokopatra" in m
               for m in said), said


def test_a_brief_disagreement_is_not_a_desync(qapp):
    """Turning a step in is not simultaneous -- one wizard clicks the NPC
    seconds before the other -- so a bare inequality would report a
    desync on every normal handover."""
    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                        seats=[SeatConfig(school="fire", deck=[],
                                          policy_name="school-aware")])
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()

    a, b = worker.seats
    a.goal, b.goal = "Talk to Zaltanna", "Defeat Krokopatra"
    for _ in range(5):
        worker._check_in_step(a)      # well inside the default grace
    assert said == [], said

    # ...and once they line up again the clock resets, so the next
    # handover gets the full grace of its own.
    b.goal = "Talk to Zaltanna"
    worker._check_in_step(a)
    assert said == [], said


def test_an_unreadable_tracker_is_never_a_desync(qapp):
    """A run with the quest tracker hidden must not report a desync
    every tick. Fewer than two readable goals is agreement."""
    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                        seats=[SeatConfig(school="fire", deck=[],
                                          policy_name="school-aware")])
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()
    worker.DESYNC_GRACE = 0.0

    a, b = worker.seats
    a.goal, b.goal = "Talk to Zaltanna", ""
    for _ in range(5):
        worker._check_in_step(a)
    assert said == [], said


def test_a_single_wizard_is_never_out_of_step_with_itself(qapp):
    from deimos_bridge.gui.live import LiveWorker

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()
    worker.DESYNC_GRACE = 0.0
    worker.seats[0].goal = "Talk to Zaltanna"
    worker._check_in_step(worker.seats[0])
    assert said == [], said


def _hurt_client(fraction, heals_after=None):
    """A client on `fraction` health, optionally recovering after N reads."""
    class _Stats:
        reads = 0

        async def current_hitpoints(self):
            _Stats.reads += 1
            f = fraction
            if heals_after is not None and _Stats.reads > heals_after:
                f = 1.0
            return 1000.0 * f

        async def max_hitpoints(self):
            return 1000.0

    class _Client:
        stats = _Stats()
    return _Client()


def test_a_wizard_on_almost_no_health_does_not_start_another_fight(qapp):
    """The reported failure: "if they die on a dungeon they might try
    again immediately even though their health is essentially 0".

    `upkeep.after_fight` tops up, and when it cannot -- an empty potion
    bottle, which it says out loud -- the run went into the next fight
    anyway, which on a dungeon means dying, walking back, and dying
    again.
    """
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()
    worker.LOW_HEALTH_POLL = 0.01
    worker.LOW_HEALTH_WAIT = 0.05

    seat = worker.seats[0]
    seat.client = _hurt_client(0.04)

    asyncio.run(worker._let_it_heal(seat))
    assert any("4% health" in m for m in said), said
    # ...and it does not block the run forever on a bottle it cannot fill
    assert any("anyway" in m for m in said), said


def test_it_stops_waiting_the_moment_the_health_is_back(qapp):
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()
    worker.LOW_HEALTH_POLL = 0.01
    worker.LOW_HEALTH_WAIT = 30.0          # would never expire in this test

    seat = worker.seats[0]
    seat.client = _hurt_client(0.10, heals_after=1)

    asyncio.run(worker._let_it_heal(seat))
    assert any("carrying on" in m for m in said), said
    assert not any("anyway" in m for m in said), said


def test_a_healthy_wizard_is_not_delayed_at_all(qapp):
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()
    seat = worker.seats[0]
    seat.client = _hurt_client(0.80)

    asyncio.run(worker._let_it_heal(seat))
    assert said == [], f"a wizard on 80% was made to wait: {said}"


def test_an_unreadable_health_bar_never_strands_the_run(qapp):
    """None rather than a guess. Refusing to fight because a stat read
    raised would strand a healthy wizard, which is worse than the
    failure this guards against."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker

    class _Broken:
        class stats:
            @staticmethod
            async def current_hitpoints():
                raise RuntimeError("the client went away")

            @staticmethod
            async def max_hitpoints():
                raise RuntimeError("the client went away")

    worker = LiveWorker(Telemetry(), "ice", [], "school-aware", 1)
    said = []
    worker.status = type("S", (), {"emit": staticmethod(said.append)})()
    seat = worker.seats[0]
    seat.client = _Broken()

    assert asyncio.run(worker._health_left(seat)) is None
    asyncio.run(worker._let_it_heal(seat))
    assert said == [], said


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


def test_buying_survives_the_keyboardinterrupt_deimos_raises(monkeypatch):
    """`src/utils.buy_potions` ends `except: raise KeyboardInterrupt`
    (utils.py:487-489). That is a BaseException, so an ordinary
    `except Exception` does not catch it and one failed purchase would
    tear down the whole run."""
    import asyncio

    from deimos_bridge import upkeep

    async def _boom(client, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(upkeep, "_refill", _boom)
    said = []
    assert asyncio.run(upkeep.buy_potions(_UpkeepClient(),
                                          on_status=said.append)) is False
    assert any("KeyboardInterrupt" in m for m in said), said

    # It has to be converted INSIDE the task. `asyncio.wait_for` runs the
    # coroutine as a Task, and a Task raising a BaseException that is not
    # an Exception propagates past the awaiting frame and out of the loop
    # -- so a handler wrapped around the `wait_for` never fires. Measured
    # here rather than asserted, because it is the whole reason
    # `_refill_guarded` exists.
    async def _escapes():
        async def boom():
            raise KeyboardInterrupt
        try:
            await asyncio.wait_for(boom(), 5)
        except (Exception, KeyboardInterrupt):
            return "caught"
        return "no exception"

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(_escapes())


def test_buying_never_swallows_a_cancellation(monkeypatch):
    """`CancelledError` is a BaseException too, and catching it would
    make stopping the run stop meaning anything."""
    import asyncio

    from deimos_bridge import upkeep

    async def _cancel(client, **kw):
        raise asyncio.CancelledError

    monkeypatch.setattr(upkeep, "_refill", _cancel)

    async def drive():
        with pytest.raises(asyncio.CancelledError):
            await upkeep.buy_potions(_UpkeepClient())

    asyncio.run(drive())


def test_a_vendor_trip_only_happens_for_an_empty_bottle(monkeypatch):
    """The one failure a vendor can fix. A click that failed or a read
    that raised will not be any better after crossing two zones."""
    import asyncio

    from deimos_bridge import upkeep

    trips = []

    async def _no_wisps(client, **kw):
        return 0

    async def _needs(client, **kw):
        return True

    async def _buy(client, on_status=None):
        trips.append(client)
        return False

    monkeypatch.setattr(upkeep, "collect_wisps", _no_wisps)
    monkeypatch.setattr(upkeep, "needs_potion", _needs)
    monkeypatch.setattr(upkeep, "buy_potions", _buy)

    async def _empty(client):
        _empty.last_error = "your potion bottle is empty — this never buys refills"
        return False
    _empty.last_error = ""

    async def _broke(client):
        _broke.last_error = "could not read your potion charges (RuntimeError: x)"
        return False
    _broke.last_error = ""

    monkeypatch.setattr(upkeep, "drink_potion", _empty)
    asyncio.run(upkeep.after_fight(_UpkeepClient(), buy=True))
    assert len(trips) == 1, "an empty bottle should send it to the vendor"

    trips.clear()
    monkeypatch.setattr(upkeep, "drink_potion", _broke)
    asyncio.run(upkeep.after_fight(_UpkeepClient(), buy=True))
    assert trips == [], "a failed READ is not something a vendor can fix"

    trips.clear()
    monkeypatch.setattr(upkeep, "drink_potion", _empty)
    asyncio.run(upkeep.after_fight(_UpkeepClient(), buy=False))
    assert trips == [], "buying must not happen unless it was asked for"


def test_buy_refills_is_off_until_it_is_switched_on(qapp):
    """It spends real gold and crosses two zone changes, either of which
    can leave the wizard somewhere the quest is not. Worth doing when a
    run would otherwise halt on an empty bottle; not worth doing behind
    anyone's back."""
    from deimos_bridge.gui.app import MainWindow
    from deimos_bridge.gui.live import LiveWorker

    win = MainWindow(Telemetry())
    assert not win.buy_potions.isChecked()
    assert LiveWorker(Telemetry(), "ice", [], "school-aware", 1).buy_potions \
        is False

    # ...and it is a LIVE toggle, like the chores beside it, so it can be
    # turned on without restarting a run.
    assert "buy_potions" in MainWindow.LIVE_TOGGLES


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
    # Board, Decisions, Damage model, Learning, Naming, Runs,
    # Questing, Party.
    assert win.tabs.count() == 8
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
        gave_up = []

        async def decide(self):
            return await decide()

        def report_failed_cast(self, name, exc):
            self.failed.append((name, type(exc).__name__))

        def report_recovery_failed(self, why):
            self.gave_up.append(why)

        def report_lost_round(self, number, reason):
            self.lost.append((number, reason))

        def report_round_timing(self, waited, planned, acted):
            self.timed.append((waited, planned, acted))

    _Backend.timed = []

    class _Client:
        mouse_handler = _Mouseless()

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    handler.client = _Client()
    handler.backend = _Backend()
    handler._last_read = None
    handler._read_failures = 0
    handler._left_round_at = None
    handler._planned = None
    handler._last_round_number = None
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
    assert handler.backend.gave_up and \
        "nothing else in hand" in handler.backend.gave_up[0], \
        "a passed round has to say whether anything else was tried"


def test_every_way_out_of_a_round_still_says_what_it_cost():
    """The round that ends badly is the one whose cost matters, and
    `_handle_round` has five ways out. So the stopwatch is a wrapper
    with a `finally`, not a line threaded through the body."""
    import asyncio

    from deimos_bridge.live_backend import PolicyDecision

    class _Card:
        async def cast(self, target, **kw):
            raise RuntimeError("the board moved")

    async def _cast_failed():
        return PolicyDecision(card_name="Sunbird", target_index=0)

    async def _read_failed():
        raise RuntimeError("could not read the board")

    async def _passed():
        return PolicyDecision(passing=True, reason="nothing worth casting")

    for decide, card in ((_cast_failed, _Card()), (_read_failed, None),
                         (_passed, None)):
        handler, _passes = _stub_handler(decide, read=object(), card=card)
        handler.backend.timed = []
        asyncio.run(handler.handle_round())
        assert len(handler.backend.timed) == 1, \
            f"{decide.__name__} left the round untimed"
        waited, planned, acted = handler.backend.timed[0]
        assert waited is None, "there was no previous round to wait after"
        assert planned is not None and planned >= 0.0
        assert acted is not None and acted >= 0.0


def test_the_wait_is_measured_from_the_last_round_leaving_not_arriving():
    """`waited` is the GAME's time — animations, the other wizards, the
    planning timer. Measured from when we let go of the previous round,
    so our own planning is never counted twice."""
    import asyncio

    from deimos_bridge.live_backend import PolicyDecision

    async def _pass():
        return PolicyDecision(passing=True, reason="pass")

    handler, _passes = _stub_handler(_pass, read=object())
    # Climbing, because a round number that does not move is a NEW duel
    # — see `_same_duel`, and the test below it.
    rounds = iter([1, 2])
    handler.round_number = lambda: _value(next(rounds))
    handler.backend.timed = []
    asyncio.run(handler.handle_round())
    left = handler._left_round_at
    assert left is not None
    asyncio.run(handler.handle_round())
    assert handler.backend.timed[1][0] is not None, \
        "the second round did not measure the gap before it"
    assert handler.backend.timed[1][0] >= 0.0


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

    src = inspect.getsource(live_backend.WizAiCombatHandler._handle_round)
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
    assert be._incoming == [0.0, 57.0, 177.0]      # the zero is IN there

    # ...and the estimate is those three weighed against the prior, which
    # is worth `INCOMING_PRIOR_WEIGHT` rounds of its own rather than
    # being discarded by the first real reading. See
    # `test_one_quiet_round_does_not_make_the_board_harmless`.
    prior = max(30.0, 589 / be.INCOMING_PRIOR_DIVISOR)
    w = be.INCOMING_PRIOR_WEIGHT
    assert per == pytest.approx((0 + 57 + 177 + prior * w) / (3 + w))

    # A healing round (hp went UP) stays out: that is the heal's
    # number, not the board's.
    be._estimate_incoming(read_at(500, 5))
    assert len(be._incoming) == 3


def test_one_quiet_round_does_not_make_the_board_harmless():
    """The prior was a fallback, so the FIRST real reading replaced it
    outright -- and a first reading of zero set the whole threat model to
    exactly zero.

    Both exports from the first live party run show it. Jeffrey's
    `incoming` reads 52.92 at fight 1 round 2, which is the prior itself
    (1270/(12*2)), then 0.0 at rounds 3 and 4 because the rounds between
    those reads happened to be quiet. Konstantin's does the same at
    fight 1 round 2.

    Zero incoming is not a small error, it is a different game: the mobs
    deal nothing, `_rollout`'s `player.alive` check can never fire,
    `died()` is unreachable and mitigation is worth literally nothing --
    so setting up costs only turns and a shield scores exactly like
    passing. The two rounds Jeffrey spent on Tower Shield at full health
    were both planned against a board the model believed was harmless,
    in the fight that then killed him.
    """
    from deimos_bridge.live_backend import WizAiBackend
    from w101_sim import Actor, State

    be = WizAiBackend(policy=lambda sim, s: None, cards={}, school="ice")
    me = Actor(name="Jeffrey", school="ice", hp=1270, max_hp=1270, team=0)
    foes = [Actor(name=f"Sand Stalker {i}", school="balance", hp=435,
                  max_hp=435, team=1) for i in range(3)]

    def read_at(hp, rnd):
        class _R:
            state = State(me, foes)
            round_number = rnd
        me.hp = hp
        return _R()

    first = be._estimate_incoming(read_at(1270, 2))
    assert first == pytest.approx(1270 / 12), "round one is still the prior"

    # Two quiet rounds in a row -- exactly Jeffrey's fight 1.
    quiet = be._estimate_incoming(read_at(1270, 3))
    quieter = be._estimate_incoming(read_at(1270, 4))
    assert be._incoming == [0.0, 0.0], "the quiet rounds are still counted"
    assert quiet > 30.0 and quieter > 30.0, (quiet, quieter)

    # It does still move -- this is a slower estimate, not a frozen one.
    assert quieter < quiet < first

    # And it converges on what the board is really doing rather than
    # sitting on the prior: 190 lost across three mobs is 63.3 each.
    for hp in (1080, 890, 700, 510, 320):
        be._estimate_incoming(read_at(hp, 5))
    assert be._estimate_incoming(read_at(130, 10)) == pytest.approx(
        (sum(be._incoming) + (1270 / 12) * be.INCOMING_PRIOR_WEIGHT)
        / (len(be._incoming) + be.INCOMING_PRIOR_WEIGHT))


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


# ------------------------------------------------- who a heal is clicked on
def _healer_read(my_hp=500, my_max=1000, ally_hp=None, ally_max=1000):
    """A duel read: me, some mobs, optionally one teammate."""
    from w101_sim import Actor, State
    from deimos_bridge.live_state import LiveRead

    me = Actor(name="Konstantin", school="fire", hp=my_hp, max_hp=my_max,
               team=0)
    mobs = [Actor(name="Sokkwi Ripper", school="balance", hp=310, max_hp=310,
                  team=1)]
    allies, ally_members = [], []
    if ally_hp is not None:
        allies.append(Actor(name="Jeffrey", school="ice", hp=ally_hp,
                            max_hp=ally_max, team=0))
        ally_members.append("member:Jeffrey")
    state = State(me, mobs, allies)
    # `get_members()` returns the enemy team FIRST, which is the trap.
    members = ["member:Sokkwi Ripper", "member:me"] + ally_members
    return LiveRead(state, {}, _Resolver(), members, "member:me", 1,
                    enemy_members=["member:Sokkwi Ripper"],
                    ally_members=ally_members)


class _Heal:
    """A card whose ops say it goes on a friend, as Pixie's do."""
    ops = [{"op": "heal", "tgt": "ally"}]


def test_a_heal_is_never_clicked_on_a_mob(qapp):
    """The stall. `read.members` is every participant in the duel, both
    sides, and the enemy team comes first — so "the first member that is
    not me" is a MOB, and in a solo fight there is nothing else in it.

    Wizard101 does not cast a friend-only spell at an enemy; the second
    click deselects the card. Pixie was selected, aimed at a Sokkwi
    Ripper, unselected, and the round passed with nothing cast — the
    duel then sat there until the round timer expired.
    """
    import asyncio

    from deimos_bridge.live_backend import WizAiCombatHandler

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)

    read = _healer_read(my_hp=500)          # solo: no teammate at all
    got = asyncio.run(handler._resolve_target(read, None, _Heal()))
    assert got == "member:me", f"a heal was aimed at {got!r}"
    assert got not in read.enemy_members


def test_a_heal_goes_to_whoever_is_hurt(qapp):
    """It is a heal. The caster whenever it is the worst off, and the
    teammate when they are — never a mob either way."""
    import asyncio

    from deimos_bridge.live_backend import WizAiCombatHandler

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)

    # teammate is on 200/1000, I am on 900/1000 -> heal them
    read = _healer_read(my_hp=900, ally_hp=200)
    assert asyncio.run(handler._resolve_target(read, None, _Heal())) \
        == "member:Jeffrey"

    # I am the hurt one -> heal me
    read = _healer_read(my_hp=100, ally_hp=950)
    assert asyncio.run(handler._resolve_target(read, None, _Heal())) \
        == "member:me"


def test_an_attack_still_goes_to_the_chosen_mob(qapp):
    """The fix must not move where a nuke lands."""
    import asyncio

    from deimos_bridge.live_backend import WizAiCombatHandler

    class _Nuke:
        ops = [{"op": "hit", "tgt": "enemy"}]

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    read = _healer_read(my_hp=500, ally_hp=500)
    assert asyncio.run(handler._resolve_target(read, 0, _Nuke())) \
        == "member:Sokkwi Ripper"


class _Shield:
    """A card whose ops say `self`, as Tower Shield's do."""
    kind = "shield"
    ops = [{"op": "ward", "percent": -0.5, "schools": None, "tgt": "self"}]


def test_a_shield_goes_on_whoever_is_nearest_to_dying(qapp):
    """The second live party run: the ice wizard cast nineteen Tower
    Shields, four of them twice in a row, and in every one of those
    pairs it was at FULL health while the fire wizard beside it was not
    -- 1270/1270 against 938/1042, 922/1042, 889/1064 -- and the fire
    wizard was dealing more of the damage. Every shield went on the
    caster because the ops say `tgt: 'self'`, which is the right default
    and not the only legal target."""
    import asyncio

    from deimos_bridge.live_backend import WizAiCombatHandler

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)

    # fight 5 round 1: me 1270/1270, teammate 938/1042
    read = _healer_read(my_hp=1270, my_max=1270, ally_hp=938, ally_max=1042)
    assert asyncio.run(handler._resolve_target(read, None, _Shield())) \
        == "member:Jeffrey"

    # ...and it comes back to the caster the moment the caster is worse.
    read = _healer_read(my_hp=300, my_max=1270, ally_hp=1000, ally_max=1042)
    assert asyncio.run(handler._resolve_target(read, None, _Shield())) \
        == "member:me"


def test_a_shield_ranks_by_fraction_left_not_health_missing(qapp):
    """The difference between a shield and a heal, and the reason they
    do not share a rule. A heal restores what is gone, so most-missing
    is right; a shield buys rounds against what is coming, so what
    matters is who is nearest to dying."""
    import asyncio

    from deimos_bridge.live_backend import WizAiCombatHandler

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)

    # I am down 400 of 1270 (68% left); the teammate is down 300 of 600
    # (50% left). Most-missing says me; nearest-to-dying says them, and
    # they are the one two hits from the floor.
    read = _healer_read(my_hp=870, my_max=1270, ally_hp=300, ally_max=600)
    assert asyncio.run(handler._resolve_target(read, None, _Shield())) \
        == "member:Jeffrey"


def test_a_second_shield_does_not_pile_onto_the_already_shielded(qapp):
    """"Two 50% Tower Shields on themselves" is the reported symptom.
    They do stack in game, but a second one on a full-health caster
    while a teammate has none is not a stack, it is a wasted card."""
    import asyncio

    from w101_sim import Hanging
    from deimos_bridge.live_backend import WizAiCombatHandler

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)

    # I am healthier AND already shielded; the teammate is neither.
    read = _healer_read(my_hp=1270, my_max=1270, ally_hp=1000, ally_max=1042)
    read.state.player.wards.append(
        Hanging(name="Tower Shield", slot="ward", kind="damage",
                percent=-0.5))
    assert asyncio.run(handler._resolve_target(read, None, _Shield())) \
        == "member:Jeffrey"

    # A trap is a ward too, so it must never read as "already
    # shielded". Posed where only that can decide it: I am the one who
    # wins on health, so the ONLY thing that could push the shield onto
    # the teammate is the +40% trap being mistaken for protection.
    read = _healer_read(my_hp=500, my_max=1000, ally_hp=900, ally_max=1000)
    read.state.player.wards.append(
        Hanging(name="Ice Trap", slot="ward", kind="damage", percent=0.40,
                schools={"ice"}))
    assert asyncio.run(handler._resolve_target(read, None, _Shield())) \
        == "member:me", "a +40% trap was counted as a shield"


def test_a_solo_wizard_still_shields_itself(qapp):
    """No teammate to consider, and the caster wins ties, so nothing
    about a wizard fighting alone changes."""
    import asyncio

    from deimos_bridge.live_backend import WizAiCombatHandler

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    read = _healer_read(my_hp=500, my_max=1270)          # no ally at all
    assert asyncio.run(handler._resolve_target(read, None, _Shield())) \
        == "member:me"

    # level on both counts -> the caster, as before
    read = _healer_read(my_hp=500, my_max=1000, ally_hp=500, ally_max=1000)
    assert asyncio.run(handler._resolve_target(read, None, _Shield())) \
        == "member:me"


def test_a_blade_is_not_moved_to_a_teammate(qapp):
    """Blades are `tgt: 'self'` too and CAN legally go on a teammate,
    where they would buff that wizard's hit instead. That is a real
    hivemind play and a much larger decision -- who is about to hit
    hardest, not who is hurt -- so it is deliberately out of scope."""
    import asyncio

    from deimos_bridge.live_backend import WizAiCombatHandler

    class _Blade:
        kind = "blade"
        ops = [{"op": "charm", "percent": 0.35, "schools": ["ice"],
                "tgt": "self"}]

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    read = _healer_read(my_hp=1270, my_max=1270, ally_hp=100, ally_max=1042)
    assert asyncio.run(handler._resolve_target(read, None, _Blade())) \
        == "member:me"


def test_the_retry_drops_the_second_click_for_a_self_cast(qapp):
    """Retrying identically but slower tests timing, and the instrumented
    run answered that: not timing. Every self-targeted cast costing no
    pips went out -- eleven of eleven -- and every self-targeted cast
    costing pips failed, three of three, all Pixie. The diagnostic ruled
    out aim, affordability and castability.

    What is left is the second click. `CombatCard.cast` with a member
    clicks the card, waits, then clicks the target's HEALTH TEXT; with
    `None` it clicks once and stops. On a spell the game has already
    aimed at the only wizard it can go on, that second click is what the
    operator described unselecting it.
    """
    from deimos_bridge.live_backend import WizAiCombatHandler

    class _Pixie:
        kind = "heal"
        ops = [{"op": "heal", "tgt": "self"}]

    class _Nuke:
        kind = "damage"
        ops = [{"op": "hit", "tgt": "enemy"}]

    class _Backend:
        cards = {"Pixie": _Pixie(), "Sunbird": _Nuke()}

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    handler.backend = _Backend()
    read = _healer_read(my_hp=651, my_max=1064)

    class _D:
        card_name = "Pixie"

    # the first attempt aimed at the caster; the retry aims at nothing,
    # which is one click instead of two
    assert handler._retry_target(read, _D(), "member:me") is None

    # ...and a cast at a mob is untouched, because nothing about it is
    # in question
    class _E:
        card_name = "Sunbird"

    assert handler._retry_target(read, _E(), "member:Sokkwi Ripper") \
        == "member:Sokkwi Ripper"


def test_an_unknown_card_keeps_the_target_it_was_given(qapp):
    """A card the catalog cannot resolve must not have its aim silently
    dropped -- that would turn a targeted nuke into a bare click."""
    from deimos_bridge.live_backend import WizAiCombatHandler

    class _Backend:
        cards = {}

    class _D:
        card_name = "Something Unresolvable"

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    handler.backend = _Backend()
    assert handler._retry_target(_healer_read(), _D(), "member:mob") \
        == "member:mob"


def test_a_failed_cast_reports_facts_not_a_theory(qapp):
    """The message this replaced asserted a cause, and asserted the wrong
    one at the card it kept being printed on.

    "Wizard101 deselects a card clicked at something it cannot be cast
    on, so this is usually a card aimed at the wrong kind of target" is
    a fine explanation of *a* failure. It is not the explanation of
    Pixie's: Pixie is `tgt: 'self'`, so it is handed `read.client_member`
    down the identical branch Fireblade and Tower Shield take, and in
    the run that prompted this the fire wizard cast Fireblade in round 1
    and Pixie failed in round 3 of the same duel on the same client. The
    line has to carry what separates the surviving explanations instead.
    """
    import asyncio

    from deimos_bridge.live_backend import WizAiCombatHandler

    class _Card:
        name, school, pips = "Pixie", "life", 2
        kind = "heal"
        ops = [{"op": "heal", "tgt": "self"}]

    class _Blade:
        name, school, pips = "Fireblade", "fire", 0
        kind = "blade"
        ops = [{"op": "charm", "tgt": "self"}]

    class _Nuke:
        name, school, pips = "Sunbird", "fire", 3
        kind = "damage"
        ops = [{"op": "hit", "tgt": "enemy"}]

    class _Backend:
        school = "fire"
        cards = {"Pixie": _Card()}

    class _Clicked:
        async def is_castable(self):
            return True

    class _Decision:
        card_name = "Pixie"
        target_index = None

    read = _healer_read(my_hp=526, my_max=897)
    read.state.player.hand = [_Card(), _Blade(), _Nuke()]
    read.state.player.norm_pips, read.state.player.pow_pips = 1, 2

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    handler.backend = _Backend()
    said = asyncio.run(handler._why_not(read, _Decision(), _Clicked()))

    # Aiming is exonerated by naming the card that shares the aim and
    # went out -- that is the whole point of the line.
    assert "aimed at self" in said
    assert "Fireblade" in said
    assert "Sunbird" not in said, "only cards with the SAME aim are relevant"
    # A power pip is worth 2 only for the caster's own school, so an
    # off-school card is the one that can be unaffordable while looking
    # affordable. Here it was affordable, and the line says so.
    assert "off school" in said and "worth 1 each" in said
    assert "= 3" in said
    assert "still calls it castable" in said
    assert "wrong kind of target" not in said


def test_the_failure_line_survives_a_client_that_will_not_answer(qapp):
    """It runs while a round is already being given away. A read that
    fails here must not replace the failure being reported."""
    import asyncio

    from deimos_bridge.live_backend import WizAiCombatHandler

    class _Backend:
        school = "fire"
        cards = {}                       # the card is not even known

    class _Broken:
        async def is_castable(self):
            raise RuntimeError("the client went away")

    class _Decision:
        card_name = "Pixie"
        target_index = None

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    handler.backend = _Backend()
    said = asyncio.run(handler._why_not(_healer_read(), _Decision(), _Broken()))
    assert "aimed at nothing" in said       # said something, raised nothing


def test_a_card_that_will_not_go_out_costs_the_best_move_not_the_round(qapp):
    """The stall, as the operator described it: "it double clicks, which
    unselects the spell, so nothing happens and the fight stalls until
    the round counter hits 0".

    Against a board out-damaging the party a free round decides fights.
    The policy already ranked every other move it weighed, so the answer
    to "the best card did not go out" is the second best, not nothing.
    """
    import asyncio

    from deimos_bridge.live_backend import WizAiCombatHandler
    from deimos_bridge.policies import Candidate

    cast = []

    class _Card:
        def __init__(self, name):
            self.name = name

        async def cast(self, target, sleep_time=None):
            cast.append((self.name, target))

    class _Decision:
        card_name = "Pixie"
        target_index = None
        candidates = [
            Candidate(card="Pixie", target=None, turns=5, damage=400, pips=2),
            Candidate(card="Sunbird", target=0, turns=4, damage=338, pips=3),
            Candidate(card="Fire Cat", target=0, turns=6, damage=197, pips=1),
            Candidate(card="pass", target=None, turns=9, damage=0, pips=0),
        ]

    told = []

    class _Backend:
        school = "fire"
        cards = {"Sunbird": None}
        RETRY = 1.0

        def report_recovered_cast(self, pick, first):
            told.append((pick.card, pick.target, first))

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    handler.backend = _Backend()
    handler._pick_card = lambda read, name: _Card(name)
    handler._cards_in_hand = lambda: _value({"Pixie", "Sunbird", "Fire Cat"})
    handler._card_left_the_hand = lambda before: _value(True)

    read = _healer_read(my_hp=526, my_max=897)
    assert asyncio.run(handler._try_the_next_best(read, _Decision()))
    # Sunbird, not Fire Cat: the runner-up is taken by SCORE. The
    # candidate list is in the order the rollout scored the hand, so
    # "the next one along" would have played whatever came next.
    assert [c[0] for c in cast] == ["Sunbird"]
    assert told == [("Sunbird", 0, "Pixie")]


def test_the_fallback_gives_up_after_one_try(qapp):
    """If the board moved under us the alternative fails the same way,
    and a loop of them spends the planning window clicking."""
    import asyncio

    from deimos_bridge.live_backend import WizAiCombatHandler
    from deimos_bridge.policies import Candidate

    tried = []

    class _Card:
        def __init__(self, name):
            self.name = name

        async def cast(self, target, sleep_time=None):
            tried.append(self.name)

    class _Decision:
        card_name = "Pixie"
        target_index = None
        candidates = [
            Candidate(card="Pixie", target=None, turns=5, damage=400, pips=2),
            Candidate(card="Sunbird", target=0, turns=4, damage=338, pips=3),
            Candidate(card="Fire Cat", target=0, turns=6, damage=197, pips=1),
        ]

    said = []

    class _Backend:
        school = "fire"
        cards = {}

        def report_recovered_cast(self, pick, first):
            raise AssertionError("nothing went out; nothing to report")

        def report_recovery_failed(self, why):
            said.append(why)

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    handler.backend = _Backend()
    handler._pick_card = lambda read, name: _Card(name)
    handler._cards_in_hand = lambda: _value(set())
    handler._card_left_the_hand = lambda before: _value(False)   # never goes

    assert not asyncio.run(
        handler._try_the_next_best(_healer_read(), _Decision()))
    assert tried == ["Sunbird"], f"tried {tried}"
    # ...and says so. Rev bb8f2b3c passed a round with two castable
    # cards still in hand and left no record of having tried either.
    assert len(said) == 1, f"said {said}"
    assert "Sunbird" in said[0] and "stayed in hand" in said[0], said[0]


def test_giving_up_on_the_runner_up_says_which_way_it_gave_up(qapp):
    """Four exits, all of them `return False`, none of them a word in
    the log. The one failed cast in rev bb8f2b3c had `alternatives:
    ["Lightning Bats", "Thunder Snake"]` recorded on the round and no
    way to tell whether either was reached."""
    import asyncio

    from deimos_bridge.live_backend import WizAiCombatHandler
    from deimos_bridge.policies import Candidate

    said = []

    class _Backend:
        school = "storm"
        cards = {}

        def report_recovery_failed(self, why):
            said.append(why)

    class _NoOthers:
        card_name = "Pixie"
        target_index = None
        candidates = [
            Candidate(card="Pixie", target=None, turns=5, damage=0, pips=2),
            Candidate(card="pass", target=None, turns=9, damage=0, pips=0),
        ]

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    handler.backend = _Backend()
    assert not asyncio.run(
        handler._try_the_next_best(_healer_read(), _NoOthers()))
    assert said and "nothing else in hand" in said[0], said

    class _Two:
        card_name = "Pixie"
        target_index = None
        candidates = [
            Candidate(card="Pixie", target=None, turns=5, damage=0, pips=2),
            Candidate(card="Thunder Snake", target=0, turns=6, damage=190,
                      pips=1),
        ]

    said.clear()
    handler._pick_card = lambda read, name: None      # no longer in hand
    assert not asyncio.run(
        handler._try_the_next_best(_healer_read(), _Two()))
    assert said and "no longer in hand" in said[0], said
    assert "Thunder Snake" in said[0], said


def test_a_round_that_could_not_recover_says_why_on_the_row(qapp):
    """The round stays the pass `note_failed_cast` recorded -- that part
    is right -- but the row read "the cast of Pixie did not go through
    ... passed instead" with two castable cards in the same hand and
    nothing about whether either was reached."""
    from deimos_bridge.telemetry import Telemetry

    tel = Telemetry()
    tel.start_fight()

    class _Decision:
        card_name = "Pixie"
        target_index = None
        target_kind = "self"
        passing = False
        reason = "policy choice"
        policy = "ttk"
        candidates = ()

    rec = tel.observe(_Decision(), _healer_read(my_hp=526, my_max=897))
    tel.note_failed_cast("Pixie did not go through — passed instead")
    tel.note_recovery_failed("the runner-up Thunder Snake stayed in hand too")

    assert rec.passing is True, "the round really was a pass; keep it one"
    assert tel.fights[-1].passes == 1
    assert "did not go through" in rec.reason
    assert "Thunder Snake stayed in hand" in rec.reason
    assert rec.clean is False


def test_a_recovered_round_stops_claiming_it_passed(qapp):
    """`note_failed_cast` ran first and was right at the time. A round
    that then played its second choice is not a pass, and differencing
    the next board against one would fold the real cast's damage into
    the round after it."""
    from deimos_bridge.telemetry import Telemetry

    tel = Telemetry()
    tel.start_fight()

    class _Decision:
        card_name = "Pixie"
        target_index = None
        target_kind = "self"
        passing = False
        reason = "policy choice"
        policy = "ttk"
        candidates = ()

    rec = tel.observe(_Decision(), _healer_read(my_hp=526, my_max=897))
    rec.predicted_damage = 400.0

    tel.note_failed_cast("Pixie did not go through")
    assert rec.passing is True
    assert tel.fights[-1].passes == 1

    tel.note_recovered_cast("Sunbird", 0, "Pixie")
    assert rec.passing is False
    assert rec.chosen == "Sunbird" and rec.target_index == 0
    assert tel.fights[-1].passes == 0
    assert not any("nothing was played" in c for c in rec.confounds)
    # The prediction stays dropped: it was priced for Pixie, and scoring
    # Sunbird's damage against it would charge the model for a number it
    # never produced.
    assert rec.predicted_damage is None
    assert rec not in [o for o in tel.damage_observations()]


def _party_client(in_fight=False, pos=(0, 0, 0), zone_name="Zafaria"):
    class _XYZ:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class _Body:
        def __init__(self, p):
            self._p = _XYZ(*p)

        async def position(self):
            return self._p

    class _C:
        def __init__(self):
            self.body = _Body(pos)
            self.tped = []

        async def in_battle(self):
            return in_fight

        async def zone_name(self):
            return zone_name

        async def teleport(self, xyz):
            self.tped.append((xyz.x, xyz.y, xyz.z))

    return _C()


def test_a_follower_lands_on_the_leader_before_stepping_into_the_duel():
    """`join_the_fight` steps in by teleporting to the CLOSEST mob, and
    closest is measured from wherever the follower is standing. A
    follower inside the radius but beside a different group walked into
    that group's fight instead, and the party spent the duel in two
    duels."""
    import asyncio

    from deimos_bridge import party

    follower = _party_client(pos=(100, 0, 0))     # well inside the radius
    leader = _party_client(in_fight=True, pos=(200, 0, 0))
    stepped = []

    real = party.join_the_fight
    party.join_the_fight = lambda f: _joined(stepped)
    try:
        moved, why = asyncio.run(party.follow(follower, leader))
    finally:
        party.join_the_fight = real

    assert follower.tped == [(200, 0, 0)], \
        "the follower stepped in from wherever it happened to be standing"
    assert stepped == [1]
    assert moved and "joined" in why


async def _joined(seen):
    seen.append(1)
    return True, ""


def test_a_follower_still_does_not_teleport_for_nothing():
    """A party standing together with nobody fighting must not
    re-teleport every tick."""
    import asyncio

    from deimos_bridge import party

    follower = _party_client(pos=(100, 0, 0))
    leader = _party_client(pos=(200, 0, 0))
    moved, why = asyncio.run(party.follow(follower, leader))
    assert follower.tped == [] and not moved and why == ""


def test_wisps_wait_when_the_leader_is_already_fighting(qapp):
    """The chores set `in_upkeep`, and the service task skips its whole
    tick while that is set — so a follower that started a wisp sweep as
    the leader walked into a duel could not follow for up to two
    minutes. The leader fought it alone and the party planned a
    coordinated round for one wizard."""
    import asyncio

    from deimos_bridge.gui.live import LiveWorker, SeatConfig

    said = []
    w = LiveWorker(Telemetry(), "ice", [], "school-aware", 1,
                   seats=[SeatConfig(school="fire", deck=[])],
                   follow_leader=True, collect_wisps=True)
    w.status = type("S", (), {"emit": staticmethod(said.append)})()
    w.seats[0].client = _party_client(in_fight=True)
    w.seats[1].client = _party_client()

    # seat 1 follows, and the leader is mid-duel
    assert asyncio.run(w._chores_can_wait(w.seats[1])) is False
    assert any("skipping the wisps" in m for m in said), said

    # the leader is free again -> the chores are fine
    w.seats[0].client = _party_client(in_fight=False)
    assert asyncio.run(w._chores_can_wait(w.seats[1])) is True

    # and the leader itself never skips them for its own sake
    assert asyncio.run(w._chores_can_wait(w.seats[0])) is True


def test_a_card_whose_ops_forgot_the_target_is_not_aimed_at_a_mob(qapp):
    """349 of the 8,148 cards in the table carry no `tgt` on any op —
    almost all auras and globals, including plain Amplify. They fell
    through to the enemy branch and were clicked on a mob, and Wizard101
    does not cast a self-buff at an enemy: the second click deselects
    the card, the round passes with nothing played, and the duel sits
    there until the round timer runs out."""
    import asyncio

    from data_full import load_spells_full
    from deimos_bridge.live_backend import WizAiCombatHandler
    from deimos_bridge.policies import primary_target

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    read = _healer_read(my_hp=500)
    amplify = load_spells_full()["Amplify"]
    assert primary_target(amplify) is None, "the card table changed"

    got = asyncio.run(handler._resolve_target(read, 0, amplify))
    assert got is None, f"an aura was aimed at {got!r}"
    assert got not in read.enemy_members


def test_every_card_in_the_table_resolves_to_somewhere_sensible(qapp):
    """No card should reach the enemy branch by accident. Whatever a
    card's ops leave out, its `kind` answers."""
    from data_full import load_spells_full
    from deimos_bridge.live_backend import _target_from_kind
    from deimos_bridge.policies import primary_target

    unknown = [c.name for c in load_spells_full().values()
               if primary_target(c) is None and _target_from_kind(c) is None]
    assert not unknown, f"{len(unknown)} cards would be clicked on a mob " \
                        f"by default, e.g. {sorted(unknown)[:5]}"


def test_a_self_cast_is_not_labelled_with_a_mobs_name():
    """`_split` gives a bare action target 0, so every move that did not
    aim at an enemy was labelled `enemies[0]`. Live that read
    "Pixie @Sokkwi Ripper" on the party plan, which is exactly what a
    heal aimed at a monster looks like from the outside — and it fed
    `_party_expected`, which sums a round's expected damage BY TARGET
    NAME, so a heal was credited to a mob."""
    from deimos_bridge.hivemind import _aims_at_an_enemy

    class _Card:
        def __init__(self, ops):
            self.ops = ops

    heal = _Card([{"op": "heal", "tgt": "self"}])
    blade = _Card([{"op": "charm", "tgt": "self"}])
    nuke = _Card([{"op": "hit", "tgt": "enemy"}])
    aoe = _Card([{"op": "hit", "tgt": "enemies"}])

    assert _aims_at_an_enemy(nuke) is True
    for c in (heal, blade, aoe):
        assert _aims_at_an_enemy(c) is False
    assert _aims_at_an_enemy(None) is False


def test_a_cast_that_does_not_take_is_noticed_rather_than_waited_out():
    """`CombatCard.cast` clicks and returns; it raises nothing when the
    click does not take. Wizard101 deselects a card clicked at something
    it cannot be cast on, so the card goes back to the hand, the round
    passes with nothing played, and the duel sits until the round timer
    expires — recorded, meanwhile, as a cast that was made."""
    import asyncio

    from deimos_bridge.live_backend import WizAiCombatHandler

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    handler.CAST_SETTLE = 0.3
    handler.CAST_POLL = 0.05

    hand = [1, 2, 3]
    handler.get_cards = lambda: _value(hand)

    # the card never left the hand -> the cast did not go through
    assert asyncio.run(handler._card_left_the_hand(3)) is False

    # it did -> fine
    hand = [1, 2]
    assert asyncio.run(handler._card_left_the_hand(3)) is True

    # a hand that will not read is not evidence of failure
    async def _boom():
        raise RuntimeError("memory read failed")

    handler.get_cards = _boom
    assert asyncio.run(handler._card_left_the_hand(3)) is True
    assert asyncio.run(handler._card_left_the_hand(None)) is True


async def _value(v):
    return v


def test_a_dead_mob_is_the_one_the_order_says_died():
    """Greedy nearest-health gets an ordinary board wrong. Three Sand
    Stalkers at [211, 435, 201]; the 211 is hit for a predicted 294 and
    dies, so the board reads [435, 201]. Nearest-health pairs the 211
    with the 201 — a drop of 10 — and declares the SURVIVOR dead, so the
    round records 10 damage against a prediction of 294 and calls it
    clean. Order-preserving alignment has one legal reading."""
    from deimos_bridge.telemetry import match_enemies, EnemyView as E

    def mob(hp):
        return E("Sand Stalker", hp, 435.0)

    before = [mob(211), mob(435), mob(201)]
    after = [mob(435), mob(201)]
    m = match_enemies(before, after)

    assert m[0] is None, "the mob that was hit and died was reported alive"
    assert m[1] is after[0] and m[2] is after[1]


def test_the_alignment_keeps_every_case_it_already_had():
    """The rules it replaced were each written for a real board."""
    from deimos_bridge.telemetry import match_enemies, EnemyView as E

    def board(hps, mx=395.0):
        return [E("mob", h, mx) for h in hps]

    def delta(before, after, i):
        got = match_enemies(before, after).get(i)
        return None if got is None else before[i].hp - got.hp

    # the twin that was stealing the measurement
    assert delta(board([395, 259]), board([376, 259]), 0) == 19
    assert delta(board([395, 395, 395]), board([171, 395, 395]), 0) == 224
    # one of two died — health says which
    assert delta(board([19, 259]), board([259]), 0) is None
    assert delta(board([19, 259]), board([259]), 1) == 0
    # a mob joined: newcomers arrive unhurt, so the 159 was already there
    assert delta(board([395]), board([159, 395]), 0) == 236


def test_an_impossible_arrival_is_not_preferred_to_a_death():
    """A mob does not appear from nowhere already hurt. Given the choice
    between reading a damaged after-entry as a newcomer and reading a
    before-entry as dead, the death is the one that happens."""
    from deimos_bridge.telemetry import match_enemies, EnemyView as E

    def mob(hp):
        return E("mob", hp, 400.0)

    # the 400 died; the 120 is the 300 having taken 180
    m = match_enemies([mob(400), mob(300)], [mob(120)])
    assert m[0] is None
    assert m[1] is not None and m[1].hp == 120


def test_the_damage_rate_counts_every_round_not_just_the_measured_ones():
    """A teammate's rollout needs damage per round *of fight*, and only
    about half of rounds yield a measurement -- the ones where something
    landed. Averaging the measurable half of the day's two-wizard run
    gives 99 and 151 a round; over every round it gives 51 and 57, and
    51 and 57 are what the boards actually lost."""
    from deimos_bridge.telemetry import Telemetry

    tel = Telemetry()
    assert tel.damage_rate() == 0.0, "nothing fought yet is not a rate"

    tel.start_fight()
    tel.fights[-1].rounds = 11
    tel.fights[-1].damage_dealt = 468.0
    tel.start_fight()
    tel.fights[-1].rounds = 11
    tel.fights[-1].damage_dealt = 789.0
    tel.start_fight()
    tel.fights[-1].rounds = 8
    tel.fights[-1].damage_dealt = 328.0
    tel.start_fight()
    tel.fights[-1].rounds = 7
    tel.fights[-1].damage_dealt = 293.0

    # Jeffrey's run: 1878 damage over 37 rounds of fight.
    assert tel.damage_rate() == pytest.approx(1878 / 37, abs=0.05)


def test_a_fight_that_dealt_nothing_still_counts_as_rounds():
    """The round that drags the rate down is the round the rate exists to
    describe. Konstantin's fight 1 was five rounds and zero damage, and a
    rate that skipped it would tell the other wizard's rollout that the
    board dies faster than it does."""
    from deimos_bridge.telemetry import Telemetry

    tel = Telemetry()
    tel.start_fight()
    tel.fights[-1].rounds = 5
    tel.fights[-1].damage_dealt = 0.0
    tel.start_fight()
    tel.fights[-1].rounds = 5
    tel.fights[-1].damage_dealt = 313.0
    assert tel.damage_rate() == pytest.approx(31.3)


def test_the_press_x_wait_ends_when_the_box_appears():
    """A flat `sleep(0.6)` after every press-X is 0.6s of standing still
    whether the box took 30ms or never came. The game answers in its own
    time and the answer is readable, which is the same argument
    `_dialogue_moved` already makes on the other side of the
    conversation."""
    import asyncio

    from deimos_bridge import questing

    calls = {"n": 0}

    class _Client:
        pass

    async def in_dialogue(client):
        calls["n"] += 1
        return calls["n"] >= 3          # the box shows up on the 3rd look

    orig = questing.in_dialogue
    questing.in_dialogue = in_dialogue
    try:
        opened = asyncio.run(questing.dialogue_opened(_Client(), settle=0.6,
                                                      poll=0.001))
    finally:
        questing.in_dialogue = orig
    assert opened is True
    assert calls["n"] == 3              # it stopped looking, not slept on

    # and a box that never comes falls through to the full settle, which
    # is exactly the old behaviour
    calls["n"] = 0

    async def never(client):
        calls["n"] += 1
        return False

    questing.in_dialogue = never
    try:
        assert asyncio.run(questing.dialogue_opened(_Client(), settle=0.05,
                                                    poll=0.01)) is False
    finally:
        questing.in_dialogue = orig


def test_being_too_far_from_the_marker_says_how_far():
    """"not at the quest marker" was the same sentence for a wizard
    standing on the NPC's toes with an odd marker read and one across
    the zone -- and the first of those is auto-dialogue looking dead
    while working exactly as written."""
    import asyncio

    from deimos_bridge import questing

    class _P:
        def __init__(self, x, y, z=0.0):
            self.x, self.y, self.z = x, y, z

    class _Body:
        async def position(self):
            return _P(4000.0, 0.0)

    class _Client:
        body = _Body()

    async def quest_position(client):
        return _P(0.0, 0.0), ""

    orig = questing.read_quest_position
    questing.read_quest_position = quest_position
    try:
        near, why = asyncio.run(questing.at_quest_marker(_Client()))
    finally:
        questing.read_quest_position = orig
    assert near is False
    assert "4,000" in why and "750" in why


# ------------------------------------------ the wizard left behind
#
# The failure the operator is losing days to: "occasionally one wizard
# might get through with a teleport, but the others might stop
# teleporting or get stuck ... not a fully automatic bot until it can run
# for days without needing to fix teleporting and questing every 5-15
# minutes."
#
# Nothing could see it. `_check_in_step` compares QUEST GOALS and a
# wizard whose teleport silently failed is on the same quest -- same
# goal, same instruction pointer, standing in the last zone.
# `_check_progress` does see it, after five minutes, and only says so.
# So this compares WHERE they are.

def _zoned_party(zones):
    """A worker whose seats read the given zone names."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from deimos_bridge import party
    from deimos_bridge.gui.live import LiveWorker
    from deimos_bridge.telemetry import Telemetry

    from deimos_bridge.gui.live import SeatConfig

    extra = [SeatConfig(school="fire", deck=[], policy_name="ttk-lookahead",
                        telemetry=Telemetry())
             for _ in zones[1:]]
    worker = LiveWorker(Telemetry(), "fire", [], "ttk-lookahead", 1,
                        seats=extra)
    assert len(worker.seats) == len(zones)
    for i, (seat, zone) in enumerate(zip(worker.seats, zones)):
        seat.client = object()
        seat.name = f"w{i}"
        seat.wizard_name = f"w{i}"
        if seat.tel is None:
            seat.tel = Telemetry()
        seat._zone = zone

    async def read_zone(client):
        for s in worker.seats:
            if s.client is client:
                return s._zone
        return None

    return worker, read_zone


def _look(worker, read_zone, monkeypatch):
    import asyncio

    from deimos_bridge import party

    monkeypatch.setattr(party, "zone", read_zone)
    return asyncio.run(worker._check_together())


def test_the_wizard_in_the_other_zone_is_the_one_that_missed_the_teleport(
        monkeypatch):
    """Two in Olde Town and one in Unicorn Way: the one on its own is
    the one whose teleport did not land, and the majority says where it
    should be."""
    worker, read_zone = _zoned_party(
        ["Olde Town", "Olde Town", "Unicorn Way"])
    # first look only starts the clock -- a zone change is not
    # simultaneous and a bare inequality fires on every door
    assert _look(worker, read_zone, monkeypatch) == (None, None)
    worker.seats[2].stranded_since -= worker.STRANDED_AFTER + 1
    seat, target = _look(worker, read_zone, monkeypatch)
    assert seat is worker.seats[2]
    assert target.zone_seen == "Olde Town"


def test_a_party_mid_transition_is_not_a_straggler(monkeypatch):
    """Half in each zone is a zone change in progress, which is most of
    every zone change. Following one of them could be the wrong way."""
    worker, read_zone = _zoned_party(
        ["Olde Town", "Unicorn Way", "Unicorn Way", "Olde Town"])
    assert _look(worker, read_zone, monkeypatch) == (None, None)


def test_two_adrift_is_a_split_not_a_straggler(monkeypatch):
    worker, read_zone = _zoned_party(
        ["Olde Town", "Olde Town", "Unicorn Way", "The Commons"])
    assert _look(worker, read_zone, monkeypatch) == (None, None)


def test_a_party_that_is_together_is_left_alone(monkeypatch):
    worker, read_zone = _zoned_party(["Olde Town"] * 3)
    assert _look(worker, read_zone, monkeypatch) == (None, None)


def test_a_zone_that_will_not_read_is_no_evidence(monkeypatch):
    """None rather than a guess. Deciding a wizard is adrift because a
    read failed would teleport a wizard that was exactly where it should
    be, which is worse than the failure being fixed."""
    worker, read_zone = _zoned_party(["Olde Town", "Olde Town", None])
    assert _look(worker, read_zone, monkeypatch) == (None, None)


def test_moving_zone_restarts_the_clock(monkeypatch):
    """A wizard walking through two zones on its own is not adrift for
    the sum of both -- it is adrift in whichever one it stops in."""
    worker, read_zone = _zoned_party(
        ["Olde Town", "Olde Town", "Unicorn Way"])
    _look(worker, read_zone, monkeypatch)
    worker.seats[2].stranded_since -= worker.STRANDED_AFTER + 1
    worker.seats[2]._zone = "The Commons"
    assert _look(worker, read_zone, monkeypatch) == (None, None)
    assert worker.seats[2].stranded_where == "The Commons"


def test_a_solo_wizard_is_never_left_behind(monkeypatch):
    worker, read_zone = _zoned_party(["Olde Town"])
    assert _look(worker, read_zone, monkeypatch) == (None, None)


def test_bringing_it_back_uses_the_ordinary_follow(monkeypatch):
    """`party.follow` and nothing new: it already handles the cross-zone
    hop, the distance close and stepping into the duel if the others are
    already fighting."""
    import asyncio

    from deimos_bridge import party

    worker, _read = _zoned_party(["Olde Town", "Olde Town", "Unicorn Way"])
    seat, target = worker.seats[2], worker.seats[0]
    seat.stranded_where = "Unicorn Way"
    target.zone_seen = "Olde Town"
    called = {}

    async def follow(follower, leader, leader_name=None, **kw):
        called["to"] = leader_name
        return True, "followed the leader into Olde Town"

    monkeypatch.setattr(party, "follow", follow)
    asyncio.run(worker._rejoin(seat, target))
    assert called["to"] == "w0"
    assert seat.stranded_since is None            # cleared once it moved
    kinds = [e["kind"] for e in seat.tel.questing]
    assert kinds == ["stranded", "rejoined"]


def test_a_rejoin_that_fails_is_recorded_rather_than_retried_silently(
        monkeypatch):
    import asyncio

    from deimos_bridge import party

    worker, _read = _zoned_party(["Olde Town", "Olde Town", "Unicorn Way"])
    seat, target = worker.seats[2], worker.seats[0]
    seat.stranded_where = "Unicorn Way"

    async def follow(follower, leader, leader_name=None, **kw):
        return False, "could not read the leader's position"

    monkeypatch.setattr(party, "follow", follow)
    seat.stranded_since = 12345.0                  # adrift, clock running
    asyncio.run(worker._rejoin(seat, target))
    assert [e["kind"] for e in seat.tel.questing] == ["stranded",
                                                      "rejoin-failed"]
    # still adrift: the clock is NOT cleared, so the next look still
    # sees it and the throttle decides when to try again
    assert seat.stranded_since == 12345.0


def test_rejoining_is_throttled_so_it_does_not_fight_the_script(monkeypatch):
    import asyncio

    from deimos_bridge import party

    worker, _read = _zoned_party(["Olde Town", "Olde Town", "Unicorn Way"])
    seat, target = worker.seats[2], worker.seats[0]
    tries = {"n": 0}

    async def follow(follower, leader, leader_name=None, **kw):
        tries["n"] += 1
        return False, "nope"

    monkeypatch.setattr(party, "follow", follow)
    asyncio.run(worker._rejoin(seat, target))
    asyncio.run(worker._rejoin(seat, target))
    assert tries["n"] == 1, "the second attempt should be inside the throttle"


def test_the_export_carries_what_happened_between_the_fights():
    """A run can be perfect in every duel and still need a human every
    ten minutes. Three wizards at rev 3c8b8087 won all seventeen of
    their fights and the exports said nothing whatever about the thing
    the operator was actually fighting with."""
    from deimos_bridge.telemetry import Telemetry

    tel = Telemetry()
    tel.note_questing("stranded", "w2 is in Unicorn Way")
    tel.note_questing("rejoined", "followed into Olde Town")
    tel.note_questing("stranded", "again")
    assert tel.questing_counts() == {"stranded": 2, "rejoined": 1}
    assert tel.summary()["questing"] == {"stranded": 2, "rejoined": 1}
    assert [e["kind"] for e in tel.questing][:2] == ["stranded", "rejoined"]
    assert all("at" in e and "fight" in e for e in tel.questing)


def test_the_questing_log_does_not_grow_without_bound():
    """Days of running would otherwise put a megabyte of "teleported"
    in every export."""
    from deimos_bridge.telemetry import Telemetry

    tel = Telemetry()
    for i in range(tel.QUESTING_LOG + 50):
        tel.note_questing("tick", str(i))
    assert len(tel.questing) == tel.QUESTING_LOG
    assert tel.questing[-1]["detail"] == str(tel.QUESTING_LOG + 49)


# ---------------------------------------------- ahead is not the same as behind
def test_a_wizard_is_not_dragged_back_to_a_zone_it_just_left(monkeypatch):
    """The run at rev 228d4f50, end to end. Sebastian went into
    WC_Firecat_T1 while the other two were in WC_Firecat, was pulled
    back out, walked in again, was pulled out again -- five times in
    four minutes, each pull throwing away the step he had just
    finished. The majority is not the same thing as the right answer."""
    worker, read_zone = _zoned_party(
        ["WC_Firecat", "WC_Firecat", "WC_Firecat"])
    _look(worker, read_zone, monkeypatch)          # everyone together

    worker.seats[2]._zone = "WC_Firecat_T1"        # he goes in
    assert _look(worker, read_zone, monkeypatch) == (None, None)
    assert worker.seats[2].zone_left[-1][0] == "WC_Firecat"

    # ...and now he is the odd one out, adrift for long enough to be
    # fetched. He must not be, because he left WC_Firecat on purpose.
    worker.seats[2].stranded_since = 0.0
    assert _look(worker, read_zone, monkeypatch) == (None, None)


def test_the_wizard_that_never_got_there_is_still_fetched(monkeypatch):
    """The case the mechanism is for, and the one the fix must not cost:
    Konstantin at t=550 was in WC_Firecat while the party had moved into
    WC_Firecat_T1, and he had never been in T1 at all."""
    worker, read_zone = _zoned_party(
        ["WC_Firecat", "WC_Firecat", "WC_Firecat"])
    _look(worker, read_zone, monkeypatch)
    for seat in worker.seats[:2]:
        seat._zone = "WC_Firecat_T1"               # the party teleports
    _look(worker, read_zone, monkeypatch)
    worker.seats[2].stranded_since -= worker.STRANDED_AFTER + 1
    seat, target = _look(worker, read_zone, monkeypatch)
    assert seat is worker.seats[2]
    assert target.zone_seen == "WC_Firecat_T1"


def test_leaving_a_zone_long_ago_does_not_protect_a_wizard_forever(
        monkeypatch):
    """A wizard that passed through a zone half an hour ago and is now
    genuinely stuck is stuck, not making progress."""
    worker, read_zone = _zoned_party(["Olde Town", "Olde Town", "Olde Town"])
    _look(worker, read_zone, monkeypatch)
    worker.seats[2]._zone = "Unicorn Way"
    _look(worker, read_zone, monkeypatch)
    stale = worker.LEFT_ON_PURPOSE + 60.0
    worker.seats[2].zone_left = [(z, at - stale)
                                 for z, at in worker.seats[2].zone_left]
    _look(worker, read_zone, monkeypatch)          # the clock starts
    worker.seats[2].stranded_since -= worker.STRANDED_AFTER + 1
    seat, _target = _look(worker, read_zone, monkeypatch)
    assert seat is worker.seats[2]


def test_three_pulls_into_one_zone_is_a_loop_and_stops(monkeypatch):
    """Belt and braces. Even if the reasoning above is wrong about a
    board it has not seen, a rescue that keeps repeating is a loop, and
    a loop must not be able to run for days."""
    import time

    worker, read_zone = _zoned_party(["Olde Town", "Olde Town", "Olde Town"])
    _look(worker, read_zone, monkeypatch)
    worker.seats[2]._zone = "Unicorn Way"
    _look(worker, read_zone, monkeypatch)
    worker.seats[2].zone_left = []                 # not the other guard
    _look(worker, read_zone, monkeypatch)          # the clock starts
    worker.seats[2].stranded_since -= worker.STRANDED_AFTER + 1
    seat, _t = _look(worker, read_zone, monkeypatch)
    assert seat is worker.seats[2], "the fixture must reach a real rescue"

    worker.seats[2].rejoin_history = [("Olde Town", time.monotonic())] * 2
    assert _look(worker, read_zone, monkeypatch) == (None, None)


# ------------------------------------------ clicking through what is actually up
def _services_root(visible=True):
    """The quest-picker an NPC with several things to say puts up.

    No `btnRight` anywhere -- that is the whole point of it.
    """
    exit_button = _Win("Exit", visible=visible)
    return _Win("root", [
        _Win("WorldView", [
            _Win("NPCServicesWin", [_Win("wndDialogMain",
                                         [exit_button])])])]), exit_button


def test_a_quest_picker_is_a_dialogue_even_without_an_advance_button():
    """Reported live as "it says it detected dialogue but does not clear
    it". wizwalker's own `Client.is_in_dialog` counts `NPCServicesWin`;
    wizAi only looked for `wndDialogMain/btnRight`, so an NPC offering
    several quests read as "no dialogue open" at a wizard that could not
    move."""
    import asyncio

    from deimos_bridge.questing import in_dialogue
    root, _ = _services_root()
    assert asyncio.run(in_dialogue(_QuestClient(root)))
    root, _ = _services_root(visible=False)
    assert not asyncio.run(in_dialogue(_QuestClient(root)))


def test_the_quest_picker_is_closed_rather_than_left_blocking():
    import asyncio

    from deimos_bridge.questing import advance_dialogue
    root, exit_button = _services_root()
    client = _QuestClient(root)
    clicks, why = asyncio.run(advance_dialogue(client, settle=0))
    assert (clicks, why) == (1, "")
    assert client.mouse_handler.clicks == [exit_button]


def test_a_click_that_changes_nothing_is_not_a_window_advanced():
    """The other half of the same report, and the reason it was slow.

    `click_window` raises nothing when the click does not reach the game
    -- it simply has no effect. The outcome of `_dialogue_moved` was
    thrown away, so forty of those counted as forty cleared windows at
    half a second each: twenty seconds of "clicking through dialogue…"
    against a box that never moved, then a report of success.
    """
    import asyncio

    from deimos_bridge.questing import advance_dialogue

    class _Deaf(_Mouse):
        async def click_window(self, window):
            self.clicks.append(window)      # lands nowhere, raises nothing

    root, _ = _dialogue_root()
    client = _QuestClient(root)
    client.mouse_handler = _Deaf()
    clicks, why = asyncio.run(
        advance_dialogue(client, max_clicks=40, settle=0.05, poll=0.01))
    assert clicks == 0, "a click that moved nothing is not a window cleared"
    assert "not reaching the game" in why
    assert len(client.mouse_handler.clicks) == 2, \
        "gave up after two dead clicks rather than forty"


def test_an_unreadable_box_is_still_clicked_through():
    """A stall verdict needs the text to have been READ. A dialogue whose
    text will not read is no evidence either way, and must not be
    mistaken for a dead click -- that would abandon a working
    conversation after two pages."""
    import asyncio

    from deimos_bridge.questing import advance_dialogue

    class _Sticky(_Mouse):
        async def click_window(self, window):
            self.clicks.append(window)

    button = _Win("btnRight", visible=True)
    root = _Win("root", [_Win("WorldView", [_Win("wndDialogMain", [button])])])
    client = _QuestClient(root)
    client.mouse_handler = _Sticky()
    clicks, why = asyncio.run(
        advance_dialogue(client, max_clicks=5, settle=0.02, poll=0.01))
    assert (clicks, why) == (5, "")


# --------------------------------------------- the run must never stop silently
def test_a_stranded_wizard_in_a_duel_is_told_about_rather_than_dropped(
        monkeypatch):
    """The run at rev 85a68184: two `stranded` entries, no `rejoined`,
    no `rejoin-failed`, no third attempt. `party.follow` returns
    (False, "") when the follower is in a duel -- leaving Sebastian
    alone with Foulgaze was right, saying nothing about it was not."""
    import asyncio

    from deimos_bridge import party

    worker, _read = _zoned_party(["Olde Town", "Olde Town", "Unicorn Way"])
    seat, target = worker.seats[2], worker.seats[0]
    seat.stranded_where, target.zone_seen = "Unicorn Way", "Olde Town"

    async def nothing_to_do(*a, **k):
        return False, ""

    async def in_a_duel(client):
        return True

    # Through monkeypatch, so these are put back. Assigning to the
    # module directly leaked into every later test that follows a
    # leader, and six of them failed on the way past.
    monkeypatch.setattr(party, "follow", nothing_to_do)
    monkeypatch.setattr(party, "in_battle", in_a_duel)
    asyncio.run(worker._rejoin(seat, target))
    kinds = [e["kind"] for e in seat.tel.questing]
    assert kinds == ["stranded", "rejoin-skipped"]
    assert "still in a duel" in seat.tel.questing[-1]["detail"]


def test_waiting_to_heal_is_written_down_where_the_export_can_see_it():
    """The one place the run deliberately stops for up to
    LOW_HEALTH_WAIT a fight, and the one place that left no trace --
    so "it gets stuck" and "it is healing, as designed" read the
    same."""
    import asyncio

    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    seat = worker.seats[0]
    worker.use_potions = worker.collect_wisps = worker.buy_potions = False
    # Short, but not zero: the give-up check runs before the first
    # report, so a zero wait would skip straight past the hold the test
    # is about. In production the wait is 150s and cannot.
    worker.LOW_HEALTH_WAIT = 0.01
    worker.LOW_HEALTH_POLL = 0.05

    async def hurt(_seat):
        return 0.05

    worker._health_left = hurt
    asyncio.run(worker._let_it_heal(seat))
    kinds = [e["kind"] for e in seat.tel.questing]
    assert "waiting-to-heal" in kinds
    assert "went-in-hurt" in kinds
    assert "5%" in seat.tel.questing[0]["detail"]


def test_a_wizard_that_heals_up_says_so_too():
    import asyncio

    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    seat = worker.seats[0]
    worker.use_potions = worker.collect_wisps = worker.buy_potions = False
    worker.LOW_HEALTH_POLL = 0.01
    reads = iter([0.05, 0.9, 0.9, 0.9])

    async def recovering(_seat):
        return next(reads, 0.9)

    worker._health_left = recovering
    asyncio.run(worker._let_it_heal(seat))
    assert [e["kind"] for e in seat.tel.questing] == ["waiting-to-heal",
                                                      "healed"]


def test_a_stage_that_keeps_timing_out_keeps_saying_so_in_the_export():
    """The run at rev 85a68184 has 99 combat rounds and eight questing
    entries to account for the rest of the session. A stage that times
    out on every tick was announced on the status line once and written
    down nowhere, so "really stuck" and "working fine" exported the
    same."""
    import asyncio

    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    seat = worker.seats[0]

    async def never_returns():
        await asyncio.sleep(10)

    async def drive():
        for _ in range(worker.STUCK_EVERY + 1):
            await worker._stage(seat, "auto-dialogue", never_returns(),
                                limit=0.001)

    asyncio.run(drive())
    stuck = [e for e in seat.tel.questing if e["kind"] == "stage-timeout"]
    assert len(stuck) == 2, "the first, then one per STUCK_EVERY"
    assert "auto-dialogue" in stuck[0]["detail"]
    assert "in a row" in stuck[1]["detail"]


def test_a_stage_that_raises_says_which_and_why():
    import asyncio

    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    seat = worker.seats[0]

    async def broken():
        raise ValueError("no child window named wndCharacter")

    asyncio.run(worker._stage(seat, "follow-the-leader", broken()))
    stuck = [e for e in seat.tel.questing if e["kind"] == "stage-failed"]
    assert len(stuck) == 1
    assert "follow-the-leader" in stuck[0]["detail"]
    assert "wndCharacter" in stuck[0]["detail"]


def test_an_ordinary_say_once_still_stays_out_of_the_questing_log():
    """Only the callers that pass a `kind` are stall reports. The rest
    are advice -- "auto-dialogue only talks to quest NPCs" -- and the
    questing log is not where those belong."""
    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    seat = worker.seats[0]
    worker._say_once(seat, "quest-arrow", "switch the quest arrow on")
    assert seat.tel.questing == []


# ------------------------------------------------------------- the timeline
def _beating(zones=("Olde Town", "Olde Town")):
    worker, _read = _zoned_party(list(zones))
    seat = worker.seats[0]

    class _C:
        async def in_battle(self):
            return False

    seat.client = _C()
    seat.zone_seen = zones[0]
    return worker, seat


def test_the_heartbeat_says_where_a_wizard_is_and_what_it_is_on():
    """Everything else in the questing log is an alarm, and alarms only
    cover the moments somebody wrote code to notice. "It's really stuck"
    is about the time between them -- rev 85a68184 had 99 combat rounds
    and eight questing entries for a whole session."""
    import asyncio

    worker, seat = _beating()
    seat.goal = "Defeat Foulgaze"

    async def half(_seat):
        return 0.5

    worker._health_left = half
    asyncio.run(worker._heartbeat(seat, driven=False))
    beat = [e for e in seat.tel.questing if e["kind"] == "heartbeat"]
    assert len(beat) == 1
    for expected in ("Olde Town", "50% health", "out of combat",
                     "Defeat Foulgaze"):
        assert expected in beat[0]["detail"], beat[0]["detail"]


def test_the_heartbeat_is_throttled_to_one_a_minute():
    import asyncio

    worker, seat = _beating()

    async def unread(_seat):
        return None

    worker._health_left = unread
    asyncio.run(worker._heartbeat(seat))
    asyncio.run(worker._heartbeat(seat))
    assert len([e for e in seat.tel.questing if e["kind"] == "heartbeat"]) == 1


def test_a_heartbeat_can_never_take_the_service_tick_down():
    """The first version read `runner.steps` unguarded, and a runner
    without it raised out of the service tick and killed the loop for
    that wizard -- the instrument for diagnosing a stuck wizard became a
    cause of one."""
    import asyncio

    worker, seat = _beating()
    worker.seats[0].runner = type("R", (), {"running": True})()  # no `steps`

    async def boom(_seat):
        raise RuntimeError("the health read exploded")

    worker._health_left = boom
    asyncio.run(worker._heartbeat(seat, driven=True))   # must not raise


def test_a_defeat_is_written_down_not_just_announced():
    """The game teleports a defeated wizard to the commons, so the
    script's next instruction is for somewhere it is not. Nothing else
    here can attribute that: the quest goal is unchanged and the zone
    change looks like any other."""
    worker, seat = _beating()
    seat.zone_seen = "WizardCity/WC_Streets/WC_OldeTown"
    worker._defeated_hook(seat)()
    lost = [e for e in seat.tel.questing if e["kind"] == "defeated"]
    assert len(lost) == 1
    assert "WC_OldeTown" in lost[0]["detail"]


def test_every_zone_change_lands_in_the_log(monkeypatch):
    """Free -- the stranded check already polls every seat's zone. It
    turns the log into a route, which is what "one of them wandered off"
    needs to be checkable rather than inferred."""
    worker, read_zone = _zoned_party(["Olde Town", "Olde Town"])
    _look(worker, read_zone, monkeypatch)
    worker.seats[1]._zone = "Unicorn Way"
    _look(worker, read_zone, monkeypatch)
    moved = [e for e in worker.seats[1].tel.questing if e["kind"] == "zone"]
    assert moved == [{"at": moved[0]["at"], "fight": 0, "kind": "zone",
                      "detail": "Olde Town -> Unicorn Way"}]


# --------------------------------------------- unwedging a stuck scripted wizard
def _wedged(dialogue=True, cleared=2, monkeypatch=None):
    """A script-driven wizard that has changed nothing for a long time."""
    import time

    from deimos_bridge import questing

    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    seat = worker.seats[0]
    seat.client = object()
    seat.progress = ("WizardCity/WC_NightSide", (1, 2, 3),
                     "Talk To Mortis in Nightside")
    seat.progress_at = time.monotonic() - worker.STUCK_AFTER - 1

    async def in_dialogue(_c):
        return dialogue

    async def near(_c):
        return True

    async def at_marker(_c):
        return False, "the marker is 900 away"

    async def advance(_c, **kw):
        return cleared, ""

    monkeypatch.setattr(questing, "in_dialogue", in_dialogue)
    monkeypatch.setattr(questing, "near_interactable", near)
    monkeypatch.setattr(questing, "at_quest_marker", at_marker)
    monkeypatch.setattr(questing, "advance_dialogue", advance)
    return worker, seat


def test_a_script_wedged_on_a_dialogue_gets_the_box_cleared(monkeypatch):
    """Rev 07ef3fa7, twice: all three wizards on "Talk To Mortis in
    Nightside" for ten minutes and then "Talk To Dworgyn in Death
    School" for five, zone and position and goal all frozen while the
    script's counter climbed 600 a minute. Both steps it died on were
    `Talk To`, and wizAi will not touch a scripted wizard's dialogue."""
    import asyncio

    worker, seat = _wedged(dialogue=True, monkeypatch=monkeypatch)
    asyncio.run(worker._unstick(seat))
    kinds = [e["kind"] for e in seat.tel.questing]
    assert kinds == ["stuck-detail", "unstuck-dialogue"]
    assert "dialogue open" in seat.tel.questing[0]["detail"]
    assert "cleared 2 window(s)" in seat.tel.questing[1]["detail"]


def test_a_wedged_script_with_no_box_open_is_only_reported(monkeypatch):
    """A retry loop with nothing open is not a dialogue problem. The log
    could not tell "the box is open and the script cannot advance it"
    from "the box never opened", and those want opposite fixes."""
    import asyncio

    worker, seat = _wedged(dialogue=False, monkeypatch=monkeypatch)
    asyncio.run(worker._unstick(seat))
    kinds = [e["kind"] for e in seat.tel.questing]
    assert kinds == ["stuck-detail"]
    assert "dialogue closed" in seat.tel.questing[0]["detail"]
    assert "looping" in seat.tel.questing[0]["detail"]
    assert "the marker is 900 away" in seat.tel.questing[0]["detail"]


def test_a_script_parked_on_a_dialogue_that_never_opened_gets_an_x(
        monkeypatch):
    """`waitfordialog` is `while not await is_in_dialog(): sleep(0.25)`
    with no timeout (`command_parser.py:67`). A script parked on that
    with no box open is waiting for one that is not coming, and pressing
    X is the thing that makes its condition true -- the one case where
    an unrequested interact is exactly what the script is asking for."""
    import asyncio

    from deimos_bridge import questing

    worker, seat = _wedged(dialogue=False, monkeypatch=monkeypatch)
    worker.seats[0].runner = type("R", (), {"running": True,
                                            "steps": 11582})()
    pressed = []

    async def press_x(_c, **kw):
        pressed.append(True)
        return True, ""

    async def opened(_c, **kw):
        return True

    monkeypatch.setattr(questing, "press_x", press_x)
    monkeypatch.setattr(questing, "dialogue_opened", opened)

    asyncio.run(worker._unstick(seat))          # first look: no baseline
    assert not pressed, "parked is unknowable on the first sample"
    seat.unstuck_at = 0.0
    asyncio.run(worker._unstick(seat))          # counter has not moved
    assert pressed == [True]
    note = [e for e in seat.tel.questing if e["kind"] == "unstuck-pressed-x"]
    assert "came up" in note[0]["detail"]


def test_a_script_still_running_instructions_never_gets_an_x(monkeypatch):
    """The dangerous direction. A climbing counter means the script is
    not waiting on anything, so an interact it did not ask for is a
    quest-state change nobody requested."""
    import asyncio

    from deimos_bridge import questing

    worker, seat = _wedged(dialogue=False, monkeypatch=monkeypatch)
    runner = type("R", (), {"running": True, "steps": 11582})()
    worker.seats[0].runner = runner
    pressed = []

    async def press_x(_c, **kw):
        pressed.append(True)
        return True, ""

    monkeypatch.setattr(questing, "press_x", press_x)

    asyncio.run(worker._unstick(seat))
    runner.steps = 12403                        # the loop is running
    seat.unstuck_at = 0.0
    asyncio.run(worker._unstick(seat))
    assert not pressed


def test_a_script_that_is_making_progress_is_left_completely_alone(
        monkeypatch):
    """The whole reason wizAi keeps its hands off a scripted wizard.
    Only measured non-progress lifts that."""
    import asyncio
    import time

    worker, seat = _wedged(monkeypatch=monkeypatch)
    seat.progress_at = time.monotonic()          # moving
    asyncio.run(worker._unstick(seat))
    assert seat.tel.questing == []


def test_unwedging_is_throttled(monkeypatch):
    import asyncio

    worker, seat = _wedged(monkeypatch=monkeypatch)
    asyncio.run(worker._unstick(seat))
    asyncio.run(worker._unstick(seat))
    assert len([e for e in seat.tel.questing
                if e["kind"] == "stuck-detail"]) == 1


def test_the_no_progress_line_does_not_mangle_the_word_instructions():
    """The live log said "while 12,620structions ran": the first version
    stripped a trailing " in" with `.replace(' in', '')`, and " in" also
    occurs inside "instructions"."""
    import time

    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    seat = worker.seats[0]
    seat.progress = ("WC_NightSide", (1, 2, 3), "Talk To Mortis")
    seat.progress_at = time.monotonic() - worker.STUCK_AFTER - 1
    worker.seats[0].runner = type("R", (), {"running": True, "steps": 12620})()
    worker._script_drives = lambda _s: True
    worker._check_progress(seat)
    note = [e for e in seat.tel.questing if e["kind"] == "no-progress"][0]
    assert "12,620 instructions ran" in note["detail"], note["detail"]
    assert "structions" not in note["detail"].replace("instructions", "")


def test_the_heartbeat_beats_while_a_wizard_is_in_a_duel():
    """Konstantin's log has one heartbeat at t=0 and the next at t=531.7
    -- nine missed beats, spent in a fourteen-round duel. A wizard
    wedged inside a fight is exactly as stuck as one wedged outside."""
    import asyncio

    worker, seat = _beating()

    async def full(_seat):
        return 1.0

    worker._health_left = full
    asyncio.run(worker._heartbeat(seat, driven=True, fighting=True))
    beat = [e for e in seat.tel.questing if e["kind"] == "heartbeat"][0]
    assert "in a duel" in beat["detail"]


def test_a_wizard_on_zero_health_is_not_described_as_fighting():
    """Rev bb8f2b3c: four consecutive minutes of `0% health · in a duel`
    after the `defeated` entry, with the script's instruction counter
    frozen the whole time. That is a corpse waiting for its allies to
    finish the fight -- the one state where a still script is correct --
    and the log made it the most alarming thing on the page."""
    import asyncio

    worker, seat = _beating()

    async def dead(_seat):
        return 0.0

    worker._health_left = dead
    asyncio.run(worker._heartbeat(seat, driven=True, fighting=True))
    beat = [e for e in seat.tel.questing if e["kind"] == "heartbeat"][0]
    assert "defeated, waiting for the duel to end" in beat["detail"]
    assert "· in a duel" not in beat["detail"], beat["detail"]


def test_a_waitfor_that_gives_up_is_reported_to_every_seat(monkeypatch):
    """A bounded wait that gives up silently is only half a fix: the
    script falls through into a retry loop and the run looks normal
    again, so the thing that went wrong has to be said."""
    from deimos_bridge import scripts

    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    worker.seats[0].runner = type("R", (), {"running": True, "steps": 1})()
    installed = []
    monkeypatch.setattr(scripts, "on_waitfor_timeout",
                        lambda hook: (installed.append(hook), (True, ""))[1])
    monkeypatch.setattr(scripts, "on_teleport_result",
                        lambda hook: (True, ""))
    monkeypatch.setattr(scripts, "on_teleport_note", lambda hook: (True, ""))
    monkeypatch.setattr(scripts, "on_party_task_failed",
                        lambda hook: (True, ""))

    worker._watch_waitfor(worker.seats[0])
    assert len(installed) == 1
    worker._watch_waitfor(worker.seats[0])
    assert len(installed) == 1, "the VM hook is module-level, install it once"

    installed[0]("dialog", 90.0, False)
    for seat in worker.seats:
        note = [e for e in seat.tel.questing
                if e["kind"] == "waitfor-gave-up"]
        assert len(note) == 1
        assert "`waitfor dialog` polled for 90s" in note[0]["detail"]


def test_a_deimos_without_the_hook_is_reported_not_ignored(monkeypatch):
    """A Deimos update that renames the hook must not stop wizAi
    starting, but it must not pass unnoticed either."""
    from deimos_bridge import scripts

    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    worker.seats[0].runner = type("R", (), {"running": True, "steps": 1})()
    monkeypatch.setattr(scripts, "on_waitfor_timeout",
                        lambda hook: (False, "this Deimos has no hook"))
    monkeypatch.setattr(scripts, "on_teleport_result",
                        lambda hook: (True, ""))
    monkeypatch.setattr(scripts, "on_teleport_note", lambda hook: (True, ""))
    monkeypatch.setattr(scripts, "on_party_task_failed",
                        lambda hook: (True, ""))

    worker._watch_waitfor(worker.seats[0])           # must not raise
    note = [e for e in worker.seats[0].tel.questing
            if e["kind"] == "stage-failed"]
    assert note and "no hook" in note[0]["detail"]


def test_a_teleport_that_did_not_land_is_reported_to_every_seat(monkeypatch):
    """`navmap_tp` returns whether it landed now. A teleport that
    silently did not is the failure that started all of this -- "one
    wizard might get through, but the others get stuck" -- and the
    script's next instruction is for wherever it meant to be."""
    from deimos_bridge import scripts

    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    worker.seats[0].runner = type("R", (), {"running": True, "steps": 1})()
    got = []
    monkeypatch.setattr(scripts, "on_waitfor_timeout", lambda h: (True, ""))
    monkeypatch.setattr(scripts, "on_teleport_result",
                        lambda hook: (got.append(hook), (True, ""))[1])
    monkeypatch.setattr(scripts, "on_teleport_note", lambda hook: (True, ""))
    monkeypatch.setattr(scripts, "on_party_task_failed",
                        lambda hook: (True, ""))

    worker._watch_waitfor(worker.seats[0])
    got[0](True, "direct", "Olde Town")             # a teleport that worked
    assert all(not s.tel.questing for s in worker.seats), \
        "a run makes thousands of these; only the failures are worth a line"

    got[0](False, "the client was not free to teleport", "Olde Town")
    for seat in worker.seats:
        note = [e for e in seat.tel.questing
                if e["kind"] == "teleport-failed"]
        assert len(note) == 1
        assert "not free" in note[0]["detail"]
        assert "Olde Town" in note[0]["detail"]


def test_a_failed_teleport_names_the_wizard_it_happened_to(monkeypatch):
    """The hook is module-level, so it fires for whichever wizard the
    script was driving. Rev bb8f2b3c has nineteen `teleport-failed`
    entries and not one of them says whose teleport it was."""
    from deimos_bridge import scripts

    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    worker.seats[0].runner = type("R", (), {"running": True, "steps": 1})()
    worker.seats[0].name = "Phönix"
    worker.seats[1].name = "Sebastian"
    got = []
    monkeypatch.setattr(scripts, "on_waitfor_timeout", lambda h: (True, ""))
    monkeypatch.setattr(scripts, "on_teleport_result",
                        lambda hook: (got.append(hook), (True, ""))[1])
    monkeypatch.setattr(scripts, "on_teleport_note", lambda hook: (True, ""))
    monkeypatch.setattr(scripts, "on_party_task_failed",
                        lambda hook: (True, ""))

    worker._watch_waitfor(worker.seats[0])
    got[0](False, "a dialogue box was open after 8s", "Olde Town",
           worker.seats[1].client)
    for seat in worker.seats:
        note = [e for e in seat.tel.questing
                if e["kind"] == "teleport-failed"]
        assert len(note) == 1
        assert note[0]["detail"].startswith("Sebastian's scripted teleport"), \
            note[0]["detail"]

    # An unrecognised client still gets a line -- naming nobody beats
    # saying nothing.
    for seat in worker.seats:
        seat.tel.questing.clear()
    got[0](False, "the wizard was in a duel", "", object())
    assert worker.seats[0].tel.questing[0]["detail"].startswith(
        "a scripted teleport did not land")


def test_a_teleport_that_pushed_through_a_stale_load_is_visible(monkeypatch):
    """`navmap_tp` deciding the client's loading flag is stale and going
    ahead anyway is a judgement call, and one a human watching a run
    should be able to see us make."""
    from deimos_bridge import scripts

    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    worker.seats[0].runner = type("R", (), {"running": True, "steps": 1})()
    worker.seats[0].name = "Phönix"
    got = []
    monkeypatch.setattr(scripts, "on_waitfor_timeout", lambda h: (True, ""))
    monkeypatch.setattr(scripts, "on_teleport_result", lambda h: (True, ""))
    monkeypatch.setattr(scripts, "on_teleport_note",
                        lambda hook: (got.append(hook), (True, ""))[1])

    worker._watch_waitfor(worker.seats[0])
    got[0]("the book (PageFlip) was open — teleporting anyway",
           worker.seats[0].client)
    for seat in worker.seats:
        note = [e for e in seat.tel.questing
                if e["kind"] == "teleport-forced"]
        assert len(note) == 1
        assert note[0]["detail"].startswith("Phönix: "), note[0]["detail"]
        assert "PageFlip" in note[0]["detail"]


def test_a_wizard_that_fell_out_of_a_mass_instruction_is_named(monkeypatch):
    """Upstream a mass instruction runs inside `asyncio.TaskGroup`, which
    cancels every sibling when one task raises -- so one wizard's
    transient failure took the other three down with it, and nothing
    said so. `PartyTaskGroup` lets them all finish; this is where the
    one that did not surfaces."""
    from deimos_bridge import scripts

    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    worker.seats[0].runner = type("R", (), {"running": True, "steps": 1})()
    got = []
    monkeypatch.setattr(scripts, "on_waitfor_timeout", lambda h: (True, ""))
    monkeypatch.setattr(scripts, "on_teleport_result", lambda h: (True, ""))
    monkeypatch.setattr(scripts, "on_teleport_note", lambda h: (True, ""))
    monkeypatch.setattr(scripts, "on_party_task_failed",
                        lambda hook: (got.append(hook), (True, ""))[1])

    worker._watch_waitfor(worker.seats[0])
    got[0]("teleport", [("tp_to_quest", RuntimeError("MemoryReadError"))])
    for seat in worker.seats:
        note = [e for e in seat.tel.questing
                if e["kind"] == "party-task-failed"]
        assert len(note) == 1
        assert "`teleport` failed for one wizard" in note[0]["detail"]
        assert "MemoryReadError" in note[0]["detail"]
        assert "the others finished it" in note[0]["detail"]


def test_a_stall_that_keeps_going_is_said_more_than_once():
    """Keyed on the situation alone this fired exactly once in a
    forty-minute stall, at the five-minute mark, while `stuck-detail`
    said the same sentence 69 times."""
    import time

    worker, _read = _zoned_party(["Krokotopia/KT_Hub"])
    worker.script = "###deimos_expertmode"
    seat = worker.seats[0]
    seat.runner = type("R", (), {"running": True, "steps": 10_000})()
    seat.progress = ("Krokotopia/KT_Hub", (0, 0, 0), "Talk To Standish")
    worker._script_drives = lambda _s: True

    said = []
    for minutes in (6, 8, 11, 21, 41, 42):
        seat.progress_at = time.monotonic() - minutes * 60
        worker._check_progress(seat)
        said.append(len([e for e in seat.tel.questing
                         if e["kind"] == "no-progress"]))
    # 6 min says it; 8 is the same band; 11, 21 and 41 each double.
    assert said == [1, 1, 2, 3, 4, 4], said


# ------------------------------------------------------------- preset scripts
_QUESTER = (
    '###deimos_expertmode\n'
    '# @name: Arc 1 Quester\n'
    'var Main_Account = "QuestingAccountName"\n'
    'var Main_Account_School = "SchoolGoesHere"\n'
    'var Questee2 = "QuestingAccountName"\n'
    'var Questee2_School = "SchoolGoesHere"\n'
    'var Questee3 = "QuestingAccountName"\n'
    'var Questee3_School = "SchoolGoesHere"\n'
    'block Go {\n'
    '  if NOT Main_Account = "QuestingAccountName" { p1 sendkey X }\n'
    '  if NOT Main_Account_School = "SchoolGoesHere" { p1 sendkey X }\n'
    '  if NOT Questee2 = "QuestingAccountName" { p2 sendkey X }\n'
    '  if NOT Questee2_School = "SchoolGoesHere" { p2 sendkey X }\n'
    '  if NOT Questee3 = "QuestingAccountName" { p3 sendkey X }\n'
    '  if NOT Questee3_School = "SchoolGoesHere" { p3 sendkey X }\n'
    '}\n')


def test_a_preset_gets_the_partys_real_names_put_into_it():
    """The structural fix for what cost rev 8e5a9c75 forty minutes. An
    operator has to type four names correctly into a 14,000-line file,
    and when they do not the script does not complain -- it skips every
    friend-teleport it has and the party can never regroup. wizAi
    already knows all four."""
    from deimos_bridge.scripts import configure

    out, filled = configure(_QUESTER, [("Phönix", "storm"),
                                       ("Konstantin", "ice")])
    assert dict(filled) == {"Main_Account": "Phönix",
                            "Main_Account_School": "storm",
                            "Questee2": "Konstantin",
                            "Questee2_School": "ice"}
    assert 'var Main_Account = "Phönix"' in out
    assert 'var Questee2 = "Konstantin"' in out
    # A seat the party does not have is left alone.
    assert 'var Questee3 = "QuestingAccountName"' in out


def test_filling_a_preset_leaves_the_scripts_own_guards_alone():
    """The guards compare against the placeholder literal -- that is how
    the script tests whether it was configured. Rewriting them would
    turn every guard permanently false and disable the very code the
    filling exists to switch on."""
    from deimos_bridge.scripts import configure

    out, _filled = configure(_QUESTER, [("Phönix", "storm")])
    assert 'if NOT Main_Account = "QuestingAccountName"' in out
    assert 'if NOT Main_Account_School = "SchoolGoesHere"' in out


def test_a_name_the_operator_typed_is_never_overwritten():
    """They may be running a script whose p1 is not wizAi's seat 1."""
    from deimos_bridge.scripts import configure

    theirs = _QUESTER.replace('var Main_Account = "QuestingAccountName"',
                              'var Main_Account = "Someone Else"')
    out, filled = configure(theirs, [("Phönix", "storm")])
    assert 'var Main_Account = "Someone Else"' in out
    assert "Main_Account" not in dict(filled)


def test_a_wizard_with_no_name_yet_still_gets_its_school():
    """Before a run the names are not knowable -- the game will not give
    them outside character select. Half the job is worth doing; the
    dialog says which half."""
    from deimos_bridge.scripts import configure

    out, filled = configure(_QUESTER, [("", "storm")])
    assert dict(filled) == {"Main_Account_School": "storm"}
    assert 'var Main_Account = "QuestingAccountName"' in out


def test_presets_are_listed_by_their_own_name(tmp_path, monkeypatch):
    from deimos_bridge import scripts

    (tmp_path / "arc1.txt").write_text(_QUESTER, encoding="utf-8")
    (tmp_path / "no_header.txt").write_text("###deimos_expertmode\n",
                                            encoding="utf-8")
    monkeypatch.setattr(scripts, "PRESET_DIR", str(tmp_path))
    got = dict((t, p) for t, p in scripts.presets())
    assert "Arc 1 Quester" in got, got
    assert "no header" in got, got


def test_no_presets_is_a_normal_answer(tmp_path, monkeypatch):
    """wizAi ships none of its own; the directory is where an operator's
    go. Listing must work even when Deimos is not importable, because
    the dialog that lists them is where you find that out."""
    from deimos_bridge import scripts

    monkeypatch.setattr(scripts, "PRESET_DIR", str(tmp_path / "nothing"))
    assert scripts.presets() == []


# --------------------------------------------------- an unconfigured script
def test_a_script_still_on_its_placeholders_is_named():
    """Rev 8e5a9c75 is 110 minutes long and the last 40 are three wizards
    standing in Krokotopia's hub while the script ran 106,000
    instructions and moved nobody. The cause is in the file's first
    thirty lines::

        var Main_Account = "QuestingAccountName"

    The quester guards its friend-teleports with `if NOT Main_Account =
    "QuestingAccountName"`, so an unconfigured script does not fail --
    it silently skips the one mechanism it has for putting a party back
    together."""
    from deimos_bridge.scripts import unconfigured

    source = (
        '###deimos_expertmode\n'
        'var Main_Account = "QuestingAccountName"\n'
        'var Questee2 = "QuestingAccountName"\n'
        'var Main_Account_School = "SchoolGoesHere"\n'
        'var SpeedDelay = 1\n'
        'block Go {\n'
        '  if NOT Main_Account = "QuestingAccountName" {\n'
        '    p1 friendtp Main_Account\n'
        '  }\n'
        '  if NOT Questee2 = "QuestingAccountName" { p2 friendtp Questee2 }\n'
        '  if NOT Main_Account_School = "SchoolGoesHere" { p1 sendkey X }\n'
        '}\n')
    got = dict(unconfigured(source))
    assert got == {"Main_Account": "QuestingAccountName",
                   "Questee2": "QuestingAccountName",
                   "Main_Account_School": "SchoolGoesHere"}, got


def test_a_filled_in_setting_is_not_called_a_placeholder():
    """No hardcoded list of placeholder strings. What marks one is that
    the script COMPARES the variable against the same literal it was
    assigned -- the author's own way of saying "this means unset"."""
    from deimos_bridge.scripts import unconfigured

    source = (
        '###deimos_expertmode\n'
        'var Main_Account = "Phoenix Deathblade"\n'
        'var Questee2 = "QuestingAccountName"\n'
        'block Go {\n'
        '  if NOT Main_Account = "QuestingAccountName" { p1 sendkey X }\n'
        '  if NOT Questee2 = "QuestingAccountName" { p2 sendkey X }\n'
        '}\n')
    got = dict(unconfigured(source))
    assert "Main_Account" not in got, "flagged a name that was filled in"
    assert got == {"Questee2": "QuestingAccountName"}


def test_a_value_that_is_never_compared_is_not_a_placeholder():
    """`var SpeedDelay = "1"` is just a value. Without a guard against
    the same literal there is nothing to say it means unset."""
    from deimos_bridge.scripts import unconfigured

    source = ('###deimos_expertmode\n'
              'var Zone = "Krokotopia/KT_Hub"\n'
              'block Go { p1 tozone Zone }\n')
    assert unconfigured(source) == []


# ------------------------------------------------- in step, and on the line
def _party_on_quests(names, goals=None):
    worker, _read = _zoned_party(["Krokotopia/KT_Pyramid/KT_PalaceOfFire"]
                                 * len(names))
    worker.script = "###deimos_expertmode"
    for i, (seat, name) in enumerate(zip(worker.seats, names)):
        seat.quest_name = name
        seat.goal = goals[i] if goals else f"goal {i}"
        seat.goals_seen = [seat.goal]
    return worker


def test_one_quest_with_five_npcs_is_not_a_desync():
    """Krokotopia #12 is "Gather the Troops" -- Primwell, Archibald,
    Livingston, Farnsworth, Standish -- so a party working through it
    correctly shows three different goal lines at all times. Comparing
    goal TEXT calls that a desync on every tick."""
    import time

    worker = _party_on_quests(
        ["Gather the Troops", "Gather the Troops", "Gather the Troops"],
        ["Talk To Private Farnsworth in Palace of Fire",
         "Talk To Private Livingston in Palace of Fire",
         "Talk To Private Archibald in Palace of Fire"])
    together, why = worker._quests_agree()
    assert together is True, why
    assert "same quest" in why

    worker._check_in_step(worker.seats[0])
    worker._in_step_since = time.monotonic() - 600
    worker._check_in_step(worker.seats[0])
    assert not [e for e in worker.seats[0].tel.questing
                if e["kind"] == "quest-desync"], "still crying desync"
    assert worker._behind is None


def test_one_quest_apart_and_staying_there_is_a_desync():
    """Rev 8e5a9c75, and the correction to the correction. Sebastian sat
    on Krokotopia #13 while the other two sat on #12 for eight unbroken
    minutes, and `BEHIND_BY = 2` called the party together for the whole
    run. A handover is a gap of one that lasts seconds; a desync is a
    gap of one that lasts twenty minutes. `DESYNC_GRACE` tells them
    apart, and it was sitting behind a floor that never let it run."""
    import time

    worker = _party_on_quests(
        ["Gather the Troops", "Gather the Troops", "Payback"],
        ["Talk To Private Farnsworth in Palace of Fire",
         "Talk To Private Livingston in Palace of Fire",
         "Defeat Flame Guardian in Palace of Fire (0 of 3)"])
    together, why = worker._quests_agree()
    assert together is False, why
    assert "1 quests apart" in why or "#12" in why

    worker._check_in_step(worker.seats[0])
    assert not [e for e in worker.seats[0].tel.questing
                if e["kind"] == "quest-desync"], "fired inside the grace period"
    worker._in_step_since = time.monotonic() - 600
    worker._check_in_step(worker.seats[0])
    said = [e for e in worker.seats[0].tel.questing
            if e["kind"] == "quest-desync"]
    assert said, "the whole-run desync is invisible again"
    # ...and BOTH wizards on #12 are named, not neither.
    assert "2 wizards are 1 quest(s) behind" in said[0]["detail"], \
        said[0]["detail"]


def test_a_real_gap_is_still_a_desync():
    import time

    worker = _party_on_quests(["Gather the Troops",      # #12
                               "Find My Colleague"])     # #7
    together, why = worker._quests_agree()
    assert together is False
    assert "5 quests apart" in why
    worker._check_in_step(worker.seats[0])
    worker._in_step_since = time.monotonic() - 600
    worker._check_in_step(worker.seats[0])
    assert [e for e in worker.seats[0].tel.questing
            if e["kind"] == "quest-desync"]


def test_a_party_the_list_cannot_place_still_falls_back_to_goals():
    """Goal text is weak evidence, but for a party the list cannot place
    it is the only evidence there is."""
    worker = _party_on_quests(["A Quest Nobody Wrote Down"] * 2,
                              ["Talk To Someone", "Talk To Someone Else"])
    together, why = worker._quests_agree()
    assert together is None, why
    assert "fewer than two" in why


# ------------------------------------------------------ losing the main line
def test_a_wizard_whose_tracker_wanders_off_the_main_line_is_reported():
    """The operator's request. Wizard101's arrow follows whichever quest
    is SELECTED, so a wizard that picks up a side quest has every `tp
    quest` aimed at it from then on and the party's main-line progress
    silently stops. Rev 8e5a9c75 has Sebastian dropping from Krokotopia
    #13 to unplaced at t=790 and staying there for seven minutes."""
    import time

    worker = _party_on_quests(["Gather the Troops", "Gather the Troops",
                               "Zeke's Quest That Is Not Main"])
    # Third wizard is on a real quest the list knows, with no place in
    # the line -- which is what a side quest looks like.
    from deimos_bridge import questlist
    side = questlist.Position(world="Krokotopia", name="Krokotopian Bundle",
                              how="by quest name", questline=None)
    worker._places = lambda: [questlist.position_of("Gather the Troops"),
                              questlist.position_of("Gather the Troops"),
                              side]
    worker._check_on_questline()                    # starts the clock
    assert not [e for e in worker.seats[2].tel.questing
                if e["kind"] == "off-questline"], "said it far too early"

    worker.seats[2].off_line_since = time.monotonic() - 300
    worker._check_on_questline()
    said = [e for e in worker.seats[0].tel.questing
            if e["kind"] == "off-questline"]
    assert len(said) == 1, said
    assert "Krokotopian Bundle" in said[0]["detail"]
    assert "side quest" in said[0]["detail"]
    assert "#12" in said[0]["detail"], "should say where the party is"
    # once per side quest, not once per tick
    worker._check_on_questline()
    assert len([e for e in worker.seats[0].tel.questing
                if e["kind"] == "off-questline"]) == 1


def test_getting_back_on_the_main_line_is_reported_too():
    import time

    from deimos_bridge import questlist

    worker = _party_on_quests(["Gather the Troops", "Gather the Troops"])
    worker.seats[1].off_line_since = time.monotonic() - 300
    worker.seats[1].said_off_line = "Krokotopian Bundle"
    worker._check_on_questline()
    back = [e for e in worker.seats[1].tel.questing
            if e["kind"] == "back-on-questline"]
    assert back and "#12" in back[0]["detail"]
    assert worker.seats[1].off_line_since is None


def test_an_unreadable_tracker_is_not_called_a_side_quest():
    """`known` False is a read that failed or a quest the list has never
    heard of. Saying "off the questline" on that would fire on every
    loading screen."""
    from deimos_bridge import questlist

    worker = _party_on_quests(["Gather the Troops", "Gather the Troops"])
    worker._places = lambda: [questlist.position_of("Gather the Troops"),
                              questlist.Position()]          # nothing read
    worker._check_on_questline()
    worker._check_on_questline()
    assert worker.seats[1].off_line_since is None
    assert not [e for e in worker.seats[1].tel.questing
                if e["kind"] == "off-questline"]


# ---------------------------------------------------- finishing a missed step
def _desynced(orders):
    """A scripted party whose wizards are on the given Krokotopia steps."""
    names = {2: "Digging in the Dirt", 4: "Fragments of a Key",
             7: "Find My Colleague", 9: "Quarter Master",
             11: "Assault on the Palace"}
    worker, _read = _zoned_party(["Krokotopia/KT_Hub"] * len(orders))
    worker.script = "###deimos_expertmode"
    for seat, order in zip(worker.seats, orders):
        seat.quest_name = names[order]
        seat.goal = f"step {order}"
        seat.goals_seen = [seat.goal]
    return worker


def test_the_desync_check_survives_a_tick_where_nothing_changed():
    """Live at rev 7888c35a, on all three wizards::

        stage-failed: the service loop: UnboundLocalError: cannot
        access local variable 'behind' where it is not associated with
        a value

    `behind` was assigned inside `if where != self._said_desync` and
    read after it, so every tick where the desync line had not CHANGED
    -- which is most of them -- took the whole service loop stage
    down."""
    import time

    worker = _desynced([11, 4])
    worker._check_in_step(worker.seats[0])          # starts the grace clock
    worker._in_step_since = time.monotonic() - 600  # ...which has now run out
    worker._check_in_step(worker.seats[0])          # first tick: says it
    worker._check_in_step(worker.seats[0])          # second: unchanged
    worker._check_in_step(worker.seats[0])          # and again
    said = [e for e in worker.seats[0].tel.questing
            if e["kind"] == "quest-desync"]
    assert len(said) == 1, "the desync line should be said once, not thrice"


def test_starting_and_stopping_a_catch_up_agree_on_where_everyone_is():
    """The start rule placed a wizard by quest name and fell back to its
    goal; the stop rule placed by quest name only. Live, the names did
    not read at all -- so the first found a five-quest gap from the
    goals and the second found nothing to compare, and
    `catch-up-started` and `catch-up-done` share a timestamp in all
    three exports."""
    worker, _read = _zoned_party(["Krokotopia/KT_Hub"] * 2)
    worker.script = "###deimos_expertmode"
    # Exactly the live case: wizard 2's quest NAME read and the other's
    # did not, so wizard 1 could only be placed from its goal -- and
    # that goal is ambiguous between #12 and #13 until a neighbour
    # narrows it.
    worker.seats[0].quest_name = ""
    worker.seats[1].quest_name = "Find My Colleague"          # #7
    worker.seats[0].goal = "Talk To Lieutenant Standish in Palace of Fire"
    worker.seats[1].goal = "Talk To Robert Lancaster in Chamber of Fire"
    for seat in worker.seats:
        seat.goals_seen = [seat.goal]

    behind = worker._who_is_behind()
    assert behind is worker.seats[1], "the goal fallback stopped working"
    worker._start_catching_up(behind, worker._behind_gap,
                              worker._behind_basis)
    worker._check_caught_up()
    assert worker._catching_up() == [worker.seats[1]], \
        "the stop rule disagreed with the start rule about who is where"


def test_the_desync_line_says_how_each_wizard_was_located():
    """All three wizards in the first live run were placed by goal text,
    which means the quest NAME read returned nothing for any of them --
    and nothing in the export said so."""
    import time

    worker = _desynced([11, 4])
    worker._check_in_step(worker.seats[0])
    worker._in_step_since = time.monotonic() - 600
    worker._check_in_step(worker.seats[0])
    said = [e for e in worker.seats[0].tel.questing
            if e["kind"] == "quest-desync"][0]["detail"]
    assert "placed:" in said
    assert "by quest name" in said, said
    assert "#11" in said and "#4" in said, said


def test_placement_does_not_depend_on_which_seat_reads_its_name():
    """One pass over the seats made the answer depend on their order: a
    wizard placed from its goal becomes a hint for the next one, so in a
    single pass only the wizards AFTER it get the benefit. Rev 7888c35a
    placed wizard 1 at #12 purely because wizard 2's name happened to
    read first."""
    ambiguous = "Talk To Lieutenant Standish in Palace of Fire"  # #12 or #13
    anchor = "Talk To Robert Lancaster in Chamber of Fire"       # #7

    for order in ((ambiguous, anchor), (anchor, ambiguous)):
        worker, _read = _zoned_party(["Krokotopia/KT_Hub"] * 2)
        for seat, goal in zip(worker.seats, order):
            seat.quest_name = ""
            seat.goal = goal
        places = worker._places()
        got = sorted(p.order for p in places if p.comparable)
        assert got == [7, 12], f"{order[0][:22]}… first gave {got}"


def test_a_wizard_that_is_behind_is_driven_through_its_own_step():
    """The operator's correction: "just teleporting back to a wizard you
    believe is behind doesn't work because I think the script then just
    continues on the same way. They need to actually quest with the
    other wizard until they catch up."

    One instruction pointer drives the whole party, so a teleport moves
    the laggard's BODY while its quest state stays put -- and the next
    `tp quest` sends it straight back."""
    worker = _desynced([11, 4])
    behind = worker._who_is_behind()
    assert behind is worker.seats[1]
    worker._start_catching_up(behind, worker._behind_gap,
                              worker._behind_basis)
    assert worker._catching_up() == [worker.seats[1]]
    said = [e for e in worker.seats[0].tel.questing
            if e["kind"] == "catch-up-started"]
    assert said and "pausing the script" in said[0]["detail"]
    # ...and every seat's export carries it, not just the laggard's.
    assert all(any(e["kind"] == "catch-up-started" for e in s.tel.questing)
               for s in worker.seats)


def test_the_catch_up_ends_when_the_party_is_level_again():
    worker = _desynced([11, 4])
    worker._start_catching_up(worker.seats[1], 7, "the questline says so")
    worker.seats[1].quest_name = "Assault on the Palace"   # caught up
    worker._check_caught_up()
    assert worker._catching_up() == []
    done = [e for e in worker.seats[0].tel.questing
            if e["kind"] == "catch-up-done"]
    assert done and "has its wizards back" in done[0]["detail"]


def test_a_catch_up_that_is_getting_nowhere_gives_the_script_its_wizards_back():
    """A paused script that is never resumed is the stall this replaces.
    Every exit puts the wizards back on the program, including this
    one."""
    import time

    worker = _desynced([11, 4])
    worker._start_catching_up(worker.seats[1], 7, "the questline says so")
    state = worker._catch_up_state
    state["started"] = time.monotonic() - worker.CATCH_UP_LIMIT - 1
    worker._check_caught_up()
    assert worker._catching_up() == []
    gave = [e for e in worker.seats[0].tel.questing
            if e["kind"] == "catch-up-gave-up"]
    assert gave and "still 7 quest(s) behind" in gave[0]["detail"]


def test_a_catch_up_that_stops_advancing_is_written_off_sooner():
    """The overall limit is generous because a missed step can be a
    dungeon boss. A step that has not moved at all is a different
    thing."""
    import time

    worker = _desynced([11, 4])
    worker._start_catching_up(worker.seats[1], 7, "the questline says so")
    worker._catch_up_state["moved"] = time.monotonic() - worker.CATCH_UP_STALL - 1
    worker._check_caught_up()
    assert worker._catching_up() == []
    gave = [e for e in worker.seats[0].tel.questing
            if e["kind"] == "catch-up-gave-up"]
    assert gave and "nothing has moved" in gave[0]["detail"]


def test_progress_towards_catching_up_restarts_the_stall_clock():
    """Closing the gap counts as movement even when the goal LINE
    happens not to change -- a multi-part step reads the same string
    throughout."""
    import time

    worker = _desynced([11, 2])
    worker._start_catching_up(worker.seats[1], 9, "the questline says so")
    worker._catch_up_state["moved"] = time.monotonic() - worker.CATCH_UP_STALL - 1
    worker.seats[1].quest_name = "Find My Colleague"       # #2 -> #7
    worker._check_caught_up()
    assert worker._catching_up() == [worker.seats[1]], \
        "gave up despite progress"
    assert worker._catch_up_state["gap"] == 4


def test_one_of_the_group_catching_up_does_not_end_the_catch_up():
    """Rev 1dcf4193: two catch-ups ended 10.6s and 0.4s after they began.
    The stop rule fired whenever the behind SET changed, and in a party
    of three somebody's quest ticks over almost immediately -- so
    `{Konstantin, Phoenix}` became `{Konstantin}` and the whole thing
    stopped with Konstantin still a quest behind.

    A subset is progress. Only an empty set, or a wizard who was not in
    the group, ends it."""
    worker = _desynced([11, 4, 4])
    behind = [worker.seats[1], worker.seats[2]]
    worker._start_catching_up(behind, 7, "the questline says so")
    assert worker._catching_up() == behind

    worker.seats[2].quest_name = "Assault on the Palace"      # caught up
    worker._check_caught_up()
    assert worker._catching_up() == [worker.seats[1]], \
        "gave up on the wizard that is still behind"
    assert not [e for e in worker.seats[0].tel.questing
                if e["kind"] == "catch-up-done"], "called it done too early"

    worker.seats[1].quest_name = "Assault on the Palace"      # and now both
    worker._check_caught_up()
    assert worker._catching_up() == []
    assert [e for e in worker.seats[0].tel.questing
            if e["kind"] == "catch-up-done"]


def test_a_new_laggard_gets_its_own_catch_up_not_this_ones_clock():
    worker = _desynced([11, 4, 9])
    worker._start_catching_up(worker.seats[1], 7, "the questline says so")
    worker.seats[1].quest_name = "Quarter Master"          # now level with s2
    worker.seats[2].quest_name = "Digging in the Dirt"     # s2 fell back
    worker._check_caught_up()
    assert worker._catching_up() == [], \
        "the old catch-up must end before a new one starts"


def test_a_desync_the_questline_cannot_measure_does_not_pause_the_script():
    """The older rules say WHO is behind, not how far -- and without a
    distance there is no way to tell when to stop. Pausing the script on
    one of those would be a pause with no exit condition."""
    worker, _read = _zoned_party(["Olde Town", "Olde Town"])
    worker.script = "###deimos_expertmode"
    for seat, seen in zip(worker.seats, [["Talk To Danforth", "Find Key Stone"],
                                         ["Talk To Danforth"]]):
        seat.goals_seen, seat.goal = list(seen), seen[-1]
        seat.quest_name = "A Quest Nobody Wrote Down"
    behind = worker._who_is_behind()
    assert behind is worker.seats[1]
    assert worker._behind_basis == "another wizard has finished that step"
    assert worker._behind_gap == 0, "a rule with no distance must not arm one"


# ------------------------------------------------------------- who is behind
def _party_on(goals_history):
    """Seats whose goal histories are the lists given, current = last."""
    worker, _read = _zoned_party(["Olde Town"] * len(goals_history))
    for seat, seen in zip(worker.seats, goals_history):
        seat.goals_seen = list(seen)
        seat.goal = seen[-1]
    return worker


def test_the_questline_outranks_the_older_guesses():
    """The party never overlapped -- no wizard has held another's goal --
    so the history rule has nothing and the old code fell through to the
    clock, which is a guess. The quest list knows: Krokotopia main #4 is
    behind main #9, as a fact."""
    worker = _party_on([["Assault on the Palace"], ["Fragments of a Key"]])
    worker.seats[0].quest_name = "Assault on the Palace"      # main #11
    worker.seats[1].quest_name = "Fragments of a Key"         # main #4
    behind = worker._who_is_behind()
    assert behind is worker.seats[1], "the questline was ignored"
    assert "the questline says so" in worker._behind_basis
    assert "#4 against #11" in worker._behind_basis
    assert worker._behind_gap == 7


def test_a_questline_that_cannot_place_a_wizard_falls_back(monkeypatch):
    """A side quest carries no order, so the questline rule must decline
    and the party's own demonstration must still be reachable. A rule
    that answers only when it knows is worth more than one that always
    answers -- but it must not block the ones underneath it."""
    worker = _party_on([["Talk To Danforth", "Find Key Stone"],
                        ["Talk To Danforth"]])
    # Neither name is in the list at all.
    worker.seats[0].quest_name = "A Quest Nobody Wrote Down"
    worker.seats[1].quest_name = "Another One"
    behind = worker._who_is_behind()
    assert behind is worker.seats[1]
    assert worker._behind_basis == "another wizard has finished that step"


def test_a_handover_is_filtered_by_TIME_not_by_size_of_gap():
    """This asserted the opposite, and the opposite was wrong.

    Turning a quest in is not simultaneous, so a party is briefly one
    quest apart on every handover -- and the first fix for that was a
    floor of two, which also swallowed a gap of one that lasted a whole
    run. A handover is a gap of one that lasts seconds; a desync is a
    gap of one that lasts twenty minutes. Only the clock separates
    them.

    So the gap IS seen immediately, and `DESYNC_GRACE` decides whether
    it is worth saying."""
    import time

    worker = _party_on([["Quarter Master"], ["Back to the Archeologist"]])
    worker.seats[0].quest_name = "Quarter Master"              # main #9
    worker.seats[1].quest_name = "Back to the Archeologist"    # main #10
    assert worker._behind_by_questline() is worker.seats[0], \
        "a one-quest gap is invisible again"
    assert worker._behind_gap == 1

    worker._check_in_step(worker.seats[0])       # starts the grace clock
    assert not [e for e in worker.seats[0].tel.questing
                if e["kind"] == "quest-desync"], "reported a handover"
    worker._in_step_since = time.monotonic() - worker.DESYNC_GRACE - 1
    worker._check_in_step(worker.seats[0])
    assert [e for e in worker.seats[0].tel.questing
            if e["kind"] == "quest-desync"], \
        "a gap that outlived the grace period is a desync"


def test_the_wizard_still_on_a_finished_step_is_the_one_behind():
    """Rev 1d28f745, the second of two desyncs sixteen seconds apart::

        w1 Find Key Stone · w2 Talk To Danforth · w3 Find Key Stone

    w1 moved ONTO w3's goal, so w3 was ahead all along and w2 missed the
    step. The old rule -- whichever goal changed least recently -- named
    w3, exactly backwards."""
    worker = _party_on([["Talk To Danforth", "Find Key Stone"],
                        ["Talk To Danforth"],
                        ["Find Key Stone"]])
    assert worker._who_is_behind() is worker.seats[1]


def test_history_beats_the_clock_when_they_disagree():
    """The clock says w3 (its goal moved least recently); the history
    says w2 (w1 finished w2's step). The history is evidence and the
    clock is a guess, so evidence wins."""
    worker = _party_on([["Talk To Danforth", "Find Key Stone"],
                        ["Talk To Danforth"],
                        ["Find Key Stone"]])
    worker.seats[0].goal_at = 100.0
    worker.seats[1].goal_at = 50.0
    worker.seats[2].goal_at = 10.0        # oldest -- the clock's answer
    assert worker._who_is_behind() is worker.seats[1]
    assert "finished that step" in worker._behind_basis


def test_two_wizards_behind_is_not_a_wizard_to_walk_the_party_to():
    """`_behind` is what `_should_catch_up` sends the others back to, so
    an honest "cannot tell" beats confidently picking one of two."""
    worker = _party_on([["A", "B"], ["A"], ["A"]])
    assert worker._who_is_behind() is None
    assert "more than one" in worker._behind_basis


def test_with_no_observed_order_the_clock_is_used_and_said_to_be_a_guess():
    """A party that has not yet been seen to advance demonstrates no
    order at all. The clock is what shipped before this; the basis is
    recorded so the export shows when a guess was made."""
    worker = _party_on([["Talk To Danforth"], ["Find Key Stone"]])
    worker.seats[0].goal_at, worker.seats[1].goal_at = 50.0, 10.0
    assert worker._who_is_behind() is worker.seats[1]
    assert "guessed" in worker._behind_basis


def test_a_party_in_step_never_reaches_the_question():
    """`_who_is_behind` is only consulted once `goals_agree` has already
    said the party has drifted, so "everyone on the same step" is the
    caller's guard rather than this one's."""
    from deimos_bridge import questing

    worker = _party_on([["A", "B"], ["A", "B"], ["A", "B"]])
    assert questing.goals_agree([s.goal for s in worker.seats])


def test_the_heartbeat_reads_the_zone_when_nothing_else_has(monkeypatch):
    """Every log at rev 1d28f745 says "zone unread" for its first eight
    minutes: `zone_seen` is filled by the stranded poll, which only runs
    once a script is driving. The zone is the first thing anybody reads
    on one of these lines."""
    import asyncio

    from deimos_bridge import party

    worker, seat = _beating()
    seat.zone_seen = None

    async def read(_client):
        return "Krokotopia/KT_Pyramid/KT_Chamber"

    async def full(_seat):
        return 1.0

    monkeypatch.setattr(party, "zone", read)
    worker._health_left = full
    asyncio.run(worker._heartbeat(seat))
    beat = [e for e in seat.tel.questing if e["kind"] == "heartbeat"][0]
    assert "KT_Chamber" in beat["detail"]


# --------------------------------------- two sigils in one zone
#
# What the last twenty minutes of rev 1dcf4193 was. Three wizards in
# Krokotopia/KT_Pyramid/KT_PalaceOfFire, all three with "Press X to
# Enter" up, party frozen -- two at Edo's Chamber and one at Akori's. No
# zone check can see it: the zone agrees, the goal agrees, and every
# wizard is doing something.

def _sigil_party(spots, prompts=None):
    """A worker whose seats stand at the given positions, all in one zone.

    `spots` are (x, y) pairs; `prompts` says which seats have the game's
    press-X window up (default: all of them).
    """
    class _XY:
        def __init__(self, x, y):
            self.x, self.y, self.z = float(x), float(y), 0.0

    worker, _read = _zoned_party(["KT_PalaceOfFire"] * len(spots))
    for seat, (x, y) in zip(worker.seats, spots):
        seat.zone_seen = "KT_PalaceOfFire"
        seat._spot = _XY(x, y)
    shown = [True] * len(spots) if prompts is None else list(prompts)
    for seat, up in zip(worker.seats, shown):
        seat._prompt = up
    return worker


def _look_at_sigils(worker, monkeypatch):
    import asyncio

    from deimos_bridge import party, questing

    def find(client):
        return next(s for s in worker.seats if s.client is client)

    async def near(client):
        return find(client)._prompt

    async def where(client):
        return find(client)._spot

    monkeypatch.setattr(questing, "near_interactable", near)
    monkeypatch.setattr(party, "position", where)
    asyncio.run(worker._check_same_sigil())
    return [e for s in worker.seats for e in s.tel.questing
            if e["kind"] == "split-sigil"]


def test_two_sigils_in_one_zone_are_named(monkeypatch):
    """The party is not stalled, stranded, off-questline or desynced.
    It is standing on two different doors."""
    import time

    worker = _sigil_party([(0, 0), (100, 0), (5000, 0)])
    assert _look_at_sigils(worker, monkeypatch) == [], \
        "it fired before the clock had run"
    for seat in worker.seats:
        if seat.apart_since is not None:
            seat.apart_since -= worker.SPLIT_AFTER + 1
    said = _look_at_sigils(worker, monkeypatch)
    assert len(said) == len(worker.seats), \
        "every wizard's export has to explain its own twenty minutes"
    detail = said[0]["detail"]
    assert "w2" in detail and "w0 and w1" in detail
    assert "KT_PalaceOfFire" in detail


def test_a_party_on_one_sigil_is_left_alone(monkeypatch):
    worker = _sigil_party([(0, 0), (150, 80), (60, 200)])
    for seat in worker.seats:
        seat.apart_since = 1.0                 # as if it had been running
    assert _look_at_sigils(worker, monkeypatch) == []
    assert all(s.apart_since is None for s in worker.seats)


def test_a_wizard_not_at_a_prompt_at_all_is_not_a_split(monkeypatch):
    """It is walking, or fighting, or stuck — all of which have their own
    report. This one is only about wizards that are each at a door."""
    worker = _sigil_party([(0, 0), (100, 0), (5000, 0)],
                          prompts=[True, True, False])
    for seat in worker.seats:
        seat.apart_since = 1.0
    assert _look_at_sigils(worker, monkeypatch) == []


def test_three_wizards_at_three_different_prompts_is_not_reported(
        monkeypatch):
    """No majority means no answer to "which sigil is the party's", and
    saying one of them is the odd one out would be a guess."""
    worker = _sigil_party([(0, 0), (5000, 0), (10000, 0)])
    for seat in worker.seats:
        seat.apart_since = 1.0
    assert _look_at_sigils(worker, monkeypatch) == []


def test_a_party_in_different_zones_is_the_other_check(monkeypatch):
    """Different zones is `_check_together`, which fetches the straggler.
    This one must not double-report it."""
    worker = _sigil_party([(0, 0), (100, 0), (5000, 0)])
    worker.seats[2].zone_seen = "KT_Pyramid"
    for seat in worker.seats:
        seat.apart_since = 1.0
    assert _look_at_sigils(worker, monkeypatch) == []


def test_walking_to_the_sigil_clears_the_clock(monkeypatch):
    """A wizard that arrives late is not a split party — it is a wizard
    that arrived late, and the clock must not carry over."""
    worker = _sigil_party([(0, 0), (100, 0), (5000, 0)])
    _look_at_sigils(worker, monkeypatch)
    assert worker.seats[2].apart_since is not None
    worker.seats[2]._spot.x = 120.0                 # it walked over
    assert _look_at_sigils(worker, monkeypatch) == []
    assert worker.seats[2].apart_since is None


def test_the_worker_wires_the_round_clock_to_the_export():
    """A hook nothing installs measures nothing. The wiring is one line
    in `_go` and it is the only thing between `report_round_timing` and
    a file that can answer "why is it slow"."""
    import inspect

    from deimos_bridge.gui.live import LiveWorker

    src = inspect.getsource(LiveWorker._go)
    assert "backend.on_round_timing = self._round_timing_hook(seat)" in src, \
        "rounds are being played without recording what they cost"


def test_the_round_clock_writes_through_to_the_record():
    """Not a source check. The hook, run, against a real Telemetry."""
    from deimos_bridge.gui.live import LiveWorker
    from deimos_bridge.telemetry import Telemetry

    class _Seat:
        tel = Telemetry()

    seat = _Seat()
    seat.tel.start_fight()
    rec = seat.tel.observe(_Decision("Sunbird"), _read(2000, 1))
    LiveWorker._round_timing_hook(object(), seat)(31.0, 3.5, 0.9)
    assert (rec.waited, rec.planned, rec.acted) == (31.0, 3.5, 0.9)


def test_a_party_split_down_the_middle_names_no_odd_one_out(monkeypatch):
    """Two at each sigil. Both halves are stuck and neither is "the
    party" — calling one of them the straggler would be a coin toss
    reported as a diagnosis."""
    worker = _sigil_party([(0, 0), (100, 0), (5000, 0), (5100, 0)])
    for seat in worker.seats:
        seat.apart_since = 1.0
    assert _look_at_sigils(worker, monkeypatch) == []


def test_the_fourth_seat_of_a_party_of_three_is_not_unconfigured():
    """Rev 1dcf4193's export opened with

        the script has 2 setting(s) still at its placeholder value
        (Questee4, Questee4_School)

    which reads as the cause of the stall below it and was nothing —
    there was no fourth wizard, and the script's guard on it was doing
    exactly its job."""
    from deimos_bridge.scripts import unconfigured, unfilled

    source = (
        '###deimos_expertmode\n'
        'var Main_Account = "QuestingAccountName"\n'
        'var Questee2 = "QuestingAccountName2"\n'
        'var Questee4 = "QuestingAccountName4"\n'
        'var Questee4_School = "SchoolPlaceholder"\n'
        'if NOT Main_Account = "QuestingAccountName" then\n'
        'if NOT Questee2 = "QuestingAccountName2" then\n'
        'if NOT Questee4 = "QuestingAccountName4" then\n'
        'if NOT Questee4_School = "SchoolPlaceholder" then\n')

    everything = [n for n, _v in unconfigured(source)]
    assert "Questee4" in everything, "the fixture is not exercising the case"

    named = [n for n, _v in unfilled(source, 3)]
    assert "Questee4" not in named and "Questee4_School" not in named
    assert "Main_Account" in named and "Questee2" in named

    # ...and with no party attached, nothing is filtered: a pre-flight on
    # a pasted script does not know how many wizards will run it.
    assert [n for n, _v in unfilled(source, None)] == everything


# ----------------------------------- a wizard cannot un-finish a quest
#
# Rev 3822cc6c. Sebastian, in step with the party all run, read `Talk To
# Robert Lancaster in Chamber of Fire` for one tick -- a real goal, and
# one that belongs to Krokotopia #7. The party declared an EIGHT quest
# gap and paused the script to catch him up. Fifteen seconds later he
# read #15 again, which is where he had been the whole time.

def _placed_party(names):
    """A worker whose seats read the given tracked quest names."""
    worker, _read = _zoned_party(["KT_Hub"] * len(names))
    for seat, name in zip(worker.seats, names):
        seat.quest_name = name
        seat.goal = ""
    return worker


def test_a_backwards_reading_is_clamped_not_believed():
    from deimos_bridge import questlist

    worker, _read = _zoned_party(["KT_Hub"])
    seat = worker.seats[0]

    ahead = questlist.Position(world="Krokotopia", order=15, name="Back to "
                               "Winthrop", how="by quest name")
    kept = worker._no_further_back(seat, ahead)
    assert kept.order == 15
    assert seat.furthest["Krokotopia"] == 15

    behind = questlist.Position(world="Krokotopia", order=7, name="Lancaster",
                                how="by goal text")
    clamped = worker._no_further_back(seat, behind)
    assert clamped.order == 15, "a wizard was allowed to un-finish 8 quests"
    assert "clamped" in clamped.how and "#7" in clamped.how, \
        "a silent clamp hides a genuine reset behind a number that never moves"


def test_going_forward_is_always_believed():
    from deimos_bridge import questlist

    worker, _read = _zoned_party(["KT_Hub"])
    seat = worker.seats[0]
    for order in (3, 4, 9, 15):
        place = questlist.Position(world="Krokotopia", order=order,
                                   how="by quest name")
        assert worker._no_further_back(seat, place).order == order
    assert seat.furthest["Krokotopia"] == 15


def test_the_next_world_starts_from_the_beginning():
    """Marleybone #1 is not eight quests behind Krokotopia #15, and a
    clamp that forgot which world it was in would pin every wizard at
    the last world's number forever."""
    from deimos_bridge import questlist

    worker, _read = _zoned_party(["KT_Hub"])
    seat = worker.seats[0]
    worker._no_further_back(seat, questlist.Position(
        world="Krokotopia", order=15, how="by quest name"))
    fresh = worker._no_further_back(seat, questlist.Position(
        world="Marleybone", order=1, how="by quest name"))
    assert fresh.order == 1 and "clamped" not in (fresh.how or "")


def test_an_unplaceable_reading_passes_straight_through():
    """A side quest has no order at all. Clamping it would invent one."""
    from deimos_bridge import questlist

    worker, _read = _zoned_party(["KT_Hub"])
    seat = worker.seats[0]
    worker._no_further_back(seat, questlist.Position(
        world="Krokotopia", order=15, how="by quest name"))
    side = questlist.Position(world=None, order=None, name="a side quest")
    assert worker._no_further_back(seat, side) is side


def test_places_applies_the_clamp():
    """Not just the helper — the function two rules and a script pause
    hang off."""
    worker = _placed_party(["Back to Winthrop"] * 3)
    first = worker._places()
    assert all(p.comparable for p in first), "the fixture did not place"
    ahead = first[0].order

    # seat 3's tracker flips to something much earlier in the line
    worker.seats[2].quest_name = ""
    worker.seats[2].goal = "Talk To Robert Lancaster in Chamber of Fire"
    after = worker._places()
    assert after[2].order >= ahead, \
        "one tick of a shared goal line moved a wizard eight quests back"


# ------------------------------- a catch-up that nothing is attempting

def test_a_catch_up_where_nobody_moves_or_fights_lets_go_early():
    """Rev 3822cc6c froze all three wizards at 8,083 instructions for the
    rest of the export, with another 140s to run before CATCH_UP_STALL
    would have released them. `hop_once` was teleporting Phönix to a
    marker he was already standing on."""
    import time

    worker, _read = _zoned_party(["KT_PalaceOfFire"] * 3)
    # Krokotopia #14 against #15 twice — the shape of rev 3822cc6c.
    for seat, quest in zip(worker.seats, ["Give em Another Round",
                                          "Back to Winthrop",
                                          "Back to Winthrop"]):
        seat.quest_name = quest
    laggard = worker.seats[0]
    now = time.monotonic()
    worker._catch_up_state = {
        "seats": [laggard], "gap": 1, "started": now - 100.0,
        "moved": now - 100.0, "goals": {id(laggard): laggard.goal}, "why": "",
    }
    for seat in worker.seats:
        seat.progress = ("KT_PalaceOfFire", (1, 2, 3), "Defeat Edo Nirini")
        seat.progress_at = now - worker.CATCH_UP_IDLE - 1
        seat.in_duel = False

    worker._check_caught_up()
    assert worker._catch_up_state is None, "the party is still frozen"
    said = [e for s in worker.seats for e in s.tel.questing
            if e["kind"] == "catch-up-gave-up"]
    assert said, "it let go without saying why"
    assert "not moved" in said[0]["detail"]


def test_a_catch_up_whose_laggard_is_fighting_is_left_alone():
    """A duel is the catch-up working. The goal line does not change
    while a boss is fought, which is exactly why CATCH_UP_STALL is
    generous — and why the short bound must not apply here."""
    import time

    worker, _read = _zoned_party(["KT_PalaceOfFire"] * 3)
    # Krokotopia #14 against #15 twice — the shape of rev 3822cc6c.
    for seat, quest in zip(worker.seats, ["Give em Another Round",
                                          "Back to Winthrop",
                                          "Back to Winthrop"]):
        seat.quest_name = quest
    laggard = worker.seats[0]
    now = time.monotonic()
    worker._catch_up_state = {
        "seats": [laggard], "gap": 1, "started": now - 100.0,
        "moved": now - 100.0, "goals": {id(laggard): laggard.goal}, "why": "",
    }
    for seat in worker.seats:
        seat.progress = ("KT_PalaceOfFire", (1, 2, 3), "Defeat Edo Nirini")
        seat.progress_at = now - worker.CATCH_UP_IDLE - 1
        seat.in_duel = False
    laggard.in_duel = True

    worker._check_caught_up()
    assert worker._catch_up_state is not None, \
        "it gave up on a wizard that was fighting the step"


def test_a_laggard_that_is_moving_keeps_its_full_budget():
    import time

    worker, _read = _zoned_party(["KT_PalaceOfFire"] * 3)
    # Krokotopia #14 against #15 twice — the shape of rev 3822cc6c.
    for seat, quest in zip(worker.seats, ["Give em Another Round",
                                          "Back to Winthrop",
                                          "Back to Winthrop"]):
        seat.quest_name = quest
    laggard = worker.seats[0]
    now = time.monotonic()
    worker._catch_up_state = {
        "seats": [laggard], "gap": 1, "started": now - 100.0,
        "moved": now - 100.0, "goals": {id(laggard): laggard.goal}, "why": "",
    }
    for seat in worker.seats:
        seat.progress = ("KT_PalaceOfFire", (1, 2, 3), "Defeat Edo Nirini")
        seat.progress_at = now - 5.0            # it moved five seconds ago
        seat.in_duel = False

    worker._check_caught_up()
    assert worker._catch_up_state is not None


# ------------------------------- the wait that was the walk to the sigil

def test_the_first_round_of_a_fight_does_not_charge_the_walk_to_it():
    """Rev 3822cc6c reports `waited: 114.82` on a 35-second round. That
    is the zone change and the walk between duels, not the game taking
    two minutes over a planning phase."""
    import asyncio

    from deimos_bridge.live_backend import PolicyDecision

    async def _pass():
        return PolicyDecision(passing=True, reason="pass")

    handler, _passes = _stub_handler(_pass, read=object())
    rounds = iter([1, 2, 1])                    # two rounds, then a new duel
    handler.round_number = lambda: _value(next(rounds))
    handler.backend.timed = []

    for _ in range(3):
        asyncio.run(handler.handle_round())

    waits = [w for w, _p, _a in handler.backend.timed]
    assert waits[0] is None, "there was no round before the first one"
    assert waits[1] is not None, "the second round of a duel does wait"
    assert waits[2] is None, "the walk between fights was charged to a round"


# ------------------------------- giving up has to mean something
#
# Rev 3822cc6c, continued. `catch-up-gave-up` and `catch-up-started`
# share a timestamp SEVEN times across thirty-five minutes -- every five
# minutes, the same wizard, the same step, the same 8,083 instructions.
# `_stop_catching_up` cleared the state and `_check_in_step` saw the same
# desync on the very next tick and started the identical catch-up again.
# The script was handed back its wizards for zero seconds each time.

def _behind_party():
    """A party where seat 1 is one quest behind the other two."""
    worker, _read = _zoned_party(["KT_PalaceOfFire"] * 3)
    worker.script = "###deimos_expertmode\n"
    for seat, quest in zip(worker.seats, ["Give em Another Round",
                                          "Back to Winthrop",
                                          "Back to Winthrop"]):
        seat.quest_name = quest
        seat.goal = f"goal for {quest}"
    return worker


def _make_it_give_up(worker):
    """Run a catch-up to its give-up, the way the live loop does."""
    import time

    laggard = worker.seats[0]
    now = time.monotonic()
    worker._catch_up_state = {
        "seats": [laggard], "gap": 1, "started": now - 100.0,
        "moved": now - 100.0, "goals": {id(laggard): laggard.goal}, "why": "",
    }
    for seat in worker.seats:
        seat.progress = ("KT_PalaceOfFire", (1, 2, 3), seat.goal)
        seat.progress_at = now - worker.CATCH_UP_IDLE - 1
        seat.in_duel = False
    worker._check_caught_up()
    assert worker._catch_up_state is None, "the fixture did not give up"
    return laggard


def test_a_catch_up_that_gave_up_is_not_started_again_on_the_same_step():
    import time

    worker = _behind_party()

    # First, the trigger DOES fire on this party — otherwise the test
    # below proves nothing.
    worker._in_step_since = time.monotonic() - worker.DESYNC_GRACE - 1
    worker._check_in_step(worker.seats[0])
    assert worker._catch_up_state is not None, \
        "the fixture never triggers a catch-up at all"

    laggard = _make_it_give_up(worker)
    assert worker._written_off([laggard]), \
        "giving up was a word rather than an act"

    # ...and now the same tick that started one must not start another.
    worker._in_step_since = time.monotonic() - worker.DESYNC_GRACE - 1
    worker._check_in_step(worker.seats[0])
    assert worker._catch_up_state is None, \
        "gave up and started the identical catch-up on the next tick"
    kinds = [e["kind"] for e in laggard.tel.questing]
    assert "catch-up-written-off" in kinds, \
        "the party stopped being paused and the export does not say why"


def test_a_wizard_that_moved_on_gets_a_fresh_chance():
    """A step that CHANGED is a different question, and the verdict on
    the old one must not carry over to it."""
    worker = _behind_party()
    laggard = _make_it_give_up(worker)
    assert worker._written_off([laggard])

    # past the cooldown floor, which is a separate guard — see below
    step, when = worker._wrote_off[id(laggard)]
    worker._wrote_off[id(laggard)] = (step, when - worker.CATCH_UP_COOLDOWN - 1)

    laggard.quest_name = "Back to Winthrop"        # it finished the step
    assert not worker._written_off([laggard]), \
        "a wizard that made progress is still being written off"
    assert id(laggard) not in worker._wrote_off, \
        "the stale verdict was not forgotten"


def test_the_same_step_stays_written_off_however_long_it_has_been():
    """Not a cooldown — a verdict. Seven attempts at an identical step
    over thirty-five minutes produced nothing, and the thirty-sixth
    minute is not going to be different."""
    worker = _behind_party()
    laggard = _make_it_give_up(worker)
    step, when = worker._wrote_off[id(laggard)]
    worker._wrote_off[id(laggard)] = (step, when - 3600.0)
    assert worker._written_off([laggard])


def test_the_cooldown_holds_even_when_the_step_reads_differently():
    """Two readings that alternate must not be able to thrash the party
    between them faster than the cooldown."""
    worker = _behind_party()
    laggard = _make_it_give_up(worker)
    laggard.quest_name = "Payback"                 # a different step
    assert worker._written_off([laggard]), \
        "a changed reading beat the cooldown floor"


def test_one_wizard_of_a_group_that_moved_on_reopens_the_question():
    """Written off only when EVERY wizard in the group is."""
    worker = _behind_party()
    import time

    now = time.monotonic()
    a, b = worker.seats[0], worker.seats[1]
    worker._wrote_off = {
        id(a): (worker._step_key(a), now - worker.CATCH_UP_COOLDOWN - 1),
    }
    assert not worker._written_off([a, b]), \
        "a group with an untried wizard in it was refused"


def test_the_written_off_step_is_said_in_the_export():
    """The operator has to be able to see WHY the party stopped being
    paused, or "it gave up" and "it never noticed" look the same."""
    worker = _behind_party()
    laggard = _make_it_give_up(worker)
    worker._behind = laggard
    worker._behind_group = [laggard]
    worker._behind_gap = 1
    worker._behind_basis = "the questline says so"
    worker._say_once(laggard, "k", "m", kind="catch-up-written-off",
                     detail="already given up on this step")
    kinds = [e["kind"] for e in laggard.tel.questing]
    assert "catch-up-written-off" in kinds


def test_x_is_not_pressed_a_fifth_time_from_a_spot_where_it_does_nothing(
        monkeypatch):
    """Rev 3822cc6c pressed X five times over twenty-four minutes while
    Konstantin stood 22,778 units from `Talk To Professor Winthrop in
    Altar of Kings` — a different zone — and every one logged "a box did
    not come up". The first press is worth making. The fifth buries the
    fact that this wizard needs moving, not interacting."""
    import asyncio

    from deimos_bridge import questing

    worker, seat = _wedged(dialogue=False, monkeypatch=monkeypatch)
    worker.seats[0].runner = type("R", (), {"running": True,
                                            "steps": 11582})()
    pressed = []

    async def press_x(_c, **kw):
        pressed.append(True)
        return True, ""

    async def never_opens(_c, **kw):
        return False

    monkeypatch.setattr(questing, "press_x", press_x)
    monkeypatch.setattr(questing, "dialogue_opened", never_opens)

    for _ in range(6):
        seat.unstuck_at = 0.0
        asyncio.run(worker._unstick(seat))

    assert len(pressed) == worker.X_TRIES_HERE, \
        f"pressed X {len(pressed)} times at something that never answered"
    gave_up = [e for e in seat.tel.questing
               if e["kind"] == "unstuck-x-does-nothing"]
    assert gave_up, "it stopped pressing X and the export does not say why"
    assert "22,778" not in gave_up[0]["detail"]     # the fixture's own number
    assert "the marker is 900 away" in gave_up[0]["detail"], \
        "the reason it is going nowhere has to travel with the report"


def test_moving_earns_a_fresh_pair_of_presses(monkeypatch):
    """The count is per SPOT. A wizard that walked somewhere else is
    asking a different question of a different interactable."""
    import asyncio

    from deimos_bridge import questing

    worker, seat = _wedged(dialogue=False, monkeypatch=monkeypatch)
    worker.seats[0].runner = type("R", (), {"running": True,
                                            "steps": 11582})()
    pressed = []

    async def press_x(_c, **kw):
        pressed.append(True)
        return True, ""

    async def never_opens(_c, **kw):
        return False

    monkeypatch.setattr(questing, "press_x", press_x)
    monkeypatch.setattr(questing, "dialogue_opened", never_opens)

    for _ in range(4):
        seat.unstuck_at = 0.0
        asyncio.run(worker._unstick(seat))
    assert len(pressed) == worker.X_TRIES_HERE

    seat.progress = ("WizardCity/WC_NightSide", (9, 9, 9), seat.progress[2])
    for _ in range(4):
        seat.unstuck_at = 0.0
        asyncio.run(worker._unstick(seat))
    assert len(pressed) == worker.X_TRIES_HERE * 2, \
        "a wizard that moved was still refused"


def test_a_box_that_does_come_up_clears_the_count(monkeypatch):
    """X working is the whole point of pressing it, and a working press
    must not count towards giving up on the next one."""
    import asyncio

    from deimos_bridge import questing

    worker, seat = _wedged(dialogue=False, monkeypatch=monkeypatch)
    worker.seats[0].runner = type("R", (), {"running": True,
                                            "steps": 11582})()

    async def press_x(_c, **kw):
        return True, ""

    async def opens(_c, **kw):
        return True

    monkeypatch.setattr(questing, "press_x", press_x)
    monkeypatch.setattr(questing, "dialogue_opened", opens)

    for _ in range(4):
        seat.unstuck_at = 0.0
        asyncio.run(worker._unstick(seat))
    assert seat.x_pressed == 0
    assert not [e for e in seat.tel.questing
                if e["kind"] == "unstuck-x-does-nothing"]


def test_a_catch_up_gets_its_own_clock_not_the_stall_that_caused_it():
    """Rev 3d026ada: `catch-up-started` and `catch-up-gave-up` on the
    same timestamp, reason "has not moved or fought for 122s". The 122s
    were spent stuck BEFORE the catch-up — which is why there is a
    catch-up at all. Every catch-up worth having is for a wizard that
    was already standing still, so an absolute idle clock kills all of
    them at birth and none ever gets a tick to teleport anywhere."""
    import time

    worker = _behind_party()
    laggard = worker.seats[0]
    now = time.monotonic()
    worker._catch_up_state = {
        "seats": [laggard], "gap": 1, "started": now,      # just began
        "moved": now, "goals": {id(laggard): laggard.goal}, "why": "",
    }
    for seat in worker.seats:
        seat.progress = ("KT_PalaceOfFire", (1, 2, 3), seat.goal)
        # stuck for a long time already, which is the reason for the
        # catch-up, not a reason to abandon it
        seat.progress_at = now - 122.0
        seat.in_duel = False

    worker._check_caught_up()
    assert worker._catch_up_state is not None, \
        "the catch-up gave up before it had run for a single tick"


def test_the_idle_clock_still_fires_once_the_catch_up_has_had_its_chance():
    import time

    worker = _behind_party()
    laggard = worker.seats[0]
    now = time.monotonic()
    began = now - worker.CATCH_UP_IDLE - 1
    worker._catch_up_state = {
        "seats": [laggard], "gap": 1, "started": began, "moved": began,
        "goals": {id(laggard): laggard.goal}, "why": "",
    }
    for seat in worker.seats:
        seat.progress = ("KT_PalaceOfFire", (1, 2, 3), seat.goal)
        seat.progress_at = now - 500.0
        seat.in_duel = False

    worker._check_caught_up()
    assert worker._catch_up_state is None, \
        "it held the party after a full CATCH_UP_IDLE of nothing happening"


def test_a_written_off_step_is_written_down_once_not_fourteen_hundred_times():
    """Rev 3d026ada spent 25 of Phönix's log entries on one sentence,
    the last of them "1440 times in a row". The verdict does not change
    while the step does not, so repeating it says nothing new."""
    import time

    worker = _behind_party()
    laggard = _make_it_give_up(worker)
    for _ in range(50):
        worker._in_step_since = time.monotonic() - worker.DESYNC_GRACE - 1
        worker._check_in_step(worker.seats[0])

    said = [e for e in laggard.tel.questing
            if e["kind"] == "catch-up-written-off"]
    assert len(said) == 1, f"the same verdict was logged {len(said)} times"
    assert "times in a row" not in said[0]["detail"]


def test_every_wizards_export_carries_the_write_off():
    """Three files get uploaded and each has to explain why its wizard
    stopped waiting for the one that is behind."""
    import time

    worker = _behind_party()
    _make_it_give_up(worker)
    worker._in_step_since = time.monotonic() - worker.DESYNC_GRACE - 1
    worker._check_in_step(worker.seats[0])
    for seat in worker.seats:
        kinds = [e["kind"] for e in seat.tel.questing]
        assert "catch-up-written-off" in kinds, f"{seat.name} was not told"


# --------------------------------------- letting go of the game clients
#
# The operator's report: "Having to open and close wizard101 constantly
# sucks, would there be a way to make it able to unhook safely with a
# button so I can relaunch once I pull the newer github repo".
#
# Three things conspired. `stop()` only set a flag; `wait_for_combat`
# polls forever and never read it; so between fights Stop did nothing,
# the window got closed instead, the worker died where it stood, and
# the teardown that unhooks never ran.

def _parked(stops_after=None):
    """A seat whose `wait_for_combat` never returns on its own."""
    import asyncio

    worker, _read = _zoned_party(["Olde Town"])
    seat = worker.seats[0]
    entered = {"n": 0}

    class _Combat:
        async def wait_for_combat(self):
            entered["n"] += 1
            await asyncio.Event().wait()          # exactly what upstream does

    seat.combat = _Combat()
    return worker, seat, entered


def test_stop_is_answered_between_fights_not_only_during_one():
    """`CombatHandler.wait_for_combat` polls `in_combat` forever, so a
    parked fight loop never looked at `_stop`. Out of combat — which is
    most of a questing run — Stop did nothing at all."""
    import asyncio

    worker, seat, _entered = _parked()
    worker.STOP_POLL = 0.01

    async def drive():
        task = asyncio.ensure_future(worker._wait_for_combat(seat))
        await asyncio.sleep(0.05)
        assert not task.done(), "it did not wait for a duel at all"
        worker.stop()
        return await asyncio.wait_for(task, 2.0)

    assert asyncio.run(drive()) is False, \
        "a stop while waiting has to end the loop, not start a fight"


def test_a_duel_that_does_start_is_still_played():
    import asyncio

    worker, seat, _entered = _parked()
    worker.STOP_POLL = 0.01
    played = []

    class _Combat:
        async def wait_for_combat(self):
            played.append(True)

    seat.combat = _Combat()
    assert asyncio.run(worker._wait_for_combat(seat)) is True
    assert played, "the fight was skipped"


def test_what_the_wait_raises_still_reaches_the_fight_loop():
    """The loop swallows memory errors by name. Losing the exception
    inside the stop-poll would turn a broken read into a silent pass."""
    import asyncio

    worker, seat, _entered = _parked()
    worker.STOP_POLL = 0.01

    class _Combat:
        async def wait_for_combat(self):
            raise RuntimeError("the duel read failed")

    seat.combat = _Combat()
    with pytest.raises(RuntimeError, match="the duel read failed"):
        asyncio.run(worker._wait_for_combat(seat))


# -- and the unhook itself

class _Client:
    def __init__(self, fails=False):
        self.fails, self.closed = fails, False

    async def close(self):
        if self.fails:
            raise RuntimeError("the process went away")
        self.closed = True


class _Handler:
    def __init__(self, clients):
        self.clients = clients


def test_one_client_that_will_not_close_does_not_strand_the_others():
    """`ClientHandler.close` is a bare loop with no guard around each
    client, so the first one that throws leaves every client after it
    hooked — and a hooked client wizAi is not driving is exactly the
    state that forces Wizard101 to be restarted."""
    import asyncio

    worker, _read = _zoned_party(["Olde Town"] * 3)
    said = []
    worker.status.connect(said.append)
    clients = [_Client(), _Client(fails=True), _Client()]

    asyncio.run(worker._unhook(_Handler(clients)))

    assert clients[0].closed and clients[2].closed, \
        "one failure stranded the clients after it"
    assert said and "unhooked 2 of 3" in said[-1]
    assert "closed and reopened" in said[-1], \
        "the operator was not told which clients still need a restart"


def test_a_clean_unhook_says_the_game_can_stay_open():
    """The one question worth answering here. "disconnected" was all it
    used to say, and that is the fact that was never in doubt."""
    import asyncio

    worker, _read = _zoned_party(["Olde Town"] * 3)
    said = []
    worker.status.connect(said.append)
    clients = [_Client(), _Client(), _Client()]

    asyncio.run(worker._unhook(_Handler(clients)))

    assert all(c.closed for c in clients)
    assert "Wizard101 can stay open" in said[-1]


def test_a_client_with_no_seat_is_still_released():
    """It connected, so it is hooked. Whether wizAi found a seat for it
    has nothing to do with whether the game gets its memory back."""
    import asyncio

    worker, _read = _zoned_party(["Olde Town"])
    clients = [_Client(), _Client()]
    asyncio.run(worker._unhook(_Handler(clients)))
    assert all(c.closed for c in clients)


def test_unhooking_nothing_says_so_rather_than_claiming_success():
    import asyncio

    worker, _read = _zoned_party(["Olde Town"])
    said = []
    worker.status.connect(said.append)
    asyncio.run(worker._unhook(_Handler([])))
    assert "nothing was hooked" in said[-1]


def test_the_fight_loop_waits_through_the_stop_aware_wrapper():
    """A guard, because the direct call is the obvious thing to write
    and it is what stranded the hooks."""
    import inspect

    from deimos_bridge.gui.live import LiveWorker

    src = inspect.getsource(LiveWorker._fight_loop)
    assert "await self._wait_for_combat(seat)" in src
    assert "await seat.combat.wait_for_combat()" not in src, \
        "the fight loop parks where Stop cannot reach it again"


def test_the_teardown_unhooks_every_client_itself():
    import inspect

    from deimos_bridge.gui.live import LiveWorker

    src = inspect.getsource(LiveWorker._go)
    assert "await self._unhook(handler)" in src
    assert "await handler.close()" not in src, \
        "back to the unguarded loop that strands clients after a failure"


def test_the_window_has_an_unhook_button_that_waits_for_the_release(qapp):
    """The operator's actual ask: a button, so the game can be left
    running while wizAi is pulled and relaunched."""
    import inspect

    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    assert hasattr(win, "unhook_btn")
    assert not win.unhook_btn.isEnabled(), \
        "nothing is hooked before a run starts"
    # Pressing it with nothing connected says so rather than pretending
    win.on_unhook_live()
    assert "not connected" in win.status.text()

    src = inspect.getsource(MainWindow.closeEvent)
    assert "UNHOOK_GRACE_MS" in src, \
        "closing the window is the path that stranded hooks most often"
    assert MainWindow.UNHOOK_GRACE_MS >= 10000


# ------------------------- a hop cannot cross a zone; the script can
#
# Rev 1843e387. Phönix two quests behind on `Defeat Edo Nirini in Palace
# of Fire`, standing in KT_Hub. The catch-up paused the whole party and
# spent 116 seconds hopping Phönix nowhere, gave up, and wrote the step
# off permanently — and then the SCRIPT moved all three into
# KT_PalaceOfFire eight seconds later, which is the one place the
# catch-up would have worked.

def test_a_catch_up_is_not_started_for_an_objective_in_another_zone():
    """A quest teleport is a within-zone move. Across a boundary the
    marker reads in the other zone's coordinate space and comes back as
    six figures — 98,813 and 110,890 and 115,018 in that run."""
    import time

    worker = _behind_party()
    laggard = worker.seats[0]
    laggard.marker_away = 110890.0

    worker._in_step_since = time.monotonic() - worker.DESYNC_GRACE - 1
    worker._check_in_step(worker.seats[0])

    assert worker._catch_up_state is None, \
        "it paused the party for a hop that cannot reach the objective"
    kinds = [e["kind"] for e in laggard.tel.questing]
    assert "catch-up-out-of-zone" in kinds
    said = [e for e in laggard.tel.questing
            if e["kind"] == "catch-up-out-of-zone"][0]
    assert "110,890" in said["detail"]


def test_an_objective_in_this_zone_still_gets_its_catch_up():
    import time

    worker = _behind_party()
    worker.seats[0].marker_away = 640.0

    worker._in_step_since = time.monotonic() - worker.DESYNC_GRACE - 1
    worker._check_in_step(worker.seats[0])
    assert worker._catch_up_state is not None, \
        "a step the hop can actually reach was refused"


def test_an_unread_marker_does_not_block_the_catch_up():
    """No evidence is not evidence of another zone, and refusing on a
    failed read would turn one bad poll into a run with no catch-up."""
    import time

    worker = _behind_party()
    worker.seats[0].marker_away = None

    worker._in_step_since = time.monotonic() - worker.DESYNC_GRACE - 1
    worker._check_in_step(worker.seats[0])
    assert worker._catch_up_state is not None


def test_arriving_somewhere_new_earns_the_step_another_attempt():
    """The verdict was about a situation, not about the step. The party
    moving into the objective's zone IS the situation changing."""
    worker = _behind_party()
    laggard = _make_it_give_up(worker)
    assert worker._written_off([laggard])

    worker._forget_write_off(laggard)
    assert not worker._written_off([laggard]), \
        "a zone change did not clear a verdict that no longer applies"
    assert id(laggard) not in worker._wrote_off


def test_the_zone_change_that_clears_it_is_noticed_by_the_poll():
    """A guard: the clearing has to happen on the per-seat poll, not in
    `_check_together`, which needs two wizards and runs every six
    seconds. A solo laggard changing zone must still clear it."""
    import inspect

    from deimos_bridge.gui.live import LiveWorker

    src = inspect.getsource(LiveWorker._read_goal)
    assert "self._forget_write_off(seat)" in src
    assert "seat.marker_away" in src, \
        "the marker distance has to be read where the position already is"


# ----------------------------------------------- the solo pilot
#
# The operator's proposal, after five runs of group-coordination
# failures: "could we code a variant that runs a version of the scripts
# for only one character and then have the others help with combat?"
# Every catastrophic stall so far has been the script COORDINATING a
# party — friend teleports that miss, one instruction pointer for four
# wizards, desync, catch-ups. The presets are documented to quest solo
# when their account settings stay at placeholders ("If you're solo
# questing, you do not have to change any of the account information
# below" — TTS Arc 1, line 23). So: script one wizard, and keep the
# rest with it using the follow and the hivemind wizAi already has.

def test_solo_source_restores_a_configured_preset_to_placeholders():
    """The dialog fills real names in; the run needs them back OUT, or
    the script's own guards enable every multi-account branch. The
    placeholder is recovered from the guard comparisons — the
    assigned-AND-compared rule read backwards."""
    import os

    from deimos_bridge import scripts

    path = os.path.join(scripts.PRESET_DIR, "TTS Arc 1.txt")
    if not os.path.exists(path):
        pytest.skip("the TTS presets are not checked in")
    src = open(path, encoding="utf-8").read()
    filled, _ = scripts.configure(
        src, [("Konstantin", "Fire"), ("Sebastian", "Life")])
    assert "Konstantin" in filled, "the fixture never configured anything"

    back, reset = scripts.solo_source(filled)
    assert "Konstantin" not in back and "Sebastian" not in back
    assert "Main_Account" in reset and "Questee2" in reset
    # ...and the result reads as unconfigured again, which is the
    # script's own solo mode.
    assert [n for n, _v in scripts.unconfigured(back)]


def test_solo_source_leaves_a_fresh_script_alone():
    from deimos_bridge import scripts

    src = ('###deimos_expertmode\n'
           'var Questee2 = "QuestingAccountName"\n'
           'if NOT Questee2 = "QuestingAccountName" { }\n')
    back, reset = scripts.solo_source(src)
    assert back == src and reset == []


def test_make_runner_waives_the_client_requirement_only_for_solo():
    """`@clients: > 1` protects a party script from running over fewer
    wizards than it names. Solo mode runs it over ONE on purpose — the
    account guards make the p2..p4 sections unreachable — so the gate
    has to step aside there and nowhere else."""
    import inspect

    from deimos_bridge import scripts

    src = inspect.getsource(scripts.make_runner)
    assert "not solo and need > len(party)" in src


def test_in_solo_mode_the_script_drives_only_the_leader():
    worker = _behind_party()
    worker.solo_script = True
    worker.seats[0].runner = type("R", (), {"running": True})()
    assert worker._script_drives(worker.seats[0])
    assert not worker._script_drives(worker.seats[1])
    assert not worker._script_drives(worker.seats[2])

    # and in party mode the same runner drives everybody, as before
    worker.solo_script = False
    assert all(worker._script_drives(s) for s in worker.seats)


def test_in_solo_mode_every_other_wizard_follows_whatever_the_box_says():
    """Following IS the mode. A follower that stood still would watch
    the pilot walk away; one questing on its own would coordinate
    beautifully with nobody."""
    worker = _behind_party()
    worker.solo_script = True
    worker.follow_leader = False                    # box unticked
    assert not worker._follows(worker.seats[0])     # the pilot leads
    assert worker._follows(worker.seats[1])
    assert worker._follows(worker.seats[2])


def test_solo_mode_reports_no_desync_and_starts_no_catch_up():
    """The followers are behind BY DESIGN — they are combat support.
    Desync reports and catch-ups for them would re-create the exact
    churn the mode exists to remove."""
    import time

    worker = _behind_party()                        # one wizard 1 behind
    worker.solo_script = True
    worker._in_step_since = time.monotonic() - worker.DESYNC_GRACE - 1
    worker._check_in_step(worker.seats[0])
    assert worker._catch_up_state is None
    assert not [e for s in worker.seats for e in s.tel.questing
                if e["kind"] in ("quest-desync", "catch-up-started")]


def test_solo_mode_still_notices_the_pilot_losing_the_questline():
    """The one wizard whose lost questline stalls the run is the pilot.
    A follower's tracker drifting is expected — nothing is questing it."""
    import time

    from deimos_bridge import questlist

    worker = _party_on_quests(["Gather the Troops"] * 3)
    worker.solo_script = True
    side = questlist.Position(world="Krokotopia", name="Krokotopian Bundle",
                              how="by quest name", questline=None)
    on_main = questlist.position_of("Gather the Troops")

    # a FOLLOWER off the line: silence
    worker._places = lambda: [on_main, side, on_main]
    worker.seats[1].off_line_since = time.monotonic() - 300
    worker._check_on_questline()
    assert not [e for e in worker.seats[0].tel.questing
                if e["kind"] == "off-questline"]

    # the PILOT off the line: said
    worker._places = lambda: [side, on_main, on_main]
    worker.seats[0].off_line_since = time.monotonic() - 300
    worker._check_on_questline()
    assert [e for e in worker.seats[0].tel.questing
            if e["kind"] == "off-questline"]


def test_solo_setup_builds_the_vm_over_the_pilot_alone(monkeypatch):
    """One client, the solo-ized source, and the waiver — the three
    things that make it the script's own documented solo mode."""
    import asyncio

    from deimos_bridge import scripts

    worker = _behind_party()
    worker.solo_script = True
    worker.script = ('###deimos_expertmode\n'
                     '# @clients: > 1\n'
                     'var Questee2 = "PlaceholderName"\n'
                     'if NOT Questee2 = "PlaceholderName" { }\n')
    # the dialog configured it, as it would live
    worker.script = worker.script.replace(
        'var Questee2 = "PlaceholderName"', 'var Questee2 = "Sebastian"')
    for i, seat in enumerate(worker.seats):
        seat.client = object()

    built = {}

    def fake_runner(clients, source, solo=False):
        built.update(clients=clients, source=source, solo=solo)
        return type("R", (), {"running": True})()

    monkeypatch.setattr(scripts, "make_runner", fake_runner)
    asyncio.run(worker._setup_script(worker.seats[0].client, worker.seats[0]))

    assert built["solo"] is True
    assert built["clients"] == [worker.seats[worker.leader].client], \
        "the VM has to see the pilot's client and nobody else's"
    assert '"Sebastian"' not in built["source"], \
        "the configured name reached the VM — the multi-account branches are live"
    assert 'var Questee2 = "PlaceholderName"' in built["source"]


def test_party_mode_is_untouched_by_the_solo_plumbing(monkeypatch):
    import asyncio

    from deimos_bridge import scripts

    worker = _behind_party()
    worker.solo_script = False
    worker.script = "###deimos_expertmode\n"
    for seat in worker.seats:
        seat.client = object()
    built = {}

    def fake_runner(clients, source, solo=False):
        built.update(clients=clients, solo=solo)
        return type("R", (), {"running": True})()

    monkeypatch.setattr(scripts, "make_runner", fake_runner)
    asyncio.run(worker._setup_script(worker.seats[0].client, worker.seats[0]))
    assert built["solo"] is False
    assert len(built["clients"]) == 3


def test_the_window_offers_the_solo_pilot_and_passes_it_through(qapp):
    import inspect

    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    assert hasattr(win, "solo_script")
    assert not win.solo_script.isChecked(), "party scripting stays the default"
    src = inspect.getsource(MainWindow.on_start_live)
    assert "solo_script=self.solo_script.isChecked()" in src


# ------------------- followers on the SAME questline, not behind it
#
# The operator's correction to the first cut of the solo pilot: "that
# wont work for leveling 3 accounts at the same time ... they should
# all be on the same questline or else you're doing 3x the questing
# having to switch back and forth who's the leader." They can be:
# defeat credit is shared in the duel circle, and every other step is
# turned in at the exact spot the pilot has just walked the party to.

def _syncing_follower(monkeypatch, goal="Talk To Hetch Al'Dim",
                      marker=640.0, dialogue=False, battle=False):
    from deimos_bridge import questing

    worker = _behind_party()
    worker.solo_script = True
    seat = worker.seats[1]
    seat.goal = goal
    seat.marker_away = marker
    calls = {"hop": 0, "advance": 0, "follow": 0}

    async def in_battle(_c):
        return battle

    async def in_dialogue(_c):
        return dialogue

    async def advance(_c, **kw):
        calls["advance"] += 1
        return 2, ""

    async def hop(_c, **kw):
        calls["hop"] += 1
        return False

    async def follow(*_a, **_kw):
        calls["follow"] += 1

    monkeypatch.setattr(questing, "in_battle", in_battle)
    monkeypatch.setattr(questing, "in_dialogue", in_dialogue)
    monkeypatch.setattr(questing, "advance_dialogue", advance)
    monkeypatch.setattr(questing, "hop_once", hop)
    monkeypatch.setattr(worker, "_follow_step", follow)
    return worker, seat, calls


def test_a_follower_at_its_own_objective_takes_the_step(monkeypatch):
    import asyncio

    worker, seat, calls = _syncing_follower(monkeypatch)
    asyncio.run(worker._sync_follower(object(), seat))
    assert calls["hop"] == 1
    assert calls["follow"] == 0, \
        "the follow teleported it away mid-turn-in"


def test_a_follower_never_hops_to_its_own_defeat_marker(monkeypatch):
    """Defeat credit comes from the PILOT's circle. A follower hopping
    to its own defeat marker starts a fight alone, which is how Phönix
    died at rev 3d026ada."""
    import asyncio

    worker, seat, calls = _syncing_follower(
        monkeypatch, goal="Defeat Edo Nirini in Palace of Fire")
    asyncio.run(worker._sync_follower(object(), seat))
    assert calls["hop"] == 0
    assert calls["follow"] == 1


def test_a_follower_with_its_objective_elsewhere_just_follows(monkeypatch):
    import asyncio

    worker, seat, calls = _syncing_follower(monkeypatch, marker=110890.0)
    asyncio.run(worker._sync_follower(object(), seat))
    assert calls["hop"] == 0 and calls["follow"] == 1


def test_an_open_dialogue_is_finished_before_anything_else(monkeypatch):
    import asyncio

    worker, seat, calls = _syncing_follower(monkeypatch, dialogue=True)
    asyncio.run(worker._sync_follower(object(), seat))
    assert calls["advance"] == 1
    assert calls["hop"] == 0 and calls["follow"] == 0


def test_a_fighting_follower_is_left_to_the_policy(monkeypatch):
    import asyncio

    worker, seat, calls = _syncing_follower(monkeypatch, battle=True)
    asyncio.run(worker._sync_follower(object(), seat))
    assert calls == {"hop": 0, "advance": 0, "follow": 0}


def test_working_a_step_does_not_fall_through_to_the_follow(monkeypatch):
    """Between hop attempts the follower WAITS at its objective. Falling
    through to the follow teleports it back to the pilot with the step
    half-done, and the sync never finishes anything."""
    import asyncio

    worker, seat, calls = _syncing_follower(monkeypatch)
    asyncio.run(worker._sync_follower(object(), seat))    # hops, stamps gate
    asyncio.run(worker._sync_follower(object(), seat))    # gate closed
    assert calls["hop"] == 1
    assert calls["follow"] == 0


def test_a_step_that_will_not_turn_in_has_a_bounded_budget(monkeypatch):
    """A follower lost to one impossible step is worse than one step
    skipped — it stops fighting too."""
    import asyncio
    import time

    worker, seat, calls = _syncing_follower(monkeypatch)
    asyncio.run(worker._sync_follower(object(), seat))
    seat.sync_began = time.monotonic() - worker.SYNC_GIVE_UP - 1
    asyncio.run(worker._sync_follower(object(), seat))
    assert calls["follow"] == 1, "it stayed on a step it cannot finish"


def test_the_quest_row_says_what_each_wizard_is_doing():
    worker = _behind_party()
    worker.solo_script = True
    worker.seats[0].runner = type("R", (), {"running": True})()
    worker.seats[1].in_duel = True
    import time

    now = time.monotonic()
    pilot = worker._quest_row(worker.seats[0], "KT_Hub", now)
    fighter = worker._quest_row(worker.seats[1], "KT_Hub", now)
    third = worker._quest_row(worker.seats[2], "KT_Hub", now)
    assert pilot["doing"] == "pilot (script)"
    assert fighter["doing"] == "fighting"
    assert third["doing"] == "following + own steps"
    assert pilot["quest"] and pilot["line"].startswith("#")


def test_the_questing_tab_mode_and_checkboxes_agree_both_ways(qapp):
    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    assert hasattr(win, "quest_mode") and hasattr(win, "quest_table")

    # dropdown -> checkboxes
    win.quest_mode.setCurrentIndex(3)
    assert win.use_script.isChecked() and win.solo_script.isChecked()
    assert not win.auto_quest.isChecked()
    # checkboxes -> dropdown
    win.solo_script.setChecked(False)
    assert win.quest_mode.currentIndex() == 2
    win.use_script.setChecked(False)
    win.auto_quest.setChecked(True)
    assert win.quest_mode.currentIndex() == 1

    # the live feed renders a row
    win.on_seat_quest(0, {"name": "Konstantin", "doing": "pilot (script)",
                          "zone": "Krokotopia/KT_Hub", "quest": "Payback",
                          "goal": "Talk To X", "line": "#13 (Krokotopia)",
                          "marker": 98813.0, "idle": 400.0})
    assert win.quest_table.item(0, 0).text() == "Konstantin"
    assert win.quest_table.item(0, 6).text() == "other zone"
    assert "min" in win.quest_table.item(0, 7).text()

    import inspect
    src = inspect.getsource(MainWindow.on_start_live)
    assert "leader=self.leader_pick.currentIndex()" in src


# ------------------------- the delay between script actions was ours
#
# The operator: "is there a long delay between actions/ teleports for
# scripts? I feel like it teleports somewhere, waits a while then
# teleports to the next place, same with dialogue". Yes: the script ran
# in 0.5s bursts with a 0.5s sleep and a tickful of reads between them
# — a ~40% duty cycle, so every action the author priced at one second
# cost about two and a half.

def test_the_script_burst_is_seconds_not_a_fraction_of_one():
    from deimos_bridge.scripts import ScriptRunner

    assert ScriptRunner.SLICE >= 2.0, \
        "back to the 40% duty cycle the operator felt as 'a long delay'"


def _stepping_runner(done, running=True):
    class _R:
        stale = False
        failures = 0
        steps = 5
        last_error = ""

        def __init__(self):
            self.running = running

        async def run_for(self, should_stop=None):
            return done

    return _R()


def test_a_working_script_marks_its_seat_hot(monkeypatch):
    """Hot means the tick comes straight back for the next burst instead
    of sleeping its usual half second between them."""
    import asyncio

    worker = _behind_party()
    seat = worker.seats[0]
    seat.script_said = True
    seat.runner = _stepping_runner(done=120)
    asyncio.run(worker._script_step(seat))
    assert seat.script_hot is True


def test_a_script_with_nothing_to_run_cools_the_seat_down(monkeypatch):
    """A parked or finished script must not keep the tick spinning at
    ten times its normal rate for nothing."""
    import asyncio

    worker = _behind_party()
    seat = worker.seats[0]
    seat.script_said = True
    seat.script_hot = True
    seat.runner = _stepping_runner(done=0, running=True)
    asyncio.run(worker._script_step(seat))
    assert seat.script_hot is False


def test_the_service_tick_hurries_while_the_script_is_hot():
    import inspect

    from deimos_bridge.gui.live import LiveWorker

    src = inspect.getsource(LiveWorker._service_loop)
    assert 'script_hot' in src, \
        "the tick sleeps its full half second between script bursts again"


# ---------------------------- the script's pacing, without the editor
#
# "make those accessible in the gui" — SpeedDelay and DialogDelay are
# the preset's own knobs, and editing a 14,000-line file is not a knob.

def test_set_pacing_rewrites_the_presets_own_settings():
    import os
    import re

    from deimos_bridge import scripts

    path = os.path.join(scripts.PRESET_DIR, "TTS Arc 1.txt")
    if not os.path.exists(path):
        pytest.skip("the TTS presets are not checked in")
    src = open(path, encoding="utf-8").read()
    out, changed = scripts.set_pacing(src, step=0.5, dialog=1.5)
    assert ("SpeedDelay", "0.5") in changed
    assert ("DialogDelay", "1.5") in changed
    assert re.search(r"^var SpeedDelay = 0\.5\b", out, re.M)
    assert re.search(r"^var DialogDelay = 1\.5\b", out, re.M)
    # ...and only those lines: every `sleep $SpeedDelay` still reads
    # the variable, which is the point of rewriting the var.
    assert out.count("$SpeedDelay") == src.count("$SpeedDelay")


def test_none_means_the_scripts_own_pacing_stands():
    from deimos_bridge import scripts

    src = "###deimos_expertmode\nvar SpeedDelay = 1\n"
    out, changed = scripts.set_pacing(src)
    assert out == src and changed == []
    out, changed = scripts.set_pacing(src, dialog=0.5)
    assert out == src and changed == [], \
        "a variable the script does not have must not be invented"


def test_the_pacing_override_reaches_the_vm_in_both_modes(monkeypatch):
    import asyncio

    from deimos_bridge import scripts

    for solo in (False, True):
        worker = _behind_party()
        worker.solo_script = solo
        worker.script_step_delay = 0.5
        worker.script_dialog_delay = 1.5
        worker.script = ('###deimos_expertmode\n'
                         'var SpeedDelay = 1\n'
                         'var DialogDelay = 1\n')
        for seat in worker.seats:
            seat.client = object()
        built = {}

        def fake_runner(clients, source, solo=False):
            built["source"] = source
            return type("R", (), {"running": True})()

        monkeypatch.setattr(scripts, "make_runner", fake_runner)
        asyncio.run(worker._setup_script(worker.seats[0].client,
                                         worker.seats[0]))
        assert "var SpeedDelay = 0.5" in built["source"], f"solo={solo}"
        assert "var DialogDelay = 1.5" in built["source"], f"solo={solo}"


def test_the_questing_tab_offers_the_pacing_and_passes_it_through(qapp):
    import inspect

    from deimos_bridge.gui.app import MainWindow

    win = MainWindow(Telemetry())
    assert hasattr(win, "pace_override")
    assert not win.pace_override.isChecked(), \
        "the script's own pacing stays the default"
    assert not win.pace_step.isEnabled(), \
        "spinners enabled while the override is off read as active"
    win.pace_override.setChecked(True)
    assert win.pace_step.isEnabled() and win.pace_dialog.isEnabled()

    src = inspect.getsource(MainWindow.on_start_live)
    assert "script_step_delay" in src and "script_dialog_delay" in src
    assert "if self.pace_override.isChecked()" in src, \
        "the override must be None when the box is unticked"


# -------------------- the third kind of dialogue nothing detected
#
# "I think sometimes the bot doesnt realize it's in dialogue and gets
# stuck waiting too long." The modal message boxes — "Do you wish to be
# transported?" when teleporting to a friend in a dungeon, "your friend
# is busy", the zone-load retry — block the wizard exactly like a
# dialogue and live under `MessageBoxModalWindow`, not `wndDialogMain`.
# `close_menus` knew their buttons; `in_dialogue` and `advance_dialogue`
# did not, so a wizard parked at one read as "no dialogue open".

def _modal_root(visible=True):
    button = _Win("centerButton", visible=visible)
    return _Win("root", [
        _Win("MessageBoxModalWindow", [
            _Win("messageBoxBG", [
                _Win("messageBoxLayout", [
                    _Win("AdjustmentWindow", [
                        _Win("Layout", [button])])])])])]), button


def test_a_modal_prompt_counts_as_being_in_dialogue():
    import asyncio

    from deimos_bridge.questing import in_dialogue

    root, _ = _modal_root(visible=True)
    assert asyncio.run(in_dialogue(_QuestClient(root)))
    root, _ = _modal_root(visible=False)
    assert not asyncio.run(in_dialogue(_QuestClient(root)))


def test_advance_dialogue_clicks_a_modal_confirm():
    """The dungeon-teleport prompt is the follower's commonest wall:
    the follow lands it at the sigil, the game asks, nothing answered."""
    import asyncio

    from deimos_bridge.questing import advance_dialogue

    root, button = _modal_root(visible=True)
    client = _QuestClient(root)

    clicked = []
    real = client.mouse_handler.click_window

    async def click(window):
        clicked.append(window)
        button._visible = False         # the prompt is answered
        return await real(window)

    client.mouse_handler.click_window = click
    clicks, why = asyncio.run(advance_dialogue(client, settle=0))
    assert clicks == 1 and why == ""
    assert clicked == [button], "it clicked something other than the confirm"


def test_the_modal_paths_are_the_same_ones_close_menus_clicks():
    """One list. The bug was three functions with three private ideas of
    what counts as a blocking window."""
    from deimos_bridge.questing import CLOSE_MENU_PATHS, MODAL_BUTTON_PATHS

    for path in MODAL_BUTTON_PATHS:
        assert path in CLOSE_MENU_PATHS
