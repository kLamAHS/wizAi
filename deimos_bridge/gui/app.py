"""The window.

Holds a `Telemetry`, one tab per panel, and the controls for configuring
and starting a run. Training happens on a worker thread so the window
stays responsive -- a Q-learning run is minutes of solid CPU and freezing
the UI for it would make the progress readout pointless.

    python -m deimos_bridge.gui              # live (needs Windows + game)
    python -m deimos_bridge.gui --demo       # canned fight, runs anywhere
"""
import argparse
import sys

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QComboBox, QFileDialog, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QProgressBar, QPushButton, QSpinBox, QCheckBox,
                             QTabWidget, QVBoxLayout, QWidget)

from ..telemetry import Telemetry
from .deckpicker import pick_deck
from .live import LiveWorker
from .panels import (BoardPanel, DecisionsPanel, ModelPanel, NamingPanel,
                     RunPanel, _label)
from .theme import PALETTE, stylesheet

SCHOOLS = ["fire", "ice", "storm", "myth", "life", "death", "balance"]
POLICIES = ["ttk-lookahead", "school-aware", "blade-stack(3)",
            "blade-stack(2)", "nuke-asap", "trained (Q)"]


class TrainWorker(QThread):
    """Runs `rl_agent.train_agent` off the UI thread."""

    progress = pyqtSignal(int, int, float, float)   # ep, total, kill%, ttk
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, cards, deck, school, episodes):
        super().__init__()
        self.cards, self.deck = cards, deck
        self.school, self.episodes = school, episodes

    def run(self):
        try:
            from rl_agent import train_agent
            from w101_sim import Boss
            agent, sim = train_agent(
                self.cards, self.deck, self.school,
                Boss(name="training dummy", hp=3000, school="ice", dmg=150),
                episodes=self.episodes)
            from w101_sim import evaluate
            kill, ttk = evaluate(sim, agent.policy(), n=800)
            self.progress.emit(self.episodes, self.episodes, kill * 100, ttk)
            self.finished_ok.emit(agent)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self, telemetry=None):
        super().__init__()
        self.setWindowTitle("wizAi — live combat lab")
        self.resize(1180, 800)
        self.tel = telemetry or Telemetry()
        self.agent = None
        self.worker = None      # training
        self.live = None        # the live fight

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.addWidget(self._build_config())

        tabs = QTabWidget()
        self.board = BoardPanel(self.tel)
        self.decisions = DecisionsPanel(self.tel)
        self.model = ModelPanel(self.tel)
        self.naming = NamingPanel(self.tel)
        self.runs = RunPanel(self.tel)
        tabs.addTab(self.board, "Board")
        tabs.addTab(self.decisions, "Decisions")
        tabs.addTab(self.model, "Damage model")
        tabs.addTab(self.naming, "Naming")
        tabs.addTab(self.runs, "Runs")
        self.tabs = tabs
        root.addWidget(tabs)

        self.status = _label("idle — press Play live, or start with --demo",
                             PALETTE["muted"])
        root.addWidget(self.status)
        self.setStyleSheet(stylesheet())
        self.refresh_all()

    # -- panel updates all happen here, on the GUI thread -----------------
    def refresh_all(self):
        """Redraw every panel from the telemetry.

        Panels do not subscribe to the telemetry themselves: a live run
        fills it from a worker thread, and touching a widget from there
        is undefined behaviour in Qt. Everything funnels through this,
        called on the GUI thread via `LiveWorker`'s queued signals.
        """
        for panel in (self.board, self.decisions, self.model, self.naming,
                      self.runs):
            try:
                panel.refresh()
            except Exception:
                pass          # a panel must never take down a live fight

    def _build_config(self):
        box = QGroupBox("run")
        outer = QVBoxLayout(box)

        row = QHBoxLayout()
        row.addWidget(QLabel("school"))
        self.school = QComboBox()
        self.school.addItems(SCHOOLS)
        row.addWidget(self.school)

        row.addWidget(QLabel("policy"))
        self.policy = QComboBox()
        self.policy.addItems(POLICIES)
        row.addWidget(self.policy)

        row.addWidget(QLabel("episodes"))
        self.episodes = QSpinBox()
        self.episodes.setRange(500, 200_000)
        self.episodes.setSingleStep(1000)
        self.episodes.setValue(8000)
        row.addWidget(self.episodes)

        row.addWidget(QLabel("fights"))
        self.fights = QSpinBox()
        self.fights.setRange(0, 999)
        self.fights.setValue(0)
        self.fights.setToolTip("0 = keep playing until you press Stop")
        row.addWidget(self.fights)

        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(self.on_train)
        row.addWidget(self.train_btn)

        self.start_btn = QPushButton("Play live")
        self.start_btn.clicked.connect(self.on_start_live)
        row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop_live)
        row.addWidget(self.stop_btn)

        self.export_btn = QPushButton("Export run")
        self.export_btn.clicked.connect(self.on_export)
        row.addWidget(self.export_btn)
        row.addStretch()
        outer.addLayout(row)

        deck_row = QHBoxLayout()
        deck_row.addWidget(QLabel("deck"))
        self.deck = QLineEdit()
        self.deck.setPlaceholderText(
            "press Choose… — or paste comma-separated card names")
        self.deck.setToolTip(
            "Required for a trained policy: the Q table is keyed on this "
            "deck's own blade and nuke positions, so a table trained for "
            "one decklist means nothing for another.")
        deck_row.addWidget(self.deck)
        self.deck_btn = QPushButton("Choose…")
        self.deck_btn.clicked.connect(self.on_pick_deck)
        deck_row.addWidget(self.deck_btn)
        outer.addLayout(deck_row)

        quest_row = QHBoxLayout()
        self.auto_quest = QCheckBox("Auto-quest between fights")
        self.auto_quest.setToolTip(
            "Between fights, teleport to the quest marker and click through "
            "dialogue until a fight starts. Not Deimos's full questing — "
            "no navigation or sigils — but enough to keep feeding the "
            "policy fights without babysitting it.")
        quest_row.addWidget(self.auto_quest)

        self.auto_dialogue = QCheckBox("Auto-dialogue")
        self.auto_dialogue.setChecked(True)
        self.auto_dialogue.setToolTip(
            "Watch for dialogue and click through it as it appears, for the "
            "whole run — not just when you press the button. Paused during "
            "combat, so it cannot fight the card clicks.")
        quest_row.addWidget(self.auto_dialogue)

        self.collect_wisps = QCheckBox("Collect wisps")
        self.collect_wisps.setChecked(True)
        self.collect_wisps.setToolTip(
            "After each fight, walk over the health and mana wisps it "
            "dropped. Skips any sitting next to a mob, so topping up does "
            "not start a second fight.")
        quest_row.addWidget(self.collect_wisps)

        self.use_potions = QCheckBox("Use potions")
        self.use_potions.setChecked(True)
        self.use_potions.setToolTip(
            "Drink one when low on health or mana, using Deimos's threshold "
            "(under 55% health, or low mana). Never buys — refilling means "
            "a vendor trip that can strand the run.")
        quest_row.addWidget(self.use_potions)

        self.tp_btn = QPushButton("Teleport to quest")
        self.tp_btn.clicked.connect(self.on_teleport)
        quest_row.addWidget(self.tp_btn)

        self.dialogue_btn = QPushButton("Advance dialogue")
        self.dialogue_btn.clicked.connect(self.on_dialogue)
        quest_row.addWidget(self.dialogue_btn)
        quest_row.addStretch()
        outer.addLayout(quest_row)

        script_row = QHBoxLayout()
        self.use_script = QCheckBox("Run script")
        self.use_script.setToolTip(
            "Run a Deimos bot script (deimoslang) alongside the policy. It "
            "steps only while out of combat, so wizAi still fights.")
        script_row.addWidget(self.use_script)
        self.script_btn = QPushButton("Paste script…")
        self.script_btn.clicked.connect(self.on_edit_script)
        script_row.addWidget(self.script_btn)
        self.script_lab = _label("no script", PALETTE["muted"])
        script_row.addWidget(self.script_lab)
        script_row.addStretch()
        outer.addLayout(script_row)
        self.script_source = ""

        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        outer.addWidget(self.train_progress)
        return box

    # -- actions ----------------------------------------------------------
    def decklist(self):
        return [d.strip() for d in self.deck.text().split(",") if d.strip()]

    def on_train(self):
        if self.worker is not None and self.worker.isRunning():
            return
        deck = self.decklist()
        if not deck:
            QMessageBox.warning(self, "wizAi", "A decklist is required.")
            return
        try:
            from data_full import load_spells_full
            cards = load_spells_full()
        except Exception as exc:
            QMessageBox.critical(self, "wizAi", f"card table failed: {exc}")
            return
        missing = [d for d in deck if d not in cards]
        if missing:
            QMessageBox.warning(
                self, "wizAi",
                "These are not in the card table, so the policy could never "
                "play them:\n\n  " + "\n  ".join(missing))
            return

        self.train_btn.setEnabled(False)
        self.train_progress.setVisible(True)
        self.train_progress.setRange(0, 0)          # indeterminate
        self.status.setText(
            f"training {self.episodes.value():,} episodes on {len(deck)} cards…")

        self.worker = TrainWorker(cards, deck, self.school.currentText(),
                                  self.episodes.value())
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_trained)
        self.worker.failed.connect(self.on_train_failed)
        self.worker.start()

    def on_progress(self, ep, total, kill, ttk):
        self.status.setText(
            f"episode {ep:,}/{total:,} — kill {kill:.1f}%, TTK {ttk:.2f}")

    def on_trained(self, agent):
        self.agent = agent
        self.train_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        self.status.setText("trained — policy ready to drive a live fight")

    def on_train_failed(self, message):
        self.train_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        self.status.setText("training failed")
        QMessageBox.critical(self, "wizAi", message)

    # -- deck ------------------------------------------------------------
    def on_pick_deck(self):
        try:
            from ..live_state import build_catalog
            catalog = build_catalog()
            cards = catalog["cards"]
        except Exception as exc:
            QMessageBox.critical(self, "wizAi", f"card table failed: {exc}")
            return
        # Cards actually seen in hand during the last live run. This is
        # the only honest "read it off the game": the client exposes the
        # deck as template ids and wizAi's table carries none, so a real
        # deck read cannot be turned into names -- but a card in combat
        # can, because CombatCard.name() returns one.
        seen = sorted({name for rec in self.tel.rounds for name in rec.hand})
        chosen = pick_deck(self, cards, self.school.currentText(),
                           self.decklist(), seen,
                           canonical=catalog.get("canonical"))
        if chosen is not None:
            self.deck.setText(",".join(chosen))
            self.status.setText(f"deck set — {len(chosen)} cards")

    # -- scripts ---------------------------------------------------------
    def on_edit_script(self):
        from .scriptdialog import edit_script
        source = edit_script(self, self.script_source)
        if source is not None:
            self.script_source = source
            lines = len([ln for ln in source.splitlines() if ln.strip()])
            self.script_lab.setText(f"{lines} line(s) loaded" if lines
                                    else "no script")
            self.use_script.setChecked(bool(lines))

    # -- questing --------------------------------------------------------
    def _quest_action(self, coro_name, label):
        """Run one questing helper against the live client.

        Only available while a run is connected: these need the hooks,
        and the worker owns the client.
        """
        if self.live is None or not self.live.isRunning():
            QMessageBox.information(
                self, "wizAi",
                "Press Play live first — teleporting needs the hooks "
                "installed, and the live run owns the client connection.")
            return
        self.live.request(coro_name)
        self.status.setText(label)

    def on_teleport(self):
        self._quest_action("teleport", "teleporting to the quest marker…")

    def on_dialogue(self):
        self._quest_action("dialogue", "clicking through dialogue…")

    # -- live ------------------------------------------------------------
    def on_start_live(self):
        if self.live is not None and self.live.isRunning():
            return
        deck = self.decklist()
        policy = self.policy.currentText()
        if policy.startswith("trained") and self.agent is None:
            QMessageBox.warning(
                self, "wizAi",
                "No trained policy yet. Press Train first, or pick another "
                "policy from the list.")
            return
        if policy.startswith("trained") and not deck:
            QMessageBox.warning(self, "wizAi", "A trained policy needs its deck.")
            return

        self.live = LiveWorker(self.tel, self.school.currentText(), deck,
                               policy, self.fights.value(), agent=self.agent,
                               auto_quest=self.auto_quest.isChecked(),
                               auto_dialogue=self.auto_dialogue.isChecked(),
                               collect_wisps=self.collect_wisps.isChecked(),
                               use_potions=self.use_potions.isChecked(),
                               script=(self.script_source
                                       if self.use_script.isChecked() else ""))
        self.live.status.connect(self.on_live_status)
        self.live.round_done.connect(self.on_round)
        self.live.fight_done.connect(lambda n: self.refresh_all())
        self.live.failed.connect(self.on_live_failed)
        self.live.finished_ok.connect(self.on_live_finished)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.train_btn.setEnabled(False)
        self.live.start()

    def on_stop_live(self):
        if self.live is not None:
            self.live.stop()
            self.status.setText("stopping after this fight…")
        self.stop_btn.setEnabled(False)

    def on_live_status(self, message):
        self.status.setText(message)

    def on_round(self, _rec):
        # Queued from the worker thread, so this runs on the GUI thread.
        self.refresh_all()

    def on_live_failed(self, message):
        self._live_over()
        self.status.setText("live run failed")
        QMessageBox.critical(self, "wizAi", message)

    def on_live_finished(self):
        self._live_over()
        self.status.setText("live run finished")

    def _live_over(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.train_btn.setEnabled(True)
        self.refresh_all()

    def closeEvent(self, event):
        if self.live is not None and self.live.isRunning():
            self.live.stop()
            self.live.wait(3000)
        super().closeEvent(event)

    def on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export run", "results_live_run.json", "JSON (*.json)")
        if path:
            self.tel.to_json(path)
            self.status.setText(f"wrote {path}")


def demo_telemetry():
    """Drive the whole window from `mock_client`, so the GUI is
    exercisable -- and testable -- with no game and no Windows."""
    import asyncio

    from w101_sim import make_blade_stack

    from ..live_backend import WizAiBackend
    from ..live_state import build_catalog
    from ..mock_client import MockCard, MockCombat, MockEffect, MockMember

    catalog = build_catalog()
    cards = catalog["cards"]
    deck = ["Fireblade"] * 3 + ["Sunbird"] * 4
    tel = Telemetry(policy_name="blade-stack(2)", school="fire", deck=deck)
    be = WizAiBackend.from_trained(school="fire", deck=deck, cards=cards,
                                   policy=make_blade_stack(2),
                                   catalog=catalog)
    tel.resolver = be.resolver
    be.on_decision = lambda d, r: tel.observe(d, r, sim=be._sim_for(r),
                                              cards=cards)
    tel.start_fight()

    hp, blades = 2400, []
    for rnd in range(1, 7):
        me = MockMember("Wizard", 3000 - rnd * 90, client=True, team_id=0,
                        normal_pips=min(rnd + 1, 7), hangings=list(blades))
        foe = MockMember("Krokopatra", hp, monster=True, team_id=1)
        combat = MockCombat(
            [me, foe],
            [MockCard("Fireblade"), MockCard("Sunbird"),
             # one of each miss kind, so the Naming panel shows the split
             MockCard("Not A Real Spell"),      # no such card anywhere
             MockCard("Summon589244")],         # real, but decoder-skipped
            round_number=rnd)
        be.attach_combat(combat)
        decision = asyncio.run(be.decide())
        if decision.card_name == "Fireblade":
            blades.append(MockEffect("modify_outgoing_damage", 35, 2343174,
                                     1000 + rnd))
        elif decision.card_name:
            # a real mob resists a bit, so predicted and actual differ --
            # which is the point of the panel
            hp -= int(325 * (1 + 0.35 * len(blades)) * 0.88)
            blades.clear()
    tel.end_fight(won=hp <= 0)
    return tel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true",
                    help="populate from mock_client instead of a live client")
    args = ap.parse_args(argv)

    app = QApplication(sys.argv[:1])
    win = MainWindow(demo_telemetry() if args.demo else None)
    if args.demo:
        win.status.setText("demo data — press Play live to use the real game")
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
