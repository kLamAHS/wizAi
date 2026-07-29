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
                             QMessageBox, QProgressBar, QPushButton, QSpinBox,
                             QTabWidget, QVBoxLayout, QWidget)

from ..telemetry import Telemetry
from .panels import (BoardPanel, DecisionsPanel, ModelPanel, NamingPanel,
                     RunPanel, _label)
from .theme import PALETTE, stylesheet

SCHOOLS = ["fire", "ice", "storm", "myth", "life", "death", "balance"]
POLICIES = ["blade-stack(3)", "blade-stack(2)", "nuke-asap", "trained (Q)"]


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
        self.worker = None

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
        root.addWidget(tabs)

        self.status = _label("idle", PALETTE["muted"])
        root.addWidget(self.status)
        self.setStyleSheet(stylesheet())

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

        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(self.on_train)
        row.addWidget(self.train_btn)

        self.export_btn = QPushButton("Export run")
        self.export_btn.clicked.connect(self.on_export)
        row.addWidget(self.export_btn)
        row.addStretch()
        outer.addLayout(row)

        deck_row = QHBoxLayout()
        deck_row.addWidget(QLabel("deck"))
        self.deck = QLineEdit(
            "Fireblade,Fireblade,Fireblade,Sunbird,Sunbird,Sunbird,Tri Blade")
        self.deck.setToolTip(
            "Comma-separated card names. Required for a trained policy: the "
            "Q table is keyed on this deck's own blade and nuke positions, "
            "so a table trained for one decklist means nothing for another.")
        deck_row.addWidget(self.deck)
        outer.addLayout(deck_row)

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

    from data_full import load_spells_full
    from w101_sim import make_blade_stack

    from ..live_backend import WizAiBackend
    from ..mock_client import MockCard, MockCombat, MockEffect, MockMember

    cards = load_spells_full()
    deck = ["Fireblade"] * 3 + ["Sunbird"] * 4
    tel = Telemetry(policy_name="blade-stack(2)", school="fire", deck=deck)
    be = WizAiBackend.from_trained(school="fire", deck=deck, cards=cards,
                                   policy=make_blade_stack(2))
    be.on_decision = lambda d, r: tel.observe(d, r, sim=be._sim_for(r),
                                              cards=cards)
    tel.start_fight()

    hp, blades = 2400, []
    for rnd in range(1, 7):
        me = MockMember("Wizard", 3000 - rnd * 90, client=True, team_id=0,
                        normal_pips=min(rnd + 1, 7), hangings=list(blades))
        foe = MockMember("Krokopatra", hp, monster=True, team_id=1)
        combat = MockCombat([me, foe],
                            [MockCard("Fireblade"), MockCard("Sunbird"),
                             MockCard("Not A Real Spell")],
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
        for panel in (win.model, win.naming, win.runs):
            panel.refresh()
        if win.tel.rounds:
            win.board.render(win.tel.rounds[-1])
            for rec in win.tel.rounds:
                win.decisions.append(rec)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
