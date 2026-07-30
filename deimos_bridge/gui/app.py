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

from ..hotkeys import DEFAULTS as HOTKEY_DEFAULTS, KEY_CHOICES as HOTKEY_CHOICES
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

    def __init__(self, cards, deck, school, episodes, player_hp=800,
                 boss_hp=1200, player_stats=None, n_enemies=1,
                 mob_hps=None):
        super().__init__()
        self.cards, self.deck = cards, deck
        self.school, self.episodes = school, episodes
        self.player_hp = player_hp
        self.boss_hp = boss_hp
        #: mobs on the training board. Has to match the live fight:
        #: `Featurizer.key` only carries its targeting tuple on a
        #: multi-enemy board, so 1v1 training and a two-mob fight produce
        #: keys of different length and share no state whatsoever.
        self.n_enemies = max(1, int(n_enemies))
        #: each mob's health, when a real board has been observed. Equal
        #: health is a degenerate board: the weakest mob is index 0 in
        #: every opening state, so the agent never sees the half of the
        #: target space a real fight starts in.
        self.mob_hps = [h for h in (mob_hps or []) if h > 0]
        #: the wizard's real gear, read off the client. Training without
        #: it prices every hit as though the wizard were naked, so the Q
        #: table is learned for a fight nobody is going to play.
        self.player_stats = dict(player_stats or {})

    def board_hps(self):
        """Health for each training mob.

        Prefers the health actually observed, mob by mob. Falling back to
        one number repeated is what produced a measured 0% coverage: with
        every mob on the same health the weakest is index 0 in every
        opening state, and a real board of 515 and 390 opens in the half
        that was never trained. So when no board has been observed, the
        mobs are spread deliberately rather than made identical.
        """
        if len(self.mob_hps) == self.n_enemies:
            return [max(1, int(h)) for h in self.mob_hps]
        # 100%, 80%, 60%, ... of the headline number.
        return [max(1, int(self.boss_hp * (1.0 - 0.2 * i)))
                for i in range(self.n_enemies)]

    def run(self):
        try:
            from rl_agent import train_agent
            from w101_sim import Boss
            # MORTAL, deliberately. train_agent defaults to
            # player_hp=10**9, and Featurizer.key writes -1 into the
            # health slot for an immortal fight and a real bucket
            # otherwise -- so a policy trained on the default shares no
            # state at all with a live wizard, its Q table reads zero
            # everywhere, and greedy falls through to PASS every turn.
            dmg = max(30, self.player_hp // 12)
            hps = self.board_hps()
            agent, sim = train_agent(
                self.cards, self.deck, self.school,
                Boss(name="training dummy", hp=hps[0], school="ice", dmg=dmg),
                enemies=[Boss(name=f"training minion {i}", hp=hp,
                              school="ice", dmg=dmg)
                         for i, hp in enumerate(hps[1:], 1)],
                episodes=self.episodes, player_hp=self.player_hp,
                player_stats=self.player_stats)
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
        #: each mob's health from the last real fight, so training does
        #: not use a degenerate board of identically-sized mobs
        self.observed_hps = []
        #: the wizard's gear, filled in from the client on connect.
        #: Training uses it, which is the whole point: a Q table learned
        #: for a naked wizard is solving a fight nobody will play.
        self.player_stats = {}

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
        self._update_policy_state()

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

        # Built before anything that can emit, because the policy combo's
        # change signal ends up in `_update_policy_state`, which draws to
        # it. Added to the layout further down, where it belongs on screen.
        self.policy_state = _label("no rounds played yet", PALETTE["muted"])
        self.policy_state.setWordWrap(True)

        row = QHBoxLayout()
        row.addWidget(QLabel("school"))
        self.school = QComboBox()
        self.school.addItems(SCHOOLS)
        row.addWidget(self.school)

        row.addWidget(QLabel("policy"))
        self.policy = QComboBox()
        self.policy.addItems(POLICIES)
        self.policy.setToolTip(
            "Changing this while a fight is running swaps the policy on "
            "the next round — no reconnect. The round in flight finishes "
            "under the policy that started it.")
        self.policy.currentTextChanged.connect(self.on_policy_changed)
        row.addWidget(self.policy)

        row.addWidget(QLabel("episodes"))
        self.episodes = QSpinBox()
        self.episodes.setRange(500, 200_000)
        self.episodes.setSingleStep(1000)
        self.episodes.setValue(8000)
        row.addWidget(self.episodes)

        row.addWidget(QLabel("my HP"))
        self.player_hp = QSpinBox()
        self.player_hp.setRange(100, 20000)
        self.player_hp.setSingleStep(100)
        self.player_hp.setValue(800)
        self.player_hp.setToolTip(
            "Your wizard's max health. Training uses it so the learned "
            "states match a live board — train immortal and the Q table "
            "shares no state with the real game at all, and the policy "
            "passes every turn. Filled in from the game on connect.")
        row.addWidget(self.player_hp)

        row.addWidget(QLabel("mob HP"))
        self.boss_hp = QSpinBox()
        self.boss_hp.setRange(100, 60000)
        self.boss_hp.setSingleStep(250)
        self.boss_hp.setValue(1200)
        self.boss_hp.setToolTip(
            "Health of the enemy to train against. The state key buckets "
            "it as health // 250, so training at 1200 and fighting a 500hp "
            "mob indexes different states for the same board. Filled in "
            "from the last fight.")
        row.addWidget(self.boss_hp)

        row.addWidget(QLabel("mobs"))
        self.n_enemies = QSpinBox()
        self.n_enemies.setRange(1, 4)
        self.n_enemies.setValue(1)
        self.n_enemies.setToolTip(
            "How many enemies to train against. This one is not a "
            "refinement — the state key carries a per-board targeting "
            "tuple ONLY when there is more than one enemy, so a table "
            "trained 1v1 and played against two mobs produces keys of a "
            "different length and matches nothing at all. Filled in from "
            "the last fight.")
        row.addWidget(self.n_enemies)

        row.addWidget(QLabel("fights"))
        self.fights = QSpinBox()
        self.fights.setRange(0, 999)
        self.fights.setValue(0)
        self.fights.setToolTip("0 = keep playing until you press Stop")
        row.addWidget(self.fights)

        self.train_btn = QPushButton("Train")
        self.train_btn.setToolTip(
            "Trains against the simulator. Works while a live run is "
            "connected — it runs at a lower thread priority so the fight "
            "keeps its timing — and the result is swapped in without "
            "dropping the connection.")
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

        key_row = QHBoxLayout()
        self.use_hotkeys = QCheckBox("Hotkeys")
        self.use_hotkeys.setChecked(True)
        self.use_hotkeys.setToolTip(
            "Do the two actions above without leaving the game. These are "
            "system-wide keys — they fire whatever window has focus, and "
            "while the run is connected the key is taken away from every "
            "other program, Wizard101 included. Pick keys the game does "
            "not use.")
        key_row.addWidget(self.use_hotkeys)

        self.hotkey_boxes = {}
        for action, label in (("teleport", "tp to quest"),
                              ("dialogue", "advance dialogue")):
            key_row.addWidget(QLabel(label))
            combo = QComboBox()
            combo.addItems(HOTKEY_CHOICES)
            combo.setCurrentText(HOTKEY_DEFAULTS[action])
            key_row.addWidget(combo)
            self.hotkey_boxes[action] = combo
        key_row.addStretch()
        outer.addLayout(key_row)

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

        # The answer to "is the model I picked actually playing?". A
        # trained policy that falls back on every board decides exactly
        # like the heuristic it falls back to, so without this the only
        # symptom is that the fight looks unremarkable.
        outer.addWidget(self.policy_state)

        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        outer.addWidget(self.train_progress)
        return box

    def _gear_line(self):
        """What the simulator thinks the wizard is wearing.

        Worth a line of its own: with no gear read, every hit is priced
        as though the wizard were naked, and the policy then optimises
        that fight rather than the one on screen.
        """
        st = self.player_stats
        if not st:
            return ("gear not read — hits are priced as if you wore none, "
                    "so the policy is solving a different fight")
        # The school the stats were *read for*, not whatever the dropdown
        # says now: gear is read once on connect, and re-keying it off a
        # combo the user can move afterwards would report 0%.
        damage = st.get("damage") or {}
        school, pct = next(iter(damage.items()), (self.school.currentText(),
                                                  0.0))
        bits = [f"{pct * 100:.0f}% {school} damage"]
        for key, label in (("pierce", "pierce"), ("accuracy", "accuracy"),
                           ("crit", "crit")):
            if st.get(key):
                bits.append(f"{st[key] * 100:.0f}% {label}")
        return "gear: " + ", ".join(bits) + " — training uses these"

    def _why_coverage_is_low(self):
        """Name the specific mismatch rather than saying "train more".

        "Train more episodes" is the wrong advice for every cause here,
        and it is expensive advice to follow before finding that out.
        The mismatches are checkable, so check them.
        """
        seen = self.tel.observed_board()
        if seen:
            n, hp = seen
            if n != self.n_enemies.value():
                return (f"cause: trained against {self.n_enemies.value()} "
                        f"mob(s), fighting {n}. The state key only carries "
                        f"its targeting tuple on a multi-enemy board, so "
                        f"these keys are different LENGTHS and can never "
                        f"match — no amount of training fixes it. Set "
                        f"mobs to {n} and retrain.")
            if abs(hp - self.boss_hp.value()) >= 250:
                return (f"cause: trained against {self.boss_hp.value():,} HP "
                        f"mobs, fighting ~{hp:,.0f}. The key buckets health "
                        f"as HP//250, so those are different states. Set "
                        f"mob HP to ~{hp:,.0f} and retrain.")
        return ("this deck's states are mostly unvisited; train more "
                "episodes, or with the deck you are actually holding")

    def _update_policy_state(self):
        """Say which policy is driving, and how often it really decided."""
        mix = self.tel.policy_mix()
        if not mix:
            self.policy_state.setText(
                "policy selected: " + self.policy.currentText() +
                " — no rounds played yet\n" + self._gear_line())
            self.policy_state.setStyleSheet(f"color: {PALETTE['muted']}")
            return
        total = sum(mix.values())
        parts = [f"{name} ×{n}" for name, n in mix.items()]
        text = (f"{total} round(s): " + "  ·  ".join(parts)
                + "\n" + self._gear_line())

        colour = PALETTE["muted"]
        live = self.live if (self.live and self.live.isRunning()) else None
        trained = getattr(live, "trained", None)
        if trained is not None and (trained.seen + trained.missed):
            cover = trained.coverage * 100
            colour = (PALETTE["good"] if cover > 66 else
                      PALETTE["warn"] if cover > 25 else PALETTE["bad"])
            text += (f"\nQ table decided {cover:.0f}% of the boards it was "
                     f"shown ({trained.missed} fell back to the heuristic)")
            if cover < 25:
                text += "\n" + self._why_coverage_is_low()
        self.policy_state.setText(text)
        self.policy_state.setStyleSheet(f"color: {colour}")

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
        fighting = self.live is not None and self.live.isRunning()
        self.status.setText(
            f"training {self.episodes.value():,} episodes on {len(deck)} cards…"
            + (" (the live fight keeps playing)" if fighting else ""))

        self.worker = TrainWorker(cards, deck, self.school.currentText(),
                                  self.episodes.value(),
                                  player_hp=self.player_hp.value(),
                                  boss_hp=self.boss_hp.value(),
                                  player_stats=self.player_stats,
                                  n_enemies=self.n_enemies.value(),
                                  mob_hps=self.observed_hps)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_trained)
        self.worker.failed.connect(self.on_train_failed)
        # Below the live worker, deliberately. Training is minutes of
        # solid CPU with no I/O to yield on; at equal priority it starves
        # the fight's event loop, and a planning phase that arrives late
        # is a turn played by the game's timeout rather than by wizAi.
        self.worker.start(QThread.Priority.LowPriority if fighting
                          else QThread.Priority.InheritPriority)

    def on_progress(self, ep, total, kill, ttk):
        self.status.setText(
            f"episode {ep:,}/{total:,} — kill {kill:.1f}%, TTK {ttk:.2f}")

    def on_trained(self, agent):
        self.agent = agent
        self.train_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        # Retrained while it was already driving: hand the new table over
        # in place. Without this, the fight would keep playing the table
        # that was current when Play live was pressed, and the retrain
        # would look like it had no effect.
        if self.live is not None and self.live.isRunning() and \
                self.policy.currentText().startswith("trained"):
            self.live.set_policy(self.policy.currentText(), agent=agent)
            self.status.setText(
                "trained — the running fight is now using the new table")
        else:
            self.status.setText("trained — policy ready to drive a live fight")
        self._update_policy_state()

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

    def hotkey_bindings(self):
        """{action: key}, or empty when hotkeys are off.

        Two actions may not share a key: `RegisterHotKey` would take the
        first and silently refuse the second, so the second action would
        appear bound and do nothing.
        """
        if not self.use_hotkeys.isChecked():
            return {}
        out, taken = {}, set()
        for action, box in self.hotkey_boxes.items():
            key = box.currentText()
            if key in taken:
                self.status.setText(
                    f"{key} is bound twice — only the first takes; give "
                    f"'{action}' a different key")
                continue
            taken.add(key)
            out[action] = key
        return out

    def on_teleport(self):
        self._quest_action("teleport", "teleporting to the quest marker…")

    def on_dialogue(self):
        self._quest_action("dialogue", "clicking through dialogue…")

    # -- live ------------------------------------------------------------
    def on_policy_changed(self, name):
        """Swap the policy on a running fight, or just remember it.

        The whole point of doing this here rather than at Play live: the
        deck picker's card list and the health the table was trained for
        both come from what a connected run observed, so disconnecting to
        change models throws away the inputs to the next decision.
        """
        if self.live is not None and self.live.isRunning():
            self.live.set_policy(name, agent=self.agent)
        self._update_policy_state()

    def on_gear_read(self, stats):
        """The wizard's real damage/accuracy/pierce, off the client."""
        self.player_stats = dict(stats or {})
        self._update_policy_state()

    def on_hp_read(self, hp):
        """The wizard's real max health, straight off the client."""
        if hp <= 0 or hp == self.player_hp.value():
            return
        self.player_hp.setMaximum(max(self.player_hp.maximum(), hp))
        self.player_hp.setValue(hp)
        self.status.setText(
            f"read your max health from the game: {hp:,} — training will "
            f"use it")

    def on_start_live(self):
        if self.live is not None and self.live.isRunning():
            return
        deck = self.decklist()
        policy = self.policy.currentText()
        if policy.startswith("trained") and self.agent is None:
            QMessageBox.warning(
                self, "wizAi",
                "No trained policy yet. Press Train first, or pick another "
                "policy from the list — you can switch to it mid-fight once "
                "it has trained, without reconnecting.")
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
                                       if self.use_script.isChecked() else ""),
                               hotkeys=self.hotkey_bindings())
        self.live.status.connect(self.on_live_status)
        self.live.round_done.connect(self.on_round)
        self.live.fight_done.connect(self.on_fight_done)
        self.live.failed.connect(self.on_live_failed)
        self.live.finished_ok.connect(self.on_live_finished)
        self.live.hp_read.connect(self.on_hp_read)
        self.live.gear_read.connect(self.on_gear_read)
        self.live.policy_changed.connect(self.on_policy_installed)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # Train stays live. Every input to a useful training run -- the
        # deck the picker learned from the last fight, the health the
        # client reported -- only exists once connected, so requiring a
        # disconnect to train meant training on guesses.
        self.live.start()

    def on_stop_live(self):
        if self.live is not None:
            self.live.stop()
            self.status.setText("stopping after this fight…")
        self.stop_btn.setEnabled(False)

    def on_live_status(self, message):
        self.status.setText(message)

    def adopt_observed_board(self):
        """Point the training board at the fight actually being fought.

        Left to hand-entry this is the single easiest way to produce a
        useless Q table, and it fails silently: train 1v1 against a
        1200hp dummy, walk into two 500hp mobs, and the learned states do
        not merely differ -- the keys are a different *length*, so
        coverage is exactly 0% and the trained policy is the fallback
        heuristic wearing its name.
        """
        seen = self.tel.observed_board()
        if not seen:
            return False
        n, hp = seen
        changed = (n != self.n_enemies.value()
                   or abs(hp - self.boss_hp.value()) >= 250)
        # The individual healths, not just the biggest: mobs that all
        # start equal never produce the opening state a real board has.
        self.observed_hps = self.tel.observed_mob_hps()
        self.n_enemies.setValue(min(max(int(n), 1), self.n_enemies.maximum()))
        self.boss_hp.setValue(min(max(int(hp), self.boss_hp.minimum()),
                                  self.boss_hp.maximum()))
        return changed

    def on_round(self, _rec):
        # Queued from the worker thread, so this runs on the GUI thread.
        self.refresh_all()
        self._update_policy_state()

    def on_fight_done(self, _n):
        if self.adopt_observed_board():
            self.status.setText(
                f"training board set from that fight: "
                f"{self.n_enemies.value()} mob(s) at ~"
                f"{self.boss_hp.value():,} HP — retrain to use it")
        self.refresh_all()
        self._update_policy_state()

    def on_policy_installed(self, name):
        """The worker says which policy it actually ended up with.

        It can differ from the dropdown -- picking a trained policy with
        nothing trained keeps the old one -- so the box is put back in
        step with reality rather than left showing a selection that did
        not take.
        """
        if name and name != self.policy.currentText():
            self.policy.blockSignals(True)      # not a fresh user choice
            self.policy.setCurrentText(name)
            self.policy.blockSignals(False)
        self._update_policy_state()

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
        self._update_policy_state()

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
                                   catalog=catalog,
                                   policy_name="blade-stack(2)")
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
