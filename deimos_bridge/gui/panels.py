"""The panels. Each one is a view over `Telemetry` and owns no state.

Panels never reach into the game or the backend -- they read a
`Telemetry` and re-render on its events. That keeps every question about
"what should this show" answerable in `telemetry.py`, where it can be
tested without a display.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QListWidget, QProgressBar, QSizePolicy,
                             QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from .theme import PALETTE


def _label(text, color=None, bold=False, size=None):
    lab = QLabel(text)
    css = []
    if color:
        css.append(f"color: {color}")
    if bold:
        css.append("font-weight: 600")
    if size:
        css.append(f"font-size: {size}pt")
    if css:
        lab.setStyleSheet(";".join(css))
    return lab


def _table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    return t


def _cell(text, color=None):
    it = QTableWidgetItem(str(text))
    if color:
        it.setForeground(QColor(color))
    return it


class BoardPanel(QWidget):
    """What the policy was shown.

    First panel for a reason: the most expensive live-run failure is not
    a bad decision, it is a decision made against a board that was read
    wrong. A card that failed to resolve is simply absent from the hand,
    and nothing else in a run will tell you that happened.
    """

    def __init__(self, telemetry):
        super().__init__()
        self.tel = telemetry
        root = QVBoxLayout(self)

        self.round_lab = _label("waiting for a planning phase…",
                                PALETTE["muted"], size=12)
        root.addWidget(self.round_lab)

        cols = QHBoxLayout()

        me = QGroupBox("wizard")
        ml = QVBoxLayout(me)
        self.hp = _label("—", size=14, bold=True)
        self.pips = _label("—")
        self.charms = _label("—", PALETTE["muted"])
        for w in (self.hp, self.pips, self.charms):
            ml.addWidget(w)
        ml.addWidget(_label("hand", PALETTE["muted"]))
        self.hand = QListWidget()
        ml.addWidget(self.hand)
        cols.addWidget(me, 1)

        foes = QGroupBox("enemies")
        fl = QVBoxLayout(foes)
        self.enemies = _table(["enemy", "health", "charms", "wards"])
        fl.addWidget(self.enemies)
        self.unres = _label("", PALETTE["bad"])
        self.unres.setWordWrap(True)
        fl.addWidget(self.unres)
        cols.addWidget(foes, 2)

        root.addLayout(cols)
        telemetry.subscribe(self._on_event)

    def _on_event(self, event, payload):
        if event == "round":
            self.render(payload)

    def render(self, rec):
        self.round_lab.setText(f"fight {rec.fight}  ·  round {rec.round}")
        pct = (rec.player_hp / rec.player_max_hp * 100) if rec.player_max_hp else 0
        colour = (PALETTE["good"] if pct > 50 else
                  PALETTE["warn"] if pct > 25 else PALETTE["bad"])
        self.hp.setText(f"{rec.player_hp:,.0f} / {rec.player_max_hp:,.0f}")
        self.hp.setStyleSheet(f"color: {colour}; font-size: 14pt; font-weight: 600")
        self.pips.setText(f"{rec.norm_pips} pips  ·  {rec.pow_pips} power")
        self.charms.setText("charms: " + (", ".join(rec.player_charms) or "none"))

        self.hand.clear()
        for name in rec.hand:
            self.hand.addItem(name)

        self.enemies.setRowCount(len(rec.enemies))
        for i, e in enumerate(rec.enemies):
            self.enemies.setItem(i, 0, _cell(e.name))
            self.enemies.setItem(i, 1, _cell(f"{e.hp:,.0f} / {e.max_hp:,.0f}"))
            self.enemies.setItem(i, 2, _cell(", ".join(e.charms) or "—"))
            self.enemies.setItem(i, 3, _cell(", ".join(e.wards) or "—"))

        if rec.unresolved:
            self.unres.setText(
                "cards the policy could NOT see: " + ", ".join(rec.unresolved))
        else:
            self.unres.setText("")


class DecisionsPanel(QWidget):
    """Every decision, with what it passed over."""

    def __init__(self, telemetry):
        super().__init__()
        self.tel = telemetry
        root = QVBoxLayout(self)
        root.addWidget(_label(
            "Each row is one planning phase. 'passed over' is what else was "
            "castable — a policy that keeps declining a nuke it could afford "
            "is the shape of a state-featurisation bug.", PALETTE["muted"]))
        self.table = _table(["fight", "round", "cast", "target", "why",
                             "passed over"])
        root.addWidget(self.table)
        telemetry.subscribe(self._on_event)

    def _on_event(self, event, payload):
        if event == "round":
            self.append(payload)

    def append(self, rec):
        r = self.table.rowCount()
        self.table.insertRow(r)
        colour = PALETTE["warn"] if rec.passing else PALETTE["text"]
        self.table.setItem(r, 0, _cell(rec.fight))
        self.table.setItem(r, 1, _cell(rec.round))
        self.table.setItem(r, 2, _cell(rec.chosen or "pass", colour))
        self.table.setItem(r, 3, _cell(rec.target_name or "—"))
        self.table.setItem(r, 4, _cell(rec.reason, PALETTE["muted"]))
        self.table.setItem(r, 5, _cell(", ".join(rec.alternatives) or "—",
                                       PALETTE["muted"]))
        self.table.scrollToBottom()


class ResidualPlot(QWidget):
    """Predicted vs actual damage, per observation.

    Deliberately a scatter of both series rather than a plot of the
    error: seeing predicted and actual side by side makes a systematic
    bias (every prediction low by the same factor) look different from
    noise, and those two have completely different causes.
    """

    def __init__(self, telemetry):
        super().__init__()
        self.tel = telemetry
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def paintEvent(self, _):
        obs = self.tel.damage_observations(clean_only=False)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        pad = 26
        p.fillRect(0, 0, w, h, QColor(PALETTE["alt_bg"]))

        if not obs:
            p.setPen(QColor(PALETTE["muted"]))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "no damage observations yet")
            return

        vals = [v for r in obs for v in (r.predicted_damage, r.actual_damage)]
        top = max(vals) * 1.1 or 1.0

        p.setPen(QPen(QColor(PALETTE["bg"]), 1))
        for frac in (0.0, 0.5, 1.0):
            y = h - pad - frac * (h - 2 * pad)
            p.drawLine(pad, int(y), w - pad, int(y))

        n = len(obs)
        step = (w - 2 * pad) / max(n, 1)
        for i, rec in enumerate(obs):
            x = pad + step * (i + 0.5)
            for value, key in ((rec.predicted_damage, "predicted"),
                               (rec.actual_damage, "actual")):
                y = h - pad - (value / top) * (h - 2 * pad)
                p.setBrush(QColor(PALETTE[key]))
                p.setPen(QPen(QColor(PALETTE[key]), 1))
                p.drawEllipse(int(x) - 3, int(y) - 3, 6, 6)
            y1 = h - pad - (rec.predicted_damage / top) * (h - 2 * pad)
            y2 = h - pad - (rec.actual_damage / top) * (h - 2 * pad)
            p.setPen(QPen(QColor(PALETTE["muted"]), 1, Qt.PenStyle.DotLine))
            p.drawLine(int(x), int(y1), int(x), int(y2))

        p.setPen(QColor(PALETTE["predicted"]))
        p.drawText(pad, 16, "predicted")
        p.setPen(QColor(PALETTE["actual"]))
        p.drawText(pad + 80, 16, "actual")
        p.setPen(QColor(PALETTE["muted"]))
        p.drawText(w - pad - 60, 16, f"max {top:,.0f}")


class ModelPanel(QWidget):
    """Did the simulator predict what the game actually did?

    The reason this project needed Deimos at all. Every other number here
    can be produced by the simulator alone; this one cannot.
    """

    def __init__(self, telemetry):
        super().__init__()
        self.tel = telemetry
        root = QVBoxLayout(self)
        root.addWidget(_label(
            "Before each cast wizAi predicts the damage; the next round's "
            "real HP says what happened. Rounds where a DoT, an AoE or a "
            "kill could have muddied the delta are marked and excluded from "
            "the headline statistics.", PALETTE["muted"]))

        stats = QGroupBox("damage model vs the live game")
        sl = QHBoxLayout(stats)
        self.stat_labels = {}
        for key, title in (("n", "clean obs"), ("mean_abs_error", "mean abs err"),
                           ("rmse", "RMSE"), ("mean_pct_error", "mean % err"),
                           ("mean_error", "bias")):
            box = QVBoxLayout()
            value = _label("—", size=16, bold=True)
            box.addWidget(value)
            box.addWidget(_label(title, PALETTE["muted"]))
            sl.addLayout(box)
            self.stat_labels[key] = value
        root.addWidget(stats)

        self.plot = ResidualPlot(telemetry)
        root.addWidget(self.plot)

        self.table = _table(["round", "cast", "target", "predicted", "actual",
                             "error", "%", "clean"])
        root.addWidget(self.table)
        telemetry.subscribe(self._on_event)

    def _on_event(self, event, payload):
        if event == "round":
            self.refresh()

    def refresh(self):
        st = self.tel.error_stats()
        for key, lab in self.stat_labels.items():
            v = st.get(key)
            if v is None:
                lab.setText("—")
            elif key == "n":
                lab.setText(str(v))
            elif key == "mean_pct_error":
                lab.setText(f"{v:+.1f}%")
            else:
                lab.setText(f"{v:,.1f}")
        bias = st.get("mean_error")
        if bias is not None:
            colour = (PALETTE["good"] if abs(bias) < 15 else
                      PALETTE["warn"] if abs(bias) < 60 else PALETTE["bad"])
            self.stat_labels["mean_error"].setStyleSheet(
                f"color: {colour}; font-size: 16pt; font-weight: 600")

        rows = self.tel.damage_observations(clean_only=False)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            err = r.error
            colour = (PALETTE["good"] if abs(err) < 15 else
                      PALETTE["warn"] if abs(err) < 60 else PALETTE["bad"])
            self.table.setItem(i, 0, _cell(r.round))
            self.table.setItem(i, 1, _cell(r.chosen or "—"))
            self.table.setItem(i, 2, _cell(r.target_name or "—"))
            self.table.setItem(i, 3, _cell(f"{r.predicted_damage:,.0f}"))
            self.table.setItem(i, 4, _cell(f"{r.actual_damage:,.0f}"))
            self.table.setItem(i, 5, _cell(f"{err:+,.0f}", colour))
            self.table.setItem(i, 6, _cell(
                f"{r.pct_error:+.1f}%" if r.pct_error is not None else "—",
                colour))
            self.table.setItem(i, 7, _cell(
                "yes" if r.clean else "; ".join(r.confounds),
                PALETTE["text"] if r.clean else PALETTE["muted"]))
        self.plot.update()


class NamingPanel(QWidget):
    """Card names the resolver could not place.

    Its own panel because the failure is silent. wizAi's card table is
    keyed on exact name; an unresolved card is not an error, it is simply
    a card the policy never had.
    """

    def __init__(self, telemetry):
        super().__init__()
        self.tel = telemetry
        root = QVBoxLayout(self)
        self.headline = _label("—", size=13, bold=True)
        root.addWidget(self.headline)
        root.addWidget(_label(
            "Every name here is a spell the policy was never offered. Add it "
            "to deimos_bridge.live_state.ALIASES if the game and the card "
            "table simply disagree on spelling, or to the scrape if the card "
            "is genuinely missing.", PALETTE["muted"]))
        self.table = _table(["game name", "times seen"])
        root.addWidget(self.table)
        telemetry.subscribe(self._on_event)

    def _on_event(self, event, payload):
        if event == "round":
            self.refresh()

    def refresh(self):
        misses = self.tel.unresolved_names()
        if misses:
            self.headline.setText(f"{len(misses)} unresolved card names")
            self.headline.setStyleSheet(
                f"color: {PALETTE['bad']}; font-size: 13pt; font-weight: 600")
        else:
            self.headline.setText("every card name resolved")
            self.headline.setStyleSheet(
                f"color: {PALETTE['good']}; font-size: 13pt; font-weight: 600")
        self.table.setRowCount(len(misses))
        for i, (name, n) in enumerate(misses.items()):
            self.table.setItem(i, 0, _cell(name, PALETTE["bad"]))
            self.table.setItem(i, 1, _cell(n))


class RunPanel(QWidget):
    """Fights, and the run summary you would paste into a results table."""

    def __init__(self, telemetry):
        super().__init__()
        self.tel = telemetry
        root = QVBoxLayout(self)
        self.headline = _label("no fights yet", size=13, bold=True)
        root.addWidget(self.headline)
        self.table = _table(["fight", "rounds", "outcome", "damage dealt",
                             "passes", "unresolved"])
        root.addWidget(self.table)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)
        telemetry.subscribe(self._on_event)

    def _on_event(self, event, payload):
        self.refresh()

    def refresh(self):
        s = self.tel.summary()
        self.headline.setText(
            f"{s['policy'] or 'policy'} · {s['school']} · "
            f"{s['fights']} fights, {s['rounds']} rounds, "
            f"{s['wins']} won, {s['passes']} passes")
        fights = self.tel.fights
        self.table.setRowCount(len(fights))
        for i, f in enumerate(fights):
            outcome = ("—" if f.won is None else "won" if f.won else "lost")
            colour = (PALETTE["muted"] if f.won is None else
                      PALETTE["good"] if f.won else PALETTE["bad"])
            self.table.setItem(i, 0, _cell(f.index))
            self.table.setItem(i, 1, _cell(f.rounds))
            self.table.setItem(i, 2, _cell(outcome, colour))
            self.table.setItem(i, 3, _cell(f"{f.damage_dealt:,.0f}"))
            self.table.setItem(i, 4, _cell(f.passes))
            self.table.setItem(i, 5, _cell(f.unresolved))
