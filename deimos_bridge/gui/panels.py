"""The panels. Each one is a view over `Telemetry` and owns no state.

Panels never reach into the game or the backend -- they read a
`Telemetry` and re-render on its events. That keeps every question about
"what should this show" answerable in `telemetry.py`, where it can be
tested without a display.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QListWidget, QProgressBar, QScrollArea,
                             QSizePolicy, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from .charts import Heatmap, LineChart, Meter, RankedBars, Scatter
from .theme import PALETTE


def scrollable(widget):
    """`widget` in a vertical scroll area, sized to its own content.

    Every tab goes through this. A panel that stacks a chart, a detail
    chart and a table is taller than a laptop window, and Qt's answer to
    "does not fit" is to squeeze every child until each is a few pixels
    tall -- which reads as a broken layout rather than as a full one.
    Scrolling keeps each piece at the size it needs to be legible.
    """
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)          # follow the width, scroll the height
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    # Horizontal as-needed, not off. `setWidgetResizable` makes the inner
    # widget follow the viewport width, but a child with a minimum width
    # of its own -- a seven-column table -- refuses to shrink, and the
    # overflow is then *clipped* rather than scrolled: the last heatmap
    # column simply vanished at a narrow window. A scrollbar that appears
    # is worse-looking and better behaved than content that disappears.
    area.setHorizontalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    return area


def _label(text, color=None, bold=False, size=None, wrap=True):
    """A label that wraps by default.

    Wrapping is not cosmetic here. A non-wrapping QLabel reports its
    whole sentence as its minimum width, so one paragraph of explanatory
    text sets a floor on the entire panel -- and inside a scroll area
    that floor becomes a horizontal scrollbar under charts that would
    otherwise have fitted.
    """
    lab = QLabel(text)
    lab.setWordWrap(wrap)
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


def _table(headers, weights=None):
    """A table whose columns get width in proportion to what they hold.

    `weights` because a blanket `Stretch` is content-blind: seven equal
    columns gave 'fight' and 'round' 165px each for a one-character
    value while 'policy' elided its reason away to
    'trained (Q) - fallback (...' and 'why' lost two thirds of its text.
    The one clause the reader needs was always the one cut.
    """
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    if weights:
        head = t.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        head.setStretchLastSection(True)
        t._weights = list(weights)
    else:
        t.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
    # Elide in the middle rather than at the end: the end is where the
    # reason lives. "trained (Q) ... in Q table)" keeps both halves of
    # the fact; "trained (Q) - fallback (..." keeps neither.
    t.setTextElideMode(Qt.TextElideMode.ElideMiddle)
    # Columns must be allowed to get narrow. Qt's default minimum section
    # width is the header text's, so a seven-column table sets a floor on
    # the whole panel's width -- which then forces a horizontal scrollbar
    # onto charts that would have fitted perfectly well.
    t.horizontalHeader().setMinimumSectionSize(46)
    t.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustIgnored)
    # Inside a scroll area an expanding table happily shrinks to a couple
    # of rows; a floor keeps it a table.
    t.setMinimumHeight(160)
    return t


def _apply_weights(table):
    """Hand out the viewport in proportion to `_weights`."""
    weights = getattr(table, "_weights", None)
    if not weights:
        return
    total = sum(weights) or 1
    width = max(1, table.viewport().width())
    for i, w in enumerate(weights):
        table.setColumnWidth(i, max(46, int(width * w / total)))


def _cell(text, color=None):
    it = QTableWidgetItem(str(text))
    if color:
        it.setForeground(QColor(color))
    # Free, and the only fix that is width-independent: whatever the
    # column does to the text, the whole string is one hover away.
    it.setToolTip(str(text))
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
        self.enemies = _table(["enemy", "health", "threat",
                               "charms", "wards"])
        fl.addWidget(self.enemies)
        self.unres = _label("", PALETTE["bad"])
        self.unres.setWordWrap(True)
        fl.addWidget(self.unres)
        cols.addWidget(foes, 2)

        root.addLayout(cols)

    def refresh(self):
        if not self.tel.rounds:
            return
        self.render(self.tel.rounds[-1])

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
            # Which offence model priced this mob -- "4+1p · casts
            # Banshee, Ghoul…" or "2p · ~136/round" -- because whether
            # the policy shields before the spike or races through the
            # drizzle hangs on exactly this, and it was invisible.
            self.enemies.setItem(i, 2, _cell(getattr(e, "threat", "")
                                             or "—"))
            self.enemies.setItem(i, 3, _cell(", ".join(e.charms) or "—"))
            self.enemies.setItem(i, 4, _cell(", ".join(e.wards) or "—"))

        if rec.unresolved:
            self.unres.setText(
                "cards the policy could NOT see: " + ", ".join(rec.unresolved))
        else:
            self.unres.setText("")


class DecisionsPanel(QWidget):
    """Every decision, with what it weighed and what it passed over.

    The matrix is the point of this panel and the table is the backup.
    A log that records only the chosen card can say *what* happened but
    never *how close it was* -- and "it played the trap" and "the trap
    beat the nuke by half a turn" are different facts, only the second of
    which can be argued with.
    """

    def __init__(self, telemetry):
        super().__init__()
        self.tel = telemetry
        root = QVBoxLayout(self)
        root.addWidget(_label(
            "Every move the lookahead weighed, round by round. Darker is "
            "better — the cell value is how many turns that move takes to "
            "clear the board. The ringed cell is the one it played; click a "
            "row to break it out below.", PALETTE["muted"]))

        self.matrix = Heatmap(
            "decision matrix",
            "rounds down, moves across · turns to clear the board")
        self.matrix.row_picked.connect(self.on_row)
        root.addWidget(self.matrix)

        self.detail = RankedBars(
            "", "how far behind the best move each option came · turns",
            height=180, lower_is_better=True)
        root.addWidget(self.detail)

        self.note = _label("", PALETTE["muted"])
        root.addWidget(self.note)

        root.addWidget(_label(
            "'passed over' is what else was castable — a policy that keeps "
            "declining a nuke it could afford is the shape of a "
            "state-featurisation bug. 'policy' names which one actually "
            "decided: a trained policy that says 'fallback' is playing the "
            "heuristic, not the table you trained.", PALETTE["muted"]))
        # Measured content widths at the default window: fight 20,
        # round 20, policy 273, cast 184, target 110, why 332,
        # passed over 229.
        self.table = _table(["fight", "round", "policy", "cast", "target",
                             "why", "passed over"],
                            weights=(1, 1, 4, 3, 2, 5, 4))
        root.addWidget(self.table)
        self._row = -1              # -1 = follow the latest round

    def on_row(self, index):
        self._row = index
        self._draw_detail()

    def _draw_detail(self):
        bars, title = self.tel.candidate_bars(self._row)
        self.detail.title = f"round detail — {title}" if title else "round detail"
        self.detail.set_bars(bars, unit=" turns")

    def resizeEvent(self, event):
        # Widths are proportions, so they have to be re-applied whenever
        # the viewport changes; Interactive columns keep whatever they
        # were last given otherwise.
        super().resizeEvent(event)
        _apply_weights(self.table)

    def refresh(self):
        rows, cols, dropped = self.tel.decision_matrix()
        self.matrix.set_matrix(rows, cols, low_is_good=True)
        _apply_weights(self.table)
        # Never truncate silently: a grid that quietly dropped six moves
        # reads as "these were all the options", which is a lie.
        self.note.setText(
            f"{dropped} rarer move(s) not shown — the table below has every one"
            if dropped else "")
        if self._row < 0 or self._row >= len(rows):
            self._row = len(rows) - 1
        self._draw_detail()

        self.table.setRowCount(0)
        for rec in self.tel.rounds:
            self.append(rec)

    def append(self, rec):
        r = self.table.rowCount()
        self.table.insertRow(r)
        colour = PALETTE["warn"] if rec.passing else PALETTE["text"]
        # A fallback row is tinted: it is the one case where the row
        # looks like a normal decision but was not made by the policy the
        # run is nominally measuring.
        policy = rec.policy or "—"
        self.table.setItem(r, 0, _cell(rec.fight))
        self.table.setItem(r, 1, _cell(rec.round))
        self.table.setItem(r, 2, _cell(
            policy, PALETTE["warn"] if "fallback" in policy else PALETTE["muted"]))
        self.table.setItem(r, 3, _cell(rec.chosen or "pass", colour))
        self.table.setItem(r, 4, _cell(rec.target_name or "—"))
        self.table.setItem(r, 5, _cell(rec.reason, PALETTE["muted"]))
        self.table.setItem(r, 6, _cell(", ".join(rec.alternatives) or "—",
                                       PALETTE["muted"]))
        self.table.scrollToBottom()


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

        self.plot = Scatter(
            "predicted vs actual",
            "x = predicted, y = actual · distance from the line is the error")
        root.addWidget(self.plot)

        self.table = _table(["round", "cast", "target", "predicted", "actual",
                             "error", "%", "clean"])
        root.addWidget(self.table)

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
        self.plot.set_points([
            (r.predicted_damage, r.actual_damage, r.clean,
             f"round {r.round} — {r.chosen or 'pass'}")
            for r in rows])


class LearningPanel(QWidget):
    """Whether the model learned anything, and whether it is being used.

    Two questions with two different answers, and conflating them is how
    a run gets misread. A table can be excellent in training and decide
    nothing at all live, which is exactly what a 0% coverage run looks
    like -- so the training curve and the live coverage sit side by side
    rather than one standing in for the other.
    """

    def __init__(self, telemetry):
        super().__init__()
        self.tel = telemetry
        root = QVBoxLayout(self)
        root.addWidget(_label(
            "Left: what training achieved against the simulator. Right: "
            "whether that table is deciding anything in the real fight. "
            "They are independent — a model can score well in training and "
            "still recognise none of the boards it is played on.",
            PALETTE["muted"]))

        cols = QHBoxLayout()

        learn = QGroupBox("training")
        ll = QVBoxLayout(learn)
        # Two charts rather than two y-axes. Kill rate is a percentage
        # and TTK is a turn count; putting them on one plot would align
        # two unrelated scales and invent a relationship between them.
        self.kill = LineChart("kill rate", "% of simulated fights won",
                              fmt=lambda v: f"{v:.0f}%", height=170)
        self.ttk = LineChart("turns to kill", "mean, over won fights",
                             fmt=lambda v: f"{v:.1f}", height=170)
        # `evaluate` reports an undefined mean as NaN, which is not
        # missing data -- it is a checkpoint that won nothing.
        self.ttk.dropped_note = "{n} checkpoint(s) won no fights"
        ll.addWidget(self.kill)
        ll.addWidget(self.ttk)
        cols.addWidget(learn, 3)

        live = QGroupBox("in the fight")
        vl = QVBoxLayout(live)
        self.coverage = Meter("of boards decided by the Q table")
        vl.addWidget(self.coverage)
        self.mix = RankedBars("who decided", "rounds played, per policy path",
                              height=150)
        self.mix.pad_l = 152
        vl.addWidget(self.mix)
        self.verdict = _label("", PALETTE["muted"])
        self.verdict.setWordWrap(True)
        vl.addWidget(self.verdict)
        vl.addStretch()
        cols.addWidget(live, 2)

        root.addLayout(cols)

    def refresh(self):
        self.kill.set_points(self.tel.training_curve())
        self.ttk.set_points(self.tel.ttk_curve())

        mix = self.tel.policy_mix()
        total = sum(mix.values())
        # Shortened to the part that differs. The full string repeats the
        # policy name on every row, which pushes the one word that
        # actually distinguishes them off the end of the label.
        self.mix.set_bars(
            [(name.split("—")[-1].strip() or name, n,
              "fallback" not in name, name)
             for name, n in mix.items()], unit="")

        decided = sum(n for name, n in mix.items()
                      if "fallback" not in name and "trained" in name)
        trained_rounds = sum(n for name, n in mix.items() if "trained" in name)
        if not trained_rounds:
            self.coverage.set_value(0.0, "no trained policy has played yet")
            self.verdict.setText("")
            return
        frac = decided / trained_rounds
        # Status colour beside a number, never instead of one -- these
        # three do not separate from each other under colour-vision
        # deficiency, and are not meant to.
        status = "good" if frac > 0.66 else "warn" if frac > 0.25 else "bad"
        self.coverage.set_value(
            frac, f"{decided} of {trained_rounds} rounds", status)
        self.verdict.setText(
            "The table is driving." if frac > 0.66 else
            "Mostly the fallback heuristic is driving — see the line under "
            "the controls for which mismatch it is.")


class NamingPanel(QWidget):
    """Cards the policy could not see, and what to do about each.

    Its own panel because the failure is silent: wizAi's card table is
    keyed on exact name, so an unresolved card is not an error, it is
    simply a card the policy never had. The headline is hand visibility
    rather than a miss count, because that is the number that decides
    whether the rest of the run means anything -- a policy shown 70% of
    its hand is not the policy you trained.

    Misses are split by cause, because they have different fixes. A name
    the game data knows but the decoder skipped is a gap in
    `data_full._map_effect` with a named cause. A name the game data has
    never heard of is a spelling problem, and belongs in `ALIASES`.
    """

    def __init__(self, telemetry):
        super().__init__()
        self.tel = telemetry
        root = QVBoxLayout(self)
        self.headline = _label("—", size=13, bold=True)
        root.addWidget(self.headline)
        self.detail = _label("", PALETTE["muted"])
        self.detail.setWordWrap(True)
        root.addWidget(self.detail)
        self.table = _table(["card", "times seen", "cause", "what to do"])
        root.addWidget(self.table)

    def _rows(self):
        """Miss rows, classified. Falls back to a flat list when no
        resolver catalog was built -- the classification is genuinely
        unavailable then, and inventing one would be worse than saying so."""
        seen = {}
        for rec in self.tel.rounds:
            for name in rec.hidden or rec.unresolved:
                seen[name] = seen.get(name, 0) + 1
        resolver = getattr(self.tel, "resolver", None)
        rows = []
        for name, n in sorted(seen.items(), key=lambda kv: -kv[1]):
            kind, detail = (resolver.classify(name) if resolver
                            else ("unknown", None))
            rows.append((name, n, kind, detail))
        return rows

    def refresh(self):
        vis = self.tel.hand_visibility()
        colour = (PALETTE["good"] if vis > 0.995 else
                  PALETTE["warn"] if vis > 0.9 else PALETTE["bad"])
        if vis > 0.995:
            self.headline.setText("the policy saw its whole hand")
        else:
            self.headline.setText(
                f"the policy saw {vis * 100:.0f}% of its hand")
        self.headline.setStyleSheet(
            f"color: {colour}; font-size: 13pt; font-weight: 600")

        rows = self._rows()
        if not rows:
            self.detail.setText("")
        elif vis <= 0.9:
            self.detail.setText(
                "This run is not measuring the policy you trained — it was "
                "planning against a hand it could only partly see. Fix the "
                "rows below before reading anything into the other tabs.")
            self.detail.setStyleSheet(f"color: {PALETTE['bad']}")
        else:
            self.detail.setText(
                "Each row is a spell the policy was never offered.")
            self.detail.setStyleSheet(f"color: {PALETTE['muted']}")

        self.table.setRowCount(len(rows))
        for i, (name, n, kind, detail) in enumerate(rows):
            if kind == "decoder":
                cause = detail or "skipped by the decoder"
                todo = ("close the gap in data_full._map_effect, "
                        "or accept the card is unmodellable")
                tint = PALETTE["warn"]
            else:
                cause = "not in the game data under this name"
                todo = "check spelling; add to live_state.ALIASES"
                tint = PALETTE["bad"]
            self.table.setItem(i, 0, _cell(name, tint))
            self.table.setItem(i, 1, _cell(n))
            self.table.setItem(i, 2, _cell(cause, PALETTE["muted"]))
            self.table.setItem(i, 3, _cell(todo, PALETTE["muted"]))


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
