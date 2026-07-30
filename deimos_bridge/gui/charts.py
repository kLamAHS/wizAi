"""Small painted charts, with no plotting dependency.

Everything here is `QPainter` on a `QWidget`. That is a deliberate choice
rather than a shortcut: matplotlib and QtCharts are both large additions
to an install that has already been the hard part of this project twice,
and neither is needed for five chart forms that each draw a few dozen
marks. The whole toolkit is a few hundred lines and renders offscreen, so
every chart in the window is testable without a display.

The visual rules are the ones the data-viz method fixes, and they are
here rather than per-chart so nothing drifts:

  * thin marks -- 2px lines, bars capped at 24px, markers at least 8px
    across, area fills as a ~10% wash rather than a saturated block;
  * hairline solid gridlines one step off the surface, never dashed;
  * a 2px surface gap between touching fills and a 2px surface ring on
    overlapping markers, so separation is done by negative space rather
    than by drawing a border around data;
  * labels in text colours, never in the series colour -- identity comes
    from a swatch beside the text;
  * selective direct labels. A number on every point is chaos and goes
    unread, so the extremes get labelled and the axis carries the rest.

`theme.CHART` supplies the colours, and those were validated against this
app's own surface rather than picked by eye.
"""
import math

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QFont, QFontMetrics, QPainter,
                         QPainterPath, QPen)
from PyQt6.QtWidgets import QSizePolicy, QToolTip, QWidget

from .theme import CHART

#: Plot padding. Left is wide enough for a y-axis label and is widened
#: per chart when the row labels are card names; right leaves room for
#: the last x-axis label to sit inside the widget instead of being
#: cropped by its own edge; bottom is the x-axis band. Sizing a chart
#: without room for its axis is how a card ends up with a tiny nested
#: scrollbar.
PAD_L, PAD_R, PAD_T, PAD_B = 54, 44, 22, 34

BAR_MAX = 24        # never fill the band; the leftover is air
GAP = 2             # the surface gap, one width everywhere
LINE_W = 2
DOT_R = 4           # >= 8px across


def _c(name):
    return QColor(CHART[name])


def finite(value, default=0.0):
    """`value` as a real number, or `default`.

    Non-finite numbers are not hypothetical here. `w101_sim.evaluate`
    returns `float("nan")` for mean turns-to-kill when a checkpoint won
    no fights at all, and `Qt`'s coordinate calls take ints -- so a NaN
    reaches `int()` and raises `ValueError: cannot convert float NaN to
    integer`, `inf` raises `OverflowError`, and both do it inside
    `paintEvent`. Everything entering a chart passes through here.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _finite_pairs(points):
    """(kept, dropped) -- points whose x AND y are both real numbers."""
    kept, dropped = [], 0
    for x, y in points:
        fx, fy = finite(x, None), finite(y, None)
        if fx is None or fy is None:
            dropped += 1
        else:
            kept.append((fx, fy))
    return kept, dropped


def nice_ticks(lo, hi, count=3):
    """`count` ticks on clean 1/2/5 numbers spanning lo..hi.

    Axis labels carry the values that are not directly labelled, so they
    have to be readable at a glance -- 0 / 50 / 100, not 0 / 47.3 / 94.6.
    """
    lo, hi = finite(lo, 0.0), finite(hi, 1.0)
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < 1e-9:
        hi = lo + 1.0
    raw = (hi - lo) / max(1, count - 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), mag * 10)
    start = math.floor(lo / step) * step
    out, v = [], start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            out.append(round(v, 10))
        v += step
    return out or [lo, hi]


def _lerp(a, b, t):
    return QColor(int(a.red() + (b.red() - a.red()) * t),
                  int(a.green() + (b.green() - a.green()) * t),
                  int(a.blue() + (b.blue() - a.blue()) * t))


def ramp_color(t: float) -> QColor:
    """A colour off the sequential ramp for `t` in 0..1.

    One hue, light to dark, interpolated between validated steps rather
    than generated -- a hue rotation would be a rainbow ramp, which is
    the classic way to make magnitude unreadable.
    """
    steps = [QColor(s) for s in CHART["ramp"]]
    t = 0.0 if t != t else max(0.0, min(1.0, t))      # NaN -> 0
    pos = t * (len(steps) - 1)
    i = min(int(pos), len(steps) - 2)
    return _lerp(steps[i], steps[i + 1], pos - i)


def readable_on(fill: QColor) -> QColor:
    """Ink or white, whichever clears contrast on this fill.

    The one place a label may sit on a data colour is inside a filled
    mark, and then it has to be picked by the fill's luminance.
    """
    lum = (0.299 * fill.red() + 0.587 * fill.green()
           + 0.114 * fill.blue()) / 255.0
    return QColor("#0b0b0b") if lum > 0.55 else QColor("#ffffff")


class Chart(QWidget):
    """Surface, padding, axes and the empty state. Subclasses paint data.

    The empty state is part of the base on purpose. A panel that has
    nothing to show yet is the most common thing a person sees, and a
    blank rectangle reads as broken.
    """

    #: what to say before there is anything to draw
    empty_text = "nothing recorded yet"

    #: widened per chart when the row labels are card names rather than
    #: numbers -- a truncated "Sun..." identifies nothing
    pad_l = PAD_L
    #: reserved strip under the x-axis for a legend, when there is one
    legend_h = 0

    def __init__(self, title="", subtitle="", parent=None, height=190):
        super().__init__(parent)
        self.title = title
        self.subtitle = subtitle
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Preferred)
        self.setMouseTracking(True)
        self._hot = None          # index of the mark under the cursor

    # -- data -------------------------------------------------------------
    def has_data(self) -> bool:
        return False

    def empty_message(self) -> str:
        """What to say when there is nothing to draw.

        Overridable because "nothing to draw" and "nothing happened" are
        not the same claim, and a chart that conflates them is lying by
        omission. The case that made this necessary: a training run in
        which every checkpoint won no fights produces an all-NaN
        turns-to-kill curve, and the count of dropped samples was drawn
        only inside `paint_data`, which the empty state never reaches --
        so the one chart that knew the run had won nothing said "no
        training run yet" instead.
        """
        return self.empty_text

    def plot_rect(self) -> QRect:
        top = PAD_T + (16 if self.title else 0) + (14 if self.subtitle else 0)
        return QRect(self.pad_l, top,
                     max(1, self.width() - self.pad_l - PAD_R),
                     max(1, self.height() - top - PAD_B - self.legend_h))

    # -- painting ---------------------------------------------------------
    def paintEvent(self, _event):
        """Paint, and never take the process down doing it.

        PyQt6 does not merely print an unhandled exception raised inside
        a virtual like this one -- it calls `qFatal` and the process
        aborts. So one chart meeting a data shape nobody anticipated
        kills a live fight mid-duel, which is the one thing this window
        must not do. The same reasoning already applies to panels
        (`MainWindow.refresh_all` swallows their errors); painting was
        the hole in it, because `refresh()` is guarded and `paintEvent`
        is called by Qt.

        The failure is drawn rather than hidden: a chart silently showing
        nothing is indistinguishable from a chart with nothing to show.

        `end()` in a `finally` rather than leaving it to the garbage
        collector: a `QPainter` still open on a widget that is then
        destroyed segfaults the process outright -- no traceback, no
        qFatal line, nothing to read afterwards. Which is the same
        outcome this method exists to prevent, arriving by the one route
        the `except` cannot cover.
        """
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.fillRect(self.rect(), _c("surface"))
            self._paint_headings(p)
            if not self.has_data():
                p.setPen(_c("muted"))
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                           self.empty_message())
                return
            self.paint_data(p, self.plot_rect())
        except Exception as exc:
            self._paint_failed(p, exc)
        finally:
            p.end()

    def _paint_failed(self, p, exc):
        p.setPen(_c("bad"))
        p.drawText(self.rect().adjusted(8, 8, -8, -8),
                   Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                   f"this chart could not be drawn\n"
                   f"{type(exc).__name__}: {exc}")

    def _paint_headings(self, p):
        y = PAD_T - 6
        if self.title:
            f = QFont(self.font())
            f.setBold(True)
            p.setFont(f)
            p.setPen(_c("ink"))
            p.drawText(max(6, self.pad_l - 40), y, self.title)
            y += 15
            p.setFont(QFont(self.font()))
        if self.subtitle:
            p.setPen(_c("muted"))
            p.drawText(max(6, self.pad_l - 40), y, self.subtitle)

    def paint_data(self, p, r):      # pragma: no cover - overridden
        raise NotImplementedError

    # -- shared chrome ----------------------------------------------------
    def grid_y(self, p, r, ticks, fmt=lambda v: f"{v:g}"):
        """Hairline horizontal gridlines with y-axis labels.

        Solid, one step off the surface, drawn under the data. Dashed
        gridlines read as a threshold or a projection when they are just
        a grid.
        """
        p.setFont(QFont(self.font()))
        fm = QFontMetrics(self.font())
        for value, y in ticks:
            p.setPen(QPen(_c("grid"), 1))
            p.drawLine(r.left(), int(y), r.right(), int(y))
            p.setPen(_c("muted"))
            label = fmt(value)
            p.drawText(QRect(0, int(y) - 8, self.pad_l - 8, 16),
                       Qt.AlignmentFlag.AlignRight
                       | Qt.AlignmentFlag.AlignVCenter, label)
        p.setPen(QPen(_c("axis"), 1))
        p.drawLine(r.left(), r.bottom(), r.right(), r.bottom())

    def x_label(self, p, x, text, width=70):
        """Centred on `x`, but never allowed off the widget.

        A label cropped by its own chart edge is worse than no label --
        "24,00" reads as a different number, not as a truncation.
        """
        left = max(0, min(int(x) - width // 2, self.width() - width))
        p.setPen(_c("muted"))
        p.drawText(QRect(left, self.plot_rect().bottom() + 6, width, 16),
                   Qt.AlignmentFlag.AlignHCenter, text)

    def legend(self, p, entries):
        """Swatch + text, always present for two or more series.

        The text wears a text colour; the swatch beside it carries the
        identity. Colouring the label itself is how a light hue becomes
        unreadable on the surface.
        """
        x = self.plot_rect().left()
        y = self.height() - 6
        fm = QFontMetrics(self.font())
        for colour, text in entries:
            p.setBrush(QBrush(colour))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(x + 4, y - 4), 4, 4)
            p.setPen(_c("ink_dim"))
            p.drawText(int(x) + 13, int(y), text)
            x += 13 + fm.horizontalAdvance(text) + 16

    # -- hover ------------------------------------------------------------
    def hit_text(self, _index):      # pragma: no cover - overridden
        return ""

    def hits(self):                  # pragma: no cover - overridden
        """[(QRect, index)] -- hover targets, deliberately bigger than the
        marks they belong to."""
        return []

    def mouseMoveEvent(self, event):
        # Guarded for the same reason `paintEvent` is: an exception in a
        # Qt virtual aborts the process, and a hover is the last thing
        # that should be able to end a fight.
        try:
            self._hover(event)
        except Exception:
            pass

    def _hover(self, event):
        pos = event.position().toPoint()
        found = None
        for rect, index in self.hits():
            if rect.contains(pos):
                found = index
                break
        if found != self._hot:
            self._hot = found
            self.update()
        if found is None:
            QToolTip.hideText()
        else:
            QToolTip.showText(event.globalPosition().toPoint(),
                              self.hit_text(found), self)

    def leaveEvent(self, _event):
        if self._hot is not None:
            self._hot = None
            self.update()
        QToolTip.hideText()


class LineChart(Chart):
    """One series over time. Area wash, 2px line, endpoint labelled.

    Single series by design. Two measures of different scale belong in
    two charts -- a second y-axis invents a correlation out of the
    arbitrary alignment of the two scales, which is the single most
    misleading thing a chart can do.
    """

    empty_text = "no training run yet"

    def __init__(self, title="", subtitle="", fmt=lambda v: f"{v:g}",
                 parent=None, height=190):
        super().__init__(title, subtitle, parent, height)
        self.points = []          # [(x, y)]
        self.dropped = 0          # samples that had no real value
        #: what an unplottable sample MEANS, which is chart-specific. An
        #: undefined turns-to-kill is not missing data, it is a
        #: checkpoint that won no fights -- worth saying in those words.
        self.dropped_note = "{n} checkpoint(s) had no value to plot"
        self.fmt = fmt

    def set_points(self, points):
        """Real numbers only, and say how many were not.

        A checkpoint that won no fights has an undefined mean
        turns-to-kill, which `evaluate` reports as NaN. Dropping it is
        right -- there is no value to plot -- but dropping it *silently*
        would leave a curve with invisible holes in it.
        """
        self.points, self.dropped = _finite_pairs(points)
        self._hot = None
        self.update()

    def has_data(self):
        return len(self.points) >= 1

    def empty_message(self):
        """An all-NaN curve is a finding, not an absence.

        Every sample dropped means every checkpoint had no value -- for
        turns-to-kill, that a run of 100,000 episodes won no fights. The
        Learning tab was showing a populated kill-rate chart reading 0%
        directly above a turns-to-kill chart reading "no training run
        yet", which is the opposite of what happened.
        """
        if self.dropped and not self.points:
            return self.dropped_note.format(n=self.dropped)
        return self.empty_text

    def _scales(self, r):
        xs = [x for x, _ in self.points]
        ys = [y for _, y in self.points]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if y1 - y0 < 1e-9:
            y0, y1 = y0 - 1, y1 + 1
        y0 = min(y0, 0.0) if y0 > 0 and y0 < (y1 - y0) else y0
        # Headroom, so the top gridline label does not collide with the
        # subtitle and the endpoint dot is not clipped by the plot edge.
        y1 += (y1 - y0) * 0.12
        span_x = (x1 - x0) or 1.0

        def px(x):
            return r.left() + (x - x0) / span_x * r.width()

        def py(y):
            return r.bottom() - (y - y0) / (y1 - y0) * r.height()

        return px, py, (x0, x1, y0, y1)

    def paint_data(self, p, r):
        px, py, (x0, x1, y0, y1) = self._scales(r)
        self.grid_y(p, r, [(v, py(v)) for v in nice_ticks(y0, y1)], self.fmt)

        path = QPainterPath()
        for i, (x, y) in enumerate(self.points):
            pt = QPointF(px(x), py(y))
            path.moveTo(pt) if i == 0 else path.lineTo(pt)

        if len(self.points) > 1:
            fill = QPainterPath(path)
            fill.lineTo(QPointF(px(self.points[-1][0]), r.bottom()))
            fill.lineTo(QPointF(px(self.points[0][0]), r.bottom()))
            fill.closeSubpath()
            wash = _c("accent")
            wash.setAlphaF(0.10)          # a wash, never a saturated block
            p.fillPath(fill, QBrush(wash))

        p.setPen(QPen(_c("accent"), LINE_W, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # Endpoint only. A dot on every sample is noise at 40 snapshots.
        ex, ey = self.points[-1]
        centre = QPointF(px(ex), py(ey))
        p.setPen(QPen(_c("surface"), GAP))      # the surface ring
        p.setBrush(QBrush(_c("accent")))
        p.drawEllipse(centre, DOT_R, DOT_R)
        p.setPen(_c("ink"))
        p.drawText(QRectF(centre.x() - 60, centre.y() - 24, 56, 16),
                   Qt.AlignmentFlag.AlignRight
                   | Qt.AlignmentFlag.AlignVCenter, self.fmt(ey))

        self.x_label(p, r.left(), f"{x0:,.0f}")
        if x1 != x0:
            self.x_label(p, r.right(), f"{x1:,.0f}")
        if self.dropped:
            p.setPen(_c("warn"))
            p.drawText(QRect(self.pad_l, self.height() - 16,
                             self.width() - self.pad_l - PAD_R, 14),
                       Qt.AlignmentFlag.AlignRight,
                       self.dropped_note.format(n=self.dropped))


class Scatter(Chart):
    """Predicted against actual, with the y = x line it is judged by.

    A residual plot rather than two lines over time: the question is
    whether the model's number matches the game's, and that is a
    relationship between two measures, not a trend in either.
    """

    empty_text = "no damage observations yet"
    legend_h = 20

    def __init__(self, title="", subtitle="", parent=None, height=250):
        super().__init__(title, subtitle, parent, height)
        self.points = []          # [(predicted, actual, clean, label)]

    def set_points(self, points):
        self.points = [(finite(a), finite(b), clean, label)
                       for a, b, clean, label in points
                       if finite(a, None) is not None
                       and finite(b, None) is not None]
        self._hot = None
        self.update()

    def has_data(self):
        return bool(self.points)

    def _scale(self, r):
        top = max(max(a, b) for a, b, _, _ in self.points) or 1.0
        top *= 1.08

        def px(v):
            return r.left() + v / top * r.width()

        def py(v):
            return r.bottom() - v / top * r.height()

        return px, py, top

    def paint_data(self, p, r):
        px, py, top = self._scale(r)
        self.grid_y(p, r, [(v, py(v)) for v in nice_ticks(0, top)],
                    lambda v: f"{v:,.0f}")

        # The reference line. Perfect prediction lies on it, so distance
        # from it *is* the error -- no arithmetic needed by the reader.
        p.setPen(QPen(_c("axis"), 1))
        p.drawLine(int(px(0)), int(py(0)), int(px(top)), int(py(top)))
        # Labelled at the midpoint, inside the plot. At the top-right end
        # it ran off the widget and rendered as "perfect predictio".
        mid = 0.62
        p.setPen(_c("muted"))
        p.drawText(QRectF(px(top * mid) + 8, py(top * mid) - 2, 130, 16),
                   Qt.AlignmentFlag.AlignLeft, "perfect prediction")

        for i, (pred, actual, clean, _label) in enumerate(self.points):
            centre = QPointF(px(pred), py(actual))
            colour = _c("accent") if clean else _c("recede")
            p.setPen(QPen(_c("surface"), GAP))
            p.setBrush(QBrush(colour))
            radius = DOT_R + (2 if i == self._hot else 0)
            p.drawEllipse(centre, radius, radius)

        self.x_label(p, r.left(), "0")
        self.x_label(p, r.right(), f"{top:,.0f}")
        self.legend(p, [(_c("accent"), "clean"),
                        (_c("recede"), "confounded (DoT, AoE, overkill)")])

    def hits(self):
        if not self.has_data():
            return []
        px, py, _ = self._scale(self.plot_rect())
        out = []
        for i, (pred, actual, _clean, _label) in enumerate(self.points):
            c = QPoint(int(px(pred)), int(py(actual)))
            out.append((QRect(c.x() - 9, c.y() - 9, 18, 18), i))
        return out

    def hit_text(self, index):
        pred, actual, clean, label = self.points[index]
        err = actual - pred
        pct = f"  ({err / actual * 100:+.0f}%)" if actual else ""
        return (f"{label}\npredicted {pred:,.0f}\nactual {actual:,.0f}\n"
                f"error {err:+,.0f}{pct}"
                + ("" if clean else "\nconfounded — not counted"))


class Heatmap(Chart):
    """Rounds down, moves across, colour = how good the move scored.

    The literal decision matrix. One hue light-to-dark, because the cell
    value is a magnitude and a categorical scheme here would say the
    columns are unrelated identities when what matters is which cell is
    darkest. The chosen cell carries a ring rather than a different hue,
    so "what it picked" and "how good it was" stay on separate channels.
    """

    empty_text = "no decisions recorded yet — play a fight"

    #: emitted with the row index when a row is clicked
    row_picked = pyqtSignal(int)

    def __init__(self, title="", subtitle="", parent=None, height=250):
        super().__init__(title, subtitle, parent, height)
        self.rows = []        # [(row label, {col: (value, chosen, note)})]
        self.cols = []
        self.low_is_good = True

    def set_matrix(self, rows, cols, low_is_good=True):
        self.rows = [(label, {k: (finite(v), chosen, note)
                              for k, (v, chosen, note) in cells.items()
                              if finite(v, None) is not None})
                     for label, cells in rows]
        self.cols = list(cols)
        self.low_is_good = low_is_good
        self._hot = None
        self.update()

    def has_data(self):
        # Rows and columns are not enough: a row whose cells are all
        # outside the shown columns contributes nothing, and a grid of
        # only those rows made `min()` raise -- inside `paintEvent`,
        # which in PyQt6 aborts the process rather than printing.
        return bool(self.rows and self.cols and self._values())

    def _values(self):
        return [v for _label, cells in self.rows
                for (v, _chosen, _note) in cells.values()]

    def _cells(self):
        """[(rect, row index, col name, value, chosen, note)]"""
        r = self.plot_rect()
        out = []
        if not self.rows or not self.cols:
            return out
        ch = min(30, r.height() / max(1, len(self.rows)))
        cw = r.width() / len(self.cols)
        for ri, (_label, cells) in enumerate(self.rows):
            for ci, col in enumerate(self.cols):
                if col not in cells:
                    continue
                value, chosen, note = cells[col]
                rect = QRectF(r.left() + ci * cw + GAP / 2,
                              r.top() + ri * ch + GAP / 2,
                              cw - GAP, ch - GAP)
                out.append((rect, ri, col, value, chosen, note))
        return out

    def paint_data(self, p, r):
        values = self._values()
        lo, hi = min(values), max(values)
        span = (hi - lo) or 1.0
        fm = QFontMetrics(self.font())

        for rect, ri, col, value, chosen, _note in self._cells():
            t = (value - lo) / span
            if self.low_is_good:
                t = 1.0 - t                 # darkest = best
            fill = ramp_color(t)
            path = QPainterPath()
            path.addRoundedRect(rect, 3, 3)
            p.fillPath(path, QBrush(fill))
            if chosen:
                # A ring, not a hue: "picked" and "scored well" are two
                # different facts and must not share the colour channel.
                p.setPen(QPen(_c("ink"), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(path)
            text = f"{value:g}"
            if fm.horizontalAdvance(text) + 8 <= rect.width():
                p.setPen(readable_on(fill))
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        # Row labels, left of the plot.
        ch = min(30, r.height() / max(1, len(self.rows)))
        p.setPen(_c("muted"))
        for ri, (label, _cells) in enumerate(self.rows):
            p.drawText(QRectF(0, r.top() + ri * ch, self.pad_l - 8, ch),
                       Qt.AlignmentFlag.AlignRight
                       | Qt.AlignmentFlag.AlignVCenter, label)

        cw = r.width() / len(self.cols)
        for ci, col in enumerate(self.cols):
            name = col if fm.horizontalAdvance(col) < cw - 4 else \
                fm.elidedText(col, Qt.TextElideMode.ElideRight, int(cw - 4))
            self.x_label(p, r.left() + ci * cw + cw / 2, name, int(cw))

    def hits(self):
        return [(c[0].toRect(), i) for i, c in enumerate(self._cells())]

    def hit_text(self, index):
        cells = self._cells()
        if index >= len(cells):
            return ""
        _rect, _ri, col, value, chosen, note = cells[index]
        return f"{col}\n{note}" + ("\n— chosen" if chosen else "")

    def mousePressEvent(self, event):
        try:
            pos = event.position().toPoint()
            cells = self._cells()
            for rect, i in self.hits():
                if rect.contains(pos) and i < len(cells):
                    self.row_picked.emit(cells[i][1])
                    return
        except Exception:
            pass


class RankedBars(Chart):
    """One round's candidates, best first, with the chosen one emphasised.

    Emphasis rather than categorical: there is exactly one winner and the
    rest are context, so giving every bar its own hue would bury the
    point the chart exists to make.
    """

    empty_text = "pick a round above"
    pad_l = 150           # card names, not numbers

    def __init__(self, title="", subtitle="", parent=None, height=200,
                 lower_is_better=False):
        super().__init__(title, subtitle, parent, height)
        self.bars = []        # [(label, value, chosen, note)]
        self.unit = ""
        #: When the metric is "turns to clear", the raw value is a bad bar:
        #: every candidate lands within a turn or two of the others, so
        #: zero-based bars are all the same length and the chart shows
        #: nothing. Drawing the gap to the best move instead magnifies
        #: exactly the quantity the panel exists to show -- how close the
        #: decision was -- and the winner sits honestly at zero.
        self.lower_is_better = lower_is_better

    def set_bars(self, bars, unit=""):
        self.bars = [(label, finite(value), chosen, note)
                     for label, value, chosen, note in bars
                     if finite(value, None) is not None]
        self.unit = unit
        self._hot = None
        self.update()

    def has_data(self):
        return bool(self.bars)

    def _plotted(self):
        """(what to draw, what to print) per bar."""
        values = [v for _l, v, _c2, _n in self.bars]
        if self.lower_is_better:
            best = min(values)
            return [v - best for v in values], values
        return list(values), values

    def _geometry(self):
        r = self.plot_rect()
        n = max(1, len(self.bars))
        band = r.height() / n
        thick = min(BAR_MAX, band - GAP * 2)
        drawn, _ = self._plotted()
        top = max(drawn) or 1.0
        return r, band, thick, top

    def paint_data(self, p, r):
        _r, band, thick, top = self._geometry()
        drawn, printed = self._plotted()
        fm = QFontMetrics(self.font())
        p.setPen(QPen(_c("axis"), 1))
        p.drawLine(r.left(), r.top(), r.left(), r.bottom())

        for i, (label, value, chosen, _note) in enumerate(self.bars):
            y = r.top() + band * i + (band - thick) / 2
            width = max(2.0, drawn[i] / top * (r.width() - 70))
            colour = _c("accent") if chosen else _c("recede")
            path = QPainterPath()
            # Square at the baseline, rounded at the data end.
            path.addRoundedRect(QRectF(r.left(), y, width, thick), 4, 4)
            path.addRect(QRectF(r.left(), y, min(4.0, width), thick))
            p.fillPath(path, QBrush(colour))

            p.setPen(_c("ink") if chosen else _c("ink_dim"))
            p.drawText(QRectF(0, y, self.pad_l - 8, thick),
                       Qt.AlignmentFlag.AlignRight
                       | Qt.AlignmentFlag.AlignVCenter,
                       fm.elidedText(label, Qt.TextElideMode.ElideRight,
                                     self.pad_l - 10))
            # Value at the tip, outside the bar -- always fits there.
            text = f"{printed[i]:g}{self.unit}"
            if self.lower_is_better and chosen:
                text += "   ← best"
            p.setPen(_c("ink") if chosen else _c("ink_dim"))
            p.drawText(QRectF(r.left() + width + 6, y, 160, thick),
                       Qt.AlignmentFlag.AlignLeft
                       | Qt.AlignmentFlag.AlignVCenter, text)

    def hits(self):
        if not self.has_data():
            return []
        r, band, _thick, _top = self._geometry()
        return [(QRect(r.left(), int(r.top() + band * i), r.width(),
                       int(band)), i) for i in range(len(self.bars))]

    def hit_text(self, index):
        label, value, chosen, note = self.bars[index]
        return f"{label}\n{note}" + ("\n— chosen" if chosen else "")


class Meter(Chart):
    """One ratio against its limit. A number with a track, not a chart.

    A single value against a maximum is the case the form guide sends to
    a meter rather than a one-bar bar chart, and the number is the
    headline -- the track only gives it somewhere to sit.
    """

    empty_text = ""

    def __init__(self, label="", parent=None, height=74):
        super().__init__("", "", parent, height)
        self.label = label
        self.value = 0.0
        self.caption = ""
        self.status = "muted"

    def set_value(self, value, caption="", status="muted"):
        self.value = max(0.0, min(1.0, finite(value, 0.0)))
        self.caption = caption
        self.status = status
        self.update()

    def has_data(self):
        return True

    def paint_data(self, p, r):
        colour = _c(self.status) if self.status in CHART else _c("accent")
        f = QFont(self.font())
        f.setPointSize(max(14, f.pointSize() + 8))
        p.setFont(f)
        p.setPen(_c("ink"))
        p.drawText(QRect(0, 2, self.width(), 34),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"  {self.value * 100:.0f}%")
        p.setFont(QFont(self.font()))
        p.setPen(_c("muted"))
        p.drawText(QRect(0, 6, self.width() - 6, 28),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   self.label)

        track = QRectF(4, 40, self.width() - 8, 8)
        path = QPainterPath()
        path.addRoundedRect(track, 4, 4)
        p.fillPath(path, QBrush(_c("recede")))
        if self.value > 0:
            filled = QPainterPath()
            filled.addRoundedRect(
                QRectF(track.left(), track.top(),
                       max(8.0, track.width() * self.value), track.height()),
                4, 4)
            p.fillPath(filled, QBrush(colour))
        if self.caption:
            p.setPen(_c("ink_dim"))
            p.drawText(QRect(4, 52, self.width() - 8, 18),
                       Qt.AlignmentFlag.AlignLeft, self.caption)
