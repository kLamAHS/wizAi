"""Building a decklist without typing one.

Three ways in, because they suit different moments:

  * **Build one for me** — runs the project's own `deck_builder.build_deck`
    for the school and level. Zero input, and it is the deck wizAi would
    recommend anyway, so a live run starts comparable to the tables.
  * **Search and click** — the 8k-card table, filtered as you type,
    with a count per card.
  * **From the last fight** — every card actually seen in hand during
    the last live run. This is the closest thing to "read my deck off
    the game" that is honest: the client exposes the deck as *template
    ids*, and wizAi's card table has no template ids to match them
    against (`spells_full.json` records carry no id field), so a real
    deck read cannot be named. Card names in combat can, because
    `CombatCard.name()` returns one.
"""
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                             QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QMessageBox,
                             QPushButton, QSpinBox, QVBoxLayout)

from .theme import PALETTE, stylesheet

SCHOOL_KINDS = ("damage", "drain", "blade", "trap", "prism", "heal",
                "shield", "aura")


class BuildWorker(QThread):
    """`deck_builder.build_deck` off the UI thread."""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, cards, school, level):
        super().__init__()
        self.cards, self.school, self.level = cards, school, level

    def run(self):
        try:
            from deck_builder import build_deck
            from w101_sim import Boss

            deck, _w, _m, _table = build_deck(
                self.cards, self.school,
                Boss(name="deck screen", hp=3000, school="ice", dmg=140),
                n_candidates=24, top_k=2, capacity=16,
                level=self.level, seed=7, log=lambda *a, **k: None)
            self.done.emit(list(deck))
        except Exception as exc:
            self.failed.emit(f"build_deck raised {type(exc).__name__}: {exc}")


class DeckPicker(QDialog):
    """Returns a decklist (a list of card names, repeats allowed)."""

    def __init__(self, cards, school="fire", deck=None, seen=(), parent=None,
                 canonical=None):
        super().__init__(parent)
        self.setWindowTitle("Deck")
        self.resize(860, 620)
        self.cards = cards
        self.school = school
        #: names a player can actually put in a deck. The card table also
        #: holds event, boss and test variants -- Iceblade - EM,
        #: IcebladeBOSS01, Wand Ice_Boss001 -- which a mob can cast but
        #: you cannot, and which otherwise bury the real cards.
        self.canonical = set(canonical) if canonical else None
        self.seen = list(seen)
        #: name -> count
        self.chosen = {}
        for name in deck or []:
            self.chosen[name] = self.chosen.get(name, 0) + 1

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.build_btn = QPushButton("Build one for me")
        self.build_btn.setToolTip(
            "Runs deck_builder.build_deck for this school and level — the "
            "deck wizAi would recommend, so a live run starts comparable "
            "to the published tables.")
        self.build_btn.clicked.connect(self.on_build)
        top.addWidget(self.build_btn)

        top.addWidget(QLabel("level"))
        self.level = QSpinBox()
        self.level.setRange(1, 170)
        self.level.setValue(50)
        top.addWidget(self.level)

        self.seen_btn = QPushButton("From the last fight")
        self.seen_btn.setEnabled(bool(self.seen))
        self.seen_btn.setToolTip(
            "Every card seen in hand during the last live run."
            if self.seen else "No live run yet.")
        self.seen_btn.clicked.connect(self.on_from_seen)
        top.addWidget(self.seen_btn)

        top.addStretch()
        root.addLayout(top)

        cols = QHBoxLayout()

        left = QVBoxLayout()
        search_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("search the card table…")
        self.search.textChanged.connect(self.refilter)
        search_row.addWidget(self.search)
        self.kind = QComboBox()
        self.kind.addItems(("any kind",) + SCHOOL_KINDS)
        self.kind.currentTextChanged.connect(self.refilter)
        search_row.addWidget(self.kind)
        self.school_only = QComboBox()
        self.school_only.addItems([f"{school} + universal", "any school"])
        self.school_only.currentTextChanged.connect(self.refilter)
        search_row.addWidget(self.school_only)
        self.player_only = QComboBox()
        self.player_only.addItems(["player cards", "every variant"])
        self.player_only.setToolTip(
            "The table also holds event, boss and test variants that a mob "
            "can cast and you cannot.")
        self.player_only.currentTextChanged.connect(self.refilter)
        search_row.addWidget(self.player_only)
        left.addLayout(search_row)

        self.results = QListWidget()
        self.results.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.results.itemDoubleClicked.connect(lambda _: self.on_add())
        left.addWidget(self.results)

        add_row = QHBoxLayout()
        self.copies = QSpinBox()
        self.copies.setRange(1, 8)
        self.copies.setValue(2)
        add_row.addWidget(QLabel("copies"))
        add_row.addWidget(self.copies)
        add_btn = QPushButton("Add →")
        add_btn.clicked.connect(self.on_add)
        add_row.addWidget(add_btn)
        add_row.addStretch()
        left.addLayout(add_row)
        cols.addLayout(left, 3)

        right = QVBoxLayout()
        self.count_lab = QLabel("deck — 0 cards")
        right.addWidget(self.count_lab)
        self.deck_list = QListWidget()
        self.deck_list.itemDoubleClicked.connect(lambda _: self.on_remove())
        right.addWidget(self.deck_list)
        rm_row = QHBoxLayout()
        rm = QPushButton("Remove one")
        rm.clicked.connect(self.on_remove)
        rm_row.addWidget(rm)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.on_clear)
        rm_row.addWidget(clear)
        rm_row.addStretch()
        right.addLayout(rm_row)
        cols.addLayout(right, 2)

        root.addLayout(cols)

        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet(f"color: {PALETTE['warn']}")
        root.addWidget(self.warn)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.setStyleSheet(stylesheet())
        self.refilter()
        self.redraw_deck()

    # -- searching --------------------------------------------------------
    def refilter(self):
        text = self.search.text().strip().lower()
        kind = self.kind.currentText()
        own_only = self.school_only.currentIndex() == 0

        player_only = (self.player_only.currentIndex() == 0
                       and self.canonical is not None)

        hits = []
        for name, card in self.cards.items():
            if text and text not in name.lower():
                continue
            if player_only and name.split("@")[0] not in self.canonical:
                continue
            if kind != "any kind" and card.kind != kind:
                continue
            if own_only and card.school not in (self.school, "balance", None):
                continue
            hits.append((name, card))
            if len(hits) >= 400:            # the table is 8k; keep it snappy
                break

        hits.sort(key=lambda nc: (nc[1].kind, -nc[1].damage, nc[0]))
        self.results.clear()
        for name, card in hits:
            label = f"{name}   ·  {card.kind}  {card.school}  {card.pips} pips"
            if card.damage:
                label += f"  ~{card.damage:.0f} dmg"
            elif card.percent:
                label += f"  {card.percent * 100:+.0f}%"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.results.addItem(item)

    # -- editing ----------------------------------------------------------
    def on_add(self):
        item = self.results.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        self.chosen[name] = self.chosen.get(name, 0) + self.copies.value()
        self.redraw_deck()

    def on_remove(self):
        item = self.deck_list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        if self.chosen.get(name, 0) > 1:
            self.chosen[name] -= 1
        else:
            self.chosen.pop(name, None)
        self.redraw_deck()

    def on_clear(self):
        self.chosen.clear()
        self.redraw_deck()

    def on_from_seen(self):
        for name in self.seen:
            if name in self.cards:
                self.chosen[name] = self.chosen.get(name, 0) + 1
        self.redraw_deck()

    def on_build(self):
        """Run `deck_builder.build_deck` on a worker thread.

        It searches a deck space against a target boss, so it needs one,
        and at its default 150 candidates it is a minutes-long job.
        `BuildWorker` uses a plain dummy boss and a much smaller search:
        the point here is a sane starting decklist, not the deck screen's
        published result.
        """
        if getattr(self, "_builder", None) is not None and \
                self._builder.isRunning():
            return
        self.build_btn.setEnabled(False)
        self.build_btn.setText("building…")
        self._builder = BuildWorker(self.cards, self.school,
                                    self.level.value())
        self._builder.done.connect(self._built)
        self._builder.failed.connect(self._build_failed)
        self._builder.start()

    def _built(self, names):
        self.chosen.clear()
        for name in names:
            self.chosen[name] = self.chosen.get(name, 0) + 1
        self.redraw_deck()
        self.build_btn.setEnabled(True)
        self.build_btn.setText("Build one for me")

    def _build_failed(self, message):
        self.build_btn.setEnabled(True)
        self.build_btn.setText("Build one for me")
        QMessageBox.critical(
            self, "wizAi",
            f"{message}\n\nPick cards by hand instead — search on the left.")

    # -- output -----------------------------------------------------------
    def redraw_deck(self):
        self.deck_list.clear()
        total = 0
        for name in sorted(self.chosen):
            n = self.chosen[name]
            total += n
            item = QListWidgetItem(f"{n}x  {name}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.deck_list.addItem(item)
        self.count_lab.setText(f"deck — {total} cards")

        nukes = sum(n for name, n in self.chosen.items()
                    if self.cards[name].kind in ("damage", "drain"))
        problems = []
        if total and not nukes:
            problems.append("no damage cards — this deck cannot kill anything")
        if total > 64:
            problems.append("over the 64-card deck limit")
        self.warn.setText("  ·  ".join(problems))

    def decklist(self):
        out = []
        for name in sorted(self.chosen):
            out.extend([name] * self.chosen[name])
        return out


def pick_deck(parent, cards, school, deck=None, seen=(), canonical=None):
    """Show the picker; returns a decklist, or None if cancelled."""
    dialog = DeckPicker(cards, school, deck, seen, parent, canonical)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.decklist()
    return None
