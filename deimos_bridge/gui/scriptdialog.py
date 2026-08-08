"""Paste a Deimos bot script.

Compiles on demand rather than only at run time: a typo on line 40 of a
pasted script should be an error in this box, not a traceback twenty
seconds into a live run.
"""
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                             QFileDialog, QHBoxLayout, QLabel,
                             QPlainTextEdit, QPushButton, QVBoxLayout)

from .theme import PALETTE, stylesheet

PLACEHOLDER = """\
# Paste a Deimos EXPERT-MODE script here -- one whose first line is
# ###deimos_expertmode. Deimos's older one-command-per-line format is a
# different language and does not run here; Check will say so.
#
# It runs alongside the combat policy and steps only while you are OUT
# of combat, so wizAi still fights every duel -- a `waitfor combat` line
# parks harmlessly until the policy has finished. That is the intended
# arrangement for a quester: scripts like TTS Arc 1 say "Combat should
# be enabled" because they expect the BOT to fight, not the script.
# Anything in the script that casts cards will not run.
#
# A multi-wizard script gets one VM over every hooked client, so p1..p4
# mean what they mean in Deimos. If its @clients header asks for more
# wizards than you have configured, loading it will say so rather than
# running it against wizards that are not there.
#
# While a script is running, auto-quest and follow-the-leader stand
# down: two things walking one wizard is how a run ends up in a doorway.
"""


class ScriptDialog(QDialog):
    def __init__(self, source="", parent=None, wizards=()):
        super().__init__(parent)
        #: [(name, school)] in seat order, so a preset can have the
        #: party's real names put into it. See `scripts.configure`.
        self._wizards = list(wizards)
        self.setWindowTitle("Bot script")
        self.resize(760, 560)

        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "Runs between fights. wizAi keeps the duel — the script steps "
            "only while out of combat."))

        self.edit = QPlainTextEdit()
        self.edit.setPlainText(source or "")
        self.edit.setPlaceholderText(PLACEHOLDER)
        self.edit.setFont(QFont("Consolas", 10))
        root.addWidget(self.edit)

        from ..scripts import presets

        self._presets = presets()
        if self._presets:
            picker = QHBoxLayout()
            picker.addWidget(QLabel("Preset:"))
            self.preset = QComboBox()
            self.preset.addItem("— pick a script —", None)
            for title, path in self._presets:
                self.preset.addItem(title, path)
            self.preset.currentIndexChanged.connect(self.on_preset)
            picker.addWidget(self.preset, 1)
            root.addLayout(picker)

        row = QHBoxLayout()
        load = QPushButton("Load from file…")
        load.clicked.connect(self.on_load)
        row.addWidget(load)
        check = QPushButton("Check")
        check.clicked.connect(self.on_check)
        row.addWidget(check)
        row.addStretch()
        root.addLayout(row)

        self.result = QLabel("")
        self.result.setWordWrap(True)
        root.addWidget(self.result)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.setStyleSheet(stylesheet())

    def on_preset(self, index):
        """Load a shipped script, with the party's names already in it.

        The filling is the point, not a convenience. A quester whose
        account names are still `"QuestingAccountName"` does not fail --
        it skips every friend-teleport it has, silently, and a party
        that cannot regroup stalls. One run lost forty minutes to that,
        and typing four names correctly into a 14,000-line file is not
        a thing to ask of somebody every session when wizAi already
        knows all four.
        """
        from ..scripts import ACCOUNT_VARS, configure, read_preset

        path = self.preset.itemData(index)
        if not path:
            return
        try:
            source = read_preset(path)
        except Exception as exc:
            self._say(f"could not read it: {exc}", ok=False)
            return
        note = ""
        if self._wizards:
            source, filled = configure(source, self._wizards)
            if filled:
                note = (" — filled in "
                        + ", ".join(f"{n}={v}" for n, v in filled[:4])
                        + (" …" if len(filled) > 4 else ""))
            # Said, not glossed over. Before a run the wizards' names
            # are not knowable -- the game will not give them outside
            # character select -- so only the schools get filled, and
            # the account names are the half that matters for
            # regrouping. Half a job quietly done is how the original
            # problem lasted forty minutes.
            missing = [f"p{i + 1}" for i, (n, _s) in
                       enumerate(self._wizards[:4]) if not n]
            if missing and any(v in ACCOUNT_VARS for v, _x in filled) is False:
                note += (f" — but {', '.join(missing)} had no name to use "
                         f"(wizAi learns those from the first duel), so "
                         f"open this again once the run has started, or "
                         f"type them in")
        self.edit.setPlainText(source)
        self._say(f"loaded {self.preset.itemText(index)}{note}", ok=True)

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a bot script", "", "All files (*)")
        if path:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    self.edit.setPlainText(f.read())
                self._say(f"loaded {path}", ok=True)
            except Exception as exc:
                self._say(f"could not read it: {exc}", ok=False)

    def on_check(self):
        from ..scripts import check
        source = self.edit.toPlainText()
        if not source.strip():
            self._say("nothing to check", ok=False)
            return
        ok, reason = check(source)
        self._say(reason, ok=ok)

    def _say(self, message, ok):
        self.result.setText(message)
        self.result.setStyleSheet(
            f"color: {PALETTE['good'] if ok else PALETTE['bad']}")

    def source(self):
        return self.edit.toPlainText()


def edit_script(parent, source="", wizards=()):
    dialog = ScriptDialog(source, parent, wizards)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.source()
    return None
