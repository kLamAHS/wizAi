"""Paste a Deimos bot script.

Compiles on demand rather than only at run time: a typo on line 40 of a
pasted script should be an error in this box, not a traceback twenty
seconds into a live run.
"""
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog,
                             QHBoxLayout, QLabel, QPlainTextEdit,
                             QPushButton, QVBoxLayout)

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
    def __init__(self, source="", parent=None):
        super().__init__(parent)
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


def edit_script(parent, source=""):
    dialog = ScriptDialog(source, parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.source()
    return None
