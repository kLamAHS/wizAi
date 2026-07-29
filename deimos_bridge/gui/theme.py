"""Palette and stylesheet.

Matches `Deimos/src/settings_manager.py:DEFAULT_THEME` so the two windows
look like they belong to the same tool, but is self-contained -- pulling
in Deimos's theme machinery would drag `src.gui.helpers`, and with it
PyQt6 widgets and Windows-only paths, into a module that wants to stay
importable anywhere.
"""

PALETTE = {
    "bg": "#1e1e1e",
    "alt_bg": "#2d2d2d",
    "text": "#ffffff",
    "accent": "#4a019e",
    "stroke": "#e0e0e0",
    "muted": "#9a9a9a",
    # semantic, and used consistently: green = the model was right,
    # amber = it was off, red = something is actually broken.
    "good": "#4caf50",
    "warn": "#ffb300",
    "bad": "#ef5350",
    "predicted": "#42a5f5",
    "actual": "#ab47bc",
}


def stylesheet(p=None) -> str:
    p = p or PALETTE
    return f"""
    QWidget {{ background-color: {p['bg']}; color: {p['text']};
               font-size: 10pt; }}
    QTabWidget::pane {{ border: none; background-color: {p['bg']}; }}
    QTabBar::tab {{ background: {p['bg']}; color: {p['muted']};
                    padding: 8px 16px; border: none; }}
    QTabBar::tab:selected {{ color: {p['text']};
                             border-bottom: 2px solid {p['accent']}; }}
    QGroupBox {{ border: 1px solid {p['alt_bg']}; border-radius: 4px;
                 margin-top: 14px; padding-top: 8px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px;
                        padding: 0 4px; color: {p['muted']}; }}
    QTableWidget {{ background-color: {p['alt_bg']};
                    gridline-color: {p['bg']};
                    selection-background-color: {p['accent']}; }}
    QHeaderView::section {{ background-color: {p['bg']};
                            color: {p['muted']}; border: none;
                            padding: 6px; }}
    QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QListWidget {{
        background-color: {p['alt_bg']}; color: {p['text']};
        border: 1px solid {p['bg']}; border-radius: 3px; padding: 4px; }}
    QPushButton {{ background-color: {p['accent']}; color: white;
                   border: none; border-radius: 3px; padding: 7px 14px; }}
    QPushButton:disabled {{ background-color: {p['alt_bg']};
                            color: {p['muted']}; }}
    QProgressBar {{ background-color: {p['alt_bg']}; border: none;
                    border-radius: 3px; text-align: center; }}
    QProgressBar::chunk {{ background-color: {p['accent']};
                           border-radius: 3px; }}
    """
