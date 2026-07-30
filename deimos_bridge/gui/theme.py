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


#: Chart colours, kept apart from the widget palette above because they
#: answer to a different standard: they are read as *data*, so they have
#: to survive colour-vision deficiency and clear contrast against the
#: surface they are painted on. Every value below was checked with the
#: data-viz palette validator against this app's own surface (#1e1e1e,
#: dark) rather than chosen by eye.
CHART = {
    #: the plot surface -- the tab pane's own background
    "surface": "#1e1e1e",
    "grid": "#2c2c2a",        # hairline, one step off the surface
    "axis": "#383835",
    "ink": "#ffffff",
    "ink_dim": "#c3c2b7",
    "muted": "#898781",

    #: Categorical identity. Only the first three slots are used, and
    #: that is a rule rather than an accident: scatter and heatmap are
    #: all-pairs forms, and past three the reference palette cannot clear
    #: the all-pairs separation floors. Validated all-pairs on this
    #: surface -- worst CVD dE 9.4, worst normal-vision dE 20.9.
    "series": ("#3987e5", "#d95926", "#199e70"),

    #: Sequential magnitude: one hue, light -> dark. Validated monotone
    #: with visible steps (every adjacent dL >= 0.06) and a light end
    #: that still clears the surface.
    "ramp": ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
             "#184f95"),

    #: Emphasis: the one mark that matters, and the grey everything else
    #: recedes to. The most useful form here -- a decision has exactly
    #: one winner, and colouring all the candidates would bury it.
    "accent": "#3987e5",
    "recede": "#4a4a48",

    #: Status. Never used for a series, always shipped beside a number or
    #: a label -- these do not clear the categorical separation floors
    #: against each other, and are not meant to: the text is what carries
    #: the meaning, the colour only reinforces it.
    "good": "#0ca30c",
    "warn": "#fab219",
    "bad": "#d03b3b",
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
