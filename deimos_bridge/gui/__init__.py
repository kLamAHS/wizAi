"""A GUI built for the ML work rather than for bot-watching.

Deimos's own window answers an operator's questions: is it questing, is
it stuck, how long has it run. Training and evaluating a policy needs a
different set of answers, and getting them wrong is expensive in a way
that is easy to miss -- a run where the policy never saw half its hand
looks exactly like a run where the policy played badly.

So this is a separate window over `deimos_bridge.telemetry`, with panels
for the four things that actually decide whether a run was worth
anything: what the policy was shown, what it decided, whether the
simulator's damage predictions survived contact with a real mob, and how
the fight went.

    python -m deimos_bridge.gui                 # plays real fights
    python -m deimos_bridge.gui --demo          # canned data, no game

Press **Play live** and the window connects to the client, installs the
hooks and takes over combat, filling every panel as the fight happens.
`--demo` runs the same window against `mock_client`, which is how it is
tested off Windows.

The fight runs on a `QThread` with its own asyncio loop (`live.py`), so a
slow memory read cannot freeze the UI. Nothing on that thread touches a
widget: the worker emits signals, `MainWindow.refresh_all` does the
drawing, and Qt queues the hop across threads. Panels therefore do not
subscribe to the telemetry themselves -- a test asserts that, because a
panel that did would be updating Qt off the GUI thread and would fail
rarely and unreproducibly.

Qt lives only in this package; `telemetry.py` is plain Python, so every
judgement the GUI makes is testable headless.
"""

__all__ = ["app", "live", "panels", "theme"]
