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

    python -m deimos_bridge.gui                 # standalone, live
    python -m deimos_bridge.gui --demo          # canned data, no game

`--demo` runs the whole window against `mock_client`, which is how it is
tested off Windows. Qt lives only in this package; `telemetry.py` is
plain Python so the interesting logic is testable headless.
"""

__all__ = ["app", "panels", "theme"]
