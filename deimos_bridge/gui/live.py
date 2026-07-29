"""Driving a real fight from the window.

Two problems to keep apart.

**asyncio inside Qt.** wizwalker is async top to bottom and Qt has its
own loop, so the fight runs on a `QThread` with its own event loop. The
window stays responsive and a hung memory read cannot freeze the UI.

**Thread affinity.** Qt widgets may only be touched from the GUI thread,
but `Telemetry.observe` is called from the worker on every planning
phase. So nothing here updates a widget: the worker emits signals, and
`MainWindow` does the drawing. `LiveWorker.round_done` is what the panels
ultimately refresh on, and because the signal crosses threads Qt queues
it onto the GUI thread automatically.
"""
import asyncio

from PyQt6.QtCore import QThread, pyqtSignal


class LiveWorker(QThread):
    """Connects to the client and plays fights until told to stop."""

    #: human-readable progress, straight to the status bar
    status = pyqtSignal(str)
    #: one planning phase completed; payload is the RoundRecord
    round_done = pyqtSignal(object)
    #: a fight ended
    fight_done = pyqtSignal(int)
    #: fatal, with a message already worth reading
    failed = pyqtSignal(str)
    #: the run stopped cleanly
    finished_ok = pyqtSignal()

    def __init__(self, telemetry, school, deck, policy_name, fights,
                 agent=None, auto_quest=False):
        super().__init__()
        self.tel = telemetry
        self.school = school
        self.deck = list(deck or [])
        self.policy_name = policy_name
        self.fights = fights
        self.agent = agent
        self.auto_quest = auto_quest
        self._stop = False
        #: one-shot questing requests from the GUI thread. A plain list
        #: rather than a queue: the GUI appends, the loop drains between
        #: fights, and CPython's list ops are atomic enough for that.
        self._requests = []
        self._client = None

    def stop(self):
        """Ask the loop to finish after the current fight."""
        self._stop = True

    def request(self, action):
        """Queue a questing action ('teleport' | 'dialogue').

        Called from the GUI thread. The loop performs it between fights,
        because the client cannot be driven from two places at once.
        """
        self._requests.append(action)

    async def _drain_requests(self):
        from .. import questing

        while self._requests:
            action = self._requests.pop(0)
            if self._client is None:
                continue
            if action == "teleport":
                ok = await questing.teleport_to_quest(self._client)
                self.status.emit("teleported" if ok else
                                 "no quest position to teleport to")
            elif action == "dialogue":
                n = await questing.advance_dialogue(self._client)
                self.status.emit(f"advanced {n} dialogue window(s)")

    # -- worker thread ----------------------------------------------------
    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._go())
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def _build_policy(self, cards):
        if self.policy_name.startswith("trained"):
            if self.agent is None:
                raise RuntimeError(
                    "No trained policy yet — press Train first, or pick "
                    "another policy.")
            return self.agent.policy()
        if self.policy_name.startswith("school-aware"):
            from ..policies import school_aware_blade_stack
            return school_aware_blade_stack(3)
        if self.policy_name.startswith("nuke"):
            from w101_sim import strat_nuke_asap
            return strat_nuke_asap
        from w101_sim import make_blade_stack
        n = 3
        if "(" in self.policy_name:
            try:
                n = int(self.policy_name.split("(")[1].split(")")[0])
            except (IndexError, ValueError):
                pass
        return make_blade_stack(n)

    async def _go(self):
        try:
            from wizwalker import ClientHandler
        except Exception as exc:
            raise RuntimeError(
                f"wizwalker did not import ({exc}). The live tab needs "
                "Windows and a running Wizard101 client; --demo works "
                "anywhere.") from exc

        from ..live_backend import WizAiBackend, make_combat_handler
        from ..live_state import build_catalog

        self.status.emit("loading the card table…")
        catalog = build_catalog()
        cards = catalog["cards"]
        policy = self._build_policy(cards)

        self.status.emit("looking for the game…")
        handler = ClientHandler()
        backend = None
        try:
            clients = handler.get_new_clients()
            if not clients:
                raise RuntimeError(
                    "No Wizard101 client found. wizwalker matches the window "
                    "class 'Wizard Graphical Client' — the game has to be "
                    "fully launched, not just the launcher.")
            client = clients[0]

            self.status.emit("activating hooks…")
            try:
                await client.activate_hooks()
            except Exception as exc:
                if "Pattern" in type(exc).__name__ or "Pattern" in str(exc):
                    raise RuntimeError(
                        "wizwalker could not install its hooks: the autobot "
                        "signature was not found in the running client.\n\n"
                        "Run  python -m deimos_bridge.diagnose_hooks  — it "
                        "tells you whether this is stale state in the process "
                        "(close the game completely) or a game patch that "
                        "outdates wizwalker."
                    ) from exc
                raise

            self.tel.policy_name = self.policy_name
            self.tel.school = self.school
            self.tel.deck = self.deck
            backend = WizAiBackend(
                policy=policy, cards=cards, school=self.school,
                decklist=self.deck, catalog=catalog,
                on_decision=self._on_decision)
            self.tel.resolver = backend.resolver
            self._backend = backend
            combat = make_combat_handler(client, backend)

            self._client = client
            self.status.emit(
                "connected — hunting for fights" if self.auto_quest
                else "connected — walk into a fight")
            fought = 0
            while not self._stop and (self.fights <= 0 or fought < self.fights):
                await self._drain_requests()
                if self._stop:
                    break
                if self.auto_quest:
                    from .. import questing
                    await questing.hop_to_next_fight(
                        client, on_status=self.status.emit)
                self.tel.start_fight()
                try:
                    # blocks until a duel starts, then plays it out
                    await combat.wait_for_combat()
                except Exception as exc:
                    name = type(exc).__name__
                    if not any(k in name for k in ("Memory", "ClientClosed",
                                                   "ReadingEnum", "Invalidated")):
                        raise
                fought += 1
                self.tel.end_fight()
                self.fight_done.emit(fought)
                self.status.emit(
                    f"fight {fought} over — waiting for the next"
                    if not self._stop else "stopping…")
            self.finished_ok.emit()
        finally:
            try:
                await handler.close()
            except Exception:
                pass
            self.status.emit("disconnected")

    #: the live backend, set once it exists. `_on_decision` needs it to
    #: build the throwaway `Sim` that produces a damage prediction.
    _backend = None

    def _on_decision(self, decision, read):
        """Runs on the worker thread: record, then signal. No widgets."""
        sim = None
        if self._backend is not None:
            try:
                sim = self._backend._sim_for(read)
            except Exception:
                sim = None          # a prediction is optional, the round is not
        rec = self.tel.observe(
            decision, read, sim=sim,
            cards=self._backend.cards if self._backend else None)
        self.round_done.emit(rec)
