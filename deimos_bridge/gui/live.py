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
                 agent=None, auto_quest=False, auto_dialogue=True,
                 collect_wisps=True, use_potions=True):
        super().__init__()
        self.tel = telemetry
        self.school = school
        self.deck = list(deck or [])
        self.policy_name = policy_name
        self.fights = fights
        self.agent = agent
        self.auto_quest = auto_quest
        self.auto_dialogue = auto_dialogue
        #: between-fights upkeep. An unattended run dies by attrition
        #: long before it runs out of quests, and a policy that lost at
        #: 12% health has told you nothing about the policy.
        self.collect_wisps = collect_wisps
        self.use_potions = use_potions
        #: Deimos's own questing, if its requirements are installed. It
        #: has the navigation ours lacks -- navmap teleports, spiral
        #: doors, dungeon entry, NPC talking -- and composes cleanly
        #: because auto_quest_solo no-ops during combat.
        self.quester = None
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

    async def _service_loop(self, client):
        """Handle requests and auto-dialogue *while* the fight loop runs.

        This is the fix for "Teleport to quest says it is teleporting and
        nothing happens". Requests used to be drained at the top of the
        fight loop, but that loop spends nearly all its time blocked
        inside `wait_for_combat`, so a request queued while waiting sat
        there until a fight had started AND finished. Servicing them on a
        concurrent task means a button press acts within a second.

        Anything that clicks is skipped while a duel is on: the combat
        handler is clicking cards then, and two coroutines driving the
        mouse at once produce misclicks.
        """
        from .. import questing

        while not self._stop:
            try:
                if await questing.in_battle(client):
                    await asyncio.sleep(0.5)
                    continue

                while self._requests:
                    action = self._requests.pop(0)
                    if action == "teleport":
                        ok, reason = await questing.teleport_to_quest(client)
                        self.status.emit("teleported to the quest marker"
                                         if ok else reason)
                    elif action == "dialogue":
                        n = await questing.advance_dialogue(client)
                        self.status.emit(
                            f"advanced {n} dialogue window(s)" if n
                            else "no dialogue open")

                if self.auto_dialogue and self.quester is None \
                        and await questing.in_dialogue(client):
                    # Deimos's questing does its own dialogue handling;
                    # a second clicker would fight it.
                    n = await questing.advance_dialogue(client)
                    if n:
                        self.status.emit(f"auto-dialogue: {n} window(s)")

                if self.auto_quest and self.quester is not None:
                    await self._quest_step(client)

                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception:
                # The service task must outlive a bad read; the fight
                # loop is the thing that matters.
                await asyncio.sleep(1.0)

    async def _setup_questing(self, client):
        """Prefer Deimos's questing; fall back to ours if it will not import."""
        from .. import deimos_questing

        ok, reason = deimos_questing.available()
        if not ok:
            self.status.emit("using the light questing — " + reason.splitlines()[0])
            return
        try:
            self.quester = await deimos_questing.make_quester(client)
            self.status.emit("questing: using Deimos's navigator")
        except Exception as exc:
            self.quester = None
            self.status.emit(f"using the light questing ({type(exc).__name__})")

    async def _quest_step(self, client):
        """One tick of whichever questing is in play.

        Deimos's is a *step*, not a loop: its own driver is
        `while questing_status: sleep(1); auto_quest_solo(...)`, and
        running that here would take the fight loop's ownership away.
        """
        from .. import questing

        if self.quester is not None:
            ok = await self.quester.step()
            if not ok and self.quester.failures in (1, 10, 50):
                self.status.emit(
                    f"questing step failed ({self.quester.failures}x): "
                    f"{self.quester.last_error}")
            return
        await questing.hop_to_next_fight(
            client, on_status=self.status.emit,
            should_stop=lambda: self._stop)

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
        servicer = None
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

            if self.auto_quest:
                await self._setup_questing(client)

            self._client = client
            self.status.emit(
                "connected — hunting for fights" if self.auto_quest
                else "connected — walk into a fight")
            servicer = asyncio.ensure_future(self._service_loop(client))
            fought = 0
            while not self._stop and (self.fights <= 0 or fought < self.fights):
                if self.auto_quest and self.quester is None:
                    # The light questing is a blocking hunt, so it runs
                    # here. Deimos's is a step and runs on the service
                    # task instead, which keeps it ticking even while
                    # this loop is parked in wait_for_combat.
                    from .. import questing
                    await questing.hop_to_next_fight(
                        client, on_status=self.status.emit,
                        should_stop=lambda: self._stop)
                    if self._stop:
                        break
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

                if not self._stop and (self.collect_wisps or self.use_potions):
                    from .. import upkeep
                    try:
                        await upkeep.after_fight(
                            client, wisps=self.collect_wisps,
                            potions=self.use_potions,
                            on_status=self.status.emit)
                    except Exception:
                        pass      # upkeep is a nicety, never a blocker
                self.status.emit(
                    f"fight {fought} over — waiting for the next"
                    if not self._stop else "stopping…")
            self.finished_ok.emit()
        finally:
            if servicer is not None:
                servicer.cancel()
                try:
                    await servicer
                except BaseException:
                    pass
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
