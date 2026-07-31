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
import threading as _threading

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
    #: the wizard's real max health, once the hooks are up. Training has
    #: to use it or the learned states share no health bucket with a live
    #: board, and typing it in by hand is a guess the game can answer.
    hp_read = pyqtSignal(int)
    #: the wizard's gear stats, so training prices hits the way the game
    #: does rather than assuming a naked wizard
    gear_read = pyqtSignal(object)
    #: the policy actually installed on the backend, after a swap
    policy_changed = pyqtSignal(str)

    def __init__(self, telemetry, school, deck, policy_name, fights,
                 agent=None, auto_quest=False, auto_dialogue=True,
                 collect_wisps=True, use_potions=True, script="",
                 hotkeys=None):
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
        #: set when a trained policy is in play, so its coverage can be
        #: reported -- "the agent had never seen 94% of these boards" is
        #: the most useful thing to know about a learned live run.
        self.trained = None
        #: a deimoslang program, stepped between fights like the quester
        self.script = script or ""
        self.runner = None
        #: {action: key name}. Global hotkeys, so the same actions the
        #: buttons perform are reachable without alt-tabbing out of a
        #: full-screen game -- which is the difference between using them
        #: and not.
        self.hotkeys = dict(hotkeys or {})
        self._hotkeys = None
        #: the wizard's gear, read off the client on connect
        self.player_stats = {}
        #: said once, not every half-second, when the quest arrow is off
        self._warned_quest_arrow = False
        self._stop = False
        #: one-shot questing requests from the GUI thread. A plain list
        #: under a lock: individual list ops are atomic, but "is it
        #: already queued" followed by "queue it" is a check-then-act,
        #: and the two callers are on different threads -- buttons on the
        #: GUI thread, hotkeys on the worker's loop.
        self._requests = []
        self._requests_lock = _threading.Lock()
        #: the action currently being performed, if any. The queue
        #: dedupe only covers the window in which an action sits
        #: *waiting*; a wisp sweep runs for seconds, and a held hotkey
        #: would otherwise queue a fresh sweep the moment the last one
        #: started.
        self._busy = None
        #: serialises upkeep against itself. The fight loop awaits
        #: `after_fight` while the service task is live, and after a
        #: fight `in_battle` is False -- so a queued wisps request was
        #: serviced *during* the automatic sweep, running two of them
        #: against one client. Created in `_go`, on the worker's loop.
        self._upkeep_lock = None
        #: set while the fight loop is doing the between-fights chores.
        #: The lock keeps two upkeep runs apart; this keeps *questing*
        #: apart from upkeep, which is a different collision -- a wisp
        #: sweep yields the loop every 0.15s, and a quest-marker teleport
        #: landing between two wisp teleports moves the wizard off the
        #: field while the sweep keeps counting.
        self._in_upkeep = False
        #: stage name -> how many times it has failed, so a broken stage
        #: is reported rather than retried silently twice a second
        self._stage_errors = {}
        #: whether the wizard's real max health was ever read
        self.hp_known = False
        self._client = None

    def stop(self):
        """Ask the loop to finish after the current fight."""
        self._stop = True

    #: what `request` accepts. Every one of these drives the mouse, so
    #: every one is serviced from the one task that owns it.
    ACTIONS = ("teleport", "dialogue", "wisps", "potion")

    def request(self, action):
        """Queue an action ('teleport' | 'dialogue' | 'wisps' | 'potion').

        Called from the GUI thread. The loop performs it out of combat,
        because the client cannot be driven from two places at once --
        the combat handler is clicking cards during a duel and a second
        coroutine reaching for the mouse produces misclicks.

        Duplicates are dropped rather than queued -- including one that
        is *running* rather than waiting. Holding a hotkey down sends a
        burst of repeats, and a queue of eight wisp sweeps would take a
        minute to work through with the fight waiting. Returns whether
        the action was accepted, so the caller can stop claiming an
        action is happening when it was dropped.
        """
        if action not in self.ACTIONS:
            return False
        with self._requests_lock:
            if action in self._requests or action == self._busy:
                return False
            self._requests.append(action)
        return True

    # -- swapping the policy without dropping the connection --------------
    def set_policy(self, name, agent=None):
        """Install a different policy on a running fight. Returns ok.

        Called from the GUI thread, and unlike `request` it does *not* go
        through the queue: building a policy touches no client and no
        event loop -- it reads the card table and constructs a closure --
        so there is nothing to hand to the worker. The swap itself is one
        attribute assignment, and `WizAiBackend.decide` reads `policy`
        into a local before calling it, so the round in flight finishes
        under the policy it started with.

        Reconnecting to change models was the alternative, and it costs
        more than the wait: the deck picker's card list and the trained
        policy's own health bucket both come from what the last run
        observed, so dropping the connection throws away the inputs to
        the next decision.
        """
        if agent is not None:
            self.agent = agent
        previous = self.policy_name
        self.policy_name = name
        try:
            policy = self._build_policy()
        except Exception as exc:
            # Selecting "trained" with nothing trained yet lands here.
            # Keeping the old policy beats installing nothing: the fight
            # is still running, and a backend with no policy cannot play.
            self.policy_name = previous
            self.status.emit(f"kept {previous} — {exc}")
            self.policy_changed.emit(previous)
            return False

        self.tel.policy_name = name
        backend = self._backend
        if backend is None:
            # Not connected yet. `_go` builds from `policy_name`, so the
            # selection is already recorded and will be honoured.
            self.policy_changed.emit(name)
            return True
        # One call, not two attribute writes: the backend keeps the
        # policy and its label in a single tuple precisely so a decision
        # in flight cannot read the new name against the old callable.
        backend.set_policy(policy, name)
        self.status.emit(f"policy is now {name} — takes effect next round")
        self.policy_changed.emit(name)
        return True

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
                if await questing.in_battle(client) or self._in_upkeep:
                    await asyncio.sleep(0.5)
                    continue

                await self._drain_requests(client)

                if self.auto_dialogue and self.quester is None:
                    # Deimos's questing does its own dialogue handling, so
                    # a second clicker would race it for the same button.
                    await self._stage("auto-dialogue",
                                      self._auto_dialogue(client))

                if self.runner is not None:
                    await self._stage("script step", self._script_step())

                if self.auto_quest:
                    await self._stage("quest step", self._quest_step(client))

                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # The service task must outlive a bad read; the fight
                # loop is the thing that matters. But it says so now.
                # One `except` around the whole body used to swallow
                # every stage below the one that raised, so a broken
                # mouse hook killed auto-dialogue, the script runner and
                # auto-quest simultaneously and forever, without a word.
                self._stage_failed("the service loop", exc)
                await asyncio.sleep(1.0)

    async def _stage(self, name, coro):
        """Run one stage of the service tick, reporting its own failure.

        Per stage rather than per tick: a stage that raises must not take
        the ones below it off the air, and the message has to name which
        one it was or "nothing works" is all anybody can report.
        """
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stage_failed(name, exc)

    def _stage_failed(self, name, exc):
        self._say_once(name, f"{name} failed — {type(exc).__name__}: {exc}")

    def _say_once(self, key, message):
        """Say it the first time, then every 20th -- twice a second is spam.

        The service tick runs twice a second, so a stage that is broken
        rather than unlucky would fill the status bar with one line and
        nothing else. Reporting the first and then thinning out keeps
        the failure visible without burying everything around it.
        """
        n = self._stage_errors.get(key, 0) + 1
        self._stage_errors[key] = n
        if n == 1 or n % 20 == 0:
            self.status.emit(
                message + (f" (still failing after {n} tries)"
                           if n > 1 else ""))

    async def _drain_requests(self, client):
        """Perform the queued button/hotkey actions, one at a time.

        `in_battle` is re-checked before each one, not once per tick: a
        wisp sweep is a multi-second, multi-teleport action, and a duel
        that starts partway through would leave this teleporting the
        wizard around while the combat handler clicks cards.
        """
        from .. import questing

        while self._requests and not self._stop:
            if await questing.in_battle(client):
                return
            with self._requests_lock:
                if not self._requests:
                    return
                action = self._requests.pop(0)
                self._busy = action
            try:
                await self._stage(f"the {action} request",
                                  self._do_request(client, action))
            finally:
                self._busy = None

    async def _do_request(self, client, action):
        from .. import questing

        if action == "teleport":
            ok, reason = await questing.teleport_to_quest(client)
            self.status.emit("teleported to the quest marker"
                             if ok else reason)
        elif action == "dialogue":
            n, why = await questing.advance_dialogue(client)
            self.status.emit(f"advanced {n} dialogue window(s)" if n
                             else (why or "no dialogue open"))
        elif action in ("wisps", "potion"):
            await self._upkeep_now(client, action)

    async def _script_step(self):
        if not await self.runner.step() and self.runner.finished:
            self.status.emit("script finished")
            self.runner = None
        elif self.runner is not None and self.runner.failures in (1, 10):
            self.status.emit(f"script error: {self.runner.last_error}")

    async def _auto_dialogue(self, client):
        """Open and clear dialogue, but only the quest's.

        The game's press-X prompt appears for every interactable in
        range, so clicking it whenever it shows greets every vendor and
        signpost walked past -- each of which then has to be clicked back
        out of. `at_quest_marker` is the discriminator.

        The gate has a failure mode worth naming out loud: with the
        in-game quest arrow switched off the quest position never reads,
        so nothing is ever "at the marker" and auto-dialogue would
        silently do nothing at all. That is reported once rather than
        left to look like a broken feature.
        """
        from .. import questing

        if not await questing.in_dialogue(client):
            if await questing.near_interactable(client):
                near, why = await questing.at_quest_marker(client)
                if near:
                    ok, pressed_why = await questing.press_x(client)
                    if ok:
                        self.status.emit("opened the quest dialogue")
                        await asyncio.sleep(0.6)
                    elif pressed_why:
                        self._say_once("press-x", pressed_why)
                elif (why and "quest marker" not in why
                        and not self._warned_quest_arrow):
                    self._warned_quest_arrow = True
                    self.status.emit(
                        "auto-dialogue only talks to quest NPCs, and " + why)

        if await questing.in_dialogue(client):
            # Whatever is already open gets cleared, quest or not --
            # dialogue blocks movement, so leaving one up strands the run.
            n, click_why = await questing.advance_dialogue(client)
            if n:
                self.status.emit(f"auto-dialogue: {n} window(s)")
            elif click_why:
                # Reported, not spun on: this used to loop twice a second
                # forever against an open dialogue it could not click,
                # with movement blocked and nothing on screen.
                self._say_once("auto-dialogue-click", click_why)

    async def _upkeep_now(self, client, action):
        """Collect wisps, or drink a potion, on demand.

        The failure is *reported* -- and reported as the helper diagnosed
        it, not as this function guesses. `collect_wisps` distinguishes
        five separate reasons for coming back with nothing; passing no
        `on_status` threw all five away and printed one invented line,
        "no safe wisps in range", at a wizard standing on a pile of them.
        The helper's own message is the whole point of the hotkey.
        """
        from .. import upkeep

        ok, why = await upkeep.available()
        if not ok:
            self.status.emit(f"upkeep unavailable — {why}")
            return

        # The same lock the automatic sweep holds. Pressing the hotkey as
        # a fight ends would otherwise run two wisp sweeps, or two potion
        # calls, against one client on one loop.
        async with self._upkeep():
            try:
                if action == "wisps":
                    said = []

                    def relay(message):
                        said.append(message)
                        self.status.emit(message)

                    await upkeep.collect_wisps(client, on_status=relay)
                    if not said:
                        self.status.emit("no wisps in range")
                else:
                    if await upkeep.drink_potion(client):
                        self.status.emit("drank a potion")
                    else:
                        why = getattr(upkeep.drink_potion, "last_error", "")
                        self.status.emit(
                            "no potion drunk"
                            + (f" — {why}" if why else ""))
            except Exception as exc:
                self.status.emit(
                    f"{action} failed — {type(exc).__name__}: {exc}")

    def _upkeep(self):
        """The upkeep lock, or a no-op when there is no loop yet.

        `_upkeep_now` is reachable before `_go` has built the lock (and
        from tests that drive it directly), and a missing lock must not
        be the reason a chore does not run.
        """
        import contextlib

        if self._upkeep_lock is None:
            return contextlib.nullcontext()
        return self._upkeep_lock

    async def _setup_hotkeys(self):
        """Bind the global hotkeys, if any were configured.

        A keypress does exactly what the button does: it lands in
        `_requests`, and the service task performs it between clicks. It
        deliberately does not touch the client directly -- a hotkey can
        arrive mid-cast, and two things driving the mouse at once
        misclick.
        """
        if not self.hotkeys:
            return
        from .. import hotkeys as hk

        self._hotkeys = hk.Hotkeys(self.hotkeys, self.request,
                                   on_status=self.status.emit)
        try:
            await self._hotkeys.start()
        except Exception as exc:
            self._hotkeys = None
            self.status.emit(f"hotkeys not installed ({type(exc).__name__})")

    async def _read_max_hp(self, client):
        """Report the wizard's real max health, once, on connect.

        Training needs it: `Featurizer.key` buckets health as a fraction
        of the maximum, so a Q table trained against a made-up 800 and
        played on a wizard with 1,300 indexes different states for the
        same board. The client knows the number -- there is no reason to
        make anyone type it, and no reason for the guess to be wrong.

        A failed read is *said*, because its symptom points at the wrong
        fix. The box keeps its default, training buckets health against
        that default, the live wizard has some other maximum, and every
        live board then keys a health bucket the table never visited --
        so the window reports "the Q table decided 0% of the boards it
        was shown" and the obvious response, train for longer, cannot
        help. Naming the unread stat is the difference between a
        two-second fix and an hour of retraining.
        """
        try:
            hp = int(await client.stats.max_hitpoints())
        except Exception as exc:
            self.status.emit(
                f"could not read your max health ({type(exc).__name__}) — "
                f"training will use whatever is in the box, and a wrong "
                f"value makes the Q table share no states with the live "
                f"board")
            return          # a nicety; never worth failing the connect
        if hp > 0:
            self.hp_known = True
            self.hp_read.emit(hp)

    async def _read_gear(self, client):
        """The wizard's damage, accuracy, pierce and resist, on connect.

        Without it the simulator prices every hit as though the wizard
        were wearing nothing, and then optimises that fight instead of
        this one. A pet giving 9% is already enough to flip which move
        kills soonest -- see `live_state.read_player_stats` for the
        measurement.
        """
        from .. import live_state

        try:
            stats = await live_state.read_player_stats(client, self.school)
        except Exception:
            return
        if not stats:
            self.status.emit(
                "could not read your gear stats — the simulator will price "
                "hits as if you were wearing none")
            return
        self.player_stats = stats
        if self._backend is not None:
            self._backend.player_stats = stats
            if stats.get("power_pip_chance"):
                self._backend.power_pip_chance = stats["power_pip_chance"]
        self.gear_read.emit(dict(stats))
        dmg = (stats.get("damage") or {}).get(self.school, 0.0)
        line = (f"read your gear: {dmg * 100:.0f}% {self.school} damage, "
                f"{stats.get('pierce', 0.0) * 100:.0f}% pierce, "
                f"{stats.get('accuracy', 0.0) * 100:.0f}% accuracy")
        unread = stats.get("unread") or []
        if unread:
            # A stat that failed to read used to be folded into 0.0, so a
            # wizard in full gear was shown a confident "0% damage" that
            # then priced every hit in training below what it lands for.
            line += (f" — but {', '.join(unread)} could not be read and "
                     f"are being treated as 0")
        self.status.emit(line)

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

    async def _setup_script(self, client):
        from .. import scripts

        try:
            self.runner = scripts.make_runner(client, self.script)
            self.status.emit("script loaded")
        except Exception as exc:
            self.runner = None
            self.status.emit(f"script not loaded: {exc}")

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
        # One hop per tick. The blocking hunt cannot run here -- it would
        # stall the request queue -- and running it from the fight loop
        # was the bug: that loop parks in wait_for_combat, so a hunt
        # placed before it fired once per fight and then never again.
        await questing.hop_once(client, on_status=self.status.emit)

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

    def _build_policy(self):
        # Cleared first: `self.trained` drives the coverage readout, and
        # a stale one left over from a previous selection would report a
        # learned policy's numbers for a heuristic that replaced it.
        self.trained = None
        if self.policy_name.startswith("trained"):
            if self.agent is None:
                raise RuntimeError(
                    "No trained policy yet — press Train first, or pick "
                    "another policy.")
            from ..policies import trained_policy
            # Wrapped, not raw: a tabular agent has no opinion about a
            # state it never visited, and QAgent.greedy turns "no
            # opinion" into PASS. See policies.TrainedPolicy.
            self.trained = trained_policy(self.agent)
            return self.trained
        if self.policy_name.startswith("ttk"):
            from ..policies import greedy_ttk
            return greedy_ttk()
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
        built_as = self.policy_name
        policy = self._build_policy()

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

            await self._read_max_hp(client)
            await self._read_gear(client)

            self.tel.policy_name = self.policy_name
            self.tel.school = self.school
            self.tel.deck = self.deck
            backend = WizAiBackend(
                policy=policy, cards=cards, school=self.school,
                decklist=self.deck, catalog=catalog,
                policy_name=built_as, on_decision=self._on_decision,
                player_stats=self.player_stats,
                on_lost_round=self._on_lost_round)
            backend.on_failed_cast = self._on_failed_cast
            self.tel.resolver = backend.resolver
            self._backend = backend
            if self.policy_name != built_as:
                # The dropdown moved while the hooks were installing.
                # `set_policy` short-circuits until `_backend` exists, so
                # that selection is sitting in `policy_name` unapplied.
                self.set_policy(self.policy_name)
            combat = make_combat_handler(client, backend)

            if self.auto_quest:
                await self._setup_questing(client)
            if self.script:
                await self._setup_script(client)
            await self._setup_hotkeys()

            self._client = client
            self.status.emit(
                "connected — hunting for fights" if self.auto_quest
                else "connected — walk into a fight")
            # Built here, not in __init__: an asyncio.Lock binds to the
            # loop it is created on, and __init__ runs on the GUI thread.
            self._upkeep_lock = asyncio.Lock()
            servicer = asyncio.ensure_future(self._service_loop(client))
            fought = 0
            while not self._stop and (self.fights <= 0 or fought < self.fights):
                # Questing of either kind runs on the service task, which
                # keeps ticking while this loop is parked in
                # wait_for_combat below.
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

                if self.trained is not None:
                    t = self.trained
                    self.status.emit(
                        f"trained policy: knew {t.coverage * 100:.0f}% of "
                        f"{t.seen + t.missed} boards "
                        f"({t.missed} fell back)")

                if not self._stop and (self.collect_wisps or self.use_potions):
                    from .. import upkeep
                    try:
                        # Under the lock, and under a flag the service
                        # task honours: a wisp sweep yields the loop
                        # every 0.15s, and the service task used to wake
                        # up inside it, decide the wizard was out of
                        # combat and free, and teleport it to the quest
                        # marker halfway through the sweep.
                        self._in_upkeep = True
                        async with self._upkeep():
                            await upkeep.after_fight(
                                client, wisps=self.collect_wisps,
                                potions=self.use_potions,
                                on_status=self.status.emit)
                    except Exception as exc:
                        # Still never a blocker -- but the reason is said
                        # out loud. Swallowing it silently is what made
                        # "collect wisps is not working" impossible to
                        # act on: no message, no log, no difference
                        # between broken and nothing-to-do.
                        self.status.emit(
                            f"upkeep failed — {type(exc).__name__}: {exc}")
                    finally:
                        self._in_upkeep = False
                self.status.emit(
                    f"fight {fought} over — waiting for the next"
                    if not self._stop else "stopping…")
            self.finished_ok.emit()
        finally:
            if self._hotkeys is not None:
                # Before anything else: a registered hotkey is taken away
                # from every other program until it is released, so it
                # must not survive a failed run.
                await self._hotkeys.stop()
                self._hotkeys = None
            if self.runner is not None:
                self.runner.stop()
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
        # The backend measures this every round; it used to go nowhere,
        # while the trainer guessed the same quantity off the wizard's
        # own health.
        rec.incoming = float(
            getattr(self._backend, "_measured_incoming", 0.0) or 0.0)
        if self.tel.fights:
            self.tel.fights[-1].damage_taken += rec.incoming
        self.round_done.emit(rec)

    def _on_lost_round(self, round_number, reason):
        """A round whose board could not be read. Recorded as that.

        It used to vanish: no row in the Decisions table, no pass
        counted, and the round after it differenced against the round
        before -- so the missing round's damage was folded into its
        predecessor's residual and scored against the damage model.
        """
        self.status.emit(reason)
        rec = self.tel.observe_lost_round(round_number, reason)
        if rec is not None:
            self.round_done.emit(rec)

    def _on_failed_cast(self, reason):
        """The card never went out, after the round said it had."""
        self.status.emit(reason)
        rec = self.tel.note_failed_cast(reason)
        if rec is not None:
            self.round_done.emit(rec)
