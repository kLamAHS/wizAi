"""Driving real fights from the window — one wizard, or a party of four.

Three problems to keep apart.

**asyncio inside Qt.** wizwalker is async top to bottom and Qt has its
own loop, so the run happens on a `QThread` with its own event loop. The
window stays responsive and a hung memory read cannot freeze the UI.

**Thread affinity.** Qt widgets may only be touched from the GUI thread,
but `Telemetry.observe` is called from the worker on every planning
phase. So nothing here updates a widget: the worker emits signals, and
`MainWindow` does the drawing. `LiveWorker.round_done` is what the panels
ultimately refresh on, and because the signal crosses threads Qt queues
it onto the GUI thread automatically.

**One worker, several clients.** A party of up to four wizards runs on
*one* thread and *one* event loop, not four. That is not a saving, it is
a requirement: a wizwalker `Client` binds its hooks to the loop that
activated them, and the hivemind's barrier has to be able to wake a
sleeping seat from whichever seat closed the round. Each wizard gets its
own `_Seat` — client, backend, combat handler, telemetry, questing,
upkeep — and the fight loops run concurrently under `asyncio.gather`.

One wizard leads and the rest follow it around: the leader quests, the
followers teleport onto it and step into its duel (`deimos_bridge/
party.py`). That is not a convenience -- the round-by-round agreement
only reaches wizards who are in the same fight, and four clients each
running the questing independently walk to four different places and
coordinate perfectly with nobody.

A run of one is exactly the run that shipped before parties existed: no
coordinator is built, no barrier is entered, nothing follows anything,
and no status line is prefixed.

Seat 0's configuration is also reachable straight off the worker
(`worker.school`, `worker.deck`, `worker._backend`, ...), so every caller
written against the single-wizard worker keeps working.
"""
import asyncio
import threading as _threading

from PyQt6.QtCore import QThread, pyqtSignal


class SeatConfig:
    """One wizard's half of a run: what it plays and what it plays with.

    A party is not four copies of one wizard. Schools differ (that is
    most of the point — a death wizard's Feint is worth casting because
    a storm wizard is going to cash it), decks differ, and a trained Q
    table is keyed on its own deck, so the policy has to be per seat
    too.
    """

    def __init__(self, school="ice", deck=(), policy_name="ttk-lookahead",
                 agent=None, continuation="", telemetry=None, name=""):
        self.school = school
        self.deck = list(deck or [])
        self.policy_name = policy_name
        self.agent = agent
        self.continuation = continuation or ""
        #: each wizard records its own run. One shared telemetry would
        #: interleave four wizards' rounds into one Decisions table and
        #: settle each round's damage against another wizard's board.
        self.telemetry = telemetry
        self.name = name

    def label(self, index):
        return self.name or f"wizard {index + 1}"


class _Seat:
    """One wizard's live state, for the duration of a run."""

    def __init__(self, index, config, telemetry):
        self.index = index
        self.name = config.label(index)
        self.school = config.school
        self.deck = list(config.deck)
        self.policy_name = config.policy_name
        self.agent = config.agent
        self.continuation = config.continuation
        self.tel = telemetry
        #: set when a trained policy is in play, so its coverage can be
        #: reported -- "the agent had never seen 94% of these boards" is
        #: the most useful thing to know about a learned live run.
        self.trained = None
        self.backend = None
        self.client = None
        self.combat = None
        self.quester = None
        self.runner = None
        #: the wizard's gear, read off the client on connect
        self.player_stats = {}
        #: whether the wizard's real max health was ever read
        self.hp_known = False
        self.fought = 0
        #: one-shot questing requests from the GUI thread. A plain list
        #: under a lock: individual list ops are atomic, but "is it
        #: already queued" followed by "queue it" is a check-then-act,
        #: and the two callers are on different threads -- buttons on the
        #: GUI thread, hotkeys on the worker's loop.
        self.requests = []
        self.lock = _threading.Lock()
        #: the action currently being performed, if any. The queue
        #: dedupe only covers the window in which an action sits
        #: *waiting*; a wisp sweep runs for seconds, and a held hotkey
        #: would otherwise queue a fresh sweep the moment the last one
        #: started.
        self.busy = None
        #: serialises upkeep against itself, per client. The fight loop
        #: awaits `after_fight` while the service task is live, and after
        #: a fight `in_battle` is False -- so a queued wisps request was
        #: serviced *during* the automatic sweep, running two of them
        #: against one client. Per seat and not per run: two wizards are
        #: two clients, and there is nothing to serialise between them.
        self.upkeep_lock = None
        #: held by whatever is moving this wizard out of combat -- a
        #: queued teleport, a wisp sweep, an auto-dialogue click, a quest
        #: hop, a follow. One at a time, because two coroutines steering
        #: one client walk it into a wall. Separate from the request
        #: QUEUE, which is the whole point: the queue is attended by its
        #: own task and only ever waits for the one action in progress.
        self.drive = None
        #: what holds `drive` right now, by name, or None. A hung game
        #: read under the lock is invisible from outside it -- the run
        #: simply stops answering -- so whoever takes the wheel says so.
        self.driver = None
        #: action -> when it was queued, so an entry that can never be
        #: serviced expires instead of wedging the hotkey forever
        self.queued_at = {}
        #: set while the fight loop is doing the between-fights chores.
        #: The lock keeps two upkeep runs apart; this keeps *questing*
        #: apart from upkeep, which is a different collision -- a wisp
        #: sweep yields the loop every 0.15s, and a quest-marker teleport
        #: landing between two wisp teleports moves the wizard off the
        #: field while the sweep keeps counting.
        self.in_upkeep = False
        #: said once, not every half-second, when the quest arrow is off
        self.warned_quest_arrow = False
        #: this wizard's quest tracker goal line, as last read, and when.
        #: Compared across seats to catch the party drifting apart -- see
        #: `LiveWorker._check_in_step`.
        self.goal = ""
        self.goal_read = 0.0
        #: when the goal last CHANGED. The wizard whose goal moved most
        #: recently is the one that got ahead -- quest names have no
        #: order, but "who advanced last" does.
        self.goal_at = 0.0
        #: (zone, rounded position, goal) as last seen, and when it last
        #: CHANGED. A script that is stepping while none of these move is
        #: a script hammering something that is not working -- see
        #: `LiveWorker._check_progress`.
        self.progress = None
        self.progress_at = 0.0
        self.said_stuck = ""
        #: what `runner` was built from, so the service tick can notice
        #: the operator turning the script on, off, or replacing it
        self.script_source = None
        #: whether "the script is running" has been said for this
        #: runner. Said once, because the alternative is once per burst.
        self.script_said = False
        #: this wizard's in-game name, learned from its first duel. The
        #: client will not say it outside the character-select screen,
        #: but a combat read names the client's own member -- and the
        #: cross-zone follow needs it to pick the leader out of the
        #: friends list.
        self.wizard_name = None
        #: the wizard's max health, off the client on connect. Kept
        #: because it is the only identity available BEFORE a duel: a
        #: name needs combat, and a record that outlives a run has to be
        #: claimed or cleared before the first round lands in it.
        self.max_hp = 0
        #: when this wizard last tried to catch up with the leader. See
        #: `LiveWorker.FOLLOW_EVERY`.
        self.followed_at = 0.0
        #: since when this wizard has been in a different zone from the
        #: rest of the party, and which zone that is. See
        #: `LiveWorker._check_together`.
        self.stranded_since = None
        self.stranded_where = None
        self.rejoined_at = 0.0
        #: the zone this seat was last read in, so a message can name
        #: where the party actually is rather than "another zone"
        self.zone_seen = None
        #: stage name -> how many times it has failed, so a broken stage
        #: is reported rather than retried silently twice a second
        self.stage_errors = {}

    def enqueue(self, action, now=None):
        with self.lock:
            if action in self.requests or action == self.busy:
                return False
            self.requests.append(action)
            if now is not None:
                self.queued_at[action] = now
        return True


def _seat_property(name):
    """Expose seat 0's field on the worker itself.

    The window, the hotkeys and every test written before parties talk
    to `worker.school` / `worker._backend` / `worker.trained`. Those are
    seat 0's, and saying so once here is better than a second copy of
    the state that can drift out of step with the seat's.
    """

    def get(self):
        return getattr(self.seats[0], name)

    def set(self, value):
        setattr(self.seats[0], name, value)

    return property(get, set)


class LiveWorker(QThread):
    """Connects to the client(s) and plays fights until told to stop."""

    #: human-readable progress, straight to the status bar
    status = pyqtSignal(str)
    #: the same line, tagged with the seat it came from
    seat_status = pyqtSignal(int, str)
    #: one planning phase completed; payload is the RoundRecord
    round_done = pyqtSignal(object)
    seat_round_done = pyqtSignal(int, object)
    #: a fight ended
    fight_done = pyqtSignal(int)
    seat_fight_done = pyqtSignal(int, int)
    #: fatal, with a message already worth reading
    failed = pyqtSignal(str)
    #: the run stopped cleanly
    finished_ok = pyqtSignal()
    #: the wizard's real max health, once the hooks are up. Training has
    #: to use it or the learned states share no health bucket with a live
    #: board, and typing it in by hand is a guess the game can answer.
    hp_read = pyqtSignal(int)
    seat_hp_read = pyqtSignal(int, int)
    #: the wizard's gear stats, so training prices hits the way the game
    #: does rather than assuming a naked wizard
    gear_read = pyqtSignal(object)
    seat_gear_read = pyqtSignal(int, object)
    #: the policy actually installed on the backend, after a swap
    policy_changed = pyqtSignal(str)
    seat_policy_changed = pyqtSignal(int, str)
    #: one round agreed by the whole party; payload is a `PartyPlan`
    party_plan = pyqtSignal(object)
    #: (seat, the wizard's in-game name), once a duel has revealed it
    seat_named = pyqtSignal(int, str)

    def __init__(self, telemetry, school, deck, policy_name, fights,
                 agent=None, auto_quest=False, auto_dialogue=True,
                 collect_wisps=True, use_potions=True,
                 buy_potions=False, script="",
                 hotkeys=None, continuation="", seats=None,
                 coordinate=True, passes=2, barrier=None,
                 follow_leader=True, leader=0, label_windows=True):
        super().__init__()
        # Seat 0 is always the arguments this was called with, so the
        # single-wizard signature is untouched; `seats` adds the rest.
        first = SeatConfig(school=school, deck=deck, policy_name=policy_name,
                           agent=agent, continuation=continuation,
                           telemetry=telemetry)
        configs = [first] + list(seats or [])
        self.seats = [
            _Seat(i, cfg, cfg.telemetry if cfg.telemetry is not None
                  else (telemetry if i == 0 else None))
            for i, cfg in enumerate(configs)]
        for seat in self.seats:
            if seat.tel is None:
                from ..telemetry import Telemetry
                seat.tel = Telemetry()
        self.fights = fights
        self.auto_quest = auto_quest
        self.auto_dialogue = auto_dialogue
        #: between-fights upkeep. An unattended run dies by attrition
        #: long before it runs out of quests, and a policy that lost at
        #: 12% health has told you nothing about the policy.
        self.collect_wisps = collect_wisps
        self.use_potions = use_potions
        #: refill an empty bottle at a vendor. OFF by default, and that
        #: is not timidity: it spends real gold and takes a navigation
        #: detour across two zone changes that can leave the wizard
        #: somewhere the quest is not. Worth it when a run would
        #: otherwise grind to a halt on an empty bottle, which is the
        #: case the operator asked for; not worth doing behind anyone's
        #: back.
        self.buy_potions = bool(buy_potions)
        #: a deimoslang program, stepped between fights like the quester
        self.script = script or ""
        #: {action: key name}. Global hotkeys, so the same actions the
        #: buttons perform are reachable without alt-tabbing out of a
        #: full-screen game -- which is the difference between using them
        #: and not. Registered once for the whole run: they are
        #: system-wide keys, so a second registration of the same key
        #: fails, and one press should reach every wizard anyway.
        self.hotkeys = dict(hotkeys or {})
        self._hotkeys = None
        #: the hivemind, built in `_go` when there is more than one
        #: wizard. See `deimos_bridge/hivemind.py`; with one wizard it
        #: stays None and the decision path is the one that shipped.
        self.coordinate = bool(coordinate)
        self.passes = int(passes)
        self.barrier = barrier
        self.hive = None
        #: which wizard sets the pace, and whether the rest chase it.
        #: Without this a party is four wizards questing independently to
        #: four different places, coordinating perfectly with nobody --
        #: see `deimos_bridge/party.py`.
        self.leader = max(0, min(int(leader), len(self.seats) - 1))
        self.follow_leader = bool(follow_leader)
        #: write which seat a client is onto its own title bar. Four
        #: identical "Wizard101" windows cannot be told apart, and the
        #: seat numbering only exists inside this program.
        self.label_windows = bool(label_windows)
        self._stop = False

    # -- seat 0, reachable where it always was -----------------------------
    tel = _seat_property("tel")
    school = _seat_property("school")
    deck = _seat_property("deck")
    policy_name = _seat_property("policy_name")
    agent = _seat_property("agent")
    trained = _seat_property("trained")
    continuation = _seat_property("continuation")
    player_stats = _seat_property("player_stats")
    hp_known = _seat_property("hp_known")
    quester = _seat_property("quester")
    runner = _seat_property("runner")
    _backend = _seat_property("backend")
    _client = _seat_property("client")
    _requests = _seat_property("requests")
    _busy = _seat_property("busy")
    _upkeep_lock = _seat_property("upkeep_lock")
    _in_upkeep = _seat_property("in_upkeep")
    _warned_quest_arrow = _seat_property("warned_quest_arrow")
    _stage_errors = _seat_property("stage_errors")

    @property
    def party(self):
        return len(self.seats)

    def stop(self):
        """Ask every seat's loop to finish after the current fight."""
        self._stop = True

    #: what `request` accepts. Every one of these drives the mouse, so
    #: every one is serviced from the one task that owns it.
    ACTIONS = ("teleport", "dialogue", "wisps", "potion")

    def request(self, action, seat=None):
        """Queue an action ('teleport' | 'dialogue' | 'wisps' | 'potion').

        Called from the GUI thread, and by default it reaches **every**
        wizard: one button is meant to sweep the whole party's wisps, not
        one quarter of them. `seat` narrows it to one.

        The loop performs it out of combat, because the client cannot be
        driven from two places at once -- the combat handler is clicking
        cards during a duel and a second coroutine reaching for the mouse
        produces misclicks.

        Duplicates are dropped rather than queued -- including one that
        is *running* rather than waiting. Holding a hotkey down sends a
        burst of repeats, and a queue of eight wisp sweeps would take a
        minute to work through with the fight waiting. Returns whether
        the action was accepted by any seat, so the caller can stop
        claiming an action is happening when it was dropped.
        """
        if action not in self.ACTIONS:
            return False
        import time

        targets = (self.seats if seat is None
                   else [self.seats[seat]] if 0 <= seat < len(self.seats)
                   else [])
        now = time.monotonic()
        return any([s.enqueue(action, now) for s in targets])

    # -- swapping the policy without dropping the connection --------------
    def set_policy(self, name, agent=None, seat=None):
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
        which = self.seats[seat] if seat is not None else self.seats[0]
        if agent is not None:
            which.agent = agent
        previous = which.policy_name
        which.policy_name = name
        try:
            policy = self._build_policy(which)
        except Exception as exc:
            # Selecting "trained" with nothing trained yet lands here.
            # Keeping the old policy beats installing nothing: the fight
            # is still running, and a backend with no policy cannot play.
            which.policy_name = previous
            self._say(which, f"kept {previous} — {exc}")
            self._policy_installed(which, previous)
            return False

        which.tel.policy_name = name
        backend = which.backend
        if backend is None:
            # Not connected yet. `_go` builds from `policy_name`, so the
            # selection is already recorded and will be honoured.
            self._policy_installed(which, name)
            return True
        # One call, not two attribute writes: the backend keeps the
        # policy and its label in a single tuple precisely so a decision
        # in flight cannot read the new name against the old callable.
        backend.set_policy(policy, name)
        self._say(which, f"policy is now {name} — takes effect next round")
        self._policy_installed(which, name)
        return True

    def _policy_installed(self, seat, name):
        self.seat_policy_changed.emit(seat.index, name)
        if seat.index == 0:
            self.policy_changed.emit(name)

    def _seat_for(self, client):
        """Whose wizard is this client? Seat 0 when nobody claims it.

        The per-client helpers below take the client and derive the seat
        from it rather than being handed both. That is not tidiness: a
        client is what every one of them actually operates on, a seat is
        bookkeeping, and keeping the signature at "the thing it drives"
        is what lets `_service_loop` be driven with a stand-in client in
        a test without the seat having to exist.
        """
        for seat in self.seats:
            if seat.client is client:
                return seat
        return self.seats[0]

    # -- talking to the window --------------------------------------------
    def _say(self, seat, message):
        """One status line, tagged with the wizard it is about.

        Only tagged in a party. With one wizard the tag would be noise on
        every line, and every message the single-wizard run ever printed
        would change shape for no reason.
        """
        index = 0 if seat is None else seat.index
        self.seat_status.emit(index, message)
        if len(self.seats) > 1 and seat is not None:
            message = f"[{seat.name}] {message}"
        self.status.emit(message)

    async def _service_loop(self, client, seat=None):
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

        seat = self._seat_for(client) if seat is None else seat
        while not self._stop:
            try:
                if await questing.in_battle(client) or seat.in_upkeep:
                    await asyncio.sleep(0.5)
                    continue

                # A person pressing a key outranks any of the automatic
                # chores below. Standing aside costs half a second of
                # questing and buys a hotkey that answers. The queue
                # itself belongs to `_request_loop` -- draining it from
                # here too let both tasks pop an action, and the second
                # pop overwrote `seat.busy`, so the dedupe stopped
                # covering whichever action was actually running.
                if seat.requests:
                    await asyncio.sleep(self.REQUEST_POLL)
                    continue

                # Building or tearing down the runner steers nothing, so
                # it is outside the drive lock -- and outside the
                # auto-dialogue branch it was mistakenly nested in, which
                # meant the "Run script" tick only took effect on a
                # wizard that also had auto-dialogue on and no Deimos
                # quester running.
                await self._stage(seat, "script", self._sync_script(seat))

                # The wheel is taken per stage, not for the whole tick.
                # It exists to stop two coroutines steering one wizard at
                # the same moment, not to make a tick atomic -- and held
                # across the tick it is held for the SUM of the stages,
                # so a press could wait out an auto-dialogue and a follow
                # before its own turn. Released between them, a queued
                # request gets the wizard at the next gap.
                # Cheap and unconditional: whatever is steering this
                # wizard -- a script, wizAi's questing, a follow -- the
                # party can drift onto different quests, and a script in
                # particular cannot notice because its own instruction
                # pointer is fine. Reading is throttled internally.
                driven = self._script_drives(seat)
                # FIRST, ahead of the reads below. An open dialogue box
                # blocks movement, so every millisecond it stays up is a
                # wizard standing still -- and this used to run after the
                # goal read, the in-step check and the script sync, so it
                # waited on all three every tick before it even looked.
                # Reported live as auto-dialogue being slow to trigger.
                if self.auto_dialogue and seat.quester is None and not driven:
                    # Deimos's questing does its own dialogue handling,
                    # so a second clicker would race it for the same
                    # button. A running script is the same problem: they
                    # all reach for the dialogue box.
                    await self._stage(seat, "auto-dialogue",
                                      self._auto_dialogue(client), wheel=True)

                await self._read_goal(seat)
                self._check_in_step(seat)
                self._check_progress(seat)

                if seat.runner is not None:
                    await self._stage(seat, "script step",
                                      self._script_step(seat), wheel=True)

                if driven and self._should_catch_up(seat):
                    # The one case where wizAi steers a wizard the script
                    # is also steering. See `_should_catch_up`.
                    await self._stage(seat, "going back for the others",
                                      self._catch_up(seat), wheel=True)
                elif driven and self._due_to_regroup():
                    # The other one, and the reason a scripted run needs a
                    # human every ten minutes: a wizard whose teleport
                    # silently did not land. It is a comparison BETWEEN
                    # seats, so it is throttled on the worker rather than
                    # run once per seat -- and driven by whichever seat's
                    # loop is alive, because the seat that owns the
                    # script may be the one in a duel, or the one adrift.
                    await self._stage(seat, "keeping the party together",
                                      self._regroup(), wheel=True)
                elif driven:
                    # The script walks this wizard. wizAi's own questing
                    # or follow would walk it somewhere else between two
                    # of the script's instructions, which is how a
                    # scripted run ends up standing in a doorway.
                    pass
                elif self._follows(seat):
                    # A follower does not quest. Two wizards taking their
                    # own quests walk to two places, and then the party
                    # coordinates beautifully with nobody.
                    await self._stage(seat, "following the leader",
                                      self._follow_step(client), wheel=True)
                elif self.auto_quest:
                    await self._stage(seat, "quest step",
                                      self._quest_step(client), wheel=True)

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
                self._stage_failed(seat, "the service loop", exc)
                await asyncio.sleep(1.0)

    #: how long a request may sit unserviced before it is dropped. A
    #: queue entry that can never run is worse than no entry at all: the
    #: dedupe refuses every further press of that key, so the hotkey goes
    #: dead and the only way back is a combat cycle -- which is exactly
    #: the "I have to walk into a fight and out again to reset the tp
    #: hotkey" this expiry exists to end.
    REQUEST_TTL = 45.0
    #: how often the request task looks at the queue
    REQUEST_POLL = 0.2

    async def _request_loop(self, client, seat=None):
        """Nothing but the button and hotkey queue, on its own task.

        It used to share a tick with auto-dialogue, the script runner and
        the quest or follow step. Every one of those can run for seconds
        -- `advance_dialogue` clicks up to forty times at half a second
        each, a quest hop settles for 1.2s, a wisp sweep teleports twelve
        times -- and the drain sat at the top of that tick waiting its
        turn. Meanwhile `enqueue` refuses a second press of an action
        already queued, so every further press of the key was dropped
        with "already running" while nothing was running at all. Walking
        into a fight and out again cleared it because the tick then
        short-circuits at the `in_battle` check and comes straight back
        round to the drain.

        So the queue gets its own task, and what the two tasks share is
        the *drive lock* rather than the tick: a press now waits for the
        one action actually in progress and nothing else.
        """
        seat = self._seat_for(client) if seat is None else seat
        while not self._stop:
            try:
                if seat.requests:
                    self._expire_requests(seat)
                    await self._drain_requests(client, seat)
                await asyncio.sleep(self.REQUEST_POLL)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stage_failed(seat, "the request queue", exc)
                await asyncio.sleep(1.0)

    def _expire_requests(self, seat):
        """Drop anything that has been queued too long to still be wanted.

        A teleport pressed a minute ago is not a teleport anybody still
        wants, and while it sits there the key is dead.
        """
        import time

        if seat.driver is not None:
            # Not stuck -- queued behind something that is running and
            # said so. A script burst can hold the wheel for longer than
            # the TTL, and dropping the press with "press it again"
            # while the thing it is waiting for is visibly working would
            # be a lie as well as a lost keypress.
            return

        now = time.monotonic()
        stale = [a for a in list(seat.requests)
                 if now - seat.queued_at.get(a, now) > self.REQUEST_TTL]
        for action in stale:
            with seat.lock:
                if action in seat.requests:
                    seat.requests.remove(action)
            seat.queued_at.pop(action, None)
            self._say(seat,
                      f"dropped the queued {action} — it waited "
                      f"{self.REQUEST_TTL:.0f}s without a chance to run, "
                      f"and a stuck entry is what makes the key stop "
                      f"responding. Press it again.")

    def _driving(self, seat, owner="something"):
        """Take the wheel for this wizard, recording who has it.

        The name is not decoration. Everything below the lock is a game
        read or a mouse click, and a hung one leaves the lock held with
        no way to tell from the outside what is holding it -- which is
        exactly the state the run was in when every hotkey stopped
        answering. `seat.driver` turns that into a status line: a press
        that has to wait says what it is waiting for, by name.
        """
        import contextlib

        if seat.drive is None:
            return contextlib.nullcontext()

        @contextlib.asynccontextmanager
        async def held():
            async with seat.drive:
                seat.driver = owner
                try:
                    yield
                finally:
                    seat.driver = None

        return held()

    #: seconds any one stage may hold the wheel before it is cut off.
    #:
    #: This is the whole reason a hotkey can go dead. Every stage below
    #: the drive lock is a real game read or a real mouse click, and some
    #: of the ones wizwalker ships **cannot** finish: the friends-list
    #: teleport clicks the page button in `while (await text()) !=
    #: "Online Friends"` with no bound at all, so a wizard whose list
    #: never reads exactly that spins there for the rest of the run. It
    #: holds the wheel while it spins, so every queued teleport, wisp
    #: sweep and potion behind it waits forever and every further press
    #: is refused as "already queued". One unbounded loop in a follow
    #: step took all four keys away.
    #:
    #: Per stage rather than one number, because a legitimate quest hop
    #: settles for over a second and a legitimate friends-list teleport
    #: opens a window, clicks through a confirmation and waits out a
    #: teleport animation. The limits are generous enough that nothing
    #: which is working gets cut off, and short enough that nothing which
    #: is stuck owns the wizard.
    #: `script step` is the odd one. It is a time-boxed burst that
    #: returns on its own (`ScriptRunner.SLICE`), so this is only a
    #: backstop -- and it has to sit ABOVE `ScriptRunner.STEP_LIMIT`,
    #: because a single deimoslang instruction legitimately blocks for
    #: a whole loading screen and cancelling one mid-flight leaves the
    #: VM half-way through it. The runner bounds and reloads itself;
    #: this only catches a burst that somehow never returns at all.
    STAGE_LIMITS = {"auto-dialogue": 45.0,
                    "script": 20.0,
                    "script step": 240.0,
                    "following the leader": 60.0,
                    "quest step": 60.0}
    #: the fallback for a stage with no entry above, including the
    #: `the <action> request` stages, which are named after the action
    DEFAULT_STAGE_LIMIT = 90.0
    #: the between-fights chores, which are not a stage but hold the same
    #: wheel. A full wisp sweep is twelve teleports with a settle after
    #: each; two minutes is several of those and still finite.
    AFTER_FIGHT_LIMIT = 120.0
    #: health fraction below which another fight is not worth starting.
    #: `upkeep.needs_potion` uses 0.55 as "should top up"; this is the
    #: lower, separate question of "walking into the next duel is just
    #: dying again", and it wants a floor that a full-health wizard can
    #: never trip.
    LOW_HEALTH = 0.35
    #: ...but 0.35 was chosen against "essentially 0", and a fight costs
    #: far more than that. Over the two-wizard run at rev 8666bda7 the
    #: fire wizard's fights took 191 to 630 health out of 1,196 -- 16% to
    #: 53% -- so he walked into his eighth on 53% and left it on nothing.
    #: The floor is therefore what the fights have ACTUALLY cost this
    #: wizard rather than a number picked once: see `_health_needed`.
    #: This stays as the answer before there is any history, and as the
    #: lower bound.
    LOW_HEALTH_CAP = 0.75
    #: how many recent fights the floor is taken from. Long enough to see
    #: a hard pull, short enough that one bad dungeon does not hold a
    #: wizard out of easy street content for the rest of the session.
    LOW_HEALTH_WINDOW = 5
    #: how long to keep trying to fix that before going anyway. A run
    #: that blocks forever on an empty potion bottle is as broken as one
    #: that suicides, so this ends -- loudly.
    LOW_HEALTH_WAIT = 150.0
    #: between health reads while waiting
    LOW_HEALTH_POLL = 10.0
    #: how often to re-read a wizard's quest tracker. It only changes
    #: when a step completes, so this is cheap to do rarely.
    GOAL_POLL = 8.0
    #: how long a scripted wizard may change nothing at all -- zone,
    #: position, quest goal -- before the run says so. Generous, because
    #: a long fight or a slow dungeon legitimately looks like this: five
    #: minutes of a wizard standing in the same spot on the same quest
    #: is not patience, it is a loop that is not working.
    STUCK_AFTER = 300.0
    #: how long the party may be on different quest goals before it is
    #: called a desync. Turning a step in is not simultaneous -- one
    #: wizard clicks the NPC seconds before the other -- so a bare
    #: inequality would cry wolf on every normal handover.
    DESYNC_GRACE = 90.0

    async def _stage(self, seat, name, coro, limit=None, wheel=False):
        """Run one stage of the service tick, reporting its own failure.

        Per stage rather than per tick: a stage that raises must not take
        the ones below it off the air, and the message has to name which
        one it was or "nothing works" is all anybody can report.

        Bounded, too. A stage that raises is visible and recoverable; a
        stage that never returns is neither, and it takes the drive lock
        with it. See `STAGE_LIMITS`.

        `wheel` for a stage that steers -- teleports, clicks, walks. It
        takes the drive lock for its own duration and gives it straight
        back, so the stage after it starts from scratch and a queued
        keypress can get in between the two.
        """
        if limit is None:
            limit = self.STAGE_LIMITS.get(name, self.DEFAULT_STAGE_LIMIT)
        if wheel:
            coro = self._at_the_wheel(seat, name, coro)
        try:
            await asyncio.wait_for(coro, limit)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self._say_once(
                seat, name,
                f"{name} ran for {limit:.0f}s without finishing and was cut "
                f"off. It holds the wheel while it runs, so everything else "
                f"for this wizard — the hotkeys included — was waiting "
                f"behind it.")
        except Exception as exc:
            self._stage_failed(seat, name, exc)

    async def _at_the_wheel(self, seat, name, coro):
        """`coro`, holding this wizard's drive lock.

        Inside the stage's own deadline rather than outside it, which is
        the whole point: the *acquisition* is bounded too, so a stage
        that cannot get the wheel is cut off and reported instead of
        queueing up behind a wedge and adding to it.
        """
        async with self._driving(seat, name):
            await coro

    def _stage_failed(self, seat, name, exc):
        self._say_once(seat, name,
                       f"{name} failed — {type(exc).__name__}: {exc}")

    def _say_once(self, seat, key, message):
        """Say it the first time, then every 20th -- twice a second is spam.

        The service tick runs twice a second, so a stage that is broken
        rather than unlucky would fill the status bar with one line and
        nothing else. Reporting the first and then thinning out keeps
        the failure visible without burying everything around it.
        """
        seat = self.seats[0] if seat is None else seat
        n = seat.stage_errors.get(key, 0) + 1
        seat.stage_errors[key] = n
        if n == 1 or n % 20 == 0:
            self._say(seat, message + (f" (still failing after {n} tries)"
                                       if n > 1 else ""))

    async def _drain_requests(self, client, seat=None):
        """Perform the queued button/hotkey actions, one at a time.

        `in_battle` is re-checked before each one, not once per tick: a
        wisp sweep is a multi-second, multi-teleport action, and a duel
        that starts partway through would leave this teleporting the
        wizard around while the combat handler clicks cards.
        """
        from .. import questing

        seat = self._seat_for(client) if seat is None else seat
        while seat.requests and not self._stop:
            if await questing.in_battle(client):
                return
            with seat.lock:
                if not seat.requests:
                    return
                action = seat.requests.pop(0)
                seat.busy = action
            seat.queued_at.pop(action, None)
            try:
                # The lock, not the tick: the operator's press waits for
                # whatever is steering the wizard right now and for
                # nothing else. And it says what that is, because a press
                # that goes quiet for thirty seconds is indistinguishable
                # from a key that is not bound.
                if seat.driver is not None:
                    self._say(seat, f"{action} is queued — waiting for "
                                    f"{seat.driver} to let go of the wheel")
                await self._stage(seat, f"the {action} request",
                                  self._do_request(client, action, seat),
                                  wheel=True)
            finally:
                seat.busy = None

    async def _do_request(self, client, action, seat=None):
        from .. import questing

        seat = self.seats[0] if seat is None else seat
        if action == "teleport":
            ok, reason = await questing.teleport_to_quest(client)
            self._say(seat, "teleported to the quest marker"
                      if ok else reason)
        elif action == "dialogue":
            n, why = await questing.advance_dialogue(client)
            self._say(seat, f"advanced {n} dialogue window(s)" if n
                      else (why or "no dialogue open"))
        elif action in ("wisps", "potion"):
            await self._upkeep_now(client, action)

    async def _chores_can_wait(self, seat):
        """False when this follower should be catching up, not sweeping.

        The between-fights chores set `in_upkeep`, and the service task
        skips its whole tick while that is set -- so a follower that
        started a wisp sweep as the leader walked into a duel cannot
        follow until the sweep finishes, which is up to two minutes. The
        leader fights it alone, and the party plans a coordinated round
        for one wizard.

        Wisps keep. A duel does not.
        """
        if not self._follows(seat):
            return True
        boss = self.seats[self.leader]
        if boss.client is None:
            return True
        from .. import party

        if not await party.in_battle(boss.client):
            return True
        self._say(seat,
                  "skipping the wisps — the leader is already in a fight and "
                  "getting into it matters more")
        return False

    def _script_drives(self, seat):
        """Is a running script steering this wizard?

        Every wizard's, not just seat 0's: the party's one VM moves
        `p1`..`p4`, so while it is running none of them should also be
        taking their own quest or chasing the leader. Two things walking
        one wizard is how a scripted run ends up in a doorway.
        """
        runner = self.seats[0].runner
        return runner is not None and runner.running

    def _scripted(self, seat):
        """Does this seat run the party's script?

        Seat 0 and only seat 0. A deimoslang program addresses the whole
        party as `p1`..`p4` and there is one of it, so there is one VM
        over every client -- not one per seat, which is four copies of
        the same quester each believing it is `p1`. See `scripts.py`.
        """
        return seat.index == 0

    async def _sync_script(self, seat):
        """Build, replace or tear down the party's script runner.

        The script was read once at Play live and never again, so
        ticking "Run script" mid-run did nothing and unticking it did
        nothing either -- the runner built at connect kept stepping.
        Done here rather than from the GUI thread because building one
        takes the clients, and they belong to this loop.
        """
        if not self._scripted(seat):
            return
        want = self.script or ""
        if seat.script_source == want:
            return
        seat.script_source = want
        if seat.runner is not None:
            seat.runner.stop()
            seat.runner = None
            if not want:
                self._say(seat, "script stopped")
        if want:
            await self._setup_script(seat.client, seat)

    async def _script_step(self, seat=None):
        """One burst of the script, not one instruction.

        `VM.step()` runs a single instruction. A real quester compiles
        to tens of thousands of them -- the TTS Arc 1 script is 18,366
        -- so one per half-second tick is two and a half hours to reach
        the end of the program once, and the wizard visibly does
        nothing. Deimos runs `while v.running: await v.step()`; this
        runs that loop for `ScriptRunner.SLICE` seconds at a time, which
        keeps the wheel available to a hotkey between bursts.
        """
        seat = self.seats[0] if seat is None else seat
        runner = seat.runner
        if runner is None:
            return
        done = await runner.run_for(
            should_stop=lambda: self._stop or seat.in_upkeep)
        if runner.stale:
            # An instruction had to be cancelled, so the VM is part-way
            # through one. Reloading is the only honest recovery.
            self._say(seat, runner.last_error)
            if not runner.restart():
                self._say(seat, "script stopped — it could not be reloaded")
                seat.runner = None
            return
        if runner.failures:
            # Thinned rather than reported at exactly the first and
            # tenth: an instruction that always raises is retried every
            # burst, and after the tenth the old rule went silent
            # forever while the script sat on the same instruction.
            self._say_once(seat, "script-error",
                           f"script error: {runner.last_error}")
            return
        if done or runner.running:
            if runner.steps and not seat.script_said:
                seat.script_said = True
                self._say(seat, f"script running — {runner.steps:,} "
                                f"instructions so far")
            return
        # It ran off the end. Deimos reloads and runs it again, and
        # questers are written expecting that; only `kill` ends a run.
        if runner.restart():
            self._say(seat, f"script reached the end — restarting it "
                            f"(pass {runner.restarts + 1})")
        else:
            self._say(seat, "script finished")
            seat.runner = None

    async def _auto_dialogue(self, client, seat=None):
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

        seat = self._seat_for(client) if seat is None else seat
        if not await questing.in_dialogue(client):
            if await questing.near_interactable(client):
                near, why = await questing.at_quest_marker(client)
                if near:
                    ok, pressed_why = await questing.press_x(client)
                    if ok:
                        self._say(seat, "opened the quest dialogue")
                        await questing.dialogue_opened(client)
                    elif pressed_why:
                        self._say_once(seat, "press-x", pressed_why)
                elif why and not seat.warned_quest_arrow:
                    # Including the too-far-from-the-marker case, which
                    # used to be filtered out here and is the one that
                    # looks like the feature is dead: the press-X prompt
                    # is up, the wizard is standing at somebody, and
                    # nothing happens. `at_quest_marker` now says how far
                    # away it thinks the marker is, which is the number
                    # that decides whether the radius is wrong or the
                    # wizard simply is not there yet.
                    seat.warned_quest_arrow = True
                    self._say(
                        seat,
                        "auto-dialogue only talks to quest NPCs, and " + why)

        if await questing.in_dialogue(client):
            # Whatever is already open gets cleared, quest or not --
            # dialogue blocks movement, so leaving one up strands the run.
            n, click_why = await questing.advance_dialogue(client)
            if n:
                self._say(seat, f"auto-dialogue: {n} window(s)")
            elif click_why:
                # Reported, not spun on: this used to loop twice a second
                # forever against an open dialogue it could not click,
                # with movement blocked and nothing on screen.
                self._say_once(seat, "auto-dialogue-click", click_why)

    async def _upkeep_now(self, client, action, seat=None):
        """Collect wisps, or drink a potion, on demand.

        The failure is *reported* -- and reported as the helper diagnosed
        it, not as this function guesses. `collect_wisps` distinguishes
        five separate reasons for coming back with nothing; passing no
        `on_status` threw all five away and printed one invented line,
        "no safe wisps in range", at a wizard standing on a pile of them.
        The helper's own message is the whole point of the hotkey.
        """
        from .. import upkeep

        seat = self._seat_for(client) if seat is None else seat
        ok, why = await upkeep.available()
        if not ok:
            self._say(seat, f"upkeep unavailable — {why}")
            return

        # The same lock the automatic sweep holds. Pressing the hotkey as
        # a fight ends would otherwise run two wisp sweeps, or two potion
        # calls, against one client on one loop.
        async with self._upkeep(seat):
            try:
                if action == "wisps":
                    said = []

                    def relay(message):
                        said.append(message)
                        self._say(seat, message)

                    await upkeep.collect_wisps(client, on_status=relay)
                    if not said:
                        self._say(seat, "no wisps in range")
                else:
                    if await upkeep.drink_potion(client):
                        self._say(seat, "drank a potion")
                    else:
                        why = getattr(upkeep.drink_potion, "last_error", "")
                        self._say(seat, "no potion drunk"
                                  + (f" — {why}" if why else ""))
            except Exception as exc:
                self._say(seat,
                          f"{action} failed — {type(exc).__name__}: {exc}")

    def _upkeep(self, seat=None):
        """The upkeep lock, or a no-op when there is no loop yet.

        `_upkeep_now` is reachable before `_go` has built the lock (and
        from tests that drive it directly), and a missing lock must not
        be the reason a chore does not run.
        """
        import contextlib

        seat = self.seats[0] if seat is None else seat
        if seat.upkeep_lock is None:
            return contextlib.nullcontext()
        return seat.upkeep_lock

    async def _setup_hotkeys(self):
        """Bind the global hotkeys, if any were configured.

        A keypress does exactly what the button does: it lands in every
        seat's request queue, and each seat's service task performs it
        between clicks. It deliberately does not touch a client directly
        -- a hotkey can arrive mid-cast, and two things driving the mouse
        at once misclick.
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

    #: how long to wait for one client's hooks before saying something.
    #: wizwalker's own default is *no* timeout, which on a background
    #: client is a hang -- see `_activate_all_hooks`.
    HOOK_TIMEOUT = 20.0
    #: bring each client to the front while its hooks are being written.
    #: Intrusive, and the alternative is the hang.
    FOCUS_TO_HOOK = True

    async def _activate_all_hooks(self):
        """Install every client's hooks, without hanging on any of them.

        This is the fix for "hooking multiple characters does not work
        first try; I have to kill the bot and re-hook every time", and
        the cause is worth writing down because nothing about it is
        visible from here.

        `client.activate_hooks()` defaults to `wait_for_ready=True,
        timeout=None`, and that `None` reaches
        `asyncio.wait_for(task, None)` -- it waits **forever** for five
        addresses to become non-zero: `player_struct`,
        `player_stat_struct`, `current_client`, `current_root_window`
        and `current_render_context`. Those are written by the game's
        own code when it next runs those paths, and the last two are the
        UI and render paths. A Wizard101 client that is not the
        foreground window throttles rendering, so on a background client
        they may simply never fire.

        Hooking the clients one after another then parks the whole run
        on client 2 forever while client 1 sits there hooked, with no
        timeout, no message and nothing to do but kill it. Alt-tabbing
        to client 2 makes it render, which is exactly the manual
        "messing around" that makes it take.

        So: bring each client to the front for the moment its hooks are
        being written, bound the wait, and put the operator's own window
        back afterwards. The focus dance is intrusive and it is still
        better than the hang; `FOCUS_TO_HOOK` turns it off.
        """
        was_in_front = self._foreground_window()
        try:
            for seat in self.seats:
                await self._activate_hooks(seat)
        finally:
            self._restore_foreground(was_in_front)

    def _foreground_window(self):
        try:
            from wizwalker import utils
            return utils.get_foreground_window()
        except Exception:
            return None

    def _restore_foreground(self, handle):
        """Give the operator their own window back."""
        if handle is None:
            return
        try:
            from wizwalker import utils
            utils.set_foreground_window(handle)
        except Exception:
            pass          # politeness, never a blocker

    def _focus(self, seat):
        try:
            seat.client.is_foreground = True
            return True
        except Exception:
            return False

    async def _activate_hooks(self, seat, tries=2):
        """One client's hooks, with a bounded wait and a way out."""
        for attempt in range(max(1, tries)):
            focused = self._focus(seat) if self.FOCUS_TO_HOOK else False
            if focused:
                # A frame or two for the client to notice it is visible
                # again and start writing the values wizwalker waits on.
                await asyncio.sleep(0.4)
            self._say(seat, "activating hooks…" if not attempt
                      else "activating hooks (retrying)…")
            try:
                await seat.client.activate_hooks(timeout=self.HOOK_TIMEOUT)
                return True
            except Exception as exc:
                name = type(exc).__name__
                if "AlreadyActivated" in name:
                    return True          # a previous run left them up
                if "Pattern" in name or "Pattern" in str(exc):
                    raise RuntimeError(
                        "wizwalker could not install its hooks: the "
                        "autobot signature was not found in the running "
                        "client.\n\n"
                        "Run  python -m deimos_bridge.diagnose_hooks  — "
                        "it tells you whether this is stale state in the "
                        "process (close the game completely) or a game "
                        "patch that outdates wizwalker."
                    ) from exc
                if not isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                    raise
                if attempt + 1 < tries:
                    self._say(
                        seat,
                        f"this client's hooks did not finish in "
                        f"{self.HOOK_TIMEOUT:.0f}s — the values wizwalker "
                        f"waits for are written by the game's own render "
                        f"loop, and a background client barely renders. "
                        f"Bringing it to the front and trying once more.")
        raise RuntimeError(
            f"{seat.name}'s client hooked but never finished writing its "
            f"hook values, twice, at {self.HOOK_TIMEOUT:.0f}s each.\n\n"
            f"wizwalker waits for the game to write five addresses, two of "
            f"which come from the UI and render paths — so a client that is "
            f"minimised, on another virtual desktop, or otherwise not "
            f"drawing will never finish. Bring that client's window up so "
            f"it is visibly rendering, leave it at the character or world "
            f"screen rather than a loading screen, and press Play live "
            f"again.")

    #: reads the run cannot do without, and where each one is needed.
    #: Probed rather than assumed, because `activate_hooks()` returning
    #: is not the same as the hooks answering -- see `_verify_hooks`.
    HOOK_PROBES = (
        ("max health", lambda c: c.stats.max_hitpoints(),
         "training buckets health as a fraction of the maximum"),
        ("position", lambda c: c.body.position(),
         "wisp sweeps, quest hops and following the leader all teleport"),
        ("zone", lambda c: c.zone_name(),
         "a follower cannot tell it is in a different zone from its "
         "leader"),
    )

    async def _verify_hooks(self, seat, tries=3, settle=1.0):
        """Check the reads this run depends on actually answer. Reports.

        `activate_hooks()` returning is not the same as the hooks being
        up. On a real party run one client hooked and then would not
        answer for its own wizard's name or school -- it reported every
        *enemy's* school on the same read -- and the run carried on
        regardless, with a wizard it could not identify and a school it
        had to fall back to guessing. The operator's only symptom was
        that something was off, and the fix was to hook and unhook until
        it took.

        So the reads are tried, the ones that do not answer are named,
        and the hooks are re-activated between attempts. It reports
        rather than refuses: a client that answers two probes out of
        three can still fight, and being told which one is missing is
        the whole point.
        """
        for attempt in range(max(1, tries)):
            missing = []
            for name, probe, _why in self.HOOK_PROBES:
                try:
                    if await probe(seat.client) is None:
                        missing.append(name)
                except Exception:
                    missing.append(name)
            if not missing:
                if attempt:
                    self._say(seat, "hooks are answering now")
                return True, ""
            if attempt + 1 < tries:
                self._say(seat,
                          f"hooks are not answering yet ({', '.join(missing)})"
                          f" — reactivating and trying again")
                try:
                    await seat.client.activate_hooks()
                except Exception:
                    pass          # the retry is the point, not the error
                await asyncio.sleep(settle)

        why = "; ".join(w for n, _p, w in self.HOOK_PROBES if n in missing)
        self._say(seat,
                  f"hooks installed but {', '.join(missing)} will not read. "
                  f"Playing anyway, but: {why}. Closing this client "
                  f"completely and relaunching is what usually fixes a "
                  f"half-installed hook.")
        return False, ", ".join(missing)

    async def _read_max_hp(self, client, seat=None):
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
        seat = self._seat_for(client) if seat is None else seat
        try:
            hp = int(await client.stats.max_hitpoints())
        except Exception as exc:
            self._say(
                seat,
                f"could not read your max health ({type(exc).__name__}) — "
                f"training will use whatever is in the box, and a wrong "
                f"value makes the Q table share no states with the live "
                f"board")
            return          # a nicety; never worth failing the connect
        if hp > 0:
            seat.hp_known = True
            seat.max_hp = hp
            self.seat_hp_read.emit(seat.index, hp)
            if seat.index == 0:
                self.hp_read.emit(hp)

    async def _fight_outcome(self, client, seat=None):
        """True/False/None: did the fight that just ended get won?

        Twelve fights exported as "wins: 0" with `won: null` on every
        one -- including a fight that took Alicane Swiftarrow from 480
        to 40 with the killing Fire Elf going in. The combat handler
        does not report outcomes, but the client answers: a defeated
        wizard leaves the duel at zero health, a winner does not. Read
        BEFORE the between-fight upkeep runs -- a potion would launder
        a defeat into a win. A fight that recorded no rounds (a
        spurious boundary) stays unknown rather than guessed.
        """
        seat = self._seat_for(client) if seat is None else seat
        try:
            if not seat.tel.fights or seat.tel.fights[-1].rounds == 0:
                return None
            hp = await client.stats.current_hitpoints()
            return bool(hp and int(hp) > 0)
        except Exception:
            return None

    #: how far a wizard's max health may move between runs and still be
    #: the same wizard. A level-up is a few percent; the two wizards that
    #: got merged into one record were 1,053 and 713, which is 32%.
    SAME_WIZARD_HP = 0.15

    def _claim_record(self, seat):
        """Is the record this seat inherited actually this wizard's?

        The window keeps one record per SEAT and reuses it across Play
        live presses, so a record outlives the run that filled it. That
        is wanted -- fights accumulate. It stops being wanted the moment
        the seat is a different wizard, and it can be: the clients come
        back in whatever order the game was launched in, and a run where
        one client would not hook is a run where the seats shift.

        `_learn_name` catches this once both runs have names. It cannot
        catch the case that actually happened, where the FIRST run never
        got a name at all -- an empty name matches everything, so two
        rounds of a 1,053 HP ice wizard stayed in the record and were
        exported under the 713 HP fire wizard's name.

        Max health is the identity available here, before any duel. It is
        coarse -- it cannot separate two wizards of the same size -- but
        it separates the case that occurs, and it is checked before the
        first round of the new run lands in the record rather than after.
        """
        tel = seat.tel
        if not tel.rounds or not seat.max_hp:
            return
        was = tel.rounds[-1].player_max_hp
        if not was:
            return
        if abs(seat.max_hp - was) <= self.SAME_WIZARD_HP * was:
            return          # the same wizard, a level or two later
        kept = len(tel.rounds)
        who = tel.wizard or "an unnamed wizard"
        tel.clear()
        self._say(seat,
                  f"this seat's record held {kept} round(s) of {who} at "
                  f"{was:,.0f} health and this client has {seat.max_hp:,} — "
                  f"a different wizard. Cleared it rather than exporting "
                  f"two wizards as one.")

    async def _read_gear(self, client, seat=None):
        """The wizard's damage, accuracy, pierce and resist, on connect.

        Without it the simulator prices every hit as though the wizard
        were wearing nothing, and then optimises that fight instead of
        this one. A pet giving 9% is already enough to flip which move
        kills soonest -- see `live_state.read_player_stats` for the
        measurement.
        """
        from .. import live_state

        seat = self._seat_for(client) if seat is None else seat
        try:
            stats = await live_state.read_player_stats(client, seat.school)
        except Exception:
            return
        if not stats:
            self._say(
                seat,
                "could not read your gear stats — the simulator will price "
                "hits as if you were wearing none")
            return
        seat.player_stats = stats
        if seat.backend is not None:
            seat.backend.player_stats = stats
            if stats.get("power_pip_chance"):
                seat.backend.power_pip_chance = stats["power_pip_chance"]
        self.seat_gear_read.emit(seat.index, dict(stats))
        if seat.index == 0:
            self.gear_read.emit(dict(stats))
        dmg = (stats.get("damage") or {}).get(seat.school, 0.0)
        line = (f"read your gear: {dmg * 100:.0f}% {seat.school} damage, "
                f"{stats.get('pierce', 0.0) * 100:.0f}% pierce, "
                f"{stats.get('accuracy', 0.0) * 100:.0f}% accuracy")
        unread = stats.get("unread") or []
        if unread:
            # A stat that failed to read used to be folded into 0.0, so a
            # wizard in full gear was shown a confident "0% damage" that
            # then priced every hit in training below what it lands for.
            line += (f" — but {', '.join(unread)} could not be read and "
                     f"are being treated as 0")
        self._say(seat, line)

    async def _setup_questing(self, client, seat=None):
        """Prefer Deimos's questing; fall back to ours if it will not import."""
        from .. import deimos_questing

        seat = self._seat_for(client) if seat is None else seat
        ok, reason = deimos_questing.available()
        if not ok:
            self._say(seat,
                      "using the light questing — " + reason.splitlines()[0])
            return
        try:
            seat.quester = await deimos_questing.make_quester(client)
            self._say(seat, "questing: using Deimos's navigator")
        except Exception as exc:
            seat.quester = None
            self._say(seat, f"using the light questing ({type(exc).__name__})")

    async def _setup_script(self, client, seat=None):
        """Compile the party's script over every hooked client.

        Every client, not this one: deimoslang addresses wizards as
        `p1`..`p4`, and a VM built over a single client answers `None`
        for the others rather than raising, so a party script fails as
        an `AttributeError` somewhere unrelated. See `scripts.py`.
        """
        from .. import scripts

        seat = self._seat_for(client) if seat is None else seat
        seat.script_source = self.script or ""
        seat.script_said = False
        party = [s.client for s in self.seats if s.client is not None]
        try:
            seat.runner = scripts.make_runner(party or [client], self.script)
            self._say(seat, "script loaded"
                      + (f" — driving {len(party)} wizard(s)"
                         if len(party) > 1 else ""))
            names = scripts.mentions_clients(self.script)
            if names > max(1, len(party)):
                # Not a refusal -- the parts that name p3 and p4 are
                # usually behind the script's own configuration flags.
                # But not always: a `teleport client 3` that does fire
                # with two wizards hooked throws before the VM advances
                # past it, and only the stuck-instruction reload gets
                # the run back.
                self._say(seat,
                          f"this script names up to p{names} and {len(party)} "
                          f"wizard(s) are hooked. Anything it does with the "
                          f"others runs against nothing — check its own "
                          f"account settings match your party.")
        except Exception as exc:
            seat.runner = None
            self._say(seat, f"script not loaded: {exc}")

    def _follows(self, seat):
        """Is this seat a follower rather than the one setting the pace?"""
        return (self.follow_leader and len(self.seats) > 1
                and seat.index != self.leader)

    #: seconds between follow attempts. The service tick runs twice a
    #: second, and a follow is not a cheap read -- it teleports, and when
    #: the leader is mid-duel it also reaches for the nearest mob. A
    #: follower that cannot get in (the circle already seats four) would
    #: otherwise retry that twice a second for the length of the fight.
    FOLLOW_EVERY = 2.5

    async def _follow_step(self, client, seat=None):
        """One tick of keeping this wizard on the leader.

        Reported only when something actually happened. The tick runs
        twice a second and a party standing together correctly is the
        normal case, so a line per tick would bury everything else in
        the status bar.
        """
        import time

        from .. import party

        seat = self._seat_for(client) if seat is None else seat
        boss = self.seats[self.leader]
        if boss is seat or boss.client is None:
            return
        now = time.monotonic()
        if now - seat.followed_at < self.FOLLOW_EVERY:
            return
        seat.followed_at = now
        moved, why = await party.follow(client, boss.client,
                                        leader_name=boss.wizard_name)
        if moved and why:
            self._say(seat, why)
        elif why:
            # A follower that cannot reach its leader is the failure that
            # makes the whole party pointless, so it is said -- but
            # thinned, because the cause is usually standing.
            self._say_once(seat, "follow", why)

    async def _quest_step(self, client, seat=None):
        """One tick of whichever questing is in play.

        Deimos's is a *step*, not a loop: its own driver is
        `while questing_status: sleep(1); auto_quest_solo(...)`, and
        running that here would take the fight loop's ownership away.
        """
        from .. import questing

        seat = self._seat_for(client) if seat is None else seat
        if seat.quester is not None:
            ok = await seat.quester.step()
            if not ok and seat.quester.failures in (1, 10, 50):
                self._say(seat,
                          f"questing step failed ({seat.quester.failures}x): "
                          f"{seat.quester.last_error}")
            return
        # One hop per tick. The blocking hunt cannot run here -- it would
        # stall the request queue -- and running it from the fight loop
        # was the bug: that loop parks in wait_for_combat, so a hunt
        # placed before it fired once per fight and then never again.
        await questing.hop_once(
            client, on_status=lambda m: self._say(seat, m))

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

    def _build_policy(self, seat=None):
        seat = self.seats[0] if seat is None else seat
        # Cleared first: `seat.trained` drives the coverage readout, and
        # a stale one left over from a previous selection would report a
        # learned policy's numbers for a heuristic that replaced it.
        seat.trained = None
        name = seat.policy_name
        if name.startswith("trained"):
            if seat.agent is None:
                raise RuntimeError(
                    "No trained policy yet — press Train first, or pick "
                    "another policy.")
            from ..policies import trained_policy
            # Wrapped, not raw: a tabular agent has no opinion about a
            # state it never visited, and QAgent.greedy turns "no
            # opinion" into PASS. See policies.TrainedPolicy.
            seat.trained = trained_policy(seat.agent)
            return seat.trained
        if name.startswith("ttk"):
            # The tuned driver: plain lookahead, or determinized search
            # where the per-deck probes measured it ahead. The seat's own
            # quartet is passed in rather than installed globally -- see
            # `_seat_search`.
            from ..policies import tuned_driver
            base, horizon, driver = self._seat_search(seat)
            return tuned_driver(continuation=base, horizon=horizon,
                                driver=driver)
        if name.startswith("school-aware"):
            from ..policies import school_aware_blade_stack
            return school_aware_blade_stack(3)
        if name.startswith("nuke"):
            from w101_sim import strat_nuke_asap
            return strat_nuke_asap
        from w101_sim import make_blade_stack
        n = 3
        if "(" in name:
            try:
                n = int(name.split("(")[1].split(")")[0])
            except (IndexError, ValueError):
                pass
        return make_blade_stack(n)

    def _seat_search(self, seat):
        """This seat's tuned quartet, parsed but deliberately NOT installed.

        The quartet -- continuation, horizon, driver -- is deck-scoped
        and worth ~14 points of kill rate, and `policies` keeps it in
        module globals that `_rollout` reads at *decision* time. For one
        wizard that is exactly right: tune mid-run and the running fight
        picks it up. For four it is exactly wrong, because four wizards
        hold four decks and four `set_continuation` calls would leave all
        four playing whichever was written last.

        So it is parsed here and handed to `tuned_driver` explicitly,
        which binds it into that seat's closure. Returns
        `(None, None, None)` for an untuned seat, which is the signal to
        `tuned_driver` to keep reading the globals as it always has.
        """
        wire = seat.continuation
        if not wire:
            return None, None, None
        from ..policies import DEFAULT_HORIZON, build_continuation

        name, horizon, driver = wire, DEFAULT_HORIZON, None
        if " @ driver " in name:
            name, driver = name.rsplit(" @ driver ", 1)
        if " @ horizon " in name:
            name, raw = name.rsplit(" @ horizon ", 1)
            try:
                horizon = int(raw)
            except ValueError:
                pass
        self._say(seat, f"search: {name} continuation, horizon {horizon}, "
                        f"driver {driver or 'ttk'}")
        return build_continuation(name), horizon, driver

    def _on_plan(self, party):
        """The Party panel, plus one thing the panel cannot keep.

        A split party is the questing failure, seen from inside combat.
        The exports at rev 3c8b8087 have seat 2 fighting two Fire Elf
        Pathfinders end to end on its own while the other two logged no
        rounds at all -- and the only trace of it was `party_hits`
        coming back empty, which is a thing you have to already suspect
        in order to look for. So it is written down as it happens, next
        to the teleports and the rejoins it is a consequence of.
        """
        self.party_plan.emit(party)
        circles = int(getattr(party, "circles", 1) or 1)
        was, self._circles = getattr(self, "_circles", 1), circles
        if circles == was:
            return
        for seat in self.seats:
            if circles > 1:
                seat.tel.note_questing(
                    "party-split",
                    f"{len(party.moves)} wizard(s) fighting in {circles} "
                    f"separate battle circles")
            else:
                seat.tel.note_questing("party-together",
                                       "one battle circle again")

    def _make_hive(self):
        """The coordinator, when there is a party to coordinate.

        Deliberately not built for one wizard: a hive of one still costs
        a barrier, a plan and a copy of the board every round, and buys
        nothing that a single wizard could not decide on its own.
        """
        if len(self.seats) < 2 or not self.coordinate:
            return None
        from ..hivemind import Hivemind

        hive = Hivemind(passes=self.passes,
                        on_status=self.status.emit,
                        on_plan=self._on_plan)
        if self.barrier is not None:
            hive.timeout = float(self.barrier)
        for seat in self.seats:
            hive.join(seat.index, seat.name)
        return hive

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

        # Before touching the game. Picking "trained" for a wizard with
        # nothing trained is a configuration mistake, and finding it out
        # after four clients have had their hooks installed means tearing
        # all four back down again.
        for seat in self.seats:
            self._build_policy(seat)

        self.status.emit("looking for the game…" if len(self.seats) == 1
                         else f"looking for {len(self.seats)} game clients…")
        handler = ClientHandler()
        servicers = []
        self.hive = hive = self._make_hive()
        try:
            clients = handler.get_new_clients()
            if not clients:
                raise RuntimeError(
                    "No Wizard101 client found. wizwalker matches the window "
                    "class 'Wizard Graphical Client' — the game has to be "
                    "fully launched, not just the launcher.")
            if len(clients) < len(self.seats):
                # Named rather than silently played short: a party of
                # four that quietly runs as a party of two coordinates
                # perfectly and farms half as fast, which is the kind of
                # failure that goes unnoticed for an evening.
                raise RuntimeError(
                    f"{len(self.seats)} wizards are configured but only "
                    f"{len(clients)} Wizard101 client(s) are running. Launch "
                    f"one client per wizard and log each one in, or set "
                    f"'wizards' back to {len(clients)}.")

            for seat, client in zip(self.seats, clients):
                seat.client = client
            # Every client hooked before any of them is set up, so a
            # client that will not hook is reported before a backend is
            # built for any of them -- and so the focus dance below
            # happens once rather than interleaved with the reads.
            await self._activate_all_hooks()

            for seat in self.seats:
                client = seat.client
                await self._verify_hooks(seat)
                await self._read_max_hp(client, seat)
                self._claim_record(seat)
                await self._read_gear(client, seat)

                built_as = seat.policy_name
                policy = self._build_policy(seat)
                seat.tel.policy_name = seat.policy_name
                seat.tel.school = seat.school
                seat.tel.deck = seat.deck
                seat.tel.seat = seat.index
                # Before a single fight: the seat number is knowable now,
                # the wizard's name is not until a duel names it. Half an
                # answer on the title bar beats none while the operator
                # is still working out which window is which.
                self._stamp_title(seat)
                backend = WizAiBackend(
                    policy=policy, cards=cards, school=seat.school,
                    decklist=seat.deck, catalog=catalog,
                    policy_name=built_as,
                    on_decision=self._decision_hook(seat),
                    player_stats=seat.player_stats,
                    on_lost_round=self._lost_round_hook(seat),
                    seat=seat.index, coordinator=hive,
                    party_size=len(self.seats))
                backend.on_failed_cast = self._failed_cast_hook(seat)
                backend.on_school_mismatch = self._school_hook(seat)
                backend.on_defeated = self._defeated_hook(seat)
                backend.on_slow_cast = self._slow_cast_hook(seat)
                # Bound late and read per round: the rate only becomes
                # non-zero once this seat has finished a fight, and the
                # coordinator asks again every round after that.
                backend.on_recovered_cast = \
                    self._recovered_cast_hook(seat)
                backend.damage_rate = seat.tel.damage_rate
                seat.tel.resolver = backend.resolver
                seat.backend = backend
                if seat.policy_name != built_as:
                    # The dropdown moved while the hooks were installing.
                    # `set_policy` short-circuits until the backend
                    # exists, so that selection is sitting in
                    # `policy_name` unapplied.
                    self.set_policy(seat.policy_name, seat=seat.index)
                seat.combat = make_combat_handler(client, backend)

                if self.auto_quest and not self.script:
                    # A script walks the party itself; a Deimos quester
                    # built beside it is a second thing steering the
                    # same wizard.
                    await self._setup_questing(client, seat)
                if self.script and self._scripted(seat):
                    # Seat 0 only. `_setup_script` builds a VM over
                    # EVERY client, so calling it once per seat gives
                    # four VMs each driving all four wizards -- four
                    # copies of one quester, which is worse than the
                    # per-seat single-client VM it replaced.
                    await self._setup_script(client, seat)
                # Built here, not in __init__: an asyncio.Lock binds to
                # the loop it is created on, and __init__ runs on the GUI
                # thread.
                seat.upkeep_lock = asyncio.Lock()
                seat.drive = asyncio.Lock()

            await self._setup_hotkeys()
            if hive is not None:
                self.status.emit(
                    f"connected {len(self.seats)} wizards — they will agree "
                    f"each round before anyone casts")
            else:
                self.status.emit(
                    "connected — hunting for fights" if self.auto_quest
                    else "connected — walk into a fight")

            for seat in self.seats:
                servicers.append(asyncio.ensure_future(
                    self._service_loop(seat.client, seat)))
                # Its own task, so a press never waits behind a quest hop
                # or a dialogue click. See `_request_loop`.
                servicers.append(asyncio.ensure_future(
                    self._request_loop(seat.client, seat)))
            # Concurrently, on one loop. Four sequential fight loops
            # would mean wizard 2 never reaching its planning phase until
            # wizard 1's duel was over, which is not a party -- it is a
            # queue, and the barrier would time out on every round.
            await asyncio.gather(*[self._fight_loop(seat)
                                   for seat in self.seats])
            self.finished_ok.emit()
        finally:
            if self._hotkeys is not None:
                # Before anything else: a registered hotkey is taken away
                # from every other program until it is released, so it
                # must not survive a failed run.
                await self._hotkeys.stop()
                self._hotkeys = None
            for seat in self.seats:
                if seat.runner is not None:
                    seat.runner.stop()
                if hive is not None:
                    hive.leave(seat.index)
            for servicer in servicers:
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

    async def _fight_loop(self, seat):
        """One wizard's duels, start to finish.

        Every seat runs one of these, concurrently. The only thing they
        share is `_stop` and the hivemind -- upkeep, questing and the
        fight count are per client, because they are per *wizard*.
        """
        hive = self.hive
        while not self._stop and (self.fights <= 0
                                  or seat.fought < self.fights):
            # Questing of either kind runs on the service task, which
            # keeps ticking while this loop is parked in
            # wait_for_combat below.
            seat.tel.start_fight()
            try:
                # blocks until a duel starts, then plays it out
                await seat.combat.wait_for_combat()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                name = type(exc).__name__
                if not any(k in name for k in ("Memory", "ClientClosed",
                                               "ReadingEnum", "Invalidated")):
                    raise
            finally:
                if hive is not None:
                    # Out of the circle: the rest of the party must stop
                    # waiting for this seat at the barrier, or every one
                    # of their rounds pays the full timeout.
                    hive.leave_combat(seat.index)
            seat.fought += 1
            if seat.wizard_name is None:
                # A whole duel and the client still would not name its
                # own wizard. That is the read that failed on the live
                # party runs, and its symptoms turn up far from here:
                # an export with no name in it, and a follower that
                # cannot pick its leader out of a friends list.
                self._say_once(
                    seat, "no-name",
                    "this client played a whole fight without naming its "
                    "wizard, so the export will say 'wizard "
                    f"{seat.index + 1}' and a cross-zone follow has "
                    "nothing to search the friends list for. The hooks "
                    "are usually the cause — close this client "
                    "completely and relaunch it.")
            seat.tel.end_fight(await self._fight_outcome(seat.client, seat))
            self.seat_fight_done.emit(seat.index, seat.fought)
            if seat.index == 0:
                self.fight_done.emit(seat.fought)

            if seat.trained is not None:
                t = seat.trained
                self._say(seat,
                          f"trained policy: knew {t.coverage * 100:.0f}% of "
                          f"{t.seen + t.missed} boards "
                          f"({t.missed} fell back)")

            if (not self._stop and (self.collect_wisps or self.use_potions)
                    and await self._chores_can_wait(seat)):
                from .. import upkeep
                try:
                    # Under the lock, and under a flag the service
                    # task honours: a wisp sweep yields the loop
                    # every 0.15s, and the service task used to wake
                    # up inside it, decide the wizard was out of
                    # combat and free, and teleport it to the quest
                    # marker halfway through the sweep.
                    seat.in_upkeep = True
                    async with self._driving(seat, "the after-fight chores"), \
                            self._upkeep(seat):
                        await asyncio.wait_for(upkeep.after_fight(
                            seat.client, wisps=self.collect_wisps,
                            potions=self.use_potions,
                            buy=self.buy_potions,
                            on_status=lambda m: self._say(seat, m)),
                            self.AFTER_FIGHT_LIMIT)
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    self._say(seat,
                              f"the after-fight chores ran for "
                              f"{self.AFTER_FIGHT_LIMIT:.0f}s without "
                              f"finishing and were cut off — the next fight "
                              f"matters more, and they hold the wheel while "
                              f"they run")
                except Exception as exc:
                    # Still never a blocker -- but the reason is said
                    # out loud. Swallowing it silently is what made
                    # "collect wisps is not working" impossible to
                    # act on: no message, no log, no difference
                    # between broken and nothing-to-do.
                    self._say(seat,
                              f"upkeep failed — {type(exc).__name__}: {exc}")
                finally:
                    seat.in_upkeep = False
            if not self._stop:
                await self._let_it_heal(seat)
            self._say(seat,
                      f"fight {seat.fought} over — waiting for the next"
                      if not self._stop else "stopping…")

    async def _read_goal(self, seat):
        """Refresh what this wizard is doing, at most every `GOAL_POLL`.

        The quest goal, its zone and roughly where it is standing --
        enough to answer "has anything happened since last time?", which
        is what `_check_progress` needs and what nothing was asking.
        """
        import time

        from .. import questing

        now = time.monotonic()
        if now - seat.goal_read < self.GOAL_POLL:
            return
        seat.goal_read = now
        try:
            goal = await questing.read_quest_goal(seat.client)
        except Exception:
            goal = ""
        if goal and goal != seat.goal:
            seat.goal_at = now
        seat.goal = goal

        zone = position = None
        try:
            zone = await seat.client.zone_name()
        except Exception:
            pass
        try:
            at = await seat.client.body.position()
            # Rounded hard: the idle animation moves a wizard by a
            # fraction constantly, and a progress check that counts
            # breathing as progress never fires.
            position = (round(at.x / 50.0), round(at.y / 50.0),
                        round(at.z / 50.0))
        except Exception:
            pass

        where = (zone, position, seat.goal)
        if where != seat.progress:
            seat.progress = where
            seat.progress_at = now
            seat.said_stuck = ""

    def _check_progress(self, seat):
        """Say so when a running script is getting nowhere.

        The scripts hammer. Across KamarJ's three arc questers there are
        2,438 bounded retry loops and 528 unbounded ones, and `tp` alone
        appears in 331 of the unbounded ones -- because the primitive
        under it, Deimos's `navmap_tp`, returns nothing at all. Every one
        of its returns is bare, including the `if not await
        is_free(client): return` at the top, so a script cannot ask
        whether a teleport landed and has to poll the game's windows and
        try again. When that never succeeds, the loop simply runs, and
        from outside the run looks alive: instructions are being
        executed, the counter climbs, and the wizard is standing still.

        That is the shape of the operator's report -- "one wizard might
        get through, while the other is still trying to" -- and nothing
        could see it. This does not fix the hammering. It ends the
        silence, which is the part that made it hard to act on.
        """
        import time

        if seat.progress is None or not self._script_drives(seat):
            return
        idle = time.monotonic() - seat.progress_at
        if idle < self.STUCK_AFTER:
            return
        zone, _where, goal = seat.progress
        note = f"{zone or 'an unreadable zone'} · {goal or 'no quest goal'}"
        if note == seat.said_stuck:
            return
        seat.said_stuck = note
        runner = self.seats[0].runner
        steps = f"{runner.steps:,} instructions in" if runner else "the script"
        self._say(seat,
                  f"nothing has changed for {idle / 60:.0f} min — same zone, "
                  f"same spot, same quest goal ({note}) — while {steps}. "
                  f"That is a retry loop that is not working, not progress")

    def _check_in_step(self, seat):
        """Say so when the party has drifted onto different quests.

        The reported failure, and the one nothing could see: "it's not
        uncommon that one wizard will get ahead a quest or 2 from the
        other because one misses a dialogue or fails a teleport". With a
        script, one program drives both clients, so the program cannot
        notice -- its own instruction pointer is fine. What has diverged
        is the GAME's quest state, and from the moment it does, every
        instruction aimed at the wizard that fell behind is aimed at the
        wrong step.

        This does not try to fix it. Nothing here can: the wizard that
        is behind has to actually complete the step it missed, and the
        script is the only thing that knows how. What it can do is stop
        the divergence being invisible, which is what it was -- the
        operator found it by watching two windows do different things.

        Held to a grace period because turning a step in is not
        simultaneous. One wizard clicks the NPC seconds before the
        other, so a bare inequality would report a desync on every
        normal handover. Only a disagreement that PERSISTS is one.
        """
        import time

        if len(self.seats) < 2:
            return
        from .. import questing

        goals = [s.goal for s in self.seats]
        now = time.monotonic()
        if questing.goals_agree(goals):
            self._in_step_since = now
            self._said_desync = ""
            self._behind = None
            return
        since = getattr(self, "_in_step_since", None)
        if since is None:
            self._in_step_since = now
            return
        if now - since < self.DESYNC_GRACE:
            return

        # WHO is behind, not just that somebody is. Quest names have no
        # order, so "ahead" is read off the clock: the wizard whose goal
        # changed most recently is the one that advanced, and the other
        # is the one still on the step it missed.
        readable = [s for s in self.seats if s.goal]
        if len(readable) >= 2:
            self._behind = min(readable, key=lambda s: s.goal_at)

        # Named, not counted. "the party is out of sync" sends you to
        # look at two windows; this says which wizard is on what.
        where = " · ".join(f"{s.name}: {s.goal or 'unreadable'}"
                           for s in self.seats)
        if where != getattr(self, "_said_desync", ""):
            self._said_desync = where
            behind = getattr(self, "_behind", None)
            self._say(seat,
                      f"the party has been on different quests for "
                      f"{now - since:.0f}s — {where}"
                      + (f". {behind.name} is the one behind; the others "
                         f"will go back and help" if behind else "")
                      + ". The script cannot see this: its own instruction "
                        "pointer is fine, and deimoslang can only ask "
                        "whether a wizard is on a NAMED quest, never "
                        "whether two wizards are on the same one")

    #: how long a wizard may be in a different zone from the rest of the
    #: party before it counts as left behind. Zone changes are not
    #: simultaneous -- one client finishes loading seconds before
    #: another -- so a bare inequality would fire on every door.
    STRANDED_AFTER = 25.0
    #: ...and how long between attempts to bring it back, so this and
    #: the script are not pulling at once every tick.
    REJOIN_EVERY = 12.0

    async def _check_together(self):
        """Which wizard, if any, the party has left behind.

        The failure the operator is actually losing days to: "occasionally
        one wizard might get through with a teleport, but the others
        might stop teleporting or get stuck". Nothing could see it.
        `_check_in_step` compares QUEST GOALS, and a wizard whose
        teleport silently failed is still on the same quest -- its goal
        is identical, its instruction pointer is fine, and the script
        keeps issuing instructions to a wizard standing in the last
        zone. `_check_progress` does see it, after five minutes, and
        only says so.

        So this compares WHERE they are. A scripted party is one party:
        if two wizards are in Olde Town and one is in Unicorn Way, the
        one on its own is the one that missed the teleport, and the
        majority is the answer to where it should be. No majority means
        the party is mid-transition -- which is most of a zone change --
        and nothing is decided.

        Returns (stranded seat, a seat to follow) or (None, None).
        """
        import time

        from .. import party

        live = [s for s in self.seats if s.client is not None]
        if len(live) < 2:
            return None, None

        zones = {}
        for seat in live:
            zones[seat] = await party.zone(seat.client)
            if zones[seat]:
                seat.zone_seen = zones[seat]
        known = [z for z in zones.values() if z]
        if len(known) < len(live):
            return None, None            # a read failed; no evidence

        counts = {}
        for z in known:
            counts[z] = counts.get(z, 0) + 1
        best, n = max(counts.items(), key=lambda kv: kv[1])
        if n < 2 or n == len(live):
            return None, None            # no majority, or nobody adrift

        odd = [s for s in live if zones[s] != best]
        if len(odd) != 1:
            return None, None            # two adrift is a split, not a
                                         # straggler, and following one
                                         # of them could be the wrong way
        seat = odd[0]

        now = time.monotonic()
        if seat.stranded_since is None or seat.stranded_where != zones[seat]:
            seat.stranded_since = now
            seat.stranded_where = zones[seat]
            return None, None
        if now - seat.stranded_since < self.STRANDED_AFTER:
            return None, None

        target = next(s for s in live if zones[s] == best)
        return seat, target

    async def _rejoin(self, seat, target):
        """Bring the wizard the party left behind back to the party.

        `party.follow`, the same teleport a follower uses, aimed at
        whichever wizard is in the majority zone -- so it handles the
        cross-zone hop, the distance close, and stepping into the duel
        if the others are already fighting.

        This breaks the rule that a script owns every wizard it drives,
        and the justification is the same one `_should_catch_up` makes:
        the rule exists because two things walking one wizard put it in
        a doorway, and a wizard in the wrong ZONE is not being walked
        anywhere useful at all. Every instruction the script issues it
        is for a place it is not.

        Bounded three ways so it cannot become a second driver: a
        majority has to exist, the wizard has to have been adrift for
        `STRANDED_AFTER`, and attempts are `REJOIN_EVERY` apart.
        """
        import time

        from .. import party

        now = time.monotonic()
        if now - seat.rejoined_at < self.REJOIN_EVERY:
            return
        seat.rejoined_at = now
        adrift = seat.stranded_where or "somewhere else"
        seat.tel.note_questing(
            "stranded",
            f"{seat.name} is in {adrift}; {target.name} and the rest of "
            f"the party are in {getattr(target, 'zone_seen', '') or 'another zone'}")
        moved, why = await party.follow(seat.client, target.client,
                                        leader_name=target.wizard_name)
        if moved:
            seat.stranded_since = None
            seat.tel.note_questing("rejoined", why or f"went to {target.name}")
            self._say(seat, f"was left behind in {adrift} — {why}")
        elif why:
            seat.tel.note_questing("rejoin-failed", why)
            self._say_once(seat, "rejoin",
                           f"left behind in {adrift} and cannot get back — "
                           f"{why}")

    #: how often the party's whereabouts are compared. Three zone reads
    #: a tick for the life of a run is a lot of memory traffic for a
    #: question whose answer changes on the scale of a zone change.
    TOGETHER_POLL = 6.0

    def _due_to_regroup(self):
        """Is it this tick's turn to check whether the party is together?

        On the worker rather than the seat: the check reads every
        client, so running it once per seat would triple the cost and
        answer the same question three times.
        """
        import time

        now = time.monotonic()
        if now - getattr(self, "_together_at", 0.0) < self.TOGETHER_POLL:
            return False
        self._together_at = now
        return True

    async def _regroup(self):
        """Find the wizard the party left behind, and go get it."""
        seat, target = await self._check_together()
        if seat is not None:
            await self._rejoin(seat, target)

    def _should_catch_up(self, seat):
        """Should this wizard abandon its own errand and go help?

        True only for a wizard that is AHEAD while the party is
        confirmed out of step. That is a deliberate exception to the
        rule that a script owns every wizard it drives, and it is worth
        being explicit about why it is safe to break here.

        The rule exists because two things walking one wizard put it in
        a doorway. But a wizard that has moved on to its next quest is
        already being walked somewhere useless: the script's instructions
        are for the step the party is on, and this wizard is past it, so
        every `tp quest` sends it to a marker the rest of the party is
        nowhere near. The choice is not between "the script drives it"
        and "we do" -- it is between it wandering off alone and it
        standing where the fight is.

        The wizard that is BEHIND is never taken over. It is the one the
        script's instructions are actually correct for, and it is the
        one that has to finish the step; only the script knows how.
        """
        behind = getattr(self, "_behind", None)
        return (behind is not None and behind is not seat
                and len(self.seats) > 1 and behind.client is not None)

    async def _catch_up(self, seat):
        """Put this wizard back with the one that fell behind.

        `party.follow` rather than anything new: the same teleport that
        keeps a follower on its leader, aimed at the laggard instead. In
        a fight it joins the circle, which is the help that matters --
        the laggard is behind because a step went wrong, and steps go
        wrong slowest when a duel is two against four.

        Throttled by `FOLLOW_EVERY` like an ordinary follow, because the
        script is still issuing this wizard's teleports and the two will
        pull against each other. Losing that tug occasionally is fine;
        the wizard ends up near the laggard either way, which is the
        whole point.
        """
        import time

        from .. import party

        behind = getattr(self, "_behind", None)
        if behind is None or behind.client is None:
            return
        now = time.monotonic()
        if now - seat.followed_at < self.FOLLOW_EVERY:
            return
        seat.followed_at = now
        moved, why = await party.follow(seat.client, behind.client,
                                        leader_name=behind.wizard_name)
        if moved and why:
            self._say(seat, f"went back for {behind.name} — {why}")
        elif why:
            self._say_once(seat, "catch-up",
                           f"trying to get back to {behind.name} — {why}")

    async def _let_it_heal(self, seat):
        """Do not walk into the next duel on almost no health.

        `upkeep.after_fight` tops up, and when it cannot -- an empty
        potion bottle is the usual reason, and it says so -- the run went
        straight into the next fight anyway. On a dungeon that means
        dying, walking back and dying again, which is what the operator
        reported: "if they die on a dungeon they might try again
        immediately even though their health is essentially 0".

        Nothing here can conjure health. What it can do is stop, say
        why, and keep asking: Wizard101 regenerates out of combat, and a
        potion charge can arrive from a drop or a wisp sweep in the
        meantime, so `after_fight` is re-run each time round rather than
        only the health being re-read.

        It gives up after `LOW_HEALTH_WAIT` and goes anyway, loudly. A
        run that blocks forever on an empty bottle is as broken as one
        that suicides -- and unlike the suicide, it produces no
        telemetry to diagnose from.
        """
        import time

        from .. import upkeep

        started = time.monotonic()
        said = False
        while not self._stop:
            left = await self._health_left(seat)
            floor = self._health_needed(seat)
            if left is None or left >= floor:
                if said:
                    self._say(seat, f"back to {left:.0%} — carrying on"
                              if left is not None else "carrying on")
                return
            if time.monotonic() - started > self.LOW_HEALTH_WAIT:
                self._say(seat,
                          f"still on {left:.0%} health after "
                          f"{self.LOW_HEALTH_WAIT:.0f}s and nothing is "
                          f"fixing it — going into the next fight anyway, "
                          f"because a run that stops here reports nothing "
                          f"at all")
                return
            if not said:
                said = True
                self._say(seat,
                          f"on {left:.0%} health and the last few fights "
                          f"have cost up to {floor:.0%} — not starting "
                          f"another one yet")
            try:
                async with self._driving(seat, "waiting to heal up"):
                    await asyncio.wait_for(
                        upkeep.after_fight(
                            seat.client, wisps=False,
                            potions=self.use_potions,
                            buy=self.buy_potions,
                            on_status=lambda m: self._say(seat, m)),
                        self.LOW_HEALTH_POLL + upkeep.BUY_POTIONS_TIMEOUT
                        if self.buy_potions else self.LOW_HEALTH_POLL)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass                  # the health read below is the check
            await asyncio.sleep(self.LOW_HEALTH_POLL)

    def _health_needed(self, seat):
        """How much health this wizard has actually needed to survive.

        The worst fraction of its own health any of the last few fights
        took off it, floored at `LOW_HEALTH` and capped at
        `LOW_HEALTH_CAP`. "Do not start a fight you could lose to",
        measured per wizard rather than guessed once -- a level-10 fire
        wizard in Triton Avenue and the same wizard in a dungeon need
        very different numbers, and only the wizard knows which it is in.

        Capped because one catastrophic pull must not wedge the run: a
        fight that took 95% would otherwise mean never fighting again.
        Floored at the old constant so this can only ever be more
        cautious than what shipped, never less.

        `damage_taken` is an estimate (`_estimate_incoming` integrated
        over the fight), not a measurement, so this is a rough number by
        construction -- which is why it decides a *threshold* and not an
        action.
        """
        fights = [f for f in (getattr(seat.tel, "fights", None) or ())
                  if f.rounds > 0 and f.damage_taken > 0]
        if not fights:
            return self.LOW_HEALTH
        top = getattr(seat, "max_hp", 0) or 0
        if not top:
            return self.LOW_HEALTH
        worst = max(f.damage_taken / top
                    for f in fights[-self.LOW_HEALTH_WINDOW:])
        return min(max(worst, self.LOW_HEALTH), self.LOW_HEALTH_CAP)

    async def _health_left(self, seat):
        """This wizard's health as a fraction, or None if it will not read.

        None rather than a guess: refusing to start a fight because a
        stat read raised would strand a healthy wizard, and that is a
        worse failure than the one this is guarding against.
        """
        try:
            now = float(await seat.client.stats.current_hitpoints())
            most = float(await seat.client.stats.max_hitpoints())
        except Exception:
            return None
        return (now / most) if most else None

    # -- per-seat callbacks the backend fires ------------------------------
    def _decision_hook(self, seat):
        return lambda decision, read: self._on_decision(decision, read, seat)

    def _lost_round_hook(self, seat):
        return lambda number, reason: self._on_lost_round(number, reason, seat)

    def _failed_cast_hook(self, seat):
        return lambda reason: self._on_failed_cast(reason, seat)

    def _school_hook(self, seat):
        return lambda actual: self._on_school_mismatch(actual, seat)

    def _slow_cast_hook(self, seat):
        """A retry notice. Status only -- the round is still in play."""
        return lambda message: self._say(seat, message)

    def _recovered_cast_hook(self, seat):
        return lambda card, target, first: self._on_recovered_cast(
            card, target, first, seat)

    def _defeated_hook(self, seat):
        return lambda: self._say(
            seat,
            "defeated — left the party's circle so the others stop waiting "
            "for it every round, and its rounds are no longer recorded. It "
            "rejoins when this fight ends.")

    def _on_school_mismatch(self, actual, seat):
        """This client is not the wizard this seat was configured as.

        `get_new_clients()` returns windows in whatever order it finds
        them, so a party's seats and clients can be crossed -- and the
        first live party run's were. The client is the authority: it is
        the one with a wizard logged into it. So the seat is corrected to
        match it rather than the other way round, and the gear is
        re-read, because it was fetched for the wrong school. On a
        low-level wizard that re-read changes nothing -- there is no
        damage or accuracy stat yet to be keyed wrongly -- but the school
        is an input to the nuke choice, to what a power pip is worth, and
        to the board a train evaluates on, and the gear starts mattering
        the moment there is any.

        The decklist is left alone. It is the operator's to fix, it only
        feeds the scarcity feature and a trained table's keying, and the
        hand the policy actually plays is read off the game either way.
        """
        was, seat.school = seat.school, actual
        seat.tel.school = actual
        if seat.backend is not None:
            seat.backend.school = actual
        named = f" ({seat.wizard_name})" if seat.wizard_name else ""
        self._say(
            seat,
            f"this client{named} is a {actual} wizard, not the {was} it was "
            f"configured as — the clients come back in whatever order the "
            f"game was launched in, so the seats were crossed. Switched to "
            f"{actual} and re-reading the gear; until now every hit was "
            f"priced with {was}'s gear bonus, which is none of it.")
        self._stamp_title(seat)
        if seat.client is not None:
            asyncio.ensure_future(self._reread_gear(seat))

    async def _reread_gear(self, seat):
        try:
            await self._read_gear(seat.client, seat)
        except Exception as exc:
            self._say(seat, f"could not re-read the gear for {seat.school} "
                            f"({type(exc).__name__}: {exc})")

    def _on_decision(self, decision, read, seat=None):
        """Runs on the worker thread: record, then signal. No widgets."""
        seat = self.seats[0] if seat is None else seat
        if seat.wizard_name is None:
            # Before the round is recorded, not after: if this seat is
            # holding a different wizard than the record does, the record
            # has to be cleared first or this round lands in it and is
            # thrown away with the rest.
            self._learn_name(seat, read)
        sim = None
        backend = seat.backend
        if backend is not None:
            try:
                sim = backend._sim_for(read)
            except Exception:
                sim = None      # a prediction is optional, the round is not
        # The party's plan for this round goes with it: the damage model
        # settles by differencing the board, and that delta carries every
        # wizard's damage, not just this one's.
        plan = getattr(self.hive, "last_plan", None) if self.hive else None
        rec = seat.tel.observe(
            decision, read, sim=sim,
            cards=backend.cards if backend else None,
            party=plan, seat=seat.index)
        # The backend measures this every round; it used to go nowhere,
        # while the trainer guessed the same quantity off the wizard's
        # own health.
        rec.incoming = float(
            getattr(backend, "_measured_incoming", 0.0) or 0.0)
        if seat.tel.fights:
            seat.tel.fights[-1].damage_taken += rec.incoming
        self.seat_round_done.emit(seat.index, rec)
        if seat.index == 0:
            self.round_done.emit(rec)

    def _learn_name(self, seat, read):
        """Take the wizard's own name off the duel it is standing in.

        The client only offers it on the character-select screen, which
        a running wizard is not on. But `read_state` already builds the
        player actor from `combat.get_client_member().name()`, so every
        round of every duel carries it for free.

        Two things need it. The cross-zone follow, to pick the leader out
        of a friends list. And the operator: "wizard 1" and "wizard 2"
        are the window's own numbering and they mean nothing once three
        exports are sitting in a folder or four clients are on the
        taskbar. So the moment the name is known it goes onto the seat,
        into the record, onto the game window's title bar, and out to the
        window as a signal.
        """
        name = getattr(getattr(read, "state", None), "player", None)
        name = getattr(name, "name", None)
        if not (isinstance(name, str) and name.strip()):
            return
        name = name.strip()
        if seat.tel.wizard and seat.tel.wizard != name:
            # The seat is a different wizard than it was last run. The
            # clients come back in whatever order the game was launched
            # in, and the record is per SEAT and outlives a run -- so
            # without this the file holds two wizards' fights under one
            # name. The first named party run's did: two rounds of a
            # 1,053 HP ice wizard, then six of a 713 HP fire one.
            was, kept = seat.tel.wizard, len(seat.tel.rounds)
            seat.tel.clear()
            self._say(seat,
                      f"this seat was {was} last run and is {name} now — "
                      f"the clients come back in whatever order the game "
                      f"was launched in. Cleared {kept} round(s) of {was}'s "
                      f"run out of this record rather than exporting two "
                      f"wizards as one.")
            seat.tel.start_fight()
        seat.wizard_name = seat.name = name
        seat.tel.wizard = name
        if self.hive is not None:
            self.hive.join(seat.index, seat.name)
        self._stamp_title(seat)
        self.seat_named.emit(seat.index, seat.wizard_name)
        self._say(seat, f"this client is {seat.wizard_name}, the "
                        f"{seat.school} wizard")

    def _stamp_title(self, seat):
        """Write who this is onto the game window itself.

        The one place the operator is already looking. Four identical
        "Wizard101" windows on a taskbar cannot be told apart, and the
        seat numbering only exists inside this program -- so it is put
        where the mapping is needed, which is on the window. Deimos does
        the same thing for the same reason.
        """
        if not self.label_windows or seat.client is None:
            return
        who = seat.wizard_name or f"wizard {seat.index + 1}"
        try:
            seat.client.title = (f"wizAi {seat.index + 1} · {who} · "
                                 f"{seat.school}")
        except Exception:
            pass          # a window title is a nicety, never a blocker

    def _on_lost_round(self, round_number, reason, seat=None):
        """A round whose board could not be read. Recorded as that.

        It used to vanish: no row in the Decisions table, no pass
        counted, and the round after it differenced against the round
        before -- so the missing round's damage was folded into its
        predecessor's residual and scored against the damage model.
        """
        seat = self.seats[0] if seat is None else seat
        self._say(seat, reason)
        rec = seat.tel.observe_lost_round(round_number, reason)
        if rec is not None:
            self.seat_round_done.emit(seat.index, rec)
            if seat.index == 0:
                self.round_done.emit(rec)

    def _on_failed_cast(self, reason, seat=None):
        """The chosen card never went out, after the round said it had."""
        seat = self.seats[0] if seat is None else seat
        self._say(seat, reason)
        rec = seat.tel.note_failed_cast(reason)
        if rec is not None:
            self.seat_round_done.emit(seat.index, rec)
            if seat.index == 0:
                self.round_done.emit(rec)

    def _on_recovered_cast(self, card, target, first, seat=None):
        """The runner-up went out; the round was not lost after all.

        Re-emitted, not merely amended: `_on_failed_cast` has already
        pushed this round to the Decisions table reading "passed", and a
        panel that keeps showing a pass for a round that played a card
        is worse than one that never mentioned the failure.
        """
        seat = self.seats[0] if seat is None else seat
        self._say(seat, f"{first} would not go out — played {card} instead")
        rec = seat.tel.note_recovered_cast(card, target, first)
        if rec is not None:
            self.seat_round_done.emit(seat.index, rec)
            if seat.index == 0:
                self.round_done.emit(rec)
