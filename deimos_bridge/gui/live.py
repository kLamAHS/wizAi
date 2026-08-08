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
        #: every goal this wizard has held, in order. A wizard still on
        #: a step another has already left is the one behind -- see
        #: `LiveWorker._who_is_behind`.
        self.goals_seen = []
        #: since when this wizard's tracker has been on something that
        #: is not the world's main line, and the last side quest that
        #: was said. See `LiveWorker._check_on_questline`.
        self.off_line_since = None
        self.said_off_line = ""
        #: the tracked quest's NAME, which is what `questlist` can place
        #: in a questline. The goal cannot be placed: "Talk to Professor
        #: Winthrop" is the objective of nine Krokotopia quests spanning
        #: main #2 to #19. See `LiveWorker._behind_by_questline`.
        self.quest_name = ""
        #: when the goal last CHANGED. The wizard whose goal moved most
        #: recently is the one that got ahead -- quest names have no
        #: order, but "who advanced last" does.
        self.goal_at = 0.0
        #: since when this wizard has been standing at a press-X prompt
        #: that is not the prompt the rest of the party is standing at.
        #: See `LiveWorker._check_same_sigil`.
        self.apart_since = None
        #: fruitless presses of X from one spot, and which spot. See
        #: `LiveWorker._x_did_nothing_here`.
        self.x_pressed = 0
        self.x_pressed_at = None
        #: whether this wizard was in a duel on its last service tick.
        #: Read from the OTHER seats' ticks -- see `CATCH_UP_IDLE`.
        self.in_duel = False
        #: {world: the furthest place in that world's line this wizard
        #: has reached}. Quests are finished in order, so a later read
        #: that is EARLIER is a tracker on a side quest or an ambiguous
        #: goal line -- never the wizard going backwards. See
        #: `LiveWorker._no_further_back`.
        self.furthest = {}
        #: (zone, rounded position, goal) as last seen, and when it last
        #: CHANGED. A script that is stepping while none of these move is
        #: a script hammering something that is not working -- see
        #: `LiveWorker._check_progress`.
        self.progress = None
        self.progress_at = 0.0
        self.said_stuck = ""
        #: when this seat last wrote a heartbeat to the questing log.
        #: See `LiveWorker._heartbeat`.
        self.beat_at = 0.0
        #: when a wedged scripted wizard was last looked at, and the
        #: script's instruction count then. A count that has not moved
        #: between two looks means the script is inside `wait_for_coro`.
        #: See `LiveWorker._unstick`.
        self.unstuck_at = 0.0
        self.steps_seen = None
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
        #: when `zone_seen` last CHANGED.
        self.zone_since = 0.0
        #: zones this seat has already been rejoined INTO, and when.
        #: The backstop against dragging a wizard in a circle.
        self.rejoin_history = []
        #: zones this seat has LEFT, and when it left them. Somewhere a
        #: wizard has just walked out of is not somewhere it was left
        #: behind. See `LiveWorker._check_together`.
        self.zone_left = []
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
        #: the catch-up in progress, or None. See `_start_catching_up`.
        self._catch_up_state = None
        #: {seat id: (the step it gave up on, when)}. A catch-up that
        #: gives up has to be REMEMBERED, or `_check_in_step` starts the
        #: identical one on the next tick -- see `_written_off`.
        self._wrote_off = {}
        #: step keys whose write-off has already been written to the
        #: exports. The verdict does not change while the step does not.
        self._said_written_off = set()
        #: whether the VM's give-up hook has been installed. Once per
        #: worker, not once per seat: it is module-level in the VM and
        #: fires for every wizard the script drives. See
        #: `LiveWorker._watch_waitfor`.
        self._waitfor_hooked = False

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
                fighting = await questing.in_battle(client)
                # Kept on the seat, because the question "is anybody in
                # this catch-up actually fighting" is asked from ANOTHER
                # seat's tick -- a wizard in a duel never reaches
                # `_check_caught_up` on its own loop. See `CATCH_UP_IDLE`.
                seat.in_duel = bool(fighting)
                # BEFORE the guard below. Konstantin's log has one
                # heartbeat at t=0 and the next at t=531.7 -- nine
                # missed beats, because he spent them in a fourteen-round
                # duel and every stage of this tick is skipped while a
                # wizard is in combat. A wizard wedged inside a fight is
                # exactly as stuck as one wedged outside it, and that
                # nine-minute hole was the one place the timeline could
                # not account for.
                await self._heartbeat(seat, self._script_drives(seat),
                                      fighting=fighting)
                if fighting or seat.in_upkeep:
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
                self._watch_waitfor(seat)

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
                # Ends any catch-up BEFORE deciding whether to start
                # one. The other order meant a catch-up started this
                # tick was judged finished on the same tick, by state
                # that had not had a chance to change -- which is
                # exactly what the first live run shows, `started` and
                # `done` sharing a timestamp.
                self._check_caught_up()
                self._check_in_step(seat)
                self._check_on_questline()
                self._check_progress(seat)

                catching = self._catching_up()
                if seat.runner is not None and not catching:
                    await self._stage(seat, "script step",
                                      self._script_step(seat), wheel=True)
                elif seat.runner is not None:
                    # Paused, not fought with. While the party is out of
                    # step the script's instructions are wrong for every
                    # wizard in it -- the laggard's most of all, since
                    # each `tp quest` drags it back to its own marker --
                    # so running them is worse than not running them.
                    self._say_once(
                        seat, "script-paused",
                        f"script paused while "
                        f"{' and '.join(c.name for c in catching)} "
                        f"finish the step they missed")

                if seat in catching:
                    # The whole point. This wizard is not followed
                    # anywhere; it is driven through its own step until
                    # its quest state catches the party's.
                    await self._stage(seat, "finishing the missed step",
                                      self._quest_the_missed_step(seat),
                                      wheel=True)
                    await asyncio.sleep(0.5)
                    continue

                if driven:
                    # Unconditional, ahead of the chain below: a wedged
                    # script needs looking at whether or not this tick
                    # also decided to catch up or regroup.
                    await self._stage(seat, "unwedging a stuck script",
                                      self._unstick(seat), wheel=True)

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
                f"behind it.",
                kind="stage-timeout",
                detail=f"{name} cut off after {limit:.0f}s")
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
                       f"{name} failed — {type(exc).__name__}: {exc}",
                       kind="stage-failed",
                       detail=f"{name}: {type(exc).__name__}: {exc}")

    #: how many repeats of the same stage failure before the questing log
    #: says so again. The status line thins out at 20 because it is read
    #: live; the export is read afterwards, and one entry per 60 is
    #: enough to show a stall lasting minutes without burying the rest.
    STUCK_EVERY = 60

    def _say_once(self, seat, key, message, kind="", detail=""):
        """Say it the first time, then every 20th -- twice a second is spam.

        The service tick runs twice a second, so a stage that is broken
        rather than unlucky would fill the status bar with one line and
        nothing else. Reporting the first and then thinning out keeps
        the failure visible without burying everything around it.

        `kind` also writes it to the questing log, and that is the half
        that was missing. A stage that times out on EVERY tick was
        announced once and then never again, anywhere -- so a wizard
        wedged for ten minutes and a wizard working normally produced
        identical exports. The operator's report was "it's really
        stuck", and the run at rev 85a68184 has 99 combat rounds and
        eight questing entries to explain the rest of the session.
        """
        seat = self.seats[0] if seat is None else seat
        n = seat.stage_errors.get(key, 0) + 1
        seat.stage_errors[key] = n
        if n == 1 or n % 20 == 0:
            self._say(seat, message + (f" (still failing after {n} tries)"
                                       if n > 1 else ""))
        if kind and (n == 1 or n % self.STUCK_EVERY == 0):
            try:
                seat.tel.note_questing(
                    kind, detail + (f" — {n} times in a row" if n > 1 else ""))
            except Exception:
                pass

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
            # Into the RUN LOG, not just the status line. Rev 8e5a9c75
            # spent its last forty minutes with three wizards standing
            # still because the script's account names were never filled
            # in, so its own friend-teleports -- the only thing it has
            # for putting a party back together -- were skipped by its
            # own guards. That is checkable in the first thirty lines of
            # the file, and it belongs where somebody reads it after the
            # run as well as before.
            # Only the slots this party HAS. `Questee4` in a party of
            # three is the script's fourth seat sitting empty, which is
            # what it is meant to do -- see `scripts.unfilled`.
            blanks = scripts.unfilled(self.script, len(party))
            if blanks and len(party) > 1:
                said = (f"the script has {len(blanks)} setting(s) still at "
                        f"its placeholder value ("
                        + ", ".join(n for n, _v in blanks[:6])
                        + (" …" if len(blanks) > 6 else "")
                        + "). While the account names are unset the script "
                          "skips its own friend-teleports, so a wizard that "
                          "falls behind cannot be pulled back by it")
                for other in self.seats:
                    try:
                        other.tel.note_questing("script-unconfigured", said)
                    except Exception:
                        pass
                self._say(seat, said)
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
                backend.on_round_timing = self._round_timing_hook(seat)
                # Bound late and read per round: the rate only becomes
                # non-zero once this seat has finished a fight, and the
                # coordinator asks again every round after that.
                backend.on_recovered_cast = \
                    self._recovered_cast_hook(seat)
                backend.on_recovery_failed = \
                    self._recovery_failed_hook(seat)
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
            await self._unhook(handler)

    async def _unhook(self, handler):
        """Release every client's hooks, one at a time, and say so.

        `ClientHandler.close` is a bare loop over `client.close()` with
        no guard around each one (`client_handler.py:122`), so the first
        client that throws leaves every client after it still hooked --
        and a hooked client that wizAi is no longer driving is exactly
        the state that forces Wizard101 to be restarted.

        So each is closed on its own, and the outcome is said out loud
        per wizard. "disconnected" was the only thing this used to
        report and it is the one fact that was never in doubt; what the
        operator needs to know before pulling a new build is whether
        the game can be left running.
        """
        clients = list(getattr(handler, "clients", None) or ())
        freed, stuck = [], []
        for seat, client in zip(self.seats, clients):
            name = seat.name if seat is not None else "a wizard"
            try:
                await client.close()
                freed.append(name)
            except Exception as exc:
                stuck.append(f"{name} ({type(exc).__name__}: {exc})")
        # Anything the seats did not cover -- a client that connected
        # but never got a seat. Still hooked, still has to be released.
        for client in clients[len(self.seats):]:
            try:
                await client.close()
                freed.append("an extra client")
            except Exception as exc:
                stuck.append(f"an extra client ({type(exc).__name__})")
        if stuck:
            self.status.emit(
                f"unhooked {len(freed)} of {len(freed) + len(stuck)} — "
                + "; ".join(stuck)
                + ". Those clients have to be closed and reopened before "
                  "wizAi can attach to them again")
        elif freed:
            self.status.emit(
                f"unhooked {len(freed)} client(s) — Wizard101 can stay "
                f"open. Pull and relaunch wizAi when ready")
        else:
            self.status.emit("disconnected — nothing was hooked")

    #: how often a fight loop parked between duels looks up to see
    #: whether the run has been asked to stop. Half a second is the
    #: service tick's own cadence and is imperceptible against a duel.
    STOP_POLL = 0.5

    async def _wait_for_combat(self, seat):
        """Wait for a duel. False if the run was stopped while waiting.

        `CombatHandler.wait_for_combat` polls `in_combat` forever
        (`wizwalker/combat/handler.py:64`) and `stop()` only sets a
        flag, so a loop parked here never looked at it. Between fights
        -- which is most of a questing run -- pressing Stop therefore
        did nothing at all until the next duel started AND finished,
        and if the party was wedged that was never.

        That is why hooks were being stranded. The run could not be
        ended, so the window was closed instead; the worker thread died
        where it stood; and `_go`'s teardown, which is the only thing
        that unhooks, never ran. Wizard101 then had to be restarted
        before wizAi could attach to it again -- which is a workflow
        cost paid on every single code change.
        """
        waiting = asyncio.ensure_future(seat.combat.wait_for_combat())
        try:
            while not self._stop:
                done, _pending = await asyncio.wait(
                    {waiting}, timeout=self.STOP_POLL)
                if done:
                    await waiting                  # re-raise what it hit
                    return True
        finally:
            if not waiting.done():
                # Cancelled part-way through `handle_combat`, which is
                # exactly what a stop is: the duel carries on without
                # us and the next launch picks it up.
                waiting.cancel()
                try:
                    await waiting
                except BaseException:
                    pass
        return False

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
                # blocks until a duel starts, then plays it out -- but
                # looks up while it waits, so Stop is answered between
                # fights and not only during one. See `_wait_for_combat`.
                if not await self._wait_for_combat(seat):
                    break
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
            # In order, and bounded. `_who_is_behind` needs to know that
            # a step was HELD and left, which a single current value
            # cannot say.
            seat.goals_seen.append(goal)
            del seat.goals_seen[:-12]
        seat.goal = goal

        # The quest NAME, alongside the goal and on the same poll. It is
        # what `questlist` can place in a questline; the goal cannot be
        # placed reliably, because one goal line belongs to as many as
        # nine quests seventeen steps apart. Failing to read it leaves
        # the previous value rather than clearing it -- a blank read is
        # not evidence the wizard changed quest, and `_behind_by_
        # questline` would silently drop that seat out of the comparison.
        try:
            name = await questing.read_quest_name(seat.client)
        except Exception:
            name = ""
        if name:
            seat.quest_name = name

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

    #: seconds between heartbeat entries per wizard. A minute is fine
    #: enough to see a stall start and end, and coarse enough that an
    #: hour of three wizards is 180 entries.
    HEARTBEAT_EVERY = 60.0

    async def _heartbeat(self, seat, driven=False, fighting=None):
        """One line a minute saying where this wizard is and what it is on.

        Everything else in the questing log is an alarm, and alarms only
        describe the moments somebody already wrote code to notice. The
        report this exists for -- "it's really stuck" -- is about the
        time BETWEEN them, and the run at rev 85a68184 had 99 combat
        rounds and eight questing entries to account for a whole
        session. There was no way to tell a wizard grinding fights from
        a wizard standing in a doorway.

        So: zone, health, whether it is in a duel, its quest goal, how
        long since any of that last moved, and what is driving it. A
        stall is then obvious by inspection -- the same line repeating
        with a climbing idle time -- without needing a detector to have
        anticipated its cause.

        Cheap on purpose. The zone and goal were already read this tick;
        only health and the duel flag are fetched here, once a minute,
        which is less memory traffic than a single round of combat.
        """
        import time

        from .. import party

        now = time.monotonic()
        if now - seat.beat_at < self.HEARTBEAT_EVERY:
            return
        seat.beat_at = now

        # Nothing a report does may take the run down. This is the whole
        # body, not just the write, and it is not defensiveness for its
        # own sake: the first version read `runner.steps` unguarded, and
        # a runner without that attribute raised straight out of the
        # service tick and killed the loop for that wizard -- turning
        # the instrument for diagnosing a stuck wizard into a cause of
        # one. A heartbeat that occasionally says less is fine; a
        # heartbeat that can stop the tick is not.
        try:
            # Read here when nothing else has. `zone_seen` is filled by
            # the stranded poll, and that only runs for a party whose
            # script is driving -- so every log at rev 1d28f745 says
            # "zone unread" for its first eight minutes, across the two
            # quest desyncs and four fights. The zone is the first thing
            # anybody reads on one of these lines.
            if not seat.zone_seen:
                try:
                    seat.zone_seen = await party.zone(seat.client)
                except Exception:
                    pass
            bits = [seat.zone_seen or "zone unread"]
            left = await self._health_left(seat)
            if left is not None:
                bits.append(f"{left:.0%} health")
            try:
                if fighting is None:
                    fighting = await party.in_battle(seat.client)
                if fighting and left is not None and left <= 0:
                    # A wizard on 0% is not fighting, whatever the duel
                    # phase says. Rev bb8f2b3c has four consecutive
                    # minutes of "0% health · in a duel" after the
                    # `defeated` entry, which reads as a wizard grinding
                    # a long fight when it is a corpse waiting for the
                    # allies to finish -- the one state where the script
                    # legitimately sits still, and the one the log made
                    # look like the bug.
                    bits.append("defeated, waiting for the duel to end")
                else:
                    bits.append("in a duel" if fighting else "out of combat")
            except Exception:
                bits.append("combat state unread")
            bits.append(seat.goal or "no quest goal")
            if seat.progress_at:
                idle = now - seat.progress_at
                bits.append(f"unchanged for {idle / 60:.0f} min"
                            if idle >= 60 else "moving")
            steps = getattr(self.seats[0].runner, "steps", None)
            if driven and isinstance(steps, int):
                bits.append(f"script at {steps:,} instructions")
            elif driven:
                bits.append("script driving")
            seat.tel.note_questing("heartbeat", " · ".join(bits))
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    def _watch_waitfor(self, seat):
        """Put a `waitfor` that gave up into the run's log.

        The VM's waits are bounded now rather than infinite (see
        `WAITFOR_TIMEOUTS` in `Deimos/src/deimoslang/vm.py`), and a
        bounded wait that gives up silently is only half a fix: the
        script falls through into a retry loop and the run looks normal
        again, so the thing that actually went wrong has to be said.

        Installed once, on whichever seat owns the runner, and it is the
        VM's own module-level hook -- so it fires for every wizard the
        script drives, not just this one.
        """
        if self._waitfor_hooked or self.seats[0].runner is None:
            return
        self._waitfor_hooked = True
        from .. import scripts

        def gave_up(kind, seconds, inverted):
            what = f"waitfor {kind}" + (" completion" if inverted else "")
            for other in self.seats:
                try:
                    other.tel.note_questing(
                        "waitfor-gave-up",
                        f"`{what}` polled for {seconds:.0f}s and gave up — "
                        f"the script has fallen through to whatever comes "
                        f"next")
                except Exception:
                    pass
            self._say(self.seats[0],
                      f"`{what}` waited {seconds:.0f}s and gave up. Upstream "
                      f"that wait has no timeout at all and would have "
                      f"blocked the script for good")

        def whose(client):
            """The name of the wizard a hook is talking about.

            The hooks are module-level, so they fire for whichever
            wizard the script was driving -- and without this every
            entry read "a scripted teleport did not land" with no way
            to tell which of four wizards it was. Nineteen of those in
            one run's log named nobody at all.
            """
            if client is None:
                return ""
            for other in self.seats:
                if other.client is client:
                    return other.name
            return ""

        def teleported(landed, how, zone, client=None):
            # Only the failures. A run makes thousands of these and the
            # ones that worked are what the `zone` entries already show.
            if landed:
                return
            name = whose(client)
            said = (f"{name}'s scripted teleport did not land" if name
                    else "a scripted teleport did not land")
            for other in self.seats:
                try:
                    other.tel.note_questing(
                        "teleport-failed",
                        f"{said} — {how}" + (f" (in {zone})" if zone else ""))
                except Exception:
                    pass
            self._say_once(
                self.seats[0], "tp-failed",
                f"{said} — {how}. Upstream this returns nothing at all, so "
                f"the script cannot tell and its next instruction is for "
                f"wherever it meant to be")

        def tp_noted(message, client=None):
            name = whose(client)
            said = f"{name}: {message}" if name else message
            for other in self.seats:
                try:
                    other.tel.note_questing("teleport-forced", said)
                except Exception:
                    pass
            self._say_once(self.seats[0], "tp-forced", said)

        def party_task_failed(what, failures):
            """Wizards that fell out of an instruction the others finished.

            Upstream this was not merely unreported, it was unsurvivable:
            `asyncio.TaskGroup` cancels every sibling when one task
            raises, so the party's instruction died with whichever
            wizard was unluckiest. Now they all finish and this says who
            did not.
            """
            for label, exc in failures:
                said = (f"`{what}` failed for one wizard "
                        f"({type(exc).__name__}: {exc}) — the others "
                        f"finished it, so the party is a step apart until "
                        f"something puts them back together")
                for other in self.seats:
                    try:
                        other.tel.note_questing("party-task-failed", said)
                    except Exception:
                        pass
                self._say_once(self.seats[0], f"party-task-{what}", said)

        installs = ((scripts.on_waitfor_timeout, "waitfor-hook", gave_up),
                    (scripts.on_teleport_result, "teleport-hook", teleported),
                    (scripts.on_teleport_note, "teleport-note", tp_noted),
                    (scripts.on_party_task_failed, "party-task-hook",
                     party_task_failed))
        for install, key, hook in installs:
            ok, why = install(hook)
            if not ok:
                self._say_once(self.seats[0], key, why,
                               kind="stage-failed", detail=why)

    #: how often a wedged scripted wizard is looked at. The condition it
    #: is waiting on changes on the scale of a conversation, not a tick.
    UNSTICK_EVERY = 30.0

    async def _unstick(self, seat):
        """Unwedge a scripted wizard, in the direction its own wait needs.

        The reason wizAi keeps its hands off a scripted wizard's
        dialogue is concrete, and it is in `command_parser.py:67`::

            async def wait_for_coro(coro, wait_for_not=False, interval=0.25):
                while not await coro():
                    await asyncio.sleep(interval)

        `waitfordialog` is that, polling `is_in_dialog` every 250ms with
        no timeout of any kind. Click the box away between two polls and
        the script's wait never satisfies -- not late, never -- so an
        override that always clears dialogue turns a ten-minute stall
        into a permanent one.

        But the same shape says which action IS safe, because the
        script's instruction counter separates the two states from
        outside. Parked on one instruction means it is inside
        `wait_for_coro`, waiting for something to become true. Climbing
        means it is running a retry loop and waiting for nothing.

        Straight off seat 2's own heartbeats at rev 07ef3fa7::

            361.1   11,582   climbing
            421.1   11,582   +0      parked
            481.6   11,582   +0      parked
            542.0   11,830   +248    climbing
            ...
            903.7   15,485   +588    climbing
            964.0   24,656   +9,171  broke out

        Both wedges open the same way -- two minutes parked, then eight
        of a retry loop -- so:

          * Box open, script parked past a full look: it is in
            `waitfordialog completion`, waiting for the box to CLOSE.
            Clearing it satisfies the wait rather than racing it.
          * Box open, script looping: nothing is waiting on it, so
            clearing it cannot deadlock anything.
          * No box, script parked: it is waiting for a box that is not
            coming. Pressing X makes its condition true -- the one case
            where an unrequested interact is exactly what the script is
            asking for. Gated on something being in range, so it is a
            press at an NPC and not into empty air.
          * No box, script looping: not a dialogue problem. Reported,
            nothing touched.

        Never before `STUCK_AFTER`, so an ordinary conversation is never
        raced, and every read is written down either way.
        """
        import time

        from .. import questing

        if seat.progress is None:
            return
        now = time.monotonic()
        if now - seat.progress_at < self.STUCK_AFTER:
            seat.unstuck_at = 0.0
            seat.steps_seen = None
            return
        if now - seat.unstuck_at < self.UNSTICK_EVERY:
            return
        seat.unstuck_at = now

        steps = getattr(self.seats[0].runner, "steps", None)
        was, seat.steps_seen = seat.steps_seen, steps
        # Only once there is a previous number to compare against.
        # "Parked" is unknowable on the first sample, and guessing it
        # would press X at a script that is merely slow.
        parked = (was is not None and steps is not None and steps == was)

        open_box = await questing.in_dialogue(seat.client)
        near = await questing.near_interactable(seat.client)
        at_marker, why = await questing.at_quest_marker(seat.client)
        seat.tel.note_questing(
            "stuck-detail",
            f"script {'parked on one instruction' if parked else 'looping'}"
            f" · dialogue {'open' if open_box else 'closed'}"
            f" · {'an interactable in range' if near else 'nothing in range'}"
            f" · {'at the quest marker' if at_marker else (why or 'not at the marker')}")

        mins = (now - seat.progress_at) / 60.0
        if open_box:
            n, click_why = await questing.advance_dialogue(seat.client)
            seat.tel.note_questing(
                "unstuck-dialogue",
                f"cleared {n} window(s) the script had left open"
                if n else (click_why or "the box would not click"))
            self._say(seat,
                      f"stuck {mins:.0f} min with a dialogue box open — "
                      f"cleared {n} window(s). wizAi normally leaves a "
                      f"scripted wizard's dialogue alone; this one was "
                      f"measurably not being handled, and it blocks movement")
            return

        if parked and near and self._x_did_nothing_here(seat, at_marker):
            # Already tried, here, and nothing came up. Rev 3822cc6c
            # pressed X five times over twenty-four minutes while
            # Konstantin stood 22,778 units from `Talk To Professor
            # Winthrop in Altar of Kings` -- a different zone -- and
            # every one of them logged "a box did not come up". The
            # first press is worth making; the fifth is a gesture at a
            # passing vendor, and it buries the fact that this wizard is
            # nowhere near the thing it is waiting for.
            seat.tel.note_questing(
                "unstuck-x-does-nothing",
                f"X has been pressed here {seat.x_pressed} time(s) and no "
                f"box has come up — {why or 'at the quest marker'}. Not "
                f"pressing it again from this spot")
            self._say(seat,
                      f"stuck {mins:.0f} min and X does nothing from here "
                      f"({why or 'at the quest marker'}) — this wizard needs "
                      f"moving, not interacting")
            return

        if parked and near:
            ok, press_why = await questing.press_x(seat.client)
            opened = await questing.dialogue_opened(seat.client)
            self._remember_x(seat, opened)
            seat.tel.note_questing(
                "unstuck-pressed-x",
                (f"the script was parked waiting for a dialogue that never "
                 f"opened; pressed X and a box "
                 f"{'came up' if opened else 'did not come up'}")
                if ok else (press_why or "could not press X"))
            self._say(seat,
                      f"stuck {mins:.0f} min with the script parked on one "
                      f"instruction and no dialogue open — pressed X. "
                      f"`waitfordialog` polls until a box exists and never "
                      f"times out, so making one exist is what it waits for")

    #: how many fruitless presses of X from one spot before wizAi stops
    #: offering. One is worth making -- `waitfordialog` polls forever
    #: and a box has to be made to exist. Two says the thing in range is
    #: not the thing the script is waiting for.
    X_TRIES_HERE = 2

    def _x_did_nothing_here(self, seat, at_marker):
        """Has X already been pressed from this spot for nothing?

        Keyed on the position stamp, so a wizard that moves gets a fresh
        pair of attempts -- and a wizard that has not moved does not get
        a fifth.
        """
        return (seat.x_pressed >= self.X_TRIES_HERE
                and seat.x_pressed_at == (seat.progress or (None, None, None)))

    def _remember_x(self, seat, opened):
        """Count a press that produced nothing, and forget one that did."""
        here = seat.progress or (None, None, None)
        if opened:
            seat.x_pressed, seat.x_pressed_at = 0, None
            return
        if seat.x_pressed_at != here:
            seat.x_pressed, seat.x_pressed_at = 0, here
        seat.x_pressed += 1

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
        # Keyed on the SITUATION and on how long it has lasted, not on
        # the situation alone. Keyed on the situation alone it is said
        # once and never again: rev 8e5a9c75 stood still for forty
        # minutes and this fired exactly once, at the five-minute mark,
        # while `stuck-detail` said the same sentence 69 times. A stall
        # that is still going after half an hour is a different fact
        # from one that has just started.
        note = f"{zone or 'an unreadable zone'} · {goal or 'no quest goal'}"
        # 5, 10, 20, 40, 80 minutes -- rare enough not to bury the log,
        # often enough that the export shows the stall growing.
        band = 1
        while band * 2 * self.STUCK_AFTER <= idle:
            band *= 2
        stamp = f"{note} @{band}"
        if stamp == seat.said_stuck:
            return
        seat.said_stuck = stamp
        runner = self.seats[0].runner
        steps = f"{runner.steps:,} instructions in" if runner else "the script"
        # Built separately rather than edited out of `steps`. The first
        # version did `steps.replace(' in', '')` to drop the trailing
        # "in", and " in" also occurs inside "instructions" -- so the
        # live log said "while 12,620structions ran".
        count = (f"{runner.steps:,} instructions" if runner
                 else "the script")
        self._say(seat,
                  f"nothing has changed for {idle / 60:.0f} min — same zone, "
                  f"same spot, same quest goal ({note}) — while {steps}. "
                  f"That is a retry loop that is not working, not progress")
        # This is THE stuck report, and it only ever went to the status
        # line -- which is read live, by somebody who is already watching
        # because something looks wrong. The export is what gets read
        # afterwards, and it had no entry for the one condition it most
        # needed to explain.
        seat.tel.note_questing(
            "no-progress",
            f"{idle / 60:.0f} min with no change — {note} — while "
            f"{count} ran")

    def _places(self):
        """Where every seat sits in the questline, one `Position` each.

        ONE function because there were two, and they disagreed. The
        rule that STARTS a catch-up placed a wizard by quest name and
        fell back to its goal line; the rule that ENDS one placed by
        quest name only. Live at rev 7888c35a the names did not read at
        all, so the first found a five-quest gap from the goals and the
        second found nothing to compare -- `catch-up-started` and
        `catch-up-done` are in that export at the same timestamp, on all
        three wizards.

        A start condition and a stop condition computed two different
        ways is a bug waiting for the day they differ, and that day was
        the first run.
        """
        from .. import questlist

        places = [questlist.position_of(s.quest_name) if s.quest_name
                  else questlist.Position() for s in self.seats]
        # A wizard whose name would not read can still be placed from
        # its goal, IF the goal is unambiguous. The rest of the party
        # says roughly where to look -- "Talk To Lieutenant Standish in
        # Palace of Fire" is Krokotopia #12 AND #13, and only a
        # neighbour tells you which.
        #
        # Repeated until it stops helping, because one pass makes the
        # answer depend on seat order: a wizard placed from its goal
        # then becomes a hint for the next one, and in a single pass
        # only the wizards after it get the benefit. Rev 7888c35a
        # placed wizard 1 at #12 purely because wizard 2's NAME
        # happened to read and seed it; had the two been swapped, the
        # party would have been unplaceable.
        for _pass in range(len(self.seats)):
            known = [p.order for p in places if p.comparable]
            if not known:
                # Nothing to hint with, so goals can only be placed on
                # their own merits. One attempt, then give up.
                near = world = None
            else:
                near = sum(known) / len(known)
                world = next(p.world for p in places if p.comparable)
            moved = False
            for i, seat in enumerate(self.seats):
                if places[i].comparable or not seat.goal:
                    continue
                found = questlist.position_from_goal(seat.goal, world, near)
                if found.comparable:
                    places[i] = found
                    moved = True
                elif not places[i].how:
                    # Keep the reason even when it did not place, so
                    # `_how_placed` can say WHY rather than "no quest".
                    places[i] = found
            if not moved:
                break
        return [self._no_further_back(seat, place)
                for seat, place in zip(self.seats, places)]

    def _no_further_back(self, seat, place):
        """A wizard cannot un-finish a quest. Clamp a backwards read.

        The questline is completed in order, so a wizard that has been
        placed at Krokotopia #15 is at #15 or later forever. A read that
        says #7 is not the wizard moving backwards -- it is the tracker
        having switched to a side quest, or a goal line that eight
        quests happen to share. Both are common and neither is a lag.

        Rev 3822cc6c is what this costs. Sebastian, in step with the
        party all run, read `Talk To Robert Lancaster in Chamber of
        Fire` for one tick -- a real goal, belonging to Krokotopia #7 --
        and the party declared an EIGHT quest gap and paused the script
        to catch him up. Fifteen seconds later he read #15 again, which
        is where he had been the whole time.

        Per world, because the order restarts at every world boundary
        and Marleybone #1 is not behind Krokotopia #15.
        """
        if not place.comparable:
            return place
        best = seat.furthest.get(place.world)
        if best is None or place.order > best:
            seat.furthest[place.world] = place.order
            return place
        if place.order == best:
            return place
        # Backwards. Keep the wizard where it has actually got to, and
        # say in `how` that the read disagreed -- a silent clamp would
        # hide a genuine questline reset behind a number that never
        # moves.
        from .. import questlist

        return questlist.Position(
            world=place.world, order=best, name=place.name,
            area=place.area, questline=place.questline,
            how=f"{place.how}, clamped — it read #{place.order}, "
                f"which is behind #{best} it has already done")

    def _how_placed(self):
        """One short line saying how each wizard was located, for the log.

        The first live run placed all three wizards by goal text, which
        means `read_quest_name` returned nothing for any of them -- and
        nothing in the export said so. Placement is now the thing two
        rules and a script pause hang off, so how it was arrived at is
        worth a few characters.
        """
        bits = []
        for seat, place in zip(self.seats, self._places()):
            where = f"#{place.order}" if place.comparable else "unplaced"
            bits.append(f"{seat.name} {where} ({place.how or 'no quest'})")
        return " · ".join(bits)

    #: how long a wizard may be off the main line before it is said.
    #: Picking a side quest up is normal and often deliberate -- the
    #: scripts do it for training points and gear. Staying on one is
    #: not, because the tracker follows the SELECTED quest and every
    #: `tp quest` goes there until something changes it.
    OFF_QUESTLINE_AFTER = 120.0

    def _check_on_questline(self):
        """Notice a wizard whose tracker has wandered off the storyline.

        The operator's request, and it is a real failure mode rather
        than a tidiness one:

            it would be beneficial since the bot sometimes loses the
            main questline to realize when it's lost it

        Wizard101's quest arrow follows whichever quest is SELECTED. A
        wizard that accepts a side quest -- or has one auto-selected
        after a dialogue -- has its arrow, and therefore every `tp
        quest` the script issues, pointed at the side quest from that
        moment. The party's main-line progress simply stops, and until
        now nothing said so: rev 8e5a9c75 has Sebastian going from
        Krokotopia main #13 to `unplaced` at t=790 and running the
        remaining seven minutes there, visible only as three characters
        in a placement string nobody was reading for that.

        Only said once per change, and only after `OFF_QUESTLINE_AFTER`,
        because being briefly off the line is how you finish a side
        quest on purpose.
        """
        import time

        from .. import questlist

        if not questlist.loaded():
            return
        now = time.monotonic()
        places = self._places()
        # Where the party as a whole is, so the message can say what
        # this wizard should be on instead of just what it is on.
        main = [p for p in places if p.on_main]
        theirs = (f"#{min(p.order for p in main)}" if main else "")
        for seat, place in zip(self.seats, places):
            if seat.client is None:
                continue
            if place.on_main or not place.known:
                # `known` False is a read that failed or a quest the
                # list has never heard of -- not evidence of anything.
                # Saying "off the questline" on an unreadable tracker
                # would fire on every load screen.
                if seat.off_line_since and place.on_main:
                    seat.tel.note_questing(
                        "back-on-questline",
                        f"{seat.name} is back on the main line at "
                        f"#{place.order} ({place.name!r}) after "
                        f"{(now - seat.off_line_since) / 60:.0f} min away")
                    self._say(seat, f"{seat.name} is back on the main line")
                seat.off_line_since = None
                seat.said_off_line = ""
                continue
            if seat.off_line_since is None:
                seat.off_line_since = now
                continue
            away = now - seat.off_line_since
            if away < self.OFF_QUESTLINE_AFTER:
                continue
            if seat.said_off_line == place.name:
                continue
            seat.said_off_line = place.name
            said = (f"{seat.name} has been off the main questline for "
                    f"{away / 60:.0f} min — its tracker is on "
                    f"{place.name!r}, which is a side quest"
                    + (f", while the party is on {theirs}" if theirs else "")
                    + ". Every `tp quest` goes to the side quest until the "
                      "main one is selected again")
            for other in self.seats:
                try:
                    other.tel.note_questing("off-questline", said)
                except Exception:
                    pass
            self._say(seat, said)

    def _quests_agree(self):
        """(together, why), or (None, why) when the list cannot say.

        The question `goals_agree` was answering badly. Two wizards on
        the same quest are together even when their goal lines differ,
        and on a multi-objective step they always differ -- that is what
        a multi-objective step IS.

        None rather than True when fewer than two wizards can be placed:
        "no evidence" and "no desync" are different, and the caller has
        a weaker rule to fall back on.
        """
        from .. import questlist

        if not questlist.loaded():
            return None, "the quest list did not load"
        places = self._places()
        orders = [p.order for p in places if p.comparable]
        if len(orders) < 2:
            return None, "fewer than two wizards could be placed"
        worlds = {p.world for p in places if p.comparable}
        if len(worlds) > 1:
            return False, (f"the party is in {len(worlds)} different worlds "
                           f"({', '.join(sorted(worlds))})")
        spread = max(orders) - min(orders)
        if spread < questlist.BEHIND_BY:
            return True, (f"all on the same quest (#{min(orders)}), whatever "
                          f"their goal lines say")
        return False, f"{spread} quests apart (#{min(orders)}–#{max(orders)})"

    def _behind_by_questline(self):
        """The seat the quest list says is furthest back, or None.

        None means "this rule has nothing to say", not "nobody is
        behind" -- the caller falls through to the older rules. It says
        nothing when the list will not load, when fewer than two wizards
        are on a quest it can place (side quests carry no order), when
        the party is split across worlds, when the gap is within
        `BEHIND_BY`, or when two wizards are equally far back.

        That is a lot of nothing, deliberately. The rule this replaces
        was a guess that named the wrong wizard; a rule that answers
        only when it knows is worth more than one that always answers.
        """
        from .. import questlist

        if not questlist.loaded():
            return None
        places = self._places()
        indices, gap, why = questlist.furthest_behind(places)
        if not indices:
            self._behind_why = why
            # Cleared, not left: `_check_in_step` reads `_behind_gap` to
            # decide whether to pause the script, and a stale distance
            # from three ticks ago would start a catch-up on a party the
            # questline can no longer place.
            self._behind_gap = 0
            self._behind_group = []
            return None
        self._behind_basis = f"the questline says so — {why}"
        self._behind_gap = gap
        self._behind_places = places
        # The whole group. Two wizards tied at the back is the ordinary
        # case in a party of three, and both of them have the same step
        # to finish -- see `questlist.furthest_behind`.
        self._behind_group = [self.seats[i] for i in indices]
        return self._behind_group[0]

    def _who_is_behind(self):
        """Which wizard is on a step the others have finished, or None.

        The clock does not answer this, and the run at rev 1d28f745 says
        so outright. Two desyncs, sixteen seconds apart::

            t=90    w1 Talk To Danforth · w2 Talk To Danforth · w3 Find Key Stone
            t=106   w1 Find Key Stone   · w2 Talk To Danforth · w3 Find Key Stone

        Both were reported as "wizard 3 is behind" by the old rule --
        whichever seat's goal changed least recently -- and at t=106 that
        is exactly backwards. w1 moved ONTO w3's goal, so w3 had been
        ahead all along and w2 is the one that missed the step. w3 looked
        stale only because it had been sitting on the later quest longer.

        This is not just a wrong line in the log. `_behind` is what
        `_should_catch_up` sends the other wizards back to, so naming the
        wrong one walks the party to a wizard that is ahead of them.

        Three rules, tried in order of how much they actually know.

        FIRST the questline. Quest names have no order in the game, but
        they have one outside it, and `questlist` carries it: the
        wizard on Krokotopia main #7 is behind the wizard on #9, as a
        fact rather than an inference. This is the only rule that can
        say HOW far behind, and the only one that works when the party
        never overlapped -- which is most of a long desync, and is why
        rev bb8f2b3c logged "which one is behind cannot be told" thirty
        times.

        SECOND the party's own demonstration: if another wizard has HELD
        this wizard's current goal and has since moved off it, that step
        is finished and this wizard is still on it.

        THIRD, and only then, the clock.

        Two wizards behind, or none identifiable, returns None -- an
        honest "cannot tell" beats confidently walking the party to the
        wrong wizard, because `_should_catch_up` MOVES them.
        """
        readable = [s for s in self.seats if s.goal]
        if len(readable) < 2:
            return None
        ranked = self._behind_by_questline()
        if ranked is not None:
            return ranked
        # The older rules name one wizard at most; the group is that one.
        behind = [s for s in readable
                  if any(o is not s and s.goal in o.goals_seen
                         and o.goal != s.goal for o in readable)]
        if len(behind) == 1:
            self._behind_basis = "another wizard has finished that step"
            return behind[0]
        if behind:
            # Two on a finished step is a party that split twice. Naming
            # one of them would walk the third to an arbitrary choice.
            self._behind_basis = "more than one wizard is on a finished step"
            return None
        # Nothing has been observed to precede anything, so fall back to
        # the clock -- the wizard whose goal moved least recently. It is
        # a guess, and it is the guess that shipped before this; the
        # basis is recorded so the next export shows when it was used
        # and whether it was right.
        self._behind_basis = "guessed from which goal moved least recently"
        return min(readable, key=lambda s: s.goal_at)

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

        And measured on the QUEST, not the goal line, whenever the
        questline can place the party. Rev 8e5a9c75 is twenty minutes of
        this being wrong: 22 `quest-desync` entries, every one of them
        three wizards inside one quest. Krokotopia #12 is "Gather the
        Troops", whose objectives are

            Private Primwell · Private Archibald · Private Livingston
            · Private Farnsworth · Lieutenant Standish

        so a party working through it correctly shows three different
        goal lines at all times, and `goals_agree` calls that a desync
        on every tick. `_behind` then flapped between all three wizards
        -- eight "another wizard has finished that step", seven guesses
        from the clock, seven "cannot be told" -- and all 22 named a
        laggard in a party that was together.

        The questline said so each time and was overruled by the older
        rules underneath it. Now it wins: within `BEHIND_BY` quests is
        in step, whatever the goal lines say. Goal text is the fallback
        for a party the list cannot place, where it is the only evidence
        there is.
        """
        import time

        if len(self.seats) < 2:
            return
        from .. import questing, questlist

        goals = [s.goal for s in self.seats]
        now = time.monotonic()
        together, why = self._quests_agree()
        if together is None:
            together = questing.goals_agree(goals)
            why = "by goal text — the questline could not place the party"
        if together:
            self._in_step_since = now
            self._said_desync = ""
            self._behind = None
            return
        self._in_step_why = why
        since = getattr(self, "_in_step_since", None)
        if since is None:
            self._in_step_since = now
            return
        if now - since < self.DESYNC_GRACE:
            return

        self._behind = self._who_is_behind()

        # Named, not counted. "the party is out of sync" sends you to
        # look at two windows; this says which wizard is on what.
        where = " · ".join(f"{s.name}: {s.goal or 'unreadable'}"
                           for s in self.seats)
        # Bound before the `if`, not inside it. It was assigned inside,
        # and read after -- so on every tick where the desync line had
        # not CHANGED, `behind` was unbound and the whole service loop
        # stage died with an UnboundLocalError. Live at rev 7888c35a
        # that fired on all three wizards, and it is the second time an
        # unguarded read in this file has taken the tick down (see
        # `_heartbeat` and `runner.steps`).
        behind = self._behind
        if where != getattr(self, "_said_desync", ""):
            self._said_desync = where
            # Every seat's log, not just the one this tick belongs to: a
            # desync is a fact about the party, and whichever export gets
            # opened first should show it.
            for other in self.seats:
                try:
                    other.tel.note_questing(
                        "quest-desync",
                        f"{now - since:.0f}s on different quests — {where}"
                        + (f" — {behind.name} is behind "
                           f"({getattr(self, '_behind_basis', '')})"
                           if behind else
                           f" — which one is behind cannot be told: "
                           f"{getattr(self, '_behind_basis', 'no evidence')}")
                        # How each wizard was located, because that is
                        # what everything above hangs off and the first
                        # live run gave no way to see it. All three were
                        # placed by goal text, which means the quest
                        # NAME read returned nothing for any of them --
                        # invisible in the export.
                        + f" — placed: {self._how_placed()}")
                except Exception:
                    pass
            self._say(seat,
                      f"the party has been on different quests for "
                      f"{now - since:.0f}s — {where}"
                      + (f". {behind.name} is the one behind; the others "
                         f"will go back and help" if behind else "")
                      + ". The script cannot see this: its own instruction "
                        "pointer is fine, and deimoslang can only ask "
                        "whether a wizard is on a NAMED quest, never "
                        "whether two wizards are on the same one")

        # A desync the questline can measure is one wizAi can actually
        # finish, so it stops reporting and starts questing. Only that
        # kind: `_behind` from the older rules says who, not how far, and
        # without a distance there is no way to tell when to stop.
        if (behind is not None and self.script
                and getattr(self, "_behind_gap", 0) >= 1
                and "questline" in getattr(self, "_behind_basis", "")):
            group = getattr(self, "_behind_group", None) or [behind]
            if self._written_off(group):
                # Tried, and it did not work. Running the script is not
                # a good answer either, but it is the only OTHER answer,
                # and pausing it forever to re-attempt a step that has
                # not moved in half an hour is the worse of the two.
                key = f"wrote-off:{self._step_key(behind)}"
                self._say_once(
                    behind, key,
                    f"{behind.name} is still behind, and a catch-up has "
                    f"already given up on this exact step — leaving it to "
                    f"the script rather than pausing the party again for "
                    f"something that did not work")
                # Written down ONCE per step, not on the `_say_once`
                # cadence. The verdict never changes while the step does
                # not, so repeating it says nothing new -- rev 3d026ada
                # spent 25 of Phönix's log entries on this one sentence,
                # the last of them "1440 times in a row".
                if key not in self._said_written_off:
                    self._said_written_off.add(key)
                    said = (f"{' and '.join(s.name for s in group)} still "
                            f"{self._behind_gap} quest(s) behind on a step a "
                            f"catch-up has already given up on. The script "
                            f"keeps its wizards — wizAi's questing cannot "
                            f"finish this one")
                    for other in self.seats:
                        try:
                            other.tel.note_questing(
                                "catch-up-written-off", said)
                        except Exception:
                            pass
                return
            self._start_catching_up(group, self._behind_gap,
                                    self._behind_basis)

    #: how long a wizard may be in a different zone from the rest of the
    #: party before it counts as left behind. Zone changes are not
    #: simultaneous -- one client finishes loading seconds before
    #: another -- so a bare inequality would fire on every door.
    #: a zone this wizard left less than this long ago is somewhere it
    #: chose to leave, not somewhere it was separated from.
    LEFT_ON_PURPOSE = 180.0
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

        now = time.monotonic()
        zones = {}
        for seat in live:
            zones[seat] = await party.zone(seat.client)
            if zones[seat]:
                if seat.zone_seen and zones[seat] != seat.zone_seen:
                    # Where it has just come FROM, and when. See the
                    # "left there on purpose" test below.
                    seat.zone_left.append((seat.zone_seen, now))
                    del seat.zone_left[:-8]
                    # Free: this poll already runs every TOGETHER_POLL
                    # seconds for the stranded check. Recording it turns
                    # the log into a route -- where each wizard went and
                    # when -- which is what "one of them wandered off"
                    # needs in order to be checkable rather than
                    # inferred from two stranded entries.
                    try:
                        seat.tel.note_questing(
                            "zone", f"{seat.zone_seen} -> {zones[seat]}")
                    except Exception:
                        pass
                if zones[seat] != seat.zone_seen:
                    seat.zone_since = now
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

        # Did it LEAVE the majority's zone, or never get there? A wizard
        # that fell behind and a wizard that walked on ahead are the
        # same shape -- one seat somewhere the others are not -- and
        # only this tells them apart.
        #
        # Rev 228d4f50 dragged the second kind backwards five times in
        # four minutes. Sebastian went into WC_Firecat_T1, was pulled
        # back out to WC_Firecat, walked in again, was pulled out again,
        # on a fifty-second cycle for the rest of the run. Every pull
        # threw away the step he had just finished and handed the script
        # a wizard behind where it thought it was, which is precisely
        # the failure this mechanism exists to prevent.
        #
        # So: somewhere it has just come from is not somewhere to send
        # it back to. Konstantin at t=550 had never been in T1 and was
        # rescued correctly; Sebastian at t=639 had been in WC_Firecat
        # minutes earlier and left under his own steam. A wizard merely
        # wandering -- through zones the party has not been in -- still
        # falls through to the stranded clock and is still fetched.
        if any(zone == best and now - at < self.LEFT_ON_PURPOSE
               for zone, at in seat.zone_left):
            seat.stranded_since = None
            return None, None

        # ...and a hard stop even if that reasoning is wrong. Two pulls
        # into the same zone inside five minutes is a loop, not a
        # rescue, and a loop must not be able to run for days.
        recent = [z for z, at in seat.rejoin_history if now - at < 300.0]
        if recent.count(best) >= 2:
            self._say_once(
                seat, "rejoin-loop",
                f"{seat.name} has been pulled back to {best} twice already "
                f"— leaving it to the script rather than dragging it in a "
                f"circle")
            seat.tel.note_questing(
                "rejoin-looping",
                f"declined to pull {seat.name} back to {best} a third time")
            return None, None

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
            seat.rejoin_history.append(
                (getattr(target, "zone_seen", "") or "", now))
            del seat.rejoin_history[:-8]
            seat.tel.note_questing("rejoined", why or f"went to {target.name}")
            self._say(seat, f"was left behind in {adrift} — {why}")
        elif why:
            seat.tel.note_questing("rejoin-failed", why)
            self._say_once(seat, "rejoin",
                           f"left behind in {adrift} and cannot get back — "
                           f"{why}")
        else:
            # `party.follow` returns (False, "") for its two "nothing to
            # do" cases -- the follower is in a duel, or they are
            # already together -- and this used to drop both. So the log
            # recorded a wizard stranded and then said nothing about it
            # ever again, which reads as the rescue having hung.
            #
            # It is what the run at rev 85a68184 shows: two `stranded`
            # entries, no `rejoined`, no `rejoin-failed`, and no third
            # attempt. Sebastian was fighting Foulgaze on his own in
            # WC_OldeTown_T2 while the other two were in WC_Hub. Leaving
            # him alone was right. Not saying so was not.
            seat.stranded_since = None
            fighting = await party.in_battle(seat.client)
            detail = ("still in a duel, so there is nothing to teleport "
                      "out of" if fighting else
                      "already close enough by the time the follow ran")
            seat.tel.note_questing("rejoin-skipped", detail)
            self._say(seat, f"left behind in {adrift} — {detail}")

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
            return
        await self._check_same_sigil()

    #: How far apart two wizards at a press-X prompt can be and still be
    #: at the SAME one. A sigil's own interact range is a few hundred
    #: units (`questing.QUEST_RADIUS` is 750 for the comparable "is this
    #: the quest NPC" question), so anything past this is two prompts.
    SAME_SIGIL_WITHIN = 1200.0
    #: ...and for how long, before it is said. Wizards reach a sigil at
    #: different times and one still walking is not a split party.
    SPLIT_AFTER = 45.0

    async def _check_same_sigil(self):
        """Everyone at a prompt, same zone, nowhere near each other.

        The failure a live run sat in for the last twenty minutes of rev
        1dcf4193, and the one the window could not name. All three
        wizards were in Krokotopia/KT_Pyramid/KT_PalaceOfFire, all three
        had "Press X to Enter" on screen, and the party never moved --
        because two of them were at Edo's Chamber and one was at Akori's.
        A sigil admits the wizards standing on IT. Pressing X at a
        different one, forever, is not a stall any zone check can see:
        the zone agrees, the goal agrees, and every wizard is doing
        something.

        Reported, not fixed. What puts the party back on one sigil is
        the script's own friend teleport -- which is exactly what was
        broken for this run (`Deimos/src/utils.py`, `_same_wizard`) --
        and dragging wizards between sigils from out here would fight
        the script for the wheel at the one moment it is about to work.
        So this says the sentence the operator needed and leaves the
        driving alone.
        """
        import time

        from .. import party, questing

        live = [s for s in self.seats if s.client is not None]
        if len(live) < 2:
            return
        zones = {s.zone_seen for s in live}
        if len(zones) != 1 or not next(iter(zones)):
            return                       # not everyone in one zone
        zone = next(iter(zones))

        at_prompt, where = [], {}
        for seat in live:
            try:
                if not await questing.near_interactable(seat.client):
                    continue
                spot = await party.position(seat.client)
            except Exception:
                continue
            if spot is None:
                continue
            at_prompt.append(seat)
            where[seat] = spot
        if len(at_prompt) < 2:
            for seat in live:
                seat.apart_since = None
            return

        def apart(a, b):
            dx = where[a].x - where[b].x
            dy = where[a].y - where[b].y
            return (dx * dx + dy * dy) ** 0.5

        # The biggest cluster is the sigil the party is at; anyone
        # further than one sigil's width from all of it is at another.
        # No majority means every wizard is on its own, which is a
        # different report and one this cannot make honestly.
        best, near = None, []
        for seat in at_prompt:
            group = [s for s in at_prompt
                     if apart(seat, s) <= self.SAME_SIGIL_WITHIN]
            if best is None or len(group) > len(near):
                best, near = seat, group
        odd = [s for s in at_prompt if s not in near]
        now = time.monotonic()
        # A strict majority, so "the party's sigil" is a fact rather than
        # a coin toss. Two and two is a party that has genuinely split
        # down the middle, and naming either half the odd one out would
        # be a guess dressed up as a diagnosis.
        if not odd or len(near) < 2 or len(near) <= len(odd):
            for seat in live:
                seat.apart_since = None
            return

        for seat in live:
            if seat not in odd:
                seat.apart_since = None
        for seat in odd:
            if seat.apart_since is None:
                seat.apart_since = now
        if now - min(s.apart_since for s in odd) < self.SPLIT_AFTER:
            return

        held = now - min(s.apart_since for s in odd)
        gap = max(apart(s, best) for s in odd)
        names = " and ".join(s.name for s in odd)
        with_them = " and ".join(s.name for s in near)
        detail = (f"{names} {'have' if len(odd) > 1 else 'has'} been at a "
                  f"different press-X prompt from "
                  f"{with_them} for {held:.0f}s — same zone ({zone}), "
                  f"{gap:,.0f} apart. Two sigils in one zone are two "
                  f"dungeons, and nobody can enter until the party is on "
                  f"one of them")
        # One status line, but an entry in EVERY wizard's log: three
        # exports get uploaded and each one has to explain the twenty
        # minutes its wizard spent pressing X at nothing.
        key = f"split-sigil:{zone}"
        self._say_once(odd[0], key, detail)
        n = odd[0].stage_errors.get(key, 0)
        if n == 1 or n % 20 == 0:
            for seat in live:
                try:
                    seat.tel.note_questing("split-sigil", detail)
                except Exception:
                    pass

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

        The wizard that is BEHIND is not followed anywhere; it is driven
        through its own step by `_quest_the_missed_step`. This used to
        say the script would do that, and it was wrong -- see
        `_start_catching_up` for what the operator pointed out.
        """
        group = getattr(self, "_behind_group", None) or []
        behind = getattr(self, "_behind", None)
        if behind is None or len(self.seats) < 2:
            return False
        # Never a wizard that is itself behind: it has its own step to
        # finish and following a wizard that is equally behind is a
        # circle. With a tied group that is two of three wizards, which
        # is why this reads the group rather than the single rally seat.
        if seat is behind or seat in group:
            return False
        return behind.client is not None

    # -- finishing the step that was missed --------------------------------
    #: how long the party may spend on one missed step before the script
    #: gets its wizards back.
    #:
    #: There has to be a bound, and it has to be generous. A missed step
    #: can be a dialogue box (seconds) or a dungeon boss (many minutes),
    #: and the script is paused throughout -- so too short abandons the
    #: catch-up half-done, and no bound at all replaces one stall with a
    #: better-documented one, which is the failure this whole project
    #: exists to remove.
    CATCH_UP_LIMIT = 900.0
    #: and how long without the laggard's quest advancing before it is
    #: written off as something wizAi's questing cannot finish.
    CATCH_UP_STALL = 300.0
    #: ...and a much shorter bound on the case where the laggard is not
    #: even MOVING. `CATCH_UP_STALL` is generous because a dungeon boss
    #: takes minutes and the goal line does not change while it is
    #: fought -- but a wizard that is standing still, out of combat,
    #: with the same zone, the same spot and the same goal is not
    #: fighting anything. `hop_once` teleports it to a marker it is
    #: already standing on and presses X at nothing.
    #:
    #: Rev 3822cc6c: Phönix one quest behind on `Defeat Edo Nirini in
    #: Palace of Fire`, whose marker is a dungeon sigil. The catch-up
    #: began at t=768 and the export ends at t=926 with all three
    #: wizards frozen and the script pinned at 8,083 instructions --
    #: with another 140s to run before `CATCH_UP_STALL` would have let
    #: go. Five minutes of a frozen party for a step nothing was
    #: attempting is worse than the desync it was answering.
    CATCH_UP_IDLE = 90.0

    def _catching_up(self):
        """The seats being caught up, as a list. Empty when none are.

        A list because two wizards tied at the back is the ordinary case
        in a party of three, not an edge case -- rev 8e5a9c75 had two on
        Krokotopia #12 and one on #13 for the whole run, and a
        single-seat catch-up could not express that at all.
        """
        state = getattr(self, "_catch_up_state", None)
        return [] if state is None else list(state.get("seats") or ())

    def _start_catching_up(self, behind, gap, why):
        """Stop taking script instructions and finish the missed step.

        The operator's correction, in full, because it overturned what
        `_catch_up` was built on:

            Just teleporting back to a wizard you believe is behind
            doesn't work because I think the script then just continues
            on the same way. They need to actually quest with the other
            wizard until they catch up.

        Exactly right, and the reason is structural. One deimoslang
        program drives the whole party from ONE instruction pointer.
        When a wizard misses a step -- a dialogue that did not clear, a
        teleport the game refused -- the pointer does not wait for it;
        it is at the party's step and the wizard is at an earlier one.
        Teleporting that wizard to the others moves its BODY. Its quest
        state stays where it was, so the next `tp quest` sends it back
        to its own marker and the next one after that, forever.

        Nothing in the script can fix this. deimoslang has no way to say
        "this wizard only, until its quest matches" -- a `waitfor` is
        the closest thing and rev bb8f2b3c showed those were only ever
        watching one wizard anyway. The quest state has to be advanced
        by actually doing the step, which means somebody has to drive
        that one wizard through it: teleport to ITS marker, clear ITS
        dialogue, press X, and let the policy fight whatever turns up.
        `questing.hop_once` is that, already written and already used by
        auto-quest.

        So the script is paused rather than fought with. It is the same
        judgement `_should_catch_up` already makes about the wizards
        that are ahead, taken to its conclusion: while the party is out
        of step the script's instructions are wrong for everybody, so
        running them is worse than not.
        """
        import time

        if getattr(self, "_catch_up_state", None) is not None:
            return
        seats = list(behind) if isinstance(behind, (list, tuple)) else [behind]
        seats = [s for s in seats if s is not None]
        if not seats:
            return
        now = time.monotonic()
        self._catch_up_state = {
            "seats": seats, "gap": gap, "started": now, "moved": now,
            "goals": {id(s): s.goal for s in seats}, "why": why,
        }
        names = " and ".join(s.name for s in seats)
        is_are = "is" if len(seats) == 1 else "are"
        said = (f"{names} {is_are} {gap} quest(s) behind — pausing the "
                f"script and finishing that step before the party goes on. "
                f"Teleporting {'it' if len(seats) == 1 else 'them'} to the "
                f"others would move {'it' if len(seats) == 1 else 'them'} "
                f"without advancing the quest, and the script would send "
                f"{'it' if len(seats) == 1 else 'them'} straight back")
        for other in self.seats:
            try:
                other.tel.note_questing("catch-up-started", said)
            except Exception:
                pass
        self._say(seats[0], said)

    def _stop_catching_up(self, kind, said):
        for other in self.seats:
            try:
                other.tel.note_questing(kind, said)
            except Exception:
                pass
        seats = self._catching_up()
        self._catch_up_state = None
        if kind == "catch-up-gave-up":
            # Remembered, or giving up is a word rather than an act.
            # Rev 3822cc6c: `catch-up-gave-up` and `catch-up-started`
            # share a timestamp seven times over thirty-five minutes,
            # because `_check_in_step` sees the same desync on the very
            # next tick and starts the same catch-up again. The script
            # was handed back its wizards for zero seconds each time,
            # and the instruction count sat at 8,083 throughout.
            import time

            now = time.monotonic()
            for one in seats:
                self._wrote_off[id(one)] = (self._step_key(one), now)
        if seats:
            self._say(seats[0], said)

    #: nothing may start another catch-up this soon after one gave up,
    #: whatever the wizards' quests say. The step-key rule below is the
    #: real guard; this is the floor under it, so two readings that
    #: alternate cannot thrash the party between them.
    CATCH_UP_COOLDOWN = 120.0

    def _step_key(self, seat):
        """What step this wizard is on, for "have we tried this already".

        The placement when there is one -- two goal lines can describe
        the same quest and the order cannot -- and the raw quest text
        when there is not, which at least changes when the wizard does.
        """
        place = dict(zip((id(s) for s in self.seats),
                         self._places())).get(id(seat))
        if place is not None and place.comparable:
            return (place.world, place.order)
        return (None, seat.quest_name or seat.goal or "")

    def _written_off(self, group):
        """Has a catch-up already given up on this exact step?

        True only when EVERY wizard in the group is one it gave up on
        and none of them has moved on since. A wizard whose step has
        changed has made progress by some other means, and a fresh
        catch-up for the new step is a different question.
        """
        import time

        now = time.monotonic()
        if not group:
            return False
        for one in group:
            wrote = self._wrote_off.get(id(one))
            if wrote is None:
                return False
            step, when = wrote
            if now - when < self.CATCH_UP_COOLDOWN:
                continue
            if step != self._step_key(one):
                # It moved. Forget the write-off so the new step gets
                # its own chance rather than inheriting this verdict --
                # and forget that it was said, so if the wizard is ever
                # written off on this step again the export says so
                # again rather than staying silent about a second one.
                del self._wrote_off[id(one)]
                self._said_written_off.discard(f"wrote-off:{step}")
                return False
        return True

    def _check_caught_up(self):
        """End the catch-up when it is done, or when it is not going to be.

        Always ends it. A paused script that is never resumed is the
        stall this replaces, so every exit here puts the wizards back on
        the program -- including the ones that give up.
        """
        import time

        from .. import questlist

        state = getattr(self, "_catch_up_state", None)
        if state is None:
            return
        seats = [s for s in state["seats"]
                 if s.client is not None and s in self.seats]
        now = time.monotonic()
        if not seats:
            self._stop_catching_up(
                "catch-up-gave-up",
                "the wizard(s) being caught up are no longer connected — "
                "the script has its wizards back")
            return
        state["seats"] = seats
        for one in seats:
            if one.goal and one.goal != state["goals"].get(id(one)):
                state["goals"][id(one)] = one.goal
                state["moved"] = now

        indices, gap, _why = questlist.furthest_behind(self._places())
        behind = {id(self.seats[i]) for i in indices}
        started = {id(s) for s in seats}
        if not behind:
            names = " and ".join(s.name for s in seats)
            self._stop_catching_up(
                "catch-up-done",
                f"{names} back with the party — the script has its "
                f"wizards back")
            return
        if not behind <= started:
            # Somebody NEW is behind, so this catch-up is answering the
            # wrong question and a fresh one should start from the top.
            self._stop_catching_up(
                "catch-up-done",
                "a different wizard is behind now — starting again rather "
                "than inheriting this catch-up's clock")
            return
        if behind != started:
            # A SUBSET. One of the group caught up and the others have
            # not, which is progress, not a reason to stop -- and
            # stopping on it is what rev 1dcf4193 did: two catch-ups
            # ended 10.6s and 0.4s after they began, because in a party
            # of three somebody's quest ticks over almost immediately.
            caught = [s for s in seats if id(s) not in behind]
            seats = [s for s in seats if id(s) in behind]
            state["seats"] = seats
            state["moved"] = now
            if caught:
                self._say(caught[0],
                          f"{' and '.join(s.name for s in caught)} caught up; "
                          f"still finishing the step for "
                          f"{' and '.join(s.name for s in seats)}")
        if gap < state["gap"]:
            # Progress, so the stall clock restarts even if the goal
            # line happened not to change.
            state["gap"] = gap
            state["moved"] = now
        waited = now - state["started"]
        stalled = now - state["moved"]
        # Not one wizard of the group has moved -- see `CATCH_UP_IDLE`.
        # `seat.progress_at` is when (zone, spot, goal) last changed, so
        # this is exactly "standing still with nothing happening", and
        # it deliberately does NOT fire while anybody is in a duel: a
        # fight is the catch-up working.
        #
        # Measured from whichever is LATER, the wizard's last movement or
        # the moment this catch-up began. Rev 3d026ada without that
        # clamp is `catch-up-started` and `catch-up-gave-up` on the same
        # timestamp with "has not moved or fought for 122s" -- the 122s
        # were spent stuck BEFORE the catch-up, which is why there is a
        # catch-up at all. Every catch-up worth having is for a wizard
        # that was already standing still, so an absolute idle clock
        # kills all of them at birth, and this one never got a single
        # tick to teleport Phönix anywhere.
        idle = min((now - max(s.progress_at, state["started"])
                    for s in seats if s.progress is not None), default=0.0)
        going_nowhere = (idle >= self.CATCH_UP_IDLE
                         and not any(s.in_duel for s in seats))
        if (waited >= self.CATCH_UP_LIMIT or stalled >= self.CATCH_UP_STALL
                or going_nowhere):
            self._stop_catching_up(
                "catch-up-gave-up",
                f"{waited / 60:.0f} min trying to finish "
                f"{' and '.join(s.name for s in seats)}'s step "
                f"and still {gap} quest(s) behind"
                + (f" — nothing has moved for {stalled / 60:.0f} min"
                   if stalled >= self.CATCH_UP_STALL else "")
                + (f" — {' and '.join(s.name for s in seats)} has not moved "
                   f"or fought for {idle:.0f}s, so nothing is attempting "
                   f"this step" if going_nowhere else "")
                + ". Giving the script its wizards back rather than holding "
                  "the whole party here")

    async def _quest_the_missed_step(self, seat):
        """Drive the laggard through its own quest step, one hop a tick.

        `questing.hop_once` and nothing new: teleport to this wizard's
        OWN quest marker, clear whatever dialogue is in the way, press X
        if something is interactable, and stop the moment a fight starts
        -- the policy takes it from there, and the wizards that came
        back are in the circle.
        """
        from .. import questing

        if await questing.in_battle(seat.client):
            return                       # the policy owns this now
        await questing.hop_once(
            seat.client,
            on_status=lambda m: self._say_once(seat, f"catch-up:{m[:24]}",
                                               f"catching up — {m}"))


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
                    seat.tel.note_questing(
                        "healed",
                        f"back to {left:.0%} after {time.monotonic() - started:.0f}s"
                        if left is not None else "carrying on")
                return
            if time.monotonic() - started > self.LOW_HEALTH_WAIT:
                self._say(seat,
                          f"still on {left:.0%} health after "
                          f"{self.LOW_HEALTH_WAIT:.0f}s and nothing is "
                          f"fixing it — going into the next fight anyway, "
                          f"because a run that stops here reports nothing "
                          f"at all")
                seat.tel.note_questing(
                    "went-in-hurt",
                    f"gave up after {self.LOW_HEALTH_WAIT:.0f}s on "
                    f"{left:.0%} health, needing {floor:.0%}")
                return
            if not said:
                said = True
                self._say(seat,
                          f"on {left:.0%} health and the last few fights "
                          f"have cost up to {floor:.0%} — not starting "
                          f"another one yet")
                # In the questing log, not just the status line. This is
                # the one place the run deliberately stops for up to
                # `LOW_HEALTH_WAIT` a fight, and it was the one place
                # that left no trace in the export -- so "it gets stuck"
                # and "it is healing, as designed" were the same picture.
                #
                # The run at rev 85a68184 is why it matters: Phönix
                # finished a fight on 27% against a 40% floor and opened
                # Lord Nightshade on 8.8%, then died; Konstantin ended
                # his on 4.5% and died too. Whether the gate held and
                # gave up, or never ran at all, is not answerable from
                # that export, and it has to be.
                seat.tel.note_questing(
                    "waiting-to-heal",
                    f"on {left:.0%} health, needing {floor:.0%} before "
                    f"the next fight")
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

    def _round_timing_hook(self, seat):
        """What the round cost, straight into its record.

        Silent: a number per round in the status bar would bury the
        messages that need reading. It goes in the export, where "the
        median round waited 34s on the game and 4s on us" is one line
        of the summary instead of a guess.
        """
        return lambda waited, planned, acted: seat.tel.time_round(
            waited, planned, acted)

    def _defeated_hook(self, seat):
        """Say a defeat out loud AND write it down.

        A defeat is the most disruptive thing that can happen to a
        scripted run, and the only one that was not in the log. The game
        teleports the wizard to the commons, so the script's next
        instruction is aimed at a place it is no longer in -- and that
        desync is invisible to everything else here. `_check_in_step`
        sees an unchanged quest goal; `_check_together` sees a zone
        change like any other and cannot say why. Six of the eighteen
        recorded fights at rev 85a68184 were losses.
        """
        def defeated():
            where = seat.zone_seen or "an unreadable zone"
            try:
                seat.tel.note_questing(
                    "defeated",
                    f"{seat.name} was defeated in {where} and sent to the "
                    f"commons — the script's next instruction is for "
                    f"somewhere else")
            except Exception:
                pass
            self._say(
                seat,
                "defeated — left the party's circle so the others stop "
                "waiting for it every round, and its rounds are no longer "
                "recorded. It rejoins when this fight ends.")

        return defeated

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

    def _recovery_failed_hook(self, seat):
        return lambda why: self._on_recovery_failed(why, seat)

    def _on_recovery_failed(self, why, seat=None):
        """The runner-up did not go out either. Say so, and say why.

        Re-emitted like `_on_recovered_cast` for the same reason: the
        Decisions table already has this round reading "passed", and the
        row is only useful with the reason on it.
        """
        seat = self.seats[0] if seat is None else seat
        self._say(seat, why)
        rec = seat.tel.note_recovery_failed(why)
        if rec is not None:
            self.seat_round_done.emit(seat.index, rec)

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
