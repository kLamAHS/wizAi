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

#: "this has never happened", for stamps compared against a cooldown.
#:
#: Zero is a LIE here, and an expensive one. `time.monotonic()` counts
#: from BOOT, not from the start of the process, so on a machine that
#: has just been turned on -- which is exactly when somebody launches
#: the game and starts a run -- `now - 0.0` is a small number and every
#: `now - stamp < COOLDOWN` test reads as "just did that". The first
#: seven minutes of uptime could not restart a script, the first five
#: could not change realm or re-arm a quest arrow, and nothing said so.
#: Minus infinity is the honest value: never, so any cooldown has
#: already elapsed.
NEVER = float("-inf")


def _door_how(door):
    """How a learned door was crossed: "walked", "sigil", or unknown.

    A door entry has grown twice -- `(spot, at, taught_by)`, then a
    sample age, then this -- and the older shapes still turn up in a
    map that outlives a reload. Reading the field positionally with a
    default keeps every one of them legible instead of raising.
    """
    return door[4] if len(door) > 4 else ""


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
        #: when the goal last read as something, blank reads excluded.
        #: A blank read keeps the previous value -- a blank is not
        #: evidence the wizard changed quest -- but only for
        #: `GOAL_MAX_AGE`, after which a kept value is invention. The
        #: same rule the quest NAME has always had; the goal did not,
        #: and cleared itself on every transient blank. See
        #: `LiveWorker._note_goal`.
        self.goal_ok_at = 0.0
        #: a changed goal that has been read ONCE. It drives nothing
        #: until a second read agrees with it. See
        #: `LiveWorker._note_goal`.
        self.goal_pending = ""
        #: every goal this wizard has held, in order. A wizard still on
        #: a step another has already left is the one behind -- see
        #: `LiveWorker._who_is_behind`.
        self.goals_seen = []
        #: since when this wizard's tracker has been on something that
        #: is not the world's main line, and the last side quest that
        #: was said. See `LiveWorker._check_on_questline`.
        self.off_line_since = None
        self.said_off_line = ""
        #: the last main-line quest this wizard's tracker held, as
        #: (world, order, name), and when. The first and best answer to
        #: "which quest was lost": a wizard on a side quest now was on
        #: THIS a moment ago, and finishing a quest hands the next one
        #: over rather than dropping the tracker onto a side quest --
        #: so a tracker that wandered never got there. See
        #: `LiveWorker._lost_quest`.
        self.last_main = None
        self.last_main_at = NEVER
        #: when a lost-questline recovery was last attempted, and the
        #: quest it was attempted for, so a cure that cannot work (the
        #: quest is not in the book) is not tried every ten minutes
        #: forever. See `LiveWorker._maybe_recover_questline`.
        self.recover_tried_at = NEVER
        self.recover_gave_up = ""
        self.recover_gave_up_at = NEVER
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
        #: when this seat was last dragged to the party's sigil, so one
        #: split episode gets one drag per wizard rather than one per
        #: check. See `LiveWorker.SIGIL_ACT`.
        self.sigil_moved_at = NEVER
        #: flat distance to this wizard's quest marker, or None. A marker
        #: in another zone reads as a six-figure nonsense distance,
        #: because the coordinates are in that zone's space -- which is
        #: precisely the discriminator `_start_catching_up` needs.
        self.marker_away = None
        #: since when the quest position has been UNREADABLE while the
        #: goal line still reads -- the dead-quest-hook signature, and a
        #: different fact from `marker_away is None` (one bad read).
        #: Cleared by any successful marker read. See
        #: `LiveWorker._marker_dead` and `questing.rearm_quest_arrow`.
        self.marker_dead_since = None
        #: when a quest-arrow re-arm was last attempted for this seat,
        #: so a cure that is not taking is retried on a cooldown rather
        #: than paging through the quest book every tick.
        self.rearm_tried_at = NEVER
        #: when a Collect step's counter last went UP, and for which
        #: collect goal. The only proof a wizard ever reached the
        #: spawns, because a Collect step has no marker to check
        #: against. See `LiveWorker._maybe_realm_hop`.
        self.collect_moved_at = 0.0
        self.collect_moved_for = ""
        #: when this client's quest hook last DID read a position, and
        #: on which goal. Proof the hook works, which is what separates
        #: "this quest has no marker" from "the arrow is off". See
        #: `LiveWorker._marker_absent_by_design`.
        self.marker_ok_at = 0.0
        self.marker_ok_goal = ""
        #: the zone this seat's write-off was recorded in, so arriving
        #: somewhere new can clear it. See `LiveWorker._written_off`.
        self.zone_for_writeoff = None
        #: fruitless presses of X from one spot, and which spot. See
        #: `LiveWorker._x_did_nothing_here`.
        self.x_pressed = 0
        self.x_pressed_at = None
        #: whether this wizard was in a duel on its last service tick.
        #: Read from the OTHER seats' ticks -- see `CATCH_UP_IDLE`.
        self.in_duel = False
        #: when this seat's service loop last came round, stamped before
        #: the one read that can hang. A loop that has stopped ticking
        #: is invisible from inside itself -- and if it is seat 0's, it
        #: takes the party's whole script with it. See `_script_seat`.
        self.ticked_at = 0.0
        #: consecutive combat reads that had to be cut off. See
        #: `LiveWorker._read_in_battle`.
        self.unanswered = 0
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
        self.unstuck_at = NEVER
        self.steps_seen = None
        #: the dialogue box's TEXT at the last unstick look, or None when
        #: no box was open. Same text across two looks = the script's own
        #: clearing is provably not advancing it. See `DIALOG_WEDGE`.
        self.box_text = None
        #: position cells this wizard has stood in recently, cell -> last
        #: seen. A script retry loop that teleports a wizard between two
        #: spots is not movement. See `LiveWorker._note_progress`.
        self.cells_seen = {}
        #: when the quest NAME last actually read. A blank read keeps the
        #: previous value, and this bounds how long. See `_note_name`.
        self.name_read_at = 0.0
        #: the spot a desperation quest-teleport was already tried from,
        #: as a `progress` stamp. See `LiveWorker._desperate_hop`.
        self.hop_tried_at = None
        #: since when this wizard has been standing essentially ON its
        #: quest marker with the zone and the goal both unchanged -- the
        #: walk-in door signature. Cleared whenever the marker reads far
        #: or something changes. See `LiveWorker._maybe_walk_through`.
        self.through_since = None
        #: the countdown guard's debounce and once-per-visit stamp, and
        #: how many times an evidenced sigil's hold has re-armed here.
        #: See `LiveWorker._maybe_count_hold`.
        self.count_hold_seen = None
        #: when this wizard last pressed X at a press-X prompt. A zone
        #: change shortly after one is a SIGIL crossing, not a door, and
        #: the difference decides whether a stranded follower can walk
        #: in behind it. See `_learn_door`.
        self.pressed_x_at = NEVER
        self.count_hold_spot = None
        self.count_hold_replays = 0
        #: the once-per-visit stamp for pulling the party onto this
        #: wizard's quest marker the moment it arrives there. See
        #: `LiveWorker._maybe_count_hold`.
        self.party_pulled_spot = None
        #: since when this follower's follow has been continuously
        #: failing, how many attempts that is, and when the stranding
        #: alarm last fired. See `LiveWorker.STRANDED_ALARM_EVERY`.
        self.follow_failing_since = None
        self.follow_fails = 0
        self.stranded_said_at = NEVER
        #: when a walk-through was last attempted, so a door that will
        #: not open is walked at on a cooldown rather than every tick.
        self.walked_through_at = NEVER
        #: whether this seat has ever built a script runner. A REBUILD
        #: is not a first start, so the program's one-time setup is
        #: trimmed from every build after the first. See `_fresh_source`.
        self.script_built = False
        #: what `runner` was built from, so the service tick can notice
        #: the operator turning the script on, off, or replacing it --
        #: and which way the debug flag pointed at build time, because
        #: `set_debug` is baked into the build. See `_sync_script`.
        self.script_source = None
        self.script_debug_built = None
        #: the wiring the runner was BUILT over: (solo?, the leader it
        #: drives when solo, else None). A solo VM holds one client and
        #: a party VM holds them all, so a mode or Leader change
        #: mid-run is a different BUILD, not a different flag -- see
        #: `_sync_script`, which rebuilds on any difference.
        self.script_wiring_built = None
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
        #: True while the script executed instructions this burst and
        #: has more to run -- the service tick shortens its sleep so the
        #: next burst starts promptly. See `LiveWorker._script_step`.
        self.script_hot = False
        #: when this follower last took its OWN quest step, so the sync
        #: does not hammer the marker. See `LiveWorker._sync_follower`.
        self.synced_at = 0.0
        #: the goal the sync is working on and since when, so a step
        #: that will not turn in has a bounded budget before the
        #: follower goes back to the pilot.
        self.sync_goal = ""
        self.sync_began = 0.0
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
        #: the last position this seat was seen at, and the zone it was
        #: in at the time -- the raw material of the door map. See
        #: `LiveWorker._doors` and `_walk_the_leaders_door`.
        self.last_spot = None
        self.last_spot_zone = None
        #: the same pair, sampled on this seat's OWN tick instead of on
        #: the six-second party poll, and when. A door is learned from
        #: the last place a wizard stood before its zone changed, so the
        #: age of that sample IS the error in the door's position -- and
        #: at six seconds it is thousands of units, which no 250-unit
        #: sweep can absorb. See `_note_spot` and `_learn_door`.
        self.spot_at = NEVER
        self.spot_at_prev = NEVER
        self.spot_seen = None
        self.spot_zone = None
        #: when this seat last walked a door route, and how many
        #: consecutive cross-zone follows have failed for it.
        self.door_walked_at = NEVER
        self.cross_zone_fails = 0
        #: consecutive door-walk failures, per (from zone, to zone). The
        #: per-seat counter above says a follow is failing; this says
        #: THIS ROUTE is not working, which is the one an escalation can
        #: act on. See `_walk_the_leaders_door`.
        self.door_fails = {}
        #: opening board -> when this seat lost to it. A fight lost
        #: twice is a wall; the outcome used to be recorded and read by
        #: nothing but the win count. See `_note_the_loss`.
        self.lost_to = {}
        #: consecutive rounds where every line the rollout tried ended
        #: with this wizard dead, and the fight that was last reported
        #: for. See `_watch_for_a_fight_that_cannot_be_won`.
        self.no_line_survives = 0
        self.unwinnable_said_for = None
        #: the fight this seat has already explained a lone plan for.
        #: See `_say_why_it_planned_alone`.
        self.alone_said_for = None
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
    #: one wizard's live questing state, every service tick — the feed
    #: behind the Questing tab. (seat index, dict); see `_quest_row`.
    seat_quest = pyqtSignal(int, object)
    #: (seat, the wizard's in-game name), once a duel has revealed it
    seat_named = pyqtSignal(int, str)

    def __init__(self, telemetry, school, deck, policy_name, fights,
                 agent=None, auto_quest=False, auto_dialogue=True,
                 collect_wisps=True, use_potions=True,
                 buy_potions=False, script="",
                 hotkeys=None, continuation="", seats=None,
                 coordinate=True, passes=2, barrier=None,
                 follow_leader=True, leader=0, label_windows=True,
                 solo_script=False, script_step_delay=None,
                 script_dialog_delay=None, script_debug=False,
                 booster_party=False):
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
        #: forward the script's own `print` commentary into the run's
        #: log. A live toggle like the others -- see `_script_logging`.
        self.script_debug = bool(script_debug)
        self._stop_capture = None
        self._logged = {}
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
        #: successful scripted-teleport landings per outcome kind, for
        #: the thinned `teleport-landed` telemetry. See `teleported`.
        self._tp_landed = {}
        #: which wizard sets the pace, and whether the rest chase it.
        #: Without this a party is four wizards questing independently to
        #: four different places, coordinating perfectly with nobody --
        #: see `deimos_bridge/party.py`.
        self.leader = max(0, min(int(leader), len(self.seats) - 1))
        self.follow_leader = bool(follow_leader)
        #: booster party: the leader is the QUESTER -- the one wizard
        #: whose questline the run levels -- and every other seat is a
        #: BOOSTER, there to join the quester's fights and end them in
        #: fewer rounds. A booster does not quest, does not doctor its
        #: own quest tracker (its journal points wherever its book was
        #: left), and is never held back by the overkill guard. The
        #: operator's framing, verbatim: "the booster needs to join
        #: combats to beat them as quick as possible (but the booster
        #: doesnt need to quest)".
        self.booster_party = bool(booster_party)
        #: solo-pilot mode: the script drives ONLY the leader, and the
        #: rest of the party follows it and joins its fights. The whole
        #: class of failure this run has been fighting -- friend
        #: teleports that miss, one instruction pointer for four
        #: wizards, desync, catch-ups -- is the script COORDINATING a
        #: party, and the presets are documented to run solo when their
        #: account settings stay at placeholders. So: coordinate with
        #: wizAi's follow + hivemind instead, and let the script do the
        #: one thing it is good at, which is the route.
        self.solo_script = bool(solo_script)
        #: A booster party with a script MEANS solo-pilot script wiring:
        #: the script quests the leader alone, and the boosters are
        #: wizAi's to keep on it. Without this, a script makes EVERY
        #: seat script-driven, the booster branch of the tick is
        #: unreachable, and each mass `tp quest` scatters the boosters
        #: to their own stale journals -- rev 8a48fd42 sent a max-level
        #: booster through the world teleporter to Krokotopia to chase
        #: "Defeat Street Player" while its quester fought in
        #: Marleybone. Forced rather than rejected, because the raw
        #: checkboxes can spell this combination even though no mode
        #: names it; said out loud in `_go`.
        self._booster_solo_forced = False
        if self.booster_party and self.script and not self.solo_script:
            self.solo_script = True
            self._booster_solo_forced = True
        #: overrides for the script's own `SpeedDelay`/`DialogDelay`
        #: settings, or None to run it as written. The author's pacing
        #: is a knob, and editing a 14,000-line file is not a knob --
        #: see `scripts.set_pacing`.
        self.script_step_delay = script_step_delay
        self.script_dialog_delay = script_dialog_delay
        #: write which seat a client is onto its own title bar. Four
        #: identical "Wizard101" windows cannot be told apart, and the
        #: seat numbering only exists inside this program.
        self.label_windows = bool(label_windows)
        self._stop = False
        #: the catch-up in progress, or None. See `_start_catching_up`.
        self._catch_up_state = None
        #: until when the script takes no instructions because a
        #: desperate quest-teleport is in flight or settling. A
        #: monotonic deadline rather than a flag, so a hop that dies
        #: mid-teleport releases the script by itself. See
        #: `_desperate_hop` and `_hop_held`.
        self._hop_pause_until = 0.0
        #: until when the script takes no instructions because a wizard
        #: may be standing on a COUNTING dungeon sigil -- the state the
        #: script cannot see (a joined wizard shows no press-X prompt)
        #: and keeps cancelling with its next teleport. Deadline, like
        #: the hop's. See `_maybe_count_hold` and `_countdown_held`.
        self._count_hold_until = 0.0
        #: has this hold already stepped the wizard off its pad and
        #: back? That move un-joins a counting sigil, so it is worth
        #: exactly one counter restart per hold and no more. See
        #: `_join_leader`'s third rung.
        self._count_hold_stepped_off = False
        self._count_hold_zone = None
        self._count_hold_seat = -1
        self._count_hold_last = NEVER
        #: whether the running hold has sigil EVIDENCE -- a helper's
        #: press-X prompt seen (and pressed) at the spot. An evidenced
        #: sigil is never swept and re-arms on expiry; see the rung.
        self._count_hold_sigil = False
        #: the seat indexes gathered under the running hold, so a mate
        #: that crosses zones ALONE mid-hold -- the countdown fired
        #: without the holder -- releases it at once instead of burning
        #: out the clock. See `_maybe_count_hold`.
        self._count_hold_party = ()
        #: the running hold's boarding state: whether the holder is
        #: provably aboard, the spot to board from, which mates are
        #: waiting for their X, and when to try again. Boarding is
        #: flaky rather than impossible -- see `_retry_boarding`.
        self._count_hold_aboard = False
        self._count_hold_pad = None
        self._count_hold_joiners = ()
        self._count_hold_retry = 0.0
        #: when the running hold engaged, so a goal that advances AFTER
        #: it releases it -- the turn-in-not-a-sigil case.
        self._count_hold_began = 0.0
        #: (from zone, to zone) -> the position a wizard was standing on
        #: the last time it made that crossing. The party's shared map
        #: of doors, learned by watching, and the answer to a dungeon
        #: that refuses friends-list teleports: a follower cannot port
        #: to its leader, but it CAN walk the door its leader just
        #: walked. See `_learn_door` and `_walk_the_leaders_door`.
        self._doors = {}
        #: when the party last changed realm, and which realms this run
        #: has already tried. See `_realm_hop_party`.
        self._realm_hopped_at = NEVER
        self._realms_tried = set()
        #: when a looping script was last forcibly restarted. See
        #: `_maybe_restart_script`.
        self._script_restarted_at = NEVER
        #: the stale-reload backoff: the last reload's failure
        #: signature and time, the hold now in force, and whether this
        #: stale episode has already escalated it. See `_script_step`.
        self._reload_sig = ""
        self._reload_at = NEVER
        self._reload_hold_until = 0.0
        self._reload_cool = 0.0
        self._reload_held = False
        #: consecutive catch-ups ended by "a different wizard is
        #: behind now", and when the last was. See `CATCH_UP_CHURN`.
        self._churn = 0
        self._churn_at = NEVER
        #: {combat first name: friends-list full name}. What the
        #: quester's account settings have to be filled with, because
        #: every teleport that reads them matches the list exactly. See
        #: `_resolve_party_names`.
        self._full_names = {}
        self._names_tried_at = NEVER
        self._names_done = False
        #: whether wizAi owns dialogue for this script, and the source
        #: the answer was worked out from — so a reloaded or rewritten
        #: script is re-examined and an unchanged one is not. See
        #: `_dialogue_is_ours`.
        self._dialogue_ours = None
        self._dialogue_for = None
        #: which seat is currently stepping the VM, when it is not the
        #: seat that owns it. Kept only so the handover is said once
        #: rather than every tick. See `_script_seat`.
        self._script_driver = None
        #: True while some seat's tick is inside the party's one VM.
        #: `_script_step` is guarded by the DRIVE lock, which is per
        #: seat -- so when the driver moves and the original owner's
        #: loop then comes back to life, two tasks hold two different
        #: locks and step one instruction pointer. See `_script_step`.
        self._vm_stepping = False
        #: the step count and when it last CHANGED, so a script that has
        #: stopped executing is a fact rather than an inference. See
        #: `_check_script_alive`.
        self._steps_seen = None
        self._steps_at = NEVER
        #: {normalised quest name: (zone, when)} -- where the party was
        #: standing while somebody tracked that quest. The saved
        #: location half of lost-quest recovery: naming the quest a
        #: wizard should be back on is only half an answer if nothing
        #: can say where it is. Fed by the goal poll, which already
        #: reads both, and read by `_maybe_recover_questline`. Bounded;
        #: see `QUEST_ZONE_MEMORY`.
        self._quest_zone = {}
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
        # The loguru sink outlives this thread otherwise, and a second
        # run would then have two of them writing the same line twice.
        self._script_logging(False)

    #: what `request` accepts. Every one of these drives the mouse, so
    #: every one is serviced from the one task that owns it. "realm" is
    #: broadcast like the rest and collapses to one party hop via the
    #: cooldown stamp in `_realm_hop_party`.
    ACTIONS = ("teleport", "dialogue", "wisps", "potion", "realm")

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
        import time

        seat = self._seat_for(client) if seat is None else seat
        while not self._stop:
            try:
                # FIRST, before the read below, because the read below is
                # the one that can stop answering. Every other await in
                # this loop is inside `_stage`, which has a deadline; this
                # clock is how anything OUTSIDE the loop can tell that
                # this seat's loop has stopped coming round -- see
                # `_script_seat`.
                seat.ticked_at = time.monotonic()
                fighting = await self._read_in_battle(seat, client)
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
                # Through `_stage`, not bare. It is the one remaining
                # unbounded await in this loop despite the comment
                # above saying every other one is inside a deadline,
                # and it is a DIAGNOSTIC -- the instrument for finding
                # a stuck wizard must never be the reason one is stuck.
                await self._stage(seat, "the heartbeat",
                                  self._heartbeat(seat,
                                                  self._script_drives(seat),
                                                  fighting=fighting))
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
                # A live toggle like auto-quest's: read every tick, so
                # the switch works during a run rather than only at
                # Play live.
                self._script_logging(self.script_debug)
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
                if (self.auto_dialogue and seat.quester is None
                        and (not driven or self._dialogue_is_ours())):
                    # Deimos's questing does its own dialogue handling,
                    # so a second clicker would race it for the same
                    # button. A running script USUALLY is the same
                    # problem -- but every preset wizAi ships has its
                    # dialogue clearer switched off by a variable
                    # nobody declared, so for those there is no second
                    # clicker and standing back means nothing presses
                    # anything. See `_dialogue_is_ours`.
                    await self._stage(seat, "auto-dialogue",
                                      self._auto_dialogue(client), wheel=True)

                # Bounded, like everything else in the tick. It is five
                # memory reads on one client -- the quest name, the goal
                # line, the zone, the body position, the quest marker --
                # and any of them can be the one that stops coming back.
                #
                # Rev 35f0fc6e: Konstantin's tick stopped coming round
                # at t=4230 with no `client-not-answering` against it, so
                # it was not the combat read at the top; it was this,
                # which was the last bare await left in the loop. His
                # heartbeats stopped with it, because the loop never got
                # back to the top to write one.
                await self._stage(seat, "reading the quest",
                                  self._read_goal(seat))
                # Two reads on this seat's own client, once a second, and
                # they are what the door map is made of. On the party
                # poll's six-second clock the doors it learns are most of
                # a room wide of the doors they name. See `_note_spot`.
                await self._stage(seat, "noting where this wizard stands",
                                  self._note_spot(seat))
                # Ends any catch-up BEFORE deciding whether to start
                # one. The other order meant a catch-up started this
                # tick was judged finished on the same tick, by state
                # that had not had a chance to change -- which is
                # exactly what the first live run shows, `started` and
                # `done` sharing a timestamp.
                self._check_caught_up()
                self._check_in_step(seat)
                self._check_on_questline()
                self._check_script_alive()
                self._check_progress(seat)

                if self._hop_held():
                    # A desperate quest-teleport or a realm change is in
                    # flight or settling, and both drive clients ACROSS
                    # seats -- a follower's follow, another seat's quest
                    # step or the script's own tp loop steering a wizard
                    # whose spellbook is open undoes the maneuver from a
                    # different task. The operator's original report of
                    # the narrow version: "it's possible and even likely
                    # for the bot to teleport away from the quest after
                    # a desperate tp back to the place it was stuck". So
                    # the hold is worker-wide: nothing below this line
                    # steers anybody until the deadline lets go.
                    self._say_once(
                        seat, "script-held-hop",
                        "holding this wizard's steering while a teleport "
                        "or realm change finishes and settles")
                    await asyncio.sleep(0.5)
                    continue

                if driven and not self._is_booster(seat):
                    # The countdown guard, deliberately AHEAD of the
                    # script step: a wizard the script just teleported
                    # onto a dungeon sigil is standing in a countdown
                    # the script cannot see (a joined wizard shows no
                    # press-X prompt), and the script's very next
                    # teleport is the one that cancels it. The hold has
                    # to land before that instruction runs.
                    await self._stage(seat, "guarding a sigil countdown",
                                      self._maybe_count_hold(seat),
                                      wheel=True, limit=90)

                catching = self._catching_up()
                # Whichever seat is CURRENTLY stepping the VM, which is
                # seat 0 unless seat 0's loop has stopped coming round.
                # See `_script_seat`.
                driving = self._script_seat() is seat
                if driving and not catching and self._countdown_held():
                    # Not stepped, not torn down: the VM keeps its state
                    # and resumes when the hold releases -- to a changed
                    # zone if the sigil fired, to the same spot if not.
                    self._say_once(
                        seat, "countdown-held",
                        "script held — this wizard may be standing on a "
                        "counting sigil, and the script's next teleport "
                        "is the one that keeps cancelling the entry")
                elif driving and not catching:
                    await self._stage(seat, "script step",
                                      self._script_step(seat), wheel=True)
                elif driving:
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

                if driven:
                    # Before anything that depends on the script being
                    # able to regroup the party, because until this runs
                    # it cannot: its account settings hold a first name
                    # and every teleport that reads them wants the full
                    # one. Gated inside on "already known", so a run
                    # pays for it once.
                    await self._stage(seat, "reading the party's names",
                                      self._resolve_party_names(seat),
                                      wheel=True, limit=90)

                if (driven or self.auto_quest) and not self._is_booster(seat):
                    # The one rung that does not need the marker,
                    # because it exists for the state where the marker
                    # is DEAD: quest position unreadable for minutes
                    # while the goal reads fine. Everything below this
                    # line aims a teleport, and none of them can aim
                    # until this fires. Gated inside on `_marker_dead`,
                    # so the stage costs one time check on a healthy
                    # tick. Boosters skip this rung and the two below:
                    # a booster's own tracker points wherever its book
                    # was left, so "curing" it pages the quest book and
                    # hops realms -- stealing the wheel, and possibly
                    # the wizard, exactly when the quester's fight
                    # needs it.
                    await self._stage(seat, "re-arming the quest arrow",
                                      self._maybe_rearm_quest_arrow(seat),
                                      wheel=True, limit=90)

                if (driven or self.auto_quest) and not self._is_booster(seat):
                    # The other lost-tracker state, and the opposite
                    # one: the hook is written perfectly and pointed at
                    # a side quest, so every teleport below aims
                    # somewhere real and wrong. Gated inside on the
                    # off-the-line clock, so a healthy tick pays two
                    # comparisons.
                    await self._stage(seat, "recovering the main questline",
                                      self._maybe_recover_questline(seat),
                                      wheel=True, limit=90)

                if (driven or self.auto_quest) and not self._is_booster(seat):
                    # The rung above the desperate teleport: a Collect
                    # goal parked at its own marker for eight minutes is
                    # a crowded realm, not a wedged wizard. Cheap gates
                    # first; the stage only costs anything on the tick
                    # it actually hops.
                    await self._stage(seat, "changing realms",
                                      self._maybe_realm_hop(seat),
                                      wheel=True, limit=360)

                if (driven or self.auto_quest) and not self._is_booster(seat):
                    # The walk-in door: a wizard parked ON its marker
                    # with no transition, because the collision-solved
                    # teleport stops exactly at the marker and the
                    # door's trigger sits just past it. The rung the
                    # operator described as "walking forward to go
                    # through like the script used to". Cheap gates
                    # first; it reads nothing until the stall is old.
                    await self._stage(seat, "walking through the door",
                                      self._maybe_walk_through(seat),
                                      wheel=True, limit=90)

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
                elif self._is_booster(seat):
                    # A booster's whole job fits in one rung: be where
                    # the quester is, and when the quester is fighting,
                    # be IN the fight. `party.follow` already does both
                    # -- cross-zone teleport, land on the leader, step
                    # into the duel -- so the rung is the follow, under
                    # a name that says why. Deliberately NOT
                    # `_sync_follower`: a booster is not levelling this
                    # questline and has no step of its own to turn in.
                    await self._stage(seat, "boosting the quester",
                                      self._follow_step(client, seat),
                                      wheel=True)
                elif self._follows(seat) and self._solo_pilot():
                    # A solo-pilot follower is not merely an escort: it
                    # is levelling the SAME questline. It turns its own
                    # steps in at the places the pilot brings it to, and
                    # follows the rest of the time. See `_sync_follower`.
                    await self._stage(seat, "questing with the pilot",
                                      self._sync_follower(client, seat),
                                      wheel=True)
                elif self._follows(seat):
                    # A follower does not quest. Two wizards taking their
                    # own quests walk to two places, and then the party
                    # coordinates beautifully with nobody.
                    await self._stage(seat, "following the leader",
                                      self._follow_step(client), wheel=True)
                elif self.auto_quest:
                    await self._stage(seat, "quest step",
                                      self._quest_step(client), wheel=True)

                await asyncio.sleep(
                    0.1 if getattr(seat, "script_hot", False) else 0.5)
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
                    # Five memory reads. Generous against a client
                    # mid-zone-load, and finite against one that has
                    # stopped answering -- which is the whole point:
                    # this runs before the heartbeat can be written
                    # again, so a read that never returns takes the
                    # seat's entire tick silently off the air.
                    "reading the quest": 30.0,
                    "script": 20.0,
                    "script step": 240.0,
                    # A handful of memory reads and a log line. Bounded
                    # because it used to be awaited bare: a read that
                    # never returns took the seat's whole tick with it,
                    # and this is the stage that exists to REPORT that.
                    "the heartbeat": 30.0,
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
    #: below this health fraction, giving up the heal wait means dying,
    #: not fighting: rev 676d6e77 "gave up after 150s on 1% health" and
    #: walked back into the pack that had just killed him, twenty-two
    #: times over four hours. The wait stretches instead -- wisps do
    #: arrive given time, and the same export proves it.
    CRITICAL_HEALTH = 0.25
    CRITICAL_HEALTH_WAIT = 600.0
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

    #: how long the combat read at the top of the service tick may take
    #: before it is cut off. A healthy read is microseconds -- it is a
    #: memory read -- so ten seconds is not a threshold anybody's client
    #: crosses while working.
    #:
    #: It needs one because it is the ONLY unbounded await left in the
    #: loop: everything below it goes through `_stage`, which has a
    #: deadline. Rev f32be436 is what the gap costs. Sebastian's client
    #: stopped answering at t=5129 and his loop never came round again --
    #: no heartbeat, no stage timeout, no exception, because none of
    #: those can fire from inside a read that never returns. His seat
    #: owns the party's script (`_script_step` runs from seat 0 alone),
    #: so the instruction pointer froze at 23,324 and stayed there for
    #: 110 minutes, 56% of the run, while the other two wizards' logs
    #: went on reporting "script at 23,324 instructions" once a minute.
    IN_BATTLE_LIMIT = 10.0
    #: consecutive cut-off combat reads before the client is called
    #: unresponsive out loud. One is a hiccup; three in a row is a client
    #: that has stopped answering, which is a different fact and the one
    #: that explains everything downstream of it.
    NOT_ANSWERING_AFTER = 3

    async def _read_in_battle(self, seat, client):
        """Is this wizard in a duel? Bounded, and never raises.

        On a read that will not come back, the answer is the LAST one:
        a client that has stopped answering has not necessarily left its
        duel, and saying "out of combat" would send every rung below
        here to click at a wizard nothing can read.
        """
        from .. import questing

        try:
            got = bool(await asyncio.wait_for(questing.in_battle(client),
                                              self.IN_BATTLE_LIMIT))
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            seat.unanswered += 1
            if seat.unanswered == self.NOT_ANSWERING_AFTER:
                said = (f"{seat.name}'s client has not answered a combat "
                        f"read in {self.NOT_ANSWERING_AFTER} tries "
                        f"({self.IN_BATTLE_LIMIT:.0f}s each). Its own tick "
                        f"cannot run while a read is outstanding, so "
                        f"everything wizAi does for this wizard is on hold "
                        f"— including the party's script if this is the "
                        f"seat driving it")
                seat.tel.note_questing("client-not-answering", said)
                self._say(seat, said)
            return bool(seat.in_duel)
        except Exception:
            return False
        # Reset on a read that came back, so a client that recovers and
        # then stops again is reported the second time too. Kept out of
        # the `except` above on purpose: a client answering once between
        # two hangs has not recovered, and the counter only clears on a
        # read that actually returned.
        seat.unanswered = 0
        return got

    #: how long a seat's service loop may go without coming round before
    #: another seat takes the script off it.
    #:
    #: This used to read "three ticks' worth of the slowest stage",
    #: which was never true: `STAGE_LIMITS["script step"]` alone is
    #: 240, and a `waitfor zonechange` is bounded at 150 inside it. A
    #: seat doing exactly what the script told it to went quiet for
    #: longer than this and lost the script to its partner -- rev
    #: 09a0af80's `script-driver-moved` immediately followed by
    #: `waitfor-gave-up`.
    #:
    #: A running stage now keeps the clock going (`_still_ticking`), so
    #: this measures what it says: a loop that is not going round at
    #: all. Ninety seconds of that is a wedged client, not a slow
    #: instruction.
    DRIVER_QUIET = 90.0

    def _dialogue_is_ours(self):
        """Should wizAi click dialogue for a wizard the script is driving?

        The operator's report, and it is two symptoms of one cause: "the
        auto dialogue works but when scripting is enabled ... it waits a
        long time to start the dialogue / go through it".

        wizAi stood back from any scripted wizard's dialogue box on the
        reasoning that a second clicker would race the script's own. For
        every preset it ships, there is no first clicker. The general
        handler is::

            if any hasdialogue {
                print "Dialogue detected. Clearing..."
                if Handle_Dialogue = True {
                    sameany sendkey SPACEBAR, .1
                }
            }

        and `Handle_Dialogue` is declared nowhere. deimoslang answers an
        undefined constant with `False` without complaining
        (`vm.py:1109`), so the script prints that it is clearing the box
        and presses nothing, then sleeps `DialogDelay` and prints it
        again. The script's banner asks the operator to switch Deimos's
        own Dialogue toggle on -- and that is Deimos's GUI, which wizAi
        does not run.

        So nothing was clearing dialogue for a scripted wizard except
        `_unstick`, which is a wedge detector: it waits `DIALOG_WEDGE`
        for the box, then another `UNSTICK_EVERY` to see the same text
        twice. Nearly a minute of standing still, by design, because it
        was built for a box somebody else was failing to handle rather
        than for one nobody was touching.

        Rev ed709013 saw the symptom and drew the wrong conclusion --
        `Dialogue detected. Clearing...` fifteen times in eight
        milliseconds while General Khaba's MORE button sat unclicked --
        and the note it left says "measurably not being handled", which
        was right about the measurement and wrong about the cause.

        Checked per script rather than assumed either way: a script that
        declares the guard, or presses the key unguarded, still owns its
        own dialogue and wizAi stays out of it.
        """
        from .. import scripts

        source = self.script or ""
        if not source:
            return False
        if self._dialogue_ours is None or self._dialogue_for != source:
            self._dialogue_for = source
            guard = scripts.dead_dialogue_guard(source)
            self._dialogue_ours = bool(guard)
            if guard:
                said = (f"this script's dialogue clearer is switched off by "
                        f"`{guard}`, which it never declares — deimoslang "
                        f"reads an undefined name as False, so it prints "
                        f"\"Dialogue detected. Clearing...\" and presses "
                        f"nothing. wizAi is clicking dialogue for these "
                        f"wizards instead of standing back for a handler "
                        f"that does not run")
                for seat in self.seats:
                    try:
                        seat.tel.note_questing("script-dialogue-dead", said)
                    except Exception:
                        pass
                self._say(self.seats[0], said)
        return self._dialogue_ours

    def _script_seat(self):
        """The seat whose loop steps the party's one VM, or None.

        Seat 0 by design: one program, one instruction pointer, one
        stepper. The failure that makes this a function rather than a
        constant is rev f32be436 -- seat 0's loop stopped coming round
        and took the whole party's script with it, because no other
        seat's loop is allowed to step the VM.

        So the seat is chosen rather than assumed. Seat 0 keeps it while
        its loop is ticking; when it goes quiet past `DRIVER_QUIET`, the
        first seat that IS ticking picks the script up. Nothing about the
        VM moves -- it is still seat 0's `runner`, still driving `p1`..
        `p4` -- only which task calls `step()` on it.
        """
        import time

        owner = self.seats[0]
        if owner.runner is None:
            return None
        now = time.monotonic()
        if owner.ticked_at and now - owner.ticked_at <= self.DRIVER_QUIET:
            return owner
        for seat in self.seats[1:]:
            if seat.client is None or not seat.ticked_at:
                continue
            if now - seat.ticked_at <= self.DRIVER_QUIET:
                if self._script_driver is not seat:
                    self._script_driver = seat
                    quiet = (now - owner.ticked_at) / 60.0 if owner.ticked_at \
                        else 0.0
                    said = (f"{owner.name}'s tick has not come round for "
                            f"{quiet:.0f} min, and that seat is the one that "
                            f"steps the party's script — so {seat.name} is "
                            f"stepping it instead. Nothing else changes: it "
                            f"is the same program driving the same wizards")
                    for other in self.seats:
                        try:
                            other.tel.note_questing("script-driver-moved",
                                                    said)
                        except Exception:
                            pass
                    self._say(seat, said)
                return seat
        # Nobody is ticking. Leave it with seat 0 rather than answering
        # None -- None would read as "there is no script", which is a
        # different thing and unlocks rungs that must not run.
        return owner

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
        # A stage that does not take the wheel starts the moment it is
        # awaited; one that does may spend its whole deadline waiting.
        started = [not wheel]
        if wheel:
            coro = self._at_the_wheel(seat, name, coro, started)
        try:
            await self._still_ticking(seat, asyncio.wait_for(coro, limit))
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            if started[0]:
                said = (f"{name} ran for {limit:.0f}s without finishing and "
                        f"was cut off. It holds the wheel while it runs, so "
                        f"everything else for this wizard — the hotkeys "
                        f"included — was waiting behind it.")
                detail = f"{name} cut off after {limit:.0f}s"
            else:
                # Never started. Saying "ran for 90s" here sends whoever
                # reads it looking for a slow stage, and the slow thing
                # is whatever has been holding this wizard's wheel for
                # the whole ninety seconds.
                said = (f"{name} waited {limit:.0f}s for this wizard's drive "
                        f"lock and never got it, so it did not run at all — "
                        f"something else has been steering this wizard that "
                        f"whole time.")
                detail = (f"{name} never started — {limit:.0f}s waiting for "
                          f"the drive lock")
            self._say_once(seat, name, said,
                           kind="stage-timeout", detail=detail)
        except Exception as exc:
            self._stage_failed(seat, name, exc)

    #: how often a running stage restamps its seat's liveness clock.
    #: Small against `DRIVER_QUIET`; large against anything the loop
    #: does per tick.
    TICK_STAMP = 5.0

    async def _still_ticking(self, seat, awaitable):
        """`awaitable`, with this seat's liveness clock kept running.

        `seat.ticked_at` is stamped once at the top of the tick and
        nowhere else, and one tick's stages sum to about 1,425 seconds
        against a `DRIVER_QUIET` of 90. `STAGE_LIMITS["script step"]`
        is 240 on its own, and a `waitfor zonechange` is bounded at 150
        INSIDE it -- so one ordinary instruction took a seat past the
        liveness threshold and handed the party's script to the other
        seat. Rev 09a0af80's export has exactly that: `script-driver-
        moved` followed by `waitfor-gave-up`, which is one event
        reported twice, and the second seat picking up a program the
        first was in the middle of.

        A stage that is RUNNING is not a loop that has stopped coming
        round. So the clock runs while it runs, and what `DRIVER_QUIET`
        now measures is what its name says: a seat whose loop is not
        going round at all -- wedged in `_read_in_battle`, or gone. A
        stage that genuinely hangs is still caught, by its own deadline
        in `STAGE_LIMITS`, which is the bound built for it.
        """
        import time

        task = asyncio.ensure_future(awaitable)
        try:
            while True:
                done, _pending = await asyncio.wait({task},
                                                    timeout=self.TICK_STAMP)
                seat.ticked_at = time.monotonic()
                if done:
                    return task.result()
        finally:
            if not task.done():
                task.cancel()

    async def _at_the_wheel(self, seat, name, coro, started=None):
        """`coro`, holding this wizard's drive lock.

        Inside the stage's own deadline rather than outside it, which is
        the whole point: the *acquisition* is bounded too, so a stage
        that cannot get the wheel is cut off and reported instead of
        queueing up behind a wedge and adding to it.

        `started` is a one-element list the caller reads afterwards, and
        it exists because those two outcomes are not the same thing.
        A stage cut off while still WAITING for the lock never ran at
        all -- and its coroutine was never awaited, which Python says
        out loud:

            RuntimeWarning: coroutine 'LiveWorker._unstick' was never
            awaited

        Rev f2b8101f printed exactly that next to `stage-timeout:
        unwedging a stuck script cut off after 90s`, and the two are one
        event: `_unstick` did not run for 90 seconds, it waited 90
        seconds for a lock somebody else was holding and was then
        dropped. Closing it here is what stops the warning; telling the
        caller is what stops the export from saying the wrong thing.
        """
        try:
            async with self._driving(seat, name):
                if started is not None:
                    started[0] = True
                await coro
        finally:
            if started is not None and not started[0]:
                # Never awaited, so close it rather than leaving Python
                # to garbage-collect an un-started coroutine.
                coro.close()

    def _stage_failed(self, seat, name, exc):
        self._say_once(seat, name,
                       f"{name} failed — {type(exc).__name__}: {exc}",
                       kind="stage-failed",
                       detail=f"{name}: {type(exc).__name__}: {exc}")

    #: how many repeats of the same stage failure before the questing log
    #: says so again. The status line thins out at 20 because it is read
    #: live; the export is read afterwards, and 60 sets where the
    #: geometric cadence starts.
    STUCK_EVERY = 60

    @staticmethod
    def _thinning(n, every):
        """The 1st repeat, then `every`, then doubling: 2×, 4×, 8×…

        Linear cadence made the pattern BE the log: rev 30e83468 wrote
        `catch-up-out-of-zone — 240 times in a row` into the export
        every twelve seconds for as long as the run lasted, one entry
        per sixty ticks of a condition that was never going to change.
        Doubling keeps a stall visible at a cost that grows with its
        length's logarithm instead.
        """
        if n == 1:
            return True
        times, remainder = divmod(n, every)
        return remainder == 0 and times & (times - 1) == 0

    def _say_once(self, seat, key, message, kind="", detail=""):
        """Say it the first time, then geometrically less often.

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
        if self._thinning(n, 20):
            self._say(seat, message + (f" (still failing after {n} tries)"
                                       if n > 1 else ""))
        if kind and self._thinning(n, self.STUCK_EVERY):
            try:
                seat.tel.note_questing(
                    kind, detail + (f" — {n} times in a row" if n > 1 else ""))
            except Exception:
                pass

    def _broadcast_once(self, key, message, kind):
        """`_say_once`, but written to EVERY seat's export.

        For facts about the party rather than about one wizard -- an
        instruction that failed for somebody, which any of the three
        exports should show. Those were written flat, once per seat per
        occurrence, and rev 35f0fc6e is what that costs: 1,336 identical
        `should_update` timeouts in each of three exports, 4,008 lines
        of one sentence, crowding out the entries that would have
        explained the run around them.

        The count lives on seat 0 so the thinning is the party's and not
        three independent schedules that drift apart.
        """
        owner = self.seats[0]
        n = owner.stage_errors.get(key, 0) + 1
        owner.stage_errors[key] = n
        if self._thinning(n, 20):
            self._say(owner, message + (f" (still failing after {n} tries)"
                                        if n > 1 else ""))
        if not self._thinning(n, self.STUCK_EVERY):
            return
        detail = message + (f" — {n} times in a row" if n > 1 else "")
        for seat in self.seats:
            try:
                seat.tel.note_questing(kind, detail)
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
            if self._hop_held():
                # A teleport or realm change is driving clients across
                # seats. The press keeps its place in the queue and runs
                # after the settle window, rather than steering a wizard
                # whose spellbook is open from a second task.
                return
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
                                  wheel=True,
                                  # A realm change is a scan plus a hop
                                  # per wizard with 45s load waits; the
                                  # default deadline cuts it mid-party.
                                  limit=360 if action == "realm" else None)
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
        elif action == "realm":
            await self._realm_hop_party(seat, "the operator asked for a "
                                              "realm change")
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

        Except in solo-pilot mode, where the VM was built over the
        leader's client alone -- the others are not `p2`..`p4` at all,
        so nothing scripted ever moves them and they are free to follow.
        """
        runner = self.seats[0].runner
        if runner is None or not runner.running:
            return False
        if self.solo_script:
            return seat.index == self.leader
        return True

    def _solo_pilot(self):
        """Is this run a solo-pilot script run with followers?"""
        return (self.solo_script and bool(self.script)
                and len(self.seats) > 1)

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
        # The debug flag is part of what the runner was BUILT from, not
        # just the text: `_fresh_source` bakes `set_debug` into the
        # build, so a runner built with the flag off keeps `DebugMode =
        # False` no matter what the checkbox says now. The 115-minute
        # run at rev f2b8101f shows the gap: "Script log" was ticked
        # mid-run, the capture attached instantly — and nothing printed
        # for the next stretch, because the running build still had
        # DebugMode off. The export's script log only starts at t=1054
        # because the name fill happened to rebuild the script 44
        # seconds after the tick. Without that coincidence it would
        # have stayed empty for the rest of the run.
        # So is the wiring: a solo VM is built over the leader's client
        # alone and a party VM over all of them, so switching the
        # questing mode or the Leader mid-run changes what the runner
        # must be BUILT from, not a flag it reads. Rev d4b5506c is the
        # cost of comparing text alone: the operator switched to
        # "Booster party + script" and pointed Leader at the quester
        # while a whole-party build kept running, and the script went
        # on walking the BOOSTER's questline with the quester in tow.
        wiring = (self._solo_pilot(),
                  self.leader if self._solo_pilot() else None)
        if seat.script_source == want \
                and seat.script_debug_built == self.script_debug \
                and seat.script_wiring_built == wiring:
            return
        if seat.script_source == want and seat.runner is not None:
            if seat.script_wiring_built != wiring:
                solo, lead = wiring
                self._say(seat,
                          ("the questing mode changed — rebuilding the "
                           "script over "
                           + (f"{self.seats[lead].name} alone"
                              if solo else "the whole party")
                           + " (a reload re-runs the route from the top; "
                             "its one-time setup stays skipped)"))
            else:
                self._say(seat,
                          f"script log turned "
                          f"{'on' if self.script_debug else 'off'}"
                          f" — reloading the script so its own DebugMode "
                          f"follows (a reload re-runs the route from the "
                          f"top; its one-time setup stays skipped)")
        seat.script_source = want
        seat.script_debug_built = self.script_debug
        seat.script_wiring_built = wiring
        if seat.runner is not None:
            seat.runner.stop()
            seat.runner = None
            if not want:
                self._say(seat, "script stopped")
        if want:
            await self._setup_script(seat.client, seat)

    def _note_reload(self, seat, why, runner=None):
        """Write a script reload into every export, with its cause.

        The gap the operator found by watching the screen instead of
        the log: "why does it go through the menu so much... it
        randomly opens up the settings". Every one of those is the
        program starting again from instruction 0 and re-running its
        setup, and a reload was only ever said on the status line --
        which nobody is reading an hour later. So the exports could
        not explain the one thing visible from across the room.
        """
        detail = f"the script was reloaded and starts again from the top — {why}"
        skipped = list(getattr(runner, "skipped_setup", None) or ())
        if skipped:
            detail += (f". Its one-time setup ({', '.join(skipped)}) is "
                       f"skipped on a reload — it has already run")
        for other in self.seats:
            try:
                other.tel.note_questing("script-reloaded", detail)
            except Exception:
                pass

    #: a reload whose script dies the same way within this much running
    #: time did not fix anything. The crash-loop cycle at rev 7d9b6d6b
    #: was ~15s: reload, march ~600 instructions back to the crashing
    #: quest check, raise 25 times, reload.
    RELOAD_THRASH = 120.0
    #: the ceiling on the doubling wait between such reloads. Ten
    #: minutes still retries -- the operator may fix the state live --
    #: without eleven reloads per export saying nothing new.
    RELOAD_COOL_MAX = 600.0

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
        import time

        seat = self.seats[0] if seat is None else seat
        # The VM belongs to seat 0 whoever is stepping it -- one program,
        # one instruction pointer. `seat` is only whose loop is calling.
        owner = self.seats[0]
        runner = owner.runner
        if runner is None:
            return
        # One stepper, enforced. The stage above holds this SEAT's drive
        # lock, and `_script_seat` can hand the script to another seat
        # and hand it back: seat 1 picks it up while seat 0 is quiet,
        # seat 0's loop comes round again, and now two tasks holding two
        # different per-seat locks are inside `run_for` on one
        # instruction pointer. Skipped rather than queued -- a second
        # burst of the same program is not wanted late either, and
        # waiting for it would spend this seat's whole stage deadline.
        if self._vm_stepping:
            return
        self._vm_stepping = True
        try:
            await self._step_the_vm(seat, owner, runner)
        finally:
            self._vm_stepping = False

    async def _step_the_vm(self, seat, owner, runner):
        """The body of `_script_step`, under its one-stepper guard."""
        import time

        if not runner.stale:
            done = await runner.run_for(
                should_stop=lambda: self._stop or seat.in_upkeep)
            # Executed something and has more to do: the tick should come
            # straight back rather than sleeping its usual half second.
            # The sleep between bursts was half of the operator's "long
            # delay between actions" -- the other half is the script's
            # own SpeedDelay -- and the script pausing 0.6s for every
            # 0.5s it ran doubled the cost of every step the author
            # priced.
            seat.script_hot = bool(done) and runner.running
        if runner.stale:
            # An instruction had to be cancelled, so the VM is part-way
            # through one. Reloading is the only honest recovery --
            # ONCE. Rev 7d9b6d6b: one wizard's journal state made a
            # quest check raise, and reload -> march back -> raise
            # cycled every 15 seconds, eleven reloads in three minutes,
            # the party frozen throughout. A reload that dies the same
            # way it died last time is not a recovery, it is a
            # metronome -- so an identical failure straight after a
            # reload backs off, doubling, and the export says which
            # instruction is doing it.
            why = runner.last_error
            sig = getattr(runner, "stale_sig", "") or why
            now = time.monotonic()
            if now < self._reload_hold_until:
                return
            thrashed = (sig and sig == self._reload_sig
                        and now - self._reload_at < self.RELOAD_THRASH
                        and not self._reload_held)
            if thrashed:
                self._reload_cool = min(max(self._reload_cool * 2, 30.0),
                                        self.RELOAD_COOL_MAX)
                self._reload_hold_until = now + self._reload_cool
                self._reload_held = True
                said = (f"the script dies the same way straight after "
                        f"every reload ({why}). Reloading is not fixing "
                        f"this, so the next reload waits "
                        f"{self._reload_cool:.0f}s — the state it crashes "
                        f"on is what needs fixing")
                for other in self.seats:
                    try:
                        other.tel.note_questing("script-reload-backoff",
                                                said)
                    except Exception:
                        pass
                self._say(seat, said)
                return
            self._say(seat, why)
            if not runner.restart():
                self._say(seat, "script stopped — it could not be reloaded")
                self._note_reload(seat, f"could NOT be reloaded — {why}")
                owner.runner = None
                return
            if (sig != self._reload_sig
                    or now - self._reload_at >= self.RELOAD_THRASH):
                # A different failure, or one the last reload survived a
                # while before -- the backoff starts over.
                self._reload_cool = 0.0
            self._reload_sig, self._reload_at = sig, now
            self._reload_held = False
            self._note_reload(seat, why, runner)
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
            owner.runner = None

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

    def _party_shape(self, seat):
        """What this wizard is FOR, and who it is following.

        Named rather than inferred, because the modes are not visible
        in the telemetry they change the meaning of. A booster's quest
        goal differing from the leader's is the design; the same two
        lines in a plain follow are a party that has walked apart. An
        export that cannot tell them apart cannot be read at all.
        """
        if len(self.seats) < 2:
            mode = "solo"
        elif self.booster_party:
            mode = "booster party"
        elif self._solo_pilot():
            mode = "solo pilot"
        elif self.follow_leader:
            mode = "follow the leader"
        else:
            mode = "independent"
        boss = self.seats[self.leader] if self.seats else None
        return {
            "mode": mode,
            "seats": len(self.seats),
            "leader_seat": self.leader + 1,
            "leader": (boss.wizard_name or boss.name) if boss else "",
            "role": ("leader" if seat.index == self.leader else
                     "booster" if self._is_booster(seat) else "follower"),
            "scripted": bool(self.script),
            "script_drives": ("nobody" if not self.script else
                              "the leader only"
                              if self._solo_pilot() or self.booster_party
                              else "every seat"),
        }

    def _restamp_party(self):
        """Rewrite every seat's party block from what is true NOW.

        `_party_shape` is stamped once at connect time, and every fact
        in it can change afterwards: `_setup_script` has not run yet,
        no duel has named a wizard yet, and the GUI writes
        `auto_quest`, `booster_party`, `solo_script`, `follow_leader`
        and `leader` straight onto the running worker while it works.

        So a block written once is a claim about a run that had not
        started. Rev 09a0af80's export said mode "follow the leader",
        `scripted: false`, `script_drives: "nobody"` and leader
        "wizard 1" -- for a scripted booster party. Every other line in
        an export means something different depending on this one, and
        this one was wrong.

        Never raises: it describes the run, and nothing that merely
        describes the run may stop it.
        """
        for seat in self.seats:
            try:
                if seat.tel is not None:
                    seat.tel.party = self._party_shape(seat)
            except Exception:
                pass

    #: how many times one opening may be lost before the party is told
    #: it is a wall rather than bad luck. Two, because the second loss
    #: is the first evidence that the first was not variance -- and the
    #: third costs another ten minutes to learn nothing.
    BOSS_WALL_LOSSES = 2

    def _note_the_loss(self, seat, won):
        """Record a defeat, and say when one keeps happening.

        A lost duel used to be written into `FightRecord.won` and read
        by nothing but the export's win count and a red label in the
        panel. So rev 8ebfcf70's Konstantin walked into `Drusilla
        Morningbane@2400+War Wyrm@685` alone three times, lost all
        three, and spent thirty of the run's forty-seven minutes doing
        it -- with no line anywhere in the log saying a fight had been
        lost at all, let alone the same one three times.

        The two facts that make it actionable are the opening (which
        is already the join key between two wizards' exports) and how
        many seats were actually in the plan. One wizard against a
        2400hp boss in a party of two is not a hard fight, it is a
        party that did not arrive.
        """
        import time

        if won is not False or not seat.tel.fights:
            return
        fight = seat.tel.fights[-1]
        opening = fight.opening or "an unnamed board"
        now = time.monotonic()
        where = seat.zone_seen or "an unknown zone"
        # How many wizards were actually in the circle for THIS fight --
        # not for the run. `planned_alone` answers the run-wide version
        # and is an export statistic; the question here is whether the
        # duel that was just lost had a party in it.
        played = [r for r in seat.tel.rounds
                  if r.fight == fight.index and not r.passing]
        alone = bool(played) and all((r.seats_in_plan or 1) <= 1
                                     for r in played)
        seat.tel.note_questing(
            "lost",
            f"{seat.name} lost to {opening} after {fight.rounds} round(s) "
            f"in {where}"
            + (" — every round of it planned alone" if alone else "")
            + ". This is the only place a defeat is written down: the "
            "fight record's outcome is read by the win count and nothing "
            "else")
        seen = seat.lost_to.setdefault(opening, [])
        seen.append(now)
        if len(seen) != self.BOSS_WALL_LOSSES:
            return
        live = [s for s in self.seats if s.client is not None]
        short = (f" — fighting it alone while {len(live)} wizard(s) were "
                 f"in the run" if alone and len(live) > 1 else "")
        for other in self.seats:
            try:
                other.tel.note_questing(
                    "boss-wall",
                    f"{seat.name} has now lost to {opening} "
                    f"{len(seen)} times in {where}{short}. Walking back in "
                    f"a third time costs the same minutes and learns "
                    f"nothing: either the party has to arrive together or "
                    f"this fight needs a different deck")
            except Exception:
                pass
        self._say(seat,
                  f"{seat.name} has lost to {opening} {len(seen)} times — "
                  f"this fight is a wall, not bad luck")

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

    def _fresh_source(self, seat, source):
        """(source, skipped) — trimmed once this seat has run the script.

        A REBUILD is no more a first start than a restart is. The name
        fill (`_fill_script_names`) changes the script text the moment
        the first duel names the party, which rebuilds the runner from
        scratch -- and rev 613ab86a shows what that cost: Sebastian's
        instruction counter reset from 7,712 to 2,114 right after
        `script-configured`, meaning the program began again at
        instruction 0 and walked the settings menus a second time. The
        setup had already run minutes earlier.

        Keyed on this seat having built a runner before, so the very
        first build of a run always executes the program as written.
        """
        from .. import scripts

        # The operator's debug switch, applied to whatever text is about
        # to be compiled -- first build, rebuild and restart alike, so a
        # toggle cannot be undone by the next reload.
        try:
            source = scripts.set_debug(source, self.script_debug)
        except Exception:
            pass
        if not getattr(seat, "script_built", False):
            return source, []
        try:
            return scripts.restart_source(source)
        except Exception:
            return source, []

    #: how many times one distinct script line is written down before
    #: it is thinned. The dispatch prints on every pass, so an
    #: unthinned capture would be the whole export.
    LOG_THIN = (1, 2, 5, 20, 100, 500, 2000)
    #: ...and a hard ceiling on how many of those markers the questing
    #: timeline may hold, whatever the script says. Thinning by CONTENT
    #: bounds nothing when every line is different -- a dispatch that
    #: prints a new zone name each pass would still write one marker
    #: apiece and evict the heartbeats it was meant to sit beside. The
    #: full stream is in `script_log` and loses nothing.
    LOG_MARKERS = 200

    def _script_logging(self, on):
        """Start or stop forwarding the script's `print` output.

        Idempotent, and safe to call from a toggle: `capture_prints`
        answers None when loguru is missing, which simply leaves the
        feature off rather than failing a run.
        """
        from .. import scripts

        if on and self._stop_capture is None:
            self._stop_capture = scripts.capture_prints(self._note_script_log)
            if self._stop_capture is None:
                self._say(self.seats[0],
                          "script debug output is on, but loguru is not "
                          "available to capture it")
            else:
                self._say(self.seats[0],
                          "script debug output on — the script's own "
                          "`print` lines now go to this log, which is how "
                          "it says which leg of its route it is running")
        elif not on and self._stop_capture is not None:
            try:
                self._stop_capture()
            except Exception:
                pass
            self._stop_capture = None

    def _note_script_log(self, text):
        """One line of the script's own commentary, thinned by content.

        Thinned rather than dropped: the SAME line repeating is what a
        wedged route looks like, and the count is the evidence -- so the
        first, second, fifth... occurrence is written and the ones
        between are not. Called from loguru's thread, so it only
        touches telemetry and the status signal, both of which are
        already used across threads.
        """
        text = (text or "").strip()
        if not text:
            return
        # EVERY line, in order, into its own stream. The operator asked
        # for all of it, and the sequence is the diagnosis: a thinned
        # sample says a leg ran often, the full stream says which legs
        # it alternates between -- the difference between a route stuck
        # in one place and a route cycling and matching nothing.
        for seat in self.seats:
            try:
                seat.tel.note_script(text)
            except Exception:
                pass
        # ...and a thinned marker in the questing timeline, so the
        # narrative the operator actually reads still shows that the
        # script was talking, and how much. `questing` is capped at
        # 2000 entries; unthinned this would evict every heartbeat and
        # stuck-detail within minutes of a chatty quester.
        n = self._logged.get(text, 0) + 1
        self._logged[text] = n
        if len(self._logged) > 400:
            self._logged.clear()
        try:
            self.status.emit(text)
        except Exception:
            pass
        if n not in self.LOG_THIN:
            return
        self._markers = getattr(self, "_markers", 0) + 1
        if self._markers > self.LOG_MARKERS:
            if self._markers == self.LOG_MARKERS + 1:
                for seat in self.seats:
                    try:
                        seat.tel.note_questing(
                            "script-log",
                            f"the script has printed {self.LOG_MARKERS} "
                            f"distinct lines — the rest are in the "
                            f"script_log stream only, so this timeline "
                            f"keeps room for the run's own events")
                    except Exception:
                        pass
            return
        said = text if n == 1 else f"{text} — {n} times"
        for seat in self.seats:
            try:
                seat.tel.note_questing("script-log", said)
            except Exception:
                pass

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
        seat.script_debug_built = self.script_debug
        seat.script_wiring_built = (self._solo_pilot(),
                                    self.leader if self._solo_pilot()
                                    else None)
        seat.script_said = False
        party = [s.client for s in self.seats if s.client is not None]
        try:
            # The operator's pacing, if any, before either path builds:
            # the same script text feeds both, and a knob that only
            # worked in one mode would read as broken in the other.
            source, paced = scripts.set_pacing(
                self.script, self.script_step_delay,
                self.script_dialog_delay)
            if paced:
                self._say(seat, "script pacing — " + ", ".join(
                    f"{n} = {v}s" for n, v in paced))
            source, steadied = scripts.steady_sigil(source)
            if steadied:
                self._say_once(
                    seat, "sigil-steadied",
                    "sigil entry hardened — the preset's fixed 10s "
                    "countdown wait becomes `waitforzonechange "
                    "completion`: it holds on the sigil until the "
                    "dungeon load actually starts, through any number "
                    "of counter restarts (a booster stepping on late "
                    "RESTARTS the counter, and the follow makes that "
                    "the normal case), bounded at 150s with a logged "
                    "give-up",
                    kind="sigil-steadied",
                    detail="the Enter_Sigil sleep-10 bet was replaced "
                           "at load; see scripts.steady_sigil")
            if self._solo_pilot():
                # The pilot's client and nobody else's. `solo_source`
                # puts the account settings back to their placeholders
                # -- the dialog may have filled real names in -- so the
                # script takes its own documented solo path and every
                # p2..p4 branch is skipped by its own guards. The rest
                # of the party is wizAi's to move: `_follows` says they
                # chase the pilot, and the hivemind has them the moment
                # they step into its duels.
                pilot = self.seats[self.leader]
                source, reset = scripts.solo_source(source)
                source, skipped_setup = self._fresh_source(seat, source)
                seat.runner = scripts.make_runner(
                    [pilot.client or client], source, solo=True)
                seat.script_built = True
                # A rebuild starts the program again from instruction 0,
                # and the whole-party branch below writes that into
                # every export. This one did not -- so rev e6201303's
                # step counter fell from 13,846 to 1,213 between two
                # heartbeats with nothing anywhere to say why, and a
                # reader can only blame the watchdog reload four
                # minutes earlier. The most consequential event in a
                # supervised script run was invisible in exactly the
                # mode the run was in.
                seat.runner.skipped_setup = list(skipped_setup)
                if skipped_setup:
                    self._note_reload(
                        seat, "the script was rebuilt (its text changed)",
                        seat.runner)
                what = ("boosters — they keep to it and join its fights, "
                        "their own journals ignored"
                        if self.booster_party else "others follow and fight")
                self._say(seat,
                          f"script loaded — "
                          + ("booster party: the script quests "
                             if self.booster_party else "solo pilot: it drives ")
                          + f"{pilot.name} alone"
                          + (f" ({len(reset)} account setting(s) reset to "
                             f"placeholders so it quests solo)"
                             if reset else "")
                          + f"; the {what}"
                          + (". Booster party with a script means exactly "
                             "this, so solo-pilot wiring was engaged even "
                             "though its checkbox was off"
                             if self._booster_solo_forced else ""))
                return
            source, skipped_setup = self._fresh_source(seat, source)
            seat.runner = scripts.make_runner(party or [client], source)
            seat.script_built = True
            # So a later restart's report agrees with this one, and does
            # not claim to skip what this build already left out.
            seat.runner.skipped_setup = list(skipped_setup)
            if skipped_setup:
                self._note_reload(
                    seat, "the script was rebuilt (its text changed)",
                    seat.runner)
            self._say(seat, "script loaded"
                      + (f" — driving {len(party)} wizard(s)"
                         if len(party) > 1 else "")
                      + (f"; its one-time setup "
                         f"({', '.join(skipped_setup)}) is skipped — it "
                         f"has already run this session" if skipped_setup
                         else ""))
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

    def _is_booster(self, seat):
        """Is this seat a booster -- muscle for the quester's fights?

        Only ever true in booster-party mode, and never of the leader:
        the leader IS the quester the boosters exist for.
        """
        return (self.booster_party and len(self.seats) > 1
                and seat.index != self.leader)

    def _follows(self, seat):
        """Is this seat a follower rather than the one setting the pace?

        In solo-pilot mode every non-leader follows, whatever the
        follow checkbox says -- following IS the mode. A follower that
        stood still would just watch the pilot walk away, and one that
        took its own quest would coordinate beautifully with nobody.
        A booster follows for the same reason with the same force: a
        booster that does not chase the quester boosts nothing.
        """
        if self._is_booster(seat):
            return True
        if self._solo_pilot():
            return seat.index != self.leader
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

        # The game may already be naming the leader for us: a follower
        # whose Quest Helper tracks the leader carries `Quest Helper
        # Following <full name>` in its own goal line -- the exact
        # spelling the friends list holds, before any duel has read a
        # name. Rev 676d6e77's booster spent four hours one zone from
        # its quester for want of this string, while displaying it.
        harvested = party.helper_followed_name(seat.goal)
        if harvested:
            known = boss.wizard_name or ""
            longer = (harvested.lower().startswith(known.lower() + " ")
                      and len(harvested) > len(known))
            if not known or longer:
                if known:
                    party._FULL_NAMES.setdefault(known, harvested)
                boss.wizard_name = harvested
                seat.tel.note_questing(
                    "leader-name-learned",
                    f"the leader's full name read off this wizard's own "
                    f"Quest Helper line: {harvested!r} — the friends-list "
                    f"teleport can aim now, no duel needed")
                self._say(seat, f"learned the leader's name from the "
                                f"Quest Helper: {harvested}")

        moved, why = await party.follow(client, boss.client,
                                        leader_name=boss.wizard_name)
        # A teleport the GAME refuses is not a wizAi failure to retry
        # harder: most dungeon instances allow friends-list ports and
        # some do not, and inside one that does not, every retry is
        # another "your friend is busy". The way in is the way the
        # leader itself went -- see `_walk_the_leaders_door`.
        if (not moved and why and seat.zone_seen and boss.zone_seen
                and seat.zone_seen != boss.zone_seen):
            seat.cross_zone_fails += 1
            walked, door_why = await self._walk_the_leaders_door(
                seat, boss, why)
            if walked:
                moved, why = True, door_why
            elif door_why:
                why = f"{why} — and {door_why}"
        if moved and why:
            self._say(seat, why)
        if moved or not why:
            seat.follow_failing_since = None
            seat.follow_fails = 0
            seat.cross_zone_fails = 0
            return
        # A follower that cannot reach its leader is the failure that
        # makes the whole party pointless, so it is said -- but
        # thinned, because the cause is usually standing. EXPORTED,
        # too: rev d3ed4d3c's Oz missed the whole Water Dojo fight
        # (Konstantin soloed Yochimo) and the export contained not
        # one word about why -- the follow's failures only went to
        # the status bar, which nobody was watching.
        self._say_once(seat, "follow", why, kind="follow-failed",
                       detail=why)
        # ...and thinning is for noise, not for emergencies. Rev
        # 676d6e77: the booster failed this follow continuously for
        # FOUR HOURS -- the quester died twenty-three times fighting
        # alone one zone over -- and the geometric thinning let exactly
        # one line into the export. A stranding that lasts minutes is
        # the booster mode not existing, so it reports on its own
        # steady clock, with the verbatim current reason, however long
        # it goes on.
        if seat.follow_failing_since is None:
            seat.follow_failing_since = now
            seat.follow_fails = 0
        seat.follow_fails += 1
        stuck = now - seat.follow_failing_since
        if (stuck >= self.STRANDED_ALARM_EVERY
                and now - seat.stranded_said_at >= self.STRANDED_ALARM_EVERY):
            seat.stranded_said_at = now
            alarm = (f"{seat.name} has been unable to reach "
                     f"{boss.wizard_name or boss.name} for "
                     f"{stuck / 60:.0f} min ({seat.follow_fails} attempts). "
                     f"Latest reason: {why}")
            for other in self.seats:
                try:
                    other.tel.note_questing("follower-stranded", alarm)
                except Exception:
                    pass
            self._say(seat, alarm)

    #: how often a follower that KEEPS failing to reach its leader says
    #: so, per seat — a steady clock deliberately outside `_say_once`'s
    #: geometric thinning, because a stranding that lasts hours is not
    #: noise to thin, it is the run's headline.
    STRANDED_ALARM_EVERY = 300.0

    #: goals a follower must NOT chase on its own. A defeat step is
    #: satisfied by the PILOT's duels -- everyone in the circle who has
    #: the quest gets the kill -- and a follower that hops to its own
    #: defeat marker starts a fight alone, which is how Phönix died at
    #: rev 3d026ada. Talk, collect and explore steps are per wizard and
    #: are exactly what a follower has to do for itself.
    FIGHT_GOALS = ("defeat", "kill")
    #: how long a follower may work one of its own steps before going
    #: back to the pilot. A talk turn-in is seconds; a step still open
    #: after this is one this pass cannot finish.
    SYNC_GIVE_UP = 45.0

    async def _sync_follower(self, client, seat=None):
        """One tick of a solo-pilot follower: same questline, no lag.

        The operator's correction to the first cut of this mode, which
        let followers drift behind: "that wont work for leveling 3
        accounts at the same time ... they should all be on the same
        questline or else you're doing 3x the questing".

        They can be, because of how Wizard101 shares quest credit:

          * Defeat steps: everyone in the duel circle with the quest
            gets the kill. The follow already puts followers in the
            pilot's fights, so these advance for free.
          * Talk / collect / explore steps: per wizard -- and the pilot
            has just walked the party to the exact objective. All the
            follower has to do is turn its OWN step in while standing
            there.

        So, in priority order: finish a dialogue that is open; take the
        wizard's own step when its own marker is within this zone and
        the step is not a fight; otherwise follow the pilot. Taking a
        step first costs nothing in keeping up, because the follow is a
        teleport -- a follower that pauses ten seconds to turn a quest
        in lands back on the pilot the next follow tick.
        """
        import time

        from .. import questing

        seat = self._seat_for(client) if seat is None else seat
        if await questing.in_battle(client):
            return                       # the policy owns this wizard now
        if await questing.in_dialogue(client):
            n, _why = await questing.advance_dialogue(client)
            if n:
                self._say_once(seat, "sync-dialogue",
                               f"{seat.name} turned its own step in")
            return
        away = getattr(seat, "marker_away", None)
        goal = (seat.goal or "").strip().lower()
        fight = any(goal.startswith(w) for w in self.FIGHT_GOALS)
        now = time.monotonic()
        if away is not None and away <= self.MARKER_IN_ZONE and not fight:
            # Its own objective is at hand and it is not a fight: work
            # the step, and do NOT fall through to the follow meanwhile
            # -- a teleport back to the pilot mid-turn-in is the follow
            # undoing the sync. `hop_once` is the same hardened
            # teleport-and-interact the catch-up uses, and marker-in-
            # zone is the reachability test rev 1843e387 taught.
            if goal != seat.sync_goal:
                seat.sync_goal, seat.sync_began = goal, now
            if now - seat.sync_began > self.SYNC_GIVE_UP:
                # The step did not turn in. Staying here forever is a
                # follower lost to the party; go back to the pilot and
                # let the next pass at this objective try again.
                self._say_once(seat, f"sync-gave-up:{goal[:24]}",
                               f"{seat.name} could not finish its own step "
                               f"({seat.goal}) in {self.SYNC_GIVE_UP:.0f}s — "
                               f"following the pilot again")
            else:
                if now - seat.synced_at >= self.FOLLOW_EVERY:
                    seat.synced_at = now
                    await questing.hop_once(
                        client,
                        on_status=lambda m: self._say_once(
                            seat, f"sync:{m[:24]}", f"own step — {m}"))
                return
        await self._follow_step(client, seat)

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
                        on_plan=self._on_plan,
                        boosters={s.index for s in self.seats
                                  if self._is_booster(s)})
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
                # How this run was WIRED, in the file. Every other line
                # in an export means something different depending on
                # it -- "the follower charged ahead and the leader
                # chased" is the design in a booster party and a bug in
                # a plain follow -- and rev 8ebfcf70's two files cannot
                # answer which mode produced them at all. It cost an
                # afternoon of reading rungs backwards to work out
                # which wizard was even supposed to be leading.
                # ...and re-stamped as the run goes on. This is the
                # connect-time snapshot: it is written BEFORE the
                # script is loaded, BEFORE the first duel learns
                # `wizard_name`, and before any GUI push -- and the GUI
                # then mutates `auto_quest`, `booster_party`,
                # `solo_script`, `follow_leader` and `leader` on the
                # RUNNING worker. Rev 09a0af80's export reported mode
                # "follow the leader", `scripted: false`,
                # `script_drives: "nobody"` and leader "wizard 1" for a
                # run that plainly had a script and was a booster
                # party. The block was added for diagnosis and it
                # blocked the diagnosis. See `_restamp_party`.
                seat.tel.party = self._party_shape(seat)
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
            # The export is written after this. Whatever the run
            # ACTUALLY was -- the mode the GUI switched it to, the
            # script it was given, the names the duels learned -- is
            # known now and was not when the block was first stamped.
            self._restamp_party()
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
            # A new board is a new verdict. The run of no-surviving-line
            # rounds belongs to the duel that produced it.
            seat.no_line_survives = 0
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
            won = await self._fight_outcome(seat.client, seat)
            seat.tel.end_fight(won)
            self._note_the_loss(seat, won)
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
            if await seat.client.is_loading():
                # Nothing below reads true mid-load: window text comes
                # back blank, positions come back in the OLD zone's
                # space, and every blank feeds a keep-the-previous-value
                # rule somewhere. A load lasts seconds; skipping the
                # poll costs one `GOAL_POLL` and poisons nothing.
                return
        except Exception:
            pass
        try:
            goal = await questing.read_quest_goal(seat.client)
        except Exception:
            goal = ""
        if goal and goal != seat.goal:
            # A CHANGE is confirmed before it is believed. This read
            # resolves a window path whose fifth element is an UNNAMED
            # list slot (`questing.py`), and `window_from_path` returns
            # the FIRST match -- so which quest's line comes back can
            # change between two reads with nothing having happened in
            # the game. Rev 09a0af80's quester alternated between a
            # Zafaria quest and a Wysteria one while standing still in
            # the Zafaria hub, and 23 rungs consume `seat.goal`,
            # including the ones that hold the script, teleport the
            # party and pause the run.
            #
            # Deimos has carried a re-read for exactly this since
            # before wizAi existed (`Deimos/src/questing.py`); it was
            # never ported. One re-read costs `GOAL_CONFIRM`, and only
            # on the poll where the goal changed.
            goal = await self._confirmed_goal(seat, goal)
        self._note_goal(seat, goal, now)

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
        self._note_name(seat, name, now)

        zone = position = None
        try:
            zone = await seat.client.zone_name()
        except Exception:
            pass
        self._note_quest_zone(seat, zone, now)
        try:
            at = await seat.client.body.position()
            # Rounded hard: the idle animation moves a wizard by a
            # fraction constantly, and a progress check that counts
            # breathing as progress never fires.
            position = (round(at.x / 50.0), round(at.y / 50.0),
                        round(at.z / 50.0))
        except Exception:
            pass

        # How far the quest marker is, on the same poll that already read
        # the body's position. Consumed by `_start_catching_up`, which
        # cannot await -- and which must not start a catch-up for a
        # marker no within-zone teleport can reach. See `MARKER_IN_ZONE`.
        was_away = seat.marker_away
        seat.marker_away = None
        marker = None
        away = None
        try:
            marker, _why = await questing.read_quest_position(seat.client)
            if marker is not None and at is not None:
                dx, dy = at.x - marker.x, at.y - marker.y
                away = (dx * dx + dy * dy) ** 0.5
        except Exception:
            pass
        if (away is not None and away <= self.AT_THE_MARKER
                and (was_away is None or was_away > self.AT_THE_MARKER)):
            # A marker that has just come within arm's reach is the
            # single most consequential read this poll makes -- it is
            # what `marker_case` gates the whole sigil rung on -- and
            # it is the least trustworthy. `read_quest_position`
            # resolves to a hook installed in the game's quest-ARROW
            # render loop: one static pointer, last writer wins, with
            # no quest identity attached to what it wrote. The Wysteria
            # quest that flapped into rev 09a0af80's Zafaria hub read
            # 81 units away, then 0.
            #
            # So arriving is confirmed and leaving is not: a marker
            # going far needs no second opinion, because nothing acts
            # on distance.
            away = await self._confirmed_marker(seat, at, away)
        seat.marker_away = away
        # The dead-hook clock. One failed read is noise -- a zone change
        # makes several fail in a row -- but a marker that will not read
        # for minutes WHILE the goal line reads fine is the quest arrow
        # being off, which starves every teleport on this client at
        # once: wizAi's hops, the catch-up's, and the script's own
        # `tp quest`, all of which read the same hook. Keyed on the
        # MARKER read, not `marker_away`: a marker that read while the
        # body position did not is still a live hook.
        if marker is not None or not seat.goal:
            seat.marker_dead_since = None
        elif seat.marker_dead_since is None:
            seat.marker_dead_since = now
        if marker is not None:
            # ...and remember that it read, and on WHAT. A hook that
            # wrote a position on another quest minutes ago is not
            # switched off, so a later goal with no position is a quest
            # without a marker -- a Collect step, most often. See
            # `_marker_absent_by_design`.
            seat.marker_ok_at = now
            seat.marker_ok_goal = seat.goal or seat.marker_ok_goal

        self._note_progress(seat, zone, position, now)
        try:
            self.seat_quest.emit(seat.index, self._quest_row(seat, zone, now))
        except Exception:
            pass
        if zone and zone != seat.zone_for_writeoff:
            # A zone change makes a written-off step a different
            # proposition -- see `_written_off`. Kept here rather than in
            # `_check_together`, which only runs when there are two
            # wizards and only every `TOGETHER_POLL` seconds.
            #
            # ...unless this seat's quest hook is dead. The write-off
            # said "nothing can attempt this step", and no zone change
            # fixes an unwritten hook -- while a script lost enough to
            # wander (its `tp quest` reads the same dead hook) churns
            # zones constantly. Rev 98b4c50c: Konstantin's write-off was
            # cleared by KT_Hub -> KT_WorldTeleporter -> KT_Hub_Sphinx
            # laps six times in one run, and each clearing re-armed the
            # identical doomed catch-up: two minutes of the whole party
            # paused to attempt nothing, six times over.
            seat.zone_for_writeoff = zone
            if not self._marker_unusable(seat, now):
                self._forget_write_off(seat)

    #: how long a position cell is remembered. A wizard back on a spot it
    #: stood on within this window, same zone and same goal, has not
    #: gone anywhere -- it has bounced.
    OSCILLATION_WINDOW = 180.0
    #: how long a quest name that will not read is kept before it counts
    #: as unknown. Keeping the previous value on a blank read is right
    #: (a blank is not evidence of change) -- for a while. Past this,
    #: the kept value is invention: rev 30e83468's Sebastian was held
    #: two quests behind by a name that had stopped reading.
    NAME_MAX_AGE = 300.0

    def _note_progress(self, seat, zone, position, now):
        """Advance the progress clock -- unless the movement is a bounce.

        The stuck clock is what every backstop hangs off (`_unstick`,
        `_check_progress`, `_desperate_hop`), and it reset on ANY
        position change. A script retry loop that TELEPORTS the wizard
        between two spots therefore reset it forever: the heartbeats
        said "moving" while the operator watched a wizard standing
        still, and the backstops built for exactly that wedge were the
        one thing the wedge switched off. The stack audit named it
        starvation (F6).

        A bounce is a return to a cell this wizard already stood in
        within `OSCILLATION_WINDOW`, in the same zone and on the same
        goal. Real progress -- a new cell, a new zone, a new goal --
        still resets the clock on sight, and `seat.progress` (the spot
        stamp the press-X and hop memories key on) deliberately stays
        on the first cell of a bounce pair: the pair is one situation,
        not two fresh chances.
        """
        where = (zone, position, seat.goal)
        if where != seat.progress:
            bounce = (seat.progress is not None
                      and position is not None
                      and zone == seat.progress[0]
                      and seat.goal == seat.progress[2]
                      and now - seat.cells_seen.get(position, -1e9)
                      < self.OSCILLATION_WINDOW)
            if not bounce:
                seat.progress = where
                seat.progress_at = now
                seat.said_stuck = ""
        if position is not None:
            seat.cells_seen[position] = now
            if len(seat.cells_seen) > 32:
                cut = now - self.OSCILLATION_WINDOW
                for cell, at in list(seat.cells_seen.items()):
                    if at < cut:
                        del seat.cells_seen[cell]

    #: how many quests the zone memory keeps. A world's main line is
    #: forty-odd quests and a run touches one world; two hundred is
    #: several worlds' worth and still a few kilobytes.
    QUEST_ZONE_MEMORY = 200

    def _note_quest_zone(self, seat, zone, now):
        """Remember where a quest was being worked, keyed by its name.

        The operator's second question -- "and the saved tp location /
        zone" -- and the cheapest possible answer: the goal poll
        already reads the tracked quest name and the zone on the same
        tick, and nothing was keeping the pair.

        It matters because naming the lost quest is only half a cure.
        `questlist` carries each quest's AREA, which is the wiki's name
        for it and not a zone id the game will accept, so on its own it
        tells the operator something and the bot nothing. A zone this
        party actually stood in while tracking that quest is the other
        half, and it comes free.

        Written by every seat into one worker-level map on purpose: the
        wizard that LOST the quest is the one that cannot answer where
        it was, and its party-mates walking the same line can.
        """
        from .. import questlist

        if not zone or not seat.quest_name:
            return
        key = questlist.key_for(seat.quest_name)
        if not key:
            return
        self._quest_zone[key] = (zone, now)
        if len(self._quest_zone) > self.QUEST_ZONE_MEMORY:
            oldest = min(self._quest_zone, key=lambda k: self._quest_zone[k][1])
            del self._quest_zone[oldest]

    #: how long a second opinion on a changed quest read is worth
    #: waiting for. Long enough that the two reads are not the same
    #: memory snapshot, short enough to disappear inside a poll that
    #: already runs every `GOAL_POLL`. Deimos waits three seconds for
    #: the same purpose; a tick has other seats to serve.
    GOAL_CONFIRM = 0.35
    #: how long a goal that will not read is kept before it counts as
    #: gone. The quest NAME has had this rule since rev 30e83468 and
    #: the goal did not: a blank read CLEARED it, unconditionally, and
    #: `maybe_text` returns blank for a zero length, a mid-write
    #: pointer or a decode failure. Shorter than `NAME_MAX_AGE`
    #: because the goal is the read every rung consults and a stale one
    #: is more expensive than an unknown one.
    GOAL_MAX_AGE = 120.0
    #: units two reads of the same marker may differ by and still be
    #: called the same marker. The idle animation and the body's own
    #: drift account for tens; a different quest's marker is hundreds
    #: to six figures away.
    MARKER_AGREE = 150.0

    async def _confirmed_goal(self, seat, first):
        """A changed goal, read twice, or the previous one.

        Returns `first` only when a second read agrees with it.
        Otherwise the old goal stands and the disagreement is recorded
        -- the next poll will read whatever is current and confirm it
        then, so a real change costs at most one `GOAL_POLL`, and a
        flap costs nothing at all.
        """
        from .. import questing

        await asyncio.sleep(self.GOAL_CONFIRM)
        try:
            again = await questing.read_quest_goal(seat.client)
        except Exception:
            again = ""
        if again == first:
            return first
        try:
            seat.tel.note_questing(
                "goal-flapped",
                f"the quest tracker read {first!r} and then {again!r} "
                f"{self.GOAL_CONFIRM:.2f}s later — keeping "
                f"{seat.goal!r} rather than letting one read of an "
                f"unnamed window slot move the party")
        except Exception:
            pass
        return seat.goal

    def _note_goal(self, seat, goal, now):
        """Keep the freshest goal, and stop keeping a dead one.

        The same rule `_note_name` has always had, for the same reason
        and with the same expiry: a blank read is not evidence the
        wizard changed quest, but a kept value becomes invention if it
        is kept forever.
        """
        from .. import questing

        if not goal:
            if (seat.goal and seat.goal_ok_at
                    and now - seat.goal_ok_at > self.GOAL_MAX_AGE):
                seat.goal = ""
            return
        seat.goal_ok_at = now
        if goal != seat.goal:
            seat.goal_at = now
            # Did a Collect COUNT go up? That is the only evidence a
            # wizard ever found the collectibles, and without it a
            # stalled count means "never got there" rather than "the
            # realm is picked clean". See `_maybe_realm_hop`.
            was = questing.collect_count(seat.goal)
            got = questing.collect_count(goal)
            if got and was and got[0] == was[0] and got[1] > was[1]:
                seat.collect_moved_at = now
                seat.collect_moved_for = got[0]
            # In order, and bounded. `_who_is_behind` needs to know that
            # a step was HELD and left, which a single current value
            # cannot say.
            seat.goals_seen.append(goal)
            del seat.goals_seen[:-12]
        seat.goal = goal

    async def _confirmed_marker(self, seat, at, first):
        """A marker that has just come near, read twice, or nothing.

        None when the two reads disagree: a marker nobody can measure
        twice is not a marker to teleport a party onto, and `None`
        already means "no usable marker" to every rung that reads it.
        """
        from .. import questing

        await asyncio.sleep(self.GOAL_CONFIRM)
        try:
            again, _why = await questing.read_quest_position(seat.client)
        except Exception:
            again = None
        if again is None or at is None:
            try:
                seat.tel.note_questing(
                    "marker-unconfirmed",
                    f"the quest marker read {first:.0f} units away and "
                    f"then would not read at all — not treating the "
                    f"wizard as standing on its objective")
            except Exception:
                pass
            return None
        dx, dy = at.x - again.x, at.y - again.y
        second = (dx * dx + dy * dy) ** 0.5
        if abs(second - first) <= self.MARKER_AGREE:
            return second
        try:
            seat.tel.note_questing(
                "marker-flapped",
                f"the quest marker read {first:.0f} units away and then "
                f"{second:.0f} — the arrow hook is a last-writer-wins "
                f"pointer with no quest attached to it, so neither read "
                f"is evidence of where this wizard's objective is")
        except Exception:
            pass
        return None

    def _note_name(self, seat, name, now):
        """Keep the freshest quest name, and stop keeping a dead one.

        A blank read keeps the previous value -- a blank is not
        evidence the wizard changed quest, and reads DO blank
        transiently (`maybe_text` returns "" for a zero length, a
        mid-write pointer or a decode failure; the audit's F3). But
        kept forever, the previous value becomes invention. Past
        `NAME_MAX_AGE` without one successful read, the honest answer
        is "unknown": placement falls to the goal line, which the same
        poll reads fresh, and `_places` treats an unplaced wizard by
        refusing to move anyone -- the safe direction.
        """
        if name:
            seat.quest_name = name
            seat.name_read_at = now
            return
        if (seat.quest_name and seat.name_read_at
                and now - seat.name_read_at > self.NAME_MAX_AGE):
            seat.quest_name = ""

    def _quest_row(self, seat, zone, now):
        """One wizard's questing state, as the Questing tab shows it.

        Pure assembly over fields the tick just read — no client I/O,
        because this runs on every service tick for every seat. The tab
        is the answer to a question the logs kept having to answer
        after the fact: what is each wizard doing RIGHT NOW, and is
        anybody stuck.
        """
        place = self._place_by_name(seat)
        if seat.in_duel:
            doing = "fighting"
        elif self._script_drives(seat):
            doing = "pilot (script)" if self._solo_pilot() else "script"
        elif seat in self._catching_up():
            doing = "catching up"
        elif self._is_booster(seat):
            doing = "boosting"
        elif self._follows(seat):
            doing = ("following + own steps" if self._solo_pilot()
                     else "following")
        elif self.auto_quest:
            doing = ("questing (boosted)" if self.booster_party
                     else "questing")
        else:
            doing = "idle"
        return {
            "name": seat.name,
            "zone": zone or "",
            "quest": seat.quest_name or "",
            "goal": seat.goal or "",
            "line": (f"#{place.order} ({place.world})" if place.comparable
                     else ("side quest" if place.known
                           and not place.on_main else "")),
            "doing": doing,
            "marker": seat.marker_away,
            "idle": (now - seat.progress_at) if seat.progress_at else 0.0,
        }

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
        # Once a minute is often enough to catch a mode the operator
        # switched mid-run, and cheap: `_party_shape` reads fields off
        # `self` and touches no client.
        self._restamp_party()

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
            away = getattr(seat, "marker_away", None)
            if away is not None:
                # How far from the objective, not merely which one. The
                # rev dbced750 stall read "Defeat Jacques the Scratcher
                # · moving" for four minutes and gave no way to tell a
                # wizard working the quest from one parked a zone away
                # -- this is the number that tells them apart.
                bits.append(f"marker {away:,.0f} away")
            if seat.progress_at:
                idle = now - seat.progress_at
                bits.append(f"unchanged for {idle / 60:.0f} min"
                            if idle >= 60 else "moving")
            steps = getattr(self.seats[0].runner, "steps", None)
            if driven and isinstance(steps, int):
                # ...and for how long it has been at that number. A
                # count on its own reads identically whether the script
                # is racing or dead, and rev f32be436 printed "script at
                # 23,324 instructions" once a minute for 110 minutes
                # while nothing executed. The heartbeat is the thing an
                # operator reads; it should not need arithmetic across
                # two hours of log to notice a stopped script.
                still = now - self._steps_at if self._steps_at > NEVER else 0.0
                bits.append(f"script at {steps:,} instructions"
                            + (f" — UNCHANGED for {still / 60:.0f} min"
                               if still >= 120 else ""))
            elif driven:
                bits.append("script driving")
            seat.tel.note_questing("heartbeat", " · ".join(bits))
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    def _whose_client(self, client):
        """The name of the wizard a hook is talking about, or ""."""
        if client is None:
            return ""
        for other in self.seats:
            if other.client is client:
                return other.name
        return ""

    def _teleport_outcome(self, landed, how, zone, client=None):
        """One scripted teleport's result, into the run's log.

        Failures verbatim; successes counted and THINNED per outcome
        kind. "Only the failures" left the rev dbced750 stall
        undiagnosable: the script teleported to the quest twenty times,
        every landing "succeeded", and the export could not say which
        path landed them (collision solve? retreat? navmap fallback?)
        or that it kept happening -- the wrong place that a teleport
        reports as success is exactly the case the counts are for.
        """
        name = self._whose_client(client)
        if landed:
            key = how.split(" (", 1)[0] if how else "unknown"
            counts = self._tp_landed
            n = counts[key] = counts.get(key, 0) + 1
            if self._thinning(n, 20):
                said = (f"{name}: " if name else "") + \
                    f"scripted teleports landing by {how!r}" + \
                    (f" in {zone}" if zone else "") + f" — {n} so far"
                for other in self.seats:
                    try:
                        other.tel.note_questing("teleport-landed", said)
                    except Exception:
                        pass
            return
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
            self._teleport_outcome(landed, how, zone, client)

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
                # Thinned, and every seat's copy with it. Written flat,
                # this is one entry per failure per wizard: rev 35f0fc6e
                # has 1,336 identical `ExceptionalTimeout: Timed out
                # waiting for coro should_update` in EACH of the three
                # exports -- 4,008 lines saying one thing, which is a
                # third of the questing log's whole budget spent burying
                # everything around the failure it was reporting.
                #
                # `_broadcast_once` keys on the message rather than the
                # instruction, so a teleport that starts failing a NEW
                # way is still said at once.
                self._broadcast_once(f"party-task-{what}-{said}",
                                     said, "party-task-failed")

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

    #: how long a wizard may stand still before a WEDGED DIALOGUE BOX
    #: comes under observation -- the fast lane, next to `STUCK_AFTER`'s
    #: five minutes for everything else. Action needs more than time: the
    #: box must show the SAME TEXT on two looks `UNSTICK_EVERY` apart.
    #: That is the receipt the stack audit found missing everywhere --
    #: rev ed709013's script "cleared" a box fifteen times in eight
    #: milliseconds and never moved it, and text-unchanged-for-30s is
    #: the proof clicks are dying that no click reports on its own. A
    #: conversation being actively advanced changes text every page and
    #: is never touched, however fast this threshold.
    DIALOG_WEDGE = 25.0

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
          * No box, script looping -- or parked with nothing in range,
            or X already tried here for nothing: not a dialogue
            problem. The wizard needs moving, and after a second look
            `_desperate_hop` moves it.

        The box arm runs from `DIALOG_WEDGE`, and acts only on a box
        showing the SAME TEXT on two looks -- the receipt that proves
        the script's clicks are dying rather than merely slow.
        Everything else waits for `STUCK_AFTER`, so an ordinary
        conversation is never raced, and every read is written down
        either way. The fast lane is rev ed709013: the script's own
        dialogue handling logged `Dialogue detected. Clearing...`
        fifteen times in eight milliseconds while General Khaba's MORE
        button sat unclicked on all three clients, and wizAi -- whose
        own clicker works -- stood by for the full five minutes it
        granted the script first.
        """
        import time

        from .. import questing

        if seat.progress is None:
            return
        now = time.monotonic()
        idle = now - seat.progress_at
        if idle < self.DIALOG_WEDGE:
            seat.unstuck_at = NEVER
            seat.steps_seen = None
            seat.box_text = None
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
        # Before STUCK_AFTER, the only actionable finding is a dialogue
        # box showing the SAME TEXT as the previous look, thirty seconds
        # apart. The text is the receipt: the script's own handling had
        # over a hundred polls at the box in that window, and a page it
        # was actually advancing would read differently by now. A page
        # that reads the same is a click that is dying in flight -- the
        # failure `click_window` itself never reports. Unreadable text
        # ("" both times) still counts: two looks at a box nothing
        # could read or move is the same wedge with worse lighting.
        text = None
        if open_box:
            try:
                text = (await questing.dialogue_text(seat.client)) or ""
            except Exception:
                text = ""
        stalled = open_box and seat.box_text is not None \
            and text == seat.box_text
        seat.box_text = text
        if idle < self.STUCK_AFTER and not stalled:
            return
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
            # ...so move it. This used to end here, with the diagnosis
            # as the deliverable.
            await self._desperate_hop(seat, mins, at_marker)
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
            return

        # No box, nothing X can reach -- or a retry loop that is not
        # working. What is left is moving the wizard -- once parked or
        # looping is actually KNOWN, so the first look stays a look and
        # the gentler fixes above are not skipped over on a guess.
        if was is not None and steps is not None:
            await self._desperate_hop(seat, mins, at_marker)
            self._maybe_restart_script(seat, mins, parked)

    #: how long between forced restarts of a looping script. A restart
    #: replays the program's opening dispatch, which takes a minute or
    #: two of walking before its effect is judgeable -- restarting
    #: faster than this is thrash, not persistence.
    SCRIPT_RESTART_EVERY = 420.0
    #: how close counts as standing ON the quest marker. The same 750
    #: the quest-marker check itself uses (`questing.at_quest_marker`),
    #: so "at the marker" means one thing everywhere.
    AT_THE_MARKER = 750.0
    #: how long a wizard may stand ON its marker, script looping and
    #: nothing changing, before the script is restarted.
    #:
    #: Generously long, because arriving at your objective and then
    #: working is the normal shape of questing -- a dungeon sigil, a
    #: dialogue chain, a fight all happen standing still on the marker.
    #: Fifteen minutes of it with the goal, the zone and the position
    #: all unchanged is not work. Rev 35f0fc6e stood there for 103.
    ON_MARKER_RESTART = 900.0

    def _maybe_restart_script(self, seat, mins, parked):
        """The operator's other manual fix, automated: restart the script.

        The run at rev 817b9f20 is the case in full. All three wizards
        stood in KT_Pyramid/KT_Hall while every quest marker read
        98,000+ away in the Krokosphinx, and the script looped ~3,100
        instructions a minute for seven straight minutes. Every rung of
        the ladder answered correctly -- the desperate hop refused
        (right: no teleport crosses a zone), nothing was in range for X
        (right), the realm change stayed quiet (right: wrong zone, not
        crowded) -- and none of them could help, because the only actor
        that can cross a zone is the script, and the script was
        spinning a retry loop it entered from a quest state that no
        longer exists.

        The operator has already named the fix, from the run where they
        applied it by hand: "when I reset it they progressed". A
        restarted deimoslang program re-runs its route dispatch from
        the top against the CURRENT quest state, and lands on the right
        leg of its route instead of the one it wandered into. Restarts
        are a move the script's design tolerates by construction -- a
        program that runs off its end restarts itself.

        Only for a LOOPING script. A parked script is waiting on
        something real (the stuck-instruction reload owns that case),
        and an in-zone marker is the desperate hop's case -- this rung
        exists precisely for the wedge the hop must refuse.

        It fires for a marker that reads ANOTHER zone, and equally for
        a step that has no marker to read at all -- which is the state
        rev 116b5866 spent its last five minutes in, and the second
        time in three rounds that a markerless Collect made the only
        rung that could help unreachable. All three wizards stood in
        `KT_WorldTeleporter`, the world portal, while the script ran
        15,785 instructions without moving anybody. p1 was on `Get Some
        Bling`, and the quester's handling of it is::

            until NOT p1 tracking_goal "collect gemstones in hall of champions" {
                if p1 inzone Krokotopia/KT_Krokosphinx/KT_ChampHall {

        an until-loop whose only body is zone-gated, containing nothing
        that would TRAVEL to that zone. Standing anywhere else, it spins
        until the goal changes, and the goal cannot change because the
        body never runs. No teleport rung can leave a loop -- only the
        instruction pointer can, and only a restart moves it.
        """
        import time

        runner = self.seats[0].runner
        if runner is None:
            return
        if parked and getattr(runner, "running", False):
            # A parked script is waiting on something real and the
            # stuck-instruction reload owns that case -- but only while
            # the VM is actually being stepped, which is the assumption
            # this used to make silently. At rev f32be436 it was false
            # for 110 minutes: the step count was frozen because nobody
            # was calling `step()`, `parked` was therefore true on every
            # look, and this returned here every time while the one
            # thing that could have crossed a zone sat still.
            #
            # `_check_script_alive` is the detector for that now, and it
            # reloads rather than restarts. This keeps its own refusal
            # for the case it was written for: a VM that is alive and
            # inside a long instruction.
            return
        away = seat.marker_away
        blind = self._marker_unusable(seat) if away is None else ""
        on_it = away is not None and away <= self.AT_THE_MARKER
        if away is not None and away <= self.MARKER_IN_ZONE and not on_it:
            # In this zone but not on top of it: the desperate hop's
            # case, not this one. It can teleport the last few thousand.
            return
        if on_it and mins < self.ON_MARKER_RESTART / 60.0:
            # ON the marker is the desperate hop's case for a while --
            # arriving and then working is what a wizard at its
            # objective is supposed to look like. Past this it is not:
            # rev 35f0fc6e spent 103 minutes with all three wizards
            # standing on the Khai Amahte marker in KT_Arena while the
            # script ran a quarter of a million instructions, and every
            # rung answered correctly and did nothing. The hop refused
            # ("teleporting to the spot this wizard is standing on"),
            # X had nothing in range, the realm change is for Collect
            # steps, and this returned above because the marker was in
            # zone. The only actor that can leave a spot the script
            # keeps walking back to is the script's own route.
            return
        if away is None and not blind:
            # One failed read. It may read on the next poll, and
            # restarting a whole program over a blink is thrash.
            return
        now = time.monotonic()
        if now - seat.progress_at < self.STUCK_AFTER + self.UNSTICK_EVERY:
            return
        if now - self._script_restarted_at < self.SCRIPT_RESTART_EVERY:
            return
        self._script_restarted_at = now
        if on_it:
            stuck_at = (f"this wizard standing ON its quest marker "
                        f"({away:,.0f} away) and nothing changing")
        elif away is not None:
            stuck_at = f"the quest marker {away:,.0f} away — another zone"
        else:
            stuck_at = f"nothing to aim at — {blind}"
        said = (f"stuck {mins:.0f} min with {stuck_at} — while the script "
                f"loops. No teleport can help from here; the script's own "
                f"route can, and it is looping on a state it cannot leave. "
                f"Restarting it so its route dispatch runs fresh against "
                f"the party's current quest — the operator's manual reset, "
                f"automated")
        ok = runner.restart()
        skipped = list(getattr(runner, "skipped_setup", None) or ())
        if not ok:
            said = ("tried to restart the looping script and the restart "
                    "failed — it will be retried in "
                    f"{self.SCRIPT_RESTART_EVERY / 60:.0f} min")
        elif skipped:
            said += (f". Its one-time setup ({', '.join(skipped)}) is "
                     f"skipped — it has already run")
        for other in self.seats:
            try:
                other.tel.note_questing("script-restarted", said)
            except Exception:
                pass
        self._say(seat, said)

    async def _desperate_hop(self, seat, mins, at_marker):
        """The operator's own last resort, automated: a quest teleport.

        "Sometimes when really stuck a simple fix is turning off the
        script for a moment and pressing the teleport to quest button,
        maybe that will help as a desperate fix if they get stuck too
        long." That is what a person does at this point, and this is
        that, with the judgement written in as guards. By the time it
        runs the wizard is past `STUCK_AFTER`, out of combat, any open
        dialogue has been handled and X has been offered everything it
        can reach -- what is left is moving the wizard.

        Each guard is a failure a live run has already had:

        -- a second look first. The gentler fixes above deserve one
           full pass (parked is unknowable on the first sample), so the
           hop waits one `UNSTICK_EVERY` beyond `STUCK_AFTER`.
        -- the marker must read, within this zone. Across a boundary
           the coordinates are another zone's space and the teleport
           lands underground -- rev 1843e387, four times over.
        -- not from on top of the marker. Rev 3822cc6c teleported
           Phönix to the spot he was already standing on, forever.
        -- not under the wizard's own heal floor. A quest teleport
           lands on sigils and mobs, and rev 85a68184's wizards died of
           entering fights hurt; one resting at a wisp is standing
           still on purpose.
        -- once per spot. A hop that helps moves the wizard, which
           resets the spot on its own; one that does not help will not
           help twice.

        And the script IS held for it -- the hop, plus a settle window
        after it lands. The first version reasoned that a parked or
        looping script could only be unblocked by the wizard arriving
        at its objective, and the operator corrected it: "it's possible
        and even likely for the bot to teleport away from the quest
        after a desperate tp back to the place it was stuck, which is
        usually 1000ish units away". A retry loop re-teleports within a
        second, aimed a thousand units off the marker -- close enough
        to look right, far enough to undo the fix before a dialogue
        opens or a sigil finishes counting down. The hold is a
        deadline, not a flag, so a hop cut off mid-teleport releases
        the script by itself when `HOP_PAUSE_CEILING` runs out.
        """
        import time

        from .. import questing

        now = time.monotonic()
        if now - seat.progress_at < self.STUCK_AFTER + self.UNSTICK_EVERY:
            return
        here = seat.progress or (None, None, None)
        if seat.hop_tried_at == here:
            return
        away = seat.marker_away
        if away is None:
            # A marker that would not read may read on the next poll,
            # and refusing permanently over one bad read would be
            # worse. A marker that will not read for a REASON is a
            # different matter: there is nothing for a quest teleport
            # to aim at, and the wizard being visibly stuck at it is
            # worth one line rather than silence -- rev 98b4c50c spent
            # 44 minutes here saying nothing at all.
            blind = self._marker_unusable(seat)
            if blind:
                seat.hop_tried_at = here
                seat.tel.note_questing(
                    "desperate-hop-refused",
                    f"stuck {mins:.0f} min and a quest teleport has "
                    f"nowhere to aim — {blind}")
            return
        if away > self.MARKER_IN_ZONE:
            seat.hop_tried_at = here
            seat.tel.note_questing(
                "desperate-hop-refused",
                f"stuck {mins:.0f} min, and the quest marker is "
                f"{away:,.0f} away — another zone. A quest teleport "
                f"cannot cross one, so the hop the operator would try by "
                f"hand has nowhere to go from here")
            return
        if at_marker:
            seat.hop_tried_at = here
            seat.tel.note_questing(
                "desperate-hop-refused",
                f"stuck {mins:.0f} min on top of the quest marker — "
                f"teleporting to the spot this wizard is standing on is "
                f"the loop rev 3822cc6c spent 24 minutes in")
            return
        left = await self._health_left(seat)
        if left is not None and left < self._health_needed(seat):
            return
        # Before the await, not after: a hop cut off mid-teleport by the
        # stage deadline must not retry from this spot forever.
        seat.hop_tried_at = here
        # ...and the script stops taking instructions NOW, before the
        # teleport, so its own tp loop cannot land one between ours and
        # the dialogue it is meant to open.
        self._hop_pause_until = now + self.HOP_PAUSE_CEILING
        self._say(seat,
                  f"stuck {mins:.0f} min and out of gentler fixes — doing "
                  f"what the operator would do by hand: script held for a "
                  f"moment, quest teleport, {away:,.0f} to the marker")
        try:
            fight = await questing.hop_once(
                seat.client, on_status=lambda m: self._say(seat, m))
        finally:
            # The settle window: a sigil the hop lands on counts down
            # for ~10s, and a turn-in needs the server round-trip. On
            # ANY exit, so a hop that raised does not hold the script
            # for the whole ceiling.
            self._hop_pause_until = time.monotonic() + self.HOP_SETTLE
        seat.tel.note_questing(
            "desperate-hop",
            f"stuck {mins:.0f} min — script held, quest-teleported "
            f"{away:,.0f} to the marker"
            + (", and a fight started" if fight else "")
            + f"; the script gets the wheel back in {self.HOP_SETTLE:.0f}s")

    #: the longest the script can be held by a hop that never returns.
    #: A deadline rather than a flag: if `hop_once` is cut off or dies,
    #: the script frees itself when this runs out.
    HOP_PAUSE_CEILING = 90.0
    #: how long the script stays held AFTER the hop lands. The operator:
    #: "it's possible and even likely for the bot to teleport away from
    #: the quest after a desperate tp back to the place it was stuck,
    #: which is usually 1000ish units away" -- and a sigil the hop lands
    #: on counts down ~10s before it admits anybody.
    HOP_SETTLE = 15.0

    def _hop_held(self):
        """Is the script waiting out a desperate quest-teleport?"""
        import time

        return time.monotonic() < self._hop_pause_until

    #: how close to the marker "standing at a walk-in door" starts. NOT
    #: `AT_THE_MARKER`'s 750: that is conversation range, and a wizard
    #: 700 units from a door is a wizard the script is still walking.
    #: The stall this rung answers reads single digits -- rev 9e13d385's
    #: Konstantin sat at "marker 2 away", then "marker 8 away" -- with
    #: room for a landing truncated at a collision volume's edge.
    WALK_THROUGH_NEAR = 120.0
    #: how long the wizard must stand there, zone and goal unchanged,
    #: before walking. Long enough for the work that legitimately
    #: happens ON a marker -- a dungeon sigil's countdown (~10s, and it
    #: RESTARTS when a partner steps on), a turn-in's server round-trip
    #: -- and comfortably shorter than the ~2 min the script's own
    #: watchdog waits before "recovering" the wizard to the world hub,
    #: which is the outcome this rung exists to beat.
    WALK_THROUGH_AFTER = 45.0
    #: how long between attempts at one door.
    WALK_THROUGH_EVERY = 120.0
    #: how far past the marker to walk. A transition trigger sits at or
    #: just behind its door; the marker sits ON the door. The script's
    #: own approach walked ~500-unit legs into doors, so 120 is well
    #: inside "what a player's W key does" and short enough not to
    #: matter when the direction is wrong.
    WALK_THROUGH_PAST = 120.0
    #: the ceiling on the script hold around the walk -- same shape as
    #: the desperate hop's, for the same reason: the script's own tp
    #: loop re-teleporting the wizard back onto the marker mid-walk is
    #: precisely the loop being broken.
    WALK_THROUGH_CEILING = 60.0
    #: the settle after a walk that did NOT cross. Short: nothing is in
    #: flight, and the script should get its own machinery back.
    WALK_THROUGH_SETTLE = 3.0

    async def _maybe_walk_through(self, seat):
        """Walk THROUGH a marker the wizard is parked on, like a player.

        The stall the collision teleport left behind, in the operator's
        words: "still standing just outside the door and teleporting
        away, not walking forward to go through like the script used
        to". A walk-in transition -- a house door, a service lift, a
        street gate -- has its quest marker ON the door and its trigger
        volume just past it. The old scripts teleported NEAR the door
        and then walked a hardcoded leg through it; a collision-solved
        teleport lands ON the marker, exactly, and then nothing crosses
        the trigger.

        Rev 9e13d385 is the case in full: Konstantin stood at "Go To
        Amy Brooks' Place in Knight's Court" with "marker 2 away", then
        "marker 8 away", for two minutes -- jittering, so the stuck
        ladder's position clock never fired -- until the script's own
        watchdog "recovered" him to MB_Hub, 10,602 units from the door
        he was touching. The fix a player would apply is the W key.

        So: standing essentially ON a marker, zone and goal unchanged
        for `WALK_THROUGH_AFTER`, no dialogue open and no press-X
        prompt in range (a prompt means a sigil or an NPC owns this
        spot, and the X rungs own those), hold the script the way the
        desperate hop does and walk through the marker -- along the
        approach line first, then, if the zone still has not changed,
        sweep the other three directions. Any leg that crosses shows up
        as a zone change or a loading screen, and the sweep stops.
        """
        import time

        from .. import questing

        away = seat.marker_away
        now = time.monotonic()
        if self._countdown_held():
            # A hold is guarding a sigil RIGHT NOW, and this rung walks
            # wizards off spots. Rev e6201303: the leader's own hold ran
            # 142.8-187.8 and this swept it four legs off the sigil at
            # 179.6, in the middle of it -- the "walking out of the
            # sigil for no reason" class, still alive through the other
            # rung because the hold only ever gated the SCRIPT's
            # teleports and never wizAi's own walk.
            seat.through_since = None
            return
        if away is None or away > self.WALK_THROUGH_NEAR:
            seat.through_since = None
            return
        since = seat.through_since
        if since is None or max(seat.zone_since, seat.goal_at) > since:
            # Just arrived, or the zone/goal moved while standing here:
            # whatever is happening on this marker is WORKING.
            seat.through_since = now
            return
        if now - since < self.WALK_THROUGH_AFTER:
            return
        if now - seat.walked_through_at < self.WALK_THROUGH_EVERY:
            return
        if await questing.in_dialogue(seat.client):
            # Dialogue is work, and walking mid-page abandons it.
            seat.through_since = now
            return
        if await questing.near_interactable(seat.client):
            # The game is offering X. A sigil, a dungeon door with a
            # prompt, the quest NPC -- every one of those is another
            # rung's case, and walking away from a counting sigil
            # un-joins it.
            seat.through_since = now
            return
        client = seat.client
        try:
            if await client.is_loading():
                return
        except Exception:
            pass
        mins = (now - since) / 60.0
        await self._sweep_through(seat, away, f"parked {mins:.0f} min")

    #: how many doors the map remembers. A dungeon is a handful of
    #: rooms and a world is a handful of streets; anything past this is
    #: history, not a route.
    DOORS_REMEMBERED = 64
    #: how long between door walks for one wizard. A walk holds the
    #: script and takes several seconds; retrying it every follow tick
    #: would be the friends-list spam with extra steps.
    DOOR_WALK_EVERY = 20.0
    #: how far past the door position each leg aims. The trigger volume
    #: sits at the door, so stopping ON it is exactly the landing that
    #: does not cross -- the walk-in-door lesson, reused.
    DOOR_WALK_PAST = 250.0
    #: how long one leg of the door walk may run.
    DOOR_WALK_LEG = 12.0
    #: how long the script is held while a door walk is in flight.
    DOOR_WALK_CEILING = 45.0
    #: how many times one (from, to) route may fail before the party is
    #: told, once, that it is impassable. Small, because the information
    #: is worth having early and the line is said once per route: rev
    #: 8ebfcf70 needed it at minute two and got 47 identical lines and
    #: no verdict instead.
    DOOR_WALK_GIVE_UP = 5

    #: how often a seat samples where it is standing, for the door map.
    #: One position read and one zone read on its OWN client, on its own
    #: tick -- not the party-wide sweep `TOGETHER_POLL` paces.
    #:
    #: The number is the whole point. A door is inferred from the last
    #: sample taken before the zone changed, so the sample's age is the
    #: door's position error, and a wizard runs ~580 units a second.
    #: Learned off the six-second party poll the error is up to ~3,500
    #: units and the sweep that has to absorb it is `DOOR_WALK_PAST`,
    #: 250. That is rev 8ebfcf70's booster: 47 door walks, 0 crossings,
    #: every one of them teleporting to a spot most of a room away from
    #: the door it was aiming at.
    SPOT_POLL = 1.0
    #: how soon after a wizard's own X press a zone change still counts
    #: as that sigil firing rather than as a door it walked. Generous:
    #: the countdown is ~10s and RESTARTS on every late join, and the
    #: instance load takes seconds more -- `COUNT_HOLD` budgets 45s for
    #: the same span and for the same reasons.
    SIGIL_CROSS_WINDOW = 60.0
    #: a door learned from a sample older than this is not a door, it is
    #: a place the wizard happened to be. Recorded anyway -- it is still
    #: the best guess available, and one room closer beats standing
    #: still -- but said out loud, so an export can tell a bad aim from
    #: a door that will not open.
    DOOR_SAMPLE_FRESH = 2.0

    async def _note_spot(self, seat):
        """Sample where this wizard is, and learn the door if it moved.

        The zone read is what makes it worth doing: a position on its
        own cannot tell which side of a door it was taken on, and a
        sample taken just AFTER a crossing is in the new room's
        coordinate space -- the one place a follower must not be sent.
        Reading both together means every sample is stamped with the
        room it belongs to, and the door is the newest sample still
        stamped with the old one.
        """
        import time

        if len(self.seats) < 2:
            # The door map exists so one wizard can walk in behind
            # another. A solo run has nobody to teach and nobody to
            # follow, so it should not pay two memory reads a second
            # for it -- `TOGETHER_POLL`'s own docstring is about
            # exactly this traffic.
            return
        now = time.monotonic()
        if now - seat.spot_at < self.SPOT_POLL:
            return
        seat.spot_at = now
        try:
            if await seat.client.is_loading():
                # Mid-load a position reads in the OLD zone's space
                # while `zone_name` may already answer the new one.
                # Pairing those two is how a door gets recorded on the
                # wrong side of itself.
                return
        except Exception:
            pass
        try:
            zone = await seat.client.zone_name()
        except Exception:
            return
        if not zone:
            return
        try:
            spot = await seat.client.body.position()
        except Exception:
            return
        was, before = seat.spot_zone, seat.spot_seen
        if was and was != zone and before is not None:
            self._learn_door(seat, zone, now, spot=before,
                             sampled=seat.spot_at_prev)
        seat.spot_at_prev = now
        seat.spot_seen, seat.spot_zone = spot, zone
        # Keep the party poll's own view fed too: it is what
        # `_walk_the_leaders_door`'s last-resort branch reads, and a
        # fresher value there is strictly better than a staler one.
        seat.last_spot, seat.last_spot_zone = spot, zone

    def _learn_door(self, seat, into, now, spot=None, sampled=None):
        """Record where this wizard was standing when it left a zone.

        The position sampled on the poll BEFORE the change is, by
        construction, the last place the wizard stood in the old zone --
        which is the door, give or take one poll of walking. Every
        wizard teaches the whole party: the leader crossing first is
        precisely the case a stranded follower needs, and it is the
        common one.

        "Give or take one poll of walking" is the load-bearing phrase,
        and it used to be one SIX-SECOND poll. The door is recorded with
        the age of the sample it came from now, and a fresher sighting
        never loses to a staler one -- see `SPOT_POLL`.
        """
        out = seat.last_spot_zone if spot is None else seat.spot_zone
        if spot is None:
            spot = seat.last_spot
        if not out or not into or out == into or spot is None:
            return
        age = None if sampled is None else max(0.0, now - sampled)
        # HOW it crossed, not just where from. A zone change that
        # follows this wizard's own X press is a sigil firing, and a
        # sigil is not a door: it admits the wizards standing on it when
        # its counter runs out, and no amount of walking through the pad
        # crosses it. The run proves that on its own, without any of the
        # door map involved -- rev 8ebfcf70 t=1922.8, Konstantin 127
        # units from the marker, four legs swept "through and past it"
        # and the zone did not change; t=1935.3 it pressed X, and at
        # t=1950.3 it was inside. Same wizard, same spot, twelve seconds
        # apart. Walking is not the way through one of these, so a
        # follower must not be sent to sweep at it.
        how = ("sigil" if now - seat.pressed_x_at <= self.SIGIL_CROSS_WINDOW
               else "walked")
        # Never trade a better-aimed door for a worse-aimed one. Both
        # `_note_spot` (fresh, on the seat's own second) and the party
        # poll (up to six seconds stale, and no age at all to declare)
        # learn doors, and the poll runs last -- so without this the
        # good sample is overwritten by the bad one within the tick.
        # An unknown age loses to a known one for the same reason: the
        # only thing that can be said about it is that it is at least
        # as stale as the poll that produced it.
        old = self._doors.get((out, into))
        if old is not None and _door_how(old) == how:
            was = old[3] if len(old) > 3 else None
            if was is not None and (age is None or was < age):
                return               # we already know this door better
        self._doors[(out, into)] = (spot, now, seat.name, age, how)
        if len(self._doors) > self.DOORS_REMEMBERED:
            for key in sorted(self._doors,
                              key=lambda k: self._doors[k][1])[:8]:
                self._doors.pop(key, None)

    async def _walk_the_leaders_door(self, seat, boss, why):
        """Walk into the leader's zone, when no teleport can get there.

        The operator's report, and the fix they asked for: "the booster
        and boostee got into a dungeon but the booster is trying to
        teleport via friends to the boostee, but some instances (most
        allow tp's) don't allow teleports, so it says your friend is
        busy on repeat. instead we can follow the tp steps of the
        leader to enter combats in this case or follow to different
        locations".

        Rev e6201303 is what that costs. Oz reached
        `DS_Arena_Gauntlet_6Room` with Konstantin and then stood in it
        for ten minutes while Konstantin walked on through
        `6Room_Sub/6Room_2` ... `6Room_6` -- the gauntlet's rooms are
        separate ZONES, an XYZ teleport cannot cross one, and the
        friends-list teleport is refused inside the instance. So the
        booster contributed two combat rounds to a 27-minute run while
        the wizard it exists to protect fought Orin Grimcaster alone,
        twice, and lost both.

        A wizard cannot port there. It can WALK there, and it does not
        need to know the way: the leader just walked it, and
        `_learn_door` wrote down where it was standing when it went.
        Teleporting to that spot is legal (same zone), and walking
        through it is the same four-leg sweep that answers a walk-in
        door -- the trigger volume sits at the door, so a landing ON it
        crosses nothing and the legs aim PAST it.

        Returns (walked, reason).
        """
        import math
        import time

        from .. import party

        here, there = seat.zone_seen, boss.zone_seen
        if not here or not there or here == there:
            return False, ""
        now = time.monotonic()
        if now - seat.door_walked_at < self.DOOR_WALK_EVERY:
            return False, ""
        door = self._doors.get((here, there))
        into = there
        if door is None:
            # The leader is several rooms ahead, which is the ordinary
            # case rather than the exception: rev e6201303's Konstantin
            # crossed FIVE doors while Oz stood in the first room, so
            # the map holds (6Room -> 6Room_2) and nothing at all keyed
            # on the room he ended up in. The way forward is still the
            # door he took out of THIS room. Walking it lands the
            # follower one room closer and the next tick answers the
            # next room, which is what following someone's route means.
            #
            # Prefer a door whose far side the leader has actually been
            # through -- a room has more than one exit, and the one
            # that matters is the one on the leader's own path.
            route = {z for z, _t in (boss.zone_left or ())} | {there}
            outs = [(to, v) for (frm, to), v in self._doors.items()
                    if frm == here]
            walked = [kv for kv in outs if kv[0] in route]
            best = max(walked or outs, key=lambda kv: kv[1][1], default=None)
            if best is not None:
                into, door = best
        if door is None:
            # Never seen anyone leave this zone at all. The leader's own
            # last spot in it is the same fact one poll earlier, and it
            # is what a leader that crossed before the map existed
            # leaves behind.
            if boss.last_spot is not None and boss.last_spot_zone == here:
                door = (boss.last_spot, now, boss.name,
                        max(0.0, now - boss.spot_at))
        if door is None:
            return False, (f"{seat.name} cannot teleport to {boss.name} in "
                           f"{there} and has no record of the way in — "
                           f"nobody has been watched crossing from {here} "
                           f"to {there} yet")
        # Four fields now; three is the shape a door learned before the
        # sample age existed still has, and it reads as "age unknown"
        # rather than as an error.
        spot, _at, taught_by = door[0], door[1], door[2]
        aim = door[3] if len(door) > 3 else None
        if _door_how(door) == "sigil":
            # Not a door. `taught_by` did not walk out of this room, it
            # stood on a sigil and pressed X, and a sigil admits the
            # wizards standing on it when its counter fires -- so the
            # four-leg sweep below cannot cross it however good the aim
            # is. Rev 8ebfcf70 swept at one for 47 attempts, holding
            # the whole party's script up to 45s each time, and the
            # export called every one of them a door.
            #
            # Nor is the answer to press X here: the counter this
            # wizard would start is its own, and a follower that enters
            # a fresh copy of the dungeon alone is the solo boss again
            # with a loading screen in front of it. What is needed is
            # for the party to be on the pad together, which is
            # `_maybe_count_hold`'s job at the moment of entry and
            # nothing this rung can retrofit once the leader is
            # already through. So: say so, exactly, and do not sweep.
            self._say_once(
                seat, f"sigil-route:{here}->{into}",
                f"{seat.name} cannot follow {boss.name} into {into}: the "
                f"way in is a sigil, not a door, and {taught_by} rode it "
                f"in on a press-X countdown",
                kind="route-is-a-sigil",
                detail=(f"{here} -> {into} was crossed by {taught_by} "
                        f"pressing X at a sigil, not by walking a door. A "
                        f"sigil admits the wizards standing on it when "
                        f"its counter fires, so no sweep crosses one and "
                        f"{seat.name} cannot enter behind {boss.name} — "
                        f"the party has to board it together. Latest "
                        f"teleport reason: {why}"))
            return False, (f"{here} -> {into} is a sigil entry, not a "
                           f"door — {seat.name} cannot walk in behind "
                           f"{boss.name}, the party has to board it "
                           f"together")
        try:
            if await party.in_battle(seat.client):
                return False, ""
        except Exception:
            pass
        seat.door_walked_at = now
        # ...and say WHY no teleport reached, in the terms the operator
        # can act on. This line used to assert the dungeon had refused,
        # every time, and rev 8ebfcf70 printed that 47 times for a
        # teleport the game was never asked for -- the friends list
        # would not open, which is wizAi's bug and not a game rule.
        # See `party.never_asked`.
        blamed = ("The teleport was never attempted, so this is not the "
                  "dungeon refusing one — but the door is still the way "
                  "in, and the party watched this one being used"
                  if party.never_asked(why) else
                  "A dungeon that refuses friends-list teleports still "
                  "has doors, and the party watched this one being used")
        # How good the aim is, said in the same line as the attempt. A
        # door sampled a second before the crossing is a door; one
        # sampled six seconds before it is wherever the wizard happened
        # to be, and the difference is the difference between this
        # working and rev 8ebfcf70's 0-for-47. Without it an export
        # cannot tell "we walked to the wrong place" from "this door
        # cannot be walked".
        aimed = ("" if aim is None else
                 f" The door was sampled {aim:.1f}s before the crossing"
                 + ("" if aim <= self.DOOR_SAMPLE_FRESH else
                    f", which is stale — at a walking pace that is "
                    f"~{aim * 580:.0f} units of error against a "
                    f"{self.DOOR_WALK_PAST:.0f}-unit sweep") + ".")
        tried = seat.door_fails.get((here, into), 0)
        seat.tel.note_questing(
            "door-walk",
            f"no teleport reaches {boss.name} in {there} ({why}) — walking "
            f"the door {taught_by} was standing on when it crossed out of "
            f"{here}" + ("" if into == there else f" into {into}, one room "
            f"closer") + f". {blamed}."
            + (f" This route has failed {tried} time(s) already." if tried
               else "") + aimed)
        self._say(seat,
                  f"{seat.name} cannot port to {boss.name} — walking the "
                  f"door into {into.rsplit('/', 1)[-1]}")

        client = seat.client
        self._hop_pause_until = now + self.DOOR_WALK_CEILING
        crossed = False
        legs = 0
        try:
            try:
                await client.teleport(spot)
            except Exception as exc:
                return False, (f"could not step to the door into {there} "
                               f"({type(exc).__name__}: {exc})")
            if await self._zone_crossed(client, here):
                crossed = True
            else:
                # Which way is through? The wizard has just landed ON
                # the door, so the leader's heading is unknown and the
                # approach line is gone. Sweep: forward along the yaw
                # it landed with, then the three other quarters. One of
                # them is into the room.
                try:
                    yaw = await client.body.yaw()
                    ux, uy = -math.sin(yaw), -math.cos(yaw)
                except Exception:
                    ux, uy = 1.0, 0.0
                for vx, vy in ((ux, uy), (-uy, ux), (uy, -ux), (-ux, -uy)):
                    legs += 1
                    try:
                        await asyncio.wait_for(
                            client.goto(spot.x + vx * self.DOOR_WALK_PAST,
                                        spot.y + vy * self.DOOR_WALK_PAST),
                            timeout=self.DOOR_WALK_LEG)
                    except Exception:
                        pass          # a wall; the next quarter answers
                    if await self._zone_crossed(client, here):
                        crossed = True
                        break
                    if await party.in_battle(client):
                        break
        finally:
            self._hop_pause_until = time.monotonic() + (
                self.HOP_SETTLE if crossed else self.WALK_THROUGH_SETTLE)
        if crossed:
            seat.cross_zone_fails = 0
            seat.door_fails.pop((here, into), None)
            # A crossing that WORKED is the route working, and a
            # follower five rooms behind has four more to walk. The
            # cooldown exists to stop a walk that goes nowhere being
            # retried twice a second, not to pace a chase that is
            # succeeding.
            seat.door_walked_at = NEVER
            seat.tel.note_questing(
                "door-walk",
                f"walked out of {here} on foot ({legs} leg(s)) — the "
                f"teleport the game refused, done the way a player would "
                f"do it")
            return True, f"walked through the door out of {here}"

        # It did not cross, and that is a fact about THIS ROUTE. Counted
        # per (from, to) rather than per seat, because the seat-wide
        # counter next door -- `cross_zone_fails` -- is written in three
        # places and read in none, which is why rev 8ebfcf70 ran the
        # same failing walk 47 times without the run ever escalating.
        fails = seat.door_fails.get((here, into), 0) + 1
        seat.door_fails[(here, into)] = fails
        if fails == self.DOOR_WALK_GIVE_UP:
            # Said once, loudly, at the point where repeating it stops
            # being a retry and becomes the run's headline. The route
            # keeps being attempted afterwards -- there is nothing else
            # to attempt, and a door that opens on the fiftieth try is
            # still a door -- but the export now names the wall instead
            # of burying it under identical lines.
            for other in self.seats:
                try:
                    other.tel.note_questing(
                        "route-impassable",
                        f"{seat.name} has failed to walk {here} -> {into} "
                        f"{fails} times running, and no teleport reaches "
                        f"{boss.name} either. {boss.name} is questing on "
                        f"the far side of a transition this wizard cannot "
                        f"cross, so every fight through it is one wizard "
                        f"short. Latest teleport reason: {why}")
                except Exception:
                    pass
            self._say(seat,
                      f"{seat.name} cannot reach {boss.name}: {here} -> "
                      f"{into} has failed {fails} times and no teleport "
                      f"works either")
        return False, (f"{seat.name} could not walk the door from {here} "
                       f"into {into} either — teleported onto the spot "
                       f"{taught_by} crossed from and swept {legs} leg(s) "
                       f"past it without the zone changing "
                       f"({fails} failure(s) on this route)")

    async def _sweep_through(self, seat, away, label):
        """The walk itself: through the marker, then sweep. Shared by
        the walk-through rung and the countdown hold's expiry -- see
        `_maybe_walk_through` for the door reasoning and rev 9e13d385;
        `label` says which stall this walk is answering ("parked 2
        min", "held 20s with no countdown"). Holds the script for its
        own duration the way the desperate hop does, stamps the
        cooldown, and reports the outcome either way. Returns whether
        a leg crossed a transition.
        """
        import math
        import time

        from .. import questing

        client = seat.client
        now = time.monotonic()
        try:
            here = await client.body.position()
            marker, _why = await questing.read_quest_position(client)
        except Exception:
            return False
        if here is None or marker is None:
            return False
        start_zone = seat.zone_seen
        try:
            start_zone = await client.zone_name() or start_zone
        except Exception:
            pass
        dx, dy = marker.x - here.x, marker.y - here.y
        dist = math.hypot(dx, dy)
        if dist >= 15.0:
            # The approach line: wizard -> marker, extended past it.
            # The wizard came AT the door, so through the marker along
            # this line is through the door.
            ux, uy = dx / dist, dy / dist
        else:
            # Standing ON it -- the line is noise. The wizard is still
            # FACING the way it walked or teleported in, so its yaw is
            # the next best approach vector (the `- sin/cos` pair is
            # wizwalker's forward, see `calc_FrontalVector`).
            try:
                yaw = await client.body.yaw()
                ux, uy = -math.sin(yaw), -math.cos(yaw)
            except Exception:
                ux, uy = 1.0, 0.0
        # Stamp and hold BEFORE the first step, like the hop: a walk
        # cut off mid-leg must not retry every tick, and the script's
        # tp loop must not land a teleport between our legs.
        seat.walked_through_at = now
        self._hop_pause_until = now + self.WALK_THROUGH_CEILING
        self._say(seat,
                  f"{label} basically ON the quest marker "
                  f"({away:,.0f} away) with no transition — holding the "
                  f"script and walking through it, the way the old "
                  f"scripts walked into doors")
        crossed = False
        legs = 0
        try:
            # Forward, then left, right, back -- each leg aims past the
            # MARKER, so every one of them crosses the door area again.
            for vx, vy in ((ux, uy), (-uy, ux), (uy, -ux), (-ux, -uy)):
                legs += 1
                try:
                    await asyncio.wait_for(
                        client.goto(marker.x + vx * self.WALK_THROUGH_PAST,
                                    marker.y + vy * self.WALK_THROUGH_PAST),
                        timeout=12.0)
                except Exception:
                    # A wall. The wrong direction refuting itself is
                    # the sweep working, not a failure to report.
                    pass
                if await self._zone_crossed(client, start_zone):
                    crossed = True
                    break
                if await questing.in_battle(client):
                    # A mob got the wizard mid-sweep. The fight loop
                    # owns it now, and a fight is at least movement.
                    break
        finally:
            self._hop_pause_until = time.monotonic() + (
                self.HOP_SETTLE if crossed else self.WALK_THROUGH_SETTLE)
        if crossed:
            seat.tel.note_questing(
                "walked-through",
                f"{label} on the marker with no transition "
                f"— walked through it ({legs} leg(s)) and the zone "
                f"changed. A walk-in door's trigger sits just past its "
                f"marker, and a collision-solved teleport stops exactly "
                f"ON the marker")
            self._say(seat, "walked through the door — the zone changed")
        else:
            seat.tel.note_questing(
                "walk-through-missed",
                f"{label} on the marker — walked {legs} "
                f"leg(s) through and past it and the zone did not "
                f"change. Whatever this marker wants, it is not a "
                f"walk-in door; the script gets the wheel back in "
                f"{self.WALK_THROUGH_SETTLE:.0f}s")
        return crossed

    async def _zone_crossed(self, client, start_zone):
        """Did a walk leg actually cross a transition?

        A loading screen counts: `zone_name` is unreadable mid-load, and
        waiting out the load here would hold the script for nothing --
        the settle window and the script's own `waitforzonechange` both
        handle the far side.
        """
        try:
            if await client.is_loading():
                return True
        except Exception:
            pass
        try:
            zone = await client.zone_name()
        except Exception:
            return False
        return bool(zone) and bool(start_zone) and zone != start_zone

    #: how long the script is held for a possible sigil countdown. The
    #: countdown is ~10s and RESTARTS on every late join -- and the
    #: gather's teleports, prompt polls and X presses ARE late joins,
    #: then the instance load takes seconds more. Rev 7e1980b5: a 20s
    #: hold expired at t+20.0 and the script's safe-area teleport
    #: landed at t+20.1, one tick after, mid-entry. The release on a
    #: zone change is immediate, so a too-long hold at a real sigil
    #: costs nothing; only a hold at a non-sigil pays the full length,
    #: once per visit.
    COUNT_HOLD = 45.0
    #: how many times an EVIDENCED sigil (a helper's prompt seen and
    #: pressed) re-arms its hold after expiring unfired, before the
    #: spot is left to the script's own entry machinery.
    COUNT_HOLD_REPLAYS = 2
    #: how long between holds, worker-wide.
    COUNT_HOLD_EVERY = 30.0
    #: the debounce between the first qualifying look and the hold. The
    #: press-X prompt takes a beat to render after a landing; holding on
    #: the very first look would classify every fresh arrival at a Talk
    #: NPC as a possible sigil.
    COUNT_HOLD_ARM = 0.4
    #: how long a wizard may have been parked on the spot before a hold
    #: guards nothing: a countdown fires in ten to twenty seconds, so a
    #: wizard that has stood still past this has no countdown running
    #: and gets the walk-through sweep directly.
    COUNT_HOLD_STALE = 45.0
    #: how far from the leader a helper's visible press-X prompt still
    #: counts as "the same sigil". The sigil circles span ~150 and the
    #: prompt shows within a few hundred; 600 is comfortably past both
    #: and short of the next street interactable.
    COUNT_HOLD_SENSE = 600.0
    #: how close the gathered party is brought to the (possibly)
    #: counting sigil. Inside the sigil circle, and far enough that a
    #: wizard already standing on it is not re-teleported -- every
    #: re-landing restarts the countdown.
    COUNT_HOLD_GATHER = 150.0
    #: prompt polls for a gathered helper, at COUNT_HOLD_POLL_GAP
    #: apiece -- the prompt takes a beat to render after a landing, and
    #: one fixed sleep missed it live (rev 7e1980b5's T5 hold gathered
    #: Oz and pressed nothing).
    COUNT_HOLD_MATE_POLLS = 6
    #: prompt polls for the holder itself once it is planted on the
    #: pad. Longer than a helper's: rev 1b1f499c's leader got 3s five
    #: times and rendered nothing in any of them, and the extra length
    #: costs nothing inside a hold that is standing still anyway.
    COUNT_HOLD_LEADER_POLLS = 8
    #: seconds between prompt polls.
    COUNT_HOLD_POLL_GAP = 0.5
    #: how far the dead-prompt refresh steps the holder off the sigil
    #: before bringing it back: past COUNT_HOLD_SENSE, where prompts
    #: provably still show, so the client has actually LEFT the range
    #: window and has to re-arm it on the way back in.
    COUNT_HOLD_REFRESH = 800.0
    #: how long the refreshed holder stands off the sigil before coming
    #: back -- one beat, enough for the client to register the exit.
    COUNT_HOLD_REFRESH_WAIT = 1.0
    #: how long between boarding attempts INSIDE a running hold. The
    #: whole attempt -- refresh out, back, poll, press -- took 2.6s the
    #: time it worked at rev e6201303, so this leaves room for the poll
    #: to run out and still gives a 45s hold several tries instead of
    #: the one it used to get.
    #:
    #: It must also EXCEED the countdown it is guarding. `COUNT_HOLD`
    #: documents that counter as ~10s, and a retry that steps the
    #: wizard off its pad restarts it -- so at the 8.0 this used to be,
    #: a real sigil could never finish counting. Rev 09a0af80: every
    #: hold ran its full 45s at six cycles apiece and not one fired.
    #: The step-off is now once per hold anyway (`_join_leader`), and
    #: this is the second guard on the same mistake.
    COUNT_HOLD_RETRY = 12.0

    def _countdown_held(self):
        """Is the script waiting out a possible sigil countdown?"""
        import time

        return time.monotonic() < self._count_hold_until

    async def _press_on_prompt(self, seat, polls):
        """Poll for this wizard's own press-X prompt; X when it shows.

        True only when the prompt was SEEN and the press went -- the
        proof `_join_leader` deals in.
        """
        import time

        from .. import questing

        for _ in range(polls):
            try:
                if await questing.near_interactable(seat.client):
                    ok, _why = await questing.press_x(seat.client)
                    if ok:
                        seat.pressed_x_at = time.monotonic()
                        seat.tel.note_questing(
                            "countdown-hold",
                            "pressed X to join the sigil as the leader")
                        self._say(seat,
                                  f"{seat.name} presses X to join its "
                                  f"own sigil")
                        return True
                    return False
            except Exception:
                return False
            await asyncio.sleep(self.COUNT_HOLD_POLL_GAP)
        return False

    async def _retry_boarding(self, seat):
        """Try again to get the holder aboard, inside its own hold.

        The boarding is FLAKY, not impossible, and rev e6201303 is the
        proof: at t=142.8 the sensor fired, the refresh ran, the
        prompt did not render, and the guard said "it cannot join, so
        no partner is pressed in without it" -- then stood there doing
        NOTHING for the remaining 39 seconds of its own hold. The hold
        expired, re-armed, and at t=189.3 ran the IDENTICAL procedure,
        which worked in 2.6 seconds and put the party in the dungeon.

        One door, 160 seconds, and the only difference between the
        attempt that failed and the attempt that worked was that the
        second one happened. So the hold retries instead of waiting
        itself out, and the moment the holder is aboard the mates get
        the X that was being withheld from them.
        """
        import time

        now = time.monotonic()
        if self._count_hold_aboard or now < self._count_hold_retry:
            return
        self._count_hold_retry = now + self.COUNT_HOLD_RETRY
        pad = self._count_hold_pad
        try:
            here = await seat.client.body.position()
        except Exception:
            here = None
        if not await self._join_leader(seat, here, pad if pad is not None
                                       else here):
            return
        self._count_hold_aboard = True
        self._count_hold_sigil = True
        seat.tel.note_questing(
            "countdown-hold",
            "boarded on a later try inside the same hold — the first "
            "attempt at this sigil failed and the guard used to stand "
            "still for the rest of its own hold rather than try again")
        waiting = [o for o in self.seats
                   if o.index in self._count_hold_joiners
                   and o.client is not None]
        await self._press_the_joiners(seat, waiting)

    async def _join_leader(self, seat, here, pad):
        """Put the holder aboard its own sigil: prompt seen, X pressed.

        True only on that proof. Rev 1b1f499c is why nothing weaker
        counts: five sensed holds at the Kyuto tower re-planted the
        promptless leader on the exact pad the booster's prompt had
        just been seen from, its own prompt never rendered once in 3s
        of polling, and the booster -- whose X had already been pressed
        -- rode every countdown in ALONE while the wizard whose quest
        it was stood outside for nineteen minutes. The escalation:

        1. Its own prompt is up (rendered after the hold armed): press
           it. The script is frozen for the hold, so its Check_X_Key
           cannot -- this X is the one nobody else will press.
        2. Off the pad a helper's prompt was seen on: re-plant there
           and poll. A hidden prompt off the pad usually just means
           misplaced (rev 7e1980b5's first tower).
        3. ON the pad and still promptless: the prompt is DEAD, and a
           re-plant provably does not revive it (rev 1b1f499c: five
           re-plants, zero prompts). Step out past the prompt's whole
           range and back, so the client leaves the range window and
           re-triggers it coming back in, then poll again. If the
           wizard was in fact joined and counting, the step off
           un-joins it -- and the X on the way back re-joins it with
           proof, at the price of one counter restart, which the
           mates' presses were about to cause anyway.
        """
        import math
        import time

        from .. import questing

        client = seat.client
        try:
            if await questing.near_interactable(client):
                ok, _why = await questing.press_x(client)
                if ok:
                    seat.pressed_x_at = time.monotonic()
                    seat.tel.note_questing(
                        "countdown-hold",
                        "this wizard's own press-X prompt rendered once "
                        "the script was held — pressed X to join, which "
                        "the frozen script could not")
                    self._say(seat,
                              f"{seat.name} presses X to join its own "
                              f"sigil")
                    return True
        except Exception:
            pass
        if pad is None:
            # No helper prompt was seen, so nothing marks the pad --
            # but the wizard is standing at its own quest marker, which
            # is the best evidence of where the sigil is that this case
            # offers, and the refresh below is exactly what eventually
            # worked at rev e6201303. Without this the marker-case hold
            # had NOTHING to try: rev e6201303's first hold stood still
            # for its full 45 seconds, with a partner on the spot, and
            # neither wizard pressed anything.
            pad = here
        if pad is None:
            return False
        gap = None
        try:
            if here is not None:
                gap = math.hypot(pad.x - here.x, pad.y - here.y)
        except Exception:
            gap = None
        if gap is None or gap > self.COUNT_HOLD_GATHER:
            stood = (f"it stood {gap:,.0f} out" if gap is not None
                     else "its own position would not read")
            try:
                await client.teleport(pad)
            except Exception:
                return False
            seat.tel.note_questing(
                "countdown-hold",
                f"re-planted this wizard on the sigil pad its helper's "
                f"prompt was seen from — {stood}, and a hidden prompt "
                f"off the pad usually just means misplaced")
            if await self._press_on_prompt(
                    seat, self.COUNT_HOLD_LEADER_POLLS):
                return True
        # On the pad -- planted there or standing there all along --
        # and the prompt still will not render: dead. Leave the range
        # window entirely and come back in.
        #
        # ONCE per hold. The docstring above budgets for "one counter
        # restart", and that is the right price -- but `_retry_boarding`
        # calls this every `COUNT_HOLD_RETRY` seconds, so the price was
        # being paid over and over against a countdown the constant at
        # `COUNT_HOLD` documents as ~10s. Stepping off UN-JOINS the
        # wizard, so an 8-second retry cadence reset a 10-second counter
        # forever: the guard prevented the entry it exists to protect.
        # Rev 09a0af80's holds ran their full 45s at six cycles apiece
        # and not one of them fired.
        #
        # The cheap rungs above (press a prompt that IS up, re-plant off
        # the pad) may still repeat -- neither of them un-joins anything.
        if self._count_hold_stepped_off:
            return False
        self._count_hold_stepped_off = True
        try:
            marker, _why = await questing.read_quest_position(client)
        except Exception:
            marker = None
        ux = uy = None
        if marker is not None:
            dx, dy = pad.x - marker.x, pad.y - marker.y
            dist = math.hypot(dx, dy)
            if dist >= 15.0:
                # The marker sits inside the dungeon, so pad-minus-
                # marker points back OUT of it, into the street the
                # wizard came from.
                ux, uy = dx / dist, dy / dist
        if ux is None:
            # Marker unreadable or sitting on the pad itself. The
            # wizard still FACES the way it came at the sigil, so
            # backward along its yaw leads back out (forward is the
            # `- sin/cos` pair, see `_sweep_through`).
            try:
                yaw = await client.body.yaw()
                ux, uy = math.sin(yaw), math.cos(yaw)
            except Exception:
                ux, uy = 1.0, 0.0
        ox = pad.x + ux * self.COUNT_HOLD_REFRESH
        oy = pad.y + uy * self.COUNT_HOLD_REFRESH
        try:
            out = type(pad)(ox, oy, pad.z)
        except Exception:
            from types import SimpleNamespace
            out = SimpleNamespace(x=ox, y=oy, z=pad.z)
        try:
            await client.teleport(out)
            await asyncio.sleep(self.COUNT_HOLD_REFRESH_WAIT)
            await client.teleport(pad)
        except Exception:
            return False
        seat.tel.note_questing(
            "countdown-hold",
            f"this wizard's prompt would not render ON the pad a "
            f"helper's prompt shows from — stepped it "
            f"{self.COUNT_HOLD_REFRESH:.0f} off the sigil and back to "
            f"make the client re-trigger the range window, the thing a "
            f"plain re-plant provably does not (rev 1b1f499c: five "
            f"re-plants, zero prompts)")
        if await self._press_on_prompt(seat, self.COUNT_HOLD_LEADER_POLLS):
            return True
        seat.tel.note_questing(
            "countdown-hold",
            "this wizard's press-X prompt never rendered — not "
            "re-planted, not walked off and back. It cannot join, so "
            "no partner is pressed in without it")
        return False

    async def _sweep_after_hold(self, seat):
        """Chain an expired countdown hold into the walk-through sweep.

        The two rungs answer the same ambiguous state -- at the marker,
        no prompt -- with the two possible cures, in the right order:
        stand still first (a countdown loses everything to a walk), and
        only then walk (a door loses nothing to 20 still seconds). An
        expired hold has finished the standing; re-verify the state and
        do the walking, instead of returning the wheel to the script
        whose next teleport is the yank being outwaited.
        """
        import time

        from .. import questing

        now = time.monotonic()
        away = seat.marker_away
        goal = (seat.goal or "").strip().lower()
        if (away is None or away > self.AT_THE_MARKER or not goal
                or any(goal.startswith(w) for w in self.FIGHT_GOALS)):
            return
        if now - seat.walked_through_at < self.WALK_THROUGH_EVERY:
            return
        if await questing.in_dialogue(seat.client):
            return
        if await questing.near_interactable(seat.client):
            return
        try:
            if await seat.client.is_loading():
                return
        except Exception:
            pass
        await self._sweep_through(
            seat, away,
            f"held {self.COUNT_HOLD:.0f}s with no countdown")

    #: how many visits to one spot may hold for a countdown and never
    #: fire before the spot is written off as not being a sigil at all.
    #:
    #: The per-VISIT bound (`COUNT_HOLD_REPLAYS`) is not enough on its
    #: own, and rev bf3b32e7 is 88 minutes of why: the script's own
    #: retry loop walks the wizard out of the cell and back, which is a
    #: NEW visit, so the guard re-armed from scratch every time. 448
    #: holds, 47 expiries, the script held for most of two hours, and
    #: the wizard never moved off `Talk To Karolak Nightspinner`.
    #:
    #: Three, because a real sigil that swallowed three full holds
    #: without once firing is not one this rung can help with either.
    COUNT_HOLD_DUDS = 3

    def _sigil_cell(self, seat, zone=None):
        """(zone, cell) for the spot memory -- WITHOUT the quest goal.

        `seat.progress` is `(zone, position, goal)`, and keying on it
        whole meant a flapping goal silently re-keyed the memory: the
        same doorway looked like a new spot every time the quest
        tracker changed its mind, so the dud count never accumulated
        and the guard never wrote anything off. Rev 09a0af80's quester
        alternated between a Zafaria quest and a Wysteria one at the
        same marker, which is exactly that.

        The physical spot is what a sigil is. The goal is not part of
        it.
        """
        spot = seat.count_hold_spot or seat.progress
        cell = spot[1] if isinstance(spot, tuple) and len(spot) > 1 else spot
        return (zone or seat.zone_seen or "", cell)

    def _sigil_dud(self, seat, add=0):
        """How many visits to this spot have held and never fired.

        Keyed on (zone, the rounded cell `progress` already uses) so it
        survives the wizard being walked out and back -- which is the
        whole point, that being exactly what the script does between
        attempts.
        """
        where = self._sigil_cell(seat)
        if not where[1]:
            return 0
        duds = getattr(self, "_sigil_duds", None)
        if duds is None:
            duds = self._sigil_duds = {}
        if add:
            duds[where] = duds.get(where, 0) + add
            if len(duds) > 64:
                for key in list(duds)[:16]:
                    duds.pop(key, None)
        return duds.get(where, 0)

    #: total seconds of held script one spot may ever cost.
    #:
    #: `COUNT_HOLD` (45) x `COUNT_HOLD_REPLAYS` (2) x
    #: `COUNT_HOLD_DUDS` (3) is 270 seconds per spot, and that product
    #: is only a bound if all three counters hold. Two of them do not
    #: survive an ordinary run: `count_hold_replays` is a per-seat
    #: field that any of a dozen paths resets, and the dud count was
    #: keyed through `seat.progress` -- which carries the quest goal --
    #: so rev 09a0af80's flapping tracker silently re-keyed the memory
    #: and the count never reached three at all.
    #:
    #: This one cannot be reset by anything except the spot actually
    #: firing. It is deliberately larger than the nominal 270: it is
    #: not a tighter policy, it is the backstop that makes the policy
    #: true.
    COUNT_HOLD_SPEND = 300.0

    def _sigil_spent(self, seat, add=0.0, zone=None):
        """Seconds of held script this spot has cost, ever.

        Same key as `_sigil_dud` and cleared by the same event -- a
        spot that fires is a sigil, and a sigil that took four tries is
        still a sigil.
        """
        where = self._sigil_cell(seat, zone=zone)
        if not where[1]:
            return 0.0
        spent = getattr(self, "_sigil_spends", None)
        if spent is None:
            spent = self._sigil_spends = {}
        if add:
            spent[where] = spent.get(where, 0.0) + add
            if len(spent) > 64:
                for key in list(spent)[:16]:
                    spent.pop(key, None)
        return spent.get(where, 0.0)

    def _sigil_fired(self, seat):
        """This spot IS a sigil after all: forget its failures.

        Keyed on the zone the hold BEGAN in, not the one the wizard is
        standing in now -- firing is a zone change, so by the time this
        runs `zone_seen` is the far side of the door and would look up
        a spot that has never been held at.
        """
        where = self._sigil_cell(seat, zone=self._count_hold_zone)
        for ledger in (getattr(self, "_sigil_duds", None),
                       getattr(self, "_sigil_spends", None)):
            if ledger:
                ledger.pop(where, None)

    async def _maybe_count_hold(self, seat):
        """Hold the script while a dungeon sigil may be counting down.

        The operator's correction that produced this rung, after the
        landing check alone did not explain their screen: "konstantin
        literally had the sigil countdown running, and was still
        teleporting away". So the wizard WAS on the sigil -- and that
        is precisely the state the script cannot recognise: a wizard
        standing on a joined sigil shows no press-X prompt (the same
        fact that broke the first steady_sigil cut), so the preset's
        `Check_X_Key_Type` -- whose dungeon test is `NPCRangeWin` +
        `TeamUpButton` visible -- reads nothing, concludes "not a
        dungeon", and the next main-loop iteration opens with
        "Teleporting all clients to a safe area", cancelling the
        countdown the wizard was standing in. Rev e786b716 did that
        twice before one attempt happened to land NEAR the sigil
        instead of ON it, which kept the prompt up and let the script's
        own path work.

        wizAi can act on the ambiguity the script cannot: a scripted
        wizard parked at its non-fight quest marker, out of combat,
        with NO dialogue and NO press-X prompt, is either standing on a
        counting sigil or standing at a door -- and in both cases the
        wrong move is teleporting away. So: hold the VM (state kept,
        nothing torn down), bring the rest of the party onto the spot
        (a sigil admits exactly the wizards ON it, and each join
        restarts the counter once), and wait. A zone change releases
        the hold early -- the sigil fired, and the script resumes to a
        world its watchdog re-syncs against, which rev e786b716's own
        log shows it does cleanly. Nothing in 20s means it was not a
        counting sigil, and the wheel goes back.

        Once per visit to a spot: the stamp clears when the wizard
        leaves the cell, so a wizard yanked away and teleported back
        gets a fresh hold -- standing on a sigil restarts its countdown,
        and the second visit is exactly as protectable as the first.
        """
        import time

        from .. import questing

        now = time.monotonic()
        if self._count_hold_until:
            holder = seat.index == self._count_hold_seat
            if now < self._count_hold_until:
                if not holder:
                    return
                if (seat.zone_seen and self._count_hold_zone
                        and seat.zone_seen != self._count_hold_zone):
                    self._count_hold_until = 0.0
                    seat.count_hold_replays = 0
                    # It fired, so this spot IS a sigil: forget any
                    # earlier visits that did not. A door that opens on
                    # the fourth try is still a door.
                    self._sigil_fired(seat)
                    seat.tel.note_questing(
                        "countdown-hold-over",
                        f"the zone changed while the script was held — a "
                        f"sigil counted down and fired, or a walk already "
                        f"in flight crossed a door. Either way, this is "
                        f"the entry the script's own loop kept cancelling")
                    self._say(seat, "the sigil fired — script released")
                    return
                if seat.goal_at and seat.goal_at > self._count_hold_began:
                    # The GOAL advancing is the zone change's equal: the
                    # spot's business is done. Rev 672d1c79 engaged at
                    # "marker 0 away" beside a Talk NPC mid-turn-in;
                    # auto-dialogue clicked the conversation through, the
                    # goal moved on within seconds — and the hold kept
                    # the script frozen at the finished NPC for its full
                    # 45s, which is the "waited so long at a dialogue"
                    # the operator watched.
                    self._count_hold_until = 0.0
                    seat.count_hold_replays = 0
                    seat.tel.note_questing(
                        "countdown-hold-over",
                        f"the goal advanced while the script was held — "
                        f"the spot's business was a turn-in that "
                        f"completed, not a sigil. Script released at once")
                    self._say(seat, "the goal advanced — script released")
                    return
                if self._count_hold_sigil:
                    # A gathered mate whose zone changed while the
                    # holder's did not: the countdown fired WITHOUT
                    # this wizard. Rev 1b1f499c burned the rest of a
                    # 45s hold plus two replays on that state, five
                    # times, while the booster bounced in and out of
                    # the tower alone. Release at once -- the follow
                    # drags the mate back within seconds, and clearing
                    # the visit stamp lets a fresh gather retry the
                    # moment it returns. (If the holder's own zone
                    # read is merely a beat behind, this release is
                    # the one the zone change was about to do anyway.)
                    for other in self.seats:
                        if other.index not in self._count_hold_party:
                            continue
                        if (other.zone_seen and self._count_hold_zone
                                and other.zone_seen
                                != self._count_hold_zone):
                            self._count_hold_until = 0.0
                            seat.count_hold_replays += 1
                            fresh = (seat.count_hold_replays
                                     < self.COUNT_HOLD_REPLAYS)
                            if fresh:
                                seat.count_hold_spot = None
                                self._count_hold_last = NEVER
                            seat.tel.note_questing(
                                "countdown-hold-over",
                                f"{other.name} crossed into another zone "
                                f"mid-hold while this wizard's did not — "
                                f"the countdown fired without it. "
                                + (f"Released at once so the party can "
                                   f"regroup for a fresh count"
                                   if fresh else
                                   f"This sigil has burned its retries; "
                                   f"the script's own machinery gets "
                                   f"the spot"))
                            self._say(seat,
                                      f"{other.name} rode the countdown "
                                      f"in without {seat.name} — "
                                      f"released to regroup")
                            return
                # Still holding, still guarding a sigil, and the holder
                # may not be on it yet. Try again rather than waiting
                # the clock out -- see `_retry_boarding`.
                await self._retry_boarding(seat)
                return
            self._count_hold_until = 0.0
            if holder:
                if self._count_hold_sigil:
                    # A helper's prompt was SEEN and pressed here: this
                    # is a sigil, not a door, and sweeping it is the
                    # "walking out of the sigil for no reason" the
                    # operator watched at rev 7e1980b5. An unfired
                    # countdown at an evidenced sigil usually means a
                    # late join restarted it, so the guard re-arms at
                    # once -- bounded, because a sigil that will not
                    # fire twice running is not going to on the third.
                    seat.count_hold_replays += 1
                    if seat.count_hold_replays < self.COUNT_HOLD_REPLAYS:
                        seat.count_hold_spot = None
                        self._count_hold_last = NEVER
                        seat.tel.note_questing(
                            "countdown-hold-over",
                            f"held {self.COUNT_HOLD:.0f}s at an evidenced "
                            f"sigil and no zone change came — a late join "
                            f"restarts the counter, so the guard re-arms "
                            f"here instead of walking. No sweep touches a "
                            f"live sigil")
                    else:
                        seat.count_hold_replays = 0
                        duds = self._sigil_dud(seat, +1)
                        seat.tel.note_questing(
                            "countdown-hold-over",
                            f"held at this sigil "
                            f"{self.COUNT_HOLD_REPLAYS} times and it never "
                            f"fired — leaving the entry to the script's "
                            f"own machinery"
                            + (f". That is {duds} visit(s) to this spot "
                               f"that have held and never fired"
                               if duds > 1 else ""))
                    return
                seat.tel.note_questing(
                    "countdown-hold-over",
                    f"held {self.COUNT_HOLD:.0f}s and no zone change came — "
                    f"not a counting sigil (or the countdown was already "
                    f"lost). The state left standing is the walk-in door's "
                    f"exactly, so the sweep gets it before the script does")
                # An expired hold IS the walk-through's evidence, already
                # aged: at the marker, no prompt, the hold's length of
                # proven nothing. Rev d3ed4d3c spent seven minutes at
                # the Emperor's Palace door in MS_Hub while four holds
                # expired one after another and the separate
                # walk-through clock never ran 45s unbroken -- the
                # script's yanks kept resetting it. Chaining here
                # converts the wasted hold into the walk that enters
                # the door.
                await self._sweep_after_hold(seat)
            return

        if seat.count_hold_spot and seat.progress != seat.count_hold_spot:
            # Left the stamped spot -- a return is a fresh visit. Ahead
            # of the gate below, because "left" is usually noticed on a
            # tick when the marker reads far, which the gate ends.
            seat.count_hold_spot = None
        if (seat.party_pulled_spot
                and seat.progress != seat.party_pulled_spot):
            seat.party_pulled_spot = None
        away = seat.marker_away
        goal = (seat.goal or "").strip().lower()
        fight = any(goal.startswith(w) for w in self.FIGHT_GOALS)
        # The marker case. Fight goals are IN, deliberately: the first
        # cut excluded them ("a defeat marker sits on the mob pack"),
        # and the operator's screenshot answered it -- "Defeat Maito in
        # Tatakai Outpost" is a boss INSIDE a dungeon, its marker sits
        # at the sigil, and the excluded hold let the script yank the
        # wizard mid-count. A street-pack defeat pays one 20s hold per
        # visit for that, and usually not even that: the pack pulls the
        # wizard into a duel long before the hold expires. Fight goals
        # stay OUT of the sweeps -- walking through a mob pack answers
        # nothing.
        marker_case = (away is not None and away <= self.AT_THE_MARKER
                       and bool(goal)
                       and not questing.is_collect_goal(seat.goal))
        if marker_case and self._marker_is_another_world(seat):
            # A near marker for a goal that names another WORLD is not
            # this wizard's objective, whatever the arrow says.
            # `MARKER_IN_ZONE` was the only guard against a marker read
            # in somebody else's coordinate space, and it assumes
            # another zone always comes back as six figures -- rev
            # 1843e387 logged 98,813 and 115,018, which is where the
            # number came from. Rev 09a0af80's Wysteria marker, read
            # from the Zafaria hub, came back as 81 and then 0 and went
            # straight through it.
            marker_case = False
            self._say_once(
                seat, f"marker-elsewhere:{seat.name}:{seat.goal}",
                f"{seat.name}'s quest marker reads close, but its goal "
                f"is in another world — not treating this spot as its "
                f"objective",
                kind="marker-another-world",
                detail=(f"{seat.goal!r} names a destination in another "
                        f"world while the wizard is in "
                        f"{seat.zone_seen or 'an unread zone'}. The "
                        f"quest arrow is one pointer with no quest "
                        f"attached to it, and a marker read in another "
                        f"world's coordinate space can land anywhere — "
                        f"including on top of the wizard"))
        # The booster-as-sensor case, and it needs no marker at all:
        # "Talk To Hoi Mang in Crimson Fields" reads its marker from
        # INSIDE the dungeon -- past every distance gate -- while the
        # party stands at the sigil. But the party itself is the
        # detector: a JOINED leader shows no press-X prompt, and an
        # unjoined helper standing beside it shows one. That asymmetry
        # is the sigil-mid-entry state, photographed twice by the
        # operator.
        mates = [s for s in self.seats
                 if s is not seat and s.client is not None
                 and self._follows(s) and s.zone_seen == seat.zone_seen]
        # The operator's ask, rev 1b1f499c post-mortem: "why wouldn't
        # we move all of the accounts to the sigil at the same time so
        # there's no chance of them missing out on entering". So the
        # moment the leader reaches a non-collect quest marker -- the
        # only places sigils live -- every follower in the zone is
        # pulled onto it AT ONCE, concurrently, before any prompt,
        # press or countdown exists to be late for. Ahead of every
        # gate below on purpose: this must fire even when the prompt
        # is up (the sigil that works first try) and even inside the
        # hold's cooldown. The gather still re-checks positions later;
        # in the common case this makes its teleports no-ops, and it
        # makes "the booster was still up the street when the counter
        # fired" impossible.
        if (marker_case and mates and seat.progress is not None
                and seat.party_pulled_spot != seat.progress):
            seat.party_pulled_spot = seat.progress
            spot = None
            try:
                if not await questing.in_battle(seat.client):
                    spot = await seat.client.body.position()
            except Exception:
                spot = None
            if spot is not None:
                async def pull(other):
                    try:
                        if await questing.in_battle(other.client):
                            return
                        at = await other.client.body.position()
                        gap = ((at.x - spot.x) ** 2
                               + (at.y - spot.y) ** 2) ** 0.5
                        if gap <= self.COUNT_HOLD_GATHER:
                            return
                        await other.client.teleport(spot)
                        other.tel.note_questing(
                            "party-pulled",
                            f"pulled onto {seat.name}'s quest marker the "
                            f"moment it arrived — a sigil admits exactly "
                            f"the wizards standing on it when its counter "
                            f"fires, so the party moves as one, before "
                            f"any countdown exists to be late for")
                        self._say(other,
                                  f"{other.name} moves with {seat.name} "
                                  f"to the marker")
                    except Exception:
                        pass
                await asyncio.gather(*[pull(o) for o in mates])
        if seat.progress is None or (not marker_case and not mates):
            seat.count_hold_seen = None
            return
        if seat.count_hold_spot == seat.progress:
            return                       # already guarded this visit
        if now - self._count_hold_last < self.COUNT_HOLD_EVERY:
            return
        if seat.count_hold_seen is None:
            seat.count_hold_seen = now   # first look arms; second holds
            return
        if now - seat.count_hold_seen < self.COUNT_HOLD_ARM:
            return
        if await questing.in_dialogue(seat.client):
            seat.count_hold_seen = None
            return
        if await questing.near_interactable(seat.client):
            # The prompt is up: NOT a joined sigil. The script's own
            # check reads this state correctly and presses X itself.
            seat.count_hold_seen = None
            return
        try:
            if await seat.client.is_loading():
                return
        except Exception:
            pass

        try:
            here = await seat.client.body.position()
        except Exception:
            here = None
        sensed = []
        shopfront = []
        if here is not None:
            for other in mates:
                try:
                    if await questing.in_battle(other.client):
                        continue
                    at = await other.client.body.position()
                    gap = ((at.x - here.x) ** 2
                           + (at.y - here.y) ** 2) ** 0.5
                    if gap > self.COUNT_HOLD_SENSE:
                        continue
                    if await questing.at_a_sigil(other.client):
                        sensed.append(other)
                    elif await questing.near_interactable(other.client):
                        # A prompt with no Team Up button: a vendor, a
                        # bank, a Talk NPC. Recorded but NOT counted as
                        # sigil evidence -- the old sensor read exactly
                        # this as "the party is at a sigil mid-entry"
                        # and spent 36% of rev 09a0af80 holding at
                        # conversations.
                        shopfront.append(other)
                except Exception:
                    continue
        if shopfront and not sensed:
            # Graded, not gated. The Team Up window path cannot be
            # checked without a live client, so a spot that shows a
            # prompt WITHOUT one is not refused outright -- it falls
            # back to the plain marker case, which is what this rung
            # did before any of this existed. What it does lose is the
            # evidenced-sigil privileges: no re-arming on expiry, and
            # the spot starts collecting duds immediately, so a Talk
            # NPC stops being revisited instead of costing 45s a time.
            self._sigil_dud(seat, +1)
            self._say_once(
                seat, f"shopfront:{seat.zone_seen}:{seat.progress}",
                f"{', '.join(o.name for o in shopfront)} is at a press-X "
                f"prompt beside {seat.name}, but it has no Team Up "
                f"button — not a sigil",
                kind="countdown-hold-refused",
                detail=(f"a partner's prompt here has no Team Up button, "
                        f"so this is a vendor, a bank or a Talk NPC and "
                        f"not a dungeon sigil the party is entering. "
                        f"Holding on the plain marker case only, with no "
                        f"re-arm — rev 09a0af80 read this exact state as "
                        f"a sigil mid-entry and spent 36% of the run "
                        f"frozen at conversations"))

        if not marker_case and not sensed:
            return

        if (marker_case and not sensed and seat.progress_at
                and now - seat.progress_at > self.COUNT_HOLD_STALE):
            # Standing on this spot for this long already means no
            # countdown is running -- one fires in ten to twenty
            # seconds, not a minute. Rev d3ed4d3c: "marker 0 away ·
            # unchanged for 2 min" at the Emperor's Palace door, and a
            # hold there guards a countdown that provably is not
            # happening. Straight to the walk -- except at a fight
            # marker, where walking through the pack answers nothing.
            seat.count_hold_spot = seat.progress
            seat.count_hold_seen = None
            self._count_hold_last = now
            if (not fight
                    and now - seat.walked_through_at
                    >= self.WALK_THROUGH_EVERY):
                await self._sweep_through(
                    seat, away,
                    f"parked {(now - seat.progress_at) / 60:.0f} min with "
                    f"no countdown possible")
            return

        seat.count_hold_spot = seat.progress
        spent = self._sigil_spent(seat)
        if spent >= self.COUNT_HOLD_SPEND:
            # The total, independent of how the visits were counted.
            # Everything else that bounds this rung is a counter some
            # other path can reset; this is a clock that only the spot
            # firing can stop.
            seat.count_hold_seen = None
            self._count_hold_last = now
            self._say_once(
                seat, f"sigil-spent:{self._sigil_cell(seat)}",
                f"{seat.name} is not holding here again — this spot has "
                f"already cost {spent / 60:.0f} min of held script and "
                f"never fired",
                kind="countdown-hold-refused",
                detail=(f"declined to hold the script at this spot: it "
                        f"has spent {spent:.0f}s frozen here across "
                        f"every visit, against a ceiling of "
                        f"{self.COUNT_HOLD_SPEND:.0f}s. Whatever is "
                        f"here, four minutes of a stopped script has "
                        f"not got the party through it"))
            return
        duds = self._sigil_dud(seat)
        if duds >= self.COUNT_HOLD_DUDS:
            # This spot has taken three full holds and never fired. It
            # is not a sigil, and holding the script here again buys
            # nothing but another 45 seconds of a wizard standing still.
            #
            # Rev bf3b32e7: `Talk To Karolak Nightspinner in Stormriven`
            # with the marker 270 away, and the evidence for a sigil was
            # Oz standing at a press-X prompt beside it -- Oz's OWN
            # prompt, for its own quest, which it was on because the two
            # wizards were never on the same one. The sensor cannot tell
            # that prompt from a shared sigil's, and this is the backstop
            # that does not have to.
            seat.count_hold_seen = None
            self._count_hold_last = now
            self._say_once(
                seat, f"sigil-dud:{seat.zone_seen}:{seat.progress}",
                f"{seat.name} is not holding here again — this spot has "
                f"held for a countdown {duds} times and never fired",
                kind="countdown-hold-refused",
                detail=(f"declined to hold the script at this spot: "
                        f"{duds} visits have held the full "
                        f"{self.COUNT_HOLD:.0f}s for a countdown and none "
                        f"of them fired. Whatever a helper's press-X "
                        f"prompt beside this wizard is for, it is not a "
                        f"sigil this wizard can enter — and every hold "
                        f"here is {self.COUNT_HOLD:.0f}s of the script "
                        f"not running"))
            return
        seat.count_hold_seen = None
        self._count_hold_last = now
        # Charged when the hold is COMMITTED, at its full length,
        # rather than measured at whichever of the half-dozen exits
        # ends it. Deliberately: the ledger's job is to be the one
        # bound nothing else can reset, and a charge that depends on
        # reaching a particular exit is a charge some path can skip. A
        # spot released early by its goal advancing is over-charged by
        # the difference -- and a spot where the goal advances is a
        # conversation, which is what this ceiling is for. A spot that
        # FIRES has its ledger cleared outright, so a real sigil never
        # pays for the tries it took.
        self._sigil_spent(seat, add=self.COUNT_HOLD)
        self._count_hold_until = now + self.COUNT_HOLD
        self._count_hold_zone = seat.zone_seen
        self._count_hold_seat = seat.index
        self._count_hold_sigil = bool(sensed)
        self._count_hold_stepped_off = False
        self._count_hold_began = now
        self._count_hold_party = tuple(s.index for s in mates)
        if sensed:
            spotted = ", ".join(o.name for o in sensed)
            saw = (f"{spotted} is standing at a press-X prompt beside "
                   f"this wizard, whose own prompt is hidden — the party "
                   f"is at a sigil mid-entry")
        else:
            saw = (f"standing at the quest marker ({away:,.0f} away) "
                   f"with no press-X prompt — the joined-sigil state "
                   f"the script cannot see")
        seat.tel.note_questing(
            "countdown-hold",
            f"{saw}. Script held up to {self.COUNT_HOLD:.0f}s so its "
            f"next teleport cannot cancel a running countdown; a zone "
            f"change releases it early")
        self._say(seat,
                  f"{seat.name} may be standing on a counting sigil — "
                  f"holding the script and gathering the party onto it")

        # A sigil admits exactly the wizards that JOINED it before the
        # counter fires -- and standing on it is not joining it, the
        # operator's screenshots say so: the booster stood at "Press X
        # to Enter" through the whole countdown, twice, and entered
        # nothing. So the gather brings the party onto the spot and
        # finds the pad (the exact position a prompt provably shows
        # from) -- but presses NOTHING until the holder itself is
        # aboard. Rev 1b1f499c ran the other order into the ground for
        # nineteen minutes: the booster's X went first, the leader's
        # prompt never rendered, and the counter fired the booster in
        # ALONE five times while the wizard whose quest it was stood
        # outside. A booster inside a dungeon without its quester
        # helps nobody.
        pad = None
        joiners = []
        for other in self.seats:
            if other is seat or other.client is None:
                continue
            if other.zone_seen != seat.zone_seen:
                continue                 # cross-zone is the follow's job
            try:
                if await questing.in_battle(other.client):
                    continue
                if here is not None:
                    at = await other.client.body.position()
                    gap = ((at.x - here.x) ** 2
                           + (at.y - here.y) ** 2) ** 0.5
                    if gap > self.COUNT_HOLD_GATHER:
                        await other.client.teleport(here)
                        other.tel.note_questing(
                            "countdown-hold",
                            f"stepped onto {seat.name}'s spot — a sigil "
                            f"admits only the wizards standing on it when "
                            f"the countdown fires, and each join restarts "
                            f"the counter once")
                        self._say(other,
                                  f"{other.name} steps onto "
                                  f"{seat.name}'s sigil")
                # The narrow read, for the same reason as the arming
                # sensor: this decides both that the spot IS a sigil
                # (`_count_hold_sigil` below, which buys it re-arms and
                # exempts it from sweeps) and that this wizard should be
                # sent to press X. At a Talk NPC the press opens a
                # conversation nobody asked for.
                shows = False
                for _ in range(self.COUNT_HOLD_MATE_POLLS):
                    if await questing.at_a_sigil(other.client):
                        shows = True
                        break
                    await asyncio.sleep(self.COUNT_HOLD_POLL_GAP)
                if not shows:
                    continue
                try:
                    pad = await other.client.body.position() or pad
                except Exception:
                    pass
                joiners.append(other)
            except Exception:
                continue

        # The holder boards first -- prompt seen, X pressed, nothing
        # weaker -- and only then do the helpers get theirs, so their
        # presses are the LAST joins and the whole party rides the
        # final count down together.
        aboard = await self._join_leader(seat, here, pad)
        if pad is not None or aboard:
            # A prompt was SEEN at this spot -- sigil evidence, whether
            # a helper's or the holder's own. Evidenced spots re-arm on
            # expiry and are never swept.
            self._count_hold_sigil = True
        # Kept for the retries: the boarding is FLAKY, not impossible,
        # and giving up for the rest of the hold is what made rev
        # e6201303 take 160 seconds to walk through one door. See
        # `_retry_boarding`.
        self._count_hold_aboard = bool(aboard)
        self._count_hold_pad = pad if pad is not None else here
        self._count_hold_joiners = tuple(o.index for o in joiners)
        self._count_hold_retry = time.monotonic() + self.COUNT_HOLD_RETRY
        if joiners and not aboard:
            self._say(seat,
                      f"{seat.name} could not join its own sigil — "
                      f"nobody else is pressed in without it")
            return
        await self._press_the_joiners(seat, joiners)

    async def _press_the_joiners(self, seat, joiners):
        """The helpers' X, once the holder is provably aboard."""
        import time

        from .. import questing

        for other in joiners:
            try:
                ok, _why = await questing.press_x(other.client)
                if ok:
                    other.pressed_x_at = time.monotonic()
                    other.tel.note_questing(
                        "countdown-hold",
                        f"pressed X to join {seat.name}'s sigil — "
                        f"standing on a sigil is not joining it, the "
                        f"prompt is")
                    self._say(other,
                              f"{other.name} presses X to join the "
                              f"sigil")
            except Exception:
                continue

    #: how long the quest position must be continuously unreadable --
    #: goal line reading fine the whole time -- before the hook counts
    #: as DEAD rather than mid-blink. Zone changes fail several reads in
    #: a row; nothing transient fails them for two minutes straight.
    MARKER_DEAD_AFTER = 120.0
    #: how long between re-arm attempts. Paging through the quest book
    #: costs ~15 held seconds, and a cure that did not take the first
    #: time needs the operator's eyes more than it needs a third try.
    REARM_EVERY = 300.0
    #: the ceiling on the worker-wide steering hold around a re-arm, and
    #: the settle after it -- same shape as the desperate hop's, for the
    #: same reason: the script clicking mid-book is the settings-menu
    #: fiasco with worse aim.
    REARM_CEILING = 60.0
    REARM_SETTLE = 3.0

    #: how long a marker read stays evidence that this client's quest
    #: hook WORKS. Fifteen minutes of quests is many steps; a hook that
    #: was writing positions that recently is not switched off.
    HOOK_ALIVE = 900.0

    def _marker_absent_by_design(self, seat, now=None):
        """Why this quest has no marker of its own, or "".

        The distinction the last round got wrong, in the operator's
        words: "there is no quest marker, this is a collect quest still
        even when you select on the quest". A quest position that will
        not read is only evidence of a dead arrow when the arrow is
        what could have written it. Two things say otherwise, and both
        are cheap:

        -- the step is a Collect. Those publish no position at all (see
           `questing.is_collect_goal`), on every client, for everybody.
        -- this client's hook read a position on some OTHER quest
           within `HOOK_ALIVE`. A hook that was writing positions ten
           minutes ago is not switched off now, so it is THIS quest
           that has nothing to point at.

        Either way the cure for a dead arrow is not the cure for this,
        and offering it -- a trip through the quest book every five
        minutes, which is how a journal ends up on the Quest Finder
        pseudo-entry -- makes the run worse rather than better.
        """
        import time

        from .. import questing

        if questing.is_collect_goal(seat.goal):
            return ("a Collect step publishes no quest position — the "
                    "game has no single place to point the arrow, so "
                    "every teleport-to-marker rung is out of the game "
                    "for this step and the script's hardcoded spots are "
                    "what finishes it")
        if now is None:
            now = time.monotonic()
        was = seat.marker_ok_goal
        if (was and was != seat.goal
                and now - seat.marker_ok_at < self.HOOK_ALIVE):
            return (f"this wizard's quest hook works — it read a position "
                    f"on {was!r} {(now - seat.marker_ok_at) / 60:.0f} min "
                    f"ago — so it is this QUEST that publishes no marker, "
                    f"not the arrow that is off")
        return ""

    def _marker_dead(self, seat, now=None):
        """Is this seat's quest hook provably dead, not merely blinking?

        True only after `MARKER_DEAD_AFTER` of continuous failed marker
        reads with a readable goal line -- the signature of the quest
        arrow being off (`read_quest_position`'s (0,0,0) case) -- and
        only when the quest itself is not the reason there is nothing
        to read. See `_marker_absent_by_design`.
        """
        import time

        since = seat.marker_dead_since
        if since is None:
            return False
        if now is None:
            now = time.monotonic()
        if now - since < self.MARKER_DEAD_AFTER:
            return False
        return not self._marker_absent_by_design(seat, now)

    def _marker_unusable(self, seat, now=None):
        """Why nothing can aim a teleport for this wizard, or "".

        The question the rungs that MOVE a wizard actually have --
        `_desperate_hop`, the catch-up's `hop_once`, the realm change's
        "am I at the spawns" -- as distinct from the question the cure
        has (`_marker_dead`). A Collect step and a dead arrow are
        different problems with different fixes, and both of them mean
        no teleport can aim.
        """
        import time

        if seat.marker_away is not None:
            return ""
        if now is None:
            now = time.monotonic()
        why = self._marker_absent_by_design(seat, now)
        if why:
            return why
        if self._marker_dead(seat, now):
            return ("the quest position has not read for minutes while "
                    "the goal line reads fine — the in-game quest arrow "
                    "is off, and every teleport on this wizard needs it")
        return ""

    async def _maybe_rearm_quest_arrow(self, seat):
        """Cure a dead quest hook by re-selecting the quest in the book.

        Rev 98b4c50c, the whole run: Konstantin's quest position read
        (0,0,0) for 44 minutes. The desperate hop returned silently
        (no marker), the realm change refused (no marker), the script
        restart refused (no marker), and six catch-ups paused the whole
        party two minutes each to attempt nothing -- because
        `hop_once` needs the same marker. Meanwhile the script's own
        `tp quest` (vm.py:1547) read the same dead hook and navmap'd
        him toward (0,0,0), which is the zone-wandering in the export.
        Every actor was starved by one unwritten hook, and the only
        cure is the one the diagnosis line already names: "switch the
        in-game quest arrow on and pick a quest". This does that.

        Held worker-wide like the desperate hop: the book is open for
        seconds, and any teleport another task lands mid-click turns
        the click into a misfire.
        """
        import time

        from .. import questing

        now = time.monotonic()
        if not self._marker_dead(seat, now):
            return
        if now - seat.rearm_tried_at < self.REARM_EVERY:
            return
        if seat.in_duel:
            return
        # Preconditions the cure cannot work under, checked BEFORE the
        # cooldown is spent -- a refusal is not an attempt.
        if await questing.in_dialogue(seat.client):
            return
        seat.rearm_tried_at = now
        dead_min = (now - (seat.marker_dead_since or now)) / 60.0
        self._say(seat,
                  f"the quest position has been unreadable for "
                  f"{dead_min:.0f} min while the goal line reads fine — "
                  f"the quest arrow is off, which starves every teleport "
                  f"on this wizard including the script's own. Doing what "
                  f"the diagnosis says: opening the quest book and "
                  f"re-selecting the quest")
        self._hop_pause_until = now + self.REARM_CEILING
        try:
            ok, why = await questing.rearm_quest_arrow(
                seat.client, goal=seat.goal or "",
                name=seat.quest_name or "",
                on_status=lambda m: self._say(seat, m))
        finally:
            self._hop_pause_until = time.monotonic() + self.REARM_SETTLE
        if ok:
            seat.tel.note_questing(
                "quest-arrow-rearmed",
                f"re-selected the tracked quest in the quest book and the "
                f"quest position reads again — the arrow had been off for "
                f"{dead_min:.0f} min, starving every teleport on this "
                f"wizard (wizAi's and the script's alike)")
            self._say(seat, "the quest arrow is back on — the quest "
                            "position reads again")
        else:
            seat.tel.note_questing(
                "quest-arrow-rearm-failed",
                f"{why} — until the quest arrow is on and a quest is "
                f"picked, no teleport on this wizard can aim: not the "
                f"script's, not the catch-up's, not the desperate hop. "
                f"Next attempt in {self.REARM_EVERY / 60:.0f} min")
            self._say(seat,
                      f"could not re-arm the quest arrow: {why}. This "
                      f"wizard needs the arrow switched on by hand — "
                      f"quest book, pick the quest")

    #: how long a COLLECT goal may sit unchanged, at its own marker and
    #: out of combat, before the realm is judged crowded rather than the
    #: wizard stuck. Deliberately after every gentler fix on the ladder:
    #: X at 5 min, the desperate teleport at ~5.5, this at 8 -- because
    #: a realm change is the one move that costs the whole party a zone
    #: reload.
    REALM_HOP_AFTER = 480.0
    #: the least time between realm changes. Also what makes the manual
    #: button and the broadcast request queue safe: every seat drains
    #: its own copy of the "realm" action, the first one hops the whole
    #: party, and the stamp turns the other two into no-ops.
    REALM_HOP_COOLDOWN = 300.0

    async def _maybe_realm_hop(self, seat):
        """Change realms when a Collect step is contested, not stuck.

        The distinction the stuck ladder cannot make on its own: a
        wizard at the right marker, out of combat, with a Collect goal
        that has not ticked in eight minutes is not wedged -- it is
        QUEUING. The collectibles respawn on a timer and every other
        player in a crowded realm takes one. No amount of pressing X or
        teleporting to the marker fixes a queue; moving to an empty
        shard does, same zone, same spot. The scan-and-pick is a Deimos
        community contribution (see `realms.py`); the trigger and the
        party coordination are wizAi's.

        The marker gate this rung shipped with made it **unreachable**,
        and the operator is the one who found it: "there is no quest
        marker, this is a collect quest still even when you select on
        the quest". Collect steps publish no quest position at all --
        so `marker_away` is None for every one of them, and a rung
        written for exactly this family refused every case it was
        built to answer. A marker that positively reads ANOTHER zone
        still refuses (that is a journey, not a queue); a marker that
        does not read because the step has none is now what it always
        was, which is normal for a Collect.
        """
        import time

        from .. import questing

        if not questing.is_collect_goal(seat.goal):
            return
        if not (self.script or self.auto_quest):
            return
        now = time.monotonic()
        if not seat.progress_at:
            return
        away = seat.marker_away
        if away is not None and away > self.MARKER_IN_ZONE:
            # A marker that reads, in ANOTHER zone: the wizard is not
            # at the spawns at all and getting there is the script's
            # journey. Some other stall, and the rest of the ladder
            # owns it.
            return
        # Frozen, or working? The two failures this rung has to tell
        # apart look OPPOSITE on the ground, and the clock the first
        # version used could only see one of them.
        #
        # A crowded realm MOVES: the quester teleports between its
        # hardcoded spots every few seconds and presses X at each, so
        # the wizard's position keeps changing while the COUNT stays at
        # (0 of 4). The clock that catches that is the goal's, not the
        # position's -- and `progress_at` resets on every one of those
        # teleports, so a stillness gate never fires for the very case
        # this rung exists for.
        #
        # A wizard in the wrong zone is STILL: the quester's
        # `if p1 inzone ...` body never runs, nothing teleports it
        # anywhere, and a realm change -- which keeps the zone and only
        # swaps the shard -- cannot put it where the collectibles are.
        # That one belongs to the script restart.
        frozen = now - seat.progress_at >= self.REALM_HOP_AFTER
        if away is not None:
            # A marker in this zone is the original, validated case:
            # standing at the spawns with nothing happening.
            if not frozen:
                return
        else:
            if frozen:
                self._say_once(
                    seat, f"collect-frozen:{seat.name}",
                    f"{seat.name}'s Collect step has not moved and neither "
                    f"has {seat.name} — a realm change only swaps the "
                    f"shard, so it cannot help a wizard that is not "
                    f"working the spots. Leaving this to the script",
                    kind="realm-hop-refused",
                    detail=(f"{seat.goal!r} is not advancing, but nothing "
                            f"is teleporting this wizard between the "
                            f"collect spots either — that is a script "
                            f"that cannot act from where it is standing, "
                            f"not a contested realm"))
                return
            if not seat.goal_at or now - seat.goal_at < self.REALM_HOP_AFTER:
                # The count is still moving, or has not been watched
                # long enough to say it is not.
                return
            here = questing.collect_count(seat.goal)
            if here and not (seat.collect_moved_for == here[0]
                             and seat.collect_moved_at):
                # ...and it has to have moved AT LEAST ONCE. A crowded
                # realm is a wizard picking up collectibles slowly; a
                # count that has never left its starting number is a
                # wizard that never reached the spawns, and swapping
                # shards cannot put it there.
                #
                # Rev 3cbb6091 is why this is not optional. Konstantin
                # sat at `(0 of 4)` for 57 minutes at the world portal
                # -- never one gemstone -- while this rung hopped the
                # party through seven realms on the claim that he was
                # "working them and finding nothing". The seventh split
                # the party across realms, which is strictly worse than
                # the crowding it was answering. Being WITH the party
                # is not being AT the spawns; only the count knows.
                self._say_once(
                    seat, f"collect-never-started:{seat.name}",
                    f"{seat.name}'s Collect count has never moved off "
                    f"{here[1]} of {here[2]} — it has not reached the "
                    f"collectibles at all, so a quieter realm is not the "
                    f"answer. Not changing realms",
                    kind="realm-hop-refused",
                    detail=(f"{seat.goal!r} has never advanced once. A "
                            f"crowded realm shows a count that moves and "
                            f"then stalls; a count still on its starting "
                            f"number is a wizard that never got to the "
                            f"spawns, and a shard swap keeps the zone"))
                return
        if away is None and not self._with_the_party(seat):
            # ...and without a marker, the party IS the check. A realm
            # change keeps the zone and only changes the shard, so it
            # can only help a wizard already standing where the
            # collectibles are. Being alone in a zone is the other
            # failure entirely, and it is `_check_together`'s.
            #
            # It is the failure the quester's own structure produces,
            # too. TTS Arc 1 wraps the gemstone spots in
            # `until NOT p1 tracking_goal ... { if p1 inzone
            # KT_ChampHall { ... } }` -- an until-loop whose only body
            # is zone-gated and which contains nothing that would
            # TRAVEL there. A wizard on that step in the wrong zone
            # spins the loop for the rest of the run, and hopping the
            # whole party to a quiet realm would answer a question
            # nobody asked.
            self._say_once(
                seat, f"collect-alone:{seat.name}",
                f"{seat.name}'s Collect step is not advancing and it is "
                f"the only wizard in its zone — that is a regroup, not a "
                f"crowded realm, so no realm change",
                kind="realm-hop-refused",
                detail=(f"{seat.goal!r} has not advanced, but this wizard "
                        f"is alone in its zone. A realm change keeps the "
                        f"zone and only changes the shard, so it cannot "
                        f"help a wizard that is not at the spawns"))
            return
        idle = ((now - seat.progress_at) if away is not None
                else (now - seat.goal_at)) / 60.0
        where = (f"{away:,.0f} from its marker, with nothing moving"
                 if away is not None else
                 f"and its count HAS moved before, so this wizard does "
                 f"reach the spawns — they are just being taken")
        await self._realm_hop_party(
            seat, f"{seat.goal!r} has not advanced in {idle:.0f} min — "
                  f"{where}. Collectibles are shared ground spawns, so a "
                  f"count that will not move is a crowded realm rather "
                  f"than a wedged wizard, and a quieter shard is the move")

    def _with_the_party(self, seat):
        """Is this wizard in a zone at least one other wizard is in?

        The cheapest "am I where I should be" there is, and it needs no
        table of zone names: the script put the party somewhere, and a
        wizard standing with the party is standing where the script
        aimed. A party of one is always with itself, and a seat whose
        zone will not read is not called adrift on a failed read.
        """
        mine = (seat.progress or (None,))[0]
        if not mine:
            return True
        others = [(s.progress or (None,))[0]
                  for s in self.seats if s is not seat and s.client is not None]
        others = [z for z in others if z]
        return not others or mine in others

    async def _realm_hop_party(self, seat, why):
        """Move EVERY hooked wizard to one quiet realm, and say so.

        The whole party or nobody, attempted in order and reported per
        wizard: a party split across realms cannot see or teleport to
        each other, which is a worse state than any crowded realm -- so
        a partial hop is the loudest line this method can produce.
        """
        import time

        from .. import realms

        now = time.monotonic()
        if now - self._realm_hopped_at < self.REALM_HOP_COOLDOWN:
            since = now - self._realm_hopped_at
            if since > 30.0:
                # A human pressing the button again, not the broadcast
                # queue's other copies of one press -- those arrive in
                # seconds. A press that does nothing silently reads as
                # a key that is not bound.
                self._say(seat, f"changed realms {since:.0f}s ago — "
                                f"waiting out the "
                                f"{self.REALM_HOP_COOLDOWN:.0f}s cooldown")
            return
        live = [s for s in self.seats if s.client is not None]
        if not live:
            return
        if any(s.in_duel for s in live):
            self._say(seat, "not changing realms while somebody is in a duel")
            return
        # Claimed before the first await, so the broadcast request queue
        # and a concurrent auto-trigger collapse to one hop.
        self._realm_hopped_at = now
        # Nothing may steer wizards whose spellbooks are open, for as
        # long as the WORST case runs: a scan plus three hops with
        # retries and 45s load waits. This must outlast the stage
        # deadline below, not the other way around.
        self._hop_pause_until = now + 420.0
        landed = []
        try:
            listed, why_not = await realms.scan_realms(seat.client)
            quiet = realms.perfect(listed)
            fresh = [r for r in quiet if r["name"] not in self._realms_tried]
            if not fresh and quiet:
                # Every quiet realm has been tried this run; start over
                # rather than never hopping again.
                self._realms_tried.clear()
                fresh = quiet
            if not fresh:
                self._say(seat, "no Perfect-population realm is listed to "
                                "hop to" + (f" — {why_not}" if why_not else ""))
                for other in live:
                    other.tel.note_questing(
                        "realm-hop-refused",
                        "wanted a quieter realm and the list offered none"
                        + (f" — {why_not}" if why_not else ""))
                return
            target = fresh[0]
            # Can EVERY wizard read the list before ANY of them moves?
            #
            # A split is the worst state this method can produce --
            # wizards in different realms cannot see or teleport to each
            # other -- and rev 3cbb6091 produced one for a reason that
            # was knowable in advance: Sebastian's realm list would not
            # read ("the realm list's page number would not read"), the
            # same failure that had already been logged five times that
            # run on that client. Two wizards hopped, he could not, and
            # the party spent the rest of the run apart.
            #
            # The scan is what the hop needs anyway, and a client that
            # cannot scan cannot hop. Checking first turns a split into
            # a refusal, which costs a realm change and nothing else.
            cannot = []
            for one in live:
                if one is seat:
                    continue                 # its scan is the one above
                seen, why_not_one = await realms.scan_realms(one.client)
                if not seen:
                    cannot.append((one, why_not_one or "the realm list "
                                                       "would not read"))
            if cannot:
                names = " and ".join(one.name for one, _r in cannot)
                said = (f"not changing realms — {names} cannot read the "
                        f"realm list ({cannot[0][1]}), and hopping the "
                        f"others without {names} would split the party "
                        f"across realms, which is worse than the crowding "
                        f"this was answering")
                for other in live:
                    other.tel.note_questing("realm-hop-refused", said)
                self._say(seat, said)
                return
            self._realms_tried.add(target["name"])
            self._say(seat, f"changing the party to realm {target['name']} "
                            f"— {why}")
            stranded = []
            for one in live:
                ok, reason = await realms.hop_to_realm(
                    one.client, target["page"], target["slot"],
                    expect_name=target["name"])
                if not ok:
                    # Once more before calling it stranded -- the first
                    # failure is usually the book or a click, not the
                    # realm.
                    ok, reason = await realms.hop_to_realm(
                        one.client, target["page"], target["slot"],
                        expect_name=target["name"])
                one.tel.note_questing(
                    "realm-hop",
                    f"to {target['name']} — {why}" if ok else
                    f"to {target['name']} FAILED: {reason}")
                if ok:
                    landed.append(one)
                else:
                    stranded.append((one, reason))
            if stranded and not landed:
                # NOBODY moved, so the party is still together -- this
                # is not a split, and calling it one sends the operator
                # chasing a state that does not exist. Either every Go
                # click died, or the party is ALREADY in the picked
                # realm: the scan cannot read which realm the party is
                # in, and a hop to your own realm shows no loading
                # screen, which is exactly the receipt that failed. The
                # realm stays on the tried list either way, so the next
                # attempt picks a different one.
                said = (f"nobody hopped — every wizard's change to "
                        f"{target['name']} reported '{stranded[0][1]}'. "
                        f"The party is still together, and may already BE "
                        f"on {target['name']}; the next attempt will pick "
                        f"a different realm")
                for other in live:
                    other.tel.note_questing("realm-hop-failed", said)
                self._say(seat, said)
            elif stranded:
                names = " and ".join(s.name for s, _r in stranded)
                said = (f"PARTY SPLIT ACROSS REALMS — {names} could not "
                        f"follow to {target['name']} "
                        f"({stranded[0][1]}). Wizards in different realms "
                        f"cannot see or teleport to each other; press "
                        f"Realm hop again or move {names} by hand")
                for other in live:
                    other.tel.note_questing("realm-split", said)
                self._say(stranded[0][0], said)
            else:
                self._say(seat, f"the whole party is on {target['name']}")
                for one in live:
                    # The crowded-judgement clock starts over: the new
                    # realm earns its own REALM_HOP_AFTER before being
                    # called contested, not the old realm's leftovers --
                    # and the zone genuinely did reload, so a fresh
                    # progress stamp is the truth, not a fudge.
                    one.progress_at = time.monotonic()
                    one.cells_seen.clear()
        except asyncio.CancelledError:
            # The stage deadline cut the maneuver mid-party. Without
            # this, the seats already hopped are in the new realm, the
            # rest never attempted, the split report below the loop
            # never runs, and the operator's retry press is refused by
            # a cooldown stamped for a hop that half-happened. Say
            # exactly where everybody is (sync calls only -- this
            # coroutine is being cancelled), and open the retry window.
            went = ", ".join(s.name for s in landed) or "nobody"
            left = [s.name for s in live if s not in landed]
            said = (f"the realm change was CUT OFF mid-party — {went} "
                    f"hopped and {', '.join(left) if left else 'nobody'} "
                    f"did not. If anyone moved, the party is split across "
                    f"realms; press Realm hop again in a minute")
            for other in live:
                try:
                    other.tel.note_questing("realm-cut-off", said)
                except Exception:
                    pass
            self._say(seat, said)
            self._realm_hopped_at = (time.monotonic()
                                     - self.REALM_HOP_COOLDOWN + 60.0)
            raise
        finally:
            self._hop_pause_until = time.monotonic() + self.HOP_SETTLE

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

    def _place_by_name(self, seat):
        """`position_of` the tracked name — unless the goal disowns it.

        One read for `_places` and the Questing tab both, so the number
        that starts a catch-up and the number on screen cannot disagree.

        The disowning is the fix for rev 30e83468. `_read_goal` keeps
        the previous quest name on a blank read, so the name can lag
        the goal by a quest -- and a stale name that still places wins
        over a fresh goal that was never consulted, which is how
        Sebastian stood on the same goal line as Konstantin and was
        held two quests behind him. A name the goal line disowns places
        nothing; the goal fallback in `_places` takes over from there.
        """
        from .. import questlist

        if not seat.quest_name:
            return questlist.Position()
        place = questlist.position_of(seat.quest_name)
        if (place.comparable and seat.goal
                and questlist.goal_disowns(seat.quest_name, seat.goal)):
            return questlist.Position(
                how=(f"the name read {seat.quest_name!r} (#{place.order}), "
                     f"but the goal line belongs to a different quest — a "
                     f"stale read places nothing"))
        return place

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

        places = [self._place_by_name(s) for s in self.seats]
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
        # A wizard nothing could place, showing the SAME goal line,
        # character for character, as a wizard that WAS placed, is on
        # that wizard's quest -- text identity is evidence when there
        # is no other evidence. Rev 30e83468's Sebastian read exactly
        # Konstantin's goal while a lagging name read held him two
        # quests back. Only the unplaceable are placed this way: two
        # wizards placed at DIFFERENT quests sharing one goal line is
        # a real thing (Krokotopia #12 and #13 both track "Talk To
        # Lieutenant Standish"), and papering over a placement with
        # text would erase a laggard the questline can actually see.
        shared = {}
        for i, seat in enumerate(self.seats):
            if seat.goal:
                shared.setdefault(seat.goal, []).append(i)
        for members in shared.values():
            placed = [places[i] for i in members if places[i].comparable]
            if len(members) < 2 or not placed:
                continue
            best = max(placed, key=lambda p: p.order)
            for i in members:
                if places[i].comparable:
                    continue
                places[i] = questlist.Position(
                    world=best.world, order=best.order, name=best.name,
                    area=best.area, questline=best.questline,
                    how=(f"the same goal line as a wizard placed at "
                         f"#{best.order}"))
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
            if ((self._solo_pilot() or self.booster_party)
                    and seat.index != self.leader):
                # A follower's tracker drifting is expected -- nothing
                # is questing it. The PILOT is still checked: it is the
                # one wizard whose lost questline stalls the run, and
                # the preset's own Auto_Find_Quest only kicks in after
                # several full loops. Boosters get the same exemption
                # for the same reason, and more so: their recovery
                # rungs are off by design, so a warning here would
                # nag about a state nothing is ever going to cure.
                continue
            if place.on_main:
                # The last place on the line this wizard actually held,
                # which is the first answer to "which quest was lost".
                # Recorded here rather than in the poll because this is
                # where a `Position` already exists, and only a
                # main-line one is worth keeping.
                seat.last_main = (place.world, place.order, place.name)
                seat.last_main_at = now
            # The one unknown name that IS evidence: "Quest Finder", the
            # journal's own pseudo-entry. Not a failed read -- the
            # journal affirmatively saying no quest is selected, which
            # the script's own lost-quest routine leaves behind when a
            # cycle fails partway. Rev f2b8101f: Sebastian's tracker sat
            # on it for the last 25 minutes of the run and every rung
            # looked past him, precisely because unknown is skipped
            # below. It gets the off-line clock, and the recovery rung
            # takes it from there.
            none_selected = questlist.no_quest_selected(seat.quest_name)
            if place.on_main or (not place.known and not none_selected):
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
                # A wizard back on the line has no give-up to honour.
                # The next loss is a fresh one, and it may be a quest
                # the book DOES list.
                seat.recover_gave_up = ""
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
            if none_selected:
                said = (f"{seat.name} has had NO quest selected for "
                        f"{away / 60:.0f} min — its journal is on the "
                        f"Quest Finder pseudo-entry, which is what the "
                        f"script's own lost-quest routine leaves behind "
                        f"when a cycle fails partway"
                        + (f", while the party is on {theirs}" if theirs
                           else "")
                        + ". Every `tp quest` has nothing to aim at until "
                          "a real quest is selected again")
            else:
                said = (f"{seat.name} has been off the main questline for "
                        f"{away / 60:.0f} min — its tracker is on "
                        f"{place.name!r}, which is a side quest"
                        + (f", while the party is on {theirs}" if theirs
                           else "")
                        + ". Every `tp quest` goes to the side quest until "
                          "the main one is selected again")
            for other in self.seats:
                try:
                    other.tel.note_questing("off-questline", said)
                except Exception:
                    pass
            self._say(seat, said)

    #: how long the script's instruction pointer may sit still, with
    #: nothing legitimately holding it, before the VM is written off and
    #: reloaded.
    #:
    #: Five minutes is far longer than any single instruction is allowed
    #: to take -- `ScriptRunner.STEP_LIMIT` cuts one off at 180s -- so
    #: reaching this means `step()` is not being CALLED, which is a
    #: different failure and the one nothing could see. Rev f32be436 sat
    #: in it for 110 minutes.
    SCRIPT_STALL_AFTER = 300.0

    def _check_script_alive(self):
        """Notice a script that has stopped executing at all.

        Every other watchdog in here asks whether the script is doing
        the RIGHT thing. This one asks whether it is doing anything, and
        it exists because the answer turned out to be unknowable from
        the inside: `ScriptRunner.STEP_LIMIT` only fires for an
        instruction that is running long, and `_unstick` reads a frozen
        step count as "parked", which is the one state
        `_maybe_restart_script` excuses itself from -- on the grounds
        that the stuck-instruction reload owns it. Nothing owned it.

        Run from EVERY seat's loop, deliberately. The seat that steps the
        VM is the seat that can stop ticking, so a check that lived on
        the driving seat would go quiet with it.

        The clock only runs while nothing legitimately holds the script:
        a catch-up pauses it by design, a duel and the after-fight chores
        skip the tick that steps it, and the steering hold stops
        everything for a few seconds after a teleport. None of those is a
        stall, and counting them as one would reload a healthy script
        every five minutes.
        """
        import time

        runner = self.seats[0].runner
        if runner is None or not self.script:
            return
        now = time.monotonic()
        steps = getattr(runner, "steps", None)
        held = (bool(self._catching_up())
                or self._hop_held()
                or all(s.in_duel or s.in_upkeep for s in self.seats
                       if s.client is not None))
        if steps != self._steps_seen or held or runner.stale:
            self._steps_seen = steps
            self._steps_at = now
            return
        if now - self._steps_at < self.SCRIPT_STALL_AFTER:
            return
        stalled = (now - self._steps_at) / 60.0
        self._steps_at = now
        said = (f"the script has not executed an instruction in "
                f"{stalled:.0f} min and nothing is holding it — it is "
                f"parked at {steps:,} with no wizard in a fight, no "
                f"catch-up running and no teleport settling. An "
                f"instruction that merely runs long is cut off after "
                f"three minutes, so this is the VM not being stepped at "
                f"all. Reloading it")
        for other in self.seats:
            try:
                other.tel.note_questing("script-stalled", said)
            except Exception:
                pass
        self._say(self.seats[0], said)
        # Hand it to the reload path rather than restarting from here:
        # that path already owns the backoff, the "could not be
        # reloaded" case and the setup-skipping restart source.
        runner.stale = True
        runner.stale_sig = f"stalled at {steps}"
        runner.last_error = said

    #: the most quests one recovery will offer the book. Each one costs
    #: a pass over four visible entries, and a wizard more than a
    #: handful of steps behind is a catch-up's problem rather than a
    #: mis-selected tracker's.
    SPAN_CANDIDATES = 6

    def _lost_quest(self, seat, places=None):
        """([(quest, how it was named)], why) for a wizard off the line.

        The middle third of the operator's question -- "can we tell if
        the main quest has been lost, and WHICH ONE". Two answers, best
        first, and neither is enough on its own:

        -- the wizard's OWN last main-line quest. The strongest
           evidence available: it was on #13 four minutes ago and it is
           on a side quest now, so #13 is what it left. It is wrong in
           exactly one way -- the wizard FINISHED #13 while the side
           quest was selected -- and that case is self-announcing,
           because a finished quest is not in the book to click.
        -- the quest the rest of the party is on. A party quests
           together, so the odd wizard belongs within a step of the
           others, and `questlist.quest_at` turns their "#13" into a
           name. This is the answer that exists even for a wizard that
           was never seen on the line at all.

        Ordered rather than chosen, because the book session that acts
        on this reads its entries once and can try both for nothing.
        """
        from .. import questlist

        if places is None:
            places = self._places()
        found = []
        seen = set()

        def add(place, source):
            if not place.name or place.name.lower() in seen:
                return
            seen.add(place.name.lower())
            found.append((place, source))

        mine = seat.last_main
        if mine and mine[2]:
            # By PLACE, not by name. Seven main-line names are reused
            # across worlds -- "The Right Combination" is Krokotopia
            # #55 and Marleybone #39 -- and a by-name lookup answers
            # with whichever came first in the data, which would hand
            # the wrong world's area to `_saved_place`. The place this
            # wizard was actually at has no such ambiguity.
            here = questlist.quest_at(mine[0], mine[1])
            add(here if here.name else questlist.Position(
                    world=mine[0], order=mine[1], name=mine[2],
                    how="the last main-line quest this wizard held",
                    questline="main"),
                f"its own last main-line quest, {mine[0]} #{mine[1]}")
        others = [p for s, p in zip(self.seats, places)
                  if s is not seat and p.on_main
                  and not self._is_booster(s)]
        if others:
            lowest = min(others, key=lambda p: p.order)
            add(questlist.quest_at(lowest.world, lowest.order),
                f"where the rest of the party is, #{lowest.order}")
            # ...and every step BETWEEN the two, in order.
            #
            # Rev f32be436 is why. Konstantin's tracker wandered onto
            # 'Trophy Quest' while he was two quests behind the party,
            # and both answers above missed for the same reason: his own
            # last main was finished and out of the book, and the
            # party's #46 was two quests further on than anything he
            # could have been given. The quest he was actually holding
            # was one of the ones in between, and nothing offered it.
            #
            # Cheap, because the book session reads its entries once and
            # tries each candidate against what it already has. Bounded
            # at `SPAN_CANDIDATES` so a wizard a whole world behind does
            # not turn one recovery into a page-through.
            if mine and mine[0] == lowest.world and mine[1] is not None:
                step = 1 if lowest.order >= mine[1] else -1
                for order in range(mine[1] + step, lowest.order, step):
                    if len(found) >= self.SPAN_CANDIDATES:
                        break
                    add(questlist.quest_at(lowest.world, order),
                        f"a step between the two, #{order}")
        if not found:
            return [], ("nothing names a main-line quest for this wizard: "
                        "it has not been seen on the line, and no other "
                        "wizard is on it either")
        return found, "; ".join(source for _p, source in found)

    def _saved_place(self, name, now=None, area=""):
        """Where a quest was last worked, as a phrase, or "".

        The last third of the operator's question -- "and the saved tp
        location / zone". Two sources, and the order matters: a zone
        THIS party actually stood in while tracking the quest beats the
        list's own area, because the list carries the wiki's prose
        ("Palace of Fire") and the zone carries the id the game and
        every teleport in the codebase speak ("KT_PalaceFire").

        `area` is passed in by callers that already hold the quest's
        `Position`, because seven main-line names are reused across
        worlds and looking one up by name here would sometimes answer
        with the other world's area.
        """
        import time

        from .. import questlist

        if not name:
            return ""
        now = time.monotonic() if now is None else now
        where = self._quest_zone.get(questlist.key_for(name))
        if where:
            return (f"the party was in {where[0]} on it "
                    f"{(now - where[1]) / 60:.0f} min ago")
        area = area or questlist.position_of(name).area
        if area:
            return f"the quest list puts it in {area}"
        return ""

    #: how long a wizard may sit off the main line, with the rest of
    #: the party still ON it, before its tracker is put back. Longer
    #: than `OFF_QUESTLINE_AFTER` on purpose: saying it costs nothing
    #: and can be said early, while doing something about it
    #: interrupts a side quest that may have been picked up on purpose.
    RECOVER_QUESTLINE_AFTER = 300.0
    #: the same, with no party-mate on the line to compare against --
    #: a solo run, or every wizard off it at once. Twenty minutes,
    #: because "all three are on side quests" is what a preset running
    #: a side-quest chain for training points looks like, and cutting
    #: that short would be wizAi fighting the script it is driving.
    RECOVER_ALONE_AFTER = 1200.0
    #: between attempts, per wizard. The quest book costs ~15 held
    #: seconds and the failure this cures is measured in tens of
    #: minutes, so there is nothing to gain by hurrying it.
    RECOVER_EVERY = 600.0
    #: the worker-wide steering hold around a recovery and the settle
    #: after it, the same shape as the re-arm's and for the same
    #: reason: the book is open for seconds, and any teleport another
    #: task lands mid-click turns the click into a misfire.
    RECOVER_CEILING = 60.0
    RECOVER_SETTLE = 3.0
    #: how long a "not among the visible entries" give-up holds before
    #: the same candidate list is worth one more look. Half an hour: a
    #: retry costs ~15 held seconds of paging the book, and the two
    #: states it recovers from -- the quest sitting on a page the read
    #: cannot reach, and the party accepting it since -- both persist
    #: far longer than a cooldown.
    RECOVER_GIVEUP_TTL = 1800.0

    async def _maybe_recover_questline(self, seat):
        """Put a lost tracker back onto the main questline.

        `_check_on_questline` has been able to SAY this since it was
        written -- "its tracker is on a side quest, and every `tp
        quest` goes there until the main one is selected again" -- and
        that is where it stopped. A diagnosis with no cure attached is
        the shape of every failure this session has had to come back
        to, and the operator named the missing half exactly:

            if we know which quest other wizards are supposed to be on
            in order, and you know for example the laggard can we tell
            if the main quest has been lost, and which one, and the
            saved tp location / zone can we have a lost quest recovery
            bit

        All three pieces existed and none of them were wired to an
        action. `questlist` orders the line, `_lost_quest` names what
        the wizard should be on, `_quest_zone` remembers where it was
        being worked, and the quest book -- already opened and clicked
        by `rearm_quest_arrow` -- is the one place a quest can be
        re-selected. This joins them.

        It does NOT move the wizard. Selection is the whole cure: the
        moment the right quest is tracked, the marker points at the
        main line again and every mover already in the ladder -- the
        script's own `tp quest`, the desperate hop, the catch-up, the
        rejoin -- aims correctly without being told. Adding a teleport
        to a remembered zone here would be a new way to split the
        party to fix a problem that no longer exists by then.
        """
        import time

        from .. import questing, questlist

        if not questlist.loaded():
            return
        # Cheap gates before the placement pass: this runs on every
        # service tick for every wizard, and on a healthy one it must
        # cost a couple of comparisons.
        if seat.client is None or seat.in_duel or seat.off_line_since is None:
            return
        now = time.monotonic()
        if now - seat.recover_tried_at < self.RECOVER_EVERY:
            return
        away = now - seat.off_line_since
        if away < self.RECOVER_QUESTLINE_AFTER:
            return
        places = self._places()
        place = dict(zip((id(s) for s in self.seats), places)).get(id(seat))
        if place is None or place.on_main:
            return
        if not place.known and not questlist.no_quest_selected(seat.quest_name):
            # Unknown is a read that failed, and selecting a quest on
            # that evidence would fire on every loading screen. The one
            # exception is the journal's Quest Finder pseudo-entry,
            # which is not a failed read -- it is the journal saying
            # nothing is selected, and putting a real quest back is
            # exactly what this rung is for. Rev f2b8101f: Sebastian
            # spent the last 25 minutes of the run in that state while
            # this line skipped him.
            return
        # Boosters are not evidence about the questline, in either
        # direction: a booster's journal is a max-level wizard's,
        # parked on whatever world it stopped in, so it reading
        # on-main is a coincidence and it reading off-main says
        # nothing. Counting it made every booster party "alone" here
        # -- and the alone wait below became the cure's ceiling.
        alone = not any(p.on_main for s, p in zip(self.seats, places)
                        if s is not seat and not self._is_booster(s))
        if alone and away < self.RECOVER_ALONE_AFTER:
            # A whole party off the line is usually the script running
            # a side chain on purpose -- and a side chain being WORKED
            # moves: goals turn in, zones change, fights start. One
            # that has moved NOTHING for the full recovery deadline is
            # not being worked by anybody. Rev d3ed4d3c's ending is
            # the case in full: the tracker fell onto 'Blue Oyster
            # Cult', the script spun ~4,000 instructions a minute at
            # an oyster it has no route for, goal and zone sat frozen
            # for nine straight minutes -- and the twenty-minute alone
            # wait outlived the run, with the booster party reading as
            # "alone" by the old rule on top.
            stalled = (bool(seat.goal_at) and bool(seat.zone_since)
                       and now - max(seat.goal_at, seat.zone_since)
                       >= self.RECOVER_QUESTLINE_AFTER)
            if not stalled:
                self._say_once(
                    seat, f"recover-wait:{seat.off_line_since:.0f}",
                    f"{seat.name} has been off the main questline for "
                    f"{away / 60:.0f} min with nobody on the line to "
                    f"compare against — waiting "
                    f"{self.RECOVER_ALONE_AFTER / 60:.0f} min in case "
                    f"the side quest is deliberate. A stall (no goal or "
                    f"zone change for "
                    f"{self.RECOVER_QUESTLINE_AFTER / 60:.0f} min) "
                    f"skips the wait",
                    kind="questline-recovery-waiting",
                    detail=(f"off the line {away / 60:.0f} min on "
                            f"{place.name!r}, nobody on the main line to "
                            f"compare against, and the side quest still "
                            f"shows signs of being worked — holding the "
                            f"cure back on purpose"))
                return
        candidates, why = self._lost_quest(seat, places)
        if not candidates:
            self._say_once(
                seat, f"recover-blind:{seat.off_line_since:.0f}",
                f"{seat.name} is off the main questline and nothing can "
                f"name the quest it should be on — {why}",
                kind="questline-recovery-blind", detail=why)
            return
        # A give-up holds for the same candidate list -- retrying it
        # every ten minutes would page through the journal forever to
        # learn the same thing -- but it EXPIRES, because "none of them
        # is among the visible entries" is evidence and not proof. The
        # book shows four quests per page and `select_quest` cannot
        # turn pages, so the quest may sit on page two the whole time;
        # and a party that turns in the previous step makes the quest
        # accepted where it was not. The 115-min run at rev f2b8101f
        # wrote Phönix off at t=4536 on exactly that reading and never
        # looked again until the candidates happened to change. A NEW
        # candidate list is still a new question and skips the wait.
        key = " | ".join(p.name for p, _s in candidates)
        if seat.recover_gave_up == key \
                and now - seat.recover_gave_up_at < self.RECOVER_GIVEUP_TTL:
            return
        if await questing.in_dialogue(seat.client):
            return

        seat.recover_tried_at = now
        target, source = candidates[0]
        told = self._saved_place(target.name, now, target.area)
        self._say(seat,
                  f"{seat.name} has been off the main questline for "
                  f"{away / 60:.0f} min on {place.name!r} — putting the "
                  f"tracker back on {target.describe()} ({source})"
                  + (f"; {told}" if told else ""))
        self._hop_pause_until = now + self.RECOVER_CEILING
        try:
            ok, why, in_book = await questing.select_quest(
                seat.client, [p.name for p, _s in candidates],
                on_status=lambda m: self._say(seat, m))
        finally:
            self._hop_pause_until = time.monotonic() + self.RECOVER_SETTLE

        if ok:
            # `why` is the candidate that took -- which of the two
            # answers was right is worth having in the export.
            took = next((p for p, _s in candidates if p.name == why), target)
            told = self._saved_place(why, now, took.area)
            seat.recover_gave_up = ""
            seat.tel.note_questing(
                "questline-recovered",
                f"{seat.name} had been off the main questline for "
                f"{away / 60:.0f} min with its tracker on {place.name!r}. "
                f"Selected {why!r} in the quest book and the tracker now "
                f"reads it, so every `tp quest` — the script's and "
                f"wizAi's — aims at the main line again ({source})"
                + (f". {told[0].upper()}{told[1:]}" if told else ""))
            self._say(seat, f"back on the main questline — the tracker "
                            f"reads {why!r} again")
            return
        if in_book is None:
            # The book was never looked in, because no candidate could
            # be matched safely -- 69 of the 2,110 main-line quests are
            # named in one word, and one word picks whatever entry
            # happens to contain it. Named for the operator and said
            # ONCE, because the answer will not change while the
            # candidates do not.
            seat.recover_gave_up = key
            seat.recover_gave_up_at = now
            seat.tel.note_questing(
                "questline-recovery-unsafe",
                f"{seat.name} has been off the main questline for "
                f"{away / 60:.0f} min and should be on "
                f"{target.describe()} ({source})"
                + (f" — {told}" if told else "")
                + f", but it cannot be selected automatically: {why}. "
                  f"Selecting it by hand in the quest book puts this "
                  f"wizard back on the line")
            self._say(seat, f"cannot select {target.name!r} safely: {why}")
            return
        if not in_book:
            # The one failure retrying cannot fix, and a different fact
            # worth its own entry: the quest is not merely unselected,
            # it was never accepted. Nothing in the journal can take
            # it; somebody has to walk to the NPC that gives it.
            seat.recover_gave_up = key
            seat.recover_gave_up_at = now
            seat.tel.note_questing(
                "questline-quest-missing",
                f"{seat.name} has been off the main questline for "
                f"{away / 60:.0f} min and the quest it should be on is "
                f"not in its book: {why}"
                + (f" — {told}" if told else "")
                + f". If it was never accepted, only its NPC can fix "
                  f"that — but the book shows four quests per page and "
                  f"the read cannot turn pages, so this is checked "
                  f"again in {self.RECOVER_GIVEUP_TTL / 60:.0f} min "
                  f"rather than never")
            self._say(seat, f"cannot recover the questline: {why}")
            return
        seat.tel.note_questing(
            "questline-recovery-failed",
            f"tried to put {seat.name} back on {target.describe()} and "
            f"could not: {why}. Every `tp quest` on this wizard still "
            f"goes to {place.name!r}. Next attempt in "
            f"{self.RECOVER_EVERY / 60:.0f} min")
        self._say(seat, f"could not put the tracker back on the main "
                        f"questline: {why}")

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

        Not in a booster party. A booster's journal points wherever it
        was left -- diverging from the quester is the DESIGN -- so the
        comparison has nothing to warn about, and at rev dbced750 it
        spent a run reporting the booster "behind" on a quest nobody
        was questing.

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

        Not in solo-pilot mode. There, only the pilot's quest state
        matters and the followers are behind BY DESIGN -- they are
        combat support, kept together by the follow rather than by the
        questline. Reporting that as desync and starting catch-ups for
        it would re-create the exact churn the mode exists to remove.
        """
        import time

        if len(self.seats) < 2 or self._solo_pilot() or self.booster_party:
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
            # Reachability is per wizard, not per group. Rev 30e83468
            # had Sebastian one turn-in from caught up and Phönix's
            # marker a zone away in the same group, and a single
            # verdict over both wastes whichever half it is wrong
            # about: refuse and Sebastian's twenty-second step is held
            # hostage to Phönix's geography, start and the catch-up's
            # one budget burns out on the wizard nothing can drive.
            near = []
            for one in group:
                blind = self._marker_unusable(one)
                if blind:
                    # A catch-up drives the laggard with `hop_once`,
                    # and `hop_once` aims at the quest position. No
                    # position, no catch-up: it would pause the whole
                    # party to attempt nothing, give up at
                    # CATCH_UP_IDLE, and repeat -- six times in one run
                    # at rev 98b4c50c. The REASON decides who owns the
                    # fix, which is why it is quoted rather than
                    # guessed at: a Collect step is the script's
                    # (hardcoded spots, and a crowded realm is the
                    # realm rung's), a dead arrow is the re-arm's.
                    self._say_once(
                        one, f"marker-unusable:{one.name}",
                        f"{one.name} is behind and nothing can aim a "
                        f"teleport for it — {blind}. Not pausing the "
                        f"party for a catch-up that cannot aim",
                        kind="catch-up-refused-no-marker",
                        detail=(f"{one.name} is behind on {one.goal!r} and "
                                f"a catch-up's hop needs a quest position: "
                                f"{blind}"))
                    continue
                away = self._marker_out_of_reach([one])
                if away is None:
                    near.append(one)
                    continue
                # Not a catch-up for this one. Getting there is the
                # script's job and the script is good at it -- rev
                # 1843e387 spent 116s failing to hop Phönix out of
                # KT_Hub and the script did it in eight seconds the
                # moment it had the wheel back.
                self._say_once(
                    one, f"marker-away:{one.name}",
                    f"{one.name} is {self._behind_gap} quest(s) behind "
                    f"and its objective is {away:,.0f} away — another zone. "
                    f"A quest teleport cannot cross one, so this is the "
                    f"script's journey to make, not a step wizAi can "
                    f"finish. Leaving it to the script",
                    kind="catch-up-out-of-zone",
                    detail=(f"{one.name}'s objective is {away:,.0f} away, "
                            f"which is another zone. Not pausing the script "
                            f"for a hop that cannot reach it"))
            if not near:
                return
            group = near
            behind = group[0]
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
            # Where this wizard is standing, kept from poll to poll.
            # Worthless on its own; the poll AFTER a zone change turns
            # the previous sample into the door that was walked
            # through -- see `_learn_door`.
            spot = None
            try:
                spot = await seat.client.body.position()
            except Exception:
                pass
            if zones[seat]:
                if seat.zone_seen and zones[seat] != seat.zone_seen:
                    # Where it has just come FROM, and when. See the
                    # "left there on purpose" test below.
                    seat.zone_left.append((seat.zone_seen, now))
                    del seat.zone_left[:-8]
                    self._learn_door(seat, zones[seat], now)
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
            if spot is not None and zones[seat]:
                seat.last_spot = spot
                seat.last_spot_zone = zones[seat]
        known = [z for z in zones.values() if z]
        if len(known) < len(live):
            return None, None            # a read failed; no evidence

        counts = {}
        for z in known:
            counts[z] = counts.get(z, 0) + 1
        best, n = max(counts.items(), key=lambda kv: kv[1])
        if len(live) == 2 and n == 1:
            # A party of two has no majority to be in. `n < 2` below
            # reads that as "no evidence" and returns, which means this
            # whole mechanism -- the stranding clock, `stranded_where`,
            # the rejoin, and every line of telemetry any of them write
            # -- has never once fired for a two-wizard party. Rev
            # 8ebfcf70 is forty-seven minutes of it: Oz in
            # `DS_Necropolis_Gauntlet_5Room2` and Konstantin four rooms
            # deeper for most of the run, and not one `stranded`,
            # `rejoined`, `rejoin-failed` or `rejoin-skipped` entry in
            # either export. The only thing that noticed was the
            # follow's own alarm, which cannot act.
            #
            # Two wizards in two zones still has an answer, and it is
            # the one the rest of the file already uses: the leader's
            # zone is the party's zone. `_follow_step` follows it,
            # `_walk_the_leaders_door` walks toward it, and in solo-pilot
            # and booster runs the leader IS the route. So the straggler
            # is simply whichever seat is not the leader.
            boss = self.seats[self.leader]
            if boss not in live or zones.get(boss) is None:
                return None, None        # no reference; decide nothing
            best = zones[boss]
        elif n < 2 or n == len(live):
            return None, None            # no majority, or nobody adrift

        odd = [s for s in live if zones[s] != best]
        if len(odd) != 1:
            return None, None            # two adrift is a split, not a
                                         # straggler, and following one
                                         # of them could be the wrong way
        seat = odd[0]

        # Is it even meant to be where they are? This mechanism was
        # built for the wizard whose TELEPORT did not land -- same
        # quest, same destination, one client that missed it. A wizard
        # on a DIFFERENT quest step is not that. Its objective is
        # somewhere else by definition, and the majority is walking a
        # later leg of the route; pulling it along destroys the one
        # thing it needs, which is to be where its own step happens.
        #
        # It is also the operator's own correction, applied to the
        # other mechanism that moves wizards: "Just teleporting back to
        # a wizard you believe is behind doesn't work because I think
        # the script then just continues on the same way. They need to
        # actually quest with the other wizard until they catch up."
        # Moving a body never advances a quest -- that is why
        # `_start_catching_up` drives the laggard through its OWN step
        # rather than dragging it, and this rung must not undo that.
        #
        # Rev 116b5866 is the cost. Konstantin was on #31 `Collect
        # Gemstones in Hall of Champions`, whose spots the quester only
        # visits `if p1 inzone KT_ChampHall`; the other two were on #32
        # and being walked elsewhere. This pulled him to the majority
        # twice in a minute -- into KT_AltarOfKings, then KT_Hub --
        # and the party finished at the world portal with his step
        # untouched.
        #
        # ...unless its own step is not reachable from where it stands
        # either. Rev cfeb9a85 is the cost of the rule without that
        # exception: Konstantin spent FORTY-TWO minutes alone inside
        # `KT_ChampHall_T3`, a dungeon instance the party had left, on
        # `Defeat Odji Sokkwi in Hall of Champions` -- whose marker read
        # 100,242 away, another zone. Every rung declined, and each was
        # right on its own terms: the catch-up refused (no teleport
        # crosses a zone), the desperate hop refused (same), the script
        # restart fired and changed nothing, and this refused because
        # his step differed. Nothing in the program could move him.
        #
        # The refusal protects a wizard whose objective is HERE. A
        # wizard whose objective is provably in another zone has nothing
        # to protect: it cannot finish its step from this spot however
        # long it stands there, and joining the party at least puts it
        # where the script is operating -- and out of a dead instance.
        mine = (seat.goal or "").strip()
        theirs = {(s.goal or "").strip()
                  for s in live if s is not seat and (s.goal or "").strip()}
        away = seat.marker_away
        adrift = away is not None and away > self.MARKER_IN_ZONE
        if mine and theirs and mine not in theirs and not adrift:
            self._say_once(
                seat, f"different-step:{seat.name}",
                f"{seat.name} is in {zones[seat]} and the others are in "
                f"{best}, but it is on a different quest step "
                f"({mine!r}) — its objective is not where they are, so "
                f"pulling it along would take it away from the only "
                f"place its step can finish",
                kind="rejoin-refused",
                detail=(f"{seat.name} is on {mine!r} while the party is on "
                        f"{' / '.join(sorted(theirs))!r}. A regroup is for "
                        f"a teleport that did not land, not for a wizard "
                        f"whose own step is elsewhere — moving a body "
                        f"never advanced a quest"))
            seat.stranded_since = None
            return None, None
        if mine and theirs and mine not in theirs and adrift:
            # Fetched anyway, and said so: this is the exception above,
            # and it should be legible in the export rather than look
            # like the refusal failing to fire.
            self._say_once(
                seat, f"stranded-adrift:{seat.name}",
                f"{seat.name} is on a different step AND its objective is "
                f"{away:,.0f} away — another zone — so it cannot finish "
                f"that step from {zones[seat]} however long it waits. "
                f"Bringing it back to the party, which is the only move "
                f"left that crosses a zone",
                kind="rejoin-adrift",
                detail=(f"{seat.name} is on {mine!r} with its marker "
                        f"{away:,.0f} away. Every rung that could move it "
                        f"refuses an out-of-zone objective, so standing "
                        f"still is not protecting anything"))

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
    #: ...and for how long before wizAi stops narrating and drags the
    #: odd wizards to the majority's sigil itself. The gap between this
    #: and `SPLIT_AFTER` is the script's window: a friend-teleport
    #: regroup that is coming lands well inside it, and one that has
    #: not come by now is not coming — rev 1dcf4193 waited twenty
    #: minutes for it.
    SIGIL_ACT = 90.0

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

        Reported at `SPLIT_AFTER`, and now FIXED at `SIGIL_ACT`: the
        odd wizards are teleported to the majority's sigil. The first
        version only reported, reasoning that the script's own friend
        teleport was about to fix it and dragging wizards would fight
        it for the wheel -- and rev 1dcf4193 is twenty minutes of that
        teleport not coming. The report still gets the first minute
        and a half to itself, so a script regroup that IS coming has
        room to land; past that, the same within-zone body write the
        VM's own `teleport client N` uses puts the party on one sigil,
        once per wizard per episode.
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

        # The fix, after the report has stood long enough for the
        # script's own regroup to have come if it was coming.
        if held < self.SIGIL_ACT:
            return
        for seat in odd:
            if now - seat.sigil_moved_at < self.SIGIL_ACT:
                continue                 # one drag per wizard per episode
            seat.sigil_moved_at = now
            try:
                await seat.client.teleport(where[best])
                moved = (f"teleported {seat.name} to the sigil "
                         f"{with_them} {'are' if len(near) > 1 else 'is'} "
                         f"standing on — {held:.0f}s at the wrong one and "
                         f"the script's own regroup never came")
                self._say(seat, moved)
                for other in live:
                    try:
                        other.tel.note_questing("sigil-regroup", moved)
                    except Exception:
                        pass
            except Exception as exc:
                seat.tel.note_questing(
                    "sigil-regroup",
                    f"tried to teleport {seat.name} to the party's sigil "
                    f"and could not — {type(exc).__name__}: {exc}")

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
        # A placement that keeps changing its mind cannot support
        # pausing the party. See the churn counter in
        # `_check_caught_up`: three answers in a minute is not three
        # laggards, it is one unstable reading, and each restart costs
        # the script another pause.
        if (self._churn >= self.CATCH_UP_CHURN
                and now - self._churn_at < self.CATCH_UP_CHURN_REST):
            self._say_once(
                seats[0], "catch-up-churn",
                f"the answer to 'who is behind' has changed "
                f"{self._churn} times in a row — the party is spread "
                f"across different content and the placement cannot "
                f"settle. Letting the script run rather than pausing it "
                f"again for a reading that will change",
                kind="catch-up-churn",
                detail=(f"{self._churn} catch-ups in a row ended because a "
                        f"DIFFERENT wizard was behind. Resting for "
                        f"{self.CATCH_UP_CHURN_REST / 60:.0f} min"))
            return
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
    #: how many times running the "who is behind" answer may change
    #: before catch-ups rest, and the window that counts as "running".
    #: Three inside a minute at rev cfeb9a85, each pausing the script.
    CATCH_UP_CHURN = 3
    CATCH_UP_CHURN_WINDOW = 90.0
    #: ...and how long they rest for. Long enough that the script gets
    #: a real run at the problem, short enough that a party which
    #: genuinely settles is caught up before it drifts further.
    CATCH_UP_CHURN_REST = 300.0

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

    def _forget_write_off(self, seat):
        """Arriving somewhere new earns the step another attempt.

        A write-off says "wizAi's questing cannot finish this step". At
        rev 1843e387 that was true and beside the point: the catch-up
        ran while Phönix was in KT_Hub and its objective was in
        KT_PalaceOfFire, where no within-zone teleport could reach it.
        It failed, was written off permanently -- and then the SCRIPT
        moved the whole party into KT_PalaceOfFire eight seconds later,
        which is the one place the catch-up would have worked.

        The verdict was about a situation, not about the step, and the
        situation changed. So a zone change clears it.
        """
        wrote = self._wrote_off.pop(id(seat), None)
        if wrote is not None:
            self._said_written_off.discard(f"wrote-off:{wrote[0]}")

    #: How far a quest marker can be and still be somewhere a within-zone
    #: teleport can reach. Inside a zone the marker is hundreds to a few
    #: thousand units off; a marker in ANOTHER zone is read in that
    #: zone's coordinate space and comes back as six figures -- rev
    #: 1843e387 logged 98,813, 110,890 and 115,018 for three wizards
    #: whose objectives were all elsewhere. Twenty thousand is far above
    #: any real within-zone distance and far below those.
    MARKER_IN_ZONE = 20000.0

    def _marker_out_of_reach(self, seats):
        """Is the laggard's objective somewhere a hop cannot go?

        `questing.hop_once` teleports to the quest marker and presses X.
        That is a within-zone move: Wizard101's quest teleport does not
        cross a zone boundary, and the script -- which knows the route,
        the sigil and the door -- is the only thing here that can.

        So a catch-up aimed across a zone boundary is not a hard case,
        it is an impossible one, and starting it spends the party's one
        attempt in the one place it cannot succeed.
        """
        for seat in seats:
            away = getattr(seat, "marker_away", None)
            if away is not None and away > self.MARKER_IN_ZONE:
                return away
            # ...and the same thing said the other way. A marker for a
            # goal in another WORLD is out of reach at any distance:
            # read in that world's coordinate space it can come back as
            # any number at all, and rev 09a0af80's came back as 81.
            # Spending the party's one catch-up attempt on it is the
            # same waste this bound exists to prevent.
            if away is not None and self._marker_is_another_world(seat):
                return away
        return None

    def _marker_is_another_world(self, seat):
        """Does this seat's goal name a destination in another world?

        False whenever either side is unknown -- an area the quest data
        does not list, an area two worlds share, a goal with no
        destination in it, a zone that will not read. This gates rungs
        that hold the script and move the party, and refusing to act on
        a guess is the whole point.
        """
        from .. import questlist

        try:
            return questlist.goal_is_elsewhere(seat.goal, seat.zone_seen)
        except Exception:
            return False

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
            # A catch-up that actually finished is evidence the
            # placement CAN settle, so the churn count starts over.
            self._churn = 0
            self._stop_catching_up(
                "catch-up-done",
                f"{names} back with the party — the script has its "
                f"wizards back")
            return
        if not behind <= started:
            # Somebody NEW is behind, so this catch-up is answering the
            # wrong question and a fresh one should start from the top.
            #
            # ...and if that keeps happening, stop starting them. Rev
            # cfeb9a85's last seven minutes are six catch-ups, three of
            # them ended by this branch within fourteen seconds of
            # starting -- 4025.7, 4049.8, 4063.4 -- each naming a
            # different wizard as the one behind (Phönix 13, then
            # Sebastian 11, then both 11). The party was genuinely
            # spread across mainline and Arena side content, so the
            # placement had no stable answer, and every restart paused
            # the script again. A reading that changes its mind three
            # times in a minute is not a laggard to rescue; it is a
            # measurement that cannot support pausing anybody.
            import time as _time

            now_churn = _time.monotonic()
            if now_churn - self._churn_at > self.CATCH_UP_CHURN_WINDOW:
                self._churn = 0
            self._churn += 1
            self._churn_at = now_churn
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
            # The give-up is for a wizard that could plausibly survive
            # the fight anyway -- NOT for one on fumes. Rev 676d6e77:
            # twenty-two "went-in-hurt" in one run, most on 0-5%
            # health, each a certain death costing three to four
            # minutes of dying, respawning and walking back -- while
            # the same export shows wisps arriving given time ("back
            # to 45% after 133s"). Below `CRITICAL_HEALTH`, waiting
            # longer is strictly cheaper than dying again, so the
            # deadline stretches to `CRITICAL_HEALTH_WAIT`.
            limit = (self.LOW_HEALTH_WAIT if left >= self.CRITICAL_HEALTH
                     else self.CRITICAL_HEALTH_WAIT)
            if time.monotonic() - started > limit:
                self._say(seat,
                          f"still on {left:.0%} health after "
                          f"{limit:.0f}s and nothing is "
                          f"fixing it — going into the next fight anyway, "
                          f"because a run that stops here reports nothing "
                          f"at all")
                seat.tel.note_questing(
                    "went-in-hurt",
                    f"gave up after {limit:.0f}s on "
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
        self._watch_for_a_fight_that_cannot_be_won(seat, rec)
        self._say_why_it_planned_alone(seat, rec)
        self.seat_round_done.emit(seat.index, rec)
        if seat.index == 0:
            self.round_done.emit(rec)

    #: how many rounds in a row where NO line survives before the run is
    #: told. Five, because one such round is a bad hand and five is the
    #: board. Rev 8ebfcf70's fight 2 opened with seven of them at full
    #: health, so this fires deep inside the FIRST attempt rather than
    #: after the second corpse -- which is the difference between losing
    #: ten minutes and losing thirty.
    UNWINNABLE_ROUNDS = 5

    def _say_why_it_planned_alone(self, seat, rec):
        """Once per fight: why this round's plan held one wizard.

        `planned_alone` was built to stop a policy's NAME being taken as
        evidence it had anybody to coordinate with, and it does that --
        but it is a count, and a count cannot be acted on. Rev 8ebfcf70:
        the two wizards shared a duel in 33 of 46 fights and fused a
        plan in 11 rounds of 161, and nothing in either export said
        whether the missing seat was waited for and late, or never
        expected at all. Those want opposite fixes.

        Once per fight, because it is a property of the fight rather
        than of the round -- the run has 151 alone rounds in it and 151
        identical lines would bury the thing they explain.
        """
        if (rec.seats_in_plan or 1) > 1 or self.hive is None:
            return
        if len([s for s in self.seats if s.client is not None]) < 2:
            return
        # This seat's reason, not the party's most recent one. They are
        # different sentences pointing at different wizards: "waited and
        # it did not submit" is a slow client, "reached the round after
        # the party had planned it" is this client being the slow one.
        # One shared field meant a seat could be filed under the exact
        # opposite of what happened to it.
        reason = getattr(self.hive, "alone_reason", None)
        why = (reason(seat.index) if callable(reason)
               else getattr(self.hive, "last_alone", None))
        if not why:
            return
        index = getattr(seat.tel.fights[-1], "index", None) \
            if seat.tel.fights else None
        if seat.alone_said_for == index:
            return
        seat.alone_said_for = index
        try:
            seat.tel.note_questing(
                "planned-alone",
                f"round {rec.round} planned without the rest of the party: "
                f"{why}")
        except Exception:
            pass

    def _watch_for_a_fight_that_cannot_be_won(self, seat, rec):
        """Say so when every line the rollout tries ends in this wizard's
        death, round after round.

        The verdict already exists and already goes in the export. A
        rollout that dies returns `_lost_score`'s sentinel rather than a
        turn count, `policies.is_sentinel` is the ready-made classifier
        for it -- and its only caller in the repo is a pip-thrift
        tie-break inside the ranking. So the one component that actually
        knows the fight is unwinnable tells nobody.

        Rev 8ebfcf70 is what that costs. Konstantin's fight 2 opens with
        rounds 1-7 where every candidate INCLUDING `pass` is the
        sentinel, at 1998 of 1998 health -- the policy knew on round one,
        at full health, that it had no line -- and 19 of the fight's 26
        rounds read that way. The same board was then walked back into
        twice more. Nothing said a word until the wizard was dead, three
        times over.

        Reported, not acted on. What to DO about it depends on why the
        line is missing -- a party that never arrived, a deck with no
        answer, a wizard ten levels light -- and only the operator can
        tell those apart. What wizAi can do is stop the verdict dying
        inside the ranking function.
        """
        from .. import policies

        cands = list(rec.candidates or ())
        if not cands:
            # No candidates is not "no line survives": a round with
            # nothing castable has nothing to roll out. It says nothing
            # either way, so it neither counts nor clears.
            return

        def died(cand):
            turns = getattr(cand, "turns", None)
            horizon = getattr(cand, "horizon", None)
            if turns is None and isinstance(cand, dict):
                turns, horizon = cand.get("turns"), cand.get("horizon")
            if turns is None or horizon is None:
                return False
            try:
                return policies.is_sentinel(turns, horizon)
            except Exception:
                return False

        if not all(died(c) for c in cands):
            seat.no_line_survives = 0
            return
        seat.no_line_survives += 1
        fight = seat.tel.fights[-1] if seat.tel.fights else None
        index = getattr(fight, "index", None)
        if (seat.no_line_survives < self.UNWINNABLE_ROUNDS
                or seat.unwinnable_said_for == index):
            return
        seat.unwinnable_said_for = index
        opening = getattr(fight, "opening", "") or "this board"
        with_me = rec.seats_in_plan or 1
        live = [s for s in self.seats if s.client is not None]
        alone = (f", and {seat.name} is planning them alone while "
                 f"{len(live)} wizard(s) are in the run"
                 if with_me <= 1 and len(live) > 1 else "")
        for other in self.seats:
            try:
                other.tel.note_questing(
                    "unwinnable",
                    f"{seat.no_line_survives} rounds in a row against "
                    f"{opening} where every line the rollout tried — "
                    f"including passing — ended with {seat.name} dead"
                    f"{alone}. At {rec.player_hp:.0f} of "
                    f"{rec.player_max_hp:.0f} health. This is the policy's "
                    f"own verdict, and it has been available since the "
                    f"round it first appeared")
            except Exception:
                pass
        self._say(seat,
                  f"{seat.name} has no surviving line against {opening} — "
                  f"{seat.no_line_survives} rounds running")

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
        self._fill_script_names()

    def _fill_script_names(self):
        """Put the party's just-learned names into a placeholder script.

        The preset dialog fills names in when it can, and before the
        first duel it cannot: the game does not give a wizard's name
        outside combat, so a run started cold gets its schools filled
        and its account names left at the placeholder -- and the script
        silently skips every friend-teleport it has. The run at rev
        ed709013 logged `script-unconfigured` at 92s and again at 403s,
        AFTER the first fight had put all three names on the seats; the
        knowledge arrived and nothing used it. The operator: "the auto
        run starts after the first combat when the names are set".

        So the moment the last name lands, the fill happens here.
        Setting `self.script` is the whole trigger: `_sync_script`
        compares each seat's `script_source` against it every tick and
        rebuilds the runner on any difference, so the configured text
        takes over within a tick. That restart is a cost the script's
        own design already pays constantly -- a deimoslang program that
        runs off its end restarts by definition.

        Not in solo-pilot mode, where `solo_source` strips the account
        names on purpose so the script never friend-teleports at all.
        """
        from .. import scripts

        if not self.script or self._solo_pilot():
            return
        party = [s for s in self.seats if s.client is not None]
        if len(party) < 2 or not all(s.wizard_name for s in party):
            return
        blanks = scripts.unfilled(self.script, len(party))
        if not any(name in scripts.ACCOUNT_VARS for name, _v in blanks):
            return
        # The FULL name, or none at all.
        #
        # `seat.wizard_name` is a combat read and combat reports a first
        # name: "Konstantin". The friends list holds "Konstantin Ice",
        # and the script's `friendtp Main_Account` ends in wizwalker's
        # `friend_name == name` -- exact. So a short name here does not
        # half-work, it fails every teleport it enables, and it enables
        # them: the guards are `if NOT Main_Account = "Questing
        # AccountName"`, so filling the variable at all switches the
        # branch on. Rev f32be436 ran three hours that way.
        #
        # Passing "" leaves the account var at its placeholder --
        # `configure` skips an empty value -- so the schools still land
        # and the friend-teleport branches stay honestly off until
        # `_resolve_party_names` has read the list.
        source, filled = scripts.configure(
            self.script,
            [(self._full_names.get(s.wizard_name, ""), s.school)
             for s in party])
        if not filled:
            return
        self.script = source
        said = ("the party's names are known now — filled "
                + ", ".join(f"{n}={v}" for n, v in filled[:4])
                + (" …" if len(filled) > 4 else "")
                + " into the script, which reloads with its "
                  "friend-teleports live. Until this moment it was "
                  "silently skipping them")
        for seat in self.seats:
            try:
                seat.tel.note_questing("script-configured", said)
            except Exception:
                pass
        self._say(party[0], said)

    #: how long between attempts to read the party's full names off a
    #: friends list. Each attempt opens a window on up to two clients,
    #: so it is not free -- and it stops entirely once every name is
    #: known, which is normally the first try.
    NAME_RESOLVE_EVERY = 120.0

    async def _resolve_party_names(self, seat):
        """Learn each wizard's name as the FRIENDS LIST spells it.

        The missing half of filling the quester's account settings in.
        wizAi knew the party's names from the first duel and wrote them
        straight into the script -- and combat reports a FIRST name
        while every teleport that consumes those settings matches the
        friends list exactly:

            p1 friendtp $Questee2   ->  teleport_to_friend_from_list(
                                            name="Sebastian")
            wizwalker:  friend_name == name

        The list holds "Sebastian Life". So the fill did not half-work,
        it turned every one of the script's friend-teleports ON and made
        each of them unfindable -- worse than the placeholder, which at
        least left the branch honestly off.

        One read answers for the whole party. Each client's friends list
        holds the OTHER wizards, so a party of three needs two reads,
        and the wizard doing the reading gets its own name from
        somebody else's list.

        Nothing here is required for the run: a name that will not
        resolve leaves that account setting at its placeholder, which is
        exactly the state the script is designed to cope with.
        """
        import time

        from .. import party, questing, scripts

        if not self.script or self._solo_pilot() or self._names_done:
            return
        seats = [s for s in self.seats
                 if s.client is not None and s.wizard_name]
        if len(seats) < 2:
            return
        if not any(name in scripts.ACCOUNT_VARS
                   for name, _v in scripts.unfilled(self.script, len(seats))):
            self._names_done = True
            return
        now = time.monotonic()
        if now - self._names_tried_at < self.NAME_RESOLVE_EVERY:
            return
        # The preconditions BEFORE the cooldown is spent -- a refusal is
        # not an attempt, the same rule `_maybe_rearm_quest_arrow`
        # follows and this one broke.
        #
        # Rev f2b8101f is what breaking it costs. The party spent that
        # run in a dungeon, out of combat for about ten seconds in
        # sixteen minutes; every tick that reached this rung found a
        # dialogue open, burned the two-minute cooldown and returned, so
        # eight attempts in a row read nothing and the account settings
        # stayed at their placeholder for the whole run.
        if seat.in_duel or await questing.in_dialogue(seat.client):
            return
        readers = [s for s in seats if not s.in_duel]
        if not readers:
            # Nobody can open a friends list right now, and finding that
            # out is not an attempt either.
            return
        # One resolver at a time for the whole party: it drives several
        # clients, and two of them racing would each open a friends list
        # on the other's wizard mid-read.
        self._names_tried_at = now

        shorts = [s.wizard_name for s in seats]
        for reader in readers:
            missing = [s for s in shorts if s not in self._full_names]
            if not missing:
                break
            try:
                got = await party.friends_list_names(reader.client, missing)
            except Exception as exc:
                self._say(reader, f"could not read {reader.name}'s friends "
                                  f"list ({type(exc).__name__}: {exc})")
                continue
            self._full_names.update(got)

        known = {s: self._full_names.get(s, "") for s in shorts}
        if not any(known.values()):
            self._say(seat,
                      "none of the party's wizards is on another one's "
                      "friends list, so the script's account settings "
                      "cannot be filled in — they have to be friends for "
                      "its own regroup to work at all")
            return
        self._names_done = all(known.values())
        for other in self.seats:
            try:
                other.tel.note_questing(
                    "party-names-read",
                    "read the party's full names off the friends list: "
                    + ", ".join(f"{short} = {full!r}"
                                for short, full in known.items() if full)
                    + (" — " + ", ".join(s for s, f in known.items() if not f)
                       + " is not on any of the others' lists"
                       if not self._names_done else ""))
            except Exception:
                pass
        self._fill_script_names()

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
