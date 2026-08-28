"""Walk-only navigation: the mode switch and the on-foot navigator.

The teleport primitives in `teleport_math` move a wizard by writing its
position into the client. This module is the other way to get there:
plan a route over the zone's walkable hex grid (`collision_math`'s A*)
and walk it leg by leg with `client.goto` -- yaw write plus a held W key,
the same input a player produces.

The walk-only principle, which the routing in `teleport_math`, the
script VM and the quester all follow:

* **No memory-write teleports for navigation.** When the mode is
  `NAV_WALK`, anything that used to teleport the wizard *toward an
  objective* walks instead.
* **In-game transport stays allowed.** Pressing X at doors and sigils,
  the spiral door UI, dungeon and hub recall, and friend-teleports made
  through the friends-list UI are things a player does; they are not
  suppressed.
* **Technical teleports stay teleports.** Out-of-bounds unstick bumps
  and the underground entity-streaming scans in collect quests move the
  wizard somewhere no path can reach *on purpose*; they run under
  `suspended()` or short-circuit on an off-mesh target.
* **Walking that fails falls back.** With `FALLBACK_AUTO` (the default),
  a target that is unreachable on foot or a wizard that stays stuck
  after re-aims and re-plans gets the old teleport for that one hop,
  and the caller says so in the run's log.

Kept deliberately free of top-level game imports (wizwalker, loguru,
shapely, the collision modules): everything heavy is imported lazily
inside the default planner, exactly like `collision_tp` does. That is
what lets the walk loop run under real driven tests on a machine with
no game installed.
"""
import asyncio
import contextlib
import contextvars
import math
import time
import weakref

# --------------------------------------------------------------- the mode

NAV_TELEPORT = "teleport"
NAV_WALK = "walk"

_mode = NAV_TELEPORT

#: What to do when walking genuinely cannot deliver the target.
#: `auto` (the default): the caller falls back to the old teleport for
#: that hop and notes it. `never`: the caller reports the failure and
#: does not teleport -- implemented for completeness, not exposed in the
#: GUI, because a run that parks at its first unreachable objective
#: needs a human watching it.
FALLBACK_AUTO = "auto"
FALLBACK_NEVER = "never"

_fallback = FALLBACK_AUTO

#: Task-local suspension depth. A contextvar rather than a bare global
#: because four wizards' coroutines interleave on one event loop: a
#: collect scan teleporting wizard 2 underground must not flip wizard 3
#: into teleport mode mid-walk.
_suspended = contextvars.ContextVar("walk_nav_suspended", default=0)


def set_nav_mode(mode: str) -> str:
    """Set the navigation mode; returns the previous one."""
    global _mode
    if mode not in (NAV_TELEPORT, NAV_WALK):
        raise ValueError(f"unknown navigation mode {mode!r}")
    previous = _mode
    _mode = mode
    return previous


def nav_mode() -> str:
    return _mode


def set_fallback(policy: str) -> str:
    """Set the walk-failure policy; returns the previous one."""
    global _fallback
    if policy not in (FALLBACK_AUTO, FALLBACK_NEVER):
        raise ValueError(f"unknown fallback policy {policy!r}")
    previous = _fallback
    _fallback = policy
    return previous


def fallback() -> str:
    return _fallback


@contextlib.contextmanager
def suspended():
    """Teleports inside this block stay teleports, walk mode or not.

    For the technical moves: underground collect scans, out-of-bounds
    bumps. Nests, and is task-local (see `_suspended`).
    """
    token = _suspended.set(_suspended.get() + 1)
    try:
        yield
    finally:
        _suspended.reset(token)


def walking() -> bool:
    """Is navigation currently supposed to go on foot?"""
    return _mode == NAV_WALK and _suspended.get() == 0


# ---------------------------------------------------------------- outcomes

WALK_ARRIVED = "arrived"          # standing at (or as near as the mesh goes to) the target
WALK_PARTIAL = "partial"          # budget ran out but ground was covered; caller re-issues
WALK_INTERRUPTED = "interrupted"  # combat/dialogue/zone change took over; not a failure
WALK_UNREACHABLE = "unreachable"  # no on-foot route can deliver this target
WALK_STUCK = "stuck"              # walked into something and re-planning did not help
WALK_ERROR = "error"              # the client stopped answering mid-walk

#: The one unreachable detail that is *expected* in normal runs: scripts
#: aim `tp XYZ(0, 100000, 0)`-style bumps at points no mesh contains,
#: and those should fall back to the raw teleport silently.
DETAIL_OFF_MESH = "off-mesh"

# ------------------------------------------------------------------ tuning

#: `wizwalker.constants.WIZARD_SPEED` -- base run speed in units/second.
#: Duplicated (not imported) so this module loads without the game.
WIZARD_SPEED = 580.0

#: Close enough to count as standing at the target. Same rationale as
#: `_WALK_GAP_MIN` in teleport_math: inside interaction AND aggro range.
ARRIVE_TOL = 20.0

#: Close enough to a *waypoint* to advance to the next. Over half the
#: 100u hex spacing, so a slightly short `goto` does not stall the leg.
WAYPOINT_TOL = 55.0

#: A leg attempt that gained less than this toward its waypoint is a
#: stall (wall, another wizard, a fence post).
MIN_PROGRESS = 30.0

#: Stall ladder: first stall re-aims (re-issues the same goto), second
#: re-plans from where the wizard actually stands. This many re-plans
#: with no net gap reduction is a wizard that walking cannot free.
REPLAN_LIMIT = 3

#: Absolute re-plan ceiling, so pathological geometry cannot alternate
#: "a little progress / a re-plan" forever.
REPLAN_HARD_CAP = 8

#: A goal that has to snap farther than this to reach walkable ground is
#: not deliverable on foot (a marker inside an instance, over water, on
#: a pad) -- the teleport machinery handles it better than a walk that
#: would end hundreds of units short.
GOAL_SNAP_MAX = 250.0

#: Same floor or not -- `collision_math.FLOOR_MATCH_TOL`'s value,
#: duplicated for the no-game import rule. The walk grid is single-level
#: by construction (`walk_z` is the LOWEST mesh surface at a node), so a
#: target on a balcony over walkable ground planes to the ground floor's
#: XY: arrival and planning both have to compare z, or the wizard "arrives"
#: a storey below the objective and the caller loops there forever.
FLOOR_TOL = 300.0

#: Route sanity: a path this many times longer than the straight-line
#: gap, or longer than this many units outright, is the A* finding a
#: technically-connected route no player would take.
MAX_DETOUR = 25.0
MAX_WALK_UNITS = 60_000.0

#: Smoothing: merge A* nodes into straight legs up to this long. Bounds
#: how stale a single `goto`'s dead reckoning can get, and how long a
#: cancellation (combat started, stage cap) waits on the current leg.
SMOOTH_LEG_MAX = 600.0

#: Per-call wall-clock ceiling, deliberately under the bridge's 60s
#: "quest step" stage limit (`STAGE_LIMITS` in live.py -- the 90s
#: default does NOT apply to quest steps): a longer route ends as
#: WALK_PARTIAL and the next tick continues from where the wizard
#: stands, instead of the stage cap cancelling the walk mid-leg.
WALK_BUDGET_CAP = 45.0

#: Per-leg goto timeout bounds. `goto` computes its own duration from
#: distance and the speed multiplier; the timeout only exists to turn a
#: wedged leg into a visible stall instead of a hang.
LEG_TIMEOUT_MIN = 3.0
LEG_TIMEOUT_MAX = 10.0


def _dist(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)


# --------------------------------------------------------- pure path helpers

def smooth(path, can_walk_straight, max_leg: float = SMOOTH_LEG_MAX):
    """Merge consecutive A* nodes into the longest straight legs that stay
    walkable, greedily: from each kept point, extend to the farthest node
    that `can_walk_straight(a, b)` accepts, stopping the scan at the first
    refusal or once the leg would exceed ``max_leg``. Adjacent A* nodes
    are walkable by construction, so the fallback step is always safe.

    ``path`` is ``[(x, y, z), ...]``; the predicate gets two such tuples.
    Pure -- run it wherever the predicate is cheap (the planner thread).
    """
    if not path or len(path) <= 2:
        return list(path)
    out = [path[0]]
    i = 0
    last = len(path) - 1
    while i < last:
        best = i + 1
        j = i + 2
        while j <= last:
            if _dist(path[i][0], path[i][1], path[j][0], path[j][1]) > max_leg:
                break
            if not can_walk_straight(path[i], path[j]):
                break
            best = j
            j += 1
        out.append(path[best])
        i = best
    return out


def truncate_at_solids(path, target_xy, contains):
    """Stop a path at the edge of the solid footprints that bounced a
    teleport -- unless the target itself sits inside one, in which case
    the volume is a door/lift/warp frame and walking in is the point.

    Same discriminator as `_walk_remaining_to_target` in teleport_math
    (see the rev 66a4fe5b note there). ``contains(x, y)`` answers "inside
    any footprint". Returns ``(path, refine)`` where ``refine`` says the
    exact-target approach is still allowed (False iff truncated).
    """
    if contains(target_xy[0], target_xy[1]):
        return list(path), True     # a door: walk all the way in
    kept = []
    for wp in path:
        if contains(wp[0], wp[1]):
            break                   # the statue rule: stop at the edge
        kept.append(wp)
    return kept, len(kept) == len(path)


# ---------------------------------------------------- remembering failures

#: How long a target that walking could not deliver stays remembered.
#: Scripts and the quester retry the same hop in tight loops; without
#: this, every retry burns a full walk budget re-proving the same
#: failure before the fallback teleport gets its turn.
WALK_FAILURE_TTL = 120.0

_failed_walks: dict = {}


def _failure_key(zone: str, xyz):
    # ~100u buckets: the same objective read twice rarely lands on the
    # exact same floats.
    return (zone or "", int(xyz.x // 100), int(xyz.y // 100))


def note_walk_failure(zone: str, xyz) -> None:
    now = time.monotonic()
    _failed_walks[_failure_key(zone, xyz)] = now
    if len(_failed_walks) > 256:
        cut = now - WALK_FAILURE_TTL
        for k, at in list(_failed_walks.items()):
            if at < cut:
                del _failed_walks[k]


def recently_failed(zone: str, xyz) -> bool:
    at = _failed_walks.get(_failure_key(zone, xyz))
    return at is not None and time.monotonic() - at < WALK_FAILURE_TTL


# ------------------------------------------------------------- the planner

#: One build per zone at a time: the grid build costs ~1.5s of CPU and
#: four wizards asked to walk at once must share it, not race it.
#: Keyed by event loop (weakly) because an asyncio.Lock binds to the
#: loop that first awaits it -- a worker restarted on a fresh loop must
#: get fresh locks, not RuntimeErrors from the dead one's.
_plan_locks: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _zone_lock(zone: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    locks = _plan_locks.get(loop)
    if locks is None:
        locks = _plan_locks[loop] = {}
    return locks.setdefault(zone, asyncio.Lock())


async def _default_planner(client, start_pos, target_xyz, avoid):
    """Plan a walkable route: ``(path, refine, detail)``.

    ``path`` is ``[(x, y, z), ...]`` or None with ``detail`` saying why
    (``DETAIL_OFF_MESH`` for a target the mesh cannot deliver). All the
    game/geometry imports live here, lazily, `collision_tp`-style.
    """
    try:
        zone = await client.zone_name()
    except Exception:
        zone = None
    if not zone:
        return None, True, "zone unreadable"
    try:
        from src.collision import CollisionWorld, get_collision_data
        from src.collision_math import get_walk_grid, straight_path_walkable
        from src import entity_collision

        collision_data = await get_collision_data(client, zone)
        async with _zone_lock(zone):
            def _build():
                world = CollisionWorld()
                world.load(collision_data)
                extra = entity_collision.build_zone_static_shapes(zone, None)
                # 45.0: get_walk_grid's own default and what prewarm_zone
                # builds with -- the cache is per zone, first build wins.
                return world, extra, get_walk_grid(world, zone, extra, 45.0)

            world, extra, grid = await asyncio.to_thread(_build)

        def _plan():
            # A goal whose own node is unwalkable ends the walk at the
            # path's last node instead of refining onto the exact target
            # -- same rule as `_walk_remaining_to_target`: never goto
            # onto a pad/boat/water the mesh says cannot be stood on.
            refine = True
            gq = grid.to_hex(target_xyz.x, target_xyz.y)
            goal_z = grid.walk_z(*gq)
            if goal_z is None:
                refine = False
                snapped = grid._nearest_walk_node(*gq)
                if snapped is None:
                    return None, True, DETAIL_OFF_MESH
                gx, gy = grid.to_world(*snapped)
                if _dist(gx, gy, target_xyz.x, target_xyz.y) > GOAL_SNAP_MAX:
                    return None, True, DETAIL_OFF_MESH
                goal_z = grid.walk_z(*snapped)
            # The floor gate. `walk_z` is single-level -- the LOWEST mesh
            # surface at that XY -- so a balcony target over walkable
            # ground planes to the alley below it: the walk would "arrive"
            # a storey under the objective and the caller would loop there
            # forever. The teleport machinery is floor-aware; hand over.
            tz = getattr(target_xyz, "z", None)
            if (tz is not None and goal_z is not None
                    and abs(goal_z - tz) > FLOOR_TOL):
                return None, True, (
                    f"wrong floor (mesh z {goal_z:.0f}, target z {tz:.0f})")
            path = grid.find_walk_path(start_pos, target_xyz)
            if not path:
                return None, True, "no path"
            straight = _dist(start_pos.x, start_pos.y,
                             target_xyz.x, target_xyz.y)
            length = sum(
                _dist(a[0], a[1], b[0], b[1])
                for a, b in zip(path, path[1:]))
            if length > MAX_WALK_UNITS or (
                    straight > ARRIVE_TOL and length > straight * MAX_DETOUR):
                return None, True, (
                    f"path too long ({length:.0f}u for a "
                    f"{straight:.0f}u gap)")
            if avoid:
                from shapely.geometry import Point as _Point

                def _inside(x, y):
                    return any(fp.contains(_Point(x, y)) for fp in avoid)

                path, kept_whole = truncate_at_solids(
                    path, (target_xyz.x, target_xyz.y), _inside)
                refine = refine and kept_whole
                if not path:
                    return None, True, "walled off by the bouncing solid"

            if not avoid:
                # No smoothing over a truncated path: the merge predicate
                # knows the navmesh but not the `avoid` footprints, so a
                # straightened leg could cut the corner back through the
                # very solid the truncation stopped at. `avoid` walks are
                # rare (post-bounce only) and short; raw nodes are fine.
                class _P:                   # straight_path_walkable wants .x/.y
                    __slots__ = ("x", "y")

                    def __init__(self, p):
                        self.x, self.y = p[0], p[1]

                path = smooth(
                    path,
                    lambda a, b: straight_path_walkable(
                        world, zone, _P(a), _P(b), static_shapes=extra))
            return path, refine, ""

        return await asyncio.to_thread(_plan)
    except Exception as e:
        return None, True, f"plan error ({e!r})"


async def _default_is_free(client) -> bool:
    from src.utils import is_free
    return await is_free(client)


# ------------------------------------------------------------- the walker

async def walk_to(client, xyz, *, avoid=None, planner=None, is_free=None,
                  budget: float = None):
    """Walk the wizard to ``xyz``. Returns ``(outcome, detail)``.

    The loop: plan (off-thread), then walk leg by leg, verifying progress
    after every `goto` instead of trusting its dead reckoning. A leg that
    gains nothing is re-aimed once, then the route is re-planned from
    where the wizard actually stands; `REPLAN_LIMIT` re-plans with no net
    gap reduction is `WALK_STUCK`. Combat, dialogue or a zone change ends
    the walk as `WALK_INTERRUPTED` -- the questing loops re-enter
    navigation when the wizard is free again, which is the resume. The
    whole call is bounded by ``budget`` seconds; running out with ground
    covered is `WALK_PARTIAL`, the caller's cue to simply try again.

    ``planner`` and ``is_free`` are injectable for tests; the defaults
    lazy-import the collision stack and `src.utils.is_free`.
    Cancellation passes straight through: `goto` releases its held key in
    a ``finally``, so the bridge's stage cap can cut a walk safely.
    """
    plan_route = planner or _default_planner
    free = is_free or _default_is_free

    def on_floor(p):
        # Arrival is 3D: the walk grid is single-level, so a balcony
        # target's XY is reachable on the alley floor below it, and an
        # XY-only test would call that standing on the objective.
        tz = getattr(xyz, "z", None)
        pz = getattr(p, "z", None)
        return tz is None or pz is None or abs(pz - tz) <= FLOOR_TOL

    try:
        pos = await client.body.position()
    except Exception as e:
        return WALK_ERROR, f"position unreadable ({e!r})"
    gap0 = _dist(pos.x, pos.y, xyz.x, xyz.y)
    if gap0 <= ARRIVE_TOL and on_floor(pos):
        return WALK_ARRIVED, "already there"

    try:
        raw = await client.client_object.speed_multiplier()
        mult = max(0.25, (raw / 100) + 1)
    except Exception:
        mult = 1.0
    auto_budget = budget is None
    if auto_budget:
        budget = min(gap0 / (WIZARD_SPEED * mult) * 3 + 15.0, WALK_BUDGET_CAP)
    started = time.monotonic()
    deadline = started + budget

    try:
        zone0 = await client.zone_name()
    except Exception:
        zone0 = None

    path, refine, detail = await plan_route(client, pos, xyz, avoid)
    if path is None:
        return WALK_UNREACHABLE, detail
    if auto_budget and len(path) > 1:
        # The straight-line guess above undershoots a detour route; now
        # that the path is known, budget for its REAL length -- still
        # capped under the quest-step stage limit, so a very long route
        # ends as a clean PARTIAL resume rather than a stage timeout.
        length = _dist(pos.x, pos.y, path[0][0], path[0][1]) + sum(
            _dist(a[0], a[1], b[0], b[1]) for a, b in zip(path, path[1:]))
        budget = min(max(budget, length / (WIZARD_SPEED * mult) * 2 + 10.0),
                     WALK_BUDGET_CAP)
        deadline = started + budget

    stalls = 0
    replans = 0
    dry_replans = 0                 # re-plans since the gap last shrank
    best_gap = gap0
    walked_units = 0.0              # ground actually covered, leg by leg
    idx = 0
    exact = False                   # walking the final exact-target leg

    while True:
        gap = _dist(pos.x, pos.y, xyz.x, xyz.y)
        if gap <= ARRIVE_TOL and on_floor(pos):
            return WALK_ARRIVED, "on the target"

        if idx >= len(path):
            # Route done. Refine onto the exact target when allowed, then
            # accept where we stand: the goal snap already guaranteed the
            # path ends within GOAL_SNAP_MAX of the target, and a target
            # the mesh cannot quite reach must not retry forever.
            if refine:
                path = [(xyz.x, xyz.y, None)]
                refine = False
                exact = True
                idx = 0
                continue
            if not on_floor(pos):
                # The planner's floor gate should have refused this
                # route; standing under the objective is not arriving.
                return WALK_UNREACHABLE, "wrong floor at the path's end"
            return WALK_ARRIVED, f"path end, {gap:.0f}u out"

        wx, wy = path[idx][0], path[idx][1]
        leg = _dist(pos.x, pos.y, wx, wy)
        # The exact-target leg is held to the arrival tolerance, not the
        # waypoint one -- the whole point of refining is closing the last
        # half-node onto the true coordinates.
        near_enough = ARRIVE_TOL if exact else WAYPOINT_TOL
        if leg <= near_enough:
            idx += 1
            stalls = 0
            continue

        try:
            if not await free(client):
                return WALK_INTERRUPTED, "no longer free"
            if zone0 is not None and await client.zone_name() != zone0:
                return WALK_INTERRUPTED, "zone changed"
        except asyncio.CancelledError:
            raise
        except Exception:
            return WALK_INTERRUPTED, "client stopped answering"

        if time.monotonic() > deadline:
            # Progress is ground COVERED, not gap closed: a legitimate
            # detour route walks away from the target first, and a
            # gap-only test would call that going nowhere and hand a
            # working walk to the teleport fallback.
            if (walked_units >= 2 * MIN_PROGRESS
                    or gap < gap0 - MIN_PROGRESS):
                return WALK_PARTIAL, (
                    f"budget spent, {walked_units:.0f}u covered")
            return WALK_STUCK, "budget spent going nowhere"

        timeout = min(max(leg / (WIZARD_SPEED * mult) + 2.0,
                          LEG_TIMEOUT_MIN), LEG_TIMEOUT_MAX)
        before_x, before_y = pos.x, pos.y
        try:
            await asyncio.wait_for(client.goto(wx, wy), timeout)
        except asyncio.TimeoutError:
            pass                        # counted as a stall below
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return WALK_ERROR, f"goto failed ({e!r})"

        try:
            pos = await client.body.position()
        except Exception:
            return WALK_INTERRUPTED, "client stopped answering"

        walked_units += _dist(before_x, before_y, pos.x, pos.y)
        now = _dist(pos.x, pos.y, wx, wy)
        if now <= near_enough:
            idx += 1
            stalls = 0
            continue
        if (_dist(before_x, before_y, wx, wy) - now) >= MIN_PROGRESS:
            stalls = 0                  # gaining ground; keep at this leg
            continue

        stalls += 1
        if stalls == 1:
            continue                    # re-aim: one more try at this leg

        # Two stalls on one leg: the route is wrong where the wizard is.
        replans += 1
        gap = _dist(pos.x, pos.y, xyz.x, xyz.y)
        if gap < best_gap - MIN_PROGRESS:
            best_gap = gap
            dry_replans = 0
        else:
            dry_replans += 1
        if dry_replans >= REPLAN_LIMIT or replans >= REPLAN_HARD_CAP:
            return WALK_STUCK, (
                f"{replans} re-plans, still {gap:.0f}u out")
        path, refine, detail = await plan_route(client, pos, xyz, avoid)
        if path is None:
            return WALK_UNREACHABLE, detail
        idx = 0
        stalls = 0
        exact = False
