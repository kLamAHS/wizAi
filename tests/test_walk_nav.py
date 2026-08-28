"""The walk-only navigator, driven.

`Deimos/src/walk_nav.py` keeps its top level free of game imports on
purpose, so unlike the rest of the vendored tree it can be LOADED here
and its walk loop actually run -- fake client, fake planner, injected
`is_free` -- instead of source-grepped. What these tests pin down is the
loop's contract: waypoints are verified rather than trusted, a stall is
re-aimed then re-planned then given up on, combat is an interruption
rather than a failure, and an off-mesh target is refused before a single
key is pressed.
"""
import asyncio
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "_walk_nav_under_test", ROOT / "Deimos/src/walk_nav.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def nav():
    return _load()


class _P:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _FakeClient:
    """Position, zone, and a `goto` whose effect the test scripts.

    `moves` is what a goto does: True teleports the fake straight to the
    waypoint (an obedient walk), False leaves it where it stands (a
    wall), a callable decides per call. `hang=True` makes goto never
    return, which is what a wedged leg looks like to `wait_for`.
    """

    def __init__(self, x=0.0, y=0.0, zone="WizardCity/WC_Streets",
                 moves=True, hang=False):
        self._pos = [float(x), float(y)]
        self.zone = zone
        self.moves = moves
        self.hang = hang
        self.gotos = []
        outer = self

        class _Body:
            async def position(self):
                return _P(outer._pos[0], outer._pos[1])

        class _Obj:
            async def speed_multiplier(self):
                return 0        # (0 / 100) + 1 = the base 1x run speed

        self.body = _Body()
        self.client_object = _Obj()

    async def zone_name(self):
        return self.zone

    async def goto(self, x, y):
        self.gotos.append((x, y))
        if self.hang:
            await asyncio.sleep(3600)
        moved = self.moves(x, y) if callable(self.moves) else self.moves
        if moved:
            self._pos = [x, y]


def _planner_of(*paths, refine=True):
    """A planner that hands out the given paths in order (last repeats).

    Each entry is a waypoint list, or None for "no path". `calls` counts
    plans, which is how a test proves a re-plan happened.
    """
    state = {"calls": 0}

    async def plan(client, start, target, avoid):
        i = min(state["calls"], len(paths) - 1)
        state["calls"] += 1
        p = paths[i]
        if p is None:
            return None, True, "no path"
        return list(p), refine, ""

    plan.state = state
    return plan


async def _free(client):
    return True


def _walk(nav, client, target, planner, is_free=None, budget=None):
    return asyncio.run(nav.walk_to(
        client, target, planner=planner,
        is_free=is_free or _free, budget=budget))


# ------------------------------------------------------------- the walk loop

def test_walk_to_walks_the_planned_waypoints_in_order(nav):
    client = _FakeClient()
    plan = _planner_of([(100, 0, 0), (200, 0, 0), (300, 0, 0)])
    outcome, detail = _walk(nav, client, _P(300, 0), plan)
    assert outcome == nav.WALK_ARRIVED, detail
    assert client.gotos == [(100, 0), (200, 0), (300, 0)]
    assert plan.state["calls"] == 1


def test_the_exact_target_is_refined_onto_after_the_path(nav):
    """The path ends on a hex node up to a node's width from the target;
    the last leg aims at the true coordinates, not the grid's version of
    them."""
    client = _FakeClient()
    plan = _planner_of([(100, 0, 0), (200, 40, 0)])
    outcome, _ = _walk(nav, client, _P(230, 45), plan)
    assert outcome == nav.WALK_ARRIVED
    assert client.gotos[-1] == (230, 45)


def test_a_leg_that_moves_nobody_is_reaimed_then_replanned(nav):
    calls = {"n": 0}

    def moves(x, y):
        calls["n"] += 1
        return calls["n"] > 2       # two dead gotos, then an obedient walk

    client = _FakeClient(moves=moves)
    plan = _planner_of([(100, 0, 0), (200, 0, 0)])
    outcome, detail = _walk(nav, client, _P(200, 0), plan)
    assert outcome == nav.WALK_ARRIVED, detail
    # goto #1 stalled (re-aim), #2 stalled (re-plan), then the fresh
    # path walked clean.
    assert plan.state["calls"] == 2
    assert len(client.gotos) == 4


def test_a_walk_that_keeps_stalling_gives_up_as_stuck(nav):
    client = _FakeClient(moves=False)
    plan = _planner_of([(100, 0, 0), (200, 0, 0)])
    outcome, detail = _walk(nav, client, _P(200, 0), plan)
    assert outcome == nav.WALK_STUCK, detail
    # Two stalled gotos per plan, REPLAN_LIMIT dry re-plans before the
    # verdict: the loop is bounded, not hopeful.
    assert plan.state["calls"] == nav.REPLAN_LIMIT
    assert len(client.gotos) == 2 * nav.REPLAN_LIMIT


def test_combat_ends_the_walk_and_is_not_a_failure(nav):
    freedom = iter([True, False])

    async def is_free(client):
        return next(freedom, False)

    client = _FakeClient()
    plan = _planner_of([(100, 0, 0), (200, 0, 0), (300, 0, 0)])
    outcome, detail = _walk(nav, client, _P(300, 0), plan, is_free=is_free)
    assert outcome == nav.WALK_INTERRUPTED
    assert "free" in detail
    assert client.gotos == [(100, 0)]   # nothing steered after the duel took over


def test_a_zone_change_mid_walk_is_a_door_firing(nav):
    client = _FakeClient()
    real_goto = client.goto

    async def goto(x, y):
        await real_goto(x, y)
        client.zone = "WizardCity/WC_Streets_02"

    client.goto = goto
    plan = _planner_of([(100, 0, 0), (200, 0, 0), (300, 0, 0)])
    outcome, detail = _walk(nav, client, _P(300, 0), plan)
    assert outcome == nav.WALK_INTERRUPTED
    assert detail == "zone changed"
    assert len(client.gotos) == 1


def test_an_off_mesh_target_is_refused_quietly(nav):
    async def plan(client, start, target, avoid):
        return None, True, nav.DETAIL_OFF_MESH

    client = _FakeClient()
    outcome, detail = _walk(nav, client, _P(0, 100000), plan)
    assert (outcome, detail) == (nav.WALK_UNREACHABLE, nav.DETAIL_OFF_MESH)
    assert client.gotos == []           # refused before any key went down


def test_already_standing_there_is_an_arrival_without_a_plan(nav):
    plan = _planner_of([(5, 5, 0)])
    client = _FakeClient(x=10, y=0)
    outcome, _ = _walk(nav, client, _P(12, 5), plan)
    assert outcome == nav.WALK_ARRIVED
    assert plan.state["calls"] == 0 and client.gotos == []


def test_a_leg_timeout_counts_as_a_stall_and_a_budget_as_partial(nav):
    # A goto that never returns is indistinguishable from a wall except
    # by clock, so the leg timeout turns it into an ordinary stall...
    nav.LEG_TIMEOUT_MIN = 0.05
    nav.LEG_TIMEOUT_MAX = 0.1
    client = _FakeClient(hang=True)
    plan = _planner_of([(100, 0, 0), (200, 0, 0)])
    outcome, _ = _walk(nav, client, _P(200, 0), plan, budget=30.0)
    assert outcome == nav.WALK_STUCK
    assert plan.state["calls"] > 1      # the timeouts drove re-plans

    # ...and a budget that runs out with ground covered is a resume, not
    # a failure: the caller simply issues the hop again.
    nav.LEG_TIMEOUT_MIN = 3.0
    nav.LEG_TIMEOUT_MAX = 10.0
    client2 = _FakeClient()
    real_goto = client2.goto

    async def slow_goto(x, y):
        await asyncio.sleep(0.2)    # each leg takes real time
        await real_goto(x, y)

    client2.goto = slow_goto
    plan2 = _planner_of([(100, 0, 0), (200, 0, 0), (300, 0, 0)])
    outcome2, detail2 = _walk(nav, client2, _P(300, 0),
                              plan2, budget=0.3)
    assert outcome2 == nav.WALK_PARTIAL, (outcome2, detail2)
    assert "covered" in detail2


# ------------------------------------------------------------- pure helpers

def test_smoothing_merges_legs_and_respects_the_cap(nav):
    path = [(0, 0, 0), (100, 0, 0), (200, 0, 0), (300, 0, 0), (400, 0, 0)]
    merged = nav.smooth(path, lambda a, b: True, max_leg=250)
    assert merged == [(0, 0, 0), (200, 0, 0), (400, 0, 0)]

    # A refused segment keeps the intermediate node: the predicate is
    # the navmesh's answer, not a suggestion.
    def around_the_corner(a, b):
        return not (a == (0, 0, 0) and b == (200, 0, 0))

    kept = nav.smooth(path, around_the_corner, max_leg=250)
    assert (100, 0, 0) in kept


def test_a_target_inside_the_solid_is_a_door_a_target_outside_a_statue(nav):
    path = [(0, 0, 0), (100, 0, 0), (200, 0, 0)]
    solid = lambda x, y: 150 <= x <= 250

    # Statue: the target stands OUTSIDE the volume that bounced the
    # teleport, so the walk stops at its edge and never grinds the face.
    kept, refine = nav.truncate_at_solids(path, (400, 0), solid)
    assert kept == [(0, 0, 0), (100, 0, 0)] and refine is False

    # Door: the target IS inside the volume -- zone doors and lifts are
    # aimed into by the quest arrow, and walking in is how they fire.
    kept, refine = nav.truncate_at_solids(path, (200, 0), solid)
    assert kept == path and refine is True


# --------------------------------------------------------------- the switch

def test_the_mode_switch_validates_and_suspension_nests(nav):
    assert nav.nav_mode() == nav.NAV_TELEPORT and not nav.walking()
    with pytest.raises(ValueError):
        nav.set_nav_mode("sprint")
    with pytest.raises(ValueError):
        nav.set_fallback("maybe")

    assert nav.set_nav_mode(nav.NAV_WALK) == nav.NAV_TELEPORT
    assert nav.walking()
    with nav.suspended():
        assert not nav.walking()
        with nav.suspended():           # nests without unwinding early
            assert not nav.walking()
        assert not nav.walking()
    assert nav.walking()


def test_suspension_is_task_local_not_party_wide(nav):
    """A collect scan teleporting wizard 2 underground must not flip
    wizard 3 into teleport mode mid-walk: four wizards interleave on one
    loop, so the suspension rides the task, not the module."""
    nav.set_nav_mode(nav.NAV_WALK)
    seen = {}

    async def scanning_wizard(entered):
        with nav.suspended():
            entered.set()
            await asyncio.sleep(0.05)
            seen["scanner"] = nav.walking()

    async def walking_wizard(entered):
        await entered.wait()
        seen["walker"] = nav.walking()

    async def run():
        entered = asyncio.Event()
        await asyncio.gather(scanning_wizard(entered),
                             walking_wizard(entered))

    asyncio.run(run())
    assert seen == {"scanner": False, "walker": True}


# ---------------------------------------------------------- the review round

def test_arrival_is_three_dimensional(nav):
    """The walk grid is single-level -- `walk_z` is the LOWEST mesh
    surface -- so a balcony objective's XY is reachable on the alley
    floor below it. Standing under the target is not arriving: the
    answer is unreachable, which hands the hop to the floor-aware
    teleport fallback instead of looping 'arrived' a storey short."""
    client = _FakeClient()
    plan = _planner_of([(100, 0, 0), (200, 0, 0), (300, 0, 0)])
    outcome, detail = _walk(nav, client, _P(300, 0, 1200.0), plan)
    assert outcome == nav.WALK_UNREACHABLE, (outcome, detail)
    assert "wrong floor" in detail


def test_a_detour_that_walks_away_is_partial_not_stuck(nav):
    """Progress at the deadline is ground covered, not gap closed: a
    legitimate route can spend its whole budget walking AWAY from the
    target (around a canyon), and a gap-only test hands that working
    walk to the teleport fallback as 'going nowhere'."""
    client = _FakeClient()
    real_goto = client.goto

    async def slow_goto(x, y):
        await asyncio.sleep(0.2)
        await real_goto(x, y)

    client.goto = slow_goto
    # Away first: the gap to (100, 0) GROWS over these first legs.
    plan = _planner_of([(0, 300, 0), (0, 600, 0), (100, 600, 0),
                        (100, 0, 0)])
    outcome, detail = _walk(nav, client, _P(100, 0), plan, budget=0.3)
    assert outcome == nav.WALK_PARTIAL, (outcome, detail)
    assert "covered" in detail


def test_walk_failures_are_remembered_briefly(nav):
    class _T:
        def __init__(self, x, y):
            self.x, self.y = x, y

    zone = "WizardCity/WC_Streets"
    spot = _T(1234.0, -560.0)
    assert not nav.recently_failed(zone, spot)
    nav.note_walk_failure(zone, spot)
    assert nav.recently_failed(zone, spot)
    # Same ~100u bucket counts; a different zone does not.
    assert nav.recently_failed(zone, _T(1260.0, -530.0))
    assert not nav.recently_failed("GrizzleheimA/Somewhere", spot)
    # And it forgets: rewind the stamp past the TTL.
    key = nav._failure_key(zone, spot)
    nav._failed_walks[key] -= nav.WALK_FAILURE_TTL + 1
    assert not nav.recently_failed(zone, spot)

