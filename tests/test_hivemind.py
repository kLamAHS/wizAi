"""Tests for the party coordinator.

The interesting claim is not "the code runs" — it is that four wizards
planning together play *differently* from four wizards planning apart,
and differently in the direction that wins fights. So most of these are
A/Bs: the same board, the same policies, the same seats, planned with
`passes=0` (which is uncoordinated play, exactly) and with the shipped
default, asserting on what changed.

Everything here is headless. `hivemind` imports no Qt and no wizwalker,
which is the whole reason the joint plan can be checked at all.
"""
import asyncio

import pytest

from deimos_bridge.hivemind import (Hivemind, KILL_CONFIDENCE, Ledger,
                                    _Submission, align_enemies, measure_cast)
from deimos_bridge.policies import greedy_ttk


# --------------------------------------------------------------------- setup
def _cards():
    from data_full import load_spells_full
    return load_spells_full()


CARDS = None


def cards():
    global CARDS
    if CARDS is None:
        CARDS = _cards()
    return CARDS


def wizard(school="fire", hand=("Fire Cat", "Fire Cat", "Fireblade"),
           pips=7, hp=1500, board=((50, "ice"), (800, "ice"))):
    """A (sim, state) pair standing in for one connected wizard."""
    from w101_sim import Actor, Boss, Rules, Sim, State

    table = cards()
    player = Actor(name="Wizard", school=school, hp=hp, max_hp=hp, team=0,
                   norm_pips=pips)
    player.hand = [table[n] for n in hand]
    foes = [Actor(name=f"Mob{i}", school=sc, hp=h, max_hp=h, team=1,
                  flat_hit=40)
            for i, (h, sc) in enumerate(board)]
    sim = Sim(cards=table, decklist=list(hand), school=school,
              boss=Boss(name="Mob0", hp=board[0][0], school=board[0][1],
                        dmg=40),
              rules=Rules(), player_hp=hp)
    return sim, State(player, foes)


def party(n=2, **kw):
    """`n` independent wizards on the same board, each with its own policy."""
    subs = []
    for i in range(n):
        sim, state = wizard(**kw)
        subs.append(_Submission(i, f"wizard {i + 1}", sim, state,
                                greedy_ttk()))
    return subs


def plan(n=2, passes=2, **kw):
    hive = Hivemind(passes=passes)
    return hive.plan(party(n, **kw))


# ------------------------------------------------------------- board alignment
def test_four_reads_of_one_board_line_up_positionally():
    _, a = wizard()
    _, b = wizard()
    assert align_enemies(a.enemies, b.enemies) == [0, 1]


def test_a_shorter_read_still_matches_by_name_and_health():
    """One client reading a round later has already dropped a corpse. The
    remaining mob still has to mean the same mob, or the ledger credits
    one wizard's damage against another wizard's other target."""
    _, a = wizard()
    _, b = wizard()
    a.enemies.pop(0)
    assert align_enemies(a.enemies, b.enemies) == [1]


def test_a_board_that_cannot_be_matched_declines_rather_than_guesses():
    _, a = wizard()
    _, b = wizard()
    a.enemies[0].name = "Somebody Else Entirely"
    a.enemies[1].name = "Also Not Here"
    assert align_enemies(a.enemies, b.enemies) is None


def test_an_unmatchable_seat_plans_alone_instead_of_being_dropped():
    subs = party(2)
    for i, enemy in enumerate(subs[0].state.enemies):
        enemy.name = f"Unrecognisable {i}"
    actions, party_plan = Hivemind(passes=2).plan(subs)
    assert set(actions) == {0, 1}
    assert len(party_plan.moves) == 2


# ------------------------------------------------------------------- measuring
def test_a_cast_is_priced_at_what_it_does_to_the_enemy_side():
    sim, state = wizard()
    effect = measure_cast(sim, state, (cards()["Fire Cat"], 1))
    assert effect.damage.get(1, 0) > 0
    assert effect.raw_damage > effect.damage[1]      # discounted by accuracy
    assert 0 < effect.accuracy < 1.0


def test_a_blade_is_not_discounted_for_accuracy():
    """Buffs do not fizzle, and pricing a Fireblade at 75% would make the
    party plan around a certainty as though it were a coin flip."""
    sim, state = wizard()
    effect = measure_cast(sim, state, (cards()["Fireblade"], 0))
    assert effect.accuracy == 1.0
    assert not effect.damage                 # it does nothing to the enemy


def test_measuring_a_cast_does_not_move_the_board_it_measured():
    sim, state = wizard()
    before = [e.hp for e in state.enemies]
    hand = [c.name for c in state.hand]
    measure_cast(sim, state, (cards()["Fire Cat"], 1))
    assert [e.hp for e in state.enemies] == before
    assert [c.name for c in state.hand] == hand


def test_damage_past_lethal_is_not_credited():
    """A 300-damage nuke into a 50 HP mob banks 50. Crediting the whole
    300 would let one oversized cast read as covering the board."""
    sim, state = wizard(board=((10, "ice"),))
    effect = measure_cast(sim, state, (cards()["Fire Cat"], 0))
    assert effect.damage[0] <= 10.0


# --------------------------------------------------------------------- ledger
def test_the_ledger_never_edits_the_state_it_was_given():
    sim, state = wizard()
    ledger = Ledger()
    ledger.add(measure_cast(sim, state, (cards()["Fire Cat"], 1)), [0, 1])
    out = ledger.apply(state, [0, 1])
    assert out is not state
    assert state.enemies[1].hp == 800


def test_a_confident_kill_is_written_down_to_dead_not_merely_reduced():
    """The Fire Cat case: ~51 damage into a 50 HP mob is lethal three
    times in four, and leaving it at 12 HP is how four wizards put four
    Fire Cats into one Lost Soul."""
    sim, state = wizard(board=((50, "ice"),))
    ledger = Ledger()
    ledger.add(measure_cast(sim, state, (cards()["Fire Cat"], 0)), [0])
    assert ledger.raw[0] >= 50
    assert not ledger.apply(state, [0]).enemies[0].alive


def test_an_unreliable_kill_leaves_the_mob_standing():
    sim, state = wizard(board=((50, "ice"),))
    ledger = Ledger(kill_confidence=0.99)   # nothing in the game is this sure
    ledger.add(measure_cast(sim, state, (cards()["Fire Cat"], 0)), [0])
    survivor = ledger.apply(state, [0]).enemies[0]
    assert survivor.alive and survivor.hp < 50


def test_a_ward_one_wizard_lays_lands_on_the_next_wizards_board():
    """The actual hivemind play, and the reason wards are shared while
    blades are not: a trap goes on the mob, so everybody cashes it."""
    sim, state = wizard(hand=("Fire Trap", "Fire Cat"), board=((800, "ice"),))
    effect = measure_cast(sim, state, (cards()["Fire Trap"], 0))
    assert effect.wards.get(0), "the trap left nothing on the enemy"

    _, other = wizard(hand=("Fire Cat",), board=((800, "ice"),))
    ledger = Ledger()
    ledger.add(effect, [0])
    assert ledger.apply(other, [0]).enemies[0].wards


def test_a_second_copy_of_the_partys_trap_adds_nothing_to_the_same_hit():
    """Casting for others, which is where the operator saw it worst.

    Traps are the party's shared currency -- one wizard lays it, every
    wizard cashes it -- so "is this trap already up?" is a question the
    coordinator asks about somebody ELSE's cast, on a board it was
    handed rather than one it read. The ice wizard's hand carries both
    `Ice Trap` and `Ice Trap - Amulet@item`, and under the old
    `(name, source)` identity those were two spells: laying the second
    on a mob the party had already trapped looked like 1.4 x 1.4.

    One spell, one template id (`spell_effect_stacking_id`), one trap
    per strike. The second is still worth carrying -- it fires on the
    NEXT hit -- but it does not double this one.
    """
    import copy
    import dataclasses

    from deimos_bridge.telemetry import predict_damage
    from w101_sim import Actor, Boss, Rules, Sim, State

    table = dict(cards())
    table["Ice Trap - Amulet@item"] = dataclasses.replace(
        table["Ice Trap"], name="Ice Trap - Amulet@item", source="item")

    def seat(hand):
        p = Actor(name="W", school="ice", hp=1500, max_hp=1500, team=0,
                  norm_pips=7)
        p.hand = [table[n] for n in hand]
        foes = [Actor(name="Mob", school="fire", hp=900, max_hp=900, team=1,
                      flat_hit=40)]
        sim = Sim(cards=table, decklist=list(hand), school="ice",
                  boss=Boss("Mob", 900, "fire", 40), rules=Rules(),
                  player_hp=1500)
        return sim, State(p, foes)

    sim_a, a = seat(["Ice Trap", "Frost Beetle"])
    ledger = Ledger()
    ledger.add(measure_cast(sim_a, a, (table["Ice Trap"], 0)), [0])

    sim_b, b = seat(["Ice Trap - Amulet@item", "Frost Beetle"])
    board = ledger.apply(b, [0])
    assert board.enemies[0].wards, "the ally's trap has to be on this board"

    def hit(state):
        beetle = next(c for c in state.player.hand if c.name == "Frost Beetle")
        return predict_damage(sim_b, state, beetle, 0)

    one_trap = hit(board)
    doubled = copy.deepcopy(board)
    sim_b.cast(doubled, next(c for c in doubled.player.hand
                             if c.name == "Ice Trap - Amulet@item"), 0)
    assert hit(doubled) == pytest.approx(one_trap), \
        "an amulet Ice Trap is an Ice Trap"
    assert len(doubled.enemies[0].wards) == 2, "and it does go up"


def test_a_shared_ward_is_copied_not_handed_round():
    """A `Hanging` carries charges and is consumed in place; one object
    shared by four rollouts lets one wizard's hit spend a trap the other
    three are still counting on."""
    sim, state = wizard(hand=("Fire Trap", "Fire Cat"), board=((800, "ice"),))
    ledger = Ledger()
    ledger.add(measure_cast(sim, state, (cards()["Fire Trap"], 0)), [0])
    _, a = wizard(hand=("Fire Cat",), board=((800, "ice"),))
    _, b = wizard(hand=("Fire Cat",), board=((800, "ice"),))
    first = ledger.apply(a, [0]).enemies[0].wards[-1]
    second = ledger.apply(b, [0]).enemies[0].wards[-1]
    assert first is not second


# ------------------------------------------------------- what coordinating buys
def test_uncoordinated_wizards_both_go_for_the_same_mob():
    """The baseline, and it is not a straw man: `focus_target` is
    deterministic and both wizards run the same arithmetic on the same
    board, so identical wizards make identical decisions."""
    _, solo = plan(n=2, passes=0)
    assert len({(m.card, m.target) for m in solo.moves}) == 1


def test_the_party_splits_up_instead():
    _, joint = plan(n=2, passes=2)
    assert joint.retargets >= 1
    moves = {(m.card, m.target) for m in joint.moves}
    assert len(moves) == 2, moves


def test_nobody_fires_into_a_mob_the_party_has_already_killed():
    """One 50 HP mob, two wizards holding a hit that kills it. The second
    hit buys nothing at all, and the card is worth more in hand."""
    _, solo = plan(n=2, passes=0, board=((50, "ice"),))
    assert all(m.card for m in solo.moves), "both should cast, uncoordinated"

    _, joint = plan(n=2, passes=2, board=((50, "ice"),))
    assert joint.saved == 1
    assert sum(1 for m in joint.moves if m.card) == 1
    assert any(m.note == "held" for m in joint.moves)


def test_a_full_circle_of_four_spreads_over_the_board():
    """Four wizards, four mobs. Uncoordinated they all pick one."""
    board = ((60, "ice"), (300, "ice"), (600, "ice"), (900, "ice"))
    _, solo = plan(n=4, passes=0, board=board)
    _, joint = plan(n=4, passes=2, board=board)
    assert len({m.target for m in solo.moves}) == 1
    assert len({(m.card, m.target) for m in joint.moves}) > 1


def test_coordination_is_off_at_passes_zero_to_the_last_decision():
    """`passes=0` has to be *exactly* uncoordinated, or the A/Bs above
    are measuring against something that is already half a hivemind."""
    subs_a = party(2)
    subs_b = party(2)
    a, _ = Hivemind(passes=0).plan(subs_a)
    b = {s.seat: s.policy(s.sim, s.state) for s in subs_b}
    for seat in a:
        card_a, target_a = (a[seat] if isinstance(a[seat], tuple)
                            else (a[seat], 0))
        card_b, target_b = (b[seat] if isinstance(b[seat], tuple)
                            else (b[seat], 0))
        assert getattr(card_a, "name", None) == getattr(card_b, "name", None)
        assert target_a == target_b


def test_a_party_of_one_decides_exactly_as_it_would_alone():
    subs = party(1)
    solo = subs[0].policy(subs[0].sim, subs[0].state)
    actions, _ = Hivemind(passes=2).plan(party(1))
    assert getattr(actions[0][0], "name", None) == \
        getattr(solo[0] if isinstance(solo, tuple) else solo, "name", None)


def test_the_party_wastes_less_damage_than_four_soloists():
    """The summary measurement: how much committed damage lands past a
    mob's health. That is the whole cost of not coordinating, and it is
    the number the Party tab is reporting in words."""
    def overkill(passes, board):
        _, out = plan(n=3, passes=passes, board=board)
        spent = {}
        for move in out.moves:
            if move.target is not None:
                spent[move.target] = spent.get(move.target, 0.0) + move.damage
        health = {i: hp for i, (_n, hp, _m) in enumerate(out.board)}
        return sum(max(0.0, v - health.get(k, 0.0)) for k, v in spent.items())

    boards = [((40, "ice"), (500, "ice")),
              ((60, "ice"), (60, "ice"), (700, "ice")),
              ((120, "ice"), (900, "ice"))]
    loose = sum(overkill(0, b) for b in boards)
    tight = sum(overkill(2, b) for b in boards)
    assert tight <= loose


# -------------------------------------------------------------------- barrier
def test_two_seats_on_one_loop_both_get_their_move():
    hive = Hivemind(passes=2, timeout=5.0)
    subs = party(2)
    for sub in subs:
        hive.join(sub.seat, sub.name)
        hive.enter_combat(sub.seat)

    async def drive():
        return await asyncio.gather(*[
            hive.decide(s.seat, s.sim, s.state, s.policy) for s in subs])

    got = asyncio.run(drive())
    assert len(got) == 2
    assert hive.rounds == 1
    assert hive.last_plan is not None and len(hive.last_plan.moves) == 2


def test_a_wizard_that_never_turns_up_is_waited_out_not_waited_on():
    """A hung client must cost one timeout, not the run. The seat that
    did arrive still gets a move."""
    hive = Hivemind(passes=1, timeout=0.2)
    subs = party(2)
    for sub in subs:
        hive.join(sub.seat, sub.name)
        hive.enter_combat(sub.seat)

    async def drive():
        s = subs[0]
        return await hive.decide(s.seat, s.sim, s.state, s.policy)

    assert asyncio.run(drive()) is not None
    assert len(hive.last_plan.moves) == 1


def test_leaving_the_circle_releases_the_others_immediately():
    """A wizard whose duel ended must stop being waited for, or every
    remaining round pays the full barrier timeout."""
    hive = Hivemind(passes=1, timeout=30.0)
    subs = party(2)
    for sub in subs:
        hive.join(sub.seat, sub.name)
        hive.enter_combat(sub.seat)

    async def drive():
        s = subs[0]
        task = asyncio.ensure_future(
            hive.decide(s.seat, s.sim, s.state, s.policy))
        await asyncio.sleep(0.05)
        hive.leave_combat(subs[1].seat)         # its fight ended
        return await asyncio.wait_for(task, 10)

    assert asyncio.run(drive()) is not None


def test_a_seat_deciding_alone_is_not_left_waiting_for_a_party():
    """Nobody has entered combat, so there is nobody to wait for."""
    hive = Hivemind(passes=2, timeout=30.0)
    sub = party(1)[0]
    hive.join(sub.seat, sub.name)

    async def drive():
        return await asyncio.wait_for(
            hive.decide(sub.seat, sub.sim, sub.state, sub.policy), 5)

    assert asyncio.run(drive()) is not None


def test_a_policy_that_raises_costs_its_own_round_and_nobody_elses():
    hive = Hivemind(passes=1, timeout=5.0)
    subs = party(2)

    def boom(sim, state):
        raise RuntimeError("the board would not read")

    subs[0].policy = boom
    for sub in subs:
        hive.join(sub.seat, sub.name)
        hive.enter_combat(sub.seat)

    async def drive():
        return await asyncio.gather(*[
            hive.decide(s.seat, s.sim, s.state, s.policy) for s in subs])

    broken, working = asyncio.run(drive())
    assert broken is None                      # passes, as a policy error does
    assert working is not None


def test_planning_stops_at_its_budget_rather_than_the_games_timer():
    """A party plan that outlasts the game's own planning phase would
    lose the round it was trying to save."""
    hive = Hivemind(passes=8, budget=0.0)
    _, out = hive.plan(party(3))
    assert out.passes <= 1


# ------------------------------------------------------------- the backend leg
def test_the_backend_asks_the_coordinator_instead_of_the_policy():
    """The wiring, end to end: a backend with a hive installed must not
    call its policy directly, or none of the above ever reaches a fight."""
    from deimos_bridge.live_backend import WizAiBackend
    from deimos_bridge.mock_client import MockCard, MockCombat, MockMember

    asked = []

    class _Hive:
        rounds = 0

        async def decide(self, seat, sim, state, policy, read=None,
                         rate=0.0):
            asked.append((seat, rate))
            return policy(sim, state)

        def last_move(self, seat):
            return None

    backend = WizAiBackend.from_trained(
        school="fire", deck=["Fire Cat"] * 4, cards=cards(),
        policy=greedy_ttk(), policy_name="ttk-lookahead",
        seat=2, coordinator=_Hive())
    backend.attach_combat(MockCombat(
        [MockMember("Wizard", 900, client=True, team_id=0, normal_pips=4),
         MockMember("Lost Soul", 450, monster=True, team_id=1)],
        [MockCard("Fire Cat")]))
    decision = asyncio.run(backend.decide())
    # Seat, and the damage rate the coordinator hands the other seats'
    # rollouts. A backend with no `damage_rate` hook reports zero rather
    # than guessing, which is what a headless one and a solo one want.
    assert asked == [(2, 0.0)]
    assert not decision.passing, decision.reason
    assert "party" in decision.policy


def test_a_held_card_is_recorded_as_held_rather_than_as_a_bare_pass():
    """"policy chose to pass" and "the other three have this dead" are
    different events, and only the second is coordination working."""
    from deimos_bridge.hivemind import SeatMove
    from deimos_bridge.live_backend import WizAiBackend
    from deimos_bridge.mock_client import MockCard, MockCombat, MockMember

    class _Hive:
        async def decide(self, seat, sim, state, policy, read=None,
                         rate=0.0):
            return None

        def last_move(self, seat):
            return SeatMove(seat=0, name="wizard 1", note="held")

    backend = WizAiBackend.from_trained(
        school="fire", deck=["Fire Cat"] * 4, cards=cards(),
        policy=greedy_ttk(), policy_name="ttk-lookahead",
        coordinator=_Hive())
    backend.attach_combat(MockCombat(
        [MockMember("Wizard", 900, client=True, team_id=0, normal_pips=4),
         MockMember("Lost Soul", 40, monster=True, team_id=1)],
        [MockCard("Fire Cat")]))
    decision = asyncio.run(backend.decide())
    assert decision.passing
    assert "already has this board dead" in decision.reason


def test_a_casting_bosss_output_is_split_across_the_party():
    """`_estimate_incoming` reads THIS wizard's health drop, so a boss
    casting one Wraith a round in a circle of four spreads it across four
    health bars. Subtracting its whole modelled output from one wizard's
    share drives the flat mobs to the floor and reads a dangerous board
    as a harmless one."""
    from deimos_bridge.live_backend import WizAiBackend
    from w101_sim import Actor, State

    def share(party_size):
        backend = WizAiBackend.from_trained(
            school="fire", deck=[], cards=cards(),
            policy=lambda sim, s: None, party_size=party_size)
        boss = Actor(name="Boss", school="death", hp=900, max_hp=900, team=1)
        boss.spell_pool = ["Dark Sprite", "Ghoul"]
        boss.power_pip_chance = 0.5
        minion = Actor(name="Minion", school="death", hp=200, max_hp=200,
                       team=1)
        minion.flat_hit = 120.0

        class _Read:
            state = State(Actor(name="W", school="fire", hp=900, max_hp=900,
                                team=0), [boss, minion])
        backend._measured_incoming = 60.0
        backend._apportion_incoming(_Read())
        return minion.flat_hit

    assert share(4) > share(1)


@pytest.mark.parametrize("seats", [1, 2, 3, 4])
def test_every_party_size_produces_a_move_for_every_seat(seats):
    actions, out = plan(n=seats, passes=2,
                        board=((80, "ice"), (400, "ice"), (900, "ice")))
    assert set(actions) == set(range(seats))
    assert len(out.moves) == seats
    assert out.seconds >= 0.0
    assert KILL_CONFIDENCE > 0.5


# ------------------------------------------------- keeping the party together
class _Pos:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


class _Body:
    def __init__(self, pos):
        self._pos = pos

    async def position(self):
        return self._pos


class _Mouse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    """The three things `party.follow` reads, and the one it calls."""

    def __init__(self, zone="Unicorn Way", pos=(0, 0, 0), fighting=False):
        self._zone = zone
        self.body = _Body(_Pos(*pos))
        self._fighting = fighting
        self.teleports = []
        self.mouse_handler = _Mouse()

    async def zone_name(self):
        return self._zone

    async def in_battle(self):
        return self._fighting

    async def teleport(self, target):
        self.teleports.append(target)
        self.body = _Body(target)


def test_a_follower_already_beside_its_leader_stays_put():
    """The tick runs twice a second; a follower that re-teleported every
    time would spend the fight teleporting instead of fighting."""
    from deimos_bridge import party

    leader = _FakeClient(pos=(100, 100, 0))
    follower = _FakeClient(pos=(140, 120, 0))
    moved, why = asyncio.run(party.follow(follower, leader))
    assert moved is False and why == ""
    assert follower.teleports == []


def test_a_follower_left_behind_teleports_onto_the_leader():
    from deimos_bridge import party

    leader = _FakeClient(pos=(5000, 0, 0))
    follower = _FakeClient(pos=(0, 0, 0))
    moved, why = asyncio.run(party.follow(follower, leader))
    assert moved is True and "regrouped" in why
    assert len(follower.teleports) == 1
    assert follower.teleports[0].x == 5000


def test_a_follower_in_its_own_duel_is_left_alone():
    """It may be in the leader's duel, and the game does not let you
    teleport out of a fight anyway."""
    from deimos_bridge import party

    leader = _FakeClient(pos=(5000, 0, 0))
    follower = _FakeClient(pos=(0, 0, 0), fighting=True)
    moved, _why = asyncio.run(party.follow(follower, leader))
    assert moved is False and follower.teleports == []


def test_a_different_zone_is_not_chased_with_a_position_teleport():
    """An XYZ teleport cannot change zone -- it silently does nothing,
    which is the worst of both answers."""
    from deimos_bridge import party

    leader = _FakeClient(zone="Triton Avenue", pos=(5000, 0, 0))
    follower = _FakeClient(zone="Unicorn Way", pos=(0, 0, 0))
    moved, why = asyncio.run(party.follow(follower, leader))
    assert moved is False
    assert "wizard name is not known" in why
    assert follower.teleports == []


def test_a_cross_zone_follow_uses_the_friends_list_when_it_has_a_name(
        monkeypatch):
    from deimos_bridge import party

    asked = {}

    async def _friends_list(follower, leader_name):
        asked["name"] = leader_name
        return True, ""

    monkeypatch.setattr(party, "teleport_to_leader_across_zones",
                        _friends_list)
    moved, why = asyncio.run(party.follow(
        _FakeClient(zone="Unicorn Way"),
        _FakeClient(zone="Triton Avenue"), leader_name="Wolf Deathblade"))
    assert moved is True and "Triton Avenue" in why
    assert asked["name"] == "Wolf Deathblade"


def test_arriving_beside_a_fighting_leader_is_not_joining_it(monkeypatch):
    """Wizard101 puts you in the circle only when you touch a sigil or a
    mob. A follower standing next to the circle looks exactly like a
    working party right up until the plan says 'one wizard'."""
    from deimos_bridge import party

    stepped = []

    async def _join(follower):
        stepped.append(follower)
        return True, ""

    monkeypatch.setattr(party, "join_the_fight", _join)
    leader = _FakeClient(pos=(5000, 0, 0), fighting=True)
    follower = _FakeClient(pos=(0, 0, 0))
    moved, why = asyncio.run(party.follow(follower, leader))
    assert moved is True and "joined" in why
    assert stepped == [follower]


def test_a_fighting_leader_is_joined_even_from_right_beside_it(monkeypatch):
    """Standing in range is not being in the duel, so the distance check
    must not short-circuit the sigil step.

    And the follower lands ON the leader first, close or not. This test
    used to assert the opposite -- no teleport, because it was already
    there -- and that was the bug: `join_the_fight` steps in via
    `tp_to_closest_mob`, and closest is measured from wherever the
    follower is STANDING. A follower inside the radius but beside a
    different group walked into that group's fight, and the party spent
    the duel in two duels. Landing on the leader makes "closest" mean
    the leader's circle.
    """
    from deimos_bridge import party

    stepped = []

    async def _join(follower):
        stepped.append(follower)
        return True, ""

    monkeypatch.setattr(party, "join_the_fight", _join)
    leader = _FakeClient(pos=(100, 100, 0), fighting=True)
    follower = _FakeClient(pos=(110, 100, 0))
    moved, _why = asyncio.run(party.follow(follower, leader))
    assert moved is True and stepped == [follower]
    assert [(p.x, p.y, p.z) for p in follower.teleports] == [(100, 100, 0)], \
        "the follower stepped in from where it happened to be standing"


def test_an_unreadable_leader_is_reported_rather_than_chased_to_zero():
    """A position that will not read used to be a teleport to (0,0,0),
    which walks the follower off the map."""
    from deimos_bridge import party

    class _Blind(_FakeClient):
        async def zone_name(self):
            return "Unicorn Way"

    leader = _Blind()
    leader.body = None
    follower = _FakeClient()
    moved, why = asyncio.run(party.follow(follower, leader))
    assert moved is False and "leader's position" in why
    assert follower.teleports == []


# ------------------------------------------ what the first live party run said
def test_the_round_one_incoming_prior_is_split_across_the_party():
    """From the first live party run. The solo wizard's prior read 57 per
    enemy against 78-84 actually measured — right, and slightly low. The
    two party wizards' priors read 85 and 58 against 15-30 and 36-53:
    over by up to five times, on the one round that has nothing measured
    yet to correct it."""
    from deimos_bridge.live_backend import WizAiBackend
    from w101_sim import Actor, State

    def prior(party_size, max_hp=1022):
        backend = WizAiBackend.from_trained(
            school="ice", deck=[], cards={}, policy=lambda sim, s: None,
            party_size=party_size)
        player = Actor(name="W", school="ice", hp=max_hp, max_hp=max_hp,
                       team=0)
        foe = Actor(name="Mob", school="death", hp=690, max_hp=690, team=1)

        class _Read:
            state = State(player, [foe])
            round_number = 1
        return backend._estimate_incoming(_Read())

    assert prior(1) == pytest.approx(1022 / 12)
    assert prior(2) == pytest.approx(1022 / 24)
    # The 30/round floor still holds, however big the party: a living mob
    # is never harmless, and dividing past that would model one.
    assert prior(4) == 30.0
    assert prior(4, max_hp=300) == 30.0

    # And it lands nearer the truth. Wizard 1 of that run measured 15-30
    # per enemy once the fight had rounds to count; the solo prior said
    # 85, and it is only round one that has to guess.
    measured = 22.0
    assert abs(prior(2) - measured) < abs(prior(1) - measured)


def test_the_party_prior_buys_the_rollout_more_rounds_to_work_with():
    """Wizard 2's round 1 in that run scored EVERY candidate at 14.0
    turns and 235 damage — the horizon sentinel, meaning "no line
    survives" — because a board dealing an imagined 115/round killed it
    inside the horizon on every line, and the comparison collapsed to the
    pip tiebreak.

    The prior is what was wrong, so the prior is what this checks: how
    many rounds the rollout thinks the wizard has. Whether that clears
    the sentinel depends on the board — on this one, at 691 health
    against two mobs, even the floored party prior still kills it inside
    twelve rounds, and saying so is not wrong. It is the factor of two
    between 6 rounds and 12 that was.
    """
    from deimos_bridge.live_backend import WizAiBackend
    from w101_sim import Actor, State

    def rounds_to_live(party_size):
        backend = WizAiBackend.from_trained(
            school="fire", deck=[], cards={}, policy=lambda sim, s: None,
            party_size=party_size)
        player = Actor(name="W", school="fire", hp=691, max_hp=691, team=0)
        foes = [Actor(name="Lord Nightshade", school="death", hp=690,
                      max_hp=690, team=1),
                Actor(name="Field Guard", school="balance", hp=395,
                      max_hp=395, team=1)]

        class _Read:
            state = State(player, foes)
            round_number = 1
        per_enemy = backend._estimate_incoming(_Read())
        return player.max_hp / (per_enemy * len(foes))

    alone = rounds_to_live(1)
    pair = rounds_to_live(2)
    assert alone == pytest.approx(6.0, abs=0.1)     # the run's own number
    # 691/24 is 28.8, under the 30/round floor, so the pair lands on the
    # floor rather than on the halving -- a living mob is never modelled
    # as harmless. Still nearly twice the rounds to work with.
    assert pair == pytest.approx(11.5, abs=0.1)
    assert pair > alone


def test_only_one_candidate_is_the_chosen_one():
    """Wizard 1's round 4 exported Frost Beetle @0 AND 'pass' both
    flagged chosen, so the decision matrix rendered two winners for one
    round."""
    from w101_sim import Actor, Boss, Rules, Sim, State

    table = cards()
    hand = ["Frost Beetle", "Snow Serpent", "Evil Snowman"]
    player = Actor(name="W", school="ice", hp=841, max_hp=1022, team=0,
                   norm_pips=4)
    player.hand = [table[n] for n in hand]
    foes = [Actor(name="Lord Nightshade", school="death", hp=690, max_hp=690,
                  team=1, flat_hit=30),
            Actor(name="Field Guard", school="balance", hp=395, max_hp=395,
                  team=1, flat_hit=30)]
    sim = Sim(table, hand, "ice",
              Boss(name="Lord Nightshade", hp=690, school="death", dmg=30),
              enemies=[Boss(name="Field Guard", hp=395, school="balance",
                            dmg=30)],
              rules=Rules(), player_hp=1022)
    policy = greedy_ttk()
    policy(sim, State(player, foes))
    chosen = [c for c in policy.last_candidates if c.chosen]
    assert len(chosen) == 1, [(c.card, c.target) for c in chosen]


def test_a_mob_two_wizards_hit_is_not_a_damage_measurement():
    """`_settle` differences the board, and in a party that delta is the
    party's damage. Silent before parties existed, because there was
    nobody else to confuse it with."""
    from deimos_bridge.hivemind import PartyPlan, SeatMove
    from deimos_bridge.telemetry import Telemetry

    class _Decision:
        card_name = "Frost Beetle"
        target_index = 0
        target_kind = "enemy"
        passing = False
        reason = ""
        policy = "ttk-lookahead"
        candidates = ()

    def read(hp):
        from w101_sim import Actor, State

        class _Resolver:
            misses = {}
        me = Actor(name="W", school="ice", hp=841, max_hp=1022, team=0)
        foe = Actor(name="Lord Nightshade", school="death", hp=hp,
                    max_hp=690, team=1)

        class _Read:
            state = State(me, [foe])
            hand_cards = {"Frost Beetle": []}
            resolver = _Resolver()
            round_number = 1
            hand_visibility = 1.0
            hidden = ()
            unreadable = ()
        return _Read()

    # `damage` is what the coordinator scored each move at, and it is
    # load-bearing: a move it priced at zero moves no health, so a
    # teammate's trap or shield is not a reason this cast's residual is
    # off. Only a damaging cast contaminates the measurement.
    plan = PartyPlan(moves=[
        SeatMove(seat=0, name="wizard 1", card="Frost Beetle", target=0,
                 target_name="Lord Nightshade", damage=85.0),
        SeatMove(seat=1, name="wizard 2", card="Fire Cat", target=0,
                 target_name="Lord Nightshade", damage=76.0)])

    tel = Telemetry()
    tel.start_fight()
    rec = tel.observe(_Decision(), read(690), party=plan, seat=0)
    assert rec.party_hits == {"Lord Nightshade": "wizard 2"}
    rec.predicted_damage = 85.0
    tel.observe(_Decision(), read(529), party=plan, seat=0)

    assert rec.actual_damage == 161.0        # the pair's damage, not the cast's
    assert rec.clean is False
    assert any("wizard 2 also hit" in c for c in rec.confounds), rec.confounds
    # ...so it is kept out of the damage model's statistics entirely.
    assert tel.damage_observations() == []


def test_collateral_damage_names_the_wizard_that_caused_it():
    """Three of wizard 1's rounds carried 'Field Guard also lost HP --
    AoE, DoT or a minion'. The cause was wizard 2."""
    from deimos_bridge.hivemind import PartyPlan, SeatMove
    from deimos_bridge.telemetry import _party_hits

    plan = PartyPlan(moves=[
        SeatMove(seat=0, name="wizard 1", card="Frost Beetle", target=0,
                 target_name="Lord Nightshade", damage=85.0),
        SeatMove(seat=1, name="wizard 2", card="Fire Elf", target=1,
                 target_name="Field Guard", damage=104.0)])
    assert _party_hits(plan, 0) == {"Field Guard": "wizard 2"}
    assert _party_hits(plan, 1) == {"Lord Nightshade": "wizard 1"}
    # A wizard that held its card moved no health and is not a cause.
    held = PartyPlan(moves=[SeatMove(seat=1, name="wizard 2", note="held")])
    assert _party_hits(held, 0) == {}
    # Nor is one whose card was aimed at an enemy but takes nothing off
    # it. A trap, a shield and a debuff all look like a cast at a mob,
    # and counting them marked 42 rounds of one live party run unclean
    # for nothing.
    trapped = PartyPlan(moves=[
        SeatMove(seat=1, name="wizard 2", card="Ice Trap", target=0,
                 target_name="Lord Nightshade", damage=0.0)])
    assert _party_hits(trapped, 0) == {}


def test_a_teammates_aoe_is_a_hit_on_every_mob_it_touched():
    """The one the name-keying could not see at all.

    An AoE aims at nothing in particular, so `_aims_at_an_enemy` gives
    it no index and `target_name` is empty -- and a lookup keyed on the
    name then reports that nobody hit anything. Every Meteor Strike
    round of the run at rev 8666bda7 did exactly that, and on one of
    them the ice wizard's Snow Serpent banked 987 damage against a 115
    prediction as a clean solo observation.
    """
    from deimos_bridge.hivemind import PartyPlan, SeatMove
    from deimos_bridge.telemetry import _party_expected, _party_hits

    plan = PartyPlan(moves=[
        SeatMove(seat=0, name="wizard 1", card="Snow Serpent", target=0,
                 target_name="Pops", damage=115.0, hit={"Pops": 115.0}),
        # no target, no target_name -- exactly what `plan` builds for one
        SeatMove(seat=1, name="wizard 2", card="Meteor Strike",
                 damage=877.0, hit={"Pops": 500.0, "Napper": 377.0})])
    assert _party_hits(plan, 0) == {"Pops": "wizard 2", "Napper": "wizard 2"}
    assert _party_expected(plan, "Pops") == 615.0     # 115 + 500, not 115
    assert _party_expected(plan, "Napper") == 377.0


def test_a_plan_without_the_per_mob_breakdown_still_reads():
    """`hit` is newer than `SeatMove`, and a plan built without it --
    an older export, a hand-made stub -- must still attribute by name
    rather than silently reporting nothing."""
    from deimos_bridge.hivemind import PartyPlan, SeatMove
    from deimos_bridge.telemetry import _party_expected, _party_hits

    plan = PartyPlan(moves=[
        SeatMove(seat=1, name="wizard 2", card="Fire Elf", target=0,
                 target_name="Pops", damage=104.0)])
    assert _party_hits(plan, 0) == {"Pops": "wizard 2"}
    assert _party_expected(plan, "Pops") == 104.0


def test_the_coordinator_fills_in_what_each_cast_touched():
    """End to end, off a real plan rather than a stub: an AoE's entry
    has to name every living mob."""
    subs = party(2, hand=("Meteor Strike", "Fire Cat"),
                 board=((400, "ice"), (400, "ice")))
    _actions, plan_out = Hivemind(passes=1).plan(subs)
    aoe = [m for m in plan_out.moves if m.card == "Meteor Strike"]
    if not aoe:
        pytest.skip("neither seat chose the AoE on this board")
    assert len(aoe[0].hit) == 2, aoe[0].hit
    assert all(v > 0 for v in aoe[0].hit.values())


# --------------------------------- what the first NAMED party run said
def test_a_record_never_holds_two_wizards():
    """The window keeps one record per SEAT and reuses it across Play
    live presses, but the clients come back in whatever order the game
    was launched in — so seat 1 can be one wizard on one run and another
    on the next. The first named party run's export was two rounds of a
    1,053 HP ice wizard followed by six of a 713 HP fire wizard, filed
    under one name."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from deimos_bridge.gui.live import LiveWorker
    from deimos_bridge.telemetry import Telemetry

    tel = Telemetry()
    tel.wizard = "Jeffrey"
    tel.start_fight()
    tel.rounds.append(object())
    tel.rounds.append(object())

    w = LiveWorker(tel, "ice", [], "ttk-lookahead", 1)
    said = []
    w.status = type("S", (), {"emit": staticmethod(said.append)})()

    class _Read:
        state = type("S", (), {
            "player": type("P", (), {"name": "Konstantin"})()})()

    w._learn_name(w.seats[0], _Read())
    assert tel.wizard == "Konstantin"
    assert tel.rounds == []                  # Jeffrey's rounds are gone
    assert any("was Jeffrey last run and is Konstantin now" in m
               for m in said), said

    # The same wizard coming back is not a different one.
    tel.rounds.append(object())
    w.seats[0].wizard_name = None
    w._learn_name(w.seats[0], _Read())
    assert len(tel.rounds) == 1


def test_the_hand_names_the_school_when_the_client_will_not():
    """`primary_magic_school_id` reported every enemy's school correctly
    on the live runs and gave nothing for the wizard's own member, so the
    crossed seats went undetected twice. A wizard holding Frost Beetle,
    Ice Trap, Snow Serpent and Evil Snowman is an ice wizard, and reading
    that off the cards is how the crossing was spotted by eye."""
    from deimos_bridge.live_backend import WizAiBackend
    from w101_sim import Actor, State

    table = cards()

    def guess(hand, configured="fire"):
        backend = WizAiBackend.from_trained(
            school=configured, deck=[], cards=table,
            policy=lambda sim, s: None)
        player = Actor(name="W", school=configured, hp=900, max_hp=900,
                       team=0)
        player.hand = [table[n] for n in hand]

        class _Read:
            state = State(player, [Actor(name="Mob", school="death", hp=1,
                                         max_hp=1, team=1)])
        return backend._school_from_hand(_Read())

    assert guess(["Frost Beetle", "Snow Serpent", "Evil Snowman"]) == "ice"
    # Buffs do not vote: a Balanceblade is a balance card in an ice deck.
    assert guess(["Frost Beetle", "Snow Serpent", "Evil Snowman",
                  "Balanceblade"]) == "ice"
    # A starter wand's off-school card does not overturn the majority...
    assert guess(["Frost Beetle", "Snow Serpent", "Evil Snowman",
                  "Frost Beetle", "Thunder Snake"]) == "ice"
    # ...but a genuinely mixed hand is not evidence of anything.
    assert guess(["Frost Beetle", "Thunder Snake", "Fire Cat"]) == ""
    # Nor is a two-card opening hand that happens to agree with itself.
    assert guess(["Frost Beetle", "Snow Serpent"]) == ""


def test_a_crossed_seat_is_caught_from_the_hand_alone():
    """End to end through the real backend: no school id anywhere, and
    the mismatch is still found."""
    from deimos_bridge.live_backend import WizAiBackend
    from deimos_bridge.mock_client import MockCard, MockCombat, MockMember

    seen = []
    backend = WizAiBackend.from_trained(
        school="fire", deck=["Frost Beetle"] * 4, cards=cards(),
        policy=lambda sim, s: None, policy_name="ttk-lookahead")
    backend.on_school_mismatch = seen.append
    backend.attach_combat(MockCombat(
        # no school_id: the accessor is simply absent, as it was live
        [MockMember("Konstantin", 1053, client=True, team_id=0,
                    normal_pips=4),
         MockMember("Biti Nirini", 510, monster=True, team_id=1)],
        [MockCard("Frost Beetle"), MockCard("Snow Serpent"),
         MockCard("Evil Snowman")]))
    asyncio.run(backend.decide())
    assert seen == ["ice"], seen


def test_a_fizzle_is_not_a_measurement_of_the_damage_model():
    """A Snow Serpent predicted at 294 into a 249 HP mob that was still
    at 249 the next round, recorded clean — and it was the run's ONLY
    observation, so the whole reported model quality was one coin flip.
    The prediction is computed through `_NoFizzle`: it is what the cast
    does when it lands, not how often it lands."""
    from deimos_bridge.telemetry import Telemetry

    class _Decision:
        card_name = "Snow Serpent"
        target_index = 0
        target_kind = "enemy"
        passing = False
        reason = ""
        policy = "ttk-lookahead"
        candidates = ()

    def read(hp):
        from w101_sim import Actor, State

        class _Resolver:
            misses = {}

        class _Read:
            state = State(Actor(name="Jeffrey", school="ice", hp=486,
                                max_hp=1053, team=0),
                          [Actor(name="Biti Nirini", school="fire", hp=hp,
                                 max_hp=510, team=1)])
            hand_cards = {"Snow Serpent": []}
            resolver = _Resolver()
            round_number = 5
            hand_visibility = 1.0
            hidden = ()
            unreadable = ()
        return _Read()

    tel = Telemetry()
    tel.start_fight()
    rec = tel.observe(_Decision(), read(249))
    rec.predicted_damage = 294.0
    tel.observe(_Decision(), read(249))          # it did not move

    assert rec.actual_damage == 0.0
    assert rec.clean is False
    assert any("fizzle" in c for c in rec.confounds), rec.confounds
    assert tel.damage_observations() == []

    # A cast that lands for less than predicted is still a real
    # measurement, and must not be swept up by this.
    tel2 = Telemetry()
    tel2.start_fight()
    landed = tel2.observe(_Decision(), read(249))
    landed.predicted_damage = 294.0
    tel2.observe(_Decision(), read(200))
    assert landed.actual_damage == 49.0
    assert landed.clean is True


# ------------------------------------------------- the party through the horizon
def test_a_lone_wizard_cannot_clear_the_board_the_party_clears_easily():
    """The defect, stated as the arithmetic that produces it.

    Three 435hp Sand Stalkers is 1305 health. A level-six wizard removes
    about 87 a round, so alone it needs fifteen turns and the rollout's
    horizon is twelve -- every candidate comes back a sentinel, `turns`
    ties for all of them, and the decision falls through to a tiebreak
    that cannot tell a Tower Shield from passing. Two wizards clear the
    same board in nine.
    """
    from deimos_bridge.policies import greedy_ttk, set_ally_rate

    board = ((435, "balance"), (435, "balance"), (435, "balance"))
    sim, state = wizard(school="ice", hand=("Frost Beetle", "Snow Serpent",
                                            "Ice Trap", "Tower Shield"),
                        pips=2, hp=1270, board=board)
    alone = greedy_ttk()
    alone(sim, state)
    horizon = alone.last_candidates[0].horizon
    assert all(c.turns > horizon for c in alone.last_candidates), \
        "this board is supposed to be out of reach alone"

    # Asserted on the rollout rather than on `last_candidates`, and the
    # difference is the point. With a teammate hitting, this board IS
    # reachable -- but every candidate reaches it on the same turn with
    # the same banked damage, passing included, because the flat rate
    # clears it whatever the wizard does. That total tie is the
    # free-rider case below, so the recorded candidate list is the
    # party-blind re-score and no longer shows these numbers. The claim
    # this test exists for is about the rollout, and here it is.
    from deimos_bridge import policies

    sim, state = wizard(school="ice", hand=("Frost Beetle", "Snow Serpent",
                                            "Ice Trap", "Tower Shield"),
                        pips=2, hp=1270, board=board)
    with policies.party_rate(120.0):
        reached = [policies._rollout(sim, state, c, horizon, 0)
                   for c in state.player.hand]
    assert any(t <= horizon for t, _ in reached), \
        "with a teammate hitting, something has to kill inside the horizon"

    # And the move it settles on is still an attack, which is the only
    # thing the fight cares about.
    sim, state = wizard(school="ice", hand=("Frost Beetle", "Snow Serpent",
                                            "Ice Trap", "Tower Shield"),
                        pips=2, hp=1270, board=board)
    together = greedy_ttk()
    try:
        set_ally_rate(120.0)
        choice = together(sim, state)
    finally:
        set_ally_rate(0.0)
    card = choice[0] if isinstance(choice, tuple) else choice
    assert getattr(card, "name", None) in ("Frost Beetle", "Snow Serpent"), \
        f"a teammate is no reason to play a shield: {card}"


def test_the_ally_rate_is_off_unless_somebody_turns_it_on():
    """Every solo path, test and tuning sweep has to be untouched."""
    from deimos_bridge import policies

    assert policies.ally_rate() == 0.0
    with policies.party_rate(90.0):
        assert policies.ally_rate() == 90.0
    assert policies.ally_rate() == 0.0


def test_a_seat_that_raises_does_not_leave_its_rate_on_the_next_one():
    from deimos_bridge import policies

    with pytest.raises(ValueError):
        with policies.party_rate(90.0):
            raise ValueError("a policy blew up mid-decision")
    assert policies.ally_rate() == 0.0


def test_allies_finish_the_weakest_mob_rather_than_spreading_out():
    """What the circle demonstrably does -- the ledger writes a mob the
    committed casts kill down to zero and the overkill guard stops the
    next wizard firing into it. Spreading the rate evenly would model a
    party that never kills anything, which is the failure being fixed."""
    from w101_sim import Actor
    from deimos_bridge.policies import _allies_hit

    class _Board:
        enemies = [Actor(name="big", school="ice", hp=400, max_hp=400, team=1),
                   Actor(name="small", school="ice", hp=50, max_hp=50, team=1)]

    _allies_hit(_Board(), 120.0)
    assert not _Board.enemies[1].alive          # the 50 died first
    assert _Board.enemies[0].hp == 330.0        # the other 70 spilled over


# ------------------------------------------------- healing when hurt
#
# The observation is real: over the run at rev 8666bda7 the ice wizard
# held a Pixie for 44 rounds and cast it never, five of those below half
# health, and the fire wizard walked into his eighth fight on 53% and
# the client read him at zero when it ended.
#
# The obvious fix was tried and MEASURED AS HARMFUL, which is why there
# is a test here and no feature. `died()` ranks a losing line by kills
# and banked damage, so crediting it for dying later looks like exactly
# the missing health term -- and it does produce healing: at 60/40/17%
# health on a 900hp board the wizard cannot clear, the same decision
# goes Ice Trap -> Pixie. But paired against the shipped behaviour on a
# brutal fire board it loses 8.7, 10.0 and 12.3 points of kill rate on
# three independent seed streams at n=600. Scaling the credit by how
# hurt the wizard already is does not rescue it, because on those boards
# the wizard IS hurt.
#
# The reason is in the ranking that is already there. `stalled` outranks
# `died` by a whole point, so a heal that genuinely SAVES the wizard
# already wins on its own -- and the credit only ever fires on the
# remaining case, a heal that delays a death it cannot prevent. There,
# racing is better, and the arithmetic says so three times over.
#
# What did happen to the fire wizard is upstream of the fight: he
# started it on 53%. That is `_health_needed` below.

def test_a_heal_at_full_health_is_not_even_offered():
    """`_is_inert` has the first word and keeps it: a Pixie at full
    health restores exactly nothing and must not win anything."""
    from deimos_bridge.policies import greedy_ttk

    sim, state = wizard(school="ice", hand=("Pixie", "Ice Trap",
                                            "Frost Beetle"),
                        pips=4, hp=1500, board=((900, "balance"),))
    policy = greedy_ttk()
    policy(sim, state)
    assert not any(c.card == "Pixie" for c in policy.last_candidates)


def test_a_heal_that_actually_saves_the_wizard_already_wins():
    """The case the credit was reaching for is covered without it: a
    Pixie that carries the wizard past the horizon moves the line from
    `died` to `stalled`, which is a whole rank better."""
    from deimos_bridge import policies

    sim, state = wizard(school="ice", hand=("Pixie", "Ice Trap",
                                            "Frost Beetle"),
                        pips=4, hp=1500, board=((150, "balance"),))
    state.player.hp = 250.0
    for e in state.enemies:
        e.flat_hit = 40
    scores = {c.name: policies._rollout(sim, state, c, 12, 0)
              for c in state.player.hand}
    assert scores["Pixie"][0] < scores["Ice Trap"][0], scores
    choice = policies.greedy_ttk()(sim, state)
    card, _t = choice if isinstance(choice, tuple) else (choice, None)
    assert card.name == "Pixie"


def test_the_floor_for_the_next_fight_is_what_the_fights_have_cost():
    """0.35 was chosen against "essentially 0". A fight costs far more:
    the fire wizard's took 191 to 630 out of 1,196, so he started his
    eighth on 53% and did not finish it."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from deimos_bridge.gui.live import LiveWorker
    from deimos_bridge.telemetry import FightRecord, Telemetry

    worker = LiveWorker(Telemetry(), "fire", [], "ttk-lookahead", 1)
    seat = worker.seats[0]
    seat.max_hp = 1196

    # nothing fought yet: the old constant, unchanged
    assert worker._health_needed(seat) == worker.LOW_HEALTH

    # a fight that cost 630 of 1196 -- 53%
    seat.tel.fights.append(FightRecord(index=1, rounds=12,
                                       damage_taken=630.0))
    assert worker._health_needed(seat) == pytest.approx(630 / 1196)

    # a cheap fight after it does not lower the floor while the hard one
    # is still in the window
    seat.tel.fights.append(FightRecord(index=2, rounds=4,
                                       damage_taken=191.0))
    assert worker._health_needed(seat) == pytest.approx(630 / 1196)

    # ...and one catastrophic pull cannot wedge the run forever
    seat.tel.fights.append(FightRecord(index=3, rounds=20,
                                       damage_taken=1190.0))
    assert worker._health_needed(seat) == worker.LOW_HEALTH_CAP


def test_the_floor_is_never_lower_than_what_shipped():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from deimos_bridge.gui.live import LiveWorker
    from deimos_bridge.telemetry import FightRecord, Telemetry

    worker = LiveWorker(Telemetry(), "fire", [], "ttk-lookahead", 1)
    seat = worker.seats[0]
    seat.max_hp = 1196
    seat.tel.fights.append(FightRecord(index=1, rounds=2, damage_taken=10.0))
    assert worker._health_needed(seat) == worker.LOW_HEALTH
    # an unreadable max health is no evidence either way
    seat.max_hp = 0
    assert worker._health_needed(seat) == worker.LOW_HEALTH


# ------------------------------------------- blading the other wizard
#
# The party's seats are not interchangeable: a level-10 ice wizard
# removes about 85 a round and the fire wizard beside it removes 200
# with one Meteor Strike. A +35% blade is worth 30 on the first and 70
# on the second -- and until now every seat could only ever put one on
# itself, so the small hitter spent rounds buffing the small hit.
#
# Priced on `rate`, the measured health-a-round each seat removes,
# rather than on this round's cast: Wizard101 resolves a round in pip
# order and the coordinator does not know that order, so a blade laid
# now may or may not be up before the teammate swings -- but it is
# certainly up for the swing after.

def _blade_party(hands, schools, rates, names=("Jeffrey", "Konstantin"),
                 pips=4):
    from w101_sim import Actor, Boss, Rules, Sim, State

    table = cards()
    subs = []
    for i, (hand, school, rate) in enumerate(zip(hands, schools, rates)):
        p = Actor(name=names[i], school=school, hp=1500, max_hp=1500, team=0,
                  norm_pips=pips)
        p.hand = [table[n] for n in hand]
        foes = [Actor(name="Mob", school="balance", hp=1200, max_hp=1200,
                      team=1, flat_hit=45)]
        sim = Sim(cards=table, decklist=list(hand), school=school,
                  boss=Boss("Mob", 1200, "balance", 45), rules=Rules(),
                  player_hp=1500)
        sub = _Submission(i, names[i], sim, State(p, foes), greedy_ttk())
        sub.rate = rate
        subs.append(sub)
    return subs


def test_an_off_school_blade_goes_to_the_wizard_who_can_use_it():
    """The headline, and the shape a real starter hand keeps producing:
    the ice wizard is holding a Fireblade. On itself it multiplies
    nothing at all -- `_consume_damage_charms` only applies a charm
    where `matches(school)` -- and on the fire wizard beside it, it
    multiplies every hit that wizard makes."""
    subs = _blade_party(hands=[("Fireblade", "Tower Shield"),
                               ("Fire Cat", "Fire Cat")],
                        schools=["ice", "fire"], rates=[40.0, 200.0])
    _actions, plan_out = Hivemind(passes=2).plan(subs)
    ice = next(m for m in plan_out.moves if m.seat == 0)
    assert ice.card == "Fireblade"
    assert ice.buff_ally == "Konstantin"
    assert plan_out.redirected == 1
    assert "buffing a teammate" in plan_out.summary()


def test_the_big_hitter_keeps_its_own_blade():
    """Nothing to redirect when the caster is already the right wizard.
    The rule ranks every seat INCLUDING itself, so this is not a special
    case -- it is the same comparison coming out the other way."""
    subs = _blade_party(hands=[("Fireblade", "Tower Shield"),
                               ("Fire Cat", "Fire Cat")],
                        schools=["fire", "fire"], rates=[200.0, 40.0])
    _actions, plan_out = Hivemind(passes=2).plan(subs)
    assert plan_out.redirected == 0
    assert not any(m.buff_ally for m in plan_out.moves)


def test_a_blade_no_teammate_can_use_is_not_moved():
    """`buff_applies` is the engine's own question. An ice wizard's ice
    blade beside a lone fire wizard has nowhere to go."""
    subs = _blade_party(hands=[("Frost Beetle", "Tower Shield"),
                               ("Fire Cat", "Fire Cat")],
                        schools=["ice", "fire"], rates=[40.0, 200.0])
    _actions, plan_out = Hivemind(passes=2).plan(subs)
    assert plan_out.redirected == 0


def test_a_party_of_one_never_redirects():
    subs = _blade_party(hands=[("Fireblade", "Tower Shield")],
                        schools=["ice"], rates=[40.0], names=("Jeffrey",))
    _actions, plan_out = Hivemind(passes=2).plan(subs)
    assert plan_out.redirected == 0


def test_a_seat_that_is_attacking_is_left_alone():
    """Trading a hit for a buff is the rollout's call, not this rule's.
    Asserted on `_redirect_blades` directly so the guard is tested
    rather than whatever the policy happened to choose."""
    from deimos_bridge.hivemind import CastEffect, _redirect_blades

    subs = _blade_party(hands=[("Fireblade", "Fire Cat"),
                               ("Fire Cat", "Fire Cat")],
                        schools=["fire", "fire"], rates=[40.0, 200.0])
    table = cards()
    actions = {0: (table["Fire Cat"], 0), 1: (table["Fire Cat"], 0)}
    effects = {0: CastEffect(card="Fire Cat", damage={0: 100.0}),
               1: CastEffect(card="Fire Cat", damage={0: 100.0})}
    notes = {0: "party", 1: "party"}
    moved = _redirect_blades(subs, actions, effects,
                             {0: 40.0, 1: 200.0}, notes)
    assert moved == 0
    assert notes[0] == "party"


def test_a_held_seat_is_not_given_a_move_by_the_redirect():
    """The overkill guard held it because the board is already dead.
    Handing it a blade puts a cast back into a round that was
    deliberately skipped."""
    from deimos_bridge.hivemind import CastEffect, _redirect_blades

    subs = _blade_party(hands=[("Fireblade", "Tower Shield"),
                               ("Fire Cat", "Fire Cat")],
                        schools=["ice", "fire"], rates=[40.0, 200.0])
    actions = {0: None, 1: (cards()["Fire Cat"], 0)}
    effects = {0: CastEffect(), 1: CastEffect(card="Fire Cat",
                                              damage={0: 100.0})}
    notes = {0: "held", 1: "party"}
    assert _redirect_blades(subs, actions, effects,
                            {0: 40.0, 1: 200.0}, notes) == 0
    assert actions[0] is None


def test_a_teammate_with_no_measured_rate_is_not_a_buff_target():
    """`rate` is zero before a seat has finished a fight. Buffing a
    wizard the coordinator knows nothing about is a guess, and the
    caster's own move is at least a decision."""
    from deimos_bridge.hivemind import CastEffect, _redirect_blades

    subs = _blade_party(hands=[("Fireblade", "Tower Shield"),
                               ("Fire Cat", "Fire Cat")],
                        schools=["ice", "fire"], rates=[0.0, 0.0])
    actions = {0: (cards()["Fireblade"], None), 1: (cards()["Fire Cat"], 0)}
    effects = {0: CastEffect(card="Fireblade"), 1: CastEffect(card="Fire Cat")}
    notes = {0: "party", 1: "party"}
    assert _redirect_blades(subs, actions, effects, {0: 0.0, 1: 0.0},
                            notes) == 0


def test_the_click_lands_on_the_named_teammate():
    """By name, because the coordinator speaks in wizard names and this
    client's participant list is its own -- seat 2 is not index 2 in
    anybody else's read."""
    from deimos_bridge.live_backend import WizAiCombatHandler

    class _Ally:
        def __init__(self, name):
            self.name = name

    class _Read:
        state = type("S", (), {"allies": [_Ally("Konstantin"),
                                          _Ally("Somebody Else")]})()
        ally_members = ["konstantin-member", "other-member"]
        client_member = "me"

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    assert handler._named_ally(_Read(), "Konstantin") == "konstantin-member"
    assert handler._named_ally(_Read(), "Somebody Else") == "other-member"
    # not in this duel's list at all -- a client that read a round late
    assert handler._named_ally(_Read(), "Nobody") is None
    assert handler._named_ally(_Read(), "") is None


def test_a_blade_for_a_teammate_is_not_clicked_on_the_caster():
    """A Fireblade reads `tgt: 'self'`, so without the override it goes
    on the wizard casting it -- which is the entire thing being
    changed."""
    from deimos_bridge.live_backend import WizAiCombatHandler

    class _Ally:
        name = "Konstantin"

    class _Read:
        state = type("S", (), {"allies": [_Ally()]})()
        ally_members = ["konstantin-member"]
        client_member = "me"
        enemy_members = ["mob"]

    handler = WizAiCombatHandler.__new__(WizAiCombatHandler)
    blade = cards()["Fireblade"]
    got = asyncio.run(handler._resolve_target(_Read(), None, blade,
                                              ally="Konstantin"))
    assert got == "konstantin-member"
    # no ally named: the card's own target, the caster
    got = asyncio.run(handler._resolve_target(_Read(), None, blade))
    assert got == "me"
    # named but absent: fall back rather than click something arbitrary
    got = asyncio.run(handler._resolve_target(_Read(), None, blade,
                                              ally="Nobody"))
    assert got == "me"


def test_the_round_record_says_who_the_blade_went_on():
    from deimos_bridge.hivemind import SeatMove
    from deimos_bridge.live_backend import WizAiBackend
    from deimos_bridge.mock_client import MockCard, MockCombat, MockMember

    class _Hive:
        async def decide(self, seat, sim, state, pol, read=None, rate=0.0):
            return pol(sim, state)

        def last_move(self, seat):
            return SeatMove(seat=0, name="wizard 1", card="Fireblade",
                            buff_ally="Konstantin", note="buff:Konstantin")

    backend = WizAiBackend.from_trained(
        school="fire", deck=["Fireblade"] * 4, cards=cards(),
        policy=greedy_ttk(), policy_name="ttk-lookahead", coordinator=_Hive())
    backend.attach_combat(MockCombat(
        [MockMember("Wizard", 900, client=True, team_id=0, normal_pips=4),
         MockMember("Lost Soul", 450, monster=True, team_id=1)],
        [MockCard("Fireblade")]))
    decision = asyncio.run(backend.decide())
    assert decision.ally_name == "Konstantin"
    assert decision.target_kind == "ally"
    assert "put on Konstantin" in decision.reason


# ------------------------------- a trap is worth what the party cashes
#
# Wards are the party's shared currency and this rate was landing as raw
# health, straight past them -- so a Feint on the mob the fire wizard is
# about to nuke was worth exactly zero to the ice wizard laying it. The
# run at rev 8666bda7 shows the price: 17 traps laid on mobs the caster
# never went on to hit (13 by the ice wizard), 7 mobs trapped by both
# wizards, and the coordinator itself moving a setup cast onto the other
# mob 7 times -- twice the two of them simply traded targets in a round.

def _lone_mob(ward=None, hp=900):
    from w101_sim import Actor, Boss, Rules, Sim, State

    mob = Actor(name="Mob", school="balance", hp=hp, max_hp=hp, team=1)
    if ward is not None:
        mob.wards.append(ward)
    me = Actor(name="W", school="ice", hp=1500, max_hp=1500, team=0)
    sim = Sim(cards=cards(), decklist=["Ice Trap"], school="ice",
              boss=Boss("Mob", hp, "balance", 40), rules=Rules(),
              player_hp=1500)
    return sim, State(me, [mob])


@pytest.mark.parametrize("label, percent, schools, left, standing", [
    ("nothing",              None, None,      100.0, 0),
    ("Feint, +70% universal", 0.70, None,      170.0, 0),
    ("Fire Trap, +40% fire",  0.40, {"fire"},  140.0, 0),
    # the one the operator described: the ice wizard's trap is not the
    # fire wizard's to spend, so it neither helps him nor is lost
    ("Ice Trap, +40% ice",    0.40, {"ice"},   100.0, 1),
])
def test_a_fire_teammate_cashes_only_the_wards_its_school_matches(
        label, percent, schools, left, standing):
    from w101_sim import Hanging
    from deimos_bridge.policies import _allies_hit

    ward = None if percent is None else Hanging(
        "w", "ward", "damage", percent=percent, schools=schools)
    sim, state = _lone_mob(ward)
    _allies_hit(state, 100.0, ("fire",), sim)
    assert 900.0 - state.enemies[0].hp == pytest.approx(left), label
    assert len(state.enemies[0].wards) == standing, label


def test_the_rate_is_split_between_the_teammates_not_given_to_each():
    """Two teammates removing 100 between them is 100, not 200 -- and
    with two schools the Feint is cashed by whichever gets there."""
    from deimos_bridge.policies import _allies_hit

    sim, state = _lone_mob()
    _allies_hit(state, 100.0, ("fire", "storm"), sim)
    assert 900.0 - state.enemies[0].hp == pytest.approx(100.0)


def test_without_a_school_the_rate_still_lands_as_raw_health():
    """Every solo path and every caller that predates this passes no
    school, and has to get the number it always got."""
    from w101_sim import Hanging
    from deimos_bridge.policies import _allies_hit

    ward = Hanging("w", "ward", "damage", percent=0.70, schools=None)
    sim, state = _lone_mob(ward)
    _allies_hit(state, 100.0)
    assert 900.0 - state.enemies[0].hp == pytest.approx(100.0)
    assert len(state.enemies[0].wards) == 1     # nothing was spent


def test_setup_moves_off_the_mob_the_teammate_is_about_to_kill():
    """The operator's report, as a decision.

    A 200hp mob and a 900hp mob, and a fire teammate removing 60 a round
    -- so the small one is gone in three turns whatever the ice wizard
    does. Blind to the teammate's school, the rollout cannot tell that
    an Ice Trap there will never be cashed, and lays it. Told the
    teammate deals fire, the same rollout puts a universal Feint on the
    mob that is still going to be standing.
    """
    from deimos_bridge.policies import greedy_ttk, party_rate

    def decide(schools):
        sim, state = wizard(school="ice",
                            hand=("Ice Trap", "Feint", "Frost Beetle"),
                            pips=2, hp=1500,
                            board=((200, "balance"), (900, "balance")))
        policy = greedy_ttk()
        with party_rate(60.0, schools):
            choice = policy(sim, state)
        card, target = choice if isinstance(choice, tuple) else (choice, None)
        return getattr(card, "name", card), target

    assert decide(()) == ("Ice Trap", 0)
    assert decide(("fire",)) == ("Feint", 1)


# ------------------------------------------------- the free-rider tie
#
# The failure these pin down, in the operator's words: "it spammed buffs
# (many of which don't stack)". The two live exports at rev e523684f
# show it exactly -- 27 of 79 in-horizon rounds chose a move that tied
# passing to the last decimal, and 24 of those played a buff, a trap or
# a shield. It is not a stacking rule the policy is missing. It is that
# a board the rest of the circle will clear anyway scores every
# candidate identically, INCLUDING doing nothing, and the thrift key
# then picks whatever costs fewest pips -- which is always a trap.
#
# `_tied_board` is that board offline: one 420hp mob, an ally rate big
# enough to clear it, and a hand holding both a 0-pip trap and real
# attacks.
def _tied_board():
    return wizard(school="ice",
                  hand=("Ice Trap", "Snow Serpent", "Frost Beetle",
                        "Tower Shield"),
                  pips=4, hp=1270, board=((420, "fire"),))


def _decide(rate):
    from deimos_bridge.policies import greedy_ttk, party_rate

    sim, state = _tied_board()
    policy = greedy_ttk()
    with party_rate(rate):
        choice = policy(sim, state)
    card = choice[0] if isinstance(choice, tuple) else choice
    return getattr(card, "name", card), policy


def test_a_board_the_party_clears_anyway_ties_every_move_with_passing():
    """The premise, asserted rather than assumed.

    Without this the fix below could pass for the wrong reason -- the
    re-score only fires on an exact tie with passing, so a test that
    never produced one would be measuring nothing.
    """
    from deimos_bridge.policies import greedy_ttk, party_rate

    sim, state = _tied_board()
    policy = greedy_ttk()
    with party_rate(900.0):
        # scored with the ally rate on and the fallback disabled, which
        # is what the shipped code did before this change
        import deimos_bridge.policies as pol
        scored = [pol._rollout(sim, state, c, 12, 0)
                  for c in state.player.hand]
        passing = pol._rollout(sim, state, None, 12)
    assert all(sc == passing for sc in scored), \
        f"the ally rate is supposed to flatten this board: {scored} vs {passing}"


def test_a_move_that_ties_doing_nothing_is_decided_as_if_alone():
    chosen, policy = _decide(900.0)
    assert policy.last_party_blind is True
    assert chosen == "Frost Beetle", \
        "an attack, not the 0-pip trap the flattened comparison preferred"


def test_the_same_board_alone_picks_the_same_move():
    """The fallback is the solo answer, not a third thing."""
    alone, policy = _decide(0.0)
    assert policy.last_party_blind is False
    together, _ = _decide(900.0)
    assert together == alone == "Frost Beetle"


def test_a_party_that_still_leaves_the_move_room_is_left_alone():
    """Not every rate flattens the board, and where the rollout can still
    tell the candidates apart the party's own arithmetic is the one that
    decides -- that is what the ally rate is for."""
    chosen, policy = _decide(200.0)
    assert policy.last_party_blind is False
    assert chosen == "Ice Trap", \
        "at this rate the trap genuinely reaches the kill a turn sooner"


def test_a_solo_decision_never_reaches_the_fallback():
    """Every published solo number, tuning sweep and regression table was
    measured without this, so it has to be unreachable at rate zero."""
    from deimos_bridge.policies import greedy_ttk

    sim, state = wizard(school="ice",
                        hand=("Frost Beetle", "Snow Serpent", "Ice Trap",
                              "Tower Shield"),
                        pips=2, hp=1270,
                        board=((435, "balance"), (435, "balance"),
                               (435, "balance")))
    policy = greedy_ttk()
    policy(sim, state)          # a board where everything is a sentinel
    assert policy.last_party_blind is False


@pytest.mark.parametrize("blind, expected", [(True, True), (False, False)])
def test_the_fallback_is_reported_on_the_round_it_decided(blind, expected):
    """A round record showing party-blind numbers beside a party plan is
    unreadable unless it says which comparison produced them.

    Both directions, because a note that is always there says nothing.
    """
    from deimos_bridge.live_backend import WizAiBackend
    from deimos_bridge.mock_client import MockCard, MockCombat, MockMember

    real = greedy_ttk()

    def policy(sim, state):
        out = real(sim, state)
        policy.last_candidates = real.last_candidates
        policy.last_party_blind = blind
        return out

    policy.last_candidates = ()
    policy.last_party_blind = blind

    class _Hive:
        async def decide(self, seat, sim, state, pol, read=None, rate=0.0):
            return pol(sim, state)

        def last_move(self, seat):
            return None

    backend = WizAiBackend.from_trained(
        school="fire", deck=["Fire Cat"] * 4, cards=cards(),
        policy=policy, policy_name="ttk-lookahead", coordinator=_Hive())
    backend.attach_combat(MockCombat(
        [MockMember("Wizard", 900, client=True, team_id=0, normal_pips=4),
         MockMember("Lost Soul", 450, monster=True, team_id=1)],
        [MockCard("Fire Cat")]))
    decision = asyncio.run(backend.decide())
    assert ("decided as if alone" in decision.reason) is expected


def test_the_rollout_does_not_hit_twice_for_the_round_already_on_the_board():
    """`Ledger.apply` has already taken this round's party damage off the
    state the policy is handed. Applying the rate on rollout turn 1 as
    well would count the same casts twice."""
    from deimos_bridge.policies import _rollout

    sim, state = wizard(school="fire", hand=("Fire Cat",), pips=7,
                        board=((10_000, "ice"),))
    # Nothing in hand and nothing to draw, so every point of the banked
    # damage is the ally's and the count is unambiguous.
    state.player.hand[:] = []
    state.player.deck[:] = []
    _turns, banked = _rollout(sim, state, None, max_turns=3, allies=100.0)
    # Three turns, ally pressure on two of them: 200, not 300.
    assert -banked == pytest.approx(200.0)

    _turns, solo = _rollout(sim, state, None, max_turns=3, allies=0.0)
    assert solo == 0.0


def test_a_seat_with_no_measured_rate_reports_its_own_rollout_instead():
    """The cold start, and it is the case that matters: the first duel of
    a session is where the party-blind rollout hurt most -- both wizards
    of the first live two-wizard run died in fight 1, before either had a
    measurement to report."""
    from deimos_bridge.policies import Candidate, rollout_throughput

    assert rollout_throughput(()) == 0.0
    assert rollout_throughput([
        Candidate(card="pass", target=None, turns=13, damage=1045,
                  pips=0, chosen=False, horizon=12),
        Candidate(card="Frost Beetle", target=0, turns=13, damage=600,
                  pips=1, chosen=False, horizon=12),
        Candidate(card="Snow Serpent", target=0, turns=13, damage=1044,
                  pips=2, chosen=True, horizon=12),
    ]) == pytest.approx(87.0)


def test_a_seats_measured_rate_reaches_the_other_seats_rollouts():
    seen = []

    class _Watching(Hivemind):
        def _decide_one(self, sub, board, ledger, step, pressure=0.0,
                        pressure_schools=()):
            seen.append((sub.seat, step, pressure, pressure_schools))
            return super()._decide_one(sub, board, ledger, step, pressure,
                                       pressure_schools)

    subs = party(2)
    subs[0].rate = 40.0
    subs[1].rate = 70.0
    _Watching(passes=1).plan(subs)

    # Pass 0 is the solo baseline and must stay solo, or "alone it would
    # have played X" is not what it says it is.
    assert [p for s, step, p, _sc in seen if step == 0] == [0.0, 0.0]
    # After that each seat sees what the OTHERS deal, never its own.
    later = {s: p for s, step, p, _sc in seen if step == 1}
    assert later == {0: 70.0, 1: 40.0}
    # ...and what they deal it in, so a trap can be priced against the
    # hit that would cash it.
    schools = {s: sc for s, step, _p, sc in seen if step == 1}
    assert schools == {0: ("fire",), 1: ("fire",)}


def test_the_rate_never_includes_the_seat_it_is_planning_for():
    """A wizard that models itself as its own teammate deals double, and
    would stop buying setup on exactly the boards that need it."""
    seen = []

    class _Watching(Hivemind):
        def _decide_one(self, sub, board, ledger, step, pressure=0.0,
                        pressure_schools=()):
            seen.append((sub.seat, step, pressure, pressure_schools))
            return super()._decide_one(sub, board, ledger, step, pressure,
                                       pressure_schools)

    subs = party(3)
    for i, sub in enumerate(subs):
        sub.rate = 100.0 * (i + 1)
    _Watching(passes=1).plan(subs)

    later = {s: p for s, step, p, _sc in seen if step == 1}
    assert later == {0: 500.0, 1: 400.0, 2: 300.0}


# ------------------------------------------------ tests that CAN see the key
#
# The suite could not. 855 tests passed both with and without d3b7962,
# a change that resolved roughly a third of their `greedy_ttk` decisions
# differently -- because 44% of decisions tie at (turns, damage) and
# every test that asserts a specific card makes one or two decisions,
# none of which tie. These force the tie so the later keys are covered
# by something.
#
# Only ONE of the three is the guard: reapplying d3b7962 fails
# `test_between_equal_outcomes_the_cheaper_card_wins` and passes the
# other two, which are the premise it rests on and a separate property.
# Checked, rather than assumed -- a test that would not have caught the
# change is not coverage of it, and claiming three guards where there is
# one is how a suite comes to be trusted for something it cannot do.
def test_a_tie_at_turns_and_damage_is_the_common_case_not_the_edge():
    """The premise the two tests below rest on, asserted rather than
    assumed. If ties were rare, the ranking tail would not matter and
    nor would testing it."""
    from deimos_bridge.policies import greedy_ttk

    tied = seen = 0
    for hp in (60, 150, 400, 800, 1600):
        for pips in (1, 3, 5, 7):
            sim, state = wizard(school="fire", pips=pips,
                                hand=("Fire Cat", "Fire Elf", "Fireblade",
                                      "Fire Trap"),
                                board=((hp, "ice"), (hp, "ice")))
            policy = greedy_ttk()
            policy(sim, state)
            cards = policy.last_candidates
            if len(cards) < 2:
                continue
            seen += 1
            best = min((c.turns, -c.damage) for c in cards)
            if sum(1 for c in cards
                   if (c.turns, -c.damage) == best) > 1:
                tied += 1

    assert seen >= 15, f"only {seen} usable boards"
    assert tied >= seen // 4, (
        f"only {tied} of {seen} boards tied at (turns, damage); if that is "
        f"really the rate, the keys after them barely matter")


def test_between_equal_outcomes_the_cheaper_card_wins(monkeypatch):
    """The third key IN HORIZON, forced. Every candidate scores the same
    real turn count and the same banked damage, so the outcomes really
    are equivalent -- and the cheaper card is better, because it leaves
    the pip banked and the big card in hand.

    Two rounds of the second live party run are exactly this and the
    cheap card is right in both: a 190hp board where Frost Beetle and
    Evil Snowman both clear on turn 1, and a 348hp board where both
    clear on turn 2.
    """
    from deimos_bridge import policies as P

    # turns=5 with a horizon of 12: a real turn count, not a sentinel.
    monkeypatch.setattr(P, "_rollout", lambda *a, **kw: (5.0, -600.0))
    sim, state = wizard(school="fire", hand=("Fire Cat", "Sunbird"),
                        pips=7, hp=1500, board=((800, "ice"),))
    chosen = P.greedy_ttk()(sim, state)
    card, _t = chosen if isinstance(chosen, tuple) else (chosen, 0)
    assert card is not None and card.name == "Fire Cat", (
        f"played {getattr(card, 'name', card)!r}: a 3-pip Sunbird beat a "
        f"1-pip Fire Cat on outcomes the rollout scored identically")


def test_above_the_horizon_the_bigger_hit_wins_instead(monkeypatch):
    """...and the same tie ABOVE the horizon means the opposite thing.

    `turns` is a sentinel, `neg_damage` is the whole rollout line's
    board delta, and a tie says only that the lookahead could not tell
    the candidates apart -- so "cheapest" is not thrift, it is a coin
    flip that always lands on the free card. Measured over the second
    live party run's 197 scored rounds, 107 have no candidate killing
    inside the horizon and 38 of those played a weaker card that tied a
    stronger one exactly; three consecutive rounds of its fight 14 spent
    4, 5 and 6 pips' worth of banked pips on 85-damage chip while Evil
    Snowman sat in hand.
    """
    from deimos_bridge import policies as P

    monkeypatch.setattr(P, "_rollout", lambda *a, **kw: (13.0, -600.0))
    sim, state = wizard(school="fire", hand=("Fire Cat", "Sunbird"),
                        pips=7, hp=1500, board=((800, "ice"),))
    chosen = P.greedy_ttk()(sim, state)
    card, _t = chosen if isinstance(chosen, tuple) else (chosen, 0)
    assert card is not None and card.name == "Sunbird", (
        f"played {getattr(card, 'name', card)!r}: nothing kills inside the "
        f"horizon, so the tie is meaningless and the free card won it")


def test_a_stall_that_killed_three_mobs_is_still_a_stall(monkeypatch):
    """`turns > horizon` is the obvious sentinel test and it is wrong.
    `_lost_score` credits a losing line 0.4 a kill, so a stalled rollout
    that killed three of four mobs scores 13 - 1.2 = 11.8 -- below the
    horizon, and indistinguishable from a real turn count by size."""
    from deimos_bridge import policies as P

    assert P.is_sentinel(11.8, 12), "a 3-kill stall reads as in-horizon"
    assert not P.is_sentinel(12, 12), "a 12-turn clear is a real count"

    monkeypatch.setattr(P, "_rollout", lambda *a, **kw: (11.8, -600.0))
    sim, state = wizard(school="fire", hand=("Fire Cat", "Sunbird"),
                        pips=7, hp=1500, board=((800, "ice"),))
    chosen = P.greedy_ttk()(sim, state)
    card, _t = chosen if isinstance(chosen, tuple) else (chosen, 0)
    assert card is not None and card.name == "Sunbird", (
        f"played {getattr(card, 'name', card)!r}: 11.8 is a stall wearing a "
        f"turn count's clothes, and the thrifty branch fired on it")


def test_the_ranking_is_total_enough_that_hand_order_does_not_decide():
    """A complete tie leaves `min` returning whatever the hand listed
    first, which is not a decision. Two copies of one card at different
    targets are the sharpest case: same pips, same damage, same
    everything but the mob."""
    from deimos_bridge import policies as P

    sim, state = wizard(school="fire", hand=("Fire Cat",), pips=7,
                        board=((400, "ice"), (120, "ice")))
    policy = P.greedy_ttk()
    policy(sim, state)
    keys = [(c.turns, -c.damage, c.pips, c.target)
            for c in policy.last_candidates if c.card != "pass"]
    assert len(keys) == len(set(keys)), (
        f"two candidates are indistinguishable to the recorded ranking: "
        f"{keys}")


# ---------------------------------------------------- pricing an X-pip card
#
# Heck Hound and 2,111 others carry `x_pips`: they spend the whole rack
# and scale with it. The card table prints 0 pips and per-pip damage,
# and the ranking key was reading both off the printed face -- so an
# X-pip card was free (it won every thrift tie) and tiny (130 against a
# Fire Cat's 100) at the same time.
#
# This used to be handled by refusing to offer the card at all, on my
# reading that the engine "gives them 0 damage at every pip count". That
# was wrong -- the fizzles in my measurement were a live RNG, not a
# broken card -- and the cost of it is in the export at rev e523684f:
# Heck Hound sat in the fire wizard's hand for 74 of 75 rounds, never
# played, out of a median hand of four.

def test_an_x_pip_card_costs_the_rack_and_scales_with_it():
    from w101_sim import cast_price, cast_reach

    sim, state = wizard(school="fire", hand=("Heck Hound", "Fire Cat"),
                        pips=6, hp=1500, board=((800, "ice"),))
    hound, cat = state.player.hand
    assert hound.x_pips and hound.pips == 0 and hound.damage == 130, \
        "premise moved"
    assert cast_price(sim, state, hound) == 6      # not the printed 0
    assert cast_reach(sim, state, hound) == 780    # 130 a pip, not 130
    assert cast_price(sim, state, cat) == 1        # an ordinary card is itself
    assert cast_reach(sim, state, cat) == 100


def test_a_rack_is_not_spent_on_a_mob_a_cheap_nuke_already_kills():
    """The tie the printed price inverts, and the reason this matters.

    Seven pips against a 300hp mob: Heck Hound and Sunbird both clear it
    on turn one banking the same 300. Priced at 0 the hound wins the
    thrift key and the whole rack goes; priced at 7 against Sunbird's 3
    it does not, and four pips stay banked.
    """
    from deimos_bridge.policies import greedy_ttk

    sim, state = wizard(school="fire",
                        hand=("Heck Hound", "Sunbird", "Fire Cat"),
                        pips=7, hp=1500, board=((300, "ice"),))
    policy = greedy_ttk()
    chosen = policy(sim, state)
    card, _t = chosen if isinstance(chosen, tuple) else (chosen, 0)
    tied = {c.card for c in policy.last_candidates if (c.turns, c.damage)
            == (1, 300.0)}
    assert {"Heck Hound", "Sunbird"} <= tied, f"the tie is the premise: {tied}"
    assert card.name == "Sunbird"


def test_a_full_rack_does_go_into_the_x_pip_card_when_it_pays():
    """The other direction, so the fix is not just "never play it".

    Six pips against an 800hp mob, and 780 of damage is worth the rack:
    the hound clears in five turns where Fire Cat and Sunbird need nine.
    """
    from deimos_bridge.policies import greedy_ttk

    sim, state = wizard(school="fire",
                        hand=("Heck Hound", "Sunbird", "Fire Cat"),
                        pips=6, hp=1500, board=((800, "ice"),))
    policy = greedy_ttk()
    chosen = policy(sim, state)
    card, _t = chosen if isinstance(chosen, tuple) else (chosen, 0)
    assert card.name == "Heck Hound"
    hound = next(c for c in policy.last_candidates if c.card == "Heck Hound")
    assert hound.turns < min(c.turns for c in policy.last_candidates
                             if c.card != "Heck Hound")


def test_one_pip_is_not_enough_and_the_rollout_says_so():
    """The live complaint -- eight Heck Hounds at one or two pips. The
    rollout rejects it on its own once the card is offered at all: 130
    of damage does not clear an 800hp board inside the horizon."""
    from deimos_bridge.policies import greedy_ttk

    sim, state = wizard(school="fire",
                        hand=("Heck Hound", "Fire Trap", "Fireblade"),
                        pips=1, hp=1064, board=((800, "ice"), (525, "balance")))
    policy = greedy_ttk()
    chosen = policy(sim, state)
    card, _t = chosen if isinstance(chosen, tuple) else (chosen, 0)
    assert card is None or card.name != "Heck Hound"
    assert any(c.card == "Heck Hound" for c in policy.last_candidates), \
        "it must be weighed and lose, not be hidden"


def test_the_recorded_price_is_the_one_that_was_paid():
    """`Candidate.pips` feeds the decision panel and `rank_candidate`'s
    runner-up ranking. Showing 0 beside a 1-pip Fire Cat reads as a free
    card losing to a paid one."""
    from deimos_bridge.policies import greedy_ttk

    sim, state = wizard(school="fire", hand=("Heck Hound", "Fire Cat"),
                        pips=5, hp=1500, board=((800, "ice"),))
    policy = greedy_ttk()
    policy(sim, state)
    hound = next(c for c in policy.last_candidates if c.card == "Heck Hound")
    assert hound.pips == 5


def test_the_ordinary_cards_are_still_offered():
    """The inert rule must not quietly swallow a normal hand."""
    from deimos_bridge.policies import _is_inert

    table = cards()
    for name in ("Fire Cat", "Sunbird", "Fireblade", "Fire Trap",
                 "Snow Serpent", "Tower Shield", "Heck Hound"):
        assert not _is_inert(table[name], None), name
