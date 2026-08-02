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

        async def decide(self, seat, sim, state, policy, read=None):
            asked.append(seat)
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
    assert asked == [2]
    assert not decision.passing, decision.reason
    assert "party" in decision.policy


def test_a_held_card_is_recorded_as_held_rather_than_as_a_bare_pass():
    """"policy chose to pass" and "the other three have this dead" are
    different events, and only the second is coordination working."""
    from deimos_bridge.hivemind import SeatMove
    from deimos_bridge.live_backend import WizAiBackend
    from deimos_bridge.mock_client import MockCard, MockCombat, MockMember

    class _Hive:
        async def decide(self, seat, sim, state, policy, read=None):
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
    must not short-circuit the sigil step."""
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
    assert follower.teleports == []          # it was already there


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
