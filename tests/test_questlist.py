"""Placing a wizard in a questline, and refusing to when it cannot be done.

The refusals matter more than the answers here. `_who_is_behind` feeds
`_should_catch_up`, which MOVES wizards, so a confident wrong answer
walks a wizard backwards through a quest it has already finished -- which
is the complaint this is meant to fix.
"""
from deimos_bridge import questlist


def test_the_list_ships_and_loads():
    assert questlist.loaded(), questlist.load_error()


def test_a_quest_name_places_a_wizard_in_its_world():
    """The reliable direction: the tracked quest name is unique enough
    (17 collisions in 2,640) to be a key."""
    p = questlist.position_of("Quarter Master")
    assert p.world == "Krokotopia"
    assert p.order == 9
    assert p.comparable
    assert p.how == "by quest name"


def test_the_name_match_ignores_case_markup_and_punctuation():
    """The HUD title-cases and wraps in markup; the data does neither."""
    for text in ("quarter master", "<center>Quarter Master</center>",
                 "  QUARTER MASTER  "):
        assert questlist.position_of(text).order == 9, text
    # apostrophes are the common case: "Give 'Em Another Round"
    assert questlist.position_of("give em another round").order == 14


def test_a_quest_the_list_has_never_heard_of_is_not_placed():
    p = questlist.position_of("Fight the Kraken of Nonexistent Bay")
    assert not p.comparable
    assert p.how == "not in the list"


def test_an_unambiguous_goal_places_a_wizard_too():
    """The fallback. `read_quest_name` can fail -- a mid-load read, a
    Deimos update -- and a goal that points at exactly one quest is
    still better than nothing."""
    p = questlist.position_from_goal(
        "Defeat Nirini Quartermaster and Collect Key Piece in Chamber of Fire")
    assert p.world == "Krokotopia" and p.order == 9
    assert p.how == "by goal text"


def test_an_ambiguous_goal_refuses_rather_than_guessing():
    """"Talk to Professor Winthrop" is the objective of NINE Krokotopia
    quests between main #2 and main #19. Matching it to the first is how
    a wizard on #15 gets called thirteen steps behind and dragged back
    through half a world."""
    p = questlist.position_from_goal("Talk To Professor Winthrop "
                                     "in Altar of Kings")
    assert not p.comparable, f"guessed {p.order}"
    assert "ambiguous" in p.how, p.how


def test_the_rest_of_the_party_disambiguates_a_goal():
    """A wizard questing with three others is within a few steps of
    them, so `near` picks the candidate that is actually plausible."""
    p = questlist.position_from_goal("Talk To Professor Winthrop "
                                     "in Altar of Kings",
                                     world="Krokotopia", near=15)
    assert p.comparable and p.order == 15, p.how


def test_the_zone_suffix_does_not_eat_the_objective():
    """"Collect Key Piece in Chamber of Fire" must lose the zone and
    keep the key piece. Only the LAST " in " is cut."""
    assert questlist._strip_zone("Talk To Winthrop in Altar of Kings") == \
        "Talk To Winthrop"
    assert questlist._strip_zone("Defeat Nirini Quartermaster and Collect "
                                 "Key Piece in Chamber of Fire") == \
        "Defeat Nirini Quartermaster and Collect Key Piece"


# ------------------------------------------------------------ who is behind
def _at(world, order):
    p = questlist.Position(world=world, order=order, name=f"#{order}")
    return p


def test_the_wizard_further_back_in_the_line_is_the_one_behind():
    behind, gap, why = questlist.furthest_behind(
        [_at("Krokotopia", 9), _at("Krokotopia", 4), _at("Krokotopia", 9)])
    assert behind == [1]
    assert gap == 5
    assert "#4 against #9" in why


def test_one_quest_apart_and_staying_there_is_a_desync():
    """This is rev 8e5a9c75. Sebastian sat one quest ahead of the other
    two for eight unbroken minutes and then wandered off the line, and
    every check said the party was together -- because the threshold was
    2, on the reasoning that a gap of one is a handover.

    A handover is a gap of one that lasts seconds. A desync is a gap of
    one that lasts twenty minutes. Duration tells them apart; magnitude
    does not."""
    behind, gap, why = questlist.furthest_behind(
        [_at("Krokotopia", 12), _at("Krokotopia", 12), _at("Krokotopia", 13)])
    assert behind == [0, 1], why
    assert gap == 1
    assert "2 wizards are 1 quest(s) behind" in why


def test_a_party_all_on_one_quest_is_in_step():
    behind, _gap, why = questlist.furthest_behind(
        [_at("Krokotopia", 9), _at("Krokotopia", 9)])
    assert behind == []
    assert "same quest" in why


def test_two_wizards_equally_far_back_are_both_named():
    """This used to refuse -- "there is no one to catch up" -- and in a
    party of three that is the ordinary case, not an edge case. Two
    wizards equally behind does not mean nobody is behind; it means two
    of them have a step to finish."""
    behind, gap, why = questlist.furthest_behind(
        [_at("Krokotopia", 9), _at("Krokotopia", 4), _at("Krokotopia", 4)])
    assert behind == [1, 2]
    assert gap == 5
    assert "2 wizards are 5 quest(s) behind" in why


def test_a_party_split_across_worlds_is_not_a_gap_in_one_line():
    """Krokotopia #9 and Wizard City #40 are not 31 steps apart; they are
    not on the same line at all."""
    behind, _gap, why = questlist.furthest_behind(
        [_at("Krokotopia", 9), _at("Wizard City", 40)])
    assert behind == []
    assert "worlds" in why


def test_a_side_quest_has_no_place_and_does_not_get_invented_one():
    """Most of Wizard City's optional line carries no order. A wizard on
    one is not comparable, and saying so is the point."""
    behind, _gap, why = questlist.furthest_behind(
        [_at("Krokotopia", 9), questlist.Position(world="Krokotopia")])
    assert behind == []
    assert "fewer than two" in why


def test_one_wizard_alone_is_never_behind():
    behind, _gap, _why = questlist.furthest_behind([_at("Krokotopia", 9)])
    assert behind == []


def test_the_krokotopia_main_line_is_complete():
    """The runs under test are all Krokotopia, so a gap in its ordering
    would silently skew every comparison made during one."""
    kt = [q for q in questlist._load().quests
          if q["world"] == "Krokotopia" and q["questline_order"]]
    orders = sorted(q["questline_order"] for q in kt)
    assert orders == list(range(1, len(orders) + 1)), \
        f"gaps in the Krokotopia line: {orders[:20]}"


# ------------------------------------------- a goal that disowns the name
#
# Rev 30e83468. `_read_goal` keeps the previous quest name on a blank
# read, because a blank is not evidence of a change -- so the name can
# lag the goal by a quest. Sebastian's goal read `Talk To Sergeant Major
# Talbot in The Oasis`, Krokotopia #20 and character for character the
# same text as the wizard he was measured against, while his name still
# read `Eye of Krok`, #18 -- and the party held a catch-up over a
# two-quest gap that did not exist.

def test_a_goal_belonging_to_another_quest_disowns_the_name():
    assert questlist.goal_disowns(
        "Eye of Krok", "Talk To Sergeant Major Talbot in The Oasis")


def test_a_quests_own_objectives_do_not_disown_it():
    """Both steps of a two-step quest, zone suffix and all."""
    for goal in ("Defeat Krokenkahmen and Collect Eye of Krok",
                 "Talk To Professor Winthrop in Throne Room of Fire"):
        assert not questlist.goal_disowns("Eye of Krok", goal), goal


def test_a_goal_the_list_has_never_heard_of_proves_nothing():
    """The data's objectives are incomplete in places -- `Into the Map
    Room` lists a prose note instead of steps -- so a goal nothing
    lists must not convict the name over the data's own gaps."""
    assert not questlist.goal_disowns("Eye of Krok", "goal for Eye of Krok")
    assert not questlist.goal_disowns(
        "Into the Map Room", "Use the Krokotech Monolith")


def test_an_unknown_quest_cannot_be_disowned():
    assert not questlist.goal_disowns(
        "Fight the Kraken of Nonexistent Bay", "Talk To Sergeant Major Talbot")


def test_the_hud_dropping_a_title_still_finds_the_quest():
    """The HUD tracks `Talk To Gordon Flemming` for quests the data
    records as `Talk to Dr. Gordon Flemming`. Every quest listing him
    is a side quest, so the answer is "known, with no place in the
    line" -- not #18, which is where a stale name read had put the
    wizard tracking him in rev 30e83468."""
    p = questlist.position_from_goal("Talk To Gordon Flemming in Royal Hall",
                                     world="Krokotopia")
    assert not p.comparable
    assert p.name, "the loose match did not find the quest at all"
    assert "no order" in p.how


def test_the_loose_match_only_narrows_from_the_goal_side():
    """The reverse direction is a trap: the data carries an objective
    that normalises to just "talk to", which is a subset of every talk
    goal in the game -- it matched `Talk To Gordon Flemming` to
    Krokotopia #12 on the first try."""
    hits = questlist._loose_lookup(
        questlist._load(), "Talk To Gordon Flemming in Royal Hall")
    assert hits, "the dropped 'Dr.' hid the quest entirely"
    assert all("Flemming" in " ".join(q.get("objectives") or ())
               for q in hits), \
        "a generic 'talk to' objective subset-matched a talk goal"
