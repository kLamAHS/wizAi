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
    index, gap, why = questlist.furthest_behind(
        [_at("Krokotopia", 9), _at("Krokotopia", 4), _at("Krokotopia", 9)])
    assert index == 1
    assert gap == 5
    assert "#4 against #9" in why


def test_a_party_a_step_apart_is_not_a_desync():
    """One step is normal -- a party rarely turns a quest in on the same
    tick -- and calling it a desync would have wizAi regrouping
    constantly."""
    index, gap, why = questlist.furthest_behind(
        [_at("Krokotopia", 9), _at("Krokotopia", 8)])
    assert index is None
    assert "normal" in why


def test_two_wizards_equally_far_back_names_neither():
    """Moving one of them is not obviously right, and the caller can
    still regroup without naming anybody."""
    index, gap, why = questlist.furthest_behind(
        [_at("Krokotopia", 9), _at("Krokotopia", 4), _at("Krokotopia", 4)])
    assert index is None
    assert gap == 5
    assert "equally far back" in why


def test_a_party_split_across_worlds_is_not_a_gap_in_one_line():
    """Krokotopia #9 and Wizard City #40 are not 31 steps apart; they are
    not on the same line at all."""
    index, _gap, why = questlist.furthest_behind(
        [_at("Krokotopia", 9), _at("Wizard City", 40)])
    assert index is None
    assert "worlds" in why


def test_a_side_quest_has_no_place_and_does_not_get_invented_one():
    """Most of Wizard City's optional line carries no order. A wizard on
    one is not comparable, and saying so is the point."""
    index, _gap, why = questlist.furthest_behind(
        [_at("Krokotopia", 9), questlist.Position(world="Krokotopia")])
    assert index is None
    assert "fewer than two" in why


def test_one_wizard_alone_is_never_behind():
    index, _gap, _why = questlist.furthest_behind([_at("Krokotopia", 9)])
    assert index is None


def test_the_krokotopia_main_line_is_complete():
    """The runs under test are all Krokotopia, so a gap in its ordering
    would silently skew every comparison made during one."""
    kt = [q for q in questlist._load().quests
          if q["world"] == "Krokotopia" and q["questline_order"]]
    orders = sorted(q["questline_order"] for q in kt)
    assert orders == list(range(1, len(orders) + 1)), \
        f"gaps in the Krokotopia line: {orders[:20]}"
