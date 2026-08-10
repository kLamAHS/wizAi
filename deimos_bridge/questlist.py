"""Where a wizard is along a questline, so "behind" can be a number.

Nothing in the game tells you that one wizard is further along than
another. A client knows its own tracked quest and its own goal text, and
that is all -- so wizAi's first two answers to "who is behind" were
inferences: the wizard on a step somebody else has already finished, or
failing that the wizard whose goal changed longest ago. Both are guesses,
and the logs say so: `quest-desync` fired thirty times in one run
alongside "which one is behind cannot be told: more than one wizard is on
a finished step".

A guess is worse than nothing here. `_catch_up` MOVES a wizard, so
naming the wrong one drags a wizard that was ahead backwards to a step it
has already done -- which is the shape of the complaint this is meant to
fix, not a new one.

So: an ordered list of quests per world, and a wizard's position in it.

Keying on the quest NAME, not the goal text
-------------------------------------------
The goal line is what the HUD shows and what wizAi already reads, and it
is the wrong key. "Talk to Professor Winthrop" is the objective of NINE
Krokotopia quests spanning main #2 to main #19; matching a goal to the
first quest that lists it puts a wizard seventeen steps from where it is.
230 of the 1,377 distinct objectives in this data are ambiguous that way.

Quest names are not: 17 collisions in 2,640. So the tracked quest name is
the key, read the way `VM._fetch_tracked_quest_text` reads it, and goal
text is a fallback that is allowed to answer "cannot tell" -- and does,
whenever the candidates disagree about where they sit.

The data
--------
`data/quests.json.gz`: names, worlds, areas and questline order, from the
Wizard101 Wiki (CC-BY-SA) and Final Bastion. Names and ordering only --
no coordinates, no walkthrough text, nothing the bot follows. It is a
sort key.

Not every quest has an order. Side quests and most of Wizard City's
optional lines have `questline_order: None`, so a wizard on one has no
comparable position -- `position_of` returns None and the callers fall
back to what they did before rather than inventing a number.
"""
import gzip
import json
import re
import threading
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "quests.json.gz"

#: How many quests apart counts as apart at all.
#:
#: One, and the number matters. This was two, on the reasoning that a
#: party rarely turns a quest in on the same tick so a gap of one is a
#: handover rather than a desync. That confuses MAGNITUDE with
#: TRANSIENCE, and rev 8e5a9c75 is what it costs: Sebastian sat one
#: quest ahead of the other two for eight unbroken minutes and then
#: wandered off the line entirely, and every check said the party was
#: together.
#:
#: A handover is a gap of one that lasts seconds. A desync is a gap of
#: one that lasts twenty minutes. Duration tells them apart and
#: magnitude does not, so duration is what filters -- see
#: `LiveWorker.DESYNC_GRACE`, which already existed for exactly this and
#: was sitting behind a floor that never let it run.
BEHIND_BY = 1

_lock = threading.Lock()
_loaded = None


def _norm(text) -> str:
    """Fold to the form both sides can be compared in.

    The HUD's text carries markup and title casing the data does not
    ("Talk To" vs "Talk to"), and the data carries punctuation the HUD
    drops. Everything that is not a letter, digit or space goes -- which
    also handles the apostrophes in "Give 'Em Another Round".
    """
    text = re.sub(r"<[^>]*>", "", text or "").lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text) -> frozenset:
    """The normalised words, for matching that survives dropped ones.

    The HUD and the data abbreviate differently: the HUD tracks `Talk
    To Gordon Flemming` for a quest the data records as `Talk to Dr.
    Gordon Flemming`, and exact-normalised comparison calls those two
    different objectives. One side's words being a subset of the
    other's is the loosest match that still cannot confuse two
    different NPCs.
    """
    return frozenset(_norm(text).split())


def _load():
    global _loaded
    with _lock:
        if _loaded is not None:
            return _loaded
        _loaded = _build()
        return _loaded


def _build():
    try:
        with gzip.open(DATA, "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:                    # pragma: no cover - packaging
        return _Index([], f"{type(exc).__name__}: {exc}")
    return _Index(raw.get("quests") or [], "")


class _Index:
    """Name -> quest, objective -> quests, both normalised."""

    def __init__(self, quests, error):
        self.error = error
        self.quests = quests
        self.by_name = {}
        self.by_objective = {}
        #: (world, order) -> the main-line quest that sits there. The
        #: inverse of `by_name`, and the half that was missing: every
        #: rule so far asked "where is this wizard", and none could ask
        #: "what should the wizard at #13 be tracking". All 2,110
        #: main-line entries have a distinct (world, order), so this is
        #: a lookup and not a guess.
        self.by_place = {}
        for quest in quests:
            name = _norm(quest.get("name"))
            # First wins, and it is not always right: seven main-line
            # names are reused across worlds ("The Right Combination"
            # is Krokotopia #55 AND Marleybone #39), so a lookup by
            # name alone can answer with the wrong world's quest. Good
            # enough for placing a wizard -- a party in Krokotopia is
            # not accidentally in Marleybone -- and NOT good enough for
            # naming a quest to click, which is why `by_place` exists
            # and `_lost_quest` uses it.
            self.by_name.setdefault(name, quest)
            if (quest.get("questline") == "main"
                    and quest.get("questline_order") is not None
                    and quest.get("world")):
                self.by_place.setdefault(
                    (quest["world"], quest["questline_order"]), quest)
            for objective in quest.get("objectives") or ():
                self.by_objective.setdefault(_norm(objective), []).append(quest)
        # The same objectives as word-sets, for `_loose_lookup`. One
        # word is not an objective ("Explore" would subset-match half
        # the list), so those never match loosely.
        self.loose = [(frozenset(key.split()), quests_)
                      for key, quests_ in self.by_objective.items()
                      if len(key.split()) >= 2]


def key_for(quest_name) -> str:
    """The stable key for a quest name, for callers keeping their own map.

    `LiveWorker._quest_zone` remembers where each quest was last being
    worked, and it has to key the way this module keys or a read misses
    every write that came through a differently-cased HUD text. One
    function so the two sides cannot drift.
    """
    return _norm(quest_name)


def loaded() -> bool:
    """Whether the list is usable. False is survivable everywhere."""
    return not _load().error and bool(_load().quests)


def load_error() -> str:
    return _load().error


class Position:
    """Where one wizard is: which world's line, and how far along it.

    `order` is None for a quest the data has no place for -- a side
    quest, mostly. Two positions only compare when both have an order
    and both are in the same world.
    """

    __slots__ = ("world", "order", "name", "area", "how", "questline")

    def __init__(self, world=None, order=None, name="", area="", how="",
                 questline=None):
        self.world = world
        self.order = order
        self.name = name
        self.area = area
        self.how = how
        #: "main" for the world's storyline, None for a side quest. The
        #: tracker follows whichever quest is SELECTED, so a wizard that
        #: picks one up has every `tp quest` aimed at it from then on --
        #: which is what "the bot loses the main questline" is.
        self.questline = questline

    def __repr__(self):                          # pragma: no cover - debug
        return f"<Position {self.world} #{self.order} {self.name!r}>"

    @property
    def comparable(self) -> bool:
        return self.world is not None and self.order is not None

    @property
    def on_main(self) -> bool:
        """Is this wizard following the world's storyline?

        A side quest is not a failure in itself -- the scripts pick up
        plenty deliberately. It becomes one when the tracker STAYS on it,
        because every quest teleport then goes to the side quest and the
        party's main-line progress stops without anything saying so.
        """
        return self.questline == "main" and self.order is not None

    @property
    def known(self) -> bool:
        """Did the list recognise the quest at all?"""
        return bool(self.name) and self.how != "not in the list"

    def describe(self) -> str:
        if not self.name:
            return "an unknown quest"
        if self.order is None:
            return f"{self.name!r} (a side quest, with no place in the line)"
        return f"{self.world} main #{self.order}, {self.name!r}"


def position_of(quest_name) -> Position:
    """The wizard's place in its world's line, from the tracked quest name.

    The reliable direction. `VM._fetch_tracked_quest_text` reads exactly
    this string out of the client, so both sides of the comparison come
    from the same place.
    """
    index = _load()
    quest = index.by_name.get(_norm(quest_name))
    if quest is None:
        return Position(name=(quest_name or "").strip(), how="not in the list")
    return Position(world=quest.get("world"),
                    order=quest.get("questline_order"),
                    name=quest.get("name") or "",
                    area=quest.get("area") or "",
                    how="by quest name",
                    questline=quest.get("questline"))


def quest_at(world, order) -> Position:
    """The main-line quest sitting at `#order` in `world`'s line.

    The inverse of `position_of`, and the lookup that turns a diagnosis
    into a cure. `_check_on_questline` could already say "this wizard
    is on a side quest while the party is on #13"; it could not say
    what #13 IS, so the only thing it could do about it was write a
    sentence. Naming the quest is what lets the quest book be opened
    and the right entry clicked.

    Returns an unplaced `Position` carrying a reason when the list has
    nothing at that spot -- a world it does not cover, or an order past
    the end of the line. Callers must not act on an unplaced one: a
    guess here re-tracks the WRONG quest, which is worse than the side
    quest it replaced.
    """
    index = _load()
    quest = index.by_place.get((world, order))
    if quest is None:
        return Position(how=f"the list has no {world} main #{order}")
    return Position(world=quest.get("world"),
                    order=quest.get("questline_order"),
                    name=quest.get("name") or "",
                    area=quest.get("area") or "",
                    how=f"{quest.get('world')} main #{order}, by place",
                    questline=quest.get("questline"))


def position_from_goal(goal, world=None, near=None) -> Position:
    """Fallback: the goal line, only when it points somewhere unambiguous.

    Deliberately willing to fail. "Talk to Professor Winthrop" belongs to
    nine Krokotopia quests between #2 and #19, so a caller that gets an
    answer from this can rely on it, and a caller that gets nothing is no
    worse off than before the list existed.

    `world` narrows by the wizard's zone, `near` by the rest of the
    party's position -- a wizard questing with three others is within a
    few steps of them, so a candidate twelve steps away is not the one.
    Both are hints; neither invents an answer on its own.
    """
    index = _load()
    text = _norm(_strip_zone(goal))
    candidates = index.by_objective.get(text)
    if not candidates:
        candidates = index.by_objective.get(_norm(goal))
    if not candidates:
        candidates = _loose_lookup(index, goal)
    if not candidates:
        return Position(name="", how="no quest lists that objective")

    if world:
        same = [q for q in candidates if q.get("world") == world]
        if same:
            candidates = same
    ordered = [q for q in candidates if q.get("questline_order") is not None]
    if not ordered:
        return Position(name=candidates[0].get("name") or "",
                        world=candidates[0].get("world"),
                        area=candidates[0].get("area") or "",
                        how="the goal matched, but that quest has no order",
                        questline=candidates[0].get("questline"))
    if near is not None and len(ordered) > 1:
        closest = min(abs((q["questline_order"]) - near) for q in ordered)
        ordered = [q for q in ordered
                   if abs(q["questline_order"] - near) == closest]

    orders = {q["questline_order"] for q in ordered}
    if len(orders) != 1:
        spread = f"{min(orders)}-{max(orders)}"
        return Position(how=f"the goal is ambiguous: {len(orders)} quests "
                            f"list it, spanning #{spread}")
    quest = ordered[0]
    return Position(world=quest.get("world"),
                    order=quest.get("questline_order"),
                    name=quest.get("name") or "",
                    area=quest.get("area") or "",
                    how="by goal text",
                    questline=quest.get("questline"))


def _strip_zone(goal) -> str:
    """"Talk To Winthrop in Altar of Kings" -> "Talk To Winthrop".

    The HUD appends where to go; the data does not carry it. Only the
    LAST " in " is cut, so "Collect Key Piece in Chamber of Fire" loses
    the zone and not the piece.
    """
    return re.sub(r"\s+in\s+[^,]+$", "", (goal or "").strip())


def _loose_lookup(index, goal):
    """The quests listing this goal, up to words the HUD dropped.

    The HUD's `Talk To Gordon Flemming` is the data's `Talk to Dr.
    Gordon Flemming`; exact lookup misses it, and the miss is not
    neutral -- a goal that cannot be recognised keeps a stale
    quest-name placement alive (see `goal_disowns`) and cost rev
    30e83468 an endless "2 quests behind" on a wizard that was not.

    ONE direction only: every word of the goal must appear in the
    data's objective. The reverse -- the objective's words a subset of
    the goal's -- reads natural but is a trap: the data carries an
    objective that normalises to just "talk to", and that is a subset
    of every talk goal in the game. It matched `Talk To Gordon
    Flemming` to Krokotopia #12 on the first try.
    """
    goal_t = _tokens(_strip_zone(goal))
    if len(goal_t) < 2:
        return []
    found = []
    for tokens, quests in index.loose:
        if goal_t <= tokens:
            for quest in quests:
                if quest not in found:
                    found.append(quest)
    return found


def goal_disowns(quest_name, goal) -> bool:
    """Does the goal line say the tracked quest NAME is a stale read?

    The name and the goal describe the same tracked quest -- when both
    reads are fresh. They are not always: `_read_goal` keeps the
    previous name on a blank read, because a blank is not evidence of a
    change, so the name can sit one quest in the past while the goal
    has moved on. Rev 30e83468 is the cost: Sebastian's goal read `Talk
    To Sergeant Major Talbot in The Oasis` -- Krokotopia #20, and
    character for character the same text as the wizard he was measured
    against -- while his name still read `Eye of Krok`, #18, and the
    party held a catch-up over a two-quest gap that did not exist.

    True only when the goal provably belongs to some OTHER quest in the
    list: not one of the named quest's own objectives, but listed by a
    different quest. A goal the list has never heard of proves nothing
    -- the data's objectives are incomplete in places (`Into the Map
    Room` lists a prose note instead of steps), and distrusting the
    name over that would unplace every wizard walking such a quest.
    """
    index = _load()
    quest = index.by_name.get(_norm(quest_name))
    if quest is None:
        return False
    goal_t = _tokens(_strip_zone(goal))
    if len(goal_t) < 2:
        return False
    for objective in quest.get("objectives") or ():
        theirs = _tokens(_strip_zone(objective))
        if len(theirs) >= 2 and (goal_t <= theirs or theirs <= goal_t):
            return False
    return bool(_loose_lookup(index, goal))


def furthest_behind(positions):
    """(indices, gap, why) for the wizards that are genuinely behind.

    `positions` is one `Position` per wizard, positionally. Returns
    `([], 0, why)` when the question has no answer.

    A LIST, not a single index, and that is the second half of the same
    correction. This used to refuse whenever two wizards tied at the
    back -- "there is no one to catch up" -- and in a party of three
    that is the ordinary case, not an edge case: rev 8e5a9c75 has two
    wizards on Krokotopia #12 and one on #13 for the whole run. Two
    wizards being equally behind does not mean nobody is behind. It
    means two of them have a step to finish.
    """
    usable = [(i, p) for i, p in enumerate(positions)
              if p is not None and p.comparable]
    if len(usable) < 2:
        return [], 0, ("fewer than two wizards have a place in the line "
                       "(side quests have no order)")
    worlds = {p.world for _i, p in usable}
    if len(worlds) > 1:
        return [], 0, (f"the party is split across {len(worlds)} worlds "
                       f"({', '.join(sorted(worlds))}), which is not a "
                       f"gap in one line")
    lowest = min(p.order for _i, p in usable)
    highest = max(p.order for _i, p in usable)
    gap = highest - lowest
    if gap < BEHIND_BY:
        return [], 0, (f"the party is all on the same quest (#{lowest})")
    at_back = [i for i, p in usable if p.order == lowest]
    who = "1 wizard is" if len(at_back) == 1 else f"{len(at_back)} wizards are"
    return at_back, gap, (f"#{lowest} against #{highest} — {who} {gap} "
                          f"quest(s) behind the furthest ahead")
