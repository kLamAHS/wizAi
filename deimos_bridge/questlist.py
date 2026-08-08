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

#: How far apart two orders have to be before it is worth acting on.
#: One step is normal -- a party rarely turns a quest in on the same
#: tick -- and calling that a desync would have wizAi regrouping
#: constantly. Two is a wizard that missed something.
BEHIND_BY = 2

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
        for quest in quests:
            name = _norm(quest.get("name"))
            # First wins. The 17 collisions are all side quests reused
            # across worlds; the main line is unique.
            self.by_name.setdefault(name, quest)
            for objective in quest.get("objectives") or ():
                self.by_objective.setdefault(_norm(objective), []).append(quest)


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


def furthest_behind(positions):
    """(index, gap, why) for the wizard that is genuinely behind.

    `positions` is one `Position` per wizard, positionally. Returns
    `(None, 0, why)` when the question has no answer -- which is most of
    the time, and is the point. It answers only when every comparable
    wizard is in the same world, exactly one is furthest back, and the
    gap is at least `BEHIND_BY`.

    A tie means two wizards are equally behind, and moving one of them
    is not obviously right; a caller that wants to regroup the party can
    still do that without naming anybody.
    """
    usable = [(i, p) for i, p in enumerate(positions)
              if p is not None and p.comparable]
    if len(usable) < 2:
        return None, 0, ("fewer than two wizards have a place in the line "
                         "(side quests have no order)")
    worlds = {p.world for _i, p in usable}
    if len(worlds) > 1:
        return None, 0, (f"the party is split across {len(worlds)} worlds "
                         f"({', '.join(sorted(worlds))}), which is not a "
                         f"gap in one line")
    lowest = min(p.order for _i, p in usable)
    highest = max(p.order for _i, p in usable)
    gap = highest - lowest
    if gap < BEHIND_BY:
        return None, 0, (f"the party is within {gap} step(s) of each other, "
                         f"which is normal")
    at_back = [i for i, p in usable if p.order == lowest]
    if len(at_back) > 1:
        return None, gap, (f"{len(at_back)} wizards are equally far back "
                           f"(#{lowest}), so there is no one to catch up")
    return at_back[0], gap, (f"#{lowest} against #{highest} — {gap} quests "
                             f"behind the furthest ahead")
