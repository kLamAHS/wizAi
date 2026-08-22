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


#: tracked-quest "names" that mean NO quest is selected at all. "Quest
#: Finder" is the journal's own pseudo-entry: the tab the game selects
#: when the quest-finder UI is used, and what `read_quest_name` returns
#: while nothing real is tracked. It reads like a quest name and is the
#: opposite of one.
PSEUDO_ENTRIES = frozenset({"quest finder"})


def no_quest_selected(quest_name) -> bool:
    """Does this tracked-quest name mean NOTHING is selected?

    The state every detector missed in the 115-minute run at rev
    f2b8101f. The script's own lost-quest routine clicks the journal's
    Quest Finder tab, and a cycle that fails partway leaves the journal
    ON that tab — no quest tracked at all. Sebastian sat there for the
    last 25 minutes of the run while every rung looked past him:
    `position_of("Quest Finder")` answers `known=False`, and unknown is
    deliberately skipped by the off-questline check, because unreadable
    must not be called a side quest. This name is not unreadable — it
    is the journal AFFIRMATIVELY saying no quest is selected, which is
    the one thing better evidence than any placement.
    """
    return _norm(quest_name) in PSEUDO_ENTRIES


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

    That last sentence was the whole prediction and it was not guarded.
    Rev 1f912030 walked Wizard City main #2, #3 and #4 and read all
    three as unplaceable:

      * `Skeleton Crew` (#3) and `Monsters and Mazes` (#4) list NO
        objectives at all. A quest that cannot name one step of its own
        can never be shown not to own this one -- and the loop below
        simply did not run, so `_loose_lookup` convicted it unopposed.
      * `Ghost Hunters` (#2) lists one summary line, `Defeat 3 Lost
        Souls`, and its goal `Talk To Private Connelly in Unicorn Way`
        was claimed by `Unicorn's Folly` -- a SIDE quest that happens to
        send you to the same soldier. Side quests share the main line's
        NPCs constantly; that is what a side quest in Unicorn Way IS.

    So both halves now need evidence. The named quest must list steps to
    be measured against, and the claimant must be ON THE LINE -- because
    the failure this exists for is a name one MAIN-LINE quest stale (rev
    30e83468: `Eye of Krok` #18 held while the goal had moved to #20),
    and a side quest sharing an NPC is not that.
    """
    index = _load()
    quest = index.by_name.get(_norm(quest_name))
    if quest is None:
        return False
    goal_t = _tokens(_strip_zone(goal))
    if len(goal_t) < 2:
        return False
    mine = [_tokens(_strip_zone(o)) for o in quest.get("objectives") or ()]
    mine = [t for t in mine if len(t) >= 2]
    if not mine:
        return False
    for theirs in mine:
        if goal_t <= theirs or theirs <= goal_t:
            return False
    return any(other.get("questline") == "main"
               and other.get("questline_order") is not None
               for other in _loose_lookup(index, goal))


#: area display names a world hub may legitimately be hiding under.
#: Taken from the quest data's own area column, which carries six:
#: Baobab Crossroads, Commons, Plaza of Conquests, Regent's Square,
#: Shopping District, The Oasis.
_HUBBISH = re.compile(r"commons|crossroads|hub|plaza|shopping|square|oasis",
                      re.I)

#: area display names that name a world but no useful area, and areas
#: whose name is a common word. Nothing is looked up under these -- an
#: ambiguous area is answered "unknown", and unknown is never evidence.
_VAGUE_AREAS = frozenset({"", "the commons", "commons", "hub", "tower",
                          "shopping district", "the shopping district"})


def goal_area(goal) -> str:
    """The area the goal line says to go to, or "".

    `"Talk To Hoi Mang in Crimson Fields"` -> `"Crimson Fields"`. The
    HUD appends it to every goal that has a destination, `_strip_zone`
    already knows where the seam is, and until now the suffix was cut
    off and thrown away. It is the only part of a quest read that says
    WHERE, independently of the quest arrow -- which matters because
    the arrow is a single last-writer-wins pointer in the game's own
    render loop, with no quest identity attached to it at all.

    A trailing count is dropped with it: Collect goals read
    `"Collect Cog in Triton Avenue (0 of 3)"`.
    """
    text = re.sub(r"<[^>]*>", "", goal or "").strip()
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    match = re.search(r"\s+in\s+([^,]+)$", text)
    return match.group(1).strip() if match else ""


def _area_key(area) -> str:
    """The form both sides of an area-name comparison fold to.

    The HUD writes the article the data leaves off -- "Defeat Belloq in
    the Tanglewood Way" against an area recorded as "Tanglewood Way" --
    and an area that will not match is answered "unknown", which
    silently disables the check it was read for.
    """
    return re.sub(r"^the\s+", "", _norm(area))


def _areas():
    """Area display name -> the one world it is in, where there is one."""
    index = _load()
    built = getattr(index, "_areas", None)
    if built is not None:
        return built
    built = {}
    for quest in index.quests:
        area = _area_key(quest.get("area") or "")
        world = quest.get("world") or ""
        if not area or not world or area in _VAGUE_AREAS:
            continue
        seen = built.get(area, world)
        # An area name that two worlds share cannot answer the
        # question. Recorded as None rather than dropped, so a later
        # quest in a third world cannot revive it.
        built[area] = world if seen == world else None
    index._areas = built
    return built


def world_of_area(area) -> str:
    """The world an area display name belongs to, or "" if unknown.

    "" for an area the data does not list, and for one that two worlds
    share. Both are "no answer", and the caller must treat them as
    such: the questline data's `objectives` are empty for every world
    past Arc 1, but `world` and `area` are populated for all 2,640
    quests, so this answers across the whole game where the goal-text
    placement path cannot.
    """
    return _areas().get(_area_key(area)) or ""


def _world_key(world) -> str:
    """A world name folded to the form both sides can be compared in.

    The data writes `"Wizard City"`, the client writes `"WizardCity"`,
    and one of them writes `"MooShu"` where the other writes
    `"Mooshu"`. Spaces and case are the whole difference.
    """
    return re.sub(r"[^a-z0-9]", "", (world or "").lower())


def world_key(world) -> str:
    """A world name folded so two spellings of it compare equal.

    Public because the comparison is wanted outside this module -- a
    tracker's world against the last main-line quest's world, where one
    side comes from the quest data ("Wizard City") and the other from a
    live zone id ("WizardCity").
    """
    return _world_key(world)


def _worlds() -> frozenset:
    """Every world the quest data names, folded by `_world_key`."""
    index = _load()
    built = getattr(index, "_worlds", None)
    if built is None:
        built = frozenset(_world_key(q.get("world"))
                          for q in index.quests if q.get("world"))
        index._worlds = built
    return built


def world_of_zone(zone) -> str:
    """The world a live zone name belongs to, or "".

    The client's zone name is a path whose first segment is the world
    in a compact form -- `"Zafaria/ZF_Z09_Drum_Jungle"`,
    `"WizardCity/WC_Ravenwood"`, `"Zafaria/Interiors/ZF_Z09_I06_..."`.
    Every zone in rev 09a0af80's export carries it.

    "" when that segment is not a world the data knows -- a housing
    instance, a PvP arena, a bare zone id with the prefix cut off. A
    caller cannot tell "another world" from "a zone I cannot place"
    unless this refuses to guess, and the callers gate rungs that hold
    the script and move the party.
    """
    head = _world_key((zone or "").split("/")[0])
    return head if head in _worlds() else ""


def area_is_this_zone(area, zone):
    """Is this area display name the zone the wizard is standing in?

    True, False or **None for "cannot tell"**, and the three are not
    interchangeable. The client's zone id squashes to the area's words
    for a plain street -- `"Drum Jungle"` inside
    `"Zafaria/ZF_Z09_Drum_Jungle"`, `"Baobab Crown"` inside
    `"Zafaria/ZF_Z01_Baobab_Crown"` -- and that positive match is
    reliable.

    The negative is not, and measuring it on rev ebc4aff8's own fifteen
    zone/goal pairs is what settles the shape of the caller: only two
    matched, and most of the misses were a wizard legitimately walking
    TOWARDS its area. Two whole families never match by name at all: an
    interior (`"Zafaria/Interiors/ZF_Z10_I02_Didos_Mausoleum"` for a
    goal in the Elephant Graveyard) and a world hub under an internal
    alias (`"Zafaria/ZF_Z00_Hub"` is the Baobab Crossroads). Both
    answer None here rather than False, and a caller must treat None as
    no answer.

    So this may VETO an action and may never authorise one. The one
    place it is worth a veto is the realm hop, where refusing costs a
    cooldown and acting wrongly cost rev ebc4aff8 seventy percent of a
    206-minute run.
    """
    area, zone = (area or "").strip(), (zone or "").strip()
    if not area or not zone:
        return None
    if world_of_area(area) and world_of_zone(zone) \
            and _world_key(world_of_area(area)) != world_of_zone(zone):
        return False                     # a different world entirely
    squashed = re.sub(r"[^a-z0-9]", "", zone.lower())
    if _area_key(area).replace(" ", "") in squashed:
        return True
    parts = [p for p in zone.split("/") if p]
    if len(parts) > 2 or any(p.lower() == "interiors" for p in parts):
        return None                      # inside something; unnameable
    # A zone id ABBREVIATES and REORDERS: `KT_ChampHall` is the Hall of
    # Champions, and a whole-name substring test calls that a mismatch
    # -- a false veto on a wizard standing exactly where it should be,
    # which is the "subtler always-on wrongness" this predicate must
    # not trade a stall for. So one shared word of four letters or more
    # is enough to withdraw the veto. It is a weak signal used only to
    # say "cannot tell", which is the direction it is safe to be wrong
    # in.
    if any(word in squashed
           for word in _area_key(area).split() if len(word) >= 4):
        return None
    if len(parts) == 2 and re.search(r"hub|commons", parts[1], re.I) \
            and _HUBBISH.search(area):
        # A world hub carries an internal alias -- `ZF_Z00_Hub` IS the
        # Baobab Crossroads -- so a hub zone cannot be called "not the
        # area" by name alone. But it can only alias a HUB area. A
        # street name against a hub zone is a real mismatch, and that
        # is rev ebc4aff8's exact pair: "Collect Dirt Mound in Cyclops
        # Lane" read for 152 minutes from `WizardCity/WC_Hub`.
        return None
    return False


def goal_is_elsewhere(goal, zone) -> bool:
    """Does the goal name a destination in a DIFFERENT world?

    False unless both sides are known and they disagree -- an unlisted
    area, a name two worlds share, a goal with no `" in "` suffix and a
    zone that will not read all answer False. Refusing to act on
    "unknown" is the whole point: this gates a rung that holds the
    script, and holding it on a guess is the failure it exists to stop.

    The read it was written for: rev 09a0af80's quester, standing in
    the Zafaria hub, alternated between a Zafaria quest whose marker
    read 7,899 units away and a WYSTERIA quest whose marker read 81 and
    then 0. `MARKER_IN_ZONE` assumes another zone always reads six
    figures; a cross-world marker at 0-81 slips straight through it,
    and everything downstream believed the wizard was standing on its
    quest objective. The goal line said "in <a Wysteria area>" the
    whole time.
    """
    mine = world_of_zone(zone)
    theirs = world_of_area(goal_area(goal))
    if not mine or not theirs:
        return False
    return _world_key(theirs) != mine


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
