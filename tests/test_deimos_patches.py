"""The patches wizAi applies to the vendored Deimos and wizwalker.

Vendored code gets updated, and an update that quietly drops one of
these would put back a bug the run has already been broken by -- with no
test failing, because everything wizAi owns still works. So each patch
has a guard here that fails loudly if the thing it fixed comes back.

Each test names the upstream shape it replaced, so re-applying the patch
after a Deimos bump is a matter of reading the failure.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _source(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


# ------------------------------------------------------- deimoslang waitfor
VM = "Deimos/src/deimoslang/vm.py"


def test_waitfor_is_still_bounded():
    """Upstream `vm.py` has, inside the `waitfor` instruction::

        async def waitfor_coro(coro, invert: bool, interval=0.25):
            while not (invert ^ await coro()):
                await asyncio.sleep(interval)

    No timeout of any kind, so a condition that never becomes true parks
    the VM forever. That is the single largest reason a scripted run
    needs a human -- wizAi's telemetry caught two wizards parked on one
    instruction for minutes on `Talk To` steps, and the run did not
    recover on its own.
    """
    src = _source(VM)
    assert "WAITFOR_TIMEOUTS" in src, \
        "the waitfor timeout patch is gone -- every `waitfor` can hang forever again"
    assert "_waitfor_gave_up(kind, waited, invert)" in src, \
        "the timeout constant survived but nothing acts on it"
    assert "while not (invert ^ await coro()):\n                        await asyncio.sleep(interval)\n\n" not in src, \
        "the unbounded upstream loop is back verbatim"


def test_every_waitfor_kind_has_an_honest_limit():
    """Per kind because the honest limits differ by two orders of
    magnitude: a dialogue box appears in seconds or not at all, and a
    `waitfor battle completion` legitimately spans a long duel."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_vm_probe", ROOT / VM)
    src = _source(VM)
    # Executed as text rather than imported: vm.py pulls in wizwalker,
    # which is Windows-only and not installed here. The constants and
    # the two helpers are self-contained.
    head = src.split("WAITFOR_TIMEOUTS", 1)[1]
    ns = {}
    exec("WAITFOR_TIMEOUTS" + head.split("\n\n\n", 1)[0], ns)
    limits = ns["WAITFOR_TIMEOUTS"]
    assert set(limits) >= {"dialog", "battle", "zonechange"}
    assert limits["dialog"] <= 120, "a dialogue box appears in seconds or not at all"
    assert limits["battle"] >= 600, "a long duel is not a hang"
    assert ns["DEFAULT_WAITFOR_TIMEOUT"] > 0, \
        "an unlisted kind must still be bounded"


def test_giving_up_returns_rather_than_raising():
    """deimoslang scripts are built out of retry loops, so falling
    through a timed-out wait lands in one and gets another attempt.
    Raising would end the program, which is the failure this exists to
    prevent."""
    src = _source(VM)
    block = src.split("async def waitfor_coro", 1)[1].split("async def waitfor_impl", 1)[0]
    assert "return" in block and "raise" not in block


# ---------------------------------------------- wizwalker friends-list waits
UTILS = "Deimos/libs/wizwalker/wizwalker/extensions/scripting/utils.py"


def test_the_friend_teleport_still_waits_long_enough_for_its_windows():
    """Both lookups in `teleport_to_friend_from_list` are a click, a flat
    one-second sleep, then a lookup that gives up 1.6s later. Three game
    clients on one machine lose that race: five of the rejoins in one run
    died on `No child window named MessageBoxModalWindow`, and one on
    `wndCharacter`.

    Checked by driving it rather than by reading it: the fixture makes
    each window appear only after a dozen polls, which the default four
    retries cannot survive and the widened wait can. See
    `_load_ww_utils` below for how the module runs without the game.

    The row-click count is the assertion that bites. Clicking the row
    again is a SECOND cure for a different fault (see
    `test_a_row_click_that_opened_nothing_is_tried_again`), and it
    would paper over a narrowed wait -- three short attempts add up to
    one long one. One click means the first wait was long enough on its
    own.
    """
    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"], slow_windows=12)
    _run_tp(mod, client, name="Konstantin Ice")
    assert client.teleported, "a slow window lost the teleport again"
    rows = [c for c in client.clicks if c.startswith("row@")]
    assert len(rows) == 1, \
        f"the first look gave up early and the retry covered for it: {rows}"


# ------------------------------------------ the friends-list teleport, driven
# Not source-grepped like the guards above: this module is the one piece
# of vendored wizwalker that can be LOADED on a machine without the game,
# because it only touches windows and the mouse. Stub those and the whole
# teleport runs, which is worth doing -- ten of these failed per run at
# rev cfeb9a85 and every one of them was reported as a bare
# `ValueError: No child window named wndCharacter`, a message that names
# the symptom and nothing else.


def _load_ww_utils():
    """The vendored wizwalker friends-list module, importable on Linux.

    Its only imports are `regex`, `WindowFlags` and one wizwalker util,
    so stubbing those three lets the real code run against fake windows.
    Loaded into a throwaway namespace rather than `sys.modules` so
    nothing else in the suite inherits the stubs.
    """
    import asyncio
    import enum
    import re
    import sys
    import types

    enums = types.ModuleType("wizwalker.memory.memory_objects.enums")

    class WindowFlags(enum.IntFlag):
        visible = 1
        noclip = 2
        disabled = 2147483648

    enums.WindowFlags = WindowFlags
    utils_mod = types.ModuleType("wizwalker.utils")

    async def maybe_wait_for_value_with_timeout(*a, **kw):
        return None

    utils_mod.maybe_wait_for_value_with_timeout = maybe_wait_for_value_with_timeout

    saved = {}
    names = ("wizwalker", "wizwalker.memory", "wizwalker.memory.memory_objects",
             "wizwalker.memory.memory_objects.enums", "wizwalker.utils", "regex")
    for name in names:
        saved[name] = sys.modules.get(name)
    try:
        for name in ("wizwalker", "wizwalker.memory",
                     "wizwalker.memory.memory_objects"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["wizwalker.memory.memory_objects.enums"] = enums
        sys.modules["wizwalker.utils"] = utils_mod
        if sys.modules.get("regex") is None:
            stub = types.ModuleType("regex")
            stub.compile = re.compile
            stub.findall = re.findall
            sys.modules["regex"] = stub
        namespace = {"__name__": "_ww_scripting_utils"}
        exec(compile(_source(UTILS), UTILS, "exec"), namespace)
        # The waits are real -- 20 retries at 0.4s is eight seconds per
        # lookup, three times over on a retried click -- and they are
        # guarded by `test_the_friend_teleport_still_waits_long_enough_
        # for_its_windows` reading the source. Nothing here is testing
        # the clock, so the module's own `asyncio` is swapped for one
        # whose sleep returns at once. Everything else is the real
        # module, so a `create_task` or a `wait_for` still works.
        shim = types.ModuleType("asyncio")
        shim.__dict__.update(asyncio.__dict__)

        async def _no_wait(_seconds):
            return None

        shim.sleep = _no_wait
        namespace["asyncio"] = shim
    finally:
        for name, was in saved.items():
            if was is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = was
    return types.SimpleNamespace(**namespace)


#: the flags a window the game has put on screen actually carries: not
#: `visible` alone but the layout bits beside it. Modelled because the
#: bug this file exists to pin down is a whole-word WRITE over them --
#: see `test_hiding_the_friends_list_keeps_the_bits_the_game_set`.
_LAID_OUT = 1 | 2 | 128 | 512          # visible | noclip | dock_left | dock_top


class _Win:
    """A game window with a name, text, children and visibility.

    Visibility is the flag word, not a separate bool. It used to be a
    separate bool and `write_flags` used to be a recorder that changed
    nothing, so the one thing the real window does on a flags write --
    stop being visible, and stop taking clicks -- was unmodellable, and
    the wedge that cost rev 8ebfcf70 its whole run could not be written
    as a test.
    """

    def __init__(self, name, children=(), text="", visible=True, flags=None):
        self._name = name
        self._children = children if callable(children) else list(children)
        self._text = text
        self._visible = visible
        self._flags = _LAID_OUT if flags is None else int(flags)
        if not callable(visible) and not visible:
            self._flags &= ~1
        #: when set, the `disabled` bit comes straight back after any
        #: write -- a window nothing wizAi does can repair.
        self.unwedgeable = False
        #: every value ever written, for tests that care how it was done
        #: rather than what it left behind.
        self.flags_written = []

    async def name(self):
        return self._name

    async def children(self):
        kids = self._children() if callable(self._children) else self._children
        return [kid for kid in kids if kid is not None]

    async def is_visible(self):
        v = self._visible
        if callable(v):
            return bool(v())
        return bool(self._flags & 1)

    async def takes_clicks(self):
        """On screen AND not disabled. The pair is the point: a window
        can be visible and still ignore the mouse, and that state reads
        as a working friends list right up until the row click does
        nothing."""
        return bool(await self.is_visible()) and not (self._flags & 2147483648)

    async def maybe_text(self):
        t = self._text
        return t() if callable(t) else t

    async def flags(self):
        return self._flags

    async def write_flags(self, flags):
        self.flags_written.append(int(flags))
        self._flags = int(flags)
        if self.unwedgeable:
            self._flags |= 2147483648

    async def get_windows_with_name(self, name):
        found = []

        async def walk(win):
            if await win.name() == name:
                found.append(win)
            for kid in await win.children():
                await walk(kid)

        await walk(self)
        return found

    async def scale_to_client(self):
        return _Rect()


class _Rect:
    y1 = 0

    def center(self):
        return (100, 100)


def _entry(icon_list, icon_index, name):
    """One friends-list row, in the markup `_friend_list_entry` parses."""
    return (f"<Y;0><X;0><indent;0><Color;ffffffff><left>"
            f"<icon;FriendsList/Friend_Icon_List_0{icon_list}.dds;1;1;"
            f"{icon_index}></left><Y;0><X;0><indent;1><Color;ffffffff>"
            f"<left><COLOR;ffffffff>{name}")


class _FriendsClient:
    """A client whose friends list and character panel actually behave.

    `panel_for` is the wizard the next row click opens -- set it to
    somebody other than the one asked for and the run should refuse to
    press "Go to Friend".
    """

    def __init__(self, friends=("Konstantin Ice",), page=1, pages=1,
                 opens_panel=True, panel_for=None, confirms=True,
                 panel_text=True, slow_windows=0,
                 tab_label="Online Friends"):
        #: how many polls the panel and the confirmation box take to
        #: appear once they have been asked for. Three game clients on
        #: one machine is what makes this non-zero live.
        self.slow_windows = slow_windows
        self._panel_polls = 0
        self._confirm_polls = 0
        self.friends = list(friends)
        self.page = page
        self.pages = pages
        self.clicks = []
        self.opens_panel = opens_panel
        self._panel_for = panel_for
        self.confirms = confirms
        self.panel_text = panel_text
        self.panel_open = False
        self.confirm_open = False
        self.teleported = False
        me = self

        def list_text():
            return "".join(_entry(1, i, n) for i, n in enumerate(me.friends))

        self.character = _Win("wndCharacter", [
            _Win("btnGoToFriend"),
            _Win("btnCharacterClose"),
            _Win("txtName", text=lambda: (me._panel_name()
                                          if me.panel_text else "")),
        ], visible=lambda: me.panel_open)
        self.confirm = _Win("MessageBoxModalWindow", [_Win("centerButton")],
                            visible=lambda: me.confirm_open)
        self.friends_window = _Win("NewFriendsListWindow", [
            _Win("lblFriendsList", text=f"<center>{tab_label}</center>"),
            _Win("btnListTypeRight"),
            _Win("listFriends", text=list_text),
            _Win("btnArrowDown"),
            _Win("PageNumber",
                 text=lambda: f"<center>{me.page} / {me.pages}</center>"),
        ])
        # The panel and the confirmation box are absent from the tree
        # until the game builds them, which is why the live failure
        # reads "No child window named wndCharacter" rather than "not
        # visible" -- `get_windows_with_name` walks children and does
        # not filter on visibility (`memory_objects/window.py:61`).
        def tree():
            panel = confirm = None
            if me.panel_open:
                me._panel_polls += 1
                if me._panel_polls > me.slow_windows:
                    panel = me.character
            if me.confirm_open:
                me._confirm_polls += 1
                if me._confirm_polls > me.slow_windows:
                    confirm = me.confirm
            return [_Win("btnFriends"), me.friends_window, panel, confirm]

        self.root_window = _Win("root", tree)

        class _Mouse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def click_window(self, win):
                name = await win.name()
                me.clicks.append(name)
                if name == "btnFriends":
                    # The game's own toggle. It re-shows the list; it
                    # does not un-disable a window somebody wrote a
                    # `disabled` bit onto, which is exactly why a wedged
                    # client cannot click its own way out.
                    me.friends_window._flags |= 1
                elif name == "btnArrowDown":
                    me.page = me.page % max(me.pages, 1) + 1
                elif name == "btnGoToFriend":
                    me.confirm_open = bool(me.confirms)
                elif name == "centerButton":
                    me.teleported = True
                    me.confirm_open = False
                elif name == "btnCharacterClose":
                    me.panel_open = False

            async def click(self, x, y):
                me.clicks.append(f"row@{y}")
                # A raw coordinate click only opens anything if the
                # window under it is on screen and taking input. This
                # used to be unconditional, which made the whole class
                # of failure -- a list that reads and does not click --
                # invisible to every test in this file.
                if not await me.friends_window.takes_clicks():
                    return
                me.panel_open = bool(me.opens_panel)

        self.mouse_handler = _Mouse()

        class _Render:
            async def ui_scale(self):
                return 1.0

        self.render_context = _Render()

    def _panel_name(self):
        if self._panel_for is not None:
            return self._panel_for
        return self.friends[0] if self.friends else ""


def _run_tp(mod, client, **kw):
    import asyncio

    return asyncio.run(mod.teleport_to_friend_from_list(client, **kw))


def _await(coro):
    """One coroutine, run to completion. For assertions about a window's
    state after the teleport has already finished."""
    import asyncio

    return asyncio.run(coro)


def test_a_tab_that_never_reads_online_friends_fails_fast_not_forever():
    """Upstream's `_cycle_to_online_friends` is `while (text) != "Online
    Friends": click`, with no exit — a label that never reads exactly
    that parks the caller for the rest of the run, and every failure
    surfaces as an opaque outer timeout. Rev 676d6e77's booster failed
    the cross-zone follow continuously for four hours with exactly that
    45s-timeout signature. Bounded now: two full laps of the tab
    carousel, then a ValueError that names what the label actually
    reads."""
    import pytest

    mod = _load_ww_utils()
    client = _FriendsClient(tab_label="Freunde")
    with pytest.raises(ValueError) as err:
        _run_tp(mod, client, name="Konstantin Ice")
    assert "never showed 'Online Friends'" in str(err.value)
    assert "Freunde" in str(err.value)


def test_the_cross_zone_follow_prefers_the_hardened_copy():
    """`wizwalker.extensions.scripting` resolves to the pip-installed
    fork's UNHARDENED copy at runtime — `deimos_path`'s overlay can only
    add missing extension subpackages, never shadow ones the fork ships
    — so every hardening in the vendored utils was dead code in wizAi's
    own follow. The loader reads the vendored file by path and execs it;
    its imports are all absolute and resolve against the installed
    wizwalker, so the follow runs hardened logic over the running
    primitives, with the stock import as the fallback."""
    src = _source("deimos_bridge/party.py")
    assert "def _hardened_friend_tp" in src
    body = src.split("async def teleport_to_leader_across_zones", 1)[1]
    body = body.split("\nasync def ", 1)[0]
    assert "_hardened_friend_tp()" in body
    assert body.index("_hardened_friend_tp()") < body.index(
        "from wizwalker.extensions.scripting import"), \
        "the stock import must be the fallback, not the first choice"


def test_the_hardened_loader_runs_against_the_installed_wizwalker():
    """Driven with the same stubs `_load_ww_utils` uses standing in for
    the installed fork: the loader execs the vendored file and hands
    back its `teleport_to_friend_from_list`."""
    import enum
    import re as _re
    import sys
    import types

    from deimos_bridge import party

    enums = types.ModuleType("wizwalker.memory.memory_objects.enums")

    class WindowFlags(enum.IntFlag):
        visible = 1

    enums.WindowFlags = WindowFlags
    utils_mod = types.ModuleType("wizwalker.utils")

    async def maybe_wait_for_value_with_timeout(*a, **kw):
        return None

    utils_mod.maybe_wait_for_value_with_timeout = \
        maybe_wait_for_value_with_timeout
    names = ("wizwalker", "wizwalker.memory", "wizwalker.memory.memory_objects",
             "wizwalker.memory.memory_objects.enums", "wizwalker.utils",
             "regex")
    saved = {name: sys.modules.get(name) for name in names}
    kept = dict(party._HARDENED)
    try:
        for name in ("wizwalker", "wizwalker.memory",
                     "wizwalker.memory.memory_objects"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["wizwalker.memory.memory_objects.enums"] = enums
        sys.modules["wizwalker.utils"] = utils_mod
        if saved["regex"] is None:
            stub = types.ModuleType("regex")
            stub.compile = _re.compile
            sys.modules["regex"] = stub
        party._HARDENED.update(fn=None, tried=False)
        fn = party._hardened_friend_tp()
        assert callable(fn), "the hardened copy did not load"
    finally:
        party._HARDENED.update(kept)
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_a_friend_teleport_that_works_closes_everything_it_opened():
    """Upstream closed the friends window on the LAST line of the happy
    path and the character panel inside `_teleport_to_friend`. Both are
    still closed -- a wizard behind its own friends list cannot walk."""
    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"])
    _run_tp(mod, client, name="Konstantin Ice")
    assert client.teleported
    assert not client.panel_open, "the character panel was left open"
    assert not _await(client.friends_window.is_visible()), \
        "the friends list was left up"


def test_a_failed_teleport_does_not_leave_the_wizard_behind_two_modals():
    """The compounding one, and the reason ten of these failed in one
    run rather than one. Upstream raised out of the middle with
    `wndCharacter` on screen on top of a friends list nothing closed
    either, so every attempt after the first started by clicking through
    two windows the wizard could not walk out of."""
    import pytest

    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"], opens_panel=False)
    with pytest.raises(ValueError):
        _run_tp(mod, client, name="Konstantin Ice")
    assert not client.panel_open
    assert not _await(client.friends_window.is_visible()), \
        "the friends list was left up after a failed teleport"


def test_a_confirmation_that_never_comes_still_closes_the_panel():
    import pytest

    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"], confirms=False)
    with pytest.raises(ValueError):
        _run_tp(mod, client, name="Konstantin Ice")
    assert not client.teleported
    assert not client.panel_open
    assert not _await(client.friends_window.is_visible())


def test_a_row_click_that_opened_nothing_is_tried_again():
    """The online friends list RE-SORTS as people change zone, so the
    row that matched the text a moment ago can hold nobody by the time
    the mouse arrives. Upstream got one click and then ended the whole
    call on `No child window named wndCharacter`."""
    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"], opens_panel=False)

    real_click = client.mouse_handler.click
    tries = []

    async def flaky(x, y):
        tries.append(y)
        if len(tries) >= 2:
            client.opens_panel = True
        await real_click(x, y)

    client.mouse_handler.click = flaky
    _run_tp(mod, client, name="Konstantin Ice")
    assert len(tries) == 2, tries
    assert client.teleported


def test_the_panel_is_checked_against_the_name_before_teleporting():
    """A stale row offset still opens a panel -- somebody else's. In a
    party of three, everyone on everyone's list, teleporting to the
    wrong one is a party split that reports as a success."""
    import pytest

    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"],
                            panel_for="Sebastian Life")
    with pytest.raises(ValueError) as caught:
        _run_tp(mod, client, name="Konstantin Ice")
    assert "somebody other than" in str(caught.value)
    assert not client.teleported, "teleported to the wrong wizard"


def test_a_panel_whose_name_will_not_read_is_still_allowed_through():
    """`None` is not `False`. Unreadable text is not evidence of the
    wrong wizard, and refusing on it would break the teleport on every
    client whose panel keeps the name somewhere this cannot walk to."""
    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"], panel_text=False)
    _run_tp(mod, client, name="Konstantin Ice")
    assert client.teleported


def test_turning_back_to_an_earlier_page_is_not_a_silent_no_op():
    """Upstream::

        for _ in range(target_page - current_page):
            await client.mouse_handler.click_window(right_button)

    `range()` of a negative number is empty, so a friend on page 1 with
    the window showing page 2 turned nothing -- and then clicked the row
    offset anyway, on the wrong page. That is not an error, it is a
    teleport to whoever happened to be sitting there."""
    mod = _load_ww_utils()
    # Twelve friends is two pages; the target is row 0, page 1, and the
    # window opens on page 2.
    friends = [f"Wizard{i} Name{i}" for i in range(12)]
    client = _FriendsClient(friends=friends, page=2, pages=2,
                            panel_for="Wizard0 Name0")
    _run_tp(mod, client, name="Wizard0 Name0")
    assert client.page == 1, "the list never turned back to page 1"
    assert client.teleported


def test_a_page_that_will_not_turn_is_reported_as_that():
    """Not as "could not find friend" -- the caller retries other
    spellings of a name that was just matched, and gives up saying the
    wizard is not on the list."""
    import pytest

    mod = _load_ww_utils()
    friends = [f"Wizard{i} Name{i}" for i in range(12)]
    client = _FriendsClient(friends=friends, page=2, pages=2)

    async def stuck(win):
        client.clicks.append(await win.name())

    client.mouse_handler.click_window = stuck
    with pytest.raises(ValueError) as caught:
        _run_tp(mod, client, name="Wizard0 Name0")
    assert "would not turn" in str(caught.value)
    assert "Could not find friend" not in str(caught.value)


def test_a_friend_who_is_not_on_the_list_still_says_so_exactly():
    """`party.py` matches this string to tell "try another spelling"
    from "this teleport is broken". Changing it silently turns every
    not-found into a fatal error."""
    import pytest

    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"])
    with pytest.raises(ValueError) as caught:
        _run_tp(mod, client, name="Nobody Here")
    assert "Could not find friend" in str(caught.value)


# ------------------------------------- the friends list wizAi wedged itself
# Rev 8ebfcf70 is the whole reason this block exists. A two-wizard run:
# seventeen cross-zone follows, seventeen failures, one error every time
# -- `No child window named wndCharacter; hardened copy` -- in a public
# hub as readily as inside a dungeon. The booster never once reached the
# quester, the quester fought a 2400hp boss alone three times and lost
# all three, and the export blamed the instance for refusing teleports
# it was never asked for.
#
# It was wizAi's own tidy-up. `_close_friend_windows` put the list away
# with `write_flags(WindowFlags.disabled)`, a WHOLE-WORD write of
# 0x80000000 over a field that also holds `visible`, `noclip` and every
# dock bit -- and it set the bit that makes a window ignore the mouse.
# Nothing restored any of it. So the list still READ (text does not need
# an enabled window, which is why the friend was always found) and no
# longer CLICKED, forever, from the first teleport of the run onward.


def test_hiding_the_friends_list_keeps_the_bits_the_game_set():
    """The fix, stated as the invariant: closing the list may clear
    `visible` and nothing else. Every other bit is the game's layout,
    and a window that comes back without it is a window the game cannot
    put on screen properly."""
    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"])
    laid_out = client.friends_window._flags
    assert laid_out & mod.WindowFlags.visible, "fixture is not on screen"
    _run_tp(mod, client, name="Konstantin Ice")
    assert client.teleported
    left = client.friends_window._flags
    assert not (left & mod.WindowFlags.visible), "the list was left up"
    assert not (left & mod.WindowFlags.disabled), \
        "the close set the bit that makes the window ignore the mouse"
    assert left == laid_out & ~mod.WindowFlags.visible, \
        (f"the close overwrote the flags word instead of clearing one "
         f"bit: {laid_out:#x} -> {left:#x}")


def test_a_second_teleport_still_works_after_the_first_closed_the_list():
    """The regression in one sentence. Two teleports in a row on one
    client -- which is every run -- and the second must work. Under the
    old whole-word close it could not: the first one wedged the window
    and the second died on `wndCharacter` without the game ever being
    asked."""
    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"])
    _run_tp(mod, client, name="Konstantin Ice")
    assert client.teleported
    client.teleported = False
    _run_tp(mod, client, name="Konstantin Ice")
    assert client.teleported, \
        "the first teleport left the friends list unable to take a click"


def test_a_wedged_friends_list_is_cleared_on_the_way_in():
    """...and a client wedged by something else is repaired rather than
    written off. `Deimos/src/utils.py` has its own copy of the
    destructive close, so a client Deimos has driven this session
    arrives already wedged -- and a wizard that has been failing this
    teleport all run has to start working again on the next attempt,
    not on the next launch."""
    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"])
    # Exactly what the old close left behind.
    client.friends_window._flags = int(mod.WindowFlags.disabled)
    _run_tp(mod, client, name="Konstantin Ice")
    assert client.teleported, "a wedged client was never un-wedged"


def test_a_list_that_will_not_come_up_says_so_instead_of_wndCharacter():
    """The message is the deliverable. `No child window named
    wndCharacter` reads like the game refusing a teleport; it is the
    character panel that the row click never opened, two steps before
    the game is asked for anything. `party.never_asked` keys on this
    wording to stop the export blaming the dungeon."""
    import pytest

    from deimos_bridge import party

    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"])
    # Wedged, and the toggle cannot fix it: `btnFriends` re-shows a
    # window, it does not re-enable one.
    client.friends_window._flags = int(mod.WindowFlags.disabled)
    client.friends_window.unwedgeable = True
    with pytest.raises(ValueError) as caught:
        _run_tp(mod, client, name="Konstantin Ice")
    said = str(caught.value)
    assert "wndCharacter" not in said, \
        f"still reporting the symptom rather than the cause: {said}"
    assert "never attempted" in said or "would not come up" in said
    assert party.never_asked(caught.value), \
        "party.py cannot tell this apart from the game refusing"


def test_a_flags_word_that_will_not_read_does_not_block_the_teleport():
    """No evidence is not evidence. The precondition refuses a window
    that READS as unusable; a window whose flags will not read at all
    is let through and judged by what happens next, because blocking
    on it would stop a client that was about to work for a reason
    nobody could act on."""
    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"])

    async def _no_read():
        raise RuntimeError("the flags word would not read")

    client.friends_window.flags = _no_read
    _run_tp(mod, client, name="Konstantin Ice")
    assert client.teleported, "an unreadable window blocked a good teleport"


def test_the_row_is_never_clicked_at_a_window_that_cannot_take_it():
    """The precondition, stated. A click aimed at a wedged window is a
    click into empty HUD, and every second spent waiting for the panel
    it cannot open is a second the booster is not with its quester."""
    import pytest

    mod = _load_ww_utils()
    client = _FriendsClient(friends=["Konstantin Ice"])
    client.friends_window._flags = int(mod.WindowFlags.disabled)
    client.friends_window.unwedgeable = True
    with pytest.raises(ValueError):
        _run_tp(mod, client, name="Konstantin Ice")
    assert not [c for c in client.clicks if c.startswith("row@")], \
        "a row was clicked at a window that ignores clicks"


# ------------------------------------------------------------ navmap_tp result
TP = "Deimos/src/teleport_math.py"


def test_navmap_tp_still_says_whether_it_landed():
    """Upstream every `return` in `navmap_tp` is bare -- including the
    `if not await is_free(client): return` on its first line -- so
    success, failure and never-attempted are the same answer. That is
    why deimoslang wraps `tp` in unbounded retry loops (331 of them
    across the arc scripts), and why "one wizard gets through with a
    teleport but the others get stuck" was invisible to everything."""
    src = _source(TP)
    assert "on_teleport_result" in src, \
        "the navmap_tp result patch is gone -- a failed teleport is silent again"
    body = src.split("async def navmap_tp", 1)[1].split("\ndef ", 1)[0]
    bare = [line for line in body.splitlines()
            if line.strip() == "return"]
    assert not bare, f"navmap_tp still has bare returns: {bare}"
    assert body.count("_tp_result(") >= 6, \
        "not every exit from navmap_tp reports a result"


def test_a_teleport_that_was_never_attempted_is_a_failure_not_a_success():
    """The first line is `if not await is_free(client): return`. A
    wizard that was not free never moved, and reporting that as success
    is how the script marches on to the next instruction."""
    src = _source(TP)
    head = src.split("async def navmap_tp", 1)[1].split("starting_zone", 1)[0]
    # Whitespace-insensitive: the call is multi-line now that it names
    # which of the three `is_free` conditions refused it.
    flat = " ".join(head.split())
    assert "_tp_result( False" in flat or "_tp_result(False" in flat, \
        "the never-attempted case is no longer reported as a failure"


def test_a_teleport_waits_for_a_transient_block_instead_of_dropping_it():
    """`is_free` is three conditions -- loading, in a duel, dialogue box
    up -- and upstream a `tp` issued while any holds is discarded on the
    first line of `navmap_tp`. Rev 1d28f745 caught that happening eight
    times to each of three wizards, within milliseconds of each other:
    the script telling the party to teleport while the game says no."""
    src = _source(TP)
    assert "wait_until_free" in src, \
        "a tp issued during a loading screen is being dropped again"
    head = src.split("async def navmap_tp", 1)[1].split("starting_zone", 1)[0]
    assert "await wait_until_free(client)" in head, \
        "navmap_tp no longer waits before giving up"


def test_the_wait_refuses_a_duel_rather_than_sitting_through_one():
    """Two of the three conditions clear in seconds. A duel lasts
    minutes and cannot be teleported out of at all, so waiting one out
    would park the script for the length of a fight."""
    src = _source(TP)
    block = src.split("async def wait_until_free", 1)[1].split("\n\n\n", 1)[0]
    assert "BLOCK_DUEL" in block, \
        "the wait no longer singles a duel out from the other blocks"
    i = block.index("BLOCK_DUEL")
    assert "return BLOCK_DUEL" in block[i:i + 120], \
        "being in a duel must end the wait, not extend it"


def test_a_stuck_loading_flag_is_told_apart_from_a_real_zone_load():
    """`client.is_loading()` is true for `TransitionWindow` -- a real
    transition, seconds -- and for `PageFlip`, which is a book and can
    sit there forever. `is_free` folds both into one bool, so rev
    bb8f2b3c dropped nineteen teleports over five minutes all saying
    "still loading or in dialogue", while the same wizard was standing
    still, out of combat, with no dialogue box and an NPC popup up."""
    src = _source(TP)
    block = src.split("async def blocked_by", 1)[1].split("\n\n\n", 1)[0]
    assert "TransitionWindow" in block and "PageFlip" in block, \
        "the two halves of is_loading() are folded together again"
    assert "BLOCK_DUEL" in block and "BLOCK_DIALOGUE" in block, \
        "blocked_by no longer names the non-loading blocks"


def test_a_loading_flag_that_never_clears_is_pushed_through():
    """A teleport dropped because the client claims to be loading, when
    it has claimed that for longer than any zone load takes, is a run
    that has stopped. Attempting it costs a write the load would
    overwrite; not attempting it costs the run."""
    src = _source(TP)
    head = src.split("async def navmap_tp", 1)[1].split("starting_zone", 1)[0]
    assert "LOADING_BLOCKS" in head, \
        "navmap_tp treats every block the same again"
    assert "_tp_note(" in head, \
        "overriding the loading gate has to be visible to a human"
    flat = " ".join(head.split())
    assert "return _tp_result( False" not in flat.split("LOADING_BLOCKS", 1)[1] \
        .split("elif", 1)[0], \
        "a stale loading flag is being reported as a dropped teleport again"


def test_the_wait_is_bounded():
    src = _source(TP)
    ns = {}
    exec("TP_FREE_WAIT" + src.split("TP_FREE_WAIT", 1)[1].split("\n\n\n", 1)[0],
         ns)
    assert 0 < ns["TP_FREE_WAIT"] <= 30, \
        "long enough for a loading screen, short enough not to be a hang"


# --------------------------------------------- one wizard's failure, one wizard
def test_a_mass_instruction_does_not_cancel_the_wizards_that_were_fine():
    """`asyncio.TaskGroup` is documented to cancel every remaining task
    when one raises. Upstream all thirteen mass instructions fan out
    inside one, so a single wizard's transient failure cancels the other
    three MID-INSTRUCTION and the VM step dies with an ExceptionGroup.

    The operator's report is the symptom exactly: "if they were all in
    the same zone, the same teleport should work for all of them, but
    some will just not move at all"."""
    src = _source(VM)
    assert "class PartyTaskGroup" in src, \
        "the non-cancelling group is gone -- one wizard's bad luck is the " \
        "party's problem again"
    # Statements only -- the class docstring quotes the upstream line.
    used = [l.strip() for l in src.splitlines()
            if l.strip().startswith("async with asyncio.TaskGroup()")]
    assert not used, \
        f"a mass instruction is back on asyncio.TaskGroup, which cancels " \
        f"the wizards that were doing fine: {used}"
    assert src.count("async with PartyTaskGroup(") >= 13, \
        "not every mass instruction was moved off TaskGroup"
    body = src.split("class PartyTaskGroup", 1)[1].split("\ndef ", 1)[0]
    assert "return_exceptions=True" in body, \
        "gather without return_exceptions has the same first-failure-wins " \
        "behaviour TaskGroup does"


def test_every_wizard_in_a_wait_is_actually_waited_on():
    """Upstream: `async def proxy(): return await method(client)` inside
    `for client in clients`. Python closes over the loop VARIABLE and no
    task runs until the group awaits it, so all N waits poll the LAST
    wizard. `waitfor dialog` returns when wizard four has a dialogue box
    whatever the other three are doing, and the script marches on with
    three wizards it never waited for. That is the desync."""
    src = _source(VM)
    block = src.split('case "waitfor":', 1)[1].split('case "sendkey":', 1)[0]
    proxies = [line.strip() for line in block.splitlines()
               if line.strip().startswith("async def proxy(")]
    assert proxies, "the waitfor proxies are gone; check this test still fits"
    for line in proxies:
        assert "client=client" in line, \
            f"this wait closes over the loop variable, so it polls only the " \
            f"last wizard: {line}"
    zone = block.split("WaitforKind.zonechange", 1)[1]
    assert "starting_zone=starting_zone" in zone, \
        "the zone-change wait compares every wizard against the last " \
        "wizard's starting zone"


def test_a_quest_teleport_reads_its_position_inside_its_own_task():
    """Upstream `pos = await client.quest_position.position()` is awaited
    in the fan-out loop, so a wizard whose quest position will not
    resolve stops the loop before the wizards after it are given a task
    at all. It is the most-used instruction in the arc scripts."""
    src = _source(VM)
    block = src.split("TeleportKind.quest:", 1)[1].split("TeleportKind.", 1)[0]
    assert "async def tp_to_quest" in block, \
        "the quest teleport reads every position up front again"
    lines = [l.strip() for l in block.splitlines()]
    body = lines.index("async def tp_to_quest(client):")
    fan = next(i for i, l in enumerate(lines) if l.startswith("for client in"))
    assert body < fan, "the read has to be inside the task, not before it"


def test_the_non_cancelling_group_really_does_not_cancel(monkeypatch):
    """Not a source check. `PartyTaskGroup` is pure asyncio, so it can be
    lifted out of vm.py and run: three wizards, one of them raising, and
    the other two must still finish."""
    import asyncio

    src = _source(VM)
    block = ("class PartyTaskGroup"
             + src.split("class PartyTaskGroup", 1)[1]
                  .split("class VMError", 1)[0])
    ns = {"asyncio": asyncio, "on_party_task_failed": None}
    exec(block, ns)

    finished, told = [], []
    ns["on_party_task_failed"] = lambda what, fails: told.append((what, fails))

    async def ok(name):
        await asyncio.sleep(0.01)
        finished.append(name)

    async def boom(name):
        raise RuntimeError(f"{name} lost a memory read")

    async def run():
        async with ns["PartyTaskGroup"]("teleport") as tg:
            tg.create_task(boom("wizard 1"))
            tg.create_task(ok("wizard 2"))
            tg.create_task(ok("wizard 3"))

    asyncio.run(run())                      # must not raise an ExceptionGroup
    assert finished == ["wizard 2", "wizard 3"], \
        f"the wizards that were fine got cancelled anyway: {finished}"
    assert told and told[0][0] == "teleport"
    assert len(told[0][1]) == 1 and isinstance(told[0][1][0][1], RuntimeError)


def test_the_upstream_group_would_have_cancelled_them():
    """The control. If this ever stops holding, `asyncio.TaskGroup`
    changed and the patch above may no longer be needed."""
    import asyncio

    finished = []

    async def ok(name):
        await asyncio.sleep(0.01)
        finished.append(name)

    async def boom():
        raise RuntimeError("lost a memory read")

    async def run():
        async with asyncio.TaskGroup() as tg:
            tg.create_task(boom())
            tg.create_task(ok("wizard 2"))
            tg.create_task(ok("wizard 3"))

    try:
        asyncio.run(run())
    except BaseException:
        pass
    assert finished == [], \
        "asyncio.TaskGroup no longer cancels siblings; re-check the patch"


# ------------------------------------------------- the friends list name match
UTILS_SRC = "Deimos/src/utils.py"


def test_a_first_name_still_finds_a_full_friends_list_entry():
    """Upstream `_cycle_friends_list` compares `friend_name == name`, and
    Wizard101's friends list holds the FULL name -- a live run's list
    reads "Sebastian S." and "Phoenix SkarabaeusSender" -- while what
    supplies a name to a friend teleport usually has only the first: a
    combat member read gives "Sebastian".

    So every regroup raised `Could not find friend with ... name
    Sebastian` on a friend that was right there in the list."""
    src = _source(UTILS_SRC)
    assert "def _same_wizard" in src, \
        "the first-name match is gone -- friend teleports fail on full names"
    block = src.split("def _cycle_friends_list", 1)[1].split("\nasync def ", 1)[0]
    assert "_same_wizard(friend_name, name)" in block, \
        "the loose match is no longer consulted"
    assert 'friend_name == name' in block, \
        "the exact match must still be tried first"


def test_a_first_name_that_two_friends_answer_to_is_refused():
    """"could be" is not "is". Taking an ambiguous first name would
    teleport the party to the wrong wizard, which is worse than not
    teleporting at all."""
    src = _source(UTILS_SRC)
    block = src.split("def _cycle_friends_list", 1)[1].split("\nasync def ", 1)[0]
    assert "ambiguous" in block, \
        "an ambiguous first name is being taken as a match again"


def test_the_name_match_runs_as_written():
    """Not a source check. Lifted out and run against the names in a
    live friends list."""
    src = _source(UTILS_SRC)
    ns = {}
    exec(src[src.index("def _same_wizard"):
             src.index("async def _cycle_friends_list")], ns)
    same = ns["_same_wizard"]

    # what a live run actually had
    assert same("sebastian s.", "sebastian")
    assert same("ph\u00f6nix skarab\u00e4ussender", "ph\u00f6nix")
    # a full name on both sides, and the list's abbreviation of it
    assert same("sebastian silverstaff", "sebastian silverstaff")
    assert same("sebastian s.", "sebastian silverstaff")
    # and the ones it must NOT match
    assert not same("sebastianus k.", "sebastian")
    assert not same("konstantin v.", "sebastian")
    assert not same("sebastian s.", "sebastian jones")
    assert not same("", "sebastian")


# ------------------------------------------------- a query is not a corpse
WW_UTILS = "Deimos/libs/wizwalker/wizwalker/utils.py"


def test_a_failed_liveness_query_is_not_a_dead_client():
    """Upstream `check_if_process_running` is::

        exit_code = ctypes.wintypes.DWORD()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        return exit_code.value == 259

    The BOOL is discarded. `GetExitCodeProcess` returns zero on failure
    and leaves `exit_code` at 0, which is not 259, which reads as "the
    client has exited" -- and `MemoryReader.read_bytes` turns that into
    `ClientClosedError`, which the deimoslang VM has no handler for, so
    the instruction re-enters and throws for the rest of the run. A live
    party did exactly that for twenty-five attempts with the client
    still on screen.
    """
    import ctypes
    import ctypes.wintypes

    src = _source(WW_UTILS)
    body = src[src.index("def check_if_process_running"):]
    body = body[:body.index("\ndef ", 1)]

    calls = []

    class _Kernel32:
        def __init__(self, ok, code):
            self._ok, self._code = ok, code

        def GetExitCodeProcess(self, handle, out):
            calls.append(handle)
            if self._code is not None:
                ctypes.cast(out, ctypes.POINTER(
                    ctypes.wintypes.DWORD)).contents.value = self._code
            return self._ok

    def run(ok, code):
        ns = {"ctypes": ctypes, "kernel32": _Kernel32(ok, code)}
        exec(body, ns)
        return ns["check_if_process_running"](1234)

    # the query failed: inconclusive, and the safe answer is "running"
    assert run(0, None) is True
    # the query worked and said STILL_ACTIVE
    assert run(1, 259) is True
    # the query worked and reported a real exit code -- this one IS dead
    assert run(1, 0) is False
    assert run(1, 1) is False
    assert calls, "the patch stopped calling GetExitCodeProcess at all"


# ------------------------------------------------- deimoslang lang keys
def test_a_literal_name_key_cannot_kill_the_program():
    """Upstream `vm.py` fetched quest names as::

        name: str = await client.cache_handler.get_langcode_name(name_key)

    and `get_langcode_name` mis-splits any key without an underscore:
    `find("_")` returns -1, so the "file" becomes the key minus its
    last character and the lookup raises `ValueError: No lang file
    named Quest Finde`. The journal's Quest Finder pseudo-entry answers
    exactly that literal key, and at rev 7d9b6d6b one wizard's journal
    in that state crash-looped the whole party's program every 15
    seconds -- eleven reloads in three minutes. Upstream even knew:
    `_fetch_quest_text` special-cased the exact string "Quest Finder",
    while `_fetch_tracked_quest_text` -- the one every `quest` check
    runs -- guarded nothing.
    """
    src = _source(VM)
    assert "async def _lang_name" in src, \
        "the literal-key guard is gone -- a Quest Finder journal kills the run again"
    assert src.count("await self._lang_name(client, name_key)") == 2, \
        "both quest-text fetchers must route through the guard"
    assert 'if name_key == "Quest Finder":' not in src, \
        "the upstream one-string special case is back instead of the guard"


def test_the_literal_key_guard_answers_the_key_itself():
    """Behavior, not just presence: a key with no underscore never
    reaches the cache handler, and a lookup that raises falls back to
    the raw key rather than out of the instruction."""
    import asyncio

    src = _source(VM)
    seg = src.split("async def _lang_name", 1)[1]
    seg = seg.split("async def _fetch_tracked_quest_text", 1)[0]
    ns = {"SprintyClient": object}
    exec("class _Probe:\n    async def _lang_name" + seg, ns)
    probe = ns["_Probe"]()

    class _Cache:
        async def get_langcode_name(self, key):
            if key == "QuestTitles_00123":
                return "The Sphinx's Riddle"
            raise ValueError(f"No lang file named {key[:key.find('_')]}")

    class _Client:
        cache_handler = _Cache()

    run = asyncio.run
    assert run(probe._lang_name(_Client(), "Quest Finder")) == "Quest Finder"
    assert run(probe._lang_name(_Client(), "QuestTitles_00123")) \
        == "The Sphinx's Riddle"
    assert run(probe._lang_name(_Client(), "Unknown_999")) == "Unknown_999"
    assert run(probe._lang_name(_Client(), None)) == ""


# ------------------------------------------------- the shipped quester presets
#
# These are vendored data, and they get updated from upstream like any
# other vendored thing — so the fixes applied to them need the same
# guard the code patches have.
SCRIPTS = sorted((ROOT / "deimos_bridge" / "data" / "scripts").glob("*.txt"))
_LOOP = re.compile(r"^\t*(until|while)\b.*\{\s*$")
_GATE = re.compile(r"^\t*if\s+p\d\s+inzone\b.*\{\s*$")


def _block_end(lines, i):
    depth = 0
    for j in range(i, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        if depth <= 0 and j > i:
            return j
    return None


def _own_level(lines, start, end, indent):
    out, depth = [], 0
    for k in range(start + 1, end):
        ln = lines[k]
        if depth == 0 and ln.strip() \
                and len(ln) - len(ln.lstrip("\t")) == indent:
            out.append(k)
        depth += ln.count("{") - ln.count("}")
    return out


def _spinning_loops(text):
    """[(line, header)] for loops that can never make progress.

    The shape: a loop whose ENTIRE body is `if pN inzone Z { ... }` with
    no else. Nothing in it travels to Z, so a wizard standing anywhere
    else spins it until the run ends.
    """
    lines = text.split("\n")
    found = []
    for i, ln in enumerate(lines):
        if not _LOOP.match(ln):
            continue
        end = _block_end(lines, i)
        if end is None:
            continue
        indent = len(ln) - len(ln.lstrip("\t"))
        ks = _own_level(lines, i, end, indent + 1)
        gates = [k for k in ks if _GATE.match(lines[k])]
        other = [k for k in ks if not _GATE.match(lines[k])
                 and not lines[k].strip().startswith("if DebugMode")]
        if len(gates) != 1 or other:
            continue
        chain_end = _block_end(lines, gates[0])
        gi = len(lines[gates[0]]) - len(lines[gates[0]].lstrip("\t"))
        if chain_end is None:
            continue
        if any(lines[k].strip().startswith("} else")
               and len(lines[k]) - len(lines[k].lstrip("\t")) == gi
               for k in range(gates[0], chain_end + 1)):
            continue
        found.append((i + 1, ln.strip()))
    return found


def test_the_presets_are_shipped():
    assert len(SCRIPTS) == 6, [p.name for p in SCRIPTS]


def test_no_preset_has_a_loop_that_cannot_make_progress():
    """The failure the operator watched for three runs: "it's just
    standing around the world portal".

    TTS Arc 1 handles Krokotopia #31 with::

        until NOT p1 tracking_goal "collect gemstones in hall of champions" {
            if p1 inzone Krokotopia/KT_Krokosphinx/KT_ChampHall {
                p1 tp XYZ(-15546.16, 5727.71, 0.0)

    Nothing in that loop travels to the Hall of Champions. Solo it never
    needed to -- the step before it ENDS there, so the guard is free --
    but three wizards hold three quest states while one instruction
    pointer serves them, so p1 reaches its own leg standing anywhere.
    In the wrong zone the guard is false, the teleport to the gems is
    never issued, and the goal cannot change because the body never
    runs: 15,785 instructions, nobody moving.

    The author's own fix is seventy lines above it, on `minion
    massacre`::

        if p1 inzone Krokotopia/KT_Krokosphinx/KT_ChampHall {
            ...
        } else {
            break
        }

    Breaking out lets the enclosing dispatch run again, and re-enter
    when the wizard IS there. `break` jumps to `_loop_label_stack[-1]`
    (`ir.py:406`), the innermost loop, so it leaves the stuck loop and
    nothing else.
    """
    offenders = {p.name: _spinning_loops(p.read_text(encoding="utf-8"))
                 for p in SCRIPTS}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, "\n".join(
        f"{name} line {ln}: {head}"
        for name, hits in offenders.items() for ln, head in hits[:5])


def test_every_preset_still_parses():
    """The guard on the guard: an edit that fixes a spin and breaks the
    compiler has made the run worse, and nothing else here would say
    so. deimoslang's parser needs no wizwalker, so this is checkable."""
    import sys

    sys.path.insert(0, str(ROOT / "Deimos"))
    from src.deimoslang.parser import Parser
    from src.deimoslang.tokenizer import Tokenizer

    for path in SCRIPTS:
        text = path.read_text(encoding="utf-8")
        program = Parser(Tokenizer().tokenize(text)).parse()
        assert program, f"{path.name} parsed to nothing"


_NOT_ANY = re.compile(r"^(\}\s*)?(el)?if\s+NOT\s+any\b.*\{\s*$")


def _dead_sameany(text):
    """[(line, stmt)] for `sameany` that provably acts on nobody.

    `sameany` resolves to `VM._any_player_client` (`vm.py:343`), which
    every `any` evaluation RESETS and then fills only with the clients
    that MATCHED (`vm.py:770-779` for `window_visible`). So inside the
    true branch of `if NOT any ...` -- taken precisely when nothing
    matched -- the list is empty and the action runs on no client at
    all.
    """
    lines = text.split("\n")
    found = []
    for i, ln in enumerate(lines):
        if not _NOT_ANY.match(ln.strip()):
            continue
        depth = 0
        for j in range(i, len(lines)):
            depth += lines[j].count("{") - lines[j].count("}")
            if depth <= 0 and j > i:
                break
        for k in range(i + 1, j):
            if lines[k].strip().startswith("sameany"):
                found.append((k + 1, lines[k].strip()))
    return found


def test_no_preset_acts_on_nobody():
    """The setup tour the operator asked about — "it randomly opens up
    the settings and clicks through but it doesnt seem to actually do
    anything beneficial" — did not, because it could not::

        times 300 {
            if NOT any windowvisible [... 'SettingPage', 'QuitButton'] {
                sameany sendkey ESC, .1
            } else { break }
            sleep .1
        }

    The branch is taken when NO client has the settings page open, so
    `_any_player_client` is empty and ESC goes to nobody; the loop then
    runs all 300 iterations and the `times 4` around it spends two
    minutes achieving nothing. The same shape made the training-menu
    recovery ("Trying different menu positions...", three attempts of
    blind clicks) entirely inert.

    `mass` is what the author meant: nothing matched, so act on all of
    them. `sameany` after a POSITIVE `any` is correct and is left
    alone -- Arc 1 still has hundreds of those.
    """
    offenders = {p.name: _dead_sameany(p.read_text(encoding="utf-8"))
                 for p in SCRIPTS}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, "\n".join(
        f"{name} line {ln}: {stmt}"
        for name, hits in offenders.items() for ln, stmt in hits[:5])


def test_the_correct_sameany_usages_were_not_touched():
    """A blanket replace would have broken every place `sameany` is
    right: acting on exactly the clients a positive `any` just
    matched."""
    arc1 = (ROOT / "deimos_bridge" / "data" / "scripts"
            / "TTS Arc 1.txt").read_text(encoding="utf-8")
    assert arc1.count("sameany") > 500, \
        "the positive-any usages went missing with the dead ones"


# ------------------------------------------- the friends-list row pattern
def test_the_local_friend_entry_pattern_still_matches_wizwalkers():
    """`party._ENTRY` is a copy of wizwalker's `_friend_list_entry`,
    kept locally because the import is the part that cannot happen off
    Windows -- `wizwalker.extensions.scripting.utils` pulls in the
    memory layer -- while reading a name out of text is not
    Windows-specific at all. That copy is what lets the party's
    name resolution be tested at all.

    A copy can drift, and drifting silently means matching nothing:
    the account settings would stay at their placeholder forever and
    the script would go on skipping every friend-teleport it has, which
    is the failure the whole thing exists to end. So the two patterns
    are compared character for character, and a Deimos bump that
    changes the game's markup fails here.
    """
    from deimos_bridge import party

    src = _source(UTILS)
    body = src.split("_friend_list_entry = regex.compile(", 1)[1]
    body = body.split("\n)", 1)[0]
    theirs = "".join(re.findall(r'r"([^"]*)"', body))
    assert theirs, "wizwalker's pattern could not be read at all"
    assert party._ENTRY == theirs, (
        "the local copy has drifted from wizwalker's:\n"
        f"  ours:   {party._ENTRY!r}\n"
        f"  theirs: {theirs!r}")


CLIENT = "Deimos/libs/wizwalker/wizwalker/client.py"


def test_a_stuck_should_update_flag_no_longer_latches_teleports_off():
    """`Client._teleport_object` waits for the game's hook to clear
    `should_update` and RAISED when it did not. The only code that
    clears the flag sits below that raise, in the
    `purge_on_after_unuser_fixer` handler — so once the flag stuck,
    every teleport on that client raised before reaching the one thing
    that would unstick it, for the rest of the run.

    Rev 35f0fc6e: 1,336 `Timed out waiting for coro should_update` over
    101 minutes, one every four and a half seconds, all on one wizard,
    while the party stood on its quest marker getting nowhere."""
    src = _source(CLIENT)
    body = src.split("async def _teleport_object", 1)[1]
    body = body.split("async def _get_je_instruction", 1)[0]
    pre = body.split("jes = await self._get_je_instruction", 1)[0]
    assert "except ExceptionalTimeout:" in pre, \
        "the pre-flight wait raises again — a stuck flag latches every " \
        "teleport on that client off for the rest of the run"
    assert "write_should_update(False)" in pre, \
        "it catches the timeout but never clears the flag it timed out on"




# ------------------------------------------------- archmastery school pips
MEMBER = "Deimos/libs/wizwalker/wizwalker/combat/member.py"


def _load_ww_member():
    """The vendored wizwalker combat member, importable on Linux.

    Its only module-level import is the `wizwalker` package itself,
    used at runtime for `MemoryInvalidated` and in annotation strings —
    so one stub module lets the real class load. Same throwaway-
    namespace discipline as `_load_ww_utils`.
    """
    import sys
    import types

    ww = types.ModuleType("wizwalker")
    ww.MemoryInvalidated = type("MemoryInvalidated", (Exception,), {})
    saved = sys.modules.get("wizwalker")
    try:
        sys.modules["wizwalker"] = ww
        namespace = {"__name__": "_ww_combat_member"}
        exec(compile(_source(MEMBER), MEMBER, "exec"), namespace)
    finally:
        if saved is None:
            sys.modules.pop("wizwalker", None)
        else:
            sys.modules["wizwalker"] = saved
    return types.SimpleNamespace(**namespace)


def _member_with_rack(counts):
    """A CombatMember whose participant's pip rack is `counts`.

    `counts` is {school: n} or None for a participant whose pip-count
    object is unreadable (older builds return None there).
    """
    import types

    mod = _load_ww_member()
    member = mod.CombatMember.__new__(mod.CombatMember)

    schools = ("balance", "death", "fire", "ice", "life", "myth", "storm")
    rack = None
    if counts is not None:
        async def _n(school):
            return counts.get(school, 0)
        rack = types.SimpleNamespace(
            **{f"{s}_pips": (lambda s=s: _n(s)) for s in schools})

    async def pip_count():
        return rack

    participant = types.SimpleNamespace(pip_count=pip_count)

    async def get_participant():
        return participant

    member.get_participant = get_participant
    return member


def test_the_member_reads_the_whole_archmastery_rack():
    """Oz, the two-account run at rev f2b8101f: a max-level storm
    wizard read 0 normal + 0 power pips for 22 consecutive rounds while
    the game showed his 7-pip Storm Lord castable, because his rack had
    converted to archmastery STORM pips — a third kind of pip stored in
    per-school counters this member never exposed. The policy priced
    him pip-starved, bought only zero-pip buffs, and passed the rest of
    the fight."""
    import asyncio

    member = _member_with_rack({"storm": 4, "balance": 1})
    rack = asyncio.run(member.school_pips())
    assert rack == {"balance": 1, "death": 0, "fire": 0, "ice": 0,
                    "life": 0, "myth": 0, "storm": 4}


def test_an_unreadable_pip_rack_reads_as_empty_not_as_an_error():
    """`pip_count()` is Optional in the vendored participant. A member
    on a build without the rack must degrade to the pre-archmastery
    arithmetic, not take the whole board read down with it."""
    import asyncio

    member = _member_with_rack(None)
    assert asyncio.run(member.school_pips()) == {}


# ------------------------------------------------- Deimos 3.14 collision tp
def test_collision_tp_answers_and_reports_like_navmap_tp():
    """The 3.14 port kept wizAi's contract: a teleport primitive that
    RETURNS whether it landed and reports through `on_teleport_result`.
    Upstream's collision_tp is bare-return like its navmap_tp was --
    success, failure and never-attempted all the same answer -- and a
    blocked client is silently dropped on the first line, the exact
    lost-instruction bug the navmap patch exists for. Here the blocked
    entry and the no-solution path both DELEGATE to navmap_tp, whose
    wait machinery and reporting answer for them."""
    src = _source(TP)
    assert "async def collision_tp" in src, "the 3.14 collision teleport is gone"
    head, body = src.split("async def collision_tp", 1)
    body = body.split("\ndef ", 1)[0]
    assert "-> bool" in body.split("\n", 1)[0], \
        "collision_tp no longer declares the landed/not answer"
    bare = [line for line in body.splitlines() if line.strip() == "return"]
    assert not bare, f"collision_tp has bare returns again: {bare}"
    assert body.count("return await navmap_tp(") == 2, \
        "blocked entry and no-solution fallback must both delegate to navmap_tp"
    assert body.count("_tp_result(") >= 3, \
        "not every collision landing reports a result"
    assert body.count("await _landed(") == 3, \
        "the three landing paths must all funnel through the landing check"


def test_the_collision_solver_stays_optional():
    """collision_tp imports the solver stack (shapely, numpy, the
    collision modules) lazily, inside the function, inside a try -- so
    a box without shapely still imports teleport_math and every
    collision teleport degrades to navmap_tp instead of the module
    failing to import at all. A top-level import here would take down
    scripts.py's `from src.teleport_math import navmap_tp` and with it
    every teleport in the program."""
    import ast

    tree = ast.parse(_source(TP))
    banned = {"shapely", "numpy", "src.collision", "src.collision_math",
              "src.entity_collision"}
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.add(node.module or "")
    hits = {t for t in top if t in banned or t.split(".")[0] in banned}
    assert not hits, f"solver imports moved to module top level: {hits}"


VM_TELEPORTS = "Deimos/src/deimoslang/vm.py"


def test_the_script_vms_teleports_solve_collision_first():
    """Deimos 3.14's own change to the script VM: the `tp quest` /
    `tp entity` instructions go through collision_tp (which falls back
    to navmap_tp by itself). The arc scripts' most-used instruction,
    and the one where a wall landing costs the most."""
    src = _source(VM_TELEPORTS)
    assert "from src.teleport_math import navmap_tp, collision_tp" in src, \
        "vm.py no longer imports the collision teleport"
    assert src.count("await collision_tp(client, pos)") == 3, \
        "the entity/vague/quest teleports are not all on collision_tp"


def test_the_314_collision_stack_is_vendored():
    """The three modules collision_tp solves with, present and parsing.
    Parse-only on purpose: collision_math imports shapely/numpy at its
    top, which this container does not carry -- exactly the situation
    the lazy import above exists for. The precise static-entity layer
    (teleporter pads, boats) additionally wants katsuba + wiztype +
    kinif, and entity_collision probes for those at import and quietly
    degrades to no colliders without them."""
    for rel, needles in (
        ("Deimos/src/collision.py",
         ("def get_collision_data", "class CollisionWorld")),
        ("Deimos/src/collision_math.py",
         ("def find_walkable_teleport_point", "def get_walk_grid",
          "def blocking_volumes_at")),
        ("Deimos/src/entity_collision.py",
         ("def build_zone_static_shapes", "def get_object_collider_radius",
          "_PRECISE_AVAILABLE = False")),
    ):
        src = _source(rel)
        compile(src, rel, "exec")
        for needle in needles:
            assert needle in src, f"{rel} lost {needle!r}"
    for rel in ("Deimos/src/questing.py", "Deimos/src/sigil.py"):
        assert "collision_tp(" in _source(rel), \
            f"{rel} did not take the 3.14 navmap->collision swaps"


def test_the_collision_walk_floor_is_aggro_range_not_interaction_range():
    """Upstream's collision teleport skips the final walk inside 120
    units — "already in range", meaning INTERACTION range. A defeat
    quest needs the wizard inside the mobs' AGGRO radius, and the
    collision landing is by construction the nearest CLEAR point,
    systematically outside the pack: rev dbced750 looped "Defeat
    Jacques the Scratcher" teleports for four minutes, every landing a
    reported success ~a hundred units short, no mob ever pulled. The
    floor is wizAi's own 20 — inside both ranges — and an upstream
    re-port that restores 120 restores the stall."""
    src = _source(TP)
    assert "_WALK_GAP_MIN = 20.0" in src, \
        "the walk floor went back above aggro range"
    assert "_WALK_GAP_MIN = 120.0" not in src


def test_a_marker_inside_the_bouncing_volume_is_a_door_not_a_statue():
    """The avoid-truncation exists for a statue the target stands
    AGAINST: stop at the edge, do not grind into the face. But the game
    aims the quest arrow INTO zone doors and service lifts, whose
    frames carry ordinary collision that rejects a teleport landing
    while welcoming a wizard on foot — walking in is how the transition
    fires. Rev 66a4fe5b: 'collision (strict re-solve)' landed both
    wizards 103 units from a marker pointing into the Scotland Yard
    service lift, the truncation parked them at its edge, and the
    script teleport-looped against a door it only needed to walk
    through. The discriminator is the target itself: inside the
    bouncing volume = destination, walk in; outside = statue, stop."""
    src = _source(TP)
    body = src.split("async def _walk_remaining_to_target", 1)[1]
    body = body.split("\nasync def ", 1)[0]
    assert "treating it " in body and "as a door and walking in" in body, \
        "the door branch is gone — lifts and zone doors truncate again"
    door_gate = body.index("as a door and walking in")
    truncation = body.index("stop at the edge of the solid")
    assert door_gate < truncation, \
        "the door check must come before the statue truncation"
    # The truncation must now be CONDITIONAL on the target being outside
    # the volume — an unconditional truncation is the rev 66a4fe5b bug.
    gate = body[:truncation]
    assert gate.count("fp.contains(_Point(target_xyz.x, target_xyz.y))") == 1


def test_a_landing_far_from_the_target_is_finished_on_foot_and_named():
    """The teleport's contract to the script is not "a teleport
    happened" — it is "the wizard now stands at the objective", because
    the script's next instruction assumes it. Rev e786b716: the preset
    teleported to a dungeon sigil, immediately checked the press-X
    prompt to classify the interactable, read nothing (the heartbeat
    put the wizard 270 units out), declared "not a dungeon", and its
    next loop iteration teleported everyone away to the safe area —
    three cycles of ~13s before one stuck. The operator: "it shouldnt
    take a billion attempts to enter a dungeon". So every landing path
    measures the wizard's ACTUAL distance to the target afterwards,
    closes an open gap on foot once (unless the walk deliberately
    stopped short — a statue truncation is a decision), and reports a
    landing still out as its own outcome kind, 'collision-short', so
    the export shows the first one instead of thinning it under the
    successes."""
    src = _source(TP)
    assert "_LANDED_NEAR = 60.0" in src, \
        "the landed-near threshold moved — the sigil press-X check needs it"
    assert "_LANDED_WALK_CAP = 800.0" in src, \
        "the cap that stops a blind cross-zone hike is gone"
    body = src.split("async def collision_tp", 1)[1].split("\ndef ", 1)[0]
    assert "collision-short (" in body, \
        "short landings are no longer their own reported kind"
    assert "await client.goto(target_xyz.x, target_xyz.y)" in body, \
        "the final on-foot approach is gone from the landing check"
    assert "not walked_short" in body, \
        "the final approach must respect a walk that chose to stop short"
    assert "_LANDED_NEAR < gap <= _LANDED_WALK_CAP" in body, \
        "the final approach lost its distance gates"


def test_the_walk_owns_its_stop_decisions():
    """`_walk_remaining_to_target` answers whether it DELIBERATELY
    stopped short — truncated at a statue's footprint, target on a
    static collider, no on-foot path, or an error — so the landing
    check knows when a still-open gap may be closed with a straight
    approach and when the stop was a decision to honour."""
    src = _source(TP)
    head, walk = src.split("async def _walk_remaining_to_target", 1)
    assert "-> bool" in walk.split('"""', 1)[0], \
        "the walk no longer declares its stopped-short answer"
    walk = walk.split("\nasync def ", 1)[0]
    assert walk.count("return True") >= 3, \
        "the deliberate stops (collider/no-path/error) must answer True"
    assert "return False" in walk, \
        "a walk that finished (or had nothing to do) must answer False"
    assert "return not refine" in walk, \
        "a truncated walk must answer stopped-short through `refine`"


# ----------------------------------------------------------- walk-only mode
# wizAi's walk-only questing: `Deimos/src/walk_nav.py` holds the mode
# switch and the on-foot navigator (driven tests in `test_walk_nav.py`),
# and the guards below pin the ROUTING -- the places in the vendored
# tree that now ask `walk_nav.walking()` before writing position memory.
# A Deimos bump that drops any of them silently turns the "Walk to
# objectives" checkbox back into a teleport.

WALK_NAV = "Deimos/src/walk_nav.py"
QUESTING = "Deimos/src/questing.py"
WIZ_NAVIGATOR = ("Deimos/libs/wizsprinter/wizwalker/extensions/"
                 "wizsprinter/wiz_navigator.py")


def test_walk_mode_walks_before_it_teleports():
    """Both teleport primitives try the hop on foot first when the mode
    is on: `collision_tp` before its solve, `navmap_tp` before its
    direct teleport -- and the exhausted-fallback delegation passes
    `_walk_first_try=not walked` so one hop never attempts the same
    failed walk twice."""
    src = _source(TP)
    assert "from src import walk_nav" in src, \
        "teleport_math no longer knows about the walk mode"

    nav = src.split("async def navmap_tp", 1)[1].split("\nasync def ", 1)[0]
    walk_gate = nav.find("_walk_first_try and walk_nav.walking()")
    direct_tp = nav.find("await client.teleport(target_xyz)")
    assert 0 <= walk_gate < direct_tp, \
        "navmap_tp teleports before it considers walking"

    body = src.split("async def collision_tp", 1)[1].split("\ndef ", 1)[0]
    assert "if walk_nav.walking():" in body, \
        "collision_tp lost its walk-first gate"
    assert body.find("walk_nav.walking()") < body.find("safe_xyz = None"), \
        "collision_tp solves for a teleport before considering the walk"
    assert "_walk_first_try=not walked" in body, \
        "a failed walk would be attempted twice in one hop"


def test_a_walk_fallback_is_spoken_not_silent():
    """When walking cannot deliver, the teleport that covers for it is
    announced through `on_teleport_note` ('walk fallback: ...'), and a
    walk that lands reports through `on_teleport_result` with its own
    outcome kinds -- except an off-mesh target, the script-bump case,
    which stays quiet on purpose."""
    src = _source(TP)
    assert 'f"walk fallback: {outcome} ({detail})' in src, \
        "the fallback teleport went silent"
    assert '_tp_result(True, "walk", zone_name, client)' in src, \
        "a walked arrival no longer reports a landing"
    assert 'f"walk-partial ({detail})"' in src, \
        "partial walks lost their own outcome kind"
    assert '"walk (interrupted)"' in src, \
        "an interrupted walk lost its own outcome kind"
    assert "DETAIL_OFF_MESH" in src, \
        "the off-mesh quiet path (script bumps) is gone"


def test_the_vms_position_tp_is_mode_aware_and_the_bump_still_bumps():
    """`tp XYZ` routes through `nav_teleport`, which walks in walk mode
    but teleports RAW for an off-mesh target -- the arc scripts' 116
    `tp XYZ(0, 100000, 0)`-style bumps rely on the game clamping an
    impossible position. `+tp`/`-tp` stay raw teleports: they are
    micro-adjust idioms, not navigation."""
    src = _source(VM)
    assert "tg.create_task(nav_teleport(client, pos))" in src, \
        "the VM's position teleport is no longer mode-aware"
    assert src.count("tg.create_task(client.teleport(newpos))") == 2, \
        "+tp/-tp must stay raw teleports"
    # The 3.14 routing guard, re-asserted here so a refactor that breaks
    # it names this feature too: the walk branches deliberately spell
    # their calls differently (`xyz=pos`).
    assert src.count("await collision_tp(client, pos)") == 3
    assert src.count("await collision_tp(client, xyz=pos)") == 2, \
        "the entitytp walk branches are gone"
    assert '"direct (walk-only bump)"' in _source(TP), \
        "nav_teleport lost the off-mesh bump short-circuit"


def test_the_questers_defeat_probe_walks_in_walk_mode():
    """The defeat-quest stagger opens with a RAW probe teleport whose
    whole purpose is detecting a collided landing; a walk makes that
    moot, so in walk mode the probe becomes a walk-first collision_tp.
    The wait-for-combat loop's mob approach walks too."""
    src = _source(QUESTING)
    assert "from src import walk_nav" in src
    stagger = src.split("staggering teleports", 1)[1]
    probe = stagger.split("detected_interact_from_popup", 1)[0]
    assert "if walk_nav.walking():" in probe, \
        "the defeat probe teleports raw even in walk mode"
    combat_wait = stagger.split("entity_detect_combat_status:", 2)[1]
    assert "find_closest_mob()" in combat_wait, \
        "the mob approach lost its walk branch"


def test_collect_scans_stay_teleports_under_walk_mode():
    """The chunk sweep parks the wizard 550u UNDER the map to stream
    entities -- off any walkable mesh -- and the grab/return hops start
    from there. The whole sequence runs under `walk_nav.suspended()`;
    the hardcoded-collect coordinates, which are ordinary surface
    points, walk instead (via `_nav_hop`)."""
    src = _source(QUESTING)
    wrapper = src.split("async def auto_collect_rewrite", 1)[1]
    wrapper = wrapper.split("async def _auto_collect_rewrite_impl", 1)[0]
    assert "with walk_nav.suspended():" in wrapper, \
        "the underground collect scan would try to walk"
    assert "_auto_collect_rewrite_impl" in wrapper
    hardcoded = src.split("async def hardcoded_collect", 1)[1]
    hardcoded = hardcoded.split("\n    async def ", 1)[0]
    assert "async def _nav_hop" in hardcoded, \
        "hardcoded collect coordinates lost their walk branch"
    assert hardcoded.count("await _nav_hop(") == 4, \
        "a hardcoded-collect hop bypasses the mode"


def test_the_zone_gates_walk_up_before_pressing_x():
    """Every gate coordinate in `gateTypeDifferentiation` used to be a
    raw `p.teleport(XYZ(...))`. They all go through `_nav_move` now:
    the same teleport with the mode off, a walk to the gate (then the
    unchanged X-press / zone-wait logic) with it on."""
    src = _source(WIZ_NAVIGATOR)
    assert "async def _nav_move" in src, \
        "wiz_navigator lost its mode-aware move"
    gate = src.split("async def gateTypeDifferentiation", 1)[1]
    gate = gate.split("async def interactiveTeleportToZone", 1)[0]
    raw = [line for line in gate.splitlines()
           if line.lstrip().startswith("await p.teleport(")]
    assert raw == [], f"gate teleports bypass the mode again: {raw}"
    assert gate.count("await _nav_move(p, XYZ(") >= 25, \
        "the gate table shrank suspiciously -- check the _nav_move sweep"
    # Proximity gates fire the transition DURING the walk, and
    # `wait_for_zone_change()` anchors on whatever zone it reads at
    # entry -- so the waits that directly follow a move must anchor on
    # the zone read BEFORE it, or they hang on a change that already
    # happened.
    assert gate.count("await p.wait_for_zone_change(zone_before)") == 4, \
        "a proximity gate's wait lost its pre-move zone anchor"
    assert "zone_before = await _nav_move(p, XYZ(" in gate


def test_walk_nav_stays_importable_without_the_game():
    """The navigator's top level must stay stdlib-only -- that is what
    lets `test_walk_nav.py` DRIVE the walk loop on a machine with no
    game, and what keeps importing it safe from anywhere in the tree
    (teleport_math, the VM, the quester all pull it in at module top)."""
    import ast
    import importlib.util

    tree = ast.parse(_source(WALK_NAV))
    banned = {"wizwalker", "loguru", "shapely", "numpy", "src"}
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.add(node.module or "")
    hits = {t for t in top if t.split(".")[0] in banned}
    assert not hits, f"walk_nav grew game imports at top level: {hits}"

    spec = importlib.util.spec_from_file_location(
        "_walk_nav_guard", ROOT / WALK_NAV)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)    # the proof: it loads right here
    assert mod.nav_mode() == mod.NAV_TELEPORT, \
        "the mode must default to the behaviour that shipped"


def test_the_wisp_hops_walk_in_walk_mode():
    """Deimos's wisp collection teleports wisp to wisp; in walk mode
    both collectors go on foot through `_wisp_hop`, bounded so a walk
    that cannot finish skips the wisp instead of stalling the chore."""
    src = _source("Deimos/src/utils.py")
    assert "async def _wisp_hop" in src, \
        "the wisp collectors lost their mode-aware hop"
    assert src.count("await _wisp_hop(client, wisp_xyz)") == 2, \
        "a wisp collector bypasses the mode"
    hop = src.split("async def _wisp_hop", 1)[1].split("\nasync def ", 1)[0]
    assert "walk_nav.walking()" in hop and "asyncio.wait_for" in hop, \
        "the wisp walk lost its mode gate or its bound"
