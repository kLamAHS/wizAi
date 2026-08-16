import asyncio
import regex
from contextlib import suppress

from wizwalker.memory.memory_objects.enums import WindowFlags
from wizwalker.utils import maybe_wait_for_value_with_timeout


_friend_list_entry = regex.compile(
    r"<Y;\d+><X;\d+><indent;0><Color;[\w\d]+><left>"
    r"<icon;FriendsList/Friend_Icon_List_0(?P<icon_list>[12])\."
    r"dds;\d+;\d+;(?P<icon_index>\d+)></left><Y;(?P<name_y>[-\d]+)><X;(?P<name_x>[-\d]+)>"
    r"<indent;\d+><Color;[\d\w]+>(<left>)?<COLOR;[\w\d]+>(?P<name>[\w ]+)"
)


async def paint_window_for(window, time: float = 2):
    """
    Paint a window for a number of seconds

    Args:
        window: The window to paint
        time: How long to paint the window
    """

    async def _paint_task():
        with suppress(asyncio.CancelledError):
            while True:
                await window.debug_paint()

    paint_task = asyncio.create_task(_paint_task())
    await asyncio.sleep(time)
    paint_task.cancel()


#: how many times the friend's row is clicked before the teleport gives
#: up. One was the upstream count, and one is a bet that the click
#: landed -- which it does not always: the online friends list RE-SORTS
#: as people change zone, and the row read a moment ago can belong to
#: somebody else (or to nobody) by the time the mouse gets there. A
#: click that opened nothing used to end the whole call with "No child
#: window named wndCharacter", and wizAi's run at rev cfeb9a85 lost ten
#: rejoins a run that way.
FRIEND_CLICK_ATTEMPTS = 3
#: how long a page turn is given to show up in the PageNumber label.
#: Upstream clicked the page button and read the list on the next line,
#: with nothing in between.
PAGE_TURN_TIMEOUT = 3.0
#: how long the character panel is given to name the friend it opened
#: on. Only used to CHECK a name, never to find one.
NAME_PROOF_WAIT = 2.0


async def _maybe_get_named_window(parent, name: str, retries: int = 4):
    async def _try_do(callback, *args, **kwargs):
        while True:
            res = await callback(*args, **kwargs)

            if not res:
                nonlocal retries
                if retries <= 0:
                    return res

                retries -= 1
                await asyncio.sleep(0.4)

            else:
                return res

    possible = await _try_do(parent.get_windows_with_name, name)

    if not possible:
        raise ValueError(f"No child window named {name}")

    if len(possible) > 1:
        raise ValueError(f"Multiple results for {name}")

    return possible[0]


async def _cycle_to_online_friends(client, friends_list):
    list_label = await _maybe_get_named_window(friends_list, "lblFriendsList")

    async def _get_text():
        current_text = await list_label.maybe_text()

        if current_text is None:
            raise Exception("Friend's list has no label")

        return current_text.replace("<center>", "").replace("</center>", "")

    right_button = await _maybe_get_named_window(friends_list, "btnListTypeRight")

    # wizAi patch: bounded. Upstream's `while ... != "Online Friends"`
    # has no exit, so a list whose label never reads exactly that --
    # a click that does not land, a label that reads empty, a different
    # tab layout -- parks the caller here for the rest of the run and
    # every failure surfaces as an opaque outer timeout. The tab
    # carousel is a handful of entries; twelve clicks is two full laps,
    # and a label that has not read "Online Friends" by then never
    # will. Raising names the actual failure, and the caller's retry
    # loops treat it like any other failed attempt.
    for _ in range(12):
        current_page = await _get_text()
        if current_page == "Online Friends":
            break
        await client.mouse_handler.click_window(right_button)
        await maybe_wait_for_value_with_timeout(
            _get_text, value=current_page, inverse_value=True, timeout=5
        )
    else:
        raise ValueError(
            f"the friends list never showed 'Online Friends' after 12 "
            f"page clicks — it reads {await _get_text()!r}"
        )


def _read_page_number(text):
    """(current, total) off the PageNumber label, or None.

    Shared by the caller and by `_turn_to_page`, which has to read it
    again after every click to know whether the click did anything.
    """
    if not text:
        return None
    cleaned = text.replace("<center>", "").replace("</center>", "").replace(" ", "")
    try:
        current, total = map(int, cleaned.split("/"))
    except (ValueError, TypeError):
        return None
    return current, total


async def _turn_to_page(client, right_button, page_number, target):
    """Click the page button until PageNumber reads `target`. True if it does.

    Upstream did this:

        if target_page != current_page:
            for _ in range(target_page - current_page):
                await client.mouse_handler.click_window(right_button)

    which has two faults and they compound. `range()` of a negative
    number is empty, so turning BACKWARDS -- the friend is on page 1 and
    the window opened on page 2 -- clicked nothing and left the wrong
    page showing. And nothing waited for the turn, so even forwards the
    row was clicked before the list redrew.

    Neither shows up as an error. Both end with `_click_on_friend`
    clicking whatever row happens to sit at that offset, which in a
    party of three -- every wizard on every other's list -- is a
    teleport to the wrong wizard, reported as a success.

    There is only a next button in this UI (no `btnArrowUp` exists in
    any layout Deimos knows), so going back means going forward until it
    wraps. Verified after every click rather than counted: a click that
    did not land, and a list that does not wrap, both stop the loop
    instead of spending the budget.
    """
    seen = _read_page_number(await page_number.maybe_text())
    if seen is None:
        return False
    current, total = seen
    if current == target:
        return True
    # At most one lap. More than that is the page not responding, and
    # clicking a button that does nothing is how the friends list ends
    # up in a state nothing else expects.
    for _click in range(max(total, 1) + 1):
        await client.mouse_handler.click_window(right_button)
        waited = 0.0
        while waited < PAGE_TURN_TIMEOUT:
            await asyncio.sleep(0.2)
            waited += 0.2
            now = _read_page_number(await page_number.maybe_text())
            if now is not None and now[0] != current:
                current = now[0]
                break
        else:
            # The click changed nothing within the timeout. Another one
            # will not either.
            return False
        if current == target:
            return True
    return False


async def _subtree_text(window, depth: int = 5, limit: int = 30) -> str:
    """Every readable text under a window, joined.

    Used to check WHO the character panel opened on. The panel keeps
    the name in a different leaf depending on whether the wizard is a
    friend, a stranger or in your group, so the check reads all of it
    rather than betting on one path -- the same reason the quest-book
    matcher does.
    """
    texts = []

    async def walk(win, left):
        if left < 0 or len(texts) >= limit:
            return
        try:
            text = await win.maybe_text()
        except Exception:
            text = ""
        if text:
            texts.append(text)
        try:
            kids = await win.children()
        except Exception:
            return
        for kid in kids:
            await walk(kid, left - 1)

    await walk(window, depth)
    return " ".join(texts)


def _name_tokens(text):
    """The lowercase words of a name, for a match that survives markup."""
    return set(regex.findall(r"[\w']+", (text or "").lower()))


async def _panel_is_for(character_window, name):
    """Does the open character panel name this friend? None = cannot tell.

    Three answers, not two. True and False decide whether to press "Go
    to Friend"; None means the panel's text would not read, and
    REFUSING on that would break the teleport on every client whose
    panel keeps the name somewhere this cannot walk to. Unreadable is
    not evidence of the wrong wizard, so it is allowed through -- the
    check exists to catch a click that landed on the wrong ROW, which
    reads a name perfectly well, just somebody else's.
    """
    if not name:
        return None
    want = _name_tokens(name)
    if not want:
        return None
    waited = 0.0
    while True:
        have = _name_tokens(await _subtree_text(character_window))
        if not have:
            if waited >= NAME_PROOF_WAIT:
                return None
            await asyncio.sleep(0.2)
            waited += 0.2
            continue
        # The list shows "First Lastname" and the panel may show either
        # part alone, so sharing a word is the match. Sharing none is a
        # different wizard.
        return bool(want & have)


async def _close_friend_windows(client, friends_window=None):
    """Best effort: put the character popup and the friends list away.

    Upstream closed the friends window on the LAST line of the happy
    path only, so every failure -- and there are five ways to fail
    before it -- left the list up, and `_teleport_to_friend` left the
    character panel up on top of it. Both block movement, and the next
    attempt starts by clicking through them.

    That is what turns one bad teleport into ten. Rev cfeb9a85 has this
    exact shape: a rejoin dies on `wndCharacter`, and the follow-ups
    behind it fail on a wizard that has been standing behind its own
    friends list since.

    Never raises. This runs in a `finally`, where an exception would
    replace the real reason the teleport failed with a tidying-up one.
    """
    with suppress(Exception):
        character_window = await _maybe_get_named_window(
            client.root_window, "wndCharacter", retries=0
        )
        close_button = await _maybe_get_named_window(
            character_window, "btnCharacterClose", retries=0
        )
        await client.mouse_handler.click_window(close_button)
        await asyncio.sleep(0.2)
    with suppress(Exception):
        if friends_window is None:
            friends_window = await _maybe_get_named_window(
                client.root_window, "NewFriendsListWindow", retries=0
            )
        await friends_window.write_flags(WindowFlags.disabled)


async def _cycle_friends_list(
    client, right_button, friends_list, icon, icon_list, name, current_page,
    page_number=None,
):
    """(match, index). Turns the list to the friend's page on the way.

    `page_number` is the PageNumber window, and passing it is what
    makes the page turn verifiable. Without it the old blind forward
    clicking is all that is possible, so a caller that has the window
    should hand it over -- see `_turn_to_page` for what the blind
    version costs.
    """
    list_text = await friends_list.maybe_text()

    match = None
    idx = 0

    for idx, friend_entry in enumerate(list(_friend_list_entry.finditer(list_text))):
        friend_icon = int(friend_entry.group("icon_index"))
        friend_icon_list = int(friend_entry.group("icon_list"))
        friend_name = friend_entry.group("name")

        if icon is not None and icon_list is not None and name:
            if (
                friend_icon == icon
                and friend_icon_list == icon_list
                and friend_name == name
            ):
                match = friend_entry
                break

        elif icon is not None and icon_list is not None:
            if friend_icon == icon and friend_icon_list == icon_list:
                match = friend_entry
                break

        elif name:
            if friend_name == name:
                match = friend_entry
                break

        else:
            raise RuntimeError("Invalid args")

    if match:
        target_page = (idx // 10) + 1
        if target_page != current_page:
            if page_number is None:
                for _ in range(target_page - current_page):
                    await client.mouse_handler.click_window(right_button)
            # Verified, and in either direction -- see `_turn_to_page`.
            # A turn that does not happen is RAISED rather than passed
            # on as "not found": the caller's next move is a click at a
            # row offset that only means anything on the right page, and
            # "could not find them" would send the caller off trying
            # other spellings of a name it just matched.
            elif not await _turn_to_page(
                client, right_button, page_number, target_page
            ):
                raise ValueError(
                    f"found the friend on page {target_page} of the "
                    f"friends list, but the list would not turn to it"
                )

    return match, idx


async def _click_on_friend(client, friends_list, friend_index):
    scaled_rect = await friends_list.scale_to_client()
    ui_scale = await client.render_context.ui_scale()

    # 12 % 10 = 2 * 30 = 60 * ui_scale
    scaled_friend_name_y = ((friend_index % 10) * 30) * ui_scale

    scaled_rect_center = scaled_rect.center()
    click_x = scaled_rect_center[0]
    # 15 is half the size of each entry's click area
    click_y = int(scaled_rect.y1 + scaled_friend_name_y + (15 * ui_scale))

    await client.mouse_handler.click(click_x, click_y)
    await asyncio.sleep(1)


async def _teleport_to_friend(client, character_window):
    teleport_button = await _maybe_get_named_window(character_window, "btnGoToFriend")
    await client.mouse_handler.click_window(teleport_button)
    # wait for confirmation window
    await asyncio.sleep(1)

    # Polled for ~8s rather than the default ~1.6s. The confirmation box
    # is not always the plain "teleport to your friend?" one: a friend
    # standing inside an instance raises the dungeon-reset warning
    # instead, which the game takes noticeably longer to put up. wizAi's
    # run at rev 228d4f50 lost three of five rejoins to
    # "No child window named MessageBoxModalWindow", every one of them
    # with the other wizard in WC_Firecat_T1 -- an interior.
    #
    # ...and the panel is closed whether or not it comes up. Upstream
    # raised from here with `wndCharacter` still on screen, on top of a
    # friends list its caller also never closed -- so a wizard that
    # lost one teleport stood behind two modals it could not walk out
    # of, and every attempt after it started by clicking through them.
    # That is the difference between one failed rejoin and the ten in
    # rev cfeb9a85.
    try:
        confirmation_window = await _maybe_get_named_window(
            client.root_window, "MessageBoxModalWindow", retries=20
        )
        yes_button = await _maybe_get_named_window(confirmation_window, "centerButton")
    except Exception:
        with suppress(Exception):
            stale = await _maybe_get_named_window(
                character_window, "btnCharacterClose", retries=0
            )
            await client.mouse_handler.click_window(stale)
        raise

    close_button = await _maybe_get_named_window(character_window, "btnCharacterClose")

    await client.mouse_handler.click_window(yes_button)
    # we need to click the close button within the 1 second of the teleport animation
    await client.mouse_handler.click_window(close_button)


# TODO: add error if friend is busy message pops up
async def teleport_to_friend_from_list(
    client, *, icon_list: int = None, icon_index: int = None, name: str = None
):
    """
    Teleport to a friend from the client's friend list

    Args:
        client: Client to teleport
        icon_list: Icon list the icon is from (1 or 2) or None
        icon_index: Index of the icon or None
        name: Name of the player or None
    """
    if (
        icon_list is None
        and icon_index is not None
        or icon_list is not None
        and icon_index is None
    ):
        raise ValueError("Icon list and icon index must both be defined or not defined")

    if all(i is None for i in (icon_list, icon_index, name)):
        raise ValueError("Must specify icon_list and icon_index or name or all")

    try:
        friends_window = await _maybe_get_named_window(
            client.root_window, "NewFriendsListWindow"
        )
    except ValueError:
        # friend's list isn't open so open it
        friend_button = await _maybe_get_named_window(client.root_window, "btnFriends")
        await client.mouse_handler.click_window(friend_button)

        friends_window = await _maybe_get_named_window(
            client.root_window, "NewFriendsListWindow"
        )
    else:
        if not await friends_window.is_visible():
            # friend's list isn't open so open it
            friend_button = await _maybe_get_named_window(client.root_window, "btnFriends")
            await client.mouse_handler.click_window(friend_button)

    # Everything from here on can fail, and every one of those failures
    # used to leave this window up -- the close was the last line of the
    # happy path. See `_close_friend_windows`.
    try:
        await _cycle_to_online_friends(client, friends_window)

        friends_list_window = await _maybe_get_named_window(friends_window, "listFriends")
        friends_list_text = await friends_list_window.maybe_text()

        # no friends online
        if not friends_list_text:
            raise ValueError("No friends online")

        right_button = await _maybe_get_named_window(friends_window, "btnArrowDown")
        page_number = await _maybe_get_named_window(friends_window, "PageNumber")

        # The row is clicked more than once if it has to be. The online
        # list RE-SORTS as people change zone, so the row that matched
        # the text read a moment ago can hold somebody else -- or
        # nothing -- by the time the mouse arrives, and a click that
        # opened no panel used to end the whole call. Each attempt
        # re-reads the list rather than re-clicking the stale offset,
        # because the point is that the offset went stale.
        last_error = None
        for attempt in range(FRIEND_CLICK_ATTEMPTS):
            if attempt:
                # Whatever the last click DID open is in the way.
                with suppress(Exception):
                    stale = await _maybe_get_named_window(
                        client.root_window, "wndCharacter", retries=0
                    )
                    close_button = await _maybe_get_named_window(
                        stale, "btnCharacterClose", retries=0
                    )
                    await client.mouse_handler.click_window(close_button)
                await asyncio.sleep(0.5)
                friends_list_text = await friends_list_window.maybe_text()
                if not friends_list_text:
                    raise ValueError("No friends online")

            seen = _read_page_number(await page_number.maybe_text())
            if seen is None:
                raise ValueError("The friends list page number would not read")
            current_page = seen[0]

            friend, friend_index = await _cycle_friends_list(
                client,
                right_button,
                friends_list_window,
                icon_index,
                icon_list,
                name,
                current_page,
                page_number,
            )

            if friend is None:
                raise ValueError(
                    f"Could not find friend with icon {icon_index} icon list {icon_list} and/or name {name}"
                )

            await _click_on_friend(client, friends_list_window, friend_index)

            # Same eight-second budget, and for the same reason as the
            # confirmation box: a click, a flat one-second sleep, then a
            # lookup that gives up after 1.6s more. wizAi's run at rev
            # 228d4f50 lost a rejoin to "No child window named
            # wndCharacter" while three game clients shared one machine.
            # It costs nothing when the window is already there --
            # `_maybe_get_named_window` returns on its first look.
            #
            # Full budget on the FIRST look only. wizAi bounds this whole
            # call at 45s (`party.TELEPORT_TIMEOUT`), and three
            # eight-second waits plus a confirmation would spend it
            # before the last attempt got its answer -- turning a clear
            # "the panel never opened" into a timeout that says nothing.
            # A second click does not need the first click's grace: by
            # then the window has had eight seconds to exist.
            try:
                character_window = await _maybe_get_named_window(
                    client.root_window, "wndCharacter",
                    retries=20 if attempt == 0 else 8,
                )
            except ValueError as exc:
                # The click landed on nothing. Read the list again and
                # try the row it says now.
                last_error = exc
                continue

            # ...and check it opened on the RIGHT wizard before pressing
            # "Go to Friend". A row offset that has gone stale still
            # opens a panel; it is somebody else's, and in a party of
            # three -- everyone on everyone's list -- teleporting to the
            # wrong one is a party split that reports as a success.
            # `None` means the panel's text would not read, which is not
            # evidence of the wrong wizard and is allowed through.
            if await _panel_is_for(character_window, name) is False:
                last_error = ValueError(
                    f"the friends list opened somebody other than {name} "
                    f"— the list re-sorted between reading it and "
                    f"clicking"
                )
                continue

            await _teleport_to_friend(client, character_window)
            return
        raise last_error or ValueError(
            f"could not open the character panel for {name} after "
            f"{FRIEND_CLICK_ATTEMPTS} attempts"
        )
    finally:
        await _close_friend_windows(client, friends_window)
