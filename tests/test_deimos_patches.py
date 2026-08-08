"""The patches wizAi applies to the vendored Deimos and wizwalker.

Vendored code gets updated, and an update that quietly drops one of
these would put back a bug the run has already been broken by -- with no
test failing, because everything wizAi owns still works. So each patch
has a guard here that fails loudly if the thing it fixed comes back.

Each test names the upstream shape it replaced, so re-applying the patch
after a Deimos bump is a matter of reading the failure.
"""
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
    `wndCharacter`."""
    src = _source(UTILS)
    assert src.count("retries=20") >= 2, \
        "the widened waits in teleport_to_friend_from_list are gone"
    assert '"wndCharacter", retries=20' in src.replace("\n", " ").replace(
        "        ", " ").replace("   ", " ").replace("  ", " ") or \
        "retries=20" in src.split("wndCharacter", 1)[1][:200], \
        "the character-window wait is back to the default"


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
    assert "in_battle()" in block and "return False" in block
    i = block.index("in_battle()")
    assert "return False" in block[i:i + 120], \
        "being in a duel must end the wait, not extend it"


def test_the_wait_is_bounded():
    src = _source(TP)
    ns = {}
    exec("TP_FREE_WAIT" + src.split("TP_FREE_WAIT", 1)[1].split("\n\n\n", 1)[0],
         ns)
    assert 0 < ns["TP_FREE_WAIT"] <= 30, \
        "long enough for a loading screen, short enough not to be a hang"
