"""Work out *why* wizwalker's hook activation failed.

`PatternFailed` on the autobot pattern has two completely different
causes with the same message, and the message's advice ("restart the
client") only fixes one of them:

  A. **Stale hook state in the running process.** `_prepare_autobot`
     finds the autobot function by signature and then overwrites 3900
     bytes of it with zeros, using it as scratch space for hook shellcode
     (`memory/handler.py:80-93`). `_rewrite_autobot` puts the original
     bytes back on a clean shutdown. If a previous attach died without
     that -- a crash, a Ctrl-C at the wrong moment, a killed terminal --
     the region is still zeroed, and the next scan finds nothing.
     Genuinely restarting the client fixes this.

  B. **The signature is stale.** The pattern is a fixed byte sequence
     matched against `WizardGraphicalClient.exe`. When the game patches,
     that code can move or change, and no amount of restarting helps --
     wizwalker itself needs updating.

The two are distinguishable, and this tells them apart: scan the *file on
disk* for the same pattern. The on-disk copy is untouched by any hook, so

    found on disk  ->  the binary still contains what wizwalker expects,
                       so the failure is state in the running process (A)
    not on disk    ->  this build of the game no longer matches the
                       signature wizwalker ships (B)

    python -m deimos_bridge.diagnose_hooks
    python -m deimos_bridge.diagnose_hooks --exe "C:\\path\\to\\WizardGraphicalClient.exe"
"""
import argparse
import glob
import os
import sys

try:
    import regex as _re
except ImportError:                     # wizwalker depends on regex, but
    import re as _re                    # the diagnostic should still run

#: Copied from `wizwalker/memory/handler.py:33-38`. Duplicated on purpose
#: -- importing it would need wizwalker, which is exactly what may be
#: broken, and this file must run even when that import fails.
AUTOBOT_PATTERN = (
    rb"\x48\x8B\xC4\x55\x41\x54\x41\x55\x41\x56\x41\x57......."
    rb"\x48......\x48.......\x48\x89\x58\x10\x48\x89"
    rb"\x70\x18\x48\x89\x78\x20.......\x48\x33\xC4....."
    rb"..\x4C\x8B\xE9.......\x80......\x0F"
)

#: The signature LaurenzLikeThat's fork looks for. Wizard101 patched and
#: the autobot function's prologue changed, so the vendored 1.8.2
#: signature stopped matching; the fork tracks the current build (and
#: grew AUTOBOT_SIZE from 3900 to 4100 with it).
#:
#: Knowing both turns a dead end into an instruction. "Not found" alone
#: only says wizwalker is out of date; finding the *other* pattern in the
#: same binary says exactly which wizwalker to install.
FORK_PATTERN = (
    rb"\x48\x89\x5C\x24.\x48\x89\x74\x24.\x48\x89\x7C\x24."
    rb"\x55\x41\x54\x41\x55\x41\x56\x41\x57"
    rb"\x48\x8D\xAC\x24....\x48\x81\xEC...."
    rb"\x48\x8B\x05....\x48\x33\xC4\x48\x89\x85...."
    rb"\x4C\x8B\xF1.......\x80......\x0F\x84...."
)

FORK_URL = "https://github.com/LaurenzLikeThat/wizwalker"

KNOWN_PATTERNS = [
    ("wizwalker 1.8.2 (the copy vendored in Deimos/libs)", AUTOBOT_PATTERN),
    (f"LaurenzLikeThat's wizwalker fork ({FORK_URL})", FORK_PATTERN),
]

COMMON_PATHS = [
    r"C:\ProgramData\KingsIsle Entertainment\Wizard101\Bin\WizardGraphicalClient.exe",
    r"C:\Program Files (x86)\KingsIsle Entertainment\Wizard101\Bin\WizardGraphicalClient.exe",
    r"C:\Program Files\KingsIsle Entertainment\Wizard101\Bin\WizardGraphicalClient.exe",
    r"C:\Users\*\AppData\Local\KingsIsle Entertainment\Wizard101\Bin\WizardGraphicalClient.exe",
]


def running_clients():
    """(pid, exe path) for every running game client."""
    out = []
    try:
        import ctypes
        import ctypes.wintypes as wt

        import psutil  # optional, nicest path
        for proc in psutil.process_iter(["pid", "name", "exe"]):
            if (proc.info.get("name") or "").lower() == "wizardgraphicalclient.exe":
                out.append((proc.info["pid"], proc.info.get("exe")))
        return out
    except Exception:
        pass

    # psutil is not a dependency, so fall back to the shell
    try:
        import subprocess
        raw = subprocess.run(
            ["wmic", "process", "where",
             "name='WizardGraphicalClient.exe'", "get", "ProcessId,ExecutablePath"],
            capture_output=True, text=True, timeout=20).stdout
        for line in raw.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            path, _, pid = line.rpartition(" ")
            if pid.strip().isdigit():
                out.append((int(pid), path.strip()))
    except Exception:
        pass
    return out


def find_exe(explicit=None):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for pid, path in running_clients():
        if path and os.path.exists(path):
            return path
    for pattern in COMMON_PATHS:
        for hit in glob.glob(pattern):
            if os.path.exists(hit):
                return hit
    return None


def scan_file(path, pattern=AUTOBOT_PATTERN):
    """Offsets where `pattern` appears in the file."""
    with open(path, "rb") as f:
        blob = f.read()
    return [m.start() for m in _re.finditer(pattern, blob, _re.DOTALL)]


def installed_pattern():
    """The signature the *installed* wizwalker actually uses, if it loads.

    Off Windows the import fails, which is fine -- the diagnostic then
    reasons purely from the known patterns.
    """
    try:
        from wizwalker.memory.handler import HookHandler
        return HookHandler.AUTOBOT_PATTERN
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", help="path to WizardGraphicalClient.exe")
    args = ap.parse_args()

    print("wizwalker hook diagnostic\n" + "=" * 60)

    if sys.platform != "win32":
        print("NOTE: not on Windows, so the running-process checks are "
              "skipped. The on-disk scan still works if you pass --exe.")

    procs = running_clients()
    print(f"\n1. running clients: {len(procs)}")
    for pid, path in procs:
        print(f"     pid {pid}  {path}")
    if len(procs) > 1:
        print("     >> More than one client is running. wizwalker attaches to "
              "the first one it finds, and a second copy is a common source "
              "of confusion. Close the ones you are not playing.")
    if not procs:
        print("     >> No client running. Start Wizard101 and log in first.")

    exe = find_exe(args.exe)
    print(f"\n2. binary: {exe or 'NOT FOUND'}")
    if not exe:
        print("     >> Could not locate WizardGraphicalClient.exe. Pass it "
              "with --exe.")
        return 2

    print("\n3. autobot signatures in the on-disk binary")
    found = {}
    for label, pattern in KNOWN_PATTERNS:
        try:
            hits = scan_file(exe, pattern)
        except Exception as exc:
            print(f"     >> could not read it: {exc}")
            return 2
        found[label] = hits
        mark = f"{len(hits)} match(es)" if hits else "not found"
        print(f"     {label}\n         -> {mark}")

    mine = installed_pattern()
    if mine is not None:
        which = next((lbl for lbl, pat in KNOWN_PATTERNS if pat == mine), None)
        print(f"\n4. the installed wizwalker uses: {which or 'an unrecognised signature'}")
    else:
        print("\n4. wizwalker is not importable here, so reasoning from the "
              "known signatures alone")

    matching = [lbl for lbl, hits in found.items() if hits]
    print("\n" + "=" * 60)

    if mine is not None and which and found.get(which):
        print("VERDICT: your installed wizwalker's signature IS in the binary.")
        print()
        print("So wizwalker is not out of date -- the failure is state in the")
        print("running process. A previous attach zeroed the autobot region")
        print("and did not restore it, so the scan finds nothing in memory.")
        print()
        print("Fix: close the game COMPLETELY and reopen it.")
        print("  - Logging out, or relaunching from the launcher, is not")
        print("    enough. The process itself has to die.")
        print("  - Check Task Manager (Ctrl+Shift+Esc) for any lingering")
        print("    WizardGraphicalClient.exe and end it.")
        print("  - Close Deimos.exe and any other wizwalker script too;")
        print("    two tools patching the same region will do this.")
        return 0

    if matching:
        print("VERDICT: wizwalker is out of date for this build of the game.")
        print()
        print("The binary does not contain the signature your wizwalker looks")
        print("for, but it DOES contain this one:")
        for label in matching:
            print(f"    {label}")
        print()
        print("Restarting the client cannot fix this -- the built-in message")
        print("saying so only anticipates the other cause.")
        print()
        if any("Laurenz" in lbl for lbl in matching):
            print("Fix: install the fork, into whichever venv you run wizAi from:")
            print("    .venv\\Scripts\\python.exe -m pip uninstall -y wizwalker")
            print("    .venv\\Scripts\\python.exe -m pip install "
                  f'"git+{FORK_URL}"')
            print()
            print("It is a drop-in wizwalker -- same package name, same")
            print("Python floor (3.11+), same pure-Python dependencies, no")
            print("Rust. Nothing in deimos_bridge changes.")
        return 1

    print("VERDICT: none of the signatures this tool knows are in the binary.")
    print()
    print("The game has been patched past every wizwalker version listed")
    print("above. Check for a newer Deimos release or a newer fork:")
    print("    https://github.com/Deimos-Wizard101/Deimos-Wizard101")
    print(f"    {FORK_URL}")
    print()
    print("Nothing else in deimos_bridge depends on this. The differential")
    print("harness, the effect audit and the GUI's --demo mode all still")
    print("run while you wait.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
