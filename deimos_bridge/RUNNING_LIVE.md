# Running a wizAi policy against real Wizard101 combat

Everything else in `deimos_bridge` runs anywhere. This does not: wizwalker
reads and writes the live game client's memory and drives the game by
clicking, so it needs Windows, the real client, and a wizard standing in
a fight.

---

## 0. What you need

| | |
|---|---|
| OS | Windows. Not negotiable — see "Why not Linux/Wine" at the bottom. |
| Python | **3.11 or newer.** That is wizwalker's floor, and wizwalker is all this needs. |
| wizwalker | the [LaurenzLikeThat fork](https://github.com/LaurenzLikeThat/wizwalker), not the copy vendored in `Deimos/libs` — the game patched past that one. `setup-windows.bat` handles it. |
| Game | Wizard101 installed, running, and **logged in to the wizard you want to play**. |
| Repo | This repository, with `Deimos/` present — it is a subtree, not a submodule, so a normal clone already has it. |

You do **not** need `Deimos.exe`, you do not launch the Deimos GUI, and
you do not need `uv`. This runs from source and takes over combat
directly.

---

## 1. Install

Double-click **`setup-windows.bat`** in the repository root. That is the
whole step. It creates `.venv`, installs the wizwalker fork, numpy and
PyQt6, and verifies the imports. It is safe to re-run.

You need `git` and Python 3.11+ on PATH first; the script checks and says
so if not.

<details>
<summary>What it does, and why the fork</summary>

Only **wizwalker** is required. `run_live.py` goes through
`WizAiCombatHandler`, which subclasses `wizwalker.combat.CombatHandler`;
wizsprinter is used only by the alternate backend path, and
`combat_api_shim.py` falls back to local stand-ins when it is absent. So
none of Deimos's heavier requirements apply — no wizsprinter (`>=3.13`),
no wizlaunch (a Rust extension), no build tools, no `uv`.

But **not** the wizwalker vendored in `Deimos/libs`. Wizard101 patched
and the autobot function's prologue changed, so that copy's byte
signature no longer matches and hook installation dies with
`PatternFailed`. [LaurenzLikeThat's
fork](https://github.com/LaurenzLikeThat/wizwalker) tracks the current
build. It is a drop-in — same package name, same `>=3.11` floor, same
pure-Python dependencies.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install "git+https://github.com/LaurenzLikeThat/wizwalker"
.venv\Scripts\python.exe -m pip install numpy PyQt6
```

`numpy` is only needed for `--policy trained` (`rl_agent` →
`dp_solver`); `PyQt6` only for the GUI.

Check it took:

```powershell
.venv\Scripts\python.exe -c "import wizwalker; print('ok')"
```

`AttributeError` on `windll` means you are not on Windows.
`ModuleNotFoundError: pymem` means the install did not complete — `pymem`
is declared `sys_platform == 'win32'`, so it only installs there.
</details>

### Running it afterwards

Two batch files in the repository root, both of which find `.venv`
relative to themselves — so a desktop **shortcut** to either works, as
long as the `.bat` stays in the folder:

| | |
|---|---|
| `wizAi-gui.bat` | opens the window; press **Play live** |
| `wizAi-live.bat` | the console runner; takes any `run_live` argument, e.g. `wizAi-live.bat --school ice --fights 5` |

They use `python.exe` rather than `pythonw.exe` on purpose: the console
stays open, so an error before the window appears is visible instead of
the app silently never starting.

---

## 2. Start the game

1. Launch Wizard101 normally.
2. Log in to the wizard whose school and deck you are going to configure
   below.
3. Leave it standing somewhere you can start a fight.

Do not start the fight yet. The runner waits for combat and takes over at
the first planning phase, which is cleaner than joining mid-duel.

---

## 3. Run it

From the **repository root** (not from `Deimos/`), so that `data_full`,
`w101_sim` and `deimos_bridge` are importable:

```powershell
.venv\Scripts\python.exe -m deimos_bridge.run_live --school fire
```

or just double-click `wizAi-live.bat`. You should see:

```
wizAi policy 'blade-stack' taking over combat (fire wizard)
walk into a fight — waiting for combat…

fight 1/1
```

Now **walk into a fight in the game.** At each planning phase the runner
prints the decision:

```
  round 1: Fireblade  (blade-stack)
  round 2: Fireblade  (blade-stack)
  round 3: Sunbird  (blade-stack)
  fight over
```

When it finishes it writes `results_live_run.json` and prints the
name-resolution report.

### The options that matter

```powershell
# the scripted baseline your tables use — start here
... -m deimos_bridge.run_live --school fire --policy blade-stack

# a trained Q-learning policy. --deck is REQUIRED: the agent's state key
# is built from this deck's own blade and nuke positions, so a table
# trained for one decklist means nothing for another.
... -m deimos_bridge.run_live --school fire --policy trained ^
      --deck "Fireblade,Fireblade,Fireblade,Sunbird,Sunbird,Sunbird,Tri Blade"

# several fights in a row
... -m deimos_bridge.run_live --school fire --fights 5

# somewhere else for the log
... -m deimos_bridge.run_live --school fire --out run_2026_07_29.json
```

`--policy trained` trains against the simulator first (a few minutes) and
then plays the live fight with the resulting table. It deliberately does
**not** learn online — exploring in real duels spends real fights.

---

## 4. Watch it with the GUI instead

```powershell
.venv\Scripts\python.exe -m deimos_bridge.gui
```

Same run, with the board, the decisions, the damage-model residuals and
the naming triage live. Try `--demo` first — it drives the whole window
from canned data and needs no game, so you can see what the panels do
before trusting one.

### Changing and training models without disconnecting

Press **Play live** once and leave it connected. From there:

- **The policy dropdown swaps mid-fight.** The change lands on the next
  planning phase; the round already in flight finishes under the policy
  that started it. No reconnect, so the run's telemetry stays continuous
  and you can compare two policies inside one session.
- **Train works while connected.** It runs at a lower thread priority so
  the fight keeps its timing, and when it finishes — if `trained (Q)` is
  the current selection — the new table is handed to the running fight
  in place.
- **Your max health is read off the client on connect** and filled into
  the *my HP* box. That number matters more than it looks: `Featurizer`
  buckets health as a fraction of the maximum, so a table trained
  against a guessed 800 and played on a 1,300 HP wizard indexes
  different states for the same board.
- **Your gear is read too** — damage, accuracy, pierce and resist, per
  school plus the "all schools" bonus the game keeps separately. Both
  the live fight and training use it. Without it the simulator prices
  every hit as though you were wearing nothing, and then optimises *that*
  fight: on a 2000hp mob with an ice deck, `ttk-lookahead` opens with a
  trap given 9% damage and 4% pierce, and opens with the hit given
  neither. The line under the controls says which it is using.

This ordering is the useful one, not a convenience. Both inputs to a good
training run — the deck (the picker learns card names from what it saw in
combat) and the health — only exist *after* you have been connected, so
requiring a disconnect to train meant training on guesses.

### Hotkeys

*Teleport to quest* and *Advance dialogue* also bind to keys (F1 and F2
by default), so you never have to alt-tab out of a full-screen client to
use them. Pick the keys next to the **Hotkeys** checkbox.

These are **system-wide** keys, via Win32 `RegisterHotKey`. Two
consequences worth knowing before you bind something:

- They fire whatever window has focus, not just the game.
- While the run is connected the key is taken *away* from every other
  program, Wizard101 included. Bind a key the game uses and you lose it
  in the game until you press Stop.

If another running program already owns the key, that binding is skipped
with a message and the rest of the run continues — pick a different key.

### Is the model actually driving?

Under the controls is a line reading something like:

```
14 round(s): trained (Q) — Q table ×12  ·  trained (Q) — fallback (state not in Q table) ×2
Q table decided 86% of the boards it was shown (2 fell back to the heuristic)
```

That is the answer to "did selecting the trained policy do anything?" —
which is otherwise unanswerable, because a Q table with no opinion falls
back to the heuristic and plays a completely ordinary-looking fight. The
**Decisions** tab carries the same thing per row, with fallback rows
tinted.

A coverage near zero means the agent has never seen boards like these.
Train more episodes, or train with the deck and health you are actually
holding rather than the defaults.

---

## 5. Read the result

The first number to look at is **hand visibility**, on the Naming tab or
as `summary.hand_visibility` in the JSON.

If it is below ~0.9, stop and fix that before reading anything else. It
means the policy was planning against a hand it could only partly see,
and nothing else in the run is measuring the policy you trained. The
panel splits each miss by cause:

- *"undecoded effect kSummonCreature"* — the card is real, the decoder
  skipped it. Either close the gap in `data_full._map_effect` or accept
  that card is unmodellable and take it out of the deck.
- *"not in the game data under this name"* — a spelling mismatch. Add it
  to `deimos_bridge.live_state.ALIASES`.

Then the **Damage model** tab, which is the thing simulation cannot tell
you: before each cast wizAi predicts the damage, and the next round's
real HP says what actually happened. A consistent bias means the model is
wrong in a fixable way; scatter with no bias usually means unmodelled
crits or damage ranges.

---

## Troubleshooting

**`no Wizard101 client found -- is the game running?`**
wizwalker finds the game by window class `"Wizard Graphical Client"`
(`utils.get_all_wizard_handles`). The game must be fully launched — the
launcher alone is not enough.

**`wizwalker.errors.PatternFailed: Pattern b'\x48\x8B\xC4...' failed.
You most likely need to restart the client.`**

This one message covers two different problems, and restarting only fixes
one of them. Run:

```powershell
.venv\Scripts\python.exe -m deimos_bridge.diagnose_hooks
```

It scans the game binary **on disk** for the same signature, which the
running process cannot have altered, and tells you which you have:

- *Signature is in the binary* → the failure is stale state in the
  running process. `_prepare_autobot` overwrites 3900 bytes of the
  autobot function with zeros to use as scratch space for hook shellcode
  (`memory/handler.py:80-93`), and `_rewrite_autobot` restores them on a
  clean shutdown. A crash or a Ctrl-C at the wrong moment skips that, so
  the region stays zeroed and the next scan finds nothing. **Close the
  game completely** — logging out or relaunching from the launcher is not
  enough, the process has to die. Check Task Manager for a lingering
  `WizardGraphicalClient.exe`, and close `Deimos.exe` or any other
  wizwalker script, since two tools patching the same region cause this.
- *Signature is not in the binary, but the fork's is* → the game has been
  patched past your wizwalker. Restarting cannot help. This is what the
  copy vendored in `Deimos/libs` does today; install the fork:

  ```powershell
  .venv\Scripts\python.exe -m pip uninstall -y wizwalker
  .venv\Scripts\python.exe -m pip install "git+https://github.com/LaurenzLikeThat/wizwalker"
  ```

  or just re-run `setup-windows.bat`.
- *Neither signature is in the binary* → the game has moved past both.
  Check for a newer Deimos release or a newer fork.

**Hooks fail some other way, or memory reads raise immediately.**
Try running the terminal as Administrator. wizwalker attaches to the game
process, patches instructions and injects a hook thread; if your Python
cannot open the process with write access, none of it works.

**The policy decides, but nothing happens in the game.**
The mouseless cursor hook is not active. `WizAiCombatHandler.handle_round`
wraps each round in `async with self.client.mouse_handler`, which is what
activates it — if you have written your own handler, do the same.

**It casts, but plays a card you did not expect.**
Check the Decisions tab's "passed over" column. If the card it wanted is
listed under Naming, it never had it.

**It casts several spells in one round.**
You are going through `WizAiBackend` as a wizsprinter backend rather than
through `WizAiCombatHandler`. `SprintyCombat` re-queries a `NamedSpell`
after each cast and plays every duplicate in hand. `run_live.py` uses the
handler for exactly this reason; do not measure a policy through the
backend.

**`ModuleNotFoundError: data_full`**
You are not in the repository root. Run from there, not from `Deimos/`.

**`'uv' is not recognized as an internal or external command`**
You do not need uv — use the venv install in step 1. If you do want it,
install it and then **close and reopen the terminal**; the installer edits
PATH and your current session will not see it.

**`ModuleNotFoundError: wizwalker`**
The venv install did not happen, or you are running the system `python`
instead of `.venv\Scripts\python.exe`.

---

## Why not Linux, or Wine

Not stubbornness — each of these was checked:

- `wizwalker/constants.py` binds `ctypes.windll.user32` at module scope,
  so `import wizwalker` raises before anything else happens.
- `pymem==1.13.1` is declared `sys_platform == 'win32'`. The memory layer
  is absent, not merely broken.
- wizwalker pattern-scans the client's PE for byte signatures, patches
  live instructions, and runs hand-assembled x86-64 shellcode via
  `CreateRemoteThread`. Wine does not carry that faithfully.
- There is no replay or headless mode anywhere in the Deimos tree to
  stand in for a live duel.

Which is why `mock_client.py` exists: the live read and the decision path
are duck-typed against the wizwalker API rather than importing it, so the
identical code runs and is tested on any machine. The only thing that
genuinely needs your Windows box is the final mouse click.
