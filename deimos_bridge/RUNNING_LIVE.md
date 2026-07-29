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
| Python | **3.11 or newer.** That is wizwalker's floor (`libs/wizwalker/pyproject.toml`), and wizwalker is all this needs. |
| Game | Wizard101 installed, running, and **logged in to the wizard you want to play**. |
| Repo | This repository, with `Deimos/` present — it is a subtree, not a submodule, so a normal clone already has it. |

You do **not** need `Deimos.exe`, you do not launch the Deimos GUI, and
you do not need `uv`. This runs from source and takes over combat
directly.

---

## 1. Install

Only **wizwalker** is required. `run_live.py` goes through
`WizAiCombatHandler`, which subclasses `wizwalker.combat.CombatHandler`;
wizsprinter is used only by the alternate backend path, and
`combat_api_shim.py` falls back to local stand-ins when it is absent.

That matters, because wizwalker is the cheap one: pure Python, `>=3.11`,
no Rust. Installing all of Deimos would drag in wizsprinter (`>=3.13`)
and wizlaunch (a Rust extension) for nothing.

```powershell
# from the repository root
python --version                     # must be 3.11+
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e Deimos\libs\wizwalker numpy
```

`numpy` is wizAi's only extra dependency, and only for `--policy
trained` (`rl_agent` → `dp_solver`). Add `PyQt6` if you want the GUI.

Check it took:

```powershell
.venv\Scripts\python.exe -c "import wizwalker; print('ok')"
```

If that raises `AttributeError` on `windll`, you are not on Windows. If it
raises `ModuleNotFoundError: pymem`, the install did not complete —
`pymem` is declared `sys_platform == 'win32'`, so it only installs there.

<details>
<summary>Or install the whole Deimos workspace with uv</summary>

Heavier, and only worth it if you also want Deimos's own bot. Needs
Python 3.13+ and [uv](https://docs.astral.sh/uv/):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# then CLOSE AND REOPEN the terminal -- the installer edits PATH
cd Deimos
uv sync
cd ..
Deimos\.venv\Scripts\python.exe -m pip install numpy
```

Substitute `Deimos\.venv\Scripts\python.exe` for `.venv\Scripts\python.exe`
everywhere below. PyQt6 comes along for free, since Deimos depends on it.
</details>

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

You should see:

```
wizAi policy 'blade-stack' taking over combat (fire wizard)
walk into a fight — waiting for combat…

fight 1/1
```

Now **walk into a fight in the game.** At each planning phase the runner
prints the decision:

```
  round 1: Fireblade  (policy choice)
  round 2: Fireblade  (policy choice)
  round 3: Sunbird  (policy choice)
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

**Hooks fail, or memory reads raise immediately.**
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
