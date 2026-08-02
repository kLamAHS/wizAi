"""Point a wizAi policy at a real Wizard101 fight.

**This one needs Windows and a running game client.** wizwalker reads the
client's memory: `wizwalker/constants.py` binds `ctypes.windll.user32` at
import, `wizwalker/__main__.py:40` refuses to start on anything but
win32, and there is no replay or offline mode anywhere in the Deimos tree
to stand in for a live duel. Everything else in `deimos_bridge` runs
anywhere; this does not.

Setup, once. Only wizwalker is needed -- this goes through
`WizAiCombatHandler`, which subclasses `wizwalker.combat.CombatHandler`,
so wizsprinter (and its 3.13 floor, and wizlaunch's Rust extension) never
enter the picture:

    python -m venv .venv
    .venv\\Scripts\\python.exe -m pip install -e Deimos\\libs\\wizwalker numpy
    # start Wizard101 and log in

See RUNNING_LIVE.md for the full walkthrough. Then, from the repo root:

    python -m deimos_bridge.run_live --school fire
    python -m deimos_bridge.run_live --school fire --policy trained \\
        --deck "Fireblade,Fireblade,Sunbird,Sunbird,Sunbird,Tri Blade"

What happens each planning phase: the board is read into a wizAi `State`,
the policy picks a card, and the pick is handed to `SprintyCombat` as a
`PriorityLine`. Every decision is logged with the state that produced it,
so a run is auditable afterwards -- which matters, because the first
thing to check on a live run is not whether the policy won but whether it
was ever shown the right board.
"""
import argparse
import asyncio
import json


def _log_decision(log, seat=0):
    def on_decision(decision, read):
        s = read.state
        log.append({
            "wizard": seat + 1,
            "round": read.round_number,
            "decision": repr(decision),
            "card": decision.card_name,
            "target": decision.target_index,
            "passed": decision.passing,
            "reason": decision.reason,
            "policy": getattr(decision, "policy", ""),
            "player_hp": s.player_hp,
            "pips": [s.norm_pips, s.pow_pips],
            "hand": [c.name for c in s.hand],
            "hidden": list(getattr(read, "hidden", [])),
            "hand_visibility": getattr(read, "hand_visibility", 1.0),
            "enemies": [(e.name, e.hp) for e in s.enemies],
            "player_charms": [h.name for h in s.player.charms],
            "enemy_wards": [h.name for h in s.enemies[0].wards] if s.enemies else [],
        })
        who = "pass" if decision.passing else decision.card_name
        why = getattr(decision, "policy", "") or decision.reason
        print(f"  round {read.round_number}: {who}  ({why})")
    return on_decision


def build_policy(kind, cards, school, deck):
    """`policy(sim, state) -> Card | str | None`."""
    if kind == "ttk":
        from .policies import greedy_ttk
        return greedy_ttk()
    if kind == "school-aware":
        from .policies import school_aware_blade_stack
        return school_aware_blade_stack(3)
    if kind == "blade-stack":
        from w101_sim import make_blade_stack
        return make_blade_stack(3)
    if kind == "nuke":
        from w101_sim import strat_nuke_asap
        return strat_nuke_asap
    if kind == "trained":
        # Train against the simulator first, then play the live fight with
        # the resulting Q table. The alternative -- learning online in a
        # real duel -- would spend real fights on exploration.
        #
        # The agent's state key is deck-specific (`Featurizer.__init__`
        # indexes blades and nukes by position in the decklist), so a
        # decklist is required rather than optional here.
        if not deck:
            raise SystemExit(
                "--policy trained needs --deck: the Q table is keyed on "
                "the deck's own blade/nuke positions, so a policy trained "
                "for one decklist means nothing for another.")
        from rl_agent import train_agent
        from w101_sim import Boss

        from .policies import trained_policy
        # MORTAL. train_agent defaults to player_hp=10**9, and
        # Featurizer.key writes -1 into the health slot for an immortal
        # fight -- so a policy trained on the default shares no state
        # with a live wizard, reads zero everywhere, and passes every
        # turn. Measured: 4% state coverage immortal, 96% mortal.
        agent, _ = train_agent(
            cards, deck, school,
            Boss(name="live", hp=1500, school="ice", dmg=90),
            episodes=8000, log=2000, player_hp=800)
        # Wrapped so an unvisited state plays the fallback rather than
        # silently passing. See policies.TrainedPolicy.
        return trained_policy(agent)
    raise SystemExit(f"unknown policy {kind!r}")


async def run(args):
    try:
        from wizwalker import ClientHandler
    except Exception as exc:
        raise SystemExit(
            f"wizwalker did not import ({exc}).\n\n"
            "  install it:  python -m venv .venv\n"
            "               .venv\\Scripts\\python.exe -m pip install "
            "-e Deimos\\libs\\wizwalker numpy\n\n"
            "This entry point needs Windows and a running Wizard101 client. "
            "Everything else in deimos_bridge -- the differential harness, "
            "the effect audit, the GUI's --demo mode, the tests against "
            "mock_client -- runs without either. See "
            "deimos_bridge/RUNNING_LIVE.md."
        )

    from .live_backend import WizAiBackend, make_combat_handler
    from .live_state import build_catalog

    # One extra pass over spells_full.json buys the difference between
    # "3 unresolved names" and "these 2 need a decoder gap closed, this 1
    # is a spelling problem" -- worth it once per run.
    catalog = build_catalog()
    cards = catalog["cards"]
    deck = [d.strip() for d in args.deck.split(",")] if args.deck else []
    policy = build_policy(args.policy, cards, args.school, deck)

    log = []
    backends = []
    handler = ClientHandler()
    wizards = max(1, int(getattr(args, "wizards", 1) or 1))
    # Above one wizard the run becomes a hivemind: every seat submits its
    # board each round and waits for the others, then chooses against a
    # board that already carries what the rest of the party committed to.
    # See deimos_bridge/hivemind.py.
    hive = None
    if wizards > 1:
        from .hivemind import Hivemind
        hive = Hivemind(passes=args.passes, on_status=lambda m: print(f"  {m}"))
    try:
        clients = handler.get_new_clients()
        if not clients:
            raise SystemExit("no Wizard101 client found -- is the game running?")
        if len(clients) < wizards:
            raise SystemExit(
                f"--wizards {wizards} needs {wizards} running, logged-in "
                f"clients; wizwalker found {len(clients)}.")
        clients = clients[:wizards]
        for i, client in enumerate(clients):
            try:
                await client.activate_hooks()
            except Exception as exc:
                # PatternFailed here has two causes with one message, and
                # the message's advice ("restart the client") only fixes
                # one of them. Rather than reprint it, hand over to the
                # diagnostic that can actually tell them apart.
                if "PatternFailed" in type(exc).__name__ \
                        or "Pattern" in str(exc):
                    raise SystemExit(
                        "wizwalker could not install its hooks: the autobot "
                        "signature was not found in the running client.\n\n"
                        "That has two causes and the built-in advice only "
                        "covers one. Run this to find out which:\n\n"
                        "    .venv\\Scripts\\python.exe -m "
                        "deimos_bridge.diagnose_hooks\n"
                    ) from exc
                raise
            backend = WizAiBackend(
                policy=policy, cards=cards, school=args.school,
                decklist=deck, on_decision=_log_decision(log, i),
                catalog=catalog, policy_name=args.policy, seat=i,
                coordinator=hive, party_size=wizards)
            if hive is not None:
                hive.join(i, f"wizard {i + 1}")
            backends.append(backend)

        # WizAiCombatHandler, not SprintyCombat: one decision must be one
        # cast, or the fight is played by a different policy than the one
        # being measured. See live_backend.WizAiCombatHandler.
        combats = [make_combat_handler(c, b)
                   for c, b in zip(clients, backends)]

        print(f"wizAi policy {args.policy!r} taking over combat "
              f"({args.school} wizard"
              + (f" x{wizards}, agreeing each round" if wizards > 1 else "")
              + ")")
        print("walk into a fight — waiting for combat…")

        async def fight_loop(seat, combat):
            for fight in range(args.fights):
                who = "" if wizards == 1 else f"[wizard {seat + 1}] "
                print(f"\n{who}fight {fight + 1}/{args.fights}")
                try:
                    # `wait_for_combat` blocks until a duel starts and
                    # then runs `handle_combat` itself (handler.py:64-73).
                    # Calling handle_combat again here would be a second,
                    # empty pass.
                    await combat.wait_for_combat()
                    print(f"  {who}fight over")
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    # A duel ending is a normal event that wizwalker often
                    # signals by a read failing -- the participant list is
                    # freed under it. Reporting that as a traceback makes
                    # a finished fight look like a crash.
                    name = type(exc).__name__
                    if any(k in name for k in ("Memory", "ClientClosed",
                                               "ReadingEnum", "Invalidated")):
                        print(f"  {who}fight ended ({name})")
                    else:
                        print(f"  {who}fight aborted: {name}: {exc}")
                        if fight == 0:
                            raise
                finally:
                    if hive is not None:
                        # Out of the circle: the rest of the party must
                        # stop waiting for this seat at the barrier, or
                        # every one of their rounds pays the full timeout.
                        hive.leave_combat(seat)

        # Concurrently. Four sequential loops would mean wizard 2 never
        # reaching its planning phase until wizard 1's duel was over,
        # which is a queue rather than a party.
        await asyncio.gather(*[fight_loop(i, c)
                               for i, c in enumerate(combats)])
    finally:
        await handler.close()
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"policy": args.policy, "school": args.school,
                           "wizards": wizards, "decisions": log}, f, indent=2)
            print(f"\nwrote {args.out} ({len(log)} decisions)")
        if hive is not None:
            print(f"party: {hive.rounds} coordinated round(s), "
                  f"{hive.retargets} move(s) changed by the party, "
                  f"{hive.saved} card(s) held back from an already-dead "
                  f"board")
        if backends:
            # The single most useful line after a live run: a card the
            # resolver could not place is a card the policy never saw.
            print(backends[0].resolver.report())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--school", default="fire")
    ap.add_argument("--policy", default="ttk",
                    choices=("ttk", "school-aware", "blade-stack", "nuke",
                             "trained"),
                    help="ttk (default) simulates every candidate move and "
                         "picks the one that kills soonest; school-aware "
                         "stacks a fixed number of buffs but only ones that "
                         "can act on the hit; blade-stack is the heuristic "
                         "the published tables use, which assumes a "
                         "school-coherent deck")
    ap.add_argument("--deck", default="",
                    help="comma-separated card names, for the scarcity "
                         "feature and for training the 'trained' policy")
    ap.add_argument("--fights", type=int, default=1)
    ap.add_argument("--wizards", type=int, default=1,
                    help="how many running clients to drive, up to the four "
                         "a battle circle seats. Above one they play as a "
                         "hivemind: each round every wizard waits for the "
                         "others and then chooses against a board carrying "
                         "what the rest of the party committed to, so a trap "
                         "one lays is cashed by the next and nobody fires "
                         "into a mob that is already dead")
    ap.add_argument("--passes", type=int, default=2,
                    help="coordination passes per round (0 = every wizard "
                         "decides alone, which is the uncoordinated "
                         "baseline)")
    ap.add_argument("--out", default="results_live_run.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
