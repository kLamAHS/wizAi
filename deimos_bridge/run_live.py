"""Point a wizAi policy at a real Wizard101 fight.

**This one needs Windows and a running game client.** wizwalker reads the
client's memory: `wizwalker/constants.py` binds `ctypes.windll.user32` at
import, `wizwalker/__main__.py:40` refuses to start on anything but
win32, and there is no replay or offline mode anywhere in the Deimos tree
to stand in for a live duel. Everything else in `deimos_bridge` runs
anywhere; this does not.

Setup, once:

    cd Deimos
    uv sync                       # or: pip install -e libs/wizwalker
                                  #     pip install -e libs/wizsprinter
    # start Wizard101, log in, and walk into a fight

Then, from the repo root:

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


def _log_decision(log):
    def on_decision(decision, read):
        s = read.state
        log.append({
            "round": read.round_number,
            "decision": repr(decision),
            "card": decision.card_name,
            "target": decision.target_index,
            "passed": decision.passing,
            "reason": decision.reason,
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
        print(f"  round {read.round_number}: {who}  ({decision.reason})")
    return on_decision


def build_policy(kind, cards, school, deck):
    """`policy(sim, state) -> Card | str | None`."""
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
        agent, _ = train_agent(
            cards, deck, school,
            Boss(name="live", hp=3000, school="ice", dmg=150),
            episodes=8000, log=2000)
        return agent.policy()
    raise SystemExit(f"unknown policy {kind!r}")


async def run(args):
    try:
        from wizwalker import ClientHandler
    except Exception as exc:
        raise SystemExit(
            f"wizwalker/wizsprinter did not import ({exc}).\n"
            "This entry point needs Windows and a running Wizard101 client. "
            "Everything else in deimos_bridge -- the differential harness, "
            "the effect audit, the backend tests against mock_client -- runs "
            "without either."
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
    backend = None
    handler = ClientHandler()
    try:
        clients = handler.get_new_clients()
        if not clients:
            raise SystemExit("no Wizard101 client found -- is the game running?")
        client = clients[0]
        await client.activate_hooks()

        backend = WizAiBackend(policy=policy, cards=cards, school=args.school,
                               decklist=deck, on_decision=_log_decision(log),
                               catalog=catalog)
        # WizAiCombatHandler, not SprintyCombat: one decision must be one
        # cast, or the fight is played by a different policy than the one
        # being measured. See live_backend.WizAiCombatHandler.
        combat = make_combat_handler(client, backend)

        print(f"wizAi policy {args.policy!r} taking over combat "
              f"({args.school} wizard)")
        print("walk into a fight — waiting for combat…")
        for fight in range(args.fights):
            print(f"\nfight {fight + 1}/{args.fights}")
            # `wait_for_combat` blocks until a duel starts and then runs
            # `handle_combat` itself (handler.py:64-73). Calling
            # handle_combat again here would be a second, empty pass.
            await combat.wait_for_combat()
            print("  fight over")
    finally:
        await handler.close()
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"policy": args.policy, "school": args.school,
                           "decisions": log}, f, indent=2)
            print(f"\nwrote {args.out} ({len(log)} decisions)")
        if backend is not None:
            # The single most useful line after a live run: a card the
            # resolver could not place is a card the policy never saw.
            print(backend.resolver.report())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--school", default="fire")
    ap.add_argument("--policy", default="blade-stack",
                    choices=("blade-stack", "nuke", "trained"))
    ap.add_argument("--deck", default="",
                    help="comma-separated card names, for the scarcity "
                         "feature and for training the 'trained' policy")
    ap.add_argument("--fights", type=int, default=1)
    ap.add_argument("--out", default="results_live_run.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
