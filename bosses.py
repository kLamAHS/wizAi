"""
Preset enemy + deck registry.

Every enemy in W101 is scripted: HP, school, cheat/attack pattern are known
in advance. That makes deck selection a *contextual* choice — the agent sees
the boss's stats and picks a loadout before the fight. Stats below are
era-appropriate ballparks for classic Wizard City -> Dragonspyre bosses
(replace with scraped values from the wiki's creature pages later).

Boss resist/boost model: 40% resist to own school, +25% from opposing school.
`stunable=None` means unspecified (research-doc convention: unknowns stay
null, they are not silently converted) — the engine treats it as stunnable.
"""
from w101_sim import Boss, CheatScript, CheatRule

BOSSES = {
    # early game: low HP, weak hits — speed decks should win here
    "Lady Blackhope":  Boss("Lady Blackhope",   400, "death",   90),
    "Rattlebones":     Boss("Rattlebones",      500, "death",  105),
    "Foulgaze":        Boss("Foulgaze",        1000, "myth",   130),
    "Lord Nightshade": Boss("Lord Nightshade", 1500, "death",  180),
    # mid game: buff math starts to matter
    "Krokopatra":      Boss("Krokopatra",      2200, "storm",  220),
    "Meowiarty":       Boss("Meowiarty",       3000, "myth",   260),
    "Jade Oni":        Boss("Jade Oni",        4000, "life",   300),
    # late: full stacks mandatory; school matchup punishes lazy picks
    "Prince Gobblestone": Boss("Prince Gobblestone", 2600, "ice", 240),
    "Ervin Flamerender":  Boss("Ervin Flamerender",  3600, "fire", 280),
    "Malistaire":      Boss("Malistaire",      6000, "death",  400),
}

# ---------------------------------------------------------------- cheats
# Illustrative cheat scripts exercising the interrupt hooks. The *shapes*
# (trap punishment, enrage threshold, periodic auto-shield) mirror real
# boss cheats; the numbers are demo values, not scraped ones.

CHEATS = {
    # "That's cheating!" — counter any trap the player lays with a free hit
    "trap_counter": CheatScript([
        CheatRule(event="after_player_cast", when={"card_kind": "trap"},
                  ops=[{"op": "hit", "amount": 320, "tgt": "enemy"}],
                  message="No traps allowed"),
    ]),
    # enrage once below 50%: blade self and shield up
    "enrage_at_half": CheatScript([
        CheatRule(event="hp_below", when={"frac": 0.5}, once=True,
                  ops=[{"op": "charm", "percent": 0.40, "schools": None,
                        "tgt": "self"},
                       {"op": "ward", "percent": -0.50, "schools": None,
                        "tgt": "self"}],
                  message="You will not defeat me"),
    ]),
    # every 4th round, throw up a tower-style shield
    "shield_cycle": CheatScript([
        CheatRule(event="round_start", when={"every": 4},
                  ops=[{"op": "ward", "percent": -0.50, "schools": None,
                        "tgt": "self"}],
                  message="Shield cycle"),
    ]),
}

def cheat_variant(name, cheat_key):
    """Clone a registry boss with a cheat script attached."""
    b = BOSSES[name]
    return Boss(b.name + f" ({cheat_key})", b.hp, b.school, b.dmg,
                b.resist_own, b.boost_opp, cheat=CHEATS[cheat_key])

# survival presets: era-appropriate player HP per school (ballparks)
PLAYER_HP = {"fire": 2900, "ice": 3600, "storm": 2500, "myth": 3000,
             "life": 3200, "death": 3100, "balance": 3150}

# ---------------------------------------------------------------- decks
# Same-name-same-source buffs don't stack, so depth comes from DIVERSITY:
# own blade + Elemental/Spirit Blade + own trap + Elemental/Spirit Trap
# + universal wards (Hex/Feint) + Fuel (fire only, 3 charges).
# Provenance variants ("Fireblade@tc") DO stack with their deck versions.
# Duplicate same-source copies are draw redundancy, not extra stacks.

DECKS = {
    "fire": {
        "speed": (["Fire Shark"] * 4 + ["Ash Bats"] * 4 + ["Fire Cat"] * 2 +
                  ["Fireblade"] * 2),
        "stack": (["Helephant"] * 2 + ["Fire Shark"] * 2 +
                  ["Fireblade"] * 2 + ["Elemental Blade"] * 2 +
                  ["Fire Trap"] * 2 + ["Elemental Trap"] * 2),
        "oneshot": (["Helephant"] * 2 + ["Immolate"] * 1 +
                    ["Fireblade"] * 2 + ["Elemental Blade"] * 2 +
                    ["Fire Trap"] * 1 + ["Elemental Trap"] * 1 +
                    ["Fuel"] * 1 + ["Feint"] * 2),
    },
    "ice": {
        "speed": (["Evil Snowman"] * 4 + ["Ice Bats"] * 4 + ["Ice Snake"] * 2 +
                  ["Iceblade"] * 2),
        "stack": (["Colossus"] * 2 + ["Frostbite"] * 2 +
                  ["Iceblade"] * 2 + ["Elemental Blade"] * 2 +
                  ["Ice Trap"] * 2 + ["Elemental Trap"] * 2),
        "oneshot": (["Colossus"] * 2 + ["Frostbite"] * 1 +
                    ["Iceblade"] * 2 + ["Elemental Blade"] * 2 +
                    ["Ice Trap"] * 1 + ["Elemental Trap"] * 1 + ["Feint"] * 2),
        # the answer to the same-school wall: traps FIRST, then prism —
        # ice damage picks up ice traps, converts to fire, hits the +25%
        # fire boost instead of the 40% ice resist
        "prism": (["Colossus"] * 2 + ["Evil Snowman"] * 1 +
                  ["Iceblade"] * 2 + ["Elemental Blade"] * 2 +
                  ["Ice Trap"] * 1 + ["Ice Prism"] * 2 + ["Feint"] * 2),
    },
    "myth": {
        "speed": (["Cyclops"] * 4 + ["Blood Bat"] * 4 + ["Mythblade"] * 2),
        "stack": (["Minotaur"] * 3 + ["Cyclops"] * 2 +
                  ["Mythblade"] * 2 + ["Spirit Blade"] * 2 +
                  ["Myth Trap"] * 2 + ["Spirit Trap"] * 2),
        "oneshot": (["Minotaur"] * 2 + ["Cyclops"] * 1 +
                    ["Mythblade"] * 2 + ["Spirit Blade"] * 2 +
                    ["Myth Trap"] * 1 + ["Spirit Trap"] * 1 + ["Feint"] * 2),
    },
    "storm": {
        "speed": (["Lightning Bats"] * 4 + ["Jolted Snowman"] * 3 +
                  ["Stormblade"] * 2),
        "stack": (["Kraken"] * 2 + ["Jolted Snowman"] * 2 +
                  ["Stormblade"] * 2 + ["Elemental Blade"] * 2 +
                  ["Storm Trap"] * 2 + ["Elemental Trap"] * 2),
        "oneshot": (["Kraken"] * 2 + ["Tempest"] * 1 +
                    ["Stormblade"] * 2 + ["Elemental Blade"] * 2 +
                    ["Storm Trap"] * 1 + ["Elemental Trap"] * 1 +
                    ["Feint"] * 2),
    },
    "death": {
        "speed": (["Banshee"] * 4 + ["Ghoul"] * 3 + ["Deathblade"] * 2),
        "stack": (["Wraith"] * 2 + ["Vampire"] * 2 +
                  ["Deathblade"] * 2 + ["Spirit Blade"] * 2 +
                  ["Death Trap"] * 2 + ["Spirit Trap"] * 2),
        # drains turn damage into sustain: the survival objective's school
        "oneshot": (["Wraith"] * 2 + ["Vampire"] * 1 +
                    ["Deathblade"] * 2 + ["Spirit Blade"] * 2 +
                    ["Death Trap"] * 1 + ["Spirit Trap"] * 1 +
                    ["Curse"] * 1 + ["Feint"] * 2),
    },
    "life": {
        "speed": (["Lifeclops"] * 4 + ["Leprechaun"] * 3 + ["Lifeblade"] * 2),
        "stack": (["Centaur"] * 2 + ["Lifeclops"] * 2 +
                  ["Lifeblade"] * 2 + ["Spirit Blade"] * 2 +
                  ["Life Trap"] * 2 + ["Spirit Trap"] * 2),
        "oneshot": (["Centaur"] * 2 + ["Lifeclops"] * 1 +
                    ["Lifeblade"] * 2 + ["Spirit Blade"] * 2 +
                    ["Life Trap"] * 1 + ["Spirit Trap"] * 1 + ["Feint"] * 2),
        "sustain": (["Centaur"] * 2 + ["Lifeclops"] * 2 +
                    ["Lifeblade"] * 2 + ["Spirit Blade"] * 1 +
                    ["Satyr"] * 2 + ["Sprite"] * 1 + ["Tower Shield"] * 2),
    },
    "balance": {
        "speed": (["Locust Swarm"] * 4 + ["Balanceblade"] * 2 +
                  ["Hex"] * 2),
        "stack": (["Hydra"] * 2 + ["Locust Swarm"] * 2 +
                  ["Balanceblade"] * 2 + ["Bladestorm"] * 2 +
                  ["Hex"] * 2 + ["Curse"] * 2),
        # Judgement is X-pip: value scales with patience, a real RL problem
        "oneshot": (["Judgement"] * 2 + ["Locust Swarm"] * 1 +
                    ["Balanceblade"] * 2 + ["Bladestorm"] * 1 +
                    ["Hex"] * 2 + ["Curse"] * 1 + ["Feint"] * 2),
    },
}
