"""
Preset enemy + deck registry.

Every enemy in W101 is scripted: HP, school, cheat/attack pattern are known
in advance. That makes deck selection a *contextual* choice — the agent sees
the boss's stats and picks a loadout before the fight. Stats below are
era-appropriate ballparks for classic Wizard City -> Dragonspyre bosses
(replace with scraped values from the wiki's creature pages later).

Boss resist/boost model: 40% resist to own school, +25% from opposing school.
"""
from w101_sim import Boss

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

# ---------------------------------------------------------------- decks
# Same-name buffs don't stack, so depth comes from DIVERSITY:
# own blade + Elemental/Spirit Blade + own trap + Elemental/Spirit Trap
# + universal wards (Hex/Feint) + Fuel (fire only, 3 charges).
# Duplicate buff copies are draw redundancy, not extra stacks.

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
}
# storm/death omitted until the fresh scrape lands — cards_clean.json is
# missing their mid/high nukes (no Kraken/Triton, no Vampire/Skeletal Pirate).
