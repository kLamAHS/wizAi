"""
Curated card mechanics for the v0.3 engine.

This is the machine-readable effect-primitive layer (the research doc's
`spell_effects` table, in code): everything the flat wiki scrape cannot
express — DoT splits, multi-hit components, drains, prisms, absorbs,
per-pip spells, dual-school shields, dispels, self-costs, stuns, summons.

Op format (list per card, executed in order):
  {'op': 'hit',    'amount': A, 'school': s|None, 'per_pip': bool,
                   'tgt': 'enemy'|'enemies', 'group': g}
  {'op': 'dot',    'total': A, 'rounds': R, ...}         damage over time
  {'op': 'hot',    'total': A, 'rounds': R, 'tgt': ...}  heal over time
  {'op': 'heal',   'amount': A, 'per_pip': bool, 'tgt': 'self'|'ally'|'allies'}
  {'op': 'drain',  'amount': A, 'fraction': 0.5, 'tgt': 'enemy'|'enemies'}
  {'op': 'charm',  'percent': p, 'schools': spec, 'kind': 'damage'|'accuracy'
                   |'heal', 'tgt': 'self'|'enemy'|'allies'|'enemies'}
  {'op': 'ward',   'percent': p, 'schools': spec, 'charges': n, 'tgt': ...}
                   (+p on enemy = trap, -p on self = shield, +p on self =
                    Feint-style backlash: uniformly "incoming damage * 1+p")
  {'op': 'prism',  'from': s1, 'to': s2, 'tgt': 'enemy'}
  {'op': 'absorb', 'amount': A, 'tgt': 'self'|'ally'|'allies'}
  {'op': 'global', 'field': 'damage'|'heal'|'pip_chance', 'schools': spec,
                   'percent': p}
  {'op': 'self_damage', 'amount': A}         unavoidable (Immolate/Dark Pact)
  {'op': 'gain_pips',   'n': k, 'tgt': 'self'|'ally'}
  {'op': 'stun',   'rounds': 1, 'tgt': 'enemy'|'enemies'}
  {'op': 'dispel', 'school': s, 'tgt': 'enemy'}
  {'op': 'remove', 'what': 'charm_pos'|'charm_neg'|'ward_pos'|'dot', 'tgt': ...}
  {'op': 'summon', 'minion': key}
  {'op': 'threat', 'mult': m, 'tgt': 'self'|'ally'}

'schools' spec: None = any school, 'own' = the card's school, or a list.
'group': damage/dot ops in the same group share ONE charm/ward consumption
(Fire Elf's 50 + DoT is one strike); different groups consume separately
(Minotaur's 50-hit eats shields/blades before the 445-hit — the community-
documented shield-breaker behavior).

Every curated entry carries a CONFIDENCE tag, per the annotation rule
"do not silently convert unknowns into zeros":
  'community' — stated in the scraped wiki description text itself
  'approx'    — well-known community value entered from memory
  'inferred'  — placeholder so the mechanic exists at all (minion blocks)
Cards whose mechanics we cannot express are listed in UNSUPPORTED and are
excluded by load_cards() rather than silently mis-modeled.
"""

ELEM = ("fire", "ice", "storm")
SPIRIT = ("life", "myth", "death")

# which schools a multi-school buff applies to (None = any school)
BUFF_SCHOOLS = {
    "Elemental Blade": set(ELEM), "Elemental Trap": set(ELEM),
    "Spirit Blade": set(SPIRIT), "Spirit Trap": set(SPIRIT),
    "Hex": None, "Feint": None, "Curse": None, "Weakness": None,
    "Plague": None, "Balanceblade": None, "Bladestorm": None,
    "Dark Pact": None,
}

# ---------------------------------------------------------------- curated ops

CARD_OPS = {
    # ---- fire
    "Fire Elf":    [{'op': 'hit', 'amount': 50, 'group': 0},
                    {'op': 'dot', 'total': 210, 'rounds': 3, 'group': 0}],
    "Fire Prism":  [{'op': 'prism', 'from': 'fire', 'to': 'ice',
                     'tgt': 'enemy'}],
    "Mass Fire Prism": [{'op': 'prism', 'from': 'fire', 'to': 'ice',
                         'tgt': 'enemies'}],
    "Quench":      [{'op': 'dispel', 'school': 'fire', 'tgt': 'enemy'}],
    "Steal Charm": [{'op': 'steal', 'what': 'charm_pos', 'tgt': 'enemy'}],
    "Tranquilize": [{'op': 'threat', 'mult': 0.5, 'tgt': 'self'}],
    "Mega Tranquilize": [{'op': 'threat', 'mult': 0.25, 'tgt': 'self'}],
    "Choke":       [{'op': 'stun', 'rounds': 1, 'tgt': 'enemies'}],
    "Fire Elemental": [{'op': 'summon', 'minion': 'fire_elemental'}],
    "Link":        [{'op': 'dot', 'total': 180, 'rounds': 3, 'group': 0},
                    {'op': 'hot', 'total': 120, 'rounds': 3, 'tgt': 'self'}],
    "Heck Hound":  [{'op': 'dot', 'total': 130, 'rounds': 3, 'per_pip': True,
                     'group': 0}],
    "Immolate":    [{'op': 'hit', 'amount': 600, 'group': 0},
                    {'op': 'self_damage', 'amount': 250}],
    "Fire Dragon": [{'op': 'hit', 'amount': 440, 'tgt': 'enemies', 'group': 0},
                    {'op': 'dot', 'total': 351, 'rounds': 3, 'tgt': 'enemies',
                     'group': 0}],
    "Fuel":        [{'op': 'ward', 'percent': 0.25, 'schools': ['fire'],
                     'charges': 3, 'tgt': 'enemy'}],

    # ---- ice
    "Frostbite":   [{'op': 'hit', 'amount': 50, 'group': 0},
                    {'op': 'dot', 'total': 450, 'rounds': 3, 'group': 0}],
    "Frost Giant": [{'op': 'hit', 'amount': 475, 'tgt': 'enemies', 'group': 0},
                    {'op': 'stun', 'rounds': 1, 'tgt': 'enemies'}],
    "Ice Prism":   [{'op': 'prism', 'from': 'ice', 'to': 'fire',
                     'tgt': 'enemy'}],
    "Mass ice prism": [{'op': 'prism', 'from': 'ice', 'to': 'fire',
                        'tgt': 'enemies'}],
    "Tower Shield": [{'op': 'ward', 'percent': -0.50, 'schools': None,
                      'tgt': 'self'}],
    "Ice Armor":   [{'op': 'absorb', 'amount': 125, 'per_pip': True,
                     'tgt': 'self'}],
    "Stun Block":  [{'op': 'stun_block', 'n': 1, 'tgt': 'self'}],
    "Melt":        [{'op': 'dispel', 'school': 'ice', 'tgt': 'enemy'}],
    "Steal Ward":  [{'op': 'steal', 'what': 'ward_pos', 'tgt': 'enemy'}],
    "Freeze":      [{'op': 'stun', 'rounds': 1, 'tgt': 'enemy'}],
    "Taunt":       [{'op': 'threat', 'add': 1000, 'tgt': 'self'}],
    "Distract":    [{'op': 'threat', 'add': 1000, 'tgt': 'ally'}],
    "Ice Guardian": [{'op': 'summon', 'minion': 'ice_guardian'}],

    # ---- storm
    "Lightning Elf": [{'op': 'hit', 'amount': 50, 'group': 0},
                      {'op': 'dot', 'total': 210, 'rounds': 3, 'group': 0}],
    "Kraken":      [{'op': 'hit', 'amount': 550, 'group': 0}],   # 520-580
    "Windstorm":   [{'op': 'ward', 'percent': 0.20, 'schools': ['storm'],
                     'tgt': 'enemies'}],
    "Storm Prism": [{'op': 'prism', 'from': 'storm', 'to': 'myth',
                     'tgt': 'enemy'}],
    "Lightning Strike": [{'op': 'charm', 'percent': 0.10, 'kind': 'accuracy',
                          'schools': ['storm'], 'tgt': 'self'}],
    "Dissipate":   [{'op': 'dispel', 'school': 'storm', 'tgt': 'enemy'}],
    "Soothe":      [{'op': 'threat', 'mult': 0.5, 'tgt': 'self'}],
    "Water Elemental": [{'op': 'summon', 'minion': 'water_elemental'}],
    "Disarm":      [{'op': 'remove', 'what': 'charm_pos', 'tgt': 'enemy'}],
    "Cleanse Charm": [{'op': 'remove', 'what': 'charm_neg', 'tgt': 'self'}],

    # ---- myth
    "Minotaur":    [{'op': 'hit', 'amount': 50, 'group': 0},
                    {'op': 'hit', 'amount': 445, 'group': 1}],
    "Stun":        [{'op': 'stun', 'rounds': 1, 'tgt': 'enemy'}],
    "Blinding Light": [{'op': 'stun', 'rounds': 1, 'tgt': 'enemies'}],
    "Pierce":      [{'op': 'remove', 'what': 'ward_pos', 'tgt': 'enemy'}],
    "Cleanse Ward": [{'op': 'remove', 'what': 'ward_neg', 'tgt': 'self'}],
    "Myth Prism":  [{'op': 'prism', 'from': 'myth', 'to': 'storm',
                     'tgt': 'enemy'}],
    "Vaporize":    [{'op': 'dispel', 'school': 'myth', 'tgt': 'enemy'}],
    "Subdue":      [{'op': 'threat', 'mult': 0.5, 'tgt': 'self'}],
    "Mend Minion": [{'op': 'heal', 'amount': 350, 'tgt': 'minion'}],
    "Draw Power":  [{'op': 'sacrifice_minion', 'pips': 2}],
    "Draw Health": [{'op': 'sacrifice_minion', 'heal': 350}],
    "Drain Health": [{'op': 'sacrifice_minion', 'heal': 450}],
    "Troll Minion": [{'op': 'summon', 'minion': 'troll'}],
    "Cyclops Minion": [{'op': 'summon', 'minion': 'cyclops_minion'}],
    "Minotaur Minion": [{'op': 'summon', 'minion': 'minotaur_minion'}],
    "Golem Minion": [{'op': 'summon', 'minion': 'golem'}],

    # ---- death
    "Ghoul":       [{'op': 'drain', 'amount': 160, 'fraction': 0.5}],
    "Vampire":     [{'op': 'drain', 'amount': 350, 'fraction': 0.5}],
    "Wraith":      [{'op': 'drain', 'amount': 500, 'fraction': 0.5}],
    "Scarecrow":   [{'op': 'drain', 'amount': 400, 'fraction': 0.5,
                     'tgt': 'enemies'}],                # wiki aoe flag is wrong
    "Feint":       [{'op': 'ward', 'percent': 0.70, 'schools': None,
                     'tgt': 'enemy'},
                    {'op': 'ward', 'percent': 0.30, 'schools': None,
                     'tgt': 'self'}],
    "Dark Pact":   [{'op': 'self_damage', 'amount': 300},
                    {'op': 'charm', 'percent': 0.30, 'schools': None,
                     'tgt': 'self'}],
    "Empower":     [{'op': 'self_damage', 'amount': 500},
                    {'op': 'gain_pips', 'n': 3, 'tgt': 'self'}],
    "Sacrifice":   [{'op': 'self_damage', 'amount': 250},
                    {'op': 'heal', 'amount': 700, 'tgt': 'ally'}],
    "Death Prism": [{'op': 'prism', 'from': 'death', 'to': 'life',
                     'tgt': 'enemy'}],
    "Strangle":    [{'op': 'dispel', 'school': 'death', 'tgt': 'enemy'}],
    "Infection":   [{'op': 'charm', 'percent': -0.50, 'kind': 'heal',
                     'schools': None, 'tgt': 'enemy'}],
    "Plague":      [{'op': 'charm', 'percent': -0.20, 'schools': None,
                     'tgt': 'enemies'}],
    "Animate":     [{'op': 'summon', 'minion': 'animate'}],
    "Pacify":      [{'op': 'threat', 'mult': 0.5, 'tgt': 'self'}],
    "Doom and Gloom": [{'op': 'global', 'field': 'heal', 'schools': None,
                        'percent': -0.65}],

    # ---- life
    "Minor Blessing": [{'op': 'heal', 'amount': 65, 'tgt': 'ally'}],
    "Fairy":       [{'op': 'heal', 'amount': 420, 'tgt': 'ally'}],
    "Pixie":       [{'op': 'heal', 'amount': 400, 'tgt': 'self'}],
    "Sprite":      [{'op': 'heal', 'amount': 30, 'tgt': 'ally'},
                    {'op': 'hot', 'total': 270, 'rounds': 3, 'tgt': 'ally'}],
    "Regenerate":  [{'op': 'heal', 'amount': 52, 'tgt': 'ally'},
                    {'op': 'hot', 'total': 1101, 'rounds': 3, 'tgt': 'ally'}],
    "Unicorn":     [{'op': 'heal', 'amount': 275, 'tgt': 'allies'}],
    "Satyr":       [{'op': 'heal', 'amount': 860, 'tgt': 'ally'}],
    "Dryad":       [{'op': 'heal', 'amount': 200, 'per_pip': True,
                     'tgt': 'ally'}],
    "Rebirth":     [{'op': 'heal', 'amount': 650, 'tgt': 'allies'},
                    {'op': 'absorb', 'amount': 400, 'tgt': 'allies'}],
    "Helping Hands": [{'op': 'hot', 'total': 540, 'rounds': 3, 'tgt': 'ally'}],
    "Sanctuary":   [{'op': 'global', 'field': 'heal', 'schools': None,
                     'percent': 0.50}],
    "Guiding Light": [{'op': 'charm', 'percent': 0.30, 'kind': 'heal',
                       'schools': None, 'tgt': 'self'}],
    "Guidance":    [{'op': 'charm', 'percent': 0.10, 'kind': 'accuracy',
                     'schools': None, 'tgt': 'allies'}],
    "Precision":   [{'op': 'charm', 'percent': 0.10, 'kind': 'accuracy',
                     'schools': None, 'tgt': 'self'}],
    "Black Mantle": [{'op': 'charm', 'percent': -0.45, 'kind': 'accuracy',
                      'schools': None, 'tgt': 'enemy'}],
    "Entangle":    [{'op': 'dispel', 'school': 'life', 'tgt': 'enemy'}],
    "Spirit Armor": [{'op': 'absorb', 'amount': 400, 'tgt': 'ally'}],
    "Triage":      [{'op': 'remove', 'what': 'dot', 'tgt': 'ally'}],
    "Life Prism":  [{'op': 'prism', 'from': 'life', 'to': 'death',
                     'tgt': 'enemy'}],
    "Sprite Guardian": [{'op': 'summon', 'minion': 'sprite_guardian'}],
    "Calm":        [{'op': 'threat', 'mult': 0.5, 'tgt': 'self'}],

    # ---- balance
    "Judgement":   [{'op': 'hit', 'amount': 100, 'per_pip': True, 'group': 0}],
    "Hydra":       [{'op': 'hit', 'amount': 190, 'school': 'fire', 'group': 0},
                    {'op': 'hit', 'amount': 190, 'school': 'ice', 'group': 1},
                    {'op': 'hit', 'amount': 190, 'school': 'storm',
                     'group': 2}],
    "Power Nova":  [{'op': 'hit', 'amount': 470, 'tgt': 'enemies', 'group': 0},
                    {'op': 'charm', 'percent': -0.25, 'schools': None,
                     'tgt': 'enemies'}],
    "Weakness":    [{'op': 'charm', 'percent': -0.25, 'schools': None,
                     'tgt': 'enemy'}],
    "Bladestorm":  [{'op': 'charm', 'percent': 0.20, 'schools': None,
                     'tgt': 'allies'}],
    "Donate Power": [{'op': 'gain_pips', 'n': 2, 'tgt': 'ally'}],
    "Power Play":  [{'op': 'global', 'field': 'pip_chance', 'schools': None,
                     'percent': 0.35}],
    "Mander Minion": [{'op': 'summon', 'minion': 'mander'}],
    "Spectral Minion": [{'op': 'summon', 'minion': 'spectral'}],
}

# multi-school SHIELDS (school coverage isn't recoverable from the flat
# scrape's single `school` column) — confidence: 'community' (description)
SHIELD_SCHOOLS = {
    "Tower Shield": None,
    "Elemental Shield": list(ELEM), "Spirit Shield": list(SPIRIT),
    "Dream Shield": ["life", "myth"], "Legend Shield": ["death", "myth"],
    "Ether Shield": ["life", "death"], "Thermic Shield": ["ice", "fire"],
    "Volcanic Shield": ["storm", "fire"], "Glacial Shield": ["storm", "ice"],
}

# headline damage totals for DP/heuristic valuation where the scrape's
# avg_damage misses DoT portions / extra components (engine uses ops)
DAMAGE_TOTALS = {
    "Fire Elf": 260, "Lightning Elf": 260, "Link": 180, "Fire Dragon": 791,
    "Hydra": 570, "Power Nova": 470, "Kraken": 550, "Frost Giant": 475,
}

CONFIDENCE = {name: "community" for name in CARD_OPS}
CONFIDENCE.update({"Tempest": "approx", "Kraken": "community"})

# cards present in the scrape whose mechanics the engine cannot express yet;
# excluded from load rather than silently mis-modeled
UNSUPPORTED = {"Beguile"}

# cards missing from the scrape, added from memory (confidence: 'approx')
EXTRA_CARDS = [
    {"name": "Tempest", "school": "storm", "pips": "X", "accuracy": 70,
     "type": "damage", "aoe": True, "avg_damage": 80, "percent": None,
     "description": "80 storm damage per pip to all enemies."},
]
EXTRA_OPS = {
    "Tempest": [{'op': 'hit', 'amount': 80, 'per_pip': True,
                 'tgt': 'enemies', 'group': 0}],
}
CARD_OPS.update(EXTRA_OPS)

# ---------------------------------------------------------------- minions
# Placeholder stat blocks so the summon mechanic exists — confidence:
# 'inferred'. Replace with creature-page scrapes when available.
MINIONS = {
    "water_elemental": dict(school="storm", hp=125, role="attack", power=70),
    "animate":         dict(school="death", hp=300, role="attack", power=60),
    "mander":          dict(school="balance", hp=300, role="attack", power=45),
    "spectral":        dict(school="fire", hp=350, role="attack", power=60),
    "sprite_guardian": dict(school="life", hp=300, role="heal", power=120),
    "fire_elemental":  dict(school="fire", hp=200, role="attack", power=55),
    "ice_guardian":    dict(school="ice", hp=300, role="attack", power=45),
    "troll":           dict(school="myth", hp=150, role="attack", power=50),
    "cyclops_minion":  dict(school="myth", hp=350, role="attack", power=60),
    "minotaur_minion": dict(school="myth", hp=450, role="attack", power=70),
    "golem":           dict(school="myth", hp=75, role="attack", power=30),
}
