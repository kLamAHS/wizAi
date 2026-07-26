"""
Wizard101 PvE combat simulator — v0.2
Objective: expected turns-to-kill (TTK) a boss; substrate for DP + RL.

Changes from v0.1:
  - SAME-NAME CHARMS/WARDS DO NOT STACK (game rule). A second identical
    pending Fireblade is uncastable; buff diversity (Fireblade + Elemental
    Blade + Fuel + Fire Trap + Feint...) is how real one-shots are built.
  - Real pip model: 7 pip slots, each normal (worth 1) or power (worth 2 for
    same-school spells, 1 for off-school). Off-school utility (Elemental
    Blade/Trap, Feint...) now costs what it really costs.
  - Multi-charge wards (Fuel = +25% to next 3 fire hits) and Feint's +30%
    incoming backlash on the caster.
  - Self-damage spells (Immolate = 600 out / 250 to self).
  - Bosses have school, resist, and boost; damage is adjusted by the
    defender's resist/boost table (bosses resist own school, take extra
    from the opposing school).

Still simplified: no criticals, no shields/heals for the player, 1v1,
no minions, X-pip spells skipped.
"""
import json, random
from dataclasses import dataclass, field

OPPOSING = {"fire": "ice", "ice": "fire", "storm": "myth", "myth": "storm",
            "life": "death", "death": "life", "balance": None}

# corrections / enrichments the flat scrape can't express
OVERRIDES = {
    "Immolate": {"damage": 600.0, "self_damage": 250.0},
    "Fuel":     {"charges": 3},
    "Feint":    {"backlash": 0.30},
}
# which schools a multi-school buff applies to (from card descriptions)
BUFF_SCHOOLS = {
    "Elemental Blade": {"fire", "ice", "storm"},
    "Elemental Trap":  {"fire", "ice", "storm"},
    "Spirit Blade":    {"life", "myth", "death"},
    "Spirit Trap":     {"life", "myth", "death"},
    "Hex": None, "Feint": None, "Curse": None,          # None = any school
    "Balanceblade": None, "Bladestorm": None, "Dark Pact": None,
}

# ---------------------------------------------------------------- card model

@dataclass
class Card:
    name: str
    school: str
    pips: int
    accuracy: float            # 0..1
    kind: str                  # 'damage' | 'blade' | 'trap'
    damage: float = 0.0
    percent: float = 0.0
    charges: int = 1           # ward charges (Fuel = 3)
    self_damage: float = 0.0
    backlash: float = 0.0      # Feint: +incoming on caster
    applies_to: object = "own" # 'own' | set of schools | None (any)

def load_cards(path):
    raw = json.load(open(path))
    cards = {}
    for r in raw:
        kind = None
        if r["type"] == "damage" and r["avg_damage"]:
            kind = "damage"
        elif r["type"] == "charm" and (r["percent"] or 0) > 0:
            kind = "blade"
        elif r["type"] == "ward" and (r["percent"] or 0) > 0:
            kind = "trap"
        if not kind:
            continue
        try:
            pip_cost = int(r["pips"])
        except (TypeError, ValueError):
            continue                                   # X-pip spells skipped
        ov = OVERRIDES.get(r["name"], {})
        cards[r["name"]] = Card(
            name=r["name"], school=r["school"], pips=pip_cost,
            accuracy=(r["accuracy"] or 100) / 100, kind=kind,
            damage=ov.get("damage", r["avg_damage"] or 0.0),
            percent=(r["percent"] or 0) / 100,
            charges=ov.get("charges", 1),
            self_damage=ov.get("self_damage", 0.0),
            backlash=ov.get("backlash", 0.0),
            applies_to=BUFF_SCHOOLS.get(r["name"], "own"),
        )
    return cards

def buff_applies(card, damage_school):
    if card.applies_to == "own":
        return card.school == damage_school
    if card.applies_to is None:
        return True
    return damage_school in card.applies_to

# ---------------------------------------------------------------- bosses

@dataclass
class Boss:
    name: str
    hp: int
    school: str
    dmg: int                    # flat hit per round
    resist_own: float = 0.40    # damage reduction vs own school
    boost_opp: float = 0.25     # extra damage from opposing school

    def incoming_mult(self, spell_school):
        if spell_school == self.school:
            return 1 - self.resist_own
        if OPPOSING.get(self.school) == spell_school:
            return 1 + self.boost_opp
        return 1.0

# ---------------------------------------------------------------- game state

@dataclass
class State:
    boss_hp: float
    player_hp: float
    norm_pips: int = 0
    pow_pips: int = 0
    blades: dict = field(default_factory=dict)   # name -> Card (pending)
    traps: dict = field(default_factory=dict)    # name -> [Card, charges_left]
    self_vuln: float = 0.0                       # Feint backlash on player
    hand: list = field(default_factory=list)
    deck: list = field(default_factory=list)
    turn: int = 0

    @property
    def pip_slots(self):
        return self.norm_pips + self.pow_pips

class Sim:
    def __init__(self, cards, decklist, school, boss, power_pip=0.85,
                 player_hp=3000, rng=None):
        self.cards, self.school, self.pp = cards, school, power_pip
        self.decklist, self.boss = decklist, boss
        self.player_hp0 = player_hp
        self.rng = rng or random.Random()

    # ---- pips ------------------------------------------------------------
    def afford(self, s, card):
        if card.school == self.school:
            return s.norm_pips + 2 * s.pow_pips >= card.pips
        return s.norm_pips + s.pow_pips >= card.pips

    def spend(self, s, card):
        c = card.pips
        if card.school == self.school:
            use_pow = min(s.pow_pips, c // 2)
            c -= 2 * use_pow
            s.pow_pips -= use_pow
            if c > 0:
                if s.norm_pips >= c:
                    s.norm_pips -= c
                else:                       # odd remainder, only power left
                    s.pow_pips -= 1
        else:                               # power pips worth 1 off-school
            use_norm = min(s.norm_pips, c)
            s.norm_pips -= use_norm
            s.pow_pips -= (c - use_norm)

    def gain_pip(self, s):
        if s.pip_slots >= 7:
            return
        if self.rng.random() < self.pp:
            s.pow_pips += 1
        else:
            s.norm_pips += 1

    # ---- casting ---------------------------------------------------------
    def can_cast(self, s, card):
        if card not in s.hand or not self.afford(s, card):
            return False
        if card.kind == "blade" and card.name in s.blades:
            return False                    # same-name charms don't stack
        if card.kind == "trap" and card.name in s.traps:
            return False
        return True

    def cast(self, s, card):
        """Returns damage dealt. Fizzle keeps card and pips (game rule)."""
        if self.rng.random() > card.accuracy:
            return 0.0
        s.hand.remove(card)
        self.spend(s, card)
        if card.kind == "blade":
            s.blades[card.name] = card
        elif card.kind == "trap":
            s.traps[card.name] = [card, card.charges]
            if card.backlash:
                s.self_vuln = card.backlash
        elif card.kind == "damage":
            mult = 1.0
            for b in list(s.blades.values()):
                if buff_applies(b, card.school):
                    mult *= 1 + b.percent
                    del s.blades[b.name]
            for name, (t, ch) in list(s.traps.items()):
                if buff_applies(t, card.school):
                    mult *= 1 + t.percent
                    if ch <= 1:
                        del s.traps[name]
                    else:
                        s.traps[name][1] = ch - 1
            dmg = card.damage * mult * self.boss.incoming_mult(card.school)
            s.boss_hp -= dmg
            if card.self_damage:
                s.player_hp -= card.self_damage
            return dmg
        return 0.0

    # ---- round loop ------------------------------------------------------
    def new_state(self):
        deck = [self.cards[n] for n in self.decklist]
        self.rng.shuffle(deck)
        s = State(float(self.boss.hp), float(self.player_hp0), deck=deck)
        self.gain_pip(s)
        self.draw(s)
        return s

    def draw(self, s):
        while len(s.hand) < 7 and s.deck:
            s.hand.append(s.deck.pop())

    def end_round(self, s):
        s.turn += 1
        self.gain_pip(s)
        if s.boss_hp > 0 and self.boss.dmg:
            s.player_hp -= self.boss.dmg * (1 + s.self_vuln)
            s.self_vuln = 0.0
        self.draw(s)

    def run(self, policy, max_turns=40):
        s = self.new_state()
        while s.turn < max_turns:
            card = policy(self, s)
            if card is not None and self.can_cast(s, card):
                self.cast(s, card)
            if s.boss_hp <= 0:
                return s.turn + 1, True, s.player_hp
            self.end_round(s)
            if s.player_hp <= 0:
                return s.turn, False, 0
        return max_turns, False, s.player_hp

# ---------------------------------------------------------------- strategies

def effective_pips(sim, s, card):
    return s.norm_pips + s.pow_pips * (2 if card.school == sim.school else 1)

def castable(sim, s, kind):
    return [c for c in s.hand if c.kind == kind and sim.can_cast(s, c)]

def strat_nuke_asap(sim, s):
    opts = castable(sim, s, "damage")
    return max(opts, key=lambda c: c.damage, default=None)

def make_blade_stack(n_buffs):
    """Stack n distinct buffs, then fire the biggest nuke once affordable."""
    def strat(sim, s):
        pend = len(s.blades) + len(s.traps)
        if pend < n_buffs:
            b = castable(sim, s, "blade") or castable(sim, s, "trap")
            if b:
                return max(b, key=lambda c: c.percent)
        nukes = [c for c in s.hand if c.kind == "damage"]
        if nukes:
            big = max(nukes, key=lambda c: c.damage)
            if sim.afford(s, big):
                return big
            b = castable(sim, s, "blade") or castable(sim, s, "trap")
            return max(b, key=lambda c: c.percent) if b else None
        return None
    return strat

def evaluate(sim, policy, n=20000, max_turns=40):
    ttk, wins = [], 0
    for _ in range(n):
        t, won, _ = sim.run(policy, max_turns=max_turns)
        if won:
            wins += 1
            ttk.append(t)
    return wins / n, (sum(ttk) / len(ttk) if ttk else float("nan"))

if __name__ == "__main__":
    cards = load_cards("cards_clean.json")
    from bosses import BOSSES, DECKS
    boss = BOSSES["Jade Oni"]
    deck = DECKS["fire"]["oneshot"]
    sim = Sim(cards, deck, school="fire", boss=Boss(boss.name, boss.hp, boss.school, 0),
              player_hp=10**9)
    print(f"{'strategy':<22}{'kill%':>8}{'mean TTK':>10}")
    strats = [("nuke ASAP", strat_nuke_asap)] + \
             [(f"buff x{n} -> nuke", make_blade_stack(n)) for n in (1, 2, 3, 4, 5)]
    for name, pol in strats:
        w, m = evaluate(sim, pol, n=10000)
        print(f"{name:<22}{w*100:>7.1f}%{m:>10.2f}")
