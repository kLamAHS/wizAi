"""
Wizard101 PvE combat simulator — v0.3
Objective: turns-to-kill / survival substrate for DP + RL.

v0.3 rebuilds the engine around the offline-ML design notes ("factored
simulator, symbolic state, effect provenance"). Changes from v0.2:

  - HANGING EFFECTS ARE STRUCTURED OBJECTS with provenance. Every charm/
    ward carries (name, source) as its stack key: a deck Fireblade, a
    treasure-card Fireblade ("Fireblade@tc") and a Sharpened Fireblade all
    stack with each other, while two copies of the same one stay illegal.
    Wards process FIFO in placement order with a running damage school, so
    trap -> prism -> shield ordering resolves exactly like the game.
  - FULL EFFECT TAXONOMY: shields (single/dual/universal), prisms (school
    conversion; the answer to same-school walls), weaknesses, mantles &
    accuracy charms, heal charms, dispels, absorbs, DoTs/HoTs as scheduled
    multi-round effects (modifiers locked at cast), drains (heal-modifier
    immune), X-pip spells, multi-hit components (Minotaur's 50-hit eats
    shields/blades before the 445-hit), AoE with per-target resolution,
    stuns + stun blocks, pip gifts, threat modifiers, minion summons.
  - MULTI-ACTOR: 1 player + ally minions vs N enemies. Enemies pick
    targets by a simple threat model. Boss attacks go through the player's
    ward/resist pipeline, so shields and Feint backlash resolve properly.
  - CHEAT HOOKS: per-enemy scripted interrupt rules (after_player_cast /
    round_start / hp_below) casting free effect ops, per the "rules engine
    plus cheat scripts" design.
  - TREASURE CARDS: per-fight sideboard; draws are random; a TC drawn this
    round cannot be discarded this round.
  - RULES/VERSION OBJECT: era tag + toggles for version-sensitive rules
    (fizzle now discards the card — community-confirmed; v0.2 kept it),
    pluggable crit/block model (off in the classic era), pierce.

Card mechanics the flat scrape can't express live in content.py as ordered
effect primitives with confidence tags. Unsupported cards are excluded at
load, never silently mis-modeled.

Still simplified: no gear/pet stat layer (fields exist, default 0), no
archmastery/shadow/school pips (post-classic era), enemy decks are flat
scripted hits + cheats, no player-side team play (allies are minions only).
"""
import copy, json, random
from dataclasses import dataclass, field

import content

SCHOOLS = ("fire", "ice", "storm", "myth", "life", "death", "balance")
OPPOSING = {"fire": "ice", "ice": "fire", "storm": "myth", "myth": "storm",
            "life": "death", "death": "life", "balance": None}

# corrections / enrichments kept from v0.2 (numbers embedded in flavor text)
OVERRIDES = {
    "Immolate": {"damage": 600.0, "self_damage": 250.0},
    "Fuel":     {"charges": 3},
    "Feint":    {"backlash": 0.30},
}
BUFF_SCHOOLS = content.BUFF_SCHOOLS


# ---------------------------------------------------------------- rules

@dataclass
class Rules:
    """Version-sensitive knobs, tagged by a named ruleset. The classic era
    (Wizard City -> Dragonspyre, pre-Celestia) has no criticals and no
    off-school power pip mastery. Every results/trajectory artifact should
    carry `ruleset_id` so numbers from different rule versions never mix."""
    ruleset_id: str = "w101-pve-classic-0.3"
    era: str = "classic"
    hand_size: int = 7
    max_pip_slots: int = 7
    fizzle_discards_card: bool = True    # community-confirmed; v0.2 kept it
    crit_multiplier: float = 2.0         # used only when crit chances > 0
    crit_resolver: object = None         # optional callable(attacker,
                                         #   defender, rng, rules) -> mult;
                                         # swap per era instead of editing code
    stun_blocks: int = 4                 # blocks granted when a stun lands
    dot_ticks_use_cast_snapshot: bool = True   # modifiers locked at cast
    damage_ranges: bool = False          # sample ops carrying 'spread'
                                         # uniformly instead of using avg
    mastery_school: object = None        # mastery amulet: this off-school
                                         # school also gets FULL power-pip
                                         # value (post-Celestia item)


# ---------------------------------------------------------------- card model

@dataclass(eq=False)                    # identity semantics for hand/deck ops
class Card:
    name: str
    school: str
    pips: int
    accuracy: float            # 0..1
    kind: str                  # primary kind, see load_cards
    damage: float = 0.0        # headline number (DP/heuristics); ops drive sim
    percent: float = 0.0
    charges: int = 1
    self_damage: float = 0.0
    backlash: float = 0.0
    applies_to: object = "own" # 'own' | set of schools | None (any)
    ops: list = field(default_factory=list)
    aoe: bool = False
    x_pips: bool = False
    source: str = "deck"       # provenance tier: deck|tc|item|pet|enchant-*
    confidence: str = "community"

    @property
    def stack_key(self):
        return (self.name, self.source)


def _kind_from_ops(ops):
    """Primary kind of a curated card, by scanning its ops in priority
    order (drives DP/RL feature spaces & the scripted heuristics)."""
    kinds = [o["op"] for o in ops]
    if "hit" in kinds or "dot" in kinds:
        return "damage"
    if "drain" in kinds:
        return "drain"
    if "heal" in kinds or "hot" in kinds:
        return "heal"
    for o in ops:
        if o["op"] == "charm":
            ck = o.get("kind", "damage")
            if ck == "accuracy":
                return "mantle" if o["percent"] < 0 else "acc_charm"
            if ck == "heal":
                return "heal_charm"
            if o.get("tgt", "self") in ("self", "ally", "allies"):
                return "blade" if o["percent"] > 0 else "shield"
            return "weakness" if o["percent"] < 0 else "blade"
        if o["op"] == "ward":
            if o.get("tgt", "self") in ("enemy", "enemies"):
                return "trap" if o["percent"] > 0 else "shield"
            return "shield" if o["percent"] < 0 else "trap"
    first = {"prism": "prism", "absorb": "shield", "global": "global",
             "dispel": "dispel", "summon": "summon"}
    for o in ops:
        if o["op"] in first:
            return first[o["op"]]
    return "util"


def _default_ops(r, kind, aoe):
    """Generic ops for cards the curated table doesn't cover."""
    tgt_e = "enemies" if aoe else "enemy"
    name = r["name"]
    # buff coverage: BUFF_SCHOOLS overrides (None = any school), else own
    buff_schools = "own"
    if name in BUFF_SCHOOLS:
        cov = BUFF_SCHOOLS[name]
        buff_schools = None if cov is None else sorted(cov)
    if kind == "damage":
        return [{"op": "hit", "amount": r["avg_damage"], "tgt": tgt_e,
                 "group": 0}]
    if kind == "drain":
        return [{"op": "drain", "amount": r["avg_damage"], "fraction": 0.5,
                 "tgt": tgt_e}]
    if kind == "blade":
        tgt = "allies" if aoe else "self"
        return [{"op": "charm", "percent": (r["percent"] or 0) / 100,
                 "schools": buff_schools, "tgt": tgt}]
    if kind == "weakness":
        return [{"op": "charm", "percent": (r["percent"] or 0) / 100,
                 "schools": None, "tgt": tgt_e}]
    if kind == "trap":
        return [{"op": "ward", "percent": (r["percent"] or 0) / 100,
                 "schools": buff_schools, "tgt": tgt_e}]
    if kind == "shield":
        schools = content.SHIELD_SCHOOLS.get(name, "own")
        return [{"op": "ward", "percent": (r["percent"] or 0) / 100,
                 "schools": schools, "tgt": "self"}]
    if kind == "global":
        return [{"op": "global", "field": "damage", "schools": "own",
                 "percent": (r["percent"] or 0) / 100}]
    return []


def load_cards(path, report=None):
    """Build the card table: base fields from the scrape, mechanics from
    content.CARD_OPS where curated, generic ops otherwise. Cards that can't
    be expressed are collected in `report['skipped']` instead of loaded."""
    raw = json.load(open(path, encoding="utf-8")) + content.EXTRA_CARDS
    cards, skipped = {}, []
    for r in raw:
        name = r["name"]
        if name in content.UNSUPPORTED:
            skipped.append((name, "unsupported mechanic"))
            continue
        pct = r["percent"] or 0
        curated = content.CARD_OPS.get(name)
        # ---- primary kind (drives DP/RL feature spaces & heuristics)
        if curated:
            kind = _kind_from_ops(curated)
        elif r["type"] == "damage" and r["avg_damage"]:
            kind = "damage"
        elif r["type"] == "steal" and r["avg_damage"]:
            kind = "drain"
        elif r["type"] == "charm" and pct > 0:
            kind = "blade"
        elif r["type"] == "charm" and pct < 0:
            kind = "weakness"
        elif r["type"] == "ward" and pct > 0:
            kind = "trap"
        elif r["type"] == "ward" and pct < 0:
            kind = "shield"
        elif r["type"] == "global" and pct:
            kind = "global"
        else:
            kind = None
        if kind is None:
            skipped.append((name, f"no curated ops for {r['type']}"))
            continue
        # ---- pip cost
        x_pips = str(r["pips"]).upper() == "X"
        try:
            pip_cost = 0 if x_pips else int(r["pips"])
        except (TypeError, ValueError):
            skipped.append((name, "bad pip cost"))
            continue
        ov = OVERRIDES.get(name, {})
        headline = content.DAMAGE_TOTALS.get(name, r["avg_damage"] or 0.0)
        card = Card(
            name=name, school=r["school"], pips=pip_cost,
            accuracy=(r["accuracy"] or 100) / 100, kind=kind,
            damage=ov.get("damage", headline),
            percent=pct / 100,
            charges=ov.get("charges", 1),
            self_damage=ov.get("self_damage", 0.0),
            backlash=ov.get("backlash", 0.0),
            applies_to=BUFF_SCHOOLS.get(name, "own"),
            aoe=bool(r["aoe"]) or any(o.get("tgt") in ("enemies", "allies")
                                      for o in (curated or [])),
            x_pips=x_pips,
            confidence=content.CONFIDENCE.get(name, "community"),
        )
        card.ops = [dict(o) for o in (curated or
                                      _default_ops(r, kind, card.aoe))]
        assign_subkeys(card.ops)
        if not card.ops:
            skipped.append((name, "no ops"))
            continue
        cards[name] = card
    if report is not None:
        report["skipped"] = skipped
    return cards


def expand_variants(cards, decklist):
    """Materialize provenance variants named in a decklist: "Fireblade@tc"
    becomes a clone with source='tc' and its own stack key."""
    for entry in decklist:
        if entry in cards or "@" not in entry:
            continue
        base_name, src = entry.rsplit("@", 1)
        base = cards[base_name]
        clone = Card(**{f: getattr(base, f) for f in
                        ("school", "pips", "accuracy", "kind", "damage",
                         "percent", "charges", "self_damage", "backlash",
                         "applies_to", "aoe", "x_pips", "confidence")},
                     name=entry, ops=[dict(o) for o in base.ops])
        clone.source = src
        cards[entry] = clone
    return cards


def sharpened(cards, name, bonus=0.10):
    """Enchanted variant with its own stack key (Sharpened Blade / Potent
    Trap semantics — post-classic, provided for provenance experiments)."""
    base = cards[name]
    vname = f"Sharpened {name}" if base.kind == "blade" else f"Potent {name}"
    if vname in cards:
        return cards[vname]
    clone = Card(**{f: getattr(base, f) for f in
                    ("school", "pips", "accuracy", "kind", "damage",
                     "percent", "charges", "self_damage", "backlash",
                     "applies_to", "aoe", "x_pips")},
                 name=vname, ops=[dict(o) for o in base.ops],
                 source="enchant", confidence="community")
    clone.percent = base.percent + bonus
    for o in clone.ops:
        if o["op"] in ("charm", "ward") and o.get("kind", "damage") == "damage":
            o["percent"] = o["percent"] + bonus
    cards[vname] = clone
    return clone


def assign_subkeys(ops):
    """Give each hanging-effect op a distinct sub index so one card can
    place several same-named effects (Tri Blade's three charms) that each
    stack-check independently."""
    for i, o in enumerate(ops):
        if o["op"] in ("charm", "ward", "prism", "absorb", "dispel"):
            o.setdefault("sub", i)
    return ops


def buff_applies(card, damage_school):
    if card.applies_to == "own":
        return card.school == damage_school
    if card.applies_to is None:
        return True
    return damage_school in card.applies_to


# ---------------------------------------------------------------- effects

@dataclass(eq=False)                    # identity semantics for removal
class Hanging:
    """One hanging effect on an actor. slot 'charm' = outgoing (blades,
    weaknesses, mantles, heal charms, dispels), slot 'ward' = incoming
    (traps, shields, prisms, absorbs, Feint backlash)."""
    name: str
    slot: str                  # 'charm' | 'ward'
    kind: str                  # 'damage'|'accuracy'|'heal'|'dispel'|'prism'|'absorb'
    percent: float = 0.0
    schools: object = None     # None = any, else set of damage schools
    charges: int = 1
    source: str = "deck"
    protected: bool = False    # Aegis/Indemnity semantics: immune to removal
    convert_to: str = None     # prisms
    amount: float = 0.0        # absorb pool
    caster: str = ""
    sub: int = 0               # discriminates multiple hangings from one
                               # card (Tri Blade = three same-named charms)

    @property
    def stack_key(self):
        return (self.name, self.source, self.sub)

    def matches(self, school):
        return self.schools is None or school in self.schools


@dataclass
class OverTime:
    name: str
    kind: str                  # 'dot' | 'hot'
    school: str
    per_tick: float            # snapshot at cast: modifiers already applied
    rounds_left: int
    caster: str = ""


@dataclass
class Bubble:
    name: str
    fld: str                   # 'damage' | 'heal' | 'pip_chance'
    schools: object            # None = any
    percent: float = 0.0


# ---------------------------------------------------------------- cheats

@dataclass
class CheatRule:
    """Scripted interrupt: when `event` fires and `when` matches, the enemy
    executes `ops` for free (out of turn), like the game's boss cheats.
    events: 'after_player_cast' (when: card_kind / card_kinds), 'round_start'
    (when: {} or {'every': n}), 'hp_below' (when: {'frac': x}, fires once)."""
    event: str
    ops: list
    when: dict = field(default_factory=dict)
    message: str = ""
    once: bool = False
    cooldown: int = 0            # min rounds between firings (0 = none)
    fired: int = 0
    last_turn: int = -10**9

    def matches_cast(self, card):
        kinds = self.when.get("card_kinds") or \
            ([self.when["card_kind"]] if "card_kind" in self.when else None)
        return kinds is None or card.kind in kinds


@dataclass
class CheatScript:
    rules: list = field(default_factory=list)

    def fresh(self):
        return CheatScript([CheatRule(r.event, [dict(o) for o in r.ops],
                                      dict(r.when), r.message, r.once,
                                      r.cooldown)
                            for r in self.rules])


# ---------------------------------------------------------------- actors

@dataclass
class Actor:
    name: str
    school: str
    hp: float
    max_hp: float
    team: int                   # 0 = player side, 1 = enemy side
    norm_pips: int = 0
    pow_pips: int = 0
    power_pip_chance: float = 0.0
    accuracy_bonus: float = 0.0
    pierce: float = 0.0
    crit_chance: float = 0.0
    block_chance: float = 0.0
    damage_bonus: dict = field(default_factory=dict)   # school|'*' -> frac
    resist: dict = field(default_factory=dict)         # school|'*' -> frac
    boost: dict = field(default_factory=dict)          # school -> frac
    flat_damage: float = 0.0     # added after % modifiers, before resist
    flat_resist: float = 0.0     # subtracted last, floor at zero
    heal_out: float = 0.0
    heal_in: float = 0.0
    charms: list = field(default_factory=list)         # outgoing Hangings
    wards: list = field(default_factory=list)          # incoming Hangings
    over_time: list = field(default_factory=list)
    stunned: int = 0
    stun_blocks: int = 0
    stunable: object = None      # None = unspecified (treated as True)
    threat: float = 0.0
    flat_hit: float = 0.0        # scripted enemy auto-attack
    cheat: object = None
    is_minion: bool = False
    minion_role: str = "attack"
    spell_pool: object = None    # living-boss card names (None = legacy)
    archetype: str = "hitter"
    discipline: float = 0.7
    minion_power: float = 0.0
    hand: list = field(default_factory=list)
    deck: list = field(default_factory=list)
    sideboard: list = field(default_factory=list)
    graveyard: list = field(default_factory=list)
    fresh_tc: list = field(default_factory=list)   # TCs drawn this round

    @property
    def alive(self):
        return self.hp > 0

    @property
    def pip_slots(self):
        return self.norm_pips + self.pow_pips

    def has_stack(self, key, slot=None):
        pools = {"charm": self.charms, "ward": self.wards}
        pool = pools.get(slot, self.charms + self.wards)
        return any(h.stack_key == key for h in pool)


# ---------------------------------------------------------------- bosses

@dataclass
class Boss:
    name: str
    hp: int
    school: str
    dmg: int                    # flat hit per round (0 = immortal-player view)
    resist_own: float = 0.40    # damage reduction vs own school
    boost_opp: float = 0.25     # extra damage from opposing school
    cheat: object = None        # CheatScript
    stunable: object = None     # None = unspecified (research-doc convention)
    pierce: float = 0.0
    crit_chance: float = 0.0
    block_chance: float = 0.0
    resist_map: object = None   # scraped per-school/universal resist table;
                                # overrides the resist_own convention
    boost_map: object = None    # scraped boost table; overrides boost_opp
    outgoing_bonus: float = 0.0 # scraped outgoing damage % (gear-like)
    # ---- living-boss layer (2026 boss-AI report; ALL confidence
    # 'modeled': the report supports pool+state-aware-AI+scripts, not
    # the exact weights) -------------------------------------------------
    pool: object = None         # card names the boss may cast; None =
                                # legacy flat `dmg` per round
    archetype: str = "hitter"   # hitter|healer|buffer|debuffer|tank
    discipline: float = 0.7     # P(follow archetype priorities) vs a
                                # uniform legal pick — the report's
                                # "imperfect but state-aware" knob
    pip_chance: float = 0.40    # enemy power-pip odds (not public;
                                # modeled at the player base cap)

    def incoming_mult(self, spell_school):
        if self.resist_map is not None or self.boost_map is not None:
            res = (self.resist_map or {}).get(spell_school, 0.0) + \
                  (self.resist_map or {}).get("*", 0.0)
            return (1 - res) * (1 + (self.boost_map or {})
                                .get(spell_school, 0.0))
        if spell_school == self.school:
            return 1 - self.resist_own
        if OPPOSING.get(self.school) == spell_school:
            return 1 + self.boost_opp
        return 1.0

    def to_actor(self):
        if self.resist_map is not None or self.boost_map is not None:
            resist = dict(self.resist_map or {})
            boost = dict(self.boost_map or {})
        else:
            resist = {self.school: self.resist_own}
            boost = {}
            opp = OPPOSING.get(self.school)
            if opp:
                boost[opp] = self.boost_opp
        return Actor(name=self.name, school=self.school, hp=float(self.hp),
                     max_hp=float(self.hp), team=1, resist=resist,
                     boost=boost, flat_hit=float(self.dmg),
                     damage_bonus={"*": self.outgoing_bonus}
                     if self.outgoing_bonus else {},
                     cheat=self.cheat.fresh() if self.cheat else None,
                     stunable=self.stunable, pierce=self.pierce,
                     crit_chance=self.crit_chance,
                     block_chance=self.block_chance,
                     spell_pool=list(self.pool) if self.pool else None,
                     archetype=self.archetype,
                     discipline=self.discipline,
                     power_pip_chance=self.pip_chance
                     if self.pool else 0.0)


# ---------------------------------------------------------------- game state

class State:
    """Multi-actor combat state with v0.2-compatible views (boss_hp,
    player_hp, pips, blades, traps, hand, deck) so the DP transfer and the
    tabular RL featurizer keep working unchanged in 1v1."""

    def __init__(self, player, enemies, allies=None):
        self.player = player
        self.enemies = enemies
        self.allies = allies or []
        self.bubble = None
        self.turn = 0
        self.events = []           # structured audit log (Sim(log_events=True))
        self.tc_casts = 0          # TCs cost real gold: usage is audited

    # ---- compat views ------------------------------------------------
    @property
    def boss_hp(self):
        return self.enemies[0].hp

    @boss_hp.setter
    def boss_hp(self, v):
        self.enemies[0].hp = v

    @property
    def player_hp(self):
        return self.player.hp

    @player_hp.setter
    def player_hp(self, v):
        self.player.hp = v

    @property
    def norm_pips(self):
        return self.player.norm_pips

    @norm_pips.setter
    def norm_pips(self, v):
        self.player.norm_pips = v

    @property
    def pow_pips(self):
        return self.player.pow_pips

    @pow_pips.setter
    def pow_pips(self, v):
        self.player.pow_pips = v

    @property
    def pip_slots(self):
        return self.player.pip_slots

    @property
    def hand(self):
        return self.player.hand

    @property
    def deck(self):
        return self.player.deck

    @property
    def blades(self):
        """Positive damage charms on the player, keyed like v0.2 (deck-
        provenance cards keep their bare name)."""
        return {h.name: h for h in self.player.charms
                if h.kind == "damage" and h.percent > 0}

    @property
    def traps(self):
        """Wards sitting on the primary enemy (positive + prisms), keyed by
        name -> [hanging, charges_left]."""
        out = {}
        for h in self.enemies[0].wards:
            if h.kind == "prism" or (h.kind == "damage" and h.percent > 0):
                out[h.name] = [h, h.charges]
        return out

    @property
    def self_vuln(self):
        """v0.2 view of Feint backlash: total +incoming on the player."""
        mult = 1.0
        for h in self.player.wards:
            if h.kind == "damage" and h.percent > 0:
                mult *= 1 + h.percent
        return mult - 1.0

    def living_enemies(self):
        return [e for e in self.enemies if e.alive]


@dataclass
class Action:
    """Rich action: optional pre-cast hand edits, then an optional cast.
    Plain policies may still return a Card (cast at enemy 0) or None."""
    card: object = None
    target: int = 0            # index into state.enemies for enemy-targeted
    ally: int = -1             # index into allies; -1 = self/player
    discards: tuple = ()
    draw_tc: int = 0


# ---------------------------------------------------------------- simulator

class Sim:
    def __init__(self, cards, decklist, school, boss, power_pip=0.85,
                 player_hp=3000, rng=None, enemies=None, sideboard=None,
                 rules=None, player_stats=None, log_events=False):
        self.cards, self.school, self.pp = cards, school, power_pip
        expand_variants(self.cards, decklist)
        if sideboard:
            expand_variants(self.cards, sideboard)
        self.decklist, self.boss = decklist, boss
        self.extra_bosses = enemies or []
        for b in [boss] + self.extra_bosses:
            missing = [n for n in (b.pool or []) if n not in self.cards]
            if missing:
                raise ValueError(f"{b.name}: pool spells not in card "
                                 f"data: {missing}")
        self.sideboard_list = sideboard or []
        self.player_hp0 = player_hp
        self.player_stats = player_stats or {}
        self.rules = rules or Rules()
        self.rng = rng or random.Random()
        self.log_events = log_events

    def _ev(self, s, type_, **kw):
        """Append a structured event to the state's audit log (opt-in)."""
        if self.log_events:
            kw["type"] = type_
            kw["turn"] = s.turn
            s.events.append(kw)

    # ---- pips ------------------------------------------------------------
    def _full_pip(self, card):
        """Does this card get 2-for-1 power pips? Own school always;
        the mastery-amulet school too when the ruleset grants one."""
        return (card.school == self.school or
                card.school == self.rules.mastery_school)

    def _pip_value(self, actor, card):
        return 2 if self._full_pip(card) else 1

    def afford(self, s, card):
        a = s.player
        if card.x_pips:
            return a.pip_slots >= 1
        if self._full_pip(card):
            return a.norm_pips + 2 * a.pow_pips >= card.pips
        return a.norm_pips + a.pow_pips >= card.pips

    def spend(self, s, card):
        """Returns effective pips spent (drives X-pip spells)."""
        a = s.player
        if card.x_pips:
            eff = a.norm_pips + a.pow_pips * self._pip_value(a, card)
            a.norm_pips = a.pow_pips = 0
            return eff
        c = card.pips
        if self._full_pip(card):
            use_pow = min(a.pow_pips, c // 2)
            c -= 2 * use_pow
            a.pow_pips -= use_pow
            if c > 0:
                if a.norm_pips >= c:
                    a.norm_pips -= c
                else:                       # odd remainder, only power left
                    a.pow_pips -= 1
        else:                               # power pips worth 1 off-school
            use_norm = min(a.norm_pips, c)
            a.norm_pips -= use_norm
            a.pow_pips -= (c - use_norm)
        return card.pips

    def gain_pip(self, s):
        a = s.player
        if a.pip_slots >= self.rules.max_pip_slots:
            return
        chance = a.power_pip_chance
        if s.bubble and s.bubble.fld == "pip_chance":
            chance = min(1.0, chance + s.bubble.percent)
        if self.rng.random() < chance:
            a.pow_pips += 1
        else:
            a.norm_pips += 1

    # ---- legality ----------------------------------------------------------
    def can_cast(self, s, card, target=0):
        if card not in s.hand or not self.afford(s, card):
            return False
        enemies = s.living_enemies()
        if not enemies:
            return False
        tgt_enemy = s.enemies[target] if target < len(s.enemies) else None
        if tgt_enemy is not None and not tgt_enemy.alive:
            tgt_enemy = enemies[0]
        if any(op["op"] == "sacrifice_minion" for op in card.ops) and \
                not any(m.alive for m in s.allies):
            return False
        # duplicate-placement rule: a PURE hanging-effect card is uncastable
        # while its primary effect is still pending on the chosen target
        # (the in-game "you may not place the same effect twice" X). Cards
        # with damage/heal components always cast; dup riders just skip.
        HANGING = ("charm", "ward", "prism", "absorb", "dispel")
        if all(op["op"] in HANGING + ("self_damage",) for op in card.ops):
            op = next(op for op in card.ops if op["op"] in HANGING)
            key = (card.name, card.source, op.get("sub", 0))
            slot = "charm" if op["op"] in ("charm", "dispel") else "ward"
            tgt = op.get("tgt", "self")
            if tgt in ("self", "allies", "ally"):
                if s.player.has_stack(key, slot):
                    return False
            elif tgt == "enemy":
                if tgt_enemy is not None and tgt_enemy.has_stack(key, slot):
                    return False
            elif tgt == "enemies":
                if all(e.has_stack(key, slot) for e in enemies):
                    return False
        return True

    # ---- resolution core ---------------------------------------------------
    def _consume_damage_charms(self, s, caster, school):
        """All matching damage charms (blades AND weaknesses) apply to one
        strike and are consumed together, like the game."""
        mult = 1.0
        keep = []
        for h in caster.charms:
            if h.kind == "damage" and h.matches(school):
                mult *= 1 + h.percent
                self._ev(s, "charm_consumed", actor=caster.name,
                         effect=h.name, percent=h.percent)
            else:
                keep.append(h)
        caster.charms[:] = keep
        return mult

    def _consume_accuracy_charms(self, s, caster, school):
        delta = 0.0
        keep = []
        for h in caster.charms:
            if h.kind == "accuracy" and h.matches(school):
                delta += h.percent
                self._ev(s, "accuracy_charm_consumed", actor=caster.name,
                         effect=h.name, percent=h.percent)
            else:
                keep.append(h)
        caster.charms[:] = keep
        return delta

    def _consume_heal_charms(self, s, caster):
        mult = 1.0
        keep = []
        for h in caster.charms:
            if h.kind == "heal":
                mult *= 1 + h.percent
                self._ev(s, "heal_charm_consumed", actor=caster.name,
                         effect=h.name, percent=h.percent)
            else:
                keep.append(h)
        caster.charms[:] = keep
        return mult

    def _take_dispel(self, s, caster, school):
        for h in caster.charms:
            if h.kind == "dispel" and h.matches(school):
                caster.charms.remove(h)
                self._ev(s, "dispel_consumed", actor=caster.name,
                         effect=h.name)
                return True
        return False

    def _ward_pass(self, s, target, school, pierce):
        """FIFO over the target's wards with a running damage school:
        traps/shields consume if they match the CURRENT school, prisms
        convert it. Pierce eats resist first (handled by caller), remainder
        shaves shield magnitude. Returns (multiplier, final_school)."""
        mult = 1.0
        keep = []
        for h in target.wards:
            if h.kind == "prism":
                if h.schools is None or school in h.schools:
                    self._ev(s, "prism_converted", target=target.name,
                             effect=h.name, to=h.convert_to)
                    school = h.convert_to
                    continue                     # consumed
                keep.append(h)
            elif h.kind == "damage":
                if h.matches(school):
                    pct = h.percent
                    if pct < 0 and pierce > 0:
                        shaved = min(pierce, -pct)
                        pct += shaved
                        pierce -= shaved
                    mult *= 1 + pct
                    if h.charges > 1:
                        h.charges -= 1
                        keep.append(h)
                    self._ev(s, "ward_consumed", target=target.name,
                             effect=h.name, percent=pct,
                             charges_left=h.charges if h.charges > 1 else 0)
                else:
                    keep.append(h)
            else:
                keep.append(h)
        target.wards[:] = keep
        return mult, school, pierce

    def _absorb_pass(self, s, target, dmg):
        keep = []
        for h in target.wards:
            if h.kind == "absorb" and dmg > 0:
                soak = min(h.amount, dmg)
                dmg -= soak
                h.amount -= soak
                self._ev(s, "absorb_soaked", target=target.name,
                         effect=h.name, amount=soak,
                         pool_left=max(h.amount, 0.0))
                if h.amount > 0:
                    keep.append(h)
            else:
                keep.append(h)
        target.wards[:] = keep
        return dmg

    def _resist_mult(self, target, school, pierce):
        res = target.resist.get(school, 0.0) + target.resist.get("*", 0.0)
        res = max(res - pierce, 0.0) if res > 0 else res
        mult = 1 - res
        mult *= 1 + target.boost.get(school, 0.0)
        return mult

    def _crit_mult(self, caster, target):
        if self.rules.crit_resolver is not None:
            return self.rules.crit_resolver(caster, target, self.rng,
                                            self.rules)
        if caster.crit_chance <= 0 or self.rng.random() >= caster.crit_chance:
            return 1.0
        if target.block_chance > 0 and self.rng.random() < target.block_chance:
            return 1.0
        return self.rules.crit_multiplier

    def _damage_bonus(self, caster, school):
        return 1 + caster.damage_bonus.get(school, 0.0) + \
            caster.damage_bonus.get("*", 0.0)

    def _bubble_damage(self, s, school):
        b = s.bubble
        if b and b.fld == "damage" and (b.schools is None or school in b.schools):
            return 1 + b.percent
        return 1.0

    def _strike(self, s, caster, target, school, base, charm_mult, crit_mult,
                ward=None):
        """One damage component against one target. Returns damage dealt.
        `ward` carries a precomputed _ward_pass result so ops in the same
        link group share ONE ward consumption (Fire Elf's hit + DoT are a
        single strike; a shield reduces both)."""
        dmg = base * charm_mult
        dmg *= self._damage_bonus(caster, school)
        dmg *= self._bubble_damage(s, school)
        dmg *= crit_mult
        if ward is None:
            ward = self._ward_pass(s, target, school, caster.pierce)
        wmult, fschool, pierce = ward
        dmg *= wmult
        if dmg > 0 and caster.flat_damage:
            dmg += caster.flat_damage           # post-%: community-described
        dmg *= self._resist_mult(target, fschool, pierce)
        dmg -= target.flat_resist
        dmg = max(dmg, 0.0)
        dmg = self._absorb_pass(s, target, dmg)
        target.hp -= dmg
        caster.threat += dmg
        self._ev(s, "damage_applied", actor=caster.name, target=target.name,
                 school=fschool, amount=round(dmg, 1))
        if not target.alive:
            self._ev(s, "defeated", target=target.name)
        return dmg

    def _heal(self, s, caster, target, amount, charm_mult=1.0):
        amt = amount * (1 + caster.heal_out) * charm_mult
        if s.bubble and s.bubble.fld == "heal":
            amt *= 1 + s.bubble.percent
        amt *= 1 + target.heal_in
        amt = max(amt, 0.0)
        healed = min(amt, target.max_hp - target.hp)
        target.hp += healed
        caster.threat += 0.5 * healed
        self._ev(s, "heal_applied", actor=caster.name, target=target.name,
                 amount=round(healed, 1))
        return healed

    def _resolve_targets(self, s, caster, spec, target_idx, ally_idx):
        """Map an op target spec to actor lists, from `caster`'s side."""
        if caster.team == 0:
            foes = s.living_enemies()
            friends = [s.player] + [m for m in s.allies if m.alive]
            me = caster
            tgt_foe = None
            if foes:
                want = s.enemies[target_idx] if target_idx < len(s.enemies) else None
                tgt_foe = want if want in foes else foes[0]
            tgt_friend = me if ally_idx < 0 else \
                (s.allies[ally_idx] if ally_idx < len(s.allies) and
                 s.allies[ally_idx].alive else me)
        else:
            friends = [e for e in s.enemies if e.alive]
            pool = [s.player] + [m for m in s.allies if m.alive]
            tgt_foe = max(pool, key=lambda a: a.threat) if pool else None
            foes = pool
            me = caster
            # ally_idx >= 0 = "the neediest living teammate" (enemy
            # healers heal their protectee, not themselves — report's
            # role-minion pattern); < 0 keeps the caster
            tgt_friend = caster if ally_idx < 0 or not friends else \
                min(friends, key=lambda f: f.hp / max(f.max_hp, 1))
        minions = [m for m in (s.allies if caster.team == 0 else [])
                   if m.alive]
        return {"self": [me], "ally": [tgt_friend], "allies": friends,
                "enemy": [tgt_foe] if tgt_foe else [],
                "enemies": foes,
                "minion": minions[:1]}.get(spec, [me])

    def _hanging_from_op(self, op, card_name, source, caster_name):
        o = op["op"]
        sub = op.get("sub", 0)
        if o == "charm":
            return Hanging(card_name, "charm", op.get("kind", "damage"),
                           percent=op["percent"], schools=op.get("_schools"),
                           source=source, caster=caster_name, sub=sub)
        if o == "ward":
            return Hanging(card_name, "ward", "damage",
                           percent=op["percent"], schools=op.get("_schools"),
                           charges=op.get("charges", 1), source=source,
                           caster=caster_name, sub=sub)
        if o == "prism":
            return Hanging(card_name, "ward", "prism",
                           schools={op["from"]}, convert_to=op["to"],
                           source=source, caster=caster_name, sub=sub)
        if o == "absorb":
            return Hanging(card_name, "ward", "absorb", amount=op["amount"],
                           source=source, caster=caster_name, sub=sub)
        if o == "dispel":
            return Hanging(card_name, "charm", "dispel",
                           schools={op["school"]}, source=source,
                           caster=caster_name, sub=sub)
        raise ValueError(o)

    def execute_ops(self, s, caster, card_name, source, ops, school,
                    pips_spent=0, accuracy_school=None, target_idx=0,
                    ally_idx=-1, crit_roll=True):
        """Run a card's op list. Damage/dot ops in the same `group` share one
        charm/ward consumption; separate groups consume separately (multi-
        hit). Returns total damage dealt to enemies."""
        total = 0.0
        group_state = {}        # group id -> (charm_mult, crit_mult)
        ward_state = {}         # (group id, target id) -> ward pass result

        def wards_for(g, t, school):
            k = (g, id(t))
            if k not in ward_state:
                ward_state[k] = self._ward_pass(s, t, school, caster.pierce)
            return ward_state[k]

        for op in ops:
            o = dict(op)
            kind = o["op"]
            spec_school = o.get("school") or school
            tgt_spec = o.get("tgt", "enemy" if kind in
                             ("hit", "dot", "drain", "stun", "dispel")
                             else "self")
            targets = self._resolve_targets(s, caster, tgt_spec, target_idx,
                                            ally_idx)

            if kind in ("hit", "dot", "drain"):
                g = o.get("group", 0)
                if g not in group_state:
                    cm = self._consume_damage_charms(s, caster, spec_school)
                    crm = self._crit_mult(caster, targets[0]) \
                        if (crit_roll and targets) else 1.0
                    group_state[g] = (cm, crm)
                charm_mult, crit_mult = group_state[g]
                amount = o.get("amount", o.get("total", 0.0))
                if o.get("outcomes") and self.rules.damage_ranges:
                    amount = self.rng.choice(o["outcomes"])   # game roll table
                elif o.get("spread") and self.rules.damage_ranges:
                    amount = self.rng.uniform(*o["spread"])   # one roll/cast
                if o.get("per_pip"):
                    amount *= pips_spent
                for t in targets:
                    ward = wards_for(g, t, spec_school)
                    if kind == "hit":
                        total += self._strike(s, caster, t, spec_school,
                                              amount, charm_mult, crit_mult,
                                              ward)
                    elif kind == "drain":
                        dealt = self._strike(s, caster, t, spec_school,
                                             amount, charm_mult, crit_mult,
                                             ward)
                        total += dealt
                        # drains ignore heal modifiers (community-confirmed)
                        healed = min(dealt * o.get("fraction", 0.5),
                                     caster.max_hp - caster.hp)
                        caster.hp += healed
                    else:                     # dot: snapshot at cast
                        wmult, fs, pierce = ward
                        dmg = amount * charm_mult
                        dmg *= self._damage_bonus(caster, spec_school)
                        dmg *= self._bubble_damage(s, spec_school)
                        dmg *= crit_mult
                        dmg *= wmult
                        dmg *= self._resist_mult(t, fs, pierce)
                        dmg = max(dmg, 0.0)
                        rounds = o.get("rounds", 3)
                        t.over_time.append(OverTime(card_name, "dot", fs,
                                                    dmg / rounds, rounds,
                                                    caster.name))
                        self._ev(s, "dot_created", target=t.name,
                                 effect=card_name, per_tick=round(
                                     dmg / rounds, 1), rounds=rounds)

            elif kind == "heal":
                amount = o["amount"] * (pips_spent if o.get("per_pip") else 1)
                cm = self._consume_heal_charms(s, caster)
                for t in targets:
                    self._heal(s, caster, t, amount, cm)

            elif kind == "hot":
                cm = self._consume_heal_charms(s, caster)
                rounds = o.get("rounds", 3)
                amt = o["total"] * (1 + caster.heal_out) * cm
                if s.bubble and s.bubble.fld == "heal":
                    amt *= 1 + s.bubble.percent
                for t in targets:
                    t.over_time.append(OverTime(card_name, "hot", school,
                                                amt / rounds, rounds,
                                                caster.name))
                    self._ev(s, "hot_created", target=t.name,
                             effect=card_name, per_tick=round(amt / rounds, 1),
                             rounds=rounds)

            elif kind in ("charm", "ward", "prism", "absorb", "dispel"):
                o["_schools"] = _op_schools_resolved(o, school)
                h = self._hanging_from_op(o, card_name, source, caster.name)
                for t in targets:
                    if t.has_stack(h.stack_key, h.slot):
                        continue              # same stack key never doubles
                    (t.charms if h.slot == "charm" else t.wards).append(
                        copy.copy(h))

            elif kind == "self_damage":
                caster.hp -= o["amount"]      # unavoidable by design

            elif kind == "gain_pips":
                for t in targets:
                    room = self.rules.max_pip_slots - t.pip_slots
                    t.norm_pips += max(0, min(o["n"], room))

            elif kind == "stun":
                for t in targets:
                    if t.stunable is False:
                        continue
                    if t.stun_blocks > 0:
                        t.stun_blocks -= 1
                        continue
                    t.stunned = max(t.stunned, o.get("rounds", 1))
                    t.stun_blocks += self.rules.stun_blocks

            elif kind in ("remove", "steal"):
                what = o["what"]
                for t in targets:
                    h = _pick_removable(t, what)
                    if h is None:
                        continue
                    if what == "dot":
                        t.over_time.remove(h)
                        continue
                    lst = t.charms if h.slot == "charm" else t.wards
                    lst.remove(h)
                    if kind == "steal" and not caster.has_stack(h.stack_key):
                        (caster.charms if h.slot == "charm"
                         else caster.wards).append(h)

            elif kind == "global":
                s.bubble = Bubble(card_name, o["field"],
                                  _op_schools_resolved(o, school),
                                  o["percent"])

            elif kind == "summon":
                blk = content.MINIONS[o["minion"]]
                m = Actor(
                    name=o["minion"], school=blk["school"], hp=float(blk["hp"]),
                    max_hp=float(blk["hp"]), team=caster.team, is_minion=True,
                    minion_role=blk["role"], minion_power=float(blk["power"]),
                    stunable=True)
                if caster.team == 0:
                    s.allies.append(m)
                else:                          # enemy summon joins their side
                    m.flat_hit = m.minion_power
                    s.enemies.append(m)

            elif kind == "stun_block":
                for t in targets:
                    t.stun_blocks += o.get("n", 1)

            elif kind == "reshuffle":
                # graveyard shuffles back into the draw pile — except the
                # reshuffle card itself (game rule: it can't restore itself)
                itself = next((c for c in reversed(caster.graveyard)
                               if c.name == card_name), None)
                back = [c for c in caster.graveyard if c is not itself]
                self.rng.shuffle(back)
                caster.deck[:0] = back
                caster.graveyard[:] = [itself] if itself else []
                self._ev(s, "reshuffled", actor=caster.name,
                         deck=len(caster.deck))

            elif kind == "sacrifice_minion":
                minions = [m for m in s.allies if m.alive] \
                    if caster.team == 0 else []
                if minions:
                    minions[0].hp = 0
                    if o.get("pips"):
                        room = self.rules.max_pip_slots - caster.pip_slots
                        caster.norm_pips += max(0, min(o["pips"], room))
                    if o.get("heal"):
                        self._heal(s, caster, caster, o["heal"],
                                   self._consume_heal_charms(s, caster))

            elif kind == "threat":
                for t in targets:
                    if "add" in o:
                        t.threat += o["add"]
                    else:
                        t.threat *= o.get("mult", 0.5)
        return total

    # ---- casting -----------------------------------------------------------
    def cast(self, s, card, target=0, ally=-1):
        """Player casts a card. Returns damage dealt (0.0 on fizzle/dispel/
        non-damage). Fizzle keeps pips; the card is discarded (Rules)."""
        caster = s.player
        if card.source == "tc":
            s.tc_casts += 1        # gold isn't free: keep TC use auditable
        self._ev(s, "cast_declared", actor=caster.name, card=card.name,
                 target=s.enemies[target].name if target < len(s.enemies)
                 else None)
        # dispel check: consumes the dispel, the pips AND the card
        if self._take_dispel(s, caster, card.school):
            s.hand.remove(card)
            caster.graveyard.append(card)
            self.spend(s, card)
            return 0.0
        acc = card.accuracy + caster.accuracy_bonus
        acc += self._consume_accuracy_charms(s, caster, card.school)
        if self.rng.random() > min(max(acc, 0.0), 1.0):
            self._ev(s, "fizzled", actor=caster.name, card=card.name,
                     accuracy=round(min(max(acc, 0.0), 1.0), 2))
            if self.rules.fizzle_discards_card:
                s.hand.remove(card)
                caster.graveyard.append(card)
            return 0.0
        s.hand.remove(card)
        caster.graveyard.append(card)
        pips_spent = self.spend(s, card)
        self._ev(s, "pips_spent", actor=caster.name, effective=pips_spent)
        dealt = self.execute_ops(s, caster, card.name, card.source, card.ops,
                                 card.school, pips_spent=pips_spent,
                                 target_idx=target, ally_idx=ally)
        self._fire_cheats(s, "after_player_cast", card=card)
        return dealt

    # ---- hand / deck / sideboard -------------------------------------------
    def draw(self, s):
        while len(s.hand) < self.rules.hand_size and s.player.deck:
            s.hand.append(s.player.deck.pop())

    def discard(self, s, card):
        """Voluntary discard. A TC drawn this round can't be discarded (game
        rule)."""
        if card in s.player.fresh_tc:
            return False
        if card in s.hand:
            s.hand.remove(card)
            s.player.graveyard.append(card)
            return True
        return False

    def draw_tc(self, s, n=1):
        """Draw up to n random treasure cards into open hand slots."""
        drawn = 0
        while drawn < n and s.player.sideboard and \
                len(s.hand) < self.rules.hand_size:
            i = self.rng.randrange(len(s.player.sideboard))
            tc = s.player.sideboard.pop(i)
            s.hand.append(tc)
            s.player.fresh_tc.append(tc)
            drawn += 1
        return drawn

    # ---- cheats --------------------------------------------------------
    def _fire_cheats(self, s, event, card=None):
        for enemy in s.enemies:
            if not enemy.alive or not enemy.cheat:
                continue
            for rule in enemy.cheat.rules:
                if rule.event != event or (rule.once and rule.fired):
                    continue
                if rule.cooldown and rule.fired and \
                        s.turn - rule.last_turn < rule.cooldown:
                    continue
                if event == "after_player_cast" and not rule.matches_cast(card):
                    continue
                if event == "hp_below" and \
                        enemy.hp > rule.when.get("frac", 0.5) * enemy.max_hp:
                    continue
                if event == "round_start" and rule.when.get("every", 1) > 1 \
                        and s.turn % rule.when["every"] != 0:
                    continue
                rule.fired += 1
                rule.last_turn = s.turn
                self._ev(s, "cheat_fired", actor=enemy.name, event=event,
                         message=rule.message)
                self.execute_ops(s, enemy, f"cheat:{rule.message or event}",
                                 "cheat", rule.ops, enemy.school)

    # ---- round loop ------------------------------------------------------
    def _build_player(self):
        deck = [self.cards[n] for n in self.decklist]
        self.rng.shuffle(deck)
        stats = self.player_stats
        return Actor(name="player", school=self.school,
                     hp=float(self.player_hp0), max_hp=float(self.player_hp0),
                     team=0, power_pip_chance=self.pp,
                     accuracy_bonus=stats.get("accuracy", 0.0),
                     pierce=stats.get("pierce", 0.0),
                     crit_chance=stats.get("crit", 0.0),
                     block_chance=stats.get("block", 0.0),
                     damage_bonus=dict(stats.get("damage", {})),
                     resist=dict(stats.get("resist", {})),
                     heal_out=stats.get("heal_out", 0.0),
                     heal_in=stats.get("heal_in", 0.0),
                     deck=deck,
                     sideboard=[self.cards[n] for n in self.sideboard_list])

    def new_state(self):
        player = self._build_player()
        enemies = [self.boss.to_actor()] + [b.to_actor()
                                            for b in self.extra_bosses]
        s = State(player, enemies)
        self.gain_pip(s)
        self.draw(s)
        return s

    def _tick_over_time(self, s, actor):
        for e in list(actor.over_time):
            if e.kind == "dot":
                dmg = self._absorb_pass(s, actor, e.per_tick)
                actor.hp -= dmg
                self._ev(s, "dot_tick", target=actor.name, effect=e.name,
                         amount=round(dmg, 1))
            else:
                healed = min(e.per_tick * (1 + actor.heal_in),
                             actor.max_hp - actor.hp)
                actor.hp += healed
                self._ev(s, "hot_tick", target=actor.name, effect=e.name,
                         amount=round(healed, 1))
            e.rounds_left -= 1
            if e.rounds_left <= 0:
                actor.over_time.remove(e)

    def _enemy_turn(self, s, enemy):
        self._tick_over_time(s, enemy)
        if not enemy.alive:
            return
        if enemy.spell_pool is not None:
            self._enemy_pip(s, enemy)   # pips accrue even while stunned
            if enemy.stunned > 0:
                enemy.stunned -= 1
                return
            card, ally_idx = self._enemy_choose(s, enemy)
            if card is not None:
                self._enemy_cast(s, enemy, card, ally_idx)
            else:
                self._ev(s, "enemy_pass", actor=enemy.name,
                         pips=enemy.norm_pips + 2 * enemy.pow_pips)
            return
        if enemy.stunned > 0:
            enemy.stunned -= 1
            return
        if enemy.flat_hit:
            pool = [s.player] + [m for m in s.allies if m.alive]
            tgt = max(pool, key=lambda a: a.threat)
            self._strike(s, enemy, tgt, enemy.school, enemy.flat_hit,
                         charm_mult=self._consume_damage_charms(
                             s, enemy, enemy.school),
                         crit_mult=self._crit_mult(enemy, tgt))

    # ---- living-boss caster (2026 boss-AI report; confidence 'modeled':
    # configured pool + legal-action AI + weighted/imperfect selection.
    # The pool is REUSABLE — no hidden hand/deck is simulated, matching
    # the report's finding that no player-like draw model is publicly
    # evidenced. Scripts (CheatRule) remain the deterministic layer.) ----

    def _enemy_pip(self, s, enemy):
        chance = enemy.power_pip_chance
        if s.bubble and s.bubble.fld == "pip_chance":
            chance = min(1.0, chance + s.bubble.percent)
        if enemy.norm_pips + enemy.pow_pips >= self.rules.max_pip_slots:
            # full rack: white pips upgrade to power (the game's escape
            # hatch; modeled) — without it a saver whose own-school nuke
            # needs 8+ effective pips can freeze into a forever-pass
            if enemy.norm_pips and self.rng.random() < chance:
                enemy.norm_pips -= 1
                enemy.pow_pips += 1
            return
        if self.rng.random() < chance:
            enemy.pow_pips += 1
        else:
            enemy.norm_pips += 1

    def _enemy_afford(self, enemy, card):
        if card.x_pips:
            return enemy.norm_pips + enemy.pow_pips >= 1
        if card.school == enemy.school:
            return enemy.norm_pips + 2 * enemy.pow_pips >= card.pips
        return enemy.norm_pips + enemy.pow_pips >= card.pips

    def _enemy_spend(self, enemy, card):
        if card.x_pips:
            eff = enemy.norm_pips + enemy.pow_pips * \
                (2 if card.school == enemy.school else 1)
            enemy.norm_pips = enemy.pow_pips = 0
            return eff
        c = card.pips
        if card.school == enemy.school:
            use_pow = min(enemy.pow_pips, c // 2)
            enemy.pow_pips -= use_pow
            c -= use_pow * 2
        use_pow2 = 0
        while c > enemy.norm_pips and enemy.pow_pips:
            enemy.pow_pips -= 1
            use_pow2 += 1
            c -= 2 if card.school == enemy.school else 1
        enemy.norm_pips -= max(c, 0)
        enemy.norm_pips = max(enemy.norm_pips, 0)
        return card.pips

    def _enemy_legal(self, s, enemy):
        """Affordable pool cards minus duplicate hanging placements."""
        tgt = self._resolve_targets(s, enemy, "enemy", 0, -1)
        tgt = tgt[0] if tgt else None
        out = []
        for name in enemy.spell_pool:
            card = self.cards.get(name)
            if card is None or not self._enemy_afford(enemy, card):
                continue
            if card.kind == "blade" and \
                    any(h.name == card.name for h in enemy.charms):
                continue
            if card.kind in ("trap", "weakness") and tgt is not None:
                lst = tgt.wards if card.kind == "trap" else tgt.charms
                if any(h.name == card.name for h in lst):
                    continue
            if card.kind == "shield" and \
                    any(h.name == card.name for h in enemy.wards):
                continue
            out.append(card)
        return out

    def _enemy_cost(self, enemy, card):
        """Real pip price of casting now: X-pip spells dump the whole
        rack, so their cost is the current effective total, never the
        card's printed 0."""
        if not card.x_pips:
            return card.pips
        return enemy.norm_pips + enemy.pow_pips * \
            (2 if card.school == enemy.school else 1)

    def _enemy_attainable(self, enemy, card):
        """Can this card EVER be paid for from a full rack? An
        off-school 8-pip spell can't (7 slots, no doubling) and must
        never be chosen as the nuke to save toward."""
        if card.x_pips:
            return True
        cap = self.rules.max_pip_slots
        return card.pips <= cap * (2 if card.school == enemy.school
                                   else 1)

    def _enemy_immediate(self, enemy, card):
        """Up-front damage of a cast right now: hit/drain ops only
        (a DoT's total lands over rounds and cannot 'kill now')."""
        eff = self._enemy_cost(enemy, card) if card.x_pips else 0
        total = 0.0
        for o in card.ops:
            if o["op"] in ("hit", "drain"):
                amt = o.get("amount", o.get("total", 0.0))
                total += amt * (eff if o.get("per_pip") else 1)
        return total

    def _enemy_choose(self, s, enemy):
        """Archetype priorities with an imperfection knob: with
        P(discipline) follow the role's bucket order, else pick
        uniformly among legal casts (the report: state-aware but
        imperfect; exact weights not public). Returns (card, ally_idx)
        — ally_idx >= 0 marks a teammate-directed cast."""
        legal = self._enemy_legal(s, enemy)
        if not legal:
            return None, -1
        cost = lambda c: self._enemy_cost(enemy, c)
        if self.rng.random() >= enemy.discipline:
            return self.rng.choice(legal), -1
        hits = [c for c in legal if c.kind in ("damage", "drain")]
        blades = [c for c in legal if c.kind == "blade"]
        traps = [c for c in legal if c.kind == "trap"]
        heals = [c for c in legal if c.kind == "heal"]
        shields = [c for c in legal if c.kind == "shield"]
        debuffs = [c for c in legal if c.kind in ("weakness", "mantle")]
        tgt = self._resolve_targets(s, enemy, "enemy", 0, -1)
        tgt = tgt[0] if tgt else None
        pool_cards = [self.cards[n] for n in enemy.spell_pool
                      if n in self.cards]
        top = max((c for c in pool_cards
                   if c.kind in ("damage", "drain")
                   and self._enemy_attainable(enemy, c)),
                  key=lambda c: c.damage, default=None)
        arch = enemy.archetype

        if arch == "healer":
            friends = [e for e in s.enemies if e.alive]
            hurt = [f for f in friends if f.hp < 0.65 * f.max_hp]
            if hurt and heals:
                return max(heals, key=lambda c: c.damage or c.pips), 0
            if shields:
                return shields[0], -1
            return (min(hits, key=cost), -1) if hits else (None, -1)
        if arch == "buffer":
            if blades:
                return blades[0], -1
            if traps:
                return traps[0], -1
            return None, -1
        if arch == "debuffer":
            if debuffs:
                return debuffs[0], -1
            return (min(hits, key=cost), -1) if hits else (None, -1)
        if arch == "tank":
            if shields:
                return shields[0], -1
            return (min(hits, key=cost), -1) if hits else (None, -1)

        # hitter (default): kill shot > setup while saving for the top
        # nuke > fire the top nuke when ready > don't waste a full rack
        if tgt is not None and hits:
            kill = [c for c in hits if self._enemy_immediate(enemy, c) *
                    self._resist_mult(tgt, c.school, enemy.pierce)
                    >= tgt.hp]
            if kill:
                return min(kill, key=cost), -1
        full = enemy.norm_pips + enemy.pow_pips >= \
            self.rules.max_pip_slots
        top_ready = top is not None and (
            full if top.x_pips else self._enemy_afford(enemy, top))
        if top is not None and not top_ready:
            if blades:
                return blades[0], -1
            if traps:
                return traps[0], -1
            return None, -1                   # save pips for the nuke
        if top is not None and top_ready and top in hits:
            return top, -1
        if full and hits:
            return max(hits, key=lambda c:
                       self._enemy_immediate(enemy, c)), -1
        return None, -1

    def _enemy_cast(self, s, enemy, card, ally_idx=-1):
        """Fizzle keeps pips (player rule, transplanted — modeled);
        the pool is reusable. Player-cast mantles and dispels bite:
        the boss consumes its incoming accuracy charms and dispels
        exactly like a player caster."""
        if self._take_dispel(s, enemy, card.school):
            self._enemy_spend(enemy, card)
            self._ev(s, "enemy_dispelled", actor=enemy.name,
                     card=card.name)
            return
        acc = min(1.0, card.accuracy + enemy.accuracy_bonus +
                  self._consume_accuracy_charms(s, enemy, card.school))
        if self.rng.random() > acc:
            self._ev(s, "enemy_fizzle", actor=enemy.name, card=card.name)
            return
        eff = self._enemy_spend(enemy, card)
        self._ev(s, "enemy_cast", actor=enemy.name, card=card.name,
                 pips_spent=eff)
        ops = card.ops
        if ally_idx >= 0:
            # teammate-directed heal: 'self'-authored heal ops follow
            # the healer's decision, not the card text's player framing
            ops = [dict(o, tgt="ally") if o["op"] in ("heal", "hot")
                   else o for o in ops]
        self.execute_ops(s, enemy, card.name, "enemy", ops,
                         card.school, pips_spent=eff, ally_idx=ally_idx)

    def _minion_turn(self, s, minion):
        self._tick_over_time(s, minion)
        if not minion.alive:
            return
        if minion.stunned > 0:
            minion.stunned -= 1
            return
        if minion.minion_role == "heal":
            self._heal(s, minion, s.player, minion.minion_power)
        else:
            foes = s.living_enemies()
            if foes:
                self._strike(s, minion, foes[0], minion.school,
                             minion.minion_power, charm_mult=1.0,
                             crit_mult=1.0)

    def end_round(self, s):
        s.turn += 1
        self._ev(s, "round_start", round=s.turn)
        self.gain_pip(s)
        self._fire_cheats(s, "round_start")
        self._fire_cheats(s, "hp_below")
        for m in s.allies:
            if m.alive:
                self._minion_turn(s, m)
        for enemy in s.enemies:
            if enemy.alive or enemy.over_time:
                self._enemy_turn(s, enemy)
        s.allies[:] = [m for m in s.allies if m.alive]
        s.player.fresh_tc.clear()
        self.draw(s)

    def _normalize_action(self, res):
        if res is None:
            return Action()
        if isinstance(res, Action):
            return res
        if isinstance(res, tuple):
            return Action(card=res[0], target=res[1])
        return Action(card=res)

    def run(self, policy, max_turns=40):
        """Play one duel. Returns (turns, won, player_hp_left). The
        final state stays on self.last_state for audit metrics
        (tc_casts, events)."""
        s = self.new_state()
        self.last_state = s
        while s.turn < max_turns:
            self._tick_over_time(s, s.player)
            if s.player.hp <= 0:
                return s.turn, False, 0
            if not any(e.alive for e in s.enemies):
                return s.turn + 1, True, s.player.hp
            if s.player.stunned > 0:
                s.player.stunned -= 1
            else:
                act = self._normalize_action(policy(self, s))
                for c in act.discards:
                    self.discard(s, c)
                if act.draw_tc:
                    self.draw_tc(s, act.draw_tc)
                if act.card is not None and \
                        self.can_cast(s, act.card, act.target):
                    self.cast(s, act.card, act.target, act.ally)
            if not any(e.alive for e in s.enemies):
                return s.turn + 1, True, s.player.hp
            self.end_round(s)
            if s.player.hp <= 0:
                return s.turn, False, 0
            if not any(e.alive for e in s.enemies):
                return s.turn, True, s.player.hp
        return max_turns, False, s.player_hp


def _op_schools_resolved(op, card_school):
    spec = op.get("schools", "own")
    if spec == "own":
        return {card_school}
    if spec is None:
        return None
    return set(spec)


def _pick_removable(actor, what):
    """Most recent unprotected effect matching a remove/steal spec.
    'ward_pos' = beneficial to the owner (shields, absorbs); 'ward_neg' =
    harmful to the owner (traps, Feint backlash)."""
    if what == "dot":
        pool = [e for e in actor.over_time if e.kind == "dot"]
        return pool[-1] if pool else None
    if what == "charm_pos":
        pool = [h for h in actor.charms if not h.protected and
                h.kind == "damage" and h.percent > 0]
    elif what == "charm_neg":
        pool = [h for h in actor.charms if not h.protected and h.percent < 0]
    elif what == "ward_pos":
        pool = [h for h in actor.wards if not h.protected and
                (h.kind == "absorb" or
                 (h.kind == "damage" and h.percent < 0))]
    elif what == "ward_neg":
        pool = [h for h in actor.wards if not h.protected and
                h.kind in ("damage", "prism") and
                (h.kind == "prism" or h.percent > 0)]
    else:
        pool = []
    return pool[-1] if pool else None


# ---------------------------------------------------------------- strategies

def effective_pips(sim, s, card):
    return s.norm_pips + s.pow_pips * (2 if card.school == sim.school else 1)

def castable(sim, s, kind):
    return [c for c in s.hand if c.kind == kind and sim.can_cast(s, c)]

def strat_nuke_asap(sim, s):
    opts = castable(sim, s, "damage") + castable(sim, s, "drain")
    return max(opts, key=lambda c: c.damage, default=None)

def make_blade_stack(n_buffs):
    """Stack n distinct buffs, then fire the biggest nuke once affordable.
    Traps are laid before prisms so the FIFO ward pass counts them
    pre-conversion."""
    def buffs(sim, s):
        return (castable(sim, s, "blade") or castable(sim, s, "trap") or
                castable(sim, s, "prism"))
    def strat(sim, s):
        pend = len(s.blades) + len(s.traps)
        if pend < n_buffs:
            b = buffs(sim, s)
            if b:
                return max(b, key=lambda c: c.percent)
        nukes = [c for c in s.hand if c.kind in ("damage", "drain")]
        if nukes:
            big = max(nukes, key=lambda c: c.damage)
            if sim.afford(s, big) and not big.x_pips:
                return big
            if big.x_pips and effective_pips(sim, s, big) >= 7:
                return big
            b = buffs(sim, s)
            return max(b, key=lambda c: c.percent) if b else None
        return None
    return strat

def tc_reflex(sim, s):
    """Greedy sideboard reflex: make room if the hand is full (the
    deck refills the hand every round, so a free slot almost never
    happens on its own — real players discard to draw) and pull one
    random treasure card. Never discards a TC drawn this round (game
    rule). Discard ranking: hanging duplicates, then repeats, then
    utility, then the smallest hit. Drawing is free in-game; WHEN to
    cast what was drawn stays the policy's decision."""
    if not s.player.sideboard:
        return
    if len(s.hand) >= sim.rules.hand_size:
        seen = set()

        def rank(cd):
            pend = (cd.kind == "blade" and cd.name in s.blades) or \
                   (cd.kind in ("trap", "prism") and cd.name in s.traps)
            dup = cd.name in seen
            seen.add(cd.name)
            v = cd.damage if cd.kind in ("damage", "drain") \
                else abs(cd.percent) or 0.4
            if pend:
                return (0, v)
            if dup:
                return (1, v)
            if cd.kind not in ("damage", "drain"):
                return (2, v)
            return (3, v)
        junk = [cd for cd in s.hand if cd not in s.player.fresh_tc]
        if not junk:
            return
        pick = min(junk, key=rank)
        s.hand.remove(pick)
        s.player.graveyard.append(pick)
    sim.draw_tc(s, 1)


def with_tc_draw(policy):
    """Wrap a policy with the treasure-card reflex (see tc_reflex)."""
    def strat(sim, s):
        tc_reflex(sim, s)
        return policy(sim, s)
    return strat


def pick_focus(s):
    """Kill-the-support-first target selection: the report's team-fight
    guidance ('healers, shielders, and blade-removers can be higher-
    priority targets than a slow hitter'). Support archetypes die
    first (lowest HP among them), then the lowest-HP attacker —
    finishing kills shrinks incoming damage fastest."""
    living = [(i, e) for i, e in enumerate(s.enemies) if e.alive]
    if not living:
        return 0
    support = [(i, e) for i, e in living
               if getattr(e, "archetype", "") in
               ("healer", "buffer", "debuffer")
               and getattr(e, "spell_pool", None) is not None]
    pool = support or living
    return min(pool, key=lambda t: t[1].hp)[0]


def with_focus(policy):
    """Wrap a card policy with focus-fire targeting: every cast is
    aimed at pick_focus(s) instead of enemy 0."""
    def strat(sim, s):
        card = policy(sim, s)
        if card is None:
            return None
        return (card, pick_focus(s))
    return strat


def make_rating_crit(k_crit=250.0, k_block=250.0, cap=0.85,
                     min_mult=1.25):
    """Post-classic criticals from RATINGS rather than flat chances.

    Confidence: MODELED. KingsIsle has never published the curves; the
    2026 research report is explicit that crit/block are gear-driven
    and their formulas are not public. What IS public and preserved
    here: a rating buys diminishing probability, block contests crit,
    and higher block shaves the multiplier rather than only cancelling
    it. `Actor.crit_chance`/`block_chance` carry the RATINGS in this
    resolver (not probabilities) — the era swap is the point of
    Rules.crit_resolver.
    """
    def resolve(caster, target, rng, rules):
        cr = max(caster.crit_chance, 0.0)
        if cr <= 0:
            return 1.0
        p = min(cap, cr / (cr + k_crit))
        if rng.random() >= p:
            return 1.0
        br = max(target.block_chance, 0.0)
        if br > 0:
            pb = min(cap, br / (br + k_block))
            if rng.random() < pb:
                return 1.0                      # blocked outright
            shave = br / (br + k_block)         # partial mitigation
            return max(min_mult,
                       1 + (rules.crit_multiplier - 1) * (1 - shave))
        return rules.crit_multiplier
    return resolve


def make_survival(inner, heal_below=0.45, shield_when_open=True):
    """Wrap a strategy with defensive triage: heal when low, keep a shield
    up, otherwise defer to the inner strategy."""
    def strat(sim, s):
        p = s.player
        if p.hp < heal_below * p.max_hp:
            heals = castable(sim, s, "heal")
            if heals:
                return max(heals, key=lambda c: c.damage or 400)
        if shield_when_open and not any(h.percent < 0 for h in p.wards):
            shields = castable(sim, s, "shield")
            if shields:
                boss = s.enemies[0]
                best = [c for c in shields
                        if any(h.get("op") == "ward" and
                               (_op_schools_resolved(h, c.school) is None or
                                boss.school in _op_schools_resolved(h, c.school))
                               for h in c.ops)]
                if best:
                    return best[0]
        return inner(sim, s)
    return strat

def evaluate(sim, policy, n=20000, max_turns=40):
    ttk, wins = [], 0
    for _ in range(n):
        t, won, _ = sim.run(policy, max_turns=max_turns)
        if won:
            wins += 1
            ttk.append(t)
    return wins / n, (sum(ttk) / len(ttk) if ttk else float("nan"))


def evaluate_paired(sim, policies, n=2000, max_turns=40, base_seed=10**6):
    """Evaluate several policies on the SAME seed stream (same shuffles,
    same RNG draws), with distribution stats — variance-reduced comparison.
    `policies`: dict name -> policy. Returns dict name -> stats."""
    out = {}
    for name, pol in policies.items():
        ttk, hp_left, wins, tc = [], [], 0, 0
        for i in range(n):
            sim.rng = random.Random(base_seed + i)
            t, won, hp = sim.run(pol, max_turns=max_turns)
            tc += sim.last_state.tc_casts
            if won:
                wins += 1
                ttk.append(t)
                hp_left.append(hp)
        ttk.sort()
        pct = lambda q: ttk[min(int(q * len(ttk)), len(ttk) - 1)] \
            if ttk else float("nan")
        out[name] = dict(
            win_rate=wins / n,
            mean_ttk=sum(ttk) / len(ttk) if ttk else float("nan"),
            median_ttk=pct(0.5), p90_ttk=pct(0.9), p99_ttk=pct(0.99),
            mean_hp_left=sum(hp_left) / len(hp_left) if hp_left else 0.0,
            mean_tc_casts=tc / n,           # gold spent is never hidden
            ruleset=sim.rules.ruleset_id)
    return out


# ---------------------------------------------------------------- analytics

def max_remaining_damage(sim, s):
    """Optimistic upper bound on the damage the player can still deliver:
    every remaining nuke lands, every distinct buff (in hand, deck,
    sideboard, or already hanging) multiplies every nuke, X-pip spells get
    full pips, and each hit meets the most favorable school multiplier any
    living enemy offers (prisms make schools fungible for bound purposes).
    Useful as a scarcity feature and for the no-impossible-kill check."""
    p = s.player
    pool = list(s.hand) + list(p.deck) + list(p.sideboard)
    max_eff = sim.rules.max_pip_slots * 2
    base = sum(c.damage * (max_eff if c.x_pips else 1)
               for c in pool if c.kind in ("damage", "drain"))
    if base <= 0:
        return 0.0
    mult = 1.0
    seen = set()
    for c in pool:
        if c.kind in ("blade", "trap") and c.stack_key not in seen:
            seen.add(c.stack_key)
            mult *= 1 + max(c.percent, 0.0)
    for h in p.charms:
        if h.kind == "damage" and h.percent > 0:
            mult *= 1 + h.percent
    # schools the remaining damage can actually arrive as: card schools,
    # expanded through prism conversions available in the pool or already
    # placed on an enemy
    reachable = set()
    for c in pool:
        if c.kind in ("damage", "drain"):
            reachable.add(c.school)
            for o in c.ops:
                if o["op"] in ("hit", "dot", "drain") and o.get("school"):
                    reachable.add(o["school"])
    prisms = [(o["from"], o["to"]) for c in pool if c.kind == "prism"
              for o in c.ops if o["op"] == "prism"]
    for e in s.living_enemies():
        prisms += [(next(iter(h.schools)), h.convert_to)
                   for h in e.wards if h.kind == "prism" and h.schools]
    changed = True
    while changed:
        changed = False
        for frm, to in prisms:
            if frm in reachable and to not in reachable:
                reachable.add(to)
                changed = True
    best_school = 0.0
    for e in s.living_enemies():
        for h in e.wards:
            if h.kind == "damage" and h.percent > 0:
                mult *= 1 + h.percent
        for school in reachable:
            res = e.resist.get(school, 0.0) + e.resist.get("*", 0.0)
            m = (1 - max(res - p.pierce, 0.0)) * \
                (1 + e.boost.get(school, 0.0))
            best_school = max(best_school, m)
    bubbles = [c.ops[0]["percent"] for c in pool
               if c.kind == "global" and c.ops[0].get("field") == "damage"]
    if s.bubble and s.bubble.fld == "damage":
        bubbles.append(s.bubble.percent)
    if bubbles:
        mult *= 1 + max(max(bubbles), 0.0)
    mult *= 1 + max(p.damage_bonus.values(), default=0.0)
    if p.crit_chance > 0:
        mult *= sim.rules.crit_multiplier
    return base * mult * best_school


def provably_unwinnable(sim, s):
    """True when even the optimistic damage bound can't cover the living
    enemies' combined health (assumes enemies take no self-damage)."""
    total = sum(e.hp for e in s.living_enemies())
    return max_remaining_damage(sim, s) < total


if __name__ == "__main__":
    cards = load_cards("cards_clean.json")
    from bosses import BOSSES, DECKS
    boss = BOSSES["Jade Oni"]
    deck = DECKS["fire"]["oneshot"]
    sim = Sim(cards, deck, school="fire",
              boss=Boss(boss.name, boss.hp, boss.school, 0),
              player_hp=10**9)
    print(f"{'strategy':<22}{'kill%':>8}{'mean TTK':>10}")
    strats = [("nuke ASAP", strat_nuke_asap)] + \
             [(f"buff x{n} -> nuke", make_blade_stack(n)) for n in (1, 2, 3, 4, 5)]
    for name, pol in strats:
        w, m = evaluate(sim, pol, n=10000)
        print(f"{name:<22}{w*100:>7.1f}%{m:>10.2f}")

    # the v0.3 headline: prisms crack the same-school wall
    print("\n-- prism demo: ice vs Prince Gobblestone (ice, 40% self-resist)")
    gb = BOSSES["Prince Gobblestone"]
    for dname in ("oneshot", "prism"):
        dl = DECKS["ice"][dname]
        sim = Sim(cards, dl, "ice", Boss(gb.name, gb.hp, gb.school, 0),
                  player_hp=10**9)
        w, m = max(((evaluate(sim, make_blade_stack(k), n=4000), k)
                    for k in range(6)), key=lambda r: (r[0][0], -r[0][1]))[0]
        print(f"   {dname:<9} kill {w*100:5.1f}%  TTK {m:6.2f}")
