"""One combat scenario, rendered into both engines' terms.

The whole value of comparing wizAi against Deimos rests on the two
engines being handed *the same fight*. Writing the inputs twice invites
the comparison to fail for translation reasons rather than modelling
reasons, so a `Scenario` is declared once here and each engine gets a
rendering of it.

wizAi is driven through the same call sequence a real cast uses
(`w101_sim.Sim._resolve_ops`, lines 1316-1334):

    charm_mult = sim._consume_damage_charms(s, caster, school)
    crit_mult  = sim._crit_mult(caster, target)
    sim._strike(s, caster, target, school, base, charm_mult, crit_mult)

`_strike` calls `_ward_pass` itself, so shields, traps and prisms are
resolved exactly as they are in a duel.
"""
from dataclasses import dataclass, field

from w101_sim import Actor, Boss, Hanging, Rules, Sim, State
from . import deimos_damage as dd


def _bare_sim(rules: Rules = None) -> Sim:
    """A `Sim` with no deck and no encounter. `_strike` and the charm/ward
    passes only touch `self.rules` and `self.rng`, so the rest is inert --
    but `Sim.__init__` validates a boss, hence the placeholder."""
    return Sim(cards={}, decklist=[], school="fire",
               boss=Boss(name="_scenario", hp=1, school="ice", dmg=0),
               rules=rules or Rules(), rng=__import__("random").Random(0))


@dataclass
class Buff:
    """A blade on the caster or a trap/shield on the target.

    `percent` is in *points* the way both engines' data carries it: +35
    is a Fireblade, -50 is a Tower Shield, +25 is a Feint. `school` None
    means universal (Deimos's `damage_type == 80289`, wizAi's
    `Hanging.schools is None`).
    """
    name: str
    percent: float
    school: str = None
    #: distinct stacking identity. Two buffs sharing one are the same
    #: effect twice and only the first counts, in both engines.
    stack: int = None


@dataclass
class Side:
    school: str = "fire"
    hp: float = 5000.0
    max_hp: float = 5000.0
    damage_pct: float = 0.0     # gear damage bonus, as a fraction
    flat_damage: float = 0.0
    pierce: float = 0.0         # fraction
    resist_pct: float = 0.0     # fraction
    flat_resist: float = 0.0
    boost: float = 0.0          # fraction; a mob's incoming-damage weakness
    crit: float = 0.0
    block: float = 0.0
    level: int = 1
    is_player: bool = False


@dataclass
class Scenario:
    name: str
    base: float                       # the spell's headline damage
    school: str                       # the spell's school
    caster: Side = field(default_factory=lambda: Side(is_player=True))
    target: Side = field(default_factory=lambda: Side(school="ice"))
    blades: list = field(default_factory=list)   # on the caster
    wards: list = field(default_factory=list)    # on the target
    prism_to: str = None              # convert the hit's school on the way in
    #: where the prism sits relative to `wards`. wizAi walks wards in list
    #: order; traps before the prism boost pre-conversion.
    prism_index: int = None
    note: str = ""

    # ---------------------------------------------------------- wizAi side
    def _wizai_actors(self):
        c, t = self.caster, self.target
        caster = Actor(
            name="player", school=c.school, hp=c.hp, max_hp=c.max_hp, team=0,
            pierce=c.pierce, crit_chance=c.crit, flat_damage=c.flat_damage,
            damage_bonus={"*": c.damage_pct} if c.damage_pct else {},
        )
        target = Actor(
            name="enemy", school=t.school, hp=t.hp, max_hp=t.max_hp, team=1,
            block_chance=t.block, flat_resist=t.flat_resist,
            resist={"*": t.resist_pct} if t.resist_pct else {},
            boost={self.school: t.boost} if t.boost else {},
        )
        for i, b in enumerate(self.blades):
            caster.charms.append(Hanging(
                name=b.name, slot="charm", kind="damage",
                percent=b.percent / 100.0,
                schools=None if b.school is None else {b.school},
                source="deck", sub=b.stack if b.stack is not None else i,
            ))
        wards = []
        for i, w in enumerate(self.wards):
            wards.append(Hanging(
                name=w.name, slot="ward", kind="damage",
                percent=w.percent / 100.0,
                schools=None if w.school is None else {w.school},
                source="deck", sub=w.stack if w.stack is not None else i,
            ))
        if self.prism_to:
            p = Hanging(name="prism", slot="ward", kind="prism",
                        schools={self.school}, convert_to=self.prism_to,
                        source="deck")
            idx = len(wards) if self.prism_index is None else self.prism_index
            wards.insert(idx, p)
        target.wards = wards
        return caster, target

    #: overrides the `rules=` argument, so a whole suite can be re-run
    #: under a different ruleset without threading it through every call.
    rules_override = None

    def wizai_damage(self, rules: Rules = None) -> float:
        rules = rules or self.rules_override
        caster, target = self._wizai_actors()
        sim = _bare_sim(rules)
        s = State(caster, [target])
        charm_mult = sim._consume_damage_charms(s, caster, self.school)
        crit_mult = sim._crit_mult(caster, target)
        return sim._strike(s, caster, target, self.school, self.base,
                           charm_mult, crit_mult)

    # --------------------------------------------------------- Deimos side
    def _deimos_effects(self):
        sid = dd.school_id(self.school)

        def dtype(school):
            return dd.UNIVERSAL_DAMAGE_TYPE if school is None else dd.school_id(school)

        caster_fx = [
            dd.Effect(dd.SpellEffects.modify_outgoing_damage, b.percent,
                      dtype(b.school),
                      spell_template_id=(b.stack if b.stack is not None else 1000 + i))
            for i, b in enumerate(self.blades)
        ]
        # combat_math.py:106-107 -- the caster list arrives newest-first and
        # is reversed so blades apply in the order they were cast.
        caster_fx.reverse()

        target_fx = [
            dd.Effect(dd.SpellEffects.modify_incoming_damage, w.percent,
                      dtype(w.school),
                      spell_template_id=(w.stack if w.stack is not None else 2000 + i))
            for i, w in enumerate(self.wards)
        ]
        if self.prism_to:
            p = dd.Effect(dd.SpellEffects.modify_incoming_damage_type,
                          dd.school_id(self.prism_to), sid,
                          spell_template_id=3000)
            idx = len(target_fx) if self.prism_index is None else self.prism_index
            target_fx.insert(idx, p)
        return caster_fx, target_fx

    def _deimos_stats(self):
        c, t = self.caster, self.target
        schools = list(dd.SCHOOL_IDS.values())
        caster = dd.Stats(
            damage={s: c.damage_pct for s in schools},
            flat_damage={s: c.flat_damage for s in schools},
            crit={s: c.crit for s in schools},
            # A FRACTION, not points. `ap_bonus_percent` is read straight
            # out of the client and `stat_viewer.py:227-229` renders it
            # through `to_percent()` (x100), so 0.15 on the wire is 15%.
            # `effect_simulation.py:189` confirms it -- a pierce *effect*
            # arrives in points and is divided by 100 before joining this
            # stat. Passing points here would paper over the ward/pierce
            # unit bug this suite is meant to surface.
            pierce={s: c.pierce for s in schools},
            level=c.level, is_player=c.is_player,
        )
        # wizAi models a mob's incoming-damage weakness as a separate
        # `boost` map; Deimos has no such field -- a negative resist *is*
        # the boost (combat_math.py:262-263). Fold it in so both engines
        # see one number.
        resist = {s: t.resist_pct for s in schools}
        if t.boost:
            resist[dd.school_id(self.school)] = t.resist_pct - t.boost
            if self.prism_to:
                resist[dd.school_id(self.prism_to)] = t.resist_pct - t.boost
        target = dd.Stats(
            resist=resist,
            flat_resist={s: t.flat_resist for s in schools},
            block={s: t.block for s in schools},
            level=t.level, is_player=t.is_player,
        )
        return caster, target

    def deimos_damage(self, force_crit: bool = False) -> float:
        caster_stats, target_stats = self._deimos_stats()
        caster_fx, target_fx = self._deimos_effects()
        return dd.deimos_damage(
            damage=self.base,
            damage_type=dd.school_id(self.school),
            caster=caster_stats, target=target_stats,
            caster_effects=caster_fx, target_effects=target_fx,
            force_crit=force_crit,
        )


# --------------------------------------------------------------------------
# The suite. Ordered so the plain cases come first: when a later scenario
# disagrees, the earlier ones say which ingredient introduced it.
# --------------------------------------------------------------------------
def suite():
    P = dict(is_player=True)
    return [
        Scenario("bare hit", 500, "fire",
                 note="no modifiers anywhere; both engines should agree exactly"),

        Scenario("one blade", 500, "fire",
                 blades=[Buff("Fireblade", 35, "fire")],
                 note="single multiplicative charm"),

        Scenario("three stacked blades", 500, "fire",
                 blades=[Buff("Fireblade", 35, "fire"),
                         Buff("Balanceblade", 25, None),
                         Buff("Sharpened Fireblade", 45, "fire")],
                 note="do blades compound, or add?"),

        Scenario("duplicate blade", 500, "fire",
                 blades=[Buff("Fireblade", 35, "fire", stack=7),
                         Buff("Fireblade", 35, "fire", stack=7)],
                 note="same stacking identity twice -- only one should count"),

        Scenario("one trap", 500, "fire",
                 wards=[Buff("Fire Trap", 25, "fire")],
                 note="single multiplicative ward"),

        Scenario("duplicate trap", 500, "fire",
                 wards=[Buff("Fire Trap", 25, "fire", stack=4),
                        Buff("Fire Trap", 25, "fire", stack=4)],
                 note="three traps on a mob is three boosted hits, not one "
                      "hit at 1.25^3 -- the second is banked, not consumed"),

        Scenario("blade and trap", 500, "fire",
                 blades=[Buff("Fireblade", 35, "fire")],
                 wards=[Buff("Fire Trap", 25, "fire")]),

        Scenario("tower shield", 500, "fire",
                 wards=[Buff("Tower Shield", -50, None),
                        Buff("Fire Trap", 25, "fire")],
                 note="negative ward"),

        Scenario("shield vs pierce", 500, "fire",
                 caster=Side(pierce=0.20, **P),
                 wards=[Buff("Tower Shield", -50, None)],
                 note="pierce eats into the shield before it multiplies"),

        Scenario("pierce exceeds shield", 500, "fire",
                 caster=Side(pierce=0.60, **P),
                 wards=[Buff("Tower Shield", -50, None)],
                 note="leftover pierce should carry to resist, not vanish"),

        Scenario("mob resist", 500, "fire",
                 target=Side(school="ice", resist_pct=0.30)),

        Scenario("mob resist vs pierce", 500, "fire",
                 caster=Side(pierce=0.15, **P),
                 target=Side(school="ice", resist_pct=0.30)),

        Scenario("mob boost", 500, "fire",
                 target=Side(school="ice", boost=0.25),
                 note="an enemy that takes extra from this school"),

        Scenario("gear damage", 500, "fire",
                 caster=Side(damage_pct=0.50, **P),
                 note="school damage stat, uncurved"),

        Scenario("gear damage and blade", 500, "fire",
                 caster=Side(damage_pct=0.50, **P),
                 blades=[Buff("Fireblade", 35, "fire")],
                 note="is gear damage a separate multiplier from charms?"),

        Scenario("flat damage", 500, "fire",
                 caster=Side(flat_damage=100, **P),
                 blades=[Buff("Fireblade", 35, "fire")],
                 wards=[Buff("Tower Shield", -50, None)],
                 note="WHERE flat damage lands in the pipeline"),

        Scenario("flat resist", 500, "fire",
                 target=Side(school="ice", resist_pct=0.30, flat_resist=75),
                 note="flat resist before or after the resist multiply"),

        Scenario("prism", 500, "fire",
                 target=Side(school="ice", resist_pct=0.50),
                 prism_to="ice",
                 note="converted school meets the target's ice resist"),

        Scenario("trap before prism", 500, "fire",
                 target=Side(school="ice", resist_pct=0.50),
                 wards=[Buff("Fire Trap", 25, "fire")],
                 prism_to="ice", prism_index=1,
                 note="a fire trap laid first should boost pre-conversion"),

        Scenario("trap after prism", 500, "fire",
                 target=Side(school="ice", resist_pct=0.50),
                 wards=[Buff("Fire Trap", 25, "fire")],
                 prism_to="ice", prism_index=0,
                 note="the same trap after the prism should strand"),

        # The two full-stack rows are a matched pair, and the pairing is
        # the point: they are identical except for pierce. If the one
        # WITH pierce diverges and the one WITHOUT agrees, the gap is the
        # pierce unit bug and nothing else -- no need to take that on
        # faith.
        Scenario("full stack (no pierce)", 500, "fire",
                 caster=Side(damage_pct=0.45, flat_damage=50, **P),
                 target=Side(school="ice", resist_pct=0.35, flat_resist=40),
                 blades=[Buff("Fireblade", 35, "fire"),
                         Buff("Balanceblade", 25, None)],
                 wards=[Buff("Fire Trap", 25, "fire"),
                        Buff("Tower Shield", -50, None)],
                 note="everything at once, pierce held out"),

        Scenario("full stack", 500, "fire",
                 caster=Side(damage_pct=0.45, flat_damage=50, pierce=0.10, **P),
                 target=Side(school="ice", resist_pct=0.35, flat_resist=40),
                 blades=[Buff("Fireblade", 35, "fire"),
                         Buff("Balanceblade", 25, None)],
                 wards=[Buff("Fire Trap", 25, "fire"),
                        Buff("Tower Shield", -50, None)],
                 note="the same board, plus 10% pierce"),
    ]
