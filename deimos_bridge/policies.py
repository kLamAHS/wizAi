"""Policies for live play.

wizAi's built-in heuristics were written against decks the builder
produced, and every one of those is **school-coherent** — a fire deck
holds fire blades and fire nukes. That assumption is invisible in the
simulator and false the moment a real wizard opens a real hand: starter
wands hand out Thunder Snake (storm), Imp (fire), Scarab (myth) and Dark
Sprite (death) regardless of school, and pets add trained cards of their
own.

`make_blade_stack` picks the biggest available buff by percentage and,
separately, the biggest available nuke by damage, with nothing tying the
two together. On a coherent deck they always match. On a real starter
hand it will happily stack a **Mythblade** and then fire a **Thunder
Snake**, and the blade does nothing at all — `_consume_damage_charms`
only applies charms where `h.matches(school)`.

`school_aware_blade_stack` closes that loop: decide the nuke first, then
only stack buffs that can actually multiply it.

On a school-coherent deck the two are **identical** — every blade matches
the only nuke school available, so "best matching buff" and "best buff"
select the same card. `tests/test_deimos_bridge.py` asserts that over the
project's own live decks, which is what makes this safe as the live
default without invalidating any published table.
"""


def _card_schools(card):
    """Damage schools a buff applies to, or None for universal.

    Read off the ops rather than `card.school`: a Tri Blade is a *balance*
    card whose three charms boost ice, storm and fire, and Balanceblade is
    a balance card that boosts everything.
    """
    out = set()
    for op in card.ops or []:
        if op.get("op") not in ("charm", "ward"):
            continue
        schools = op.get("schools")
        if schools is None:
            return None                  # universal
        out.update(schools)
    return out or None


def buff_matches(card, school):
    """Would this buff multiply a hit of `school`?"""
    schools = _card_schools(card)
    return schools is None or school in schools


def pending_for(state, school):
    """Buffs already on the board that apply to `school`.

    Counts the same things `make_blade_stack` counts — charms on the
    caster, wards on the primary enemy, the aura — but only those that
    can actually act on this school, which it does not check.

    That distinction is worth more than it sounds, and not only on mixed
    hands. A **Tri Trap** places three ward legs: fire, ice and storm. An
    ice wizard's hit consumes the ice leg; the fire and storm legs stay
    on the enemy for the rest of the duel, because nothing in an ice deck
    will ever trigger them. `State.traps` counts all three, so
    `make_blade_stack` believes it has a full buff stack while holding
    one live multiplier and two corpses — and fires early. On the
    project's own ice/stack deck that costs 17 points of kill rate.
    """
    n = 0
    for h in state.player.charms:
        if h.kind == "damage" and h.percent > 0 and h.matches(school):
            n += 1
    enemy = state.enemies[0] if state.enemies else None
    if enemy is not None:
        for h in enemy.wards:
            if h.kind == "prism" or (h.kind == "damage" and h.percent > 0
                                     and h.matches(school)):
                n += 1
    if state.player.aura is not None:
        n += 1
    return n


def choose_nuke(sim, state):
    """The hit this turn is being built around.

    Prefers the caster's own school on a tie-ish call, because power pips
    are worth two there and one elsewhere (`effective_pips`) — an ice
    wizard's Frost Beetle is cheaper to fire than a storm wand card of
    similar size, and only the former can be blade-stacked with the ice
    blades an ice deck carries.
    """
    nukes = [c for c in state.hand if c.kind in ("damage", "drain")]
    if not nukes:
        return None
    own = [c for c in nukes if c.school == sim.school]
    pool = own or nukes
    return max(pool, key=lambda c: c.damage)


def _buff_options(sim, s, school):
    """The buffs worth casting toward a hit of `school`.

    Mirrors `make_blade_stack.buffs` exactly -- blades and traps compete
    on value, an aura counts when none is up, prisms are the fallback
    when nothing else is available -- with one change: anything that
    cannot act on `school` is filtered out.

    Prisms are deliberately *not* filtered. A prism converts the hit on
    the way in, and charms are consumed against the card's own school
    before the ward pass converts it (`_strike` computes `charm_mult`
    from `spec_school`, then `_ward_pass` changes it). So an ice blade
    still multiplies an ice hit that a prism is about to turn into myth,
    and a prism is never the wrong school for the hit it converts.
    """
    from w101_sim import castable

    options = [c for c in castable(sim, s, "blade") + castable(sim, s, "trap")
               if buff_matches(c, school)]
    if s.player.aura is None:
        options = options + castable(sim, s, "aura")
    return options or castable(sim, s, "prism")


def school_aware_blade_stack(n_buffs=3):
    """Stack up to `n_buffs` buffs that apply to the nuke, then fire it."""

    def strat(sim, s):
        from w101_sim import castable, effective_pips

        nuke = choose_nuke(sim, s)
        if nuke is None:
            # No hit in hand: bank the best buff on offer, else pass.
            options = castable(sim, s, "blade") + castable(sim, s, "trap")
            return max(options, key=lambda c: c.percent, default=None)

        school = nuke.school
        if pending_for(s, school) < n_buffs:
            options = _buff_options(sim, s, school)
            if options:
                return max(options, key=lambda c: c.percent)

        if sim.afford(s, nuke) and not nuke.x_pips:
            return nuke
        if nuke.x_pips and effective_pips(sim, s, nuke) >= 7:
            return nuke

        # Cannot fire yet -- keep building rather than waste the round.
        options = _buff_options(sim, s, school)
        return max(options, key=lambda c: c.percent, default=None)

    return strat
