"""
Deck-construction search: the outer half of the bilevel problem.

Replaces "pick one of a few hand-built decks" with a LEGAL deck space
(equipped-deck capacity, per-card copy limits, unlocked pool) searched in
two stages:

  1. SCREEN: sample legal candidates from role templates (hits, blades,
     traps, prism, heals, redundancy) and simulate each cheaply with the
     scripted blade-stack policy on paired seeds — a fast proxy scorer.
  2. FINE-TUNE: train the tabular combat policy on the shortlist and
     re-evaluate; score lexicographically by (win rate, speed, SMALLER
     deck) so extra cards must pay for themselves in reliability.

Also provides random_boss() and a held-out generalization check: build
against random opponents, evaluate on unseen seeds, so what transfers is
the RULE ("compact prism deck into same-school walls, redundancy priced
by fizzle risk"), not a memorized answer per boss.

Bilevel rung 3 lives in deck_scorer.py: a ridge surrogate of the
screen, trained from screen_log JSONL rows; pass it as
build_deck(scorer=...) to simulate only the predicted-best slice of
candidates. Rung 4 lives in generalist.py: a deck-conditioned linear-Q
combat policy trained across random (school, deck, boss) triples; pass
it as build_deck(generalist=...) to evaluate the top-k zero-shot
instead of running a per-deck RL fine-tune. The buildable pool is the
curated TRAINED whitelist below (the dump carries no trainability
marker — see the comment on TRAINED); level gating is max(curated
unlock floor, game data's level_restriction).
"""
import math
import random
import re

from w101_sim import (Sim, evaluate, evaluate_paired, make_blade_stack,
                      Boss, OPPOSING, DMG_ENCHANTS, PCT_ENCHANTS,
                      enchant_base, enchant_card, enchanted_deck_size,
                      is_enchanted)
from rl_agent import QAgent


# ------------------------------------------------------------- trainable set
# The dump has NO reliable trainability marker: boss/encounter spells,
# pet cards (Firezilla), mutations (Ice Cat) and cross-school reskins
# (Skeletal Dragon Fire) are all variant='core', and every candidate
# field fails somewhere — training_cost is 0 for school quest spells
# (Fireblade) AND for pet cards; pve_flag marks PvE-only restrictions,
# not trainability; level_restriction is null below ~30. After three
# rounds of heuristic whack-a-mole the honest fix is explicit: the
# trained-spell quest lines, curated by name against the dump's dev
# names, with approximate unlock levels (floors — the data's own
# level_restriction still applies on top via max()). Utility spells the
# loader skips (Steal Charm, spears, Scions) are omitted.
TRAINED = {
    "fire": {
        "Fire Cat": 1, "Fire Elf": 5, "Sunbird": 10, "Fire Shield": 10,
        "Fire Trap": 15, "Fireblade": 20, "Fire Prism": 20, "Link": 20,
        "Heck Hound": 20, "Meteor Strike": 25, "Smoke Screen": 25,
        "Scald": 30, "Immolate": 30, "Helephant": 30, "Wyldfire": 30,
        "Power Link": 35, "Phoenix": 40, "Fuel": 40, "Backdraft": 45,
        "Fire Dragon": 50, "Efreet": 60, "Rain of Fire": 70,
        "Fire from Above": 100,
    },
    "storm": {
        "Thunder Snake": 1, "Lightning Bats": 5, "Storm Shark": 10,
        "Storm Shield": 10, "Disarm": 15, "Storm Trap": 15,
        "Lightning Strike": 15, "Stormblade": 20, "Storm Prism": 20,
        "Cleanse Charm": 20, "Kraken": 25, "Windstorm": 30, "Tempest": 30,
        "Darkwind": 35, "Stormzilla": 35, "Supercharge": 40,
        "Wild Bolt": 40, "Storm Lord": 45, "Triton": 50, "Leviathan": 60,
        "Insane Bolt": 60, "Sirens": 70, "Storm Owl": 90,
    },
    "ice": {
        "Frost Beetle": 1, "Snow Serpent": 5, "Evil Snowman": 10,
        "Snow Shield": 10, "Stun Block": 10, "Ice Trap": 15, "Freeze": 15,
        "Iceblade": 20, "Ice Prism": 20, "Tower Shield": 20,
        "Ice Wyvern": 25, "Ice Armor": 30, "Frostbite": 35,
        "Colossus": 35, "Blizzard": 40, "Legion Shield": 40,
        "Frost Giant": 55, "Frozen Armor": 55, "Snow Angel": 60,
        "Snowball Barrage": 75, "Lord of Winter": 90,
    },
    "myth": {
        "Bloodbat": 1, "Troll": 5, "Cyclops": 10, "Myth Shield": 10,
        "Myth Trap": 15, "Pierce": 20, "Mythblade": 20, "Myth Prism": 20,
        "Humongofrog": 25, "Blinding Light": 25, "Minotaur": 30,
        "Shatter": 40, "Earthquake": 40, "Orthrus": 50, "Medusa": 60,
        "Basilisk": 70,
    },
    "life": {
        "Imp": 1, "Minor Blessing": 1, "Sprite": 5, "Leprechaun": 10,
        "Life Shield": 10, "Pixie": 10, "Life Trap": 15, "Unicorn": 15,
        "Lifeblade": 20, "Life Prism": 20, "Nature's Wrath": 25,
        "Seraph": 30, "Regenerate": 30, "Satyr": 35, "Centaur": 40,
        "Dryad": 40, "Sanctuary": 50, "Forest Lord": 50, "Rebirth": 60,
        "Gnomes": 70, "Spinysaur": 90,
    },
    "death": {
        "Dark Sprite": 1, "Ghoul": 5, "Banshee": 10, "Death Shield": 10,
        "Death Trap": 15, "Sacrifice": 15, "Infection": 20,
        "Deathblade": 20, "Death Prism": 20, "Poison": 25, "Plague": 25,
        "Vampire": 25, "Skeletal Pirate": 35, "Doom and Gloom": 40,
        "Wraith": 50, "Scarecrow": 50, "Skeletal Dragon": 60,
        "Virulent Plague": 60, "Avenging Fossil": 90,
        "Call of Khrulhu": 100,
    },
    "balance": {
        "Scarab": 1, "Scorpion": 5, "Weakness": 10, "Locust Swarm": 10,
        "Sandstorm": 15, "Tri Shield": 15, "Spirit Shield": 15,
        "Helping Hands": 20, "Reshuffle": 20, "Hex": 20,
        "Balanceblade": 20, "Tri Trap": 20, "Spirit Trap": 20,
        "Spectral Blast": 25, "Judgement": 30, "Bladestorm": 30,
        "Tri Blade": 30, "Spirit Blade": 30, "Hydra": 35,
        "Power Play": 35, "Power Nova": 45, "Ra": 60,
        "Availing Hands": 60, "Chimera": 70, "Gaze of Fate": 100,
    },
}

# cross-trained staples any school picks up from other schools' trainers
UNIVERSAL_BUFFS = {
    "Tower Shield": 20, "Sprite": 5, "Pixie": 10, "Reshuffle": 20,
    "Tri Blade": 30, "Tri Trap": 30, "Spirit Blade": 30, "Spirit Trap": 30,
    "Hex": 20, "Curse": 20, "Feint": 30, "Balanceblade": 20,
    "Bladestorm": 30,
}

# defense in depth behind the whitelist: catches a whitelisted name being
# re-dumped with boss-scale values or an internal-marker rename.
_INTERNAL = re.compile(
    r"(-|_|\bNA\b|BOSS|Tutorial|Mutate|Mashup|FUSE|Loremaster|Token|"
    r"Polymorph|Test|\d|\bAdv\b|\bMass\b|"
    r"\s(Sun|Moon|Star|Shadow)$)", re.IGNORECASE)   # enchant variants


def _player_plausible(name, c):
    if _INTERNAL.search(name):
        return False
    if c.kind in ("damage", "drain"):
        per_pip = c.damage if c.x_pips else c.damage / max(c.pips, 1)
        if per_pip > 200:
            return False
    if c.kind == "blade" and c.percent > 0.45:
        return False
    if c.kind in ("trap", "weakness") and abs(c.percent) > 0.75:
        return False
    return True


# Sun enchantments unlock by world, and the owner named the worlds
# rather than the levels: Strong/Giant "early Celestia", Monstrous/
# Gargantuan "mid-level Celestia", Colossal "Zafaria", Epic
# "Arcanum/Polaris". The level floors below are INFERRED from those
# world bands (Celestia 48-60, Zafaria 60-70, Polaris 100-110) and are
# the least-sourced numbers in this module.
ENCHANT_UNLOCK = {"Sharpen Blade": 50, "Potent Trap": 50,
                  "Strong": 48, "Giant": 52, "Monstrous": 55,
                  "Gargantuan": 58, "Colossal": 68, "Epic": 100}


def _best_damage_enchant(level):
    """The strongest flat enchant a wizard of this level owns — players
    carry their best, not their whole collection."""
    ok = [(DMG_ENCHANTS[n], n) for n in DMG_ENCHANTS
          if level is None or ENCHANT_UNLOCK[n] <= level]
    return max(ok)[1] if ok else None


def add_enchanted(pool, level=None):
    """Widen a pool with the enchanted variants a wizard of this level
    could actually apply. Returns a NEW pool; the caller decides whether
    the builder gets to see it.

    Deliberately narrow: the best unlocked flat enchant on damage
    spells, Sharpen Blade on blades, Potent Trap on traps. Offering
    every tier of every enchant would blow up the candidate space
    without modeling anything a player does."""
    out = dict(pool)
    best = _best_damage_enchant(level)
    for name, c in list(pool.items()):
        try:
            if c.kind == "blade" and (level is None
                                      or ENCHANT_UNLOCK["Sharpen Blade"]
                                      <= level):
                e = enchant_card(c, "Sharpen Blade")
            elif c.kind == "trap" and (level is None
                                       or ENCHANT_UNLOCK["Potent Trap"]
                                       <= level):
                e = enchant_card(c, "Potent Trap")
            elif c.kind in ("damage", "drain") and best:
                e = enchant_card(c, best)
            else:
                continue
        except ValueError:
            continue          # per-pip spells and other refusals
        out[e.name] = e
    return out


def legal_pool(cards, school, level=None, mastery=None, enchants=False):
    """Unlocked, deck-buildable cards for a school: the curated TRAINED
    quest line plus the cross-trained universal staples — nothing else,
    so pet cards (Firezilla), mutations and cross-school reskins
    (Skeletal Dragon Fire) can never leak in. `level` gates by
    max(curated unlock floor, game data's level_restriction) — the
    progression knob: legal_pool(cards, 'fire', level=12) is a level-12
    wizard."""
    trained = dict(TRAINED.get(school, {}))
    if mastery:                      # amulet: that school's trained
        trained.update(TRAINED.get(mastery, {}))   # line is packable
    pool = {}
    for name, c in cards.items():
        floor = trained.get(name, UNIVERSAL_BUFFS.get(name))
        if floor is None:                     # not on the trainable list
            continue
        if c.source != "deck" or not _player_plausible(name, c):
            continue
        unlock = max(floor, getattr(c, "level", 1))
        if level is not None and unlock > level:
            continue
        if c.school == school or name in UNIVERSAL_BUFFS or \
                (mastery and c.school == mastery):
            if c.kind in ("damage", "drain", "blade", "trap", "prism",
                          "shield", "heal", "weakness"):
                pool[name] = c
    if enchants:
        pool = add_enchanted(pool, level)
    return pool


def sample_deck(pool, school, boss, rng, capacity=16, copy_limit=3):
    """One legal candidate from a role template with sampled counts."""
    hits = sorted((c for c in pool.values()
                   if c.kind in ("damage", "drain") and not c.x_pips),
                  key=lambda c: -c.damage)[:4]
    xpips = [c for c in pool.values() if c.x_pips and
             c.kind in ("damage", "drain")]
    blades = sorted((c for c in pool.values() if c.kind == "blade"),
                    key=lambda c: -c.percent)[:4]
    traps = sorted((c for c in pool.values() if c.kind == "trap"),
                   key=lambda c: -c.percent)[:4]
    prisms = [c for c in pool.values() if c.kind == "prism"
              and boss.incoming_mult(school) < 1.0]
    if not hits and not xpips:
        return None
    deck = []

    def add(card, n):
        cost = 2 if is_enchanted(card.name) else 1
        room = (capacity - enchanted_deck_size(deck)) // cost
        base = enchant_base(card.name)
        used = sum(1 for x in deck if enchant_base(x) == base)
        deck.extend([card.name] * max(0, min(n, copy_limit - used, room)))

    n_hits = rng.randint(2, 5)
    top = hits[0] if hits else xpips[0]
    add(top, rng.randint(1, min(n_hits, copy_limit)))
    backups = [c for c in hits[1:]] + [c for c in xpips if c is not top]
    rng.shuffle(backups)
    for c in backups:
        if sum(1 for n in deck if pool[n].kind in ("damage", "drain")) \
                >= n_hits:
            break
        add(c, 1)
    for c in rng.sample(blades, k=min(len(blades), rng.randint(0, 3))):
        add(c, rng.randint(1, 2))
    for c in rng.sample(traps, k=min(len(traps), rng.randint(0, 3))):
        add(c, rng.randint(1, 2))
    if prisms and rng.random() < 0.8:
        add(prisms[0], rng.randint(1, 2))
    heals = [c for c in pool.values() if c.kind == "heal"]
    if boss.dmg > 0:
        # lethal opponent: the template offers defensive roles and the
        # screen prices whether they pay for their slots
        shields = [c for c in pool.values() if c.kind == "shield"]
        for c in rng.sample(shields,
                            k=min(len(shields), rng.randint(0, 2))):
            add(c, rng.randint(1, 2))
        if heals and rng.random() < 0.8:
            add(rng.choice(heals), rng.randint(1, 2))
    elif heals and rng.random() < 0.3:
        add(rng.choice(heals), 1)
    return deck if deck else None


def check_legal(deck, capacity, copy_limit):
    """Legality in REAL deck slots and per-SPELL copies.

    An enchanted entry costs two slots (spell + enchantment) and counts
    against its base spell's copy limit, because the physical deck holds
    the plain spell and the enchant is applied in hand. Bit-identical to
    the old rule for any deck with no enchants."""
    from collections import Counter
    return enchanted_deck_size(deck) <= capacity and \
        all(v <= copy_limit
            for v in Counter(map(enchant_base, deck)).values())


def screen(cards, decks, school, boss, rules=None, n=250, base_seed=7000,
           progress=None, player_hp=None, power_pip=None, enemies=None,
           player_stats=None):
    """Cheap proxy scores: scripted policy on paired seeds. A mortal
    `player_hp` screens each candidate under BOTH the race proxy and
    the triage-wrapped survival proxy and keeps the better score —
    burst bosses are raced, chip bosses are triaged, and the screen
    must not presuppose which (a triage-only survival screen went
    blind on a burst boss: every candidate scored 0-6% and the
    ranking was noise)."""
    from w101_sim import make_survival, with_focus
    pols = [make_blade_stack(3)]
    if player_hp is not None:
        pols.append(make_survival(make_blade_stack(3)))
    if enemies:
        # team fights need focus-fire proxies: a target-blind screen
        # scores every candidate ~0 against a healer and ranks noise
        pols = [with_focus(p) for p in pols]
    out = []
    for i, dl in enumerate(decks):
        if progress and i and i % 20 == 0:
            progress(f"screened {i}/{len(decks)} candidates...")
        sim = Sim(dict(cards), dl, school, boss,
                  player_hp=player_hp or 10**9, rules=rules,
                  power_pip=0.85 if power_pip is None else power_pip,
                  enemies=enemies, player_stats=player_stats)
        best = (0.0, 99.0)
        for pol in pols:
            wins, ttk = 0, []
            for j in range(n):
                sim.rng = random.Random(base_seed + j)
                t, won, _ = sim.run(pol)
                if won:
                    wins += 1
                    ttk.append(t)
            cand = (wins / n, sum(ttk) / len(ttk) if ttk else 99.0)
            if (round(cand[0], 2), -cand[1]) > (round(best[0], 2),
                                                -best[1]):
                best = cand
        out.append((best[0], best[1], dl))
    return sorted(out, key=lambda r: (-round(r[0], 2), r[1], len(r[2])))


def fine_tune(cards, dl, school, boss, rules=None, episodes=8000, seed=0,
              player_hp=10**9, sideboard=None, power_pip=None,
              enemies=None, advisor=None, player_stats=None):
    """Train the combat policy on one candidate; return (win, ttk, pol).
    `advisor` is an optional scripted policy (e.g. make_blade_stack(3))
    followed with decaying probability early in training — against
    stochastic living bosses, sparse wins starve tabular MC of credit
    and the prior supplies the missing bootstrap."""
    sim = Sim(dict(cards), dl, school, boss, player_hp=player_hp,
              rules=rules, rng=random.Random(seed), sideboard=sideboard,
              power_pip=0.85 if power_pip is None else power_pip,
              enemies=enemies, player_stats=player_stats)
    agent = QAgent(dict(cards), dl, school, dp_pol=advisor,
                   rng=random.Random(seed + 1))
    for ep in range(episodes):
        frac = ep / episodes
        agent.alpha = 0.30 * (1 - 0.9 * frac)
        dp_w = max(0.0, 0.5 * (1 - 2 * frac)) if advisor else 0.0
        agent.train_episode(sim, eps=max(0.02, 0.3 * (1 - frac)),
                            dp_w=dp_w)
    w, m = evaluate(sim, agent.policy(), n=2000)
    return w, m, agent


def build_deck(cards, school, boss, rules=None, n_candidates=150,
               top_k=5, capacity=16, copy_limit=3, seed=0, log=print,
               level=None, scorer=None, screen_frac=1 / 3,
               screen_log=None, generalist=None, objective="mean",
               player_hp=None, power_pip=None, enemies=None,
               player_stats=None, mastery=None, enchants=False):
    """Two-stage search over the legal deck space. Returns
    (deck, win, ttk, screen_table). `level` gates the unlocked pool.
    A fitted DeckScorer (`scorer`) pre-ranks candidates so only the top
    `screen_frac` slice is simulated; `screen_log` appends the screen's
    (deck, boss, result) rows to a JSONL file as surrogate training
    data; a trained GeneralistQ (`generalist`) evaluates the top-k
    zero-shot instead of running a per-deck RL fine-tune.
    `objective` is the risk stance for the final pick: 'mean' ranks by
    (win, mean TTK, size), 'p90' ranks by (win, p90 TTK, size) — the
    reliability build that prices the slow tail instead of the average
    fight. `player_hp` = None builds for the immortal speed objective;
    a real HP total builds for SURVIVAL — boss damage counts, the
    screen proxy gains triage, and the template offers shields/heals."""
    rng = random.Random(seed)
    pool = legal_pool(cards, school, level=level, mastery=mastery,
                      enchants=enchants)
    # enchanted variants are DERIVED cards: they exist in the pool but
    # not in the caller's registry, and every downstream Sim looks its
    # decklist up by name, so fold them in before anything is simulated
    if enchants:
        cards = {**cards, **pool}
    seen, cands = set(), []
    tries = 0                       # a level-1 pool may only support a
    while len(cands) < n_candidates and tries < n_candidates * 40:
        tries += 1                  # handful of distinct decks — cap it
        dl = sample_deck(pool, school, boss, rng, capacity, copy_limit)
        if dl is None:
            break
        key = tuple(sorted(dl))
        if key in seen or not check_legal(dl, capacity, copy_limit):
            continue
        seen.add(key)
        cands.append(dl)
    if scorer is not None and len(cands) > top_k:
        keep = max(top_k * 2, int(len(cands) * screen_frac))
        cands.sort(key=lambda dl: scorer.rank_key(dl, cards, school,
                                                  boss))
        if log:
            log(f"surrogate pruned {len(cands)} -> {keep} candidates "
                f"(simulating the predicted-best slice only)")
        cands = cands[:keep]
    if log:
        log(f"screening {len(cands)} legal candidates "
            f"(capacity {capacity}, copy limit {copy_limit}"
            + (f", survival at {player_hp} HP" if player_hp else "") + ")")
    table = screen(cards, cands, school, boss, rules, progress=log,
                   player_hp=player_hp, power_pip=power_pip,
                   enemies=enemies, player_stats=player_stats)
    if log and table and round(table[0][0], 2) == 0:
        log("WARNING: screen signal collapsed (best candidate 0%) — "
            "ranking is noise and the size tiebreak then favors SMALL "
            "decks; the encounter may be infeasible for this pool")
    if screen_log:
        import json
        from deck_scorer import boss_to_dict
        with open(screen_log, "a", encoding="utf-8") as f:
            for w0, m0, dl in table:
                f.write(json.dumps(dict(
                    school=school, boss=boss_to_dict(boss), level=level,
                    deck=dl, win=w0, ttk=m0)) + "\n")
    best = None
    if generalist is not None and log:
        log("evaluating top-k with the generalist policy "
            "(zero-shot, no per-deck training)")
    php = player_hp or 10**9
    for w0, m0, dl in table[:top_k]:
        if generalist is not None:
            pol = generalist.policy()
            gsim = Sim(dict(cards), dl, school, boss, player_hp=php,
                       rules=rules, power_pip=0.85 if power_pip is None else power_pip,
                       player_stats=player_stats)
            w, m = evaluate(gsim, pol, n=2000)
        else:
            w, m, agent = fine_tune(cards, dl, school, boss, rules,
                                    seed=seed, player_hp=php,
                                    power_pip=power_pip, enemies=enemies,
                                    player_stats=player_stats)
            pol = agent.policy()
        tail = m
        if objective == "p90":
            psim = Sim(dict(cards), dl, school, boss, player_hp=php,
                       rules=rules, power_pip=0.85 if power_pip is None else power_pip,
                       player_stats=player_stats)
            st = evaluate_paired(psim, {"pol": pol}, n=1000)["pol"]
            w, m = st["win_rate"], st["mean_ttk"]
            tail = st["p90_ttk"]
            if math.isnan(tail):
                tail = 99.0
        if log:
            extra = f"  p90 {tail:5.2f}" if objective == "p90" else ""
            log(f"  size {len(dl):>2}  screen {w0*100:3.0f}%/{m0:5.2f}  "
                f"RL {w*100:5.1f}%/{m:5.2f}{extra}  {sorted(set(dl))}")
        score = (round(w, 2), -tail, -len(dl))
        if best is None or score > best[0]:
            best = (score, dl, w, m)
    return best[1], best[2], best[3], table


# ---------------------------------------------------------------- opponents

def random_boss(rng, name="rand"):
    """Randomized opponent for generalization training/testing."""
    school = rng.choice(("fire", "ice", "storm", "myth", "life", "death"))
    hp = rng.randrange(500, 8001, 100)
    b = Boss(f"{name}-{school}-{hp}", hp, school, 0)
    b.resist_map = {school: round(rng.uniform(0.2, 0.8), 2)}
    if rng.random() < 0.5:
        b.resist_map["*"] = round(rng.uniform(0.0, 0.3), 2)
    opp = OPPOSING.get(school)
    b.boost_map = {opp: round(rng.uniform(0.0, 0.5), 2)} if opp else {}
    return b


if __name__ == "__main__":
    from data_full import load_spells_full, load_bosses_full, LIVE_DECKS, \
        LIVE_RULES
    import copy as _copy
    cards = load_spells_full()
    bosses, _ = load_bosses_full()

    print("== deck search: death vs Jade Oni (live rules) ==")
    boss = _copy.copy(bosses["Jade Oni"])
    boss.dmg = 0
    dl, w, m, _ = build_deck(cards, "death", boss, LIVE_RULES,
                             n_candidates=120, top_k=4, seed=0)
    print(f"built (size {len(dl)}): win {w*100:.1f}%  ttk {m:.2f}")
    hand = LIVE_DECKS["death"]["oneshot"]
    w0, m0, _ = fine_tune(cards, hand, "death", boss, LIVE_RULES, seed=0)
    print(f"hand-built oneshot (size {len(hand)}): win {w0*100:.1f}%  "
          f"ttk {m0:.2f}")

    print("\n== generalization: build vs 3 random bosses, held-out seeds ==")
    rng = random.Random(42)
    for i in range(3):
        rb = random_boss(rng)
        dl, w, m, _ = build_deck(cards, "ice", rb, LIVE_RULES,
                                 n_candidates=60, top_k=3, seed=i, log=None)
        prisms = sum(1 for n in dl if cards[n].kind == "prism")
        print(f"{rb.name:<18} resist={rb.resist_map} -> size {len(dl)}, "
              f"prisms {prisms}, win {w*100:.1f}%, ttk {m:.2f}")
