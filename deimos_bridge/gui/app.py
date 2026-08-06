"""The window.

Holds one `Telemetry` per wizard, one tab per panel, and the controls for
configuring and starting a run. Training happens on a worker thread so
the window stays responsive -- a Q-learning run is minutes of solid CPU
and freezing the UI for it would make the progress readout pointless.

    python -m deimos_bridge.gui              # live (needs Windows + game)
    python -m deimos_bridge.gui --demo       # canned fight, runs anywhere

**Up to four wizards.** The `wizards` box says how many Wizard101 clients
to drive, and the `wizard` selector beside it says which one the
controls, the tabs and the Train button are talking about. Every wizard
gets its own school, deck, policy, gear, trained table and telemetry,
because all six of those genuinely differ -- a Q table is keyed on its
own decklist, and a party of four identical wizards is the one party
worth nothing.

With more than one, the run also becomes a *hivemind*: the wizards agree
each round before any of them clicks a card, so a trap laid by one is
cashed by the next and nobody fires into a mob that is already dead. The
Party tab shows what that agreement changed. See
`deimos_bridge/hivemind.py`.
"""
import argparse
import sys

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QComboBox, QFileDialog, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QProgressBar, QPushButton, QSpinBox, QCheckBox,
                             QTabWidget, QVBoxLayout, QWidget)

from ..hotkeys import DEFAULTS as HOTKEY_DEFAULTS, KEY_CHOICES as HOTKEY_CHOICES
from ..telemetry import Telemetry
from .deckpicker import pick_deck
from .live import LiveWorker, SeatConfig
from .panels import (BoardPanel, DecisionsPanel, HivemindPanel,
                     LearningPanel, ModelPanel, NamingPanel, RunPanel,
                     _label, scrollable)
from .theme import PALETTE, stylesheet

#: How many game clients one window will drive. Four is the game's own
#: limit -- a battle circle seats four wizards -- so it is the ceiling
#: rather than a chosen one.
MAX_WIZARDS = 4

SCHOOLS = ["fire", "ice", "storm", "myth", "life", "death", "balance"]
#: `ttk-lookahead` first, so it is what the window starts on. It keys on
#: nothing -- it simulates the fight rather than looking it up -- so it
#: cannot be out of band on a game with 1,912 creatures in it, and it
#: measured best across a real-creature benchmark (63.4% on trained
#: boards, 68.1% on held-out ones, at 61 ms a fight). The table stays on
#: the menu as an overlay for a board you have actually trained.
POLICIES = ["ttk-lookahead", "school-aware", "blade-stack(3)",
            "blade-stack(2)", "nuke-asap", "trained (Q)"]


def _duration(seconds):
    """A rough time, in the units a person would say it in.

    Deliberately coarse. An estimate accurate to the second would be
    claiming a precision it does not have -- the run's own checkpoints
    are irregular -- and "about 4 min" is the whole of what the number
    is for.
    """
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


class TrainWorker(QThread):
    """Runs `rl_agent.train_agent` off the UI thread."""

    progress = pyqtSignal(int, int, float, float)   # ep, total, kill%, ttk
    snapshot = pyqtSignal(int, float, float)        # ep, kill rate, ttk
    #: episodes done, episodes total, seconds left (<0 = not yet known).
    #: Carries no measurement -- that is what `snapshot` is for, and it
    #: costs a 2,000-fight evaluation. This one only counts, which is why
    #: it can fire while the checkpoints are 5,000 episodes apart.
    tick = pyqtSignal(int, int, float)
    #: what the run is doing when it is not counting episodes
    stage = pyqtSignal(str)
    #: (chosen rollout continuation, {name: kill rate}) for this deck
    continuation = pyqtSignal(str, object)
    #: (trained kill rate, heuristic kill rate) on the same eval board.
    #: The comparison the window never made — a table that keys 95% of
    #: boards and plays them worse than the fallback reads as a success
    #: on every number the Learning tab used to show.
    verdict = pyqtSignal(float, float)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)
    #: the preflight's verdict, distinct from a crash: "this board is
    #: not winnable at these settings" is a finding about the fight,
    #: and a status bar that renders it as "training failed" reads as
    #: a bug in the tool rather than a fact about the board
    refused = pyqtSignal(str)

    def __init__(self, cards, deck, school, episodes, player_hp=800,
                 boss_hp=1200, player_stats=None, n_enemies=1,
                 mob_hps=None, generalize=True, mob_schools=None,
                 mob_damage=0, party_size=1):
        super().__init__()
        self.cards, self.deck = cards, deck
        self.school, self.episodes = school, episodes
        self.player_hp = player_hp
        self.boss_hp = boss_hp
        #: biggest board to train for. With `generalize` on, every count
        #: from 1 to this is sampled -- which is what stops the mob count
        #: from being load-bearing, since `Featurizer.key` only carries
        #: its targeting tuple on a multi-enemy board and a table that
        #: never saw two mobs cannot key a two-mob fight at all.
        self.n_enemies = max(1, int(n_enemies))
        #: healths seen in real fights, used to centre the range rather
        #: than to pin the board
        self.mob_hps = [h for h in (mob_hps or []) if h > 0]
        #: schools seen in real fights. This used to be the literal
        #: "ice" for every training mob regardless of anything, which
        #: is not a cosmetic default: `Boss.resist_own` is 0.40, so an
        #: ice wizard was training against a board that resisted 40% of
        #: every card in the deck. On the operator's settings that made
        #: the board unwinnable by *every* policy in the repo, at every
        #: enemy damage down to zero -- which is what a flat 0% kill
        #: rate across 40,000 episodes was.
        self.mob_schools = [s for s in (mob_schools or []) if s]
        #: measured incoming damage per enemy per round, off the live
        #: fight. 0 means "never measured", and only then is the wizard's
        #: own health used as a stand-in.
        self.mob_damage = int(mob_damage or 0)
        #: wizards in the circle. It only reaches the *unmeasured*
        #: incoming prior and the deck-ceiling refusal, because those are
        #: the two places a solo assumption is quietly wrong for a party
        #: member -- see `enemy_damage` and `preflight`.
        self.party_size = max(1, int(party_size or 1))
        #: Resample the board every episode instead of training one.
        #: On by default, because the alternative is retraining before
        #: every fight -- which needs you to know the board before you
        #: can learn to fight it, and is not a workflow anyone will keep
        #: doing.
        self.generalize = bool(generalize)
        #: the wizard's real gear, read off the client. Training without
        #: it prices every hit as though the wizard were naked, so the Q
        #: table is learned for a fight nobody is going to play.
        self.player_stats = dict(player_stats or {})
        #: set when the episode loop starts, so the estimate measures
        #: episodes rather than the warm-start solve that precedes them
        self._t0 = None

    def _on_stage(self, name):
        if name == "training":
            import time
            self._t0 = time.monotonic()
        self.stage.emit(name)

    def _on_tick(self, done, total):
        """Episodes done, and how long the rest will take.

        The estimate is elapsed/fraction rather than a per-episode rate,
        which amortises the periodic checkpoints instead of pretending
        they are free -- each one is a 2,000-fight evaluation, so a rate
        measured between them would promise a finish it cannot meet.
        """
        import time

        left = -1.0
        if self._t0 is not None and done > 0:
            spent = time.monotonic() - self._t0
            left = max(0.0, spent * (total - done) / done)
        self.tick.emit(done, total, left)

    def hp_range(self):
        """The band of mob health to train across.

        Wide on purpose. The health enters the state key as a bucket
        (`hp // 250`), so what has to be covered is a span of buckets, not
        a number -- and a band that only just covers the last fight would
        put the next one outside it again.
        """
        centre = max(1, int(self.boss_hp))
        lo, hi = int(centre * 0.4), int(centre * 1.8)
        if self.mob_hps:
            # Stretch to include what has actually been fought, with room
            # either side; observed mobs centre the band, they do not
            # define its edges.
            lo = min(lo, int(min(self.mob_hps) * 0.6))
            hi = max(hi, int(max(self.mob_hps) * 1.6))
        return max(1, lo), max(lo + 1, hi)

    #: Below this kill rate a board has effectively no winning line, so
    #: episodes spent there collect no reward and teach nothing. Not zero:
    #: a board the deck clears one time in twenty is still a board, and
    #: the edge of the envelope is exactly where the interesting states
    #: are.
    ENVELOPE_FLOOR = 0.15

    def envelope(self, dmg=None, n=140, on_probe=None):
        """{mob count: (lo, hi)} the deck can actually win.

        The answer to "why do I have to train for a specific health".
        You should not: the band was `mob HP` x0.4 to x1.8, so typing 235
        bought 94-423 and a 480 HP mob fell off the end and keyed nothing.

        The right band is not a wider guess either -- it is the range
        this deck, at this health, can actually clear, and that is
        measurable. It is also sharply different per mob count: measured
        on one starter ice deck at 1,022 health, one mob is winnable to
        1,400 HP, two to 700, three to 480. A single band across all
        counts either spends most of a three-mob budget on boards with no
        winning line -- the same zero-reward trap the hardcoded school
        was -- or caps the one-mob range at a third of what the deck can
        clear.

        Found by bisection on a scripted policy rather than by training,
        because it is a property of the deck and the wizard, not of the
        table. Costs about a second per mob count.
        """
        from w101_sim import Boss, Sim, evaluate
        from ..policies import school_aware_blade_stack

        dmg = self.enemy_damage() if dmg is None else dmg
        schools = self.school_pool()
        policy = school_aware_blade_stack(3)

        def wins(hp, count):
            import random as _random

            board = Boss(name="probe", hp=hp, school=schools[0], dmg=dmg)
            extra = [Boss(name=f"probe {i}", hp=hp,
                          school=schools[i % len(schools)], dmg=dmg)
                     for i in range(1, count)]
            sim = Sim(self.cards, self.deck, self.school, board,
                      player_hp=self.player_hp,
                      player_stats=self.player_stats, enemies=extra,
                      # Seeded per probe point, so the same deck at the
                      # same settings bisects to the SAME bands every
                      # run. The bands are stamped onto the trained
                      # table and quoted back at the operator ("above
                      # the 40-1,500 band this table was trained on");
                      # edges that wobbled with each run's evaluation
                      # luck made those messages disagree between
                      # sessions about what the deck could clear.
                      rng=_random.Random(hash((count, hp))))
            kill, _ = evaluate(sim, policy, n=n)
            if on_probe:
                on_probe(count, hp, kill)
            return kill >= self.ENVELOPE_FLOOR

        bands, floor = {}, 40
        for count in range(1, self.n_enemies + 1):
            if not wins(floor, count):
                continue          # cannot clear even the smallest board
            lo, hi = floor, 6000
            while hi - lo > 60:
                mid = (lo + hi) // 2
                if wins(mid, count):
                    lo = mid
                else:
                    hi = mid
            # Up to the top of the bucket the frontier lands in. The
            # key cannot tell 365 from 480 -- both are `hp // 250 == 1`
            # -- so stopping the band at 365 trains part of a bucket and
            # then reports a 480 mob as outside it, which is a
            # distinction the model does not make. Costs at most one
            # bucket of ground the deck clears less often; buys a band
            # whose edges mean the same thing to the trainer and to the
            # state key.
            from rl_agent import HP_BUCKET
            hi = (lo // HP_BUCKET + 1) * HP_BUCKET
            # ...and at least as far as the boards you say you will meet.
            # Typing 780 into "biggest mob HP" and being told a 480 mob
            # is outside the trained band is the box doing nothing, and
            # that was fair to call annoying: the envelope answered
            # "what can this deck reliably win" when the question is
            # "what will this table be asked about".
            #
            # Training past the winnable frontier used to be a bad trade
            # -- more coverage measured as worse play -- because a state
            # visited twice drove exactly like a state visited ten
            # thousand times. With `TrainedPolicy.MIN_VISITS` gating on
            # evidence, thin states hand the round to the heuristic on
            # their own, so the only cost of reaching further is
            # episodes.
            want = max([int(self.boss_hp)] + [int(h) for h in self.mob_hps])
            hi = max(hi, (want // HP_BUCKET + 1) * HP_BUCKET)
            bands[count] = (floor, hi)
        return bands

    def compare(self, agent, bands, dmg, schools, n=300):
        """(trained, heuristic) kill rate, where the board can tell them apart.

        Scoring at one point is why "98% against 100%" read as a tie. On
        an easy board every policy is at the ceiling and the comparison
        ranks nothing; the same two policies on a board near the edge of
        what the deck can clear are 30% against 76%. Measured across one
        deck: at 235 HP x2 every policy scored 96-100%, at 480 HP x2 the
        table scored 30% and the heuristic 76%, at 620 HP x2 it was 1%
        against 61%.

        So this walks the envelope and reports the point of **largest
        disagreement** rather than an average or an endpoint. An average
        would dilute the informative boards with the saturated ones,
        which is the same mistake in a different shape.
        """
        from w101_sim import Boss, Sim, evaluate_paired
        from ..policies import school_aware_blade_stack, trained_policy

        counts = sorted(bands) or [self.n_enemies]
        probes = []
        for count in counts:
            lo, hi = bands.get(count, self.hp_range())
            for frac in (0.35, 0.6, 0.85):
                probes.append((int(lo + (hi - lo) * frac), count))
        if not probes:
            probes = [(self.boss_hp, self.n_enemies)]

        worst = (0.0, 0.0, -1.0)          # trained, rival, gap
        for hp, count in probes:
            board = Boss(name="probe", hp=hp, school=schools[0], dmg=dmg)
            extra = [Boss(name=f"probe {i}", hp=hp,
                          school=schools[i % len(schools)], dmg=dmg)
                     for i in range(1, count)]
            sim = Sim(self.cards, self.deck, self.school, board,
                      player_hp=self.player_hp,
                      player_stats=self.player_stats, enemies=extra)
            # The wrapped policy, because that is what plays live -- the
            # raw table passes on an unseen state, which is not a move
            # anyone makes on purpose. Paired seeds, because both the
            # verdict AND the choice of which probe to report ride on
            # the trained-minus-rival difference: on independent streams
            # a lucky run inflates a gap and the largest-disagreement
            # rule then reports the probe with the loudest noise.
            stats = evaluate_paired(
                sim, {"trained": trained_policy(agent),
                      "rival": school_aware_blade_stack(3)}, n=n)
            t = stats["trained"]["win_rate"]
            r = stats["rival"]["win_rate"]
            if abs(t - r) > worst[2]:
                worst = (t, r, abs(t - r))
        return worst[0], worst[1]

    def probe_boards(self, bands, schools, fracs=(0.55, 0.8)):
        """Boards to compare policies on: hard enough to tell them apart.

        Taken from the envelope rather than from the settings, because a
        board every candidate clears ranks nothing. Measured directly:
        on probe boards near the ceiling the five continuations scored
        97.5-99.0% -- a 1.5 point spread that is noise -- while on
        boards near the edge of the same deck's envelope the spread was
        60.0-68.0%.
        """
        out = []
        for count, (lo, hi) in sorted((bands or {}).items()):
            for frac in fracs:
                hp = max(lo, int(lo + (hi - lo) * frac))
                out.append((hp, count, schools[0]))
        return out or [(self.boss_hp, self.n_enemies, schools[0])]

    def describe_envelope(self, bands):
        if not bands:
            return ("this deck cannot clear a single mob at these settings "
                    "— nothing to train")
        parts = [f"{n} mob{'s' if n > 1 else ''} to {hi:,} HP"
                 for n, (_lo, hi) in sorted(bands.items())]
        return "training over " + ", ".join(parts)

    def school_pool(self):
        """The schools to draw training mobs from.

        Observed schools first -- if the run has actually fought a death
        boss, train against death. Failing that, all seven, because a
        table trained against one school has learned that school's
        resist as if it were a property of the game. Never the wizard's
        own school on its own: that is the 0.40 own-school resist, the
        worst matchup available, and it was the shipped default.
        """
        from rl_agent import MOB_SCHOOLS

        seen = [s for s in dict.fromkeys(self.mob_schools) if s in MOB_SCHOOLS]
        return seen or list(MOB_SCHOOLS)

    def board_schools(self):
        """Schools for the fixed evaluation board, one per mob.

        The wizard's own school is excluded when the real one is not
        known. `Boss.resist_own` is 0.40, so a same-school mob is the
        worst matchup in the game, and putting one on a *guessed* board
        makes the guess harder than any fight it stands in for. Cycling
        the full seven-school pool did exactly that: an ice wizard's
        two-mob eval board came out "fire + ice".
        """
        pool = self.school_pool()
        if len(pool) > 1:
            pool = [s for s in pool if s != self.school] or pool
        return [pool[i % len(pool)] for i in range(self.n_enemies)]

    def damage_ceiling(self):
        """The most damage this decklist could ever deliver, optimistically.

        An upper bound, computed generously on purpose: every damage card
        lands, gear applies, and every buff in the deck is spent on the
        biggest hits available. Nothing in a real fight beats it -- draw
        order, accuracy and the enemy all take away from it.

        It exists because a board can be unwinnable for a reason none of
        the knobs in this window addresses. A 9-card deck of 3 Frost
        Beetles, 3 Ice Traps and 3 Snow Serpents tops out near 1,080
        damage; a board of 780 + 624 has 1,404 health. No mob HP, mob
        count, enemy school or incoming-damage setting changes that, and
        the run that prompted this said "lower the mob HP or the mob
        count, raise your health, or check the enemy school" -- four
        suggestions, none of them the answer. Adding the three Evil
        Snowmen the deck had lost takes the same board from 0% to 98%.
        """
        gear = 1.0 + (self.player_stats.get("damage") or {}).get(
            self.school, 0.0)
        hits, buffs = [], []
        for name in self.deck:
            card = self.cards.get(name)
            if card is None:
                continue
            if card.kind in ("damage", "drain") and card.damage:
                hits.append(card.damage * gear)
            elif card.kind in ("blade", "trap", "prism") and card.percent > 0:
                buffs.append(card.percent)
        hits.sort(reverse=True)
        buffs.sort(reverse=True)
        total = sum(hits)
        # Each buff spent on the biggest remaining hit. One per hit --
        # the engine allows only one of a stacking identity per strike.
        for i, pct in enumerate(buffs[:len(hits)]):
            total += hits[i] * pct
        return total

    def board_hps(self):
        """The fixed board, for `generalize=False` and for evaluation.

        Spread rather than uniform: mobs that all start on the same
        health make the weakest index 0 in every opening state, and a
        real board of 515 beside 390 opens in the half that was never
        visited. Measured at 0% coverage.
        """
        if len(self.mob_hps) == self.n_enemies:
            return [max(1, int(h)) for h in self.mob_hps]
        # 100%, 80%, 60%, ... of the headline number.
        return [max(1, int(self.boss_hp * (1.0 - 0.2 * i)))
                for i in range(self.n_enemies)]

    def enemy_damage(self):
        """How hard a training mob hits, per round.

        This was `max(30, player_hp // 12)` -- the *wizard's* health
        divided by a constant, which is not a model of an enemy at all.
        Its specific defect is that it makes the fight exactly as hard
        whatever the wizard wears: the death clock sits at 12/n_mobs
        rounds at every level, so a table trained on it can never learn
        that more health buys more turns. Measured: kill rate stays in a
        68-80% band across a 15x range of player health, where a fixed
        enemy number gives the correct 0% -> 92% progression.

        Measured incoming damage first -- the live run counts it every
        round. Then the wizard's health, which is at least in the right
        order of magnitude for the level ranges where the two grow
        together, and is what this used to always do.

        Both are "damage to *this* wizard, per enemy, per round", and in
        a party that is not the board's output -- an enemy picking one
        of four wizards to hit lands on this one a quarter of the time.
        The measured number already knows that, because it is read off
        this wizard's own health bar. The stand-in did not, so a party
        member with no fight measured yet trained against a board
        hitting four times as hard as the one it plays: the death clock
        sits four times too early, and every line that needs a setup
        turn is scored as though it dies before the payoff. Divided for
        the same reason `WizAiBackend._apportion_incoming` divides.
        """
        if self.mob_damage and self.mob_damage > 0:
            return int(self.mob_damage)
        return max(30, self.player_hp // (12 * self.party_size))

    def deck_advice(self, deficit):
        """Name cards that would close a damage deficit, from the pool.

        The boards no policy can win are lost in the deck box, not in
        play -- 19 of 32 game-spanning boards were lost by every policy
        in the repo, and better play cannot buy a single one of them.
        `deck_builder.legal_pool` knows what this school can actually
        put in a deck, so the refusal can say "add these" instead of
        "add damage cards", which is the difference between advice and
        a shrug.
        """
        try:
            from deck_builder import legal_pool

            pool = legal_pool(self.cards, self.school)
        except Exception:
            pool = {n: c for n, c in self.cards.items()
                    if getattr(c, "school", "") == self.school}
        gear = 1.0 + (self.player_stats.get("damage") or {}).get(
            self.school, 0.0)
        have = {}
        for name in self.deck:
            have[name] = have.get(name, 0) + 1
        picks, closed = [], 0.0
        # Cheap pips first, biggest damage within a cost. Sorting by
        # raw damage suggested Lord of Winter to a level-5 wizard --
        # the pool is not level-gated, but pip cost is a decent proxy
        # for "castable by whoever is running this deck".
        hitters = sorted(
            (c for n, c in pool.items()
             if c.kind in ("damage", "drain") and c.damage
             and not c.x_pips and c.pips <= 4
             and have.get(c.name, 0) < 3),
            key=lambda c: (c.pips, -c.damage))
        for c in hitters:
            room = 3 - have.get(c.name, 0)
            take = min(room, max(1, int(deficit // max(1, c.damage * gear))))
            for _ in range(take):
                if closed >= deficit * 1.15:
                    break
                picks.append(c.name)
                closed += c.damage * gear
            if closed >= deficit * 1.15:
                break
        if not picks:
            return ""
        counts = {}
        for n in picks:
            counts[n] = counts.get(n, 0) + 1
        listed = ", ".join(f"{v}x {k}" for k, v in counts.items())
        return (f" Adding {listed} would close the ~{deficit:,.0f} damage "
                f"gap with room to spare.")

    def preflight(self, board, extra, n=200):
        """(feasible, note) -- can anything win this board?

        The check that would have saved the operator 40,000 episodes.
        A learner on an unwinnable board does not fail loudly: it
        explores normally, builds tens of thousands of Q entries,
        collects zero reward and draws a flat line. Scoring one scripted
        policy on the same evaluation board separates "the learner did
        not learn" from "no policy can win this", which are different
        problems with different fixes.
        """
        from w101_sim import Boss, Sim, evaluate      # noqa: F401
        from ..policies import school_aware_blade_stack

        try:
            sim = Sim(self.cards, self.deck, self.school, board,
                      player_hp=self.player_hp,
                      player_stats=self.player_stats, enemies=extra)
            kill, _ttk = evaluate(sim, school_aware_blade_stack(3), n=n)
        except Exception:
            return True, ""       # never block a run over the check itself
        if kill > 0.0:
            return True, ""
        # The scripted canary is the WEAKEST policy in the repo, and
        # refusing on its word alone overclaims badly: measured on a
        # live operator's board (480 + 235 at 77/round, a heal-less
        # fire deck), the canary won 0.0% of 500 while greedy_ttk --
        # the policy that actually drives live fights -- won 60.4%.
        # "This board cannot be won" must be checked against the
        # strongest cheap policy before it is said. The escalation
        # only runs when the canary reads zero, so healthy boards
        # never pay for it.
        try:
            from ..policies import greedy_ttk
            kill2, _ = evaluate(sim, greedy_ttk(6), n=max(60, n // 3))
            if kill2 > 0.0:
                return True, ""
        except Exception:
            return True, ""
        mobs = " + ".join(f"{b.hp:,} HP {b.school}" for b in [board] + extra)
        health = sum(b.hp for b in [board] + extra)
        head = (f"this board cannot be won at these settings, so training "
                f"it would draw a flat 0% however long it ran.\n\n"
                f"Board: {mobs}, each hitting for {board.dmg}/round, "
                f"against {self.player_hp:,} HP.\n\n")

        # The matchup, named. A fire wizard's all-fire deck against a
        # fire board loses ~40% of every hit to own-school resist, and
        # no amount of same-school deck advice fixes a school wall --
        # `deck_advice` draws from this school's own pool, so without
        # this line the refusal recommends more of the resisted school.
        # Measured on the fight that earned the message (Alicane
        # Swiftarrow + Magma Man vs a level-7 fire wizard): 0.2% for
        # every policy in the repo; the deck is the reason, and the
        # only real fixes are off-school.
        pool = [b for b in [board] + extra if b.hp > 0]
        wall = sum(b.incoming_mult(self.school) * b.hp for b in pool) \
            / max(1, sum(b.hp for b in pool))
        matchup = ""
        if wall <= 0.75:
            own = [self.cards[n] for n in self.deck
                   if n in self.cards
                   and self.cards[n].kind in ("damage", "drain")]
            share = (sum(1 for c in own if c.school == self.school)
                     / len(own)) if own else 0.0
            if share >= 0.7:
                matchup = (
                    f"\n\nThe matchup is most of it: this board takes "
                    f"only ~{wall * 100:.0f}% from {self.school} damage, "
                    f"and this deck's damage is "
                    f"{'all' if share == 1.0 else 'mostly'} "
                    f"{self.school}. Nothing from the {self.school} "
                    f"pool fixes a school wall — off-school damage "
                    f"does: treasure cards from the bazaar, or a wand "
                    f"that hits in another school.")

        # Name the cause rather than listing the knobs. When the deck
        # simply cannot deliver the board's health, none of the knobs is
        # the answer and suggesting them sends you round in circles.
        ceiling = self.damage_ceiling()
        # In a party this wizard is not asked to deliver the whole
        # board. Three other wizards are hitting it, and refusing to
        # train a deck for a board its share of which it clears
        # comfortably is a refusal to the wrong question -- the one the
        # simulator can ask, rather than the one being played.
        owed = health / self.party_size
        if ceiling < owed:
            hint = self.deck_advice(owed - ceiling)
            share = ("" if self.party_size == 1 else
                     f" Your share of it, across {self.party_size} "
                     f"wizards, is about {owed:,.0f}.")
            return False, (
                head +
                f"Your deck is the reason, not the board. Every damage "
                f"card in it, landing, with your gear, and with every "
                f"buff spent on the biggest hits, comes to about "
                f"{ceiling:,.0f} damage — and this board has {health:,} "
                f"health.{share} No play order wins that, and no mob HP, "
                f"mob count or enemy school setting changes it.{hint}"
                f"{matchup}\n\n"
                f"Add damage cards, or train against a smaller board.")

        alone = ("" if self.party_size == 1 else
                 f"\n\nThis is a SOLO verdict: the simulator has one "
                 f"wizard in it, and you are training one of "
                 f"{self.party_size}. The incoming damage is already "
                 f"priced as this wizard's share, but the other "
                 f"{self.party_size - 1} wizards' damage is not — so a "
                 f"board your party clears together can still be refused "
                 f"here. Train against a smaller board and the table will "
                 f"still key the real one, because the trained band is "
                 f"stretched to cover the boards you say you will meet.")
        return False, (
            head +
            f"Neither the scripted policy ({n} fights) nor the search "
            f"policy could win one, and your deck could deliver about "
            f"{ceiling:,.0f} damage against its {health:,} health — so "
            f"this is a race it loses on time, not on damage. Lower the "
            f"mob HP or the mob count, raise your health, or check the "
            f"incoming damage: at {board.dmg}/round each you last "
            f"{self.player_hp / max(1, board.dmg * (1 + len(extra))):.0f} "
            f"rounds.{matchup}{alone}")

    def run(self):
        try:
            from rl_agent import train_agent
            from w101_sim import Boss
            # MORTAL, deliberately. train_agent defaults to
            # player_hp=10**9, and Featurizer.key writes -1 into the
            # health slot for an immortal fight and a real bucket
            # otherwise -- so a policy trained on the default shares no
            # state at all with a live wizard, its Q table reads zero
            # everywhere, and greedy falls through to PASS every turn.
            dmg = self.enemy_damage()
            hps = self.board_hps()
            schools = self.board_schools()
            sampler = None
            bands = {}
            if self.generalize:
                from rl_agent import make_board_sampler
                # Discovered, not typed. See `envelope`.
                self.stage.emit("finding what this deck can clear")
                bands = self.envelope(dmg)
                self.stage.emit(self.describe_envelope(bands))
                sampler = make_board_sampler(
                    schools[0], self.hp_range(), max_mobs=self.n_enemies,
                    dmg=dmg, schools=self.school_pool(), bands=bands)

            board = Boss(name="training dummy", hp=hps[0],
                         school=schools[0], dmg=dmg)
            extra = [Boss(name=f"training minion {i}", hp=hp,
                          school=schools[i], dmg=dmg)
                     for i, hp in enumerate(hps[1:], 1)]

            # Before 40,000 episodes: can this board be won at all? A
            # learner given a board with no winning line explores
            # normally, builds a table of tens of thousands of entries,
            # collects zero reward, and reports a flat 0% -- which reads
            # as "training failed" when it means "this fight is
            # arithmetically impossible". Three seconds of checking beats
            # twenty minutes of it.
            feasible, note = self.preflight(board, extra)
            if not feasible:
                self.stage.emit(note)
                self.refused.emit(note)
                return

            # The rollout's continuation, picked for this deck. It is one
            # small policy reused on every board and every rollout, so it
            # needs no coverage of anything -- which makes it the one
            # place learning fits a game with 1,912 creatures in it.
            # Measured worth ~14 points of kill rate, and deck-specific:
            # the choice that is +5.2 on one deck is -7.6 on another.
            # Seconds, against episodes.
            self.stage.emit("tuning the search for this deck")
            try:
                from ..policies import choose_search
                picked, horizon, scores = choose_search(
                    self.cards, self.deck, self.school,
                    self.probe_boards(bands, schools), n=60, dmg=dmg)
                from ..policies import driver_name
                self.continuation.emit(
                    f"{picked} @ horizon {horizon} @ driver {driver_name()}",
                    dict(scores))
                self.stage.emit(
                    f"search tuned: {picked}, horizon {horizon}")
            except Exception:
                pass          # a nicety; never worth failing a train over

            self.stage.emit("training")

            agent, sim = train_agent(
                self.cards, self.deck, self.school, board, enemies=extra,
                episodes=self.episodes, player_hp=self.player_hp,
                player_stats=self.player_stats, board_sampler=sampler,
                on_snapshot=lambda ep, kill, ttk:
                    self.snapshot.emit(ep, kill, ttk),
                on_tick=self._on_tick, on_stage=self._on_stage)
            from w101_sim import evaluate
            self.stage.emit("scoring the trained table")
            kill, ttk = evaluate(sim, agent.policy(), n=800)
            # Against the heuristic it would displace, on the same board.
            # Coverage was the only number this ever reported, and
            # coverage is not competence: a table can key 95% of boards
            # and play every one of them worse than the fallback it is
            # keeping out of the driver's seat. Measured on the
            # operator's board: 89% coverage, 6.4% kill, against 82.4%
            # for the heuristic. That comparison is the answer to "is
            # the model stupid?", and nothing in the window had it.
            # What the table was actually trained on, stamped onto it
            # so a live miss can say which fact it did not recognise.
            # "it always goes to fallback" is not actionable; "a 1,500 HP
            # mob is above the 276-1,242 band this table was trained on"
            # is, and points at a box already on screen.
            lo, hi = self.hp_range()
            agent.trained_on = {"hp": (lo, hi), "mobs": self.n_enemies,
                                "schools": list(self.school_pool()),
                                "player_hp": self.player_hp,
                                # per-count, when the envelope was
                                # discovered: a miss can then say which
                                # count's band it fell outside
                                "bands": dict(bands)}
            tk, rk = self.compare(agent, bands, dmg, schools)
            self.verdict.emit(tk, rk)
            self.progress.emit(self.episodes, self.episodes, kill * 100, ttk)
            self.finished_ok.emit(agent)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DeckWorker(QThread):
    """Runs `deck_builder.build_deck` for the observed board.

    The last lever: 19 of 32 game-spanning boards were lost by every
    policy in the repo, and no amount of better play buys one of them --
    those fights are lost in the deck box. The repo has had a two-stage
    deck search all along (`build_deck`: sample the legal pool, screen by
    simulation, fine-rank the survivors); what it never had was the real
    fight to build FOR. The live run now measures the board -- healths,
    schools, incoming damage, the wizard's health and gear -- so the
    search can finally optimise the deck for the fight being farmed
    rather than a hypothetical one.
    """

    status = pyqtSignal(str)
    finished_ok = pyqtSignal(object, float, float)   # deck, win, ttk
    failed = pyqtSignal(str)

    def __init__(self, cards, school, player_hp, player_stats,
                 mob_hps, mob_schools, mob_damage, boss_hp, n_enemies,
                 mob_names=None, encounter_name=""):
        super().__init__()
        self.cards, self.school = cards, school
        self.player_hp = player_hp
        self.player_stats = dict(player_stats or {})
        self.mob_hps = list(mob_hps or [])
        self.mob_schools = list(mob_schools or [])
        self.mob_names = list(mob_names or [])
        self.mob_damage = int(mob_damage or 0)
        self.boss_hp, self.n_enemies = boss_hp, n_enemies
        #: build for a NAMED catalog fight instead of the measured one:
        #: the boss and the creatures the catalog says fight beside it,
        #: before ever walking in
        self.encounter_name = str(encounter_name or "").strip()

    def level_guess(self):
        """The wizard's level, inverted from their health curve.

        `legal_pool` gates by level so the search cannot propose a spell
        the wizard has not trained -- but nothing in the window knows the
        level. The health curve does, near enough: `school_hp` maps
        level to base health, so the measured maximum inverts to a level
        within a few of the truth, and a few levels of slack only ever
        UNDER-gates. Erring low can only hide a card the wizard has;
        erring high proposes one they do not, which is worse.
        """
        try:
            from player_curves import school_hp

            for level in range(1, 121):
                if school_hp(self.school, level) >= self.player_hp:
                    return max(1, level - 2)
            return 120
        except Exception:
            return None

    def run(self):
        try:
            from deck_builder import build_deck
            from w101_sim import Boss

            hps = self.mob_hps or [self.boss_hp] + [
                int(self.boss_hp * 0.8)] * (self.n_enemies - 1)
            schools = (list(self.mob_schools)
                       + ["balance"] * len(hps))[:len(hps)]
            dmg = self.mob_damage or max(30, self.player_hp // 12)
            names = (self.mob_names + [""] * len(hps))[:len(hps)]
            board = sorted(zip(hps, schools, names), key=lambda t: -t[0])

            def mk(hp, sc, name, i):
                """A Boss carrying the catalog's exact defences.

                Resist decides which school of damage a deck should
                slot at all, so a search that priced Lord Nightshade as
                a generic death mob would happily fill the deck with
                the one school he halves. And a named catalog boss is a
                CASTING boss: it fights with its scraped spell pool and
                opening pips instead of a flat hit, so the search prices
                the round-one Wraith a 6-pip opener makes legal -- the
                exact tempo a shield-or-race deck choice hangs on.
                Observed health and school stay authoritative; only the
                flat-damage stand-ins use the measured per-round hit,
                because a casting boss's damage IS its pool.
                """
                if name:
                    try:
                        from ..bestiary import full_boss
                        b = full_boss(name, hp)
                        if b is not None:
                            b.school = sc or b.school
                            if not b.pool:
                                b.dmg = dmg   # measured beats rank guess
                            return b
                    except Exception:
                        pass
                resist_map = boost_map = None
                if name:
                    try:
                        from ..bestiary import stat_overrides
                        found = stat_overrides(name, hp)
                        if found:
                            resist_map = dict(found[0]) or None
                            boost_map = dict(found[1]) or None
                    except Exception:
                        pass
                return Boss(name=name or f"observed mob {i}", hp=int(hp),
                            school=sc, dmg=dmg, resist_map=resist_map,
                            boost_map=boost_map)

            if self.encounter_name:
                # A NAMED fight, before ever walking in: the catalog
                # boss and the creatures it says fight beside it. The
                # observed path below knows more once a fight has
                # happened; this one knows the whole encounter first.
                from ..bestiary import cheat_warning, full_encounter
                self.status.emit(
                    f"looking up '{self.encounter_name}' in the catalog…")
                found = full_encounter(self.encounter_name,
                                       self.boss_hp or None)
                if not found:
                    self.failed.emit(
                        f"'{self.encounter_name}' is not in the catalog "
                        f"— check the spelling, or fight it once and "
                        f"build from the measured board")
                    return
                boss, rest = found
                warn = cheat_warning(self.encounter_name,
                                     self.boss_hp or None)
                if warn:
                    self.status.emit(warn)
            else:
                if any(nm for _, _, nm in board):
                    # the first catalog hit loads the full registry
                    # (~2s); without a line the button just looks dead
                    self.status.emit("pricing the board against the "
                                     "boss catalog…")
                boss = mk(board[0][0], board[0][1], board[0][2], 0)
                rest = [mk(h, sc, nm, i)
                        for i, (h, sc, nm) in enumerate(board[1:], 1)]
            level = self.level_guess()
            self.status.emit(
                f"searching decks for {boss.hp:,} {boss.school}"
                + (f" + {len(rest)} mob(s)" if rest else "")
                + (f", level ~{level}" if level else ""))
            deck, win, ttk, _table = build_deck(
                self.cards, self.school, boss, enemies=rest or None,
                n_candidates=48, top_k=4, level=level,
                player_hp=self.player_hp, player_stats=self.player_stats,
                log=lambda *a: self.status.emit(
                    " ".join(str(x) for x in a)[:120]))
            self.finished_ok.emit(list(deck), float(win), float(ttk))
        except Exception as exc:
            self.failed.emit(f"deck search failed — "
                             f"{type(exc).__name__}: {exc}")


class TuneWorker(QThread):
    """Tunes the search quartet for the observed fight, in the background.

    The quartet -- continuation, horizon, driver, width -- is worth ~14
    points of kill rate and deck-specific, but it was only ever picked
    during a TRAIN. A wizard who connects and just fights on an
    untrained deck plays the untuned defaults indefinitely, on the
    exact boards the run is measuring for it. Round one hands over
    everything the tuner needs (healths, schools, incoming damage),
    and with the sweep no longer installing candidates globally it is
    safe to run under the live fight: about a minute at low priority,
    against every subsequent fight played with a measured pick.
    """

    tuned = pyqtSignal(str, object)     # wire format, {probe: kill rate}
    failed = pyqtSignal(str)

    def __init__(self, school, deck, mob_hps, mob_schools, mob_damage):
        super().__init__()
        self.school, self.deck = school, list(deck)
        self.mob_hps = list(mob_hps or [])
        self.mob_schools = list(mob_schools or [])
        self.mob_damage = int(mob_damage or 0)

    def run(self):
        try:
            from ..live_state import build_catalog
            from ..policies import choose_search, driver_name

            cards = build_catalog()["cards"]
            hp = int(max(self.mob_hps))
            count = len(self.mob_hps)
            sch = (self.mob_schools or ["balance"])[0]
            # The observed fight and a softer cousin: probes at a single
            # ceiling rank nothing, and 0.7x is the envelope lesson
            # applied without an envelope to bisect.
            boards = [(hp, count, sch),
                      (max(1, int(hp * 0.7)), count, sch)]
            picked, horizon, scores = choose_search(
                cards, self.deck, self.school, boards, n=60,
                dmg=self.mob_damage)
            self.tuned.emit(
                f"{picked} @ horizon {horizon} @ driver {driver_name()}",
                dict(scores))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    #: the last training checkpoint's numbers, appended to the progress
    #: bar. A class attribute so the bar can be driven before a run has
    #: ever been started.
    _last_checkpoint = ""
    #: how the last trained table scored against the heuristic it
    #: displaces, on the same board. Empty until a run has finished.
    verdict_text = ""
    trained_kill = 0.0
    rival_kill = 0.0
    #: the rollout continuation picked for the current deck, if a train
    #: has run. Deck-scoped: the choice that is +5.2 points on one deck
    #: is -7.6 on another, so there is no global answer to store.
    continuation = ""
    #: which deck each wizard's installed quartet was measured for, so
    #: the auto-tuner knows an untuned deck when it sees one and never
    #: re-tunes a deck a train already tuned. Per wizard, because the
    #: quartet is deck-scoped and four wizards hold four decks -- one
    #: shared entry would tune wizard 1 and then declare wizard 3 done.
    _tuned_decks = None
    _tuning_decks = None
    _autotunes = None
    #: which wizard the running train belongs to, so the others still
    #: auto-tune while it works
    _training_seat = None
    #: cheat warnings already shown, so a farmed boss is announced once
    #: per session rather than once per fight
    _cheats_warned = None
    #: which wizard the school/deck/policy boxes and the tabs are showing.
    #: Not read off the combo: the combo's index moves *during* a party
    #: resize, and the whole point of this attribute is to know which
    #: wizard the widgets still hold so its values can be stored before
    #: the new one is loaded over them.
    _seat_showing = 0
    #: set while the widgets are being written to from stored config, so
    #: the change signals do not read that back as the user editing --
    #: which would save wizard 1's deck over wizard 2's on every switch,
    #: and swap a running fight's policy for good measure
    _loading = False

    def __init__(self, telemetry=None):
        super().__init__()
        self._cheats_warned = set()
        self.setWindowTitle("wizAi — live combat lab")
        self.resize(1180, 800)
        #: one record per wizard. Built up front rather than on demand so
        #: the selector, the panels and the export all have something to
        #: point at before a run has ever started. `self.tel` stays
        #: wizard 1's, which is what every caller that predates parties
        #: means by "the telemetry".
        self.tels = [telemetry or Telemetry()] + [
            Telemetry() for _ in range(MAX_WIZARDS - 1)]
        self.tel = self.tels[0]
        #: each wizard's own trained table and gear. A Q table is keyed
        #: on its deck, and gear is read per client, so neither can be
        #: shared even between two wizards of the same school.
        self.agents = [None] * MAX_WIZARDS
        self.stats = [{} for _ in range(MAX_WIZARDS)]
        #: each wizard's school/deck/policy/boss, so moving the selector
        #: does not lose what was typed for the other three. Filled in
        #: once the controls exist, at the bottom of `__init__`.
        self.seat_configs = [None] * MAX_WIZARDS
        #: the rollout continuation tuned per wizard (it is deck-scoped,
        #: and the decks differ)
        self.continuations = [""] * MAX_WIZARDS
        self._tuned_decks = [None] * MAX_WIZARDS
        self._tuning_decks = [None] * MAX_WIZARDS
        self._autotunes = {}
        self.worker = None      # training
        self.live = None        # the live fight
        #: each mob's health from the last real fight, so training does
        #: not use a degenerate board of identically-sized mobs
        self.observed_hps = []
        #: mob schools and measured incoming damage, off the live run.
        #: Training used to invent both -- the school as a hardcoded
        #: literal and the damage from the wizard's own health -- and
        #: each of those was on its own enough to make the trained board
        #: a different fight from the played one.
        self.observed_schools = []
        self.observed_names = []
        self.observed_incoming = 0.0
        #: whether the incoming number came off a real fight
        self.mob_damage_measured = False

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.addWidget(self._build_config())

        tabs = QTabWidget()
        self.board = BoardPanel(self.tel)
        self.decisions = DecisionsPanel(self.tel)
        self.model = ModelPanel(self.tel)
        self.learning = LearningPanel(self.tel)
        self.naming = NamingPanel(self.tel)
        self.runs = RunPanel(self.tel)
        self.party = self.hivemind = HivemindPanel()
        #: per-wizard live state for the Hivemind roster, filled from the
        #: worker's seat signals. Kept here rather than read off
        #: `live.seats` because those live on the worker's thread.
        self.seat_live = [{} for _ in range(MAX_WIZARDS)]
        # Every tab scrolls. The Decisions and Learning panels stack a
        # chart, a second chart and a table, which is taller than a laptop
        # window -- and Qt's answer to "does not fit" is to squeeze all
        # three until none of them is readable.
        for panel, name in ((self.board, "Board"),
                            (self.decisions, "Decisions"),
                            (self.model, "Damage model"),
                            (self.learning, "Learning"),
                            (self.naming, "Naming"),
                            (self.runs, "Runs"),
                            (self.party, "Hivemind")):
            tabs.addTab(scrollable(panel), name)
        self.tabs = tabs
        #: the Hivemind tab only means something with a party in it
        self.party_tab = self.hivemind_tab = tabs.count() - 1
        tabs.setTabVisible(self.party_tab, False)
        root.addWidget(tabs)

        self.status = _label("idle — press Play live, or start with --demo",
                             PALETTE["muted"])
        root.addWidget(self.status)
        self.setStyleSheet(stylesheet())
        # Every wizard starts as a copy of what the boxes say, so a party
        # that is never configured per wizard still runs four wizards
        # rather than three empty ones.
        self.seat_configs = [self._snapshot() for _ in range(MAX_WIZARDS)]
        self._wire_live_toggles()
        self.refresh_all()
        self._update_policy_state()

    # -- panel updates all happen here, on the GUI thread -----------------
    def refresh_all(self):
        """Redraw every panel from the telemetry.

        Panels do not subscribe to the telemetry themselves: a live run
        fills it from a worker thread, and touching a widget from there
        is undefined behaviour in Qt. Everything funnels through this,
        called on the GUI thread via `LiveWorker`'s queued signals.
        """
        for panel in (self.board, self.decisions, self.model, self.learning,
                      self.naming, self.runs, self.party):
            try:
                panel.refresh()
            except Exception:
                pass          # a panel must never take down a live fight

    def _build_config(self):
        box = QGroupBox("run")
        outer = QVBoxLayout(box)

        # Built before anything that can emit, because the policy combo's
        # change signal ends up in `_update_policy_state`, which draws to
        # it. Added to the layout further down, where it belongs on screen.
        self.policy_state = _label("no rounds played yet", PALETTE["muted"])
        self.policy_state.setWordWrap(True)

        row = QHBoxLayout()

        # How many clients, and which one everything below is about.
        # One box would not do: "drive three wizards" and "show me the
        # second one" are different questions, and conflating them means
        # you cannot look at wizard 2 without changing the party size.
        row.addWidget(QLabel("wizards"))
        self.wizards = QSpinBox()
        self.wizards.setRange(1, MAX_WIZARDS)
        self.wizards.setValue(1)
        self.wizards.setToolTip(
            "How many Wizard101 clients to drive at once, up to the four a "
            "battle circle seats.\n\n"
            "Above one they play as a hivemind: each round every wizard "
            "submits its board, waits for the others, and then chooses "
            "against a board that already carries what the rest of the "
            "party committed to. A trap one wizard lays is in the next "
            "wizard's rollout and gets cashed; nobody spends a nuke on a "
            "mob another wizard has already killed this round. The Party "
            "tab shows what that changed.\n\n"
            "You need one running, logged-in client per wizard — the run "
            "says so and stops rather than quietly playing short.")
        self.wizards.valueChanged.connect(self.on_wizard_count)
        row.addWidget(self.wizards)

        self.which = QComboBox()
        self.which.setToolTip(
            "Which wizard the boxes below, the tabs above and the Train "
            "button are about. Each wizard keeps its own school, deck, "
            "policy, gear, trained table and run record — switching this "
            "does not lose what you set for the others.")
        self.which.addItem("wizard 1")
        self.which.currentIndexChanged.connect(self.on_which_wizard)
        self.which.setVisible(False)
        row.addWidget(self.which)

        row.addWidget(QLabel("school"))
        self.school = QComboBox()
        self.school.addItems(SCHOOLS)
        self.school.currentTextChanged.connect(self._on_seat_edited)
        row.addWidget(self.school)

        row.addWidget(QLabel("policy"))
        self.policy = QComboBox()
        self.policy.addItems(POLICIES)
        self.policy.setToolTip(
            "Changing this while a fight is running swaps the policy on "
            "the next round — no reconnect. The round in flight finishes "
            "under the policy that started it.")
        self.policy.currentTextChanged.connect(self.on_policy_changed)
        row.addWidget(self.policy)

        # "live fights", not "fights": it sits beside the training
        # controls and reads as an episode count otherwise, which is the
        # one thing it is not.
        row.addWidget(QLabel("live fights"))
        self.fights = QSpinBox()
        self.fights.setRange(0, 999)
        self.fights.setValue(0)
        self.fights.setToolTip(
            "How many real duels to play after pressing Play live. "
            "0 = keep playing until you press Stop. Nothing to do with "
            "training — that is 'episodes'.")
        row.addWidget(self.fights)

        self.train_btn = QPushButton("Train")
        self.train_btn.setToolTip(
            "Trains against the simulator. Works while a live run is "
            "connected — it runs at a lower thread priority so the fight "
            "keeps its timing — and the result is swapped in without "
            "dropping the connection.")
        self.train_btn.clicked.connect(self.on_train)
        row.addWidget(self.train_btn)

        self.start_btn = QPushButton("Play live")
        self.start_btn.clicked.connect(self.on_start_live)
        row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop_live)
        row.addWidget(self.stop_btn)

        self.export_btn = QPushButton("Export run")
        self.export_btn.clicked.connect(self.on_export)
        row.addWidget(self.export_btn)
        row.addStretch()
        outer.addLayout(row)

        # Everything past the first row lives behind a toggle. Five rows
        # of controls is ~250px gone before the tabs begin, and they are
        # set once and then not touched -- while the panels below them are
        # what a run is actually watched through.
        self.more_btn = QPushButton("▾ options")
        self.more_btn.setCheckable(True)
        self.more_btn.setChecked(True)
        self.more_btn.setFlat(True)
        self.more_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: "
            f"{PALETTE['muted']}; border: none; padding: 2px 0; "
            f"text-align: left; }}")
        self.more_btn.toggled.connect(self.on_toggle_options)
        row.addWidget(self.more_btn)

        self.more = QWidget()
        outer.addWidget(self.more)
        outer = QVBoxLayout(self.more)
        outer.setContentsMargins(0, 0, 0, 0)

        # The training parameters. Set once per wizard and then left
        # alone, which is exactly what belongs behind the toggle -- and
        # keeping ten controls out of the top row is what lets the window
        # be narrower than a desk.
        train_row = QHBoxLayout()
        train_row.addWidget(QLabel("episodes"))
        self.episodes = QSpinBox()
        # The old 200k ceiling contradicted the window's own advice: a
        # coverage miss said "raise episodes and retrain" to an
        # operator whose box was already at its maximum.
        self.episodes.setRange(500, 2_000_000)
        self.episodes.setSingleStep(1000)
        self.episodes.setValue(20000)
        self.episodes.setToolTip(
            "How many simulated fights to learn from. Diminishing "
            "returns are measured, not a warning label: coverage grows "
            "as roughly episodes^0.43, so ten times the episodes buys "
            "about 2.7x the coverage. If misses keep coming at high "
            "episode counts, the table cannot key this board range — "
            "the ttk policy needs no training and usually plays it "
            "better.")
        train_row.addWidget(self.episodes)

        train_row.addWidget(QLabel("my HP"))
        self.player_hp = QSpinBox()
        self.player_hp.setRange(100, 20000)
        self.player_hp.setSingleStep(100)
        self.player_hp.setValue(800)
        self.player_hp.setToolTip(
            "Your wizard's max health. Training uses it so the learned "
            "states match a live board — train immortal and the Q table "
            "shares no state with the real game at all, and the policy "
            "passes every turn. Filled in from the game on connect.")
        train_row.addWidget(self.player_hp)

        train_row.addWidget(QLabel("biggest mob HP"))
        self.boss_hp = QSpinBox()
        self.boss_hp.setRange(100, 60000)
        self.boss_hp.setSingleStep(250)
        self.boss_hp.setValue(1200)
        self.boss_hp.setToolTip(
            "The biggest mob on the board. The rest of the training "
            "board is filled in around it — after a real fight from the "
            "healths actually seen, so a 690 boss beside a 255 minion is "
            "trained as exactly that, and before one at 100%/80%/60% of "
            "this number.\n\n"
            "With 'any board' on, the range trained over is discovered "
            "rather than derived from this: the run measures what your "
            "deck can actually clear at each mob count and trains that "
            "span. The board line below says what it settled on.")
        train_row.addWidget(self.boss_hp)

        train_row.addWidget(QLabel("up to mobs"))
        self.n_enemies = QSpinBox()
        self.n_enemies.setRange(1, 4)
        self.n_enemies.setValue(3)
        self.n_enemies.setToolTip(
            "The biggest board to train for. With 'any board' on, every "
            "count from 1 to this is trained, so one model handles a lone "
            "mob and a pack.")
        train_row.addWidget(self.n_enemies)

        self.generalize = QCheckBox("any board")
        self.generalize.setChecked(True)
        self.generalize.setToolTip(
            "Resample the mobs every episode — count and health — instead "
            "of training one board.\n\n"
            "This is what lets a single model cover many fights. Trained "
            "on one board it covers exactly that board: the state key "
            "holds an absolute health bucket, and a targeting tuple that "
            "is only present at all when more than one mob is up, so a "
            "different fight produces keys of a different length or a "
            "different bucket and the table matches nothing.\n\n"
            "Measured on one deck: a fixed-board model covered 0% of five "
            "different boards; a randomised one covered 100% of all five. "
            "Costs more episodes — there are more states to fill.")
        train_row.addWidget(self.generalize)

        train_row.addStretch()
        outer.addLayout(train_row)

        deck_row = QHBoxLayout()
        deck_row.addWidget(QLabel("deck"))
        self.deck = QLineEdit()
        self.deck.setPlaceholderText(
            "press Choose… — or paste comma-separated card names")
        self.deck.setToolTip(
            "Required for a trained policy: the Q table is keyed on this "
            "deck's own blade and nuke positions, so a table trained for "
            "one decklist means nothing for another.")
        self.deck.textChanged.connect(self._on_seat_edited)
        deck_row.addWidget(self.deck)
        self.deck_btn = QPushButton("Choose…")
        self.deck_btn.clicked.connect(self.on_pick_deck)
        deck_row.addWidget(self.deck_btn)
        self.boss_name = QLineEdit()
        self.boss_name.setPlaceholderText("boss name (optional)")
        self.boss_name.setMaximumWidth(170)
        self.boss_name.setToolTip(
            "Type a boss's name to build the deck for its CATALOG "
            "encounter — the real casting boss plus the creatures the "
            "catalog says fight beside it — before ever walking in. "
            "Leave empty to build for the measured board instead.")
        self.boss_name.textChanged.connect(self._on_seat_edited)
        deck_row.addWidget(self.boss_name)
        self.build_btn = QPushButton("Build deck…")
        self.build_btn.setToolTip(
            "Search this school's legal card pool for the strongest deck "
            "against the board the live run measured (or the boxes above, "
            "before a fight). With a boss name typed, it builds for that "
            "boss's catalog encounter instead. Takes a couple of minutes; "
            "the result lands in the deck box to accept or edit — nothing "
            "is applied until you train or play with it.")
        self.build_btn.clicked.connect(self.on_build_deck)
        deck_row.addWidget(self.build_btn)
        outer.addLayout(deck_row)

        quest_row = QHBoxLayout()
        self.follow_leader = QCheckBox("Followers chase wizard 1")
        self.follow_leader.setChecked(True)
        self.follow_leader.setVisible(False)
        self.follow_leader.setToolTip(
            "Wizard 1 quests; the rest teleport onto it and step into its "
            "fight.\n\n"
            "This is what makes a party a party. Four clients each running "
            "the questing independently walk to four different places, take "
            "four different quests, and coordinate perfectly with nobody — "
            "the round-by-round agreement can only help wizards who are in "
            "the same duel.\n\n"
            "Inside one zone the follow is a plain position teleport. Across "
            "zones it goes through the friends list and needs the leader's "
            "wizard name, which is read off the first duel — so the party "
            "has to fight once together, or already be friends and in the "
            "same zone, before a door can be followed through.")
        quest_row.addWidget(self.follow_leader)

        self.auto_quest = QCheckBox("Auto-quest between fights")
        self.auto_quest.setToolTip(
            "Between fights, teleport to the quest marker and click through "
            "dialogue until a fight starts. Not Deimos's full questing — "
            "no navigation or sigils — but enough to keep feeding the "
            "policy fights without babysitting it.")
        quest_row.addWidget(self.auto_quest)

        self.auto_dialogue = QCheckBox("Auto-dialogue")
        self.auto_dialogue.setChecked(True)
        self.auto_dialogue.setToolTip(
            "Watch for dialogue and click through it as it appears, for the "
            "whole run — not just when you press the button. Paused during "
            "combat, so it cannot fight the card clicks.\n\n"
            "Only starts conversations at the quest marker. The game shows "
            "its press-X prompt for every interactable in range, so without "
            "that it greets every vendor and signpost you walk past. Needs "
            "the in-game quest arrow switched on; it says so if it is not.")
        quest_row.addWidget(self.auto_dialogue)

        self.collect_wisps = QCheckBox("Collect wisps")
        self.collect_wisps.setChecked(True)
        self.collect_wisps.setToolTip(
            "After each fight, walk over the health and mana wisps it "
            "dropped. Skips any sitting next to a mob, so topping up does "
            "not start a second fight.")
        quest_row.addWidget(self.collect_wisps)

        self.use_potions = QCheckBox("Use potions")
        self.use_potions.setChecked(True)
        self.use_potions.setToolTip(
            "Drink one when low on health or mana, using Deimos's threshold "
            "(under 55% health, or low mana). Never buys — refilling means "
            "a vendor trip that can strand the run.")
        quest_row.addWidget(self.use_potions)

        self.tp_btn = QPushButton("Teleport to quest")
        self.tp_btn.clicked.connect(self.on_teleport)
        quest_row.addWidget(self.tp_btn)

        self.dialogue_btn = QPushButton("Advance dialogue")
        self.dialogue_btn.clicked.connect(self.on_dialogue)
        quest_row.addWidget(self.dialogue_btn)

        # The same two chores as buttons, not only as automatic
        # between-fights behaviour. Automatic upkeep runs after a fight
        # ends; nothing could top the wizard up during a long questing
        # stretch, or when it silently was not working.
        self.wisps_btn = QPushButton("Collect wisps")
        self.wisps_btn.setToolTip(
            "Sweep the health and mana wisps in range now, rather than "
            "waiting for a fight to end. Says what it found either way.")
        self.wisps_btn.clicked.connect(self.on_wisps)
        quest_row.addWidget(self.wisps_btn)

        self.potion_btn = QPushButton("Drink potion")
        self.potion_btn.setToolTip(
            "Use one potion charge now. Never buys — refilling means a "
            "vendor trip that can strand the run.")
        self.potion_btn.clicked.connect(self.on_potion)
        quest_row.addWidget(self.potion_btn)
        quest_row.addStretch()
        outer.addLayout(quest_row)

        key_row = QHBoxLayout()
        self.use_hotkeys = QCheckBox("Hotkeys")
        self.use_hotkeys.setChecked(True)
        self.use_hotkeys.setToolTip(
            "Do the actions above without leaving the game. These are "
            "system-wide keys — they fire whatever window has focus, and "
            "while the run is connected the key is taken away from every "
            "other program, Wizard101 included. Pick keys the game does "
            "not use.")
        key_row.addWidget(self.use_hotkeys)

        self.hotkey_boxes = {}
        for action, label in (("teleport", "tp to quest"),
                              ("dialogue", "dialogue"),
                              ("wisps", "wisps"),
                              ("potion", "potion")):
            key_row.addWidget(QLabel(label))
            combo = QComboBox()
            combo.addItems(HOTKEY_CHOICES)
            combo.setCurrentText(HOTKEY_DEFAULTS[action])
            key_row.addWidget(combo)
            self.hotkey_boxes[action] = combo
        key_row.addStretch()
        outer.addLayout(key_row)

        script_row = QHBoxLayout()
        self.use_script = QCheckBox("Run script")
        self.use_script.setToolTip(
            "Run a Deimos bot script (deimoslang) alongside the policy. It "
            "steps only while out of combat, so wizAi still fights.")
        script_row.addWidget(self.use_script)
        self.script_btn = QPushButton("Paste script…")
        self.script_btn.clicked.connect(self.on_edit_script)
        script_row.addWidget(self.script_btn)
        self.script_lab = _label("no script", PALETTE["muted"])
        script_row.addWidget(self.script_lab)
        script_row.addStretch()
        outer.addLayout(script_row)
        self.script_source = ""

        # The answer to "is the model I picked actually playing?". A
        # trained policy that falls back on every board decides exactly
        # like the heuristic it falls back to, so without this the only
        # symptom is that the fight looks unremarkable.
        # Back on the group box itself: the coverage/gear readout and the
        # progress bar answer "is this working right now", so they stay
        # visible when the options are folded away.
        box.layout().addWidget(self.policy_state)

        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        # The count goes *in* the bar. A bare bar answers "is it moving",
        # which is only half of what anyone watching a twenty-minute
        # train wants to know -- the other half is how far in it is and
        # how much longer, and both of those are numbers.
        self.train_progress.setTextVisible(True)
        box.layout().addWidget(self.train_progress)
        return box

    #: checkbox -> the attribute it drives on a running `LiveWorker`.
    #: The worker reads every one of these on each service tick, so
    #: keeping them in step is all a live toggle needs -- and without it
    #: they were read once at Play live and never again, which is why
    #: auto-quest, auto-dialogue, the upkeep chores and the follow could
    #: not be turned on or off during a run.
    LIVE_TOGGLES = {"auto_quest": "auto_quest",
                    "auto_dialogue": "auto_dialogue",
                    "collect_wisps": "collect_wisps",
                    "use_potions": "use_potions",
                    "follow_leader": "follow_leader"}

    def _wire_live_toggles(self):
        for box_name, attr in self.LIVE_TOGGLES.items():
            box = getattr(self, box_name)
            box.toggled.connect(
                lambda on, a=attr, n=box_name: self._on_live_toggle(a, n, on))
        self.use_script.toggled.connect(lambda _on: self._push_script())

    def _on_live_toggle(self, attr, box_name, on):
        # Not while a seat's saved configuration is being restored into
        # the boxes. `setChecked` fires `toggled` exactly like a click
        # does, so switching the wizard dropdown mid-run would push
        # whatever that seat's snapshot happened to hold onto the live
        # worker -- turning auto-quest off for the whole party because
        # wizard 3 was configured without it.
        if self._loading:
            return
        if self.live is None or not self.live.isRunning():
            return
        setattr(self.live, attr, bool(on))
        label = box_name.replace("_", "-")
        self.status.setText(f"{label} is {'on' if on else 'off'} — takes "
                            f"effect on the next tick, no reconnect")

    def _push_script(self):
        """Hand the running worker whatever the script box says now.

        The worker rebuilds or tears down each seat's runner on its own
        loop when this changes; see `LiveWorker._sync_script`.
        """
        if self._loading:
            return                    # a restore, not a person: see above
        if self.live is None or not self.live.isRunning():
            return
        want = self.script_source if self.use_script.isChecked() else ""
        self.live.script = want
        self.status.setText("script started" if want else "script stopped")

    def on_toggle_options(self, shown, by_user=True):
        self.more.setVisible(shown)
        self.more_btn.setText("▾ options" if shown else "▸ options")
        if by_user:
            # A deliberate choice is never overridden by the auto-fold
            # below. Adaptive layout that undoes what someone just did is
            # worse than no adaptive layout.
            self._options_pinned = True

    #: set once the toggle has been pressed by hand
    _options_pinned = False
    #: below this the config block is more than a third of the window
    FOLD_BELOW = 700

    def resizeEvent(self, event):
        """Fold the options away on a short window.

        Five rows of set-once controls is ~250px; on a 520px-tall window
        that is half the screen spent on things nobody is looking at
        while the panels underneath are the point.
        """
        super().resizeEvent(event)
        # Qt can deliver a resize before `_build_config` has run, and a
        # virtual like this one aborts the process on an unhandled
        # AttributeError rather than printing it.
        if self._options_pinned or not hasattr(self, "more_btn"):
            return
        try:
            want = self.height() >= self.FOLD_BELOW
            if want != self.more_btn.isChecked():
                self.more_btn.blockSignals(True)
                self.more_btn.setChecked(want)
                self.more_btn.blockSignals(False)
                self.on_toggle_options(want, by_user=False)
        except Exception:
            pass

    def _board_line(self):
        """The board training will actually build, spelled out.

        One spinbox cannot say "a 690 HP boss beside a 255 HP minion",
        and the box does not have to: after a real fight the healths are
        taken from what was seen, and `board_hps` uses them whenever the
        count matches. What was missing was any way to *tell* -- the box
        showed one number and the board was derived silently, so a
        mismatched board looked like a typo in a field that was not
        being used.
        """
        worker = TrainWorker(
            {}, [], self.school.currentText(), 0,
            player_hp=self.player_hp.value(), boss_hp=self.boss_hp.value(),
            n_enemies=self.n_enemies.value(), mob_hps=self.observed_hps,
            mob_schools=self.observed_schools,
            mob_damage=int(self.observed_incoming),
            party_size=self.party_size())
        hps = worker.board_hps()
        schools = worker.board_schools()
        board = " + ".join(f"{hp:,} {sc}" for hp, sc in zip(hps, schools))
        source = ("from your last fight" if
                  len(self.observed_hps) == self.n_enemies.value()
                  else "spread around the biggest")
        dmg = worker.enemy_damage()
        how = ("measured live" if self.mob_damage_measured
               else "estimated — no fight measured yet")
        line = (f"training board: {board}, {dmg}/round each "
                f"({how}) — {source}")
        if self.party_size() > 1:
            # Said rather than left to be discovered. The incoming
            # damage is this wizard's share and the simulator knows it;
            # the other wizards' *damage* is not modelled, because the
            # simulator has one wizard in it. So the trained table is
            # pessimistic about the fight, which is the safe direction
            # but not a free one -- it will not learn to leave a mob to
            # somebody else.
            line += (f"\ntraining models ONE wizard: incoming is your "
                     f"share of a {self.party_size()}-wizard circle, but "
                     f"the other {self.party_size() - 1} wizards' damage "
                     f"is not in it — the table is trained for a harder "
                     f"fight than the party plays")
        return line

    def _gear_line(self):
        """What the simulator thinks the wizard is wearing.

        Worth a line of its own: with no gear read, every hit is priced
        as though the wizard were naked, and the policy then optimises
        that fight rather than the one on screen.
        """
        st = self.player_stats
        if not st:
            return ("gear not read — hits are priced as if you wore none, "
                    "so the policy is solving a different fight")
        # The school the stats were *read for*, not whatever the dropdown
        # says now: gear is read once on connect, and re-keying it off a
        # combo the user can move afterwards would report 0%.
        damage = st.get("damage") or {}
        school, pct = next(iter(damage.items()), (self.school.currentText(),
                                                  0.0))
        bits = [f"{pct * 100:.0f}% {school} damage"]
        for key, label in (("pierce", "pierce"), ("accuracy", "accuracy"),
                           ("crit", "crit")):
            if st.get(key):
                bits.append(f"{st[key] * 100:.0f}% {label}")
        return "gear: " + ", ".join(bits) + " — training uses these"

    def _why_coverage_is_low(self, reasons=None):
        """Name the specific mismatch rather than saying "train more".

        "Train more episodes" is the wrong advice for every cause here,
        and it is expensive advice to follow before finding that out.
        The mismatches are checkable, so check them.

        `reasons` are the ones the misses *recorded as they happened*,
        and they come first because they are observations rather than
        inferences. The run that prompted this printed "a 480 HP mob is
        above the 94-423 band this table was trained on" on one line and
        "the states are mostly unvisited -- raise episodes and retrain"
        on the next: the second is guesswork, it contradicted the first,
        and it is the one fix that could not have helped.
        """
        if reasons:
            why, n = next(iter(reasons.items()))
            rest = sum(reasons.values()) - n
            return (f"cause: {why} ({n} round(s)"
                    + (f", plus {rest} for other reasons" if rest else "")
                    + ")")
        # First: did training learn anything at all? A table whose kill
        # rate never left zero has nothing to apply, and every other
        # explanation below is a distraction from that.
        curve = self.current_tel().training_curve()
        if curve and max(k for _ep, k in curve) <= 0.0:
            return ("cause: training never won a fight — kill rate stayed "
                    "at 0% for every checkpoint, so the table learned "
                    "nothing to apply. Check the Learning tab: if the "
                    "board is unwinnable at these settings, lower mob HP "
                    "or the mob count.")
        # Second: was the wizard's own health ever read? The key buckets
        # player health as a fraction of the maximum, so training against
        # the box's default while the wizard has some other maximum makes
        # every live board key a bucket the table never visited. This
        # cause is invisible in the board numbers below -- mob count and
        # mob HP can both be perfectly in range -- and its symptom looks
        # exactly like an under-trained table, which is the one fix that
        # cannot help.
        if self.live is not None and getattr(self.live, "hp_known", True) \
                is False:
            return ("cause: your max health was never read off the client "
                    "(the status bar said so on connect), so training used "
                    "whatever was in the box. The key buckets health as a "
                    "fraction of the maximum — a wrong maximum shares no "
                    "states with the live board. Fix the health box and "
                    "retrain.")
        seen = self.current_tel().observed_board()
        if not self.generalize.isChecked():
            return ("cause: trained on one fixed board. Tick 'any board' "
                    "and retrain — a fixed-board model covered 0% of five "
                    "different boards in testing and a randomised one "
                    "covered 100% of all five.")
        if seen:
            n, hp = seen
            if n > self.n_enemies.value():
                return (f"cause: fighting {n} mobs, trained for up to "
                        f"{self.n_enemies.value()}. The state key only "
                        f"carries its targeting tuple on a multi-enemy "
                        f"board, so a count never trained produces keys of "
                        f"a length the table has none of. Raise 'up to "
                        f"mobs' to {n} and retrain.")
            lo, hi = TrainWorker(
                {}, [], "ice", 0, boss_hp=self.boss_hp.value(),
                mob_hps=self.observed_hps).hp_range()
            if not (lo <= hp <= hi):
                return (f"cause: fighting ~{hp:,.0f} HP mobs, trained over "
                        f"{lo:,}–{hi:,}. The key buckets health as HP//250, "
                        f"so outside that band nothing matches. Set mob HP "
                        f"nearer {hp:,.0f} and retrain.")
        # The dead-end version of this advice told an operator at the
        # episode box's MAXIMUM to "raise episodes and retrain". The
        # scaling is measured: coverage grows as ~episodes^0.43, so at
        # high counts the honest reading is that the table cannot key
        # this range — and the window already measures who plays those
        # rounds better (the fallback IS the ttk lookahead).
        if self.episodes.value() >= 100_000:
            return ("the states are mostly unvisited even at "
                    f"{self.episodes.value():,} episodes. Coverage grows "
                    "as ~episodes^0.43 (measured), so more training buys "
                    "little here — this board range is wider than a "
                    "tabular key can fill. The rounds that miss are "
                    "played by the ttk lookahead, which needs no "
                    "training; picking the ttk policy outright is "
                    "usually the stronger driver on boards like these.")
        return ("the states are mostly unvisited — a wider board range "
                "needs more episodes to fill; raise episodes and retrain")

    def _update_policy_state(self):
        """Say which policy is driving, and how often it really decided."""
        mix = self.current_tel().policy_mix()
        if not mix:
            self.policy_state.setText(
                "policy selected: " + self.policy.currentText() +
                " — no rounds played yet\n" + self._gear_line())
            self.policy_state.setStyleSheet(f"color: {PALETTE['muted']}")
            return
        total = sum(mix.values())
        parts = [f"{name} ×{n}" for name, n in mix.items()]
        text = (f"{total} round(s): " + "  ·  ".join(parts)
                + "\n" + self._gear_line()
                + "\n" + self._board_line())

        colour = PALETTE["muted"]
        # Off the round records, which is where the Learning tab reads
        # it too. Taking it off the live policy object instead put two
        # different answers to one question on screen at once, because
        # those counters reset on every `set_policy` and the records do
        # not.
        decided, missed, reasons = self.current_tel().trained_coverage()
        if decided + missed:
            cover = decided * 100.0 / (decided + missed)
            colour = (PALETTE["good"] if cover > 66 else
                      PALETTE["warn"] if cover > 25 else PALETTE["bad"])
            text += (f"\nQ table decided {cover:.0f}% of the boards it was "
                     f"shown ({missed} fell back to the heuristic)")
            if cover < 66:
                text += "\n" + self._why_coverage_is_low(reasons)
        if self.verdict_text:
            # Under the coverage line on purpose. Coverage answers "does
            # the table recognise this board"; this answers "is it any
            # good at it", and when the answer is no, coverage is the
            # bad news rather than the good.
            text += "\n" + self.verdict_text
            if self.rival_kill > self.trained_kill:
                colour = PALETTE["bad"]
        self.policy_state.setText(text)
        self.policy_state.setStyleSheet(f"color: {colour}")

    # -- which wizard the window is talking about --------------------------
    #
    # One selector governs the config boxes AND the tabs, deliberately.
    # Two would let the window show wizard 2's decisions beside wizard
    # 1's deck, which is exactly the confusion a party invites.
    @property
    def agent(self):
        """The selected wizard's trained table. A Q table is keyed on its
        own decklist, so there is no such thing as the party's agent."""
        return self.agents[self._seat_showing]

    @agent.setter
    def agent(self, value):
        self.agents[self._seat_showing] = value

    @property
    def player_stats(self):
        """The selected wizard's gear, read off its own client."""
        return self.stats[self._seat_showing]

    @player_stats.setter
    def player_stats(self, value):
        self.stats[self._seat_showing] = dict(value or {})

    @property
    def continuation(self):
        return self.continuations[self._seat_showing]

    @continuation.setter
    def continuation(self, value):
        self.continuations[self._seat_showing] = value or ""

    def current_tel(self):
        """The record the tabs are showing."""
        return self.tels[self._seat_showing]

    @property
    def _tuned_deck(self):
        return self._tuned_decks[self._seat_showing]

    @_tuned_deck.setter
    def _tuned_deck(self, value):
        self._tuned_decks[self._seat_showing] = value

    @property
    def _tuning_deck(self):
        return self._tuning_decks[self._seat_showing]

    @_tuning_deck.setter
    def _tuning_deck(self, value):
        self._tuning_decks[self._seat_showing] = value

    @property
    def _autotune(self):
        return self._autotunes.get(self._seat_showing)

    @_autotune.setter
    def _autotune(self, value):
        if value is None:
            self._autotunes.pop(self._seat_showing, None)
        else:
            self._autotunes[self._seat_showing] = value

    def _snapshot(self):
        return {"school": self.school.currentText(),
                "policy": self.policy.currentText(),
                "deck": self.deck.text(),
                "boss": self.boss_name.text()}

    def _on_seat_edited(self, *_):
        """The boxes are the truth for whichever wizard they are showing."""
        if self._loading or self.seat_configs is None:
            return
        self.seat_configs[self._seat_showing] = self._snapshot()

    def on_which_wizard(self, index):
        """Point the boxes and the tabs at a different wizard."""
        if self._loading:
            return
        index = max(0, min(int(index), MAX_WIZARDS - 1))
        if index == self._seat_showing:
            return
        # Store before loading, or the wizard being left behind keeps
        # whatever the wizard being switched to has.
        self.seat_configs[self._seat_showing] = self._snapshot()
        self._seat_showing = index
        cfg = self.seat_configs[index] or {}
        self._loading = True
        try:
            self.school.setCurrentText(cfg.get("school") or SCHOOLS[0])
            self.policy.setCurrentText(cfg.get("policy") or POLICIES[0])
            self.deck.setText(cfg.get("deck") or "")
            self.boss_name.setText(cfg.get("boss") or "")
        finally:
            self._loading = False
        for panel in (self.board, self.decisions, self.model, self.learning,
                      self.naming, self.runs):
            try:
                panel.set_telemetry(self.tels[index])
            except Exception:
                pass          # a panel must never take down a live fight
        self.status.setText(
            f"showing wizard {index + 1} — its own school, deck, policy, "
            f"gear and run record")
        self._update_policy_state()

    def on_wizard_count(self, n):
        """Resize the party. The selector and the Party tab follow."""
        n = max(1, min(int(n), MAX_WIZARDS))
        self._loading = True
        try:
            while self.which.count() > n:
                self.which.removeItem(self.which.count() - 1)
            while self.which.count() < n:
                i = self.which.count()
                named = (self.tels[i].wizard
                         if i < len(self.tels) else "")
                self.which.addItem(f"wizard {i + 1} — {named}" if named
                                   else f"wizard {i + 1}")
        finally:
            self._loading = False
        self.which.setVisible(n > 1)
        self.follow_leader.setVisible(n > 1)
        self.tabs.setTabVisible(self.party_tab, n > 1)
        if n > 1:
            # The roster is the party's size, so it redraws when the
            # party changes size -- including before a run, where it is
            # the only confirmation that four wizards were configured.
            self.refresh_hivemind()
        if self._seat_showing >= n:
            # The wizard the boxes were showing is no longer in the
            # party; Qt has already moved the combo, so this only has to
            # make the window agree with it.
            self.on_which_wizard(self.which.currentIndex())
        if n > 1:
            self.status.setText(
                f"{n} wizards — they will agree each round before any of "
                f"them casts. You need {n} running, logged-in clients.")
        else:
            self.status.setText("one wizard — deciding for itself")

    def party_size(self):
        return max(1, min(self.wizards.value(), MAX_WIZARDS))

    def seat_configs_now(self):
        """Every wizard in the party, with the visible one refreshed."""
        self._on_seat_edited()
        out = []
        for i in range(self.party_size()):
            cfg = dict(self.seat_configs[i] or self._snapshot())
            cfg["deck"] = [d.strip() for d in (cfg.get("deck") or "").split(",")
                           if d.strip()]
            out.append(cfg)
        return out

    # -- actions ----------------------------------------------------------
    def decklist(self):
        return [d.strip() for d in self.deck.text().split(",") if d.strip()]

    def on_train(self):
        if self.worker is not None and self.worker.isRunning():
            return
        deck = self.decklist()
        if not deck:
            QMessageBox.warning(self, "wizAi", "A decklist is required.")
            return
        try:
            from data_full import load_spells_full
            cards = load_spells_full()
        except Exception as exc:
            QMessageBox.critical(self, "wizAi", f"card table failed: {exc}")
            return
        missing = [d for d in deck if d not in cards]
        if missing:
            QMessageBox.warning(
                self, "wizAi",
                "These are not in the card table, so the policy could never "
                "play them:\n\n  " + "\n  ".join(missing))
            return

        self._tuning_deck = list(deck)
        self._training_seat = self._seat_showing
        self.train_btn.setEnabled(False)
        self.train_progress.setVisible(True)
        # Indeterminate only until the first tick. The warm-start solve
        # runs before episode 1 and has nothing to count, so a bar that
        # started at 0/N would sit at zero looking stalled through the
        # one phase that genuinely has no progress to report.
        self.train_progress.setRange(0, 0)
        self.train_progress.setFormat("starting…")
        self._last_checkpoint = ""
        fighting = self.live is not None and self.live.isRunning()
        self.status.setText(
            f"training {self.episodes.value():,} episodes on {len(deck)} cards…"
            + (" (the live fight keeps playing)" if fighting else ""))

        self.worker = TrainWorker(cards, deck, self.school.currentText(),
                                  self.episodes.value(),
                                  player_hp=self.player_hp.value(),
                                  boss_hp=self.boss_hp.value(),
                                  player_stats=self.player_stats,
                                  n_enemies=self.n_enemies.value(),
                                  mob_hps=self.observed_hps,
                                  generalize=self.generalize.isChecked(),
                                  # Both read off the real fight. The
                                  # schools especially: every training
                                  # mob used to be the literal "ice".
                                  mob_schools=self.observed_schools,
                                  mob_damage=self.observed_incoming,
                                  party_size=self.party_size())
        self.current_tel().clear_curve()
        self.worker.snapshot.connect(self.on_snapshot)
        self.worker.progress.connect(self.on_progress)
        self.worker.tick.connect(self.on_tick)
        self.worker.stage.connect(self.on_stage)
        self.worker.verdict.connect(self.on_verdict)
        self.worker.continuation.connect(self.on_continuation)
        self.worker.finished_ok.connect(self.on_trained)
        self.worker.failed.connect(self.on_train_failed)
        self.worker.refused.connect(self.on_train_refused)
        # Below the live worker, deliberately. Training is minutes of
        # solid CPU with no I/O to yield on; at equal priority it starves
        # the fight's event loop, and a planning phase that arrives late
        # is a turn played by the game's timeout rather than by wizAi.
        self.worker.start(QThread.Priority.LowPriority if fighting
                          else QThread.Priority.InheritPriority)

    def on_snapshot(self, episode, kill, ttk):
        """One training checkpoint. Queued from the training thread, so
        this runs on the GUI thread and may touch widgets."""
        self.current_tel().record_snapshot(episode, kill, ttk)
        # Kept for the bar to append. A checkpoint is the only thing in a
        # training run that says whether it is going anywhere, and it
        # used to be visible only as a new point on a chart on another
        # tab -- so a run left on the Live tab reported nothing but a
        # moving bar for its whole length.
        self._last_checkpoint = (
            f"kill {kill * 100:.0f}%"
            + (f", {ttk:.1f} turns" if ttk == ttk else ", won nothing"))
        try:
            self.learning.refresh()
        except Exception:
            pass          # a watching panel never interrupts training

    def on_stage(self, name):
        """A phase of training that is not measured in episodes."""
        if name == "training":
            return          # the bar takes over from here
        self.train_progress.setRange(0, 0)
        self.train_progress.setFormat(f"{name}…")
        self.status.setText(f"{name}…")

    def on_tick(self, done, total, seconds_left):
        """Episodes done, in the bar itself."""
        if self.train_progress.maximum() != total:
            self.train_progress.setRange(0, total)
        self.train_progress.setValue(done)
        text = f"{done:,} / {total:,} episodes  ({done * 100 // max(1, total)}%)"
        if seconds_left >= 0 and done < total:
            text += f" — {_duration(seconds_left)} left"
        if self._last_checkpoint:
            text += f" — {self._last_checkpoint}"
        self.train_progress.setFormat(text)

    def on_continuation(self, name, scores, seat=None):
        """The rollout continuation picked for one wizard's deck.

        Kept on the window rather than in a module global so a live run
        can be handed it, and so changing decks changes it. Per wizard
        for the same reason it is per deck: the choice that is +5.2
        points on one deck is -7.6 on another, and a party holds four.
        """
        seat = self._seat_showing if seat is None else seat
        self.continuations[seat] = name
        self._tuned_decks[seat] = tuple(sorted(
            self._tuning_decks[seat] or self.decklist()))
        ranked = sorted((scores or {}).items(), key=lambda kv: -kv[1])
        spread = ((ranked[0][1] - ranked[-1][1]) * 100) if len(ranked) > 1 else 0
        who = ("this deck" if self.party_size() == 1
               else f"wizard {seat + 1}'s deck")
        self.status.setText(
            f"rollout continuation for {who}: {name} "
            f"({spread:.0f} points better than the worst of "
            f"{len(ranked)} tried)")

    def on_verdict(self, kill, rival):
        """Is the table worth playing, against the heuristic it displaces?

        The one number the window never showed. Coverage answers "does
        the table recognise this board", which is a different question
        and the less useful one: a table can key 95% of boards and play
        every one of them worse than the fallback it is keeping out of
        the driver's seat, and every readout in the Learning tab would
        still look healthy.
        """
        self.trained_kill, self.rival_kill = kill, rival
        if rival <= 0.0 and kill <= 0.0:
            self.verdict_text = ""
        elif kill >= rival:
            self.verdict_text = (
                f"the trained table wins {kill * 100:.0f}% where the "
                f"school-aware heuristic wins {rival * 100:.0f}% — worth "
                f"playing")
        else:
            self.verdict_text = (
                f"the trained table wins {kill * 100:.0f}% where the "
                f"school-aware heuristic wins {rival * 100:.0f}% on the "
                f"same board — the table is the weaker player here, so "
                f"high coverage makes the fight worse, not better")
        self._update_policy_state()

    def on_progress(self, ep, total, kill, ttk):
        self.status.setText(
            f"episode {ep:,}/{total:,} — kill {kill:.1f}%, TTK {ttk:.2f}")

    def on_trained(self, agent):
        self.agent = agent
        self.train_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        # Retrained while it was already driving: hand the new table over
        # in place. Without this, the fight would keep playing the table
        # that was current when Play live was pressed, and the retrain
        # would look like it had no effect.
        if self.live is not None and self.live.isRunning() and \
                self.policy.currentText().startswith("trained"):
            self.live.set_policy(self.policy.currentText(), agent=agent)
            self.status.setText(
                "trained — the running fight is now using the new table")
        else:
            self.status.setText("trained — policy ready to drive a live fight")
        self._update_policy_state()

    def on_train_failed(self, message):
        self.train_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        self.status.setText("training failed")
        QMessageBox.critical(self, "wizAi", message)

    def on_train_refused(self, message):
        """The preflight said no. A verdict about the BOARD, not a
        crash -- the old path rendered it as "training failed", which
        a live operator reasonably read as the tool breaking."""
        self.train_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        self.status.setText(
            "training refused — the board is not winnable at these "
            "settings (see message)")
        QMessageBox.warning(self, "wizAi", message)

    # -- deck ------------------------------------------------------------
    def on_build_deck(self):
        """Search for the strongest deck against the measured board."""
        if getattr(self, "deck_worker", None) is not None \
                and self.deck_worker.isRunning():
            return
        try:
            from ..live_state import build_catalog
            cards = build_catalog()["cards"]
        except Exception as exc:
            QMessageBox.critical(self, "wizAi", f"card table failed: {exc}")
            return
        self.build_btn.setEnabled(False)
        self.deck_worker = DeckWorker(
            cards, self.school.currentText(), self.player_hp.value(),
            self.player_stats, self.observed_hps, self.observed_schools,
            int(self.observed_incoming), self.boss_hp.value(),
            self.n_enemies.value(), mob_names=self.observed_names,
            encounter_name=self.boss_name.text())
        self.deck_worker.status.connect(self.status.setText)
        self.deck_worker.finished_ok.connect(self.on_deck_built)
        self.deck_worker.failed.connect(self.on_deck_build_failed)
        self.deck_worker.start()

    def on_deck_built(self, deck, win, ttk):
        self.build_btn.setEnabled(True)
        self.deck.setText(",".join(deck))
        self.status.setText(
            f"built a {len(deck)}-card deck — wins {win * 100:.0f}% of the "
            f"measured fight at {ttk:.1f} turns. It is in the deck box; "
            f"edit it freely, then Train or Play live.")

    def on_deck_build_failed(self, message):
        self.build_btn.setEnabled(True)
        self.status.setText(message)

    def on_pick_deck(self):
        try:
            from ..live_state import build_catalog
            catalog = build_catalog()
            cards = catalog["cards"]
        except Exception as exc:
            QMessageBox.critical(self, "wizAi", f"card table failed: {exc}")
            return
        # Cards actually seen in hand during the last live run. This is
        # the only honest "read it off the game": the client exposes the
        # deck as template ids and wizAi's table carries none, so a real
        # deck read cannot be turned into names -- but a card in combat
        # can, because CombatCard.name() returns one.
        seen = sorted({name for rec in self.current_tel().rounds
                       for name in rec.hand})
        chosen = pick_deck(self, cards, self.school.currentText(),
                           self.decklist(), seen,
                           canonical=catalog.get("canonical"))
        if chosen is not None:
            self.deck.setText(",".join(chosen))
            self.status.setText(f"deck set — {len(chosen)} cards")

    # -- scripts ---------------------------------------------------------
    def on_edit_script(self):
        from .scriptdialog import edit_script
        source = edit_script(self, self.script_source)
        if source is not None:
            self.script_source = source
            lines = len([ln for ln in source.splitlines() if ln.strip()])
            self.script_lab.setText(f"{lines} line(s) loaded" if lines
                                    else "no script")
            self.use_script.setChecked(bool(lines))
            self._push_script()

    # -- questing --------------------------------------------------------
    def _quest_action(self, coro_name, label):
        """Run one questing helper against the live client.

        Only available while a run is connected: these need the hooks,
        and the worker owns the client.
        """
        if self.live is None or not self.live.isRunning():
            QMessageBox.information(
                self, "wizAi",
                "Press Play live first — these need the hooks installed, "
                "and the live run owns the client connection.")
            return
        # Honoured, not discarded: `request` refuses an action that is
        # already queued or already running, and telling the user it is
        # happening anyway is how a dropped press became "the button
        # does nothing".
        if not self.live.request(coro_name):
            self.status.setText(
                f"{coro_name} is already queued or running — it will "
                f"finish on its own")
            return
        self.status.setText(label)

    def hotkey_bindings(self):
        """{action: key}, or empty when hotkeys are off.

        Two actions may not share a key: `RegisterHotKey` would take the
        first and silently refuse the second, so the second action would
        appear bound and do nothing.

        Every collision is collected and reported together. With four
        actions on four default keys, retargeting one onto another's
        default drops that *other* action -- one the user never touched
        -- and reporting them one at a time meant each message overwrote
        the last, so only the final collision was ever seen.
        """
        if not self.use_hotkeys.isChecked():
            return {}
        out, taken, clashes = {}, set(), []
        for action, box in self.hotkey_boxes.items():
            key = box.currentText()
            if key in taken:
                clashes.append((action, key))
                continue
            taken.add(key)
            out[action] = key
        if clashes:
            self.status.setText(
                "; ".join(f"{key} is bound twice — '{action}' is unbound"
                          for action, key in clashes)
                + ". Give each action its own key.")
        return out

    def on_teleport(self):
        self._quest_action("teleport", "teleporting to the quest marker…")

    def on_dialogue(self):
        self._quest_action("dialogue", "clicking through dialogue…")

    def on_wisps(self):
        self._quest_action("wisps", "sweeping for wisps…")

    def on_potion(self):
        self._quest_action("potion", "drinking a potion…")

    # -- live ------------------------------------------------------------
    def on_policy_changed(self, name):
        """Swap the policy on a running fight, or just remember it.

        The whole point of doing this here rather than at Play live: the
        deck picker's card list and the health the table was trained for
        both come from what a connected run observed, so disconnecting to
        change models throws away the inputs to the next decision.

        It swaps the *selected* wizard's policy, not the party's. Four
        wizards in a circle are meant to play differently -- that is most
        of what makes a party worth more than one wizard four times.
        """
        if self._loading:
            return          # loading another wizard's config, not a choice
        self._on_seat_edited()
        if self.live is not None and self.live.isRunning():
            seat = {} if self._seat_showing == 0 \
                else {"seat": self._seat_showing}
            self.live.set_policy(name, agent=self.agent, **seat)
        self._update_policy_state()

    def on_gear_read(self, stats, seat=0):
        """A wizard's real damage/accuracy/pierce, off its own client."""
        self.stats[max(0, min(int(seat), MAX_WIZARDS - 1))] = dict(stats or {})
        self._update_policy_state()

    def on_hp_read(self, hp, seat=0):
        """A wizard's real max health, straight off its own client.

        Only written into the box when it is that wizard's box on screen.
        The box drives training, and training is for the selected wizard
        -- wizard 3's 2,100 landing in it while wizard 1 is shown would
        silently train wizard 1's deck against wizard 3's health, which
        is the exact mismatch reading the health off the client exists to
        prevent.
        """
        if hp <= 0 or seat != self._seat_showing:
            return
        if hp == self.player_hp.value():
            return
        self.player_hp.setMaximum(max(self.player_hp.maximum(), hp))
        self.player_hp.setValue(hp)
        self.status.setText(
            f"read your max health from the game: {hp:,} — training will "
            f"use it")

    def on_start_live(self):
        if self.live is not None and self.live.isRunning():
            return
        configs = self.seat_configs_now()
        for i, cfg in enumerate(configs):
            policy, deck = cfg["policy"], cfg["deck"]
            who = "" if len(configs) == 1 else f"Wizard {i + 1}: "
            if policy.startswith("trained") and self.agents[i] is None:
                QMessageBox.warning(
                    self, "wizAi",
                    f"{who}no trained policy yet. Press Train first, or pick "
                    "another policy from the list — you can switch to it "
                    "mid-fight once it has trained, without reconnecting."
                    + ("" if not who else "\n\nEach wizard has its own "
                       "trained table: a Q table is keyed on its own "
                       "decklist, so wizard 1's does not drive wizard 3."))
                return
            if policy.startswith("trained") and not deck:
                QMessageBox.warning(self, "wizAi",
                                    f"{who}a trained policy needs its deck.")
                return

        first = configs[0]
        rest = [SeatConfig(school=cfg["school"], deck=cfg["deck"],
                           policy_name=cfg["policy"], agent=self.agents[i],
                           continuation=self.continuations[i],
                           telemetry=self.tels[i], name=f"wizard {i + 1}")
                for i, cfg in enumerate(configs) if i > 0]
        self.live = LiveWorker(self.tels[0], first["school"], first["deck"],
                               first["policy"], self.fights.value(),
                               agent=self.agents[0],
                               auto_quest=self.auto_quest.isChecked(),
                               auto_dialogue=self.auto_dialogue.isChecked(),
                               collect_wisps=self.collect_wisps.isChecked(),
                               use_potions=self.use_potions.isChecked(),
                               script=(self.script_source
                                       if self.use_script.isChecked() else ""),
                               hotkeys=self.hotkey_bindings(),
                               continuation=self.continuations[0],
                               seats=rest,
                               follow_leader=self.follow_leader.isChecked())
        self.live.status.connect(self.on_live_status)
        # Per wizard as well as into the one-line status bar: with four
        # of them talking the bar holds whichever spoke last, and a
        # follower stuck repeating itself is invisible the moment anyone
        # else says anything.
        self.live.seat_status.connect(self.on_seat_status)
        # Seat-aware throughout: four wizards fill four records, and a
        # round routed to the wrong one settles its damage against
        # another wizard's board.
        self.live.seat_round_done.connect(self.on_seat_round)
        self.live.seat_fight_done.connect(self.on_seat_fight_done)
        self.live.failed.connect(self.on_live_failed)
        self.live.finished_ok.connect(self.on_live_finished)
        self.live.seat_hp_read.connect(self.on_seat_hp_read)
        self.live.seat_gear_read.connect(self.on_seat_gear_read)
        self.live.seat_policy_changed.connect(self.on_seat_policy_installed)
        self.live.party_plan.connect(self.on_party_plan)
        self.live.seat_named.connect(self.on_seat_named)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # Train stays live. Every input to a useful training run -- the
        # deck the picker learned from the last fight, the health the
        # client reported -- only exists once connected, so requiring a
        # disconnect to train meant training on guesses.
        self.live.start()

    def on_stop_live(self):
        if self.live is not None:
            self.live.stop()
            self.status.setText("stopping after this fight…")
        self.stop_btn.setEnabled(False)

    def on_live_status(self, message):
        self.status.setText(message)

    def adopt_observed_board(self):
        """Centre the training range on the fights actually being fought.

        With 'any board' on this is a nudge, not a requirement -- the
        range is trained across, so a fight near the middle of it needs
        no retraining at all. It only widens: `up to mobs` never comes
        down, because a model that already covers three mobs should not
        forget them because the last fight had one.
        """
        tel = self.current_tel()
        seen = tel.observed_board()
        if not seen:
            return False
        n, hp = seen
        changed = (n > self.n_enemies.value()
                   or abs(hp - self.boss_hp.value()) >= 250)
        n = max(n, self.n_enemies.value())
        # The individual healths, not just the biggest: mobs that all
        # start equal never produce the opening state a real board has.
        self.observed_hps = tel.observed_mob_hps()
        self.observed_schools = tel.observed_mob_schools()
        self.observed_names = tel.observed_mob_names()
        self.observed_incoming = tel.observed_incoming()
        self.mob_damage_measured = self.observed_incoming > 0
        self.n_enemies.setValue(min(max(int(n), 1), self.n_enemies.maximum()))
        self.boss_hp.setValue(min(max(int(hp), self.boss_hp.minimum()),
                                  self.boss_hp.maximum()))
        return changed

    def on_seat_hp_read(self, seat, hp):
        if 0 <= seat < len(self.seat_live):
            self.seat_live[seat]["max_hp"] = hp
        self.on_hp_read(hp, seat)

    def on_seat_gear_read(self, seat, stats):
        self.on_gear_read(stats, seat)

    def on_party_plan(self, plan):
        """One round the whole party agreed. Queued onto the GUI thread."""
        self.party.show_plan(plan, getattr(self.live, "hive", None))
        self.refresh_hivemind()

    # -- the Hivemind roster ----------------------------------------------
    def on_seat_status(self, seat, message):
        """The last thing one wizard said, kept per wizard.

        The status bar holds one line for the whole window, so with four
        wizards talking it holds whichever of them spoke last -- and a
        follower repeating "could not read the leader's position" is
        invisible the moment anyone else says anything. The roster keeps
        one line each, which is the difference between noticing a stuck
        wizard and not.
        """
        if not (0 <= seat < len(self.seat_live)):
            return
        self.seat_live[seat]["last_said"] = message
        self.refresh_hivemind()

    def _seat_state(self, seat, hive):
        """What this wizard is doing, as one of `HivemindPanel._STATES`."""
        row = self.seat_live[seat]
        hp = row.get("hp")
        if hp is not None and hp <= 0:
            return "defeated"
        fighting = set(hive.fighting()) if hive is not None else set()
        if seat in fighting:
            # Alone in the circle is a different thing from in the party's
            # circle, and it is the one worth flagging: a wizard fighting
            # by itself is being planned for by itself.
            return "fighting" if len(fighting) > 1 else "alone"
        if self.live is not None and self.live.isRunning():
            if (self.follow_leader.isChecked() and self.party_size() > 1
                    and seat != 0):
                return "following"
            if self.auto_quest.isChecked():
                return "questing"
        return "waiting"

    def refresh_hivemind(self):
        """Redraw the roster from what the signals have told us so far."""
        if not self.tabs.isTabVisible(self.party_tab):
            return
        hive = getattr(self.live, "hive", None)
        rows = []
        for seat in range(self.party_size()):
            live = self.seat_live[seat]
            tel = self.tels[seat]
            cfg = self.seat_configs[seat] if self.seat_configs else None
            move = hive.last_move(seat) if hive is not None else None
            last_move = ""
            if move is not None and move.card:
                last_move = move.card + (f" @{move.target_name}"
                                         if move.target_name else "")
            elif move is not None:
                last_move = "pass"
            rows.append({
                "name": (tel.wizard or live.get("name")
                         or f"wizard {seat + 1}"),
                "school": tel.school or (cfg or {}).get("school", ""),
                "policy": tel.policy_name or (cfg or {}).get("policy", ""),
                "hp": live.get("hp"),
                "max_hp": live.get("max_hp", 0),
                "state": self._seat_state(seat, hive),
                "fights": len(tel.fights),
                "rounds": len(tel.rounds),
                "last_move": last_move,
                "last_said": live.get("last_said", ""),
            })
        self.party.show_party(rows, hive)
        self.party.show_model(self._party_model())

    def _party_model(self):
        """The party's damage-model number, counted once per round.

        Every seat that fired into the shared mob records the same
        claim about the same board delta, so adding the seats' series
        together would count one measurement twice for two wizards and
        four times for four. Deduped on the round it describes: which
        duel (by opening board, since fight numbers are per seat), which
        round, which mob.
        """
        import statistics

        seen, obs = set(), []
        for seat in range(self.party_size()):
            tel = self.tels[seat]
            by_index = {f.index: f.opening for f in tel.fights}
            for r in tel.party_observations():
                key = (by_index.get(r.fight, r.fight), r.round, r.target_name)
                if key in seen:
                    continue
                seen.add(key)
                obs.append(r)
        if not obs:
            return {"n": 0}
        errs = [r.party_error for r in obs]
        pcts = [100.0 * r.party_error / r.party_predicted
                for r in obs if r.party_predicted]
        worst = max(obs, key=lambda r: abs(r.party_error))
        return {"n": len(obs),
                "mean_error": statistics.fmean(errs),
                "mean_pct_error": statistics.fmean(pcts) if pcts else None,
                "worst": f"f{worst.fight}r{worst.round}"}

    def on_seat_named(self, seat, name):
        """A duel told us which wizard this client is actually driving.

        "wizard 1" and "wizard 2" are this window's own numbering and
        mean nothing to anyone looking at four game clients. Once the
        game has named one, the selector says so too -- the game window's
        own title bar is stamped by the worker at the same moment, so the
        two agree.
        """
        if not name or not (0 <= seat < self.which.count()):
            return
        if 0 <= seat < len(self.seat_live):
            self.seat_live[seat]["name"] = name
        self._loading = True
        try:
            self.which.setItemText(seat, f"wizard {seat + 1} — {name}")
        finally:
            self._loading = False
        self.refresh_hivemind()

    def on_round(self, rec):
        # Queued from the worker thread, so this runs on the GUI thread.
        if getattr(rec, "round", 0) == 1:
            self._warn_cheats(rec)
        self.refresh_all()
        self._update_policy_state()

    def on_seat_round(self, seat, rec):
        """A round from one wizard.

        Only the wizard on screen redraws. Four wizards each finishing a
        planning phase would otherwise repaint every panel four times a
        round, three of those with data the panels are not showing.
        """
        if getattr(rec, "round", 0) == 1:
            self._warn_cheats(rec)
        # Every wizard's health, from every wizard's round: the roster is
        # the one place all four are shown at once, so it cannot be fed
        # only by whichever one happens to be on screen.
        if 0 <= seat < len(self.seat_live):
            live = self.seat_live[seat]
            live["hp"] = getattr(rec, "player_hp", None)
            if getattr(rec, "player_max_hp", 0):
                live["max_hp"] = rec.player_max_hp
        if seat == self._seat_showing:
            self.refresh_all()
            self._update_policy_state()
        else:
            try:
                self.party.refresh()
            except Exception:
                pass
        self.refresh_hivemind()

    def _warn_cheats(self, rec):
        """Say so when the catalog knows this enemy cheats.

        The catalog has carried scraped cheat notes for 1,912 creatures
        the whole time, and the run reads enemy names every round; the
        two just never met. The sim cannot model an arbitrary cheat, but
        the operator can be told -- a known interrupt is survivable in a
        way a surprise one is not.
        """
        from ..bestiary import cheat_warning

        for e in getattr(rec, "enemies", []) or []:
            try:
                line = cheat_warning(e.name, e.max_hp)
            except Exception:
                continue
            if line and line not in self._cheats_warned:
                self._cheats_warned.add(line)
                self.status.setText(line)

    def on_fight_done(self, _n):
        if self.adopt_observed_board():
            self.status.setText(
                f"training range centred on that fight: up to "
                f"{self.n_enemies.value()} mob(s) around "
                f"{self.boss_hp.value():,} HP")
        self._maybe_autotune()
        self.refresh_all()
        self._update_policy_state()

    def on_seat_fight_done(self, seat, n):
        """One wizard's duel ended.

        The board it just fought is the same board every wizard in the
        circle fought, so the training range is adopted from whichever
        wizard is on screen and not from all four -- four adoptions of
        one fight would say the same thing four times and race each
        other into the spinboxes.
        """
        if seat == self._seat_showing:
            self.on_fight_done(n)
        else:
            self.refresh_all()

    def _maybe_autotune(self):
        """Tune each wizard's quartet for the observed fight.

        A train tunes as part of its run; this covers the wizard who
        connects and just fights. It fires once per deck, off the first
        finished fight's measured board, at low priority under the live
        run -- which the sweep is safe for now that it never installs
        the candidate it is measuring.

        Once per **wizard**, not once per window. The quartet is
        deck-scoped and worth ~14 points of kill rate, and a party holds
        four decks: tuning only the wizard that happens to be selected
        leaves the other three playing the untuned defaults for the
        whole run, on the exact boards the run is measuring for them.
        """
        if self.live is None or not self.live.isRunning():
            return
        if not self.observed_hps:
            return
        seats = getattr(self.live, "seats", None)
        configs = self.seat_configs_now()
        if seats is not None:
            configs = configs[:len(seats)]
        started = []
        for seat, cfg in enumerate(configs):
            if self._start_autotune(seat, cfg):
                started.append(seat)
        if started:
            who = ("the search" if len(configs) == 1 else
                   "wizard " + ", ".join(str(i + 1) for i in started) +
                   "'s search")
            self.status.setText(
                f"tuning {who} for the observed fight (in the "
                f"background — the fight keeps playing)…")

    def _start_autotune(self, seat, cfg):
        """One wizard's tuner, if it wants one. Returns whether it started."""
        deck = cfg["deck"]
        if not deck:
            return False
        if seat == self._training_seat and self.worker is not None \
                and self.worker.isRunning():
            return False              # its train will tune on its own
        if tuple(sorted(deck)) == self._tuned_decks[seat]:
            return False
        running = self._autotunes.get(seat)
        if running is not None and running.isRunning():
            return False
        dmg = int(self.observed_incoming) or max(
            30, self.player_hp.value() // 12)
        worker = TuneWorker(cfg["school"], deck, self.observed_hps,
                            self.observed_schools, dmg)
        self._autotunes[seat] = worker
        # Bound to the seat, not to whichever wizard happens to be
        # selected when the sweep finishes a minute later.
        worker.tuned.connect(
            lambda wire, scores, s=seat: self.on_autotuned(wire, scores, s))
        worker.failed.connect(lambda _m: None)          # a nicety;
        worker.start(QThread.Priority.LowPriority)      # stay quiet
        return True

    def on_autotuned(self, wire, scores, seat=None):
        seat = self._seat_showing if seat is None else seat
        cfg = self.seat_configs[seat] or {}
        self._tuning_decks[seat] = [d.strip() for d
                                    in (cfg.get("deck") or "").split(",")
                                    if d.strip()]
        # The pick is this wizard's, and so is the reinstall. Storing it
        # first: `set_policy` rebuilds the closure, and the closure is
        # where the quartet gets bound for a party -- reinstalling before
        # the pick is recorded would rebuild with the old one.
        self.continuations[seat] = wire
        # A driver change only takes effect through a rebuilt policy
        # closure; reinstalling the current name does exactly that
        # without dropping the connection. Before on_continuation, so
        # the status line the operator is left with is the tuned pick.
        if self.live is not None and self.live.isRunning():
            seats = getattr(self.live, "seats", None)
            if seats is not None and seat < len(seats):
                seats[seat].continuation = wire
                name = seats[seat].policy_name
            else:
                name = self.live.policy_name
            if seat == 0:
                self.live.set_policy(name)
            else:
                self.live.set_policy(name, seat=seat)
        self.on_continuation(wire, scores, seat)

    def on_policy_installed(self, name):
        """The worker says which policy it actually ended up with.

        It can differ from the dropdown -- picking a trained policy with
        nothing trained keeps the old one -- so the box is put back in
        step with reality rather than left showing a selection that did
        not take.
        """
        if name and name != self.policy.currentText():
            self.policy.blockSignals(True)      # not a fresh user choice
            self.policy.setCurrentText(name)
            self.policy.blockSignals(False)
            self._on_seat_edited()
        self._update_policy_state()

    def on_seat_policy_installed(self, seat, name):
        """Only the wizard on screen puts its box back in step.

        Seat 3's policy landing in the box while wizard 1 is shown would
        misreport wizard 1 and, worse, be saved as wizard 1's choice the
        next time anything touched the config.
        """
        if not name:
            return
        self.seat_configs[seat] = dict(self.seat_configs[seat] or {},
                                       policy=name)
        if seat == self._seat_showing:
            self.on_policy_installed(name)

    def on_live_failed(self, message):
        self._live_over()
        self.status.setText("live run failed")
        QMessageBox.critical(self, "wizAi", message)

    def on_live_finished(self):
        self._live_over()
        self.status.setText("live run finished")

    def _live_over(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.train_btn.setEnabled(True)
        self.refresh_all()
        self._update_policy_state()

    def closeEvent(self, event):
        if self.live is not None and self.live.isRunning():
            self.live.stop()
            self.live.wait(3000)
        super().closeEvent(event)

    def on_export(self):
        """Write the run out. In a party, one file per wizard.

        Not one merged file: the records are per wizard because a round
        settles its damage against the board *that* wizard was shown, and
        interleaving four wizards' rounds into one file would make every
        residual in it meaningless. The names are suffixed rather than
        overwritten, which is the alternative that silently exports one
        wizard four times.
        """
        path, _ = QFileDialog.getSaveFileName(
            self, "Export run", "results_live_run.json", "JSON (*.json)")
        if not path:
            return
        n = self.party_size()
        if n == 1:
            self.tels[0].to_json(path)
            self.status.setText(f"wrote {path}")
            return
        import os

        stem, ext = os.path.splitext(path)
        written = []
        for i in range(n):
            # The name when the game has given one. Three files called
            # "-wizard1/2/3" have to be identified by reading the hands
            # and guessing, which is what happened to the first party
            # run's exports.
            who = "".join(ch for ch in (self.tels[i].wizard or "")
                          if ch.isalnum())
            out = f"{stem}-wizard{i + 1}{'-' + who if who else ''}" \
                  f"{ext or '.json'}"
            self.tels[i].to_json(out)
            written.append(os.path.basename(out))
        self.status.setText(f"wrote {len(written)} files: "
                            + ", ".join(written))


def demo_telemetry():
    """Drive the whole window from `mock_client`, so the GUI is
    exercisable -- and testable -- with no game and no Windows."""
    import asyncio

    from w101_sim import make_blade_stack

    from ..live_backend import WizAiBackend
    from ..live_state import build_catalog
    from ..mock_client import MockCard, MockCombat, MockEffect, MockMember

    catalog = build_catalog()
    cards = catalog["cards"]
    deck = ["Fireblade"] * 3 + ["Sunbird"] * 4
    tel = Telemetry(policy_name="blade-stack(2)", school="fire", deck=deck)
    be = WizAiBackend.from_trained(school="fire", deck=deck, cards=cards,
                                   policy=make_blade_stack(2),
                                   catalog=catalog,
                                   policy_name="blade-stack(2)")
    tel.resolver = be.resolver
    be.on_decision = lambda d, r: tel.observe(d, r, sim=be._sim_for(r),
                                              cards=cards)
    tel.start_fight()

    hp, blades = 2400, []
    for rnd in range(1, 7):
        me = MockMember("Wizard", 3000 - rnd * 90, client=True, team_id=0,
                        normal_pips=min(rnd + 1, 7), hangings=list(blades))
        foe = MockMember("Krokopatra", hp, monster=True, team_id=1)
        combat = MockCombat(
            [me, foe],
            [MockCard("Fireblade"), MockCard("Sunbird"),
             # one of each miss kind, so the Naming panel shows the split
             MockCard("Not A Real Spell"),      # no such card anywhere
             MockCard("Summon589244")],         # real, but decoder-skipped
            round_number=rnd)
        be.attach_combat(combat)
        decision = asyncio.run(be.decide())
        if decision.card_name == "Fireblade":
            blades.append(MockEffect("modify_outgoing_damage", 35, 2343174,
                                     1000 + rnd))
        elif decision.card_name:
            # a real mob resists a bit, so predicted and actual differ --
            # which is the point of the panel
            hp -= int(325 * (1 + 0.35 * len(blades)) * 0.88)
            blades.clear()
    tel.end_fight(won=hp <= 0)
    return tel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true",
                    help="populate from mock_client instead of a live client")
    args = ap.parse_args(argv)

    app = QApplication(sys.argv[:1])

    # Before the window, so an error building it is still reported. Qt
    # loses errors unusually well: an exception inside a virtual aborts
    # the process outright, which from a desktop shortcut is a window
    # vanishing with nothing written down.
    from . import crashlog

    def show_crash(text):
        QMessageBox.critical(
            None, "wizAi hit an error",
            text[-2000:] + f"\n\nWritten to {crashlog.log_path()}")

    crashlog.install(show=show_crash)

    win = MainWindow(demo_telemetry() if args.demo else None)
    if args.demo:
        win.status.setText("demo data — press Play live to use the real game")
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
