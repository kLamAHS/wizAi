"""The window.

Holds a `Telemetry`, one tab per panel, and the controls for configuring
and starting a run. Training happens on a worker thread so the window
stays responsive -- a Q-learning run is minutes of solid CPU and freezing
the UI for it would make the progress readout pointless.

    python -m deimos_bridge.gui              # live (needs Windows + game)
    python -m deimos_bridge.gui --demo       # canned fight, runs anywhere
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
from .live import LiveWorker
from .panels import (BoardPanel, DecisionsPanel, LearningPanel, ModelPanel,
                     NamingPanel, RunPanel, _label, scrollable)
from .theme import PALETTE, stylesheet

SCHOOLS = ["fire", "ice", "storm", "myth", "life", "death", "balance"]
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
    #: (trained kill rate, heuristic kill rate) on the same eval board.
    #: The comparison the window never made — a table that keys 95% of
    #: boards and plays them worse than the fallback reads as a success
    #: on every number the Learning tab used to show.
    verdict = pyqtSignal(float, float)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, cards, deck, school, episodes, player_hp=800,
                 boss_hp=1200, player_stats=None, n_enemies=1,
                 mob_hps=None, generalize=True, mob_schools=None,
                 mob_damage=0):
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
            board = Boss(name="probe", hp=hp, school=schools[0], dmg=dmg)
            extra = [Boss(name=f"probe {i}", hp=hp,
                          school=schools[i % len(schools)], dmg=dmg)
                     for i in range(1, count)]
            sim = Sim(self.cards, self.deck, self.school, board,
                      player_hp=self.player_hp,
                      player_stats=self.player_stats, enemies=extra)
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
            bands[count] = (floor, lo)
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
        from w101_sim import Boss, Sim, evaluate
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
            # anyone makes on purpose.
            t, _ = evaluate(sim, trained_policy(agent), n=n)
            r, _ = evaluate(sim, school_aware_blade_stack(3), n=n)
            if abs(t - r) > worst[2]:
                worst = (t, r, abs(t - r))
        return worst[0], worst[1]

    def describe_envelope(self, bands):
        if not bands:
            return ("this deck cannot clear a single mob at these settings "
                    "— nothing to train")
        parts = [f"{n} mob{'s' if n > 1 else ''} to ~{hi:,} HP"
                 for n, (_lo, hi) in sorted(bands.items())]
        return "this deck clears " + ", ".join(parts)

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
        """
        if self.mob_damage and self.mob_damage > 0:
            return int(self.mob_damage)
        return max(30, self.player_hp // 12)

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
        mobs = " + ".join(f"{b.hp:,} HP {b.school}" for b in [board] + extra)
        health = sum(b.hp for b in [board] + extra)
        head = (f"this board cannot be won at these settings, so training "
                f"it would draw a flat 0% however long it ran.\n\n"
                f"Board: {mobs}, each hitting for {board.dmg}/round, "
                f"against {self.player_hp:,} HP.\n\n")

        # Name the cause rather than listing the knobs. When the deck
        # simply cannot deliver the board's health, none of the knobs is
        # the answer and suggesting them sends you round in circles.
        ceiling = self.damage_ceiling()
        if ceiling < health:
            missing = [n for n in ("Evil Snowman", "Snow Serpent",
                                   "Frost Beetle")
                       if n in self.cards and n not in self.deck]
            hint = (f" You have no {missing[0]} in the deck."
                    if missing else "")
            return False, (
                head +
                f"Your deck is the reason, not the board. Every damage "
                f"card in it, landing, with your gear, and with every "
                f"buff spent on the biggest hits, comes to about "
                f"{ceiling:,.0f} damage — and this board has {health:,} "
                f"health. No play order wins that, and no mob HP, mob "
                f"count or enemy school setting changes it.{hint}\n\n"
                f"Add damage cards, or train against a smaller board.")

        return False, (
            head +
            f"A scripted policy wins 0 of {n} fights on it, and your deck "
            f"could deliver about {ceiling:,.0f} damage against its "
            f"{health:,} health — so this is a race it loses on time, not "
            f"on damage. Lower the mob HP or the mob count, raise your "
            f"health, or check the incoming damage: at {board.dmg}/round "
            f"each you last {self.player_hp / max(1, board.dmg * (1 + len(extra))):.0f} "
            f"rounds.")

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
            self.stage.emit(note if note else "training")
            if not feasible:
                self.failed.emit(note)
                return

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

    def __init__(self, telemetry=None):
        super().__init__()
        self.setWindowTitle("wizAi — live combat lab")
        self.resize(1180, 800)
        self.tel = telemetry or Telemetry()
        self.agent = None
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
        self.observed_incoming = 0.0
        #: whether the incoming number came off a real fight
        self.mob_damage_measured = False
        #: the wizard's gear, filled in from the client on connect.
        #: Training uses it, which is the whole point: a Q table learned
        #: for a naked wizard is solving a fight nobody will play.
        self.player_stats = {}

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
        # Every tab scrolls. The Decisions and Learning panels stack a
        # chart, a second chart and a table, which is taller than a laptop
        # window -- and Qt's answer to "does not fit" is to squeeze all
        # three until none of them is readable.
        for panel, name in ((self.board, "Board"),
                            (self.decisions, "Decisions"),
                            (self.model, "Damage model"),
                            (self.learning, "Learning"),
                            (self.naming, "Naming"),
                            (self.runs, "Runs")):
            tabs.addTab(scrollable(panel), name)
        self.tabs = tabs
        root.addWidget(tabs)

        self.status = _label("idle — press Play live, or start with --demo",
                             PALETTE["muted"])
        root.addWidget(self.status)
        self.setStyleSheet(stylesheet())
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
                      self.naming, self.runs):
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
        row.addWidget(QLabel("school"))
        self.school = QComboBox()
        self.school.addItems(SCHOOLS)
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
        self.episodes.setRange(500, 200_000)
        self.episodes.setSingleStep(1000)
        self.episodes.setValue(20000)
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
        deck_row.addWidget(self.deck)
        self.deck_btn = QPushButton("Choose…")
        self.deck_btn.clicked.connect(self.on_pick_deck)
        deck_row.addWidget(self.deck_btn)
        outer.addLayout(deck_row)

        quest_row = QHBoxLayout()
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
            mob_damage=int(self.observed_incoming))
        hps = worker.board_hps()
        schools = worker.board_schools()
        board = " + ".join(f"{hp:,} {sc}" for hp, sc in zip(hps, schools))
        source = ("from your last fight" if
                  len(self.observed_hps) == self.n_enemies.value()
                  else "spread around the biggest")
        dmg = worker.enemy_damage()
        how = ("measured live" if self.mob_damage_measured
               else "estimated — no fight measured yet")
        return (f"training board: {board}, {dmg}/round each "
                f"({how}) — {source}")

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
        curve = self.tel.training_curve()
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
        seen = self.tel.observed_board()
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
        return ("the states are mostly unvisited — a wider board range "
                "needs more episodes to fill; raise episodes and retrain")

    def _update_policy_state(self):
        """Say which policy is driving, and how often it really decided."""
        mix = self.tel.policy_mix()
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
        decided, missed, reasons = self.tel.trained_coverage()
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
                                  mob_damage=self.observed_incoming)
        self.tel.clear_curve()
        self.worker.snapshot.connect(self.on_snapshot)
        self.worker.progress.connect(self.on_progress)
        self.worker.tick.connect(self.on_tick)
        self.worker.stage.connect(self.on_stage)
        self.worker.verdict.connect(self.on_verdict)
        self.worker.finished_ok.connect(self.on_trained)
        self.worker.failed.connect(self.on_train_failed)
        # Below the live worker, deliberately. Training is minutes of
        # solid CPU with no I/O to yield on; at equal priority it starves
        # the fight's event loop, and a planning phase that arrives late
        # is a turn played by the game's timeout rather than by wizAi.
        self.worker.start(QThread.Priority.LowPriority if fighting
                          else QThread.Priority.InheritPriority)

    def on_snapshot(self, episode, kill, ttk):
        """One training checkpoint. Queued from the training thread, so
        this runs on the GUI thread and may touch widgets."""
        self.tel.record_snapshot(episode, kill, ttk)
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

    # -- deck ------------------------------------------------------------
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
        seen = sorted({name for rec in self.tel.rounds for name in rec.hand})
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
        """
        if self.live is not None and self.live.isRunning():
            self.live.set_policy(name, agent=self.agent)
        self._update_policy_state()

    def on_gear_read(self, stats):
        """The wizard's real damage/accuracy/pierce, off the client."""
        self.player_stats = dict(stats or {})
        self._update_policy_state()

    def on_hp_read(self, hp):
        """The wizard's real max health, straight off the client."""
        if hp <= 0 or hp == self.player_hp.value():
            return
        self.player_hp.setMaximum(max(self.player_hp.maximum(), hp))
        self.player_hp.setValue(hp)
        self.status.setText(
            f"read your max health from the game: {hp:,} — training will "
            f"use it")

    def on_start_live(self):
        if self.live is not None and self.live.isRunning():
            return
        deck = self.decklist()
        policy = self.policy.currentText()
        if policy.startswith("trained") and self.agent is None:
            QMessageBox.warning(
                self, "wizAi",
                "No trained policy yet. Press Train first, or pick another "
                "policy from the list — you can switch to it mid-fight once "
                "it has trained, without reconnecting.")
            return
        if policy.startswith("trained") and not deck:
            QMessageBox.warning(self, "wizAi", "A trained policy needs its deck.")
            return

        self.live = LiveWorker(self.tel, self.school.currentText(), deck,
                               policy, self.fights.value(), agent=self.agent,
                               auto_quest=self.auto_quest.isChecked(),
                               auto_dialogue=self.auto_dialogue.isChecked(),
                               collect_wisps=self.collect_wisps.isChecked(),
                               use_potions=self.use_potions.isChecked(),
                               script=(self.script_source
                                       if self.use_script.isChecked() else ""),
                               hotkeys=self.hotkey_bindings())
        self.live.status.connect(self.on_live_status)
        self.live.round_done.connect(self.on_round)
        self.live.fight_done.connect(self.on_fight_done)
        self.live.failed.connect(self.on_live_failed)
        self.live.finished_ok.connect(self.on_live_finished)
        self.live.hp_read.connect(self.on_hp_read)
        self.live.gear_read.connect(self.on_gear_read)
        self.live.policy_changed.connect(self.on_policy_installed)
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
        seen = self.tel.observed_board()
        if not seen:
            return False
        n, hp = seen
        changed = (n > self.n_enemies.value()
                   or abs(hp - self.boss_hp.value()) >= 250)
        n = max(n, self.n_enemies.value())
        # The individual healths, not just the biggest: mobs that all
        # start equal never produce the opening state a real board has.
        self.observed_hps = self.tel.observed_mob_hps()
        self.observed_schools = self.tel.observed_mob_schools()
        self.observed_incoming = self.tel.observed_incoming()
        self.mob_damage_measured = self.observed_incoming > 0
        self.n_enemies.setValue(min(max(int(n), 1), self.n_enemies.maximum()))
        self.boss_hp.setValue(min(max(int(hp), self.boss_hp.minimum()),
                                  self.boss_hp.maximum()))
        return changed

    def on_round(self, _rec):
        # Queued from the worker thread, so this runs on the GUI thread.
        self.refresh_all()
        self._update_policy_state()

    def on_fight_done(self, _n):
        if self.adopt_observed_board():
            self.status.setText(
                f"training range centred on that fight: up to "
                f"{self.n_enemies.value()} mob(s) around "
                f"{self.boss_hp.value():,} HP")
        self.refresh_all()
        self._update_policy_state()

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
        self._update_policy_state()

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
        path, _ = QFileDialog.getSaveFileName(
            self, "Export run", "results_live_run.json", "JSON (*.json)")
        if path:
            self.tel.to_json(path)
            self.status.setText(f"wrote {path}")


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
