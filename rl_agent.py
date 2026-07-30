"""
Phase 3: RL on top of the simulator.

Two nested learning problems:
  1. In-fight policy — tabular Q-learning over a featurized state that
     includes what the DP abstraction cannot see: which cards are in hand
     and how many damage cards remain (scarcity). Warm-started by the
     DP-transfer policy (probability of following it decays over training).
  2. Deck selection — every boss is preset (HP/school/damage known), so
     loadout choice is a per-boss bandit. UCB1 over the school's candidate
     decks, trained jointly with the in-fight Q-tables.

Objective: primarily kill the boss, secondarily fast.
Reward: -1 per turn, terminal 0 on kill, -FAIL_PENALTY on timeout/defeat.
"""
import random, math
from collections import defaultdict
from w101_sim import (Sim, Boss, load_cards, evaluate, strat_nuke_asap,
                      make_blade_stack, tc_reflex)
from dp_solver import solve, dp_policy

FAIL_PENALTY = 25.0
MAX_TURNS = 40
PASS = "__pass__"


# ---------------------------------------------------------------- state feats

DMG_KINDS = ("damage", "drain")


class Featurizer:
    def __init__(self, cards, decklist):
        self.names = list(dict.fromkeys(decklist))
        self.blades = [n for n in self.names if cards[n].kind == "blade"]
        self.traps = [n for n in self.names
                      if cards[n].kind in ("trap", "prism")]
        self.dmg = [n for n in self.names if cards[n].kind in DMG_KINDS]

    def key(self, sim, s):
        """Compact state. Hand bits only for damage cards: buff availability
        is already encoded by the legal-action set, but WHICH nuke is in hand
        (vs still in deck) drives the wait/dig/fire decision. nukes_left is
        the scarcity signal the DP abstraction lacks. Treasure cards in
        hand enter by name (the sideboard is small and WHICH TC is up
        decides the turn)."""
        hb = min(int(s.boss_hp // 250), 24)
        p = min(s.norm_pips + 2 * s.pow_pips +
                2 * getattr(s, 'school_pips', 0), 14)
        bmask = sum(1 << i for i, n in enumerate(self.blades) if n in s.blades)
        tsig = tuple(s.traps[n][1] if n in s.traps else 0 for n in self.traps)
        dmask = sum(1 << i for i, n in enumerate(self.dmg)
                    if any(c.name == n for c in s.hand))
        nukes_left = min(sum(1 for c in s.hand if c.kind in DMG_KINDS) +
                         sum(1 for c in s.deck if c.kind in DMG_KINDS), 8)
        tcs = tuple(sorted({c.name for c in s.hand if c.source == "tc"}))
        # own-HP bucket only in mortal fights (heal/shield timing needs
        # it); immortal fights keep the old, smaller state space
        php = min(int(s.player_hp // 300), 9) \
            if sim.player_hp0 < 10**9 else -1
        # living-boss opponents are STATEFUL: their hanging blades
        # (incoming spike) and self-shields (my hits blunted) drive
        # timing decisions the old state couldn't express. Flat bosses
        # keep the sentinel so their state space is unchanged.
        # per-foe health, ONLY on a multi-enemy board — the agent
        # cannot choose a target it cannot see. Single-enemy fights keep
        # the shorter tuple, so their state space is untouched.
        # Multi-enemy targeting signal, kept DELIBERATELY coarse. The
        # first cut encoded a health bucket per enemy; with three foes
        # that made almost every state unique, so nothing was ever
        # visited twice, Q stayed all-zero, and greedy fell through to
        # the first legal action — PASS. The agent scored a clean 0.0%.
        # What focus-fire actually needs is which foe is closest to
        # dying and how many are left, so that is all it gets.
        foes = ()
        if len(s.enemies) > 1:
            alive = [(i, e) for i, e in enumerate(s.enemies) if e.alive]
            if alive:
                weakest = min(alive, key=lambda ie: ie[1].hp)[0]
                foes = (len(alive), weakest,
                        min(int(min(e.hp for _, e in alive) // 600), 6))
        boss = s.enemies[0]
        if getattr(boss, "spell_pool", None) is not None:
            eb = min(sum(1 for h in boss.charms
                         if h.kind == "damage" and h.percent > 0), 3)
            es = 1 if any(h.kind == "damage" and h.percent < 0
                          for h in boss.wards) else 0
        else:
            eb = es = -1
        base = (hb, p, bmask, tsig, dmask, nukes_left, tcs, php, eb, es)
        return base + (foes,) if foes else base

    def legal(self, sim, s):
        """Legal actions. On a multi-enemy board a single-target
        offensive card is expanded per living foe as "name@i", so the
        agent can commit to killing one thing instead of always hitting
        slot 0 — the deficit that cost it 54 points at Krokotopia and
        that mob_generalist.py isolated as representational.

        Single-enemy fights emit bare names exactly as before, so every
        1v1 result predating this is bit-identical."""
        acts = [PASS]
        foes = [i for i, e in enumerate(s.enemies) if e.alive]
        seen = set()
        for c in s.hand:
            if c.name in seen or not sim.can_cast(s, c):
                continue
            seen.add(c.name)
            if len(foes) > 1 and _single_target_offense(c):
                acts.extend(f"{c.name}@{i}" for i in foes)
            else:
                acts.append(c.name)
        return acts


def _single_target_offense(card):
    """A card worth aiming: it hangs or lands something on ONE enemy.
    AoE ops (tgt 'enemies') hit everything, so a target index is noise."""
    ops = card.ops or []
    return (any(o.get("tgt") == "enemy" for o in ops)
            and not any(o.get("tgt") == "enemies" for o in ops))


def split_target(act):
    """('Kraken@2') -> ('Kraken', 2). Treasure cards carry '@tc' in
    their own names, so the suffix only counts when it is an integer."""
    if "@" in act:
        head, tail = act.rsplit("@", 1)
        if tail.isdigit():
            return head, int(tail)
    return act, 0


def apply_action(sim, s, act, dig_keep=None):
    """Execute an action name in the sim (PASS digs the worst card)."""
    if act == PASS:
        _dig(s, keep=dig_keep)
        return
    name, target = split_target(act)
    for c in s.hand:
        if c.name == name and sim.can_cast(s, c, target):
            sim.cast(s, c, target=target)
            return


def _card_value(cd):
    """Heuristic keep-value for dig ranking (higher = keep longer)."""
    if cd.kind in DMG_KINDS:
        return cd.damage
    if cd.kind == "prism":
        return 0.35
    if cd.kind == "heal":
        return 0.40
    if cd.kind == "shield":
        return abs(cd.percent)
    return cd.percent


def _dig(s, keep=None):
    if not s.deck or len(s.hand) < 7:
        return
    seen = set()
    def rank(cd):
        pend = (cd.kind == "blade" and cd.name in s.blades) or \
               (cd.kind in ("trap", "prism") and cd.name in s.traps)
        dup = cd.name in seen
        seen.add(cd.name)
        if pend: return (0, _card_value(cd))
        if dup:  return (1, _card_value(cd))
        if cd.kind not in DMG_KINDS: return (2, _card_value(cd))
        return (3, cd.damage)
    junk = [cd for cd in s.hand if cd.name != keep
            and cd not in s.player.fresh_tc]    # game rule: fresh TC stays
    if junk:
        pick = min(junk, key=rank)
        s.hand.remove(pick)
        s.player.graveyard.append(pick)      # keep deck conservation exact


# ---------------------------------------------------------------- Q-learning

class QAgent:
    turn_cost = 1.0

    def __init__(self, cards, decklist, school, dp_pol=None,
                 alpha=0.25, gamma=1.0, rng=None):
        self.feat = Featurizer(cards, decklist)
        self.Q = defaultdict(float)
        self.school, self.alpha, self.gamma = school, alpha, gamma
        self.dp_pol = dp_pol                    # warm-start advisor
        self.rng = rng or random.Random()       # seeded => reproducible
                                                # exploration, not just eval

    def greedy(self, sim, s, legal):
        k = self.feat.key(sim, s)
        return max(legal, key=lambda a: self.Q[(k, a)])

    def act(self, sim, s, eps, dp_w):
        legal = self.feat.legal(sim, s)
        if self.dp_pol and self.rng.random() < dp_w:
            pick = self.dp_pol(sim, s)          # advisor digs internally on pass
            if pick is None:
                return PASS, legal, False       # ...so don't dig again
            # a TARGET-AWARE advisor (with_focus) returns (card, index);
            # map it onto the same "name@i" action the agent uses, or
            # its demonstration names an action the agent cannot take
            # and the whole advisor silently does nothing
            card, tgt = pick if isinstance(pick, tuple) else (pick, None)
            for cand in ((f"{card.name}@{tgt}", card.name)
                         if tgt is not None else (card.name,)):
                if cand in legal:
                    return cand, legal, True
        if self.rng.random() < eps:
            return self.rng.choice(legal), legal, True
        return self.greedy(sim, s, legal), legal, True

    def train_episode(self, sim, eps, dp_w):
        """Backward Monte-Carlo updates: with a ~10-15 step horizon and sparse
        state visits, full-return backups propagate value in one episode where
        one-step Q-learning needs thousands."""
        s = sim.new_state()
        traj, won = [], False
        while True:
            tc_reflex(sim, s)           # TC reflex: draw free, cast learned
            k = self.feat.key(sim, s)
            a, legal, do_dig = self.act(sim, s, eps, dp_w)
            traj.append((k, a))
            if a == PASS and not do_dig:
                pass
            else:
                apply_action(sim, s, a)
            # The BOARD, not enemy 0. `State.boss_hp` is `enemies[0].hp`,
            # so on a multi-mob board this used to declare victory the
            # moment the first mob fell and hand out the full reward with
            # the rest still swinging -- the agent was being trained to
            # kill one thing, which is not the game. Identical on a 1v1
            # board, where enemy 0 dying IS the board clearing.
            if not any(e.alive for e in s.enemies):
                won = True
                turns = s.turn + 1
                break
            sim.end_round(s)
            if s.player_hp <= 0 or s.turn >= MAX_TURNS:
                turns = s.turn
                break
        G = 0.0 if won else -FAIL_PENALTY
        for k, a in reversed(traj):
            # `turn_cost` is the per-turn penalty that makes the agent
            # prefer fast kills. It is a knob because on a board the
            # agent cannot win, EVERY episode ends at -FAIL_PENALTY and
            # the only thing left to optimise is turn count — which
            # rewards dying sooner rather than finding the win.
            G = -self.turn_cost + self.gamma * G
            self.Q[(k, a)] += self.alpha * (G - self.Q[(k, a)])
        return turns, won

    def policy(self):
        """The greedy policy, as `(card, target)` — wizAi's aimed move.

        This used to match `c.name == a` against the raw action, and
        `Featurizer.legal` emits `"Snow Serpent@1"` on a multi-enemy
        board. So the comparison never matched, the policy returned None,
        and the agent **passed every single turn on any board with more
        than one mob** -- in `evaluate`, which is why a training run
        reported a flat 0% kill rate across 100,000 episodes, and in live
        play, where the same `policy()` is what `TrainedPolicy` calls.

        `train_episode` was unaffected: it goes through `apply_action`,
        which has always split the target. So the agent was learning a
        table it could then not use, and the only visible symptom was a
        kill rate that never moved.
        """
        def pol(sim, s):
            tc_reflex(sim, s)
            legal = self.feat.legal(sim, s)
            a = self.greedy(sim, s, legal)
            if a == PASS:
                _dig(s)
                return None
            name, target = split_target(a)
            for c in s.hand:
                if c.name == name and sim.can_cast(s, c, target):
                    # Aimed. `Sim._normalize_action` unpacks the tuple, and
                    # returning the bare card would throw away the target
                    # the agent just chose -- which is most of what the
                    # multi-enemy action space is for.
                    return c, target
            return None
        return pol


def make_board_sampler(school, hp_range, max_mobs=3, dmg=60):
    """A fresh board per episode, so one table covers many fights.

    Training against a single fixed board produces a table for that board
    and nothing else, because `Featurizer.key` is built from absolute
    quantities: the enemy health bucket (`hp // 250`), and on a
    multi-enemy board a `(living, weakest_index, ...)` tuple that is only
    present at all when more than one mob is up. Walk into a fight with a
    different mob count and the keys are a different *length*; walk into
    one with different health and they are a different bucket. Either way
    the table matches nothing and the policy is its own fallback.

    Retraining before every fight is the alternative, and it is not a
    real option -- you would have to know the board before you could
    learn to fight it. So the board is randomised instead: mob count and
    per-mob health resampled each episode, over the range the wizard is
    actually going to meet.

    Two details that look cosmetic and are not:

      * healths are drawn **independently**, not scaled from one number.
        Mobs that all start equal make `weakest_index` 0 in every opening
        state, and a real board of 515 beside 390 opens in the half that
        was never visited. Measured: 0% coverage.
      * they are deliberately **not sorted**. Ordering them big-to-small
        would put the weakest at the last index every time, which is the
        same degeneracy wearing a different hat.
    """
    lo, hi = int(min(hp_range)), int(max(hp_range))
    lo = max(1, lo)
    hi = max(lo, hi)

    def sample(rng):
        n = rng.randint(1, max(1, max_mobs))
        hps = [rng.randint(lo, hi) for _ in range(n)]
        board = [Boss(name=f"mob {i}", hp=hp, school=school, dmg=dmg)
                 for i, hp in enumerate(hps)]
        return board[0], board[1:]

    return sample


def train_agent(cards, decklist, school, boss, episodes=60000,
                warm=True, seed=0, player_hp=10**9, log=None,
                snap_every=5000, sideboard=None, player_stats=None,
                enemies=None, board_sampler=None, on_snapshot=None):
    """`player_stats` is the wizard's gear, as `Sim` takes it. Left out,
    training solves the fight for a wizard wearing nothing: every hit is
    priced below what it really lands for, so the table is optimal for a
    fight nobody is going to play.

    `enemies` are extra mobs beside `boss`, and the count is not a
    detail: `Featurizer.key` appends a `foes` tuple **only** when the
    board holds more than one enemy, so a table trained 1v1 and played
    against two mobs produces keys of a different length and cannot match
    a single state.

    `board_sampler(rng) -> (boss, extra_enemies)` resamples the board
    every episode, which is how one table comes to cover more than one
    fight -- see `make_board_sampler`. `boss`/`enemies` are still used
    for the periodic evaluation, so the checkpoint metric stays
    comparable across snapshots instead of drifting with whatever board
    happened to be up."""
    rng = random.Random(seed)
    sim = Sim(cards, decklist, school, boss, player_hp=player_hp, rng=rng,
              sideboard=sideboard, player_stats=player_stats,
              enemies=enemies)
    dp_pol = None
    if warm:
        V, pol, meta = solve(cards, decklist, boss, school)
        dp_pol = dp_policy(V, pol, meta, school)
    agent = QAgent(cards, decklist, school, dp_pol=dp_pol,
                   rng=random.Random(seed + 1))
    best = (-1.0, float("inf"), None)               # (kill%, ttk, Q snapshot)
    #: the board `evaluate` scores on. Held fixed while the training
    #: board is resampled, so successive snapshots are comparable and the
    #: "best checkpoint" pick is not decided by which board came up.
    eval_board = (sim.boss, list(sim.extra_bosses))

    def use_eval_board():
        sim.boss, sim.extra_bosses = eval_board[0], list(eval_board[1])

    for ep in range(episodes):
        frac = ep / episodes
        eps = max(0.02, 0.30 * (1 - frac))          # explore -> exploit
        dp_w = max(0.0, 0.50 * (1 - 2 * frac)) if warm else 0.0
        agent.alpha = 0.30 * (1 - 0.9 * frac)       # settle late training
        if board_sampler is not None:
            sim.boss, sim.extra_bosses = board_sampler(rng)
        agent.train_episode(sim, eps, dp_w)
        if (ep + 1) % snap_every == 0:
            use_eval_board()
            w, m = evaluate(sim, agent.policy(), n=2000)
            if on_snapshot is not None:
                try:
                    on_snapshot(ep + 1, w, m)
                except Exception:
                    pass          # a watching view never breaks training
            score = w - m / 1000.0                  # kill% first, speed second
            if score > best[0] - best[1] / 1000.0:
                best = (w, m, dict(agent.Q))
            if log and (ep + 1) % log == 0:
                print(f"    ep {ep+1:>6}: kill {w*100:5.1f}%  TTK {m:6.2f}")
    use_eval_board()
    if best[2] is not None:
        agent.Q = defaultdict(float, best[2])       # keep the best checkpoint
    return agent, sim


# ---------------------------------------------------------------- deck bandit

class DeckBandit:
    """UCB1 per boss over candidate decks; arms share nothing (different
    hand-mask spaces), each arm owns its Q-agent."""
    def __init__(self, cards, school, decks, boss, warm=True, player_hp=10**9):
        self.arms = {}
        self.stats = {}                              # name -> [n, mean_reward]
        for name, dl in decks.items():
            dp_pol = None
            if warm:
                V, pol, meta = solve(cards, dl, boss, school)
                dp_pol = dp_policy(V, pol, meta, school)
            self.arms[name] = (QAgent(cards, dl, school, dp_pol=dp_pol),
                               Sim(cards, dl, school, boss, player_hp=player_hp))
            self.stats[name] = [0, 0.0]
        self.t = 0

    WARMUP = 9000        # round-robin episodes before UCB takes over
    EMA = 1 / 1500.0     # recency weight: arms IMPROVE as they train, so the
                         # bandit must track a moving target, not lifetime mean

    def pick(self):
        self.t += 1
        names = list(self.stats)
        if self.t <= self.WARMUP:
            return names[self.t % len(names)]
        return max(names, key=lambda n: self.stats[n][1] +
                   2.5 * math.sqrt(math.log(self.t) / max(self.stats[n][0], 1)))

    def train(self, episodes=60000, per_arm_horizon=15000):
        for ep in range(episodes):
            name = self.pick()
            agent, sim = self.arms[name]
            n = self.stats[name][0]                 # this arm's own clock
            frac = min(n / per_arm_horizon, 1.0)
            agent.alpha = 0.30 * (1 - 0.9 * frac)
            t, won = agent.train_episode(sim, eps=max(0.02, 0.3 * (1 - frac)),
                                         dp_w=max(0.0, 0.5 * (1 - 2 * frac)))
            r = -t if won else -(t + FAIL_PENALTY)
            n, mu = self.stats[name]
            self.stats[name] = [n + 1, mu + self.EMA * (r - mu)]

    def best(self):
        name = max(self.stats, key=lambda n: self.stats[n][1])
        agent, sim = self.arms[name]
        return name, agent, sim
