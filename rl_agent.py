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
from w101_sim import Sim, Boss, load_cards, evaluate, strat_nuke_asap, make_blade_stack
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
        the scarcity signal the DP abstraction lacks."""
        hb = min(int(s.boss_hp // 250), 24)
        p = min(s.norm_pips + 2 * s.pow_pips, 14)
        bmask = sum(1 << i for i, n in enumerate(self.blades) if n in s.blades)
        tsig = tuple(s.traps[n][1] if n in s.traps else 0 for n in self.traps)
        dmask = sum(1 << i for i, n in enumerate(self.dmg)
                    if any(c.name == n for c in s.hand))
        nukes_left = min(sum(1 for c in s.hand if c.kind in DMG_KINDS) +
                         sum(1 for c in s.deck if c.kind in DMG_KINDS), 8)
        return (hb, p, bmask, tsig, dmask, nukes_left)

    def legal(self, sim, s):
        acts = [PASS]
        seen = set()
        for c in s.hand:
            if c.name not in seen and sim.can_cast(s, c):
                seen.add(c.name)
                acts.append(c.name)
        return acts


def apply_action(sim, s, act, dig_keep=None):
    """Execute an action name in the sim (PASS digs the worst card)."""
    if act == PASS:
        _dig(s, keep=dig_keep)
        return
    for c in s.hand:
        if c.name == act and sim.can_cast(s, c):
            sim.cast(s, c)
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
    junk = [cd for cd in s.hand if cd.name != keep]
    if junk:
        pick = min(junk, key=rank)
        s.hand.remove(pick)
        s.player.graveyard.append(pick)      # keep deck conservation exact


# ---------------------------------------------------------------- Q-learning

class QAgent:
    def __init__(self, cards, decklist, school, dp_pol=None,
                 alpha=0.25, gamma=1.0):
        self.feat = Featurizer(cards, decklist)
        self.Q = defaultdict(float)
        self.school, self.alpha, self.gamma = school, alpha, gamma
        self.dp_pol = dp_pol                    # warm-start advisor

    def greedy(self, sim, s, legal):
        k = self.feat.key(sim, s)
        return max(legal, key=lambda a: self.Q[(k, a)])

    def act(self, sim, s, eps, dp_w):
        legal = self.feat.legal(sim, s)
        if self.dp_pol and random.random() < dp_w:
            card = self.dp_pol(sim, s)          # advisor digs internally on pass
            if card is None:
                return PASS, legal, False       # ...so don't dig again
            if card.name in legal:
                return card.name, legal, True
        if random.random() < eps:
            return random.choice(legal), legal, True
        return self.greedy(sim, s, legal), legal, True

    def train_episode(self, sim, eps, dp_w):
        """Backward Monte-Carlo updates: with a ~10-15 step horizon and sparse
        state visits, full-return backups propagate value in one episode where
        one-step Q-learning needs thousands."""
        s = sim.new_state()
        traj, won = [], False
        while True:
            k = self.feat.key(sim, s)
            a, legal, do_dig = self.act(sim, s, eps, dp_w)
            traj.append((k, a))
            if a == PASS and not do_dig:
                pass
            else:
                apply_action(sim, s, a)
            if s.boss_hp <= 0:
                won = True
                turns = s.turn + 1
                break
            sim.end_round(s)
            if s.player_hp <= 0 or s.turn >= MAX_TURNS:
                turns = s.turn
                break
        G = 0.0 if won else -FAIL_PENALTY
        for k, a in reversed(traj):
            G = -1.0 + self.gamma * G
            self.Q[(k, a)] += self.alpha * (G - self.Q[(k, a)])
        return turns, won

    def policy(self):
        def pol(sim, s):
            legal = self.feat.legal(sim, s)
            a = self.greedy(sim, s, legal)
            if a == PASS:
                _dig(s)
                return None
            for c in s.hand:
                if c.name == a and sim.can_cast(s, c):
                    return c
            return None
        return pol


def train_agent(cards, decklist, school, boss, episodes=60000,
                warm=True, seed=0, player_hp=10**9, log=None, snap_every=5000):
    rng = random.Random(seed)
    sim = Sim(cards, decklist, school, boss, player_hp=player_hp, rng=rng)
    dp_pol = None
    if warm:
        V, pol, meta = solve(cards, decklist, boss, school)
        dp_pol = dp_policy(V, pol, meta, school)
    agent = QAgent(cards, decklist, school, dp_pol=dp_pol)
    best = (-1.0, float("inf"), None)               # (kill%, ttk, Q snapshot)
    for ep in range(episodes):
        frac = ep / episodes
        eps = max(0.02, 0.30 * (1 - frac))          # explore -> exploit
        dp_w = max(0.0, 0.50 * (1 - 2 * frac)) if warm else 0.0
        agent.alpha = 0.30 * (1 - 0.9 * frac)       # settle late training
        agent.train_episode(sim, eps, dp_w)
        if (ep + 1) % snap_every == 0:
            w, m = evaluate(sim, agent.policy(), n=2000)
            score = w - m / 1000.0                  # kill% first, speed second
            if score > best[0] - best[1] / 1000.0:
                best = (w, m, dict(agent.Q))
            if log and (ep + 1) % log == 0:
                print(f"    ep {ep+1:>6}: kill {w*100:5.1f}%  TTK {m:6.2f}")
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
