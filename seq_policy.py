"""
Sequence-capable student: a recurrent policy over the same features.

HYPOTHESIS UNDER TEST (and largely refuted — read on): three
negatives suggested the linear generalist could not hold a plan
across steps, and the offline ladder's ~75% ceiling suggested the
binding constraint was student capacity.

This is the minimal student that adds capacity while changing nothing
else — same features, same demonstrations, same benchmark. What the
experiment actually found is that a memoryless linear policy, simply
REFIT on the same demonstrations, reaches 76.8% on the X-pip bar the
class was said to be unable to express; recurrence adds only ~4
points there and loses ground in aggregate. The capacity story did
not survive its own ablation. A hidden state h carries the plan and MODULATES the
linear ranking:

    w_eff(t) = w0 + U h(t)              scoring weights this step
    q_i(t)   = w_eff(t) . phi_i(t)      rank legal actions
    h(t+1)   = tanh(Wh h(t) + Wx x(t) + b)      x(t) = phi[chosen]

so the policy's preferences at step t depend on what it committed to
at steps < t. With U = 0 it IS the linear policy, so training
warm-starts from the linear BC solution (`from_linear`).

IMPORTANT: w0 keeps training too, so a gain over the linear baseline
is NOT automatically a recurrent effect — `freeze_U=True` runs the
same optimizer with the recurrence pinned off, and that ablation is
what attributes the difference. On this repo's benchmark it turned
out to matter enormously: almost all of the improvement was the
episode-level Adam optimizer, not the memory (see
results_sequence_model.json and the README).

Trained by filtered behavior cloning over whole episodes with
hand-written backprop-through-time (numpy only, no autodiff). The
numerical gradient check in tests/test_seq.py is what makes these
gradients trustworthy; it does not, by itself, validate the
experimental claims built on top of them.
"""
import json

import numpy as np

from generalist import FEATS, legal_actions, phi
from rl_agent import PASS, _dig
from w101_sim import tc_reflex

H_DEFAULT = 8


def _softmax_masked(q, mask):
    if not mask.any():                   # no legal action: uniform,
        return np.full(len(q), 1.0 / len(q))   # never NaN
    z = np.where(mask, q, -np.inf)
    z = z - z.max()
    e = np.where(mask, np.exp(z), 0.0)
    return e / e.sum()


class RecurrentQ:
    """Linear ranking whose weights are modulated by a recurrent state."""

    def __init__(self, H=H_DEFAULT, w0=None, U=None, Wh=None, Wx=None,
                 b=None, seed=0):
        F = len(FEATS)
        rng = np.random.default_rng(seed)
        self.H = H
        # np.array (not asarray): Adam mutates these in place, and
        # asarray would ALIAS a caller's array — from_linear(w) would
        # then silently overwrite the linear baseline during training
        self.p = {
            "w0": np.zeros(F) if w0 is None else np.array(w0, float),
            # U = 0 => exactly the linear policy at init
            "U": np.zeros((F, H)) if U is None else np.array(U, float),
            "Wh": (rng.normal(0, 0.1, (H, H)) if Wh is None
                   else np.array(Wh, float)),
            "Wx": (rng.normal(0, 0.1, (H, F)) if Wx is None
                   else np.array(Wx, float)),
            "b": np.zeros(H) if b is None else np.array(b, float),
        }

    @classmethod
    def from_linear(cls, w_linear, H=H_DEFAULT, seed=0):
        """Warm start: identical to the linear policy until U moves."""
        return cls(H=H, w0=np.asarray(w_linear, float), seed=seed)

    # ---------------------------------------------------------- scoring
    def w_eff(self, h):
        return self.p["w0"] + self.p["U"] @ h

    def step_h(self, h, x):
        return np.tanh(self.p["Wh"] @ h + self.p["Wx"] @ x + self.p["b"])

    # ------------------------------------------------- loss + gradients
    def episode_grads(self, P, M, C):
        """Forward+backward over ONE episode. P (T,A,F), M (T,A) bool,
        C (T,) chosen index. Returns (loss, grads dict)."""
        T = len(C)
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        hs = [np.zeros(self.H)]
        ps, xs = [], []
        loss = 0.0
        for t in range(T):
            we = self.w_eff(hs[t])
            q = P[t] @ we
            pr = _softmax_masked(q, M[t])
            loss -= np.log(max(pr[C[t]], 1e-12))
            ps.append(pr)
            x = P[t][C[t]]
            xs.append(x)
            hs.append(self.step_h(hs[t], x))
        dh = np.zeros(self.H)
        for t in range(T - 1, -1, -1):
            da = dh * (1.0 - hs[t + 1] ** 2)
            g["Wh"] += np.outer(da, hs[t])
            g["Wx"] += np.outer(da, xs[t])
            g["b"] += da
            dh_prev = self.p["Wh"].T @ da
            dq = ps[t].copy()
            dq[C[t]] -= 1.0
            dq = np.where(M[t], dq, 0.0)
            dwe = P[t].T @ dq
            g["w0"] += dwe
            g["U"] += np.outer(dwe, hs[t])
            dh = dh_prev + self.p["U"].T @ dwe
        return loss, g

    # ---------------------------------------------------------- inference
    def policy(self):
        """Fresh hidden state per fight (detected by state identity)."""
        box = {"s": None, "h": np.zeros(self.H)}

        def pol(sim, s):
            if s is not box["s"]:
                box["s"] = s
                box["h"] = np.zeros(self.H)
            tc_reflex(sim, s)
            acts = legal_actions(sim, s)
            we = self.w_eff(box["h"])
            feats = [phi(sim, s, a) for a in acts]
            i = int(np.argmax([f @ we for f in feats]))
            box["h"] = self.step_h(box["h"], feats[i])
            a = acts[i]
            if a == PASS:
                _dig(s)
                return None
            return a
        return pol

    # ------------------------------------------------------------- io
    def save(self, path):
        json.dump({"features": FEATS, "H": self.H,
                   **{k: v.tolist() for k, v in self.p.items()}},
                  open(path, "w", encoding="utf-8"), indent=1)

    @classmethod
    def load(cls, path):
        """Zero-pads a policy saved before new features existed (the
        same forward-compatibility contract GeneralistQ.load has)."""
        d = json.load(open(path, encoding="utf-8"))
        F, H = len(FEATS), d["H"]
        w0 = np.zeros(F)
        w0[:len(d["w0"])] = d["w0"][:F]
        U = np.zeros((F, H))
        U_old = np.asarray(d["U"], float)
        U[:U_old.shape[0]] = U_old[:F]
        Wx = np.zeros((H, F))
        Wx_old = np.asarray(d["Wx"], float)
        Wx[:, :Wx_old.shape[1]] = Wx_old[:, :F]
        return cls(H=H, w0=w0, U=U, Wh=d["Wh"], Wx=Wx, b=d["b"])


# ------------------------------------------------------------- training

def episodes_from(data, winners_only=True):
    """Split a flat dataset into per-episode (P, M, C) tuples. The
    generators log steps in REVERSE order within an episode (they
    accumulate return-to-go backwards), so each episode is flipped
    back to forward time — recurrence is meaningless otherwise."""
    ep = data["ep"]
    out = []
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        if winners_only and not data["won"][idx[0]]:
            continue
        idx = idx[::-1]                      # restore forward time
        out.append((data["phis"][idx], data["mask"][idx],
                    data["chosen"][idx]))
    return out


class _Adam:
    def __init__(self, params, lr=0.01):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, clip=5.0):
        self.t += 1
        norm = np.sqrt(sum(float((g ** 2).sum())
                           for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        for k in params:
            gk = grads[k] * scale
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk
            self.v[k] = 0.999 * self.v[k] + 0.001 * gk ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def train_seq_bc(data, w_linear=None, H=H_DEFAULT, epochs=12, lr=0.01,
                 batch=16, seed=0, log=None, val=None, freeze_U=False):
    """Filtered BC with BPTT. `val` is an optional callable taking the
    model and returning a score; the best-scoring snapshot is kept.
    `freeze_U=True` pins U (and the recurrent parameters) at zero, so
    the model stays EXACTLY linear while w0 still gets the identical
    Adam-over-episodes optimizer — the ablation that separates
    'recurrence helped' from 'the optimizer helped'."""
    eps = episodes_from(data, winners_only=True)
    model = (RecurrentQ.from_linear(w_linear, H=H, seed=seed)
             if w_linear is not None else RecurrentQ(H=H, seed=seed))
    opt = _Adam(model.p, lr=lr)
    rng = np.random.default_rng(seed)
    best = (-np.inf, None)
    if val is not None:                  # epoch 0 = the warm start:
        s0 = val(model)                  # training must be able to
        best = (s0, {k: v.copy() for k, v in model.p.items()})
        if log:                          # fall back to it
            log(f"    epoch  0 (warm start)      val {s0*100:5.1f}%")
    for epoch in range(epochs):
        order = rng.permutation(len(eps))
        tot, nsteps = 0.0, 0
        for lo in range(0, len(order), batch):
            chunk = order[lo:lo + batch]
            acc = {k: np.zeros_like(v) for k, v in model.p.items()}
            for j in chunk:
                P, M, C = eps[j]
                loss, g = model.episode_grads(P, M, C)
                for k in acc:
                    acc[k] += g[k]
                tot += loss
                nsteps += len(C)
            for k in acc:
                acc[k] /= max(len(chunk), 1)
            if freeze_U:
                for k in ("U", "Wh", "Wx", "b"):
                    acc[k][...] = 0.0
            opt.step(model.p, acc)
        score = None
        if val is not None:
            score = val(model)
            if score > best[0]:
                best = (score, {k: v.copy() for k, v in model.p.items()})
        if log:
            log(f"    epoch {epoch+1:>2}/{epochs}  loss/step "
                f"{tot/max(nsteps,1):.4f}"
                + (f"  val {score*100:5.1f}%" if score is not None
                   else ""))
    if best[1] is not None:
        model.p = best[1]
    return model
