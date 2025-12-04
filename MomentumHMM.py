#!/usr/bin/env python3
"""
3-state HMM with EM for NBA momentum.

- Hidden states (K=3): HOT / NEUTRAL / COLD
- Emissions (M=3): S (score), 0 (no score), T (turnover)
- Input: a .txt file containing a sequence like "SS0ST0SSS0T0SS..."
- Output:
    - <outbase>_results.npz : A, B, pi, v_path, logL
    - <outbase>_states.csv  : idx, outcome, state
"""

import argparse
import csv
from typing import List, Dict, Tuple, Optional

import numpy as np

# --------------------------------------------------
# HMM core
# --------------------------------------------------

class DiscreteHMM:
    """
    Discrete Hidden Markov Model with:
      - Baum–Welch (EM) training
      - Viterbi decoding
      - Posterior state probabilities
    """

    def __init__(
        self,
        n_states: int,
        n_obs: int,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        pi: Optional[np.ndarray] = None,
        seed: int = 0,
        smoothing: float = 1e-8,
    ):
        self.n_states = n_states
        self.n_obs = n_obs
        self.rng = np.random.default_rng(seed)
        self.smoothing = smoothing

        # Random initialization if not provided
        if A is None:
            A = self.rng.random((n_states, n_states))
            A = A / A.sum(axis=1, keepdims=True)
        if B is None:
            B = self.rng.random((n_states, n_obs))
            B = B / B.sum(axis=1, keepdims=True)
        if pi is None:
            pi = self.rng.random(n_states)
            pi = pi / pi.sum()

        self.A = A.astype(float)
        self.B = B.astype(float)
        self.pi = pi.astype(float)

    def _log(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.clip(x, self.smoothing, None))

    # ---------- forward-backward ----------

    def _forward(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        T = len(obs)
        K = self.n_states

        alpha = np.zeros((T, K))
        scales = np.zeros(T)

        # t = 0
        alpha[0] = self.pi * self.B[:, obs[0]]
        scales[0] = alpha[0].sum()
        alpha[0] /= max(scales[0], self.smoothing)

        # t >= 1
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            scales[t] = alpha[t].sum()
            alpha[t] /= max(scales[t], self.smoothing)

        log_likelihood = np.sum(np.log(np.clip(scales, self.smoothing, None)))
        return alpha, scales, log_likelihood

    def _backward(self, obs: np.ndarray, scales: np.ndarray) -> np.ndarray:
        T = len(obs)
        K = self.n_states
        beta = np.zeros((T, K))

        # t = T - 1
        beta[-1] = 1.0 / max(scales[-1], self.smoothing)

        # t <= T - 2
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A * self.B[:, obs[t + 1]] * beta[t + 1]).sum(axis=1)
            beta[t] /= max(scales[t], self.smoothing)

        return beta

    def _expectation(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        alpha, scales, log_likelihood = self._forward(obs)
        beta = self._backward(obs, scales)

        T = len(obs)
        K = self.n_states

        # gamma[t, i] ∝ alpha[t, i] * beta[t, i]
        gamma = alpha * beta
        gamma /= np.clip(gamma.sum(axis=1, keepdims=True), self.smoothing, None)

        # xi[t, i, j] ∝ alpha[t,i] * A[i,j] * B[j, o_{t+1}] * beta[t+1, j]
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            numer = (alpha[t][:, None] * self.A) * (
                self.B[:, obs[t + 1]] * beta[t + 1]
            )[None, :]
            denom = np.clip(numer.sum(), self.smoothing, None)
            xi[t] = numer / denom

        return gamma, xi, log_likelihood

    # ---------- EM (Baum–Welch) ----------

    def fit(
        self,
        obs: List[int],
        n_iter: int = 300,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> Dict[str, float]:
        obs = np.asarray(obs, dtype=int)
        last_ll = -np.inf
        history: Dict[str, float] = {}

        for it in range(1, n_iter + 1):
            gamma, xi, ll = self._expectation(obs)

            # pi update
            self.pi = np.clip(gamma[0], self.smoothing, None)
            self.pi /= self.pi.sum()

            # A update
            xi_sum = xi.sum(axis=0)
            A_new = xi_sum + self.smoothing
            A_new /= A_new.sum(axis=1, keepdims=True)
            self.A = A_new

            # B update
            K = self.n_states
            M = self.n_obs
            B_new = np.zeros((K, M)) + self.smoothing
            for k in range(K):
                for t, o in enumerate(obs):
                    B_new[k, o] += gamma[t, k]
            B_new /= B_new.sum(axis=1, keepdims=True)
            self.B = B_new

            if verbose:
                print(f"Iter {it:3d}  logL={ll:.6f}")

            if abs(ll - last_ll) < tol:
                last_ll = ll
                break
            last_ll = ll

        history["log_likelihood"] = float(last_ll)
        history["iterations"] = it
        return history

    # ---------- Viterbi ----------

    def viterbi(self, obs: List[int]) -> List[int]:
        obs = np.asarray(obs, dtype=int)
        T = len(obs)
        K = self.n_states

        logA = self._log(self.A)
        logB = self._log(self.B)
        logpi = self._log(self.pi)

        dp = np.full((T, K), -np.inf)
        ptr = np.zeros((T, K), dtype=int)

        # init
        dp[0] = logpi + logB[:, obs[0]]

        # DP
        for t in range(1, T):
            for j in range(K):
                scores = dp[t - 1] + logA[:, j] + logB[j, obs[t]]
                ptr[t, j] = int(np.argmax(scores))
                dp[t, j] = np.max(scores)

        # backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = ptr[t + 1, path[t + 1]]

        return path.tolist()

    def posterior_state_probs(self, obs: List[int]) -> np.ndarray:
        obs = np.asarray(obs, dtype=int)
        gamma, _, _ = self._expectation(obs)
        return gamma


# --------------------------------------------------
# S/0/T helpers + I/O
# --------------------------------------------------

EVENT_TO_ID = {"S": 0, "0": 1, "T": 2}
ID_TO_EVENT = {v: k for k, v in EVENT_TO_ID.items()}


def read_s0t_sequence(path: str) -> List[str]:
    """
    Read a .txt file containing S/0/T sequence.
    Accepts:
      - a single string like 'SS0ST0SS'
      - or with whitespace/newlines, which will be stripped out.
    """
    text = open(path, "r").read()
    # strip whitespace and newlines
    chars = [ch for ch in text if not ch.isspace()]
    for ch in chars:
        if ch not in EVENT_TO_ID:
            raise ValueError(
                f"Invalid character '{ch}' in sequence. Allowed: {list(EVENT_TO_ID.keys())}"
            )
    return chars


def encode_events(events: List[str]) -> List[int]:
    return [EVENT_TO_ID[e] for e in events]


def write_states_csv(
    out_path: str, events: List[str], v_path: List[int], labels: List[str]
) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "outcome", "state"])
        for i, (e, s) in enumerate(zip(events, v_path)):
            w.writerow([i, e, labels[s]])


# --------------------------------------------------
# Main CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3-state HMM with EM for S/0/T possession sequences."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input .txt file with S/0/T sequence (e.g., from dataPull).",
    )
    parser.add_argument(
        "--states",
        type=int,
        default=3,
        help="Number of hidden states (default 3 = HOT/NEUTRAL/COLD).",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=300,
        help="Max EM iterations (default 300).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="EM convergence tolerance (default 1e-6).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default 0)."
    )
    parser.add_argument(
        "--outbase",
        default="hmm_3state",
        help="Base name for outputs (default 'hmm_3state').",
    )
    args = parser.parse_args()

    # 1) Read sequence as S/0/T characters
    events = read_s0t_sequence(args.in_path)
    obs = encode_events(events)
    T = len(obs)
    print(f"Loaded {T} possessions from {args.in_path}")

    K = args.states
    M = 3  # S/0/T

    # 2) Initialize HMM
    A0 = None
    B0 = None
    pi0 = None

    # Semi-informative init for 3 states: HOT, NEUTRAL, COLD
    if K == 3:
        A0 = np.array(
            [
                [0.85, 0.10, 0.05],  # HOT
                [0.15, 0.70, 0.15],  # NEUTRAL
                [0.05, 0.10, 0.85],  # COLD
            ],
            dtype=float,
        )
        B0 = np.array(
            [
                [0.65, 0.25, 0.10],  # HOT: more scores
                [0.45, 0.35, 0.20],  # NEUTRAL
                [0.25, 0.45, 0.30],  # COLD: fewer scores, more 0/T
            ],
            dtype=float,
        )
        pi0 = np.array([0.34, 0.32, 0.34], dtype=float)

    hmm = DiscreteHMM(
        n_states=K,
        n_obs=M,
        A=A0,
        B=B0,
        pi=pi0,
        seed=args.seed,
        smoothing=1e-6,
    )

    # 3) Train with EM
    hist = hmm.fit(obs, n_iter=args.iter, tol=args.tol, verbose=False)

    # 4) Relabel states by descending P(S | state), so:
    #    state 0 = HOT, 1 = NEUTRAL, 2 = COLD (when K=3).
    order = np.argsort(-hmm.B[:, 0])  # column 0 is 'S'
    A = hmm.A[order][:, order]
    B = hmm.B[order]
    pi = hmm.pi[order]
    remap = {old: new for new, old in enumerate(order)}
    v_path_raw = hmm.viterbi(obs)
    v_path = [remap[s] for s in v_path_raw]

    if K == 3:
        labels = ["HOT", "NEUTRAL", "COLD"]
    else:
        labels = [f"STATE_{i}" for i in range(K)]

    # 5) Save results
    np.savez(
        f"{args.outbase}_results.npz",
        A=A,
        B=B,
        pi=pi,
        v_path=np.array(v_path, dtype=int),
        logL=hist["log_likelihood"],
    )

    write_states_csv(f"{args.outbase}_states.csv", events, v_path, labels)

    # 6) Print summary
    np.set_printoptions(precision=3, suppress=True)
    print("Training complete.")
    print("logL:", hist["log_likelihood"])
    print("A (rows in label order):\n", A)
    print("B (cols = S,0,T):\n", B)
    print("pi:", pi)
    print("First 20 decoded states:", [labels[s] for s in v_path[:20]])
    print(f"Saved: {args.outbase}_results.npz")
    print(f"Saved: {args.outbase}_states.csv")


if __name__ == "__main__":
    main()
