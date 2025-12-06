#!/usr/bin/env python3

import os
import csv
import numpy as np
import pandas as pd
from collections import defaultdict

POSSESSIONS_SUFFIX = "_possessions.csv"
STATES_SUFFIX = "_states.csv"   # e.g. "0022400073_WAS_states.csv"

STATE_LABELS = ["HOT", "NEUTRAL", "COLD"]

OUTCOME_INDEX = {"S": 0, "0": 1, "T": 2}
OUTCOME_LABELS = ["S", "0", "T"]


def find_game_output_dirs(root="."):
    """
    Find all directories in `root` that end with '_outputs'.
    Returns a list of absolute paths.
    """
    dirs = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and name.endswith("_outputs"):
            dirs.append(path)
    return sorted(dirs)


def parse_game_and_team(filename):

    base = os.path.basename(filename)
    if not base.endswith(POSSESSIONS_SUFFIX):
        return None, None
    core = base[:-len(POSSESSIONS_SUFFIX)]  # '0022400073_WAS'
    parts = core.split("_")
    if len(parts) < 2:
        return None, None
    game_id = parts[0]
    team_abbr = parts[1]
    return game_id, team_abbr


def load_states(states_path):
    df = pd.read_csv(states_path)

    if "state" not in df.columns:
        raise ValueError(f"'state' column not found in {states_path}")

    vals = df["state"].tolist()

    if all(isinstance(v, (int, np.integer)) for v in vals):
        return [int(v) for v in vals]

    label_to_idx = {lab: i for i, lab in enumerate(STATE_LABELS)}
    states_idx = []
    for v in vals:
        if isinstance(v, str) and v in label_to_idx:
            states_idx.append(label_to_idx[v])
        else:
            raise ValueError(
                f"Unexpected state value '{v}' in {states_path}. "
                f"Adjust STATE_LABELS or file format."
            )
    return states_idx


def compute_transitions(paths, n_states=3):
    counts = np.zeros((n_states, n_states), dtype=int)
    total_poss = 0

    for path in paths:
        if len(path) < 2:
            continue
        total_poss += len(path)
        for t in range(len(path) - 1):
            i = path[t]
            j = path[t + 1]
            if 0 <= i < n_states and 0 <= j < n_states:
                counts[i, j] += 1

    probs = counts.astype(float)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs /= row_sums

    return counts, probs, total_poss


def compute_emissions(paths, outcome_lists, n_states=3):
    assert len(paths) == len(outcome_lists), "paths and outcomes length mismatch"

    counts = np.zeros((n_states, len(OUTCOME_LABELS)), dtype=int)
    totals = np.zeros(n_states, dtype=int)

    for path, outs in zip(paths, outcome_lists):
        if len(path) != len(outs):
            continue
        for st, out in zip(path, outs):
            if st < 0 or st >= n_states:
                continue
            if out not in OUTCOME_INDEX:
                continue
            k = OUTCOME_INDEX[out]
            counts[st, k] += 1
            totals[st] += 1

    probs = counts.astype(float)
    for i in range(n_states):
        if totals[i] > 0:
            probs[i, :] /= float(totals[i])
        else:
            probs[i, :] = 0.0

    return counts, totals, probs


def main():

    team_paths = defaultdict(list)      
    team_outcomes = defaultdict(list)   

    game_dirs = find_game_output_dirs(".")
    print(f"Found {len(game_dirs)} *_outputs directories.")

    for gdir in game_dirs:
        game_dirname = os.path.basename(gdir)              
        game_id = game_dirname.split("_outputs")[0]       

        for fname in os.listdir(gdir):
            if not fname.endswith(POSSESSIONS_SUFFIX):
                continue

            poss_path = os.path.join(gdir, fname)
            gid, team_abbr = parse_game_and_team(fname)
            if gid is None or team_abbr is None:
                continue

            states_name = f"{gid}_{team_abbr}{STATES_SUFFIX}"
            states_path = os.path.join(gdir, states_name)
            if not os.path.exists(states_path):
                print(f"[WARN] Missing states file for {gid} {team_abbr}: {states_path}")
                continue
                
            try:
                df_poss = pd.read_csv(poss_path)
            except Exception as e:
                print(f"[ERROR] reading possessions {poss_path}: {e}")
                continue

            if "outcome" not in df_poss.columns:
                print(f"[WARN] No 'outcome' column in {poss_path}, skipping.")
                continue

            outcomes = df_poss["outcome"].tolist()

            try:
                states_idx = load_states(states_path)
            except Exception as e:
                print(f"[ERROR] reading states {states_path}: {e}")
                continue

            if len(states_idx) != len(outcomes):
                print(
                    f"[WARN] length mismatch {gid} {team_abbr}: "
                    f"{len(states_idx)} states vs {len(outcomes)} outcomes, skipping."
                )
                continue

            team_paths[team_abbr].append(states_idx)
            team_outcomes[team_abbr].append(outcomes)

    if not team_paths:
        print("No valid (states, outcome) pairs found for any team.")
        return

    print(f"\nFound data for {len(team_paths)} teams: {', '.join(sorted(team_paths.keys()))}")

    for team_abbr in sorted(team_paths.keys()):
        paths = team_paths[team_abbr]
        outs = team_outcomes[team_abbr]

        print(f"\n=== TEAM {team_abbr} ===")
        n_states = len(STATE_LABELS)

        counts_T, probs_T, total_poss = compute_transitions(paths, n_states=n_states)
        print("Transition counts (from -> to):")
        print(counts_T)
        print("Transition probabilities (from -> to):")
        for i in range(n_states):
            row = "  " + "  ".join(f"{probs_T[i, j]:.3f}" for j in range(n_states))
            print(row)
        print(f"Total possessions for {team_abbr}: {total_poss}")

        counts_E, totals_E, probs_E = compute_emissions(paths, outs, n_states=n_states)
        print("\nEmission probabilities P(outcome | state):")
        for i, lab in enumerate(STATE_LABELS):
            print(f"  State {i} ({lab}), N={totals_E[i]}:")
            line = "    " + ", ".join(
                f"P({OUTCOME_LABELS[k]}|{lab})={probs_E[i, k]:.3f}"
                for k in range(len(OUTCOME_LABELS))
            )
            print(line)

        trans_out = f"{team_abbr}_agg_HMM_transitions.csv"
        with open(trans_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["from_state", "to_state", "from_label", "to_label", "count", "P_to_given_from"])
            for i in range(n_states):
                for j in range(n_states):
                    w.writerow([i, j, STATE_LABELS[i], STATE_LABELS[j],
                                int(counts_T[i, j]), float(probs_T[i, j])])
        print(f"[OK] Wrote transitions to {trans_out}")

        emis_out = f"{team_abbr}_agg_HMM_emissions.csv"
        with open(emis_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "state_idx",
                "state_label",
                "total_in_state",
                "count_S",
                "count_0",
                "count_T",
                "P_S_given_state",
                "P_0_given_state",
                "P_T_given_state",
            ])
            for i in range(n_states):
                w.writerow([
                    i,
                    STATE_LABELS[i],
                    int(totals_E[i]),
                    int(counts_E[i, 0]),
                    int(counts_E[i, 1]),
                    int(counts_E[i, 2]),
                    float(probs_E[i, 0]),
                    float(probs_E[i, 1]),
                    float(probs_E[i, 2]),
                ])
        print(f"[OK] Wrote emissions to {emis_out}")


if __name__ == "__main__":
    main()

