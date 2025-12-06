#!/usr/bin/env python3

import argparse
import os
import csv
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from specificTeam import (
    TEAM_ABBR_TO_ID,
    get_team_game_ids,
    train_hmm_for_game_team,
)

STATE_LABELS = ["HOT", "NEUTRAL", "COLD"]

OUTCOME_LABELS = ["score", "no_score", "turnover"]
OUTCOME_INDEX = {lab: i for i, lab in enumerate(OUTCOME_LABELS)}

MAX_WORKERS = 16


def compute_empirical_viterbi_transitions(v_paths, n_states=3):
    """
    v_paths: list of lists of integer states (0..n_states-1) from Viterbi.

    Returns:
      counts[i, j] = # of i->j transitions in Viterbi paths
      probs[i, j]  = counts row-normalized
      total_possessions = total # of states across all paths
    """
    counts = np.zeros((n_states, n_states), dtype=int)
    total_poss = 0

    for path in v_paths:
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


def compute_empirical_viterbi_emissions(v_paths, outcome_lists, n_states=3):
    """
    v_paths: list of state sequences from Viterbi
    outcome_lists: matching list of outcome sequences (strings 'score','no_score','turnover')

    Returns:
      counts[i, k] = # of outcome k when in state i
      totals[i]    = # of times we are in state i
      probs[i, k]  = P(outcome k | state i)
    """
    assert len(v_paths) == len(outcome_lists), "paths vs outcomes length mismatch"

    counts = np.zeros((n_states, len(OUTCOME_LABELS)), dtype=int)
    totals = np.zeros(n_states, dtype=int)

    for path, outs in zip(v_paths, outcome_lists):
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


def run_for_team(team_abbr: str, season: str):
    team_abbr = team_abbr.upper()
    print(f"\n======================")
    print(f"TEAM: {team_abbr}, SEASON: {season}")
    print(f"======================")

    try:
        game_ids = get_team_game_ids(team_abbr, season=season)
    except Exception as e:
        print(f"[ERROR] fetching game IDs for {team_abbr}: {e}")
        return team_abbr

    if not game_ids:
        print(f"[WARN] No games found for {team_abbr} in {season}")
        return team_abbr

    print(f"Found {len(game_ids)} games for {team_abbr} in {season}")

    all_paths = []
    all_outcomes = []

    for gid in game_ids:
        print(f"\n=== {team_abbr}: Processing game {gid} ===")

        # 1) Get Viterbi path from HMM
        try:
            v_path = train_hmm_for_game_team(gid, team_abbr)
        except Exception as e:
            print(f"  [ERROR] HMM/Viterbi failed for game {gid}: {e}")
            continue

        if v_path is None or len(v_path) < 2:
            print(f"  [WARN] skipping game {gid}: not enough possessions.")
            continue

        # 2) Load possessions outcomes from [gid]_outputs/[gid]_[TEAM]_possessions.csv
        game_dir = f"{gid}_outputs"
        poss_file = os.path.join(game_dir, f"{gid}_{team_abbr}_possessions.csv")

        if not os.path.exists(poss_file):
            print(f"  [WARN] possessions file not found: {poss_file}, skipping this game.")
            continue

        try:
            df_poss = pd.read_csv(poss_file)
        except Exception as e:
            print(f"  [ERROR] reading possessions {poss_file}: {e}")
            continue

        # Your files use 'result' with values 'score', 'no_score', 'turnover'
        if "result" not in df_poss.columns:
            print(f"  [WARN] no 'result' column in {poss_file}, skipping this game.")
            continue

        outcomes = df_poss["result"].tolist()
        if len(outcomes) != len(v_path):
            print(
                f"  [WARN] length mismatch in game {gid}: "
                f"{len(outcomes)} results vs {len(v_path)} states. Skipping."
            )
            continue

        all_paths.append(list(v_path))
        all_outcomes.append(outcomes)

    if not all_paths:
        print(f"[WARN] No valid HMM+result pairs for {team_abbr}")
        return team_abbr

    # ---- transitions ----
    counts_T, probs_T, total_poss = compute_empirical_viterbi_transitions(all_paths, n_states=3)

    print(f"\n=== {team_abbr} VITERBI transition probabilities (from -> to) ===")
    for i in range(3):
        row = "  " + "  ".join(f"{probs_T[i, j]:.3f}" for j in range(3))
        print(row)
    print(f"Total possessions contributing: {total_poss}")

    # ---- emissions ----
    counts_E, totals_E, probs_E = compute_empirical_viterbi_emissions(all_paths, all_outcomes, n_states=3)

    print(f"\n=== {team_abbr} VITERBI emission probabilities P(result | state) ===")
    for i, lab in enumerate(STATE_LABELS):
        print(f"  State {i} ({lab}), N={totals_E[i]}:")
        print("    " + ", ".join(
            f"P({OUTCOME_LABELS[k]}|{lab})={probs_E[i, k]:.3f}"
            for k in range(len(OUTCOME_LABELS))
        ))

    # ---- save to CSVs ----
    season_clean = season.replace("-", "")
    trans_out = f"{team_abbr}_{season_clean}_HMM_transitions.csv"
    emis_out = f"{team_abbr}_{season_clean}_HMM_emissions.csv"

    with open(trans_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from_state", "to_state", "from_label", "to_label", "count", "P_to_given_from"])
        for i in range(3):
            for j in range(3):
                w.writerow([i, j, STATE_LABELS[i], STATE_LABELS[j],
                            int(counts_T[i, j]), float(probs_T[i, j])])

    with open(emis_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "state_idx", "state_label", "total_in_state",
            "count_score", "count_no_score", "count_turnover",
            "P_score_given_state", "P_no_score_given_state", "P_turnover_given_state",
        ])
        for i in range(3):
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

    print(f"[OK] Saved transitions to {trans_out}")
    print(f"[OK] Saved emissions to {emis_out}")

    return team_abbr


def main():
    ap = argparse.ArgumentParser(description="Run Viterbi-based momentum pipeline for ALL 30 NBA teams.")
    ap.add_argument(
        "season",
        nargs="?",
        default="2024-25",
        help="Season string, e.g. '2024-25' (default: 2024-25).",
    )
    args = ap.parse_args()
    season = args.season

    team_list = sorted(TEAM_ABBR_TO_ID.keys())
    print(f"Running for {len(team_list)} teams: {', '.join(team_list)}")

    # Parallel over teams
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_for_team, team, season): team for team in team_list}
        for fut in as_completed(futures):
            team = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] team {team} crashed: {e}")

    print("\n[DONE] All teams processed.")


if __name__ == "__main__":
    main()

