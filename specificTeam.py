

import os
import sys
import glob
import argparse
import numpy as np

from nba_api.stats.endpoints import leaguegamefinder

from dataPull import save_possessions_csv
from probabilities import probs_for_game
from MomentumHMM import DiscreteHMM, encode_events

TEAM_ABBR_TO_ID = {
    "ATL": 1610612737, "BOS": 1610612738, "CLE": 1610612739, "NOP": 1610612740,
    "CHI": 1610612741, "DAL": 1610612742, "DEN": 1610612743, "GSW": 1610612744,
    "HOU": 1610612745, "LAC": 1610612746, "LAL": 1610612747, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "BKN": 1610612751, "NYK": 1610612752,
    "ORL": 1610612753, "IND": 1610612754, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "OKC": 1610612760,
    "TOR": 1610612761, "UTA": 1610612762, "MEM": 1610612763, "WAS": 1610612764,
    "DET": 1610612765, "CHA": 1610612766,
}


def get_team_game_ids(team_abbr: str,
                      season: str = "2024-25",
                      season_type: str = "Regular Season") -> list:
    """
    Use nba_api to get all game IDs for a team in a given season.
    """
    team_abbr = team_abbr.upper()
    if team_abbr not in TEAM_ABBR_TO_ID:
        raise ValueError(f"Unknown team abbreviation '{team_abbr}'")

    team_id = TEAM_ABBR_TO_ID[team_abbr]

    gf = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable=season_type,
    )
    games = gf.get_data_frames()[0]
    game_ids = games["GAME_ID"].unique().tolist()
    return game_ids


def train_hmm_for_game_team(game_id: str,
                            team_abbr: str,
                            seed: int = 0) -> list:
    team_abbr = team_abbr.upper()

    print(f"  [dataPull] pulling possessions for game {game_id} ...")
    save_possessions_csv(game_id)

    out_dir = f"{game_id}_outputs"
    os.makedirs(out_dir, exist_ok=True)

    print(f"  [probs] computing raw + conditional probs for {game_id} ...")
    probs_for_game(game_id)

    seq_pattern = os.path.join(out_dir, f"{game_id}_{team_abbr}_sequence.txt")
    matches = glob.glob(seq_pattern)
    if not matches:
        raise FileNotFoundError(
            f"No S/0/T sequence file for team {team_abbr} in game {game_id} "
            f"(expected pattern {seq_pattern})"
        )
    seq_path = matches[0]
    print(f"  [HMM] using sequence file: {seq_path}")

    with open(seq_path, "r") as f:
        text = f.read()
    events = [ch for ch in text if ch in ("S", "0", "T")]
    if len(events) < 2:
        print("  [WARN] fewer than 2 possessions for this team in this game; skipping.")
        return []

    obs = encode_events(events)

    K = 3
    M = 3  # S, 0, T

    #assumption: team that are hot stay hot and teams that are cold stay cold
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
        seed=seed,
        smoothing=1e-6,
    )

    hist = hmm.fit(obs, n_iter=300, tol=1e-6, verbose=False)
    print(f"  [HMM] log-likelihood for {game_id}, {team_abbr}: {hist['log_likelihood']:.3f}")

    # 6) Relabel states by descending P(S | state):
    #    state 0 = HOT, 1 = NEUTRAL, 2 = COLD.
    order = np.argsort(-hmm.B[:, 0])  # column 0 is 'S'
    remap = {old: new for new, old in enumerate(order)}

    v_path_raw = hmm.viterbi(obs)
    v_path = [remap[s] for s in v_path_raw]

    return v_path


def accumulate_hmm_transitions(v_paths: list,
                               n_states: int = 3):
    counts = np.zeros((n_states, n_states), dtype=int)
    total_possessions = 0

    for path in v_paths:
        if len(path) < 2:
            continue
        total_possessions += len(path)
        for t in range(len(path) - 1):
            i, j = path[t], path[t + 1]
            counts[i, j] += 1

    probs = counts.astype(float)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid divide-by-zero
    probs /= row_sums

    return counts, probs, total_possessions


def main():
    ap = argparse.ArgumentParser(description="Run dataPull, probabilities, and HMM across all games for a team.")
    ap.add_argument("team", help="Team abbreviation (e.g., SAC, LAL, BOS).")
    ap.add_argument("season", nargs="?", default="2024-25",
                    help="Season string (default: 2024-25). Format like '2024-25'.")
    args = ap.parse_args()

    team_abbr = args.team.upper()
    season = args.season

    if team_abbr not in TEAM_ABBR_TO_ID:
        print(f"Unknown team abbreviation: {team_abbr}")
        sys.exit(1)

    print(f"=== TeamPipeline for {team_abbr}, season {season} ===")

    try:
        game_ids = get_team_game_ids(team_abbr, season=season)
    except Exception as e:
        print(f"Error fetching game IDs via nba_api: {e}")
        sys.exit(1)

    if not game_ids:
        print(f"No games found for {team_abbr} in {season}")
        sys.exit(0)

    print(f"Found {len(game_ids)} games for {team_abbr} in {season}")

    all_paths = []

    for gid in game_ids:
        print(f"\n=== Processing game {gid} ===")
        try:
            v_path = train_hmm_for_game_team(gid, team_abbr)
        except Exception as e:
            print(f"  [ERROR] skipping game {gid} due to error: {e}")
            continue

        if len(v_path) >= 2:
            all_paths.append(v_path)
        else:
            print(f"  [WARN] skipping game {gid}: not enough possessions for HMM transitions.")

    if not all_paths:
        print("No valid HMM paths collected across games. Nothing to aggregate.")
        sys.exit(0)

    counts, probs, total_possessions = accumulate_hmm_transitions(all_paths, n_states=3)

    labels = ["HOT", "NEUTRAL", "COLD"]

    print("\n=== HMM transition counts (from -> to) ===")
    header = "from\\to   " + "  ".join(f"{lab:>8}" for lab in labels)
    print(header)
    for i, lab_from in enumerate(labels):
        row = "  ".join(f"{counts[i, j]:8d}" for j in range(3))
        print(f"{lab_from:>8}  {row}")

    print("\n=== HMM transition probabilities (from -> to) ===")
    print(header)
    for i, lab_from in enumerate(labels):
        row = "  ".join(f"{probs[i, j]:8.3f}" for j in range(3))
        print(f"{lab_from:>8}  {row}")

    print(f"\nTotal possessions used for {team_abbr} across all games: {total_possessions}")

    out_name = f"{team_abbr}_{season.replace('-', '')}_HMM_transitions.csv"
    import csv
    with open(out_name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from_state", "to_state", "from_label", "to_label", "count", "P_to_given_from"])
        for i in range(3):
            for j in range(3):
                w.writerow([i, j, labels[i], labels[j], int(counts[i, j]), float(probs[i, j])])

    print(f"Saved HMM transition matrix to: {out_name}")


if __name__ == "__main__":
    main()

