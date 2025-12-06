#!/usr/bin/env python3

import os
import math
import argparse
import pandas as pd

STATES = ["score", "no_score", "turnover"]


def phi_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_prop_z_test(x1: int, n1: int, x2: int, n2: int):
    """
    Pooled two-proportion z-test.

    Group 1: x1 successes out of n1
    Group 2: x2 successes out of n2

    Returns: (p1, p2, z, p_two_sided)
    """
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    p1 = x1 / n1
    p2 = x2 / n2

    total_n = n1 + n2
    if total_n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    p_pool = (x1 + x2) / total_n
    if p_pool == 0.0 or p_pool == 1.0:
        # degenerate case
        if p1 == p2:
            return (p1, p2, 0.0, 1.0)
        return (p1, p2, float("inf"), 0.0)

    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / denom
    p_two = 2 * (1 - phi_cdf(abs(z)))
    return (p1, p2, z, p_two)


def load_cond_for_game(game_id: str) -> pd.DataFrame:
    """
    Load <game_id>_cond_probs.csv from <game_id>_outputs/.
    These come from probabilities.probs_for_game(game_id).
    """
    path = os.path.join(f"{game_id}_outputs", f"{game_id}_cond_probs.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing cond probs: {path} (run probabilities.probs_for_game for this game)"
        )
    return pd.read_csv(path)


def auto_discover_games() -> list:
    """
    Find all '*_outputs' dirs containing '<gid>_cond_probs.csv'
    and return the game_ids.
    """
    games = []
    for d in os.listdir("."):
        if d.endswith("_outputs") and os.path.isdir(d):
            gid = d[:-8]  # strip '_outputs'
            cond_path = os.path.join(d, f"{gid}_cond_probs.csv")
            if os.path.isfile(cond_path):
                games.append(gid)
    return sorted(games)


def aggregate_counts(game_ids, team_filter=None) -> pd.DataFrame:
    """
    Aggregate transition counts across all games in game_ids.

    Returns DataFrame: ['team','from','to','count'].
    """
    frames = []
    for gid in game_ids:
        df = load_cond_for_game(gid)
        if team_filter is not None:
            df = df[df["team"] == team_filter]
        frames.append(df[["team", "from", "to", "count"]])

    if not frames:
        return pd.DataFrame(columns=["team", "from", "to", "count"])

    agg = pd.concat(frames, ignore_index=True)
    agg = agg.groupby(["team", "from", "to"], as_index=False)["count"].sum()
    return agg

def ztests_vs_raw_for_team(agg_counts: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    For a single team, compute for all from,to:

        P(to | from)   vs   raw/unconditional P(to)

    using aggregated transition counts across all games.
    """

    team_counts = agg_counts[agg_counts["team"] == team].copy()
    if team_counts.empty:
        return pd.DataFrame()


    idx = pd.MultiIndex.from_product([[team], STATES, STATES],
                                     names=["team", "from", "to"])
    team_counts = (
        team_counts
        .set_index(["team", "from", "to"])
        .reindex(idx, fill_value=0)
        .reset_index()
    )


    n_from = team_counts.groupby("from")["count"].sum().to_dict()

    total_transitions = int(team_counts["count"].sum())
    to_totals = team_counts.groupby("to")["count"].sum().to_dict()

    rows = []
    for to_state in STATES:
        x2 = to_totals.get(to_state, 0)
        n2 = total_transitions

        for from_state in STATES:
            mask = (team_counts["from"] == from_state) & (team_counts["to"] == to_state)
            x1 = int(team_counts.loc[mask, "count"].sum())
            n1 = int(n_from.get(from_state, 0))

            p1, p2, z, pval = two_prop_z_test(x1, n1, x2, n2)
            rows.append({
                "team": team,
                "from": from_state,
                "to": to_state,
                "x1": x1, "n1": n1, "p1_cond": p1,
                "x2": x2, "n2": n2, "p2_raw": p2,
                "z": z,
                "p_two_sided": pval,
            })

    out = pd.DataFrame(rows)
    out = out[[
        "team", "from", "to",
        "x1", "n1", "p1_cond",
        "x2", "n2", "p2_raw",
        "z", "p_two_sided",
    ]]
    return out


def run(game_ids, team=None, save_path=None) -> pd.DataFrame:
    agg = aggregate_counts(game_ids, team_filter=team)
    if agg.empty:
        print("No data aggregated. Did you run probabilities.probs_for_game first?")
        return pd.DataFrame()

    if team is not None:
        teams = [team]
    else:
        teams = sorted(agg["team"].unique())

    frames = []
    for t in teams:
        df_t = ztests_vs_raw_for_team(agg, t)
        if not df_t.empty:
            frames.append(df_t)

    if not frames:
        print("No z-tests computed (empty data after filtering).")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    print(out.to_string(index=False))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        out.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Team-level z-tests: P(to | from) vs raw P(to) across games."
    )
    ap.add_argument(
        "game_ids",
        nargs="*",
        help="Game IDs (e.g., 0022400785). If omitted, use --auto."
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="Auto-discover *_outputs folders with <gid>_cond_probs.csv"
    )
    ap.add_argument(
        "--team",
        type=str,
        help="Limit to a single team abbreviation (e.g., SAC)"
    )
    ap.add_argument(
        "--save",
        type=str,
        help="Optional CSV path to save results."
    )

    args = ap.parse_args()

    if args.auto:
        gids = auto_discover_games()
    else:
        gids = args.game_ids

    if not gids:
        print("Provide at least one game_id or use --auto.")
        return

    run(gids, team=args.team, save_path=args.save)


if __name__ == "__main__":
    main()

