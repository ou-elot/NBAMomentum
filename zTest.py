# z_tests_all.py

import os
import glob
import math
import argparse
import pandas as pd

STATES = ["score", "no_score", "turnover"]
PAIRS  = [("score","no_score"), ("score","turnover"), ("no_score","turnover")]

def phi_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def two_prop_z_test(x1: int, n1: int, x2: int, n2: int):
    """Pooled two-proportion z-test."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2) if (n1 + n2) else float("nan")
    denom = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2)) if p_pool not in (0,1) else 0.0
    if denom == 0:
        if p1 == p2:
            return (p1, p2, 0.0, 1.0)
        return (p1, p2, float("inf"), 0.0)
    z = (p1 - p2) / denom
    p_two = 2 * (1 - phi_cdf(abs(z)))
    return (p1, p2, z, p_two)

def load_cond_for_game(game_id: str) -> pd.DataFrame:
    path = os.path.join(f"{game_id}_outputs", f"{game_id}_cond_probs.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing cond probs: {path} (run analyze_probs.py first)")
    return pd.read_csv(path)

def auto_discover_games() -> list:
    games = []
    for d in os.listdir("."):
        if d.endswith("_outputs") and os.path.isdir(d):
            gid = d[:-8]
            if os.path.isfile(os.path.join(d, f"{gid}_cond_probs.csv")):
                games.append(gid)
    return sorted(games)

def aggregate_counts(game_ids, team_filter=None) -> pd.DataFrame:
    frames = []
    for gid in game_ids:
        df = load_cond_for_game(gid)
        if team_filter:
            df = df[df["team"] == team_filter]
        # keep only what we need
        frames.append(df[["team","from","to","count"]])
    if not frames:
        return pd.DataFrame(columns=["team","from","to","count"])
    agg = pd.concat(frames, ignore_index=True)
    agg = agg.groupby(["team","from","to"], as_index=False)["count"].sum()
    return agg

def ztests_all_to_for_team(agg_counts: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    For one team, compute z-tests for ALL destination states.
    Returns long-form table with one row per (to_state, pair).
    """
    team_counts = agg_counts[agg_counts["team"] == team].copy()
    idx = pd.MultiIndex.from_product([[team], STATES, STATES], names=["team","from","to"])
    team_counts = team_counts.set_index(["team","from","to"]).reindex(idx, fill_value=0).reset_index()

    n_from = team_counts.groupby("from")["count"].sum().to_dict()

    rows = []
    for to_state in STATES:
        succ = team_counts[team_counts["to"] == to_state].set_index("from")["count"].to_dict()
        for a, b in PAIRS:
            x1, n1 = succ.get(a, 0), n_from.get(a, 0)
            x2, n2 = succ.get(b, 0), n_from.get(b, 0)
            p1, p2, z, pval = two_prop_z_test(x1, n1, x2, n2)
            rows.append({
                "team": team,
                "to": to_state,
                "from_a": a, "from_b": b,
                "x1": x1, "n1": n1, "p1": p1,
                "x2": x2, "n2": n2, "p2": p2,
                "z": z, "p_two_sided": pval
            })
    return pd.DataFrame(rows)

def run(game_ids, team=None, save_path=None):
    agg = aggregate_counts(game_ids, team_filter=team)
    if agg.empty:
        print("No data aggregated. Did you run analyze_probs.py to create *_cond_probs.csv?")
        return pd.DataFrame()

    teams = [team] if team else sorted(agg["team"].unique())
    out_frames = [ztests_all_to_for_team(agg, t) for t in teams]
    out = pd.concat(out_frames, ignore_index=True)
    out = out[["team","to","from_a","from_b","x1","n1","p1","x2","n2","p2","z","p_two_sided"]]

    print(out.to_string(index=False))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        out.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")
    return out

def main():
    ap = argparse.ArgumentParser(description="Two-proportion z-tests for ALL 'to' states")
    ap.add_argument("game_ids", nargs="*", help="Game IDs (e.g., 0022400785). If omitted, use --auto.")
    ap.add_argument("--auto", action="store_true", help="Auto-discover *_outputs folders with cond probs")
    ap.add_argument("--team", type=str, help="Limit to a single team (e.g., SAC)")
    ap.add_argument("--save", type=str, help="Path to save the combined CSV (e.g., outputs/ztests_all.csv)")
    args = ap.parse_args()

    if args.auto:
        gids = auto_discover_games()
    else:
        gids = args.game_ids

    if not gids:
        print("Provide at least one game_id or use --auto")
        return

    run(gids, team=args.team, save_path=args.save)

if __name__ == "__main__":
    main()

