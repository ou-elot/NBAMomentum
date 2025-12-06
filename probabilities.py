import os
import sys
import glob
import pandas as pd

STATES = ["score", "no_score", "turnover"]

def raw_possession_probs(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    vc = df["result"].value_counts(normalize=True)
    counts = df["result"].value_counts()
    return {
        "P_score":     float(vc.get("score", 0.0)),
        "P_no_score":  float(vc.get("no_score", 0.0)),
        "P_turnover":  float(vc.get("turnover", 0.0)),
        "n_score":     int(counts.get("score", 0)),
        "n_no_score":  int(counts.get("no_score", 0)),
        "n_turnover":  int(counts.get("turnover", 0)),
        "n_possessions": int(len(df)),
    }

def conditional_probs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["result"].isin(STATES)].reset_index(drop=True)

    df["next_result"] = df["result"].shift(-1)
    trans = df.dropna(subset=["next_result"]).copy()

    counts = (
        trans.groupby(["result", "next_result"])
             .size()
             .rename("count")
             .reset_index()
             .rename(columns={"result": "from", "next_result": "to"})
    )

    idx = pd.MultiIndex.from_product([STATES, STATES], names=["from","to"])
    counts = counts.set_index(["from","to"]).reindex(idx, fill_value=0).reset_index()

    row_totals = counts.groupby("from")["count"].transform("sum").replace(0, 1)
    counts["P_to_given_from"] = counts["count"] / row_totals

    return counts

def probs_for_game(game_id: str) -> pd.DataFrame:
    folder = f"{game_id}_outputs"
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"{folder} not found. Run dataPull.py first.")

    team_files = [
        f for f in glob.glob(os.path.join(folder, "*_possessions.csv"))
        if "_ALL_" not in f  # skip combined file
    ]
    if not team_files:
        raise FileNotFoundError(f"No team possession files in {folder}")

    raw_rows = []
    for f in sorted(team_files):
        team = os.path.basename(f).split("_")[-2]
        stats = raw_possession_probs(f)
        stats.update({"game_id": game_id, "team": team})
        raw_rows.append(stats)

    raw_out = pd.DataFrame(raw_rows, columns=[
        "game_id","team","P_score","P_no_score","P_turnover",
        "n_score","n_no_score","n_turnover","n_possessions"
    ])

    probs_path = os.path.join(folder, f"{game_id}_probs.csv")
    raw_out.to_csv(probs_path, index=False)
    print(f"Saved: {probs_path}")

    cond_frames = []
    for f in sorted(team_files):
        team = os.path.basename(f).split("_")[-2]
        cond = conditional_probs(f)
        cond["team"] = team
        cond["game_id"] = game_id
        cond_frames.append(cond)

    cond_out = pd.concat(cond_frames, ignore_index=True)
    cond_out = cond_out[["game_id","team","from","to","count","P_to_given_from"]]

    cond_path = os.path.join(folder, f"{game_id}_cond_probs.csv")
    cond_out.to_csv(cond_path, index=False)
    print(f"Saved: {cond_path}")

    return raw_out, cond_out

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_probs.py <GAME_ID> [<GAME_ID> ...]")
        sys.exit(1)
    for gid in sys.argv[1:]:
        probs_for_game(gid)

if __name__ == "__main__":
    main()

