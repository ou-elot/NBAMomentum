# pip install pbpstats pandas
from pbpstats.client import Client
import pandas as pd
import os

# --- Static map: team_id -> abbreviation (30 teams) ---
TEAM_ID_TO_ABBR = {
    1610612737: "ATL", 1610612738: "BOS", 1610612739: "CLE", 1610612740: "NOP",
    1610612741: "CHI", 1610612742: "DAL", 1610612743: "DEN", 1610612744: "GSW",
    1610612745: "HOU", 1610612746: "LAC", 1610612747: "LAL", 1610612748: "MIA",
    1610612749: "MIL", 1610612750: "MIN", 1610612751: "BKN", 1610612752: "NYK",
    1610612753: "ORL", 1610612754: "IND", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612760: "OKC",
    1610612761: "TOR", 1610612762: "UTA", 1610612763: "MEM", 1610612764: "WAS",
    1610612765: "DET", 1610612766: "CHA"
}

def save_possessions_csv(game_id: str):
    settings = {
        "Possessions": {"source": "web", "data_provider": "data_nba"},  # <--- you changed this
        "Pbp": {"source": "web", "data_provider": "data_nba"},
    }
    client = Client(settings)
    game = client.Game(game_id)

    # --- Build base table from pbpstats possessions (your original approach) ---
    poss_dicts = [p.data for p in game.possessions.items]
    df = pd.DataFrame(poss_dicts)

    # Event sequence column (list of event class names)
    df["sequence"] = [[type(e).__name__ for e in p.events] for p in game.possessions.items]

    # --- Offense team columns (from possession, then map to abbrev) ---
    offense_ids, offense_abbrevs = [], []
    for p in game.possessions.items:
        oid = getattr(p, "offense_team_id", None) or p.data.get("offense_team_id")
        offense_ids.append(oid)
        offense_abbrevs.append(TEAM_ID_TO_ABBR.get(oid))
    df["offense_team_id"] = offense_ids
    df["offense_team_abbreviation"] = offense_abbrevs

    # --- 3-way possession result using LAST EVENT ONLY ---
    # FieldGoal/FreeThrow -> "score"; Turnover -> "turnover"; Rebound/other -> "no_score"
    def classify_result(seq):
        if not seq:
            return "no_score"
        last = seq[-1]
        if "Turnover" in last:
            return "turnover"
        if "FieldGoal" in last or "FreeThrow" in last:
            return "score"
        if "Rebound" in last:
            return "no_score"
        return "no_score"

    df["result"] = df["sequence"].apply(classify_result)

    # --- Rename / reorder exactly like your original flow ---
    rename_map = {
        "offense_team_id": "team_id",
        "offense_team_abbreviation": "team_abbrev",
        "possession_number": "index",
        "start_event_number": "start_eventnum",
        "end_event_number": "end_eventnum",
        "possession_end_type": "ended_by",
        "num_events": "num_events",
        "possession_has_turnover": "had_turnover",
        "possession_has_missed_fg": "had_missed_shot",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    preferred_cols = [
        "game_id", "period", "index",
        "team_id", "team_abbrev",
        "start_eventnum", "end_eventnum", "start_time", "end_time",
        "points", "ended_by", "num_events", "had_turnover",
        "had_missed_shot", "result", "sequence"
    ]
    df = df[[c for c in preferred_cols if c in df.columns] +
            [c for c in df.columns if c not in preferred_cols]]

    # --- Output directory and files ---
    out_dir = f"{game_id}_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Combined (all possessions)
    combined_path = os.path.join(out_dir, f"{game_id}_ALL_possessions.csv")
    df.to_csv(combined_path, index=False)

    # 🔹 NEW: combined S/0/T sequence file (all possessions in order)
    combined_outcomes = df["result"].replace({
        "score": "S",
        "no_score": "0",
        "turnover": "T"
    }).tolist()
    combined_txt = os.path.join(out_dir, f"{game_id}_ALL_sequence.txt")
    with open(combined_txt, "w") as f:
        f.write("".join(combined_outcomes))
    print(f"Saved ALL sequence file: {combined_txt}")

    # Split by offensive team (one file per team in this game)
    team_ids = [tid for tid in df["team_id"].dropna().unique().tolist() if pd.notna(tid)]
    for tid in team_ids:
        team_df = df[df["team_id"] == tid].reset_index(drop=True)
        # choose label: abbrev if present, otherwise the numeric id
        team_label = team_df["team_abbrev"].iloc[0] if ("team_abbrev" in team_df.columns and pd.notna(team_df["team_abbrev"].iloc[0])) else str(int(tid))
        path = os.path.join(out_dir, f"{game_id}_{team_label}_possessions.csv")
        team_df.to_csv(path, index=False)
        print(f"Saved team file: {path}")

        # 🔹 NEW: per-team S/0/T sequence
        team_outcomes = team_df["result"].replace({
            "score": "S",
            "no_score": "0",
            "turnover": "T"
        }).tolist()
        txt_path = os.path.join(out_dir, f"{game_id}_{team_label}_sequence.txt")
        with open(txt_path, "w") as f:
            f.write("".join(team_outcomes))
        print(f"Saved HMM sequence file: {txt_path}")

    print(f"Saved combined file: {combined_path}")

# --- example usage ---
# --- command-line usage ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataPull.py <GAME_ID> [GAME_ID2 GAME_ID3 ...]")
        sys.exit(1)

    game_ids = sys.argv[1:]

    for gid in game_ids:
        print(f"\n=== Pulling game {gid} ===")
        save_possessions_csv(gid)
