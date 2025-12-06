#!/usr/bin/env python3

import os
import pandas as pd
import teamZ  # this is your team_ztest_vs_raw.py (renamed teamZ.py on your machine)

# Full set of NBA team abbreviations for 2024-25
TEAM_LIST = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET",
    "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
    "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS",
    "TOR", "UTA", "WAS"
]

def main():
    print("=== Discovering valid games ===")
    game_ids = teamZ.auto_discover_games()
    print(f"Found {len(game_ids)} games with cond_probs.")

    if not game_ids:
        print("No game outputs found. Make sure *_outputs folders exist.")
        return

    os.makedirs("outputs", exist_ok=True)

    for team in TEAM_LIST:
        print(f"\n===== Running z-tests for {team} =====")
        out_path = f"outputs/ztests_{team}.csv"

        df_trans = teamZ.run(game_ids, team=team, save_path=None)
        if df_trans is None or df_trans.empty:
            print(f"No data for team {team}, skipping.")
            continue

        emission_rows = []
        for to_state, grp in df_trans.groupby("to"):
            x2 = grp["x2"].iloc[0]
            n2 = grp["n2"].iloc[0]
            p2 = grp["p2_raw"].iloc[0]
            emission_rows.append({
                "team": team,
                "from": "EMIT",          
                "to": to_state,
                "x1": None,
                "n1": None,
                "p1_cond": None,
                "x2": x2,
                "n2": n2,
                "p2_raw": p2,
                "z": None,
                "p_two_sided": None,
            })

        df_emit = pd.DataFrame(emission_rows)

        df_out = pd.concat([df_trans, df_emit], ignore_index=True)

        df_out.to_csv(out_path, index=False)
        print(f"Saved transitions + emissions to: {out_path}")

    print("\n=== DONE ===")
    print("All teams processed. See 'outputs/' folder for individual CSVs.")


if __name__ == "__main__":
    main()

