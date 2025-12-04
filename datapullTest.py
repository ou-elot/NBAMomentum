#!/usr/bin/env python3
"""
datapull_pbpstats.py
Pull possessions for a single NBA game using pbpstats and save a clean CSV.

Outputs columns:
- game_id, period, index, team_id, team_abbrev
- start_eventnum, end_eventnum, start_time, end_time
- points, ended_by, num_events
- had_turnover, had_missed_shot

Usage:
  pip install pbpstats pandas
  python datapull_pbpstats.py --game-id 0022400785 --csv-out 0022400785_possessions.csv
"""

import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import pandas as pd
from pbpstats.client import Client


# ---------- robust helpers ----------

def _safe_get(obj: Any, *names, default=None):
    """Return the first present attribute/key from names; else default."""
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        else:
            if hasattr(obj, n):
                return getattr(obj, n)
    return default

def _to_lower(s: Optional[str]) -> str:
    return (s or "").lower()

def _event_description(evt: Any) -> str:
    return _safe_get(evt, "description", "desc", "text", default="") or ""

def _event_msg_type(evt: Any) -> Optional[int]:
    emt = _safe_get(evt, "event_msg_type", "eventmsgtype", default=None)
    if emt is None:
        raw = _safe_get(evt, "data", "raw", default=None)
        if isinstance(raw, dict):
            emt = raw.get("EVENTMSGTYPE")
    try:
        return int(emt) if emt is not None else None
    except Exception:
        return None

def _event_team_id(evt: Any) -> Optional[int]:
    tid = _safe_get(evt, "team_id", "teamId", "team", default=None)
    if isinstance(tid, dict):
        tid = tid.get("id") or tid.get("team_id")
    try:
        return int(tid) if tid is not None else None
    except Exception:
        return None

def _is_free_throw(evt: Any) -> bool:
    d = _to_lower(_event_description(evt))
    return "free throw" in d or bool(_safe_get(evt, "is_free_throw", default=False))

def _is_missed_shot(evt: Any) -> bool:
    d = _to_lower(_event_description(evt))
    if "misses" in d and "free throw" not in d:
        return True
    return _event_msg_type(evt) == 2  # 2 = missed shot in many schemas

def _is_turnover(evt: Any) -> bool:
    d = _to_lower(_event_description(evt))
    if "turnover" in d:
        return True
    return _event_msg_type(evt) == 5  # 5 = turnover in many schemas

def _is_made_shot(evt: Any) -> bool:
    made_flag = _safe_get(evt, "is_made_shot", "made", default=None)
    if isinstance(made_flag, bool):
        return made_flag
    d = _to_lower(_event_description(evt))
    return ("makes" in d or "made" in d) and ("3pt" in d or "2pt" in d or "layup" in d or "dunk" in d or "jump" in d)

def _shot_points_from_desc(evt: Any) -> int:
    d = _to_lower(_event_description(evt))
    if "3pt" in d or "3-pt" in d or "3 pt" in d or "three point" in d:
        return 3
    return 2

def _free_throw_points(evt: Any) -> int:
    d = _to_lower(_event_description(evt))
    return 0 if "miss" in d else 1

def _event_points(evt: Any) -> int:
    # Prefer direct fields; fall back to description parsing.
    for name in ("points", "score_value", "shot_value", "points_scored"):
        val = _safe_get(evt, name, default=None)
        if isinstance(val, (int, float)):
            return int(val)
    if _is_made_shot(evt):
        return _shot_points_from_desc(evt)
    if _is_free_throw(evt):
        return _free_throw_points(evt)
    return 0


# ---------- data classes ----------

@dataclass
class PossessionRow:
    game_id: str
    period: int
    index: int
    team_id: Optional[int]
    team_abbrev: Optional[str]
    start_eventnum: Optional[int]
    end_eventnum: Optional[int]
    start_time: Optional[str]
    end_time: Optional[str]
    points: int
    ended_by: Optional[str]
    num_events: int
    had_turnover: bool
    had_missed_shot: bool


# ---------- pbpstats wrappers ----------

def build_settings(provider: str, source: str, data_dir: str) -> dict:
    """
    provider: 'stats_nba' (default) | 'data_nba' | 'live'
    source:   'web' (default) | 'file'
    """
    return {
        "dir": data_dir,
        "Boxscore":    {"source": source, "data_provider": provider},
        "Possessions": {"source": source, "data_provider": provider},
        # If you want raw pbp attached too, uncomment next line and later access game.pbp
        # "Pbp":         {"source": source, "data_provider": provider},
    }

def load_game(game_id: str, provider: str = "stats_nba", source: str = "web", data_dir: str = "./data"):
    client = Client(build_settings(provider, source, data_dir))
    game = client.Game(game_id)
    # Trigger loads for resources we use
    _ = game.possessions
    _ = game.boxscore
    return game

def get_game_metadata(game) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int], Optional[int]]:
    """
    Returns (away_abbr, home_abbr, game_date, away_team_id, home_team_id)
    """
    bs = _safe_get(game, "boxscore", default=None)
    away_abbr = home_abbr = game_date = None
    away_id = home_id = None

    if bs is not None:
        away = _safe_get(bs, "away_team", "awayTeam", default=None)
        home = _safe_get(bs, "home_team", "homeTeam", default=None)
        if away:
            away_abbr = _safe_get(away, "abbreviation", "abbr", "tri_code", "triCode", default=None)
            away_id = _safe_get(away, "id", "team_id", "teamId", default=None)
        if home:
            home_abbr = _safe_get(home, "abbreviation", "abbr", "tri_code", "triCode", default=None)
            home_id = _safe_get(home, "id", "team_id", "teamId", default=None)
        game_date = _safe_get(bs, "game_date_est", "game_date", "date", default=None)

    if away_id is None:
        away_id = _safe_get(game, "away_team_id", default=None)
    if home_id is None:
        home_id = _safe_get(game, "home_team_id", default=None)

    try:
        away_id = int(away_id) if away_id is not None else None
    except Exception:
        pass
    try:
        home_id = int(home_id) if home_id is not None else None
    except Exception:
        pass

    return away_abbr, home_abbr, game_date, away_id, home_id

def team_abbrev_map(game) -> Dict[int, str]:
    away_abbr, home_abbr, _, away_id, home_id = get_game_metadata(game)
    mapping = {}
    if away_id is not None and away_abbr:
        mapping[int(away_id)] = str(away_abbr)
    if home_id is not None and home_abbr:
        mapping[int(home_id)] = str(home_abbr)
    return mapping


# ---------- possession extraction ----------

def possession_to_row(game_id: str, poss: Any, idx_in_period: int, id_to_abbrev: Dict[int, str]) -> PossessionRow:
    period = int(_safe_get(poss, "period", "Period", default=0) or 0)
    team_id = _safe_get(poss, "offense_team_id", "team_id", "teamId", default=None)
    try:
        team_id = int(team_id) if team_id is not None else None
    except Exception:
        team_id = None
    team_abbrev = id_to_abbrev.get(team_id)

    start_eventnum = _safe_get(poss, "start_event_num", "start_eventnum", "start_event_id", default=None)
    end_eventnum = _safe_get(poss, "end_event_num", "end_eventnum", "end_event_id", default=None)
    start_time = _safe_get(poss, "start_time", "start_time_str", "start_time_string", default=None)
    end_time = _safe_get(poss, "end_time", "end_time_str", "end_time_string", default=None)

    events = _safe_get(poss, "events", "possession_events", "pbp_events", default=[]) or []

    points = 0
    had_to = False
    had_miss = False
    for e in events:
        points += _event_points(e)
        if _is_turnover(e):
            had_to = True
        if _is_missed_shot(e):
            had_miss = True

    ended_by = _safe_get(poss, "end_reason", "ended_by", default=None)

    return PossessionRow(
        game_id=game_id,
        period=period,
        index=idx_in_period,
        team_id=team_id,
        team_abbrev=team_abbrev,
        start_eventnum=start_eventnum,
        end_eventnum=end_eventnum,
        start_time=start_time,
        end_time=end_time,
        points=points,
        ended_by=ended_by,
        num_events=len(events),
        had_turnover=had_to,
        had_missed_shot=had_miss,
    )

def extract_possession_table(game_id: str, provider: str, source: str, data_dir: str) -> Tuple[pd.DataFrame, Any]:
    game = load_game(game_id, provider=provider, source=source, data_dir=data_dir)
    id_to_abbrev = team_abbrev_map(game)

    possessions = game.possessions.items
    rows: List[PossessionRow] = []
    cur_period = None
    idx_in_period = 0
    for poss in possessions:
        p = int(_safe_get(poss, "period", "Period", default=0) or 0)
        if p != cur_period:
            cur_period = p
            idx_in_period = 1
        else:
            idx_in_period += 1
        rows.append(possession_to_row(game_id, poss, idx_in_period, id_to_abbrev))

    df = pd.DataFrame([asdict(r) for r in rows])
    return df, game


# ---------- printing helpers ----------

def sum_pts_from_distribution(points_series: pd.Series) -> int:
    dist = Counter(points_series.astype(int).tolist())
    total = 0
    for pts, cnt in dist.items():
        total += int(pts) * int(cnt)
    return total

def print_game_header(game, df: pd.DataFrame):
    away_abbr, home_abbr, game_date, away_id, home_id = get_game_metadata(game)
    hdr = f"{(away_abbr or 'AWAY')} @ {(home_abbr or 'HOME')}"
    line = "=" * max(40, len(hdr))
    print(line)
    print(hdr)
    if game_date:
        print(f"Date: {game_date}")
    print(line)

    # compute scores from possessions
    away_pts = None
    home_pts = None
    if away_id is not None:
        away_pts = sum_pts_from_distribution(df.loc[df["team_id"] == away_id, "points"])
    if home_id is not None:
        home_pts = sum_pts_from_distribution(df.loc[df["team_id"] == home_id, "points"])
    if away_pts is not None and home_pts is not None:
        print(f"Score (from possessions): {(away_abbr or 'AWAY')} {away_pts} — {(home_abbr or 'HOME')} {home_pts}")
    print()


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True, help="NBA GAME_ID (e.g., 0022400785)")
    ap.add_argument("--csv-out", required=True, help="Path to save possessions CSV")
    ap.add_argument("--provider", default="stats_nba", choices=["stats_nba", "data_nba", "live"], help="pbpstats data provider")
    ap.add_argument("--source", default="web", choices=["web", "file"], help="pbpstats source")
    ap.add_argument("--data-dir", default="./data", help="Cache directory used by pbpstats")
    args = ap.parse_args()

    df, game = extract_possession_table(
        game_id=args.game_id,
        provider=args.provider,
        source=args.source,
        data_dir=args.data_dir,
    )

    print_game_header(game, df)

    # quick per-team summary
    for tid, g in df.groupby("team_id", dropna=True):
        team = g["team_abbrev"].dropna().iloc[0] if g["team_abbrev"].notna().any() else str(int(tid))
        poss = len(g)
        scored = int((g["points"] > 0).sum())
        p_score = scored / poss if poss else 0.0
        misses = int(((g["points"] == 0) & g["had_missed_shot"] & (~g["had_turnover"])).sum())
        turnovers = int(((g["points"] == 0) & g["had_turnover"]).sum())
        print(f"[{team}] possessions={poss}, scored={scored} (P(score)={p_score:.3f}), misses={misses}, turnovers={turnovers}")

    # write CSV
    df.to_csv(args.csv_out, index=False)
    print(f"\nSaved possessions to {args.csv_out}")


if __name__ == "__main__":
    main()
