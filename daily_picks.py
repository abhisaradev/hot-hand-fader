"""
daily_picks.py — Production daily prediction tool for Hot Hand Fader.
Identifies NBA/WNBA hot and cold streak candidates, outputs tiered picks
with actionable bet thresholds, and auto-grades yesterday's predictions.
Run once per day before games start: python daily_picks.py
"""

from nba_api.stats.endpoints import (
    playergamelog, commonteamroster, scoreboardv3
)
from nba_api.stats.static import players
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import re
import json
import difflib
import subprocess

# ==============================================================================
# REGRESSION MODEL (v2.1) — loaded once at startup; soft dependency on joblib
# ==============================================================================

MODEL_PKL        = "model_regression.pkl"
MODEL_META       = "model_features.json"
PLAYER_RATES_PKL = "player_regression_rates.json"

# Ordinal stat encoding — must match train_model.py STAT_ENCODE
STAT_ENCODE = {"FG3M": 0, "PTS": 1, "PR": 2, "PA": 3, "PRA": 4}

_REGRESSION_MODEL  = None   # GradientBoosting / XGBoost / LogReg pipeline
_REGRESSION_META   = None   # {"features": [...], "train_means": {...}}
_PLAYER_RATES      = None   # {"global_mean": float, "players": {name: rate}}


def _load_regression_model():
    """
    Load the trained GradientBoosting model and supporting files from disk into
    global variables so they're available for every pick scored today.

    Loads three files:
      model_regression.pkl           — the sklearn pipeline (scaler + classifier)
      model_features.json            — feature list + training means used for imputation
      player_regression_rates.json   — per-player historical regression rates

    Uses joblib for the pkl file (Anaconda environment only).
    Fails silently if files are missing or joblib isn't installed — the tool
    still works, it just skips v2 probability scores.
    """
    global _REGRESSION_MODEL, _REGRESSION_META, _PLAYER_RATES
    try:
        import joblib  # Anaconda env only; not in system python3.11
        if os.path.exists(MODEL_PKL) and os.path.exists(MODEL_META):
            _REGRESSION_MODEL = joblib.load(MODEL_PKL)
            with open(MODEL_META) as fh:
                _REGRESSION_META = json.load(fh)
            model_type = _REGRESSION_META.get("model_type", "model")
            print(f"  🤖 v2.1 model loaded  ({model_type}, "
                  f"{len(_REGRESSION_META['features'])} features)")
        else:
            print(f"  ℹ️  Regression model files not found — skipping v2 scoring")
            return

        # Load player regression rate lookup (new in v2.1)
        if os.path.exists(PLAYER_RATES_PKL):
            with open(PLAYER_RATES_PKL) as fh:
                _PLAYER_RATES = json.load(fh)
            n = len(_PLAYER_RATES.get("players", {}))
            print(f"  🤖 Player regression rates loaded  ({n} players)")
        else:
            print(f"  ℹ️  player_regression_rates.json not found — using global mean")

    except ImportError:
        print("  ℹ️  joblib not available in this Python env — skipping v2 scoring")
    except Exception as exc:
        print(f"  ⚠️  Regression model load failed: {exc}")


def score_regression_probability(pick: dict) -> float | None:
    """
    Return P(regression towards fair line) from the v2.1 model.

    New in v2.1:
      • player_regression_rate — looked up from player_regression_rates.json
      • season_position — proxy: current_games / 82 (normalised 0→1)

    Contextual features (opp defense, pace, home/away) are imputed with
    training-set means — they contribute zero variance to the prediction
    but keep the feature vector the right shape.

    Returns None if the model is not loaded.
    """
    if _REGRESSION_MODEL is None or _REGRESSION_META is None:
        return None
    try:
        features = _REGRESSION_META["features"]
        means    = _REGRESSION_META["train_means"]

        gap = pick["recent_avg"] - pick["baseline_avg"]

        # Feature 1 — player regression rate (v2.1)
        player_rate = means.get("player_regression_rate",
                                means.get("global_mean", 0.746))
        if _PLAYER_RATES:
            lookup = _PLAYER_RATES.get("players", {})
            global_mean = _PLAYER_RATES.get("global_mean", player_rate)
            player_rate = lookup.get(pick["player"], global_mean)

        # Feature 2 — season position proxy (v2.1)
        # current_games from pick (how many games baseline is built from)
        # normalised to 0–1 scale using 82-game regular season
        current_games = pick.get("current_games", 41)  # default mid-season
        season_position = min(float(current_games) / 82.0, 1.0)

        # Stat identity (v2.2) — use actual stat encoding, not training mean
        stat_enc = float(STAT_ENCODE.get(pick.get("stat", "FG3M"), 0))

        # Build full feature vector
        feat_vals = {
            "gap_from_baseline":        gap,
            "z_score":                  pick["z_score"],
            "abs_z_score":              abs(gap),
            "season_avg":               pick["baseline_avg"],
            "during_hot":               pick["recent_avg"],
            "stat_encoded":             stat_enc,
            "minutes_trend":            pick.get("minutes_trend",
                                                 means.get("minutes_trend", 0.0)),
            "player_regression_rate":   player_rate,
            "season_position":          season_position,
            # contextual — not fetched in daily workflow; use training means
            "opp_3p_pct_allowed":       means.get("opp_3p_pct_allowed", 0.363),
            "opp_pace":                 means.get("opp_pace", 99.4),
            "is_home":                  means.get("is_home", 0.487),
            "blowout_flag":             means.get("blowout_flag", 0.229),
            "def_tier_ord":             means.get("def_tier_ord", 1.042),
        }

        X = np.array([[feat_vals.get(f, means.get(f, 0.0)) for f in features]])
        prob = float(_REGRESSION_MODEL.predict_proba(X)[0, 1])
        return round(prob, 3)
    except Exception:
        return None

# ==============================================================================
# CONSTANTS & CONFIG
# ==============================================================================

NBA_LEAGUE_ID = "00"
WNBA_LEAGUE_ID = "10"

CURRENT_NBA_SEASON = "2025-26"
PREVIOUS_NBA_SEASON = "2024-25"
CURRENT_WNBA_SEASON = "2026"
PREVIOUS_WNBA_SEASON = "2025"

STATS_TO_CHECK = ["FG3M", "PTS", "PRA"]

# Tier thresholds (z-score absolute value)
STRONG_THRESHOLD = 1.2
MODERATE_THRESHOLD = 1.0
WEAK_THRESHOLD = 0.85

# Bet threshold buffer (multiplied by stat std deviation)
BET_BUFFER_MULTIPLIER = 0.5

MIN_RECENT_MPG = 15

GAMES_FOR_PURE_CURRENT = 40
GAMES_FOR_BLENDED = 20

# ==============================================================================
# WNBA-SPECIFIC FILTER CONSTANTS
# WNBA season is 40 games (vs NBA 82); games are 40 min (vs NBA 48).
# Do NOT change the NBA constants above.
# ==============================================================================

WNBA_MIN_RECENT_MPG        = 12   # role players often 10-20 MPG; starters 25-32
WNBA_GAMES_FOR_PURE_CURRENT = 20  # enough for a reliable current-season avg mid-season
WNBA_GAMES_FOR_BLENDED      = 10  # blend with prior season if 10-19 games played

PREDICTIONS_FILE = "daily_predictions.csv"
GRADED_FILE = "graded_predictions.csv"
PERFORMANCE_FILE = "model_performance.csv"
PERFORMANCE_BY_STAT_FILE = "model_performance_by_stat.csv"


# ==============================================================================
# INJURY REPORT
# ==============================================================================

def _norm_player(name):
    """Lowercase + strip punctuation, for injury lookup keys."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def _injury_fuzzy_match(player_name, injury_report, threshold=0.75):
    """
    Fuzzy-match player_name against normalised keys in injury_report.
    Returns (injury_dict, score) or (None, 0.0) if no match found.
    """
    t = _norm_player(player_name)
    best_key, best_score = None, 0.0
    for key in injury_report:
        s = difflib.SequenceMatcher(None, t, key).ratio()
        if s > best_score:
            best_score, best_key = s, key
    if best_key and best_score >= threshold:
        return injury_report[best_key], best_score
    return None, 0.0


def get_injury_report():
    """
    Fetch current NBA injury report from ESPN's public injury API.

    Returns dict:  normalised_player_name → {status, description, team, raw_name}
        status values: 'OUT' | 'DOUBTFUL' | 'GTD' | raw uppercase string

    Always fetched fresh — never cached. Prints a warning and returns {} on error.
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    try:
        result = subprocess.run(
            ["curl", "-sk", "--max-time", "10", url],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ⚠️  Injury API fetch failed (curl returned {result.returncode})")
            return {}
        data = json.loads(result.stdout)
    except Exception as e:
        print(f"  ⚠️  Injury API error: {e}")
        return {}

    STATUS_MAP = {
        "out":          "OUT",
        "doubtful":     "DOUBTFUL",
        "questionable": "GTD",
        "day-to-day":   "GTD",
        "probable":     "GTD",
    }

    report = {}
    for team_entry in data.get("injuries", []):
        team_name = team_entry.get("displayName", "")
        for inj in team_entry.get("injuries", []):
            player = inj.get("athlete", {}).get("displayName", "")
            if not player:
                continue

            raw_status = inj.get("status", "")
            status = STATUS_MAP.get(raw_status.lower(), raw_status.upper() or "UNKNOWN")

            details = inj.get("details", {})
            parts = [
                p for p in [
                    details.get("side", ""),
                    details.get("type", ""),
                    details.get("detail", ""),
                ]
                if p and p.lower() not in ("", "not specified")
            ]
            description = " ".join(parts) if parts else inj.get("shortComment", "")[:80]

            report[_norm_player(player)] = {
                "status":      status,
                "description": description,
                "team":        team_name,
                "raw_name":    player,
            }

    return report


def _check_returning(recent_df):
    """
    Return True if there's a 5+ day gap between any two consecutive games
    in recent_df — signals a player returning from a recent absence.
    """
    if recent_df is None or len(recent_df) < 2:
        return False
    dates = sorted(recent_df["GAME_DATE"].dt.date.tolist())
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days >= 5:
            return True
    return False


# ==============================================================================
# DATA PULLING
# ==============================================================================

def _get_nba_games_espn(date_str):
    """
    Fetch NBA games for date_str using ESPN's public scoreboard API.

    Used instead of stats.nba.com (ScoreboardV3) which blocks Python requests
    without browser-level auth. ESPN's scoreboard is unauthenticated and
    reliable — same source odds_compare.py already uses successfully.

    Maps ESPN team abbreviations back to NBA API team IDs (needed for
    CommonTeamRoster calls downstream) using nba_api's static team list.
    """
    from nba_api.stats.static import teams as nba_teams_static

    all_teams = nba_teams_static.get_teams()
    # Primary lookup: abbreviation (e.g. "SAS"). Fallback: normalised full name.
    # ESPN uses shortened abbreviations like "SA"/"NY" vs NBA API's "SAS"/"NYK",
    # so the name-based fallback is essential.
    abbrev_to_id   = {t["abbreviation"].lower(): t["id"] for t in all_teams}
    fullname_to_id = {t["full_name"].lower(): t["id"] for t in all_teams}
    nickname_to_id = {t["nickname"].lower(): t["id"] for t in all_teams}

    def _resolve_espn_team(team_dict):
        abbrev = team_dict.get("abbreviation", "").lower()
        if abbrev in abbrev_to_id:
            return abbrev_to_id[abbrev]
        # Fall back to display name ("San Antonio Spurs") or short name ("Spurs")
        display = team_dict.get("displayName", "").lower()
        if display in fullname_to_id:
            return fullname_to_id[display]
        short = team_dict.get("shortDisplayName", team_dict.get("name", "")).lower()
        return nickname_to_id.get(short)

    date_compact = date_str.replace("-", "")
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball"
        f"/nba/scoreboard?dates={date_compact}"
    )
    try:
        result = subprocess.run(
            ["curl", "-sk", "--max-time", "10", url],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ESPN scoreboard fetch failed (curl exit {result.returncode})")
            return []
        data = json.loads(result.stdout)
    except Exception as e:
        print(f"  ESPN scoreboard error: {str(e)[:80]}")
        return []

    games = []
    for event in data.get("events", []):
        status = event.get("status", {})
        status_id = str(status.get("type", {}).get("id", "1"))
        status_text = status.get("type", {}).get("detail", "Scheduled")
        already_started = status_id in ("2", "3")

        comps = event.get("competitions", [])
        if not comps:
            continue
        competitors = comps[0].get("competitors", [])
        if len(competitors) < 2:
            continue

        home_team_id = away_team_id = None
        for c in competitors:
            nba_id = _resolve_espn_team(c.get("team", {}))
            if nba_id is None:
                continue
            if c.get("homeAway") == "home":
                home_team_id = nba_id
            else:
                away_team_id = nba_id

        if home_team_id is None or away_team_id is None:
            continue

        games.append({
            "game_id":            event.get("id", ""),
            "home_team_id":       home_team_id,
            "away_team_id":       away_team_id,
            "game_status":        status_text,
            "game_is_predictable": not already_started,
        })
    return games


def _get_wnba_games_espn(date_str):
    """
    Fetch WNBA games for date_str using ESPN's public WNBA scoreboard.

    CommonTeamRoster and ScoreboardV3 are NBA-only endpoints — they return empty
    data for WNBA team IDs. ESPN's WNBA scoreboard is the reliable alternative.
    Returns game dicts with ESPN WNBA team IDs (used by _get_wnba_roster_espn).
    """
    date_compact = date_str.replace("-", "")
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball"
        f"/wnba/scoreboard?dates={date_compact}"
    )
    try:
        result = subprocess.run(
            ["curl", "-sk", "--max-time", "10", url],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
    except Exception as e:
        print(f"  ESPN WNBA scoreboard error: {str(e)[:80]}")
        return []

    games = []
    for event in data.get("events", []):
        status = event.get("status", {})
        status_id = str(status.get("type", {}).get("id", "1"))
        status_text = status.get("type", {}).get("detail", "Scheduled")
        already_started = status_id in ("2", "3")

        comps = event.get("competitions", [])
        if not comps:
            continue
        competitors = comps[0].get("competitors", [])
        if len(competitors) < 2:
            continue

        home_team_id = away_team_id = None
        for c in competitors:
            tid = c.get("team", {}).get("id")
            if tid is None:
                continue
            if c.get("homeAway") == "home":
                home_team_id = int(tid)
            else:
                away_team_id = int(tid)

        if home_team_id is None or away_team_id is None:
            continue

        games.append({
            "game_id":            event.get("id", ""),
            "home_team_id":       home_team_id,
            "away_team_id":       away_team_id,
            "game_status":        status_text,
            "game_is_predictable": not already_started,
        })
    return games


def _get_wnba_roster_espn(espn_team_id):
    """
    Fetch a WNBA team's roster from ESPN.
    Returns a DataFrame with PLAYER_NAME and PLAYER_ID (ESPN athlete IDs).
    """
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball"
        f"/wnba/teams/{espn_team_id}/roster"
    )
    try:
        result = subprocess.run(
            ["curl", "-sk", "--max-time", "10", url],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return pd.DataFrame()
        data = json.loads(result.stdout)
    except Exception:
        return pd.DataFrame()

    rows = []
    for athlete in data.get("athletes", []):
        name = athlete.get("displayName", "")
        pid  = athlete.get("id", "")
        if name and pid:
            rows.append({"PLAYER_NAME": name, "PLAYER_ID": int(pid)})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _get_wnba_player_gamelog_espn(espn_athlete_id, season_year):
    """
    Fetch a WNBA player's game log from ESPN's public athlete gamelog API.

    Returns a DataFrame with columns: GAME_DATE, PTS, REB, AST, FG3M, MIN, PRA
    matching the schema of get_player_gamelog() so existing analysis code reuses it.
    Returns None on any error or if the player has no logged games.

    3PT stat comes as 'made-attempted' string (e.g. '3-7') — parses the made count.
    MIN may be a number or 'MM:SS' string — both handled.
    """
    url = (
        f"https://site.web.api.espn.com/apis/common/v3/sports/basketball"
        f"/wnba/athletes/{espn_athlete_id}/gamelog"
        f"?region=us&lang=en-US&season={season_year}"
    )
    try:
        result = subprocess.run(
            ["curl", "-sk", "--max-time", "10", url],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
    except Exception:
        return None

    labels      = data.get("labels", [])
    events_dict = data.get("events", {})

    rows = []
    for st in data.get("seasonTypes", []):
        for cat in st.get("categories", []):
            for ev in cat.get("events", []):
                eid  = str(ev.get("eventId", ""))
                stats = ev.get("stats", [])
                if len(stats) != len(labels):
                    continue
                ev_info  = events_dict.get(eid, {})
                date_str = ev_info.get("gameDate", "")
                if not date_str:
                    continue

                raw = dict(zip(labels, stats))

                # 3PT is 'made-attempted' string — extract made count
                try:
                    fg3m = float(str(raw.get("3PT", "0-0")).split("-")[0])
                except (ValueError, IndexError):
                    fg3m = 0.0

                # MIN may be a float or 'MM:SS' string
                try:
                    min_raw = str(raw.get("MIN", "0"))
                    if ":" in min_raw:
                        parts = min_raw.split(":")
                        minutes = float(parts[0]) + float(parts[1]) / 60
                    else:
                        minutes = float(min_raw)
                except (ValueError, TypeError):
                    minutes = 0.0

                rows.append({
                    "GAME_DATE": date_str,
                    "PTS":       float(raw.get("PTS", 0) or 0),
                    "REB":       float(raw.get("REB", 0) or 0),
                    "AST":       float(raw.get("AST", 0) or 0),
                    "FG3M":      fg3m,
                    "MIN":       minutes,
                })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], utc=True).dt.tz_convert(None)
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df[["GAME_DATE", "PTS", "REB", "AST", "FG3M", "MIN", "PRA"]]


def get_games_for_date(date_str, league_id):
    """
    Return all games for the given date and league, with a predictability flag.

    date_str  — "YYYY-MM-DD"
    league_id — NBA_LEAGUE_ID ("00") or WNBA_LEAGUE_ID ("10")

    NBA games use ESPN's scoreboard API (stats.nba.com times out without
    browser-level auth headers; ESPN is public and already used by odds_compare.py).
    WNBA games still use nba_api ScoreboardV3.

    Returns ALL games (including in-progress and final) so the script always
    knows tonight's teams even when run mid-game.  Each game dict includes:
      game_id, home_team_id, away_team_id, game_status, game_is_predictable

    game_is_predictable = True  → game hasn't started; generate picks normally
    game_is_predictable = False → in progress or final; show rosters only,
                                  no new picks generated
    Returns an empty list only if no games exist or the API call fails.
    """
    if league_id == NBA_LEAGUE_ID:
        return _get_nba_games_espn(date_str)

    if league_id == WNBA_LEAGUE_ID:
        return _get_wnba_games_espn(date_str)

    # Other leagues: nba_api fallback (not currently used)
    try:
        scoreboard = scoreboardv3.ScoreboardV3(
            game_date=date_str,
            league_id=league_id
        )
        games_data = scoreboard.get_dict()
        games_list = games_data.get("scoreboard", {}).get("games", [])

        games = []
        for game in games_list:
            status_text = game.get("gameStatusText", "Unknown")
            already_started = any(
                s in status_text.lower()
                for s in ["final", "in progress", "qtr", "half"]
            )
            games.append({
                "game_id": game["gameId"],
                "home_team_id": game["homeTeam"]["teamId"],
                "away_team_id": game["awayTeam"]["teamId"],
                "game_status": status_text,
                "game_is_predictable": not already_started,
            })
        return games
    except Exception as e:
        print(f"  Error pulling games for {date_str}: {str(e)[:80]}")
        return []


def find_next_game_date(start_date_str, days_ahead=14):
    """
    Scan forward from start_date_str until a day with NBA or WNBA games is found.

    Checks up to days_ahead days into the future.  Returns a human-readable
    string like "2026-06-11 — NBA (2 games), WNBA (3 games)" so the terminal
    output tells you when to run the tool next.  Returns None if no games are
    found within the window (rare — usually the off-season).
    """
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    for i in range(1, days_ahead + 1):
        check_date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        nba_games = get_games_for_date(check_date, NBA_LEAGUE_ID)
        wnba_games = get_games_for_date(check_date, WNBA_LEAGUE_ID)
        if nba_games or wnba_games:
            messages = []
            if nba_games:
                messages.append(f"NBA ({len(nba_games)} game{'s' if len(nba_games) > 1 else ''})")
            if wnba_games:
                messages.append(f"WNBA ({len(wnba_games)} game{'s' if len(wnba_games) > 1 else ''})")
            return f"{check_date} — {', '.join(messages)}"
        time.sleep(0.3)
    return None


def get_team_roster_safe(team_id, season):
    """
    Fetch the roster for one team and season from the NBA API.

    Returns a DataFrame with PLAYER_NAME and PLAYER_ID columns.
    On any error (network timeout, unknown team, etc.) returns an empty
    DataFrame so the calling loop can skip this team without crashing.
    """
    try:
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        df = roster.get_data_frames()[0]
        return df[["PLAYER", "PLAYER_ID"]].rename(columns={"PLAYER": "PLAYER_NAME"})
    except Exception as e:
        print(f"    Roster fetch failed for team {team_id}: {str(e)[:80]}")
        return pd.DataFrame()


def get_player_gamelog(player_id, season, season_type):
    """
    Pull a player's game-by-game log from the NBA API and return it as a DataFrame.

    player_id   — NBA player ID (integer)
    season      — season string like "2025-26"
    season_type — "Regular Season" or "Playoffs"

    Adds a PRA column (Points + Rebounds + Assists) computed on the fly.
    Sorts games chronologically (oldest first) so rolling averages work correctly.
    Returns None on any error or if the player has no logged games that season.
    """
    try:
        log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type
        )
        df = log.get_data_frames()[0]
        if len(df) == 0:
            return None
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
        return df
    except Exception:
        return None


# ==============================================================================
# MODEL ANALYSIS
# ==============================================================================

def get_player_baseline(player_id, current_season, previous_season):
    """
    Decide which games to use as the 'true skill' baseline for a player.

    The right baseline changes through the year:
      Playoffs (5+ games):  use playoff games only — these are the most
                            relevant to tonight's game context.
      Full season (40+ games current RS): use current season only — enough
                            data to have a reliable average.
      Mid-season (20-39 games current RS): blend current + previous season
                            60/40 (3 parts current, 2 parts previous) to
                            stabilise the average while it's still small.
      Early season (<20 games current RS): fall back to last year's full
                            season if available (20+ games), because this
                            year's sample is too small to be reliable.
      No data: return None to skip this player entirely.

    Returns:
      baseline_df    — DataFrame of games to compute the season average from
      baseline_label — human-readable description of what was used, e.g.
                       "blended (25cur + 72prev)"
      current_games  — integer count of current regular-season games played
    """
    current_rs = get_player_gamelog(player_id, current_season, "Regular Season")
    current_po = get_player_gamelog(player_id, current_season, "Playoffs")
    previous_rs = get_player_gamelog(player_id, previous_season, "Regular Season")
    
    current_games = len(current_rs) if current_rs is not None else 0
    
    if current_po is not None and len(current_po) >= 5:
        return current_po, f"current_playoffs ({len(current_po)}g)", current_games
    
    if current_games >= GAMES_FOR_PURE_CURRENT:
        return current_rs, f"current_only ({current_games}g)", current_games
    
    elif current_games >= GAMES_FOR_BLENDED:
        if previous_rs is not None and len(previous_rs) >= 20:
            current_weight = 3
            previous_weight = 2
            blended_parts = [current_rs] * current_weight + [previous_rs] * previous_weight
            blended_df = pd.concat(blended_parts, ignore_index=True)
            return blended_df, f"blended ({current_games}cur + {len(previous_rs)}prev)", current_games
        else:
            return current_rs, f"current_only ({current_games}g, no prev)", current_games
    
    else:
        if previous_rs is not None and len(previous_rs) >= 20:
            return previous_rs, f"previous ({len(previous_rs)}g)", current_games
        elif current_games >= 5:
            return current_rs, f"current_only ({current_games}g, low conf)", current_games
        else:
            return None, "insufficient_data", current_games


def classify_tier(z_score):
    """Return tier label based on z-score magnitude."""
    abs_z = abs(z_score)
    if abs_z >= STRONG_THRESHOLD:
        return "STRONG"
    elif abs_z >= MODERATE_THRESHOLD:
        return "MODERATE"
    elif abs_z >= WEAK_THRESHOLD:
        return "WEAK"
    else:
        return "NORMAL"


def get_bet_recommendation(status, fair_line, season_std):
    """
    Build the actionable bet threshold.
    For HOT: bet UNDER if sportsbook line > fair_line + buffer
    For COLD: bet OVER if sportsbook line < fair_line - buffer
    """
    buffer = BET_BUFFER_MULTIPLIER * season_std
    
    if status == "HOT":
        threshold = fair_line + buffer
        return f"bet UNDER if line > {threshold:.1f}"
    elif status == "COLD":
        threshold = fair_line - buffer
        return f"bet OVER if line < {threshold:.1f}"
    return None


def analyze_player(player_name, player_id, current_season, previous_season):
    """
    Run the full hot/cold streak analysis for one player across all tracked stats.

    Steps:
      1. Get the player's baseline (true skill level) using get_player_baseline()
      2. Get their 3 most recent games (from playoffs if in playoffs, else RS)
      3. For each stat (FG3M, PTS, PRA): compute season average, std, rolling
         3-game average, and z-score
      4. Classify each stat as HOT (z > 0, above threshold), COLD (z < 0,
         above threshold), or NORMAL
      5. Compute the bet recommendation threshold for HOT/COLD picks

    Returns a flat dict with keys like FG3M_status, FG3M_zscore, FG3M_fair_line,
    plus player-level info (recent_mpg, baseline_used, returning flag).
    Returns None if the player doesn't have enough recent data to analyse.
    """
    baseline_df, baseline_label, current_games = get_player_baseline(
        player_id, current_season, previous_season
    )
    
    if baseline_df is None:
        return None
    
    current_rs = get_player_gamelog(player_id, current_season, "Regular Season")
    current_po = get_player_gamelog(player_id, current_season, "Playoffs")
    
    if current_po is not None and len(current_po) >= 1:
        recent_df = current_po.tail(min(3, len(current_po)))
        recent_minutes = current_po["MIN"].tail(min(3, len(current_po))).mean()
    elif current_rs is not None and len(current_rs) >= 3:
        recent_df = current_rs.tail(3)
        recent_minutes = current_rs["MIN"].tail(3).mean()
    else:
        return None
    
    baseline_minutes = baseline_df["MIN"].mean()
    
    results = {
        "player": player_name,
        "player_id": int(player_id),
        "current_games": current_games,
        "baseline_used": baseline_label,
        "recent_mpg": round(recent_minutes, 1),
        "baseline_mpg": round(baseline_minutes, 1),
        "returning": _check_returning(recent_df),
    }
    
    for stat in STATS_TO_CHECK:
        if stat not in baseline_df.columns:
            continue
        
        season_avg = baseline_df[stat].mean()
        season_std = baseline_df[stat].std()
        recent_avg = recent_df[stat].mean()
        z_score = (recent_avg - season_avg) / season_std if season_std > 0 else 0
        
        # Classify HOT/COLD based on direction + tier
        tier = classify_tier(z_score)
        if z_score > 0 and tier != "NORMAL":
            status = "HOT"
        elif z_score < 0 and tier != "NORMAL":
            status = "COLD"
        else:
            status = "NORMAL"
        
        bet_rec = get_bet_recommendation(status, season_avg, season_std) if status != "NORMAL" else None
        
        results[f"{stat}_baseline"] = round(season_avg, 2)
        results[f"{stat}_std"] = round(season_std, 2)
        results[f"{stat}_recent"] = round(recent_avg, 2)
        results[f"{stat}_zscore"] = round(z_score, 2)
        results[f"{stat}_status"] = status
        results[f"{stat}_tier"] = tier
        results[f"{stat}_fair_line"] = round(season_avg, 2)
        results[f"{stat}_bet_rec"] = bet_rec
    
    return results


# ==============================================================================
# WNBA ANALYSIS — parallel to the NBA functions above but using ESPN data
# CommonTeamRoster / PlayerGameLog are NBA-only; WNBA uses ESPN throughout.
# ==============================================================================

def get_player_baseline_wnba(espn_athlete_id, current_season_year, previous_season_year):
    """
    WNBA equivalent of get_player_baseline(), using ESPN game logs and WNBA thresholds.

    WNBA season is 40 games (not 82), so the tier breakpoints are halved:
      Full season (20+ games): current season only
      Mid-season (10-19 games): blend current + previous 60/40
      Early (<10 games): fall back to previous season
    """
    current_rs  = _get_wnba_player_gamelog_espn(espn_athlete_id, current_season_year)
    previous_rs = _get_wnba_player_gamelog_espn(espn_athlete_id, previous_season_year)

    current_games = len(current_rs) if current_rs is not None else 0

    if current_games >= WNBA_GAMES_FOR_PURE_CURRENT:
        return current_rs, f"current_only ({current_games}g)", current_games

    if current_games >= WNBA_GAMES_FOR_BLENDED:
        if previous_rs is not None and len(previous_rs) >= 10:
            blended_df = pd.concat(
                [current_rs] * 3 + [previous_rs] * 2, ignore_index=True
            )
            return blended_df, f"blended ({current_games}cur + {len(previous_rs)}prev)", current_games
        return current_rs, f"current_only ({current_games}g, no prev)", current_games

    if previous_rs is not None and len(previous_rs) >= 10:
        return previous_rs, f"previous ({len(previous_rs)}g)", current_games
    if current_games >= 5:
        return current_rs, f"current_only ({current_games}g, low conf)", current_games
    return None, "insufficient_data", current_games


def analyze_wnba_player(player_name, espn_athlete_id, current_season_year, previous_season_year):
    """
    WNBA equivalent of analyze_player(), using ESPN game logs.

    Returns the same dict schema as analyze_player() so all downstream pick-building
    and CSV-saving code works unchanged.  Returns None if not enough data.
    """
    baseline_df, baseline_label, current_games = get_player_baseline_wnba(
        espn_athlete_id, current_season_year, previous_season_year
    )
    if baseline_df is None:
        return None

    current_rs = _get_wnba_player_gamelog_espn(espn_athlete_id, current_season_year)

    if current_rs is not None and len(current_rs) >= 3:
        recent_df      = current_rs.tail(3)
        recent_minutes = current_rs["MIN"].tail(3).mean()
    else:
        return None

    baseline_minutes = baseline_df["MIN"].mean()

    results = {
        "player":        player_name,
        "player_id":     int(espn_athlete_id),
        "current_games": current_games,
        "baseline_used": baseline_label,
        "recent_mpg":    round(recent_minutes, 1),
        "baseline_mpg":  round(baseline_minutes, 1),
        "returning":     _check_returning(recent_df),
    }

    for stat in STATS_TO_CHECK:
        if stat not in baseline_df.columns:
            continue

        season_avg = baseline_df[stat].mean()
        season_std = baseline_df[stat].std()
        recent_avg = recent_df[stat].mean()
        z_score    = (recent_avg - season_avg) / season_std if season_std > 0 else 0

        tier = classify_tier(z_score)
        if z_score > 0 and tier != "NORMAL":
            status = "HOT"
        elif z_score < 0 and tier != "NORMAL":
            status = "COLD"
        else:
            status = "NORMAL"

        bet_rec = get_bet_recommendation(status, season_avg, season_std) if status != "NORMAL" else None

        results[f"{stat}_baseline"] = round(season_avg, 2)
        results[f"{stat}_std"]      = round(season_std, 2)
        results[f"{stat}_recent"]   = round(recent_avg, 2)
        results[f"{stat}_zscore"]   = round(z_score, 2)
        results[f"{stat}_status"]   = status
        results[f"{stat}_tier"]     = tier
        results[f"{stat}_fair_line"] = round(season_avg, 2)
        results[f"{stat}_bet_rec"]  = bet_rec

    return results


# ==============================================================================
# GRADING (proxy backtest methodology + per-stat performance log)
# ==============================================================================

def grade_predictions_from_date(date_str):
    """
    Grade yesterday's predictions against actual game results.

    Reads daily_predictions.csv, finds picks for date_str, then fetches the
    actual stat from each player's game log for that date.  Scores each pick
    using the proxy backtest methodology:

      WIN  — actual outcome was closer to our fair_line (season avg) than to
              the recent_avg (the hot-streak line) — our mean-reversion call
              was correct
      LOSS — actual outcome was closer to the recent_avg
      PUSH — equidistant (rare)
      DNP  — player didn't play (game log for that date came back empty)

    Appends results to graded_predictions.csv and updates model_performance.csv
    (overall) and model_performance_by_stat.csv (FG3M / PTS / PRA breakdown).

    Returns nothing — all output is written to CSVs and printed to terminal.
    """
    if not os.path.exists(PREDICTIONS_FILE):
        return
    
    all_preds = pd.read_csv(PREDICTIONS_FILE)
    target_preds = all_preds[all_preds["date"] == date_str]
    
    if len(target_preds) == 0:
        return
    
    print(f"\nGrading predictions from {date_str}...")
    
    graded_rows = []
    
    for _, pred in target_preds.iterrows():
        player_name = pred["player"]
        season = pred["season"]
        
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            continue
        player_id = player_dict[0]["id"]
        
        actual = None
        for season_type in ["Playoffs", "Regular Season"]:
            try:
                df = get_player_gamelog(player_id, season, season_type)
                if df is not None:
                    match = df[df["GAME_DATE"].dt.strftime("%Y-%m-%d") == date_str]
                    if len(match) > 0:
                        actual = match.iloc[0]
                        break
            except Exception:
                pass
            time.sleep(0.3)
        
        if actual is None:
            graded_rows.append({**pred.to_dict(), "actual": None, "result": "DNP"})
            continue
        
        stat = pred["stat"]
        actual_value = float(actual[stat]) if stat in actual else None

        # Proxy backtest methodology: a prediction is a WIN if the actual
        # outcome lands closer to fair_line (mean reversion) than to recent_avg
        # (the hot/cold streak), and a LOSS if it lands closer to recent_avg.
        result = "PUSH"
        if actual_value is not None:
            recent = float(pred["recent_avg"])
            fair = float(pred["fair_line"])
            dist_to_fair = abs(actual_value - fair)
            dist_to_recent = abs(actual_value - recent)

            if dist_to_fair < dist_to_recent:
                result = "WIN"
            elif dist_to_recent < dist_to_fair:
                result = "LOSS"
            # Equidistant stays PUSH

        graded_rows.append({
            **pred.to_dict(),
            "actual": actual_value,
            "result": result
        })
    
    graded_df = pd.DataFrame(graded_rows)
    
    if os.path.exists(GRADED_FILE):
        existing = pd.read_csv(GRADED_FILE)
        graded_df = pd.concat([existing, graded_df], ignore_index=True)
    graded_df.to_csv(GRADED_FILE, index=False)
    
    results = graded_df[graded_df["date"] == date_str]
    if len(results) > 0:
        wins = (results["result"] == "WIN").sum()
        losses = (results["result"] == "LOSS").sum()
        pushes = (results["result"].isin(["PUSH", "DNP"])).sum()
        played = wins + losses
        win_rate = (100 * wins / played) if played > 0 else 0
        
        print(f"  {date_str} results: {wins}W - {losses}L - {pushes} pushes/DNP")
        if played > 0:
            print(f"  Win rate (excluding pushes/DNP): {win_rate:.1f}%")
        
        perf_row = {
            "date": date_str,
            "predictions": len(results),
            "wins": int(wins),
            "losses": int(losses),
            "pushes_dnp": int(pushes),
            "win_rate": round(win_rate, 1)
        }
        perf_df = pd.DataFrame([perf_row])
        if os.path.exists(PERFORMANCE_FILE):
            existing_perf = pd.read_csv(PERFORMANCE_FILE)
            perf_df = pd.concat([existing_perf, perf_df], ignore_index=True)
        perf_df.to_csv(PERFORMANCE_FILE, index=False)

        # Per-stat performance log — track which stat (FG3M, PTS, PRA) is
        # performing best over time.
        by_stat_rows = []
        for stat, stat_results in results.groupby("stat"):
            s_wins = (stat_results["result"] == "WIN").sum()
            s_losses = (stat_results["result"] == "LOSS").sum()
            s_pushes = (stat_results["result"].isin(["PUSH", "DNP"])).sum()
            s_played = s_wins + s_losses
            s_win_rate = (100 * s_wins / s_played) if s_played > 0 else 0
            by_stat_rows.append({
                "date": date_str,
                "stat": stat,
                "wins": int(s_wins),
                "losses": int(s_losses),
                "pushes_dnp": int(s_pushes),
                "win_rate": round(s_win_rate, 1)
            })

        if by_stat_rows:
            by_stat_df = pd.DataFrame(by_stat_rows)
            if os.path.exists(PERFORMANCE_BY_STAT_FILE):
                existing_by_stat = pd.read_csv(PERFORMANCE_BY_STAT_FILE)
                by_stat_df = pd.concat([existing_by_stat, by_stat_df], ignore_index=True)
            by_stat_df.to_csv(PERFORMANCE_BY_STAT_FILE, index=False)


# ==============================================================================
# MAIN
# ==============================================================================

def run_for_league(league_name, league_id, current_season, previous_season, today_str):
    """
    Run the full pick-generation pipeline for one league (NBA or WNBA).

    For each game tonight:
      1. Gets rosters for both teams
      2. Runs analyze_player() on every player in those rosters
      3. Filters out players averaging fewer than MIN_RECENT_MPG minutes (15)
      4. Checks the injury report — OUT players are removed entirely;
         DOUBTFUL/GTD players are flagged; returning players are demoted to WEAK
      5. For each player's HOT/COLD stat, scores a regression probability from
         the v2 GradientBoosting model (FG3M picks only)
      6. Collects all flagged picks into a list

    Returns a list of pick dicts, one per player-stat pair that met the threshold.
    Returns an empty list if no games tonight or no players qualified.
    """
    print(f"\n{'='*70}")
    print(f"  {league_name} — {today_str}")
    print(f"{'='*70}")

    games = get_games_for_date(today_str, league_id)
    if not games:
        print(f"  No upcoming games today.")
        return []

    print(f"  {len(games)} game(s) tonight")

    # Fetch injury report once for the whole league run (always fresh)
    print(f"  Fetching injury report from ESPN...")
    injury_report = get_injury_report()
    if injury_report:
        print(f"  Injury report: {len(injury_report)} player(s) on report")
    else:
        print(f"  Injury report: none available (API may be down)")

    is_wnba     = (league_id == WNBA_LEAGUE_ID)
    mpg_min     = WNBA_MIN_RECENT_MPG if is_wnba else MIN_RECENT_MPG
    # WNBA season identifier is a plain year string ("2026"); NBA is "2025-26"
    cur_season_id  = current_season
    prev_season_id = previous_season

    all_picks = []
    out_skipped = 0

    for game in games:
        if not game.get("game_is_predictable", True):
            # Game is live or already final — show rosters without generating picks
            status = game.get("game_status", "In progress")
            print(f"\n  ⏱️  {status}")
            print(f"     Game already started — showing rosters only, no new picks generated.")
            print(f"     Run tomorrow to grade results.")
            for team_id in [game["home_team_id"], game["away_team_id"]]:
                if is_wnba:
                    roster = _get_wnba_roster_espn(team_id)
                else:
                    roster = get_team_roster_safe(team_id, cur_season_id)
                if not roster.empty:
                    names = ", ".join(roster["PLAYER_NAME"].head(6).tolist())
                    print(f"     Team {team_id}: {names}…")
            continue

        team_ids = [game["home_team_id"], game["away_team_id"]]
        for team_id in team_ids:
            if is_wnba:
                roster = _get_wnba_roster_espn(team_id)
            else:
                roster = get_team_roster_safe(team_id, cur_season_id)

            for _, player_row in roster.iterrows():
                try:
                    if is_wnba:
                        result = analyze_wnba_player(
                            player_row["PLAYER_NAME"],
                            player_row["PLAYER_ID"],
                            cur_season_id,
                            prev_season_id,
                        )
                    else:
                        result = analyze_player(
                            player_row["PLAYER_NAME"],
                            player_row["PLAYER_ID"],
                            cur_season_id,
                            prev_season_id,
                        )
                except Exception:
                    result = None
                time.sleep(0.3)

                if not result:
                    continue
                if result["recent_mpg"] < mpg_min:
                    continue

                player_name = result["player"]

                # ── Injury check ──────────────────────────────────────────
                inj_info, _ = _injury_fuzzy_match(player_name, injury_report)
                inj_status  = inj_info["status"]      if inj_info else None
                inj_desc    = inj_info["description"] if inj_info else ""

                # OUT → skip entirely; don't even add to picks
                if inj_status == "OUT":
                    print(f"    ⛔  {player_name} — OUT ({inj_desc}) — removed from picks")
                    out_skipped += 1
                    continue

                # Returning-from-injury signal (5+ day gap in recent log)
                is_returning = result.get("returning", False)

                # Minutes trend for regression model feature
                minutes_trend = result["recent_mpg"] - result["baseline_mpg"]

                for stat in STATS_TO_CHECK:
                    status = result.get(f"{stat}_status", "NORMAL")
                    if status not in ("HOT", "COLD"):
                        continue

                    tier = result[f"{stat}_tier"]

                    # Determine injury flag for this pick
                    injury_flag = ""
                    injury_desc = ""
                    if inj_status == "DOUBTFUL":
                        injury_flag = "DOUBTFUL"
                        injury_desc = f"⚠️ DOUBTFUL — high DNP risk ({inj_desc})" if inj_desc else "⚠️ DOUBTFUL — high DNP risk"
                    elif inj_status == "GTD":
                        injury_flag = "GTD"
                        injury_desc = f"⚠️ GTD — confirm active before betting ({inj_desc})" if inj_desc else "⚠️ GTD — confirm active before betting"
                    elif is_returning:
                        injury_flag = "RETURNING"
                        injury_desc = "↩️ RETURNING — first game back, demoted to WEAK"
                        tier = "WEAK"   # demote returning players

                    # Build pick; score regression probability (v2 model)
                    pick_partial = {
                        "date":             today_str,
                        "league":           league_name,
                        "season":           current_season,
                        "player":           player_name,
                        "team_id":          int(team_id),
                        "current_games":    result["current_games"],
                        "baseline_used":    result["baseline_used"],
                        "stat":             stat,
                        "status":           status,
                        "tier":             tier,
                        "baseline_avg":     result[f"{stat}_baseline"],
                        "season_std":       result[f"{stat}_std"],
                        "recent_avg":       result[f"{stat}_recent"],
                        "z_score":          result[f"{stat}_zscore"],
                        "fair_line":        result[f"{stat}_fair_line"],
                        "bet_recommendation": result[f"{stat}_bet_rec"],
                        "recent_mpg":       result["recent_mpg"],
                        "minutes_trend":    round(minutes_trend, 2),
                        # injury fields (dropped before CSV save)
                        "_injury_flag":     injury_flag,
                        "_injury_desc":     injury_desc,
                    }
                    pick_partial["regression_probability"] = score_regression_probability(
                        pick_partial
                    )
                    all_picks.append(pick_partial)

    if out_skipped:
        print(f"\n  ⛔  {out_skipped} player(s) removed — OUT on injury report")

    return all_picks


def print_full_pick(pick):
    """Print a single pick in full detail (for STRONG/MODERATE tiers)."""
    icon = "🔥" if pick["status"] == "HOT" else "❄️"
    print(f"  {icon} [{pick['tier']}] {pick['player']} ({pick['league']}) — {pick['stat']}")
    print(f"      Baseline: {pick['baseline_avg']} (std {pick['season_std']}) | Recent: {pick['recent_avg']} | z={pick['z_score']}")
    print(f"      Source: {pick['baseline_used']} | MPG: {pick['recent_mpg']}")
    print(f"      📊 FAIR LINE: {pick['fair_line']}")
    print(f"      💰 ACTION: {pick['bet_recommendation']}")
    reg_prob = pick.get("regression_probability")
    if reg_prob is not None:
        pct = round(reg_prob * 100, 1)
        conf = "HIGH" if pct >= 70 else ("MED" if pct >= 60 else "LOW")
        print(f"      🤖 v2 MODEL: {pct}% regression probability [{conf} confidence]")
    if pick.get("_injury_flag"):
        print(f"      {pick['_injury_desc']}")
    print()


def print_summary_pick(pick):
    """Print a single pick in summary form (one line, for WEAK tier)."""
    icon = "🔥" if pick["status"] == "HOT" else "❄️"
    inj = f"  {pick['_injury_desc']}" if pick.get("_injury_flag") else ""
    print(f"  {icon} {pick['player']} ({pick['stat']}) z={pick['z_score']} | "
          f"fair {pick['fair_line']} | {pick['bet_recommendation']}{inj}")


def main():
    """
    Entry point for the daily prediction workflow.

    Runs in this order:
      1. Load the v2 ML model (soft dependency — skipped if joblib unavailable)
      2. Check whether picks already exist for today and prompt to replace if so
      3. Grade yesterday's predictions against actual results
      4. Run pick generation for NBA and WNBA
      5. Sort picks by tier (STRONG → MODERATE → WEAK), then by z-score
      6. Print the pick report to terminal
      7. Save picks to daily_predictions.csv for odds_compare.py to read

    Intended to be run once per day before games start (e.g. by 5 PM ET).
    """
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"\n{'#'*70}")
    print(f"#  BETTING THE REGRESSION — daily_picks.py v1.4")
    print(f"#  Running on {today}")
    print(f"#  Auto-grading yesterday ({yesterday})")
    print(f"{'#'*70}")

    _load_regression_model()
    
    # Duplicate prevention
    if os.path.exists(PREDICTIONS_FILE):
        existing = pd.read_csv(PREDICTIONS_FILE)
        if today in existing["date"].astype(str).values:
            print(f"\n⚠️  Predictions already exist for {today}.")
            print(f"   Re-running would create duplicates.")
            response = input("   Type 'y' to replace today's picks, or 'n' to keep existing: ").strip().lower()
            if response != "y":
                print("   Skipping today's analysis.")
                grade_predictions_from_date(yesterday)
                return
            existing = existing[existing["date"].astype(str) != today]
            existing.to_csv(PREDICTIONS_FILE, index=False)
    
    grade_predictions_from_date(yesterday)
    
    nba_picks = run_for_league(
        "NBA", NBA_LEAGUE_ID,
        CURRENT_NBA_SEASON, PREVIOUS_NBA_SEASON,
        today
    )
    wnba_picks = run_for_league(
        "WNBA", WNBA_LEAGUE_ID,
        CURRENT_WNBA_SEASON, PREVIOUS_WNBA_SEASON,
        today
    )
    all_picks = nba_picks + wnba_picks
    
    if not all_picks:
        print(f"\n{'='*70}")
        print(f"  NO PICKS FOR TODAY ({today})")
        print(f"{'='*70}")
        print(f"\n  Either no games scheduled, or no players met our criteria.")
        next_game_info = find_next_game_date(today)
        if next_game_info:
            print(f"  Next scheduled games: {next_game_info}")
        else:
            print(f"  No games found in either league for the next 14 days.")
        return
    
    picks_df = pd.DataFrame(all_picks)
    picks_df["abs_z"] = picks_df["z_score"].abs()
    
    # Sort: tier first (STRONG > MODERATE > WEAK), then by abs(z_score) within tier
    tier_order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2}
    picks_df["tier_rank"] = picks_df["tier"].map(tier_order)
    picks_df = picks_df.sort_values(["tier_rank", "abs_z"], ascending=[True, False])
    
    # Save — drop internal sort keys and injury fields (ephemeral, fetched fresh each run)
    # minutes_trend and regression_probability are kept in CSV
    drop_cols = ["abs_z", "tier_rank", "_injury_flag", "_injury_desc"]
    save_cols = [c for c in drop_cols if c in picks_df.columns]
    if os.path.exists(PREDICTIONS_FILE):
        existing = pd.read_csv(PREDICTIONS_FILE)
        save_df = pd.concat([existing, picks_df.drop(columns=save_cols)], ignore_index=True)
    else:
        save_df = picks_df.drop(columns=save_cols)
    save_df.to_csv(PREDICTIONS_FILE, index=False)
    
    print(f"\n{'='*70}")
    print(f"  TODAY'S PICKS — {today}")
    print(f"{'='*70}")
    
    # Split by tier
    strong_picks = picks_df[picks_df["tier"] == "STRONG"]
    moderate_picks = picks_df[picks_df["tier"] == "MODERATE"]
    weak_picks = picks_df[picks_df["tier"] == "WEAK"]
    
    if len(strong_picks) > 0:
        print(f"\n🎯 STRONG PICKS (|z| ≥ {STRONG_THRESHOLD}) — {len(strong_picks)} found\n")
        for _, pick in strong_picks.iterrows():
            print_full_pick(pick)
    
    if len(moderate_picks) > 0:
        print(f"\n✅ MODERATE PICKS ({MODERATE_THRESHOLD} ≤ |z| < {STRONG_THRESHOLD}) — {len(moderate_picks)} found\n")
        for _, pick in moderate_picks.iterrows():
            print_full_pick(pick)
    
    if len(weak_picks) > 0:
        print(f"\n📋 WEAK PICKS — SUMMARY ({WEAK_THRESHOLD} ≤ |z| < {MODERATE_THRESHOLD}) — {len(weak_picks)} found\n")
        for _, pick in weak_picks.iterrows():
            print_summary_pick(pick)
    
    print(f"\nSaved {len(picks_df)} picks to {PREDICTIONS_FILE}")
    print(f"  Tiers: {len(strong_picks)} STRONG | {len(moderate_picks)} MODERATE | {len(weak_picks)} WEAK")
    
    if os.path.exists(PERFORMANCE_FILE):
        perf = pd.read_csv(PERFORMANCE_FILE)
        total_wins = perf["wins"].sum()
        total_losses = perf["losses"].sum()
        if total_wins + total_losses > 0:
            overall = 100 * total_wins / (total_wins + total_losses)
            print(f"\n📊 MODEL TRACK RECORD: {total_wins}W - {total_losses}L ({overall:.1f}%)")


if __name__ == "__main__":
    main()