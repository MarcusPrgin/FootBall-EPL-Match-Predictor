# main file for generating league table and team details from match data
# assumes match data CSV has been created using scraping.py

import argparse
import pandas as pd
from tabulate import tabulate
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

def _available_feature_cols(df):
    # Use whatever exists in your CSV (FBref scraping can vary a bit)
    candidate = ["xg", "xga", "poss", "sh", "sot", "dist", "fk", "pk", "pkatt", "attendance"]
    return [c for c in candidate if c in df.columns]


def train_goal_models(df):
    """
    Train 2 regressors:
      - predicts gf (goals for)
      - predicts ga (goals against)

    Uses numeric match features + categorical venue.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    feature_cols = _available_feature_cols(df)
    if not feature_cols:
        raise RuntimeError(
            "No usable numeric features found (expected columns like xg, xga, sh, sot, poss...)."
        )

    # Keep only rows with targets
    work = df.dropna(subset=["gf", "ga"]).copy()

    X = work[feature_cols + (["venue"] if "venue" in work.columns else [])]
    y_gf = work["gf"].astype(float)
    y_ga = work["ga"].astype(float)

    numeric_features = feature_cols
    categorical_features = ["venue"] if "venue" in X.columns else []

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ],
        remainder="drop",
    )

    # RandomForest works well as a simple baseline (no scaling needed)
    reg = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=3
    )

    model_gf = Pipeline(steps=[("pre", pre), ("reg", reg)])
    model_ga = Pipeline(steps=[("pre", pre), ("reg", reg)])

    model_gf.fit(X, y_gf)
    model_ga.fit(X, y_ga)

    return model_gf, model_ga, feature_cols


def build_team_profiles(df, feature_cols):
    """
    Build per-team average stats split by venue.
    We'll use these profiles to create features for simulated games.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "venue" not in df.columns:
        # If venue missing, just build one profile
        prof = df.groupby("team")[feature_cols].mean(numeric_only=True)
        return {"ANY": prof}

    # Two profiles: home and away
    home_prof = df[df["venue"].astype(str).str.lower().str.startswith("h")].groupby("team")[feature_cols].mean(numeric_only=True)
    away_prof = df[df["venue"].astype(str).str.lower().str.startswith("a")].groupby("team")[feature_cols].mean(numeric_only=True)

    return {"HOME": home_prof, "AWAY": away_prof}


def round_robin_fixtures(teams, seed=42):
    """
    Create a double round-robin schedule (home/away).
    Simple circle method.
    Returns list of (home, away).
    """
    teams = list(sorted(teams))
    rng = np.random.default_rng(seed)
    rng.shuffle(teams)

    if len(teams) % 2 == 1:
        teams.append("BYE")

    n = len(teams)
    half = n // 2
    fixtures = []

    arr = teams[:]
    for _round in range(n - 1):
        left = arr[:half]
        right = arr[half:][::-1]

        for a, b in zip(left, right):
            if a != "BYE" and b != "BYE":
                fixtures.append((a, b))  # a home, b away

        # rotate
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]

    # Double round robin: reverse venues
    fixtures_return = fixtures + [(away, home) for (home, away) in fixtures]
    return fixtures_return


def simulate_season(df, season_year=None, seed=42):
    """
    Train ML on the provided df (season 1),
    then simulate a new season (season 2) schedule.
    Returns a simulated matchlog (team-perspective) and a league table.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    model_gf, model_ga, feature_cols = train_goal_models(df)
    profiles = build_team_profiles(df, feature_cols)

    teams = sorted(df["team"].dropna().unique().tolist())
    fixtures = round_robin_fixtures(teams, seed=seed)

    rng = np.random.default_rng(seed)

    sim_rows = []

    def _get_profile(team, venue_key):
        # fallback: if a team has no home/away rows, use overall mean from ANY
        if venue_key in profiles and team in profiles[venue_key].index:
            return profiles[venue_key].loc[team]
        if "ANY" in profiles and team in profiles["ANY"].index:
            return profiles["ANY"].loc[team]
        # last resort: global mean
        if venue_key in profiles and not profiles[venue_key].empty:
            return profiles[venue_key].mean()
        # if everything is empty, zeros
        return pd.Series({c: 0.0 for c in feature_cols})

    for home, away in fixtures:
        # Build a single “match feature row” for each team perspective.
        # Home team features: its HOME profile + (optional) set venue="Home"
        home_feat = _get_profile(home, "HOME")
        away_feat = _get_profile(away, "AWAY")

        # We predict home gf/ga using home profile (and venue Home),
        # and away gf/ga using away profile (and venue Away).
        X_home = pd.DataFrame([{**home_feat.to_dict(), "venue": "Home"}])
        X_away = pd.DataFrame([{**away_feat.to_dict(), "venue": "Away"}])

        # Predicted means (clip to avoid negative)
        mu_home_gf = max(0.05, float(model_gf.predict(X_home)[0]))
        mu_home_ga = max(0.05, float(model_ga.predict(X_home)[0]))
        mu_away_gf = max(0.05, float(model_gf.predict(X_away)[0]))
        mu_away_ga = max(0.05, float(model_ga.predict(X_away)[0]))

        # Combine into a consistent scoreline:
        # Home goals should align with Away goals against, etc.
        mu_home = (mu_home_gf + mu_away_ga) / 2.0
        mu_away = (mu_away_gf + mu_home_ga) / 2.0

        hg = int(rng.poisson(mu_home))
        ag = int(rng.poisson(mu_away))

        # Determine results
        if hg > ag:
            home_res, away_res = "W", "L"
        elif hg < ag:
            home_res, away_res = "L", "W"
        else:
            home_res, away_res = "D", "D"

        # Save team-perspective rows (like your dataset)
        sim_rows.append({
            "season": (season_year + 1) if season_year is not None else None,
            "team": home,
            "opponent": away,
            "venue": "Home",
            "gf": hg,
            "ga": ag,
            "result": home_res
        })
        sim_rows.append({
            "season": (season_year + 1) if season_year is not None else None,
            "team": away,
            "opponent": home,
            "venue": "Away",
            "gf": ag,
            "ga": hg,
            "result": away_res
        })

    sim_df = pd.DataFrame(sim_rows)
    sim_table = build_league_table(sim_df)

    return sim_df, sim_table


def load_matches(path, season=None):
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Your CSV uses lowercase 'season'
    if season is not None:
        if "season" in df.columns:
            df = df[df["season"] == season]
        elif "Season" in df.columns:
            df = df[df["Season"] == season]

    return df


def build_league_table(df):
    # normalize headers
    df = df.copy()
    df.columns = df.columns.str.strip()

    required = ["team", "gf", "ga", "result"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Available columns: {list(df.columns)}")

    table = {}

    for _, row in df.iterrows():
        team = row["team"]
        gf = int(row["gf"])
        ga = int(row["ga"])
        res = str(row["result"]).strip().upper()

        if team not in table:
            table[team] = {
                "GP": 0, "W": 0, "D": 0, "L": 0,
                "GF": 0, "GA": 0, "Pts": 0
            }

        table[team]["GP"] += 1
        table[team]["GF"] += gf
        table[team]["GA"] += ga

        r0 = res[:1]
        if r0 == "W":
            table[team]["W"] += 1
            table[team]["Pts"] += 3
        elif r0 == "L":
            table[team]["L"] += 1
        else:
            # Treat anything else as a draw (FBref uses "D" for draw)
            table[team]["D"] += 1
            table[team]["Pts"] += 1

    table_df = pd.DataFrame.from_dict(table, orient="index")
    table_df["GD"] = table_df["GF"] - table_df["GA"]

    table_df = table_df.sort_values(
        by=["Pts", "GD", "GF"],
        ascending=False
    ).reset_index().rename(columns={"index": "Team"})

    table_df.insert(0, "Pos", range(1, len(table_df) + 1))
    return table_df


def print_league_table(table_df):
    display_df = table_df[["Pos", "Team", "GP", "W", "D", "L", "Pts"]]
    print("\n=== LEAGUE TABLE ===\n")
    print(tabulate(display_df, headers="keys", tablefmt="github", showindex=False))


def team_details(df, table_df, team_query):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Find best match team name
    team = None
    for t in table_df["Team"]:
        if team_query.lower() in str(t).lower():
            team = t
            break

    if team is None:
        print("Team not found.")
        return

    row = table_df[table_df["Team"] == team].iloc[0]
    print(f"\n=== {str(team).upper()} ===")
    print(
        f"GP: {row.GP}  W: {row.W}  D: {row.D}  "
        f"L: {row.L}  Pts: {row.Pts}  "
        f"GD: {row.GD}"
    )

    # Show match list for that team (rows are already team-perspective)
    required = ["date", "venue", "opponent", "result", "gf", "ga", "team"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nCannot show match list, missing columns: {missing}")
        return

    matches = []
    team_rows = df[df["team"] == team]

    for _, r in team_rows.iterrows():
        v_raw = str(r.get("venue", "")).strip().lower()
        v = "H" if v_raw.startswith("h") else "A" if v_raw.startswith("a") else ""

        res = str(r.get("result", "")).strip().upper()[:1]
        matches.append([
            r.get("date", ""),
            v,
            r.get("opponent", ""),
            res,
            f'{int(r["gf"])}-{int(r["ga"])}'
        ])

    matches_df = pd.DataFrame(matches, columns=["Date", "V", "Opponent", "Res", "Score"])

    print("\nMatches:")
    print(tabulate(matches_df, headers="keys", tablefmt="github", showindex=False))


def main():
    print(">>> prediction.py starting...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/matches.csv")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(">>> args:", args)

    df = load_matches(args.data, args.season)
    print(">>> loaded df shape:", df.shape)
    print(">>> columns:", list(df.columns))

    table_df = build_league_table(df)
    print(">>> built table rows:", len(table_df))

    print_league_table(table_df)

    if args.simulate:
        print("\n>>> SIMULATION MODE ON")
        sim_df, sim_table = simulate_season(df, season_year=args.season, seed=args.seed)
        print_league_table(sim_table)
        return

    print("\n>>> entering interactive loop (type q to quit)")
    while True:
        cmd = input("\nType a team name for details, or 'q' to quit: ").strip()
        if cmd.lower() == "q":
            break
        team_details(df, table_df, cmd)


if __name__ == "__main__":
    main()
