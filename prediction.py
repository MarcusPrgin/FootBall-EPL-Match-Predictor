import argparse
import pandas as pd
from tabulate import tabulate


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/matches.csv")
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()

    df = load_matches(args.data, args.season)
    table_df = build_league_table(df)

    print_league_table(table_df)

    while True:
        cmd = input("\nType a team name for details, or 'q' to quit: ").strip()
        if cmd.lower() == "q":
            break
        team_details(df, table_df, cmd)


if __name__ == "__main__":
    main()
