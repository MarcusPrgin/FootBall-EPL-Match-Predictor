import argparse
import pandas as pd
from tabulate import tabulate


def load_matches(path, season=None):
    df = pd.read_csv(path)

    if season is not None and "Season" in df.columns:
        df = df[df["Season"] == season]

    return df


def build_league_table(df):
    table = {}

    for _, row in df.iterrows():
        home = row["Home"]
        away = row["Away"]
        hg = row["HomeGoals"]
        ag = row["AwayGoals"]

        for team in [home, away]:
            if team not in table:
                table[team] = {
                    "GP": 0, "W": 0, "D": 0, "L": 0,
                    "GF": 0, "GA": 0, "Pts": 0
                }

        table[home]["GP"] += 1
        table[away]["GP"] += 1

        table[home]["GF"] += hg
        table[home]["GA"] += ag
        table[away]["GF"] += ag
        table[away]["GA"] += hg

        if hg > ag:
            table[home]["W"] += 1
            table[home]["Pts"] += 3
            table[away]["L"] += 1
        elif ag > hg:
            table[away]["W"] += 1
            table[away]["Pts"] += 3
            table[home]["L"] += 1
        else:
            table[home]["D"] += 1
            table[away]["D"] += 1
            table[home]["Pts"] += 1
            table[away]["Pts"] += 1

    table_df = pd.DataFrame.from_dict(table, orient="index")
    table_df["GD"] = table_df["GF"] - table_df["GA"]

    table_df = table_df.sort_values(
        by=["Pts", "GD", "GF"],
        ascending=False
    ).reset_index().rename(columns={"index": "Team"})

    table_df.insert(0, "Pos", range(1, len(table_df) + 1))
    return table_df


def print_league_table(table_df):
    display_df = table_df[
        ["Pos", "Team", "GP", "W", "D", "L", "Pts"]
    ]
    print("\n=== LEAGUE TABLE ===\n")
    print(tabulate(display_df, headers="keys", tablefmt="github", showindex=False))


def team_details(df, table_df, team_query):
    team = None
    for t in table_df["Team"]:
        if team_query.lower() in t.lower():
            team = t
            break

    if team is None:
        print("Team not found.")
        return

    row = table_df[table_df["Team"] == team].iloc[0]
    print(f"\n=== {team.upper()} ===")
    print(
        f"GP: {row.GP}  W: {row.W}  D: {row.D}  "
        f"L: {row.L}  Pts: {row.Pts}  "
        f"GD: {row.GD}"
    )

    matches = []
    for _, r in df.iterrows():
        if r["Home"] == team:
            res = "W" if r["HomeGoals"] > r["AwayGoals"] else "L" if r["HomeGoals"] < r["AwayGoals"] else "D"
            matches.append([
                r.get("Date", ""),
                "H",
                r["Away"],
                res,
                f'{r["HomeGoals"]}-{r["AwayGoals"]}'
            ])
        elif r["Away"] == team:
            res = "W" if r["AwayGoals"] > r["HomeGoals"] else "L" if r["AwayGoals"] < r["HomeGoals"] else "D"
            matches.append([
                r.get("Date", ""),
                "A",
                r["Home"],
                res,
                f'{r["AwayGoals"]}-{r["HomeGoals"]}'
            ])

    matches_df = pd.DataFrame(
        matches,
        columns=["Date", "V", "Opponent", "Res", "Score"]
    )

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
