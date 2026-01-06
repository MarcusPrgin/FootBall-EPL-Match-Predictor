"""Train a simple EPL match outcome model and print evaluation metrics.

Usage:
  python prediction.py --data data/matches.csv

This is a direct Python-script version of the original notebook.
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


def add_basic_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])

    # target: 1 if win else 0
    matches["target"] = (matches["result"] == "W").astype(int)

    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes
    matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype(int)
    matches["day_code"] = matches["date"].dt.dayofweek

    return matches


def rolling_averages(group: pd.DataFrame, cols: list[str], new_cols: list[str]) -> pd.DataFrame:
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed="left").mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


def make_predictions(data: pd.DataFrame, predictors: list[str], *, cutoff: str) -> tuple[pd.DataFrame, float]:
    train = data[data["date"] < cutoff]
    test = data[data["date"] > cutoff]

    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    rf.fit(train[predictors], train["target"])

    preds = rf.predict(test[predictors])

    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error


class MissingDict(dict):
    __missing__ = lambda self, key: key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/matches.csv", help="Path to matches.csv")
    parser.add_argument("--cutoff", default="2022-01-01", help="Train/test date cutoff (YYYY-MM-DD)")
    args = parser.parse_args()

    matches = pd.read_csv(args.data, index_col=0)

    # Drop columns the notebook removed
    for col in ["comp", "notes"]:
        if col in matches.columns:
            del matches[col]

    matches = add_basic_features(matches)

    predictors = ["venue_code", "opp_code", "hour", "day_code"]

    # Rolling features
    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
    new_cols = [f"{c}_rolling" for c in cols]

    matches_rolling = matches.groupby("team", group_keys=False).apply(
        lambda x: rolling_averages(x, cols, new_cols)
    )
    matches_rolling = matches_rolling.reset_index(drop=True)

    combined, precision = make_predictions(matches_rolling, predictors + new_cols, cutoff=args.cutoff)
    combined = combined.merge(
        matches_rolling[["date", "team", "opponent", "result"]],
        left_index=True,
        right_index=True,
    )

    print(f"Precision: {precision:.3f}")
    print("\nSample predictions:")
    print(combined.tail(10).to_string(index=False))

    # Join the two sides of each match, like in the notebook
    map_values = {
        "Brighton and Hove Albion": "Brighton",
        "Manchester United": "Manchester Utd",
        "Newcastle United": "Newcastle Utd",
        "Sheffield United": "Sheffield Utd",
        "Tottenham Hotspur": "Tottenham",
        "West Bromwich Albion": "West Brom",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves",
    }
    mapping = MissingDict(**map_values)

    combined["new_team"] = combined["team"].map(mapping)
    merged = combined.merge(
        combined,
        left_on=["date", "new_team"],
        right_on=["date", "opponent"],
        suffixes=("_team", "_opp"),
    )

    # Show games where the model predicts a win for the team but not for the opponent
    interesting = merged[(merged["predicted_team"] == 1) & (merged["predicted_opp"] == 0)]
    print("\nGames where model favors team (win) over opponent:")
    if interesting.empty:
        print("(none)")
    else:
        cols_out = ["date", "team_team", "opponent_team", "result_team", "actual_team"]
        print(interesting[cols_out].tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
