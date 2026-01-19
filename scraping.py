"""Scrape Premier League match + shooting stats from FBref and save to CSV.

Usage:
  python scraping.py --out data/matches.csv --seasons 2022 2021

Notes:
- FBref may block aggressive scraping. This script sleeps between requests.
- If the site layout changes, you may need to tweak the selectors.
"""

# import annotations for Python 3.7+

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup



BASE = "https://fbref.com"
START_STANDINGS_URL = f"{BASE}/en/comps/9/Premier-League-Stats"


def get_soup(url: str, *, session: requests.Session) -> BeautifulSoup:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def extract_team_urls(standings_soup: BeautifulSoup) -> List[str]:
    standings_table = standings_soup.select_one("table.stats_table")
    if standings_table is None:
        raise RuntimeError("Could not find standings table on page.")
    links = [a.get("href") for a in standings_table.find_all("a")]
    links = [l for l in links if l and "/squads/" in l]
    return [f"{BASE}{l}" for l in links]


def scrape_team_season(team_url: str, season_year: int, *, session: requests.Session) -> pd.DataFrame | None:
    """Return match rows for one team in one season, or None if unavailable."""
    team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")

    team_soup = get_soup(team_url, session=session)
    team_html = str(team_soup)

    # Scores & Fixtures table
    try:
        matches = pd.read_html(team_html, match="Scores & Fixtures")[0]
    except ValueError:
        return None

    # Find the shooting stats page link
    links = [a.get("href") for a in team_soup.find_all("a")]
    links = [l for l in links if l and "all_comps/shooting/" in l]
    if not links:
        return None

    shooting_soup = get_soup(f"{BASE}{links[0]}", session=session)
    shooting = pd.read_html(str(shooting_soup), match="Shooting")[0]
    shooting.columns = shooting.columns.droplevel()

    # Merge shooting columns into match rows
    try:
        team_data = matches.merge(
            shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]],
            on="Date",
            how="left",
        )
    except Exception:
        return None

    # Keep only EPL
    team_data = team_data[team_data["Comp"] == "Premier League"].copy()

    # Add season + team
    team_data["Season"] = season_year
    team_data["Team"] = team_name
    return team_data


def scrape(seasons: List[int], *, sleep_s: float = 1.0) -> pd.DataFrame:
    all_matches: list[pd.DataFrame] = []
    standings_url = START_STANDINGS_URL

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; epl-match-predictor/1.0; +https://example.com)"
    }

    with requests.Session() as session:
        session.headers.update(headers)

        for year in seasons:
            standings_soup = get_soup(standings_url, session=session)
            team_urls = extract_team_urls(standings_soup)

            # advance to previous season for the next loop.S
            prev = standings_soup.select_one("a.prev")
            if prev and prev.get("href"):
                standings_url = f"{BASE}{prev.get('href')}"

            for team_url in team_urls:
                df = scrape_team_season(team_url, year, session=session)
                if df is not None and not df.empty:
                    all_matches.append(df)
                time.sleep(sleep_s)

    if not all_matches:
        raise RuntimeError("No matches were scraped. FBref layout may have changed or requests were blocked.")

    match_df = pd.concat(all_matches, ignore_index=True)

    # Lowercase columns so prediction.py can rely on: team, opponent, venue, result, gf, ga, date, season...
    match_df.columns = [c.lower() for c in match_df.columns]
    return match_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/matches.csv", help="Output CSV path.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2022, 2021],
        help="Season years to scrape (e.g., 2022 2021).",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep between team requests (seconds).")
    args = parser.parse_args()

    df = scrape(args.seasons, sleep_s=args.sleep)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Saved {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
