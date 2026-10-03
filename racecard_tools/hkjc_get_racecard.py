"""Download official HKJC local racecard entries into a model-input CSV.

Example for 27 Sep 2026:
  python hkjc_get_racecard.py --date 2026-09-27 --venue ST -o hkjc_20260927_racecard.csv

Official source: HKJC public local racecard and odds pages. The script tries to
include current WIN/PLACE odds; if the odds service is unavailable, those fields
stay blank and can be refreshed later in the Streamlit app.
"""
from __future__ import annotations

import argparse
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://racing.hkjc.com/zh-hk/local/information/racecard"
GRAPHQL_URL = "https://info.cld.hkjc.com/graphql/base/"
ODDS_QUERY = """query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
  raceMeetings(date: $date, venueCode: $venueCode) {
    pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
      id status sellStatus oddsType lastUpdateTime guarantee minTicketCost name_en name_ch
      leg { number races }
      cWinSelections { composite name_ch name_en starters }
      oddsNodes { combString oddsValue hotFavourite oddsDropValue bankerOdds { combString oddsValue } }
    }
  }
}"""
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "zh-HK,zh;q=0.9,en;q=0.6",
}


def clean_person(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    # Strip apprentice allowance suffixes such as "(-7)"; the feature's
    # historical jockey key stores the jockey's base name.
    return re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).replace("\n", " ").strip() for c in frame.columns]
    return frame


def scrape_race(session: requests.Session, date_text: str, venue: str, race_no: int) -> pd.DataFrame:
    params = {"racedate": date_text.replace("-", "/"), "Racecourse": venue, "RaceNo": race_no}
    response = session.get(BASE_URL, params=params, headers=HEADERS, timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    page_text = soup.get_text(" ", strip=True)

    # Restrict metadata parsing to the selected race heading and following text.
    heading = re.search(rf"第\s*{race_no}\s*場", page_text)
    if not heading:
        raise RuntimeError(f"官方頁找不到 {date_text} {venue} 第 {race_no} 場標題。")
    race_text = page_text[heading.start():heading.start() + 700]
    distance_match = re.search(r"(\d{3,4})\s*米", race_text)
    if not distance_match:
        raise RuntimeError(f"無法從官方第 {race_no} 場排位頁讀取路程。")
    distance = int(distance_match.group(1))
    surface = "泥地" if ("全天候" in race_text or "全天候跑道" in race_text) else "草地"

    tables = pd.read_html(StringIO(response.text))
    runners = None
    for table in tables:
        candidate = normalize_columns(table)
        columns = set(candidate.columns.astype(str))
        if {"馬匹編號", "馬名", "負磅", "騎師", "檔位", "練馬師"}.issubset(columns):
            runners = candidate
            break
    if runners is None:
        raise RuntimeError(f"第 {race_no} 場排位頁找不到預期的馬匹資料表。")

    rows = []
    race_id = f"{date_text.replace('-', '')}-{race_no:02d}"
    for _, runner in runners.iterrows():
        horse_no = pd.to_numeric(runner.get("馬匹編號"), errors="coerce")
        if pd.isna(horse_no):
            continue
        horse_name = str(runner.get("馬名", "")).strip()
        if not horse_name or horse_name.lower() == "nan":
            continue
        weight = pd.to_numeric(runner.get("負磅"), errors="coerce")
        draw = pd.to_numeric(runner.get("檔位"), errors="coerce")
        rows.append({
            "賽事編號": race_id,
            "名次": "",  # Unknown before the race; intentionally blank.
            "馬號": int(horse_no),
            "馬名": horse_name,
            "騎師": clean_person(runner.get("騎師")),
            "練馬師": clean_person(runner.get("練馬師")),
            # Pre-race assigned weight from the official racecard; not a result.
            "實際負磅": float(weight) if pd.notna(weight) else pd.NA,
            "排位檔位": int(draw) if pd.notna(draw) else pd.NA,
            "距離": distance,
            "場地": surface,
            "獨贏賠率": pd.NA,
            "位置賠率": pd.NA,
            "racecourse_code": venue,
            "official_distance_m": distance,
        })
    if not rows:
        raise RuntimeError(f"第 {race_no} 場沒有讀取到任何正選馬匹。")
    return pd.DataFrame(rows)


def fetch_current_odds(session: requests.Session, date_text: str, venue: str):
    """Return {(race_no, horse_no): (win, place)} plus official update time."""
    payload = {
        "operationName": "racing",
        "variables": {
            "date": date_text,
            "venueCode": venue,
            "raceNo": None,
            "oddsTypes": ["WIN", "PLA"],
        },
        "query": ODDS_QUERY,
    }
    headers = {
        **HEADERS,
        "Accept": "*/*",
        "Content-Type": "application/json",
        "Origin": "https://bet.hkjc.com",
        "Referer": "https://bet.hkjc.com/",
    }
    response = session.post(GRAPHQL_URL, json=payload, headers=headers, timeout=25)
    response.raise_for_status()
    data = response.json()
    if data.get("errors"):
        raise RuntimeError(f"HKJC odds GraphQL errors: {data['errors']}")
    meetings = (data.get("data") or {}).get("raceMeetings") or []
    if not meetings:
        raise RuntimeError("HKJC odds API 尚未回傳賽事資料。")

    odds = {}
    timestamps = []
    for pool in meetings[0].get("pmPools") or []:
        pool_type = str(pool.get("oddsType", "")).upper()
        if pool_type not in {"WIN", "PLA"}:
            continue
        if pool.get("lastUpdateTime"):
            timestamps.append(str(pool["lastUpdateTime"]))
        race_numbers = (pool.get("leg") or {}).get("races") or []
        for race_no in race_numbers:
            for node in pool.get("oddsNodes") or []:
                horse_no = str(node.get("combString", "")).strip()
                value = pd.to_numeric(node.get("oddsValue"), errors="coerce")
                if not horse_no or pd.isna(value):
                    continue
                key = (int(race_no), int(float(horse_no)))
                current = odds.setdefault(key, {"獨贏賠率": pd.NA, "位置賠率": pd.NA})
                current["獨贏賠率" if pool_type == "WIN" else "位置賠率"] = float(value)
    return odds, (max(timestamps) if timestamps else None)


def download_racecard(date_text: str, venue: str, output: str | Path) -> pd.DataFrame:
    venue = venue.upper()
    if venue not in {"ST", "HV"}:
        raise ValueError("venue 只接受 ST（沙田）或 HV（跑馬地）。")
    date_text = pd.to_datetime(date_text).strftime("%Y-%m-%d")
    session = requests.Session()

    # Discover available local race numbers from HKJC's day navigation links.
    response = session.get(
        BASE_URL,
        params={"racedate": date_text.replace("-", "/"), "Racecourse": venue, "RaceNo": 1},
        headers=HEADERS,
        timeout=25,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    pattern = re.compile(rf"racedate={re.escape(date_text.replace('-', '/'))}.*?Racecourse={venue}.*?RaceNo=(\d+)", re.I)
    numbers = sorted({int(m.group(1)) for a in soup.find_all("a", href=True) for m in [pattern.search(a["href"])] if m})
    numbers = sorted(set(numbers) | {1})
    if len(numbers) <= 1:
        # Fallback: common HKJC site renders links with encoded separators or
        # the selected race itself is the only parsed link.
        numbers = list(range(1, 13))

    all_runners = []
    for i, race_no in enumerate(numbers):
        try:
            race_df = scrape_race(session, date_text, venue, race_no)
            all_runners.append(race_df)
            print(f"取得第 {race_no} 場：{len(race_df)} 匹正選馬")
        except RuntimeError as exc:
            # Stop at the first unavailable race after at least one valid race;
            # gaps can occur in the navigation where no local race exists.
            if all_runners:
                print(f"停止於場次 {race_no}：{exc}")
                break
            raise
        if i < len(numbers) - 1:
            time.sleep(0.25)

    if not all_runners:
        raise RuntimeError("沒有抓取到任何排位資料，請核對賽日、場地及 HKJC 頁面狀態。")
    result = pd.concat(all_runners, ignore_index=True)
    odds_update_time = None
    try:
        current_odds, odds_update_time = fetch_current_odds(session, date_text, venue)
        for idx, row in result.iterrows():
            race_no = int(str(row["賽事編號"]).rsplit("-", 1)[1])
            horse_no = int(row["馬號"])
            values = current_odds.get((race_no, horse_no))
            if values:
                result.loc[idx, "獨贏賠率"] = values["獨贏賠率"]
                result.loc[idx, "位置賠率"] = values["位置賠率"]
        print(f"已讀取 HKJC 即時賠率；API 更新時間：{odds_update_time or '未提供'}")
    except Exception as exc:
        print(f"提示：本次未能讀取即時賠率（{exc}）；賠率欄保持空白，可稍後在 Streamlit 刷新。")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    result.attrs["odds_update_time"] = odds_update_time
    print(f"\n完成：{len(result)} 匹馬、{result['賽事編號'].nunique()} 場。輸出：{output}")
    print("請注意賠率是抓取時的市場快照，賽前會變動；預測測試請記錄查詢時間。")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取 HKJC 官方本地賽事排位表")
    parser.add_argument("--date", required=True, help="賽日，例如 2026-09-27")
    parser.add_argument("--venue", default="ST", help="ST 沙田或 HV 跑馬地")
    parser.add_argument("-o", "--output", default=None, help="輸出 CSV")
    args = parser.parse_args()
    date_text = pd.to_datetime(args.date).strftime("%Y-%m-%d")
    output = args.output or f"hkjc_{date_text.replace('-', '')}_racecard.csv"
    download_racecard(date_text, args.venue, output)


if __name__ == "__main__":
    main()
