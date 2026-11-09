import os
import sys
import glob
import shutil
import zipfile
import urllib.request
import json
import pandas as pd
import numpy as np

WORKSPACE = r"c:\Users\andre\Documents\GitHub\Progetto-Data-Science"
RAW_DIR = os.path.join(WORKSPACE, "data", "football", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

REQUIRED_FILES = [
    "competitions.csv",
    "games.csv",
    "clubs.csv",
    "player_valuations.csv",
    "transfers.csv",
    "players.csv",
]

def find_existing_dataset():
    user_home = os.path.expanduser("~")
    search_roots = [
        WORKSPACE,
        os.path.join(user_home, "Downloads"),
        os.path.join(user_home, "Desktop"),
        os.path.join(user_home, "Documents"),
        os.path.join(user_home, "OneDrive"),
    ]
    # 1. Look for already extracted folder containing player_valuations.csv and games.csv
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "player_valuations.csv" in filenames and "games.csv" in filenames:
                if os.path.abspath(dirpath) != os.path.abspath(RAW_DIR):
                    return ("dir", dirpath)
    # 2. Look for archive*.zip in Downloads / Desktop / Workspace
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                low = fn.lower()
                if low.startswith("archive") and low.endswith(".zip"):
                    full_zip = os.path.join(dirpath, fn)
                    try:
                        with zipfile.ZipFile(full_zip, "r") as zf:
                            names = zf.namelist()
                            if any("player_valuations.csv" in n for n in names):
                                return ("zip", full_zip)
                    except Exception:
                        pass
    return (None, None)

def ensure_data():
    if all(os.path.exists(os.path.join(RAW_DIR, f)) for f in REQUIRED_FILES):
        print("All required CSV files are already in", RAW_DIR)
        return

    kind, path = find_existing_dataset()
    if kind == "dir":
        print(f"Found existing extracted dataset at: {path}")
        for fn in os.listdir(path):
            if fn.endswith(".csv"):
                src = os.path.join(path, fn)
                dst = os.path.join(RAW_DIR, fn)
                if not os.path.exists(dst):
                    print(f"Copying {fn}...")
                    shutil.copy2(src, dst)
        return
    elif kind == "zip":
        print(f"Found existing zip dataset at: {path}")
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(RAW_DIR)
        print("Extracted to", RAW_DIR)
        return

    # Otherwise download from Kaggle API v1 public download URL
    url = "https://www.kaggle.com/api/v1/datasets/download/davidcariboo/player-scores"
    zip_path = os.path.join(RAW_DIR, "player-scores.zip")
    print(f"Downloading dataset from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 / total
                print(f"Downloaded {downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB ({pct:.1f}%)", flush=True)
    print("Extracting zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    os.remove(zip_path)
    print("Done extracting!")

def inspect_data():
    summary = {}
    summary["files_in_raw"] = sorted(os.listdir(RAW_DIR))

    # 1. Inspect games.csv (Attendance in Top-5 leagues & Serie A)
    games_path = os.path.join(RAW_DIR, "games.csv")
    games = pd.read_csv(games_path, low_memory=False)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    top5 = ["IT1", "GB1", "ES1", "L1", "FR1"]
    g_top5 = games[games["competition_id"].isin(top5) & (games["season"] >= 2012)].copy()
    g_it1 = games[(games["competition_id"] == "IT1") & (games["season"] >= 2012)].copy()

    g_it1["year_month"] = g_it1["date"].dt.to_period("M").astype(str)
    att_it1_monthly = (
        g_it1.groupby("year_month")
        .agg(
            matches=("game_id", "count"),
            att_non_null=("attendance", lambda x: int(x.notna().sum())),
            att_gt_zero=("attendance", lambda x: int((x.fillna(0) > 0).sum())),
            att_mean=("attendance", "mean"),
            att_median=("attendance", "median"),
        )
        .reset_index()
    )

    summary["games_date_range"] = [str(games["date"].min()), str(games["date"].max())]
    summary["seasons_available"] = sorted(games["season"].dropna().unique().astype(int).tolist())
    summary["it1_attendance_monthly_sample"] = att_it1_monthly.tail(36).to_dict(orient="records")
    summary["it1_total_months_with_matches"] = len(att_it1_monthly)

    # 2. Inspect player_valuations.csv
    val_path = os.path.join(RAW_DIR, "player_valuations.csv")
    vals = pd.read_csv(val_path, low_memory=False)
    vals["date"] = pd.to_datetime(vals["date"], errors="coerce")
    summary["valuations_columns"] = list(vals.columns)
    summary["valuations_date_range"] = [str(vals["date"].min()), str(vals["date"].max())]
    summary["valuations_total_rows"] = int(len(vals))

    vals_recent = vals[(vals["date"] >= "2012-01-01") & (vals["player_club_domestic_competition_id"].isin(top5))].copy()
    vals_recent["year_month"] = vals_recent["date"].dt.to_period("M").astype(str)
    raw_updates_per_month = vals_recent.groupby("year_month").size().tail(36).to_dict()
    summary["raw_valuation_updates_per_month_top5_last36m"] = raw_updates_per_month

    # Build monthly panel for IT1 and Top-5 using forward-fill per player
    # Create monthly grid from 2012-01 to 2024-12
    vals_sorted = vals[vals["date"] >= "2010-01-01"].sort_values("date").copy()
    vals_sorted["year_month"] = vals_sorted["date"].dt.to_period("M")
    # Take last valuation of each player in each month
    monthly_player = (
        vals_sorted.groupby(["player_id", "year_month"])
        .last()
        .reset_index()[["player_id", "year_month", "market_value_in_eur", "current_club_id", "player_club_domestic_competition_id"]]
    )

    # Pivot to player x month grid
    all_months = pd.period_range("2012-01", vals_sorted["year_month"].max(), freq="M")
    piv_val = monthly_player.pivot(index="year_month", columns="player_id", values="market_value_in_eur").reindex(all_months)
    piv_comp = monthly_player.pivot(index="year_month", columns="player_id", values="player_club_domestic_competition_id").reindex(all_months)

    # Forward fill up to 12 months
    piv_val_ff = piv_val.ffill(limit=12)
    piv_comp_ff = piv_comp.ffill(limit=12)

    it1_mask = piv_comp_ff == "IT1"
    top5_mask = piv_comp_ff.isin(top5)

    it1_total_val = piv_val_ff.where(it1_mask).sum(axis=1) / 1e9  # in Billions EUR
    it1_mean_val = piv_val_ff.where(it1_mask).mean(axis=1) / 1e6   # in Millions EUR
    top5_total_val = piv_val_ff.where(top5_mask).sum(axis=1) / 1e9

    # Check how often consecutive months have identical values
    it1_diff_pct = it1_total_val.pct_change().abs() * 100
    top5_diff_pct = top5_total_val.pct_change().abs() * 100

    summary["panel_it1_ZeroChangeMonths_out_of_total"] = f"{int((it1_diff_pct == 0).sum())} / {len(it1_diff_pct)}"
    summary["panel_it1_MedianMonthlyPctChange"] = float(it1_diff_pct.median())
    summary["panel_top5_MedianMonthlyPctChange"] = float(top5_diff_pct.median())
    summary["panel_it1_last24m_billions"] = {str(k): round(float(v), 3) for k, v in it1_total_val.tail(24).items()}
    summary["panel_top5_last24m_billions"] = {str(k): round(float(v), 3) for k, v in top5_total_val.tail(24).items()}

    # 3. Inspect transfers.csv
    tr_path = os.path.join(RAW_DIR, "transfers.csv")
    if os.path.exists(tr_path):
        tr = pd.read_csv(tr_path, low_memory=False)
        tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
        summary["transfers_columns"] = list(tr.columns)
        summary["transfers_date_range"] = [str(tr["transfer_date"].min()), str(tr["transfer_date"].max())]
        tr_rec = tr[(tr["transfer_date"] >= "2012-01-01") & (tr["transfer_date"] <= "2025-12-31")].copy()
        tr_rec["year_month"] = tr_rec["transfer_date"].dt.to_period("M").astype(str)
        tr_monthly = tr_rec.groupby("year_month").agg(
            n_transfers=("player_id", "count"),
            total_fee_m=("transfer_fee", lambda x: round(float(x.fillna(0).sum() / 1e6), 2)),
        ).tail(24).to_dict(orient="index")
        summary["transfers_monthly_last24m"] = tr_monthly

    out_json = os.path.join(WORKSPACE, "data", "football_inspection.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Saved inspection summary to", out_json)

if __name__ == "__main__":
    ensure_data()
    inspect_data()

