"""
========================================================================================
Progetto di Data Science - Parte 2: Analisi delle Serie Storiche (Transfermarkt)
Corso di Laurea Magistrale in Ingegneria Informatica e dell'Automazione (UnivPM)
Fase 1: Data Cleaning, Ricostruzione Storica delle Rose ed ETL Mensile (2012 - 2026)
========================================================================================
Questo script esegue l'intera pipeline di ETL sul dataset grezzo di Transfermarkt:
1. Organizza i CSV grezzi in Parte_2_Time_Series/data/raw/ e pulisce eventuali file temporanei.
2. Ricostruisce dinamicamente l'appartenenza storica di ciascun calciatore al proprio club
   e campionato mese per mese, incrociando valutazioni (player_valuations.csv),
   trasferimenti ufficiali (transfers.csv), presenze in campo (appearances.csv) e club (clubs.csv).
3. Costruisce la serie storica mensile regolare (Gennaio 2012 - Giugno 2026, 174 mesi consecutivi,
   s = 12 mesi) del valore di mercato per i Top-5 campionati europei e per ciascuna singola lega
   (con focus dedicato alla Serie A), affiancata dai volumi di spesa del calciomercato.
4. Esporta i dataset puliti in Parte_2_Time_Series/data/processed/ e genera i grafici diagnostici
   ad alta risoluzione (300 DPI) per il Capitolo 1 della Relazione della Parte 2.
========================================================================================
"""

import os
import sys
import json
import shutil
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Percorsi principali confinati in Parte_2_Time_Series
WORKSPACE_ROOT = r"c:\Users\andre\Documents\GitHub\Progetto-Data-Science"
PART2_ROOT = os.path.join(WORKSPACE_ROOT, "Parte_2_Time_Series")
RAW_DIR = os.path.join(PART2_ROOT, "data", "raw")
PROC_DIR = os.path.join(PART2_ROOT, "data", "processed")
FIG_DIR = os.path.join(PART2_ROOT, "figures")
CAP1_IMG_DIR = os.path.join(PART2_ROOT, "Relazione", "capitolo1", "img")
CAP6_IMG_DIR = os.path.join(WORKSPACE_ROOT, "Relazione", "capitolo6", "img")

for d in [RAW_DIR, PROC_DIR, FIG_DIR, CAP1_IMG_DIR, CAP6_IMG_DIR]:
    os.makedirs(d, exist_ok=True)

# Stile grafico coerente con il progetto (600 DPI Ultra-HD, DejaVu Sans, whitegrid)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.dpi"] = 600


def setup_and_clean_raw_files():
    """Sposta i CSV grezzi da data/football/raw a Parte_2_Time_Series/data/raw e pulisce file temp."""
    old_raw = os.path.join(WORKSPACE_ROOT, "data", "football", "raw")
    if os.path.exists(old_raw):
        for fn in os.listdir(old_raw):
            src = os.path.join(old_raw, fn)
            dst = os.path.join(RAW_DIR, fn)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(os.path.join(WORKSPACE_ROOT, "data", "football"), ignore_errors=True)

    # Pulizia file temporanei creati nella fase esplorativa preliminare
    temp_files = [
        os.path.join(WORKSPACE_ROOT, "data", "football_inspection.json"),
        os.path.join(WORKSPACE_ROOT, "scripts", "download_and_inspect.py"),
    ]
    for tf in temp_files:
        if os.path.exists(tf):
            try:
                os.remove(tf)
            except Exception:
                pass


def run_etl():
    print(">>> [FASE 1 - ETL] Avvio caricamento tabelle grezze Transfermarkt...")
    setup_and_clean_raw_files()

    # 1. Caricamento anagrafiche competitions e clubs
    comp_df = pd.read_csv(os.path.join(RAW_DIR, "competitions.csv"), low_memory=False)
    clubs_df = pd.read_csv(os.path.join(RAW_DIR, "clubs.csv"), low_memory=False)

    top5_codes = ["IT1", "GB1", "ES1", "L1", "FR1"]
    top5_names = {
        "IT1": "Serie A",
        "GB1": "Premier League",
        "ES1": "LaLiga",
        "L1": "Bundesliga",
        "FR1": "Ligue 1",
    }

    # Mappa club_id -> domestic_competition_id
    club_to_comp = clubs_df.set_index("club_id")["domestic_competition_id"].to_dict()
    top5_club_ids = set(clubs_df[clubs_df["domestic_competition_id"].isin(top5_codes)]["club_id"].unique())

    # 2. Caricamento player_valuations.csv
    vals_path = os.path.join(RAW_DIR, "player_valuations.csv")
    vals = pd.read_csv(vals_path, low_memory=False)
    raw_vals_count = len(vals)
    vals["date"] = pd.to_datetime(vals["date"], errors="coerce")
    vals = vals.dropna(subset=["player_id", "date", "market_value_in_eur"]).copy()
    vals = vals[vals["market_value_in_eur"] > 0].copy()
    vals["year_month"] = vals["date"].dt.to_period("M")

    # 3. Caricamento transfers.csv
    tr_path = os.path.join(RAW_DIR, "transfers.csv")
    tr = pd.read_csv(tr_path, low_memory=False)
    raw_tr_count = len(tr)
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    tr = tr.dropna(subset=["player_id", "transfer_date"]).copy()
    tr["year_month"] = tr["transfer_date"].dt.to_period("M")
    tr["from_comp"] = tr["from_club_id"].map(club_to_comp).fillna("OTHER")
    tr["to_comp"] = tr["to_club_id"].map(club_to_comp).fillna("OTHER")

    # 4. Caricamento appearances.csv per ricostruire il club storico effettivo in partita
    app_path = os.path.join(RAW_DIR, "appearances.csv")
    app = pd.read_csv(
        app_path,
        usecols=["player_id", "date", "player_club_id", "competition_id"],
        low_memory=False,
    )
    raw_app_count = len(app)
    app["date"] = pd.to_datetime(app["date"], errors="coerce")
    app = app.dropna(subset=["player_id", "date", "player_club_id"]).copy()
    app["year_month"] = app["date"].dt.to_period("M")

    print(f"Righe grezze lette -> Valuations: {raw_vals_count:,} | Transfers: {raw_tr_count:,} | Appearances: {raw_app_count:,}")

    # =========================================================================
    # PASSO 2: RICOSTRUZIONE STORICA DEL CLUB E DEL VALORE MESE PER MESE
    # =========================================================================
    print(">>> Costruzione griglia temporale mensile (warm-up 2009-01 -> 2026-05, 180 mesi target = 15 anni esatti)...")
    grid_months = pd.period_range("2009-01", "2026-05", freq="M")
    target_months = pd.period_range("2011-06", "2026-05", freq="M")  # 180 mesi consecutivi (15 cicli annuali completi)

    # Filtra dati dal 2009 in avanti per un warm-up completo prima del 2011-06
    vals_sub = vals[(vals["year_month"] >= "2009-01") & (vals["year_month"] <= "2026-05")].sort_values("date")
    app_sub = app[(app["year_month"] >= "2009-01") & (app["year_month"] <= "2026-05")].sort_values("date")
    tr_sub = tr[(tr["year_month"] >= "2009-01") & (tr["year_month"] <= "2026-05")].sort_values("transfer_date")

    # 2A. Valore di mercato mensile per giocatore
    val_m = vals_sub.groupby(["player_id", "year_month"])["market_value_in_eur"].last().unstack(level=0)
    val_m = val_m.reindex(grid_months)
    # Forward-fill fino a 18 mesi per coprire interamente la finestra tra due revisioni per giocatori in attività
    val_ff = val_m.ffill(limit=18)

    # 2B. Serie statica "naive" (usando solo player_club_domestic_competition_id statico del 2026 e ffill corto)
    player_static_comp = vals_sub.groupby("player_id")["player_club_domestic_competition_id"].last()
    static_it1_players = player_static_comp[player_static_comp == "IT1"].index.intersection(val_ff.columns)
    naive_serie_a_bn = (val_m.ffill(limit=12)[static_it1_players].sum(axis=1) / 1e9).reindex(target_months)

    # 2C. Ricostruzione storica del club di appartenenza mese per mese (Appearances + Transfers)
    # Eventi da presenze in campo
    app_club_m = app_sub.groupby(["player_id", "year_month"])["player_club_id"].last()
    # Eventi da trasferimenti ufficiali (to_club_id nel mese di trasferimento)
    tr_club_m = tr_sub.groupby(["player_id", "year_month"])["to_club_id"].last()

    # Uniamo gli eventi dando priorità al trasferimento nel mese di mercato e alla presenza nei mesi giocati
    club_events = pd.concat([app_club_m, tr_club_m]).groupby(level=[0, 1]).last().unstack(level=0)
    club_events = club_events.reindex(grid_months)

    # Allineiamo le colonne (player_id) con val_ff
    common_players = val_ff.columns
    club_events = club_events.reindex(columns=common_players)

    # Per ogni giocatore attivo (coperto da valutazione valida in val_ff), il club nel mese t
    # è l'ultimo club registrato da trasferimenti o presenze, con fallback sul club di valutazione
    val_club_m = vals_sub.groupby(["player_id", "year_month"])["current_club_id"].last().unstack(level=0)
    val_club_m = val_club_m.reindex(index=grid_months, columns=common_players).ffill().bfill()
    club_hist = club_events.ffill().bfill().fillna(val_club_m)

    # Mappiamo ogni club_id storico al rispettivo campionato (IT1, GB1, ES1, L1, FR1 o OTHER)
    print(">>> Mappatura club storici -> campionati nazionali Top-5...")
    comp_hist = club_hist.apply(lambda col: col.map(club_to_comp)).fillna("OTHER")

    # Tagliamo sulla finestra di analisi: Gennaio 2012 - Giugno 2026 (174 mesi)
    val_target = val_ff.reindex(target_months)
    comp_target = comp_hist.reindex(target_months)

    # Calcolo valore aggregato mensile per ciascuna lega e per l'aggregato Top-5
    league_series = {}
    league_players_count = {}
    for code in top5_codes:
        mask = (comp_target == code) & (val_target.notna())
        league_series[code] = (val_target.where(mask).sum(axis=1) / 1e9)  # Miliardi di EUR
        league_players_count[code] = mask.sum(axis=1)

    top5_mask = comp_target.isin(top5_codes) & (val_target.notna())
    top5_val_bn = val_target.where(top5_mask).sum(axis=1) / 1e9
    top5_active_players = top5_mask.sum(axis=1)
    top5_mean_player_m = (val_target.where(top5_mask).mean(axis=1)) / 1e6  # Milioni di EUR

    # =========================================================================
    # PASSO 3: AGGREGAZIONE MENSILE DEI TRASFERIMENTI E SPESE DI MERCATO
    # =========================================================================
    print(">>> Aggregazione mensile dei trasferimenti e dei volumi di spesa...")
    # Trasferimenti che coinvolgono almeno un club dei Top-5 campionati (in entrata o in uscita)
    tr_top5 = tr_sub[(tr_sub["to_comp"].isin(top5_codes)) | (tr_sub["from_comp"].isin(top5_codes))].copy()
    tr_it1 = tr_sub[(tr_sub["to_comp"] == "IT1") | (tr_sub["from_comp"] == "IT1")].copy()

    # Calcolo overpay ratio (transfer_fee / market_value_in_eur) sui trasferimenti onerosi
    tr_paid = tr_top5[(tr_top5["transfer_fee"] > 0) & (tr_top5["market_value_in_eur"] > 0)].copy()
    tr_paid["overpay"] = (tr_paid["transfer_fee"] / tr_paid["market_value_in_eur"]).clip(0.1, 10.0)

    # Aggregazione mensile Top-5
    tr_m_top5 = tr_top5.groupby("year_month").agg(
        n_transfers_top5=("player_id", "count"),
        paid_transfers_top5=("transfer_fee", lambda x: int((x.fillna(0) > 0).sum())),
        transfer_spend_top5_m=("transfer_fee", lambda x: float(x.fillna(0).sum() / 1e6)),
        mean_transfer_fee_m=("transfer_fee", lambda x: float(x[x > 0].mean() / 1e6) if (x > 0).any() else 0.0),
    ).reindex(target_months).fillna(0.0)

    overpay_m = tr_paid.groupby("year_month")["overpay"].median().reindex(target_months).fillna(1.0)

    # Aggregazione mensile Serie A (acquisti e movimenti IT1)
    tr_m_it1 = tr_it1.groupby("year_month").agg(
        n_transfers_serie_a=("player_id", "count"),
        transfer_spend_serie_a_m=("transfer_fee", lambda x: float(x.fillna(0).sum() / 1e6)),
    ).reindex(target_months).fillna(0.0)

    # Conteggio aggiornamenti di prezzo grezzi rilasciati da Transfermarkt in ogni mese nei Top-5
    vals_top5_raw = vals_sub[vals_sub["player_club_domestic_competition_id"].isin(top5_codes)]
    updates_count_m = vals_top5_raw.groupby("year_month").size().reindex(target_months).fillna(0).astype(int)

    # =========================================================================
    # PASSO 4: COSTRUZIONE TABELLA MASTER (market_monthly.csv)
    # =========================================================================
    df_master = pd.DataFrame(index=target_months)
    df_master.index.name = "year_month"
    df_master["date"] = df_master.index.to_timestamp()
    df_master["year"] = df_master["date"].dt.year
    df_master["month"] = df_master["date"].dt.month

    df_master["top5_market_value_bn"] = top5_val_bn.round(4)
    df_master["serie_a_market_value_bn"] = league_series["IT1"].round(4)
    df_master["premier_league_market_value_bn"] = league_series["GB1"].round(4)
    df_master["laliga_market_value_bn"] = league_series["ES1"].round(4)
    df_master["bundesliga_market_value_bn"] = league_series["L1"].round(4)
    df_master["ligue1_market_value_bn"] = league_series["FR1"].round(4)
    df_master["serie_a_naive_static_bn"] = naive_serie_a_bn.round(4)

    df_master["active_players_top5"] = top5_active_players.astype(int)
    df_master["active_players_serie_a"] = league_players_count["IT1"].astype(int)
    df_master["mean_player_value_m"] = top5_mean_player_m.round(3)

    # Variabili di calciomercato ed esogene
    df_master["valuation_updates_count"] = updates_count_m
    df_master["transfer_spend_top5_m"] = tr_m_top5["transfer_spend_top5_m"].round(2)
    df_master["transfer_spend_serie_a_m"] = tr_m_it1["transfer_spend_serie_a_m"].round(2)
    df_master["n_transfers_top5"] = tr_m_top5["n_transfers_top5"].astype(int)
    df_master["paid_transfers_top5"] = tr_m_top5["paid_transfers_top5"].astype(int)
    df_master["n_transfers_serie_a"] = tr_m_it1["n_transfers_serie_a"].astype(int)
    df_master["mean_transfer_fee_m"] = tr_m_top5["mean_transfer_fee_m"].round(3)
    df_master["overpay_ratio"] = overpay_m.round(3)

    # Variazioni percentuali mensili (Change %)
    df_master["pct_change_top5"] = (df_master["top5_market_value_bn"].pct_change() * 100).fillna(0.0).round(3)
    df_master["pct_change_serie_a"] = (df_master["serie_a_market_value_bn"].pct_change() * 100).fillna(0.0).round(3)

    # Variabili Dummy di intervento (Shock storici e calendario mercato)
    df_master["shock_neymar_post2017"] = (df_master["date"] >= "2017-08-01").astype(int)
    df_master["shock_covid_2020"] = ((df_master["date"] >= "2020-03-01") & (df_master["date"] <= "2021-05-01")).astype(int)
    df_master["shock_covid_cut_apr2020"] = (df_master["date"] == "2020-04-01").astype(int)
    df_master["shock_saudi_boom_post2023"] = (df_master["date"] >= "2023-07-01").astype(int)
    df_master["is_transfer_window"] = df_master["month"].isin([1, 7, 8]).astype(int)

    # Salvataggio market_monthly.csv
    master_csv_path = os.path.join(PROC_DIR, "market_monthly.csv")
    df_master.reset_index().to_csv(master_csv_path, index=False)
    print(f"Salvato dataset master: {master_csv_path} ({len(df_master)} righe, {df_master.shape[1]} colonne)")

    # Salvataggio formato long per le 5 leghe (leagues_monthly.csv)
    long_rows = []
    for code, name in top5_names.items():
        for ym in target_months:
            long_rows.append({
                "year_month": str(ym),
                "date": ym.to_timestamp(),
                "competition_id": code,
                "league_name": name,
                "market_value_bn": round(float(league_series[code].loc[ym]), 4),
                "active_players": int(league_players_count[code].loc[ym]),
            })
    df_leagues = pd.DataFrame(long_rows)
    leagues_csv_path = os.path.join(PROC_DIR, "leagues_monthly.csv")
    df_leagues.to_csv(leagues_csv_path, index=False)

    # =========================================================================
    # PASSO 5: STATISTICHE DI CERTIFICAZIONE DELL'ETL (etl_summary.json)
    # =========================================================================
    zero_change_top5 = int((df_master["pct_change_top5"].iloc[1:] == 0).sum())
    zero_change_it1_dynamic = int((df_master["pct_change_serie_a"].iloc[1:] == 0).sum())
    naive_pct_change = df_master["serie_a_naive_static_bn"].pct_change().iloc[1:]
    zero_change_it1_naive = int((naive_pct_change.abs() < 1e-6).sum())

    etl_summary = {
        "raw_counts": {
            "player_valuations": int(raw_vals_count),
            "transfers": int(raw_tr_count),
            "appearances": int(raw_app_count),
            "clubs": int(len(clubs_df)),
            "competitions": int(len(comp_df)),
        },
        "time_horizon": {
            "start_month": "2011-06",
            "end_month": "2026-05",
            "total_months": int(len(df_master)),
            "missing_months": 0,
            "missing_values_in_master": int(df_master.isna().sum().sum()),
        },
        "quality_checks": {
            "zero_change_months_top5": f"{zero_change_top5} / {len(df_master)-1}",
            "zero_change_months_serie_a_dynamic": f"{zero_change_it1_dynamic} / {len(df_master)-1}",
            "zero_change_months_serie_a_naive_static": f"{zero_change_it1_naive} / {len(df_master)-1}",
            "top5_value_start_2012_bn": round(float(df_master["top5_market_value_bn"].iloc[0]), 2),
            "top5_value_end_2026_bn": round(float(df_master["top5_market_value_bn"].iloc[-1]), 2),
            "top5_value_max_bn": round(float(df_master["top5_market_value_bn"].max()), 2),
            "top5_covid_drop_apr2020_pct": round(float(df_master.loc[pd.Period("2020-04", freq="M"), "pct_change_top5"]), 2),
            "serie_a_value_start_2012_bn": round(float(df_master["serie_a_market_value_bn"].iloc[0]), 2),
            "serie_a_value_end_2026_bn": round(float(df_master["serie_a_market_value_bn"].iloc[-1]), 2),
            "serie_a_covid_drop_apr2020_pct": round(float(df_master.loc[pd.Period("2020-04", freq="M"), "pct_change_serie_a"]), 2),
            "total_transfer_spend_2012_2026_bn": round(float(df_master["transfer_spend_top5_m"].sum() / 1000.0), 2),
            "total_transfers_top5_2012_2026": int(df_master["n_transfers_top5"].sum()),
        },
    }

    summary_path = os.path.join(PROC_DIR, "etl_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(etl_summary, f, indent=2, ensure_ascii=False)
    print(f"Salvato sommario ETL: {summary_path}")

    # =========================================================================
    # PASSO 6: GENERAZIONE DEI GRAFICI DIAGNOSTICI DELL'ETL (300 DPI)
    # =========================================================================
    print(">>> Generazione grafici diagnostici per il Capitolo 1 (ETL)...")

    month_labels = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]
    seasonal_profile = df_master.groupby("month").agg(
        avg_updates=("valuation_updates_count", "mean"),
        avg_transfers=("n_transfers_top5", "mean"),
        avg_spend_m=("transfer_spend_top5_m", "mean"),
    )

    # FIGURA 1: Complementarità stagionale tra aggiornamenti di listino e trasferimenti di mercato
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(1, 13)
    width = 0.38
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    b1 = ax1.bar(
        x - width / 2,
        seasonal_profile["avg_updates"],
        width=width,
        color="#2b5c8f",
        alpha=0.88,
        label="Aggiornamenti listino (player_valuations)",
    )
    b2 = ax1_twin.bar(
        x + width / 2,
        seasonal_profile["avg_transfers"],
        width=width,
        color="#d95f02",
        alpha=0.88,
        label="Trasferimenti completati (transfers)",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(month_labels, fontsize=10)
    ax1.set_ylabel("Numero medio aggiornamenti di prezzo", color="#2b5c8f", fontweight="bold", fontsize=11)
    ax1_twin.set_ylabel("Numero medio trasferimenti ufficiali", color="#d95f02", fontweight="bold", fontsize=11)
    ax1.set_title("a) Complementarità mensile: listini vs sessioni di mercato", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylim(0, seasonal_profile["avg_updates"].max() * 1.24)
    ax1_twin.set_ylim(0, seasonal_profile["avg_transfers"].max() * 1.24)
    ax1_twin.grid(False)

    # Legenda unificata
    handles = [b1, b2]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper left", frameon=True, fontsize=9.5)

    # Pannello B: Spesa media mensile per trasferimenti (in Milioni di EUR)
    ax2 = axes[1]
    colors_bar = ["#e7298a" if m in [1, 7, 8] else "#66a61e" if m in [2, 6, 9] else "#8da0cb" for m in range(1, 13)]
    bars_spend = ax2.bar(x, seasonal_profile["avg_spend_m"], color=colors_bar, alpha=0.9, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(month_labels, fontsize=10)
    ax2.set_ylabel("Volume medio di spesa mensile (Milioni €)", fontweight="bold", fontsize=11)
    ax2.set_title("b) Distribuzione mensile del volume di spesa sul mercato", fontsize=12, fontweight="bold", pad=10)

    for bar in bars_spend:
        h = bar.get_height()
        if h > 80:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 35,
                f"{h:.0f}M€",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
    ax2.set_ylim(0, seasonal_profile["avg_spend_m"].max() * 1.16)

    plt.suptitle(
        "Diagnostica ETL: integrazione tra aggiornamenti di valutazione e finestre di calciomercato (2012–2026)",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    for out_dir in [FIG_DIR, CAP1_IMG_DIR, CAP6_IMG_DIR]:
        fig.savefig(os.path.join(out_dir, "fig01_etl_complementarita_fonti.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    # FIGURA 2: Confronto tra approccio statico naive e ricostruzione storica dinamica (Serie A e Top-5)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True)

    ax_top = axes[0]
    ax_top.plot(
        df_master["date"],
        df_master["serie_a_market_value_bn"],
        color="#1b9e77",
        linewidth=2.3,
        label="Serie A – Ricostruzione storica dinamica (Valuations + Transfers + Appearances)",
    )
    ax_top.plot(
        df_master["date"],
        df_master["serie_a_naive_static_bn"],
        color="#d95f02",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        label="Serie A – Aggregazione statica naive (basata solo sul club attuale del 2026)",
    )
    ax_top.set_ylabel("Valore di mercato (Miliardi €)", fontweight="bold", fontsize=11)
    ax_top.set_title(
        "a) Effetto della ricostruzione storica dei trasferimenti sul valore della Serie A (2012–2026)",
        fontsize=12,
        fontweight="bold",
    )
    ax_top.legend(loc="upper left", frameon=True, fontsize=10)

    # Annotazione distorsione retrospettiva del metodo naive
    ax_top.annotate(
        "Distorsione retrospettiva (bias del club 2026)\neliminata dalla ricostruzione storica",
        xy=(pd.Timestamp("2016-06-01"), float(df_master.loc[pd.Period("2016-06", freq="M"), "serie_a_market_value_bn"])),
        xytext=(pd.Timestamp("2013-03-01"), 4.2),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#bbbbbb"),
    )

    ax_bot = axes[1]
    ax_bot.plot(
        df_master["date"],
        df_master["top5_market_value_bn"],
        color="#2b5c8f",
        linewidth=2.5,
        label="Aggregato Top-5 Campionati Europei (IT1, GB1, ES1, L1, FR1)",
    )
    ax_bot.fill_between(
        df_master["date"],
        df_master["top5_market_value_bn"],
        color="#2b5c8f",
        alpha=0.12,
    )
    ax_bot.set_ylabel("Valore Top-5 (Miliardi €)", fontweight="bold", fontsize=11)
    ax_bot.set_xlabel("Data (Frequenza mensile regolare: 180 mesi consecutivi = 15 anni, Δt = 1 mese)", fontweight="bold", fontsize=11)
    ax_bot.set_title(
        "b) Serie storica principale post-ETL: valore di mercato aggregato dei Top-5 campionati europei",
        fontsize=12,
        fontweight="bold",
    )
    ax_bot.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    for out_dir in [FIG_DIR, CAP1_IMG_DIR, CAP6_IMG_DIR]:
        fig.savefig(os.path.join(out_dir, "fig02_etl_ricostruzione_storica_confronto.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(">>> [FASE 1 - ETL] Completata con successo!")


if __name__ == "__main__":
    run_etl()

