"""
========================================================================================
Progetto di Data Science - Parte 2: Analisi delle Serie Storiche (Transfermarkt)
Corso di Laurea Magistrale in Ingegneria Informatica e dell'Automazione (UnivPM)
Fase 1: Data Cleaning, Ricostruzione Storica delle Rose ed ETL Mensile (2013 - 2026)
========================================================================================
Questo script esegue l'intera pipeline di ETL sul dataset grezzo di Transfermarkt:
1. Legge le 7 tabelle grezze necessarie da Parte_2_Time_Series/data/raw/
   (player_valuations, transfers, appearances, games, clubs, competitions, national_teams).
2. Ricostruisce mese per mese, in modo puramente causale (zero bfill / zero look-ahead):
   - il valore di mercato del singolo calciatore (ffill max 15 mesi);
   - il club di appartenenza come evento piu' recente tra presenze di club (escluse
     Nazionali, valide 12 mesi) e trasferimenti (validi 36 mesi, con i rientri gratuiti
     di fine prestito del 30 giugno allineati al 1° luglio della nuova stagione);
   - il campionato effettivamente disputato dal club in quella stagione tramite games.csv
     (gestendo promozioni, retrocessioni e il prolungamento pandemico di luglio 2020).
3. Costruisce la serie storica mensile regolare (Giugno 2013 - Maggio 2026, 156 mesi) del
   valore delle rose dei Top-5 campionati e di ciascuna lega, affiancata dalla spesa per
   ACQUISTI a titolo oneroso (solo trasferimenti in entrata nei club del campionato).
4. Esporta i dataset in Parte_2_Time_Series/data/processed/ e genera le figure del
   Capitolo 6 (Sezione 1 della Parte 2) della Relazione a 600 DPI.
========================================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

PART2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PART2_ROOT not in sys.path:
    sys.path.insert(0, PART2_ROOT)

from src.config import (
    DATA_RAW_DIR as RAW_DIR,
    DATA_PROCESSED_DIR as PROC_DIR,
    FIGURES_DIR as FIG_DIR,
    CAP6_IMG_DIR,
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.dpi"] = 600

# -----------------------------------------------------------------------------
# Parametri della ricostruzione causale
# -----------------------------------------------------------------------------
TOP5_CODES = ["IT1", "GB1", "ES1", "L1", "FR1"]
TOP5_NAMES = {"IT1": "Serie A", "GB1": "Premier League", "ES1": "LaLiga", "L1": "Bundesliga", "FR1": "Ligue 1"}

TR_WARMUP_START = "2009-01"   # warm-up trasferimenti (copre finestra 36 mesi prima del 2013-06)
VAL_WARMUP_START = "2011-01"  # warm-up valutazioni e spesa mobile 12m
EVENTS_START = "2012-07"      # prima presenza disponibile in appearances.csv (stagione 2012/13)
TARGET_START = "2013-06"      # primo mese con finestra presenze 12m completamente coperta
TARGET_END = "2026-05"        # ultimo mese completo dell'estrazione

APP_ACTIVITY_MONTHS = 12      # una presenza di club resta valida per 12 mesi (limit=11)
TR_ACTIVITY_MONTHS = 36       # un trasferimento resta valido fino a 36 mesi (limit=35, mitiga buco neopromosse)
VALUATION_FFILL = 15          # una valutazione resta valida per max 15 mesi (evita buchi tra aggiornamenti >12m)


def costruisci_mappa_stagione_mese(games):
    """
    Associa ogni mese alla stagione calcistica di competenza:
    regola standard (luglio-dicembre -> anno, gennaio-giugno -> anno-1), corretta in modo
    fattuale sui mesi in cui il calendario Top-5 di games.csv ha disputato gare della stagione
    precedente (es. luglio 2020 durante il prolungamento COVID della stagione 2019/2020: 211 gare).
    """
    gl = games[games["competition_id"].isin(TOP5_CODES)].copy()
    gl["date"] = pd.to_datetime(gl["date"], errors="coerce")
    gl["ym"] = gl["date"].dt.to_period("M")
    dom = gl.groupby(["ym", "season"]).size().unstack(fill_value=0)

    override = {}
    for ym, row in dom.iterrows():
        if row.sum() > 0:
            std_s = ym.year if ym.month >= 7 else ym.year - 1
            dom_s = int(row.idxmax())
            # Override solo quando la netta maggioranza delle gare ufficiali del mese appartiene all'altra stagione
            if dom_s != std_s and row[dom_s] >= 0.75 * row.sum():
                override[ym] = dom_s
    return override


def stagione_da_mese(periodi, override_stagione=None):
    """Restituisce l'anno di inizio stagione per ciascun mese (Period M)."""
    override_stagione = override_stagione or {}
    periodi = pd.Series(periodi)
    unici = periodi.unique()
    mappa = {
        p: override_stagione.get(p, p.year if p.month >= 7 else p.year - 1)
        for p in unici
    }
    return periodi.map(mappa).to_numpy()


def carica_dati_grezzi():
    """Legge le sette tabelle grezze necessarie e applica la pulizia di base."""
    comp_df = pd.read_csv(os.path.join(RAW_DIR, "competitions.csv"), low_memory=False)
    clubs_df = pd.read_csv(os.path.join(RAW_DIR, "clubs.csv"), low_memory=False)

    vals = pd.read_csv(os.path.join(RAW_DIR, "player_valuations.csv"), low_memory=False)
    raw_counts = {"player_valuations": len(vals)}
    vals["date"] = pd.to_datetime(vals["date"], errors="coerce")
    vals = vals.dropna(subset=["player_id", "date", "market_value_in_eur"])
    vals = vals[vals["market_value_in_eur"] > 0].copy()
    vals["year_month"] = vals["date"].dt.to_period("M")

    tr = pd.read_csv(os.path.join(RAW_DIR, "transfers.csv"), low_memory=False)
    raw_counts["transfers"] = len(tr)
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    tr = tr.dropna(subset=["player_id", "transfer_date", "to_club_id"]).copy()
    # I rientri gratuiti da prestito datati 30 giugno (92% dei trasferimenti di giugno)
    # appartengono contrattualmente alla nuova stagione che inizia il 1° luglio:
    # posticiparli di 1 giorno e' strettamente causale ed elimina il picco fittizio di giugno.
    mask_j30 = (
        (tr["transfer_date"].dt.month == 6)
        & (tr["transfer_date"].dt.day == 30)
        & (tr["transfer_fee"].fillna(0.0) <= 0.0)
    )
    n_j30_shifted = int(mask_j30.sum())
    tr.loc[mask_j30, "transfer_date"] = tr.loc[mask_j30, "transfer_date"] + pd.Timedelta(days=1)
    tr["year_month"] = tr["transfer_date"].dt.to_period("M")

    app = pd.read_csv(
        os.path.join(RAW_DIR, "appearances.csv"),
        usecols=["player_id", "date", "player_club_id", "competition_id"],
        low_memory=False,
    )
    raw_counts["appearances"] = len(app)
    # Esclude le presenze in Nazionale (EURO, Mondiali, Coppa d'Africa, ecc.),
    # dove player_club_id contiene l'ID della Nazionale invece del club di appartenenza.
    nt_comps = set(comp_df.loc[comp_df["type"] == "national_team_competition", "competition_id"])
    app = app[~app["competition_id"].isin(nt_comps)].copy()
    nt_path = os.path.join(RAW_DIR, "national_teams.csv")
    n_nt = 0
    if os.path.exists(nt_path):
        nt_df = pd.read_csv(nt_path, usecols=["national_team_id"])
        n_nt = len(nt_df)
        nt_ids = set(nt_df["national_team_id"])
        app = app[~app["player_club_id"].isin(nt_ids)].copy()
    app["date"] = pd.to_datetime(app["date"], errors="coerce")
    app = app.dropna(subset=["player_id", "date", "player_club_id"]).copy()
    app["year_month"] = app["date"].dt.to_period("M")

    games = pd.read_csv(
        os.path.join(RAW_DIR, "games.csv"),
        usecols=["competition_id", "season", "date", "home_club_id", "away_club_id"],
        low_memory=False,
    )
    raw_counts["games"] = len(games)
    raw_counts["clubs"] = len(clubs_df)
    raw_counts["competitions"] = len(comp_df)
    raw_counts["national_teams"] = n_nt
    return vals, tr, app, games, clubs_df, raw_counts, n_j30_shifted


def costruisci_mappa_club_stagione(games):
    """(stagione, club_id) -> campionato Top-5 in cui il club ha giocato quella stagione (da games.csv)."""
    gl = games[games["competition_id"].isin(TOP5_CODES)]
    casa = gl[["season", "competition_id", "home_club_id"]].rename(columns={"home_club_id": "club_id"})
    fuori = gl[["season", "competition_id", "away_club_id"]].rename(columns={"away_club_id": "club_id"})
    mappa = pd.concat([casa, fuori]).drop_duplicates(["season", "club_id"])
    mappa = mappa.rename(columns={"competition_id": "league"})
    return mappa[["season", "club_id", "league"]]


def run_etl():
    print(">>> [FASE 1 - ETL] Caricamento tabelle grezze Transfermarkt...")
    vals, tr, app, games, clubs_df, raw_counts, n_j30_shifted = carica_dati_grezzi()
    print("Righe grezze lette -> " + " | ".join(f"{k}: {v:,}" for k, v in raw_counts.items()))
    print(f"Rientri prestiti gratuiti del 30 giugno allineati al 1° luglio: {n_j30_shifted:,}")

    tr_grid = pd.period_range(TR_WARMUP_START, TARGET_END, freq="M")
    val_grid = pd.period_range(VAL_WARMUP_START, TARGET_END, freq="M")
    target_months = pd.period_range(TARGET_START, TARGET_END, freq="M")
    mappa_club_stagione = costruisci_mappa_club_stagione(games)
    override_stagione = costruisci_mappa_stagione_mese(games)
    print(f"Override stagionali da calendario gare effettivo: { {str(k): v for k, v in override_stagione.items()} }")

    # =========================================================================
    # PASSO 1: valore di mercato mensile per giocatore (propagazione causale max 15 mesi)
    # =========================================================================
    vals_sub = vals[(vals["year_month"] >= VAL_WARMUP_START) & (vals["year_month"] <= TARGET_END)].sort_values("date")
    val_m = vals_sub.groupby(["player_id", "year_month"])["market_value_in_eur"].last().unstack(level=0)
    val_m = val_m.reindex(val_grid)
    val_ff = val_m.ffill(limit=VALUATION_FFILL)

    # =========================================================================
    # PASSO 2: club di appartenenza mese per mese (evento piu' recente tra presenze 12m e trasferimenti 36m)
    # =========================================================================
    ev_app = app[(app["year_month"] >= EVENTS_START) & (app["year_month"] <= TARGET_END)]
    ev_app = ev_app[ev_app["player_id"].isin(val_ff.columns)].sort_values("date", kind="mergesort")
    app_last = ev_app.groupby(["player_id", "year_month"]).last()[["player_club_id", "date"]]
    app_club = app_last["player_club_id"].unstack(0).reindex(index=tr_grid, columns=val_ff.columns).ffill(limit=APP_ACTIVITY_MONTHS - 1)
    app_date = app_last["date"].astype("int64").unstack(0).reindex(index=tr_grid, columns=val_ff.columns).ffill(limit=APP_ACTIVITY_MONTHS - 1)

    ev_tr = tr[(tr["year_month"] >= TR_WARMUP_START) & (tr["year_month"] <= TARGET_END)]
    ev_tr = ev_tr[ev_tr["player_id"].isin(val_ff.columns)].sort_values("transfer_date", kind="mergesort")
    tr_last = ev_tr.groupby(["player_id", "year_month"]).last()[["to_club_id", "transfer_date"]]
    tr_club = tr_last["to_club_id"].unstack(0).reindex(index=tr_grid, columns=val_ff.columns).ffill(limit=TR_ACTIVITY_MONTHS - 1)
    tr_date = tr_last["transfer_date"].astype("int64").unstack(0).reindex(index=tr_grid, columns=val_ff.columns).ffill(limit=TR_ACTIVITY_MONTHS - 1)

    ac = app_club.reindex(target_months)
    ad = app_date.reindex(target_months).fillna(-1)
    tc = tr_club.reindex(target_months)
    td = tr_date.reindex(target_months).fillna(-1)
    # Tra presenza valida (<=12m) e trasferimento valido (<=36m), prevale l'evento con data piu' recente
    # (a parita' di giorno prevale il trasferimento).
    use_tr = (td >= ad) & (td > 0)
    use_app = (ad > td) & (ad > 0)
    club_hist = pd.DataFrame(
        np.where(use_tr, tc, np.where(use_app, ac, np.nan)),
        index=target_months,
        columns=val_ff.columns,
    )

    # =========================================================================
    # PASSO 3: unione in formato lungo e assegnazione del campionato della stagione
    # =========================================================================
    print(">>> Assegnazione del campionato stagione per stagione (games.csv)...")
    long_club = club_hist.stack().rename("club_id").reset_index()
    long_club.columns = ["year_month", "player_id", "club_id"]
    long_val = val_ff.reindex(target_months).stack().rename("value").reset_index()
    long_val.columns = ["year_month", "player_id", "value"]

    rose = long_club.merge(long_val, on=["year_month", "player_id"], how="inner")
    rose["season"] = stagione_da_mese(rose["year_month"], override_stagione)
    rose = rose.merge(mappa_club_stagione, on=["season", "club_id"], how="left")
    rose = rose[rose["league"].isin(TOP5_CODES)].copy()

    agg_lega = rose.groupby(["year_month", "league"]).agg(value=("value", "sum"), players=("value", "size"))
    league_value_bn = (agg_lega["value"].unstack("league").reindex(target_months).fillna(0.0)) / 1e9
    league_players = agg_lega["players"].unstack("league").reindex(target_months).fillna(0).astype(int)

    top5_val_bn = league_value_bn.sum(axis=1)
    top5_players = league_players.sum(axis=1)
    top5_mean_m = (rose.groupby("year_month")["value"].mean().reindex(target_months)) / 1e6
    top5_median_m = (rose.groupby("year_month")["value"].median().reindex(target_months)) / 1e6

    # Numero effettivo medio di club nei Top-5 sui 156 mesi target (98 fino al 2022/23, 96 dal 2023/24)
    clubs_by_season = mappa_club_stagione.groupby("season")["club_id"].nunique()
    target_seasons = stagione_da_mese(target_months, override_stagione)
    avg_top5_clubs = float(pd.Series(target_seasons).map(clubs_by_season).mean())

    # Aggiornamenti di listino effettivi (non propagati) per i giocatori in rosa nel mese
    upd = val_m.reindex(target_months).stack().rename("raw").reset_index()
    upd.columns = ["year_month", "player_id", "raw"]
    upd = upd.merge(rose[["year_month", "player_id"]], on=["year_month", "player_id"], how="inner")
    updates_count_m = upd.groupby("year_month").size().reindex(target_months).fillna(0).astype(int)

    # =========================================================================
    # PASSO 4: confronto con l'aggregazione statica "naive" (club attuale del 2026)
    # =========================================================================
    player_static_comp = vals_sub.groupby("player_id")["player_club_domestic_competition_id"].last()
    static_it1 = player_static_comp[player_static_comp == "IT1"].index.intersection(val_ff.columns)
    naive_serie_a_bn = (val_ff[static_it1].sum(axis=1) / 1e9).reindex(target_months)

    # =========================================================================
    # PASSO 5: acquisti mensili (solo trasferimenti IN ENTRATA nei club del campionato)
    # =========================================================================
    print(">>> Aggregazione mensile degli acquisti (trasferimenti in entrata)...")
    tr_sub = tr[(tr["year_month"] >= VAL_WARMUP_START) & (tr["year_month"] <= TARGET_END)].copy()
    tr_sub["season"] = stagione_da_mese(tr_sub["year_month"], override_stagione)
    tr_sub = tr_sub.merge(
        mappa_club_stagione.rename(columns={"club_id": "to_club_id", "league": "to_league"}),
        on=["season", "to_club_id"], how="left",
    )
    tr_sub["fee"] = tr_sub["transfer_fee"].fillna(0.0).clip(lower=0.0)
    acq = tr_sub[tr_sub["to_league"].isin(TOP5_CODES)].copy()

    def aggrega_acquisti(df_acq):
        g = df_acq.groupby("year_month")
        out = pd.DataFrame({
            "spend_m": g["fee"].sum() / 1e6,
            "n_in": g.size(),
            "n_paid": g["fee"].apply(lambda x: int((x > 0).sum())),
        })
        return out.reindex(val_grid).fillna(0.0)

    acq_top5 = aggrega_acquisti(acq)
    acq_it1 = aggrega_acquisti(acq[acq["to_league"] == "IT1"])
    spesa_12m_top5_bn = acq_top5["spend_m"].rolling(12).sum() / 1000.0
    spesa_12m_it1_bn = acq_it1["spend_m"].rolling(12).sum() / 1000.0
    acq_top5, acq_it1 = acq_top5.reindex(target_months), acq_it1.reindex(target_months)

    # =========================================================================
    # PASSO 6: tabella master (market_monthly.csv) — zero colonne inutilizzate
    # =========================================================================
    df = pd.DataFrame(index=target_months)
    df.index.name = "year_month"
    df["date"] = df.index.to_timestamp()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df["top5_market_value_bn"] = top5_val_bn.round(4)
    df["serie_a_market_value_bn"] = league_value_bn["IT1"].round(4)
    df["premier_league_market_value_bn"] = league_value_bn["GB1"].round(4)
    df["laliga_market_value_bn"] = league_value_bn["ES1"].round(4)
    df["bundesliga_market_value_bn"] = league_value_bn["L1"].round(4)
    df["ligue1_market_value_bn"] = league_value_bn["FR1"].round(4)
    df["serie_a_naive_static_bn"] = naive_serie_a_bn.round(4)

    df["active_players_top5"] = top5_players.astype(int)
    for code in TOP5_CODES:
        df[f"active_players_{code}"] = league_players[code].astype(int)
    df["mean_player_value_m"] = top5_mean_m.round(3)
    df["median_player_value_m"] = top5_median_m.round(3)
    df["valuation_updates_count"] = updates_count_m

    df["transfer_spend_top5_m"] = acq_top5["spend_m"].round(2)
    df["transfer_spend_serie_a_m"] = acq_it1["spend_m"].round(2)
    df["transfer_spend_top5_12m_bn"] = spesa_12m_top5_bn.reindex(target_months).round(4)
    df["transfer_spend_serie_a_12m_bn"] = spesa_12m_it1_bn.reindex(target_months).round(4)
    df["n_incoming_top5"] = acq_top5["n_in"].astype(int)
    df["n_paid_incoming_top5"] = acq_top5["n_paid"].astype(int)
    df["n_incoming_serie_a"] = acq_it1["n_in"].astype(int)

    df["pct_change_top5"] = (df["top5_market_value_bn"].pct_change() * 100).round(3)
    df["pct_change_serie_a"] = (df["serie_a_market_value_bn"].pct_change() * 100).round(3)

    # Intervento deterministico a gradino (noto ex-ante, diventa impulso unitario dopo d=1)
    df["step_covid_apr2020"] = (df["date"] >= "2020-04-01").astype(int)

    df.reset_index().to_csv(os.path.join(PROC_DIR, "market_monthly.csv"), index=False)
    print(f"Salvato dataset master: market_monthly.csv ({len(df)} righe, {df.shape[1]} colonne)")

    long_rows = []
    for code, name in TOP5_NAMES.items():
        for ym in target_months:
            long_rows.append({
                "year_month": str(ym), "date": ym.to_timestamp(), "competition_id": code, "league_name": name,
                "market_value_bn": round(float(league_value_bn.loc[ym, code]), 4),
                "active_players": int(league_players.loc[ym, code]),
            })
    pd.DataFrame(long_rows).to_csv(os.path.join(PROC_DIR, "leagues_monthly.csv"), index=False)

    # =========================================================================
    # PASSO 7: statistiche di controllo dell'ETL (etl_summary.json)
    # =========================================================================
    stagioni_acq = acq.assign(season=acq["season"]).groupby("season")["fee"].sum() / 1e9
    giugni = df[df["month"] == 6]
    profilo_spesa = df.groupby("month")["transfer_spend_top5_m"].mean()
    etl_summary = {
        "raw_counts": {k: int(v) for k, v in raw_counts.items()},
        "parametri": {
            "finestra_presenze_mesi": APP_ACTIVITY_MONTHS,
            "finestra_trasferimenti_mesi": TR_ACTIVITY_MONTHS,
            "ffill_valutazioni_mesi": VALUATION_FFILL,
            "inizio_eventi_presenze": EVENTS_START,
            "rientri_30giu_spostati_1lug": n_j30_shifted,
            "override_stagione": {str(k): int(v) for k, v in override_stagione.items()},
        },
        "time_horizon": {
            "start_month": TARGET_START,
            "end_month": TARGET_END,
            "total_months": int(len(df)),
            "missing_values_in_master": int(df.drop(columns=["pct_change_top5", "pct_change_serie_a"]).isna().sum().sum()),
        },
        "copertura": {
            "giocatori_attivi_top5_min": int(df["active_players_top5"].min()),
            "giocatori_attivi_top5_max": int(df["active_players_top5"].max()),
            "giocatori_attivi_top5_medi": round(float(df["active_players_top5"].mean()), 1),
            "club_top5_medi_periodo": round(avg_top5_clubs, 2),
            "giocatori_per_club_medi": round(float(df["active_players_top5"].mean() / avg_top5_clubs), 1),
            "giocatori_attivi_serie_a_medi": round(float(df["active_players_IT1"].mean()), 1),
            "giocatori_per_club_serie_a_medi": round(float(df["active_players_IT1"].mean() / 20.0), 1),
            "giocatori_giugno": {str(int(r.year)): int(r.active_players_top5) for r in giugni.itertuples()},
        },
        "spesa_media_mensile_top5_m": {
            "gennaio": round(float(profilo_spesa.loc[1]), 0),
            "luglio": round(float(profilo_spesa.loc[7]), 0),
            "agosto": round(float(profilo_spesa.loc[8]), 0),
            "settembre": round(float(profilo_spesa.loc[9]), 0),
        },
        "quality_checks": {
            "zero_change_months_top5": int((df["pct_change_top5"].iloc[1:] == 0).sum()),
            "top5_value_start_bn": round(float(df["top5_market_value_bn"].iloc[0]), 2),
            "top5_value_end_bn": round(float(df["top5_market_value_bn"].iloc[-1]), 2),
            "top5_value_max_bn": round(float(df["top5_market_value_bn"].max()), 2),
            "top5_value_max_month": str(df["top5_market_value_bn"].idxmax()),
            "top5_covid_drop_apr2020_pct": round(float(df.loc[pd.Period("2020-04", freq="M"), "pct_change_top5"]), 2),
            "serie_a_value_start_bn": round(float(df["serie_a_market_value_bn"].iloc[0]), 2),
            "serie_a_value_end_bn": round(float(df["serie_a_market_value_bn"].iloc[-1]), 2),
            "serie_a_covid_drop_apr2020_pct": round(float(df.loc[pd.Period("2020-04", freq="M"), "pct_change_serie_a"]), 2),
            "spesa_acquisti_top5_totale_bn": round(float(df["transfer_spend_top5_m"].sum() / 1000.0), 2),
            "spesa_acquisti_serie_a_totale_bn": round(float(df["transfer_spend_serie_a_m"].sum() / 1000.0), 2),
            "acquisti_top5_totali": int(df["n_incoming_top5"].sum()),
            "acquisti_onerosi_top5_totali": int(df["n_paid_incoming_top5"].sum()),
            "spesa_acquisti_top5_per_stagione_bn": {str(int(k)): round(float(v), 2) for k, v in stagioni_acq.items() if 2012 <= k <= 2025},
            "gap_statico_vs_dinamico_serie_a_max_bn": round(float((df["serie_a_market_value_bn"] - df["serie_a_naive_static_bn"]).max()), 2),
            "gap_statico_vs_dinamico_serie_a_max_month": str((df["serie_a_market_value_bn"] - df["serie_a_naive_static_bn"]).idxmax()),
        },
    }
    with open(os.path.join(PROC_DIR, "etl_summary.json"), "w", encoding="utf-8") as f:
        json.dump(etl_summary, f, indent=2, ensure_ascii=False)
    print("Salvato sommario ETL: etl_summary.json")

    genera_figure_etl(df)
    print(">>> [FASE 1 - ETL] Completata con successo!")


def salva(fig, nome):
    for out_dir in [FIG_DIR, CAP6_IMG_DIR]:
        fig.savefig(os.path.join(out_dir, nome), dpi=600, bbox_inches="tight")
    plt.close(fig)


def genera_figure_etl(df):
    print(">>> Generazione grafici diagnostici per la Sezione 1 / Capitolo 6 (ETL)...")
    month_labels = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]
    profilo = df.groupby("month").agg(
        avg_updates=("valuation_updates_count", "mean"),
        avg_transfers=("n_incoming_top5", "mean"),
        avg_spend_m=("transfer_spend_top5_m", "mean"),
    )

    # FIGURA 1: complementarita' stagionale tra aggiornamenti di listino e acquisti
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(1, 13)
    w = 0.38
    ax1 = axes[0]
    ax1t = ax1.twinx()
    b1 = ax1.bar(x - w / 2, profilo["avg_updates"], width=w, color="#2b5c8f", alpha=0.88, label="Aggiornamenti di listino (player_valuations)")
    b2 = ax1t.bar(x + w / 2, profilo["avg_transfers"], width=w, color="#d95f02", alpha=0.88, label="Trasferimenti in entrata (transfers)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(month_labels, fontsize=10)
    ax1.set_ylabel("Numero medio aggiornamenti di prezzo", color="#2b5c8f", fontweight="bold", fontsize=11)
    ax1t.set_ylabel("Numero medio trasferimenti in entrata", color="#d95f02", fontweight="bold", fontsize=11)
    ax1.set_title("a) Complementarità mensile: listini vs sessioni di mercato", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylim(0, profilo["avg_updates"].max() * 1.24)
    ax1t.set_ylim(0, profilo["avg_transfers"].max() * 1.24)
    ax1t.grid(False)
    ax1.legend([b1, b2], [b1.get_label(), b2.get_label()], loc="upper left", frameon=True, fontsize=9.5)

    ax2 = axes[1]
    colori = ["#e7298a" if m in [1, 7, 8] else "#66a61e" if m in [2, 6, 9] else "#8da0cb" for m in range(1, 13)]
    barre = ax2.bar(x, profilo["avg_spend_m"], color=colori, alpha=0.9, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(month_labels, fontsize=10)
    ax2.set_ylabel("Spesa media mensile per acquisti (Milioni €)", fontweight="bold", fontsize=11)
    ax2.set_title("b) Distribuzione mensile della spesa per acquisti", fontsize=12, fontweight="bold", pad=10)
    for b in barre:
        h = b.get_height()
        if h > 80:
            ax2.text(b.get_x() + b.get_width() / 2, h + 35, f"{h:.0f}M€", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0, profilo["avg_spend_m"].max() * 1.16)
    plt.suptitle("Integrazione tra aggiornamenti di valutazione e finestre di calciomercato (2013–2026)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    salva(fig, "fig01_etl_complementarita_fonti.png")

    # FIGURA 2: ricostruzione dinamica vs statica (Serie A) e controllo di copertura delle rose
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.4), sharex=True)
    ax = axes[0]
    ax.plot(df["date"], df["serie_a_market_value_bn"], color="#1b9e77", linewidth=2.3,
            label="Serie A – ricostruzione dinamica (club e campionato stagione per stagione)")
    ax.plot(df["date"], df["serie_a_naive_static_bn"], color="#d95f02", linestyle="--", linewidth=1.8, alpha=0.85,
            label="Serie A – aggregazione statica (club attuale del 2026)")
    ax.set_ylabel("Valore di mercato (Miliardi €)", fontweight="bold", fontsize=11)
    ax.set_title("a) Effetto della ricostruzione storica sul valore della Serie A (2013–2026)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    ax = axes[1]
    colori_lega = {"GB1": "#1f77b4", "ES1": "#ff7f0e", "IT1": "#2ca02c", "L1": "#d62728", "FR1": "#9467bd"}
    for code, col in colori_lega.items():
        ax.plot(df["date"], df[f"active_players_{code}"], color=col, linewidth=1.9, label=TOP5_NAMES[code])
    ax.set_ylabel("Calciatori conteggiati nelle rose", fontweight="bold", fontsize=11)
    ax.set_xlabel("Data (frequenza mensile regolare: 156 mesi consecutivi, Giugno 2013 – Maggio 2026)", fontweight="bold", fontsize=11)
    ax.set_title("b) Controllo di copertura: numero di calciatori attivi conteggiati per campionato", fontsize=12, fontweight="bold")
    ax.set_ylim(0, df[[f"active_players_{c}" for c in TOP5_CODES]].values.max() * 1.25)
    ax.legend(loc="lower right", ncol=5, frameon=True, fontsize=9.5)
    plt.tight_layout()
    salva(fig, "fig02_etl_ricostruzione_storica_confronto.png")


if __name__ == "__main__":
    run_etl()
