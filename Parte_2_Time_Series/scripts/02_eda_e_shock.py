
"""
========================================================================================
Progetto di Data Science - Parte 2: Analisi delle Serie Storiche (Transfermarkt)
Corso di Laurea Magistrale in Ingegneria Informatica e dell'Automazione (UnivPM)
Fase 2 (Capitolo 7): Analisi Esplorativa (EDA), Shock Storici e Correlazioni
========================================================================================
Questo script genera le 9 figure del Capitolo 7 ad alta risoluzione (600 DPI) sia in
Parte_2_Time_Series/figures/ che in Relazione/capitolo7/img/, calcolando dinamicamente
tutte le annotazioni e le statistiche di sintesi (salvate in eda_shock_summary.json):
- fig03_eda_valore_campionati.png
- fig04_eda_quote_mercato.png
- fig05_eda_shock_neymar_2017.png
- fig06_eda_shock_covid_2020.png
- fig07_eda_shock_boom_2026.png
- fig08_eda_change_pct_volatilita.png
- fig09_eda_distribuzione_kde.png
- fig10_eda_spese_calciomercato_impulsi.png
- fig11_eda_matrice_correlazione.png
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

from src.config import DATA_PROCESSED_DIR as PROC_DIR, FIGURES_DIR as FIG_DIR, CAP7_IMG_DIR

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 600


def save_fig(fig, name):
    for out_dir in [FIG_DIR, CAP7_IMG_DIR]:
        fig.savefig(os.path.join(out_dir, name), dpi=600, bbox_inches="tight")
    plt.close(fig)


def run_eda_single_figures():
    print(">>> [FASE 2 - EDA & SHOCK] Caricamento market_monthly.csv...")
    df = pd.read_csv(os.path.join(PROC_DIR, "market_monthly.csv"))
    df["date"] = pd.to_datetime(df["date"])

    league_cols = {
        "Premier League": ("premier_league_market_value_bn", "#1f77b4"),
        "LaLiga": ("laliga_market_value_bn", "#ff7f0e"),
        "Serie A": ("serie_a_market_value_bn", "#2ca02c"),
        "Bundesliga": ("bundesliga_market_value_bn", "#d62728"),
        "Ligue 1": ("ligue1_market_value_bn", "#9467bd"),
    }

    covid_drop_pct = float(df.loc[df["year_month"] == "2020-04", "pct_change_top5"].iloc[0])
    covid_drop_bn = float(
        df.loc[df["year_month"] == "2020-04", "top5_market_value_bn"].iloc[0]
        - df.loc[df["year_month"] == "2020-03", "top5_market_value_bn"].iloc[0]
    )
    peak_precovid = float(df.loc[df["year_month"] == "2020-03", "top5_market_value_bn"].iloc[0])
    idx_precovid = int(df.index[df["year_month"] == "2020-03"][0])
    rec_rows = df[(df["date"] > "2020-04-01") & (df["top5_market_value_bn"] >= peak_precovid)]
    recovery_month = str(rec_rows.iloc[0]["year_month"]) if len(rec_rows) else "non raggiunto"
    recovery_months_n = int(rec_rows.index[0] - idx_precovid) if len(rec_rows) else None

    val_max_top5 = float(df["top5_market_value_bn"].max())
    val_max_month = str(df.loc[df["top5_market_value_bn"].idxmax(), "year_month"])

    # -------------------------------------------------------------------------
    # FIGURA 3: Valore di mercato assoluto dei 5 campionati
    # -------------------------------------------------------------------------
    print(">>> Generazione fig03_eda_valore_campionati.png...")
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for name, (col, color) in league_cols.items():
        lw = 3.0 if name in ["Premier League", "Serie A"] else 2.2
        alpha = 1.0 if name in ["Premier League", "Serie A"] else 0.85
        ax.plot(df["date"], df[col], label=name, color=color, linewidth=lw, alpha=alpha)

    ymax_l = max(df[col].max() for col, _ in league_cols.values())
    ax.axvline(pd.Timestamp("2017-08-01"), color="#e41a1c", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(pd.Timestamp("2017-09-01"), ymax_l * 0.86, "Accelerazione\n2017–2019\n(Ago 2017)", color="#b30000", fontsize=9.5, fontweight="bold")

    ax.axvline(pd.Timestamp("2020-04-01"), color="#333333", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(pd.Timestamp("2020-05-15"), ymax_l * 0.90, f"Shock COVID-19\n(Apr 2020: {covid_drop_pct:.1f}%)", color="#222222", fontsize=9.8, fontweight="bold")

    ax.axvline(pd.Timestamp("2023-07-01"), color="#ff7f00", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(pd.Timestamp("2023-08-15"), ymax_l * 0.70, "Espansione 2023–2026\ne nuovi ricavi UEFA", color="#b35806", fontsize=9.5, fontweight="bold")

    ax.set_ylabel("Valore di mercato (Miliardi EUR)", fontweight="bold", fontsize=12)
    ax.set_xlabel(f"Data (Giugno 2013 – Maggio 2026, {len(df)} mesi)", fontweight="bold", fontsize=12)
    ax.set_title("Evoluzione storica del valore di mercato delle rose nei cinque maggiori campionati europei", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", ncol=3, frameon=True, fontsize=10.5)
    ax.set_ylim(0.8, ymax_l * 1.15)
    save_fig(fig, "fig03_eda_valore_campionati.png")

    # -------------------------------------------------------------------------
    # FIGURA 4: Quote percentuali (%) su area stackplot
    # -------------------------------------------------------------------------
    print(">>> Generazione fig04_eda_quote_mercato.png...")
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    shares = pd.DataFrame(index=df.index)
    for name, (col, _) in league_cols.items():
        shares[name] = (df[col] / df["top5_market_value_bn"]) * 100.0

    ax.stackplot(
        df["date"],
        [shares[name] for name in league_cols.keys()],
        labels=list(league_cols.keys()),
        colors=[c for _, c in league_cols.values()],
        alpha=0.85,
    )
    ax.set_ylabel("Quota sul totale Top-5 (%)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Data (Giugno 2013 – Maggio 2026)", fontweight="bold", fontsize=12)
    ax.set_title("Ripartizione percentuale delle quote di mercato tra le cinque leghe europee", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncol=5, frameon=True, fontsize=10)

    pl_start, pl_end = float(shares["Premier League"].iloc[0]), float(shares["Premier League"].iloc[-1])
    sa_start, sa_end = float(shares["Serie A"].iloc[0]), float(shares["Serie A"].iloc[-1])
    y_sa_start = pl_start + float(shares["LaLiga"].iloc[0]) + sa_start / 2.0
    y_sa_end = pl_end + float(shares["LaLiga"].iloc[-1]) + sa_end / 2.0

    ax.text(pd.Timestamp("2013-11-01"), pl_start / 2.0, f"Premier: {pl_start:.1f}%", color="white", fontweight="bold", fontsize=10)
    ax.text(pd.Timestamp("2024-03-01"), pl_end / 2.0, f"Premier: {pl_end:.1f}%", color="white", fontweight="bold", fontsize=10)
    ax.text(pd.Timestamp("2013-11-01"), y_sa_start, f"Serie A: {sa_start:.1f}%", color="white", fontweight="bold", fontsize=10)
    ax.text(pd.Timestamp("2024-03-01"), y_sa_end, f"Serie A: {sa_end:.1f}%", color="white", fontweight="bold", fontsize=10)
    save_fig(fig, "fig04_eda_quote_mercato.png")

    # -------------------------------------------------------------------------
    # FIGURA 5: Zoom Shock 1 - Accelerazione 2017-2019 (confronto descrittivo pendenze)
    # -------------------------------------------------------------------------
    print(">>> Generazione fig05_eda_shock_neymar_2017.png...")
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    df_s1 = df[(df["date"] >= "2013-06-01") & (df["date"] <= "2020-02-01")].copy()
    pre_2017 = df_s1[df_s1["date"] <= "2017-07-01"]
    post_2017 = df_s1[df_s1["date"] >= "2017-08-01"]

    ax.plot(df_s1["date"], df_s1["top5_market_value_bn"], color="#2b5c8f", linewidth=3.0, label="Valore aggregato Top-5 reale (Mld EUR)")

    t_pre = np.arange(len(pre_2017))
    slope_pre, intercept_pre = np.polyfit(t_pre, pre_2017["top5_market_value_bn"], 1)
    t_all = np.arange(len(df_s1))
    ax.plot(
        df_s1["date"],
        intercept_pre + slope_pre * t_all,
        color="#555555",
        linestyle=":",
        linewidth=2.4,
        label=f"Trend lineare 2013–2017 proiettato (+{slope_pre*12:.2f} Mld EUR/anno)",
    )

    t_post = np.arange(len(post_2017))
    slope_post, intercept_post = np.polyfit(t_post, post_2017["top5_market_value_bn"], 1)
    ratio_slope = (slope_post / slope_pre) if slope_pre != 0 else np.nan
    ax.plot(
        post_2017["date"],
        intercept_post + slope_post * t_post,
        color="#e41a1c",
        linestyle="--",
        linewidth=2.6,
        label=f"Trend lineare 2017–2020 (+{slope_post*12:.2f} Mld EUR/anno, {ratio_slope:.1f}×)",
    )

    ax.axvline(pd.Timestamp("2017-08-01"), color="#e41a1c", linestyle="-.", linewidth=1.6, alpha=0.8)
    ax.text(pd.Timestamp("2017-09-01"), 13.5, "Estate 2017: trasferimento Neymar\n(222 Mln EUR) e inflazione listini", color="#b30000", fontsize=10, fontweight="bold")

    ax.set_title("Fase 1: accelerazione descrittiva della pendenza tra l'estate 2017 e l'inizio del 2020", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Valore rose Top-5 (Miliardi EUR)", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Anno", fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    save_fig(fig, "fig05_eda_shock_neymar_2017.png")

    # -------------------------------------------------------------------------
    # FIGURA 6: Zoom Shock 2 - Primavera 2020 (COVID-19) fino al recupero completo
    # -------------------------------------------------------------------------
    print(">>> Generazione fig06_eda_shock_covid_2020.png...")
    fig, ax = plt.subplots(figsize=(11.2, 5.5))
    df_s2 = df[df["date"] >= "2019-06-01"].copy()
    l1 = ax.plot(df_s2["date"], df_s2["top5_market_value_bn"], color="#2b5c8f", linewidth=2.6, marker="o", markersize=3.5, label="Valore Top-5 (Mld EUR)")
    ax_twin = ax.twinx()
    l2 = ax_twin.plot(df_s2["date"], df_s2["serie_a_market_value_bn"], color="#2ca02c", linewidth=2.2, linestyle="--", marker="s", markersize=3.2, label="Valore Serie A (Mld EUR)")
    ax_twin.grid(False)

    l3 = [ax.axhline(peak_precovid, color="#555555", linestyle=":", linewidth=1.6, label=f"Picco pre-COVID (Mar 2020: {peak_precovid:.2f} Mld EUR)")]
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01"), color="#e41a1c", alpha=0.15)
    ax.annotate(
        f"Taglio COVID-19\n(Apr 2020: {covid_drop_pct:.2f}%)",
        xy=(pd.Timestamp("2020-04-01"), float(df.loc[df["year_month"] == "2020-04", "top5_market_value_bn"].iloc[0])),
        xytext=(pd.Timestamp("2020-11-01"), 23.3),
        arrowprops=dict(arrowstyle="->", color="#b30000", lw=1.4),
        color="#b30000", fontweight="bold", fontsize=9.8,
    )

    if len(rec_rows):
        rec_date = rec_rows.iloc[0]["date"]
        rec_val = float(rec_rows.iloc[0]["top5_market_value_bn"])
        ax.scatter([rec_date], [rec_val], color="#d95f02", s=70, zorder=6)
        ax.annotate(
            f"Recupero picco\n({recovery_month}: {recovery_months_n} mesi)",
            xy=(rec_date, rec_val),
            xytext=(pd.Timestamp("2023-03-01"), 32.0),
            arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.5),
            fontsize=9.5,
            fontweight="bold",
            color="#b34800",
        )

    ax.set_title("Fase 2: contrazione pandemica da COVID-19 (Aprile 2020) e recupero completo", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Valore Top-5 (Miliardi EUR)", color="#2b5c8f", fontweight="bold", fontsize=11.5)
    ax_twin.set_ylabel("Valore Serie A (Miliardi EUR)", color="#2ca02c", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Anno", fontweight="bold", fontsize=11.5)
    ax.set_ylim(22.5, 34.5)
    ax_twin.set_ylim(4.1, 6.2)

    lines = l1 + l2 + l3
    ax.legend(lines, [line.get_label() for line in lines], loc="lower right", frameon=True, fontsize=9.5)
    save_fig(fig, "fig06_eda_shock_covid_2020.png")

    # -------------------------------------------------------------------------
    # FIGURA 7: Zoom Fase 3 - Espansione 2022-2026 e valore medio/mediano per giocatore
    # -------------------------------------------------------------------------
    print(">>> Generazione fig07_eda_shock_boom_2026.png...")
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    df_s3 = df[df["date"] >= "2021-06-01"].copy()
    l1 = ax.plot(df_s3["date"], df_s3["top5_market_value_bn"], color="#2b5c8f", linewidth=2.8, label="Valore totale Top-5 (Mld EUR)")
    ax_twin = ax.twinx()
    l2 = ax_twin.plot(df_s3["date"], df_s3["mean_player_value_m"], color="#d95f02", linewidth=2.4, linestyle="-.", label="Valore medio per calciatore (Mln EUR)")
    l3 = ax_twin.plot(df_s3["date"], df_s3["median_player_value_m"], color="#66a61e", linewidth=2.0, linestyle=":", label="Valore mediano per calciatore (Mln EUR)")
    ax_twin.grid(False)

    ax.axvline(pd.Timestamp("2023-07-01"), color="#d95f02", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(pd.Timestamp("2023-08-15"), 25.1, "Estate 2023: mercato saudita\ne nuova UEFA Champions League", color="#b35806", fontweight="bold", fontsize=9.4)

    ax.set_title(f"Fase 3: espansione 2023–2026, massimo a {val_max_top5:.2f} Mld e dinamica medio/mediano per calciatore", fontsize=12.2, fontweight="bold", pad=12)
    ax.set_ylabel("Valore Top-5 (Miliardi EUR)", color="#2b5c8f", fontweight="bold", fontsize=11.5)
    ax_twin.set_ylabel("Valore per calciatore attivo (Milioni EUR)", color="#d95f02", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Anno", fontweight="bold", fontsize=11.5)
    ax.set_ylim(24.5, 34.8)
    ax_twin.set_ylim(2.0, 14.5)

    lines = l1 + l2 + l3
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left", frameon=True, fontsize=9.5)
    save_fig(fig, "fig07_eda_shock_boom_2026.png")

    # -------------------------------------------------------------------------
    # FIGURA 8: Variazioni percentuali mensili (Change %) con soglie classiche e robuste (MAD)
    # -------------------------------------------------------------------------
    print(">>> Generazione fig08_eda_change_pct_volatilita.png...")
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    pct_top5 = df["pct_change_top5"].iloc[1:]
    pct_it1 = df["pct_change_serie_a"].iloc[1:]
    dates_pct = df["date"].iloc[1:]

    mu_pct = float(pct_top5.mean())
    sd_pct = float(pct_top5.std())
    med_pct = float(pct_top5.median())
    mad_n = float(1.4826 * np.median(np.abs(pct_top5 - med_pct)))
    z_covid_classic = (covid_drop_pct - mu_pct) / sd_pct
    z_covid_robust = (covid_drop_pct - med_pct) / mad_n

    ax.axhline(0, color="black", linewidth=1.0)
    ax.axhline(mu_pct + 2 * sd_pct, color="#d95f02", linestyle="--", linewidth=1.3, label=f"Soglia μ±2σ ({mu_pct - 2*sd_pct:.2f}% / +{mu_pct + 2*sd_pct:.2f}%)")
    ax.axhline(mu_pct - 2 * sd_pct, color="#d95f02", linestyle="--", linewidth=1.3)

    ax.plot(dates_pct, pct_top5, color="#2b5c8f", linewidth=1.9, label="Variazione % mensile Top-5")
    ax.plot(dates_pct, pct_it1, color="#2ca02c", linewidth=1.2, alpha=0.65, linestyle=":", label="Variazione % mensile Serie A")

    outliers = df.iloc[1:][(pct_top5 > mu_pct + 2 * sd_pct) | (pct_top5 < mu_pct - 2 * sd_pct)]
    ax.scatter(outliers["date"], outliers["pct_change_top5"], color="#e41a1c", s=48, zorder=5, label=f"Mesi fuori μ±2σ ({len(outliers)} mesi)")

    max_pos_row = df.iloc[1:].loc[pct_top5.idxmax()]
    ax.annotate(
        f"Crollo COVID-19\n(Apr 2020: {covid_drop_pct:.2f}%)",
        xy=(pd.Timestamp("2020-04-01"), covid_drop_pct),
        xytext=(pd.Timestamp("2016-06-01"), -13.5),
        arrowprops=dict(arrowstyle="->", color="#e41a1c", lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#b30000",
    )
    ax.annotate(
        f"Picco rivalutazioni\n({max_pos_row['year_month']}: +{max_pos_row['pct_change_top5']:.2f}%)",
        xy=(max_pos_row["date"], max_pos_row["pct_change_top5"]),
        xytext=(pd.Timestamp("2019-06-01"), 9.8),
        arrowprops=dict(arrowstyle="->", color="#2b5c8f", lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#1b3b5c",
    )

    ax.set_title("Variazioni percentuali mensili e individuazione degli outlier (soglia μ±2σ)", fontsize=12.3, fontweight="bold", pad=12)
    ax.set_ylabel("Variazione percentuale mensile (%)", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Data (2013–2026)", fontweight="bold", fontsize=11.5)
    ax.legend(loc="lower right", ncol=1, frameon=True, fontsize=8.5)
    ax.set_ylim(-21.5, 14.0)
    save_fig(fig, "fig08_eda_change_pct_volatilita.png")

    # -------------------------------------------------------------------------
    # FIGURA 9: Distribuzione empirica di densità KDE
    # -------------------------------------------------------------------------
    print(">>> Generazione fig09_eda_distribuzione_kde.png...")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.histplot(pct_top5, bins=32, kde=True, color="#2b5c8f", stat="density", alpha=0.45, ax=ax, label="Top-5 campionati (istogramma + KDE)")
    sns.kdeplot(pct_it1, color="#2ca02c", linewidth=2.2, linestyle="--", ax=ax, label="Serie A (IT1)")
    ax.axvline(mu_pct, color="#e41a1c", linestyle="-", linewidth=1.6, label=f"Media Top-5: {mu_pct:+.2f}%/mese")
    ax.axvline(med_pct, color="#7570b3", linestyle=":", linewidth=1.6, label=f"Mediana Top-5: {med_pct:+.2f}%/mese")

    ax.set_title("Distribuzione empirica di densità (KDE) delle variazioni percentuali mensili", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Variazione percentuale mensile (%)", fontweight="bold", fontsize=11.5)
    ax.set_ylabel("Densità empirica", fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9.8)
    save_fig(fig, "fig09_eda_distribuzione_kde.png")

    # -------------------------------------------------------------------------
    # FIGURA 10: Impulsi stagionali spese acquisti calciomercato vs valore rose
    # -------------------------------------------------------------------------
    print(">>> Generazione fig10_eda_spese_calciomercato_impulsi.png...")
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax_twin = ax.twinx()

    ax.bar(
        df["date"],
        df["transfer_spend_top5_m"] / 1000.0,
        width=24,
        color="#d95f02",
        alpha=0.62,
        label="Spesa mensile per acquisti Top-5 (Mld EUR)",
    )
    ax_twin.plot(
        df["date"],
        df["top5_market_value_bn"],
        color="#2b5c8f",
        linewidth=2.8,
        label="Valore di mercato rose Top-5 (Mld EUR)",
    )
    ax_twin.grid(False)

    ax.set_title("Spesa mensile per acquisti nelle finestre di mercato e valore delle rose Top-5", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Spesa mensile acquisti (Miliardi EUR)", color="#d95f02", fontweight="bold", fontsize=11.5)
    ax_twin.set_ylabel("Valore rose Top-5 (Miliardi EUR)", color="#2b5c8f", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Data (2013–2026)", fontweight="bold", fontsize=11.5)
    ax.set_ylim(0, (df["transfer_spend_top5_m"].max() / 1000.0) * 1.25)
    ax_twin.set_ylim(9.0, 36.0)

    h1, lb1 = ax.get_legend_handles_labels()
    h2, lb2 = ax_twin.get_legend_handles_labels()
    ax.legend(h1 + h2, lb1 + lb2, loc="upper left", frameon=True, fontsize=10)
    save_fig(fig, "fig10_eda_spese_calciomercato_impulsi.png")

    # -------------------------------------------------------------------------
    # FIGURA 11: Confronto correlazione di Pearson in LIVELLI vs DIFFERENZE PRIME
    # -------------------------------------------------------------------------
    print(">>> Generazione fig11_eda_matrice_correlazione.png...")
    df_lvl = pd.DataFrame({
        "Valore Top-5": df["top5_market_value_bn"],
        "Valore Serie A": df["serie_a_market_value_bn"],
        "Valore Premier": df["premier_league_market_value_bn"],
        "Valore medio gioc.": df["mean_player_value_m"],
        "Spesa mensile": df["transfer_spend_top5_m"],
        "Acquisti 12m": df["transfer_spend_top5_12m_bn"],
        "Acquisti 12m IT1": df["transfer_spend_serie_a_12m_bn"],
    })
    corr_lvl = df_lvl.corr()
    corr_diff = df_lvl.diff().dropna().corr()

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    sns.heatmap(
        corr_lvl, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1.0, vmax=1.0, center=0.0,
        cbar=True, square=True, ax=axes[0], annot_kws={"size": 9.2, "weight": "bold"},
    )
    axes[0].set_title("a) Correlazione sui livelli\n(influenzata dal trend comune)", fontsize=11.0, fontweight="bold", pad=10)
    axes[0].tick_params(axis="x", rotation=35, labelsize=9.2)
    axes[0].tick_params(axis="y", rotation=0, labelsize=9.2)

    sns.heatmap(
        corr_diff, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1.0, vmax=1.0, center=0.0,
        cbar=True, square=True, ax=axes[1], annot_kws={"size": 9.2, "weight": "bold"},
    )
    axes[1].set_title("b) Correlazione sulle differenze prime Δ\n(senza trend comune)", fontsize=11.0, fontweight="bold", pad=10)
    axes[1].tick_params(axis="x", rotation=35, labelsize=9.2)
    axes[1].tick_params(axis="y", rotation=0, labelsize=9.2)

    plt.suptitle("Verifica della correlazione spuria: livelli vs differenze prime mensili (2013–2026)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig11_eda_matrice_correlazione.png")

    # Tabella date chiave per il Capitolo 7 (percentuali calcolate sui valori esatti non arrotondati)
    key_dates = ["2013-06", "2017-07", "2020-03", "2020-04", "2026-05"]
    tabella_leghe = {}
    for name, (col, _) in list(league_cols.items()) + [("Totale Top-5", ("top5_market_value_bn", ""))]:
        raw_k = {ym: float(df.loc[df["year_month"] == ym, col].iloc[0]) for ym in key_dates}
        vals_k = {ym: round(v, 2) for ym, v in raw_k.items()}
        d_covid = (raw_k["2020-04"] - raw_k["2020-03"]) / raw_k["2020-03"] * 100.0
        d_tot = (raw_k["2026-05"] - raw_k["2013-06"]) / raw_k["2013-06"] * 100.0
        vals_k["delta_covid_pct"] = round(float(d_covid), 2)
        vals_k["delta_totale_pct"] = round(float(d_tot), 1)
        vals_k["max_bn"] = round(float(df[col].max()), 2)
        vals_k["max_month"] = str(df.loc[df[col].idxmax(), "year_month"])
        tabella_leghe[name] = vals_k

    # Recupero sostenuto: primo mese dopo il quale la serie non scende più sotto il picco pre-COVID
    sustained_rec = None
    for idx_r in range(idx_precovid + 2, len(df)):
        if (df["top5_market_value_bn"].iloc[idx_r:] >= peak_precovid).all():
            sustained_rec = {
                "month": str(df["year_month"].iloc[idx_r]),
                "months_from_mar2020": int(idx_r - idx_precovid),
                "value_bn": round(float(df["top5_market_value_bn"].iloc[idx_r]), 2),
            }
            break

    eda_summary = {
        "n_months": int(len(df)),
        "horizon": f"{df['year_month'].iloc[0]} .. {df['year_month'].iloc[-1]}",
        "tabella_leghe": tabella_leghe,
        "quote_mercato_pct": {
            "premier_start": round(pl_start, 1),
            "premier_end": round(pl_end, 1),
            "serie_a_start": round(sa_start, 1),
            "serie_a_end": round(sa_end, 1),
        },
        "slope_pre2017_bn_per_year": round(float(slope_pre * 12), 2),
        "slope_post2017_bn_per_year": round(float(slope_post * 12), 2),
        "slope_ratio": round(float(ratio_slope), 2),
        "covid_drop_top5_pct": round(covid_drop_pct, 2),
        "covid_drop_top5_bn": round(covid_drop_bn, 2),
        "covid_recovery_month": recovery_month,
        "covid_recovery_months": recovery_months_n,
        "covid_recovery_value_bn": round(float(rec_rows.iloc[0]["top5_market_value_bn"]), 2) if len(rec_rows) else None,
        "covid_recovery_sustained": sustained_rec,
        "mean_player_value_start_m": round(float(df["mean_player_value_m"].iloc[0]), 2),
        "mean_player_value_end_m": round(float(df["mean_player_value_m"].iloc[-1]), 2),
        "mean_player_value_growth_pct": round(
            float((df["mean_player_value_m"].iloc[-1] - df["mean_player_value_m"].iloc[0]) / df["mean_player_value_m"].iloc[0] * 100.0), 1
        ),
        "median_player_value_start_m": round(float(df["median_player_value_m"].iloc[0]), 2),
        "median_player_value_end_m": round(float(df["median_player_value_m"].iloc[-1]), 2),
        "pct_change_mean_top5": round(mu_pct, 3),
        "pct_change_std_top5": round(sd_pct, 3),
        "pct_change_median_top5": round(med_pct, 3),
        "pct_change_mad_norm_top5": round(mad_n, 3),
        "pct_change_std_serie_a": round(float(pct_it1.std()), 3),
        "pct_change_skew_top5": round(float(pct_top5.skew()), 2),
        "pct_change_excess_kurt_top5": round(float(pct_top5.kurtosis()), 2),
        "z_covid_classic": round(float(z_covid_classic), 2),
        "z_covid_robust_mad": round(float(z_covid_robust), 2),
        "max_pos_pct_month": str(max_pos_row["year_month"]),
        "max_pos_pct_value": round(float(max_pos_row["pct_change_top5"]), 2),
        "n_outliers_2sigma": int(len(outliers)),
        "outliers_2sigma": [
            {"year_month": str(r.year_month), "pct_change": round(float(r.pct_change_top5), 2)}
            for r in outliers.itertuples()
        ],
        "corr_levels_top5_spend12m": round(float(corr_lvl.loc["Valore Top-5", "Acquisti 12m"]), 2),
        "corr_levels_top5_spend_monthly": round(float(corr_lvl.loc["Valore Top-5", "Spesa mensile"]), 2),
        "corr_levels_premier_spend12m": round(float(corr_lvl.loc["Valore Premier", "Acquisti 12m"]), 2),
        "corr_levels_serie_a_spend12m_it1": round(float(corr_lvl.loc["Valore Serie A", "Acquisti 12m IT1"]), 2),
        "corr_diff_top5_spend12m": round(float(corr_diff.loc["Valore Top-5", "Acquisti 12m"]), 2),
        "corr_diff_serie_a_spend12m_it1": round(float(corr_diff.loc["Valore Serie A", "Acquisti 12m IT1"]), 2),
        "max_monthly_spend_top5_m": round(float(df["transfer_spend_top5_m"].max()), 2),
        "max_monthly_spend_month": str(df.loc[df["transfer_spend_top5_m"].idxmax(), "year_month"]),
        "max_monthly_spend_serie_a_m": round(float(df["transfer_spend_serie_a_m"].max()), 2),
        "max_monthly_spend_serie_a_month": str(df.loc[df["transfer_spend_serie_a_m"].idxmax(), "year_month"]),
        "total_spend_top5_bn": round(float(df["transfer_spend_top5_m"].sum() / 1000.0), 2),
        "total_spend_serie_a_bn": round(float(df["transfer_spend_serie_a_m"].sum() / 1000.0), 2),
        "serie_a_spend_share_pct": round(float(df["transfer_spend_serie_a_m"].sum() / df["transfer_spend_top5_m"].sum() * 100.0), 1),
    }
    with open(os.path.join(PROC_DIR, "eda_shock_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2, ensure_ascii=False)

    print(">>> [FASE 2 - EDA & SHOCK] Tutte le 9 figure e il riepilogo JSON sono stati aggiornati!")


if __name__ == "__main__":
    run_eda_single_figures()
