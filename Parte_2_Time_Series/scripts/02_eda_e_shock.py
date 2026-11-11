"""
========================================================================================
Progetto di Data Science - Parte 2: Analisi delle Serie Storiche (Transfermarkt)
Corso di Laurea Magistrale in Ingegneria Informatica e dell'Automazione (UnivPM)
Fase 2 (Capitolo 7): Generazione Figure Singole ad Alta Risoluzione e Massima Leggibilità
========================================================================================
Questo script genera le figure per il Capitolo 7 in formato SINGOLO (non schiacciate,
con font grandi e margini chiari) sia in Parte_2_Time_Series/figures/ che in
Relazione/capitolo7/img/:
- fig03_eda_valore_campionati.png: Valore assoluto (Mld EUR) dei 5 campionati con shock
- fig04_eda_quote_mercato.png: Quote percentuali (%) su area stackplot
- fig05_eda_shock_neymar_2017.png: Zoom Shock 1 Neymar (confronto pendenze pre/post 2017)
- fig06_eda_shock_covid_2020.png: Zoom Shock 2 COVID-19 (taglio Apr 2020 e recupero)
- fig07_eda_shock_boom_2026.png: Zoom Shock 3 Espansione 2022-2026 e valore medio giocatore
- fig08_eda_change_pct_volatilita.png: Serie storica mensile Change % con bande +-2 sigma e outlier
- fig09_eda_distribuzione_kde.png: Distribuzione empirica di densità KDE di Change %
- fig10_eda_spese_calciomercato_impulsi.png: Impulsi stagionali spese calciomercato vs valore rose
- fig11_eda_matrice_correlazione.png: Matrice di correlazione di Pearson valore e spesa
========================================================================================
"""

import os
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

WORKSPACE_ROOT = r"c:\Users\andre\Documents\GitHub\Progetto-Data-Science"
PART2_ROOT = os.path.join(WORKSPACE_ROOT, "Parte_2_Time_Series")
PROC_DIR = os.path.join(PART2_ROOT, "data", "processed")
FIG_DIR = os.path.join(PART2_ROOT, "figures")
REL_DIR = os.path.join(WORKSPACE_ROOT, "Relazione")
CAP7_IMG_DIR = os.path.join(REL_DIR, "capitolo7", "img")

for d in [PROC_DIR, FIG_DIR, CAP7_IMG_DIR]:
    os.makedirs(d, exist_ok=True)

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

    # -------------------------------------------------------------------------
    # FIGURA 3: Valore di mercato assoluto dei 5 campionati
    # -------------------------------------------------------------------------
    print(">>> Generazione fig03_eda_valore_campionati.png...")
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for name, (col, color) in league_cols.items():
        lw = 3.0 if name in ["Premier League", "Serie A"] else 2.2
        alpha = 1.0 if name in ["Premier League", "Serie A"] else 0.85
        ax.plot(df["date"], df[col], label=name, color=color, linewidth=lw, alpha=alpha)

    ax.axvline(pd.Timestamp("2017-08-01"), color="#e41a1c", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(pd.Timestamp("2017-09-01"), 12.6, "Shock Neymar\n(Ago 2017)", color="#b30000", fontsize=10, fontweight="bold")

    ax.axvline(pd.Timestamp("2020-04-01"), color="#333333", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(pd.Timestamp("2020-05-01"), 12.6, "Shock COVID-19\n(Apr 2020: -18,2%)", color="#222222", fontsize=10, fontweight="bold")

    ax.axvline(pd.Timestamp("2023-07-01"), color="#ff7f00", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(pd.Timestamp("2023-08-01"), 10.4, "Super-ciclo 2023-2026\ne boom mercati", color="#b35806", fontsize=10, fontweight="bold")

    ax.set_ylabel("Valore di mercato (Miliardi EUR)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Data (Giugno 2011 – Maggio 2026, 180 mesi)", fontweight="bold", fontsize=12)
    ax.set_title("Evoluzione storica del valore di mercato delle rose nei cinque maggiori campionati europei", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", ncol=3, frameon=True, fontsize=10.5)
    ax.set_ylim(1.0, 15.2)
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
    ax.set_xlabel("Data (Giugno 2011 – Maggio 2026)", fontweight="bold", fontsize=12)
    ax.set_title("Ripartizione percentuale delle quote patrimoniali tra le cinque leghe europee", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncol=5, frameon=True, fontsize=10)

    ax.text(pd.Timestamp("2012-03-01"), 14, "Premier: 28,3%", color="white", fontweight="bold", fontsize=10)
    ax.text(pd.Timestamp("2024-06-01"), 18, "Premier: 39,3%", color="white", fontweight="bold", fontsize=10)
    ax.text(pd.Timestamp("2012-03-01"), 61, "Serie A: 20,7%", color="white", fontweight="bold", fontsize=10)
    ax.text(pd.Timestamp("2024-06-01"), 63, "Serie A: 15,7%", color="white", fontweight="bold", fontsize=10)
    save_fig(fig, "fig04_eda_quote_mercato.png")

    # -------------------------------------------------------------------------
    # FIGURA 5: Zoom Shock 1 - Estate 2017 (Neymar)
    # -------------------------------------------------------------------------
    print(">>> Generazione fig05_eda_shock_neymar_2017.png...")
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    df_s1 = df[(df["date"] >= "2014-01-01") & (df["date"] <= "2020-02-01")].copy()
    pre_2017 = df_s1[df_s1["date"] <= "2017-07-01"]
    post_2017 = df_s1[df_s1["date"] >= "2017-08-01"]

    ax.plot(df_s1["date"], df_s1["top5_market_value_bn"], color="#2b5c8f", linewidth=3.0, label="Valore aggregato Top-5 reale (Mld EUR)")

    t_pre = np.arange(len(pre_2017))
    slope_pre, intercept_pre = np.polyfit(t_pre, pre_2017["top5_market_value_bn"], 1)
    t_all = np.arange(len(df_s1))
    trend_pre_proj = intercept_pre + slope_pre * t_all
    ax.plot(
        df_s1["date"],
        trend_pre_proj,
        color="#555555",
        linestyle=":",
        linewidth=2.4,
        label=f"Trend pre-2017 proiettato (+{slope_pre*12:.2f} Mld EUR/anno)",
    )

    t_post = np.arange(len(post_2017))
    slope_post, intercept_post = np.polyfit(t_post, post_2017["top5_market_value_bn"], 1)
    ax.plot(
        post_2017["date"],
        intercept_post + slope_post * t_post,
        color="#e41a1c",
        linestyle="--",
        linewidth=2.6,
        label=f"Trend post-Neymar (+{slope_post*12:.2f} Mld EUR/anno: quadruplicato)",
    )

    ax.axvline(pd.Timestamp("2017-08-01"), color="#e41a1c", linestyle="-.", linewidth=1.6, alpha=0.8)
    ax.text(pd.Timestamp("2017-08-15"), 15.0, "Trasferimento Neymar (222 Mln EUR)\ne nuovi diritti TV", color="#b30000", fontsize=10, fontweight="bold")

    ax.set_title("Shock 1: cambio strutturale di pendenza dopo il trasferimento di Neymar (Agosto 2017)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Valore rose Top-5 (Miliardi EUR)", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Anno", fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    save_fig(fig, "fig05_eda_shock_neymar_2017.png")

    # -------------------------------------------------------------------------
    # FIGURA 6: Zoom Shock 2 - Primavera 2020 (COVID-19)
    # -------------------------------------------------------------------------
    print(">>> Generazione fig06_eda_shock_covid_2020.png...")
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    df_s2 = df[(df["date"] >= "2019-06-01") & (df["date"] <= "2023-06-01")].copy()
    l1 = ax.plot(df_s2["date"], df_s2["top5_market_value_bn"], color="#2b5c8f", linewidth=2.8, marker="o", markersize=4.5, label="Valore Top-5 (Mld EUR)")
    ax_twin = ax.twinx()
    l2 = ax_twin.plot(df_s2["date"], df_s2["serie_a_market_value_bn"], color="#2ca02c", linewidth=2.4, linestyle="--", marker="s", markersize=4.0, label="Valore Serie A (Mld EUR)")
    ax_twin.grid(False)

    peak_precovid = float(df.loc[df["date"] == "2020-03-01", "top5_market_value_bn"].iloc[0])
    l3 = [ax.axhline(peak_precovid, color="#555555", linestyle=":", linewidth=1.6, label="Picco pre-COVID (33,79 Mld EUR)")]
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01"), color="#e41a1c", alpha=0.15)
    ax.text(pd.Timestamp("2020-04-15"), 26.2, "Taglio COVID-19\n(Apr 2020: -18,17%)", color="#b30000", fontweight="bold", fontsize=10)

    ax.set_title("Shock 2: contrazione pandemica da COVID-19 (Aprile 2020) e dinamica di recupero", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Valore Top-5 (Miliardi EUR)", color="#2b5c8f", fontweight="bold", fontsize=11.5)
    ax_twin.set_ylabel("Valore Serie A (Miliardi EUR)", color="#2ca02c", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Anno", fontweight="bold", fontsize=11.5)
    ax.set_ylim(25.5, 35.5)
    ax_twin.set_ylim(4.6, 6.4)

    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="lower right", frameon=True, fontsize=9.5)
    save_fig(fig, "fig06_eda_shock_covid_2020.png")

    # -------------------------------------------------------------------------
    # FIGURA 7: Zoom Shock 3 - Espansione 2022-2026 e Valore medio per giocatore
    # -------------------------------------------------------------------------
    print(">>> Generazione fig07_eda_shock_boom_2026.png...")
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    df_s3 = df[df["date"] >= "2021-06-01"].copy()
    l1 = ax.plot(df_s3["date"], df_s3["top5_market_value_bn"], color="#2b5c8f", linewidth=2.8, label="Valore totale Top-5 (Mld EUR)")
    ax_twin = ax.twinx()
    l2 = ax_twin.plot(df_s3["date"], df_s3["mean_player_value_m"], color="#d95f02", linewidth=2.4, linestyle="-.", label="Valore medio per calciatore (Mln EUR)")
    ax_twin.grid(False)

    ax.axvline(pd.Timestamp("2023-07-01"), color="#d95f02", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(pd.Timestamp("2023-07-20"), 30.0, "Ingresso fondi sauditi\ne riforma UEFA Champions League", color="#b35806", fontweight="bold", fontsize=9.5)

    ax.set_title("Shock 3: espansione 2022–2026, nuovo massimo storico a 34,44 Mld e inflazione del valore medio", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Valore Top-5 (Miliardi EUR)", color="#2b5c8f", fontweight="bold", fontsize=11.5)
    ax_twin.set_ylabel("Valore medio per calciatore (Milioni EUR)", color="#d95f02", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Anno", fontweight="bold", fontsize=11.5)

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="upper left", frameon=True, fontsize=10)
    save_fig(fig, "fig07_eda_shock_boom_2026.png")

    # -------------------------------------------------------------------------
    # FIGURA 8: Variazioni percentuali mensili (Change %) e bande +-2 sigma
    # -------------------------------------------------------------------------
    print(">>> Generazione fig08_eda_change_pct_volatilita.png...")
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    pct_top5 = df["pct_change_top5"].iloc[1:]
    pct_it1 = df["pct_change_serie_a"].iloc[1:]
    dates_pct = df["date"].iloc[1:]

    mu_pct = float(pct_top5.mean())
    sd_pct = float(pct_top5.std())

    ax.axhline(0, color="black", linewidth=1.0)
    ax.axhline(mu_pct + 2 * sd_pct, color="#d95f02", linestyle="--", linewidth=1.4, label=f"Soglia anomalia +2σ (+{mu_pct + 2*sd_pct:.2f}%)")
    ax.axhline(mu_pct - 2 * sd_pct, color="#d95f02", linestyle="--", linewidth=1.4, label=f"Soglia anomalia -2σ ({mu_pct - 2*sd_pct:.2f}%)")

    ax.plot(dates_pct, pct_top5, color="#2b5c8f", linewidth=1.9, label="Change % mensile Top-5")
    ax.plot(dates_pct, pct_it1, color="#2ca02c", linewidth=1.3, alpha=0.7, linestyle=":", label="Change % mensile Serie A")

    outliers = df.iloc[1:][(df["pct_change_top5"].iloc[1:].abs() > mu_pct + 2 * sd_pct)]
    ax.scatter(outliers["date"], outliers["pct_change_top5"], color="#e41a1c", s=50, zorder=5, label="Mesi outlier (|Δ%| > 2σ)")

    ax.annotate(
        "Crollo COVID-19\n(Apr 2020: -18,17%)",
        xy=(pd.Timestamp("2020-04-01"), -18.17),
        xytext=(pd.Timestamp("2016-06-01"), -16.5),
        arrowprops=dict(arrowstyle="->", color="#e41a1c", lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#b30000",
    )
    ax.annotate(
        "Rivalutazione post-Neymar\n(Gen 2018: +9,24%)",
        xy=(pd.Timestamp("2018-01-01"), 9.24),
        xytext=(pd.Timestamp("2013-06-01"), 8.2),
        arrowprops=dict(arrowstyle="->", color="#2b5c8f", lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#1b3b5c",
    )

    ax.set_title("Variazioni percentuali mensili (Change %) e individuazione degli shock statistici oltre ±2σ", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Variazione percentuale mensile (%)", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Data (2011–2026)", fontweight="bold", fontsize=11.5)
    ax.legend(loc="lower left", ncol=2, frameon=True, fontsize=9.5)
    ax.set_ylim(-21.0, 12.5)
    save_fig(fig, "fig08_eda_change_pct_volatilita.png")

    # -------------------------------------------------------------------------
    # FIGURA 9: Distribuzione empirica di densità KDE
    # -------------------------------------------------------------------------
    print(">>> Generazione fig09_eda_distribuzione_kde.png...")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.histplot(pct_top5, bins=34, kde=True, color="#2b5c8f", stat="density", alpha=0.45, ax=ax, label="Top-5 campionati (KDE)")
    sns.kdeplot(pct_it1, color="#2ca02c", linewidth=2.2, linestyle="--", ax=ax, label="Serie A (IT1)")
    ax.axvline(mu_pct, color="#e41a1c", linestyle="-", linewidth=1.6, label=f"Media Top-5: +{mu_pct:.2f}%/mese")

    ax.set_title("Distribuzione empirica di densità (KDE) delle variazioni percentuali mensili", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Variazione percentuale mensile (%)", fontweight="bold", fontsize=11.5)
    ax.set_ylabel("Densità empirica", fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    save_fig(fig, "fig09_eda_distribuzione_kde.png")

    # -------------------------------------------------------------------------
    # FIGURA 10: Impulsi stagionali spese calciomercato vs valore rose
    # -------------------------------------------------------------------------
    print(">>> Generazione fig10_eda_spese_calciomercato_impulsi.png...")
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax_twin = ax.twinx()

    b_sp = ax.bar(
        df["date"],
        df["transfer_spend_top5_m"] / 1000.0,
        width=25,
        color="#d95f02",
        alpha=0.6,
        label="Volume spesa mensile trasferimenti (Mld EUR)",
    )
    l_val = ax_twin.plot(
        df["date"],
        df["top5_market_value_bn"],
        color="#2b5c8f",
        linewidth=2.8,
        label="Valore di mercato rose Top-5 (Mld EUR)",
    )
    ax_twin.grid(False)

    ax.set_title("Impulsi stagionali di spesa nelle finestre di calciomercato e crescita del valore patrimoniale delle rose", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Spesa mensile trasferimenti (Miliardi EUR)", color="#d95f02", fontweight="bold", fontsize=11.5)
    ax_twin.set_ylabel("Valore rose Top-5 (Miliardi EUR)", color="#2b5c8f", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Data (2011–2026)", fontweight="bold", fontsize=11.5)
    ax.set_ylim(0, 5.2)
    ax_twin.set_ylim(9.0, 37.0)

    h1, lb1 = ax.get_legend_handles_labels()
    h2, lb2 = ax_twin.get_legend_handles_labels()
    ax.legend(h1 + h2, lb1 + lb2, loc="upper left", frameon=True, fontsize=10)
    save_fig(fig, "fig10_eda_spese_calciomercato_impulsi.png")

    # -------------------------------------------------------------------------
    # FIGURA 11: Matrice di correlazione di Pearson
    # -------------------------------------------------------------------------
    print(">>> Generazione fig11_eda_matrice_correlazione.png...")
    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    df_corr = pd.DataFrame({
        "Valore Top-5": df["top5_market_value_bn"],
        "Valore Serie A": df["serie_a_market_value_bn"],
        "Valore Premier": df["premier_league_market_value_bn"],
        "Valore medio gioc.": df["mean_player_value_m"],
        "Spesa mensile": df["transfer_spend_top5_m"],
        "Spesa mobile 12m": df["transfer_spend_top5_m"].rolling(12, min_periods=1).sum(),
        "N. trasferimenti": df["n_transfers_top5"],
    })
    corr_mat = df_corr.corr()

    sns.heatmap(
        corr_mat,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-0.2,
        vmax=1.0,
        cbar=True,
        square=True,
        ax=ax,
        annot_kws={"size": 10.5, "weight": "bold"},
    )
    ax.set_title("Matrice di correlazione lineare di Pearson tra valore patrimoniale e metriche di mercato", fontsize=12.0, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    save_fig(fig, "fig11_eda_matrice_correlazione.png")

    # Salvataggio metriche
    eda_summary = {
        "slope_pre2017_bn_per_year": round(float(slope_pre * 12), 2),
        "slope_post2017_bn_per_year": round(float(slope_post * 12), 2),
        "pct_change_mean_top5": round(mu_pct, 3),
        "pct_change_std_top5": round(sd_pct, 3),
        "pct_change_skew_top5": round(float(pct_top5.skew()), 2),
        "pct_change_kurt_top5": round(float(pct_top5.kurtosis()), 2),
        "corr_val_rolling12m_spend": round(float(corr_mat.loc["Valore Top-5", "Spesa mobile 12m"]), 3),
        "corr_val_serie_a_top5": round(float(corr_mat.loc["Valore Top-5", "Valore Serie A"]), 3),
        "max_monthly_spend_top5_m": round(float(df["transfer_spend_top5_m"].max()), 2),
        "max_monthly_spend_month": str(df.loc[df["transfer_spend_top5_m"].idxmax(), "year_month"]),
    }
    with open(os.path.join(PROC_DIR, "eda_shock_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2, ensure_ascii=False)

    print(">>> [FASE 2 - EDA & SHOCK] Tutte le 9 figure singole ad alta leggibilità sono state create con successo!")


if __name__ == "__main__":
    run_eda_single_figures()
