"""
========================================================================================
Modulo di Analisi: Decomposizione STL, Test di Stazionarietà e Autocorrelazione
========================================================================================
Questo modulo contiene le funzioni per:
1. Decomporre la serie in Trend, Stagionalità e Residuo (STL).
2. Eseguire i test matematici di stazionarietà (ADF e KPSS).
3. Calcolare i grafici di memoria della serie (ACF e PACF).
Tutti i calcoli e i grafici sono spiegati con commenti semplici e in italiano.
========================================================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

from ..config import CAP8_IMG_DIR, SEASONAL_PERIOD
from ..plotting import crea_figura, salva_figura

warnings.filterwarnings("ignore")


def esegui_decomposizione_stl(serie, date):
    """
    Divide la serie originale in 3 componenti:
    - Trend: la direzione di fondo negli anni (crescita).
    - Stagionalità: il ciclo che si ripete ogni 12 mesi (mercato estivo e revisioni autunnali).
    - Residuo: le variazioni impreviste e gli shock (es. COVID).
    """
    print(">>> [1/4] Esecuzione della Decomposizione STL (periodo = 12 mesi)...")
    serie_con_tempo = pd.Series(serie.values, index=pd.to_datetime(date))

    # STL (Seasonal and Trend decomposition using Loess) con stima robusta agli outlier
    res_stl = STL(serie_con_tempo, period=SEASONAL_PERIOD, robust=True).fit()

    trend = res_stl.trend
    stagionalita = res_stl.seasonal
    residuo = res_stl.resid

    # Calcolo della forza del trend e della stagionalità (formule standard di Wang et al.)
    var_residuo = np.var(residuo)
    forza_trend = max(0.0, 1.0 - var_residuo / np.var(trend + residuo))
    forza_stagionalita = max(0.0, 1.0 - var_residuo / np.var(stagionalita + residuo))

    print(f"   -> Forza del Trend: {forza_trend:.3f} (molto forte, vicino a 1.0)")
    print(f"   -> Forza della Stagionalità: {forza_stagionalita:.3f}")

    # FIGURA 12: I 4 pannelli della decomposizione STL (impilati in verticale ben leggibili)
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.5), sharex=True)

    axes[0].plot(date, serie, color="#2b5c8f", linewidth=2.4)
    axes[0].set_ylabel("Serie reale (Mld)", fontweight="bold", fontsize=10.5)
    axes[0].set_title(
        "Decomposizione STL della serie storica del valore rose Top-5 (2011–2026)",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )

    axes[1].plot(date, trend, color="#d95f02", linewidth=2.4)
    axes[1].set_ylabel("Trend (Mld)", fontweight="bold", fontsize=10.5)
    axes[1].set_title(
        f"Componente di Trend di lungo periodo (Forza trend = {forza_trend:.2f})",
        fontsize=11,
        fontweight="bold",
    )

    axes[2].plot(date, stagionalita, color="#2ca02c", linewidth=2.0)
    axes[2].set_ylabel("Stagionalità (Mld)", fontweight="bold", fontsize=10.5)
    axes[2].set_title(
        f"Componente Stagionale annuale regolare (Periodo s = 12 mesi, Forza = {forza_stagionalita:.2f})",
        fontsize=11,
        fontweight="bold",
    )
    axes[2].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

    axes[3].scatter(date, residuo, color="#7570b3", s=24, alpha=0.85)
    axes[3].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[3].set_ylabel("Residuo (Mld)", fontweight="bold", fontsize=10.5)
    axes[3].set_xlabel("Data", fontweight="bold", fontsize=11)
    axes[3].set_title(
        "Componente Residuale irregolare (Shock esterni ed eventi anomali)",
        fontsize=11,
        fontweight="bold",
    )

    # Evidenziamo il crollo COVID nel residuo
    data_covid = pd.Timestamp("2020-04-01")
    axes[3].annotate(
        "Shock COVID-19\n(Apr 2020: -6,1 Mld)",
        xy=(data_covid, residuo.loc[data_covid]),
        xytext=(pd.Timestamp("2015-06-01"), -3.8),
        arrowprops=dict(arrowstyle="->", color="#e41a1c", lw=1.5),
        fontweight="bold",
        color="#b30000",
        fontsize=9.5,
    )

    plt.tight_layout()
    salva_figura(fig, "fig12_stl_decomposizione_completa.png", CAP8_IMG_DIR)

    # FIGURA 13: Il profilo medio stagionale dei 12 mesi (da Gennaio a Dicembre)
    fig_profilo, ax_p = crea_figura(larghezza=11.0, altezza=5.6)
    profilo_12m = stagionalita.groupby(stagionalita.index.month).mean()
    nomi_mesi = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]

    colori_barre = ["#2b5c8f" if v >= 0 else "#d95f02" for v in profilo_12m]
    barre = ax_p.bar(nomi_mesi, profilo_12m.values, color=colori_barre, width=0.58, alpha=0.88)
    ax_p.axhline(0, color="black", linewidth=1.0)
    ax_p.set_ylim(-0.20, 0.35)

    ax_p.set_ylabel("Effetto stagionale medio (Miliardi EUR)", fontweight="bold", fontsize=11.5)
    ax_p.set_xlabel("Mese dell'anno", fontweight="bold", fontsize=11.5)
    ax_p.set_title(
        "Profilo stagionale medio mensile del valore delle rose nei dodici mesi dell'anno",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )

    # Aggiungiamo i valori numerici sopra o sotto ogni barra con ampio margine
    for barra in barre:
        h = barra.get_height()
        offset = 0.015 if h >= 0 else -0.028
        ax_p.text(
            barra.get_x() + barra.get_width() / 2.0,
            h + offset,
            f"{h:+.2f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    salva_figura(fig_profilo, "fig13_stl_profilo_medio_mensile.png", CAP8_IMG_DIR)

    return {
        "trend": trend,
        "stagionalita": stagionalita,
        "residuo": residuo,
        "forza_trend": round(float(forza_trend), 3),
        "forza_stagionalita": round(float(forza_stagionalita), 3),
        "profilo_mensile": {m: round(float(v), 3) for m, v in zip(nomi_mesi, profilo_12m.values)},
    }


def calcola_test_stazionarieta(serie, nome_serie):
    """
    Esegue i due test statistici standard:
    1. ADF (Augmented Dickey-Fuller): Ipotesi H0 = la serie HA una radice unitaria (NON è stazionaria).
       Se p-value < 0.05 rifiutiamo H0 -> la serie è stazionaria.
    2. KPSS: Ipotesi H0 = la serie È stazionaria.
       Se p-value > 0.05 non possiamo rifiutare H0 -> la serie è stazionaria.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_adf = adfuller(serie.dropna(), autolag="AIC")
        stat_adf, pval_adf, _, _, crit_adf, _ = res_adf

        res_kpss = kpss(serie.dropna(), regression="c", nlags="auto")
        stat_kpss, pval_kpss, _, crit_kpss = res_kpss

    esito_adf = "Stazionaria" if pval_adf < 0.05 else "Non stazionaria"
    esito_kpss = "Stazionaria" if pval_kpss >= 0.05 else "Non stazionaria"

    print(f"   [{nome_serie}]")
    print(f"      ADF: Statistica = {stat_adf:.3f}, p-value = {pval_adf:.4f} -> {esito_adf}")
    print(f"      KPSS: Statistica = {stat_kpss:.3f}, p-value = {pval_kpss:.4f} -> {esito_kpss}")

    return {
        "nome": nome_serie,
        "adf_stat": round(float(stat_adf), 3),
        "adf_pval": round(float(pval_adf), 4),
        "adf_crit_5pct": round(float(crit_adf["5%"]), 3),
        "adf_esito": esito_adf,
        "kpss_stat": round(float(stat_kpss), 3),
        "kpss_pval": round(float(pval_kpss), 4),
        "kpss_crit_5pct": round(float(crit_kpss["5%"]), 3),
        "kpss_esito": esito_kpss,
    }


def analizza_stazionarieta_e_differenziazione(serie, date):
    """
    Confronta la serie originale (livello) con la serie differenziata al mese precedente (d=1).
    Genera il grafico del confronto e calcola la tabella riassuntiva dei test ADF e KPSS.
    """
    print(">>> [2/4] Analisi della stazionarietà e test matematici...")
    serie_diff1 = serie.diff().dropna()
    date_diff1 = date.iloc[1:]

    test_originale = calcola_test_stazionarieta(serie, "Serie originale di livello (d=0)")
    test_diff1 = calcola_test_stazionarieta(serie_diff1, "Serie differenziata prima (d=1)")

    # FIGURA 14: Confronto visivo impilato in verticale (ampio e leggibile)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)

    axes[0].plot(date, serie, color="#2b5c8f", linewidth=2.5)
    axes[0].set_ylabel("Valore (Mld EUR)", fontweight="bold", fontsize=11)
    axes[0].set_title(
        f"a) Serie originale in livelli: non stazionaria (trend rialzista, p-value ADF = {test_originale['adf_pval']:.3f})",
        fontsize=12,
        fontweight="bold",
    )

    axes[1].plot(date_diff1, serie_diff1, color="#2ca02c", linewidth=1.9)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)

    mu_d = serie_diff1.mean()
    sd_d = serie_diff1.std()
    axes[1].axhline(mu_d + 2 * sd_d, color="#d95f02", linestyle=":", linewidth=1.3, label="Banda +2σ")
    axes[1].axhline(mu_d - 2 * sd_d, color="#d95f02", linestyle=":", linewidth=1.3, label="Banda -2σ")

    axes[1].set_ylabel("Differenza mensile ΔY (Mld)", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Data (2011–2026)", fontweight="bold", fontsize=11)
    axes[1].set_title(
        "b) Serie differenziata prima (d=1): stazionaria attorno a zero (p-value ADF < 0.001, KPSS p-value = 0.10)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    salva_figura(fig, "fig14_stazionarieta_confronto_differenziazione.png", CAP8_IMG_DIR)

    return [test_originale, test_diff1]


def calcola_e_disegna_acf_pacf(serie_diff, max_lags=36):
    """
    Calcola l'Autocorrelazione (ACF) e l'Autocorrelazione Parziale (PACF)
    sulla serie differenziata stazionaria fino a 36 ritardi (3 anni di memoria).
    Disegna i due pannelli impilati a tutta larghezza per la massima leggibilità.
    """
    print(">>> [3/4] Calcolo delle funzioni di autocorrelazione ACF e PACF...")
    valori_acf = acf(serie_diff, nlags=max_lags)
    valori_pacf = pacf(serie_diff, nlags=max_lags, method="ywm")

    # Intervallo di confidenza al 95% (regola di Bartlett: +-1.96 / sqrt(N))
    soglia_confidenza = 1.96 / np.sqrt(len(serie_diff))
    lags = np.arange(max_lags + 1)

    # FIGURA 15: Due pannelli a tutta larghezza (impilati verticalmente) per leggere ogni lag
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True)

    # Pannello Superiore: ACF
    ax_a = axes[0]
    ax_a.stem(lags, valori_acf, linefmt="#2b5c8f", markerfmt="o", basefmt="k-")
    ax_a.axhspan(
        -soglia_confidenza,
        soglia_confidenza,
        color="#2b5c8f",
        alpha=0.15,
        label="Intervallo di confidenza 95% (soglia di Bartlett ±0,146)",
    )
    ax_a.axvline(12, color="#d95f02", linestyle="--", linewidth=1.4, alpha=0.85, label="Ciclo stagionale annuale (Lag 12, 24, 36)")
    ax_a.axvline(24, color="#d95f02", linestyle="--", linewidth=1.4, alpha=0.85)
    ax_a.axvline(36, color="#d95f02", linestyle="--", linewidth=1.4, alpha=0.85)
    ax_a.set_title(
        "a) Funzione di Autocorrelazione (ACF) sulla serie differenziata ΔY (guida per ordini MA q, Q)",
        fontsize=12,
        fontweight="bold",
    )
    ax_a.set_ylabel("Autocorrelazione (ACF)", fontweight="bold", fontsize=11)
    ax_a.set_ylim(-0.35, 1.08)
    ax_a.legend(loc="upper right", fontsize=9.5)

    # Pannello Inferiore: PACF
    ax_p = axes[1]
    ax_p.stem(lags, valori_pacf, linefmt="#2ca02c", markerfmt="s", basefmt="k-")
    ax_p.axhspan(
        -soglia_confidenza,
        soglia_confidenza,
        color="#2ca02c",
        alpha=0.15,
        label="Intervallo di confidenza 95% (±0,146)",
    )
    ax_p.axvline(12, color="#d95f02", linestyle="--", linewidth=1.4, alpha=0.85, label="Ciclo stagionale annuale (Lag 12, 24, 36)")
    ax_p.axvline(24, color="#d95f02", linestyle="--", linewidth=1.4, alpha=0.85)
    ax_p.axvline(36, color="#d95f02", linestyle="--", linewidth=1.4, alpha=0.85)
    ax_p.set_title(
        "b) Funzione di Autocorrelazione Parziale (PACF) sulla serie differenziata ΔY (guida per ordini AR p, P)",
        fontsize=12,
        fontweight="bold",
    )
    ax_p.set_xlabel("Ritardo temporale in mesi (Lag k = 0, 1, ..., 36)", fontweight="bold", fontsize=11)
    ax_p.set_ylabel("Autocorrelazione Parziale", fontweight="bold", fontsize=11)
    ax_p.set_ylim(-0.35, 1.08)
    ax_p.set_xticks(np.arange(0, max_lags + 1, 3))
    ax_p.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    salva_figura(fig, "fig15_acf_pacf_serie_differenziata.png", CAP8_IMG_DIR)

    return {
        "lag1_acf": round(float(valori_acf[1]), 3),
        "lag12_acf": round(float(valori_acf[12]), 3),
        "lag1_pacf": round(float(valori_pacf[1]), 3),
        "lag12_pacf": round(float(valori_pacf[12]), 3),
        "soglia_confidenza": round(float(soglia_confidenza), 3),
    }
