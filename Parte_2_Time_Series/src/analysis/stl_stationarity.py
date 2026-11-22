"""
========================================================================================
Modulo di Analisi: Decomposizione STL, Test di Stazionarietà e Autocorrelazione
========================================================================================
Tutte le analisi diagnostiche di identificazione del modello vengono eseguite sul solo
Training Set (primi 120 mesi, Giugno 2013 – Maggio 2023) per evitare data leakage dal
Test Set nelle scelte di differenziazione e di ordine (p, d, q)(P, D, Q)_12.
Si verifica inoltre la coerenza dei risultati anche in scala logaritmica e sull'intera
serie storica (salvati nel riepilogo JSON).
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


def _forza_stl(trend, stagionalita, residuo):
    var_r = float(np.var(residuo))
    ft = max(0.0, 1.0 - var_r / float(np.var(trend + residuo)))
    fs = max(0.0, 1.0 - var_r / float(np.var(stagionalita + residuo)))
    return ft, fs


def esegui_decomposizione_stl(serie, date, label_campione="Training Set, 2013–2023"):
    """
    Applica la decomposizione STL (Seasonal and Trend decomposition using Loess) con
    pesi robusti iterativi (robust=True) e periodo s = 12 mesi.
    Confronta inoltre le metriche di forza del trend e della stagionalità anche in scala
    logaritmica (decomposizione moltiplicativa) ed escludendo l'outlier di aprile 2020.
    """
    print(f">>> [1/4] Decomposizione STL ({label_campione}, s = {SEASONAL_PERIOD} mesi)...")
    idx_time = pd.to_datetime(date)
    serie_t = pd.Series(np.asarray(serie, dtype=float), index=idx_time)

    res_stl = STL(serie_t, period=SEASONAL_PERIOD, robust=True).fit()
    trend, stagionalita, residuo = res_stl.trend, res_stl.seasonal, res_stl.resid
    ft, fs = _forza_stl(trend, stagionalita, residuo)

    # Verifica escludendo il singolo outlier di aprile 2020
    mask_nc = residuo.index != pd.Timestamp("2020-04-01")
    _, fs_nc = _forza_stl(trend[mask_nc], stagionalita[mask_nc], residuo[mask_nc])

    # Verifica in scala logaritmica (equivalente a modello moltiplicativo)
    res_log = STL(np.log(serie_t), period=SEASONAL_PERIOD, robust=True).fit()
    ft_log, fs_log = _forza_stl(res_log.trend, res_log.seasonal, res_log.resid)
    _, fs_log_nc = _forza_stl(res_log.trend[mask_nc], res_log.seasonal[mask_nc], res_log.resid[mask_nc])

    print(f"   -> Livelli: F_T = {ft:.3f} | F_S = {fs:.3f} (escl. Apr 2020: {fs_nc:.3f})")
    print(f"   -> Log:     F_T = {ft_log:.3f} | F_S = {fs_log:.3f} (escl. Apr 2020: {fs_log_nc:.3f})")

    # FIGURA 12: I 4 pannelli della decomposizione STL
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.5), sharex=True)

    axes[0].plot(idx_time, serie_t.values, color="#2b5c8f", linewidth=2.4)
    axes[0].set_ylabel("Serie reale (Mld)", fontweight="bold", fontsize=10.5)
    axes[0].set_title(
        f"Decomposizione STL additiva robusta della serie Top-5 ({label_campione})",
        fontsize=12.5, fontweight="bold", pad=10,
    )

    axes[1].plot(idx_time, trend.values, color="#d95f02", linewidth=2.4)
    axes[1].set_ylabel("Trend (Mld)", fontweight="bold", fontsize=10.5)
    axes[1].set_title(f"Componente di trend di lungo periodo (forza F_T = {ft:.3f})", fontsize=11, fontweight="bold")

    axes[2].plot(idx_time, stagionalita.values, color="#2ca02c", linewidth=2.0)
    axes[2].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[2].set_ylabel("Stagionalità (Mld)", fontweight="bold", fontsize=10.5)
    axes[2].set_title(
        f"Componente stagionale annuale (s = 12 mesi, F_S = {fs:.3f}; in scala log: {fs_log:.3f})",
        fontsize=11, fontweight="bold",
    )

    axes[3].scatter(idx_time, residuo.values, color="#7570b3", s=26, alpha=0.85)
    axes[3].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[3].set_ylabel("Residuo (Mld)", fontweight="bold", fontsize=10.5)
    axes[3].set_xlabel("Data", fontweight="bold", fontsize=11)
    axes[3].set_title("Componente residuale irregolare (shock esterni ed eventi anomali)", fontsize=11, fontweight="bold")

    data_covid = pd.Timestamp("2020-04-01")
    if data_covid in residuo.index:
        val_cov = float(residuo.loc[data_covid])
        axes[3].annotate(
            f"Shock COVID-19\n(Apr 2020: {val_cov:.2f} Mld)",
            xy=(data_covid, val_cov),
            xytext=(pd.Timestamp("2016-01-01"), val_cov * 0.78),
            arrowprops=dict(arrowstyle="->", color="#e41a1c", lw=1.5),
            fontweight="bold", color="#b30000", fontsize=9.5,
        )

    plt.tight_layout()
    salva_figura(fig, "fig12_stl_decomposizione_completa.png", CAP8_IMG_DIR)

    # FIGURA 13: Profilo stagionale medio mensile (su Training Set)
    fig_p, ax_p = crea_figura(larghezza=11.0, altezza=5.6)
    profilo_12m = stagionalita.groupby(stagionalita.index.month).mean()
    nomi_mesi = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]

    colori_barre = ["#2b5c8f" if v >= 0 else "#d95f02" for v in profilo_12m]
    barre = ax_p.bar(nomi_mesi, profilo_12m.values, color=colori_barre, width=0.58, alpha=0.88)
    ax_p.axhline(0, color="black", linewidth=1.0)
    ymin_p = min(float(profilo_12m.min()), -0.1) * 1.35
    ymax_p = max(float(profilo_12m.max()), 0.1) * 1.30
    ax_p.set_ylim(ymin_p, ymax_p)

    ax_p.set_ylabel("Effetto stagionale medio (Miliardi EUR)", fontweight="bold", fontsize=11.5)
    ax_p.set_xlabel("Mese dell'anno", fontweight="bold", fontsize=11.5)
    ax_p.set_title(
        f"Profilo stagionale medio mensile della componente S_t ({label_campione})",
        fontsize=12.5, fontweight="bold", pad=12,
    )

    span = ymax_p - ymin_p
    for barra in barre:
        h = barra.get_height()
        offset = 0.03 * span if h >= 0 else -0.055 * span
        ax_p.text(
            barra.get_x() + barra.get_width() / 2.0,
            h + offset,
            f"{h:+.2f}",
            ha="center", fontsize=9.8, fontweight="bold",
        )

    salva_figura(fig_p, "fig13_stl_profilo_medio_mensile.png", CAP8_IMG_DIR)

    return {
        "forza_trend_livelli": round(ft, 3),
        "forza_stagionalita_livelli": round(fs, 3),
        "forza_stagionalita_livelli_no_covid": round(fs_nc, 3),
        "forza_trend_log": round(ft_log, 3),
        "forza_stagionalita_log": round(fs_log, 3),
        "forza_stagionalita_log_no_covid": round(fs_log_nc, 3),
        "residuo_covid_apr2020_bn": round(float(residuo.loc[data_covid]), 2) if data_covid in residuo.index else None,
        "profilo_mensile": {m: round(float(v), 3) for m, v in zip(nomi_mesi, profilo_12m.values)},
    }


def calcola_test_stazionarieta(serie, nome_serie, regression="c"):
    """
    Esegue i test ADF (H0: presenza di radice unitaria / non stazionarietà) e KPSS
    (H0: stazionarietà intorno a costante 'c' o a trend lineare 'ct').
    Gestisce esplicitamente i limiti tabellari del p-value KPSS (<= 0.01 e >= 0.10).
    """
    s = pd.Series(serie).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat_adf, pval_adf, _, _, crit_adf, _ = adfuller(s, regression=regression, autolag="AIC")
        stat_kpss, pval_kpss, _, crit_kpss = kpss(s, regression=regression, nlags="auto")

    esito_adf = "Stazionaria" if pval_adf < 0.05 else "Non stazionaria"
    esito_kpss = "Stazionaria" if pval_kpss >= 0.05 else "Non stazionaria"
    kpss_pval_str = "≤ 0.01" if pval_kpss <= 0.01001 else ("≥ 0.10" if pval_kpss >= 0.09999 else f"{pval_kpss:.4f}")

    print(f"   [{nome_serie} | reg='{regression}']")
    print(f"      ADF:  stat = {stat_adf:.3f}, p = {pval_adf:.4f} (crit 5% = {crit_adf['5%']:.3f}) -> {esito_adf}")
    print(f"      KPSS: stat = {stat_kpss:.3f}, p {kpss_pval_str} (crit 5% = {crit_kpss['5%']:.3f}) -> {esito_kpss}")

    return {
        "nome": nome_serie,
        "regression": regression,
        "adf_stat": round(float(stat_adf), 3),
        "adf_pval": round(float(pval_adf), 4),
        "adf_crit_5pct": round(float(crit_adf["5%"]), 3),
        "adf_esito": esito_adf,
        "kpss_stat": round(float(stat_kpss), 3),
        "kpss_pval": round(float(pval_kpss), 4),
        "kpss_pval_label": kpss_pval_str,
        "kpss_crit_5pct": round(float(crit_kpss["5%"]), 3),
        "kpss_esito": esito_kpss,
    }


def analizza_stazionarieta_e_differenziazione(serie_train, date_train, serie_full=None):
    """
    Esegue i test ADF e KPSS sul Training Set per:
    - livelli d=0 con solo costante ('c') e con costante + trend ('ct'),
    - differenze prime d=1 ('c'),
    - verifica in scala logaritmica (livelli 'ct' e differenze logaritmiche 'c').
    Genera la Figura 14.
    """
    print(">>> [2/4] Test di stazionarietà ADF e KPSS sul Training Set...")
    s_tr = pd.Series(np.asarray(serie_train, dtype=float), index=pd.to_datetime(date_train))
    s_diff1 = s_tr.diff().dropna()
    s_diff1_nc = s_diff1.copy()
    data_cov = pd.Timestamp("2020-04-01")
    if data_cov in s_diff1_nc.index:
        s_diff1_nc.loc[data_cov] = float(s_diff1.median())
    date_tr = pd.to_datetime(date_train)

    risultati = [
        calcola_test_stazionarieta(s_tr, "Livelli Y_t (d=0, costante)", regression="c"),
        calcola_test_stazionarieta(s_tr, "Livelli Y_t (d=0, costante + trend)", regression="ct"),
        calcola_test_stazionarieta(s_diff1, "Differenze prime ΔY_t (d=1, costante)", regression="c"),
        calcola_test_stazionarieta(np.log(s_tr), "Log-livelli ln(Y_t) (d=0, costante + trend)", regression="ct"),
        calcola_test_stazionarieta(np.log(s_tr).diff().dropna(), "Differenze logaritmiche Δln(Y_t) (d=1, costante)", regression="c"),
        calcola_test_stazionarieta(s_diff1_nc, "Verifica ΔY_t depurata da Apr 2020 (d=1, c)", regression="c"),
        calcola_test_stazionarieta(s_tr.diff().diff(SEASONAL_PERIOD).dropna(), "Verifica doppia diff. ΔΔ12 Y_t (d=1, D=1)", regression="c"),
    ]
    if serie_full is not None:
        s_f = pd.Series(np.asarray(serie_full, dtype=float))
        calcola_test_stazionarieta(s_f, "Verifica intera serie Y_t (d=0, ct)", regression="ct")
        calcola_test_stazionarieta(s_f.diff().dropna(), "Verifica intera serie ΔY_t (d=1, c)", regression="c")

    # FIGURA 14: Confronto visivo tra livelli (d=0) e differenze prime (d=1) sul Training Set
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)

    axes[0].plot(date_tr, s_tr.values, color="#2b5c8f", linewidth=2.5)
    axes[0].set_ylabel("Valore (Mld EUR)", fontweight="bold", fontsize=11)
    axes[0].set_title(
        f"a) Serie originale in livelli sul Training Set (d=0): non stazionaria (ADF ct p = {risultati[1]['adf_pval']:.3f}, KPSS ct p {risultati[1]['kpss_pval_label']})",
        fontsize=11.5, fontweight="bold",
    )

    axes[1].plot(date_tr.iloc[1:], s_diff1.values, color="#2ca02c", linewidth=1.9)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    mu_d, sd_d = float(s_diff1.mean()), float(s_diff1.std())
    axes[1].axhline(mu_d + 2 * sd_d, color="#d95f02", linestyle=":", linewidth=1.3, label=f"Fascia μ±2σ ({mu_d - 2*sd_d:.2f} / +{mu_d + 2*sd_d:.2f} Mld)")
    axes[1].axhline(mu_d - 2 * sd_d, color="#d95f02", linestyle=":", linewidth=1.3)
    axes[1].set_ylabel("Differenza mensile ΔY_t (Mld)", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Data (Training Set: Giugno 2013 – Maggio 2023)", fontweight="bold", fontsize=11)
    adf_d1_str = "< 0.0001" if risultati[2]["adf_pval"] < 0.0001 else f"= {risultati[2]['adf_pval']:.4f}"
    axes[1].set_title(
        f"b) Serie differenziata prima (d=1): stazionaria (ADF p {adf_d1_str}, KPSS p {risultati[2]['kpss_pval_label']})",
        fontsize=11.5, fontweight="bold",
    )
    axes[1].legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    salva_figura(fig, "fig14_stazionarieta_confronto_differenziazione.png", CAP8_IMG_DIR)
    return risultati


def calcola_e_disegna_acf_pacf(serie_train, date_train, max_lags=36):
    """
    Calcola ACF e PACF sulla serie differenziata del Training Set in due versioni:
    1. Grezza (con il crollo anomalo di aprile 2020);
    2. Depurata dal singolo shock di aprile 2020 (sostituito con la mediana di ΔY_t),
       per mostrare come l'outlier pandemico attenui il segnale stagionale a lag 12.
    La banda ±1,96/sqrt(N) rappresenta l'intervallo di confidenza al 95% sotto ipotesi di
    rumore bianco.
    """
    print(">>> [3/4] Calcolo ACF e PACF (grezza vs depurata dallo shock COVID-19)...")
    s_tr = pd.Series(np.asarray(serie_train, dtype=float), index=pd.to_datetime(date_train))
    d_raw = s_tr.diff().dropna()
    d_clean = d_raw.copy()
    data_cov = pd.Timestamp("2020-04-01")
    if data_cov in d_clean.index:
        d_clean.loc[data_cov] = float(d_raw.median())

    acf_raw = acf(d_raw.values, nlags=max_lags)
    pacf_raw = pacf(d_raw.values, nlags=max_lags, method="ywm")
    acf_cl = acf(d_clean.values, nlags=max_lags)
    pacf_cl = pacf(d_clean.values, nlags=max_lags, method="ywm")

    # Soglia al 95% per processo rumore bianco: +-1.96 / sqrt(N)
    soglia = 1.96 / np.sqrt(len(d_raw))
    lags = np.arange(1, max_lags + 1)

    # FIGURA 15: 2x2 confronto ACF/PACF grezza (sinistra) vs depurata da Apr 2020 (destra)
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.2), sharex=True, sharey=True)

    pannelli = [
        (axes[0, 0], acf_raw[1:], "#2b5c8f", "o", "a) ACF su ΔY_t grezza (con shock Apr 2020)"),
        (axes[0, 1], acf_cl[1:], "#2b5c8f", "o", "b) ACF su ΔY_t depurata dallo shock Apr 2020"),
        (axes[1, 0], pacf_raw[1:], "#2ca02c", "s", "c) PACF su ΔY_t grezza (con shock Apr 2020)"),
        (axes[1, 1], pacf_cl[1:], "#2ca02c", "s", "d) PACF su ΔY_t depurata dallo shock Apr 2020"),
    ]
    for ax, vals, col, mk, titolo in pannelli:
        ax.stem(lags, vals, linefmt=col, markerfmt=mk, basefmt="k-")
        ax.axhspan(-soglia, soglia, color=col, alpha=0.14, label=f"Banda rumore bianco 95% (±{soglia:.3f})")
        for s_lag in [12, 24, 36]:
            ax.axvline(s_lag, color="#d95f02", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(titolo, fontsize=11.2, fontweight="bold")
        ax.set_ylim(-0.35, 0.55)
        ax.set_xticks(np.arange(0, max_lags + 1, 6))
        ax.legend(loc="upper right", fontsize=8.8)

    axes[0, 0].set_ylabel("Autocorrelazione (ACF)", fontweight="bold", fontsize=10.5)
    axes[1, 0].set_ylabel("Autocorrelazione parziale (PACF)", fontweight="bold", fontsize=10.5)
    axes[1, 0].set_xlabel("Ritardo in mesi (lag k)", fontweight="bold", fontsize=10.5)
    axes[1, 1].set_xlabel("Ritardo in mesi (lag k)", fontweight="bold", fontsize=10.5)

    plt.suptitle(
        "Funzioni ACF e PACF sulla serie differenziata ΔY_t (Training Set, N = 119): effetto di mascheramento dello shock COVID-19",
        fontsize=12.4, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    salva_figura(fig, "fig15_acf_pacf_serie_differenziata.png", CAP8_IMG_DIR)

    return {
        "n_diff_train": int(len(d_raw)),
        "soglia_rumore_bianco_95": round(float(soglia), 3),
        "raw": {
            "lag1_acf": round(float(acf_raw[1]), 3),
            "lag2_acf": round(float(acf_raw[2]), 3),
            "lag12_acf": round(float(acf_raw[12]), 3),
            "lag24_acf": round(float(acf_raw[24]), 3),
            "lag1_pacf": round(float(pacf_raw[1]), 3),
            "lag2_pacf": round(float(pacf_raw[2]), 3),
            "lag12_pacf": round(float(pacf_raw[12]), 3),
        },
        "no_covid": {
            "lag1_acf": round(float(acf_cl[1]), 3),
            "lag2_acf": round(float(acf_cl[2]), 3),
            "lag5_acf": round(float(acf_cl[5]), 3),
            "lag7_acf": round(float(acf_cl[7]), 3),
            "lag12_acf": round(float(acf_cl[12]), 3),
            "lag24_acf": round(float(acf_cl[24]), 3),
            "lag1_pacf": round(float(pacf_cl[1]), 3),
            "lag2_pacf": round(float(pacf_cl[2]), 3),
            "lag12_pacf": round(float(pacf_cl[12]), 3),
            "lag24_pacf": round(float(pacf_cl[24]), 3),
        },
    }
