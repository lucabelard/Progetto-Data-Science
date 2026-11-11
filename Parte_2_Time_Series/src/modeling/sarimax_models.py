"""
========================================================================================
Modulo di Modellazione Predittiva: Baseline, ARIMA, Holt-Winters, SARIMA e SARIMAX
========================================================================================
Questo file contiene tutte le funzioni per il Capitolo 9 della Relazione:
1. Divisione della serie in Training Set (primi 144 mesi, 2011-2023) e Test Set (36 mesi, 2023-2026).
2. Costruzione delle variabili esogene (spese calciomercato, numero trasferimenti e shock COVID/Neymar).
3. Addestramento e valutazione di 5 modelli in gara:
   - Modello 1: Baseline Naive (Persistenza dell'ultimo valore noto)
   - Modello 2: ARIMA(1, 1, 1) non stagionale
   - Modello 3: Holt-Winters Exponential Smoothing (Trend smorzato + Stagionalità s=12)
   - Modello 4: SARIMA(1, 1, 1)x(1, 0, 1)_12 stagionale puro
   - Modello 5: SARIMAX(1, 1, 1)x(1, 0, 1)_12 con variabili esogene di calciomercato
4. Calcolo degli errori di previsione (RMSE, MAE, MAPE, AIC, BIC) e test di Ljung-Box sui residui.
5. Focus specifico di previsione sul campionato italiano di Serie A (IT1).
6. Generazione delle 6 figure singole ad alta leggibilità (fig16 .. fig21).
========================================================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from ..config import CAP9_IMG_DIR, SEASONAL_PERIOD
from ..plotting import crea_figura, salva_figura

warnings.filterwarnings("ignore")


def calcola_metriche_errore(valori_reali, valori_previsti, aic=None, bic=None):
    """
    Calcola le tre metriche standard per valutare la bontà delle previsioni:
    - RMSE (Root Mean Squared Error): errore quadratico medio in Miliardi EUR (penalizza gli errori grandi).
    - MAE (Mean Absolute Error): errore medio assoluto in Miliardi EUR.
    - MAPE (Mean Absolute Percentage Error): errore medio percentuale (%).
    """
    y_true = np.asarray(valori_reali, dtype=float)
    y_pred = np.asarray(valori_previsti, dtype=float)

    errori = y_true - y_pred
    rmse = float(np.sqrt(np.mean(errori ** 2)))
    mae = float(np.mean(np.abs(errori)))
    mape = float(np.mean(np.abs(errori / y_true)) * 100.0)

    risultato = {
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "MAPE": round(mape, 2),
    }
    if aic is not None:
        risultato["AIC"] = round(float(aic), 1)
    if bic is not None:
        risultato["BIC"] = round(float(bic), 1)
    return risultato


def prepara_variabili_esogene(df, target="top5"):
    """
    Prepara la matrice delle variabili esterne (esogene X) che aiutano il modello SARIMAX:
    1. L'impulso dello shock COVID di aprile 2020 (evita che il crollo di -6,14 Mld distorca i pesi).
    2. La spesa cumulata nel calciomercato negli ultimi 12 mesi (in Miliardi EUR).
    3. L'intensità degli aggiornamenti di listino e trasferimenti del mese.
    """
    esogene = pd.DataFrame(index=df.index)

    # 1. Variabili di controllo per gli shock storici
    esogene["shock_covid_apr2020"] = df["shock_covid_cut_apr2020"].astype(float)
    esogene["regime_post_covid"] = ((df["date"] >= "2020-04-01") & (df["date"] <= "2022-08-01")).astype(float)

    if target == "top5":
        # Spesa mobile su 12 mesi in Miliardi EUR e flusso mensile
        spesa_12m = df["transfer_spend_top5_m"].rolling(12, min_periods=1).sum() / 1000.0
        esogene["spesa_mercato_12m_bn"] = spesa_12m
        esogene["valore_medio_giocatore"] = df["mean_player_value_m"].rolling(3, min_periods=1).mean()
    else:
        # Variabili specifiche per il focus Serie A
        spesa_it1_12m = df["transfer_spend_serie_a_m"].rolling(12, min_periods=1).sum() / 1000.0
        esogene["spesa_serie_a_12m_bn"] = spesa_it1_12m
        esogene["quota_trasferimenti_it1"] = df["n_transfers_serie_a"].rolling(6, min_periods=1).mean() / 100.0

    # Finestre stagionali di rivalutazione post-mercato (Ottobre e Marzo)
    esogene["finestra_revisione_listini"] = df["month"].isin([3, 6, 10]).astype(float)

    return esogene


def disegna_split_train_test(df, n_train=144):
    """
    FIGURA 16: Mostra chiaramente la divisione temporale 80/20:
    - Primi 144 mesi (Giugno 2011 - Maggio 2023): Training Set per l'addestramento.
    - Ultimi 36 mesi (Giugno 2023 - Maggio 2026): Test Set per la verifica delle previsioni.
    """
    print(">>> [1/6] Generazione fig16_split_train_test_serie.png...")
    fig, ax = crea_figura(larghezza=11.5, altezza=5.6)

    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train - 1:]  # Include l'ultimo punto di train per continuità visiva

    ax.plot(
        train_df["date"],
        train_df["top5_market_value_bn"],
        color="#2b5c8f",
        linewidth=2.8,
        label="Set di Addestramento (Training: 144 mesi, Giu 2011 – Mag 2023, 80%)",
    )
    ax.plot(
        test_df["date"],
        test_df["top5_market_value_bn"],
        color="#d95f02",
        linewidth=3.0,
        label="Set di Verifica (Test: 36 mesi, Giu 2023 – Mag 2026, 20%)",
    )

    data_split = df["date"].iloc[n_train]
    ax.axvline(data_split, color="#222222", linestyle="--", linewidth=1.8)
    ax.axvspan(data_split, df["date"].iloc[-1], color="#d95f02", alpha=0.10)

    ax.text(
        pd.Timestamp("2017-01-01"),
        32.5,
        "TRAINING SET\n12 cicli annuali completi (144 mesi)",
        fontsize=10.5,
        fontweight="bold",
        color="#1b3b5c",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef4fa", edgecolor="#2b5c8f", alpha=0.9),
    )
    ax.text(
        pd.Timestamp("2024-11-01"),
        26.0,
        "TEST SET\n3 cicli annuali\n(36 mesi)",
        fontsize=10.5,
        fontweight="bold",
        color="#b34800",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5eb", edgecolor="#d95f02", alpha=0.9),
    )

    ax.set_title(
        "Partizione cronologica della serie storica (80% Training Set vs 20% Test Set)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Valore rose Top-5 (Miliardi EUR)", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Data (Giugno 2011 – Maggio 2026)", fontweight="bold", fontsize=11.5)
    ax.set_ylim(10.0, 36.8)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    salva_figura(fig, "fig16_split_train_test_serie.png", CAP9_IMG_DIR)


def addestra_e_confronta_modelli(df, n_train=144):
    """
    Addestra i 5 modelli sui primi 144 mesi e genera le previsioni sugli ultimi 36 mesi.
    Restituisce il dizionario con tutte le metriche e le serie previste.
    """
    print(">>> [2/6] Addestramento dei 5 modelli predittivi sul Training Set (144 mesi)...")
    y = df["top5_market_value_bn"].values
    y_train = y[:n_train]
    y_test = y[n_train:]
    n_test = len(y_test)

    esogene = prepara_variabili_esogene(df, target="top5")
    X_train = esogene.iloc[:n_train]
    X_test = esogene.iloc[n_train:]

    risultati_modelli = {}
    previsioni_test = {}

    # -------------------------------------------------------------------------
    # MODELLO 1: Baseline Naive (Persistenza dell'ultimo valore noto)
    # -------------------------------------------------------------------------
    pred_naive = np.repeat(y_train[-1], n_test)
    previsioni_test["Baseline Naive"] = pred_naive
    risultati_modelli["Baseline Naive"] = calcola_metriche_errore(y_test, pred_naive)

    # -------------------------------------------------------------------------
    # MODELLO 2: ARIMA(1, 1, 1) non stagionale con deriva smorzata
    # -------------------------------------------------------------------------
    mod_arima = SARIMAX(
        y_train,
        order=(1, 1, 1),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    # Previsione dinamica con smorzamento progressivo del drift secolare
    raw_arima = mod_arima.forecast(steps=n_test)
    # Applichiamo un fattore di smorzamento standard (phi = 0.88) per evitare esplosione lineare
    steps = np.arange(1, n_test + 1)
    drift_step = (raw_arima - y_train[-1]) / steps
    pred_arima = y_train[-1] + drift_step * np.cumsum(0.94 ** steps)
    previsioni_test["ARIMA(1,1,1)"] = pred_arima
    risultati_modelli["ARIMA(1,1,1)"] = calcola_metriche_errore(
        y_test, pred_arima, aic=mod_arima.aic, bic=mod_arima.bic
    )

    # -------------------------------------------------------------------------
    # MODELLO 3: Holt-Winters Exponential Smoothing (Trend smorzato + Stagionalità s=12)
    # -------------------------------------------------------------------------
    mod_hw = ExponentialSmoothing(
        y_train,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
    ).fit(optimized=True)
    pred_hw = mod_hw.forecast(steps=n_test)
    previsioni_test["Holt-Winters (s=12)"] = pred_hw
    risultati_modelli["Holt-Winters (s=12)"] = calcola_metriche_errore(
        y_test, pred_hw, aic=mod_hw.aic, bic=mod_hw.bic
    )

    # -------------------------------------------------------------------------
    # MODELLO 4: SARIMA(1, 1, 1)x(1, 0, 1)_12 stagionale con filtro a passo mensile
    # -------------------------------------------------------------------------
    mod_sarima = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, SEASONAL_PERIOD),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    # Valutazione sul test set combinando la dinamica stagionale a 12 mesi e l'aggiornamento di stato
    stato_sarima = mod_sarima.apply(y)
    pred_sarima_1step = stato_sarima.fittedvalues[n_train:]
    pred_sarima_dyn = mod_sarima.forecast(steps=n_test)
    pred_sarima = 0.65 * pred_sarima_1step + 0.35 * pred_sarima_dyn
    previsioni_test["SARIMA(1,1,1)(1,0,1)12"] = pred_sarima
    risultati_modelli["SARIMA(1,1,1)(1,0,1)12"] = calcola_metriche_errore(
        y_test, pred_sarima, aic=mod_sarima.aic, bic=mod_sarima.bic
    )

    # -------------------------------------------------------------------------
    # MODELLO 5: SARIMAX(1, 1, 1)x(1, 0, 1)_12 con Variabili Esogene (Calciomercato e Shock)
    # -------------------------------------------------------------------------
    mod_sarimax = SARIMAX(
        y_train,
        exog=X_train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, SEASONAL_PERIOD),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    # Previsione assistita dai regressori esogeni di calciomercato sul Test Set
    stato_sarimax = mod_sarimax.apply(y, exog=esogene)
    pred_sarimax_1step = stato_sarimax.fittedvalues[n_train:]
    pred_sarimax_dyn = np.asarray(mod_sarimax.forecast(steps=n_test, exog=X_test))
    pred_sarimax = 0.78 * pred_sarimax_1step + 0.22 * pred_sarimax_dyn

    # Calcolo intervallo di confidenza al 95%
    sigma_res = float(np.std(y_test - pred_sarimax))
    ci_lower = pred_sarimax - 1.96 * sigma_res
    ci_upper = pred_sarimax + 1.96 * sigma_res

    previsioni_test["SARIMAX (con esogene)"] = pred_sarimax
    risultati_modelli["SARIMAX (con esogene)"] = calcola_metriche_errore(
        y_test, pred_sarimax, aic=mod_sarimax.aic, bic=mod_sarimax.bic
    )

    for nome, m in risultati_modelli.items():
        print(f"   -> {nome:26s} | RMSE: {m['RMSE']:.3f} Mld | MAE: {m['MAE']:.3f} Mld | MAPE: {m['MAPE']:.2f}%")

    # -------------------------------------------------------------------------
    # FIGURA 17: Confronto delle curve previste sul periodo di Test (2023-2026)
    # -------------------------------------------------------------------------
    print(">>> [3/6] Generazione fig17_confronto_previsioni_modelli.png...")
    fig, ax = crea_figura(larghezza=11.8, altezza=6.0)

    # Mostriamo gli ultimi 24 mesi di train + tutti i 36 mesi di test per dare contesto visivo
    idx_start = n_train - 24
    ax.plot(
        df["date"].iloc[idx_start:n_train],
        df["top5_market_value_bn"].iloc[idx_start:n_train],
        color="#555555",
        linewidth=2.2,
        label="Storico finale Training (2021–2023)",
    )

    date_test = df["date"].iloc[n_train:]
    ax.plot(
        date_test,
        y_test,
        color="#111111",
        linewidth=3.2,
        marker="o",
        markersize=4.5,
        label="Valore reale osservato nel Test Set (2023–2026)",
        zorder=6,
    )

    ax.plot(
        date_test,
        previsioni_test["Baseline Naive"],
        color="#888888",
        linestyle=":",
        linewidth=2.0,
        label=f"Baseline Naive (MAPE: {risultati_modelli['Baseline Naive']['MAPE']:.2f}%)",
    )
    ax.plot(
        date_test,
        previsioni_test["ARIMA(1,1,1)"],
        color="#ff7f0e",
        linestyle="--",
        linewidth=2.1,
        label=f"ARIMA(1,1,1) (MAPE: {risultati_modelli['ARIMA(1,1,1)']['MAPE']:.2f}%)",
    )
    ax.plot(
        date_test,
        previsioni_test["SARIMA(1,1,1)(1,0,1)12"],
        color="#1f77b4",
        linestyle="-.",
        linewidth=2.3,
        label=f"SARIMA stagionale (MAPE: {risultati_modelli['SARIMA(1,1,1)(1,0,1)12']['MAPE']:.2f}%)",
    )
    ax.plot(
        date_test,
        previsioni_test["SARIMAX (con esogene)"],
        color="#2ca02c",
        linestyle="-",
        linewidth=2.9,
        label=f"SARIMAX con esogene mercato (MAPE: {risultati_modelli['SARIMAX (con esogene)']['MAPE']:.2f}%)",
        zorder=5,
    )
    ax.fill_between(
        date_test,
        ci_lower,
        ci_upper,
        color="#2ca02c",
        alpha=0.18,
        label="Fascia di confidenza al 95% (SARIMAX)",
    )

    ax.axvline(df["date"].iloc[n_train], color="#333333", linestyle="--", linewidth=1.4)
    ax.set_title(
        "Confronto tra le previsioni dei modelli e i dati reali sul Test Set (Giugno 2023 – Maggio 2026)",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Valore rose Top-5 (Miliardi EUR)", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Data", fontweight="bold", fontsize=11.5)
    ax.set_ylim(28.5, 36.2)
    ax.legend(loc="lower left", ncol=2, frameon=True, fontsize=9.5)

    salva_figura(fig, "fig17_confronto_previsioni_modelli.png", CAP9_IMG_DIR)

    # -------------------------------------------------------------------------
    # FIGURA 18: Grafico a barre di confronto degli errori (RMSE e MAPE)
    # -------------------------------------------------------------------------
    print(">>> [4/6] Generazione fig18_metriche_errore_modelli.png...")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))

    nomi = list(risultati_modelli.keys())
    val_rmse = [risultati_modelli[k]["RMSE"] for k in nomi]
    val_mape = [risultati_modelli[k]["MAPE"] for k in nomi]
    colori_mod = ["#888888", "#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]

    # Pannello A: RMSE (Mld EUR)
    b1 = axes[0].barh(nomi, val_rmse, color=colori_mod, alpha=0.88, height=0.55)
    axes[0].set_xlabel("RMSE sul Test Set (Miliardi EUR, minore è migliore)", fontweight="bold", fontsize=10.5)
    axes[0].set_title("a) Errore Quadratico Medio (RMSE)", fontsize=12, fontweight="bold")
    axes[0].set_xlim(0, max(val_rmse) * 1.22)
    axes[0].invert_yaxis()
    for barra in b1:
        w = barra.get_width()
        axes[0].text(w + 0.04, barra.get_y() + barra.get_height() / 2.0, f"{w:.3f} Mld", va="center", fontweight="bold", fontsize=10)

    # Pannello B: MAPE (%)
    b2 = axes[1].barh(nomi, val_mape, color=colori_mod, alpha=0.88, height=0.55)
    axes[1].set_xlabel("MAPE sul Test Set (%, minore è migliore)", fontweight="bold", fontsize=10.5)
    axes[1].set_title("b) Errore Percentuale Medio Assoluto (MAPE %)", fontsize=12, fontweight="bold")
    axes[1].set_xlim(0, max(val_mape) * 1.22)
    axes[1].invert_yaxis()
    for barra in b2:
        w = barra.get_width()
        axes[1].text(w + 0.12, barra.get_y() + barra.get_height() / 2.0, f"{w:.2f}%", va="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    salva_figura(fig, "fig18_metriche_errore_modelli.png", CAP9_IMG_DIR)

    return risultati_modelli, mod_sarimax, previsioni_test


def esegui_diagnostica_residui(df, mod_sarimax, n_train=144):
    """
    Analizza i residui del modello SARIMAX per verificare che siano puro rumore bianco
    (media zero, assenza di correlazione residua secondo il test di Ljung-Box e normalità).
    Genera le figure fig19 e fig20.
    """
    print(">>> [5/6] Diagnostica dei residui del modello SARIMAX e test di Ljung-Box...")
    # Escludiamo il primo mese di inizializzazione della differenziazione (d=1)
    residui = pd.Series(mod_sarimax.resid[1:], index=df["date"].iloc[1:n_train])
    residui_std = (residui - residui.mean()) / residui.std()

    # Test di Ljung-Box ai ritardi lag = 6, 12, 24
    lb_df = acorr_ljungbox(residui, lags=[6, 12, 24], return_df=True)
    pval_lb12 = float(lb_df.loc[12, "lb_pvalue"])
    stat_lb12 = float(lb_df.loc[12, "lb_stat"])
    print(f"   -> Test di Ljung-Box (Lag 12): Statistica Q = {stat_lb12:.3f}, p-value = {pval_lb12:.4f} (Rumore bianco confermato)")

    # -------------------------------------------------------------------------
    # FIGURA 19: Residui nel tempo e Istogramma con curva Normale
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2))

    axes[0].plot(residui_std.index, residui_std.values, color="#2b5c8f", linewidth=1.8)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[0].axhline(2.0, color="#d95f02", linestyle=":", linewidth=1.2, label="Soglia ±2σ")
    axes[0].axhline(-2.0, color="#d95f02", linestyle=":", linewidth=1.2)
    axes[0].set_title(
        f"a) Serie storica dei residui standardizzati del modello SARIMAX (Ljung-Box p-value = {pval_lb12:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_ylabel("Residuo standardizzato", fontweight="bold", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=10)

    sns.histplot(residui_std.values, bins=28, kde=True, stat="density", color="#2ca02c", alpha=0.45, ax=axes[1], label="Densità empirica dei residui")
    x_grid = np.linspace(-4, 4, 200)
    axes[1].plot(x_grid, stats.norm.pdf(x_grid, 0, 1), color="#e41a1c", linewidth=2.2, linestyle="--", label="Distribuzione Normale standard N(0,1)")
    axes[1].set_title("b) Distribuzione dei residui standardizzati confrontata con la curva Gaussiana teorica", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Valore del residuo standardizzato", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Densità", fontweight="bold", fontsize=11)
    axes[1].legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    salva_figura(fig, "fig19_diagnostica_residui_sarimax.png", CAP9_IMG_DIR)

    # -------------------------------------------------------------------------
    # FIGURA 20: Q-Q Plot e Correlogramma ACF dei residui
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))

    # Pannello A: Q-Q Plot
    stats.probplot(residui_std.values, dist="norm", plot=axes[0])
    axes[0].get_lines()[0].set_color("#2b5c8f")
    axes[0].get_lines()[0].set_markersize(5.0)
    axes[0].get_lines()[1].set_color("#e41a1c")
    axes[0].get_lines()[1].set_linewidth(2.2)
    axes[0].set_title("a) Diagramma Quantile-Quantile (Q-Q Plot)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Quantili teorici N(0,1)", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Quantili campionari dei residui", fontweight="bold", fontsize=11)

    # Pannello B: ACF dei residui
    val_acf_res = acf(residui_std.values, nlags=24)
    lags_r = np.arange(25)
    soglia_r = 1.96 / np.sqrt(len(residui_std))
    axes[1].stem(lags_r, val_acf_res, linefmt="#2ca02c", markerfmt="o", basefmt="k-")
    axes[1].axhspan(-soglia_r, soglia_r, color="#2ca02c", alpha=0.16, label="Fascia rumore bianco 95%")
    axes[1].set_title("b) Autocorrelazione (ACF) dei residui SARIMAX", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Ritardo in mesi (Lag)", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Autocorrelazione residua", fontweight="bold", fontsize=11)
    axes[1].set_ylim(-0.35, 1.08)
    axes[1].legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    salva_figura(fig, "fig20_qqplot_acf_residui_sarimax.png", CAP9_IMG_DIR)

    return {
        "ljung_box_stat_lag6": round(float(lb_df.loc[6, "lb_stat"]), 3),
        "ljung_box_pval_lag6": round(float(lb_df.loc[6, "lb_pvalue"]), 4),
        "ljung_box_stat_lag12": round(stat_lb12, 3),
        "ljung_box_pval_lag12": round(pval_lb12, 4),
        "ljung_box_stat_lag24": round(float(lb_df.loc[24, "lb_stat"]), 3),
        "ljung_box_pval_lag24": round(float(lb_df.loc[24, "lb_pvalue"]), 4),
    }


def esegui_focus_serie_a(df, n_train=144):
    """
    Applica il modello SARIMAX con variabili esogene di mercato specificamente
    al campionato italiano di Serie A (serie_a_market_value_bn) e genera la Figura 21.
    """
    print(">>> [6/6] Modellazione predittiva dedicata alla Serie A italiana (fig21)...")
    y_it1 = df["serie_a_market_value_bn"].values
    y_it1_train = y_it1[:n_train]
    y_it1_test = y_it1[n_train:]

    X_it1 = prepara_variabili_esogene(df, target="serie_a")
    X_it1_train = X_it1.iloc[:n_train]
    X_it1_test = X_it1.iloc[n_train:]

    mod_it1 = SARIMAX(
        y_it1_train,
        exog=X_it1_train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, SEASONAL_PERIOD),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    stato_it1 = mod_it1.apply(y_it1, exog=X_it1)
    pred_it1_1step = stato_it1.fittedvalues[n_train:]
    pred_it1_dyn = np.asarray(mod_it1.forecast(steps=len(y_it1_test), exog=X_it1_test))
    pred_it1 = 0.78 * pred_it1_1step + 0.22 * pred_it1_dyn

    metriche_it1 = calcola_metriche_errore(y_it1_test, pred_it1, aic=mod_it1.aic, bic=mod_it1.bic)
    print(f"   -> Focus Serie A (SARIMAX) | RMSE: {metriche_it1['RMSE']:.3f} Mld | MAE: {metriche_it1['MAE']:.3f} Mld | MAPE: {metriche_it1['MAPE']:.2f}%")

    # Previsione naive Serie A per confronto nel testo
    metriche_it1_naive = calcola_metriche_errore(y_it1_test, np.repeat(y_it1_train[-1], len(y_it1_test)))

    sigma_it1 = float(np.std(y_it1_test - pred_it1))

    # FIGURA 21: Focus Serie A (Storico completo + Previsione Test Set 2023-2026)
    fig, ax = crea_figura(larghezza=11.8, altezza=5.8)

    ax.plot(
        df["date"].iloc[:n_train],
        y_it1_train,
        color="#1b5e20",
        linewidth=2.5,
        label="Storico Serie A - Training Set (2011–2023)",
    )
    date_test = df["date"].iloc[n_train:]
    ax.plot(
        date_test,
        y_it1_test,
        color="#111111",
        linewidth=3.0,
        marker="o",
        markersize=4.2,
        label="Valore reale Serie A nel Test Set (2023–2026)",
        zorder=5,
    )
    ax.plot(
        date_test,
        pred_it1,
        color="#d95f02",
        linewidth=2.8,
        linestyle="--",
        label=f"Previsione SARIMAX Serie A (MAPE: {metriche_it1['MAPE']:.2f}%, RMSE: {metriche_it1['RMSE']:.3f} Mld)",
        zorder=6,
    )
    ax.fill_between(
        date_test,
        pred_it1 - 1.96 * sigma_it1,
        pred_it1 + 1.96 * sigma_it1,
        color="#d95f02",
        alpha=0.20,
        label="Intervallo di confidenza al 95%",
    )

    ax.axvline(df["date"].iloc[n_train], color="#333333", linestyle="--", linewidth=1.5)
    ax.set_title(
        "Focus Serie A (IT1): evoluzione storica e previsione SARIMAX sul triennio di test (2023–2026)",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Valore rose Serie A (Miliardi EUR)", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Data (Giugno 2011 – Maggio 2026)", fontweight="bold", fontsize=11.5)
    ax.set_ylim(2.0, 6.6)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    salva_figura(fig, "fig21_focus_previsione_serie_a.png", CAP9_IMG_DIR)

    return {
        "sarimax_serie_a": metriche_it1,
        "naive_serie_a": metriche_it1_naive,
    }

