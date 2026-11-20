"""
========================================================================================
Modulo di Modellazione Predittiva: Baseline, Holt-Winters, SARIMA e SARIMAX
========================================================================================
Funzioni per la Sezione 4 (Capitolo 9) della Relazione:
1. Divisione cronologica Training Set (120 mesi, Giu 2013 – Mag 2023) vs Test Set
   (36 mesi, Giu 2023 – Mag 2026) -> fig16.
2. Addestramento e confronto di 7 modelli sul mercato Top-5 con ablazione PURA a
   parita' di ordine parsimonioso (0,1,0)(1,0,0)_12 (100% convergente, privo di
   cancellazioni di radici AR/MA) -> fig17.
3. Valutazione multi-orizzonte a finestra espandibile (rolling-origin CV a h = 1, 6, 12
   mesi) su tutti i 7 modelli (incluso Seasonal Naive), con test di Diebold-Mariano
   (per confronti non annidati) e test di Clark-West (2007, per modelli annidati) -> fig18.
4. Diagnostica dei residui (Ljung-Box con correzione gradi di liberta' model_df,
   verifica dell'effetto di mascheramento dell'outlier COVID sui modelli senza dummy e
   test di normalita' Jarque-Bera) -> fig19, fig20.
5. Focus sulla Serie A (IT1) con gli stessi modelli parsimoniosi (0,1,0)(1,0,0)_12,
   holdout a 36 mesi + rolling-origin CV (h = 1, 6, 12) + diagnostica completa -> fig21.
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
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from ..config import CAP9_IMG_DIR, SEASONAL_PERIOD, N_TRAIN_MONTHS
from ..plotting import crea_figura, salva_figura


def calcola_metriche_errore(valori_reali, valori_previsti, aic=None, bic=None):
    """Calcola RMSE, MAE e MAPE (%), oltre ad AIC/BIC quando confrontabili."""
    y_true = np.asarray(valori_reali, dtype=float)
    y_pred = np.asarray(valori_previsti, dtype=float)
    errori = y_true - y_pred

    risultato = {
        "RMSE": round(float(np.sqrt(np.mean(errori ** 2))), 3),
        "MAE": round(float(np.mean(np.abs(errori))), 3),
        "MAPE": round(float(np.mean(np.abs(errori / y_true)) * 100.0), 2),
    }
    if aic is not None:
        risultato["AIC"] = round(float(aic), 1)
    if bic is not None:
        risultato["BIC"] = round(float(bic), 1)
    return risultato


def estrai_tabella_parametri(modello_fit):
    """Estrae coefficienti, errori standard, statistiche z e p-value da un SARIMAX stimato."""
    out = {}
    for nome, val, se, z, pv in zip(
        modello_fit.param_names,
        modello_fit.params,
        modello_fit.bse,
        modello_fit.zvalues,
        modello_fit.pvalues,
    ):
        out[str(nome)] = {
            "coef": round(float(val), 4),
            "se": round(float(se), 4),
            "z": round(float(z), 2),
            "pval": round(float(pv), 4),
        }
    return out


def prepara_variabili_esogene(df, target="top5"):
    """
    Costruisce la matrice delle variabili esogene X per i modelli SARIMAX:
    1. step_covid_apr2020: gradino deterministico da aprile 2020 in poi. Dopo la
       differenziazione d=1 nella formulazione RegARIMA diventa un impulso unitario
       esatto su aprile 2020, senza rimbalzi artificiali nei mesi successivi.
    2. spesa_acquisti_12m_bn: spesa cumulata negli ultimi 12 mesi per soli ACQUISTI
       (trasferimenti in entrata, calcolata in ETL sulla griglia estesa 2011–2026 senza
       distorsioni di avvio). Sul Test Set la valutazione e' condizionata ex-post.
    """
    esogene = pd.DataFrame(index=df.index)
    esogene["step_covid_apr2020"] = df["step_covid_apr2020"].astype(float)
    if target == "top5":
        esogene["spesa_acquisti_12m_bn"] = df["transfer_spend_top5_12m_bn"].astype(float)
    else:
        esogene["spesa_acquisti_12m_bn"] = df["transfer_spend_serie_a_12m_bn"].astype(float)
    return esogene


def _newey_west_var(d, h):
    """Stima della varianza di lungo periodo con pesi di Bartlett (troncamento a h-1)."""
    n = len(d)
    d_mean = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    for k in range(1, h):
        w = 1.0 - k / float(h)
        cov_k = float(np.sum((d[k:] - d_mean) * (d[:-k] - d_mean)) / (n - 1))
        var_d += 2.0 * w * cov_k
    return var_d


def test_diebold_mariano(e1, e2, h=1):
    """
    Test di Diebold-Mariano (con correzione Harvey-Leybourne-Newbold per campioni finiti)
    sulla differenza di errore quadratico d_t = e1_t^2 - e2_t^2.
    Indicato per modelli NON annidati. DM < 0 favorisce il modello 1.
    """
    d = np.asarray(e1, dtype=float) ** 2 - np.asarray(e2, dtype=float) ** 2
    n = len(d)
    d_mean = float(np.mean(d))
    var_d = _newey_west_var(d, h)
    if var_d <= 0:
        return {"dm_stat": 0.0, "pval": 1.0, "n": int(n)}
    dm = d_mean / np.sqrt(var_d / n)
    hln = np.sqrt((n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n)
    dm_hln = dm * hln
    pval = float(2.0 * stats.t.sf(abs(dm_hln), df=max(1, n - 1)))
    return {"dm_stat": round(float(dm_hln), 3), "pval": round(pval, 4), "n": int(n)}


def test_clark_west(y_true, yhat_restricted, yhat_unrestricted, h=1):
    """
    Test MSPE-adjusted di Clark & West (2007) per modelli ANNIDATI (nested):
    confronta un modello parsimonioso (restricted, 1) con una sua estensione (unrestricted, 2).
    f_t = e1_t^2 - [e2_t^2 - (yhat1_t - yhat2_t)^2].
    H0: il modello esteso NON migliora il modello parsimonioso;
    H1 (unilaterale destro): il modello esteso migliora il modello parsimonioso (CW > 0).
    """
    yt = np.asarray(y_true, dtype=float)
    y1 = np.asarray(yhat_restricted, dtype=float)
    y2 = np.asarray(yhat_unrestricted, dtype=float)
    e1 = yt - y1
    e2 = yt - y2
    f = (e1 ** 2) - (e2 ** 2 - (y1 - y2) ** 2)
    n = len(f)
    f_mean = float(np.mean(f))
    var_f = _newey_west_var(f, h)
    if var_f <= 0:
        return {"cw_stat": 0.0, "pval_one_sided": 0.5, "n": int(n)}
    cw = f_mean / np.sqrt(var_f / n)
    pval_one_sided = float(stats.norm.sf(cw))
    return {"cw_stat": round(float(cw), 3), "pval_one_sided": round(pval_one_sided, 4), "n": int(n)}


def disegna_split_train_test(df, n_train=N_TRAIN_MONTHS):
    """FIGURA 16: Partizione cronologica Training Set (120 mesi) vs Test Set (36 mesi)."""
    print(">>> [1/6] Generazione fig16_split_train_test_serie.png...")
    fig, ax = crea_figura(larghezza=11.5, altezza=5.6)

    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train - 1:]
    n_test = len(df) - n_train

    ax.plot(
        train_df["date"], train_df["top5_market_value_bn"],
        color="#2b5c8f", linewidth=2.8,
        label=f"Training Set ({n_train} mesi, Giu 2013 – Mag 2023, {n_train/len(df)*100:.1f}%)",
    )
    ax.plot(
        test_df["date"], test_df["top5_market_value_bn"],
        color="#d95f02", linewidth=3.0,
        label=f"Test Set ({n_test} mesi, Giu 2023 – Mag 2026, {n_test/len(df)*100:.1f}%)",
    )

    data_split = df["date"].iloc[n_train]
    ax.axvline(data_split, color="#222222", linestyle="--", linewidth=1.8)
    ax.axvspan(data_split, df["date"].iloc[-1], color="#d95f02", alpha=0.10)

    ax.text(
        pd.Timestamp("2018-01-01"), 31.8,
        f"TRAINING SET\n10 cicli annuali completi ({n_train} mesi)",
        fontsize=10.5, fontweight="bold", color="#1b3b5c", ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef4fa", edgecolor="#2b5c8f", alpha=0.9),
    )
    ax.text(
        pd.Timestamp("2024-11-01"), 23.0,
        f"TEST SET\n3 cicli annuali\n({n_test} mesi)",
        fontsize=10.5, fontweight="bold", color="#b34800", ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5eb", edgecolor="#d95f02", alpha=0.9),
    )

    ax.set_title("Partizione cronologica della serie storica (Training Set vs Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Valore rose Top-5 (Miliardi EUR)", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Data (Giugno 2013 – Maggio 2026)", fontweight="bold", fontsize=11.5)
    ax.set_ylim(10.0, 35.5)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    salva_figura(fig, "fig16_split_train_test_serie.png", CAP9_IMG_DIR)


def _fit_sarimax(y_tr, order, seasonal_order=(0, 0, 0, 0), exog_tr=None):
    """
    Stima SARIMAX con vincoli di stazionarieta'/invertibilita' attivi, cov_type='oim' e
    verifica esplicita di convergenza MLE (fallback automatico su powell / nm se lbfgs
    non converge). Non nasconde eventuali mancate convergenze.
    """
    mod = SARIMAX(
        y_tr,
        exog=exog_tr,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    tentativi = [("lbfgs", 300), ("powell", 500), ("nm", 800)]
    res = None
    for meth, maxit in tentativi:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            res = mod.fit(method=meth, maxiter=maxit, disp=False, cov_type="oim")
        if bool(res.mle_retvals.get("converged", False)):
            return res
    print(f"   [ATTENZIONE] Mancata convergenza per ordine {order}x{seasonal_order}!")
    return res


def addestra_e_confronta_modelli(df, n_train=N_TRAIN_MONTHS):
    """
    Addestra i 7 modelli sul Training Set (120 mesi) e valuta le previsioni multi-step
    sul Test Set (36 mesi) con ablazione PURA a parita' di ordine (0,1,0)(1,0,0)_12.
    Verifica inoltre: scala logaritmica, ordine AR(2) stagionale e spesa ritardata ex-ante.
    Genera la Figura 17.
    """
    print(f">>> [2/6] Addestramento dei 7 modelli sul Training Set ({n_train} mesi)...")
    y = df["top5_market_value_bn"].values
    y_train, y_test = y[:n_train], y[n_train:]
    n_test = len(y_test)

    X = prepara_variabili_esogene(df, target="top5")
    X_tr_cov, X_te_cov = X[["step_covid_apr2020"]].iloc[:n_train], X[["step_covid_apr2020"]].iloc[n_train:]
    X_tr_full, X_te_full = X.iloc[:n_train], X.iloc[n_train:]

    risultati = {}
    previsioni = {}
    convergenza = {}

    # 1. Baseline Naive (persistenza)
    p_naive = np.repeat(y_train[-1], n_test)
    previsioni["Baseline Naive"] = p_naive
    risultati["Baseline Naive"] = calcola_metriche_errore(y_test, p_naive)

    # 2. Seasonal Naive (s = 12)
    p_snaive = np.tile(y_train[-SEASONAL_PERIOD:], int(np.ceil(n_test / SEASONAL_PERIOD)))[:n_test]
    previsioni["Seasonal Naive (s=12)"] = p_snaive
    risultati["Seasonal Naive (s=12)"] = calcola_metriche_errore(y_test, p_snaive)

    # 3. Random Walk con drift = ARIMA(0,1,0) + c
    mod_rwd = _fit_sarimax(y_train, order=(0, 1, 0))
    convergenza["Random Walk + drift"] = bool(mod_rwd.mle_retvals.get("converged", False))
    p_rwd = np.asarray(mod_rwd.forecast(steps=n_test))
    previsioni["Random Walk + drift"] = p_rwd
    risultati["Random Walk + drift"] = calcola_metriche_errore(y_test, p_rwd, aic=mod_rwd.aic, bic=mod_rwd.bic)

    # 4. Holt-Winters additivo con trend smorzato (s = 12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod_hw = ExponentialSmoothing(
            y_train, trend="add", damped_trend=True, seasonal="add", seasonal_periods=SEASONAL_PERIOD
        ).fit(optimized=True)
    p_hw = np.asarray(mod_hw.forecast(steps=n_test))
    previsioni["Holt-Winters (s=12)"] = p_hw
    risultati["Holt-Winters (s=12)"] = calcola_metriche_errore(y_test, p_hw)

    # 5. SARIMA(0,1,0)(1,0,0)_12 (identificato da ACF/PACF e criteri informativi)
    mod_sarima = _fit_sarimax(y_train, order=(0, 1, 0), seasonal_order=(1, 0, 0, SEASONAL_PERIOD))
    convergenza["SARIMA(0,1,0)(1,0,0)12"] = bool(mod_sarima.mle_retvals.get("converged", False))
    p_sarima = np.asarray(mod_sarima.forecast(steps=n_test))
    previsioni["SARIMA(0,1,0)(1,0,0)12"] = p_sarima
    risultati["SARIMA(0,1,0)(1,0,0)12"] = calcola_metriche_errore(
        y_test, p_sarima, aic=mod_sarima.aic, bic=mod_sarima.bic
    )

    # 6. SARIMAX(0,1,0)(1,0,0)_12 + solo step COVID (ABLAZIONE PURA a parita' di ordine!)
    mod_cov = _fit_sarimax(y_train, order=(0, 1, 0), seasonal_order=(1, 0, 0, SEASONAL_PERIOD), exog_tr=X_tr_cov)
    convergenza["SARIMAX + COVID"] = bool(mod_cov.mle_retvals.get("converged", False))
    fc_cov = mod_cov.get_forecast(steps=n_test, exog=X_te_cov)
    p_cov = np.asarray(fc_cov.predicted_mean)
    previsioni["SARIMAX + COVID"] = p_cov
    risultati["SARIMAX + COVID"] = calcola_metriche_errore(y_test, p_cov, aic=mod_cov.aic, bic=mod_cov.bic)

    # 7. SARIMAX(0,1,0)(1,0,0)_12 + step COVID + Spesa acquisti 12m (ABLAZIONE PURA ex-post)
    mod_full = _fit_sarimax(y_train, order=(0, 1, 0), seasonal_order=(1, 0, 0, SEASONAL_PERIOD), exog_tr=X_tr_full)
    convergenza["SARIMAX + COVID + Spesa 12m"] = bool(mod_full.mle_retvals.get("converged", False))
    fc_full = mod_full.get_forecast(steps=n_test, exog=X_te_full)
    p_full = np.asarray(fc_full.predicted_mean)
    ci_full = np.asarray(fc_full.conf_int(alpha=0.05))
    previsioni["SARIMAX + COVID + Spesa 12m"] = p_full
    risultati["SARIMAX + COVID + Spesa 12m"] = calcola_metriche_errore(
        y_test, p_full, aic=mod_full.aic, bic=mod_full.bic
    )

    # Verifiche secondarie di robustezza:
    # (a) scala logaritmica, (b) spesa ritardata ex-ante di 1 mese, (c) SARIMAX(2,1,0)(1,0,0)12 (per lag 2)
    ly_tr = np.log(y_train)
    mod_log_sarima = _fit_sarimax(ly_tr, order=(0, 1, 0), seasonal_order=(1, 0, 0, SEASONAL_PERIOD))
    p_log_sarima = np.exp(np.asarray(mod_log_sarima.forecast(steps=n_test)))

    X_lag1 = pd.DataFrame({
        "step_covid_apr2020": df["step_covid_apr2020"].astype(float),
        "spesa_12m_lag1": df["transfer_spend_top5_12m_bn"].shift(1).bfill().astype(float),
    })
    mod_lag1 = _fit_sarimax(y_train, order=(0, 1, 0), seasonal_order=(1, 0, 0, SEASONAL_PERIOD), exog_tr=X_lag1.iloc[:n_train])
    p_lag1 = np.asarray(mod_lag1.forecast(steps=n_test, exog=X_lag1.iloc[n_train:]))
    # Rolling-origin CV strettamente ex-ante a h=1 per spesa ritardata di 1 mese
    errs_lag1_h1 = []
    for t0 in range(n_train, len(y)):
        m_k = _fit_sarimax(y[:t0], order=(0, 1, 0), seasonal_order=(1, 0, 0, SEASONAL_PERIOD), exog_tr=X_lag1.iloc[:t0])
        pred_k = float(np.asarray(m_k.forecast(steps=1, exog=X_lag1.iloc[t0:t0 + 1]))[0])
        errs_lag1_h1.append(y[t0] - pred_k)
    cv_h1_rmse_lag1 = round(float(np.sqrt(np.mean(np.square(errs_lag1_h1)))), 3)

    mod_ar2 = _fit_sarimax(y_train, order=(2, 1, 0), seasonal_order=(1, 0, 0, SEASONAL_PERIOD), exog_tr=X_tr_full)
    p_ar2 = np.asarray(mod_ar2.forecast(steps=n_test, exog=X_te_full))

    verifiche_robustezza = {
        "SARIMA_010_100_log": calcola_metriche_errore(y_test, p_log_sarima),
        "RW_drift_log": calcola_metriche_errore(
            y_test, np.exp(ly_tr[-1] + np.mean(np.diff(ly_tr)) * np.arange(1, n_test + 1))
        ),
        "SARIMAX_010_100_spesa_lag1_ex_ante": {
            **calcola_metriche_errore(y_test, p_lag1, aic=mod_lag1.aic, bic=mod_lag1.bic),
            "coef_spesa_lag1": round(float(mod_lag1.params.iloc[2]), 4),
            "pval_spesa_lag1": round(float(mod_lag1.pvalues.iloc[2]), 4),
            "cv_h1_rmse": cv_h1_rmse_lag1,
            "converged": bool(mod_lag1.mle_retvals.get("converged", False)),
        },
        "SARIMAX_210_100_covid_spesa": {
            **calcola_metriche_errore(y_test, p_ar2, aic=mod_ar2.aic, bic=mod_ar2.bic),
            "pval_spesa": round(float(mod_ar2.pvalues.iloc[2]), 4),
            "converged": bool(mod_ar2.mle_retvals.get("converged", False)),
        },
    }

    for nome, m in risultati.items():
        aic_s = f"AIC={m['AIC']:5.1f} BIC={m['BIC']:5.1f}" if "AIC" in m else "AIC=  --  BIC=  -- "
        conv_s = f"conv={convergenza[nome]}" if nome in convergenza else ""
        print(f"   -> {nome:28s} | RMSE={m['RMSE']:.3f} | MAE={m['MAE']:.3f} | MAPE={m['MAPE']:.2f}% | {aic_s} {conv_s}")

    # -------------------------------------------------------------------------
    # FIGURA 17: Confronto multi-step sul Test Set (2023–2026) con raccordo continuo
    # -------------------------------------------------------------------------
    print(">>> [3/6] Generazione fig17_confronto_previsioni_modelli.png...")
    fig, ax = crea_figura(larghezza=11.8, altezza=6.1)

    idx_start = n_train - 24
    ax.plot(
        df["date"].iloc[idx_start:n_train], df["top5_market_value_bn"].iloc[idx_start:n_train],
        color="#555555", linewidth=2.2, label="Storico finale Training (2021–2023)",
    )
    # Linea reale connessa dall'ultimo punto di training (n_train-1) per evitare stacco visivo
    ax.plot(
        df["date"].iloc[n_train - 1:], y[n_train - 1:],
        color="#111111", linewidth=2.6, zorder=6,
    )
    date_test = df["date"].iloc[n_train:]
    ax.plot(
        date_test, y_test, color="#111111", linewidth=3.0, marker="o", markersize=4.0,
        label="Valore reale osservato nel Test Set (2023–2026)", zorder=7,
    )

    stili = [
        ("Baseline Naive", "#888888", ":", 1.8),
        ("Random Walk + drift", "#ff7f0e", "--", 2.1),
        ("Holt-Winters (s=12)", "#9467bd", (0, (3, 1, 1, 1)), 2.0),
        ("SARIMA(0,1,0)(1,0,0)12", "#1f77b4", "-", 2.6),
        ("SARIMAX + COVID", "#e377c2", "-.", 2.2),
        ("SARIMAX + COVID + Spesa 12m", "#2ca02c", "-", 2.2),
    ]
    for nome, col, ls, lw in stili:
        ax.plot(
            date_test, previsioni[nome], color=col, linestyle=ls, linewidth=lw,
            label=f"{nome} (MAPE: {risultati[nome]['MAPE']:.2f}%)", zorder=5,
        )
    ax.fill_between(
        date_test, ci_full[:, 0], ci_full[:, 1], color="#2ca02c", alpha=0.14,
        label="Fascia 95% condizionata (SARIMAX + COVID + Spesa)",
    )
    ax.axvline(df["date"].iloc[n_train], color="#333333", linestyle="--", linewidth=1.4)
    ax.set_title("Confronto multi-step sul Test Set (Giugno 2023 – Maggio 2026)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Valore rose Top-5 (Miliardi EUR)", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Data", fontweight="bold", fontsize=11.5)
    ymin_17 = min(24.5, float(np.min(ci_full[:, 0])) - 0.3)
    ymax_17 = max(38.0, float(np.max(ci_full[:, 1])) + 0.5)
    ax.set_ylim(ymin_17, ymax_17)
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=8.3)
    salva_figura(fig, "fig17_confronto_previsioni_modelli.png", CAP9_IMG_DIR)

    # Drift medi (per spiegare la differenza tra modelli con e senza step COVID)
    d_ytr = np.diff(y_train)
    idx_apr20 = int(np.where(df["year_month"].iloc[1:n_train].values == "2020-04")[0][0])
    drift_info = {
        "mean_monthly_diff_train_raw_bn": round(float(np.mean(d_ytr)), 4),
        "mean_monthly_diff_train_excl_apr2020_bn": round(float(np.mean(np.delete(d_ytr, idx_apr20))), 4),
        "mean_monthly_diff_test_bn": round(float((y_test[-1] - y_train[-1]) / n_test), 4),
    }

    dettagli = {
        "convergenza_holdout": convergenza,
        "drift_comparison": drift_info,
        "loglikelihood_burn": {
            "RW_drift": int(mod_rwd.loglikelihood_burn),
            "SARIMA": int(mod_sarima.loglikelihood_burn),
            "SARIMAX_COVID": int(mod_cov.loglikelihood_burn),
            "SARIMAX_COVID_Spesa": int(mod_full.loglikelihood_burn),
        },
        "parametri_rwd": estrai_tabella_parametri(mod_rwd),
        "parametri_sarima": estrai_tabella_parametri(mod_sarima),
        "parametri_sarimax_covid": estrai_tabella_parametri(mod_cov),
        "parametri_sarimax_full": estrai_tabella_parametri(mod_full),
        "parametri_holt_winters": {
            "alpha": round(float(mod_hw.params["smoothing_level"]), 3),
            "beta": round(float(mod_hw.params["smoothing_trend"]), 4),
            "gamma": round(float(mod_hw.params["smoothing_seasonal"]), 3),
            "phi_damping": round(float(mod_hw.params["damping_trend"]), 3),
        },
        "verifiche_robustezza": verifiche_robustezza,
    }
    return risultati, mod_sarima, mod_hw, mod_cov, mod_full, dettagli


def valuta_rolling_origin(df, risultati_holdout, n_train=N_TRAIN_MONTHS, orizzonti=(1, 6, 12)):
    """
    Validazione a finestra espandibile (rolling-origin CV) su tutti i 7 modelli (incluso
    Seasonal Naive) lungo l'intero triennio di test (origini da n_train a len(df)-h) per
    orizzonti h = 1, 6, 12 mesi + test di Diebold-Mariano (non annidati) e Clark-West
    (annidati). Genera la Figura 18.
    """
    print(">>> [4/6] Validazione rolling-origin CV (h = 1, 6, 12 mesi) e Figura 18...")
    y = df["top5_market_value_bn"].values
    X = prepara_variabili_esogene(df, target="top5")
    X_cov = X[["step_covid_apr2020"]]

    failed_conv = {}

    def _fn_naive(t0, H):
        return np.repeat(y[t0 - 1], H), True

    def _fn_snaive(t0, H):
        return np.tile(y[t0 - SEASONAL_PERIOD:t0], int(np.ceil(H / SEASONAL_PERIOD)))[:H], True

    def _fn_rwd(t0, H):
        return y[t0 - 1] + np.mean(np.diff(y[:t0])) * np.arange(1, H + 1), True

    def _fn_hw(t0, H):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ExponentialSmoothing(
                y[:t0], trend="add", damped_trend=True, seasonal="add", seasonal_periods=SEASONAL_PERIOD
            ).fit(optimized=True)
        return np.asarray(m.forecast(H)), True

    def _fn_sarima(t0, H):
        m = _fit_sarimax(y[:t0], (0, 1, 0), (1, 0, 0, SEASONAL_PERIOD))
        return np.asarray(m.forecast(H)), bool(m.mle_retvals.get("converged", False))

    def _fn_cov(t0, H):
        m = _fit_sarimax(y[:t0], (0, 1, 0), (1, 0, 0, SEASONAL_PERIOD), exog_tr=X_cov.iloc[:t0])
        return np.asarray(m.forecast(H, exog=X_cov.iloc[t0:t0 + H])), bool(m.mle_retvals.get("converged", False))

    def _fn_full(t0, H):
        m = _fit_sarimax(y[:t0], (0, 1, 0), (1, 0, 0, SEASONAL_PERIOD), exog_tr=X.iloc[:t0])
        return np.asarray(m.forecast(H, exog=X.iloc[t0:t0 + H])), bool(m.mle_retvals.get("converged", False))

    modelli_cv = {
        "Baseline Naive": _fn_naive,
        "Seasonal Naive (s=12)": _fn_snaive,
        "Random Walk + drift": _fn_rwd,
        "Holt-Winters (s=12)": _fn_hw,
        "SARIMA(0,1,0)(1,0,0)12": _fn_sarima,
        "SARIMAX + COVID": _fn_cov,
        "SARIMAX + COVID + Spesa 12m": _fn_full,
    }

    rolling_summary = {}
    errori_raw = {nome: {h: [] for h in orizzonti} for nome in modelli_cv}
    preds_raw = {nome: {h: [] for h in orizzonti} for nome in modelli_cv}
    trues_raw = {h: [] for h in orizzonti}

    for t0 in range(n_train, len(y)):
        for h in orizzonti:
            if t0 + h <= len(y):
                trues_raw[h].append(y[t0 + h - 1])

    for nome, fn in modelli_cv.items():
        n_fail = 0
        for t0 in range(n_train, len(y)):
            h_ok = [h for h in orizzonti if t0 + h <= len(y)]
            if not h_ok:
                break
            fc, ok = fn(t0, max(h_ok))
            if not ok:
                n_fail += 1
            for h in h_ok:
                preds_raw[nome][h].append(fc[h - 1])
                errori_raw[nome][h].append(y[t0 + h - 1] - fc[h - 1])
        failed_conv[nome] = int(n_fail)
        rolling_summary[nome] = {}
        for h in orizzonti:
            e = np.asarray(errori_raw[nome][h], dtype=float)
            yt = np.asarray(trues_raw[h], dtype=float)
            rolling_summary[nome][f"h{h}"] = {
                "RMSE": round(float(np.sqrt(np.mean(e ** 2))), 3),
                "MAE": round(float(np.mean(np.abs(e))), 3),
                "MAPE": round(float(np.mean(np.abs(e / yt)) * 100.0), 2),
                "n_origini": int(len(e)),
            }
        r = rolling_summary[nome]
        print(f"   -> CV {nome:28s} | fail={n_fail:2d}/36 | h=1: {r['h1']['RMSE']:.3f} | h=6: {r['h6']['RMSE']:.3f} | h=12: {r['h12']['RMSE']:.3f}")

    # Test di accuratezza predittiva:
    # - Diebold-Mariano (HLN) per confronti NON annidati
    # - Clark-West (2007) per confronti ANNIDATI (nested)
    test_accuratezza = {
        "non_nested_diebold_mariano": {
            "HW_vs_RWdrift_h1": test_diebold_mariano(
                errori_raw["Holt-Winters (s=12)"][1], errori_raw["Random Walk + drift"][1], h=1
            ),
            "HW_vs_SARIMAX_COVID_h1": test_diebold_mariano(
                errori_raw["Holt-Winters (s=12)"][1], errori_raw["SARIMAX + COVID"][1], h=1
            ),
            "SARIMA_vs_HW_h12": test_diebold_mariano(
                errori_raw["SARIMA(0,1,0)(1,0,0)12"][12], errori_raw["Holt-Winters (s=12)"][12], h=12
            ),
        },
        "nested_clark_west": {
            "SARIMAX_Spesa_over_COVID_h1": test_clark_west(
                trues_raw[1], preds_raw["SARIMAX + COVID"][1], preds_raw["SARIMAX + COVID + Spesa 12m"][1], h=1
            ),
            "SARIMAX_Spesa_over_COVID_h6": test_clark_west(
                trues_raw[6], preds_raw["SARIMAX + COVID"][6], preds_raw["SARIMAX + COVID + Spesa 12m"][6], h=6
            ),
            "SARIMAX_COVID_over_RWdrift_h1": test_clark_west(
                trues_raw[1], preds_raw["Random Walk + drift"][1], preds_raw["SARIMAX + COVID"][1], h=1
            ),
            "SARIMA_over_Naive_h12": test_clark_west(
                trues_raw[12], preds_raw["Baseline Naive"][12], preds_raw["SARIMA(0,1,0)(1,0,0)12"][12], h=12
            ),
            "RWdrift_over_Naive_h12": test_clark_west(
                trues_raw[12], preds_raw["Baseline Naive"][12], preds_raw["Random Walk + drift"][12], h=12
            ),
        },
        "failed_convergences_cv": failed_conv,
    }

    # FIGURA 18: Pannello (a) Holdout 36m RMSE | Pannello (b) Rolling-origin RMSE a h=1, 6, 12
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.5))

    nomi_h = list(risultati_holdout.keys())
    val_rmse = [risultati_holdout[k]["RMSE"] for k in nomi_h]
    colori_h = ["#888888", "#bbbbbb", "#ff7f0e", "#9467bd", "#1f77b4", "#e377c2", "#2ca02c"]
    b1 = axes[0].barh(nomi_h, val_rmse, color=colori_h, alpha=0.88, height=0.58)
    axes[0].set_xlabel("RMSE sul Test Set a 36 mesi (Miliardi EUR)", fontweight="bold", fontsize=10.5)
    axes[0].set_title("a) Previsione multi-step 36 mesi (holdout unico)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlim(0, max(val_rmse) * 1.24)
    axes[0].invert_yaxis()
    for barra in b1:
        w = barra.get_width()
        axes[0].text(w + 0.05, barra.get_y() + barra.get_height() / 2.0, f"{w:.3f}", va="center", fontweight="bold", fontsize=9.5)

    nomi_cv = list(modelli_cv.keys())
    x_pos = np.arange(len(nomi_cv))
    w_bar = 0.25
    for i, (h, col_h, lbl_h) in enumerate([(1, "#2b5c8f", "h = 1 mese"), (6, "#d95f02", "h = 6 mesi"), (12, "#2ca02c", "h = 12 mesi")]):
        vals_h = [rolling_summary[m][f"h{h}"]["RMSE"] for m in nomi_cv]
        axes[1].bar(x_pos + (i - 1) * w_bar, vals_h, width=w_bar, color=col_h, alpha=0.88, label=lbl_h)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(
        ["Naive", "Seas.\nNaive", "RW+\ndrift", "Holt-W.", "SARIMA", "SARIMAX\nCOVID", "SARIMAX\n+Spesa"],
        fontsize=8.6, fontweight="bold",
    )
    axes[1].set_ylabel("RMSE medio rolling-origin (Miliardi EUR)", fontweight="bold", fontsize=10.5)
    axes[1].set_title("b) Validazione rolling-origin su più orizzonti (2023–2026)", fontsize=11.5, fontweight="bold")
    axes[1].set_ylim(0, max(rolling_summary[m]["h12"]["RMSE"] for m in nomi_cv) * 1.22)
    axes[1].legend(loc="upper right", frameon=True, fontsize=9.2)

    plt.tight_layout()
    salva_figura(fig, "fig18_metriche_errore_modelli.png", CAP9_IMG_DIR)
    return rolling_summary, test_accuratezza


def _diagnostica_singolo_modello(residui, dates_sub, model_df, burn_in=13):
    """
    Calcola Ljung-Box (con correzione model_df) ai lag 6, 12, 24 e Jarque-Bera,
    sia sui residui grezzi sia escludendo il singolo outlier di aprile 2020 (per
    smascherare l'eventuale effetto di mascheramento nei modelli senza dummy COVID).
    """
    r = np.asarray(residui, dtype=float)[burn_in:]
    r_std = (r - np.mean(r)) / np.std(r)
    lb = acorr_ljungbox(r_std, lags=[6, 12, 24], model_df=model_df, return_df=True)
    jb_stat, jb_pval = stats.jarque_bera(r_std)

    # Verifica escludendo il solo residuo di aprile 2020 (sostituito con la mediana)
    r_cl = r.copy()
    mask_apr = (pd.to_datetime(dates_sub) == pd.Timestamp("2020-04-01"))
    apr_z = None
    if np.any(mask_apr):
        idx_apr = int(np.where(mask_apr)[0][0])
        apr_z = round(float(r_std[idx_apr]), 2)
        r_cl[idx_apr] = float(np.median(r_cl))
    r_cl_std = (r_cl - np.mean(r_cl)) / np.std(r_cl)
    lb_cl = acorr_ljungbox(r_cl_std, lags=[6, 12, 24], model_df=model_df, return_df=True)

    return r_std, {
        "model_df": int(model_df),
        "n_residui": int(len(r_std)),
        "apr2020_resid_z": apr_z,
        "ljung_box": {
            f"lag{int(lag)}": {"Q": round(float(q), 3), "df": int(lag - model_df), "pval": round(float(p), 4)}
            for lag, q, p in zip([6, 12, 24], lb["lb_stat"], lb["lb_pvalue"])
        },
        "ljung_box_excl_apr2020": {
            f"lag{int(lag)}": {"Q": round(float(q), 3), "df": int(lag - model_df), "pval": round(float(p), 4)}
            for lag, q, p in zip([6, 12, 24], lb_cl["lb_stat"], lb_cl["lb_pvalue"])
        },
        "jarque_bera": {"stat": round(float(jb_stat), 2), "pval": float(f"{jb_pval:.2e}")},
    }


def esegui_diagnostica_residui(df, mod_sarima, mod_hw, mod_cov, mod_sarimax_full, n_train=N_TRAIN_MONTHS):
    """
    Esegue la diagnostica dei residui con correzione dei gradi di liberta' (model_df) per:
    - SARIMAX(0,1,0)(1,0,0)_12 + COVID + Spesa (model_df = 1),
    - SARIMAX(0,1,0)(1,0,0)_12 + COVID (model_df = 1),
    - SARIMA(0,1,0)(1,0,0)_12 (model_df = 1, raw e senza outlier COVID),
    - Holt-Winters smorzato (model_df = 0 / descrittivo, raw e senza outlier COVID).
    Genera le figure fig19 e fig20 sul modello SARIMAX(0,1,0)(1,0,0)_12 + COVID + Spesa.
    """
    print(">>> [5/6] Diagnostica residui (Ljung-Box con model_df, verifica outlier e Jarque-Bera)...")
    burn_in = 13
    date_resid = pd.to_datetime(df["date"].iloc[burn_in:n_train].values)

    r_std_full, diag_full = _diagnostica_singolo_modello(mod_sarimax_full.resid, date_resid, model_df=1, burn_in=burn_in)
    _, diag_cov = _diagnostica_singolo_modello(mod_cov.resid, date_resid, model_df=1, burn_in=burn_in)
    _, diag_sarima = _diagnostica_singolo_modello(mod_sarima.resid, date_resid, model_df=1, burn_in=burn_in)
    _, diag_hw = _diagnostica_singolo_modello(
        df["top5_market_value_bn"].values[:n_train] - mod_hw.fittedvalues, date_resid, model_df=0, burn_in=burn_in
    )

    for lbl, d in [
        ("SARIMAX+COVID+Spesa (df=1)", diag_full),
        ("SARIMAX+COVID       (df=1)", diag_cov),
        ("SARIMA puro         (df=1)", diag_sarima),
        ("Holt-Winters        (df=0)", diag_hw),
    ]:
        lb = d["ljung_box"]
        lbc = d["ljung_box_excl_apr2020"]
        print(
            f"   -> {lbl}: RAW LB6={lb['lag6']['pval']:.4f}, LB12={lb['lag12']['pval']:.4f}, LB24={lb['lag24']['pval']:.4f}"
            f" | NO-COV LB6={lbc['lag6']['pval']:.4f}, LB12={lbc['lag12']['pval']:.4f}, LB24={lbc['lag24']['pval']:.4f}"
            f" | JB={d['jarque_bera']['stat']:.1f}"
        )

    # FIGURA 19: Residui standardizzati nel tempo e distribuzione vs N(0,1)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2))
    axes[0].plot(date_resid, r_std_full, color="#2b5c8f", linewidth=1.8)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[0].axhline(2.0, color="#d95f02", linestyle=":", linewidth=1.3, label="Fascia ±2σ")
    axes[0].axhline(-2.0, color="#d95f02", linestyle=":", linewidth=1.3)
    axes[0].set_title(
        f"a) Residui standardizzati SARIMAX(0,1,0)(1,0,0)_12 (Ljung-Box lag 24, df=23: p = {diag_full['ljung_box']['lag24']['pval']:.3f})",
        fontsize=11.5, fontweight="bold",
    )
    axes[0].set_ylabel("Residuo standardizzato", fontweight="bold", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=10)

    sns.histplot(r_std_full, bins=24, kde=True, stat="density", color="#2ca02c", alpha=0.45, ax=axes[1], label="Densità empirica dei residui")
    x_grid = np.linspace(-4, 4, 200)
    axes[1].plot(x_grid, stats.norm.pdf(x_grid, 0, 1), color="#e41a1c", linewidth=2.2, linestyle="--", label="Normale standard N(0,1)")
    axes[1].set_title(
        f"b) Distribuzione dei residui vs curva Gaussiana (Jarque-Bera = {diag_full['jarque_bera']['stat']:.1f}, code pesanti)",
        fontsize=11.5, fontweight="bold",
    )
    axes[1].set_xlabel("Valore del residuo standardizzato", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Densità", fontweight="bold", fontsize=11)
    axes[1].legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    salva_figura(fig, "fig19_diagnostica_residui_sarimax.png", CAP9_IMG_DIR)

    # FIGURA 20: Q-Q Plot e ACF dei residui SARIMAX
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    stats.probplot(r_std_full, dist="norm", plot=axes[0])
    axes[0].get_lines()[0].set_color("#2b5c8f")
    axes[0].get_lines()[0].set_markersize(5.0)
    axes[0].get_lines()[1].set_color("#e41a1c")
    axes[0].get_lines()[1].set_linewidth(2.2)
    axes[0].set_title("a) Diagramma Quantile-Quantile (Q-Q Plot)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Quantili teorici N(0,1)", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Quantili campionari dei residui", fontweight="bold", fontsize=11)

    val_acf_res = acf(r_std_full, nlags=24)[1:]
    lags_r = np.arange(1, 25)
    soglia_r = 1.96 / np.sqrt(len(r_std_full))
    axes[1].stem(lags_r, val_acf_res, linefmt="#2ca02c", markerfmt="o", basefmt="k-")
    axes[1].axhspan(-soglia_r, soglia_r, color="#2ca02c", alpha=0.16, label=f"Banda rumore bianco 95% (±{soglia_r:.3f})")
    axes[1].set_title("b) Autocorrelazione (ACF) dei residui SARIMAX", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Ritardo in mesi (lag)", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Autocorrelazione residua", fontweight="bold", fontsize=11)
    axes[1].set_ylim(-0.35, 0.40)
    axes[1].legend(loc="upper right", fontsize=9.8)
    plt.tight_layout()
    salva_figura(fig, "fig20_qqplot_acf_residui_sarimax.png", CAP9_IMG_DIR)

    return {
        "sarimax_covid_spesa": diag_full,
        "sarimax_covid": diag_cov,
        "sarima_010_100": diag_sarima,
        "holt_winters": diag_hw,
    }


def esegui_focus_serie_a(df, n_train=N_TRAIN_MONTHS, orizzonti=(1, 6, 12)):
    """
    Focus sulla Serie A italiana (IT1) con modello parsimonioso SARIMA/SARIMAX(0,1,0)(1,0,0)_12,
    confrontato con Baseline Naive, Random Walk + drift e ablazione COVID vs COVID + Spesa 12m,
    sia su holdout a 36 mesi sia in rolling-origin CV (h = 1, 6, 12). Genera la Figura 21.
    """
    print(">>> [6/6] Focus previsionale sulla Serie A italiana (holdout + rolling CV, fig21)...")
    y_it = df["serie_a_market_value_bn"].values
    y_tr, y_te = y_it[:n_train], y_it[n_train:]
    n_test = len(y_te)

    X_it = prepara_variabili_esogene(df, target="serie_a")
    X_tr_cov, X_te_cov = X_it[["step_covid_apr2020"]].iloc[:n_train], X_it[["step_covid_apr2020"]].iloc[n_train:]
    X_tr_full, X_te_full = X_it.iloc[:n_train], X_it.iloc[n_train:]

    p_naive = np.repeat(y_tr[-1], n_test)
    mod_rwd = _fit_sarimax(y_tr, order=(0, 1, 0))
    p_rwd = np.asarray(mod_rwd.forecast(steps=n_test))

    mod_sarima = _fit_sarimax(y_tr, order=(0, 1, 0), seasonal_order=(1, 0, 0, SEASONAL_PERIOD))
    p_sarima = np.asarray(mod_sarima.forecast(steps=n_test))

    mod_cov = _fit_sarimax(y_tr, order=(0, 1, 0), seasonal_order=(1, 0, 0, SEASONAL_PERIOD), exog_tr=X_tr_cov)
    p_cov = np.asarray(mod_cov.forecast(steps=n_test, exog=X_te_cov))

    mod_full = _fit_sarimax(y_tr, order=(0, 1, 0), seasonal_order=(1, 0, 0, SEASONAL_PERIOD), exog_tr=X_tr_full)
    fc_full = mod_full.get_forecast(steps=n_test, exog=X_te_full)
    p_full = np.asarray(fc_full.predicted_mean)
    ci_full = np.asarray(fc_full.conf_int(alpha=0.05))

    burn_in = 13
    date_resid = pd.to_datetime(df["date"].iloc[burn_in:n_train].values)
    _, diag_it_full = _diagnostica_singolo_modello(mod_full.resid, date_resid, model_df=1, burn_in=burn_in)
    _, diag_it_cov = _diagnostica_singolo_modello(mod_cov.resid, date_resid, model_df=1, burn_in=burn_in)

    metriche = {
        "Baseline Naive": calcola_metriche_errore(y_te, p_naive),
        "Random Walk + drift": calcola_metriche_errore(y_te, p_rwd, aic=mod_rwd.aic, bic=mod_rwd.bic),
        "SARIMA(0,1,0)(1,0,0)12": calcola_metriche_errore(y_te, p_sarima, aic=mod_sarima.aic, bic=mod_sarima.bic),
        "SARIMAX + COVID": calcola_metriche_errore(y_te, p_cov, aic=mod_cov.aic, bic=mod_cov.bic),
        "SARIMAX + COVID + Spesa 12m": calcola_metriche_errore(y_te, p_full, aic=mod_full.aic, bic=mod_full.bic),
    }

    # Rolling-origin CV anche sulla Serie A (h = 1, 6, 12)
    X_it_cov = X_it[["step_covid_apr2020"]]
    cv_fns = {
        "Baseline Naive": lambda t0, H: (np.repeat(y_it[t0 - 1], H), True),
        "Random Walk + drift": lambda t0, H: (y_it[t0 - 1] + np.mean(np.diff(y_it[:t0])) * np.arange(1, H + 1), True),
        "SARIMA(0,1,0)(1,0,0)12": lambda t0, H: (
            lambda m: (np.asarray(m.forecast(H)), bool(m.mle_retvals.get("converged", False)))
        )(_fit_sarimax(y_it[:t0], (0, 1, 0), (1, 0, 0, SEASONAL_PERIOD))),
        "SARIMAX + COVID": lambda t0, H: (
            lambda m: (np.asarray(m.forecast(H, exog=X_it_cov.iloc[t0:t0 + H])), bool(m.mle_retvals.get("converged", False)))
        )(_fit_sarimax(y_it[:t0], (0, 1, 0), (1, 0, 0, SEASONAL_PERIOD), exog_tr=X_it_cov.iloc[:t0])),
        "SARIMAX + COVID + Spesa 12m": lambda t0, H: (
            lambda m: (np.asarray(m.forecast(H, exog=X_it.iloc[t0:t0 + H])), bool(m.mle_retvals.get("converged", False)))
        )(_fit_sarimax(y_it[:t0], (0, 1, 0), (1, 0, 0, SEASONAL_PERIOD), exog_tr=X_it.iloc[:t0])),
    }
    rolling_it = {}
    failed_it = {}
    for nome, fn in cv_fns.items():
        eh = {h: [] for h in orizzonti}
        n_fail = 0
        for t0 in range(n_train, len(y_it)):
            h_ok = [h for h in orizzonti if t0 + h <= len(y_it)]
            if not h_ok:
                break
            fc, ok = fn(t0, max(h_ok))
            if not ok:
                n_fail += 1
            for h in h_ok:
                eh[h].append(y_it[t0 + h - 1] - fc[h - 1])
        failed_it[nome] = int(n_fail)
        rolling_it[nome] = {
            f"h{h}": {"RMSE": round(float(np.sqrt(np.mean(np.asarray(eh[h]) ** 2))), 3)}
            for h in orizzonti
        }
        m = metriche[nome]
        r = rolling_it[nome]
        print(
            f"   -> Serie A {nome:28s} | 36m RMSE={m['RMSE']:.3f} (MAPE={m['MAPE']:.2f}%)"
            f" | CV h1={r['h1']['RMSE']:.3f} h6={r['h6']['RMSE']:.3f} h12={r['h12']['RMSE']:.3f} | fail={n_fail}"
        )

    # FIGURA 21
    fig, ax = crea_figura(larghezza=11.8, altezza=5.8)
    ax.plot(df["date"].iloc[:n_train], y_tr, color="#1b5e20", linewidth=2.4, label="Storico Serie A – Training Set (2013–2023)")
    ax.plot(df["date"].iloc[n_train - 1:], y_it[n_train - 1:], color="#111111", linewidth=2.5, zorder=5)
    date_test = df["date"].iloc[n_train:]
    ax.plot(date_test, y_te, color="#111111", linewidth=3.0, marker="o", markersize=4.0, label="Valore reale Serie A nel Test Set (2023–2026)", zorder=6)
    ax.plot(date_test, p_rwd, color="#1f77b4", linewidth=2.3, linestyle="-.", label=f"Random Walk + drift (MAPE: {metriche['Random Walk + drift']['MAPE']:.2f}%)", zorder=5)
    ax.plot(date_test, p_sarima, color="#9467bd", linewidth=2.1, linestyle=":", label=f"SARIMA(0,1,0)(1,0,0)12 (MAPE: {metriche['SARIMA(0,1,0)(1,0,0)12']['MAPE']:.2f}%)", zorder=5)
    ax.plot(date_test, p_full, color="#d95f02", linewidth=2.5, linestyle="--", label=f"SARIMAX + COVID + Spesa (MAPE: {metriche['SARIMAX + COVID + Spesa 12m']['MAPE']:.2f}%)", zorder=5)
    ax.fill_between(date_test, ci_full[:, 0], ci_full[:, 1], color="#d95f02", alpha=0.18, label="Fascia 95% condizionata (SARIMAX)")

    ax.axvline(df["date"].iloc[n_train], color="#333333", linestyle="--", linewidth=1.4)
    ax.set_title("Focus Serie A (IT1): confronto multi-step tra Random Walk con drift, SARIMA e SARIMAX (2023–2026)", fontsize=12.2, fontweight="bold", pad=12)
    ax.set_ylabel("Valore rose Serie A (Miliardi EUR)", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Data (Giugno 2013 – Maggio 2026)", fontweight="bold", fontsize=11.5)
    ax.set_ylim(1.9, max(6.8, float(np.max(ci_full[:, 1])) + 0.25))
    ax.legend(loc="upper left", frameon=True, fontsize=9.2)
    salva_figura(fig, "fig21_focus_previsione_serie_a.png", CAP9_IMG_DIR)

    return {
        "convergenza_holdout": {
            "Random Walk + drift": bool(mod_rwd.mle_retvals.get("converged", False)),
            "SARIMA(0,1,0)(1,0,0)12": bool(mod_sarima.mle_retvals.get("converged", False)),
            "SARIMAX + COVID": bool(mod_cov.mle_retvals.get("converged", False)),
            "SARIMAX + COVID + Spesa 12m": bool(mod_full.mle_retvals.get("converged", False)),
        },
        "failed_convergences_cv": failed_it,
        "metriche": metriche,
        "rolling_origin_cv": rolling_it,
        "parametri_sarimax_covid": estrai_tabella_parametri(mod_cov),
        "parametri_sarimax_full": estrai_tabella_parametri(mod_full),
        "diagnostica_sarimax_covid": diag_it_cov,
        "diagnostica_sarimax_full": diag_it_full,
    }
