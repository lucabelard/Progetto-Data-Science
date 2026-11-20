"""
========================================================================================
Runner 04: Modellazione Predittiva, Validazione Rolling-Origin e Focus Serie A (Cap. 9)
========================================================================================
Esegue l'intero flusso del Capitolo 9 usando il modulo src/modeling/sarimax_models.py:
1. Carica market_monthly.csv (156 mesi, Giu 2013 – Mag 2026).
2. Disegna la partizione temporale Training (120 mesi) / Test (36 mesi) -> fig16.
3. Addestra e confronta i 7 modelli sul mercato Top-5 -> fig17.
4. Esegue la validazione rolling-origin (h = 1, 6, 12 mesi) + Diebold-Mariano -> fig18.
5. Esegue la diagnostica dei residui (Ljung-Box con model_df e Jarque-Bera) -> fig19, fig20.
6. Esegue il focus previsionale sulla Serie A italiana -> fig21.
7. Esporta tutte le metriche e i parametri in modeling_summary.json.
========================================================================================
"""

import os
import sys
import json
import pandas as pd

CARTELLA_PARTE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CARTELLA_PARTE2 not in sys.path:
    sys.path.insert(0, CARTELLA_PARTE2)

from src.config import MARKET_MONTHLY_CSV, DATA_PROCESSED_DIR, N_TRAIN_MONTHS
from src.modeling.sarimax_models import (
    disegna_split_train_test,
    addestra_e_confronta_modelli,
    valuta_rolling_origin,
    esegui_diagnostica_residui,
    esegui_focus_serie_a,
)


def main():
    print("========================================================================================")
    print(" AVVIO MODELLAZIONE CAPITOLO 9: BASELINE, HOLT-WINTERS, SARIMA, SARIMAX E ROLLING CV")
    print("========================================================================================")

    if not os.path.exists(MARKET_MONTHLY_CSV):
        print(f"ERRORE: File {MARKET_MONTHLY_CSV} non trovato. Esegui prima l'ETL.")
        sys.exit(1)

    df = pd.read_csv(MARKET_MONTHLY_CSV)
    df["date"] = pd.to_datetime(df["date"])
    n_train = N_TRAIN_MONTHS
    n_test = len(df) - n_train

    # 1. Split temporale (fig16)
    disegna_split_train_test(df, n_train=n_train)

    # 2. Confronto 7 modelli multi-step 36 mesi (fig17)
    risultati_top5, mod_sarima, mod_hw, mod_cov, mod_full, dettagli = addestra_e_confronta_modelli(df, n_train=n_train)

    # 3. Validazione rolling-origin CV h=1, 6, 12 + Diebold-Mariano / Clark-West (fig18)
    rolling_cv, test_accuratezza = valuta_rolling_origin(df, risultati_top5, n_train=n_train)

    # 4. Diagnostica residui con correzione model_df e verifica outlier COVID (fig19, fig20)
    diagnostica = esegui_diagnostica_residui(df, mod_sarima, mod_hw, mod_cov, mod_full, n_train=n_train)

    # 5. Focus Serie A (fig21)
    focus_it1 = esegui_focus_serie_a(df, n_train=n_train)

    sommario_modelli = {
        "n_train_months": int(n_train),
        "n_test_months": int(n_test),
        "train_period": f"{df['year_month'].iloc[0]} .. {df['year_month'].iloc[n_train - 1]}",
        "test_period": f"{df['year_month'].iloc[n_train]} .. {df['year_month'].iloc[-1]}",
        "modelli_top5_holdout_36m": risultati_top5,
        "rolling_origin_cv": rolling_cv,
        "test_accuratezza": test_accuratezza,
        "dettagli_stima_top5": dettagli,
        "diagnostica_residui": diagnostica,
        "focus_serie_a": focus_it1,
    }

    percorso_json = os.path.join(DATA_PROCESSED_DIR, "modeling_summary.json")
    with open(percorso_json, "w", encoding="utf-8") as f:
        json.dump(sommario_modelli, f, indent=2, ensure_ascii=False)

    print(f"   [OK] Sommario metriche modelli salvato in: {percorso_json}")
    print("========================================================================================")
    print(" COMPLETATO CON SUCCESSO! Tutte le 6 figure del Capitolo 9 sono pronte in Relazione.")
    print("========================================================================================")


if __name__ == "__main__":
    main()
