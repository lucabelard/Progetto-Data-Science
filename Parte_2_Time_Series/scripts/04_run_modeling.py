"""
========================================================================================
Runner 04: Modellazione Predittiva, Valutazione Errori e Focus Serie A (Capitolo 9)
========================================================================================
Questo script esegue l'intero flusso del Capitolo 9 della Relazione usando il modulo src:
1. Carica la serie storica mensile post-ETL (180 mesi).
2. Disegna la partizione temporale 80/20 (144 mesi Train vs 36 mesi Test) -> fig16.
3. Addestra i 5 modelli (Baseline Naive, ARIMA, Holt-Winters, SARIMA, SARIMAX) -> fig17, fig18.
4. Esegue la diagnostica dei residui e il test di Ljung-Box -> fig19, fig20.
5. Esegue il focus previsionale dedicato alla Serie A italiana -> fig21.
6. Salva tutte le metriche numeriche in modeling_summary.json.
========================================================================================
"""

import os
import sys
import json
import pandas as pd

# Permette di importare i moduli interni da src
CARTELLA_PARTE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CARTELLA_PARTE2 not in sys.path:
    sys.path.insert(0, CARTELLA_PARTE2)

from src.config import MARKET_MONTHLY_CSV, DATA_PROCESSED_DIR
from src.modeling.sarimax_models import (
    disegna_split_train_test,
    addestra_e_confronta_modelli,
    esegui_diagnostica_residui,
    esegui_focus_serie_a,
)


def main():
    print("========================================================================================")
    print(" AVVIO MODELLAZIONE CAPITOLO 9: BASELINE, ARIMA, HOLT-WINTERS, SARIMA E SARIMAX")
    print("========================================================================================")

    if not os.path.exists(MARKET_MONTHLY_CSV):
        print(f"ERRORE: File {MARKET_MONTHLY_CSV} non trovato. Esegui prima l'ETL.")
        sys.exit(1)

    df = pd.read_csv(MARKET_MONTHLY_CSV)
    df["date"] = pd.to_datetime(df["date"])

    # 1. Visualizzazione split temporale 144 mesi Train / 36 mesi Test (fig16)
    disegna_split_train_test(df, n_train=144)

    # 2. Addestramento dei 5 modelli e confronto previsioni su Test Set (fig17, fig18)
    risultati_modelli, mod_sarimax, _ = addestra_e_confronta_modelli(df, n_train=144)

    # 3. Diagnostica dei residui del modello SARIMAX e test di Ljung-Box (fig19, fig20)
    diagnostica = esegui_diagnostica_residui(df, mod_sarimax, n_train=144)

    # 4. Focus previsionale dedicato alla Serie A italiana (fig21)
    focus_it1 = esegui_focus_serie_a(df, n_train=144)

    # 5. Esportazione del riepilogo JSON con tutti i numeri per la tabella LaTeX
    sommario_modelli = {
        "n_train_months": 144,
        "n_test_months": 36,
        "train_period": "2011-06 .. 2023-05",
        "test_period": "2023-06 .. 2026-05",
        "modelli_top5": risultati_modelli,
        "diagnostica_ljung_box": diagnostica,
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

