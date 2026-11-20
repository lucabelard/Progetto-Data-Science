"""
========================================================================================
Runner 03: Decomposizione STL, Test di Stazionarietà e Autocorrelazioni (Capitolo 8)
========================================================================================
Esegue tutte le analisi di identificazione del Capitolo 8 sul solo Training Set
(primi 120 mesi, Giugno 2013 – Maggio 2023) per evitare leakage nella scelta dei modelli,
salvando le figure fig12..fig15 e il riepilogo stl_stationarity_summary.json.
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
from src.analysis.stl_stationarity import (
    esegui_decomposizione_stl,
    analizza_stazionarieta_e_differenziazione,
    calcola_e_disegna_acf_pacf,
)


def main():
    print("========================================================================================")
    print(" AVVIO ANALISI CAPITOLO 8: STL, STAZIONARIETA (ADF/KPSS) E AUTOCORRELAZIONI (ACF/PACF)")
    print("========================================================================================")

    if not os.path.exists(MARKET_MONTHLY_CSV):
        print(f"ERRORE: File {MARKET_MONTHLY_CSV} non trovato. Esegui prima l'ETL.")
        sys.exit(1)

    df = pd.read_csv(MARKET_MONTHLY_CSV)
    df["date"] = pd.to_datetime(df["date"])

    df_train = df.iloc[:N_TRAIN_MONTHS].copy()
    serie_train = df_train["top5_market_value_bn"]
    date_train = df_train["date"]

    # 1. Decomposizione STL sul Training Set (fig12, fig13)
    risultati_stl = esegui_decomposizione_stl(
        serie_train, date_train, label_campione=f"Training Set, {N_TRAIN_MONTHS} mesi: 2013–2023"
    )

    # 2. Test di stazionarietà ADF e KPSS (fig14)
    risultati_test = analizza_stazionarieta_e_differenziazione(
        serie_train, date_train, serie_full=df["top5_market_value_bn"]
    )

    # 3. ACF e PACF sulla serie differenziata (grezza e depurata dal COVID, fig15)
    risultati_acf_pacf = calcola_e_disegna_acf_pacf(serie_train, date_train, max_lags=36)

    sommario = {
        "campione_identificazione": f"Training Set ({N_TRAIN_MONTHS} mesi: 2013-06 .. 2023-05)",
        "stl": risultati_stl,
        "test_stazionarieta": risultati_test,
        "acf_pacf": risultati_acf_pacf,
    }

    percorso_sommario = os.path.join(DATA_PROCESSED_DIR, "stl_stationarity_summary.json")
    with open(percorso_sommario, "w", encoding="utf-8") as f:
        json.dump(sommario, f, indent=2, ensure_ascii=False)

    print(f"   [OK] Sommario risultati esportato in: {percorso_sommario}")
    print("========================================================================================")
    print(" COMPLETATO CON SUCCESSO! Tutte le 4 figure del Capitolo 8 sono pronte in Relazione.")
    print("========================================================================================")


if __name__ == "__main__":
    main()
