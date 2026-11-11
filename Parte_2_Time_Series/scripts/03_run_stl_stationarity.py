"""
========================================================================================
Runner 03: Decomposizione STL, Test di Stazionarietà e Autocorrelazioni (Capitolo 8)
========================================================================================
Questo script esegue l'analisi per il Capitolo 8 della Relazione usando il modulo src:
1. Carica la serie storica mensile post-ETL.
2. Esegue la Decomposizione STL e salva le figure fig12 e fig13.
3. Esegue i test ADF e KPSS e salva la figura fig14.
4. Calcola ACF e PACF e salva la figura fig15.
5. Esporta un riassunto dei risultati in un file JSON pulito.
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
from src.analysis.stl_stationarity import (
    esegui_decomposizione_stl,
    analizza_stazionarieta_e_differenziazione,
    calcola_e_disegna_acf_pacf,
)


def main():
    print("========================================================================================")
    print(" AVVIO ANALISI CAPITOLO 8: STL, STAZIONARIETA (ADF/KPSS) E AUTOCORRELAZIONI (ACF/PACF)")
    print("========================================================================================")
    
    # 1. Carichiamo i dati dal CSV pulito post-ETL
    if not os.path.exists(MARKET_MONTHLY_CSV):
        print(f"ERRORE: File {MARKET_MONTHLY_CSV} non trovato. Esegui prima l'ETL.")
        sys.exit(1)
        
    df = pd.read_csv(MARKET_MONTHLY_CSV)
    df["date"] = pd.to_datetime(df["date"])
    
    serie = df["top5_market_value_bn"]
    date = df["date"]
    
    # 2. Decomposizione STL (fig12, fig13)
    risultati_stl = esegui_decomposizione_stl(serie, date)
    
    # 3. Test di stazionarietà ADF e KPSS (fig14)
    risultati_test = analizza_stazionarieta_e_differenziazione(serie, date)
    
    # 4. Calcolo ACF e PACF sulla serie differenziata (fig15)
    serie_differenziata = serie.diff().dropna()
    risultati_acf_pacf = calcola_e_disegna_acf_pacf(serie_differenziata, max_lags=36)
    
    # 5. Salvataggio del sommario dei risultati per la compilazione del testo LaTeX
    sommario = {
        "forza_trend": risultati_stl["forza_trend"],
        "forza_stagionalita": risultati_stl["forza_stagionalita"],
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

