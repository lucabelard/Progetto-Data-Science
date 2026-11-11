"""
========================================================================================
Master Script: Esecuzione Completa della Pipeline Time Series (Parte 2)
========================================================================================
Questo script lancia in sequenza tutti i moduli di analisi e modellazione della Parte 2:
- Capitolo 7 (EDA e Shock Storici): genera fig03 .. fig11 in Relazione/capitolo7/img/
- Capitolo 8 (STL, Stazionarietà e ACF/PACF): genera fig12 .. fig15 in Relazione/capitolo8/img/
- Capitolo 9 (Modelli ARIMA/SARIMAX e Serie A): genera fig16 .. fig21 in Relazione/capitolo9/img/
========================================================================================
"""

import os
import sys

CARTELLA_PARTE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARTELLA_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in [CARTELLA_PARTE2, CARTELLA_SCRIPTS]:
    if p not in sys.path:
        sys.path.insert(0, p)

from importlib import import_module


def main():
    print("========================================================================================")
    print(" PIPELINE COMPLETA PARTE 2 (TIME SERIES TRANSFERMARKT): CAPITOLI 7, 8 E 9")
    print("========================================================================================")

    # 1. Capitolo 7: EDA e Shock Storici (fig03 .. fig11)
    mod_eda = import_module("02_eda_e_shock")
    mod_eda.run_eda_single_figures()

    # 2. Capitolo 8: Decomposizione STL, ADF/KPSS e ACF/PACF (fig12 .. fig15)
    mod_stl = import_module("03_run_stl_stationarity")
    mod_stl.main()

    # 3. Capitolo 9: Modellazione SARIMAX, Diagnostica e Serie A (fig16 .. fig21)
    mod_modeling = import_module("04_run_modeling")
    mod_modeling.main()

    print("\n========================================================================================")
    print(" TUTTA LA PIPELINE DELLA PARTE 2 E STATA ESEGUITA CON SUCCESSO!")
    print(" Tutte le 21 figure ad alta risoluzione sono aggiornate nelle cartelle di Relazione/.")
    print("========================================================================================")


if __name__ == "__main__":
    main()

