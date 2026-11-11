"""
========================================================================================
Configurazione generale del progetto Time Series (Transfermarkt)
========================================================================================
Questo file contiene tutte le impostazioni comuni (cartelle, date, colori)
in un unico punto, così da non doverle riscrivere in ogni script.
========================================================================================
"""

import os

# 1. Percorsi principali delle cartelle del progetto
WORKSPACE_ROOT = r"c:\Users\andre\Documents\GitHub\Progetto-Data-Science"
PART2_ROOT = os.path.join(WORKSPACE_ROOT, "Parte_2_Time_Series")
DATA_RAW_DIR = os.path.join(PART2_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PART2_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PART2_ROOT, "figures")

# Cartelle della Relazione unificata in LaTeX
REL_DIR = os.path.join(WORKSPACE_ROOT, "Relazione")
CAP6_IMG_DIR = os.path.join(REL_DIR, "capitolo6", "img")
CAP7_IMG_DIR = os.path.join(REL_DIR, "capitolo7", "img")
CAP8_IMG_DIR = os.path.join(REL_DIR, "capitolo8", "img")
CAP9_IMG_DIR = os.path.join(REL_DIR, "capitolo9", "img")

# Assicuriamoci che tutte le cartelle necessarie esistano su disco
ALL_DIRS = [
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    CAP6_IMG_DIR,
    CAP7_IMG_DIR,
    CAP8_IMG_DIR,
    CAP9_IMG_DIR,
]
for folder in ALL_DIRS:
    os.makedirs(folder, exist_ok=True)

# 2. File di dati principale post-ETL
MARKET_MONTHLY_CSV = os.path.join(DATA_PROCESSED_DIR, "market_monthly.csv")

# 3. Parametri temporali per i modelli (180 mesi = 15 anni esatti)
START_DATE = "2011-06-01"
SPLIT_TRAIN_END = "2023-05-01"    # Primi 12 anni (144 mesi) per allenare i modelli
SPLIT_TEST_START = "2023-06-01"   # Ultimi 3 anni (36 mesi) per testare le previsioni
END_DATE = "2026-05-01"
SEASONAL_PERIOD = 12              # Ciclo annuale (12 mesi)

# 4. Colori ufficiali per le 5 leghe europee (chiari e distinti)
LEAGUE_COLORS = {
    "Premier League": {"col": "premier_league_market_value_bn", "color": "#1f77b4"},
    "LaLiga": {"col": "laliga_market_value_bn", "color": "#ff7f0e"},
    "Serie A": {"col": "serie_a_market_value_bn", "color": "#2ca02c"},
    "Bundesliga": {"col": "bundesliga_market_value_bn", "color": "#d62728"},
    "Ligue 1": {"col": "ligue1_market_value_bn", "color": "#9467bd"},
}

