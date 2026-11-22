"""
========================================================================================
Configurazione generale del progetto Time Series (Transfermarkt)
========================================================================================
Questo file contiene tutte le impostazioni comuni (cartelle, date, colori)
in un unico punto, così da non doverle riscrivere in ogni script.
========================================================================================
"""

import os

# 1. Percorsi principali delle cartelle del progetto (relativi alla posizione del repo, portabili su ogni PC)
PART2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(PART2_ROOT)
DATA_RAW_DIR = os.path.join(PART2_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PART2_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PART2_ROOT, "figures")

# Cartelle della Relazione unificata in LaTeX
REL_DIR = os.path.join(WORKSPACE_ROOT, "Relazione")
CAP6_IMG_DIR = os.path.join(REL_DIR, "capitolo6", "img")
CAP7_IMG_DIR = os.path.join(REL_DIR, "capitolo7", "img")
CAP8_IMG_DIR = os.path.join(REL_DIR, "capitolo8", "img")
CAP9_IMG_DIR = os.path.join(REL_DIR, "capitolo9", "img")

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

# 3. Parametri temporali (156 mesi = 13 cicli annuali completi: Giu 2013 – Mag 2026)
START_DATE = "2013-06-01"
SPLIT_TRAIN_END = "2023-05-01"    # Primi 10 anni (120 mesi, 76,9%) per addestramento
SPLIT_TEST_START = "2023-06-01"   # Ultimi 3 anni (36 mesi, 23,1%) per test multi-step
END_DATE = "2026-05-01"
N_TRAIN_MONTHS = 120
SEASONAL_PERIOD = 12              # Ciclo annuale (12 mesi)

# 4. Colori ufficiali per le 5 leghe europee
LEAGUE_COLORS = {
    "Premier League": {"col": "premier_league_market_value_bn", "color": "#1f77b4"},
    "LaLiga": {"col": "laliga_market_value_bn", "color": "#ff7f0e"},
    "Serie A": {"col": "serie_a_market_value_bn", "color": "#2ca02c"},
    "Bundesliga": {"col": "bundesliga_market_value_bn", "color": "#d62728"},
    "Ligue 1": {"col": "ligue1_market_value_bn", "color": "#9467bd"},
}
