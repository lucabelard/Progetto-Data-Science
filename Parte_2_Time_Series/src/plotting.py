"""
========================================================================================
Funzioni grafiche e di salvataggio per la Parte 2
========================================================================================
Questo file imposta lo stile dei grafici (font grandi, linee chiare, alta risoluzione)
e fornisce la funzione per salvare ogni figura automaticamente sia nell'archivio
centrale figures/ che nella cartella specifica del capitolo della Relazione.
========================================================================================
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES_DIR

# Impostiamo lo stile grafico elegante e leggibile alla massima risoluzione di stampa (600 DPI)
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 600


def crea_figura(larghezza=11.5, altezza=5.8):
    """Crea una nuova figura pulita con dimensioni adatte alla pagina della relazione."""
    fig, ax = plt.subplots(figsize=(larghezza, altezza))
    return fig, ax


def salva_figura(fig, nome_file, cartella_capitolo):
    """
    Salva la figura alla massima risoluzione Ultra-HD (600 DPI) in due posti:
    1. Nella cartella centrale: Parte_2_Time_Series/figures/
    2. Nella cartella delle immagini del capitolo: Relazione/capitoloX/img/
    """
    percorsi = [
        os.path.join(FIGURES_DIR, nome_file),
        os.path.join(cartella_capitolo, nome_file),
    ]
    for p in percorsi:
        fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"   [OK] Immagine salvata con successo (600 DPI Ultra-HD): {nome_file}")

