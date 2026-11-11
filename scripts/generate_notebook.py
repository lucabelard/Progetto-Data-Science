"""
========================================================================================
Progetto di Data Science: Spotify Audio Analysis
Script di Generazione del Jupyter Notebook in Stile Tesi Accademica (LaTeX Mirror)
Corso di Laurea Magistrale in Ingegneria Informatica e dell'Automazione (UnivPM)
========================================================================================
Questo script compila 'Spotify_Audio_Analysis_Classification_Clustering.ipynb' replicando
fedelmente la struttura, i titoli, le formule, le tabelle e le didascalie del documento
accademico di riferimento (LaTeX).
========================================================================================
"""

import os
import json
import base64
import re

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}
cells = []

def _to_lines(s):
    lines = s.splitlines(keepends=True)
    return lines if lines else [""]

def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _to_lines(text.strip())
    }

def code(text):
    source_code = text.strip()
    outputs = []
    img_matches = re.findall(r"Image\(filename=['\"]([^'\"]+)['\"]\)", source_code)
    for img_path in img_matches:
        if os.path.exists(img_path):
            with open(img_path, "rb") as img_f:
                b64_data = base64.b64encode(img_f.read()).decode('utf-8')
            outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": b64_data,
                    "text/plain": [f"<IPython.core.display.Image object: {img_path}>"]
                },
                "metadata": {}
            })
    return {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": outputs,
        "source": _to_lines(source_code)
    }

# ======================================================================================
# FRONTESPIZIO ACCADEMICO
# ======================================================================================
cells.append(md(r"""
<div align="center">

# Università Politecnica delle Marche
### Facoltà di Ingegneria
#### Dipartimento di Ingegneria dell'Informazione
##### Corso di Laurea Magistrale in Ingegneria Informatica e dell'Automazione

<br/>

---

# **Spotify tracks analysis:**
# **Classificazione e clustering**

---

<br/>

<table width="100%" style="border: none; background: transparent;">
<tr>
<td align="left" style="border: none;">
<b>Docenti:</b><br/>
Prof. Ursino Domenico<br/>
Dott. Marchetti Michele
</td>
<td align="right" style="border: none;">
<b>Componenti del gruppo:</b><br/>
Bellante Luca<br/>
Coccia Giansimone<br/>
Ferretti Laura
</td>
</tr>
</table>

<br/>

### **Anno Accademico 2024-2025**

</div>
"""))

# ======================================================================================
# INDICE DEI CONTENUTI (TABLE OF CONTENTS)
# ======================================================================================
cells.append(md(r"""
## Contents

* [**1 Spotify tracks analysis: Introduzione**](#1-spotify-tracks-analysis-introduzione)
  * [1.1 Obiettivi del progetto](#11-obiettivi-del-progetto)
* [**2 Spotify tracks analysis: Dataset**](#2-spotify-tracks-analysis-dataset)
  * [2.1 ETL](#21-etl)
  * [2.2 Analisi descrittiva](#22-analisi-descrittiva)
    * [2.2.1 Distribuzione dei generi musicali](#221-distribuzione-dei-generi-musicali)
    * [2.2.2 Distribuzione delle feature continue](#222-distribuzione-delle-feature-continue)
    * [2.2.3 Distribuzione delle variabili discrete](#223-distribuzione-delle-variabili-discrete)
    * [2.2.4 Matrice di correlazione](#224-matrice-di-correlazione)
* [**3 Spotify tracks analysis: Classificazione**](#3-spotify-tracks-analysis-classificazione)
  * [3.1 Preprocessing del Dataset](#31-preprocessing-del-dataset)
    * [3.1.1 Oversampling con SMOTE](#311-oversampling-con-smote)
    * [3.1.2 Undersampling](#312-undersampling)
    * [3.1.3 Dataset Finale](#313-dataset-finale)
    * [3.1.4 Analisi della distribuzione degli attributi](#314-analisi-della-distribuzione-degli-attributi)
  * [3.2 Scenari di Classificazione](#32-scenari-di-classificazione)
  * [3.3 Processo di Classificazione](#33-processo-di-classificazione)
    * [3.3.1 Confronto Modelli di Machine Learning](#331-confronto-modelli-di-machine-learning)
    * [3.3.2 Ensemble Learning (Voting e Stacking Classifier)](#332-ensemble-learning)
    * [3.3.3 Diagnostica e Commento ai Risultati](#333-diagnostica-e-commento-ai-risultati)
* [**4 Spotify tracks analysis: Clustering**](#4-spotify-tracks-analysis-clustering)
  * [4.1 Fase di ETL per il clustering](#41-fase-di-etl-per-il-clustering)
    * [4.1.1 Operazioni preliminari sul dataset](#411-operazioni-preliminari-sul-dataset)
    * [4.1.2 Riduzione di dimensionalità (PCA e t-SNE)](#412-riduzione-di-dimensionalità)
  * [4.2 Kmeans](#42-kmeans)
    * [4.2.1 Applicazione dell'algoritmo K-means ed Elbow Method](#421-applicazione-dellalgoritmo-k-means)
    * [4.2.2 Analisi dei risultati ottenuti e Profiling dei Cluster](#422-analisi-dei-risultati-ottenuti)
    * [4.2.3 Grafico della Silhouette](#423-grafico-della-silhouette)
    * [4.2.4 Evoluzione probabilistica con GMM](#424-evoluzione-probabilistica-con-gmm)
  * [4.3 DBSCAN](#43-dbscan)
    * [4.3.1 Applicazione dell'algoritmo DBSCAN](#431-applicazione-dellalgoritmo-dbscan)
    * [4.3.2 Analisi dei risultati ottenuti e Diagnostica del Rumore](#432-analisi-dei-risultati-ottenuti)
    * [4.3.3 Evoluzione con HDBSCAN e affiliazioni soft](#433-evoluzione-con-hdbscan-e-affiliazioni-soft)
* [**5 Business Case finale e Conclusioni**](#5-business-case-finale-e-conclusioni)
"""))

# Setup Libraries
cells.append(md(r"""
---
### Configurazione dell'Ambiente di Esecuzione e Librerie
"""))

cells.append(code(r"""
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

# Preprocessing e Machine Learning
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Riduzione di dimensionalità e Clustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import hdbscan
from hdbscan import all_points_membership_vectors
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples

# Stile visuale coerente con il report accademico
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

print("Ambiente di lavoro inizializzato con successo!")
"""))

# ======================================================================================
# CAPITOLO 1: INTRODUZIONE
# ======================================================================================
cells.append(md(r"""
---
# 1 Spotify tracks analysis: Introduzione

Nel panorama contemporaneo dello streaming musicale, la fruizione e la catalogazione dei brani audio sono guidate in misura sempre maggiore da modelli matematici e algoritmi di machine learning. Piattaforme leader a livello globale come Spotify estraggono e analizzano sistematicamente caratteristiche acustiche e psicoacustiche a partire dal segnale audio grezzo di milioni di tracce musicali.

Comprendere come queste variabili continue e discrete interagiscano tra loro permette sia di automatizzare la categorizzazione dei brani all'interno dei rispettivi generi musicali, sia di scoprire pattern latenti di affinità sonica non rilevabili dai soli metadati commerciali. In questo contesto, il presente progetto si propone di analizzare un dataset contenente oltre 42.000 tracce musicali descritte da molteplici attributi tecnici (frequenza, tempo in BPM, volume sonoro, presenza vocale, valenza emotiva e ballabilità), integrando un percorso metodologico completo che spazia dalla pulizia e preparazione dei dati, alla modellazione supervisionata e non supervisionata.

## 1.1 Obiettivi del progetto

L'obiettivo principale del progetto è l'applicazione di tecniche di machine learning, in particolare di classificazione e clustering, per individuare schemi e relazioni nei dati. Nello specifico, il progetto si articola nei seguenti obiettivi:

* **Classificazione:** Creare modelli predittivi in grado di stimare il genere musicale di appartenenza sulla base delle variabili acustiche disponibili, affrontando il problema dello sbilanciamento tramite campionamento avanzato ed ensemble learning.
* **Clustering:** Raggruppare i brani in cluster significativi, basati su caratteristiche condivise, per identificare profili comuni di sonorità e comporre automaticamente playlist basate sul *mood*.
* **Analisi interpretativa:** Esplorare i fattori che contribuiscono maggiormente alla definizione del genere (Feature Importance), evidenziando il peso relativo delle variabili incluse nel dataset e identificando eventuali brani anomali (outlier acustici).

Attraverso l'impiego di queste tecniche, il progetto mira a fornire spunti utili per una migliore comprensione delle proprietà del catalogo e per la definizione di raccomandazioni musicali personalizzate.
"""))

# ======================================================================================
# CAPITOLO 2: DATASET
# ======================================================================================
cells.append(md(r"""
---
# 2 Spotify tracks analysis: Dataset

Il dataset utilizzato per l'analisi delle caratteristiche musicali è stato reperito dalla piattaforma Kaggle (*Spotify Audio Features / Genres v2*). Il dataset in questione contiene informazioni dettagliate su **42.305 brani musicali** categorizzati in 15 generi, includendo misurazioni fisiche, attributi di produzione e metadati.

Nella Tabella 1 vediamo una rappresentazione delle colonne del dataset e il relativo formato dei dati.

#### Table 1: Colonne del dataset e relativo formato dei dati.

| Colonna | Formato / Descrizione |
| :--- | :--- |
| `danceability` | Float [0.0 - 1.0] (Ballabilità del brano) |
| `energy` | Float [0.0 - 1.0] (Intensità ed energia percepita) |
| `key` | Intero [0 - 11] (Tonalità musicale / Pitch Class) |
| `loudness` | Float (Volume complessivo in decibel dB) |
| `mode` | Intero [0 = Minore, 1 = Maggiore] |
| `speechiness` | Float [0.0 - 1.0] (Presenza di parole parlate) |
| `acousticness` | Float [0.0 - 1.0] (Confidenza che il brano sia acustico) |
| `instrumentalness` | Float [0.0 - 1.0] (Probabilità di assenza di parti vocali) |
| `liveness` | Float [0.0 - 1.0] (Probabilità che sia registrato dal vivo) |
| `valence` | Float [0.0 - 1.0] (Positività / Umore del brano) |
| `tempo` | Float (Velocità ritmica in battiti al minuto BPM) |
| `duration_ms` | Intero (Durata della traccia in millisecondi) |
| `time_signature` | Intero (Metro ritmico, es. 4/4) |
| `id` | Stringa (Identificativo univoco Spotify) |
| `genre` | Categoriale (Target di classificazione: 15 generi) |
"""))

cells.append(md(r"""
## 2.1 ETL

Il processo di ETL è stato fondamentale per garantire la qualità dei dati e la loro adeguatezza per le analisi successive. Durante questa fase, sono state applicate diverse operazioni per preparare i dati:

* **Eliminazione dei duplicati basata su Track ID:** La presenza di brani duplicati è stata eliminata filtrando la chiave univoca `id`. Sono state rimosse **6.428 tracce duplicate**, portando il dataset a **35.877 brani unici**. Questo passaggio previene il data leakage tra training e test set.
* **Eliminazione delle variabili di servizio:** Le variabili `Unnamed: 0`, `type`, `uri`, `track_href`, `analysis_url`, `song_name` e `title` sono state rimosse poiché rappresentavano metadati o identificativi privi di rilevanza analitica acustica.
* **Gestione dei valori nulli e verifica della bontà dei dati:** Il dataset è stato analizzato al fine di individuare l'eventuale presenza di valori nulli: tutte le feature acustiche e il target risultano completi (0 valori mancanti).
"""))

cells.append(code(r"""
# Esecuzione e verifica del processo di ETL
df_clean = pd.read_csv('data/genres_cleaned.csv')
print(f"Dataset pulito caricato con successo: {df_clean.shape[0]} righe e {df_clean.shape[1]} colonne.")
print(f"Verifica valori nulli totali: {df_clean.isnull().sum().sum()}")
display(df_clean.head(5))
"""))

cells.append(md(r"""
## 2.2 Analisi descrittiva

L'analisi descrittiva svolta ha avuto lo scopo di esplorare e sintetizzare le principali caratteristiche del dataset, fornendo una comprensione preliminare delle variabili e delle loro distribuzioni.

### 2.2.1 Distribuzione dei generi musicali
La distribuzione dei generi è stata visualizzata tramite il grafico a barre riportato in **Figura 1**. I dati evidenziano uno sbilanciamento significativo, con una netta prevalenza dei generi *Underground Rap* (5.044 brani) e *Dark Trap* (4.578 brani), a fronte di classi meno frequenti come *Pop* (404 brani). Questo sbilanciamento guiderà la successiva fase di campionamento.
"""))

cells.append(code(r"""
# Figure 1: Distribuzione dei generi musicali
display(Image(filename='figures/fig01_distribuzione_generi.png'))
print("Conteggio per classe di genere:")
display(df_clean['genre'].value_counts().to_frame(name='Numero di Brani'))
"""))

cells.append(md(r"""
### 2.2.2 Distribuzione delle feature continue
Per analizzare le variabili numeriche continue, è stata creata una serie di istogrammi con curve di densità KDE riportata in **Figura 2**. 
La distribuzione mostra come la `danceability` sia approssimativamente normale, mentre `energy` e `loudness` evidenziano un'asimmetria verso valori elevati. La `speechiness` riflette la bimodalità tipica di tracce strumentali/cantate rispetto a brani a base rap, e l'`acousticness` è quasi ovunque prossima allo zero.
"""))

cells.append(code(r"""
# Figure 2: Distribuzione delle feature audio continue
display(Image(filename='figures/fig02_audio_features_hist_kde.png'))
"""))

cells.append(md(r"""
### 2.2.3 Distribuzione delle variabili discrete
In **Figura 3** vengono esaminate le distribuzioni della tonalità (`key`), della modalità (`mode`) e del metro (`time_signature`), attestando la netta prevalenza del tempo 4/4 e una leggera maggioranza di tonalità maggiori.
"""))

cells.append(code(r"""
# Figure 3: Distribuzione delle variabili discrete
display(Image(filename='figures/fig03_discrete_features.png'))
"""))

cells.append(md(r"""
### 2.2.4 Matrice di correlazione
È stata calcolata, infine, la matrice di correlazione di Pearson per le feature audio, visualizzata tramite heatmap in **Figura 4**. 
Dall'analisi emerge la forte relazione lineare positiva tra `energy` e `loudness` ($r \approx 0.70$), la correlazione negativa tra `energy` e `acousticness` ($r \approx -0.49$) e il legame positivo tra `danceability` e `valence`.
"""))

cells.append(code(r"""
# Figure 4: Matrice di correlazione iniziale
display(Image(filename='figures/fig04_matrice_correlazione_iniziale.png'))
"""))

# ======================================================================================
# CAPITOLO 3: CLASSIFICAZIONE
# ======================================================================================
cells.append(md(r"""
---
# 3 Spotify tracks analysis: Classificazione

La classificazione è stata condotta con l'obiettivo di sviluppare modelli predittivi in grado di stimare la classe di genere musicale dei brani sulla base delle feature audio disponibili. Il processo ha incluso un preprocessing rigoroso contro il data leakage, il bilanciamento delle classi, la selezione e il confronto di 8 algoritmi, e l'integrazione di tecniche di Ensemble Learning.

## 3.1 Preprocessing del Dataset

Il dataset è stato suddiviso in Training Set (80%, 28.701 brani) e Test Set (20%, 7.176 brani) con partizione stratificata.
Lo `StandardScaler` è stato calcolato (**fit**) esclusivamente sui campioni di training:
\begin{equation}
z = \frac{x - \mu_{train}}{\sigma_{train}} \tag{1}
\end{equation}
e successivamente applicato (**transform**) su train e test set, preservando l'integrità della valutazione.

### 3.1.1 Oversampling con SMOTE
Per affrontare il problema dello sbilanciamento delle classi nella variabile target `genre`, è stata utilizzata la tecnica SMOTE (*Synthetic Minority Oversampling Technique*) per incrementare i campioni delle classi minoritarie (*Pop, Trap Metal, Rap, Emo, RnB*) tramite interpolazione vettoriale.

### 3.1.2 Undersampling
Per le classi maggioritarie (*Underground Rap, Dark Trap*), è stato applicato un undersampling controllato per evitare che il modello sviluppasse una distorsione sistematica verso le etichette dominanti.

### 3.1.3 Dataset Finale
Dopo l'applicazione di entrambe le tecniche, il training set bilanciato presenta esattamente **2.000 campioni per ciascuna delle 15 classi** (30.000 osservazioni complessive). In **Figura 5** è possibile vedere il risultato della distribuzione dei generi prima e dopo il processo di campionamento.
"""))

cells.append(code(r"""
# Figure 5: Distribuzione del training set pre e post sampling
display(Image(filename='figures/fig05_distribuzione_sampling.png'))

with open('data/project_summary.json', 'r') as f:
    stats = json.load(f)
print(f"Campioni di training bilanciati: {stats['balanced_train_samples']} (2.000 per ciascuna delle 15 classi)")
print(f"Campioni di test set preservati: {stats['test_samples']}")
"""))

cells.append(md(r"""
### 3.1.4 Analisi della distribuzione degli attributi
Al termine delle operazioni di campionamento, è stata verificata la coerenza della distribuzione degli attributi nel dataset bilanciato. In **Figura 6** è riportata la matrice di correlazione post-sampling: la struttura lineare tra le variabili si è mantenuta intatta e coerente con quella originale (Figura 4), attestando che il bilanciamento non ha compromesso l'integrità del dataset.
"""))

cells.append(code(r"""
# Figure 6: Matrice di correlazione dopo le operazioni di sampling
display(Image(filename='figures/fig06_matrice_correlazione_post_sampling.png'))
"""))

cells.append(md(r"""
## 3.2 Scenari di Classificazione

Nel procedere con la classificazione, sono stati considerati due scenari per valutare l'impatto degli attributi:
1. **Scenario 1 (Set Completo):** Include tutte le 13 feature acustiche standardizzate.
2. **Scenario 2 (Esclusione Collinearità):** Esclude `loudness` per via della forte collinearità con `energy` ($r \approx 0.70$), verificando se la ridondanza informativa alteri le prestazioni predittive.

## 3.3 Processo di Classificazione

Gli algoritmi di machine learning selezionati per la classificazione sono:
* Logistic Regression (Baseline lineare)
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes
* AdaBoost Classifier
* Gradient Boosting Classifier
* XGBoost (Extreme Gradient Boosting)

I risultati ottenuti sul Test Set indipendente sono riassunti in **Tabella 2** e illustrati in **Figura 7**.

#### Table 2: Confronto delle performance dei modelli sul Test Set (7.176 campioni).

| Indice | Modello | Accuratezza Media | Weighted F1 | Macro F1 |
| :---: | :--- | :---: | :---: | :---: |
| 0 | **XGBoost** | **0.6745** | **0.6763** | 0.6480 |
| 1 | **Random Forest** | **0.6734** | **0.6726** | 0.6498 |
| 2 | Gradient Boosting | 0.6630 | 0.6634 | 0.6397 |
| 3 | Decision Tree | 0.5916 | 0.5960 | 0.5734 |
| 4 | Logistic Regression | 0.5516 | 0.5560 | 0.5279 |
| 5 | K-Nearest Neighbors | 0.5394 | 0.5425 | 0.5213 |
| 6 | Naive Bayes | 0.5332 | 0.5364 | 0.5198 |
| 7 | AdaBoost | 0.3686 | 0.3513 | 0.3261 |
"""))

cells.append(code(r"""
# Figure 7: Confronto dell'accuratezza media tra i modelli baseline
display(Image(filename='figures/fig07_confronto_modelli_baseline.png'))
"""))

cells.append(md(r"""
### 3.3.2 Ensemble Learning

Gli approcci di Ensemble Learning adottati per massimizzare le performance sono due:

* **Voting Classifier:** Implementato con la modalità *soft voting*, che sfrutta le probabilità predette dai singoli modelli per determinare la classe finale:
\begin{equation}
\hat{y} = \arg\max_{c} \frac{1}{M} \sum_{m=1}^M P_m(y=c \mid x) \tag{2}
\end{equation}
Integrando XGBoost, Random Forest e Gradient Boosting, raggiunge un'accuratezza del **68.00%** e un Weighted F1 di **0.6810**.

* **Stacking Classifier:** Combina le predizioni di probabilità dei modelli di base fornendole a un meta-modello (Regressione Logistica) addestrato tramite 3-fold cross-validation. Questo approccio raggiunge un'accuratezza del **68.90%** e un Weighted F1 di **0.6890**, costituendo il modello più performante in assoluto.
"""))

cells.append(code(r"""
print(f"Voting Classifier (Soft):    Accuratezza = {stats['voting_acc']*100:.2f}% | F1-Weighted = {stats['voting_f1']:.4f}")
print(f"Stacking Classifier (Meta LR): Accuratezza = {stats['stacking_acc']*100:.2f}% | F1-Weighted = {stats['stacking_f1']:.4f}")
"""))

cells.append(md(r"""
### 3.3.3 Diagnostica e Commento ai Risultati

La diagnostica approfondita del modello è visualizzata attraverso i seguenti grafici:
* **Matrice di Confusione Normalizzata (Figura 8):** Mostra un'eccellente concentrazione delle predizioni lungo la diagonale principale. Gli errori riflettono similarità acustiche reali del mondo musicale (es. *Dark Trap* scambiato con *Underground Rap* o *Trap Metal*, e *Techhouse* con *Techno*).
* **Classification Report (Figura 9):** F1-Score elevati per generi con firme soniche distinte (*dnb, hardstyle, psytrance* compresi tra **0.75 e 0.88**).
* **Curve ROC e Precision-Recall (Figure 10 e 11):** Il valore di **Macro-Average ROC-AUC è pari a 0.94**, confermando un'elevata capacità di discriminazione globale.
* **Learning Curve (Figura 12):** La convergenza progressiva della curva di validazione verso quella di training attesta la robustezza del modello ed esclude problemi di overfitting.
* **Feature Importance (Figura 13):** `tempo` (BPM) e `speechiness` risultano i fattori discriminanti più determinanti, seguiti da `danceability`, `loudness` ed `energy`.
"""))

cells.append(code(r"""
# Visualizzazione Figure 8, 9, 10, 11, 12, 13 (Diagnostica Completa)
display(Image(filename='figures/fig08_confusion_matrix_best_model.png'))
display(Image(filename='figures/fig09_classification_report_heatmap.png'))
display(Image(filename='figures/fig10_roc_curves_multiclass.png'))
display(Image(filename='figures/fig11_precision_recall_curves.png'))
display(Image(filename='figures/fig12_learning_curve.png'))
display(Image(filename='figures/fig13_feature_importance.png'))
"""))

# ======================================================================================
# CAPITOLO 4: CLUSTERING
# ======================================================================================
cells.append(md(r"""
---
# 4 Spotify tracks analysis: Clustering

Questo capitolo illustra tecniche di clustering non supervisionato per raggruppare i brani in base alle caratteristiche sonore, senza usare il target commerciale `genre` per l'addestramento. K-Means e DBSCAN sono estesi rispettivamente con GMM e HDBSCAN per analizzare sovrapposizioni tra mood e densità variabile.

## 4.1 Fase di ETL per il clustering
### 4.1.1 Operazioni preliminari e selezione delle feature
Prima di applicare gli algoritmi basati su distanza euclidea (K-Means e DBSCAN), il dataset deduplicato ($N = 35.877$) è stato sottoposto a una selezione mirata delle feature per prevenire distorsioni metriche:
* **Esclusione di `genre` e `id`:** per garantire un approccio puramente non supervisionato.
* **Esclusione delle 3 variabili discrete/categoriche (`key`, `mode`, `time_signature`):** la loro codifica intera/binaria introduce discontinuità artificiali nello spazio euclideo.
* **Esclusione di `loudness` e `liveness`:** `loudness` viene rimossa per l'elevata multicollinearità con `energy` ($r = 0.70$, che duplicherebbe il peso del volume nei centroidi), mentre `liveness` descrive il contesto di registrazione (pubblico dal vivo) e non la struttura musicale del brano.
* **Standardizzazione Obbligatoria (`StandardScaler`) e Rischio di Dominanza della Varianza:** Poiché `duration_ms` ($\sim 200.000\text{ ms}$, $\sigma^2 \approx 10^{10}$) e `tempo` ($\sim 140\text{ BPM}$, $\sigma^2 \approx 900$) presentano varianze enormemente superiori agli indici in $[0, 1]$, applicare la PCA sui dati grezzi farebbe assorbire a `duration_ms` il $99.99\%$ della `PC1`. Ogni feature $j \in \{1,\dots,8\}$ viene quindi portata a media $\mu_j = 0$ e varianza unitaria $\sigma_j^2 = 1$ (varianza totale dello spazio $= 8$):
\begin{equation}
\boxed{z_{i,j} = \frac{x_{i,j} - \mu_j}{\sigma_j} \quad \Longrightarrow \quad \mu(z_j) = 0, \quad \sigma^2(z_j) = 1}
\end{equation}

### 4.1.2 Riduzione di dimensionalità: PCA e t-SNE
* **PCA (Principal Component Analysis):** applicata sulle 8 feature standardizzate come riduzione dimensionale effettiva prima del clustering. Come mostrato in **Tabella 3** e nello Scree Plot di **Figura 14**, le prime **4 Componenti Principali (`PC1`–`PC4`)** catturano il **73.96% della varianza cumulativa totale** (superando la soglia del 70%). K-Means, GMM, DBSCAN e HDBSCAN sono confrontati nello spazio PCA a 4 dimensioni; il 26,04% residuo non viene assunto automaticamente come rumore privo di informazione.

#### Table 3: Quota di varianza spiegata individuale e cumulativa (PCA sulle 8 feature selezionate).

| Componente Principale | Varianza Spiegata (%) | Varianza Cumulativa (%) | Stato Selezione |
| :--- | :---: | :---: | :---: |
| **PC1** | 32.00% | 32.00% | **Selezionata** |
| **PC2** | 16.54% | 48.54% | **Selezionata** |
| **PC3** | 15.21% | 63.75% | **Selezionata** |
| **PC4** | 10.22% | **73.96%** | **Selezionata (>70%)** |
| PC5 | 9.03% | 83.00% | Esclusa |
| PC6 | 8.04% | 91.04% | Esclusa |
| PC7 | 4.89% | 95.92% | Esclusa |
| PC8 | 4.08% | 100.00% | Esclusa |

#### Table 4: Matrice dei Loadings PCA (`PC1`–`PC4`) sulle 8 feature acustiche.

| Feature Acustica | PC1 (32.00%) | PC2 (16.54%) | PC3 (15.21%) | PC4 (10.22%) |
| :--- | :---: | :---: | :---: | :---: |
| `danceability` | -0.2179 | **+0.6721** | +0.2303 | +0.1679 |
| `energy` | **+0.3767** | -0.2699 | **+0.5122** | -0.1522 |
| `speechiness` | **-0.3724** | -0.0696 | +0.2895 | -0.0380 |
| `acousticness` | **-0.3482** | +0.0925 | **-0.5224** | +0.2265 |
| `instrumentalness` | **+0.4838** | +0.2391 | -0.0819 | +0.3038 |
| `valence` | -0.2888 | +0.2422 | **+0.5533** | +0.1739 |
| `tempo` | -0.1770 | **-0.5300** | +0.1223 | **+0.7626** |
| `duration_ms` | **+0.4494** | +0.2555 | +0.0092 | **+0.4380** |

* **t-SNE:** impiegata esclusivamente a scopo di visualizzazione topologica non lineare in 2D (`n_components=2`, `perplexity=40`, `random_state=42` su un campione rappresentativo di $N=6.000$ brani, **Figura 15**).
"""))

cells.append(code(r"""
# Figure 14 e 15: PCA Scree Plot e mappa t-SNE 2D
display(Image(filename='figures/fig14_pca_varianza_cumulativa.png'))
display(Image(filename='figures/fig15_tsne_2d_overview.png'))
"""))

cells.append(md(r"""
## 4.2 K-Means nello Spazio PCA (4 Componenti)

Il metodo K-Means minimizza la distanza intra-cluster calcolando la varianza interna WCSS (*Within-Cluster Sum of Squares*) nello spazio delle prime 4 componenti principali:
\begin{equation}
\text{WCSS} = \sum_{j=1}^k \sum_{\mathbf{z}_i \in C_j} \|\mathbf{z}_i - \boldsymbol{\mu}_j\|^2 \tag{3}
\end{equation}

### 4.2.1 Applicazione dell'algoritmo, Elbow Method e Silhouette Score
Il valore ottimale di $k$ è stato selezionato valutando congiuntamente la curva **WCSS (Elbow Method)** e il **Coefficiente di Silhouette medio** per $k \in [2, 10]$ (**Figura 16**). In corrispondenza di **$k = 5$** si osserva sia il gomito della curva WCSS ($13.981,8$), sia un **massimo locale del Silhouette Score pari a `0.2829`**, prima del netto calo a $k=6$ (`0.2395`).

### 4.2.2 Analisi dei risultati ottenuti e Profiling delle 5 "Mood Playlists"
Applicando K-Means con $k=5$ nello spazio PCA 4D e proiettando i cluster sia nel piano lineare PCA (`PC1` vs `PC2` con centroidi) sia sulla mappa t-SNE (**Figura 17**), si ottengono 5 profili acustici. La **Figura 18** riporta la heatmap: le annotazioni sono medie originali, mentre i colori sono riscalati separatamente per feature. La matrice include `loudness` e `liveness` solo come descrittori retrospettivi; non sono input del clustering. Gli ID C0-C4 nella legenda sono accompagnati da nomi sintetici basati sui tratti dominanti.
* **Cluster 0 — *High-Intensity Bass & Workout* (1.710 brani campione / 10.583 catalogo, 29.50%):** Massima `energy` (`0.86`) e `loudness` (`-4.83 dB`), `tempo` elevato (`152.40 BPM`), bassa `danceability` (`0.48`) e `valence` (`0.24`). Dominato da *Hardstyle, Drum and Bass e Trap*.
* **Cluster 1 — *Extended Club & Instrumental Electronic* (1.517 brani campione / 8.964 catalogo, 24.99%):** Altissima `instrumentalness` (`0.79`) e durata estesa (`390.803 ms`, ~6.5 min), alta `energy` (`0.84`) e `danceability` (`0.66`), `speechiness` minima (`0.06`). Dominato da *Techno, Psytrance, Trance e Techhouse*.
* **Cluster 2 — *Acoustic, Chill & Atmospheric* (549 brani campione / 3.189 catalogo, 8.89%):** Massima `acousticness` (`0.49`) e minimi assoluti di `energy` (`0.45`) e `loudness` (`-9.87 dB`). Raccoglie brani acustici, *RnB, Emo e Dark Trap/Lo-Fi* atmosferica.
* **Cluster 3 — *Groove, Melodic Urban & Dance-Pop* (1.250 brani campione / 7.354 catalogo, 20.50%):** Massima `danceability` (`0.76`), `tempo` cadenzato (`130.31 BPM`), `valence` positiva (`0.49`) e parlato moderato (`0.18`). Raccoglie *Pop, R&B, Rap melodico e Techhouse* radiofonica.
* **Cluster 4 — *Fast-Flow Rap & High-Tempo Urban* (974 brani campione / 5.787 catalogo, 16.13%):** Massimo `tempo` (`179.12 BPM`), massima `speechiness` (`0.26`) e massima `valence` (`0.55`). Composto da *Underground Rap, Hiphop e Dark Trap* ad alta velocità.

### 4.2.3 Grafico della Silhouette
La qualità del partizionamento nello spazio PCA 4D è attestata dal diagramma di Silhouette (**Figura 20**), con valore medio **`0.2829`** e picchi superiori a `0.50` per tutti i 5 cluster.

### 4.2.4 Evoluzione probabilistica con GMM
K-Means assegna ciascun brano a un solo cluster. Per rappresentare le tracce che condividono più profili acustici, è stato aggiunto un **Gaussian Mixture Model (GMM) a 5 componenti** nello stesso spazio PCA 4D. Per ogni brano il modello stima un vettore di affiliazioni ai componenti. Sul campione, **409 brani (6,82%)** hanno le due affiliazioni maggiori distanti non più di 0,15; la mediana dell'affiliazione massima è 0,883. Fra i 462 punti noise DBSCAN, la mediana dell'affiliazione massima è 0,928 e 39 (8,44%) sono candidati a un’affiliazione trasversale. Le probabilità sono stime del modello, non etichette di mood validate dagli ascoltatori. La **Figura 25** mostra le componenti e la distribuzione dell’affiliazione massima, con una legenda che ne riassume i tratti acustici. La scelta di 5 componenti mantiene il confronto con K-Means diretto; il GMM può descrivere covarianze e sovrapposizioni, ma assume componenti gaussiane e non garantisce che ogni componente corrisponda a un mood musicale distinto.
"""))

cells.append(code(r"""
# Visualizzazione Figure 16-18: selezione k, mappe cluster e heatmap K-Means
display(Image(filename='figures/fig16_kmeans_elbow_wcss.png'))
display(Image(filename='figures/fig17_kmeans_tsne_clusters.png'))
display(Image(filename='figures/fig18_kmeans_feature_profiling.png'))

"""))

cells.append(md(r"""
La heatmap rende leggibile il confronto delle medie: C1 è più strumentale e lungo, C2 più acustico e meno energetico, C3 più ballabile e C4 più rapido e vocale. Il radar della **Figura 19** mostra gli stessi profili sulle sole 8 feature usate nel modello; i valori sono normalizzati per asse, quindi si confrontano le forme tra cluster e non le unità assolute.
"""))

cells.append(code(r"""
display(Image(filename='figures/fig19_kmeans_feature_radar.png'))
display(Image(filename='figures/fig19_kmeans_silhouette_plot.png'))
"""))

cells.append(md(r"""
## 4.3 DBSCAN nello Spazio PCA (4 Componenti)

DBSCAN (*Density-Based Spatial Clustering of Applications with Noise*) raggruppa i punti in base alla densità locale nello spazio PCA a 4 dimensioni. La label `-1` indica che un brano non soddisfa i criteri di densità del modello; non è un giudizio sulla sua qualità musicale.

### 4.3.1 Scelta dei parametri e tuning ($\epsilon = 0.60, \text{min\_samples} = 5$)
* Dal grafico delle distanze ordinate 5-NN nello spazio PCA 4D (**Figura 21**), il punto di ginocchio si colloca a **$\epsilon = 0.60$**.
* La doppia matrice di tuning $(\epsilon \times \text{min\_samples})$ in **Figura 22** mostra il numero di cluster e il Silhouette Score dei punti non-rumore: con $\epsilon = 0.60, \text{min\_samples} = 5$ si identificano **6 cluster** (un macro-cluster centrale ad alta densità di 5.509 brani + 5 micro-cluster periferici) e si marcano come noise **462 brani (7.70% del campione)**.

### 4.3.2 Analisi dei risultati ottenuti e Diagnostica del Rumore
In **Figura 23** è mostrata la distribuzione spaziale DBSCAN su PCA 2D e t-SNE 2D con le 462 tracce noise evidenziate in rosso. I boxplot in **Figura 24** ne descrivono le durate, la `speechiness` e l'`acousticness`; i valori periferici non dimostrano che le tracce siano errori o vadano scartate.

### 4.3.3 Evoluzione con HDBSCAN e affiliazioni soft
La label `-1` di DBSCAN indica bassa densità rispetto ai parametri scelti; non significa che il brano sia privo di valore musicale. **HDBSCAN** è stato aggiunto nello stesso spazio PCA 4D (`min_cluster_size=30`, `min_samples=5`): sul campione trova 10 cluster, ma marca ancora come noise 4.847 brani (80,78%). Nessuno dei 462 punti noise DBSCAN entra in un cluster HDBSCAN con questa configurazione. Questo esito mostra che HDBSCAN non risolve automaticamente il problema e che i risultati dipendono dalla geometria e dai parametri.

Le affiliazioni soft HDBSCAN per quei 462 punti sono deboli: 22 superano il criterio esplorativo adottato per essere ispezionati. Per non escluderli a priori da ogni playlist, sono state quindi considerate anche le affiliazioni GMM ai cinque profili: ogni traccia riceve una proposta di profilo sonoro e, per 39 brani, i due profili più probabili sono vicini. La Figura 26 mostra i cluster HDBSCAN e l'affiliazione GMM massima per i punti noise DBSCAN. Le affiliazioni restano suggerimenti del modello: quando l'affinità è debole o le componenti sono poco interpretabili, il brano va mantenuto come caso periferico, non assegnato forzatamente.
"""))

cells.append(code(r"""
# Visualizzazione Figure 21-26: DBSCAN, GMM e HDBSCAN
display(Image(filename='figures/fig20_dbscan_knn_distance.png'))
display(Image(filename='figures/fig21_dbscan_grid_search.png'))
display(Image(filename='figures/fig22_dbscan_tsne_clusters_noise.png'))
display(Image(filename='figures/fig23_dbscan_noise_profiling.png'))
display(Image(filename='figures/fig24_gmm_soft_membership.png'))
display(Image(filename='figures/fig25_hdbscan_soft_membership.png'))
"""))

# =========================================================================
# CAPITOLO 5: CONCLUSIONI E BUSINESS CASE
# =========================================================================
cells.append(md(r"""
---
# 5 Business Case finale e Conclusioni

L'integrazione di tecniche supervisionate e non supervisionate dimostra come la Data Science applicata all'audio intelligence trasformi semplici numeri in decisioni strategiche per l'industria discografica:

1. **Playlist per affinità acustica (K-Means e GMM):** K-Means offre profili compatti; il GMM aggiunge appartenenze probabilistiche, utili per proporre tracce ibride in più playlist basate sul suono.
2. **Scoperta di nicchie e brani periferici (DBSCAN e HDBSCAN):** gli algoritmi di densità individuano comunità e punti poco rappresentati. La label noise è una proprietà geometrica del modello, non un giudizio sulla qualità musicale; le affiliazioni GMM e HDBSCAN aiutano a suggerire possibili famiglie senza imporre un’assegnazione quando il modello non offre supporto sufficiente.
3. **Valutazione Quantitativa del Potenziale Commerciale (Classificazione Stacking):** Con un'accuratezza vicina al **69% su 15 generi** e una **Macro-AUC di 0.94**, il modello fornisce a produttori ed etichette discografiche uno strumento oggettivo per posizionare sul mercato un nuovo brano demo prima di allocare budget pubblicitari.
"""))

# Salvataggio del Notebook
nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.14.3"
    }
}

target_path = 'Spotify_Audio_Analysis_Classification_Clustering.ipynb'
with open(target_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Jupyter Notebook salvato con successo in '{target_path}' con {len(cells)} celle in perfetto stile LaTeX!")
