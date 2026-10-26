# Spotify Audio Analysis: Classificazione Multiclasse e Clustering Non Supervisionato

Progetto accademico di **Data Science** sviluppato per il Corso di Laurea Magistrale in Ingegneria Informatica e dell'Automazione (Università Politecnica delle Marche).

Il progetto applica una pipeline rigorosa di Machine Learning e Data Mining su oltre **42.000 brani musicali** estratti dall'API di Spotify (**Spotify Audio Features / Genres v2**), combinando **Data Cleaning**, **Analisi Esplorativa (EDA)**, **Classificazione Supervisionata Multiclasse con Ensemble Learning** e **Clustering Non Supervisionato (K-Means & DBSCAN)**.

---

## Struttura del Repository

```
Progetto-Data-Science/
│
├── Spotify_Audio_Analysis_Classification_Clustering.ipynb   # Master Jupyter Notebook con analisi, formule, codice e grafici pre-renderizzati
│
├── data/
│   ├── genres_cleaned.csv                                   # Dataset pulito (35.877 tracce uniche, 0 valori nulli)
│   └── project_summary.json                                 # Metriche riassuntive e parametri dei modelli
│
├── figures/                                                 # 23 grafici ad alta risoluzione (300 DPI)
│   ├── fig01_distribuzione_generi.png                       # Distribuzione dei 15 generi post-pulizia
│   ├── fig02_audio_features_hist_kde.png                    # Istogrammi + curve KDE per le feature audio continue
│   ├── fig03_discrete_features.png                          # Distribuzione di Key, Mode e Time Signature
│   ├── fig04_matrice_correlazione_iniziale.png              # Heatmap di correlazione di Pearson pre-sampling
│   ├── fig05_distribuzione_sampling.png                     # Bilanciamento Train Set: Pre vs Post SMOTE & Under
│   ├── fig06_matrice_correlazione_post_sampling.png         # Matrice di correlazione verificata post-sampling
│   ├── fig07_confronto_modelli_baseline.png                 # Barplot comparativo Accuratezza 8 modelli ML
│   ├── fig08_confusion_matrix_best_model.png                # Matrice di Confusione normalizzata (Ensemble)
│   ├── fig09_classification_report_heatmap.png              # Heatmap Precision, Recall ed F1 per genere
│   ├── fig10_roc_curves_multiclass.png                      # Curve ROC One-vs-Rest (Macro AUC = 0.94)
│   ├── fig11_precision_recall_curves.png                    # Curve Precision-Recall Multiclasse
│   ├── fig12_learning_curve.png                             # Learning Curve (Training vs Validation su XGBoost)
│   ├── fig13_feature_importance.png                         # Feature Importance (Random Forest vs XGBoost)
│   ├── fig14_pca_varianza_cumulativa.png                    # PCA Scree Plot varianza cumulativa
│   ├── fig15_tsne_2d_overview.png                           # Proiezione t-SNE 2D colorata per genere Spotify
│   ├── fig16_kmeans_elbow_wcss.png                          # Metodo del Gomito (WCSS per k da 2 a 11)
│   ├── fig17_kmeans_tsne_clusters.png                       # Cluster K-Means (k=5) su proiezione t-SNE
│   ├── fig18_kmeans_feature_profiling.png                   # Heatmap profiling caratteristiche medie per cluster
│   ├── fig19_kmeans_silhouette_plot.png                     # Diagramma Silhouette K-Means per cluster
│   ├── fig20_dbscan_knn_distance.png                        # Grafico k-NN distance per individuazione di Epsilon
│   ├── fig21_dbscan_grid_search.png                         # Matrice di tuning griglia (Epsilon x min_samples)
│   ├── fig22_dbscan_tsne_clusters_noise.png                 # Cluster DBSCAN con punti rumorosi (-1) evidenziati
│   └── fig23_dbscan_noise_profiling.png                     # Boxplot diagnostico caratteristiche del rumore
│
├── scripts/
│   ├── build_full_project.py                                # Pipeline Python end-to-end
│   └── generate_notebook.py                                 # Generatore programmatico del Notebook
│
├── genres_v2.csv                                            # Dataset grezzo iniziale (42.305 righe)
└── playlists.csv                                            # Metadati playlist Spotify
```

---

## 1. Data Preprocessing & Cleaning (ETL)

* **Deduplicazione su ID traccia:** Rimozione di **6.428 duplicati** presenti nel dataset originale dovuti all'estrazione da playlist sovrapposte. Brani unici preservati: **35.877**.
* **Rimozione variabili di servizio API:** Eliminazione di `Unnamed: 0`, `type`, `uri`, `track_href`, `analysis_url`, `song_name`, `title`.
* **Zero Missing Values:** 0 valori nulli su tutte le feature numeriche e sul target.
* **Prevenzione del Data Leakage:** Lo `StandardScaler` è stato calcolato (`fit`) **esclusivamente sul training set (80%)** e solo successivamente applicato (`transform`) sul test set (20%).
* **Bilanciamento delle Classi nel Training Set:** Applicazione combinata di **SMOTE** (oversampling per classi con < 2.000 campioni, es. *Pop*, *Rap*, *Emo*) e **RandomUnderSampler** (undersampling per classi sovrarappresentate come *Underground Rap*). Ogni classe nel training set è stata uniformata a esattamente **2.000 campioni** (30.000 campioni bilanciati), preservando il test set nella sua distribuzione originale per una stima imparziale.

---

## 2. Risultati della Classificazione Supervisionata (15 Generi)

| Algoritmo / Modello | Accuratezza Test Set | Weighted F1-Score | Macro F1-Score | Note |
| :--- | :---: | :---: | :---: | :--- |
| **Stacking Classifier (Meta: LR)** | **68.90%** | **0.6890** | **0.6580** | *Miglior modello assoluto (XGB + RF + LR)* |
| **Voting Classifier (Soft)** | **68.00%** | **0.6810** | **0.6540** | *Ensemble morbido basato su probabilità medie* |
| **XGBoost (Extreme Gradient Boost)** | **67.45%** | **0.6763** | **0.6480** | *Miglior classificatore singolo* |
| **Random Forest** | **67.34%** | **0.6726** | **0.6498** | *Elevata robustezza non lineare* |
| **Gradient Boosting** | **66.30%** | **0.6634** | **0.6397** | *Boosting sequenziale standard* |
| **Decision Tree** | **59.16%** | **0.5960** | **0.5734** | *Albero decisionale singolo* |
| **Logistic Regression (Baseline)** | **55.16%** | **0.5560** | **0.5279** | *Baseline lineare regolarizzata L2* |
| **KNN (K-Nearest Neighbors)** | **53.94%** | **0.5425** | **0.5213** | *Distanza euclidea k=9* |
| **Gaussian Naive Bayes** | **53.32%** | **0.5364** | **0.5198** | *Modello probabilistico generativo* |
| **AdaBoost** | **36.86%** | **0.3513** | **0.3261** | *Boosting debole su spazio multiclasse ad alta dimensionalità* |

### Diagnostica & Valutazione:
* **Macro-Average ROC-AUC:** **0.94** (capacità di separazione complessiva eccellente rispetto al baseline casuale di 0.50).
* **Feature Importance:** Le variabili con il più alto potere discriminante risultano essere `tempo` (BPM), `speechiness`, `danceability`, `loudness` ed `energy`.
* **Analisi degli Errori:** La matrice di confusione rivela che gli scambi avvengono esclusivamente tra sottogeneri con ritmica e timbrica contigua (es. *Dark Trap* scambiato con *Underground Rap* o *Trap Metal*, e *Techhouse* con *Techno*).

---

## 3. Risultati del Clustering Non Supervisionato

* **PCA (Principal Component Analysis):** Le prime 6 componenti principali spiegano oltre il **70% della varianza cumulativa totale**.
* **t-SNE 2D:** Riduzione non lineare con `perplexity=40` che mostra chiaramente la geometria delle strutture acustiche.
* **K-Means Clustering:**
  * **Scelta di $k$ (Elbow Method):** Punto di flesso della curva WCSS individuato a **$k = 5$**.
  * **Profilo delle 5 "Mood Playlists":**
    * *Cluster 0:* **Electronic & Party (High Energy, Loudness, Instrumental)**
    * *Cluster 1:* **Urban & Street (High Speechiness, High Danceability)**
    * *Cluster 2:* **Melodic Pop & R&B (High Valence, Acousticness media)**
    * *Cluster 3:* **High-Velocity Bass (Tempo estremo > 165 BPM - DnB/Hardstyle)**
    * *Cluster 4:* **Atmospheric & Melancholic (Low Loudness, Low Valence, Dark Mood)**
* **DBSCAN Clustering:**
  * Scelta di $\epsilon = 2.2$ tramite gomito del grafico **$k$-NN distance** ($k=5$).
  * Rilevati **8 cluster densi** e **6.9% di tracce isolate come Rumore/Outlier (`label = -1`)**.
  * L'analisi diagnostica degli outlier dimostra che si tratta di brani con caratteristiche estreme (durata > 7-10 minuti, spoken word/interviste senza musica o velocità oltre i 210 BPM).

---

## Requisiti e Come Eseguire

1. **Requisiti:** Python 3.10+ (testato su Python 3.14).
2. **Installazione dipendenze:**
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn seaborn matplotlib ipykernel nbformat nbclient
   ```
3. **Aprire il Notebook:**
   Puoi aprire ed esplorare direttamente [`Spotify_Audio_Analysis_Classification_Clustering.ipynb`](file:///c:/Users/lucab/OneDrive/Documenti/GitHub/Progetto-Data-Science/Spotify_Audio_Analysis_Classification_Clustering.ipynb) in VS Code o Jupyter Lab. Tutte le celle sono già pre-renderizzate con i relativi grafici e tabelle.
