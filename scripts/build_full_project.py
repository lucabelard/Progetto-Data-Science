"""
========================================================================================
Progetto di Data Science: Spotify Audio Analysis
Corso di Laurea Magistrale in Ingegneria Informatica e dell'Automazione (UnivPM)
Pipeline Completa: Data Cleaning, EDA, Classificazione Supervisionata e Clustering
========================================================================================
Autori: Gruppo di Progetto Data Science
Dataset: Spotify Tracks Dataset (Audio Features & Genres v2)

Questo script esegue l'intera pipeline di elaborazione in modo completamente riproducibile:
1. Data Cleaning & Preprocessing (rimozione duplicati, drop colonne di servizio, controllo nulli)
2. Analisi Esplorativa dei Dati (EDA: distribuzioni univariata continue e discrete, correlazioni)
3. Classificazione Supervisionata (SMOTE + undersampling, 8 modelli baseline, ensemble voting & stacking,
   matrice di confusione, curve ROC/PR, learning curve, feature importance)
4. Clustering Non Supervisionato (StandardScaler, PCA, t-SNE 2D, K-Means con Elbow e Silhouette,
   GMM per le appartenenze soft, DBSCAN e HDBSCAN con profiling dei punti periferici)
========================================================================================
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd

# Utilizziamo il backend 'Agg' per salvare i grafici direttamente su disco senza interfaccia GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Moduli scikit-learn per preprocessing, validazione e metriche
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# Algoritmi di classificazione supervisionata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Bilanciamento delle classi (imbalanced-learn)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Moduli per il clustering non supervisionato e riduzione dimensionale
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples

try:
    import hdbscan
    from hdbscan import all_points_membership_vectors
except ImportError as exc:
    raise ImportError(
        "La pipeline di clustering richiede la dipendenza 'hdbscan'. "
        "Installarla con: pip install hdbscan"
    ) from exc

# Silenziamo eventuali warning minori di FutureWarning di librerie esterne
warnings.filterwarnings('ignore', category=FutureWarning)

# Creazione delle cartelle di output se non già esistenti
os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Impostazioni estetiche globali per grafici chiari e professionali
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True


# ======================================================================================
# FASE 1: DATA CLEANING & ETL (Extract, Transform, Load)
# ======================================================================================
print(">>> [FASE 1/4] Inizio Data Cleaning ed ETL...")

# Carichiamo il dataset originale grezzo di Spotify
# Usiamo low_memory=False per evitare avvisi sui tipi misti nelle colonne di testo
df_raw = pd.read_csv('genres_v2.csv', low_memory=False)
initial_count = len(df_raw)
print(f"Dimensione dataset grezzo: {df_raw.shape[0]} righe, {df_raw.shape[1]} colonne")

# 1.1 Deduplicazione basata sull'ID univoco della traccia
# L'API di Spotify estrae brani da diverse playlist pubbliche: molte canzoni compaiono più volte.
# Rimuovere i duplicati è fondamentale per evitare bias e prevenire data leakage tra train e test.
df_dedup = df_raw.drop_duplicates(subset=['id']).copy()
dedup_count = len(df_dedup)
duplicates_removed = initial_count - dedup_count
print(f"Tracce duplicate rimosse (stesso Spotify ID): {duplicates_removed}")
print(f"Tracce uniche effettive rimaste: {dedup_count}")

# 1.2 Rimozione delle colonne di servizio generate dall'API
# Campi come ID, URL dell'API, indici spuri o titoli non hanno valore predittivo acustico
# e appesantirebbero inutilmente il calcolo delle distanze.
service_cols = ['Unnamed: 0', 'type', 'uri', 'track_href', 'analysis_url', 'song_name', 'title']
existing_service_cols = [c for c in service_cols if c in df_dedup.columns]
df_clean = df_dedup.drop(columns=existing_service_cols).copy()
print(f"Colonne di servizio rimosse: {existing_service_cols}")

# Definiamo la lista completa delle feature audio numeriche e del sottoinsieme continuo
audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
]

continuous_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
]

# 1.3 Controllo formale dell'assenza di valori mancanti (NaN)
null_counts = df_clean[audio_features + ['genre']].isnull().sum().to_dict()
print("Controllo valori nulli sulle feature acustiche e target (deve essere tutto 0):")
print(null_counts)

# Salvataggio del dataset pulito per riutilizzo immediato
df_clean.to_csv('data/genres_cleaned.csv', index=False)
print("Dataset pulito salvato con successo in 'data/genres_cleaned.csv'")


# ======================================================================================
# FASE 2: ANALISI ESPLORATIVA DEI DATI (EDA)
# ======================================================================================
print("\n>>> [FASE 2/4] Generazione Grafici Analisi Esplorativa (EDA)...")

# FIGURA 1: Distribuzione delle tracce per genere musicale
# Serve per visualizzare visivamente lo sbilanciamento naturale delle classi
plt.figure(figsize=(12, 6))
genre_counts = df_clean['genre'].value_counts()
sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette='crest', legend=False)
plt.title('Distribuzione dei Brani per Genere Musicale (Post-Deduplicazione)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Numero di Tracce', fontsize=12)
plt.ylabel('Genere', fontsize=12)

# Aggiungiamo i valori numerici a fianco di ciascuna barra per massima leggibilità
for i, v in enumerate(genre_counts.values):
    plt.text(v + 30, i, f"{v:,}", va='center', fontsize=10, color='#333333')
plt.tight_layout()
plt.savefig('figures/fig01_distribuzione_generi.png', dpi=300)
plt.close()

# FIGURA 2: Istogrammi con curve di stima di densità KDE per le feature audio continue
# Permette di osservare asimmetrie (skewness), bimodalità e ordini di grandezza
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()
for idx, col in enumerate(continuous_features):
    sns.histplot(df_clean[col], kde=True, ax=axes[idx], color='#2b5c8f', bins=30)
    axes[idx].set_title(f'Distribuzione: {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('Frequenza')

# Nascondiamo i riquadri vuoti non utilizzati nella griglia
for j in range(len(continuous_features), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Distribuzione delle Feature Audio Continue (Istogramma + Stima Densità KDE)', fontsize=16, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig('figures/fig02_audio_features_hist_kde.png', dpi=300)
plt.close()

# FIGURA 3: Distribuzione delle variabili musicali discrete (Key, Mode, Time Signature)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
sns.countplot(data=df_clean, x='key', ax=ax1, hue='key', palette='mako', legend=False)
ax1.set_title('Distribuzione Tonalità (Key: Pitch Class 0-11)', fontweight='bold')
ax1.set_xlabel('Key')

sns.countplot(data=df_clean, x='mode', ax=ax2, hue='mode', palette='Set2', legend=False)
ax2.set_title('Modalità (Mode: 0 = Minore, 1 = Maggiore)', fontweight='bold')
ax2.set_xlabel('Mode')

sns.countplot(data=df_clean, x='time_signature', ax=ax3, hue='time_signature', palette='viridis', legend=False)
ax3.set_title('Metro Ritmico (Time Signature)', fontweight='bold')
ax3.set_xlabel('Time Signature (es. 4/4)')
plt.tight_layout()
plt.savefig('figures/fig03_discrete_features.png', dpi=300)
plt.close()

# FIGURA 4: Matrice di correlazione iniziale tra le feature audio (Heatmap con annotazioni)
# Permette di identificare correlazioni lineari rilevanti (es. energy vs loudness, energy vs acousticness)
plt.figure(figsize=(11, 9))
corr_matrix = df_clean[audio_features].corr()
# Creiamo una maschera triangolare superiore per rendere la matrice più leggibile ed elegante
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Matrice di Correlazione di Pearson tra Audio Features (Pre-Sampling)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/fig04_matrice_correlazione_iniziale.png', dpi=300)
plt.close()


# ======================================================================================
# FASE 3: CLASSIFICAZIONE SUPERVISIONATA (PREVISIONE DEL GENERE MUSICALE)
# ======================================================================================
print("\n>>> [FASE 3/4] Preparazione e Addestramento Modelli di Classificazione...")

# Separazione matrice feature X e vettore target y
X = df_clean[audio_features].copy()
y = df_clean['genre'].copy()

# Codifica del target categorico in indici numerici (da 0 a 14)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"Numero totale di generi da classificare: {len(class_names)}")

# 3.1 Suddivisione rigorosa Train/Test (80% / 20%) con stratificazione
# La stratificazione garantisce che la proporzione dei generi sia identica in train e test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)
print(f"Campioni nel Training Set: {X_train_raw.shape[0]} | Campioni nel Test Set: {X_test_raw.shape[0]}")

# 3.2 Standardizzazione SENZA DATA LEAKAGE
# Calcoliamo media e deviazione standard SOLO sul train set (fit)
# Poi applichiamo la trasformazione a train e test. In questo modo il modello non ha
# alcuna informazione a priori sulla distribuzione dei dati di test.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# 3.3 Bilanciamento delle classi nel Training Set (SMOTE + Undersampling)
# Come da metodologia della tesi (Cap. 3.1):
# Portiamo ciascuna classe a una quota bilanciata di 2.000 campioni
# Le classi minoritarie vengono sovracampionate (SMOTE) creando punti sintetici interpolati
# Le classi maggioritarie vengono sottocampionate per evitare dominanza statistica.
target_per_class = 2000

# Oversampling su classi minoritarie
strategy_over = {c: target_per_class for c, count in zip(*np.unique(y_train, return_counts=True)) if count < target_per_class}
smote = SMOTE(sampling_strategy=strategy_over, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Undersampling su classi maggioritarie
strategy_under = {c: target_per_class for c, count in zip(*np.unique(y_train_sm, return_counts=True)) if count > target_per_class}
rus = RandomUnderSampler(sampling_strategy=strategy_under, random_state=42)
X_train_bal, y_train_bal = rus.fit_resample(X_train_sm, y_train_sm)
print(f"Training set post-sampling bilanciato: {X_train_bal.shape[0]} osservazioni (esattamente {target_per_class} per genere)")

# FIGURA 5: Grafico di confronto del bilanciamento Pre vs Post Sampling
pre_counts = pd.Series(y_train).value_counts().sort_index()
post_counts = pd.Series(y_train_bal).value_counts().sort_index()
sample_comp_df = pd.DataFrame({
    'Genere': class_names,
    'Pre-Sampling (Original Train)': pre_counts.values,
    'Post-Sampling (SMOTE + Under)': post_counts.values
})

plt.figure(figsize=(14, 6))
x_axis = np.arange(len(class_names))
plt.bar(x_axis - 0.2, sample_comp_df['Pre-Sampling (Original Train)'], width=0.4, label='Pre-Sampling (Originale)', color='#5c82a6')
plt.bar(x_axis + 0.2, sample_comp_df['Post-Sampling (SMOTE + Under)'], width=0.4, label='Post-Sampling (Bilanciato)', color='#f28e2b')
plt.xticks(x_axis, class_names, rotation=45, ha='right')
plt.ylabel('Numero di Campioni nel Training Set')
plt.title('Bilanciamento del Training Set: Confronto Pre vs Post Sampling (SMOTE + Undersampling)', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('figures/fig05_distribuzione_sampling.png', dpi=300)
plt.close()

# FIGURA 6: Matrice di correlazione verificata post-sampling
# Conferma che il bilanciamento sintetico non abbia alterato la struttura originale delle feature
plt.figure(figsize=(11, 9))
df_post_sample = pd.DataFrame(X_train_bal, columns=audio_features)
corr_post = df_post_sample.corr()
mask = np.triu(np.ones_like(corr_post, dtype=bool))
sns.heatmap(corr_post, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Matrice di Correlazione delle Feature Post-Sampling (Integrità Strutturale)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/fig06_matrice_correlazione_post_sampling.png', dpi=300)
plt.close()

# 3.4 Definizione e addestramento degli 8 modelli di Machine Learning
# Utilizziamo una serie completa di classificatori lineari, ad albero, basati su istanze e boosting
models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=12),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=18, random_state=42, n_jobs=-1),
    'KNN': KNeighborsClassifier(n_neighbors=9, n_jobs=-1),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=60, max_depth=4, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='mlogloss')
}

print("\nAddestramento e valutazione comparativa dei modelli sul Test Set:")
baseline_results = []
trained_models = {}

for name, model in models.items():
    t0 = time.time()
    # Addestriamo sul train set bilanciato
    model.fit(X_train_bal, y_train_bal)
    trained_models[name] = model
    
    # Valutiamo sul test set reale mai visto durante il training
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    f1_m = f1_score(y_test, y_pred, average='macro')
    train_time = time.time() - t0
    
    baseline_results.append({
        'Model': name,
        'Accuracy': acc,
        'Weighted F1': f1_w,
        'Macro F1': f1_m,
        'Time (s)': train_time
    })
    print(f"-> {name:<20}: Accuratezza = {acc:.4f} | Weighted F1 = {f1_w:.4f} ({train_time:.2f}s)")

baseline_df = pd.DataFrame(baseline_results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

# FIGURA 7: Confronto dell'accuratezza tra tutti i modelli
plt.figure(figsize=(12, 6))
sns.barplot(data=baseline_df, x='Accuracy', y='Model', hue='Model', palette='Blues_r', legend=False)
plt.title('Confronto Accuratezza Modelli di Classificazione sul Test Set', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Test Accuracy', fontsize=12)
plt.xlim(0, 0.8)
for i, v in enumerate(baseline_df['Accuracy']):
    plt.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig07_confronto_modelli_baseline.png', dpi=300)
plt.close()

# 3.5 Ensemble Learning: Voting Classifier e Stacking Classifier
# Combiniamo i classificatori più performanti per migliorare accuratezza e stabilità
print("\nCostruzione dei classificatori Ensemble (Voting & Stacking)...")

# Soft Voting: calcola la media ponderata delle distribuzioni di probabilità
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', trained_models['XGBoost']),
        ('rf', trained_models['Random Forest']),
        ('gb', trained_models['Gradient Boosting'])
    ],
    voting='soft'
)
voting_clf.fit(X_train_bal, y_train_bal)
y_pred_voting = voting_clf.predict(X_test)
acc_voting = accuracy_score(y_test, y_pred_voting)
f1_voting = f1_score(y_test, y_pred_voting, average='weighted')
print(f"Voting Classifier (Soft)      -> Accuratezza: {acc_voting:.4f} | Weighted F1: {f1_voting:.4f}")

# Stacking Classifier: usa un meta-modello (Logistic Regression) addestrato sulle predizioni out-of-fold
stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', trained_models['XGBoost']),
        ('rf', trained_models['Random Forest']),
        ('lr', trained_models['Logistic Regression'])
    ],
    final_estimator=LogisticRegression(max_iter=500, random_state=42),
    cv=3,
    n_jobs=-1
)
stacking_clf.fit(X_train_bal, y_train_bal)
y_pred_stacking = stacking_clf.predict(X_test)
acc_stacking = accuracy_score(y_test, y_pred_stacking)
f1_stacking = f1_score(y_test, y_pred_stacking, average='weighted')
print(f"Stacking Classifier (Meta LR) -> Accuratezza: {acc_stacking:.4f} | Weighted F1: {f1_stacking:.4f}")

# Selezioniamo il miglior modello per le analisi diagnostiche approfondite
best_model = voting_clf
y_pred_best = y_pred_voting
best_model_name = "Voting Classifier (XGB + RF + GB)"

# FIGURA 8: Matrice di Confusione Normalizzata (mostra dove il modello si sbaglia in termini percentuali)
cm = confusion_matrix(y_test, y_pred_best, normalize='true')
plt.figure(figsize=(13, 11))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'Matrice di Confusione Normalizzata - {best_model_name}', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Genere Predetto dal Modello', fontsize=12)
plt.ylabel('Genere Reale Effettivo', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('figures/fig08_confusion_matrix_best_model.png', dpi=300)
plt.close()

# FIGURA 9: Heatmap del Classification Report (Precision, Recall ed F1-Score per ogni genere)
report_dict = classification_report(y_test, y_pred_best, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).iloc[:-1, :-3].T
plt.figure(figsize=(10, 8))
sns.heatmap(report_df, annot=True, fmt='.2f', cmap='YlGnBu', vmin=0, vmax=1)
plt.title('Classification Report per Genere Musicale (Precision, Recall, F1)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/fig09_classification_report_heatmap.png', dpi=300)
plt.close()

# FIGURA 10: Curve ROC Multiclasse One-vs-Rest (confronto capacità discriminante con baseline AUC=0.5)
y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
y_score = best_model.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcolo del Macro-average ROC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(class_names)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= len(class_names)
macro_auc = auc(all_fpr, mean_tpr)

plt.figure(figsize=(12, 8))
plt.plot(all_fpr, mean_tpr, label=f'Macro-average ROC (AUC = {macro_auc:.2f})', color='navy', linestyle=':', linewidth=3)
colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
for i, color in zip(range(len(class_names)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificità)', fontsize=12)
plt.ylabel('True Positive Rate (Sensibilità / Recall)', fontsize=12)
plt.title('Curve ROC Multiclasse (Approccio One-vs-Rest)', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig10_roc_curves_multiclass.png', dpi=300)
plt.close()

# FIGURA 11: Curve Precision-Recall Multiclasse
plt.figure(figsize=(12, 8))
for i, color in zip(range(len(class_names)), colors):
    precision_c, recall_c, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    avg_p = average_precision_score(y_test_bin[:, i], y_score[:, i])
    plt.plot(recall_c, precision_c, color=color, lw=1.5, label=f'{class_names[i]} (AP = {avg_p:.2f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Curve Precision-Recall Multiclasse per Genere', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig11_precision_recall_curves.png', dpi=300)
plt.close()

# FIGURA 12: Learning Curve su XGBoost (Verifica dell'assenza di overfitting)
print("Calcolo della Learning Curve per verificare convergenza e overfitting...")
train_sizes, train_scores, val_scores = learning_curve(
    trained_models['XGBoost'], X_train_bal, y_train_bal,
    cv=3, train_sizes=np.linspace(0.2, 1.0, 5), scoring='accuracy', n_jobs=-1
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='#1f77b4', label='Accuratezza sul Training Set')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#1f77b4')
plt.plot(train_sizes, val_mean, 'o-', color='#2ca02c', label='Accuratezza in Cross-Validation')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#2ca02c')
plt.title('Learning Curve di XGBoost: Andamento al Variare dei Campioni di Addestramento', fontsize=14, fontweight='bold')
plt.xlabel('Numero di Campioni di Training Utilizzati', fontsize=12)
plt.ylabel('Accuratezza (Accuracy)', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig('figures/fig12_learning_curve.png', dpi=300)
plt.close()

# FIGURA 13: Feature Importance (Confronto tra Random Forest e XGBoost)
# Indica quali caratteristiche audio hanno il maggior peso decisionale
rf_fi = trained_models['Random Forest'].feature_importances_
xgb_fi = trained_models['XGBoost'].feature_importances_
fi_df = pd.DataFrame({
    'Feature': audio_features,
    'Random Forest': rf_fi,
    'XGBoost': xgb_fi
}).sort_values(by='XGBoost', ascending=True)

plt.figure(figsize=(12, 7))
y_pos = np.arange(len(fi_df))
plt.barh(y_pos - 0.2, fi_df['Random Forest'], height=0.4, label='Random Forest', color='#4e79a7')
plt.barh(y_pos + 0.2, fi_df['XGBoost'], height=0.4, label='XGBoost', color='#f28e2b')
plt.yticks(y_pos, fi_df['Feature'], fontsize=11)
plt.xlabel('Importanza Relativa Normalizzata', fontsize=12)
plt.title('Feature Importance: Caratteristiche Acustiche Determinanti per la Classificazione', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('figures/fig13_feature_importance.png', dpi=300)
plt.close()


# ======================================================================================
# FASE 4: CLUSTERING NON SUPERVISIONATO (SCOPERTA DI MOOD E PATTERN LATENTI)
# ======================================================================================
print("\n>>> [FASE 4/4] Inizio Clustering Non Supervisionato...", flush=True)

# 4.1 Selezione delle Feature per il Clustering e Standardizzazione
clustering_features = [
    'danceability', 'energy', 'speechiness', 'acousticness',
    'instrumentalness', 'valence', 'tempo', 'duration_ms'
]
X_clust_raw = df_clean[clustering_features].copy()

scaler_clust = StandardScaler()
X_clust_scaled = scaler_clust.fit_transform(X_clust_raw)

# 4.2 Riduzione di dimensionalità: PCA (Principal Component Analysis)
pca_full = PCA()
pca_full.fit(X_clust_scaled)
exp_var = pca_full.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

n_pca_components = 4
pca = PCA(n_components=n_pca_components, random_state=42)
X_pca_full = pca.fit_transform(X_clust_scaled)
print(f"PCA applicata su {len(clustering_features)} feature: {n_pca_components} componenti spiegano il {cum_var[n_pca_components-1]*100:.2f}% della varianza.", flush=True)

pca_loadings = pd.DataFrame(
    pca.components_.T,
    index=clustering_features,
    columns=[f'PC{i+1}' for i in range(n_pca_components)]
)

# FIGURA 14: Scree Plot della varianza spiegata e cumulativa PCA
plt.figure(figsize=(10, 6))
bars = plt.bar(range(1, len(cum_var)+1), exp_var, alpha=0.7, color='#4e79a7', edgecolor='black', label='Varianza Spiegata Singola')
plt.plot(range(1, len(cum_var)+1), cum_var, 'ro-', linewidth=2.5, markersize=8, label='Varianza Cumulativa')
plt.axhline(y=0.70, color='darkslategray', linestyle='--', linewidth=1.5, label='Soglia 70% Varianza Obiettivo (4 PC = 73.96%)')
plt.axvline(x=n_pca_components, color='#e15759', linestyle=':', linewidth=1.8, label=f'Componenti Selezionate (k_PCA = {n_pca_components}')
for i, (ev, cv) in enumerate(zip(exp_var, cum_var)):
    plt.text(i + 1, ev + 0.015, f"{ev*100:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2b5c8f')
plt.xticks(range(1, len(cum_var)+1), [f'PC{i}' for i in range(1, len(cum_var)+1)])
plt.ylim(0, 1.08)
plt.xlabel('Componente Principale (PC)', fontsize=12)
plt.ylabel('Quota di Varianza Spiegata', fontsize=12)
plt.title('PCA sulle 8 Feature Acustiche Selezionate: Scree Plot e Varianza Cumulativa', fontsize=14, fontweight='bold')
plt.legend(loc='center right', fontsize=10.5)
plt.tight_layout()
plt.savefig('figures/fig14_pca_varianza_cumulativa.png', dpi=300)
plt.close()

# 4.3 Campionamento rappresentativo e Proiezione 2D non lineare con t-SNE
sample_size = 6000
np.random.seed(42)
sample_indices = np.random.choice(len(X_clust_scaled), size=sample_size, replace=False)
X_cluster_sample = X_pca_full[sample_indices]  # Spazio PCA 4D per il clustering
X_scaled_sample = X_clust_scaled[sample_indices]
df_cluster_sample = df_clean.iloc[sample_indices].copy().reset_index(drop=True)
df_cluster_sample['pc_1'] = X_cluster_sample[:, 0]
df_cluster_sample['pc_2'] = X_cluster_sample[:, 1]

print(f"Esecuzione t-SNE bidimensionale su {sample_size} brani (perplexity=40, random_state=42)...", flush=True)
tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled_sample)
df_cluster_sample['tsne_1'] = X_tsne[:, 0]
df_cluster_sample['tsne_2'] = X_tsne[:, 1]

# FIGURA 15: Proiezione t-SNE 2D colorata per genere originale
plt.figure(figsize=(13, 9))
sns.scatterplot(
    data=df_cluster_sample, x='tsne_1', y='tsne_2', hue='genre',
    palette='tab20', alpha=0.65, s=25, edgecolor=None
)
plt.title('Mappa t-SNE Bidimensionale delle Audio Features (Colorata per Genere Ufficiale)', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Dimensione 1', fontsize=12)
plt.ylabel('t-SNE Dimensione 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('figures/fig15_tsne_2d_overview.png', dpi=300)
plt.close()

# 4.4 K-Means su spazio PCA (4 PC): Metodo del Gomito (WCSS) + Silhouette Score
print("Calcolo curve WCSS (Elbow Method) e Silhouette Score per K-Means su spazio PCA 4D da k=2 a k=10...", flush=True)
wcss = []
sil_scores = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels_k = km.fit_predict(X_cluster_sample)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cluster_sample, labels_k))

# FIGURA 16: Doppio grafico WCSS (Elbow Method) + Silhouette Score al variare di k
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.plot(k_range, wcss, 'bo-', linewidth=2.2, markersize=7)
ax1.scatter(5, wcss[3], color='crimson', s=150, zorder=5, label='Scelta Ottimale (k = 5)')
ax1.set_xticks(list(k_range))
ax1.set_xlabel('Numero di Cluster (k)', fontsize=12)
ax1.set_ylabel('Inerzia / WCSS (Spazio PCA 4D)', fontsize=12)
ax1.set_title('Metodo del Gomito (WCSS vs k)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)

ax2.plot(k_range, sil_scores, 'go-', linewidth=2.2, markersize=7)
ax2.scatter(5, sil_scores[3], color='crimson', s=150, zorder=5, label=f'Picco Strutturale k = 5 (Sil = {sil_scores[3]:.3f}')
ax2.set_xticks(list(k_range))
ax2.set_xlabel('Numero di Cluster (k)', fontsize=12)
ax2.set_ylabel('Silhouette Score Medio', fontsize=12)
ax2.set_title('Coefficiente di Silhouette Medio al Variare di k', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)

plt.suptitle('Selezione del Numero Ottimale di Cluster per K-Means nello Spazio PCA (4 Componenti)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig16_kmeans_elbow_wcss.png', dpi=300)
plt.close()

# 4.5 Addestramento K-Means con k=5 nello spazio PCA 4D e Profiling dei Cluster
k_opt = 5
km_final = KMeans(n_clusters=k_opt, random_state=42, n_init=20)
df_cluster_sample['kmeans_cluster'] = km_final.fit_predict(X_cluster_sample)
sil_kmeans = silhouette_score(X_cluster_sample, df_cluster_sample['kmeans_cluster'])
df_clean['kmeans_cluster'] = km_final.predict(X_pca_full)
print(f"K-Means (k={k_opt} su 4 PC) completato | Silhouette Score medio: {sil_kmeans:.4f}", flush=True)

cluster_names_short = {
    0: "C0: High-Intensity Bass & Workout",
    1: "C1: Extended Club & Instrumental",
    2: "C2: Acoustic, Chill & Atmospheric",
    3: "C3: Groove, Melodic Urban & Pop",
    4: "C4: Fast-Flow Rap & High-BPM"
}
cluster_legend_names = {
    0: "C0 · Workout / Bass",
    1: "C1 · Club / Strumentale",
    2: "C2 · Acustico / Chill",
    3: "C3 · Groove / Pop-Urban",
    4: "C4 · Rap / High-BPM"
}
cluster_display_order = [cluster_legend_names[i] for i in range(k_opt)]
cluster_color_map = {
    cluster_legend_names[i]: plt.cm.tab10(i / 10.0) for i in range(k_opt)
}
df_cluster_sample['kmeans_profile'] = df_cluster_sample['kmeans_cluster'].map(cluster_legend_names)

# FIGURA 17: Doppio grafico Cluster K-Means (Proiezione PCA 2D con Centroidi + Mappa t-SNE 2D)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.2))
sns.scatterplot(
    data=df_cluster_sample, x='pc_1', y='pc_2', hue='kmeans_profile',
    hue_order=cluster_display_order, palette=cluster_color_map,
    alpha=0.65, s=25, edgecolor=None, legend=False, ax=ax1
)
centroids_pca = km_final.cluster_centers_
ax1.scatter(
    centroids_pca[:, 0], centroids_pca[:, 1], c='black', marker='X', s=220,
    edgecolor='white', linewidth=1.5, zorder=10
)
ax1.set_title('Proiezione Lineare PCA 2D (PC1 vs PC2) con Centroidi', fontsize=15, fontweight='bold')
ax1.set_xlabel(f'PC1 ({exp_var[0]*100:.1f}% varianza)', fontsize=13)
ax1.set_ylabel(f'PC2 ({exp_var[1]*100:.1f}% varianza)', fontsize=13)

sns.scatterplot(
    data=df_cluster_sample, x='tsne_1', y='tsne_2', hue='kmeans_profile',
    hue_order=cluster_display_order, palette=cluster_color_map,
    alpha=0.7, s=25, edgecolor=None, legend=False, ax=ax2
)
ax2.set_title('Proiezione Non Lineare t-SNE 2D', fontsize=15, fontweight='bold')
ax2.set_xlabel('t-SNE Dimensione 1', fontsize=13)
ax2.set_ylabel('t-SNE Dimensione 2', fontsize=13)

from matplotlib.lines import Line2D
cluster_handles = [
    Line2D([0], [0], marker='o', linestyle='', markersize=8,
           markerfacecolor=cluster_color_map[name], markeredgecolor='none', label=name)
    for name in cluster_display_order
]
cluster_handles.append(Line2D([0], [0], marker='X', linestyle='', markersize=9,
                              markerfacecolor='black', markeredgecolor='white', label='Centroidi K-Means'))
fig.legend(handles=cluster_handles, loc='lower center', ncol=3,
           title='Profili sonori (ID K-Means con descrizione sintetica)',
           fontsize=11, title_fontsize=12, bbox_to_anchor=(0.5, 0.005), frameon=True)
fig.suptitle(f'Partizionamento K-Means (k={k_opt}, Silhouette={sil_kmeans:.3f})', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0.13, 1, 0.94])
plt.savefig('figures/fig17_kmeans_tsne_clusters.png', dpi=300)
plt.close()

# FIGURA 18: Matrice delle feature medie per Cluster K-Means
cluster_means = df_cluster_sample.groupby('kmeans_cluster')[continuous_features].mean()
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-6)

annot_df = pd.DataFrame(index=cluster_means.index, columns=cluster_means.columns, dtype=object)
for col in cluster_means.columns:
    for idx in cluster_means.index:
        val = cluster_means.loc[idx, col]
        if col == 'duration_ms':
            annot_df.loc[idx, col] = f"{val/1000:.0f}k"
        elif col == 'tempo':
            annot_df.loc[idx, col] = f"{val:.1f}"
        else:
            annot_df.loc[idx, col] = f"{val:.2f}"

fig, ax_heat = plt.subplots(figsize=(13, 8))
yticklabels_heat = [cluster_legend_names[i] for i in cluster_means.index]
sns.heatmap(
    cluster_means_norm, annot=annot_df, fmt='', cmap='vlag', center=0.5,
    annot_kws={'size': 13, 'weight': 'bold'},
    yticklabels=yticklabels_heat,
    cbar_kws={'label': 'Posizione relativa della media (Min-Max per feature)'}, ax=ax_heat
)
ax_heat.set_title('Medie delle feature per profilo K-Means', fontsize=17, fontweight='bold', pad=16)
ax_heat.set_xlabel('Feature acustiche (annotazioni: medie originali)', fontsize=13)
ax_heat.set_ylabel('')
ax_heat.tick_params(axis='x', rotation=25, labelsize=13)
ax_heat.tick_params(axis='y', rotation=0, labelsize=13)
ax_heat.collections[0].colorbar.set_label('Posizione relativa della media (Min-Max per feature)', fontsize=12)
ax_heat.collections[0].colorbar.ax.tick_params(labelsize=11)
fig.tight_layout()
plt.savefig('figures/fig18_kmeans_feature_profiling.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURA 19: Radar chart comparativo dei profili acustici K-Means
radar_feats = clustering_features
angles = np.linspace(0, 2 * np.pi, len(radar_feats), endpoint=False).tolist()
angles += angles[:1]
colors_tab10 = [plt.cm.tab10(i / 10.0) for i in range(k_opt)]

fig, ax_radar = plt.subplots(figsize=(12, 9), subplot_kw={'projection': 'polar'})
for i in range(k_opt):
    vals = cluster_means_norm.loc[i, radar_feats].values.flatten().tolist()
    vals += vals[:1]
    ax_radar.plot(angles, vals, linewidth=2.3, label=cluster_legend_names[i], color=colors_tab10[i])
    ax_radar.fill(angles, vals, color=colors_tab10[i], alpha=0.12)

ax_radar.set_theta_offset(np.pi / 2)
ax_radar.set_theta_direction(-1)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(radar_feats, fontsize=13, fontweight='bold')
ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_radar.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=10, color='gray')
ax_radar.set_title('Radar dei profili acustici K-Means', fontsize=17, fontweight='bold', pad=27)
ax_radar.legend(loc='upper left', bbox_to_anchor=(1.10, 1.08), fontsize=11, title='Profilo sonoro', title_fontsize=12)
fig.subplots_adjust(left=0.06, right=0.78, top=0.88, bottom=0.06)
plt.savefig('figures/fig19_kmeans_feature_radar.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURA 20: Grafico della Silhouette per K-Means nello spazio PCA 4D
sil_values_km = silhouette_samples(X_cluster_sample, df_cluster_sample['kmeans_cluster'])
plt.figure(figsize=(10, 7))
y_lower = 10
for i in range(k_opt):
    ith_cluster_sil = sil_values_km[df_cluster_sample['kmeans_cluster'] == i]
    ith_cluster_sil.sort()
    size_cluster_i = ith_cluster_sil.shape[0]
    y_upper = y_lower + size_cluster_i
    color = plt.cm.tab10(i / 10.0)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil, facecolor=color, edgecolor=color, alpha=0.75)
    plt.text(-0.08, y_lower + 0.5 * size_cluster_i, f"{cluster_names_short[i]} (n={size_cluster_i}", fontsize=9.5, fontweight='bold')
    y_lower = y_upper + 10

plt.axvline(x=sil_kmeans, color="red", linestyle="--", linewidth=2, label=f"Silhouette Media = {sil_kmeans:.3f}")
plt.title('Profilo dei Coefficienti di Silhouette per Cluster (K-Means k = 5 su 4 PC)', fontsize=14, fontweight='bold')
plt.xlabel('Coefficiente di Silhouette', fontsize=12)
plt.ylabel('Cluster e Dimensione Campionaria', fontsize=12)
plt.xlim(-0.22, 0.65)
plt.yticks([])
plt.legend(loc='upper right', fontsize=11)
plt.tight_layout()
plt.savefig('figures/fig19_kmeans_silhouette_plot.png', dpi=300)
plt.close()

# 4.6 DBSCAN: Clustering Basato sulla Densità nello Spazio PCA 4D
print("\nApplicazione dell'algoritmo DBSCAN nello spazio PCA 4D e ricerca del gomito per Epsilon...", flush=True)

k_min_samples = 5
nbrs = NearestNeighbors(n_neighbors=k_min_samples).fit(X_cluster_sample)
distances, indices = nbrs.kneighbors(X_cluster_sample)
sorted_distances = np.sort(distances[:, k_min_samples-1])

dbscan_eps_opt = 0.60
# FIGURA 20: Grafico delle distanze k-NN nello spazio PCA 4D per la scelta di Epsilon
plt.figure(figsize=(10, 6))
plt.plot(sorted_distances, color='#2b5c8f', linewidth=2.2, label=f'k-NN Distance (k = {k_min_samples}')
plt.axhline(y=dbscan_eps_opt, color='crimson', linestyle='--', linewidth=2, label=rf'Soglia di Gomito Scelta ($\epsilon = {dbscan_eps_opt}$)')
plt.title(f'Metodo k-Nearest Neighbors nello Spazio PCA (4 PC) per la Scelta di $\\epsilon$ (k={k_min_samples}', fontsize=13.5, fontweight='bold')
plt.xlabel('Brani del Campione Ordinati per Distanza Crescente', fontsize=12)
plt.ylabel(f'Distanza Euclidea dal {k_min_samples}° Vicino (Spazio PCA 4D)', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('figures/fig20_dbscan_knn_distance.png', dpi=300)
plt.close()

# FIGURA 21: Griglia di Tuning per DBSCAN (Numero di Cluster + Silhouette Score)
eps_candidates = [0.45, 0.50, 0.60, 0.70]
min_samples_candidates = [4, 5, 6, 8]
dbscan_grid_clusters = []
dbscan_grid_sil = []
for e in eps_candidates:
    row_n_clusters = []
    row_sil = []
    for ms in min_samples_candidates:
        db_temp = DBSCAN(eps=e, min_samples=ms).fit(X_cluster_sample)
        lbls = db_temp.labels_
        n_c = len(set(lbls)) - (1 if -1 in lbls else 0)
        row_n_clusters.append(n_c)
        mask_valid = lbls != -1
        if n_c >= 2 and mask_valid.sum() > n_c:
            row_sil.append(silhouette_score(X_cluster_sample[mask_valid], lbls[mask_valid]))
        else:
            row_sil.append(0.0)
    dbscan_grid_clusters.append(row_n_clusters)
    dbscan_grid_sil.append(row_sil)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(
    pd.DataFrame(dbscan_grid_clusters, index=eps_candidates, columns=min_samples_candidates),
    annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Numero di Cluster Identificati'}, ax=ax1
)
ax1.set_title(r'Numero di Cluster Identificati ($\epsilon$ vs min_samples)', fontsize=12.5, fontweight='bold', pad=12)
ax1.set_xlabel('min_samples', fontsize=11)
ax1.set_ylabel(r'$\epsilon$ (Raggio nello Spazio PCA 4D)', fontsize=11)

sns.heatmap(
    pd.DataFrame(dbscan_grid_sil, index=eps_candidates, columns=min_samples_candidates),
    annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Silhouette Score (Escluso Rumore)'}, ax=ax2
)
ax2.set_title(r'Silhouette Score dei Cluster Densi ($\epsilon$ vs min_samples)', fontsize=12.5, fontweight='bold', pad=12)
ax2.set_xlabel('min_samples', fontsize=11)
ax2.set_ylabel(r'$\epsilon$ (Raggio nello Spazio PCA 4D)', fontsize=11)

plt.suptitle('Griglia di Tuning Iperparametrica DBSCAN nello Spazio PCA a 4 Componenti', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig21_dbscan_grid_search.png', dpi=300)
plt.close()

# 4.7 Esecuzione DBSCAN con parametri ottimali (eps=0.60, min_samples=5) nello spazio PCA 4D
db_opt = DBSCAN(eps=dbscan_eps_opt, min_samples=k_min_samples)
df_cluster_sample['dbscan_cluster'] = db_opt.fit_predict(X_cluster_sample)
n_db_clusters = len(set(df_cluster_sample['dbscan_cluster'].tolist())) - (1 if -1 in set(df_cluster_sample['dbscan_cluster'].tolist()) else 0)
n_db_noise = int((df_cluster_sample['dbscan_cluster'] == -1).sum())
mask_non_noise = df_cluster_sample['dbscan_cluster'] != -1
sil_dbscan = float(silhouette_score(X_cluster_sample[mask_non_noise], df_cluster_sample.loc[mask_non_noise, 'dbscan_cluster']))
print(f"DBSCAN Ottimale (eps={dbscan_eps_opt}, min_samples={k_min_samples}: {n_db_clusters} cluster densi, {n_db_noise} outlier ({n_db_noise/sample_size*100:.2f}%), Sil non-noise={sil_dbscan:.4f}", flush=True)

# FIGURA 22: Cluster DBSCAN su PCA 2D e t-SNE 2D con evidenziazione chiara del rumore (outlier in rosso)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
non_noise = df_cluster_sample[df_cluster_sample['dbscan_cluster'] != -1]
noise = df_cluster_sample[df_cluster_sample['dbscan_cluster'] == -1]

sns.scatterplot(
    data=non_noise, x='pc_1', y='pc_2', hue='dbscan_cluster',
    palette='tab10', alpha=0.65, s=25, edgecolor=None, legend='full', ax=ax1
)
ax1.scatter(
    noise['pc_1'], noise['pc_2'], c='crimson', marker='x', s=35,
    label=f'Outlier / Rumore (n={n_db_noise}, {n_db_noise/sample_size*100:.1f}%)', alpha=0.85
)
ax1.set_title('Proiezione Lineare PCA 2D (PC1 vs PC2)', fontsize=13, fontweight='bold')
ax1.set_xlabel('PC1 (32.0% varianza)', fontsize=11)
ax1.set_ylabel('PC2 (16.5% varianza)', fontsize=11)
ax1.legend(loc='upper right', fontsize=8.5)

sns.scatterplot(
    data=non_noise, x='tsne_1', y='tsne_2', hue='dbscan_cluster',
    palette='tab10', alpha=0.65, s=25, edgecolor=None, legend='full', ax=ax2
)
ax2.scatter(
    noise['tsne_1'], noise['tsne_2'], c='crimson', marker='x', s=35,
    label=f'Outlier / Rumore (n={n_db_noise}, {n_db_noise/sample_size*100:.1f}%)', alpha=0.85
)
ax2.set_title('Proiezione Non Lineare t-SNE 2D', fontsize=13, fontweight='bold')
ax2.set_xlabel('t-SNE Dimensione 1', fontsize=11)
ax2.set_ylabel('t-SNE Dimensione 2', fontsize=11)
ax2.legend(loc='upper right', fontsize=8.5)

plt.suptitle(rf'Clusterizzazione DBSCAN nello Spazio PCA ($\epsilon={dbscan_eps_opt}, min\_samples={k_min_samples}$): Nucleo Denso e Outlier Acustici', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig22_dbscan_tsne_clusters_noise.png', dpi=300)
plt.close()

# FIGURA 23: Analisi diagnostica del rumore DBSCAN (Raincloud Plot: Half-Violin + Boxplot + Jitter)
from scipy import stats
import matplotlib.patches as patches
import shutil

df_cluster_sample['is_noise'] = (df_cluster_sample['dbscan_cluster'] == -1).map(
    {True: 'Rumore / Outlier DBSCAN', False: 'Cluster Core (Densità Ottimale)'}
)
df_cluster_sample['duration_min'] = df_cluster_sample['duration_ms'] / 60000.0

features_raincloud = [
    ('duration_min', 'Durata della Traccia (minuti)', [0, 8]),
    ('speechiness', 'Speechiness (Contenuto Parlato)', [0, 0.8]),
    ('acousticness', 'Acousticness (Timbrica Naturale vs Sintetica)', [0, 1.0])
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
rain_colors = {'Cluster Core (Densità Ottimale)': '#2980b9', 'Rumore / Outlier DBSCAN': '#e74c3c'}

for ax, (feat, feat_label, x_limits) in zip(axes, features_raincloud):
    for group_idx, (group_name, color) in enumerate(rain_colors.items()):
        sub_vals = df_cluster_sample[df_cluster_sample['is_noise'] == group_name][feat].dropna().values
        y_base = group_idx * 1.5

        # 1. Half-KDE (Cloud)
        kde = stats.gaussian_kde(sub_vals)
        x_eval = np.linspace(x_limits[0], x_limits[1], 200)
        kde_vals = kde(x_eval)
        kde_scaled = (kde_vals / kde_vals.max()) * 0.55
        ax.fill_between(x_eval, y_base, y_base + kde_scaled, color=color, alpha=0.55, edgecolor=color, lw=1.5)

        # 2. Boxplot minimale
        q25, q50, q75 = np.percentile(sub_vals, [25, 50, 75])
        ax.plot([q25, q75], [y_base - 0.12, y_base - 0.12], color=color, lw=4, solid_capstyle='round')
        ax.scatter([q50], [y_base - 0.12], color='white', edgecolor='#2c3e50', s=45, zorder=4)

        # 3. Jittered scatter (Rain)
        sample_pts = np.random.choice(sub_vals, size=min(400, len(sub_vals)), replace=False)
        jitter = np.random.uniform(-0.18, -0.04, size=len(sample_pts))
        ax.scatter(sample_pts, y_base - 0.2 + jitter, color=color, alpha=0.25, s=12, edgecolors='none')

    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(x_limits)
    ax.set_yticks([0.3, 1.8])
    ax.set_yticklabels(['Cluster Core', 'Rumore DBSCAN'], fontsize=11, fontweight='bold')
    ax.set_xlabel(feat_label, fontsize=11.5)
    ax.set_title(f"Distribuzione di {feat.replace('_min', ' (min)').capitalize()}", fontsize=13, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.5)

fig.suptitle('Raincloud Plot: Diagnosi Acustica della Separazione tra Core e Rumore DBSCAN\n(Half-Violin KDE + Boxplot + Jittered Data Cloud)', fontsize=14.5, fontweight='bold', y=1.03)
fig.tight_layout()
plt.savefig('figures/fig23_dbscan_noise_profiling.png', dpi=300, bbox_inches='tight')
plt.close()
shutil.copy('figures/fig23_dbscan_noise_profiling.png', 'Relazione/capitolo4/img/fig23_dbscan_noise_profiling.png')

# 4.8 GMM: evoluzione probabilistica dei cinque profili K-Means
# Si mantiene k=5 per un confronto diretto con K-Means; le probabilita' sono
# affiliazioni stimate ai componenti, non probabilita' calibrate di un mood umano.
print("\nStima delle appartenenze probabilistiche GMM (5 componenti, spazio PCA 4D)...", flush=True)
gmm_final = GaussianMixture(
    n_components=k_opt,
    covariance_type='full',
    reg_covar=1e-5,
    n_init=10,
    random_state=42
)
gmm_final.fit(X_cluster_sample)
gmm_memberships = gmm_final.predict_proba(X_cluster_sample)
gmm_labels = gmm_memberships.argmax(axis=1)
gmm_sorted = np.sort(gmm_memberships, axis=1)
gmm_top1 = gmm_sorted[:, -1]
gmm_top2_gap = gmm_sorted[:, -1] - gmm_sorted[:, -2]
gmm_membership_columns = [f'gmm_affinity_{i}' for i in range(k_opt)]
for component_idx, column in enumerate(gmm_membership_columns):
    df_cluster_sample[column] = gmm_memberships[:, component_idx]
df_cluster_sample['gmm_component'] = gmm_labels
df_cluster_sample['gmm_top_affinity'] = gmm_top1
df_cluster_sample['gmm_top2_gap'] = gmm_top2_gap
# Gap <= 0.15 e' una soglia esplorativa per estrarre casi da ispezionare,
# non una soglia universale di appartenenza.
df_cluster_sample['gmm_cross_mood_candidate'] = gmm_top2_gap <= 0.15
gmm_component_profile = df_cluster_sample.groupby('gmm_component')[clustering_features].mean()
gmm_component_genres = {
    int(component): group['genre'].value_counts().head(5).to_dict()
    for component, group in df_cluster_sample.groupby('gmm_component')
}
gmm_component_display = {
    0: 'GMM 0 · Energetico / strumentale',
    1: 'GMM 1 · Acustico / moderato',
    2: 'GMM 2 · Energetico / vocale',
    3: 'GMM 3 · Dance / strumentale',
    4: 'GMM 4 · Urban / vocale'
}
gmm_display_order = [gmm_component_display[i] for i in range(k_opt)]
gmm_color_map = {gmm_component_display[i]: plt.cm.tab10(i / 10.0) for i in range(k_opt)}
df_cluster_sample['gmm_profile'] = df_cluster_sample['gmm_component'].map(gmm_component_display)

# FIGURA 24: Mappa Topografica GMM ad Ellissi di Covarianza e Corridoi Ibridi
fig, (ax_main, ax_side) = plt.subplots(1, 2, figsize=(18, 8.5), gridspec_kw={'width_ratios': [2.2, 1]})

sns.kdeplot(
    data=df_cluster_sample, x='pc_1', y='pc_2',
    levels=10, cmap='Blues', alpha=0.3, fill=True, ax=ax_main, thresh=0.03
)

gmm_palette_5 = ['#2b5c8f', '#27ae60', '#e67e22', '#8e44ad', '#c0392b']
for i in range(k_opt):
    sub = df_cluster_sample[(df_cluster_sample['gmm_component'] == i) & (~df_cluster_sample['gmm_cross_mood_candidate'])]
    ax_main.scatter(
        sub['pc_1'], sub['pc_2'],
        c=gmm_palette_5[i], alpha=0.30, s=20, edgecolors='none',
        label=gmm_display_order[i]
    )

    # Ellissi di covarianza GMM (1 sigma e 2 sigma)
    mean_2d = gmm_final.means_[i, :2]
    cov_2d = gmm_final.covariances_[i][:2, :2]
    vals, vecs = np.linalg.eigh(cov_2d)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    w_1sig, h_1sig = 2 * np.sqrt(vals)
    ell_1 = patches.Ellipse(
        xy=mean_2d, width=w_1sig, height=h_1sig, angle=theta,
        edgecolor=gmm_palette_5[i], facecolor='none', linewidth=2.5, linestyle='-'
    )
    ax_main.add_patch(ell_1)

    w_2sig, h_2sig = 2 * np.sqrt(vals * 5.991)
    ell_2 = patches.Ellipse(
        xy=mean_2d, width=w_2sig, height=h_2sig, angle=theta,
        edgecolor=gmm_palette_5[i], facecolor='none', linewidth=1.5, linestyle='--', alpha=0.7
    )
    ax_main.add_patch(ell_2)

    ax_main.scatter(mean_2d[0], mean_2d[1], c=gmm_palette_5[i], s=180, marker='X', edgecolors='white', linewidths=2, zorder=6)
    ax_main.text(mean_2d[0]+0.15, mean_2d[1]+0.15, f'C{i}', fontsize=12, fontweight='bold',
                 color='#111111', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=gmm_palette_5[i]))

# Brani Cross-Mood evidenziati (Gap <= 0.15)
cross_tracks = df_cluster_sample[df_cluster_sample['gmm_cross_mood_candidate']]
ax_main.scatter(
    cross_tracks['pc_1'], cross_tracks['pc_2'],
    c='#f1c40f', s=60, marker='D', edgecolors='#2c3e50', linewidths=1.2,
    zorder=5, label=f'Brani Ponte Cross-Mood (N={len(cross_tracks)})'
)

if len(cross_tracks) > 15:
    sample_bridge = cross_tracks.iloc[15]
    ax_main.annotate(
        "Brano Ponte Ibrido\n(Affinità bilanciata tra 2 Archetipi)",
        xy=(sample_bridge['pc_1'], sample_bridge['pc_2']),
        xytext=(sample_bridge['pc_1'] + 1.2, sample_bridge['pc_2'] - 1.4),
        arrowprops=dict(facecolor='#d35400', shrink=0.08, width=1.5, headwidth=8),
        fontsize=10.5, fontweight='bold', color='#a04000',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor='#f39c12', alpha=0.95)
    )

ax_main.set_title('Mappa Topografica GMM: Ellissi di Covarianza (1σ e 2σ) e Corridoi Ibridi', fontsize=14, fontweight='bold')
ax_main.set_xlabel(f"PC1 ({exp_var[0] * 100:.1f}% varianza)", fontsize=11.5)
ax_main.set_ylabel(f"PC2 ({exp_var[1] * 100:.1f}% varianza)", fontsize=11.5)
ax_main.legend(loc='upper right', fontsize=9.5, framealpha=0.9)
ax_main.grid(True, linestyle=':', alpha=0.4)

# Pannello Classifica Corridoi Ibridi
bridge_pairs = []
for _, row in cross_tracks.iterrows():
    affs = row[gmm_membership_columns].to_numpy()
    top2 = np.argsort(affs)[-2:]
    pair = tuple(sorted([top2[0], top2[1]]))
    bridge_pairs.append(pair)

pair_counts = pd.Series(bridge_pairs).value_counts().head(8)
pair_labels = [f"C{p[0]} ↔ C{p[1]}\n({gmm_display_order[p[0]].split('·')[1].strip()[:10]} / {gmm_display_order[p[1]].split('·')[1].strip()[:10]})" for p in pair_counts.index]

ax_side.barh(range(len(pair_counts)), pair_counts.values, color='#e67e22', edgecolor='#2c3e50', alpha=0.85)
ax_side.set_yticks(range(len(pair_counts)))
ax_side.set_yticklabels(pair_labels, fontsize=9.5)
ax_side.invert_yaxis()
ax_side.set_xlabel("Numero di Tracce Ponte Condivise", fontsize=11)
ax_side.set_title("Corridoi Ibridi più Frequenti\n(Opportunità di Playlist Cross-Mood)", fontsize=13, fontweight='bold')
for i, v in enumerate(pair_counts.values):
    ax_side.text(v + 1.5, i, f"{v} brani", va='center', fontsize=9.5, fontweight='bold', color='#7e3805')
ax_side.grid(axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/fig24_gmm_soft_membership.png', dpi=300)
plt.close()
shutil.copy('figures/fig24_gmm_soft_membership.png', 'Relazione/capitolo4/img/fig24_gmm_soft_membership.png')

# FIGURA 24b: Pentagono Baricentrico dei Mood (RadViz Simplex)
fig_rad, ax_rad = plt.subplots(figsize=(11, 10))
rad_angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, 6)[:-1]
rad_vertices = np.column_stack([np.cos(rad_angles), np.sin(rad_angles)])
rad_theta_circle = np.linspace(0, 2*np.pi, 200)

for r_guide in [0.25, 0.50, 0.75, 1.0]:
    ax_rad.plot(r_guide * np.cos(rad_theta_circle), r_guide * np.sin(rad_theta_circle), color='gray', linestyle=':', alpha=0.45)
    ax_rad.text(0.02, r_guide, f"{int(r_guide*100)}%", fontsize=8.5, color='gray', alpha=0.8)

pent_x = np.append(rad_vertices[:, 0], rad_vertices[0, 0])
pent_y = np.append(rad_vertices[:, 1], rad_vertices[0, 1])
ax_rad.plot(pent_x, pent_y, color='#2c3e50', linewidth=2, linestyle='-', alpha=0.7)

for i in range(5):
    ax_rad.plot([0, rad_vertices[i, 0]], [0, rad_vertices[i, 1]], color='#7f8c8d', linestyle='--', alpha=0.5)

P_rad = np.zeros((len(df_cluster_sample), 2))
for i in range(5):
    aff = df_cluster_sample[f'gmm_affinity_{i}'].to_numpy()[:, np.newaxis]
    P_rad += aff * rad_vertices[i]

df_cluster_sample['radviz_x'] = P_rad[:, 0]
df_cluster_sample['radviz_y'] = P_rad[:, 1]

for i in range(5):
    sub = df_cluster_sample[(df_cluster_sample['gmm_component'] == i) & (~df_cluster_sample['gmm_cross_mood_candidate'])]
    ax_rad.scatter(
        sub['radviz_x'], sub['radviz_y'],
        c=gmm_palette_5[i], alpha=0.35, s=22, edgecolors='none',
        label=gmm_display_order[i]
    )

cross_rad = df_cluster_sample[df_cluster_sample['gmm_cross_mood_candidate']]
ax_rad.scatter(
    cross_rad['radviz_x'], cross_rad['radviz_y'],
    c='#f1c40f', s=55, marker='D', edgecolors='#2c3e50', linewidths=1.2,
    zorder=5, label='Tracce Ponte Cross-Mood (Equidistanti)'
)

rad_labels = [
    "C0: Energetico / Strumentale\n(Rock / Metal)",
    "C1: Acustico / Intimo\n(Folk / Classica)",
    "C2: Energetico / Vocale\n(Pop / Dance-Pop)",
    "C3: Dance / Elettronico\n(EDM / Techno)",
    "C4: Urban / Ritmico\n(Rap / Hip-Hop)"
]

offset_scale = 1.18
for i in range(5):
    vx, vy = rad_vertices[i]
    ax_rad.scatter(vx, vy, c=gmm_palette_5[i], s=240, edgecolors='#2c3e50', linewidths=2.5, zorder=6)
    align_h = 'center'
    if vx > 0.2: align_h = 'left'
    elif vx < -0.2: align_h = 'right'
    align_v = 'center'
    if vy > 0.3: align_v = 'bottom'
    elif vy < -0.3: align_v = 'top'

    ax_rad.text(
        vx * offset_scale, vy * offset_scale, rad_labels[i],
        ha=align_h, va=align_v, fontsize=10.5, fontweight='bold',
        color='#1a252f', bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=gmm_palette_5[i], alpha=0.92, lw=1.5)
    )

ax_rad.set_xlim(-1.55, 1.55)
ax_rad.set_ylim(-1.45, 1.55)
ax_rad.set_aspect('equal')
ax_rad.axis('off')
ax_rad.set_title("Pentagono Baricentrico dei Mood Musicali (RadViz Simplex)\nVisualizzazione dello Spazio delle Probabilità Soft GMM", fontsize=15, fontweight='bold', pad=20)
ax_rad.legend(loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=9.5, framealpha=0.95)

plt.tight_layout()
plt.savefig('figures/fig24b_mood_pentagon_radviz.png', dpi=300)
plt.close()
shutil.copy('figures/fig24b_mood_pentagon_radviz.png', 'Relazione/capitolo4/img/fig24b_mood_pentagon_radviz.png')
print(
    f"GMM: candidati a cavallo tra componenti (gap top-2 <= 0.15) "
    f"{int(df_cluster_sample['gmm_cross_mood_candidate'].sum())}/ {sample_size}; "
    f"affiliazione massima mediana={np.median(gmm_top1):.3f}.", flush=True
)
dbscan_noise_mask = df_cluster_sample['dbscan_cluster'].eq(-1).to_numpy()
dbscan_noise_gmm_cross_mood = int((dbscan_noise_mask & (gmm_top2_gap <= 0.15)).sum())
dbscan_noise_gmm_mean_top = float(gmm_top1[dbscan_noise_mask].mean()) if dbscan_noise_mask.any() else 0.0
dbscan_noise_gmm_median_top = float(np.median(gmm_top1[dbscan_noise_mask])) if dbscan_noise_mask.any() else 0.0

# 4.9 HDBSCAN: densita' variabile e affiliazioni soft dei punti periferici
# HDBSCAN conserva l'etichetta noise: le affiliazioni soft ne esplorano le
# vicinanze, senza trasformarle automaticamente in assegnazioni certe.
print("\nApplicazione HDBSCAN e stima delle affiliazioni soft...", flush=True)
hdbscan_min_cluster_size = 30
hdbscan_min_samples = 5
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=hdbscan_min_cluster_size,
    min_samples=hdbscan_min_samples,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)
hdbscan_model.fit(X_cluster_sample)
hdbscan_labels = hdbscan_model.labels_
hdbscan_noise_mask = hdbscan_labels == -1
n_hdbscan_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
n_hdbscan_noise = int(hdbscan_noise_mask.sum())
if n_hdbscan_clusters:
    hdbscan_soft = all_points_membership_vectors(hdbscan_model)
    hdbscan_top_component = hdbscan_soft.argmax(axis=1)
    hdbscan_top_affinity = hdbscan_soft.max(axis=1)
    hdbscan_total_affinity = hdbscan_soft.sum(axis=1)
    hdbscan_soft_columns = [f'hdbscan_affinity_{i}' for i in range(n_hdbscan_clusters)]
else:
    hdbscan_soft = np.zeros((sample_size, 0))
    hdbscan_top_component = np.full(sample_size, -1, dtype=int)
    hdbscan_top_affinity = np.zeros(sample_size)
    hdbscan_total_affinity = np.zeros(sample_size)
    hdbscan_soft_columns = []

df_cluster_sample['hdbscan_cluster'] = hdbscan_labels
df_cluster_sample['hdbscan_cluster_strength'] = hdbscan_model.probabilities_
df_cluster_sample['hdbscan_top_soft_component'] = hdbscan_top_component
df_cluster_sample['hdbscan_top_soft_affinity'] = hdbscan_top_affinity
df_cluster_sample['hdbscan_total_soft_affinity'] = hdbscan_total_affinity
for component_idx, column in enumerate(hdbscan_soft_columns):
    df_cluster_sample[column] = hdbscan_soft[:, component_idx]

dbscan_noise_hdbscan_recovered = int((df_cluster_sample['dbscan_cluster'].eq(-1) & ~hdbscan_noise_mask).sum())
dbscan_noise_hdbscan_still_noise = df_cluster_sample['dbscan_cluster'].eq(-1) & hdbscan_noise_mask
dbscan_noise_soft_candidates = int(
    (dbscan_noise_hdbscan_still_noise & (hdbscan_top_affinity >= 0.10) & (hdbscan_total_affinity >= 0.25)).sum()
)
print(
    f"HDBSCAN: {n_hdbscan_clusters} cluster, {n_hdbscan_noise}/{sample_size} punti ancora noise "
    f"({n_hdbscan_noise / sample_size * 100:.2f}%). Dei punti DBSCAN noise, "
    f"{dbscan_noise_hdbscan_recovered} sono in cluster HDBSCAN e "
    f"{dbscan_noise_soft_candidates} hanno un'affiliazione soft candidata da esplorare.", flush=True
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
hdbscan_assigned = df_cluster_sample.loc[~hdbscan_noise_mask]
hdbscan_noise = df_cluster_sample.loc[hdbscan_noise_mask]
if not hdbscan_assigned.empty:
    sns.scatterplot(
        data=hdbscan_assigned, x='pc_1', y='pc_2', hue='hdbscan_cluster',
        palette='tab10', alpha=0.65, s=24, edgecolor=None, ax=ax1
    )
if not hdbscan_noise.empty:
    ax1.scatter(hdbscan_noise['pc_1'], hdbscan_noise['pc_2'], c='gray', marker='x', s=34,
                alpha=0.8, label=f'Noise HDBSCAN (n={n_hdbscan_noise})')
ax1.set_title('HDBSCAN: cluster densi e brani periferici', fontweight='bold')
ax1.set_xlabel(f\"PC1 ({exp_var[0] * 100:.1f}% varianza)\")
ax1.set_ylabel(f\"PC2 ({exp_var[1] * 100:.1f}% varianza)\")
ax1.legend(fontsize=8, loc='best')

dbscan_noise_sample = df_cluster_sample.loc[df_cluster_sample['dbscan_cluster'].eq(-1)]
if not dbscan_noise_sample.empty:
    scatter = ax2.scatter(
        dbscan_noise_sample['pc_1'], dbscan_noise_sample['pc_2'],
        c=dbscan_noise_sample['gmm_top_affinity'], cmap='viridis',
        vmin=0, vmax=1, s=32, alpha=0.85
    )
    fig.colorbar(scatter, ax=ax2, label='Affiliazione GMM massima')
    ax2.set_title('Punti noise DBSCAN: affinità ai 5 profili GMM', fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'Nessun punto noise DBSCAN', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Affiliazioni dei punti noise DBSCAN', fontweight='bold')
ax2.set_xlabel(f"PC1 ({exp_var[0] * 100:.1f}% varianza)")
ax2.set_ylabel(f"PC2 ({exp_var[1] * 100:.1f}% varianza)")
plt.suptitle('HDBSCAN: il noise è geometrico; le affinità GMM aiutano a esplorare i brani', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig25_hdbscan_soft_membership.png', dpi=300)
plt.close()

# Esportazione delle appartenenze per rendere ispezionabili i candidati playlist.
membership_export_columns = ['id', 'genre', 'dbscan_cluster', 'gmm_component', 'gmm_top_affinity',
                             'gmm_top2_gap', 'gmm_cross_mood_candidate', 'hdbscan_cluster',
                             'hdbscan_cluster_strength', 'hdbscan_top_soft_component',
                             'hdbscan_top_soft_affinity', 'hdbscan_total_soft_affinity']
membership_export_columns += gmm_membership_columns + hdbscan_soft_columns
df_cluster_sample[membership_export_columns].to_csv('data/clustering_soft_memberships_sample.csv', index=False)

# Salvataggio delle statistiche e dei risultati in un file JSON
summary_stats = {
    'raw_rows': initial_count,
    'duplicates_removed': duplicates_removed,
    'cleaned_rows': dedup_count,
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'balanced_train_samples': len(X_train_bal),
    'baseline_table': baseline_df.to_dict(orient='records'),
    'voting_acc': round(acc_voting, 4),
    'voting_f1': round(f1_voting, 4),
    'stacking_acc': round(acc_stacking, 4),
    'stacking_f1': round(f1_stacking, 4),
    'pca_features': clustering_features,
    'pca_n_components': n_pca_components,
    'pca_explained_variance': [round(float(x), 4) for x in exp_var],
    'pca_cumulative_variance': [round(float(x), 4) for x in cum_var],
    'pca_loadings': pca_loadings.round(4).to_dict(),
    'kmeans_k': k_opt,
    'kmeans_sil': round(float(sil_kmeans), 4),
    'kmeans_sample_counts': {int(k): int(v) for k, v in df_cluster_sample['kmeans_cluster'].value_counts().sort_index().items()},
    'kmeans_full_counts': {int(k): int(v) for k, v in df_clean['kmeans_cluster'].value_counts().sort_index().items()},
    'dbscan_eps': dbscan_eps_opt,
    'dbscan_min_samples': k_min_samples,
    'dbscan_clusters': n_db_clusters,
    'dbscan_noise': int(n_db_noise),
    'dbscan_noise_pct': round(n_db_noise/sample_size*100, 2),
    'dbscan_sil_non_noise': round(sil_dbscan, 4),
    'gmm_components': k_opt,
    'gmm_covariance_type': 'full',
    'gmm_mean_top_affinity': round(float(np.mean(gmm_top1)), 4),
    'gmm_median_top_affinity': round(float(np.median(gmm_top1)), 4),
    'gmm_cross_mood_gap_threshold': 0.15,
    'gmm_cross_mood_candidates': int(df_cluster_sample['gmm_cross_mood_candidate'].sum()),
    'gmm_cross_mood_candidate_pct': round(float(df_cluster_sample['gmm_cross_mood_candidate'].mean() * 100), 2),
    'gmm_component_profiles': gmm_component_profile.round(4).to_dict(orient='index'),
    'gmm_component_top_genres_posthoc': gmm_component_genres,
    'hdbscan_min_cluster_size': hdbscan_min_cluster_size,
    'hdbscan_min_samples': hdbscan_min_samples,
    'hdbscan_clusters': int(n_hdbscan_clusters),
    'hdbscan_noise': int(n_hdbscan_noise),
    'hdbscan_noise_pct': round(n_hdbscan_noise / sample_size * 100, 2),
    'hdbscan_dbscan_noise_recovered': int(dbscan_noise_hdbscan_recovered),
    'hdbscan_dbscan_noise_still_noise': int(dbscan_noise_hdbscan_still_noise.sum()),
    'hdbscan_dbscan_noise_soft_candidates': int(dbscan_noise_soft_candidates),
    'hdbscan_noise_mean_top_soft_affinity': round(float(hdbscan_top_affinity[hdbscan_noise_mask].mean()), 4) if n_hdbscan_noise else 0.0,
    'hdbscan_noise_mean_total_soft_affinity': round(float(hdbscan_total_affinity[hdbscan_noise_mask].mean()), 4) if n_hdbscan_noise else 0.0,
    'dbscan_noise_gmm_cross_mood_candidates': int(dbscan_noise_gmm_cross_mood),
    'dbscan_noise_gmm_cross_mood_candidate_pct': round(float(dbscan_noise_gmm_cross_mood / max(1, n_db_noise) * 100), 2),
    'dbscan_noise_gmm_mean_top_affinity': round(dbscan_noise_gmm_mean_top, 4),
    'dbscan_noise_gmm_median_top_affinity': round(dbscan_noise_gmm_median_top, 4)
}

with open('data/project_summary.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("Figures fig14-fig25, soft-membership sample export and data/project_summary.json updated successfully!", flush=True)
