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
   DBSCAN con k-NN distance graph, griglia di tuning e profiling degli outlier)
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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples

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
print("\n>>> [FASE 4/4] Inizio Clustering Non Supervisionato...")

# Standardizziamo l'intero dataset numerico, omettendo totalmente il target 'genre'
# L'approccio è rigorosamente agnostico rispetto all'etichetta di mercato
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X)

# 4.1 Riduzione di dimensionalità: PCA (Principal Component Analysis)
pca = PCA()
pca.fit(X_full_scaled)
exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

# FIGURA 14: Scree Plot della varianza cumulativa PCA (individua quante componenti spiegano il 70%)
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cum_var)+1), exp_var, alpha=0.6, color='#4e79a7', label='Varianza Spiegata Singola')
plt.step(range(1, len(cum_var)+1), cum_var, where='mid', color='#e15759', linewidth=2, label='Varianza Cumulativa')
plt.axhline(y=0.70, color='gray', linestyle='--', label='Soglia 70% Varianza Obiettivo')
plt.xticks(range(1, len(cum_var)+1))
plt.xlabel('Componente Principale (PC)', fontsize=12)
plt.ylabel('Quota di Varianza Spiegata', fontsize=12)
plt.title('PCA: Scree Plot e Analisi della Varianza Cumulativa', fontsize=14, fontweight='bold')
plt.legend(loc='center right', fontsize=11)
plt.tight_layout()
plt.savefig('figures/fig14_pca_varianza_cumulativa.png', dpi=300)
plt.close()

# 4.2 Proiezione 2D non lineare con t-SNE
# Utilizziamo un campione rappresentativo di 6.000 brani per calcolo rapido e massima nitidezza visiva
sample_size = 6000
np.random.seed(42)
sample_indices = np.random.choice(len(X_full_scaled), size=sample_size, replace=False)
X_cluster_sample = X_full_scaled[sample_indices]
df_cluster_sample = df_clean.iloc[sample_indices].copy().reset_index(drop=True)

print(f"Esecuzione t-SNE bidimensionale su {sample_size} brani (perplexity=40, random_state=42)...")
tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_cluster_sample)
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

# 4.3 K-Means: Metodo del Gomito (Elbow Method) per determinare il k ottimale
print("Calcolo curva WCSS (Elbow Method) per K-Means da k=2 a k=11...")
wcss = []
k_range = range(2, 12)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster_sample)
    wcss.append(km.inertia_)

# FIGURA 16: Metodo del Gomito (Elbow Method)
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=7)
plt.scatter(5, wcss[3], color='red', s=150, zorder=5, label='Punto di Gomito Scelto (k = 5)')
plt.xticks(k_range)
plt.xlabel('Numero di Cluster (k)', fontsize=12)
plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
plt.title('Metodo del Gomito (Elbow Method) per la Scelta di K nel K-Means', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('figures/fig16_kmeans_elbow_wcss.png', dpi=300)
plt.close()

# 4.4 Addestramento K-Means con k=5 e Profiling dei Cluster
k_opt = 5
km_final = KMeans(n_clusters=k_opt, random_state=42, n_init=20)
df_cluster_sample['kmeans_cluster'] = km_final.fit_predict(X_cluster_sample)
sil_kmeans = silhouette_score(X_cluster_sample, df_cluster_sample['kmeans_cluster'])
print(f"K-Means (k={k_opt}) completato | Silhouette Score medio: {sil_kmeans:.4f}")

# FIGURA 17: Visualizzazione dei cluster K-Means su spazio t-SNE 2D
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_cluster_sample, x='tsne_1', y='tsne_2', hue='kmeans_cluster',
    palette='tab10', alpha=0.7, s=30, edgecolor=None
)
plt.title(f'Cluster K-Means (k={k_opt}) Proiettati nello Spazio 2D t-SNE', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Dimensione 1', fontsize=12)
plt.ylabel('t-SNE Dimensione 2', fontsize=12)
plt.legend(title='Cluster ID', fontsize=10)
plt.tight_layout()
plt.savefig('figures/fig17_kmeans_tsne_clusters.png', dpi=300)
plt.close()

# FIGURA 18: Profiling delle Feature Medie per Cluster (Creazione automatica "Mood Playlists")
# Calcoliamo la media delle feature audio continue per ciascun cluster e normalizziamo per confronto
cluster_means = df_cluster_sample.groupby('kmeans_cluster')[continuous_features].mean()
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-6)

plt.figure(figsize=(14, 7))
sns.heatmap(cluster_means_norm, annot=cluster_means.round(2), fmt='', cmap='coolwarm', cbar_kws={'label': 'Intensità Relativa (0-1)'})
plt.title('Profiling Acustico dei Cluster K-Means (Identificazione delle Mood Playlists)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Caratteristiche Sonore', fontsize=12)
plt.ylabel('Cluster K-Means', fontsize=12)
plt.tight_layout()
plt.savefig('figures/fig18_kmeans_feature_profiling.png', dpi=300)
plt.close()

# FIGURA 19: Grafico della Silhouette per K-Means (compattezza e separazione dei singoli cluster)
sil_values_km = silhouette_samples(X_cluster_sample, df_cluster_sample['kmeans_cluster'])
plt.figure(figsize=(10, 7))
y_lower = 10
for i in range(k_opt):
    ith_cluster_sil = sil_values_km[df_cluster_sample['kmeans_cluster'] == i]
    ith_cluster_sil.sort()
    size_cluster_i = ith_cluster_sil.shape[0]
    y_upper = y_lower + size_cluster_i
    color = plt.cm.tab10(float(i) / k_opt)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil, facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}", fontsize=10, fontweight='bold')
    y_lower = y_upper + 10

plt.axvline(x=sil_kmeans, color="red", linestyle="--", label=f"Silhouette Media = {sil_kmeans:.2f}")
plt.title('Grafico dei Coefficienti di Silhouette per Cluster (K-Means k = 5)', fontsize=14, fontweight='bold')
plt.xlabel('Coefficiente di Silhouette', fontsize=12)
plt.ylabel('Cluster', fontsize=12)
plt.yticks([])
plt.legend(loc='upper right', fontsize=11)
plt.tight_layout()
plt.savefig('figures/fig19_kmeans_silhouette_plot.png', dpi=300)
plt.close()

# 4.5 DBSCAN: Clustering Basato sulla Densità
print("\nApplicazione dell'algoritmo DBSCAN e ricerca del gomito per Epsilon...")

# Determiniamo analiticamente Epsilon tramite il metodo Nearest-Neighbors con k=min_samples
k_min_samples = 5
nbrs = NearestNeighbors(n_neighbors=k_min_samples).fit(X_cluster_sample)
distances, indices = nbrs.kneighbors(X_cluster_sample)
sorted_distances = np.sort(distances[:, k_min_samples-1])

# FIGURA 20: Grafico delle distanze k-NN per la scelta di Epsilon
plt.figure(figsize=(10, 6))
plt.plot(sorted_distances, color='#2b5c8f', linewidth=2)
plt.axhline(y=2.2, color='red', linestyle='--', label=r'Gomito Ottimale ($\epsilon = 2.2$)')
plt.title(f'Metodo Nearest-Neighbors per Scelta Parametro $\\epsilon$ (k={k_min_samples})', fontsize=14, fontweight='bold')
plt.xlabel('Campioni Ordinati per Distanza Crescente', fontsize=12)
plt.ylabel(f'Distanza dal {k_min_samples}° Vicino', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('figures/fig20_dbscan_knn_distance.png', dpi=300)
plt.close()

# FIGURA 21: Griglia di Tuning per DBSCAN (Epsilon vs min_samples)
eps_candidates = [1.8, 2.0, 2.2, 2.5]
min_samples_candidates = [4, 5, 6, 8]
dbscan_grid_results = []
for e in eps_candidates:
    row_n_clusters = []
    for ms in min_samples_candidates:
        db_temp = DBSCAN(eps=e, min_samples=ms).fit(X_cluster_sample)
        n_c = len(set(db_temp.labels_)) - (1 if -1 in db_temp.labels_ else 0)
        row_n_clusters.append(n_c)
    dbscan_grid_results.append(row_n_clusters)

plt.figure(figsize=(9, 6))
sns.heatmap(
    pd.DataFrame(dbscan_grid_results, index=eps_candidates, columns=min_samples_candidates),
    annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Numero di Cluster Identificati'}
)
plt.title(r'DBSCAN Tuning: Numero di Cluster al Variare di $\epsilon$ e min_samples', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('min_samples', fontsize=11)
plt.ylabel(r'$\epsilon$ (Raggio di Ricerca)', fontsize=11)
plt.tight_layout()
plt.savefig('figures/fig21_dbscan_grid_search.png', dpi=300)
plt.close()

# 4.6 Esecuzione DBSCAN con parametri ottimali (eps=2.2, min_samples=5)
db_opt = DBSCAN(eps=2.2, min_samples=5)
df_cluster_sample['dbscan_cluster'] = db_opt.fit_predict(X_cluster_sample)
n_db_clusters = len(set(df_cluster_sample['dbscan_cluster'])) - (1 if -1 in df_cluster_sample['dbscan_cluster'] else 0)
n_db_noise = (df_cluster_sample['dbscan_cluster'] == -1).sum()
print(f"DBSCAN Ottimale: {n_db_clusters} cluster densi, {n_db_noise} punti di rumore/outlier ({n_db_noise/sample_size*100:.1f}%)")

# FIGURA 22: Cluster DBSCAN su t-SNE con evidenziazione chiara del rumore (outlier in rosso)
plt.figure(figsize=(12, 8))
non_noise = df_cluster_sample[df_cluster_sample['dbscan_cluster'] != -1]
noise = df_cluster_sample[df_cluster_sample['dbscan_cluster'] == -1]

sns.scatterplot(
    data=non_noise, x='tsne_1', y='tsne_2', hue='dbscan_cluster',
    palette='tab10', alpha=0.7, s=30, edgecolor=None, legend='full'
)
plt.scatter(
    noise['tsne_1'], noise['tsne_2'], c='crimson', marker='x', s=45,
    label=f'Punti Rumorosi / Outlier (n={n_db_noise})', alpha=0.85
)
plt.title(r'Clusterizzazione DBSCAN ($\epsilon=2.2, min\_samples=5$) con Outlier Evidenziati', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Dimensione 1', fontsize=12)
plt.ylabel('t-SNE Dimensione 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('figures/fig22_dbscan_tsne_clusters_noise.png', dpi=300)
plt.close()

# FIGURA 23: Analisi diagnostica del rumore DBSCAN (Perché sono considerati outlier?)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
df_cluster_sample['is_noise'] = (df_cluster_sample['dbscan_cluster'] == -1).map({True: 'Rumore / Outlier', False: 'Punti di Cluster'})

sns.boxplot(data=df_cluster_sample, x='is_noise', y='duration_ms', ax=ax1, hue='is_noise', palette='Set1', legend=False)
ax1.set_title('Durata Brano (ms, scala logaritmica)', fontweight='bold')
ax1.set_yscale('log')
ax1.set_xlabel('')

sns.boxplot(data=df_cluster_sample, x='is_noise', y='speechiness', ax=ax2, hue='is_noise', palette='Set1', legend=False)
ax2.set_title('Speechiness (Presenza Parlato)', fontweight='bold')
ax2.set_xlabel('')

sns.boxplot(data=df_cluster_sample, x='is_noise', y='tempo', ax=ax3, hue='is_noise', palette='Set1', legend=False)
ax3.set_title('Tempo (BPM)', fontweight='bold')
ax3.set_xlabel('')
plt.suptitle('Analisi Diagnostica del Rumore DBSCAN: Caratteristiche Acustiche degli Outlier', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig23_dbscan_noise_profiling.png', dpi=300)
plt.close()

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
    'kmeans_k': k_opt,
    'kmeans_sil': round(sil_kmeans, 4),
    'dbscan_eps': 2.2,
    'dbscan_min_samples': 5,
    'dbscan_clusters': n_db_clusters,
    'dbscan_noise': int(n_db_noise),
    'dbscan_noise_pct': round(n_db_noise/sample_size*100, 2)
}

with open('data/project_summary.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("\n>>> PIPELINE COMPLETATA CON SUCCESSO! Tutte le 23 figure e le metriche sono state generate e salvate.")
