import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy import stats

# Set styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('figures/previews', exist_ok=True)

# 1. Caricamento dati
print("Caricamento dataset...", flush=True)
df_clean = pd.read_csv('data/genres_cleaned.csv')
df_sample = pd.read_csv('data/clustering_soft_memberships_sample.csv')

clustering_features = [
    'danceability', 'energy', 'speechiness', 'acousticness',
    'instrumentalness', 'valence', 'tempo', 'duration_ms'
]

# Standardizzazione e PCA coerente con build_full_project.py
scaler = StandardScaler()
X_clust_scaled = scaler.fit_transform(df_clean[clustering_features])
pca = PCA(n_components=4, random_state=42)
X_pca_full = pca.fit_transform(X_clust_scaled)

np.random.seed(42)
sample_indices = np.random.choice(len(df_clean), size=6000, replace=False)
X_pca_sample = X_pca_full[sample_indices]

# Merge dataframe
df = df_clean.iloc[sample_indices].copy().reset_index(drop=True)
for col in df_sample.columns:
    if col not in df.columns:
        df[col] = df_sample[col].values

df['pc_1'] = X_pca_sample[:, 0]
df['pc_2'] = X_pca_sample[:, 1]
df['pc_3'] = X_pca_sample[:, 2]
df['pc_4'] = X_pca_sample[:, 3]

# Fit GMM sui 4 PC per recuperare medie e matrici di covarianza
gmm = GaussianMixture(n_components=5, covariance_type='full', reg_covar=1e-5, n_init=10, random_state=42)
gmm.fit(X_pca_sample)

# Nomi e colori degli archetipi
archetype_names = {
    0: 'C0 · Energetico / Strumentale (Rock/Metal)',
    1: 'C1 · Acustico / Intimo (Folk/Classica)',
    2: 'C2 · Energetico / Vocale (Pop/Dance-Pop)',
    3: 'C3 · Dance / Elettronico (EDM/Techno)',
    4: 'C4 · Urban / Ritmico (Rap/Hip-Hop)'
}
df['archetype_label'] = df['gmm_component'].map(archetype_names)

palette = ['#2b5c8f', '#27ae60', '#e67e22', '#8e44ad', '#c0392b']
color_dict = {archetype_names[i]: palette[i] for i in range(5)}

print("Generazione Anteprima 1: Mappa Topografica ed Ellissi GMM con Ponti Cross-Mood...", flush=True)
# -------------------------------------------------------------------------
# FIGURA ANTEPRIMA 1: Mappa Topografica ed Ellissi GMM
# -------------------------------------------------------------------------
fig, (ax_main, ax_side) = plt.subplots(1, 2, figsize=(18, 8.5), gridspec_kw={'width_ratios': [2.2, 1]})

# Background KDE isolinee
sns.kdeplot(
    data=df, x='pc_1', y='pc_2',
    levels=10, cmap='Blues', alpha=0.3, fill=True, ax=ax_main, thresh=0.03
)

# Scatter standard tracks (soft colors)
for i in range(5):
    sub = df[(df['gmm_component'] == i) & (~df['gmm_cross_mood_candidate'])]
    ax_main.scatter(
        sub['pc_1'], sub['pc_2'],
        c=palette[i], alpha=0.30, s=20, edgecolors='none',
        label=archetype_names[i]
    )

# Disegna ellissi di covarianza GMM (1 sigma e 2 sigma)
for i in range(5):
    mean_2d = gmm.means_[i, :2]
    cov_2d = gmm.covariances_[i][:2, :2]
    vals, vecs = np.linalg.eigh(cov_2d)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
    # 1 sigma (68% confidenza)
    w_1sig, h_1sig = 2 * np.sqrt(vals)
    ell_1 = patches.Ellipse(
        xy=mean_2d, width=w_1sig, height=h_1sig, angle=theta,
        edgecolor=palette[i], facecolor='none', linewidth=2.5, linestyle='-'
    )
    ax_main.add_patch(ell_1)
    
    # 2 sigma (95% confidenza)
    w_2sig, h_2sig = 2 * np.sqrt(vals * 5.991)
    ell_2 = patches.Ellipse(
        xy=mean_2d, width=w_2sig, height=h_2sig, angle=theta,
        edgecolor=palette[i], facecolor='none', linewidth=1.5, linestyle='--', alpha=0.7
    )
    ax_main.add_patch(ell_2)
    
    # Centroid marker
    ax_main.scatter(mean_2d[0], mean_2d[1], c=palette[i], s=180, marker='X', edgecolors='white', linewidths=2, zorder=6)
    ax_main.text(mean_2d[0]+0.15, mean_2d[1]+0.15, f'C{i}', fontsize=12, fontweight='bold',
                 color='#111111', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=palette[i]))

# Brani Cross-Mood evidenziati (Gap <= 0.15)
cross_tracks = df[df['gmm_cross_mood_candidate']]
ax_main.scatter(
    cross_tracks['pc_1'], cross_tracks['pc_2'],
    c='#f1c40f', s=60, marker='D', edgecolors='#2c3e50', linewidths=1.2,
    zorder=5, label=f'Brani Ponte Cross-Mood (N={len(cross_tracks)})'
)

# Esempio annotazione ponte
sample_bridge = cross_tracks.iloc[15]
ax_main.annotate(
    f"Brano Ponte Ibrido\n(Affinità bilanciata tra 2 Archetipi)",
    xy=(sample_bridge['pc_1'], sample_bridge['pc_2']),
    xytext=(sample_bridge['pc_1'] + 1.2, sample_bridge['pc_2'] - 1.4),
    arrowprops=dict(facecolor='#d35400', shrink=0.08, width=1.5, headwidth=8),
    fontsize=10.5, fontweight='bold', color='#a04000',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor='#f39c12', alpha=0.95)
)

ax_main.set_title("Mappa Topografica GMM: Ellissi di Covarianza (1σ e 2σ) e Corridoi Ibridi", fontsize=14, fontweight='bold')
ax_main.set_xlabel("Prima Componente Principale PC1 (29.8% Varianza)", fontsize=11.5)
ax_main.set_ylabel("Seconda Componente Principale PC2 (18.5% Varianza)", fontsize=11.5)
ax_main.legend(loc='upper right', fontsize=9.5, framealpha=0.9)
ax_main.grid(True, linestyle=':', alpha=0.4)

# Pannello Laterale: Matrice delle Transizioni / Coppie di Archetipi nei brani ponte
# Calcoliamo per i brani ponte quali sono i due archetipi con massima affinità
affinity_cols = [f'gmm_affinity_{i}' for i in range(5)]
bridge_pairs = []
for _, row in cross_tracks.iterrows():
    affs = row[affinity_cols].to_numpy()
    top2 = np.argsort(affs)[-2:]
    pair = tuple(sorted([top2[0], top2[1]]))
    bridge_pairs.append(pair)

pair_counts = pd.Series(bridge_pairs).value_counts().head(8)
pair_labels = [f"C{p[0]} ↔ C{p[1]}\n({archetype_names[p[0]].split('·')[1].split('(')[0].strip()[:10]} / {archetype_names[p[1]].split('·')[1].split('(')[0].strip()[:10]})" for p in pair_counts.index]

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
plt.savefig('figures/previews/preview1_gmm_topographic_map.png', dpi=300)
plt.close()

print("Generazione Anteprima 2: Pentagono Baricentrico dei Mood (RadViz Simplex)...", flush=True)
# -------------------------------------------------------------------------
# FIGURA ANTEPRIMA 2: Pentagono Baricentrico dei Mood (RadViz Simplex)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 10))

# 5 vertici disposti sul cerchio unitario
angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, 6)[:-1]
R = 1.0
vertices = np.column_stack([R * np.cos(angles), R * np.sin(angles)])

# Ragnatela pentagonale e cerchi guida
theta_circle = np.linspace(0, 2*np.pi, 200)
for r_guide in [0.25, 0.50, 0.75, 1.0]:
    ax.plot(r_guide * np.cos(theta_circle), r_guide * np.sin(theta_circle), color='gray', linestyle=':', alpha=0.45)
    ax.text(0.02, r_guide, f"{int(r_guide*100)}%", fontsize=8.5, color='gray', alpha=0.8)

# Disegna perimetro del pentagono
pent_x = np.append(vertices[:, 0], vertices[0, 0])
pent_y = np.append(vertices[:, 1], vertices[0, 1])
ax.plot(pent_x, pent_y, color='#2c3e50', linewidth=2, linestyle='-', alpha=0.7)

# Disegna raggi verso il centro
for i in range(5):
    ax.plot([0, vertices[i, 0]], [0, vertices[i, 1]], color='#7f8c8d', linestyle='--', alpha=0.5)

# Calcolo coordinate baricentriche per ogni canzone
P = np.zeros((len(df), 2))
for i in range(5):
    aff = df[f'gmm_affinity_{i}'].to_numpy()[:, np.newaxis]
    P += aff * vertices[i]

df['radviz_x'] = P[:, 0]
df['radviz_y'] = P[:, 1]

# Scatter tracce normali
for i in range(5):
    sub = df[(df['gmm_component'] == i) & (~df['gmm_cross_mood_candidate'])]
    ax.scatter(
        sub['radviz_x'], sub['radviz_y'],
        c=palette[i], alpha=0.35, s=22, edgecolors='none',
        label=archetype_names[i]
    )

# Scatter tracce ponte cross-mood (in centro o lungo i bordi intermedi)
cross = df[df['gmm_cross_mood_candidate']]
ax.scatter(
    cross['radviz_x'], cross['radviz_y'],
    c='#f1c40f', s=55, marker='D', edgecolors='#2c3e50', linewidths=1.2,
    zorder=5, label='Tracce Ponte Cross-Mood (Equidistanti)'
)

# Etichette sui 5 vertici
labels = [
    "C0: Energetico / Strumentale\n(Rock / Metal)",
    "C1: Acustico / Intimo\n(Folk / Classica)",
    "C2: Energetico / Vocale\n(Pop / Dance-Pop)",
    "C3: Dance / Elettronico\n(EDM / Techno)",
    "C4: Urban / Ritmico\n(Rap / Hip-Hop)"
]

offset_scale = 1.18
for i in range(5):
    vx, vy = vertices[i]
    ax.scatter(vx, vy, c=palette[i], s=240, edgecolors='#2c3e50', linewidths=2.5, zorder=6)
    align_h = 'center'
    if vx > 0.2: align_h = 'left'
    elif vx < -0.2: align_h = 'right'
    
    align_v = 'center'
    if vy > 0.3: align_v = 'bottom'
    elif vy < -0.3: align_v = 'top'
    
    ax.text(
        vx * offset_scale, vy * offset_scale, labels[i],
        ha=align_h, va=align_v, fontsize=10.5, fontweight='bold',
        color='#1a252f', bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=palette[i], alpha=0.92, lw=1.5)
    )

ax.set_xlim(-1.55, 1.55)
ax.set_ylim(-1.45, 1.55)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Pentagono Baricentrico dei Mood Musicali (RadViz Simplex)\nVisualizzazione dello Spazio delle Probabilità Soft GMM", fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=9.5, framealpha=0.95)

plt.tight_layout()
plt.savefig('figures/previews/preview2_mood_pentagon_radviz.png', dpi=300)
plt.close()

print("Generazione Anteprima 3: Raincloud Plot per Diagnostica Noise DBSCAN...", flush=True)
# -------------------------------------------------------------------------
# FIGURA ANTEPRIMA 3: Raincloud Plot per Diagnostica Noise DBSCAN
# -------------------------------------------------------------------------
# Confronto DBSCAN Core (0) vs Noise (-1) su 3 feature: Duration (min), Speechiness, Acousticness
df['dbscan_group'] = df['dbscan_cluster'].map({0: 'Cluster Core (Densità Ottimale)', -1: 'Rumore / Outlier DBSCAN'})
df['duration_min'] = df['duration_ms'] / 60000.0

features_raincloud = [
    ('duration_min', 'Durata della Traccia (minuti)', [0, 8]),
    ('speechiness', 'Speechiness (Contenuto Parlato)', [0, 0.8]),
    ('acousticness', 'Acousticness (Timbrica Naturale vs Sintetica)', [0, 1.0])
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
rain_colors = {'Cluster Core (Densità Ottimale)': '#2980b9', 'Rumore / Outlier DBSCAN': '#e74c3c'}

for ax, (feat, feat_label, x_limits) in zip(axes, features_raincloud):
    for group_idx, (group_name, color) in enumerate(rain_colors.items()):
        sub_vals = df[df['dbscan_group'] == group_name][feat].dropna().values
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
        # Sottocampionamento per nitidezza visiva
        sample_pts = np.random.choice(sub_vals, size=min(400, len(sub_vals)), replace=False)
        jitter = np.random.uniform(-0.18, -0.04, size=len(sample_pts))
        ax.scatter(sample_pts, y_base - 0.2 + jitter, color=color, alpha=0.25, s=12, edgecolors='none')
    
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(x_limits)
    ax.set_yticks([0.3, 1.8])
    ax.set_yticklabels(['Cluster Core', 'Rumore DBSCAN'], fontsize=11, fontweight='bold')
    ax.set_xlabel(feat_label, fontsize=11.5)
    ax.set_title(f"Distribuzione di {feat.capitalize()}", fontsize=13, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.5)

plt.suptitle("Raincloud Plot: Diagnosi Acustica della Separazione tra Core e Rumore DBSCAN\n(Half-Violin KDE + Boxplot + Jittered Data Cloud)", fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('figures/previews/preview3_dbscan_noise_raincloud.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generazione Anteprima 4: JointGrid Bivariata PCA con Marginali KDE...", flush=True)
# -------------------------------------------------------------------------
# FIGURA ANTEPRIMA 4: JointGrid Bivariata PCA con Marginali KDE
# -------------------------------------------------------------------------
g = sns.JointGrid(data=df, x='pc_1', y='pc_2', hue='archetype_label', palette=palette, height=9, ratio=4)
g.plot_joint(sns.scatterplot, alpha=0.45, s=22, edgecolor='none')
g.plot_joint(sns.kdeplot, levels=5, alpha=0.6, linewidths=1.2)
g.plot_marginals(sns.kdeplot, fill=True, common_norm=False, alpha=0.35, linewidth=1.5)

g.ax_joint.set_xlabel("PC1: Asse Energia / Acusticità (29.8% Varianza Spiegata)", fontsize=12)
g.ax_joint.set_ylabel("PC2: Asse Vocale / Strumentale (18.5% Varianza Spiegata)", fontsize=12)
g.ax_joint.grid(True, linestyle=':', alpha=0.4)

# Legenda pulita
g.ax_joint.legend(title="Archetipo Musicale", loc='upper right', fontsize=9.5, title_fontsize=10.5, framealpha=0.9)
plt.suptitle("JointGrid PCA: Paesaggio Continuo e Densità Marginali degli Archetipi Musicali", fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('figures/previews/preview4_pca_jointgrid_density.png', dpi=300, bbox_inches='tight')
plt.close()

print("Tutte e 4 le anteprime sono state create con successo in figures/previews/!", flush=True)
