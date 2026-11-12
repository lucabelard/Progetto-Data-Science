"""Regenerate the high-impact clustering plots used in Chapter 4.

Run from any directory with:
    python scripts/regenerate_clustering_profiles.py

This script produces the updated, high-aesthetic figures for Chapter 4:
- fig17_kmeans_tsne_clusters.png (PCA + t-SNE)
- fig18_kmeans_feature_profiling.png (Heatmap)
- fig19_kmeans_feature_radar.png (Radar Chart)
- fig23_dbscan_noise_profiling.png (Raincloud Plot: Half-Violin + Boxplot + Jitter)
- fig24_gmm_soft_membership.png (Mappa Topografica GMM con Ellissi di Covarianza e Corridoi Ibridi)
- fig24b_mood_pentagon_radviz.png (Pentagono Baricentrico dei Mood / RadViz Simplex)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "genres_cleaned.csv"
FIGURE_DIRS = [ROOT / "figures", ROOT / "Relazione" / "capitolo4" / "img"]

CLUSTER_FEATURES = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "valence", "tempo", "duration_ms",
]
PROFILE_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]
SAMPLE_SIZE = 6000
N_CLUSTERS = 5

KMEANS_NAMES = {
    0: "C0 · Workout / Bass",
    1: "C1 · Club / Strumentale",
    2: "C2 · Acustico / Chill",
    3: "C3 · Groove / Pop-Urban",
    4: "C4 · Rap / High-BPM",
}
GMM_NAMES = {
    0: "GMM 0 · Energetico / strumentale",
    1: "GMM 1 · Acustico / moderato",
    2: "GMM 2 · Energetico / vocale",
    3: "GMM 3 · Dance / strumentale",
    4: "GMM 4 · Urban / vocale",
}

GMM_DISPLAY_LABELS = {
    0: "C0 · Energetico / Strumentale (Rock/Metal)",
    1: "C1 · Acustico / Intimo (Folk/Classica)",
    2: "C2 · Energetico / Vocale (Pop/Dance-Pop)",
    3: "C3 · Dance / Elettronico (EDM/Techno)",
    4: "C4 · Urban / Ritmico (Rap/Hip-Hop)"
}

PALETTE_5 = ["#2b5c8f", "#27ae60", "#e67e22", "#8e44ad", "#c0392b"]


def save_to_project(image: plt.Figure, filename: str, *, tight: bool = True) -> None:
    for directory in FIGURE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        image.savefig(directory / filename, dpi=300, bbox_inches="tight" if tight else None)
    plt.close(image)


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    print("Caricamento dataset e preparazione trasformazioni...", flush=True)
    df = pd.read_csv(DATA_PATH)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])
    pca = PCA(n_components=4, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(df), size=SAMPLE_SIZE, replace=False)
    sample = df.iloc[sample_indices].copy().reset_index(drop=True)
    X_sample_pca = X_pca[sample_indices]
    X_sample_scaled = X_scaled[sample_indices]
    sample["pc_1"], sample["pc_2"] = X_sample_pca[:, 0], X_sample_pca[:, 1]
    sample["pc_3"], sample["pc_4"] = X_sample_pca[:, 2], X_sample_pca[:, 3]

    print("Calcolo t-SNE 2D...", flush=True)
    X_tsne = TSNE(
        n_components=2, perplexity=40, random_state=42, max_iter=1000
    ).fit_transform(X_sample_scaled)
    sample["tsne_1"], sample["tsne_2"] = X_tsne[:, 0], X_tsne[:, 1]

    # -------------------------------------------------------------------------
    # K-MEANS
    # -------------------------------------------------------------------------
    print("Addestramento K-Means...", flush=True)
    kmeans_palette = {KMEANS_NAMES[i]: plt.get_cmap("tab10")(i) for i in range(N_CLUSTERS)}
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    sample["kmeans_cluster"] = km.fit_predict(X_sample_pca)
    sample["kmeans_profile"] = sample["kmeans_cluster"].map(KMEANS_NAMES)
    cluster_order = [KMEANS_NAMES[i] for i in range(N_CLUSTERS)]

    # Figura 17: PCA e t-SNE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.2))
    for ax, x, y, title in [
        (ax1, "pc_1", "pc_2", "Proiezione PCA 2D con centroidi"),
        (ax2, "tsne_1", "tsne_2", "Proiezione t-SNE 2D"),
    ]:
        sns.scatterplot(
            data=sample, x=x, y=y, hue="kmeans_profile", hue_order=cluster_order,
            palette=kmeans_palette, alpha=0.65, s=25, edgecolor=None, legend=False, ax=ax,
        )
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% varianza)"
            if x == "pc_1" else "t-SNE Dimensione 1", fontsize=13
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% varianza)"
            if y == "pc_2" else "t-SNE Dimensione 2", fontsize=13
        )
    ax1.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker="X", s=220,
        c="black", edgecolor="white", linewidth=1.5, zorder=10,
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor=kmeans_palette[name], markeredgecolor="none", label=name)
        for name in cluster_order
    ]
    handles.append(Line2D([0], [0], marker="X", linestyle="", markersize=9,
                          markerfacecolor="black", markeredgecolor="white", label="Centroidi K-Means"))
    fig.legend(
        handles=handles, loc="lower center", ncol=3,
        title="Profili sonori (ID K-Means con descrizione sintetica)",
        fontsize=11, title_fontsize=12, bbox_to_anchor=(0.5, 0.005), frameon=True,
    )
    fig.suptitle("Partizionamento K-Means (k=5)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.13, 1, 0.94])
    save_to_project(fig, "fig17_kmeans_tsne_clusters.png")

    # Figura 18: Heatmap dei profili
    profile_means = sample.groupby("kmeans_cluster")[PROFILE_FEATURES].mean()
    normalized = (profile_means - profile_means.min()) / (
        profile_means.max() - profile_means.min() + 1e-6
    )
    annotations = profile_means.copy().astype(object)
    for feature in PROFILE_FEATURES:
        for cluster in profile_means.index:
            value = profile_means.loc[cluster, feature]
            if feature == "duration_ms":
                annotations.loc[cluster, feature] = f"{value / 1000:.0f}k"
            elif feature == "tempo":
                annotations.loc[cluster, feature] = f"{value:.1f}"
            else:
                annotations.loc[cluster, feature] = f"{value:.2f}"

    fig, ax = plt.subplots(figsize=(13, 8))
    sns.heatmap(
        normalized, annot=annotations, fmt="", cmap="vlag", center=0.5,
        annot_kws={"size": 13, "weight": "bold"},
        yticklabels=[KMEANS_NAMES[i] for i in profile_means.index],
        cbar_kws={"label": "Posizione relativa della media (Min-Max per feature)"}, ax=ax,
    )
    ax.set_title("Medie delle feature per profilo K-Means", fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("Feature acustiche (annotazioni: medie originali)", fontsize=13)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25, labelsize=13)
    ax.tick_params(axis="y", rotation=0, labelsize=13)
    ax.collections[0].colorbar.set_label("Posizione relativa della media (Min-Max per feature)", fontsize=12)
    ax.collections[0].colorbar.ax.tick_params(labelsize=11)
    fig.tight_layout()
    save_to_project(fig, "fig18_kmeans_feature_profiling.png")

    # Figura 19: Radar dei profili
    radar_features = CLUSTER_FEATURES
    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={"projection": "polar"})
    for cluster in range(N_CLUSTERS):
        values = normalized.loc[cluster, radar_features].tolist()
        values += values[:1]
        color = plt.get_cmap("tab10")(cluster)
        ax.plot(angles, values, linewidth=2.3, label=KMEANS_NAMES[cluster], color=color)
        ax.fill(angles, values, color=color, alpha=0.12)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_features, fontsize=13, fontweight="bold")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=10, color="gray")
    ax.set_title("Radar dei profili acustici K-Means", fontsize=17, fontweight="bold", pad=27)
    ax.legend(loc="upper left", bbox_to_anchor=(1.10, 1.08), fontsize=11, title="Profilo sonoro", title_fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.78, top=0.88, bottom=0.06)
    save_to_project(fig, "fig19_kmeans_feature_radar.png")

    # -------------------------------------------------------------------------
    # DBSCAN & RAINCLOUD PLOT (FIGURA 23)
    # -------------------------------------------------------------------------
    print("Addestramento DBSCAN e generazione Raincloud Plot (Figura 23)...", flush=True)
    db = DBSCAN(eps=0.60, min_samples=5).fit(X_sample_pca)
    sample["dbscan_cluster"] = db.labels_
    sample["dbscan_group"] = sample["dbscan_cluster"].map({-1: "Rumore / Outlier DBSCAN", 0: "Cluster Core (Densità Ottimale)"}).fillna("Micro-Cluster")
    sample["duration_min"] = sample["duration_ms"] / 60000.0

    features_raincloud = [
        ("duration_min", "Durata della Traccia (minuti)", [0, 8]),
        ("speechiness", "Speechiness (Contenuto Parlato)", [0, 0.8]),
        ("acousticness", "Acousticness (Timbrica Naturale vs Sintetica)", [0, 1.0])
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    rain_colors = {"Cluster Core (Densità Ottimale)": "#2980b9", "Rumore / Outlier DBSCAN": "#e74c3c"}

    for ax, (feat, feat_label, x_limits) in zip(axes, features_raincloud):
        for group_idx, (group_name, color) in enumerate(rain_colors.items()):
            sub_vals = sample[sample["dbscan_group"] == group_name][feat].dropna().values
            y_base = group_idx * 1.5

            # 1. Half-KDE (Cloud)
            kde = stats.gaussian_kde(sub_vals)
            x_eval = np.linspace(x_limits[0], x_limits[1], 200)
            kde_vals = kde(x_eval)
            kde_scaled = (kde_vals / kde_vals.max()) * 0.55
            ax.fill_between(x_eval, y_base, y_base + kde_scaled, color=color, alpha=0.55, edgecolor=color, lw=1.5)

            # 2. Boxplot minimale
            q25, q50, q75 = np.percentile(sub_vals, [25, 50, 75])
            ax.plot([q25, q75], [y_base - 0.12, y_base - 0.12], color=color, lw=4, solid_capstyle="round")
            ax.scatter([q50], [y_base - 0.12], color="white", edgecolor="#2c3e50", s=45, zorder=4)

            # 3. Jittered scatter (Rain)
            sample_pts = np.random.choice(sub_vals, size=min(400, len(sub_vals)), replace=False)
            jitter = np.random.uniform(-0.18, -0.04, size=len(sample_pts))
            ax.scatter(sample_pts, y_base - 0.2 + jitter, color=color, alpha=0.25, s=12, edgecolors="none")

        ax.set_ylim(-0.5, 2.5)
        ax.set_xlim(x_limits)
        ax.set_yticks([0.3, 1.8])
        ax.set_yticklabels(["Cluster Core", "Rumore DBSCAN"], fontsize=11, fontweight="bold")
        ax.set_xlabel(feat_label, fontsize=11.5)
        ax.set_title(f"Distribuzione di {feat.replace('_min', ' (min)').capitalize()}", fontsize=13, fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    fig.suptitle("Raincloud Plot: Diagnosi Acustica della Separazione tra Core e Rumore DBSCAN\n(Half-Violin KDE + Boxplot + Jittered Data Cloud)", fontsize=14.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_to_project(fig, "fig23_dbscan_noise_profiling.png")

    # -------------------------------------------------------------------------
    # GMM: MAPPA TOPOGRAFICA AD ELLISSI E CORRIDOI IBRIDI (FIGURA 24)
    # -------------------------------------------------------------------------
    print("Addestramento GMM e generazione Mappa Topografica (Figura 24)...", flush=True)
    gmm = GaussianMixture(
        n_components=N_CLUSTERS, covariance_type="full", reg_covar=1e-5,
        n_init=10, random_state=42,
    ).fit(X_sample_pca)

    memberships = gmm.predict_proba(X_sample_pca)
    gmm_sorted = np.sort(memberships, axis=1)
    gmm_top1 = gmm_sorted[:, -1]
    gmm_top2_gap = gmm_sorted[:, -1] - gmm_sorted[:, -2]
    gmm_labels = memberships.argmax(axis=1)

    for i in range(N_CLUSTERS):
        sample[f"gmm_affinity_{i}"] = memberships[:, i]
    sample["gmm_component"] = gmm_labels
    sample["gmm_top_affinity"] = gmm_top1
    sample["gmm_top2_gap"] = gmm_top2_gap
    sample["gmm_cross_mood_candidate"] = gmm_top2_gap <= 0.15

    fig, (ax_main, ax_side) = plt.subplots(1, 2, figsize=(18, 8.5), gridspec_kw={"width_ratios": [2.2, 1]})

    # Background KDE isolinee
    sns.kdeplot(
        data=sample, x="pc_1", y="pc_2",
        levels=10, cmap="Blues", alpha=0.3, fill=True, ax=ax_main, thresh=0.03
    )

    # Scatter brani standard
    for i in range(N_CLUSTERS):
        sub = sample[(sample["gmm_component"] == i) & (~sample["gmm_cross_mood_candidate"])]
        ax_main.scatter(
            sub["pc_1"], sub["pc_2"],
            c=PALETTE_5[i], alpha=0.30, s=20, edgecolors="none",
            label=GMM_DISPLAY_LABELS[i]
        )

    # Ellissi di covarianza GMM (1 sigma e 2 sigma)
    for i in range(N_CLUSTERS):
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
            edgecolor=PALETTE_5[i], facecolor="none", linewidth=2.5, linestyle="-"
        )
        ax_main.add_patch(ell_1)

        # 2 sigma (95% confidenza)
        w_2sig, h_2sig = 2 * np.sqrt(vals * 5.991)
        ell_2 = patches.Ellipse(
            xy=mean_2d, width=w_2sig, height=h_2sig, angle=theta,
            edgecolor=PALETTE_5[i], facecolor="none", linewidth=1.5, linestyle="--", alpha=0.7
        )
        ax_main.add_patch(ell_2)

        # Centroid marker
        ax_main.scatter(mean_2d[0], mean_2d[1], c=PALETTE_5[i], s=180, marker="X", edgecolors="white", linewidths=2, zorder=6)
        ax_main.text(mean_2d[0]+0.15, mean_2d[1]+0.15, f"C{i}", fontsize=12, fontweight="bold",
                     color="#111111", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor=PALETTE_5[i]))

    # Brani Cross-Mood evidenziati (Gap <= 0.15)
    cross_tracks = sample[sample["gmm_cross_mood_candidate"]]
    ax_main.scatter(
        cross_tracks["pc_1"], cross_tracks["pc_2"],
        c="#f1c40f", s=60, marker="D", edgecolors="#2c3e50", linewidths=1.2,
        zorder=5, label=f"Brani Ponte Cross-Mood (N={len(cross_tracks)})"
    )

    # Annotazione
    sample_bridge = cross_tracks.iloc[15]
    ax_main.annotate(
        "Brano Ponte Ibrido\n(Affinità bilanciata tra 2 Archetipi)",
        xy=(sample_bridge["pc_1"], sample_bridge["pc_2"]),
        xytext=(sample_bridge["pc_1"] + 1.2, sample_bridge["pc_2"] - 1.4),
        arrowprops=dict(facecolor="#d35400", shrink=0.08, width=1.5, headwidth=8),
        fontsize=10.5, fontweight="bold", color="#a04000",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9e7", edgecolor="#f39c12", alpha=0.95)
    )

    ax_main.set_title("Mappa Topografica GMM: Ellissi di Covarianza (1σ e 2σ) e Corridoi Ibridi", fontsize=14, fontweight="bold")
    ax_main.set_xlabel(f"Prima Componente Principale PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Varianza)", fontsize=11.5)
    ax_main.set_ylabel(f"Seconda Componente Principale PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Varianza)", fontsize=11.5)
    ax_main.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax_main.grid(True, linestyle=":", alpha=0.4)

    # Classifica corridoi ibridi
    affinity_cols = [f"gmm_affinity_{i}" for i in range(5)]
    bridge_pairs = []
    for _, row in cross_tracks.iterrows():
        affs = row[affinity_cols].to_numpy()
        top2 = np.argsort(affs)[-2:]
        pair = tuple(sorted([top2[0], top2[1]]))
        bridge_pairs.append(pair)

    pair_counts = pd.Series(bridge_pairs).value_counts().head(8)
    pair_labels = [f"C{p[0]} ↔ C{p[1]}\n({GMM_DISPLAY_LABELS[p[0]].split('·')[1].split('(')[0].strip()[:10]} / {GMM_DISPLAY_LABELS[p[1]].split('·')[1].split('(')[0].strip()[:10]})" for p in pair_counts.index]

    ax_side.barh(range(len(pair_counts)), pair_counts.values, color="#e67e22", edgecolor="#2c3e50", alpha=0.85)
    ax_side.set_yticks(range(len(pair_counts)))
    ax_side.set_yticklabels(pair_labels, fontsize=9.5)
    ax_side.invert_yaxis()
    ax_side.set_xlabel("Numero di Tracce Ponte Condivise", fontsize=11)
    ax_side.set_title("Corridoi Ibridi più Frequenti\n(Opportunità di Playlist Cross-Mood)", fontsize=13, fontweight="bold")
    for i, v in enumerate(pair_counts.values):
        ax_side.text(v + 1.5, i, f"{v} brani", va="center", fontsize=9.5, fontweight="bold", color="#7e3805")
    ax_side.grid(axis="x", linestyle=":", alpha=0.6)

    fig.tight_layout()
    save_to_project(fig, "fig24_gmm_soft_membership.png")

    # -------------------------------------------------------------------------
    # GMM: PENTAGONO BARICENTRICO RADVIZ (FIGURA 24b)
    # -------------------------------------------------------------------------
    print("Generazione Pentagono Baricentrico dei Mood (Figura 24b)...", flush=True)
    fig, ax = plt.subplots(figsize=(11, 10))

    angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, 6)[:-1]
    R = 1.0
    vertices = np.column_stack([R * np.cos(angles), R * np.sin(angles)])

    theta_circle = np.linspace(0, 2*np.pi, 200)
    for r_guide in [0.25, 0.50, 0.75, 1.0]:
        ax.plot(r_guide * np.cos(theta_circle), r_guide * np.sin(theta_circle), color="gray", linestyle=":", alpha=0.45)
        ax.text(0.02, r_guide, f"{int(r_guide*100)}%", fontsize=8.5, color="gray", alpha=0.8)

    pent_x = np.append(vertices[:, 0], vertices[0, 0])
    pent_y = np.append(vertices[:, 1], vertices[0, 1])
    ax.plot(pent_x, pent_y, color="#2c3e50", linewidth=2, linestyle="-", alpha=0.7)

    for i in range(5):
        ax.plot([0, vertices[i, 0]], [0, vertices[i, 1]], color="#7f8c8d", linestyle="--", alpha=0.5)

    P = np.zeros((len(sample), 2))
    for i in range(5):
        aff = sample[f"gmm_affinity_{i}"].to_numpy()[:, np.newaxis]
        P += aff * vertices[i]

    sample["radviz_x"] = P[:, 0]
    sample["radviz_y"] = P[:, 1]

    for i in range(5):
        sub = sample[(sample["gmm_component"] == i) & (~sample["gmm_cross_mood_candidate"])]
        ax.scatter(
            sub["radviz_x"], sub["radviz_y"],
            c=PALETTE_5[i], alpha=0.35, s=22, edgecolors="none",
            label=GMM_DISPLAY_LABELS[i]
        )

    cross = sample[sample["gmm_cross_mood_candidate"]]
    ax.scatter(
        cross["radviz_x"], cross["radviz_y"],
        c="#f1c40f", s=55, marker="D", edgecolors="#2c3e50", linewidths=1.2,
        zorder=5, label="Tracce Ponte Cross-Mood (Equidistanti)"
    )

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
        ax.scatter(vx, vy, c=PALETTE_5[i], s=240, edgecolors="#2c3e50", linewidths=2.5, zorder=6)
        align_h = "center"
        if vx > 0.2: align_h = "left"
        elif vx < -0.2: align_h = "right"

        align_v = "center"
        if vy > 0.3: align_v = "bottom"
        elif vy < -0.3: align_v = "top"

        ax.text(
            vx * offset_scale, vy * offset_scale, labels[i],
            ha=align_h, va=align_v, fontsize=10.5, fontweight="bold",
            color="#1a252f", bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=PALETTE_5[i], alpha=0.92, lw=1.5)
        )

    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.45, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Pentagono Baricentrico dei Mood Musicali (RadViz Simplex)\nVisualizzazione dello Spazio delle Probabilità Soft GMM", fontsize=15, fontweight="bold", pad=20)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=9.5, framealpha=0.95)

    fig.tight_layout()
    save_to_project(fig, "fig24b_mood_pentagon_radviz.png")

    print("Tutte le figure del capitolo 4 sono state rigenerate con successo in figures/ e Relazione/capitolo4/img/!")


if __name__ == "__main__":
    main()
