"""Regenerate the K-Means and GMM plots used in Chapter 4.

Run from any directory with:
    python scripts/regenerate_clustering_profiles.py

This focused entry point reproduces the clustering figures without rerunning
the supervised classification pipeline.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
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


def save_to_project(image: plt.Figure, filename: str, *, tight: bool = True) -> None:
    for directory in FIGURE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        image.savefig(directory / filename, dpi=300, bbox_inches="tight" if tight else None)
    plt.close(image)


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
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
    X_tsne = TSNE(
        n_components=2, perplexity=40, random_state=42, max_iter=1000
    ).fit_transform(X_sample_scaled)
    sample["tsne_1"], sample["tsne_2"] = X_tsne[:, 0], X_tsne[:, 1]

    palette = {KMEANS_NAMES[i]: plt.get_cmap("tab10")(i) for i in range(N_CLUSTERS)}
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    sample["kmeans_cluster"] = km.fit_predict(X_sample_pca)
    sample["kmeans_profile"] = sample["kmeans_cluster"].map(KMEANS_NAMES)
    cluster_order = [KMEANS_NAMES[i] for i in range(N_CLUSTERS)]

    # Figure 17: PCA and t-SNE projections with a descriptive, shared legend.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.2))
    for ax, x, y, title in [
        (ax1, "pc_1", "pc_2", "Proiezione PCA 2D con centroidi"),
        (ax2, "tsne_1", "tsne_2", "Proiezione t-SNE 2D"),
    ]:
        sns.scatterplot(
            data=sample, x=x, y=y, hue="kmeans_profile", hue_order=cluster_order,
            palette=palette, alpha=0.65, s=25, edgecolor=None, legend=False, ax=ax,
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
               markerfacecolor=palette[name], markeredgecolor="none", label=name)
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

    # Common normalized profiles: annotations retain the original means.
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

    # Figure 18: readable heatmap on its own page/figure.
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

    # Figure 19: radar chart only uses the eight model-input features.
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

    # Fit GMM and use concise descriptions in a separate figure legend.
    gmm = GaussianMixture(
        n_components=N_CLUSTERS, covariance_type="full", reg_covar=1e-5,
        n_init=10, random_state=42,
    ).fit(X_sample_pca)
    memberships = gmm.predict_proba(X_sample_pca)
    gmm_top1 = memberships.max(axis=1)
    gmm_labels = memberships.argmax(axis=1)
    sample["gmm_profile"] = pd.Series(gmm_labels).map(GMM_NAMES)
    gmm_order = [GMM_NAMES[i] for i in range(N_CLUSTERS)]
    gmm_palette = {GMM_NAMES[i]: plt.get_cmap("tab10")(i) for i in range(N_CLUSTERS)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8.5))
    sns.scatterplot(
        data=sample, x="pc_1", y="pc_2", hue="gmm_profile", hue_order=gmm_order,
        palette=gmm_palette, alpha=0.55, s=22, edgecolor=None, legend=False, ax=ax1,
    )
    ax1.set_title("GMM a 5 componenti nella proiezione PCA 2D", fontsize=15, fontweight="bold")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% varianza)", fontsize=13)
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% varianza)", fontsize=13)
    ax2.hist(gmm_top1, bins=np.linspace(0.2, 1.0, 17), color="#4e79a7", edgecolor="white")
    ax2.axvline(np.median(gmm_top1), color="crimson", linestyle="--", linewidth=2,
                label=f"Mediana = {np.median(gmm_top1):.2f}")
    ax2.set_title("Forza dell’affiliazione più alta per brano", fontsize=15, fontweight="bold")
    ax2.set_xlabel("Probabilità stimata del componente più probabile", fontsize=13)
    ax2.set_ylabel("Numero di brani", fontsize=13)
    ax2.legend(fontsize=11)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor=gmm_palette[name], markeredgecolor="none", label=name)
        for name in gmm_order
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2,
        title="Profili sintetici delle componenti (ID GMM arbitrari)",
        fontsize=11, title_fontsize=12, bbox_to_anchor=(0.5, 0.0), frameon=True,
    )
    fig.suptitle("GMM: profili musicali con appartenenze probabilistiche", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.16, 1, 0.94])
    save_to_project(fig, "fig24_gmm_soft_membership.png")

    print("Figure K-Means/GMM rigenerate in figures/ e Relazione/capitolo4/img/.")


if __name__ == "__main__":
    main()
