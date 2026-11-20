"""Regenerate the clustering plots used in Chapter 4.

Run from any directory with:
    python scripts/regenerate_clustering_profiles.py
or to regenerate only the 4 updated GMM/DBSCAN/HDBSCAN figures quickly:
    python scripts/regenerate_clustering_profiles.py --only-critical

Outputs are saved in both `figures/` and `Relazione/capitolo4/img/`:
- fig17_kmeans_tsne_clusters.png (PCA + t-SNE)
- fig18_kmeans_feature_profiling.png (Heatmap)
- fig19_kmeans_feature_radar.png (Radar Chart)
- fig23_dbscan_noise_profiling.png (Raincloud Plot: Acousticness, Energy, Speechiness)
- fig24_gmm_soft_membership.png (Mappa Topografica GMM con Ellissi e Corridoi Ibridi)
- fig24b_mood_pentagon_radviz.png (Pentagono Baricentrico dei Mood ordinato per continuità)
- fig25_hdbscan_soft_membership.png (HDBSCAN cluster densi in primo piano + affinità GMM noise DBSCAN)
"""

import argparse
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
MEMBERSHIP_PATH = ROOT / "data" / "clustering_soft_memberships_sample.csv"
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

# Coerenti al 100% con Tabella 4 (tab:gmm_profile) e con i 15 generi reali del dataset
GMM_DISPLAY_LABELS = {
    0: "GMM 0 · Energetico / Strumentale (Psytrance, DnB, Trance)",
    1: "GMM 1 · Acustico / Moderato (Dark Trap, Undergr. Rap, RnB)",
    2: "GMM 2 · Energetico / Vocale (Hardstyle, Trap, Emo)",
    3: "GMM 3 · Dance / Strumentale (Techno, Techhouse)",
    4: "GMM 4 · Urban / Vocale (Undergr. Rap, Dark Trap, Hiphop)",
}

GMM_SHORT_NAMES = {
    0: "Energ./Strum.",
    1: "Acust./Mod.",
    2: "Energ./Voc.",
    3: "Dance/Strum.",
    4: "Urban/Voc.",
}

GMM_RADVIZ_LABELS = {
    0: "GMM 0 · Energetico / Strumentale\n(Psytrance, DnB, Trance)",
    1: "GMM 1 · Acustico / Moderato\n(Dark Trap, Undergr. Rap, RnB)",
    2: "GMM 2 · Energetico / Vocale\n(Hardstyle, Trap, Emo)",
    3: "GMM 3 · Dance / Strumentale\n(Techno, Techhouse)",
    4: "GMM 4 · Urban / Vocale\n(Undergr. Rap, Dark Trap, Hiphop)",
}

PALETTE_5 = ["#2b5c8f", "#27ae60", "#e67e22", "#8e44ad", "#c0392b"]


def save_to_project(image: plt.Figure, filename: str, *, tight: bool = True) -> None:
    for directory in FIGURE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        image.savefig(directory / filename, dpi=300, bbox_inches="tight" if tight else None)
    plt.close(image)


def bounded_kde(values: np.ndarray, x_eval: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """KDE con riflessione ai bordi per feature limitate in [low, high]."""
    reflected = np.concatenate([values, 2 * low - values, 2 * high - values])
    kde = stats.gaussian_kde(reflected)
    density = kde(x_eval) * 3.0
    return density


def generate_raincloud_fig23(sample: pd.DataFrame) -> None:
    """Figura 23 (Fig. 29 PDF): Raincloud Plot sulle 3 feature più discriminanti (Acousticness, Energy, Speechiness)."""
    print("Generazione Raincloud Plot DBSCAN (Figura 23)...", flush=True)
    rng_rain = np.random.RandomState(42)

    features_raincloud = [
        ("acousticness", "Acousticness (Timbrica Naturale vs Sintetica)", (0.0, 1.0), 1.0),
        ("energy", "Energy (Intensità ed Energia Sonora)", (0.0, 1.0), 1.0),
        ("speechiness", "Speechiness (Contenuto Parlato / Rap)", (0.0, 0.85), 1.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    groups = [
        ("Cluster Core (n=5.538)", sample["dbscan_cluster"] != -1, "#2980b9"),
        ("Rumore DBSCAN (n=462)", sample["dbscan_cluster"] == -1, "#e74c3c"),
    ]

    for ax, (feat, feat_label, x_limits, hi_bound) in zip(axes, features_raincloud):
        for group_idx, (group_label, mask, color) in enumerate(groups):
            sub_vals = sample.loc[mask, feat].dropna().to_numpy()
            y_base = group_idx * 1.55

            # 1. Half-KDE (Cloud) con riflessione ai bordi
            x_eval = np.linspace(x_limits[0], x_limits[1], 240)
            kde_vals = bounded_kde(sub_vals, x_eval, low=0.0, high=hi_bound)
            kde_scaled = (kde_vals / (kde_vals.max() + 1e-12)) * 0.62
            ax.fill_between(x_eval, y_base, y_base + kde_scaled, color=color, alpha=0.50, edgecolor=color, lw=1.6)

            # 2. Barra IQR + Mediana
            q25, q50, q75 = np.percentile(sub_vals, [25, 50, 75])
            ax.plot([q25, q75], [y_base - 0.13, y_base - 0.13], color=color, lw=4.5, solid_capstyle="round", zorder=3)
            ax.scatter([q50], [y_base - 0.13], color="white", edgecolor="#1a252f", s=55, linewidths=1.5, zorder=5)

            # Etichetta quantitativa mediana/IQR
            txt_x = min(q75 + 0.03 * (x_limits[1] - x_limits[0]), x_limits[1] - 0.28 * (x_limits[1] - x_limits[0]))
            if feat == "energy" and group_idx == 0:
                txt_x = q25 - 0.30 * (x_limits[1] - x_limits[0])
            ax.text(
                txt_x, y_base + 0.68,
                f"med = {q50:.2f}  (IQR: {q25:.2f}–{q75:.2f})",
                fontsize=9.2, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, alpha=0.88, lw=0.9),
            )

            # 3. Jittered scatter (Rain)
            n_pts = min(400, len(sub_vals))
            sample_pts = rng_rain.choice(sub_vals, size=n_pts, replace=False)
            jitter = rng_rain.uniform(-0.22, -0.05, size=n_pts)
            ax.scatter(sample_pts, y_base - 0.22 + jitter, color=color, alpha=0.24, s=11, edgecolors="none")

        ax.set_ylim(-0.55, 2.55)
        ax.set_xlim(x_limits)
        ax.set_yticks([0.20, 1.75])
        ax.set_yticklabels(["Cluster Core\n(n=5.538)", "Rumore DBSCAN\n(n=462)"], fontsize=10.5, fontweight="bold")
        ax.set_xlabel(feat_label, fontsize=11.5)
        ax.set_title(f"Distribuzione di {feat.capitalize()}", fontsize=13.5, fontweight="bold", pad=10)
        ax.grid(axis="x", linestyle=":", alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Raincloud Plot: Diagnosi Acustica della Separazione tra Core e Rumore DBSCAN\n"
        "(Le 3 feature a maggiore divario: Acousticness, Energy e Speechiness)",
        fontsize=14.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_to_project(fig, "fig23_dbscan_noise_profiling.png")


def generate_gmm_map_fig24(sample: pd.DataFrame, gmm: GaussianMixture, exp_var: np.ndarray) -> None:
    """Figura 24: Mappa GMM pulita con ellissi di covarianza + classifica leggibile dei corridoi ibridi."""
    print("Generazione Mappa GMM e Corridoi Ibridi (Figura 24)...", flush=True)
    fig, (ax_main, ax_side) = plt.subplots(
        1, 2, figsize=(18.5, 8.6), gridspec_kw={"width_ratios": [1.85, 1.05]}
    )

    # Scatter brani standard (sottile e leggibile, senza KDE blu confusa di sfondo)
    for i in range(N_CLUSTERS):
        sub = sample[(sample["gmm_component"] == i) & (~sample["gmm_cross_mood_candidate"])]
        ax_main.scatter(
            sub["pc_1"], sub["pc_2"],
            c=PALETTE_5[i], alpha=0.22, s=15, edgecolors="none",
            label=GMM_DISPLAY_LABELS[i], zorder=2,
        )

    # Ellissi di covarianza GMM (1 sigma continua, 2 sigma tratteggiata leggera)
    for i in range(N_CLUSTERS):
        mean_2d = gmm.means_[i, :2]
        cov_2d = gmm.covariances_[i][:2, :2]
        vals, vecs = np.linalg.eigh(cov_2d)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # 1 sigma (68%)
        w_1sig, h_1sig = 2 * np.sqrt(vals)
        ell_1 = patches.Ellipse(
            xy=mean_2d, width=w_1sig, height=h_1sig, angle=theta,
            edgecolor=PALETTE_5[i], facecolor="none", linewidth=2.4, linestyle="-", zorder=5,
        )
        ax_main.add_patch(ell_1)

        # 2 sigma (95%)
        w_2sig, h_2sig = 2 * np.sqrt(vals * 5.991)
        ell_2 = patches.Ellipse(
            xy=mean_2d, width=w_2sig, height=h_2sig, angle=theta,
            edgecolor=PALETTE_5[i], facecolor="none", linewidth=1.2, linestyle="--", alpha=0.48, zorder=3,
        )
        ax_main.add_patch(ell_2)

    # Brani ponte cross-mood (dimensione calibrata per non seppellire i centroidi)
    cross_tracks = sample[sample["gmm_cross_mood_candidate"]]
    ax_main.scatter(
        cross_tracks["pc_1"], cross_tracks["pc_2"],
        c="#f1c40f", s=26, marker="D", edgecolors="#2c3e50", linewidths=0.65,
        alpha=0.82, zorder=6, label=f"Brani Ponte Cross-Mood (gap ≤ 0,15; N={len(cross_tracks)})",
    )

    # Centroidi ed etichette GMM 0..4 SEMPRE IN PRIMO PIANO con offset dedicati anti-sovrapposizione
    label_offsets = {
        0: (0.35, -0.55),   # GMM 0 (destra-basso)
        1: (-1.35, -0.75),  # GMM 1 (sinistra-basso, ben separato da GMM 4)
        2: (0.25, -0.75),   # GMM 2 (basso-centro)
        3: (0.35, 0.45),    # GMM 3 (destra-alto)
        4: (-1.35, 0.75),   # GMM 4 (sinistra-alto, ben separato da GMM 1)
    }
    for i in range(N_CLUSTERS):
        mean_2d = gmm.means_[i, :2]
        dx, dy = label_offsets[i]
        ax_main.scatter(
            mean_2d[0], mean_2d[1], c=PALETTE_5[i], s=190, marker="X",
            edgecolors="white", linewidths=2.0, zorder=9,
        )
        ax_main.annotate(
            f"GMM {i}",
            xy=(mean_2d[0], mean_2d[1]),
            xytext=(mean_2d[0] + dx, mean_2d[1] + dy),
            fontsize=10.5, fontweight="bold", color="#111111", zorder=10,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.95, edgecolor=PALETTE_5[i], lw=1.8),
            arrowprops=dict(arrowstyle="-", color=PALETTE_5[i], lw=1.3, alpha=0.85),
        )

    ax_main.text(
        0.02, 0.02,
        "Ellissi: linea continua = 1σ (68%), tratteggiata = 2σ (95%).\n"
        "Nota: GMM 1 e GMM 4 si sovrappongono in PC1–PC2 ma si separano su PC3–PC4 (acusticità ed energia).",
        transform=ax_main.transAxes, fontsize=8.8, color="#34495e",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9fa", edgecolor="#bdc3c7", alpha=0.92),
        zorder=11,
    )

    ax_main.set_title("Mappa GMM nello Spazio PCA 2D: Ellissi di Covarianza e Brani Ponte", fontsize=13.5, fontweight="bold", pad=10)
    ax_main.set_xlabel(f"Prima Componente Principale PC1 ({exp_var[0] * 100:.1f}% varianza)", fontsize=11.5)
    ax_main.set_ylabel(f"Seconda Componente Principale PC2 ({exp_var[1] * 100:.1f}% varianza)", fontsize=11.5)
    ax_main.grid(True, linestyle=":", alpha=0.45)

    # Legenda pulita sotto il grafico principale
    handles, labels = ax_main.get_legend_handles_labels()
    # Rendi visibili i pallini in legenda (alpha=1)
    clean_handles = []
    for idx_h, h in enumerate(handles):
        if idx_h < N_CLUSTERS:
            clean_handles.append(
                Line2D([0], [0], marker="o", linestyle="", markersize=8,
                       markerfacecolor=PALETTE_5[idx_h], markeredgecolor="none", label=labels[idx_h])
            )
        else:
            clean_handles.append(
                Line2D([0], [0], marker="D", linestyle="", markersize=7,
                       markerfacecolor="#f1c40f", markeredgecolor="#2c3e50", label=labels[idx_h])
            )

    # Pannello destro: classifica corridoi ibridi con etichette intere e leggibili
    affinity_cols = [f"gmm_affinity_{i}" for i in range(N_CLUSTERS)]
    bridge_pairs = []
    for _, row in cross_tracks.iterrows():
        affs = row[affinity_cols].to_numpy()
        top2 = np.argsort(affs)[-2:]
        pair = tuple(sorted([int(top2[0]), int(top2[1])]))
        bridge_pairs.append(pair)

    pair_counts = pd.Series(bridge_pairs).value_counts().head(8)
    total_bridges = len(cross_tracks)
    pair_labels = [
        f"GMM {p[0]} ↔ GMM {p[1]}\n({GMM_SHORT_NAMES[p[0]]} ↔ {GMM_SHORT_NAMES[p[1]]})"
        for p in pair_counts.index
    ]

    bars = ax_side.barh(
        range(len(pair_counts)), pair_counts.values,
        color="#e67e22", edgecolor="#2c3e50", alpha=0.88, height=0.72,
    )
    ax_side.set_yticks(range(len(pair_counts)))
    ax_side.set_yticklabels(pair_labels, fontsize=9.8)
    ax_side.invert_yaxis()
    ax_side.set_xlim(0, pair_counts.max() * 1.32)
    ax_side.set_xlabel("Numero di Tracce Ponte Condivise (gap ≤ 0,15)", fontsize=11)
    ax_side.set_title("Corridoi Ibridi più Frequenti\n(Opportunità di Playlist Cross-Mood)", fontsize=13.5, fontweight="bold", pad=10)
    for idx_b, v in enumerate(pair_counts.values):
        pct = v / total_bridges * 100.0
        ax_side.text(v + 2.5, idx_b, f"{v} ({pct:.1f}%)", va="center", fontsize=9.8, fontweight="bold", color="#7e3805")
    ax_side.grid(axis="x", linestyle=":", alpha=0.6)
    ax_side.spines["top"].set_visible(False)
    ax_side.spines["right"].set_visible(False)

    fig.legend(
        handles=clean_handles, loc="lower center", ncol=2,
        title="Componenti GMM (profili coerenti con Tabella 4)",
        fontsize=10, title_fontsize=11, bbox_to_anchor=(0.5, -0.02), frameon=True,
    )
    fig.tight_layout(rect=[0, 0.13, 1, 0.98])
    save_to_project(fig, "fig24_gmm_soft_membership.png")


def generate_radviz_fig24b(sample: pd.DataFrame) -> None:
    """Figura 24b (Fig. 25 PDF): Pentagono Baricentrico dei Mood ordinato per continuità acustica,
    senza cerchi guida fuorvianti e con i generi reali del dataset."""
    print("Generazione Pentagono Baricentrico dei Mood (Figura 24b)...", flush=True)
    fig, ax = plt.subplots(figsize=(12, 10.5))

    # Ordine dei vertici lungo il perimetro (in senso orario da in alto):
    # GMM 0 (Top) -> GMM 2 (Top-Right) -> GMM 4 (Bottom-Right) -> GMM 1 (Bottom-Left) -> GMM 3 (Top-Left)
    # In questo modo l'87,8% (359/409) dei brani ponte cade direttamente lungo i lati esterni!
    vertex_order = [0, 2, 4, 1, 3]
    angles = [np.pi / 2 - k * (2 * np.pi / 5) for k in range(5)]
    vertices = {
        comp: np.array([np.cos(ang), np.sin(ang)])
        for comp, ang in zip(vertex_order, angles)
    }

    # Perimetro del pentagono e raggi sottili dal baricentro (NESSUN cerchio 50%/75% fuorviante!)
    pent_coords = np.array([vertices[c] for c in vertex_order] + [vertices[vertex_order[0]]])
    ax.plot(pent_coords[:, 0], pent_coords[:, 1], color="#2c3e50", linewidth=2.2, linestyle="-", alpha=0.75, zorder=2)

    for c in vertex_order:
        v = vertices[c]
        ax.plot([0, v[0]], [0, v[1]], color="#95a5a6", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.scatter([0], [0], color="#7f8c8d", s=30, zorder=2)

    # Coordinate baricentriche per ogni traccia
    P = np.zeros((len(sample), 2))
    for c in range(N_CLUSTERS):
        aff = sample[f"gmm_affinity_{c}"].to_numpy()[:, np.newaxis]
        P += aff * vertices[c]
    sample["radviz_x"] = P[:, 0]
    sample["radviz_y"] = P[:, 1]

    # Scatter tracce standard
    for c in range(N_CLUSTERS):
        sub = sample[(sample["gmm_component"] == c) & (~sample["gmm_cross_mood_candidate"])]
        ax.scatter(
            sub["radviz_x"], sub["radviz_y"],
            c=PALETTE_5[c], alpha=0.28, s=18, edgecolors="none",
            label=GMM_DISPLAY_LABELS[c], zorder=3,
        )

    # Scatter tracce ponte cross-mood
    cross = sample[sample["gmm_cross_mood_candidate"]]
    ax.scatter(
        cross["radviz_x"], cross["radviz_y"],
        c="#f1c40f", s=38, marker="D", edgecolors="#2c3e50", linewidths=0.85,
        alpha=0.90, zorder=5, label=f"Brani Ponte Cross-Mood (gap ≤ 0,15; N={len(cross)})",
    )

    # Calcola conteggi sui 5 lati esterni per annotare i corridoi principali lungo il perimetro
    affinity_cols = [f"gmm_affinity_{i}" for i in range(N_CLUSTERS)]
    bridge_pairs = []
    for _, row in cross.iterrows():
        top2 = np.argsort(row[affinity_cols].to_numpy())[-2:]
        bridge_pairs.append(tuple(sorted([int(top2[0]), int(top2[1])])))
    pair_counts = pd.Series(bridge_pairs).value_counts().to_dict()

    for idx_edge in range(5):
        c_a = vertex_order[idx_edge]
        c_b = vertex_order[(idx_edge + 1) % 5]
        cnt = pair_counts.get(tuple(sorted([c_a, c_b])), 0)
        if cnt >= 20:
            mid = 0.5 * (vertices[c_a] + vertices[c_b])
            outward = mid / (np.linalg.norm(mid) + 1e-12)
            pos = mid + 0.14 * outward
            ax.text(
                pos[0], pos[1], f"{cnt} brani\nponte",
                ha="center", va="center", fontsize=9.0, fontweight="bold", color="#7e3805",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fef9e7", edgecolor="#e67e22", alpha=0.95, lw=1.2),
                zorder=7,
            )

    # Vertici ed etichette dei 5 poli reali
    label_positions = {
        0: (0.0, 1.18, "center", "bottom"),
        2: (1.14, 0.36, "left", "center"),
        4: (0.72, -1.04, "left", "top"),
        1: (-0.72, -1.04, "right", "top"),
        3: (-1.14, 0.36, "right", "center"),
    }
    for c in vertex_order:
        vx, vy = vertices[c]
        ax.scatter(vx, vy, c=PALETTE_5[c], s=260, edgecolors="#2c3e50", linewidths=2.4, zorder=6)
        lx, ly, ha, va = label_positions[c]
        ax.text(
            lx, ly, GMM_RADVIZ_LABELS[c],
            ha=ha, va=va, fontsize=10.2, fontweight="bold", color="#1a252f",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=PALETTE_5[c], alpha=0.95, lw=1.8),
            zorder=8,
        )

    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.42, 1.52)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Pentagono Baricentrico dei Mood Musicali (RadViz Simplex)\n"
        "Vertici ordinati per continuità acustica: 87,8% dei brani ponte lungo i lati di transizione",
        fontsize=14, fontweight="bold", pad=16,
    )

    clean_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor=PALETTE_5[i], markeredgecolor="none", label=GMM_DISPLAY_LABELS[i])
        for i in range(N_CLUSTERS)
    ]
    clean_handles.append(
        Line2D([0], [0], marker="D", linestyle="", markersize=7.5,
               markerfacecolor="#f1c40f", markeredgecolor="#2c3e50",
               label=f"Brani Ponte Cross-Mood (gap ≤ 0,15; N={len(cross)})")
    )
    ax.legend(
        handles=clean_handles, loc="lower center", bbox_to_anchor=(0.5, -0.07),
        ncol=2, fontsize=9.5, framealpha=0.95,
    )

    fig.tight_layout()
    save_to_project(fig, "fig24b_mood_pentagon_radviz.png")


def generate_hdbscan_fig25(sample: pd.DataFrame, exp_var: np.ndarray) -> None:
    """Figura 25 (Fig. 30 PDF): HDBSCAN con noise di sfondo leggero e 10 cluster densi in primo piano
    + affinità GMM massima dei 462 punti noise DBSCAN con scala leggibile e riepilogo."""
    print("Generazione Grafico HDBSCAN + Affinità GMM (Figura 25)...", flush=True)

    # Carica etichette HDBSCAN già calcolate sul campione esatto (o ricalcola se assenti)
    if "hdbscan_cluster" not in sample.columns:
        if MEMBERSHIP_PATH.exists():
            saved = pd.read_csv(MEMBERSHIP_PATH)
            if len(saved) == len(sample) and (saved["id"].values == sample["id"].values).all():
                sample["hdbscan_cluster"] = saved["hdbscan_cluster"].values
            else:
                merged = pd.merge(sample[["id"]], saved[["id", "hdbscan_cluster"]], on="id", how="left")
                sample["hdbscan_cluster"] = merged["hdbscan_cluster"].fillna(-1).astype(int).values
        else:
            import hdbscan
            model = hdbscan.HDBSCAN(
                min_cluster_size=30, min_samples=5, metric="euclidean",
                cluster_selection_method="eom", prediction_data=True,
            )
            X_4d = sample[["pc_1", "pc_2", "pc_3", "pc_4"]].to_numpy()
            sample["hdbscan_cluster"] = model.fit_predict(X_4d)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.6))

    # --- PANNELLO SINISTRO: Noise HDBSCAN SOTTO, 10 cluster densi SOPRA ---
    hdb_noise = sample[sample["hdbscan_cluster"] == -1]
    hdb_clusters = sample[sample["hdbscan_cluster"] != -1]
    cluster_ids = sorted(hdb_clusters["hdbscan_cluster"].unique())
    cmap_hdb = plt.get_cmap("tab10")

    ax1.scatter(
        hdb_noise["pc_1"], hdb_noise["pc_2"],
        c="#cfd8dc", marker="o", s=11, alpha=0.38, edgecolors="none",
        zorder=1, label=f"Noise HDBSCAN (n={len(hdb_noise)})",
    )

    hdb_handles = []
    for cid in cluster_ids:
        sub = hdb_clusters[hdb_clusters["hdbscan_cluster"] == cid]
        col = cmap_hdb(cid % 10)
        ax1.scatter(
            sub["pc_1"], sub["pc_2"],
            c=[col], s=30, alpha=0.88, edgecolors="white", linewidths=0.3,
            zorder=3,
        )
        hdb_handles.append(
            Line2D([0], [0], marker="o", linestyle="", markersize=7.5,
                   markerfacecolor=col, markeredgecolor="white", label=f"H{cid} (n={len(sub)})")
        )
    hdb_handles.append(
        Line2D([0], [0], marker="o", linestyle="", markersize=6,
               markerfacecolor="#cfd8dc", markeredgecolor="#90a4ae",
               label=f"Noise HDBSCAN (n={len(hdb_noise)}, {len(hdb_noise)/len(sample)*100:.1f}%)")
    )

    ax1.set_title("HDBSCAN: 10 Micro-Cluster Densi (in primo piano) vs Sfondo Periferico", fontsize=12.8, fontweight="bold", pad=10)
    ax1.set_xlabel(f"PC1 ({exp_var[0] * 100:.1f}% varianza)", fontsize=11.5)
    ax1.set_ylabel(f"PC2 ({exp_var[1] * 100:.1f}% varianza)", fontsize=11.5)
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(
        handles=hdb_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=4, fontsize=8.8, frameon=True, title="Comunità HDBSCAN (min_cluster_size=30, min_samples=5)",
        title_fontsize=9.5,
    )

    # --- PANNELLO DESTRO: 462 punti noise DBSCAN colorati per affinità GMM massima ---
    ax2.scatter(
        sample["pc_1"], sample["pc_2"],
        c="#e0e6ea", s=10, alpha=0.30, edgecolors="none", zorder=1,
        label="Resto del catalogo (sfondo)",
    )
    db_noise = sample[sample["dbscan_cluster"] == -1]
    sc = ax2.scatter(
        db_noise["pc_1"], db_noise["pc_2"],
        c=db_noise["gmm_top_affinity"], cmap="viridis",
        vmin=0.35, vmax=1.0, s=38, alpha=0.92,
        edgecolors="#2c3e50", linewidths=0.35, zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax2, pad=0.02, shrink=0.92)
    cbar.set_label("Affiliazione GMM massima (scala 0,35 – 1,00)", fontsize=10.5)

    med_aff = float(db_noise["gmm_top_affinity"].median())
    pct_gt80 = float((db_noise["gmm_top_affinity"] > 0.80).mean() * 100.0)
    n_cross_noise = int((db_noise["gmm_top2_gap"] <= 0.15).sum())

    ax2.text(
        0.97, 0.03,
        f"462 brani noise DBSCAN:\n"
        f"• Mediana affinità GMM: {med_aff:.3f}\n"
        f"• {pct_gt80:.1f}% con affinità > 0,80\n"
        f"• {n_cross_noise} brani ({n_cross_noise/len(db_noise)*100:.1f}%) cross-mood",
        transform=ax2.transAxes, ha="right", va="bottom", fontsize=9.2, color="#1a252f",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#2c3e50", alpha=0.93, lw=1.0),
        zorder=5,
    )

    ax2.set_title("I 462 Brani Noise DBSCAN: Affinità ai 5 Profili GMM", fontsize=12.8, fontweight="bold", pad=10)
    ax2.set_xlabel(f"PC1 ({exp_var[0] * 100:.1f}% varianza)", fontsize=11.5)
    ax2.set_ylabel(f"PC2 ({exp_var[1] * 100:.1f}% varianza)", fontsize=11.5)
    ax2.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle(
        "HDBSCAN e Rivalutazione Probabilistica: il Noise è Geometrico, non Musicale",
        fontsize=14.5, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    save_to_project(fig, "fig25_hdbscan_soft_membership.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-critical", action="store_true",
        help="Rigenera rapidamente solo fig23, fig24, fig24b e fig25 (saltando t-SNE per fig17-19).",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    print("Caricamento dataset e preparazione trasformazioni PCA...", flush=True)
    df = pd.read_csv(DATA_PATH)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])
    pca = PCA(n_components=4, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    exp_var = pca.explained_variance_ratio_

    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(df), size=SAMPLE_SIZE, replace=False)
    sample = df.iloc[sample_indices].copy().reset_index(drop=True)
    X_sample_pca = X_pca[sample_indices]
    X_sample_scaled = X_scaled[sample_indices]
    sample["pc_1"], sample["pc_2"] = X_sample_pca[:, 0], X_sample_pca[:, 1]
    sample["pc_3"], sample["pc_4"] = X_sample_pca[:, 2], X_sample_pca[:, 3]

    if not args.only_critical:
        print("Calcolo t-SNE 2D...", flush=True)
        X_tsne = TSNE(
            n_components=2, perplexity=40, random_state=42, max_iter=1000
        ).fit_transform(X_sample_scaled)
        sample["tsne_1"], sample["tsne_2"] = X_tsne[:, 0], X_tsne[:, 1]

        print("Addestramento K-Means e rigenerazione Fig. 17, 18, 19...", flush=True)
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
                f"PC1 ({exp_var[0] * 100:.1f}% varianza)"
                if x == "pc_1" else "t-SNE Dimensione 1", fontsize=13
            )
            ax.set_ylabel(
                f"PC2 ({exp_var[1] * 100:.1f}% varianza)"
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

    # DBSCAN
    print("Addestramento DBSCAN (eps=0.60, min_samples=5)...", flush=True)
    db = DBSCAN(eps=0.60, min_samples=5).fit(X_sample_pca)
    sample["dbscan_cluster"] = db.labels_
    generate_raincloud_fig23(sample)

    # GMM
    print("Addestramento GMM (k=5, full)...", flush=True)
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

    generate_gmm_map_fig24(sample, gmm, exp_var)
    generate_radviz_fig24b(sample)
    generate_hdbscan_fig25(sample, exp_var)

    print("Rigenerazione completata con successo in figures/ e Relazione/capitolo4/img/!", flush=True)


if __name__ == "__main__":
    main()
