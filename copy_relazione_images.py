import os
import shutil

WORKSPACE = r"c:\Users\andre\Documents\GitHub\Progetto-Data-Science"
REL_DIR = os.path.join(WORKSPACE, "Relazione")
SRC_FIG = os.path.join(WORKSPACE, "figures")
P2_IMG = os.path.join(WORKSPACE, "Parte_2_Time_Series", "Relazione", "capitolo1", "img")

# 1. Copia immagini per capitoli Spotify (2, 3, 4)
mapping_spotify = {
    "capitolo2": [
        "fig01_distribuzione_generi.png",
        "fig02_audio_features_hist_kde.png",
        "fig03_discrete_features.png",
        "fig04_matrice_correlazione_iniziale.png",
    ],
    "capitolo3": [
        "fig05_distribuzione_sampling.png",
        "fig06_matrice_correlazione_post_sampling.png",
        "fig07_confronto_modelli_baseline.png",
        "fig08_confusion_matrix_best_model.png",
        "fig09_classification_report_heatmap.png",
        "fig10_roc_curves_multiclass.png",
        "fig11_precision_recall_curves.png",
        "fig12_learning_curve.png",
        "fig13_feature_importance.png",
    ],
    "capitolo4": [
        "fig14_pca_varianza_cumulativa.png",
        "fig15_tsne_2d_overview.png",
        "fig16_kmeans_elbow_wcss.png",
        "fig17_kmeans_tsne_clusters.png",
        "fig18_kmeans_feature_profiling.png",
        "fig19_kmeans_silhouette_plot.png",
        "fig20_dbscan_knn_distance.png",
        "fig21_dbscan_grid_search.png",
        "fig22_dbscan_tsne_clusters_noise.png",
        "fig23_dbscan_noise_profiling.png",
    ],
}

for cap, files in mapping_spotify.items():
    dst_dir = os.path.join(REL_DIR, cap, "img")
    os.makedirs(dst_dir, exist_ok=True)
    for f in files:
        src = os.path.join(SRC_FIG, f)
        dst = os.path.join(dst_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)

# 2. Copia immagini Capitolo 6 (ETL Transfermarkt)
dst_cap6 = os.path.join(REL_DIR, "capitolo6", "img")
os.makedirs(dst_cap6, exist_ok=True)
if os.path.exists(P2_IMG):
    for f in os.listdir(P2_IMG):
        if f.endswith(".png"):
            shutil.copy2(os.path.join(P2_IMG, f), os.path.join(dst_cap6, f))

# 3. Assicura .gitkeep per cartelle img ancora vuote (capitolo1, 5, 7, 8, 9)
for cap in ["capitolo1", "capitolo5", "capitolo7", "capitolo8", "capitolo9"]:
    dst_dir = os.path.join(REL_DIR, cap, "img")
    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(dst_dir, ".gitkeep"), "w") as fp:
        pass

print("Copia immagini completata con successo!")

