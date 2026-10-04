"""Unsupervised learning pipeline: K-Means + PCA on the Wine dataset.

Labels are loaded only for an optional sanity check AFTER clustering;
they are never used to fit K-Means or PCA.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

RANDOM_STATE = 42
FIG = "figures"

# 1. Load data, hide labels ---------------------------------------------------
wine = load_wine(as_frame=True)
X = wine.data.copy()
hidden_labels = wine.target  # only used for the post-hoc check
print(f"Shape: {X.shape}; features: {list(X.columns)}\n")

# 2. Scale --------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Elbow + silhouette over k ------------------------------------------------
ks = range(2, 11)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, km.labels_))
    print(f"k={k:2d}  inertia={km.inertia_:8.2f}  silhouette={sils[-1]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), inertias, "o-"); ax[0].set_title("Elbow method")
ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks), sils, "o-", color="tab:orange"); ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k"); ax[1].set_ylabel("Mean silhouette")
plt.tight_layout(); plt.savefig(f"{FIG}/elbow_silhouette.png", dpi=150); plt.close()

best_k = list(ks)[int(np.argmax(sils))]
print(f"\nBest k by silhouette: {best_k}")

# 4. Final K-Means ------------------------------------------------------------
km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
X["cluster"] = km.labels_

# 5. PCA ----------------------------------------------------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
evr = pca.explained_variance_ratio_
print(f"\nExplained variance ratio: PC1={evr[0]:.4f}, PC2={evr[1]:.4f}, total={evr.sum():.4f}")

full = PCA().fit(X_scaled)
print("Cumulative variance (all PCs):", np.round(np.cumsum(full.explained_variance_ratio_), 3))

# 6. Scatter plot -------------------------------------------------------------
plt.figure(figsize=(7, 6))
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km.labels_, cmap="viridis", s=45, edgecolor="k", linewidth=0.3)
centers = pca.transform(km.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Centroids")
plt.xlabel(f"PC1 ({evr[0]*100:.1f}% variance)"); plt.ylabel(f"PC2 ({evr[1]*100:.1f}% variance)")
plt.title(f"K-Means (k={best_k}) clusters in PCA space ({evr.sum()*100:.1f}% variance captured)")
plt.legend(*sc.legend_elements(), title="Cluster", loc="best")
plt.tight_layout(); plt.savefig(f"{FIG}/pca_clusters.png", dpi=150); plt.close()

# PCA loadings (what do PC1/PC2 mean?)
loadings = pd.DataFrame(pca.components_.T, index=wine.data.columns, columns=["PC1", "PC2"])
print("\nTop PCA loadings:\n", loadings.reindex(loadings.abs().sum(axis=1).sort_values(ascending=False).index).head(6).round(3))

# 7. Interpret clusters -------------------------------------------------------
profile = X.groupby("cluster").mean().round(2)
print("\nCluster sizes:\n", X["cluster"].value_counts().sort_index())
print("\nCluster means (original units):\n", profile.T)
profile.to_csv("cluster_profiles.csv")

z = pd.DataFrame(X_scaled, columns=wine.data.columns); z["cluster"] = km.labels_
zprof = z.groupby("cluster").mean()
plt.figure(figsize=(9, 6))
import seaborn as sns
sns.heatmap(zprof.T, annot=True, fmt=".1f", cmap="coolwarm", center=0)
plt.title("Cluster profiles (z-scores per feature)")
plt.tight_layout(); plt.savefig(f"{FIG}/cluster_heatmap.png", dpi=150); plt.close()

print("\nSample rows per cluster:")
for c in range(best_k):
    print(f"\n--- Cluster {c} ---")
    print(X[X.cluster == c].sample(3, random_state=RANDOM_STATE).round(2).to_string())

# Optional sanity check against hidden labels (NOT used for fitting)
print(f"\nAdjusted Rand Index vs hidden cultivar labels: {adjusted_rand_score(hidden_labels, km.labels_):.3f}")
print("Crosstab (cluster vs cultivar):\n", pd.crosstab(X["cluster"], hidden_labels.rename("cultivar")))
