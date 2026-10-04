# Unsupervised Learning: K-Means + PCA on Wine Chemistry

Clustering 178 wines from their 13 chemical measurements, with no labels used during fitting, then visualizing the structure with PCA.

**Dataset:** UCI Wine (`sklearn.datasets.load_wine`), 13 numeric features. The cultivar labels are hidden during fitting and used only once, at the end, as a sanity check.

## Run it
```bash
pip install -r requirements.txt
python unsupervised_pipeline.py
```
Outputs: `figures/`, `cluster_profiles.csv`, `run_output.txt` (full console log).

## Method
1. **Scale** all features with `StandardScaler`. Proline (hundreds to 1,600) would otherwise swamp features like hue (~1).
2. **Choose k** for k = 2..10 using the elbow method and silhouette score.
3. **Fit K-Means** with the chosen k (`n_init=10`, `random_state=42`).
4. **PCA** to 2 components on the same scaled data.
5. **Plot** PCA space colored by cluster, then profile each cluster.

## Choosing k
![elbow and silhouette](figures/elbow_silhouette.png)

| k | Inertia | Silhouette |
|---|---------|-----------|
| 2 | 1658.8 | 0.259 |
| **3** | **1277.9** | **0.285** |
| 4 | 1175.4 | 0.260 |
| 5 | 1109.5 | 0.202 |
| 6 | 1046.0 | 0.237 |

Silhouette peaks at **k = 3**. The elbow drops sharply from k=2 to k=3 and flattens afterward. The elbow is soft rather than razor-sharp, which is typical for real data. A silhouette of 0.285 indicates moderate, somewhat overlapping cluster structure, not perfectly separated blobs.

## PCA and explained variance
![PCA clusters](figures/pca_clusters.png)

- PC1: **36.2%**, PC2: **19.2%**
- **The 2D projection captures 55.4% of the total variance.** The remaining ~45% is not visible in the plot, so some overlap between clusters in 2D may be a projection artifact.
- For reference, 7 components are needed to reach ~89% and 8 for ~92%.
- Main loadings: PC1 is driven by OD280/OD315, hue, proline and flavanoid-related features (positive) versus malic acid (negative). PC2 is driven by color intensity, alcohol and proline.

## Cluster interpretation
![cluster heatmap](figures/cluster_heatmap.png)

Means in original units (from `cluster_profiles.csv`):

| Feature | Cluster 0 (n=65) | Cluster 1 (n=51) | Cluster 2 (n=62) |
|---|---|---|---|
| alcohol | 12.25 | 13.13 | **13.68** |
| malic acid | 1.90 | **3.31** | 2.00 |
| flavanoids | 2.05 | **0.82** | **3.00** |
| color intensity | **2.97** | **7.23** | 5.45 |
| hue | 1.06 | **0.69** | 1.07 |
| proline | 510 | 619 | **1100** |

- **Cluster 2: full-bodied, high-alcohol, phenol-rich wines.** Highest alcohol, proline, flavanoids and total phenols, with low alcalinity of ash. These are the "rich, structured" wines.
- **Cluster 0: light, pale wines.** Lowest alcohol and by far the lowest color intensity (~3), moderate phenols, low proline. Lighter, milder wines.
- **Cluster 1: deeply colored, acidic, low-phenol wines.** Highest malic acid and color intensity, lowest flavanoids and hue. Sharper, darker wines with a distinct chemical signature.

Example rows (from the script output) agree with these profiles: e.g. row 0 (alcohol 14.23, proline 1065, flavanoids 3.06) sits in cluster 2; row 173 (malic acid 5.65, color 7.7, flavanoids 0.61) in cluster 1; row 116 (alcohol 11.82, color 1.95) in cluster 0.

### Sanity check against the hidden labels
The dataset's three cultivars (grape varieties) were *not* used for fitting. Comparing afterwards gives an Adjusted Rand Index of **0.897**: clusters 2, 0 and 1 map almost one-to-one onto cultivars 0, 1 and 2 (only 6 of 178 wines land in a different cluster). So the clusters found purely from chemistry correspond to real grape varieties.

## Limitations
- Silhouette of 0.285 is moderate, so cluster boundaries are fuzzy.
- K-Means assumes roughly spherical clusters of similar size.
- The 2D PCA plot keeps only 55% of variance.
- Only 178 samples, so k is fairly stable here but less certain than it would be with more data.
