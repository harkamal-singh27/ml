import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data cleaning
df = pd.read_csv('pca.csv')
diagnosis = df['diagnosis'] if 'diagnosis' in df.columns else None
X = df.drop(['id', 'diagnosis'], axis=1, errors='ignore')
X = X.dropna(axis=1, how='all')
X = X.dropna(axis=0)

X_mean = X.mean()
X_std = X.std()
X_scaled = (X - X_mean) / X_std

# covarmatrix
n_samples = X_scaled.shape[0]
cov_matrix = np.dot(X_scaled.T, X_scaled) / (n_samples - 1)

# eigen vector
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# pca
pc_eigenvectors = eigenvectors[:, :2]
pca_applied = np.dot(X_scaled, pc_eigenvectors)

# k-means
def manual_kmeans(X, k, max_iters=100, seed=42):
    np.random.seed(seed)
    
    # initialize centroids
    random_idx = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[random_idx, :].copy()
    
    labels = np.zeros(X.shape[0])
    
    for _ in range(max_iters):
        # assign clusters
        distances = np.sqrt(np.sum((X[:, np.newaxis, :] - centroids)**2, axis=2))
        new_labels = np.argmin(distances, axis=1)
        
        # convergence check
        if np.all(labels == new_labels):
            break
        labels = new_labels
            
        # update centroids
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)

    # inertia WCSS
    inertia = 0
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - centroids[i])**2)
            
    return centroids, labels, inertia

# elbow method
inertias = []
max_k = 10
for k in range(1, max_k + 1):
    _, _, inertia = manual_kmeans(pca_applied, k=k)
    inertias.append(inertia)

# plot elbow
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_k + 1), inertias, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig('kmeans_elbow.png')
plt.close()

# plot k-means
best_k = 2
final_centroids, final_labels, _ = manual_kmeans(pca_applied, k=best_k)

plt.figure(figsize=(8, 6))

colors = ['blue', 'red']
for i in range(best_k):
    cluster_points = pca_applied[final_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], alpha=0.5, label=f'Cluster {i}')

# plot centroids
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='black', marker='X', s=200, label='Centroids', linewidths=2)

plt.title('K-Means Clusters Map')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.savefig('kmeans_clusters.png')
plt.close()
