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


pc1_loadings = np.abs(eigenvectors[:, 0])
top_2_pc1 = pc1_loadings.argsort()[::-1][:2]

print("Top 2 attributes with max contribution:")
for i in top_2_pc1:
    print(f"{X.columns[i]}")

# plot scatter
plt.figure(figsize=(8, 6))
if diagnosis is not None:
    colors = {'M': 'red', 'B': 'blue'}
    plt.scatter(pca_applied[:, 0], pca_applied[:, 1], c=diagnosis.map(colors), alpha=0.5)
else:
    plt.scatter(pca_applied[:, 0], pca_applied[:, 1], alpha=0.5)
plt.title('PCA: PC1 vs PC2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.savefig('pca_scatter.png')
plt.close()

# plot variance
plt.figure(figsize=(8, 5))
explained_variance = eigenvalues / np.sum(eigenvalues)
plt.plot(np.cumsum(explained_variance), marker='o', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Variance Explained')
plt.grid(True)
plt.savefig('pca_variance.png')
plt.close()
