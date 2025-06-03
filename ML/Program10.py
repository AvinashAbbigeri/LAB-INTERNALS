import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

data = load_breast_cancer()
X_scaled = StandardScaler().fit_transform(data.data)
y = data.target

kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

print("Confusion Matrix:\n", confusion_matrix(y, y_kmeans))
print("\nClassification Report:\n", classification_report(y, y_kmeans))

X_pca = PCA(n_components=2).fit_transform(X_scaled)
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['Cluster'], df['True Label'] = y_kmeans, y

fig, axes = plt.subplots(1, 3, figsize=(24, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=100, edgecolor='black', alpha=0.7, ax=axes[0])
axes[0].set_title('K-Means Clustering')
sns.scatterplot(data=df, x='PC1', y='PC2', hue='True Label', palette='coolwarm', s=100, edgecolor='black', alpha=0.7, ax=axes[1])
axes[1].set_title('True Labels')
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=100, edgecolor='black', alpha=0.7, ax=axes[2])
centers = PCA(n_components=2).fit(X_scaled).transform(kmeans.cluster_centers_)
axes[2].scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', label='Centroids')
axes[2].set_title('K-Means with Centroids')
for ax in axes:
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()
plt.tight_layout()
plt.show()
