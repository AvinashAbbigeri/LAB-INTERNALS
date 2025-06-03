import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(iris.data)

plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i, label in enumerate(np.unique(iris.target)):
    plt.scatter(
        data_reduced[iris.target == label, 0],
        data_reduced[iris.target == label, 1],
        label=iris.target_names[label],
        color=colors[i]
    )
plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()
