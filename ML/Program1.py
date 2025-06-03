import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame
num_cols = df.select_dtypes(include=[np.number]).columns

fig, axes = plt.subplots(2, len(num_cols), figsize=(20, 6))
for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, bins=30, color='blue', ax=axes[0, i])
    axes[0, i].set_title(f'Distribution of {col}')
    sns.boxplot(x=df[col], color='orange', ax=axes[1, i])
    axes[1, i].set_title(f'Box Plot of {col}')
plt.tight_layout(); plt.show()

print("Outliers Detection:")
for col in num_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
    print(f"{col}: {len(outliers)} outliers")

print("\nDataset Summary:\n", df.describe())
