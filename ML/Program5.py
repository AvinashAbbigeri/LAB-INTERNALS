import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

data = np.random.rand(100)
labels = ["Class1" if x <= 0.5 else "Class2" for x in data[:50]]
train_data, test_data = data[:50], data[50:]
k_values = [1, 2, 3, 4, 5, 20, 30]

def knn_classifier(train_data, train_labels, test_point, k):
    distances = sorted(((abs(test_point - x), label) for x, label in zip(train_data, train_labels)), key=lambda x: x[0])
    return Counter(label for _, label in distances[:k]).most_common(1)[0][0]

print("--- k-Nearest Neighbors Classification ---\n")
results = {}

for k in k_values:
    classified_labels = [knn_classifier(train_data, labels, x, k) for x in test_data]
    results[k] = classified_labels
    print(f"Results for k = {k}:")
    for i, label in enumerate(classified_labels, start=51):
        print(f"Point x{i} (value: {test_data[i-51]:.4f}) is classified as {label}")
    print()

print("Classification complete.\n")

for k in k_values:
    classified_labels = results[k]
    plt.figure(figsize=(10, 6))
    plt.scatter(train_data, [0]*50, c=["blue" if l=="Class1" else "red" for l in labels], label="Training Data", marker="o")
    plt.scatter(test_data, [1]*50, c=["blue" if l=="Class1" else "red" for l in classified_labels], label="Test Data", marker="x")
    plt.title(f"k-NN Classification Results for k = {k}")
    plt.xlabel("Data Points")
    plt.ylabel("Classification Level")
    plt.legend()
    plt.grid(True)
    plt.show()
