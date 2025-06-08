from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

corpus = [
    "The patient was diagnosed with diabetes and hypertension.",
    "MRI scans reveal abnormalities in the brain tissue.",
    "The treatment involves antibiotics and regular monitoring.",
    "Symptoms include fever, fatigue, and muscle pain.",
    "The vaccine is effective against several viral infections.",
    "Doctors recommend physical therapy for recovery.",
    "The clinical trial results were published in the journal.",
    "The surgeon performed a minimally invasive procedure.",
    "The prescription includes pain relievers and anti-inflammatory drugs.",
    "The diagnosis confirmed a rare genetic disorder."
]

sentences = [s.lower().split() for s in corpus]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=50)

words = model.wv.index_to_key
embeddings = np.array([model.wv[w] for w in words])
tsne_result = TSNE(n_components=2, random_state=42, perplexity=5, max_iter=300).fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], color="blue")
for i, word in enumerate(words):
    plt.text(tsne_result[i, 0] + 0.02, tsne_result[i, 1] + 0.02, word, fontsize=12)
plt.title("Word Embeddings Visualization (Medical Domain)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()

def find_similar(word, n=5):
    try:
        print(f"Words similar to '{word}':")
        for w, sim in model.wv.most_similar(word, topn=n):
            print(f"{w} ({sim:.2f})")
    except KeyError:
        print(f"'{word}' not found in vocabulary.")

find_similar("treatment")
find_similar("vaccine")
