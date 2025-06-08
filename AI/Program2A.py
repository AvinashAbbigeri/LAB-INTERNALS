import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

print("Loading pre-trained word vectors...")
wv = api.load("word2vec-google-news-300")

def explore_word_relationships(w1, w2, w3):
    try:
        res = wv[w1] - wv[w2] + wv[w3]
        sim = [(w, s) for w, s in wv.similar_by_vector(res, topn=10) if w not in (w1, w2, w3)]
        print(f"\nWord Relationship: {w1} - {w2} + {w3}\nMost similar words (excluding input):")
        for w, s in sim[:5]: print(f"{w}: {s:.4f}")
        return sim
    except KeyError as e:
        print(f"Error: {e} not in vocabulary."); return []

def visualize(words, method='pca'):
    reducer = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, random_state=42, perplexity=3)
    vecs = np.array([wv[w] for w in words])
    reduced = reducer.fit_transform(vecs)
    plt.figure(figsize=(10,8))
    for i, w in enumerate(words):
        plt.scatter(*reduced[i], color='blue')
        plt.text(reduced[i,0]+.02, reduced[i,1]+.02, w, fontsize=12)
    plt.title(f"Word Embeddings Visualization using {method.upper()}")
    plt.xlabel("Component 1"); plt.ylabel("Component 2"); plt.grid(True); plt.show()

words = ["king", "man", "woman", "queen", "prince", "princess", "royal", "throne"]
filtered = explore_word_relationships("king", "man", "woman")
all_words = words + [w for w, _ in filtered]
visualize(all_words, 'pca')
visualize(all_words, 'tsne')
