import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

word_vectors = api.load("word2vec-google-news-300")
domain_words = ["computer", "software", "hardware", "algorithm", "data", "network", "programming", "machine", "learning", "artificial"]
domain_vectors = np.array([word_vectors[w] for w in domain_words])

def visualize(words, vectors, method='pca', perplexity=5):
    reducer = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced = reducer.fit_transform(vectors)
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        plt.scatter(*reduced[i], color="blue")
        plt.text(reduced[i,0]+0.02, reduced[i,1]+0.02, word, fontsize=12)
    plt.title(f"Word Embeddings Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()

visualize(domain_words, domain_vectors, 'pca')
visualize(domain_words, domain_vectors, 'tsne', perplexity=3)

def similar_words(word):
    try:
        print(f"\nTop 5 semantically similar words to '{word}':")
        for w, s in word_vectors.most_similar(word, topn=5):
            print(f"{w}: {s:.4f}")
    except KeyError:
        print(f"Error: '{word}' not found in the vocabulary.")

similar_words("computer")
similar_words("learning")
