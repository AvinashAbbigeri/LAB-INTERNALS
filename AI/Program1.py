import gensim.downloader as api

print("Loading pre-trained word vectors...")
wv = api.load("word2vec-google-news-300")

def explore_word_relationships(w1, w2, w3):
    try:
        res = wv.similar_by_vector(wv[w1] - wv[w2] + wv[w3], topn=10)
        print(f"\nWord Relationship: {w1} - {w2} + {w3}")
        print("Most similar words to the result (excluding input words):")
        for word, sim in [(w, s) for w, s in res if w not in (w1, w2, w3)][:5]:
            print(f"{word}: {sim:.4f}")
    except KeyError as e:
        print(f"Error: {e} not found in the vocabulary.")

def analyze_similarity(w1, w2):
    try:
        print(f"\nSimilarity between {w1} and {w2}: {wv.similarity(w1, w2):.4f}")
    except KeyError as e:
        print(f"Error: {e} not found in the vocabulary.")

def find_most_similar(w):
    try:
        print(f"\nMost similar words to '{w}':")
        for word, sim in wv.most_similar(w, topn=5):
            print(f"{word}: {sim:.4f}")
    except KeyError as e:
        print(f"Error: {e} not found in the vocabulary.")

explore_word_relationships("king", "man", "woman")
explore_word_relationships("paris", "france", "germany")
explore_word_relationships("apple", "fruit", "carrot")

analyze_similarity("cat", "dog")
analyze_similarity("computer", "keyboard")
analyze_similarity("music", "art")

find_most_similar("happy")
find_most_similar("sad")
find_most_similar("technology")
