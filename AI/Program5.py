import gensim.downloader as api
import random
import nltk

nltk.download('punkt')
word_vectors = api.load("glove-wiki-gigaword-100")

def get_similar_words(seed, n=5):
    try:
        return [w for w, _ in word_vectors.most_similar(seed, topn=n)]
    except KeyError:
        print(f"'{seed}' not found in vocabulary. Try another word.")
        return []

def generate_paragraph(seed):
    words = get_similar_words(seed)
    if not words: return "Could not generate a paragraph. Try another seed word."
    templates = [
        f"The {seed} was surrounded by {words[0]} and {words[1]}.",
        f"People often associate {seed} with {words[2]} and {words[3]}.",
        f"In the land of {seed}, {words[4]} was a common sight.",
        f"A story about {seed} would be incomplete without {words[1]} and {words[0]}."
    ]
    return " ".join(random.choices(templates, k=4))

seed = input("Enter a seed word: ")
print("\nGenerated Paragraph:\n")
print(generate_paragraph(seed))
