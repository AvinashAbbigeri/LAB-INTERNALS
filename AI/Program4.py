!pip install gensim transformers nitk torch torchvision torchaudio

import gensim.downloader as api
from transformers import pipeline
import nltk
import string
from nltk.tokenize import word_tokenize

nltk.download('punkt')

word_vectors = api.load("glove-wiki-gigaword-100")
generator = pipeline("text-generation", model="gpt2")

def replace_keyword(prompt, keyword, vectors, topn=1):
    def get_replacement(w):
        w_clean = w.lower().strip(string.punctuation)
        if w_clean == keyword.lower():
            try:
                return vectors.most_similar(w_clean, topn=topn)[0][0]
            except KeyError:
                pass
        return w
    words = word_tokenize(prompt)
    enriched = [get_replacement(w) for w in words]
    return " ".join(enriched)

original_prompt = "Who is king."
key_term = "king"
enriched_prompt = replace_keyword(original_prompt, key_term, word_vectors)

def generate(prompt):
    try:
        return generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    except Exception as e:
        print(f"Error: {e}")
        return None

print(f"\nOriginal Prompt: {original_prompt}")
print(f"\nEnriched Prompt: {enriched_prompt}")

orig_resp = generate(original_prompt)
enr_resp = generate(enriched_prompt)

print("\nOriginal Prompt Response:\n", orig_resp)
print("\nEnriched Prompt Response:\n", enr_resp)
print("\nComparison of Responses:")
print("Original Length:", len(orig_resp))
print("Enriched Length:", len(enr_resp))
print("Original Detail:", orig_resp.count("."))
print("Enriched Detail:", enr_resp.count("."))
