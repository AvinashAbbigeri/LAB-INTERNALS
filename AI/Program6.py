from transformers import pipeline

analyze = pipeline("sentiment-analysis")
reviews = [
    "The product is amazing! I love it so much.",
    "I'm very disappointed. The service was terrible.",
    "It was an average experience, nothing special.",
    "Absolutely fantastic quality! Highly recommended.",
    "Not great, but not the worst either."
]

print("\nCustomer Sentiment Analysis Results:")
for text in reviews:
    res = analyze(text)[0]
    print(f"\nInput Text: {text}")
    print(f"Sentiment: {res['label']} (Confidence: {res['score']:.4f})\n")
