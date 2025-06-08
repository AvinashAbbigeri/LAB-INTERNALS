from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    text = " ".join(text.split())
    configs = [
        {"do_sample": False},
        {"do_sample": True, "temperature": 0.9},
        {"do_sample": False, "num_beams": 5},
        {"do_sample": True, "top_k": 50, "top_p": 0.95}
    ]
    summaries = [summarizer(text, max_length=150, min_length=30, **cfg)[0]['summary_text'] for cfg in configs]
    print("\nOriginal Text:\n", text)
    print("\nSummarized Text:")
    print("\nDefault:", summaries[0])
    print("\nHigh randomness:", summaries[1])
    print("\nConservative:", summaries[2])
    print("\nDiverse sampling:", summaries[3])

long_text = """
Artificial Intelligence (AI) is a rapidly evolving field of computer science focused on creating intelligent machines capable of mimicking human cognitive functions such as learning, problem-solving, and decision-making. In recent years, AI has significantly impacted various industries, including healthcare, finance, education, and entertainment. AI-powered applications, such as chatbots, self-driving cars, and recommendation systems, have transformed the way we interact with technology. Machine learning and deep learning, subsets of AI, enable systems to learn from data and improve over time without explicit programming.
However, AI also poses ethical challenges, such as bias in decision-making and concerns over job displacement. As AI technology continues to advance, it is crucial to balance innovation with ethical considerations to ensure its responsible development and deployment.
"""

summarize_text(long_text)
