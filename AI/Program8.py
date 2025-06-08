from langchain import PromptTemplate
from langchain.llms import Cohere
import getpass

try:
    text_content = open("Teaching.txt", encoding="utf-8").read()
    print("File loaded successfully!")
except Exception as e:
    print("Error loading file:", e)

cohere_llm = Cohere(cohere_api_key=getpass.getpass("Enter your Cohere API Key:"), model="command")

template = """
You are an AI assistant helping to summarize and analyze a text document.
Here is the document content:
{text}
Summary:
- Provide a concise summary of the document.
Key Takeaways:
List 3 important points from the text.
Sentiment Analysis:
- Determine if the sentiment of the document is Positive, Negative, or Neutral.
"""

response = cohere_llm.predict(PromptTemplate(input_variables=["text"], template=template).format(text=text_content))

print("\n**Formatted Output**")
print(response)
