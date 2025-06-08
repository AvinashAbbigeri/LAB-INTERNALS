import fitz, os
from langchain.llms import Cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from sentence_transformers import SentenceTransformer

# Extract text from PDF
with fitz.open("ipc.pdf") as pdf:
    ipc_text = "".join(page.get_text() for page in pdf)
with open('IPC_text.txt', 'w', encoding="utf-8") as f:
    f.write(ipc_text)
print("Text extracted and saved!")

# Split text and create embeddings
texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(ipc_text)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, convert_to_tensor=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.cpu().numpy())

# LLM setup
os.environ["COHERE_API_KEY"] = "YrqBdTypjvdKMc7bB1jLwihs5TS54JCN8qjrLVQ5"
llm = Cohere(model="command-xlarge-nightly", temperature=0.7)

def get_chat_response(query):
    D, I = index.search(model.encode([query], convert_to_tensor=True).cpu().numpy(), k=1)
    prompt = f"""
The user has asked a question related to the Indian Penal Code.
Below is the relevant section from the Indian Penal Code:
{texts[I[0][0]]}
The user's question: {query}
Please provide an answer based on the above IPC section.
"""
    return llm(prompt)

response = get_chat_response(input("Ask a question about the Indian Penal Code: "))
print(f"Chatbot Response: {response}")
