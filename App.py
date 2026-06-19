import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import OpenAI

load_dotenv()

# -------------------------------
# GROQ CLIENT SETUP
# -------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY environment variable not set. Please set it first.")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# -------------------------------
# CHROMA DB (PERSISTENT)
# -------------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# -------------------------------
# BETTER EMBEDDINGS
# -------------------------------

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="enterprise_docs",
    embedding_function=embedding_function
)

# -------------------------------
# TEXT CHUNKING FUNCTION
# -------------------------------

def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

# -------------------------------
# DATA INGESTION (ONLY ONCE)
# -------------------------------

if collection.count() == 0:

    print("📥 Ingesting data...")

    data_folder = "data"
    doc_id = 0

    for file_name in os.listdir(data_folder):

        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(data_folder, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)

        for chunk in chunks:
            doc_id += 1

            collection.add(
                documents=[chunk],
                ids=[str(doc_id)],
                metadatas=[{
                    "source": file_name,
                    "category": file_name.split(".")[0]
                }]
            )


    print("✅ Data ingestion complete!")

else:
    print("⚡ Using existing database (fast mode)")

# -------------------------------
# QUERY INPUT
# -------------------------------

query = input("\n💬 Ask your question: ")

# -------------------------------
# RETRIEVE
# -------------------------------

results = collection.query(
    query_texts=[query],
    n_results=3
)

print("\n🔍 Retrieved Chunks:\n")

for i, doc in enumerate(results["documents"][0]):
    print(f"{i+1}. {doc}\n")

# -------------------------------
# CONTEXT
# -------------------------------

context = " ".join(results["documents"][0])

# -------------------------------
# PROMPT
# -------------------------------

prompt = f"""
You are an HR assistant.

Answer clearly and concisely using the context.

You may use logical reasoning if needed.

If the answer cannot be determined, say:
"Not mentioned in policy"

Context:
{context}

Question:
{query}
"""

# -------------------------------
# LLM CALL (GROQ)
# -------------------------------

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a helpful HR assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

# -------------------------------
# FINAL OUTPUT
# -------------------------------

print("\n🤖 AI Answer:\n")
print(response.choices[0].message.content)