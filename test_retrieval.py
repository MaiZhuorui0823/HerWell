import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("medical_knowledge")

print(f"Total chunks: {collection.count()}")

def get_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

query = "dysmenorrhea pain management"
embedding = get_embedding(query)

results = collection.query(
    query_embeddings=[embedding],
    n_results=3
)

print(f"\nTop 3 results for '{query}':")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\n[{i+1}] Source: {meta['source']} | Category: {meta['category']}")
    print(f"     {doc[:150]}...")