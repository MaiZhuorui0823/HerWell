import os
import fitz  # pymupdf
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# ── Config ──
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("medical_knowledge")

# ── Helper: split text into chunks ──
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ── Helper: get embedding from OpenAI ──
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# ── Load PDF ──
def load_pdf(filepath, source, category):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text, source, category

# ── Load TXT ──
def load_txt(filepath, source, category):
    with open(filepath, encoding="utf-8") as f:
        text = f.read()
    return text, source, category

# ── All documents ──
documents = [
    # ACOG
    ("docs/acog/Dysmenorrhea_ Painful Periods _ ACOG.pdf",                    "acog", "dysmenorrhea"),
    ("docs/acog/Painful Periods _ ACOG.pdf",                                  "acog", "dysmenorrhea"),
    ("docs/acog/Abnormal Uterine Bleeding _ ACOG.pdf",                        "acog", "abnormal_bleeding"),
    ("docs/acog/Heavy Menstrual Bleeding _ ACOG.pdf",                         "acog", "abnormal_bleeding"),
    ("docs/acog/Polycystic Ovary Syndrome (PCOS) _ ACOG.pdf",                 "acog", "pcos"),
    ("docs/acog/Endometriosis _ ACOG.pdf",                                    "acog", "endometriosis"),
    ("docs/acog/PFSI033 Menstruation Ovulation and How Pregnancy Occurs.pdf", "acog", "general"),
    # ESHRE
    ("docs/eshre/Evidence-Based-Guidelines-2023.pdf",                         "eshre", "pcos"),
    # MedlinePlus
    ("docs/medlineplus/menstruation.txt",  "medlineplus", "general"),
    ("docs/medlineplus/dysmenorrhea.txt",  "medlineplus", "dysmenorrhea"),
    ("docs/medlineplus/amenorrhea.txt",    "medlineplus", "amenorrhea"),
    ("docs/medlineplus/pcos.txt",          "medlineplus", "pcos"),
    # Asian population (Plan C)
    ("docs/asian_population/pcos_genome_meta_analysis.pdf",         "asian_population", "pcos"),
    ("docs/asian_population/pcos_bmi_singapore_women.pdf",          "asian_population", "pcos"),
    ("docs/asian_population/pcos_criteria_phenotypes_ethnicity.pdf","asian_population", "pcos"),
    ("docs/asian_population/pcos_east_asian_phenotype_2023.pdf",    "asian_population", "pcos"),
]

# ── Main: process and store ──
total_chunks = 0

for item in documents:
    filepath, source, category = item
    print(f"\nProcessing: {filepath}")

    if not os.path.exists(filepath):
        print(f"  ⚠️  File not found, skipping.")
        continue

    if filepath.endswith(".pdf"):
        text, source, category = load_pdf(filepath, source, category)
    else:
        text, source, category = load_txt(filepath, source, category)

    chunks = chunk_text(text)
    print(f"  → {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 50:
            continue

        embedding = get_embedding(chunk)
        doc_id = f"{source}_{category}_{os.path.basename(filepath)}_{i}"

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": source, "category": category, "filename": os.path.basename(filepath)}]
        )
        total_chunks += 1

    print(f"  Done")

print(f"\nAll done. Total chunks stored: {total_chunks}")