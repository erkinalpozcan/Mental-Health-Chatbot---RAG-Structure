import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

df = pd.read_csv("Mental_Health_FAQ.csv")
df.columns = df.columns.str.strip().str.lower()

print("Sütunlar:", df.columns)

texts = df["answers"].astype(str).tolist()

model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# embedings
embeddings = model.encode(texts, batch_size=16, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# normalization
embeddings_normalized = normalize(embeddings, norm='l2')

# vectoral db
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings_normalized)

faiss.write_index(index, "mental_health_index.faiss")

