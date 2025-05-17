import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import openai
import os
from dotenv import load_dotenv

# API KEY 
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Question embeding model upload
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
index = faiss.read_index("mental_health_index.faiss")
df = pd.read_csv("Mental_Health_FAQ.csv")

client = openai.OpenAI()

def answer_question(query: str) -> str:
    query_embedding = model.encode([query])
    query_embedding = normalize(query_embedding, norm='l2')
    
    top_k = 3
    distances, indices = index.search(query_embedding, top_k)
    
    # summing answers
    retrieved_answers = "\n".join([df.iloc[i]["Answers"] for i in indices[0]])
    
    # PROMPT Generated with Turkish Language
    prompt = f"""
Aşağıda bir kullanıcı sorusu ve ona karşılık gelen bilgi parçaları var.
Bu bilgilere dayanarak, doğal, açık ve yardımcı bir cevap ver. Ayrıca çok uzun cavaplar vererek müşteriyi sıkma. 
Yardımsever ol, müşteriyi anlamak için soruduğu sorularla ilişkili detaylı sorular sor.

Soru:
{query}

Bilgi:
{retrieved_answers}

Cevap:
"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Sen müşterilere psikolojik danışmanlık veren bir chatbot'sun."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
   
    return response.choices[0].message.content.strip()
