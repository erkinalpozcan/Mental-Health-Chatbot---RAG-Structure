# Mental-Health-Chatbot - RAG-Structure

🧠 Mental Health Assistant — RAG Powered AI Chatbot
This project allows you to develop an AI-powered assistant that can answer questions about mental health, working with the Retrieval-Augmented Generation (RAG) architecture. The system interprets the question from the user, finds the most appropriate pieces of information with vector search, and produces a natural and helpful response with OpenAI GPT-3.5 Turbo.

🔧 Technologies Used:

- 🧠 OpenAI GPT-3.5 Turbo (natural language response)
- 🔍 FAISS (document search with vector similarity)
- 🗂️ SentenceTransformers (embedding generation)
- ⚡ FastAPI (API development)
- 🐍 Python 3.10+

# 📁 Project Design
  
├── main.py

├── rag_structure.py

├── embedding.py

├── Mental_Health_FAQ.csv

├── mental_health_index.faiss 

└── .env   

### 1. Step: Install All the Requirements
`pip install -r requirements.txt`

### 2. Step: Add your OpenAI and HuggingFace API keys in .env file

### 3. Step: RUN embedding.py Python file to make a faiss file. (Just one run is enough!)

### 4. Step: Start your FastAPI app
write this code in bash
`uvicorn main:app --reload`

### Last Step: Try it out in your browser
go to -> http://127.0.0.1:8000/docs

## Features
- Turkish Q&A support
- Data-based knowledge-based answers with health content
- Searching and combining the best 3 answers to a user's question with a vector
- Creating contextual answers with GPT


