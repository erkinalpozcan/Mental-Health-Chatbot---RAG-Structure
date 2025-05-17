from fastapi import FastAPI
from pydantic import BaseModel
from RAG_structure import answer_question

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    answer = answer_question(query.question)
    answer = answer.replace("\n", " ")
    return {"answer": answer}
