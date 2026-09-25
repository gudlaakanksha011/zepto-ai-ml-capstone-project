from fastapi import FastAPI
from .models import AskRequest, AskResponse
from .retrieval import build_index
from .graph import ask

app=FastAPI(title="Zepto Policy Support Assistant")

@app.on_event("startup")
def startup():
    build_index()

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    return ask(request.query)

if __name__=="__main__":
    build_index()
    print(ask("How long does delivery take?"))
    print(ask("What is the capital of France?"))
