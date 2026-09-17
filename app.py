from fastapi import FastAPI, Request, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn
from fastapi import FastAPI
import json
import os
from dotenv import load_dotenv
from QASystem.retrieval_generation import get_result

#loading the environment variable
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
print("Import Successfully")

#creating the app
app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")

#creating the routes with bind functions
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )
    
@app.post("/get_answer")
async def get_answer(request: Request, question: str = Form(...)):
    print(question)
    answer= get_result(question)
    print(answer)
    return {
        "answer": answer
    }
    
if __name__ == "__main__":
    uvicorn.run("app:app",host="localhost",port=8000,reload=True)