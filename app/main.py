from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import json
import os
from .rag_pipeline import call_llm
app = FastAPI()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if os.getenv("GROQ_API_KEY") is None:
    try:
        import winreg
        registry_key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Environment",
        )
        value, _ = winreg.QueryValueEx(registry_key, "GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = value
    except Exception as e:
        print("Could not read GROQ_API_KEY from registry:", e)
print(GROQ_API_KEY) 
print("GROQ KEY FOUND?", bool(GROQ_API_KEY))

@app.get("/")
def home():
    return {"message": "FastAPI working!"}

class Prompt(BaseModel):
    prompt: str
#https://unlubricant-ignacio-semidomestically.ngrok-free.dev -> http://localhost:8000 
@app.post("/generate")
def generate(data: Prompt):
    try:
        response = call_llm(data.prompt)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
