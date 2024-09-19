from fastapi import FastAPI, Request, HTTPException
from transformer import Model  # Assuming your custom transformer model
import json

app = FastAPI()

model = Model()
model.loadModel("paraphrase-MiniLM-L3-v2")

@app.post("/vectorEmbeddings")
async def generate_embeddings(request: Request):
    try:
        json_data = await request.json()
        data_str = json.dumps(json_data)  # basic json to string conversion
        print(data_str)
        return {'embed': model.getEmbeddings([data_str]).tolist()[0]}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")

@app.get("/")
async def generate_embeddings():
    return {"status": "successful"}