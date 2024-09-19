from fastapi import FastAPI, Request, HTTPException
from transformer import Model
import json
import time

app = FastAPI()

model = Model()
model.loadModel("paraphrase-MiniLM-L3-v2")

@app.post("/vectorEmbeddings")
async def generate_embeddings(request: Request):
    try:
        json_data = await request.json()
        data_str = json.dumps(json_data)  # basic json to string conversion

        t1 = time.time()
        embd =  model.getEmbeddings([data_str]).tolist()[0]
        t2 = time.time()

        send_data = json_data
        send_data['vector'] = embd
        send_data['inferenceTime'] = t2-t1
        return send_data

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")

@app.get("/")
async def generate_embeddings():
    return {"connection": "successful"}