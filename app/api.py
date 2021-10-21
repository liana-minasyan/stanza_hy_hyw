from fastapi import FastAPI
from fastapi import FastAPI
from starlette.requests import Request
import torch
from run_pipeline import run_east, run_west

app = FastAPI(
    title="TSA",
    version="1.0",
    description="Target-Based sentiment analysis"
)

@app.post("/api/east")
async def classify(request: Request):
    """Parsing textual data"""
    body = await request.body()
    data = body.decode("utf-8")
    try:
        result = run_east(data)
        return result
        print(result)
    except Exception as e:
        return('invalid data format')

@app.post("/api/west")
async def classify(request: Request):
    """Parsing textual data"""
    body = await request.body()
    data = body.decode("utf-8")
    try:
        result = run_west(data)
        return result
    except Exception as e:
        return('invalid data format')

        # GPU compatibility
        # data.to(device=device)