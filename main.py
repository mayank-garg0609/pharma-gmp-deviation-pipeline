import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from src.gmp_dev_generator import deviation_generation
from src.brainstorming import brain
from src.add_content import add_data


app = FastAPI(
    title="GMP Deviation Brainstorming API",
    description="API for GMP deviation ingestion and expert brainstorming",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "https://report-companion.vercel.app",
                  "https://happy-field-05ba9fc00.4.azurestaticapps.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BrainstormingRequest(BaseModel):
    data: Dict[str, Any]


class AddDataRequest(BaseModel):
    data: Dict[str, Any]

class GMPResponse(BaseModel):
    data: Dict[str, Any]

@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "GMP Brainstorming API is live"
    }
@app.post("/brainstorming")
def run_brainstorming(request: BrainstormingRequest):
    try:
        result = brain(request.data)
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.post("/adddata")
def ingest_deviation(request: AddDataRequest):
    try:
        response = add_data(request.data)
        return {
            "status": "success",
            "response": response
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.post("/gmpgeneration")
def generate_gmp_deviation(request: GMPResponse):
    try:
        result = deviation_generation(request.data)
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0",  port=int(os.environ.get("PORT", 3000)))

