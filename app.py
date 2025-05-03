from fastapi import FastAPI, Request
import uvicorn
import sys
import os
import pandas as pd
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response, HTMLResponse
from src.textSummarizer.pipeline.stage_5_prediction_pipeline import PredictionPipeline

text: str = "What is Text Summarization?"

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # <== Make sure this folder exists

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict_route(text: str):
    try:
        obj = PredictionPipeline()
        prediction = obj.predict(text)
        return {"summary": prediction}
    except Exception as e:
        return {"error": str(e)}


@app.get("/metrics", response_class=HTMLResponse)
async def display_metrics(request: Request):
    try:
        df = pd.read_csv("artifacts/model_evaluation/metrics.csv")
        table_html = df.to_html(classes="table table-striped", index=False)
        return templates.TemplateResponse("metrics.html", {"request": request, "table_html": table_html})
    except Exception as e:
        return Response(f"Could not read metrics.csv: {e}")
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
