import random
import warnings
from pathlib import Path
from typing import Literal

warnings.filterwarnings('ignore')

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
if not (BASE_DIR / 'Mental_Health_Model.pkl').exists() and (BASE_DIR.parent / 'Mental_Health_Model.pkl').exists():
    BASE_DIR = BASE_DIR.parent

MODEL_PATH = BASE_DIR / 'Mental_Health_Model.pkl'
DATASET_PATH = BASE_DIR / 'Student Social Media And Mental Health Impact.csv'

# Load trained pipeline
model = joblib.load(MODEL_PATH)

# Load dataset for integration, validation, and sample inspection
df_dataset = None
if DATASET_PATH.exists():
    try:
        df_dataset = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f"Warning: Failed to load dataset: {e}")

top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']

app = FastAPI(
    title="Mental Health Score Predictor",
    description="Machine learning API predicting student mental health score based on digital habits and lifestyle.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic schema matching the training features
class StudentData(BaseModel):
    age                     : int = Field(..., ge=10, le=100, description="Age in years (10-100)")
    gender                  : Literal['Male', 'Female']
    country                 : str
    academic_level          : Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform      : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter', 'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat']
    purpose_of_use          : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours   : float = Field(..., ge=0, le=24)
    daily_unlocks           : int   = Field(..., ge=0)
    study_hours             : float = Field(..., ge=0, le=24)
    physical_activity_hours : float = Field(..., ge=0, le=24)
    sleep_hours_per_night   : float = Field(..., ge=0, le=24)
    stress_level            : Literal['Medium', 'Low', 'Very High', 'High']


class PredictionResponse(BaseModel):
    predicted_mental_health_score: float
    band: str = ""
    context: str = ""


def get_band_info(score: float):
    if score < 4.0:
        return "Signal: strained", "Your responses suggest elevated strain right now. Small shifts in sleep or screen time can go a long way."
    elif score < 7.0:
        return "Signal: balanced", "Your rhythm looks fairly steady, with some room to recover and reset."
    else:
        return "Signal: strong", "Your habits point to a well-supported, resilient baseline. Keep it up."


@app.get('/')
def serve_index():
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Mental Health Score Predictor API is running", "status": "ok"}


@app.get('/style.css')
def serve_css():
    css_file = BASE_DIR / "style.css"
    if css_file.exists():
        return FileResponse(css_file, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")


@app.get('/script.js')
def serve_js():
    js_file = BASE_DIR / "script.js"
    if js_file.exists():
        return FileResponse(js_file, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="script.js not found")


@app.get('/api/health')
@app.get('/health')
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "dataset_loaded": df_dataset is not None,
        "total_records": len(df_dataset) if df_dataset is not None else 0,
        "metrics": {
            "r2_score": 0.9500,
            "mae": 0.1897,
            "rmse": 0.2860
        }
    }


@app.get('/api/sample-student')
@app.get('/sample-student')
def sample_student():
    if df_dataset is None or len(df_dataset) == 0:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
    
    idx = random.randint(0, len(df_dataset) - 1)
    row = df_dataset.iloc[idx].to_dict()
    
    return {
        "index": int(idx + 1),
        "age": int(row['Age']),
        "gender": str(row['Gender']),
        "country": str(row['Country']),
        "academic_level": str(row['Academic_Level']),
        "most_used_platform": str(row['Most_Used_Platform']),
        "purpose_of_use": str(row['Purpose_Of_Use']),
        "avg_daily_usage_hours": float(row['Avg_Daily_Usage_Hours']),
        "daily_unlocks": int(row['Daily_Unlocks']),
        "study_hours": float(row['Study_Hours']),
        "physical_activity_hours": float(row['Physical_Activity_Hours']),
        "sleep_hours_per_night": float(row['Sleep_Hours_Per_Night']),
        "stress_level": str(row['Stress_Level']),
        "actual_mental_health_score": round(float(row['Mental_Health_Score']), 2)
    }


@app.post('/predict', response_model=PredictionResponse)
@app.post('/api/predict', response_model=PredictionResponse)
def predict(data: StudentData):
    country_group = data.country if data.country in top_countries else "Other"

    input_row = pd.DataFrame([{
        'Age'                       : data.age,
        'Gender'                    : data.gender,
        'Country'                   : data.country,
        'Academic_Level'            : data.academic_level,
        'Most_Used_Platform'        : data.most_used_platform,
        'Purpose_Of_Use'            : data.purpose_of_use,
        'Avg_Daily_Usage_Hours'     : data.avg_daily_usage_hours,
        'Daily_Unlocks'             : data.daily_unlocks,
        'Study_Hours'               : data.study_hours,
        'Physical_Activity_Hours'   : data.physical_activity_hours,
        'Sleep_Hours_Per_Night'     : data.sleep_hours_per_night,
        'Stress_Level'              : data.stress_level,
        'Grouped_country'           : country_group
    }])

    pred = float(model.predict(input_row)[0])
    score = round(max(0.0, min(10.0, pred)), 2)
    band, context = get_band_info(score)

    return PredictionResponse(
        predicted_mental_health_score=score,
        band=band,
        context=context
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)