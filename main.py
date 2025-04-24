# ======= Top Level: Imports and Setup =======
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pdfplumber
from PyPDF2 import PdfReader
import numpy as np
import pandas as pd
import requests
import pickle
import os
import re
import io
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

crop_model = None
label_encoder = None
fertilizer_model = None
soil_encoder = None
crop_encoder = None

fertilizer_map = {
    0: "10-26-26", 1: "14-35-14", 2: "17-17-17",
    3: "20-20", 4: "28-28", 5: "DAP", 6: "Urea",
}

# class CropInput(BaseModel):
#     N: float
#     P: float
#     K: float
#     temperature: float
#     humidity: float
#     ph: float
#     rainfall: float

@app.on_event("startup")
def load_models():
    global crop_model, label_encoder, fertilizer_model, soil_encoder, crop_encoder
    with open("models/CropRecommendation.pkl", "rb") as f:
        crop_model = pickle.load(f)
    with open("encoders/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("models/FertilizerRecommendation.pkl", "rb") as f:
        fertilizer_model = pickle.load(f)
    with open("encoders/soil_encoder_FR.pkl", "rb") as f:
        soil_encoder = pickle.load(f)
    with open("encoders/crop_encoder_FR.pkl", "rb") as f:
        crop_encoder = pickle.load(f)


def predict_crop_from_input(data):
    input_array = np.array([[ data["N"], data["P"], data["K"],
        data["temperature"], data["humidity"],
        data["ph"], data["rainfall"]]])
    prediction = crop_model.predict(input_array)
    return label_encoder.inverse_transform(prediction)[0]

def get_soil_data(content):
    pdf_file = io.BytesIO(content)
    extracted = {}
    with pdfplumber.open(pdf_file) as file:
        text = file.pages[0].extract_text()
        table = file.pages[0].extract_tables()[0]

        soil_type_match = re.search(r"Sample Description:\s*(.*) soil", text)
        extracted['soil_type'] = soil_type_match.group(1).strip() if soil_type_match else "Unknown"

        for row in table[1:]:
            param = row[1].strip()
            value = float(row[3].strip())
            if "pH" in param:
                extracted["pH"] = value
            elif "Nitrogen" in param:
                extracted["N"] = value
            elif "Phosphorus" in param:
                extracted["P"] = value
            elif "Potassium" in param:
                extracted["K"] = value
    return extracted


def get_weather(location):
    lat, lon = location
    api_url = f"http://api.weatherapi.com/v1/forecast.json?key=6bac00c67d4144e5ad2180607240809&q={lat},{lon}&days=1&aqi=no&alerts=no"
    res = requests.get(api_url)
    if res.status_code != 200:
        raise HTTPException(status_code=res.status_code, detail="Weather API failed")
    data = res.json()
    try:
        temperature = data["current"]["temp_c"]
        humidity = data["current"]["humidity"]
        moisture = data["current"]["precip_mm"]
        rainfall = data["forecast"]["forecastday"][0]["day"].get("totalprecip_mm", 0.0)
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Weather data format error: {e}")

    return {
        "Temperature": temperature,
        "Humidity": humidity,
        "Moisture": moisture,
        "Rainfall" : rainfall
    }


def preprocess_fertilizer_data(data):
    df = pd.DataFrame(data)
    df["Soil Type"] = soil_encoder.transform(df["Soil Type"])
    df["Crop Type"] = crop_encoder.transform(df["Crop Type"])

    features = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
    return df[features + ["Soil Type", "Crop Type"]].values.astype(float)

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.post("/crop-prediction/")
async def crop_prediction(
    file: UploadFile = File(...),
    lat: float = Query(...),
    lon: float = Query(...),):
    content = await file.read()
    soil_data = get_soil_data(content)
    weather = get_weather((lat, lon))
    #     N: float
    # P: float
    # K: float
    # temperature: float
    # humidity: float
    # ph: float
    # rainfall: float
    data = {
        "temperature": weather["Temperature"],
        "humidity": weather["Humidity"],
        "rainfall": weather["Rainfall"],
        "N": soil_data["N"],
        "P": soil_data["K"],
        "K": soil_data["P"],
        "ph" : soil_data["pH"]
    }
    crop = predict_crop_from_input(data)
    return {"recommended_crop": crop}


@app.post("/fertilizer-prediction/")
async def fertilizer_prediction(
    file: UploadFile = File(...),
    lat: float = Query(...),
    lon: float = Query(...),
    crop_type: str = Query(...)
):
    content = await file.read()
    soil_data = get_soil_data(content)
    weather = get_weather((lat, lon))

    data = {
        "Temperature": [weather["Temperature"]],
        "Humidity": [weather["Humidity"]],
        "Moisture": [weather["Moisture"]],
        "Soil Type": [soil_data["soil_type"]],
        "Crop Type": [crop_type],
        "Nitrogen": [soil_data["N"]],
        "Potassium": [soil_data["K"]],
        "Phosphorous": [soil_data["P"]],
    }

    processed = preprocess_fertilizer_data(data)
    pred = fertilizer_model.predict(processed)[0]
    return {"recommended_fertilizer": fertilizer_map[pred]}

@app.post("/fertilizer-prediction-64/")
async def fertilizer_prediction_base64(
    base64_pdf: str = Body(...),
    lat: float = Query(...),
    lon: float = Query(...),
    crop_type: str = Query(...)
):
    try:
        if "," in base64_pdf:
            base64_pdf = base64_pdf.split(",")[1]
        pdf_bytes = b64decode(base64_pdf)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF string")

    soil_data = get_soil_data(pdf_bytes)
    weather = get_weather((lat, lon))

    data = {
        "Temperature": [weather["Temperature"]],
        "Humidity": [weather["Humidity"]],
        "Moisture": [weather["Moisture"]],
        "Soil Type": [soil_data["soil_type"]],
        "Crop Type": [crop_type],
        "Nitrogen": [soil_data["N"]],
        "Potassium": [soil_data["K"]],
        "Phosphorous": [soil_data["P"]],
    }

    processed = preprocess_fertilizer_data(data)
    pred = fertilizer_model.predict(processed)[0]
    return {"recommended_fertilizer": fertilizer_map[pred]}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.pdf", "wb") as f:
        f.write(contents)
    reader = PdfReader("temp.pdf")
    text = "".join([page.extract_text() or "" for page in reader.pages])
    return {"filename": file.filename, "content": text[:500]}
