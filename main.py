from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
import pandas
import pdfplumber
import uvicorn
from PyPDF2 import PdfReader
import pandas as pd
import io
from pydantic import BaseModel
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import re
from dotenv import load_dotenv 
import pickle
import os
import requests
import base64

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
fertilizer_map = {
    0: "10-26-26",
    1: "14-35-14",
    2: "17-17-17",
    3: "20-20",
    4: "28-28",
    5: "DAP",
    6: "Urea",
}

class PDFRequest(BaseModel):
    base64_pdf: str


class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

def get_soil_data(content):
    pdf_file = io.BytesIO(content)
    extracted_values = dict()
    with pdfplumber.open(pdf_file) as file:
      table = file.pages[0].extract_tables()[0]
      text = file.pages[0].extract_text()
      # print("Extracted Text:\n", text)

      soil_type_match = re.search(r"Sample Description:\s*(.*) soil", text)
      soil_type = soil_type_match.group(1).strip() if soil_type_match else "Not Found"
      extracted_values['soil_type'] = soil_type

    #   print("\nExtracted Soil Type:", soil_type)


      # print(f"table : {table} tables : {len(table)} tablel[] : {table[0][:]}")
      headers = table[0]
      data = table[1:]
      # print(f"headers : {headers} data : {data}")

      for row in data:
          parameter = row[1].strip()
          value = float(row[3].strip())
          if "pH" in parameter:
              extracted_values["pH"] = value
          elif "Available Nitrogen" in parameter:
              extracted_values["N"] = value
          elif "Available Phosphorus" in parameter:
              extracted_values["P"] = value
          elif "Available Potassium" in parameter:
              extracted_values["K"] = value
    # print("\nExtracted Soil Data:")
    # print(f"pH: {extracted_values.get('pH', 'Not Found')}")
    # print(f"Nitrogen (N): {extracted_values.get('Nitrogen (N)', 'Not Found')} ppm")
    # print(f"Phosphorus (P): {extracted_values.get('Phosphorus (P)', 'Not Found')} ppm")
    # print(f"Potassium (K): {extracted_values.get('Potassium (K)', 'Not Found')} ppm")
    return extracted_values


def get_weather(location):
    api_url = f"http://api.weatherapi.com/v1/forecast.json?key={os.getenv('WEATHER_MAP_API')}&q={location[0]},{location[1]}&days=1&aqi=no&alerts=no"
    print(api_url)
    response = requests.get(api_url)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching weather data")
    weather_data = response.json()
    current_weather = weather_data.get('current', {})
    temperature = current_weather.get('temp_c', None)
    humidity = current_weather.get('humidity', None)
    precipitation = current_weather.get('precip_mm', None)
    data = {
        'Temperature': temperature,
        'Humidity': humidity,
        'Moisture': precipitation
    }
    print(data)
    return data

# print(get_weather("Coimbatore"))

def load_cr_model():
  with open("models/CropRecommendation.pkl", "rb") as model_file:
    model = pickle.load(model_file, fix_imports=True)
  with open("models\label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
    return (model, label_encoder)


def preprocess_data(data):
    df = pd.DataFrame(data)


    with open("encoders/soil_encoder_FR.pkl", "rb") as file:
        encode_soil = pickle.load(file)

    with open("encoders/crop_encoder_FR.pkl", "rb") as file:
        encode_crop = pickle.load(file)

    df["Soil Type"] = encode_soil.transform(df["Soil Type"])
    df["Crop Type"] = encode_crop.transform(df["Crop Type"])

    numeric_features = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]

    transformed_data = df[numeric_features + ["Soil Type", "Crop Type"]].values.astype(float)

    print(f"Final Preprocessed Data:\n{transformed_data}")

    return transformed_data

def get_fertilizer_prediction(data):
    with open("models/FertilizerRecommendation.pkl", "rb") as model_file:
        model = pickle.load(model_file)
        preprocessed_data = preprocess_data(data)
        prediction = model.predict(preprocessed_data)
        print("Recommended Fertilizer:", prediction[0])
    return prediction[0]


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()

    # Save it temporarily
    with open("temp.pdf", "wb") as f:
        f.write(contents)

    # Extract PDF text
    reader = PdfReader("temp.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    return {"filename": file.filename, "content": text[:500]}  # Just return first 500 chars

@app.get("/")
def home():
    return {"message": "As you can see, I'm Not Dead!"}


@app.post("/crop-prediction")
def predict_crop(data: CropInput):
    model, label_encoder = load_cr_model()
    input_array = np.array([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])
    prediction = model.predict(input_array)
    crop_name = label_encoder.inverse_transform(prediction)[0]
    return {"recommended_crop": crop_name}

@app.post("/fertilizer-prediction/")
async def predict_fertilizer(
    file : UploadFile = File(...),
    lat: float = Query(..., description="Latitude of the location"),
    lon: float = Query(..., description="Longitude of the location"),
    crop_type: str = Query(..., description="Type of the crop")
    ):

    # print(f"file : {file} lat : {lat} lon: {lon} crop : {crop_type}")
    content = await file.read()
    soil_data = get_soil_data(content)
    print(f"soil data: {soil_data}")

    soil_type = soil_data["soil_type"]
    n = soil_data["N"]
    p = soil_data["P"]
    k = soil_data["K"]

    weather_data = get_weather([lat, lon])
    t = weather_data['Temperature']
    h = weather_data['Humidity']
    m = weather_data['Moisture']

    data = {
    "Temperature": [t],
    "Humidity": [h],
    "Moisture": [m],
    "Soil Type": [soil_type],
    "Crop Type": [crop_type],
    "Nitrogen": [n],
    "Potassium": [k],
    "Phosphorous": [p],
    }
    print(f"data : {data}")
    # df = pd.DataFrame(data)
    prediction = fertilizer_map[get_fertilizer_prediction(data)]
    print("Prediction : ",prediction)
    return JSONResponse(content={"Response": "Function completed","FR":prediction})
