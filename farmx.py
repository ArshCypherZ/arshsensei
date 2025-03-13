from fastapi import APIRouter, UploadFile, File
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io
import os
from google import genai
from pydantic import BaseModel
import pickle
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

# =========================== 🌿 Plant Disease Prediction =========================== #

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load Image Model
preprocessor = AutoImageProcessor.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
model = AutoModelForImageClassification.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")

@router.post("/predict/")
async def predict_plant_disease(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    inputs = preprocessor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    result = model.config.id2label[predicted_class_idx]
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=f"{result}. Explain about this plant disease and give measures, preventives. Do not put up or say anything extra.",
    )    
    return {"disease": result, "info": response.text}


# =========================== 🌾 Farm Waste Volume Prediction =========================== #

# Load Trained Model and Features
with open('wastevol_model.pkl', 'rb') as model_file:
    waste_model = pickle.load(model_file)

with open('wastevol_model_features.pkl', 'rb') as feature_file:
    expected_features = pickle.load(feature_file)

# Rainfall Data
rainfall_data = {
    "Andaman and Nicobar Islands": 2967, "Andhra Pradesh": 900, "Arunachal Pradesh": 2782,
    "Assam": 2818, "Bihar": 1027, "Chandigarh": 1111, "Chhattisgarh": 1447,
    "Dadra and Nagar Haveli": 2500, "Daman and Diu": 1700, "Delhi": 792,
    "Goa": 3005, "Gujarat": 800, "Haryana": 509, "Himachal Pradesh": 1367,
    "Jammu and Kashmir": 1295, "Jharkhand": 1351, "Karnataka": 1500,
    "Kerala": 3055, "Ladakh": 100, "Lakshadweep": 1648, "Madhya Pradesh": 1144,
    "Maharashtra": 1700, "Manipur": 1467, "Meghalaya": 2818, "Mizoram": 2555,
    "Nagaland": 2034, "Odisha": 1525, "Pondicherry": 998, "Punjab": 506,
    "Rajasthan": 300, "Sikkim": 2739, "Tamil Nadu": 998, "Tripura": 2143,
    "Uttarakhand": 1742, "Uttar Pradesh": 900, "West Bengal": 1900,
}

# Input Schema
class FarmerInput(BaseModel):
    crop: str
    season: str
    state: str
    area: float  # in hectares
    fertilizer: float  # in kilograms
    pesticide: float  # in kilograms

@router.post("/predict_waste_volume/")
def predict_waste_volume(data: FarmerInput):
    # Determine annual rainfall based on state
    annual_rainfall = rainfall_data.get(data.state, 500)  # Default to 500 if no data

    # Prepare input data
    input_data = {
        'Crop': [data.crop], 'Season': [data.season], 'State': [data.state],
        'Area': [data.area], 'Annual_Rainfall': [annual_rainfall],
        'Fertilizer': [data.fertilizer], 'Pesticide': [data.pesticide]
    }
    input_df = pd.DataFrame(input_data)

    # One-hot encode categorical columns
    categorical_columns = ['Crop', 'Season', 'State']
    input_encoded = pd.get_dummies(input_df, columns=categorical_columns)

    # Align input features with model
    for feature in expected_features:
        if feature not in input_encoded:
            input_encoded[feature] = 0
    input_encoded = input_encoded[expected_features]

    # Predict yield and calculate waste
    predicted_yield = waste_model.predict(input_encoded)[0]
    total_yield = predicted_yield * data.area
    waste = (10.1 * total_yield) / 100

    return {"predicted_yield_per_unit_area": predicted_yield, "total_yield": total_yield, "waste": waste}
