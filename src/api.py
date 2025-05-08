from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your processed data
with open("web/data.json", "r") as f:
    data = json.load(f)

@app.get("/data")
def get_data():
    return data

@app.get("/countries")
def get_countries():
    return [country["name"] for country in data.get("countries", [])]

@app.get("/country/{country_name}")
def get_country(country_name: str):
    for country in data.get("countries", []):
        if country["name"].lower() == country_name.lower():
            return country
    return {"error": "Country not found"}

@app.get("/regions")
def get_regions():
    return data.get("regions", [])
