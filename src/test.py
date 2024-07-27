import os
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')

def test_solar_radiation_api(lat, lon):
    solar_url = f"https://api.openweathermap.org/data/2.5/solar_radiation/forecast"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': API_KEY
    }
    response = requests.get(solar_url, params=params)
    if response.status_code == 200:
        print("Solar radiation forecast data retrieved successfully.")
        print(response.json())
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    lat, lon = 41.089075, -100.773775
    test_solar_radiation_api(lat, lon)
