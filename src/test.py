import os
import sys
import requests
import json
from dotenv import load_dotenv, find_dotenv

def get_api_key():
    # Try to find and load the .env file
    dotenv_path = find_dotenv()
    if not dotenv_path:
        raise FileNotFoundError(".env file not found. Please create one with your API key.")
    
    load_dotenv(dotenv_path, override=True)
    api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    
    if not api_key:
        raise ValueError("API key not found in .env file. Please set OPENWEATHERMAP_API_KEY.")
    
    return api_key

def test_solar_radiation_api(lat, lon, api_key):
    solar_url = "https://api.openweathermap.org/data/2.5/solar_radiation/forecast"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key
    }
    try:
        response = requests.get(solar_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 401:
            print(f"Error 401: Unauthorized. Please check your API key.")
        elif response.status_code == 404:
            print(f"Error 404: API endpoint not found. Please check the URL.")
        else:
            print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError:
        print("Error: Unable to connect to the API. Please check your internet connection.")
    except requests.exceptions.Timeout:
        print("Error: API request timed out. Please try again later.")
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}")
    return None

def display_solar_data(data):
    if not data:
        return
    
    print("\nSolar Radiation Forecast Data:")
    print(json.dumps(data, indent=2))
    
    if 'list' in data:
        print("\nSummary:")
        for item in data['list'][:5]:  # Display first 5 entries
            dt = item.get('dt', 'N/A')
            ghi = item.get('radiation', {}).get('ghi', 'N/A')
            print(f"Timestamp: {dt}, Global Horizontal Irradiance: {ghi} W/m²")

def main():
    try:
        api_key = get_api_key()
        print(f"API Key loaded successfully: {api_key[:5]}...{api_key[-5:]}")
        
        lat, lon = 41.089075, -100.773775
        print(f"\nTesting Solar Radiation API for coordinates: {lat}, {lon}")
        
        solar_data = test_solar_radiation_api(lat, lon, api_key)
        if solar_data:
            display_solar_data(solar_data)
        else:
            print("Failed to retrieve solar radiation data.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()