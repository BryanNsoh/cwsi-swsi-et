import os
import requests
import pandas as pd
import numpy as np
import pyet
import logging
from datetime import datetime
import pytz
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenWeatherMap API details
BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Metadata for ET calculation
ELEVATION = 876  # meters
LATITUDE = 41.15  # degrees
LONGITUDE = -100.77  # degrees
WIND_HEIGHT = 3  # meters

def get_forecast(lat, lon):
    params = {
        'lat': lat,
        'lon': lon,
        'appid': API_KEY,
        'units': 'metric'
    }
    logger.info(f"Requesting forecast data for coordinates: {lat}, {lon}")
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    logger.info("Successfully retrieved forecast data")
    return response.json()

def get_solar_radiation_forecast(lat, lon):
    solar_url = f"https://api.openweathermap.org/data/2.5/solar_radiation/forecast"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': API_KEY
    }
    logger.info(f"Requesting solar radiation forecast data for coordinates: {lat}, {lon}")
    response = requests.get(solar_url, params=params)
    response.raise_for_status()
    logger.info("Successfully retrieved solar radiation forecast data")
    return response.json()

def map_forecast_data(forecast_item, collection_time):
    return {
        'collection_time': collection_time.isoformat(),
        'TIMESTAMP': datetime.utcfromtimestamp(forecast_item['dt']).isoformat(),
        'Ta_2m_Avg': forecast_item['main']['temp'],
        'TaMax_2m': forecast_item['main']['temp_max'],
        'TaMin_2m': forecast_item['main']['temp_min'],
        'RH_2m_Avg': forecast_item['main']['humidity'],
        'Dp_2m_Avg': forecast_item['main'].get('dew_point'),
        'WndAveSpd_3m': forecast_item['wind']['speed'],
        'WndAveDir_3m': forecast_item['wind']['deg'],
        'WndMaxSpd5s_3m': forecast_item['wind'].get('gust'),
        'PresAvg_1pnt5m': forecast_item['main']['pressure'],
        'Rain_1m_Tot': forecast_item['rain']['3h'] if 'rain' in forecast_item else 0,
        'Visibility': forecast_item.get('visibility', 0),
        'Clouds': forecast_item['clouds']['all']
    }

def map_solar_forecast_data(solar_item):
    return {
        'TIMESTAMP': datetime.utcfromtimestamp(solar_item['dt']).isoformat(),
        'Solar_2m_Avg': solar_item['radiation']['ghi']  # Assuming GHI is used for solar radiation
    }

def merge_weather_and_solar_data(weather_data, solar_data):
    weather_df = pd.DataFrame(weather_data)
    solar_df = pd.DataFrame(solar_data)
    merged_df = pd.merge(weather_df, solar_df, on='TIMESTAMP', how='left')
    return merged_df

def calculate_et(df):
    df_daily = df.set_index('TIMESTAMP').resample('D').mean()
    
    required_columns = ['Ta_2m_Avg', 'TaMax_2m', 'TaMin_2m', 'RH_2m_Avg', 'WndAveSpd_3m', 'Solar_2m_Avg']
    df_daily = df_daily.dropna(subset=required_columns)
    
    if df_daily.empty:
        logger.warning("Dataframe is empty after resampling and dropping NaN values.")
        return pd.DataFrame(columns=['TIMESTAMP', 'et'])
    
    # Convert solar radiation to MJ/m^2/day
    df_daily['Solar_2m_Avg_MJ'] = df_daily['Solar_2m_Avg'] * 0.0864
    
    lat_rad = LATITUDE * np.pi / 180
    
    inputs = {
        'tmean': df_daily['Ta_2m_Avg'],
        'wind': df_daily['WndAveSpd_3m'],
        'rs': df_daily['Solar_2m_Avg_MJ'],
        'tmax': df_daily['TaMax_2m'],
        'tmin': df_daily['TaMin_2m'],
        'rh': df_daily['RH_2m_Avg'],
        'elevation': ELEVATION,
        'lat': lat_rad
    }
    
    df_daily['et'] = pyet.combination.pm_asce(**inputs)
    
    return df_daily.reset_index()[['TIMESTAMP', 'et']]

def four_day_forecast_static(lat, lon):
    try:
        logger.info("Starting static 4-day forecast function")
        
        weather_forecast_data = get_forecast(lat, lon)
        solar_forecast_data = get_solar_radiation_forecast(lat, lon)
        
        collection_time = datetime.now(pytz.UTC)
        
        mapped_weather_data = [map_forecast_data(item, collection_time) for item in weather_forecast_data['list']]
        mapped_solar_data = [map_solar_forecast_data(item) for item in solar_forecast_data['list']]
        
        logger.info(f"Processed {len(mapped_weather_data)} weather forecast entries")
        logger.info(f"Processed {len(mapped_solar_data)} solar radiation forecast entries")
        
        df_forecast = merge_weather_and_solar_data(mapped_weather_data, mapped_solar_data)
        
        # Calculate daily precipitation sum
        df_forecast['TIMESTAMP'] = pd.to_datetime(df_forecast['TIMESTAMP'])
        df_forecast.set_index('TIMESTAMP', inplace=True)
        df_daily_precip = df_forecast['Rain_1m_Tot'].resample('D').sum().reset_index()
        
        et_df = calculate_et(df_forecast)
        
        if not et_df.empty:
            df_forecast = pd.merge(df_forecast.reset_index(), et_df, on='TIMESTAMP', how='left')
            df_forecast = pd.merge(df_forecast, df_daily_precip, on='TIMESTAMP', how='left', suffixes=('', '_daily_sum'))
        
        logger.info("Static forecast data processed successfully")
        return df_forecast
    except Exception as e:
        logger.error(f"Error processing static forecast data: {str(e)}", exc_info=True)
        return pd.DataFrame()

if __name__ == "__main__":
    lat, lon = 41.089075, -100.773775
    df_result = four_day_forecast_static(lat, lon)
    print(df_result)
