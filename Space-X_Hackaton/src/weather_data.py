import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
import time

class WeatherDataFetcher:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the weather data fetcher."""
        self.api_key = api_key or os.getenv('OPENWEATHERMAP_API_KEY')
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key is required. Set it as OPENWEATHERMAP_API_KEY environment variable.")
        
        # Use current weather endpoint instead of historical
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_current_weather(self, lat: float, lon: float) -> Dict:
        """Get current weather data for a specific location."""
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'  # Use metric units
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def get_weather_features(self, lat: float, lon: float, date: datetime) -> Dict:
        """Extract relevant weather features for a launch."""
        # Get current weather data
        weather_data = self.get_current_weather(lat, lon)
        if not weather_data:
            return {
                'temperature': None,
                'humidity': None,
                'wind_speed': None,
                'wind_direction': None,
                'clouds': None,
                'precipitation': None,
                'visibility': None
            }
        
        # Extract weather features
        return {
            'temperature': weather_data.get('main', {}).get('temp'),
            'humidity': weather_data.get('main', {}).get('humidity'),
            'wind_speed': weather_data.get('wind', {}).get('speed'),
            'wind_direction': weather_data.get('wind', {}).get('deg'),
            'clouds': weather_data.get('clouds', {}).get('all'),
            'precipitation': weather_data.get('rain', {}).get('1h', 0),
            'visibility': weather_data.get('visibility')
        }
    
    def get_launch_weather_data(self, launches_df: pd.DataFrame) -> pd.DataFrame:
        """Get weather data for all launches."""
        weather_data = []
        
        # Get unique launch sites to minimize API calls
        launch_sites = launches_df[['latitude', 'longitude']].drop_duplicates()
        
        # Create a dictionary to store weather data for each site
        site_weather = {}
        
        for _, site in launch_sites.iterrows():
            # Add delay to respect API rate limits
            time.sleep(1)
            
            # Get weather features for this site
            weather_features = self.get_weather_features(
                site['latitude'],
                site['longitude'],
                datetime.now()
            )
            
            # Store weather data for this site
            site_key = (site['latitude'], site['longitude'])
            site_weather[site_key] = weather_features
        
        # Assign weather data to each launch
        for _, launch in launches_df.iterrows():
            site_key = (launch['latitude'], launch['longitude'])
            weather_features = site_weather.get(site_key, {})
            weather_features['flight_number'] = launch['flight_number']
            weather_data.append(weather_features)
        
        return pd.DataFrame(weather_data)

def main():
    """Test the weather data fetcher."""
    # Example usage
    fetcher = WeatherDataFetcher()
    
    # Test coordinates (Kennedy Space Center)
    lat, lon = 28.5728, -80.6490
    
    # Get weather features
    features = fetcher.get_weather_features(lat, lon, datetime.now())
    print("Weather features:", features)

if __name__ == "__main__":
    main() 