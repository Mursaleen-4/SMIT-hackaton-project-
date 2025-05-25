import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder
import json
import os

class DataPreprocessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.label_encoders = {}
        
    def load_json_data(self, filename: str) -> List[Dict]:
        """Load data from JSON file."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV file."""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {filename} to {filepath}")
    
    def prepare_and_save_all_data(self):
        """Prepare and save all data as CSV files."""
        # Prepare all data
        launches_df = self.prepare_launches_data()
        launchpads_df = self.prepare_launchpads_data()
        rockets_df = self.prepare_rockets_data()
        capsules_df = self.prepare_capsules_data()
        cores_df = self.prepare_cores_data()
        
        # Save to CSV
        self.save_to_csv(launches_df, 'launches.csv')
        self.save_to_csv(launchpads_df, 'launchpads.csv')
        self.save_to_csv(rockets_df, 'rockets.csv')
        self.save_to_csv(capsules_df, 'capsules.csv')
        self.save_to_csv(cores_df, 'cores.csv')
        
        # Prepare and save merged data
        X, y = self.prepare_training_data()
        merged_data = pd.concat([X, y], axis=1)
        self.save_to_csv(merged_data, 'merged_data.csv')
        
        return {
            'launches': launches_df,
            'launchpads': launchpads_df,
            'rockets': rockets_df,
            'capsules': capsules_df,
            'cores': cores_df,
            'merged_data': merged_data
        }
    
    def prepare_launches_data(self) -> pd.DataFrame:
        """Prepare launches data for analysis."""
        print("\n[DEBUG] Starting prepare_launches_data...")
        launches = self.load_json_data('launches.json')
        payloads_data = self.load_json_data('payloads.json')
        payload_mass_dict = {p['id']: p.get('mass_kg', 0) or 0 for p in payloads_data}
        
        # Convert to DataFrame
        df = pd.DataFrame(launches)
        print("[DEBUG] Initial launches DataFrame columns:", df.columns.tolist())
        
        try:
            # Select relevant columns
            df = df[[
                'flight_number', 'name', 'date_utc', 'success', 'failures',
                'rocket', 'launchpad', 'payloads', 'cores', 'capsules',
                'ships', 'crew', 'details', 'links'
            ]]
        except KeyError as e:
            missing_cols = str(e).strip("'")
            print(f"[DEBUG] WARNING: Missing columns in launches data: {missing_cols}")
            # Continue with available columns
            df = df[[col for col in [
                'flight_number', 'name', 'date_utc', 'success', 'failures',
                'rocket', 'launchpad', 'payloads', 'cores', 'capsules',
                'ships', 'crew', 'details', 'links'
            ] if col in df.columns]]
        
        # Convert date to datetime
        df['date_utc'] = pd.to_datetime(df['date_utc'])
        
        # Extract year and month
        df['year'] = df['date_utc'].dt.year
        df['month'] = df['date_utc'].dt.month
        
        # Handle failures
        df['failure_reason'] = df['failures'].apply(
            lambda x: x[0]['reason'] if x and len(x) > 0 else None
        )
        
        # Extract payload mass using payload_mass_dict
        def get_payload_mass(payload_ids):
            if not payload_ids:
                return 0
            return sum(payload_mass_dict.get(pid, 0) for pid in payload_ids)
        
        df['payload_mass'] = df['payloads'].apply(get_payload_mass)
        
        # Extract core information
        df['core_reused'] = df['cores'].apply(
            lambda x: x[0].get('reused', False) if x and len(x) > 0 else False
        )
        
        # Extract capsule information
        df['has_capsule'] = df['capsules'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        
        # Extract crew information
        df['has_crew'] = df['crew'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        
        # Clean up
        df = df.drop(['failures', 'payloads', 'cores', 'capsules', 'ships', 'crew', 'links'], axis=1)
        print("[DEBUG] Final launches DataFrame columns:", df.columns.tolist())
        
        return df
    
    def prepare_launchpads_data(self) -> pd.DataFrame:
        """Prepare launchpads data for analysis."""
        launchpads = self.load_json_data('launchpads.json')
        
        # Convert to DataFrame
        df = pd.DataFrame(launchpads)
        
        # Select relevant columns
        columns = ['id', 'name', 'full_name', 'locality', 'region', 'latitude', 'longitude', 'launch_attempts', 'launch_successes']
        df = df[columns]
        
        # Calculate success rate
        df['success_rate'] = df['launch_successes'] / df['launch_attempts']
        
        return df
    
    def prepare_rockets_data(self) -> pd.DataFrame:
        """Prepare rockets data for analysis."""
        rockets = self.load_json_data('rockets.json')
        
        # Convert to DataFrame
        df = pd.DataFrame(rockets)
        
        # Select relevant columns
        columns = ['id', 'name', 'type', 'active', 'stages', 'cost_per_launch', 'success_rate_pct', 'first_flight', 'country']
        df = df[columns]
        
        return df
    
    def prepare_capsules_data(self) -> pd.DataFrame:
        """Prepare capsules data for analysis."""
        capsules = self.load_json_data('capsules.json')
        
        # Convert to DataFrame
        df = pd.DataFrame(capsules)
        
        # Select relevant columns
        columns = ['id', 'serial', 'status', 'type', 'reuse_count', 'water_landings', 'land_landings']
        df = df[columns]
        
        return df
    
    def prepare_cores_data(self) -> pd.DataFrame:
        """Prepare cores data for analysis."""
        cores = self.load_json_data('cores.json')
        
        # Convert to DataFrame
        df = pd.DataFrame(cores)
        
        # Select relevant columns
        columns = ['id', 'serial', 'status', 'reuse_count', 'rtls_attempts', 'rtls_landings', 'asds_attempts', 'asds_landings']
        df = df[columns]
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder."""
        df_encoded = df.copy()
        
        for column in columns:
            if column in df.columns:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                df_encoded[column] = self.label_encoders[column].fit_transform(df[column].astype(str))
        
        return df_encoded
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by removing NA rows, filling NA values, and creating dummy variables."""
        print("[DEBUG] Columns at start of clean_data:", df.columns.tolist())
        # Remove rows with NA in target variable if present
        if 'success' in df.columns:
            df = df.dropna(subset=['success'])
        # Fill NA values in numeric columns with mean
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        # Fill NA values in categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'success']  # Exclude 'success' from categorical columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        # Separate target variable before creating dummies
        target = None
        if 'success' in df.columns:
            target = df['success']
            features_df = df.drop(columns=['success'])
        else:
            features_df = df
        # Create dummy variables for categorical features
        features_df = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)
        # Concatenate target variable back if it was separated
        if target is not None:
            features_df['success'] = target.values
        print("[DEBUG] Columns at end of clean_data:", features_df.columns.tolist())
        return features_df
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare the data for training."""
        print("\n[DEBUG] Starting data preparation...")
        
        # Load and merge data
        launches_df = self.prepare_launches_data()
        rockets_df = self.prepare_rockets_data()
        launchpads_df = self.prepare_launchpads_data()
        
        print("[DEBUG] Data loaded:")
        print(f"Launches shape: {launches_df.shape}")
        print(f"Rockets shape: {rockets_df.shape}")
        print(f"Launchpads shape: {launchpads_df.shape}")
        
        # Merge data
        df = launches_df.merge(
            rockets_df,
            left_on='rocket',
            right_on='id',
            how='left',
            suffixes=('', '_rocket')
        )
        
        df = df.merge(
            launchpads_df,
            left_on='launchpad',
            right_on='id',
            how='left',
            suffixes=('', '_launchpad')
        )
        
        print("[DEBUG] After merging:")
        print(f"Final shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Get weather data
        try:
            from weather_data import WeatherDataFetcher
            weather_fetcher = WeatherDataFetcher()
            weather_df = weather_fetcher.get_launch_weather_data(df)
            
            # Merge weather data
            df = df.merge(
                weather_df,
                on='flight_number',
                how='left'
            )
            print("[DEBUG] Added weather data")
            
            # Add weather-related features
            df['is_windy'] = df['wind_speed'] > 20  # Wind speed > 20 m/s
            df['is_cloudy'] = df['clouds'] > 50  # Cloud coverage > 50%
            df['is_rainy'] = df['precipitation'] > 0  # Any precipitation
            df['is_clear'] = df['visibility'] > 10000  # Visibility > 10km
            
            # Add season based on month
            df['season'] = pd.cut(
                df['month'],
                bins=[0, 3, 6, 9, 12],
                labels=['Winter', 'Spring', 'Summer', 'Fall']
            )
            
        except Exception as e:
            print(f"[WARNING] Could not fetch weather data: {e}")
            # Add empty weather columns
            weather_columns = [
                'temperature', 'humidity', 'wind_speed',
                'wind_direction', 'clouds', 'precipitation', 'visibility',
                'is_windy', 'is_cloudy', 'is_rainy', 'is_clear', 'season'
            ]
            for col in weather_columns:
                df[col] = None
        
        # Feature engineering
        df['year'] = pd.to_datetime(df['date_utc']).dt.year
        df['month'] = pd.to_datetime(df['date_utc']).dt.month
        df['payload_mass'] = df['payload_mass'].fillna(0)
        df['core_reused'] = df['core_reused'].fillna(False)
        df['has_capsule'] = df['has_capsule'].fillna(False)
        df['has_crew'] = df['has_crew'].fillna(False)
        
        # Fill missing weather data with median values
        weather_columns = [
            'temperature', 'humidity', 'wind_speed',
            'wind_direction', 'clouds', 'precipitation', 'visibility'
        ]
        for col in weather_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing boolean weather features with False
        boolean_weather_columns = ['is_windy', 'is_cloudy', 'is_rainy', 'is_clear']
        for col in boolean_weather_columns:
            if col in df.columns:
                df[col] = df[col].fillna(False)
        
        # Fill missing season with mode, handling empty case
        if 'season' in df.columns:
            if df['season'].isna().all():
                # If all values are NA, fill with 'Summer' as default
                df['season'] = df['season'].fillna('Summer')
            else:
                # If there are some values, fill NA with mode
                mode_value = df['season'].mode()
                if not mode_value.empty:
                    df['season'] = df['season'].fillna(mode_value[0])
                else:
                    df['season'] = df['season'].fillna('Summer')
        
        # Calculate success rate for each rocket
        rocket_success = df.groupby('rocket')['success'].mean()
        df['success_rate'] = df['rocket'].map(rocket_success)
        
        # Select features for training
        features = [
            'year', 'month', 'payload_mass', 'core_reused',
            'latitude', 'longitude', 'stages', 'cost_per_launch',
            'has_capsule', 'has_crew', 'success_rate',
            'locality', 'region', 'type', 'country',
            'temperature', 'humidity', 'wind_speed',
            'wind_direction', 'clouds', 'precipitation', 'visibility',
            'is_windy', 'is_cloudy', 'is_rainy', 'is_clear',
            'season'
        ]
        
        # Ensure target variable is boolean
        df['success'] = df['success'].astype(bool)
        
        # Create feature matrix and target vector
        X = df[features].copy()
        y = df['success'].copy()
        
        # One-hot encode categorical features
        categorical_features = ['locality', 'region', 'type', 'country', 'season']
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        
        print("[DEBUG] Final data shape:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"y values: {y.value_counts()}")
        
        return X, y

def main():
    """Main function to test preprocessing and save data."""
    preprocessor = DataPreprocessor()
    
    # First fetch the data if not already done
    from data_acquisition import SpaceXDataFetcher
    fetcher = SpaceXDataFetcher()
    fetcher.fetch_all_data()
    
    # Prepare and save all data
    data = preprocessor.prepare_and_save_all_data()
    
    print("\nData saved to CSV files in the 'data' directory:")
    print("1. launches.csv - Launch data")
    print("2. launchpads.csv - Launchpad information")
    print("3. rockets.csv - Rocket specifications")
    print("4. capsules.csv - Capsule data")
    print("5. cores.csv - Core information")
    print("6. merged_data.csv - Combined data for ML model")
    
    print(f"\nMerged dataset shape: {data['merged_data'].shape}")
    print(f"Target variable distribution:\n{data['merged_data']['success'].value_counts(normalize=True)}")

if __name__ == "__main__":
    main() 