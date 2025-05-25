import requests
import json
import os
from typing import Dict, List, Optional

class SpaceXDataFetcher:
    def __init__(self, data_dir: str = "data"):
        self.base_url = "https://api.spacexdata.com/v4"
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_launches(self) -> List[Dict]:
        """Fetch all launches data."""
        try:
            response = requests.get(f"{self.base_url}/launches")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            print(f"Error fetching launches: {e}")
            return []
    
    def fetch_launchpads(self) -> List[Dict]:
        """Fetch all launchpads data."""
        try:
            response = requests.get(f"{self.base_url}/launchpads")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            print(f"Error fetching launchpads: {e}")
            return []
    
    def fetch_rockets(self) -> List[Dict]:
        """Fetch all rockets data."""
        try:
            response = requests.get(f"{self.base_url}/rockets")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            print(f"Error fetching rockets: {e}")
            return []
    
    def fetch_capsules(self) -> List[Dict]:
        """Fetch all capsules data."""
        try:
            response = requests.get(f"{self.base_url}/capsules")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            print(f"Error fetching capsules: {e}")
            return []
    
    def fetch_cores(self) -> List[Dict]:
        """Fetch all cores data."""
        try:
            response = requests.get(f"{self.base_url}/cores")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            print(f"Error fetching cores: {e}")
            return []
    
    def fetch_payloads(self) -> List[Dict]:
        try:
            response = requests.get(f"{self.base_url}/payloads")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            print(f"Error fetching payloads: {e}")
            return []
    
    def save_data(self, data: List[Dict], filename: str):
        """Save data to JSON file."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} records to {filepath}")
    
    def fetch_all_data(self) -> Dict[str, List[Dict]]:
        """Fetch and save all SpaceX data."""
        print("Fetching SpaceX data...")
        
        # Fetch all data
        launches = self.fetch_launches()
        launchpads = self.fetch_launchpads()
        rockets = self.fetch_rockets()
        capsules = self.fetch_capsules()
        cores = self.fetch_cores()
        payloads = self.fetch_payloads()
        
        # Save data
        self.save_data(launches, 'launches.json')
        self.save_data(launchpads, 'launchpads.json')
        self.save_data(rockets, 'rockets.json')
        self.save_data(capsules, 'capsules.json')
        self.save_data(cores, 'cores.json')
        self.save_data(payloads, 'payloads.json')
        
        print("\nData fetching complete!")
        print(f"Launches: {len(launches)}")
        print(f"Launchpads: {len(launchpads)}")
        print(f"Rockets: {len(rockets)}")
        print(f"Capsules: {len(capsules)}")
        print(f"Cores: {len(cores)}")
        print(f"Payloads: {len(payloads)}")
        
        return {
            'launches': launches,
            'launchpads': launchpads,
            'rockets': rockets,
            'capsules': capsules,
            'cores': cores,
            'payloads': payloads
        }

def main():
    """Main function to test data fetching."""
    fetcher = SpaceXDataFetcher()
    data = fetcher.fetch_all_data()
    print("\nAll data has been fetched and saved to the 'data' directory.")

if __name__ == "__main__":
    main() 