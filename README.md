# SpaceX Launch Analysis & Prediction Platform

An advanced Python application that visualizes SpaceX launch data and predicts the success of future launches using machine learning models. This platform provides comprehensive insights into SpaceX's launch history and helps predict the success probability of future launches.

## Live Demo:
https://space-x.streamlit.app
## Features

- **Interactive Dashboard**
  - Historical launch data visualization
  - Success rate trends over time
  - Payload mass distribution analysis
  - Launch success rate heatmap by month and year

- **Launch Site Analysis**
  - Interactive map showing all launch sites
  - Site-specific success rates
  - Geographic distribution of launches
  - Launch site statistics and metrics

- **Prediction Model**
  - Machine learning-based launch success prediction
  - Feature importance visualization
  - Real-time prediction with confidence scores
  - Model performance metrics (accuracy, precision, recall)

- **Data Analysis**
  - Comprehensive launch data analysis
  - Rocket and payload statistics
  - Core reuse analysis
  - Crew and capsule information

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd spacex-launch-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── data/                  # Data storage directory
│   ├── launches.json     # Launch data
│   ├── launchpads.json   # Launchpad information
│   ├── rockets.json      # Rocket specifications
│   ├── capsules.json     # Capsule data
│   └── cores.json        # Core information
├── src/                  # Source code
│   ├── data_acquisition.py    # SpaceX API data fetching
│   ├── preprocessing.py       # Data cleaning and feature engineering
│   ├── model.py              # ML model implementation
│   └── visualization.py      # Data visualization functions
├── app.py                # Main Streamlit application
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the dashboard:
   - Open your web browser
   - Navigate to `http://localhost:8501`

3. Using the Platform:
   - Navigate through different sections using the sidebar
   - View historical data and trends
   - Make predictions for future launches
   - Explore launch site statistics
   - Analyze feature importance

## Data Sources

- **SpaceX API v5**
  - Launch data
  - Rocket specifications
  - Launchpad information
  - Capsule data
  - Core information

## Technologies Used

- **Backend & Data Processing**
  - Python 3.x
  - Pandas & NumPy for data manipulation
  - Scikit-learn for machine learning
  - Requests for API communication

- **Frontend & Visualization**
  - Streamlit for interactive dashboard
  - Plotly for interactive charts
  - Folium for geospatial visualization
  - Streamlit-folium for map integration

## Model Details

The prediction model uses a Random Forest Classifier with the following features:
- Launch parameters (year, month, payload mass)
- Core reuse information
- Launch site details (latitude, longitude)
- Rocket specifications (stages, cost)
- Capsule and crew presence
- Historical success rates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
