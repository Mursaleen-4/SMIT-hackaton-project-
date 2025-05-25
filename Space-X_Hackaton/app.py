import streamlit as st
import pandas as pd
import numpy as np
from src.data_acquisition import SpaceXDataFetcher
from src.preprocessing import DataPreprocessor
from src.model import LaunchPredictor
from src.visualization import LaunchVisualizer
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="SpaceX Launch Analysis & Prediction",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Title and description
st.title("🚀 SpaceX Launch Analysis & Prediction Platform")
st.markdown("""
This platform provides comprehensive analysis and prediction capabilities for SpaceX launches.
Explore historical data, visualize trends, and predict the success of future launches.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Data Overview", "Launch Analysis", "Prediction Model", "Launch Sites Map"]
)

# Data fetching
@st.cache_data
def fetch_data():
    fetcher = SpaceXDataFetcher()
    return fetcher.fetch_all_data()

# Load and prepare data
@st.cache_data
def prepare_data():
    preprocessor = DataPreprocessor()
    launches_df = preprocessor.prepare_launches_data()
    launchpads_df = preprocessor.prepare_launchpads_data()
    rockets_df = preprocessor.prepare_rockets_data()
    
    # Merge data
    df = launches_df.merge(
        launchpads_df,
        left_on='launchpad',
        right_on='id',
        how='left',
        suffixes=('', '_launchpad')
    )
    
    df = df.merge(
        rockets_df,
        left_on='rocket',
        right_on='id',
        how='left',
        suffixes=('', '_rocket')
    )
    
    return df

# Train model
@st.cache_data
def train_model():
    """Train the model and return both the predictor and metrics."""
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_training_data()
    
    predictor = LaunchPredictor()
    metrics = predictor.train(X, y)
    
    return predictor, metrics

def show_prediction_model():
    """Show the prediction model interface."""
    st.title("Launch Success Prediction Model")
    
    # Train model
    predictor, metrics = train_model()
    
    # Display model metrics
    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.2%}")
    with col4:
        st.metric("F1 Score", f"{metrics['f1']:.2%}")
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = predictor.get_feature_importance()
    fig = px.bar(
        importance.head(10),
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Features'
    )
    st.plotly_chart(fig)
    
    # Prediction interface
    st.subheader("Make a Prediction")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic launch parameters
        year = st.number_input("Launch Year", min_value=2000, max_value=2024, value=2024)
        month = st.number_input("Launch Month", min_value=1, max_value=12, value=1)
        payload_mass = st.number_input("Payload Mass (kg)", min_value=0.0, value=1000.0)
        core_reused = st.checkbox("Core Reused")
        stages = st.number_input("Number of Stages", min_value=1, max_value=3, value=2)
        cost_per_launch = st.number_input("Cost per Launch (USD)", min_value=0, value=50000000)
        has_capsule = st.checkbox("Has Capsule")
        has_crew = st.checkbox("Has Crew")
        success_rate = st.slider("Historical Success Rate", 0.0, 1.0, 0.8)
    
    with col2:
        # Launch site parameters
        st.write("Launch Site Location")
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.5728)
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-80.6490)
        
        # Weather parameters
        st.write("Weather Conditions")
        temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
        humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=60)
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=100.0, value=5.0)
        wind_direction = st.number_input("Wind Direction (degrees)", min_value=0, max_value=359, value=180)
        clouds = st.number_input("Cloud Coverage (%)", min_value=0, max_value=100, value=20)
        precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=100.0, value=0.0)
        visibility = st.number_input("Visibility (m)", min_value=0, max_value=100000, value=10000)
        
        # Derived weather features
        is_windy = wind_speed > 20
        is_cloudy = clouds > 50
        is_rainy = precipitation > 0
        is_clear = visibility > 10000
        
        # Season
        season = pd.cut(
            [month],
            bins=[0, 3, 6, 9, 12],
            labels=['Winter', 'Spring', 'Summer', 'Fall']
        )[0]
    
    # Create prediction button
    if st.button("Predict Launch Success"):
        # Prepare input data with raw values
        raw_input = {
            'year': [year],
            'month': [month],
            'payload_mass': [payload_mass],
            'core_reused': [int(core_reused)],
            'latitude': [latitude],
            'longitude': [longitude],
            'stages': [stages],
            'cost_per_launch': [cost_per_launch],
            'has_capsule': [int(has_capsule)],
            'has_crew': [int(has_crew)],
            'success_rate': [success_rate],
            'temperature': [temperature],
            'humidity': [humidity],
            'wind_speed': [wind_speed],
            'wind_direction': [wind_direction],
            'clouds': [clouds],
            'precipitation': [precipitation],
            'visibility': [visibility],
            'is_windy': [int(is_windy)],
            'is_cloudy': [int(is_cloudy)],
            'is_rainy': [int(is_rainy)],
            'is_clear': [int(is_clear)],
            'season': [season]
        }
        
        # Create DataFrame with raw input
        input_data = pd.DataFrame(raw_input)
        
        # One-hot encode categorical features
        input_data = pd.get_dummies(input_data, columns=['season'], drop_first=True)
        
        # Make prediction
        prediction, probability = predictor.predict(input_data)
        
        # Display results
        st.subheader("Prediction Results")
        success_prob = probability[0][1] * 100
        
        if prediction[0] == 1:
            st.success(f"Launch is predicted to be successful! (Confidence: {success_prob:.1f}%)")
        else:
            st.error(f"Launch is predicted to fail. (Confidence: {100-success_prob:.1f}%)")
        
        # Show probability distribution
        fig = go.Figure(data=[
            go.Bar(
                x=['Failure', 'Success'],
                y=[100-success_prob, success_prob],
                marker_color=['#e74c3c', '#2ecc71']
            )
        ])
        fig.update_layout(
            title="Prediction Probability Distribution",
            yaxis_title="Probability (%)",
            showlegend=False
        )
        st.plotly_chart(fig)
        
        # Show weather impact
        st.subheader("Weather Impact Analysis")
        weather_metrics = {
            'Temperature': temperature,
            'Humidity': humidity,
            'Wind Speed': wind_speed,
            'Cloud Coverage': clouds,
            'Precipitation': precipitation,
            'Visibility': visibility
        }
        
        # Create weather radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(weather_metrics.values()),
            theta=list(weather_metrics.keys()),
            fill='toself',
            name='Current Conditions'
        ))
        
        # Add ideal conditions
        ideal_conditions = {
            'Temperature': 25,
            'Humidity': 50,
            'Wind Speed': 5,
            'Cloud Coverage': 20,
            'Precipitation': 0,
            'Visibility': 10000
        }
        
        fig.add_trace(go.Scatterpolar(
            r=list(ideal_conditions.values()),
            theta=list(ideal_conditions.keys()),
            fill='toself',
            name='Ideal Conditions'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(weather_metrics.values()), max(ideal_conditions.values()))]
                )
            ),
            showlegend=True
        )
        
        st.plotly_chart(fig)
        
        # Weather warnings
        st.subheader("Weather Warnings")
        warnings = []
        if is_windy:
            warnings.append("⚠️ High wind speed may affect launch")
        if is_cloudy:
            warnings.append("⚠️ High cloud coverage may affect visibility")
        if is_rainy:
            warnings.append("⚠️ Precipitation detected")
        if not is_clear:
            warnings.append("⚠️ Poor visibility conditions")
        
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("No weather warnings - conditions are favorable")

# Main content
if page == "Data Overview":
    st.header("Data Overview")
    
    if not st.session_state.data_fetched:
        with st.spinner("Fetching data..."):
            data = fetch_data()
            st.session_state.data_fetched = True
    
    df = prepare_data()
    
    # Display basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Launches", len(df))
    with col2:
        st.metric("Success Rate", f"{df['success'].mean():.1%}")
    with col3:
        st.metric("Unique Launch Sites", df['launchpad'].nunique())
    
    # Display recent launches
    st.subheader("Recent Launches")
    recent_launches = df.sort_values('date_utc', ascending=False).head(5)
    st.dataframe(recent_launches[['name', 'date_utc', 'success', 'rocket', 'full_name']])

elif page == "Launch Analysis":
    st.header("Launch Analysis")
    
    df = prepare_data()
    visualizer = LaunchVisualizer()
    
    # Success rate over time
    st.subheader("Launch Success Rate Over Time")
    fig = px.line(
        df.groupby('year')['success'].mean().reset_index(),
        x='year',
        y='success',
        title='Launch Success Rate by Year'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Payload mass distribution
    st.subheader("Payload Mass Distribution")
    fig = px.box(
        df,
        x='success',
        y='payload_mass',
        title='Payload Mass Distribution by Launch Success'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Success rate heatmap
    st.subheader("Success Rate by Month and Year")
    success_matrix = df.pivot_table(
        values='success',
        index='year',
        columns='month',
        aggfunc='mean'
    )
    fig = px.imshow(
        success_matrix,
        labels=dict(x="Month", y="Year", color="Success Rate"),
        title="Launch Success Rate Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Prediction Model":
    show_prediction_model()

elif page == "Launch Sites Map":
    st.header("Launch Sites Map")
    
    df = prepare_data()
    visualizer = LaunchVisualizer()
    
    # Create map
    m = visualizer.create_launch_sites_map(df)
    folium_static(m)
    
    # Display launch site statistics
    st.subheader("Launch Site Statistics")
    site_stats = df.groupby('full_name').agg({
        'success': ['count', 'mean'],
        'payload_mass': 'mean'
    }).round(2)
    site_stats.columns = ['Total Launches', 'Success Rate', 'Avg Payload Mass']
    st.dataframe(site_stats)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit") 