import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import os

class LaunchVisualizer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        plt.style.use('seaborn-v0_8')
        self.colors = ['#2ecc71', '#e74c3c']  # Green for success, Red for failure
        
    def plot_launch_success_rate(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot launch success rate over time."""
        # Calculate success rate by year
        success_rate = df.groupby('year')['success'].mean()
        
        plt.figure(figsize=(12, 6))
        success_rate.plot(kind='line', marker='o')
        plt.title('SpaceX Launch Success Rate Over Time')
        plt.xlabel('Year')
        plt.ylabel('Success Rate')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_payload_mass_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot payload mass distribution by launch success."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='success', y='payload_mass', data=df)
        plt.title('Payload Mass Distribution by Launch Success')
        plt.xlabel('Launch Success')
        plt.ylabel('Payload Mass (kg)')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def create_launch_sites_map(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create an interactive map of launch sites."""
        # Create a map centered on the mean of all launch sites
        m = folium.Map(
            location=[df['latitude'].mean(), df['longitude'].mean()],
            zoom_start=4
        )
        
        # Create a marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each launch site
        for _, row in df.drop_duplicates('launchpad').iterrows():
            # Create popup content
            popup_content = f"""
            <b>Launch Site:</b> {row['full_name']}<br>
            <b>Location:</b> {row['locality']}, {row['region']}<br>
            <b>Total Launches:</b> {len(df[df['launchpad'] == row['launchpad']])}<br>
            <b>Success Rate:</b> {df[df['launchpad'] == row['launchpad']]['success'].mean():.2%}
            """
            
            # Add marker
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(
                    color='green' if df[df['launchpad'] == row['launchpad']]['success'].mean() > 0.5 else 'red',
                    icon='rocket',
                    prefix='fa'
                )
            ).add_to(marker_cluster)
        
        if save_path:
            m.save(save_path)
        else:
            return m
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot feature importance scores."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title('Feature Importance for Launch Success Prediction')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def create_interactive_launch_timeline(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create an interactive timeline of launches."""
        # Prepare data
        timeline_data = df.copy()
        timeline_data['date'] = pd.to_datetime(timeline_data['date_utc'])
        
        # Create figure
        fig = px.scatter(
            timeline_data,
            x='date',
            y='payload_mass',
            color='success',
            hover_data=['name', 'rocket', 'launchpad'],
            title='SpaceX Launch Timeline',
            labels={
                'date': 'Launch Date',
                'payload_mass': 'Payload Mass (kg)',
                'success': 'Launch Success'
            }
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def create_success_rate_heatmap(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create a heatmap of launch success rates by month and year."""
        # Calculate success rate by month and year
        success_matrix = df.pivot_table(
            values='success',
            index='year',
            columns='month',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            success_matrix,
            annot=True,
            fmt='.2%',
            cmap='RdYlGn',
            center=0.5
        )
        plt.title('Launch Success Rate by Month and Year')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def main():
    """Main function to test visualizations."""
    from preprocessing import DataPreprocessor
    
    # Prepare data
    preprocessor = DataPreprocessor()
    launches_df = preprocessor.prepare_launches_data()
    launchpads_df = preprocessor.prepare_launchpads_data()
    
    # Merge data
    df = launches_df.merge(
        launchpads_df,
        left_on='launchpad',
        right_on='id',
        how='left'
    )
    
    # Create visualizer
    visualizer = LaunchVisualizer()
    
    # Create visualizations
    visualizer.plot_launch_success_rate(df)
    visualizer.plot_payload_mass_distribution(df)
    visualizer.create_launch_sites_map(df)
    visualizer.create_interactive_launch_timeline(df)
    visualizer.create_success_rate_heatmap(df)

if __name__ == "__main__":
    main() 