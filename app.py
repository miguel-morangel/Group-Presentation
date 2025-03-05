"""
Streamlit App for Cat Marketing Analytics
This app provides an interactive interface for:
1. Model predictions and insights
2. Data visualization
3. Business recommendations
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path to enable absolute imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import logging
import joblib
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
from datetime import datetime, timedelta

# Import our modules using absolute imports
from model_trainer import CatMarketingPredictor
from train_model import train_model
from model_creation import generate_business_insights
from engineer_features import FeatureEngineer
from model_prediction import load_predictor, prepare_and_predict
from data_utils import (
    handle_outliers,
    calculate_time_features,
    calculate_engagement_score,
    validate_categorical_value,
    calculate_content_score,
    normalize_budget,
    ENGAGEMENT_BASELINES
)

# Import analysis modules
import data_analysis
import clustering
import nlp_processing
import time_series_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Cat Marketing Analytics",
    page_icon="🐱",
    layout="wide"
)

# Initialize model at startup
def initialize_model():
    """Initialize or train the model if needed"""
    model_path = Path('models/model.joblib')
    if not model_path.exists():
        st.warning("Training model for first use... This may take a minute.")
        try:
            train_model(
                campaigns_path='data/raw/catfluencer_campaigns.csv',
                engagement_path='data/raw/catfluencer_engagement.csv',
                posts_path='data/raw/catfluencer_posts.csv'
            )
            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return False
    return True

# Check if model is ready
model_ready = initialize_model()

def load_data():
    """Load and preprocess the data"""
    try:
        # Load data from data/raw directory
        data_dir = Path(__file__).parent / 'data' / 'raw'
        campaigns_df = pd.read_csv(data_dir / 'catfluencer_campaigns.csv')
        engagement_df = pd.read_csv(data_dir / 'catfluencer_engagement.csv')
        posts_df = pd.read_csv(data_dir / 'catfluencer_posts.csv')
        
        # Convert timestamps
        for df in [engagement_df, posts_df]:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return campaigns_df, engagement_df, posts_df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error("Error loading data. Please ensure the data files exist in the data/raw directory.")
        return None, None, None

def get_basic_stats(campaigns_df, posts_df):
    """Calculate basic statistics"""
    return {
        'total_campaigns': len(campaigns_df),
        'avg_budget': campaigns_df['budget'].mean(),
        'avg_likes': posts_df['likes'].mean(),
        'avg_comments': posts_df['comments'].mean(),
        'avg_shares': posts_df['shares'].mean()
    }

def analyze_content_performance(posts_df):
    """Analyze performance by content type"""
    return posts_df.groupby('content_type').agg({
        'likes': 'mean',
        'comments': 'mean',
        'shares': 'mean'
    }).rename(columns={
        'likes': 'Avg Likes',
        'comments': 'Avg Comments',
        'shares': 'Avg Shares'
    })

def prepare_time_features(df):
    """Prepare time-based features using data_utils"""
    df = df.copy()
    time_features = df['timestamp'].apply(calculate_time_features)
    for feature, values in pd.DataFrame(time_features.tolist()).items():
        df[feature] = values
    return df

def get_optimal_posting_times(df):
    """Calculate optimal posting times using time series analysis"""
    return time_series_analysis.get_optimal_posting_times(df)

def initialize_predictor():
    """Initialize and load the predictor"""
    try:
        predictor = load_predictor()
        if predictor is None:
            st.warning("⚠️ No trained model found. Please train the model first.")
            return None
        return predictor
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def make_prediction(predictor, input_data):
    """Make a prediction using the model"""
    try:
        # Make prediction using the predictor
        prediction, metadata = prepare_and_predict(predictor, input_data)
        return prediction, metadata
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

def show_overview():
    """Show overview page"""
    st.title("Cat Marketing Analytics Dashboard")
    
    # Load data
    campaigns_df, engagement_df, posts_df = load_data()
    if campaigns_df is None:
        return
        
    # Create tabs
    overview_tab, clustering_tab, nlp_tab, temporal_tab = st.tabs([
        "Overview Stats", "Engagement Clusters", "Content Analysis", "Temporal Patterns"
    ])
    
    with overview_tab:
        # Get basic stats
        stats = get_basic_stats(campaigns_df, posts_df)
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Campaigns", stats['total_campaigns'])
        with col2:
            st.metric("Average Engagement Rate", f"{engagement_df['engagement_rate'].mean():.2%}")
        with col3:
            st.metric("Total Budget", f"${stats['avg_budget']:,.2f}")
        with col4:
            st.metric("Average Sentiment", f"{engagement_df['sentiment_score'].mean():.2f}")
        
        # Engagement over time
        st.subheader("Engagement Trends")
        daily_engagement = engagement_df.copy()
        daily_engagement['date'] = pd.to_datetime(daily_engagement['timestamp']).dt.date
        daily_engagement = daily_engagement.groupby('date')['engagement_rate'].mean().reset_index()
        daily_engagement['date'] = pd.to_datetime(daily_engagement['date'])
        
        fig = px.line(daily_engagement, x='date', y='engagement_rate',
                     title='Daily Average Engagement Rate')
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Engagement Rate",
            yaxis_tickformat='.2%'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Platform performance
        st.subheader("Platform Performance")
        platform_metrics = campaigns_df.groupby('platform').agg({
            'budget': 'sum',
            'campaign_id': 'count'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(platform_metrics, values='budget', names='platform',
                        title='Budget Distribution by Platform')
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig)
        
        with col2:
            fig = px.pie(platform_metrics, values='campaign_id', names='platform',
                        title='Campaign Distribution by Platform')
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig)
    
    with clustering_tab:
        st.subheader("Engagement Clustering Analysis")
        clustered_posts = clustering.perform_clustering(posts_df)
        
        # Get the clustering plots
        elbow_fig, scatter_fig, box_fig = clustering.plot_clustering_analysis(clustered_posts)
        
        # Display the plots in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(elbow_fig)
            st.pyplot(scatter_fig)
        
        with col2:
            st.pyplot(box_fig)
            
            # Add cluster statistics
            st.subheader("Cluster Statistics")
            cluster_stats = clustered_posts.groupby('engagement_cluster').agg({
                'likes': 'mean',
                'comments': 'mean',
                'shares': 'mean'
            }).round(2)
            
            st.dataframe(cluster_stats)
    
    with nlp_tab:
        st.subheader("Content Analysis")
        nlp_processing.initialize_nlp()
        processed_posts = nlp_processing.prepare_engagement_data(posts_df)
        analysis_results = nlp_processing.analyze_caption_impact(processed_posts)
        
        col1, col2 = st.columns(2)
        with col1:
            plt.close('all')
            nlp_processing.create_word_impact_plot(analysis_results['top_words'])
            fig_impact = plt.gcf()
            fig_impact.set_size_inches(10, 6)
            plt.tight_layout()
            st.pyplot(fig_impact)
        
        with col2:
            plt.close('all')
            wordcloud = nlp_processing.create_word_cloud(analysis_results['words'], analysis_results['importance'])
            fig_cloud = plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            st.pyplot(fig_cloud)
        
        # Add top performing posts with better formatting
        st.subheader("Top Performing Posts")
        top_posts = nlp_processing.get_top_performing_posts(processed_posts)
        
        # Format the dataframe for better display
        if 'timestamp' in top_posts.columns:
            top_posts['timestamp'] = pd.to_datetime(top_posts['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        if 'engagement_rate' in top_posts.columns:
            top_posts['engagement_rate'] = top_posts['engagement_rate'].map('{:.2%}'.format)
        
        st.dataframe(
            top_posts,
            use_container_width=True,
            hide_index=True
        )
    
    with temporal_tab:
        st.subheader("Temporal Analysis")
        # Ensure timestamp is datetime
        engagement_df['timestamp'] = pd.to_datetime(engagement_df['timestamp'])
        engagement_df = time_series_analysis.prepare_time_features(engagement_df)
        
        # Convert hour and day_of_week to numeric explicitly
        engagement_df['hour'] = pd.to_numeric(engagement_df['hour'])
        engagement_df['day_of_week'] = pd.to_numeric(engagement_df['day_of_week'])
        
        col1, col2 = st.columns(2)
        with col1:
            plt.close('all')
            hourly_engagement = time_series_analysis.analyze_hourly_engagement(engagement_df)
            # Convert hour to numeric if it's not already
            if not pd.api.types.is_numeric_dtype(hourly_engagement.index):
                hourly_engagement.index = pd.to_numeric(hourly_engagement.index)
            time_series_analysis.plot_hourly_engagement(hourly_engagement)
            fig_hourly = plt.gcf()
            fig_hourly.set_size_inches(10, 6)
            plt.tight_layout()
            st.pyplot(fig_hourly)
        
        with col2:
            plt.close('all')
            daily_engagement = time_series_analysis.analyze_daily_engagement(engagement_df)
            # Convert day_of_week to numeric if it's not already
            if not pd.api.types.is_numeric_dtype(daily_engagement.index):
                daily_engagement.index = pd.to_numeric(daily_engagement.index)
            time_series_analysis.plot_daily_engagement(daily_engagement)
            fig_daily = plt.gcf()
            fig_daily.set_size_inches(10, 6)
            plt.tight_layout()
            st.pyplot(fig_daily)
        
        plt.close('all')
        heatmap_data = time_series_analysis.create_engagement_heatmap(engagement_df)
        time_series_analysis.plot_engagement_heatmap(heatmap_data)
        fig_heatmap = plt.gcf()
        fig_heatmap.set_size_inches(12, 6)
        plt.tight_layout()
        st.pyplot(fig_heatmap)
        
        # Add optimal posting times
        optimal_times = time_series_analysis.get_optimal_posting_times(engagement_df)
        
        st.subheader("Best Times to Post")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Best Hours:")
            for hour in optimal_times['best_hours']:
                st.write(f"• {hour}")
        with col2:
            st.write("Best Days:")
            for day in optimal_times['best_days']:
                st.write(f"• {day}")

def show_predictions():
    """Show predictions page"""
    st.title("Predict Campaign Performance")
    
    predictor = initialize_predictor()
    if predictor is None:
        return
    
    # Input form
    st.subheader("Enter Campaign Details")
    
    # Create input form
    input_data = {}
    
    col1, col2 = st.columns(2)
    with col1:
        # Platform selection
        platform_options = ['Instagram', 'Facebook', 'TikTok']
        input_data['platform'] = st.selectbox('Platform', platform_options)
        
        # Content type selection
        content_options = ['Photo', 'Video', 'Story', 'Reel']
        input_data['content_type'] = st.selectbox('Content Type', content_options)
        
        # Campaign goal selection
        goal_options = ['Sales', 'Brand Awareness', 'Engagement']
        input_data['campaign_goal'] = st.selectbox('Campaign Goal', goal_options)
        
        # Target audience selection
        audience_options = ['Young Adults', 'Pet Owners', 'General']
        input_data['target_audience'] = st.selectbox('Target Audience', audience_options)
        
        # Campaign dates
        st.markdown("### 📅 Campaign Duration")
        start_date = st.date_input("Campaign Start Date", value=pd.Timestamp.now())
        end_date = st.date_input("Campaign End Date", value=pd.Timestamp.now() + pd.Timedelta(days=7))
        
        if start_date >= end_date:
            st.error("End date must be after start date")
            return
            
        input_data['start_date'] = start_date
        input_data['end_date'] = end_date
        
    with col2:
        # Budget input
        input_data['budget'] = st.number_input('Budget ($)', min_value=100, max_value=10000, value=1000)
        
        # Content metrics
        input_data['hashtag_count'] = st.number_input('Number of Hashtags', min_value=0, max_value=30, value=5)
        input_data['sentiment_score'] = st.slider('Content Sentiment (-1 to 1)', min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
        input_data['is_sponsored'] = st.checkbox('Is Sponsored Content', value=False)
        
        # Timing
        st.markdown("### ⏰ Timing")
        post_time = st.time_input("Post Time", value=pd.Timestamp.now().replace(hour=12, minute=0))
        input_data['hour'] = post_time.hour
        input_data['is_weekend'] = st.checkbox("Post on Weekend", value=False)
        
        # Calculate time-based features using data_utils
        current_time = pd.Timestamp.now().replace(hour=post_time.hour)
        time_features = calculate_time_features(current_time)
        input_data.update(time_features)
        
        # Add interaction features
        input_data['weekend_peak'] = float(input_data['is_weekend']) * float(input_data['is_peak_hours'])
        input_data['sentiment_business'] = float(input_data['sentiment_score']) * float(input_data['is_business_hours'])
        input_data['hashtag_weekend'] = float(input_data['hashtag_count']) * float(input_data['is_weekend'])
        input_data['budget_peak'] = float(input_data['budget']) * float(input_data['is_peak_hours'])
        input_data['time_score'] = (
            0.4 * float(input_data['is_peak_hours']) +
            0.3 * float(input_data['is_business_hours']) +
            0.3 * (1 - float(input_data['is_weekend']))
        )
    
    if st.button("Predict Engagement"):
        try:
            # Make prediction
            prediction, metadata = make_prediction(predictor, input_data)
            
            # Display results
            st.success(f"Predicted Engagement Rate: {metadata['clipped_prediction']:.2%}")
            
            # Create three columns for key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Campaign Duration",
                    f"{max(0, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days)} days",
                    delta=None,
                    help="Total duration of the campaign"
                )
            
            with col2:
                confidence = min(1.0, 1.0 - abs(metadata['raw_prediction'] - metadata['clipped_prediction']))
                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.1%}",
                    delta=None,
                    help="Higher confidence means more reliable prediction"
                )
            
            with col3:
                campaign_days = max(1, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days)
                st.metric(
                    "Expected Daily Engagement",
                    f"{metadata['clipped_prediction'] / campaign_days:.3%}",
                    delta=None,
                    help="Average daily engagement rate"
                )
            
            # Show feature importance with improved visualization
            if metadata['feature_importance']:
                st.subheader("🔍 Key Factors Influencing Engagement")
                
                # Prepare feature importance data and ensure proper types
                importance_df = pd.DataFrame({
                    'Feature': list(metadata['feature_importance'].keys()),
                    'Importance': pd.to_numeric(list(metadata['feature_importance'].values()))  # Ensure numeric
                }).sort_values('Importance', ascending=True)
                
                # Create a horizontal bar chart with custom formatting
                fig = go.Figure(go.Bar(
                    x=importance_df['Importance'].values,  # Use .values to avoid categorical warning
                    y=importance_df['Feature'].values,     # Use .values to avoid categorical warning
                    orientation='h',
                    marker_color='rgb(55, 83, 109)'
                ))
                
                fig.update_layout(
                    title='Top Factors Affecting Engagement',
                    xaxis_title='Relative Importance',
                    yaxis_title='Feature',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(
                        type='category',  # Explicitly set as categorical
                        categoryorder='total ascending'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed insights
            with st.expander("📊 Detailed Analysis"):
                # Create tabs for different aspects
                stats_tab, features_tab, timing_tab = st.tabs([
                    "Statistics", "Features", "Timing Analysis"
                ])
                
                with stats_tab:
                    st.markdown("### 📈 Prediction Statistics")
                    # Convert all values to strings to avoid Arrow conversion issues
                    stats_df = pd.DataFrame({
                        'Metric': [
                            'Raw Prediction Score',
                            'Adjusted Prediction',
                            'Confidence Score',
                            'Feature Count',
                            'Campaign Duration',
                            'Daily Engagement Target'
                        ],
                        'Value': [
                            f"{metadata['raw_prediction']:.3f}",
                            f"{metadata['clipped_prediction']:.2%}",
                            f"{confidence:.1%}",
                            str(metadata['feature_shape'][1]),  # Convert to string
                            f"{max(0, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days)} days",
                            f"{metadata['clipped_prediction'] / max(1, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days):.3%}"
                        ]
                    })
                    st.dataframe(
                        stats_df.astype(str),  # Ensure all values are strings
                        hide_index=True,
                        use_container_width=True
                    )
                
                with features_tab:
                    st.markdown("### 🎯 Feature Analysis")
                    # Group features by type
                    feature_groups = {
                        'Content': ['sentiment', 'hashtag', 'content_score'],
                        'Timing': ['hour_', 'day_', 'month_', 'weekend', 'peak'],
                        'Campaign': ['budget', 'duration', 'progress'],
                        'Engagement': ['likes', 'comments', 'shares'],
                        'Platform': ['platform_', 'content_type_']
                    }
                    
                    for group, patterns in feature_groups.items():
                        matching_features = [
                            feat for feat in metadata['features_used']
                            if any(pattern in feat.lower() for pattern in patterns)
                        ]
                        if matching_features:
                            st.markdown(f"**{group} Features:**")
                            st.write(", ".join(f"`{f}`" for f in matching_features))
                
                with timing_tab:
                    st.markdown("### ⏰ Timing Analysis")
                    # Create a timeline visualization
                    timeline_df = pd.DataFrame([{
                        'Start': pd.Timestamp(start_date),
                        'End': pd.Timestamp(end_date),
                        'Current': pd.Timestamp.now()
                    }])
                    
                    fig = go.Figure()
                    
                    # Add campaign duration bar
                    campaign_label = np.array(['Campaign'])  # Use numpy array instead of list
                    fig.add_trace(go.Bar(
                        x=[(timeline_df['End'].iloc[0] - timeline_df['Start'].iloc[0]).days],
                        y=campaign_label,
                        orientation='h',
                        base=timeline_df['Start'].iloc[0],
                        width=0.3,
                        marker_color='lightblue',
                        hovertemplate='Campaign Duration: %{x} days<extra></extra>',
                        name='Campaign Duration'
                    ))
                    
                    # Add current time marker
                    fig.add_trace(go.Scatter(
                        x=[timeline_df['Current'].iloc[0]],
                        y=campaign_label,
                        mode='markers',
                        marker=dict(size=12, symbol='line-ns', line_width=2),
                        name='Current Time',
                        hovertemplate='Current Time<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title='Campaign Timeline',
                        showlegend=True,
                        height=100,
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis=dict(
                            type='date',
                            tickformat='%Y-%m-%d'
                        ),
                        yaxis=dict(
                            type='category',
                            categoryarray=campaign_label,
                            categoryorder='array'
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add timing metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Days Since Start",
                            f"{max(0, (pd.Timestamp.now() - pd.Timestamp(start_date)).days)} days",
                            help="Number of days since campaign start"
                        )
                    with col2:
                        st.metric(
                            "Days Until End",
                            f"{max(0, (pd.Timestamp(end_date) - pd.Timestamp.now()).days)} days",
                            help="Number of days until campaign ends"
                        )
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            logger.error(f"Prediction error details: {str(e)}", exc_info=True)

def show_insights():
    """Show model insights page with detailed analytics and recommendations"""
    st.title("🔍 Model Insights & Analytics")
    
    predictor = initialize_predictor()
    if predictor is None:
        st.warning("⚠️ No trained model found. Please train the model first.")
        return
    
    # Load data for time-based analysis
    try:
        data_dir = Path(__file__).parent / 'data' / 'raw'
        engagement_df = pd.read_csv(data_dir / 'catfluencer_engagement.csv')
        engagement_df['timestamp'] = pd.to_datetime(engagement_df['timestamp'])
    except Exception as e:
        logger.error(f"Error loading engagement data: {str(e)}")
        engagement_df = None
    
    # Generate insights
    insights = generate_business_insights(predictor)
    
    # Create tabs for different types of insights
    overview_tab, features_tab, recommendations_tab = st.tabs([
        "Overview 📊", "Feature Analysis 🎯", "Recommendations 💡"
    ])
    
    with overview_tab:
        st.subheader("Model Performance Overview")
        
        # Get model metrics safely
        metrics = predictor.get_model_metrics() if hasattr(predictor, 'get_model_metrics') else {}
        model_info = predictor.get_model_info() if hasattr(predictor, 'get_model_info') else {}
        
        # Display model metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Model Performance",
                f"{metrics.get('performance', 0):.2%}",
                help="Overall model performance score"
            )
        with col2:
            st.metric(
                "Prediction Accuracy",
                f"{metrics.get('accuracy', 0):.2%}",
                help="Accuracy of predictions within acceptable range"
            )
        with col3:
            st.metric(
                "Average Error",
                f"{metrics.get('mean_error', 0):.3f}",
                help="Average prediction error"
            )
        
        # Display model version and training info
        st.info(
            f"""
            **Model Information:**
            - Version: {model_info.get('version', 'N/A')}
            - Last Training: {model_info.get('last_training_date', 'N/A')}
            - Features Used: {len(predictor.feature_names) if hasattr(predictor, 'feature_names') else 'N/A'}
            - Training Samples: {model_info.get('n_samples', 'N/A')}
            """
        )
    
    with features_tab:
        st.subheader("Feature Importance Analysis")
        
        # Get feature importance safely
        feature_importance = (
            predictor.get_feature_importance() 
            if hasattr(predictor, 'get_feature_importance') 
            else {}
        )
        
        if feature_importance:
            # Convert feature importance to DataFrame for better handling
            importance_df = pd.DataFrame([
                {'Feature': feat, 'Importance': imp}
                for feat, imp in feature_importance.items()
            ])
            
            # Sort by absolute importance
            importance_df['Abs_Importance'] = abs(importance_df['Importance'])
            importance_df = importance_df.sort_values('Abs_Importance', ascending=True).drop('Abs_Importance', axis=1)
            
            # Create bar plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker_color=np.where(
                    importance_df['Importance'] > 0,
                    'rgba(55, 83, 109, 0.8)',
                    'rgba(219, 64, 82, 0.8)'
                )
            ))
            
            fig.update_layout(
                title='Feature Impact on Engagement',
                xaxis_title='Impact Magnitude',
                yaxis_title='Feature',
                height=max(400, len(importance_df) * 20),
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance data not available")
        
        # Feature correlation analysis
        st.subheader("Feature Correlations")
        correlation_matrix = (
            predictor.get_feature_correlations()
            if hasattr(predictor, 'get_feature_correlations')
            else pd.DataFrame()
        )
        
        if not correlation_matrix.empty:
            # Create correlation heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            
            fig.update_layout(
                title='Feature Correlation Matrix',
                height=600,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature correlation data not available")
        
        # Feature distributions
        st.subheader("Feature Distributions")
        feature_names = (
            predictor.model_info.get('feature_names', [])
            if hasattr(predictor, 'model_info')
            else []
        )
        
        if feature_names:
            selected_feature = st.selectbox(
                "Select Feature to Analyze",
                options=feature_names
            )
            
            if selected_feature:
                feature_data = (
                    predictor.get_feature_distribution(selected_feature)
                    if hasattr(predictor, 'get_feature_distribution')
                    else None
                )
                
                if feature_data is not None and len(feature_data) > 0:
                    # Convert to numeric explicitly
                    feature_data = pd.to_numeric(feature_data, errors='coerce')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=feature_data,
                        nbinsx=30,
                        name='Distribution',
                        marker_color='rgba(55, 83, 109, 0.6)'
                    ))
                    
                    fig.update_layout(
                        title=f'Distribution of {selected_feature}',
                        xaxis_title=selected_feature,
                        yaxis_title='Count',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature statistics
                    stats = (
                        predictor.get_feature_statistics(selected_feature)
                        if hasattr(predictor, 'get_feature_statistics')
                        else {}
                    )
                    if stats:
                        st.write("Feature Statistics:")
                        stats_df = pd.DataFrame([{
                            'Metric': k.title(),
                            'Value': f"{v:.3f}"
                        } for k, v in stats.items()])
                        st.dataframe(stats_df, hide_index=True)
                else:
                    st.warning("Feature distribution data not available")
        else:
            st.warning("No feature names available")
    
    with recommendations_tab:
        st.subheader("Actionable Recommendations")
        
        if insights:
            # Group insights by category
            for insight in insights:
                with st.expander(f"📌 {insight['title']}", expanded=True):
                    if 'description' in insight:
                        st.markdown(f"**Overview:** {insight['description']}")
                    
                    if 'items' in insight:
                        for item in insight['items']:
                            st.markdown(f"• {item}")
                    
                    if 'metrics' in insight:
                        cols = st.columns(len(insight['metrics']))
                        for col, (metric, value) in zip(cols, insight['metrics'].items()):
                            col.metric(
                                metric,
                                value,
                                delta=insight['metrics_delta'].get(metric) if 'metrics_delta' in insight else None
                            )
                    
                    if 'chart_data' in insight:
                        fig = go.Figure(data=insight['chart_data'])
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Add time-based recommendations
            if engagement_df is not None:
                st.subheader("Timing Optimization")
                # Prepare time features
                engagement_df = time_series_analysis.prepare_time_features(engagement_df)
                optimal_times = time_series_analysis.get_optimal_posting_times(engagement_df)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Best Times to Post:**")
                    for hour, rate in zip(optimal_times['best_hours'], optimal_times['best_hours_rates']):
                        st.write(f"• {hour} ({rate:.2%} engagement)")
                
                with col2:
                    st.write("**Best Days to Post:**")
                    for day, rate in zip(optimal_times['best_days'], optimal_times['best_days_rates']):
                        st.write(f"• {day} ({rate:.2%} engagement)")
            
            # Add key takeaways
            if hasattr(predictor, 'get_key_takeaways'):
                st.subheader("Key Takeaways")
                takeaways = predictor.get_key_takeaways()
                for i, takeaway in enumerate(takeaways, 1):
                    st.markdown(f"{i}. {takeaway}")
        else:
            st.warning("No insights available")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Overview", "Make Predictions", "Model Insights"]
)

# Main app logic
if page == "Overview":
    show_overview()
elif page == "Make Predictions":
    show_predictions()
else:
    show_insights()