"""
Model prediction module for the Streamlit app
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import Dict, Any, Tuple
import logging
from model_trainer import CatMarketingPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_predictor(model_dir: str = 'models') -> CatMarketingPredictor:
    """Load the trained predictor"""
    try:
        model_path = os.path.join(model_dir, 'model.joblib')
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}. Please train the model first.")
            return None
            
        predictor = CatMarketingPredictor()
        predictor.load_model(model_path)
        return predictor
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_prediction_input() -> Dict[str, Any]:
    """
    Get prediction input from Streamlit UI
    
    Returns:
        Dictionary containing the input features
    """
    st.markdown("### 📊 Post Details")
    col1, col2 = st.columns(2)
    
    with col1:
        platform = st.selectbox("Platform", ['Instagram', 'Facebook', 'TikTok'])
        content_type = st.selectbox("Content Type", ['Photo', 'Video', 'Story', 'Reel'])
        campaign_goal = st.selectbox("Campaign Goal", ['Sales', 'Brand Awareness', 'Engagement'])
        target_audience = st.selectbox("Target Audience", ['Young Adults', 'Pet Owners', 'Cat Lovers', 'General'])
        budget = st.number_input("Budget ($)", min_value=100, max_value=10000, value=1000, step=100)
    
    with col2:
        hashtag_count = st.number_input("Number of Hashtags", min_value=0, max_value=30, value=5)
        sentiment_score = st.slider("Content Sentiment (-1 to 1)", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
        is_sponsored = st.checkbox("Is Sponsored Content", value=False)
        
        st.markdown("### ⏰ Timing")
        post_time = st.time_input("Post Time", value=pd.Timestamp.now().replace(hour=12, minute=0))
        is_weekend = st.checkbox("Post on Weekend", value=False)
    
    # Calculate derived time features
    hour = post_time.hour
    is_business_hours = 9 <= hour <= 17
    is_peak_hours = hour in [12, 13, 17, 18, 19, 20]
    
    return {
        'platform': platform,
        'content_type': content_type,
        'campaign_goal': campaign_goal,
        'target_audience': target_audience,
        'budget': budget,
        'hashtag_count': hashtag_count,
        'sentiment_score': sentiment_score,
        'is_sponsored': int(is_sponsored),
        'hour': hour,
        'is_weekend': int(is_weekend),
        'is_business_hours': int(is_business_hours),
        'is_peak_hours': int(is_peak_hours)
    }

def display_prediction_results(prediction: float, metadata: Dict[str, Any]) -> None:
    """
    Display prediction results in Streamlit UI
    
    Args:
        prediction: The predicted engagement rate
        metadata: Additional prediction metadata
    """
    # Main prediction display
    st.markdown("### 🎯 Prediction Results")
    st.success(f"Predicted Engagement Rate: {prediction:.2%}")
    
    # Detailed metrics
    st.markdown("### 📈 Detailed Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Raw Score", f"{metadata['raw_prediction']:.3f}")
    with col2:
        st.metric("Adjusted Score", f"{metadata['clipped_prediction']:.3f}")
    with col3:
        confidence = min(1.0, 1.0 - abs(metadata['raw_prediction'] - metadata['clipped_prediction']))
        st.metric("Confidence", f"{confidence:.2%}")
    
    # Feature importance
    if 'feature_importance' in metadata:
        st.markdown("### 🔍 Key Factors")
        importance_df = pd.DataFrame({
            'Feature': metadata['feature_importance'].keys(),
            'Impact': metadata['feature_importance'].values()
        }).sort_values('Impact', ascending=False).head(5)
        
        st.bar_chart(importance_df.set_index('Feature'))
    
    # Additional insights
    with st.expander("See Technical Details"):
        st.markdown("#### Feature Information")
        features_df = pd.DataFrame({
            'Feature': metadata['features_used'],
            'Status': ['✅' for _ in metadata['features_used']]
        })
        st.dataframe(features_df)
        
        st.markdown("#### Model Details")
        st.write({
            'Feature Count': metadata['feature_shape'][1],
            'Processing Time': f"{metadata.get('processing_time', 0):.3f}s"
        })

def validate_input_data(input_data: Dict[str, Any]) -> None:
    """
    Validate the input data for prediction
    
    Args:
        input_data: Dictionary containing the input features
    
    Raises:
        ValueError: If any required fields are missing or invalid
    """
    required_fields = {
        'platform': ['Instagram', 'Facebook', 'TikTok'],
        'content_type': ['Photo', 'Video', 'Story', 'Reel'],
        'campaign_goal': ['Sales', 'Brand Awareness', 'Engagement'],
        'target_audience': ['Young Adults', 'Pet Owners', 'Cat Lovers', 'General'],
        'budget': (100, 10000),
        'hashtag_count': (0, 30),
        'sentiment_score': (-1, 1),
        'is_sponsored': (0, 1),
        'hour': (0, 23),
        'is_weekend': (0, 1),
        'is_business_hours': (0, 1),
        'is_peak_hours': (0, 1)
    }
    
    for field, valid_values in required_fields.items():
        if field not in input_data:
            raise ValueError(f"Missing required field: {field}")
            
        value = input_data[field]
        if isinstance(valid_values, list):
            if value not in valid_values:
                raise ValueError(f"Invalid value for {field}. Must be one of: {valid_values}")
        elif isinstance(valid_values, tuple):
            min_val, max_val = valid_values
            if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                raise ValueError(f"Invalid value for {field}. Must be between {min_val} and {max_val}")

def prepare_and_predict(
    predictor: CatMarketingPredictor,
    input_data: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Prepare features and make a prediction
    
    Args:
        predictor: The CatMarketingPredictor instance
        input_data: Dictionary containing the input features
        
    Returns:
        Tuple containing:
        - float: Predicted engagement rate (between 0 and 1)
        - Dict: Additional prediction metadata
    """
    try:
        start_time = pd.Timestamp.now()
        
        # Validate input data
        validate_input_data(input_data)
        logger.info("Input validation passed")
        
        # Let the feature engineer prepare the features
        features_df = predictor.feature_engineer.prepare_prediction_features(input_data)
        logger.info(f"Features prepared - shape: {features_df.shape}")
        
        # Get model and make prediction
        model = predictor.get_model('stacking_model')
        if model is None:
            raise ValueError("No trained model found")
            
        raw_pred = model.predict(features_df)[0]
        logger.info(f"Raw prediction: {raw_pred}")
        
        # Ensure prediction is between 0 and 1
        prediction = np.clip(raw_pred, 0, 1)
        logger.info(f"Clipped prediction: {prediction}")
        
        # Calculate processing time
        processing_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            for feat, imp in zip(features_df.columns, importance):
                if imp > 0:  # Only include features with non-zero importance
                    feature_importance[feat] = imp
        
        # Prepare metadata
        metadata = {
            'feature_shape': features_df.shape,
            'features_used': list(features_df.columns),
            'raw_prediction': float(raw_pred),
            'clipped_prediction': float(prediction),
            'processing_time': processing_time,
            'feature_importance': feature_importance
        }
        
        return prediction, metadata
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise 