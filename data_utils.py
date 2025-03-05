"""
Shared utilities for data processing and transformations used across the codebase.
Contains common functionality for cleaning, transforming, and validating data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for data processing
ENGAGEMENT_BASELINES = {
    'likes': {'median': 616.0, 'mean': 633.7, 'std': 271.9},
    'comments': {'median': 62.0, 'mean': 63.2, 'std': 27.3},
    'shares': {'median': 36.0, 'mean': 36.5, 'std': 13.8}
}

PEAK_HOURS = [12, 13, 17, 18, 19, 20]
BUSINESS_HOURS = range(9, 18)
WEEKEND_DAYS = [5, 6]

def handle_outliers(df: pd.DataFrame, column: str, stats: Dict[str, float], 
                   std_multiplier: float = 2.5) -> pd.DataFrame:
    """
    Handle outliers for a single column using statistical methods.
    
    Args:
        df: Input DataFrame
        column: Column name to process
        stats: Dictionary containing 'mean', 'std', and 'median' for the column
        std_multiplier: Number of standard deviations to use for bounds
    
    Returns:
        DataFrame with outliers handled
    """
    df = df.copy()
    
    # Calculate bounds
    mean, std = stats['mean'], stats['std']
    lower_bound = max(0, mean - std_multiplier * std)
    upper_bound = mean + std_multiplier * std
    
    # Handle missing values
    if df[column].isna().any():
        df[column] = df[column].fillna(stats['median'])
        logger.info(f"Filled {df[column].isna().sum()} missing values in {column}")
    
    # Handle outliers
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_count = outlier_mask.sum()
    if outlier_count > 0:
        df.loc[outlier_mask, column] = df.loc[outlier_mask, column].clip(lower_bound, upper_bound)
        logger.info(f"Handled {outlier_count} outliers in {column}")
    
    return df

def calculate_time_features(timestamp: pd.Timestamp) -> Dict[str, float]:
    """
    Calculate time-based features from a timestamp.
    
    Args:
        timestamp: Input timestamp
    
    Returns:
        Dictionary of time features
    """
    hour = timestamp.hour
    day = timestamp.dayofweek
    month = timestamp.month
    
    return {
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'day_sin': np.sin(2 * np.pi * day / 7),
        'day_cos': np.cos(2 * np.pi * day / 7),
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'is_weekend': int(day in WEEKEND_DAYS),
        'is_business_hours': int(hour in BUSINESS_HOURS),
        'is_peak_hours': int(hour in PEAK_HOURS)
    }

def calculate_engagement_score(metrics: Dict[str, float]) -> float:
    """
    Calculate normalized engagement score from engagement metrics.
    
    Args:
        metrics: Dictionary containing 'likes', 'comments', and 'shares'
    
    Returns:
        Normalized engagement score between 0 and 1
    """
    try:
        # Calculate relative metrics
        likes_score = metrics['likes'] / ENGAGEMENT_BASELINES['likes']['median'] if metrics['likes'] > 0 else 0
        comments_score = metrics['comments'] / ENGAGEMENT_BASELINES['comments']['median'] if metrics['comments'] > 0 else 0
        shares_score = metrics['shares'] / ENGAGEMENT_BASELINES['shares']['median'] if metrics['shares'] > 0 else 0
        
        # Weights based on inverse of standard deviation
        likes_weight = 1.0
        comments_weight = ENGAGEMENT_BASELINES['likes']['std'] / ENGAGEMENT_BASELINES['comments']['std']
        shares_weight = ENGAGEMENT_BASELINES['likes']['std'] / ENGAGEMENT_BASELINES['shares']['std']
        
        # Calculate weighted score
        total_weight = likes_weight + comments_weight + shares_weight
        engagement_score = (
            (likes_score * likes_weight +
             comments_score * comments_weight +
             shares_score * shares_weight) / total_weight
        )
        
        # Normalize using sigmoid function
        return 1 / (1 + np.exp(-engagement_score + 1))
        
    except Exception as e:
        logger.error(f"Error calculating engagement score: {str(e)}")
        return 0.0

def validate_categorical_value(value: str, valid_categories: List[str], feature_name: str) -> str:
    """
    Validate and clean categorical values.
    
    Args:
        value: Input categorical value
        valid_categories: List of valid categories
        feature_name: Name of the feature for logging
    
    Returns:
        Cleaned and validated category string
    """
    if pd.isna(value) or value == '':
        logger.warning(f"Missing value for {feature_name}, using 'unknown'")
        return 'unknown'
    
    clean_value = str(value).strip()
    if clean_value not in valid_categories:
        logger.warning(f"Invalid {feature_name} value: {clean_value}, using 'unknown'")
        return 'unknown'
    
    return clean_value

def calculate_content_score(sentiment_score: float, hashtag_count: int,
                          sentiment_mean: float, sentiment_std: float,
                          hashtag_mean: float, hashtag_std: float) -> float:
    """
    Calculate content quality score based on sentiment and hashtags.
    
    Args:
        sentiment_score: Post sentiment score
        hashtag_count: Number of hashtags
        sentiment_mean: Mean sentiment score from training data
        sentiment_std: Standard deviation of sentiment scores
        hashtag_mean: Mean hashtag count from training data
        hashtag_std: Standard deviation of hashtag counts
    
    Returns:
        Normalized content quality score
    """
    sentiment_z = (sentiment_score - sentiment_mean) / sentiment_std if sentiment_std > 0 else 0
    hashtag_z = (hashtag_count - hashtag_mean) / hashtag_std if hashtag_std > 0 else 0
    
    return sentiment_z * 0.6 + hashtag_z * 0.4

def normalize_budget(budget: float, mean_budget: float, std_budget: float) -> float:
    """
    Normalize budget values using z-score normalization.
    
    Args:
        budget: Input budget value
        mean_budget: Mean budget from training data
        std_budget: Standard deviation of budgets
    
    Returns:
        Normalized budget value
    """
    if std_budget > 0:
        return (budget - mean_budget) / std_budget
    return 0.0 