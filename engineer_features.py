import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
from typing import Dict, List, Tuple, Any
import joblib
import os
from data_utils import (
    handle_outliers,
    calculate_time_features,
    calculate_engagement_score,
    validate_categorical_value,
    calculate_content_score,
    normalize_budget,
    ENGAGEMENT_BASELINES,
    PEAK_HOURS,
    BUSINESS_HOURS,
    WEEKEND_DAYS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.categorical_features = {}
        self.numeric_features = []
        self.scalers = {}
        self.engagement_stats = {}
        self.last_features = None  # Store last engineered features
        # Add missing attributes with default values
        self.caption_feature_names = []  # We're not using text features in this version
        self.vectorizer = None
        self.caption_impact = {}
        
    def save_state(self, path: str) -> None:
        """Save the feature engineer state
        
        Args:
            path: Path to save the state to
        """
        try:
            state = {
                'scalers': self.scalers,
                'categorical_features': self.categorical_features,
                'numerical_features': self.numeric_features,
                'feature_stats': self.engagement_stats,
                'last_features': self.last_features
            }
            joblib.dump(state, path)
            logger.info(f"Feature engineer state saved to {path}")
        except Exception as e:
            logger.error(f"Error saving feature engineer state: {e}")
            raise

    def load_state(self, path: str) -> None:
        """Load fitted transformers and statistics
        
        Args:
            path: Path to the state file
        """
        try:
            if not os.path.exists(path):
                raise ValueError(f"No saved state found at {path}")
                
            state = joblib.load(path)
            self.scalers = state['scalers']
            self.categorical_features = state['categorical_features']
            self.numeric_features = state['numerical_features']
            self.engagement_stats = state.get('feature_stats', {})
            self.last_features = state.get('last_features', None)
            
            logger.info(f"Feature engineer state loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading feature engineer state: {e}")
            raise

    def handle_outliers(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """Handle outliers based on actual data distributions"""
        df_clean = df.copy()
        stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # Log initial stats
            logger.info(f"\nProcessing {col}:")
            logger.info(f"Missing values: {df[col].isna().sum()}")
            logger.info(f"Initial mean: {df[col].mean():.2f}")
            logger.info(f"Initial median: {df[col].median():.2f}")
            
            # Use shared utility for outlier handling
            df_clean = handle_outliers(df_clean, col, ENGAGEMENT_BASELINES[col])
            stats[col] = ENGAGEMENT_BASELINES[col]
            
            # Log final stats
            logger.info(f"Final mean: {df_clean[col].mean():.2f}")
            logger.info(f"Final median: {df_clean[col].median():.2f}")
            
        return df_clean, stats

    def calculate_engagement_metrics(self, row: pd.Series, stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate engagement metrics based on actual data distributions"""
        try:
            metrics = {
                'likes': float(row['likes']),
                'comments': float(row['comments']),
                'shares': float(row['shares'])
            }
            
            # Use shared utility for engagement score calculation
            engagement_rate = calculate_engagement_score(metrics)
            
            return {
                'engagement_rate': engagement_rate,
                'likes_ratio': metrics['likes'] / ENGAGEMENT_BASELINES['likes']['median'] if metrics['likes'] > 0 else 0,
                'comments_ratio': metrics['comments'] / ENGAGEMENT_BASELINES['comments']['median'] if metrics['comments'] > 0 else 0,
                'shares_ratio': metrics['shares'] / ENGAGEMENT_BASELINES['shares']['median'] if metrics['shares'] > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating engagement metrics: {str(e)}")
            return {
                'engagement_rate': 0.0,
                'likes_ratio': 0.0,
                'comments_ratio': 0.0,
                'shares_ratio': 0.0
            }

    def engineer_features(self, campaigns_df: pd.DataFrame, 
                         engagement_df: pd.DataFrame, 
                         posts_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features with proper handling of engagement metrics"""
        try:
            # Validate input data
            if any(df.empty for df in [campaigns_df, engagement_df, posts_df]):
                raise ValueError("Empty DataFrame provided")
            
            # Create working copies
            posts_df = posts_df.copy()
            campaigns_df = campaigns_df.copy()
            engagement_df = engagement_df.copy()
            
            # Convert all timestamps and dates
            for df in [posts_df, engagement_df]:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            campaigns_df['start_date'] = pd.to_datetime(campaigns_df['start_date'])
            campaigns_df['end_date'] = pd.to_datetime(campaigns_df['end_date'])
            
            # Calculate campaign duration features
            campaigns_df['campaign_duration'] = (campaigns_df['end_date'] - campaigns_df['start_date']).dt.days
            
            # Handle engagement metrics first
            engagement_cols = ['likes', 'comments', 'shares']
            posts_df, engagement_stats = self.handle_outliers(posts_df, engagement_cols)
            
            # Calculate engagement metrics
            engagement_data = posts_df.apply(
                lambda row: self.calculate_engagement_metrics(row, engagement_stats), 
                axis=1
            )
            
            # Extract engagement metrics
            posts_df['engagement_rate'] = engagement_data.apply(lambda x: x['engagement_rate'])
            posts_df['likes_ratio'] = engagement_data.apply(lambda x: x['likes_ratio'])
            posts_df['comments_ratio'] = engagement_data.apply(lambda x: x['comments_ratio'])
            posts_df['shares_ratio'] = engagement_data.apply(lambda x: x['shares_ratio'])
            
            # Log engagement rate distribution
            logger.info("\nEngagement Rate Distribution:")
            logger.info(f"Min: {posts_df['engagement_rate'].min():.3f}")
            logger.info(f"Max: {posts_df['engagement_rate'].max():.3f}")
            logger.info(f"Mean: {posts_df['engagement_rate'].mean():.3f}")
            logger.info(f"Median: {posts_df['engagement_rate'].median():.3f}")
            
            # Handle target variable
            target = posts_df['engagement_rate']
            target = (target - target.min()) / (target.max() - target.min())  # Normalize to [0,1]
            
            # Merge relevant features
            posts_df = posts_df.merge(
                engagement_df[['timestamp', 'sentiment_score', 'is_sponsored']],
                on='timestamp', 
                how='left',
                validate='1:1'
            )
            
            # Merge campaign features and calculate time-based campaign features
            campaign_features = [
                'campaign_id', 'platform', 'target_audience', 'campaign_goal', 
                'budget', 'start_date', 'end_date', 'campaign_duration'
            ]
            posts_df = posts_df.merge(
                campaigns_df[campaign_features],
                on='campaign_id',
                how='left',
                validate='m:1'
            )
            
            # Calculate campaign timing features
            posts_df['days_since_campaign_start'] = (posts_df['timestamp'] - posts_df['start_date']).dt.days
            posts_df['days_until_campaign_end'] = (posts_df['end_date'] - posts_df['timestamp']).dt.days
            posts_df['campaign_progress'] = posts_df['days_since_campaign_start'] / posts_df['campaign_duration']
            
            # Fill missing values and create derived features
            posts_df['sentiment_score'] = posts_df['sentiment_score'].fillna(posts_df['sentiment_score'].median())
            posts_df['hashtag_count'] = posts_df['hashtags'].str.count('#').fillna(0)
            
            # Store engagement statistics for prediction
            self.engagement_stats = {
                'sentiment_mean': posts_df['sentiment_score'].mean(),
                'sentiment_std': posts_df['sentiment_score'].std(),
                'hashtag_mean': posts_df['hashtag_count'].mean(),
                'hashtag_std': posts_df['hashtag_count'].std(),
                'budget_mean': posts_df['budget'].mean(),
                'budget_std': posts_df['budget'].std(),
                'campaign_duration_mean': posts_df['campaign_duration'].mean(),
                'campaign_duration_std': posts_df['campaign_duration'].std()
            }
            
            # Enhanced time features using shared utility
            time_features = posts_df['timestamp'].apply(calculate_time_features)
            for feature, values in pd.DataFrame(time_features.tolist()).items():
                posts_df[feature] = values
            
            # Calculate content score using shared utility
            posts_df['content_score'] = posts_df.apply(
                lambda row: calculate_content_score(
                    row['sentiment_score'],
                    row['hashtag_count'],
                    self.engagement_stats['sentiment_mean'],
                    self.engagement_stats['sentiment_std'],
                    self.engagement_stats['hashtag_mean'],
                    self.engagement_stats['hashtag_std']
                ),
                axis=1
            )
            
            # Define numeric features (updated list)
            self.numeric_features = [
                'budget', 'hashtag_count', 'sentiment_score', 'is_sponsored',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                'month_sin', 'month_cos',
                'is_weekend', 'is_business_hours', 'is_peak_hours',
                'likes_ratio', 'comments_ratio', 'shares_ratio',
                'campaign_duration', 'days_since_campaign_start',
                'days_until_campaign_end', 'campaign_progress',
                'content_score',
                'weekend_peak', 'sentiment_business', 'hashtag_weekend',
                'budget_peak', 'time_score'
            ]
            
            # Handle categorical features
            categorical_columns = {
                'platform': ['Instagram', 'Facebook', 'TikTok'],
                'content_type': sorted(posts_df['content_type'].unique().tolist()),
                'campaign_goal': ['Sales', 'Brand Awareness', 'Engagement'],
                'target_audience': sorted(posts_df['target_audience'].unique().tolist())
            }
            
            # Create dummies for categorical features
            for col, categories in categorical_columns.items():
                posts_df[col] = posts_df[col].fillna('unknown')
                dummies = pd.get_dummies(posts_df[col], prefix=col)
                expected_cols = [f"{col}_{cat}" for cat in categories]
                for expected_col in expected_cols:
                    if expected_col not in dummies.columns:
                        dummies[expected_col] = 0
                self.categorical_features[col] = expected_cols
                posts_df = pd.concat([posts_df, dummies[expected_cols]], axis=1)
            
            # Scale numeric features
            self.scalers = {}
            scaled_numeric = pd.DataFrame(index=posts_df.index)
            
            # First create all base features
            for col in self.numeric_features:
                if col not in ['weekend_peak', 'sentiment_business', 'hashtag_weekend', 'budget_peak', 'time_score']:
                    if col in posts_df.columns:
                        self.scalers[col] = RobustScaler()
                        scaled_values = self.scalers[col].fit_transform(posts_df[col].values.reshape(-1, 1))
                        scaled_numeric[col] = scaled_values.flatten()
            
            # Create interaction features in the exact same order as numeric_features
            scaled_numeric['weekend_peak'] = scaled_numeric['is_weekend'] * scaled_numeric['is_peak_hours']
            scaled_numeric['sentiment_business'] = scaled_numeric['sentiment_score'] * scaled_numeric['is_business_hours']
            scaled_numeric['hashtag_weekend'] = scaled_numeric['hashtag_count'] * scaled_numeric['is_weekend']
            scaled_numeric['budget_peak'] = scaled_numeric['budget'] * scaled_numeric['is_peak_hours']
            scaled_numeric['time_score'] = (
                0.4 * scaled_numeric['is_peak_hours'] +
                0.3 * scaled_numeric['is_business_hours'] +
                0.3 * (1 - scaled_numeric['is_weekend'])
            )
            
            # Scale interaction features
            for col in ['weekend_peak', 'sentiment_business', 'hashtag_weekend', 'budget_peak', 'time_score']:
                self.scalers[col] = RobustScaler()
                scaled_values = self.scalers[col].fit_transform(scaled_numeric[col].values.reshape(-1, 1))
                scaled_numeric[col] = scaled_values.flatten()
            
            # Combine features in exact order
            categorical_features = [col for cols in self.categorical_features.values() for col in cols]
            final_features = pd.concat([
                posts_df[categorical_features],
                scaled_numeric[self.numeric_features]  # Use numeric_features to maintain order
            ], axis=1)
            
            # Add target
            final_features['engagement_rate'] = target
            
            # Verify feature order
            expected_features = categorical_features + self.numeric_features
            assert list(final_features.columns[:-1]) == expected_features, "Feature order mismatch"
            
            logger.info(f"Feature engineering completed. Shape: {final_features.shape}")
            logger.info(f"Features in order: {list(final_features.columns)}")
            
            # Store the final features for later use
            self.last_features = final_features.copy()
            
            return final_features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise

    def prepare_prediction_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            # Initialize features dictionary
            features = {}
            
            # Get the exact feature order from training
            expected_features = (
                [col for cols in self.categorical_features.values() for col in cols] +
                self.numeric_features
            )
            
            # Process categorical features using shared utility
            for category, values in self.categorical_features.items():
                input_value = validate_categorical_value(
                    input_data.get(category, ''),
                    [v.split('_')[-1] for v in values],
                    category
                )
                for col in values:
                    features[col] = 1.0 if col.endswith(input_value) else 0.0
            
            # Process numeric features
            # Handle budget
            if 'budget' in input_data:
                budget = float(input_data['budget'])
                if 'budget' in self.scalers:
                    features['budget'] = self.scalers['budget'].transform([[budget]])[0][0]
                else:
                    features['budget'] = normalize_budget(
                        budget,
                        self.engagement_stats.get('budget_mean', 1000),
                        self.engagement_stats.get('budget_std', 500)
                    )
            
            # Handle sentiment and hashtags
            sentiment_score = float(input_data.get('sentiment_score', 0))
            hashtag_count = int(input_data.get('hashtag_count', 0))
            is_sponsored = int(input_data.get('is_sponsored', 0))
            
            # Scale features using stored scalers
            for feature, value in [
                ('sentiment_score', sentiment_score),
                ('hashtag_count', hashtag_count),
                ('is_sponsored', is_sponsored)
            ]:
                if feature in self.scalers:
                    features[feature] = self.scalers[feature].transform([[value]])[0][0]
                else:
                    features[feature] = value
            
            # Calculate content score using stored statistics
            features['content_score'] = calculate_content_score(
                sentiment_score,
                hashtag_count,
                self.engagement_stats.get('sentiment_mean', 0),
                self.engagement_stats.get('sentiment_std', 1),
                self.engagement_stats.get('hashtag_mean', 5),
                self.engagement_stats.get('hashtag_std', 2)
            )
            
            # Process time features using shared utility
            current_time = pd.Timestamp.now()
            time_features = calculate_time_features(current_time)
            features.update(time_features)
            
            # Handle campaign duration and timing features
            if 'start_date' in input_data and 'end_date' in input_data:
                start_date = pd.to_datetime(input_data['start_date'])
                end_date = pd.to_datetime(input_data['end_date'])
                
                # Calculate campaign duration
                campaign_duration = (end_date - start_date).days
                if 'campaign_duration' in self.scalers:
                    features['campaign_duration'] = self.scalers['campaign_duration'].transform([[campaign_duration]])[0][0]
                else:
                    features['campaign_duration'] = normalize_budget(
                        campaign_duration,
                        self.engagement_stats.get('campaign_duration_mean', 7),
                        self.engagement_stats.get('campaign_duration_std', 3)
                    )
                
                # Calculate days since start and until end
                days_since_start = (current_time - start_date).days
                days_until_end = (end_date - current_time).days
                
                # Scale these features if scalers exist
                if 'days_since_campaign_start' in self.scalers:
                    features['days_since_campaign_start'] = self.scalers['days_since_campaign_start'].transform([[days_since_start]])[0][0]
                else:
                    features['days_since_campaign_start'] = days_since_start
                    
                if 'days_until_campaign_end' in self.scalers:
                    features['days_until_campaign_end'] = self.scalers['days_until_campaign_end'].transform([[days_until_end]])[0][0]
                else:
                    features['days_until_campaign_end'] = days_until_end
                
                # Calculate campaign progress
                features['campaign_progress'] = days_since_start / campaign_duration if campaign_duration > 0 else 0
            else:
                # If no campaign dates provided, set defaults
                for feature in ['campaign_duration', 'days_since_campaign_start', 'days_until_campaign_end', 'campaign_progress']:
                    features[feature] = 0.0
            
            # Create interaction features
            features['weekend_peak'] = features['is_weekend'] * features['is_peak_hours']
            features['sentiment_business'] = features['sentiment_score'] * features['is_business_hours']
            features['hashtag_weekend'] = features['hashtag_count'] * features['is_weekend']
            features['budget_peak'] = features['budget'] * features['is_peak_hours']
            
            # Calculate time score (weighted combination of time features)
            features['time_score'] = (
                0.4 * features['is_peak_hours'] +
                0.3 * features['is_business_hours'] +
                0.3 * (1 - features['is_weekend'])  # Weekdays tend to have better engagement
            )
            
            # Add engagement ratios (set to 0 for prediction as they're not available)
            for ratio in ['likes_ratio', 'comments_ratio', 'shares_ratio']:
                features[ratio] = 0.0
            
            # Create DataFrame with exact feature order
            prediction_df = pd.DataFrame([features])
            
            # Ensure all expected features are present with default value 0
            for feature in expected_features:
                if feature not in prediction_df.columns:
                    prediction_df[feature] = 0.0
                    logger.warning(f"Missing feature {feature} set to 0")
            
            # Reorder columns to match training data exactly
            prediction_df = prediction_df[expected_features]
            
            logger.info(f"Prepared features for prediction. Shape: {prediction_df.shape}")
            logger.info(f"Features in order: {list(prediction_df.columns)}")
            
            # Store prediction features and feature names
            self.last_features = prediction_df.copy()
            self.feature_names = expected_features
            
            return prediction_df
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {str(e)}")
            raise