"""
Train and validate the predictive model for cat marketing analytics
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import logging
import joblib
import os
from engineer_features import FeatureEngineer
from model_trainer import CatMarketingPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(campaigns_path: str, engagement_path: str, posts_path: str, 
                model_dir: str = 'models') -> None:
    """Train a model for predicting engagement using improved metrics"""
    try:
        # Clean up and recreate model directory
        if os.path.exists(model_dir):
            # Remove all files in the directory
            for file in os.listdir(model_dir):
                file_path = os.path.join(model_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.warning(f"Error cleaning up {file_path}: {e}")
            
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize predictor
        predictor = CatMarketingPredictor()
        
        # Load datasets
        campaigns_df = pd.read_csv(campaigns_path)
        engagement_df = pd.read_csv(engagement_path)
        posts_df = pd.read_csv(posts_path)
        
        # Engineer features using predictor's feature engineer
        features_df = predictor.feature_engineer.engineer_features(
            campaigns_df=campaigns_df,
            engagement_df=engagement_df,
            posts_df=posts_df
        )
        
        # Split features and target
        target_col = 'engagement_rate'
        feature_cols = [col for col in features_df.columns if col != target_col]
        
        X = features_df[feature_cols]
        y = features_df[target_col]
        
        # Update model info with feature names
        predictor.model_info['feature_names'] = feature_cols
        predictor.model_info['n_samples'] = len(X)
        predictor.model_info['last_training_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Split data with stratification on binned engagement rates
        y_binned = pd.qcut(y, q=5, labels=False)  # 5 equal-sized bins
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y_binned
        )
        
        # Initialize model with carefully tuned parameters
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        
        # Create sample weights to balance training
        median_engagement = y_train.median()
        sample_weights = 1 + np.abs(y_train - median_engagement)
        sample_weights /= sample_weights.mean()  # Normalize weights
        
        # Train model
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Evaluate model
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_r2': r2_score(y_train, train_preds),
            'test_r2': r2_score(y_test, test_preds),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_preds))
        }
        
        # Update predictor with metrics
        predictor.update_metrics(metrics)
        
        # Log metrics
        logger.info("Model Performance:")
        logger.info(f"Train R² Score: {metrics['train_r2']:.3f}")
        logger.info(f"Test R² Score: {metrics['test_r2']:.3f}")
        logger.info(f"Train RMSE: {metrics['train_rmse']:.3f}")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.3f}")
        
        # Analyze feature importance by groups
        feature_groups = {
            'Temporal': ['hour_', 'day_', 'month_', 'weekend', 'business_hours', 'peak_hours'],
            'Content': ['sentiment', 'hashtag', 'content_score', 'likes_ratio', 'comments_ratio', 'shares_ratio'],
            'Campaign': ['budget', 'platform_', 'campaign_goal_'],
            'Interaction': ['_business', '_weekend', '_peak']
        }
        
        # Calculate group importance
        group_importance = {}
        for group, patterns in feature_groups.items():
            group_importance[group] = sum(
                imp for feat, imp in zip(X.columns, model.feature_importances_)
                if any(pattern in feat for pattern in patterns)
            )
            logger.info(f"{group}: {group_importance[group]:.3%}")
        
        # Get individual feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Update predictor with feature importance
        predictor.update_feature_importance(feature_importance)
        
        # Update predictor with model
        predictor.update_model('stacking_model', model)
        
        # Save predictor (which includes model and feature engineer)
        model_path = os.path.join(model_dir, 'model.joblib')
        predictor.save_model(model_path)
        
        logger.info("Model training and saving completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model(
        campaigns_path='data/raw/catfluencer_campaigns.csv',
        engagement_path='data/raw/catfluencer_engagement.csv',
        posts_path='data/raw/catfluencer_posts.csv'
    )