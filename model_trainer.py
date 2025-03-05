"""
Model trainer module for cat marketing analytics
"""
import logging
import joblib
import os
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from engineer_features import FeatureEngineer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CatMarketingPredictor:
    """Main class for managing the predictive modeling pipeline"""
    
    def __init__(self):
        self.models = {}
        self.feature_importances = {}
        self.shap_values = {}
        self.metrics = {}
        self.feature_engineer = FeatureEngineer()
        self.model_info = {
            'version': '1.0.0',
            'last_training_date': None,
            'n_samples': 0,
            'feature_names': []
        }
        
    def update_metrics(self, metrics: Dict[str, float]):
        """Update model metrics"""
        self.metrics.update(metrics)
        logger.info("Updated model metrics")
        
    def update_feature_importance(self, importance_df: pd.DataFrame):
        """Update feature importance"""
        self.feature_importances['feature_importance'] = importance_df
        logger.info("Updated feature importance")
        
    def update_model(self, model_name: str, model: Any):
        """Update stored model"""
        self.models[model_name] = model
        logger.info(f"Updated model: {model_name}")
        
    def get_model(self, model_name: str) -> Any:
        """Get stored model"""
        return self.models.get(model_name)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance as a dictionary"""
        if 'feature_importance' in self.feature_importances:
            df = self.feature_importances['feature_importance']
            return dict(zip(df['feature'], df['importance']))
        return {}
        
    def get_model_metrics(self) -> Dict[str, float]:
        """Get model performance metrics"""
        return {
            'performance': self.metrics.get('test_r2', 0),
            'accuracy': 1 - self.metrics.get('test_rmse', 0),
            'mean_error': self.metrics.get('test_rmse', 0)
        }
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info
        
    def get_feature_correlations(self) -> pd.DataFrame:
        """Get feature correlations"""
        try:
            model = self.get_model('stacking_model')
            if model is None:
                return pd.DataFrame()
                
            # Get feature names from feature engineer
            feature_names = (
                [col for cols in self.feature_engineer.categorical_features.values() for col in cols] +
                self.feature_engineer.numeric_features
            )
            
            # Create correlation matrix using model's feature importances
            correlations = np.outer(model.feature_importances_, model.feature_importances_)
            return pd.DataFrame(correlations, columns=feature_names, index=feature_names)
            
        except Exception as e:
            logger.error(f"Error getting feature correlations: {e}")
            return pd.DataFrame()
            
    def get_feature_distribution(self, feature_name: str) -> np.ndarray:
        """Get distribution data for a specific feature"""
        try:
            if hasattr(self.feature_engineer, 'last_features') and feature_name in self.feature_engineer.last_features:
                return self.feature_engineer.last_features[feature_name].values
            return np.array([])
        except Exception as e:
            logger.error(f"Error getting feature distribution: {e}")
            return np.array([])
            
    def get_feature_statistics(self, feature_name: str) -> Dict[str, float]:
        """Get statistics for a specific feature"""
        try:
            if hasattr(self.feature_engineer, 'last_features') and feature_name in self.feature_engineer.last_features:
                series = self.feature_engineer.last_features[feature_name]
                return {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'median': float(series.median())
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting feature statistics: {e}")
            return {}
            
    def get_key_takeaways(self) -> List[str]:
        """Get key insights and takeaways from the model"""
        try:
            takeaways = []
            
            # Get feature importance
            importance = self.get_feature_importance()
            if importance:
                # Sort features by importance
                sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
                
                # Add top feature insights
                top_features = sorted_features[:3]
                takeaways.append(
                    f"Top performing features are: {', '.join(f'{feat} ({imp:.1%})' for feat, imp in top_features)}"
                )
            
            # Add model performance insights
            metrics = self.get_model_metrics()
            if metrics:
                performance = metrics.get('performance', 0)
                accuracy = metrics.get('accuracy', 0)
                takeaways.append(
                    f"Model achieves {performance:.1%} R² score with {accuracy:.1%} accuracy"
                )
            
            # Add timing insights if available
            if hasattr(self.feature_engineer, 'engagement_stats'):
                stats = self.feature_engineer.engagement_stats
                takeaways.append(
                    f"Optimal campaign duration is {stats.get('campaign_duration_mean', 7):.1f} days"
                )
            
            return takeaways
            
        except Exception as e:
            logger.error(f"Error getting key takeaways: {e}")
            return []
            
    def save_model(self, path: str) -> None:
        """Save the model and feature engineer state
        
        Args:
            path: Path to save the model to
        """
        try:
            # Update model info before saving
            self.model_info['last_training_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            if hasattr(self.feature_engineer, 'last_features'):
                self.model_info['n_samples'] = len(self.feature_engineer.last_features)
                self.model_info['feature_names'] = list(self.feature_engineer.last_features.columns)
            
            # Save feature engineer state in the models directory
            model_dir = os.path.dirname(path)
            feature_engineer_path = os.path.join(model_dir, 'feature_engineer_state.joblib')
            
            # Save feature engineer state
            self.feature_engineer.save_state(feature_engineer_path)
            
            # Create state dictionary
            state = {
                'models': self.models,
                'metrics': self.metrics,
                'feature_importance': self.feature_importances['feature_importance'],
                'feature_engineer_path': feature_engineer_path,
                'model_info': self.model_info
            }
            
            # Save state
            joblib.dump(state, path)
            logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, filename: str):
        """Load model and associated artifacts"""
        try:
            if not os.path.exists(filename):
                raise ValueError(f"Model file not found: {filename}")
                
            model_artifacts = joblib.load(filename)
            
            # Load model artifacts
            self.models = model_artifacts['models']
            self.metrics = model_artifacts['metrics']
            self.feature_importances = {'feature_importance': model_artifacts['feature_importance']}
            self.model_info = model_artifacts.get('model_info', {
                'version': '1.0.0',
                'last_training_date': None,
                'n_samples': 0,
                'feature_names': []
            })
            
            # Load feature engineer state
            feature_engineer_path = model_artifacts['feature_engineer_path']
            if os.path.exists(feature_engineer_path):
                self.feature_engineer.load_state(feature_engineer_path)
            else:
                logger.warning(f"Feature engineer state not found at {feature_engineer_path}")
            
            logger.info(f"Model loaded successfully from {filename}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 