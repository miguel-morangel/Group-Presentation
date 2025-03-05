# Engage  with  the  dataset,  develop  a  predictive  model,  find  significant  insights, and provide 
# visuals  that  can  explain  the  findings.  The  data  visualization  has to be clear and distinct to 
# enhance the presentation of the findings effectively. 
# ●  Presenting Data Insights (Specifications for the explanations of the model results). 
# ○  Make Business Recommendations: translate analytics into actionable decisions. 
# ○  Develop Model and deploy it: do this like companies do when they put models 
# into production using tools like Streamlit. 
# ○  The business relies on data-driven decisions, so insights are only helpful if they 
# are communicated clearly. 
# ○  Stakeholders need to understand why a model was chosen, its reliability, and what 
# actions to take based on the results.* 
# ○  Knowing  how  to  present  insights  effectively  is  as  important  as  building  the 
# models themselves

from model_trainer import CatMarketingPredictor
import logging
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_business_insights(predictor: CatMarketingPredictor) -> list:
    """Generate business insights from the model"""
    insights = []
    
    # Get feature importance
    importance_dict = predictor.get_feature_importance()
    if importance_dict:
        # Convert dictionary to sorted list of tuples
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]  # Get top 5 features
        
        insights.append({
            'title': 'Key Performance Drivers',
            'description': 'Most influential features in predicting engagement',
            'items': [f"{feat}: {imp:.2%}" for feat, imp in sorted_features]
        })
    
    # Get model metrics
    metrics = predictor.get_model_metrics()
    if metrics:
        insights.append({
            'title': 'Model Performance',
            'description': f"Overall Model Performance: {metrics.get('performance', 0):.2%}",
            'items': [
                f"Prediction Accuracy: {metrics.get('accuracy', 0):.2%}",
                f"Average Error Rate: {metrics.get('mean_error', 0):.3f}"
            ]
        })
    
    # Get model info
    model_info = predictor.get_model_info()
    if model_info:
        insights.append({
            'title': 'Model Information',
            'description': f"Model Version: {model_info.get('version', 'N/A')}",
            'items': [
                f"Last Training: {model_info.get('last_training_date', 'N/A')}",
                f"Training Samples: {model_info.get('n_samples', 0):,}",
                f"Features Used: {len(model_info.get('feature_names', []))}"
            ]
        })
    
    # Get engagement stats if available
    if hasattr(predictor.feature_engineer, 'engagement_stats'):
        stats = predictor.feature_engineer.engagement_stats
        insights.append({
            'title': 'Engagement Patterns',
            'description': 'Key engagement metrics and patterns',
            'items': [
                f"Optimal Campaign Duration: {stats.get('campaign_duration_mean', 7):.1f} days",
                f"Average Budget: ${stats.get('budget_mean', 0):,.2f}",
                f"Average Hashtag Count: {stats.get('hashtag_mean', 0):.1f}"
            ]
        })
    
    return insights

def main():
    """
    Main function to create and analyze the model
    """
    logger.info("Starting model creation and analysis")
    
    # Initialize predictor
    predictor = CatMarketingPredictor()
    
    # Load data
    data_dir = Path('data/raw')
    campaigns_df = pd.read_csv(data_dir / 'catfluencer_campaigns.csv')
    engagement_df = pd.read_csv(data_dir / 'catfluencer_engagement.csv')
    posts_df = pd.read_csv(data_dir / 'catfluencer_posts.csv')
    
    # Print summary statistics
    print("\nData Analysis Summary")
    print("=" * 50)
    print(f"Campaign Statistics:")
    print(f"- Total campaigns: {len(campaigns_df):,}")
    print(f"- Average campaign budget: ${campaigns_df['budget'].mean():,.2f}")
    print(f"- Most common platform: {campaigns_df['platform'].mode()[0]}")
    print(f"- Most common campaign goal: {campaigns_df['campaign_goal'].mode()[0]}")

    print(f"\nPost Statistics:")
    print(f"- Total posts analyzed: {len(posts_df):,}")
    print(f"- Average engagement:")
    print(f"  • Likes: {posts_df['likes'].mean():,.1f}")
    print(f"  • Comments: {posts_df['comments'].mean():,.1f}")
    print(f"  • Shares: {posts_df['shares'].mean():,.1f}")

if __name__ == "__main__":
    main()
