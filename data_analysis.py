# Core imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set simple, clean plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

def load_data():
    """
    Load all datasets with proper date parsing
    Returns:
        Tuple of (campaigns_df, engagement_df, posts_df)
    """
    campaigns_df = pd.read_csv('catfluencer_campaigns.csv', parse_dates=['start_date', 'end_date'])
    engagement_df = pd.read_csv('catfluencer_engagement.csv', parse_dates=['timestamp'])
    posts_df = pd.read_csv('catfluencer_posts.csv', parse_dates=['timestamp'])
    return campaigns_df, engagement_df, posts_df

def clean_data(df, date_columns=None):
    """
    Comprehensive data cleaning function
    Args:
        df: DataFrame to clean
        date_columns: List of date columns to process
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Handle date columns
    if date_columns:
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Replace negative values with NaN
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"Found {neg_count} negative values in {col}")
            df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
        
        # Handle outliers using IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"Found {outliers} outliers in {col}")
            df[col] = df[col].apply(lambda x: np.nan if x < lower_bound or x > upper_bound else x)
        
        # Fill NaN with median
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"Filling {nan_count} NaN values in {col} with median")
            df[col] = df[col].fillna(df[col].median())
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Convert to lowercase and strip whitespace
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower().str.strip()
        
        # Fill NaN with "not specified"
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"Filling {nan_count} NaN values in {col} with 'not specified'")
            df[col] = df[col].fillna("not specified")
    
    return df

def get_basic_stats(campaigns_df, posts_df):
    """
    Get basic statistics about campaigns and posts
    Args:
        campaigns_df: Campaigns DataFrame
        posts_df: Posts DataFrame
    Returns:
        Dictionary of statistics
    """
    stats = {
        'total_campaigns': len(campaigns_df),
        'avg_budget': campaigns_df['budget'].mean(),
        'common_platform': campaigns_df['platform'].mode()[0],
        'common_goal': campaigns_df['campaign_goal'].mode()[0],
        'total_posts': len(posts_df),
        'avg_likes': posts_df['likes'].mean(),
        'avg_comments': posts_df['comments'].mean(),
        'avg_shares': posts_df['shares'].mean()
    }
    return stats

def analyze_content_performance(posts_df):
    """
    Analyze content type performance
    Args:
        posts_df: Posts DataFrame
    Returns:
        DataFrame with content performance metrics
    """
    content_performance = posts_df.groupby('content_type').agg({
        'likes': ['mean', 'count'],
        'comments': 'mean',
        'shares': 'mean'
    }).round(1)

    content_performance.columns = ['Avg Likes', 'Post Count', 'Avg Comments', 'Avg Shares']
    content_performance = content_performance.sort_values('Avg Likes', ascending=False)
    return content_performance

def plot_content_performance(content_performance):
    """
    Create visualization of content performance
    Args:
        content_performance: Content performance DataFrame
    """
    plt.figure(figsize=(12, 6))
    content_performance_melted = content_performance.reset_index().melt(
        id_vars=['content_type'],
        value_vars=['Avg Likes', 'Avg Comments', 'Avg Shares'],
        var_name='Metric',
        value_name='Count'
    )

    # Create bar plot
    ax = sns.barplot(
        data=content_performance_melted,
        x='content_type',
        y='Count',
        hue='Metric'
    )

    # Customize plot
    plt.title('Average Engagement by Content Type', pad=20)
    plt.xlabel('Content Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f')

    plt.tight_layout()
    plt.show()

def process_all_data():
    """
    Load, clean, and process all data
    Returns:
        Tuple of processed DataFrames (campaigns_df, engagement_df, posts_df)
    """
    # Load data
    campaigns_df, engagement_df, posts_df = load_data()
    
    # Clean datasets with appropriate date columns
    print("\nCleaning Campaigns Dataset:")
    campaigns_df = clean_data(campaigns_df, date_columns=['start_date', 'end_date'])

    print("\nCleaning Engagement Dataset:")
    engagement_df = clean_data(engagement_df, date_columns=['timestamp'])

    print("\nCleaning Posts Dataset:")
    posts_df = clean_data(posts_df, date_columns=['timestamp'])
    
    return campaigns_df, engagement_df, posts_df

if __name__ == "__main__":
    # Example usage when run as a script
    campaigns_df, engagement_df, posts_df = process_all_data()
    
    # Get and print basic stats
    stats = get_basic_stats(campaigns_df, posts_df)
    print("\nData Analysis Summary")
    print("=" * 50)
    print(f"Campaign Statistics:")
    print(f"- Total campaigns: {stats['total_campaigns']:,}")
    print(f"- Average campaign budget: ${stats['avg_budget']:,.2f}")
    print(f"- Most common platform: {stats['common_platform']}")
    print(f"- Most common campaign goal: {stats['common_goal']}")
    
    print(f"\nPost Statistics:")
    print(f"- Total posts analyzed: {stats['total_posts']:,}")
    print(f"- Average engagement:")
    print(f"  • Likes: {stats['avg_likes']:,.1f}")
    print(f"  • Comments: {stats['avg_comments']:,.1f}")
    print(f"  • Shares: {stats['avg_shares']:,.1f}")
    
    # Analyze and visualize content performance
    content_performance = analyze_content_performance(posts_df)
    print("\nContent Type Performance:")
    print("=" * 50)
    print(content_performance)
    plot_content_performance(content_performance)