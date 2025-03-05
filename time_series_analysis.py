import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union

def prepare_time_features(engagement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare time-based features from engagement data
    Args:
        engagement_df: DataFrame with engagement data
    Returns:
        DataFrame with added time features
    """
    # Make a copy and ensure timestamp is datetime
    df = engagement_df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        df.set_index('timestamp', inplace=True)
    
    # Extract time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    df['is_peak_hours'] = df['hour'].isin([11, 12, 13, 17, 18, 19, 20]).astype(int)
    
    return df

def analyze_hourly_engagement(engagement_df: pd.DataFrame) -> pd.Series:
    """
    Analyze engagement patterns by hour
    Args:
        engagement_df: DataFrame with engagement data
    Returns:
        Series with hourly engagement rates
    """
    if 'engagement_rate' not in engagement_df.columns:
        engagement_df['engagement_rate'] = engagement_df['engagement_score']
    
    hourly_engagement = engagement_df.groupby('hour')['engagement_rate'].mean()
    return hourly_engagement

def analyze_daily_engagement(engagement_df: pd.DataFrame) -> pd.Series:
    """
    Analyze engagement patterns by day of week
    Args:
        engagement_df: DataFrame with engagement data
    Returns:
        Series with daily engagement rates
    """
    if 'engagement_rate' not in engagement_df.columns:
        engagement_df['engagement_rate'] = engagement_df['engagement_score']
    
    daily_engagement = engagement_df.groupby('day_of_week')['engagement_rate'].mean()
    return daily_engagement

def create_engagement_heatmap(engagement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create heatmap data for engagement patterns
    Args:
        engagement_df: DataFrame with engagement data
    Returns:
        DataFrame with heatmap data
    """
    if 'engagement_rate' not in engagement_df.columns:
        engagement_df['engagement_rate'] = engagement_df['engagement_score']
    
    return engagement_df.groupby(['day_of_week', 'hour'])['engagement_rate'].mean().unstack()

def calculate_rolling_average(engagement_df: pd.DataFrame, window: int = 24) -> pd.Series:
    """
    Calculate rolling average of engagement
    Args:
        engagement_df: DataFrame with engagement data
        window: Rolling window size in hours
    Returns:
        Series with rolling average
    """
    if 'engagement_rate' not in engagement_df.columns:
        engagement_df['engagement_rate'] = engagement_df['engagement_score']
    
    return engagement_df['engagement_rate'].rolling(window=window, min_periods=1).mean()

def plot_hourly_engagement(hourly_engagement: pd.Series) -> None:
    """
    Plot engagement by hour
    Args:
        hourly_engagement: Series with hourly engagement data
    """
    plt.figure(figsize=(10, 5))
    hour_labels = [f"{hour%12 or 12}{' AM' if hour < 12 else ' PM'}" for hour in hourly_engagement.index]
    
    sns.lineplot(x=hour_labels, y=hourly_engagement.values, marker="o")
    plt.title("Average Engagement Rate by Hour of the Day", pad=20)
    plt.xlabel("Hour of the Day")
    plt.ylabel("Engagement Rate")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_daily_engagement(daily_engagement: pd.Series) -> None:
    """
    Plot engagement by day of week
    Args:
        daily_engagement: Series with daily engagement data
    """
    plt.figure(figsize=(10, 5))
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    sns.barplot(
        x=days,
        y=daily_engagement.values,
        palette="Blues_r",
        edgecolor="black"
    )

    for index, value in enumerate(daily_engagement.values):
        plt.text(index, value + 0.001, f"{value:.3f}", ha='center', fontsize=10)

    plt.title("Average Engagement Rate by Day of the Week", pad=20)
    plt.xlabel("Day of the Week")
    plt.ylabel("Engagement Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()

def plot_engagement_heatmap(heatmap_data: pd.DataFrame) -> None:
    """
    Plot engagement heatmap
    Args:
        heatmap_data: DataFrame with heatmap data
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={'label': 'Engagement Rate'}
    )
    plt.title("Engagement Heatmap: Day of the Week vs Hour of the Day", pad=20)
    plt.xlabel("Hour of the Day")
    plt.ylabel("Day of the Week")
    
    # Update x-axis labels
    plt.xticks(
        ticks=range(24),
        labels=[f"{h%12 or 12}{' AM' if h < 12 else ' PM'}" for h in range(24)],
        rotation=45
    )
    
    # Update y-axis labels
    plt.yticks(
        ticks=range(7),
        labels=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        rotation=0
    )
    plt.tight_layout()

def get_optimal_posting_times(engagement_df: pd.DataFrame) -> Dict[str, Union[List[str], np.ndarray]]:
    """
    Get optimal posting times based on historical engagement
    Args:
        engagement_df: DataFrame with engagement data
    Returns:
        Dictionary with optimal posting times and their engagement rates
    """
    if 'engagement_rate' not in engagement_df.columns:
        engagement_df['engagement_rate'] = engagement_df['engagement_score']
    
    hourly_eng = analyze_hourly_engagement(engagement_df)
    daily_eng = analyze_daily_engagement(engagement_df)
    
    best_hours = hourly_eng.nlargest(3)
    best_days = daily_eng.nlargest(3)
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    return {
        'best_hours': [f"{h%12 or 12}{' AM' if h < 12 else ' PM'}" for h in best_hours.index],
        'best_days': [days[day] for day in best_days.index],
        'best_hours_rates': best_hours.values,
        'best_days_rates': best_days.values
    }

def plot_rolling_average(engagement_df: pd.DataFrame) -> None:
    """
    Plot rolling average engagement
    Args:
        engagement_df: DataFrame with engagement data
    """
    plt.figure(figsize=(12, 6))
    rolling_avg = calculate_rolling_average(engagement_df)
    
    plt.plot(rolling_avg.index, rolling_avg.values, linewidth=2)
    plt.title("24-Hour Rolling Average of Engagement Over Time", pad=20)
    plt.xlabel("Date")
    plt.ylabel("Engagement Rate")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

if __name__ == "__main__":
    # Example usage
    campaigns_df, engagement_df, posts_df = process_all_data()  # Assuming this function exists
    
    # Prepare time features
    engagement_df = prepare_time_features(engagement_df)
    
    # Analyze patterns
    hourly_engagement = analyze_hourly_engagement(engagement_df)
    daily_engagement = analyze_daily_engagement(engagement_df)
    heatmap_data = create_engagement_heatmap(engagement_df)
    
    # Generate plots
    plot_hourly_engagement(hourly_engagement)
    plot_daily_engagement(daily_engagement)
    plot_engagement_heatmap(heatmap_data)
    plot_rolling_average(engagement_df)
    
    # Get optimal posting times
    optimal_times = get_optimal_posting_times(engagement_df)
    print("\nOptimal Posting Times:")
    print(f"Best Hours: {', '.join(optimal_times['best_hours'])}")
    print(f"Best Days: {', '.join(optimal_times['best_days'])}")
