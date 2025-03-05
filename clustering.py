"""
This module provides functionality to group social media posts by their engagement levels using K-means clustering.
It analyzes posts based on their likes, comments, and shares to identify distinct groups of posts with similar engagement patterns.

The posts are automatically grouped into three engagement levels:
- High Engagement: Posts with the highest combined likes, comments, and shares
- Medium Engagement: Posts with moderate levels of engagement
- Low Engagement: Posts with the lowest combined likes, comments, and shares

The grouping is determined by analyzing the total engagement (sum of likes, comments, and shares) of each post.
Posts are clustered using K-means algorithm, which groups them based on their engagement patterns.
The groups are then labeled as High, Medium, or Low based on their average total engagement.
This helps identify which posts are performing best and understand the distribution of engagement across your content.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def print_engagement_group_stats(posts_df):
    """
    Print clear statistics about each engagement level group.
    
    Args:
        posts_df: DataFrame with engagement_level column
    """
    print("\nEngagement Level Groups Analysis:")
    print("=================================")
    
    for level in ["High", "Medium", "Low"]:
        group_data = posts_df[posts_df["engagement_level"] == level]
        print(f"\n{level} Engagement Group:")
        print(f"Number of posts: {len(group_data)}")
        print(f"Average likes: {group_data['likes'].mean():.0f}")
        print(f"Average comments: {group_data['comments'].mean():.0f}")
        print(f"Average shares: {group_data['shares'].mean():.0f}")
        print(f"Total engagement (likes + comments + shares): {group_data[['likes', 'comments', 'shares']].sum(axis=1).mean():.0f}")
        print("-" * 40)

def perform_clustering(posts_df, n_clusters=3):
    """
    Group social media posts into clusters based on their engagement levels (likes, comments, shares).
    
    This function uses K-means clustering to identify groups of posts that have similar engagement patterns.
    Posts are grouped based on their normalized likes, comments, and shares counts.
    
    Args:
        posts_df: DataFrame containing posts data with columns: likes, comments, shares
        n_clusters: Number of engagement level groups to create (default 3)
    Returns:
        DataFrame with engagement_cluster and engagement_level columns added, where each post is assigned to a group
        based on its engagement level (High, Medium, or Low)
    """
    # Selecting the engagement-related data we care about
    engagement_features = posts_df[["likes", "comments", "shares"]]
    
    # Ensure all values are numeric
    engagement_features = engagement_features.apply(pd.to_numeric, errors='coerce')
    
    # Handle any NaN values with mean imputation
    engagement_features = engagement_features.fillna(engagement_features.mean())

    # Standardizing the numbers so they are on the same scale
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(engagement_features)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to the dataframe
    posts_df = posts_df.copy()
    posts_df["engagement_cluster"] = cluster_labels
    
    # Calculate mean engagement for each cluster to determine which is high/medium/low
    cluster_means = posts_df.groupby("engagement_cluster")[["likes", "comments", "shares"]].mean()
    cluster_means["total_engagement"] = cluster_means.sum(axis=1)
    
    # Sort clusters by total engagement and assign labels
    sorted_clusters = cluster_means.sort_values("total_engagement", ascending=False)
    engagement_levels = ["High", "Medium", "Low"]
    cluster_to_level = dict(zip(sorted_clusters.index, engagement_levels))
    
    # Add descriptive engagement level labels
    posts_df["engagement_level"] = posts_df["engagement_cluster"].map(cluster_to_level)
    
    # Print statistics about the groups
    print_engagement_group_stats(posts_df)
    
    return posts_df

def plot_clustering_analysis(posts_df):
    """
    Create visualizations to analyze how posts are grouped by engagement levels.
    
    Generates three plots:
    1. Elbow plot to determine optimal number of engagement level groups
    2. Scatter plot showing how posts are distributed across engagement levels
    3. Box plot comparing likes distribution across different engagement level groups
    
    Args:
        posts_df: DataFrame with engagement_cluster and engagement_level columns (from perform_clustering)
    Returns:
        tuple: (elbow_fig, scatter_fig, box_fig) - Three matplotlib figures showing engagement level analysis
    """
    # Clear any existing plots
    plt.close('all')
    
    # Finding the best number of groups (clusters) for our posts
    engagement_features = posts_df[["likes", "comments", "shares"]]
    engagement_features = engagement_features.apply(pd.to_numeric, errors='coerce')
    engagement_features = engagement_features.fillna(engagement_features.mean())
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(engagement_features)
    
    inertia = []  # This measures how well the groups are formed
    k_range = range(1, 11)  # Testing 1 to 10 clusters
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

    # Plotting a graph to find the best number of clusters
    elbow_fig = plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='--', color='b')
    plt.xlabel("Number of Engagement Level Groups")
    plt.ylabel("Clustering Quality (Lower is Better)")
    plt.title("Determining Optimal Number of Engagement Level Groups")
    plt.grid(True)
    plt.xticks(k_range)
    plt.annotate("Recommended: 3 Groups", xy=(3, inertia[2]), xytext=(4, inertia[2] + 1000),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)
    elbow_fig.tight_layout()

    # Scatter plot showing how posts are grouped
    scatter_fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=posts_df, x="likes", y="comments", 
                    hue="engagement_level", palette="viridis", 
                    alpha=0.7, edgecolor='black')
    plt.xlabel("Number of Likes", fontsize=12)
    plt.ylabel("Number of Comments", fontsize=12)
    plt.title("Posts Grouped by Engagement Levels", fontsize=14)
    plt.legend(title="Engagement Level", loc='upper left')
    plt.grid(True)
    scatter_fig.tight_layout()

    # Boxplot to show how likes are distributed in each group
    box_fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=posts_df, x="engagement_level", y="likes",
                hue="engagement_level", palette="Set2", dodge=False)
    plt.xlabel("Engagement Level", fontsize=12)
    plt.ylabel("Number of Likes", fontsize=12)
    plt.title("Likes Distribution Across Engagement Levels", fontsize=14)
    plt.legend(title="Engagement Level", loc='upper right')
    plt.grid(True)
    box_fig.tight_layout()

    return elbow_fig, scatter_fig, box_fig