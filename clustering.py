import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering(posts_df, n_clusters=3):
    """
    Perform clustering on engagement metrics
    Args:
        posts_df: DataFrame containing posts data with likes, comments, shares
        n_clusters: Number of clusters to create (default 3)
    Returns:
        DataFrame with engagement_cluster column added
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
    
    return posts_df

def plot_clustering_analysis(posts_df):
    """
    Create visualizations for clustering analysis
    Args:
        posts_df: DataFrame with engagement_cluster column
    Returns:
        tuple: (elbow_fig, scatter_fig, box_fig) - Three matplotlib figures
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
    plt.xlabel("Number of Groups")
    plt.ylabel("Grouping Quality (Lower is Better)")
    plt.title("Finding the Best Number of Groups for Engagement")
    plt.grid(True)
    plt.xticks(k_range)
    plt.annotate("Best Choice Here", xy=(3, inertia[2]), xytext=(4, inertia[2] + 1000),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)
    elbow_fig.tight_layout()

    # Scatter plot showing how posts are grouped
    scatter_fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=posts_df, x="likes", y="comments", 
                    hue="engagement_cluster", palette="viridis", 
                    alpha=0.7, edgecolor='black')
    plt.xlabel("Likes on a Post", fontsize=12)
    plt.ylabel("Comments on a Post", fontsize=12)
    plt.title("Grouping Posts Based on Likes and Comments", fontsize=14)
    plt.legend(title="Group Number", loc='upper left')
    plt.grid(True)
    scatter_fig.tight_layout()

    # Boxplot to show how likes are distributed in each group
    box_fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=posts_df, x="engagement_cluster", y="likes",
                hue="engagement_cluster", palette="Set2", dodge=False)
    plt.xlabel("Group Number", fontsize=12)
    plt.ylabel("Number of Likes", fontsize=12)
    plt.title("How Different Groups Compare in Likes", fontsize=14)
    plt.legend(title="Group Number", loc='upper right')
    plt.grid(True)
    box_fig.tight_layout()

    return elbow_fig, scatter_fig, box_fig
