import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

def initialize_nlp():
    """
    Initialize NLP resources
    """
    nltk.download('stopwords', quiet=True)

def clean_text(text):
    """
    Clean and preprocess text data
    Args:
        text: String text to clean
    Returns:
        Cleaned text string
    """
    text = str(text).lower()
    return ' '.join([word for word in text.split() 
                    if word not in stopwords.words('english')])

def prepare_engagement_data(posts_df):
    """
    Prepare post data for NLP analysis
    Args:
        posts_df: DataFrame containing post data
    Returns:
        DataFrame with processed text features
    """
    processed_posts = posts_df.copy()
    
    # Clean captions
    processed_posts['clean_caption'] = processed_posts['caption'].apply(clean_text)
    
    # Calculate caption length
    processed_posts['caption_length'] = processed_posts['caption'].str.len().fillna(0)
    
    # Calculate total engagement
    processed_posts['total_engagement'] = (
        processed_posts['likes'].fillna(0) + 
        processed_posts['comments'].fillna(0) + 
        processed_posts['shares'].fillna(0)
    )
    
    return processed_posts

def analyze_caption_impact(posts_df, max_features=1000, top_n=15):
    """
    Analyze the impact of words on engagement
    Args:
        posts_df: DataFrame with post data
        max_features: Maximum number of features for TF-IDF
        top_n: Number of top words to return
    Returns:
        Dictionary containing analysis results
    """
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=max_features)
    text_features = vectorizer.fit_transform(posts_df['clean_caption'])
    
    # Fit model
    model = LinearRegression()
    model.fit(text_features.toarray(), posts_df['total_engagement'])
    
    # Get top words
    importance = np.abs(model.coef_).astype(float)
    words = vectorizer.get_feature_names_out()
    top_words = sorted(zip(importance, words), reverse=True)[:top_n]
    
    return {
        'importance': importance,
        'words': words,
        'top_words': top_words,
        'vectorizer': vectorizer,
        'model': model
    }

def create_word_impact_plot(top_words):
    """
    Create visualization of word impact on engagement
    Args:
        top_words: List of (importance, word) tuples
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(12, 6))
    words = [word for _, word in top_words]
    scores = np.array([score for score, _ in top_words])

    # Create horizontal bars
    plt.barh(range(len(words)), scores, color='#3498db', alpha=0.6)
    plt.yticks(range(len(words)), words)
    plt.xlabel('Impact on Engagement')
    plt.title('Words That Drive the Most Engagement')

    # Add value labels
    max_score = float(scores.max())
    for i, score in enumerate(scores):
        if score > max_score * 0.3:
            # Inside bar
            plt.text(score - max_score * 0.02, i, f'{int(score):,}', 
                    va='center', ha='right', color='white')
        else:
            # Outside bar
            plt.text(score + max_score * 0.02, i, f'{int(score):,}', 
                    va='center', ha='left', color='black')

    plt.grid(axis='x', linestyle='--', alpha=0.2)
    plt.tight_layout()
    return plt.gcf()

def create_word_cloud(words, scores):
    """
    Create engagement-weighted word cloud
    Args:
        words: List of words
        scores: List of importance scores
    Returns:
        WordCloud object
    """
    word_freq = dict(zip(words, scores))
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='YlOrRd',
        max_words=30,
        prefer_horizontal=0.7,
        min_font_size=12
    ).generate_from_frequencies(word_freq)
    return wordcloud

def get_top_performing_posts(posts_df, n=5):
    """
    Get the top performing posts by total engagement
    Args:
        posts_df: DataFrame with post data
        n: Number of top posts to return
    Returns:
        DataFrame with top posts
    """
    return posts_df.nlargest(n, 'total_engagement')[
        ['caption', 'likes', 'comments', 'shares', 'total_engagement']
    ]

def format_post_metrics(post):
    """
    Format post metrics for display
    Args:
        post: Series containing post data
    Returns:
        Tuple of (caption, metrics_string)
    """
    metrics = f"Likes: {int(post['likes']):,}, Comments: {int(post['comments']):,}, Shares: {int(post['shares']):,}"
    caption = post['caption'][:100] + "..." if len(post['caption']) > 100 else post['caption']
    return caption, metrics

if __name__ == "__main__":
    # Example usage
    initialize_nlp()
    
    # Assuming posts_df is loaded
    posts_df = prepare_engagement_data(posts_df)
    
    # Analyze caption impact
    analysis_results = analyze_caption_impact(posts_df)
    
    # Create visualizations
    impact_fig = create_word_impact_plot(analysis_results['top_words'])
    impact_fig.show()
    
    # Print top words
    print("\nMost Impactful Words in Captions:")
    for score, word in analysis_results['top_words']:
        print(f"{word}: {int(score):,}")
    
    # Generate and display word cloud
    print("\nGenerating engagement-weighted word cloud...")
    wordcloud = create_word_cloud(analysis_results['words'], analysis_results['importance'])
    
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Words Sized by Their Impact on Engagement', pad=20)
    plt.tight_layout()
    plt.show()
    
    # Show top performing posts
    print("\nTop 5 Most Engaging Posts:")
    top_posts = get_top_performing_posts(posts_df)
    for _, post in top_posts.iterrows():
        caption, metrics = format_post_metrics(post)
        print(f"\n{metrics}")
        print(f"Caption: {caption}")
