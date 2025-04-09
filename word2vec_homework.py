#!/usr/bin/env python
# coding: utf-8

# # Word2Vec Implementation, Visualization and Application

# ## Import necessary libraries
import gensim
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
import re
import urllib.request

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ## 1. Data Preparation
print("1. Data Preparation")

# Download and load a dataset (Amazon reviews)
print("Downloading and preparing dataset...")

# Download the dataset
url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Amazon-Fine-Food-Reviews/master/Reviews.csv"
urllib.request.urlretrieve(url, "amazon_reviews.csv")

# Load the dataset
df = pd.read_csv("amazon_reviews.csv")
print(f"Dataset loaded with {len(df)} reviews")

# Sample a smaller portion for faster processing
df = df.sample(5000, random_state=42)
print(f"Working with a sample of {len(df)} reviews")

# Preprocess the text
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        return tokens
    return []

# Create sentences for Word2Vec
sentences = df['Text'].apply(preprocess_text).tolist()
print(f"Preprocessed {len(sentences)} sentences for Word2Vec training")

# Remove empty lists
sentences = [s for s in sentences if len(s) > 0]
print(f"Final dataset contains {len(sentences)} non-empty sentences")

# ## 2. Word2Vec Model Implementation with Different Hyperparameters
print("\n2. Word2Vec Model Implementation")

# Function to train and save Word2Vec model
def train_word2vec(sentences, vector_size, window, min_count, workers, epochs, model_name):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    
    # Print model information
    vocab_size = len(model.wv.key_to_index)
    print(f"Model '{model_name}' trained with {vocab_size} words in vocabulary")
    
    # Example word vector
    if 'good' in model.wv.key_to_index:
        print(f"Example embedding for 'good':", model.wv['good'][:5], "...")
    
    return model

# Train models with different hyperparameters
print("\nTraining models with different hyperparameters:")

# Model 1: Default parameters
model1 = train_word2vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    epochs=5,
    model_name="Default"
)

# Model 2: Larger vector size
model2 = train_word2vec(
    sentences=sentences,
    vector_size=200,
    window=5,
    min_count=5,
    workers=4,
    epochs=5,
    model_name="Large vector size"
)

# Model 3: Larger window size
model3 = train_word2vec(
    sentences=sentences,
    vector_size=100,
    window=10,
    min_count=5,
    workers=4,
    epochs=5,
    model_name="Large window size"
)

# Select the best model for further analysis (using model1 as default)
best_model = model1
print("\nUsing the default model for further analysis")

# ## 3. Word Embeddings Visualization
print("\n3. Word Embeddings Visualization")

def visualize_embeddings(model, num_words=100, perplexity=30):
    # Get the most common words
    words = list(model.wv.key_to_index.keys())[:num_words]
    word_vectors = [model.wv[word] for word in words]
    
    # Apply t-SNE for dimensionality reduction
    print(f"Applying t-SNE on {len(words)} word vectors...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(word_vectors)
    
    # Visualization
    plt.figure(figsize=(12, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.5)
    
    # Annotate points with words
    for i, word in enumerate(words):
        plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 2), textcoords='offset points',
                    ha='right', va='bottom', fontsize=8)
    
    plt.title("t-SNE Visualization of Word Embeddings")
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.tight_layout()
    plt.savefig('word_embeddings_visualization.png')
    plt.show()

# Visualize word embeddings
visualize_embeddings(best_model, num_words=100, perplexity=30)

# ## 4. Word Similarity Analysis
print("\n4. Word Similarity Analysis")

def analyze_word_similarities(model, words=None):
    if words is None:
        # Define some interesting words to explore
        words = ['good', 'bad', 'delicious', 'excellent', 'terrible', 'quality', 'price']
    
    # Filter for words in the vocabulary
    words = [word for word in words if word in model.wv.key_to_index]
    
    if not words:
        print("None of the specified words are in the vocabulary.")
        return
    
    print("Word similarity analysis:")
    for word in words:
        print(f"\nMost similar words to '{word}':")
        similar_words = model.wv.most_similar(word, topn=5)
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.4f}")

# Analyze word similarities
analyze_word_similarities(best_model)

# ## 5. Application: Sentiment Analysis using Word2Vec Embeddings
print("\n5. Application: Sentiment Analysis with Word2Vec")

# Prepare data for sentiment analysis
# Convert the Score to a binary sentiment (1-3 as negative, 4-5 as positive)
df['Sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
print(f"Positive reviews: {sum(df['Sentiment'] == 1)}")
print(f"Negative reviews: {sum(df['Sentiment'] == 0)}")

# Function to get document embedding by averaging word vectors
def get_document_vector(text, model, vector_size):
    tokens = preprocess_text(text)
    vec = np.zeros(vector_size)
    count = 0
    for token in tokens:
        if token in model.wv.key_to_index:
            vec += model.wv[token]
            count += 1
    if count > 0:
        vec /= count
    return vec

# Create document vectors
vector_size = best_model.vector_size
X = df['Text'].apply(lambda text: get_document_vector(text, best_model, vector_size))
X = np.array(X.tolist())
y = df['Sentiment']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a classifier
classifier = LogisticRegression(random_state=42, max_iter=1000)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nSentiment Analysis Results:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# ## 6. Discussion: Strengths and Weaknesses of Word2Vec
print("\n6. Discussion: Strengths and Weaknesses of Word2Vec")

print("""
Strengths of Word2Vec:
1. Captures semantic relationships between words, as demonstrated by the word similarity analysis.
2. Relatively efficient to train compared to more complex models.
3. Creates dense vector representations that work well in downstream NLP tasks like sentiment analysis.
4. Can capture analogies and relationships between words (king - man + woman ≈ queen).
5. The embeddings are useful features for various machine learning models.

Weaknesses of Word2Vec:
1. Cannot handle out-of-vocabulary words - any word not seen during training has no representation.
2. Word embeddings are context-independent, meaning each word has a single vector regardless of its context.
3. Doesn't capture polysemy (multiple meanings of the same word).
4. Requires substantial amounts of training data for good quality embeddings.
5. The choice of hyperparameters (vector size, window size, etc.) significantly affects performance.
6. Struggles with rare words that appear few times in the corpus.

Challenges encountered in this implementation:
1. Balancing model complexity with training time for different hyperparameters.
2. Finding the right visualization parameters for t-SNE to meaningfully represent the high-dimensional word vectors.
3. Dealing with preprocessing decisions that affect the quality of embeddings.
4. Handling the computational requirements for training on larger datasets.

Potential improvements:
1. Use more sophisticated models like BERT or FastText that can handle context and subword information.
2. Experiment with domain-specific corpora for targeted applications.
3. Implement dynamic window sizes or attention mechanisms to better capture word relationships.
4. Incorporate syntactic and grammatical information into the embeddings.
""")

print("\nAssignment completed successfully!") 