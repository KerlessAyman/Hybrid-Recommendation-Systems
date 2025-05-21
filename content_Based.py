from preprocessing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Initialize TF-IDF Vectorizer to ignore common English stop words
tfidf = TfidfVectorizer(stop_words='english')

# Transform the 'genres' column from the movies DataFrame into TF-IDF feature vectors
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute the cosine similarity matrix between all movie genre vectors
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a reverse lookup Series to get movie indices based on movie titles
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function based on content similarity
def get_recommendations(title, movies, cosine_sim, indices, top_n=10):
    # Check if the movie title exists in the dataset
    if title not in indices:
        return pd.Series([], name='title')
    
    # Get the index of the movie that matches the title
    idx = indices[title]
    
    # Get a list of similarity scores for this movie with all others
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Skip the first movie (itself) and take the next top_n movies
    sim_scores = sim_scores[1:top_n+1]
    
    # Extract the indices of the recommended movies
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the titles of the recommended movies
    return movies['title'].iloc[movie_indices]

# Example usage
print(get_recommendations("Toy Story (1995)", movies, cosine_sim, indices, top_n=10))
