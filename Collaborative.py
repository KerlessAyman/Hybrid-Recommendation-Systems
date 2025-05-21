from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import pandas as pd
from content_Based import *  # Assuming this is your content-based module

# 1. Prepare the data for Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# 2. Split data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 3. Initialize and train the SVD model
model = SVD()
model.fit(trainset)

# 4. Predict on the test set
predictions = model.test(testset)

# 5. Evaluate the model
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

# 6. Function to get top-N recommendations for a specific user
def get_top_n_recommendations(predictions, user_id, n=10, movies=None):
    user_id_str = str(user_id)
    # Filter predictions for this user
    user_predictions = [pred for pred in predictions if str(pred.uid) == user_id_str]
    # Sort predictions by estimated rating in descending order
    user_predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = user_predictions[:n]

    if movies is not None:
        # Create a mapping from movieId to title
        movie_id_to_title = pd.Series(movies.title.values, index=movies.movieId.astype(str)).to_dict()
        # Return list of (title, estimated rating)
        return [(movie_id_to_title.get(str(pred.iid), "Unknown Title"), pred.est) for pred in top_n]
    else:
        # Return list of (movieId, estimated rating)
        return [(pred.iid, pred.est) for pred in top_n]

# 7. Generate predictions for all unrated movies by user 1
user_id = 1
all_movie_ids = ratings['movieId'].unique()
rated_movie_ids = ratings[ratings['userId'] == user_id]['movieId'].values
unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]

# Generate predictions for all unrated movies
predictions_for_user = [model.predict(str(user_id), str(movie_id)) for movie_id in unrated_movie_ids]

# 8. Get top 10 recommendations
top_movies = get_top_n_recommendations(predictions_for_user, user_id=user_id, n=10, movies=movies)

# 9. Display the recommendations
print(f"Top movie recommendations for user {user_id}:")
for title, score in top_movies:
    print(f"{title}: {score:.2f}")
