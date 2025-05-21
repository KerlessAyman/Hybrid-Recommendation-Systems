from Collaborative import *

def hybrid_recommendations(userId, title, movies, model, cosine_sim, indices, top_n=10, content_weight=0.5, cf_weight=0.5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n*5]
    
    hybrid_scores = []
    for movie_idx, sim_score in sim_scores:
        movie_id = movies.iloc[movie_idx]['movieId']
        try:
            cf_pred = model.predict(userId, movie_id).est
            final_score = (content_weight * sim_score) + (cf_weight * cf_pred)
            hybrid_scores.append((movie_idx, final_score))
        except:
            continue
    
    if not hybrid_scores:
        return pd.Series([], name='title')
    
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]
    top_indices = [idx for idx, _ in hybrid_scores]
    return movies['title'].iloc[top_indices]
