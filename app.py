from Hybrid import *
import streamlit as st

st.title("🎬 Hybrid Movie Recommendation System")

user_id = st.number_input("Enter User ID", min_value=1, value=1)
movie_title = st.selectbox("Choose a movie you like", movies['title'].sort_values())

recommend_type = st.radio("Recommendation Type", ['Content-Based', 'Collaborative', 'Hybrid'])

top_n = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Recommend"):
    if recommend_type == 'Content-Based':
        recs = get_recommendations(movie_title, movies, cosine_sim, indices, top_n)
    elif recommend_type == 'Collaborative':
        recs = get_top_n_recommendations(predictions, user_id, top_n, movies)
    else:
        recs = hybrid_recommendations(user_id, movie_title, movies, model, cosine_sim, indices, top_n)
    
    st.write("### Recommended Movies:")
    for r in recs:
        st.write(f"🎥 {r}")