#  app.py
import streamlit as st
import pandas as pd
import pickle
import requests
import json


# Load API keys
with open("config.json", "r") as f:
    config = json.load(f)

OMDB_KEY = config.get("OMDb_API_KEY")
TMDB_KEY = config.get("TMDB_API_KEY")


# Fetch Poster & Plot
def get_movie_details(title):
    plot = ""
    poster = ""

    # 1 Try OMDb
    omdb_url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_KEY}"
    try:
        response = requests.get(omdb_url)
        data = response.json()
        if data.get("Response") == "True":
            plot = data.get("Plot", "")
            poster = data.get("Poster", "")
    except Exception:
        pass

    # 2️ Fallback to TMDB
    if not poster or poster == "N/A":
        try:
            tmdb_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_KEY}&query={title}"
            tmdb_response = requests.get(tmdb_url).json()
            if tmdb_response["results"]:
                poster_path = tmdb_response["results"][0].get("poster_path")
                if poster_path:
                    poster = f"https://image.tmdb.org/t/p/w500{poster_path}"
        except Exception:
            poster = "https://upload.wikimedia.org/wikipedia/commons/f/fc/No_picture_available.png"

    # 3️ Final fallback
    if not poster or poster == "N/A":
        poster = "https://upload.wikimedia.org/wikipedia/commons/f/fc/No_picture_available.png"

    return plot, poster



# Load saved data
movies = pickle.load(open("movie_data.pkl", "rb"))
similarity = pickle.load(open("cosine_sim.pkl", "rb"))


# Recommend Function
def recommend(movie):
    if movie not in movies['title'].values:
        return []
    idx = movies[movies['title'] == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:11]  # 10 recommendations

    recommended = []
    for i in movie_list:
        title = movies.iloc[i[0]].title
        genre = movies.iloc[i[0]].genre
        plot, poster = get_movie_details(title)
        recommended.append({
            "title": title,
            "genre": genre,
            "plot": plot,
            "poster": poster
        })
    return recommended



# Streamlit UI  
st.set_page_config(page_title="Movie Recommender", layout="wide")

#  CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
        color: #fff;
        font-family: 'Poppins', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #e50914;
        font-weight: 700;
    }
    .movie-card {
        background-color: #141414;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        transition: 0.3s;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    .movie-card:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 15px rgba(229,9,20,0.4);
    }
    img {
        object-fit: cover;
        width: 180px !important;
        height: 270px !important;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Personalized Movie Recommendation System ")
st.subheader("Find your next favorite movie ")

st.markdown("---")

# Input field
selected_movie = st.text_input("Enter a movie name:", "")

if st.button("Recommend"):
    if selected_movie:
        recs = recommend(selected_movie)
        if recs:
            st.subheader("Recommended Movies:")
            cols = st.columns(5)
            for i, rec in enumerate(recs):
                with cols[i % 5]:
                    st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                    st.image(rec["poster"])
                    st.markdown(f"{rec['title']}")
                    st.caption(f"Genre: {rec['genre']}")
                    short_plot = rec["plot"][:150] + "..." if len(rec["plot"]) > 150 else rec["plot"]
                    st.write(short_plot)
                    with st.expander("Read More"):
                        st.write(rec["plot"])
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Movie not found in dataset.")
    else:
        st.info("Please enter a movie name.")
