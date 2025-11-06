from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pickle
import joblib
import pandas as pd

# --- App setup ---
app = FastAPI(title="Movie Recommender API")

# --- CORS setup ---
origins = [
    "http://localhost:5173",  # Vite frontend
    "http://localhost:3000",  # React dev server
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Data ---
try:
    similarity = joblib.load(open("similarity_compressed.pkl", "rb"))
    movies_dict = joblib.load(open("movies_dict.pkl", "rb"))
    movies = pd.DataFrame(movies_dict)
except Exception as e:
    raise RuntimeError(f"❌ Error loading data: {e}")

# --- Routes ---

@app.get("/")
def root():
    return {"message": "🎬 Welcome to My Next Movie API"}


# ✅ 1️⃣ Lazy Loading Endpoint (for infinite scroll)
@app.get("/movies/all")
def get_movies(skip: int = 0, limit: int = 50):
    """Return a random chunk of movie titles"""
    total = len(movies)
    # Dropna then sample random titles
    titles = movies["title"].dropna().sample(n=min(limit, total), replace=False).tolist()
    # In random mode, has_more doesn't really make sense; set based on whether more unique titles exist
    has_more = total > limit
    return {"data": titles, "has_more": has_more, "total": total}

# ✅ 2️⃣ Search Endpoint (for React Select autocomplete)
@app.get("/movies/search")
def search_movies(q: str = Query("", description="Search for movies by title")):
    """Search movies by partial title"""
    if not q:
        # return first few if no query
        results = movies["title"].dropna().head(10).tolist()
    else:
        results = (
            movies[movies["title"].str.contains(q, case=False, na=False)]["title"]
            .dropna()
            .head(10)
            .tolist()
        )
    return results


# ✅ 3️⃣ Movie Recommendation Endpoint
@app.get("/recommend/{movie_name}")
def recommend(movie_name: str):
    """Recommend top 5 similar movies"""
    movie_name = movie_name.lower()
    movies["title_lower"] = movies["title"].str.lower()

    if movie_name not in movies["title_lower"].values:
        return {"error": f"Movie '{movie_name}' not found in database."}

    # Get index of movie
    idx = movies[movies["title_lower"] == movie_name].index[0]

    # Fetch and sort similarity scores
    distances = similarity[idx]
    recs = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = [movies.iloc[i[0]].title for i in recs]
    return {
        "your_movie": movies.iloc[idx].title,
        "recommendations": recommendations,
    }
