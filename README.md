
# 🎬 Movie Recommendation System (SBERT + TMDb API)

This project builds a **content-based movie recommendation system** powered by **semantic embeddings from Sentence-BERT** and enriched with **real movie metadata** fetched from the **TMDb API**.

---

## 📦 Project Overview

- 🔄 **Data Collection**: Fetches 10,000 popular movies using TMDb API with rich metadata like genre, keywords, directors, cast, ratings, revenue, and more.
- 🧠 **Movie Embeddings**: Generates SBERT embeddings for each movie using `all-MiniLM-L6-v2`.
- 🎯 **Recommendation Engine**: Computes cosine similarity between watched movies and all others for personalized recommendations.
- 🖥️ **GUI**: An interactive `Tkinter` interface allows users to select movies (random or manual) and receive top 10 similar movies.

---

## 📁 Directory Structure

```
📦 Movie-Recommendation-System/
┣ 📄 get_data.py                 ← Collects 10,000 movies using TMDb API
┣ 📄 movie_recommendation_system.py  ← Main GUI-based recommendation app
┣ 📄 tmdb_10000_movies_full.csv  ← Generated dataset with movie metadata
┣ 📄 movie_embeddings_sbert.pkl  ← Cached embeddings for fast inference
┣ 📄 README.md                   ← You're reading it!
```

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install dependencies

```bash
pip install pandas numpy pillow sentence-transformers scikit-learn requests
```

---

## 📥 Part 1: Fetching Movie Data (Optional but Recommended)

To fetch your own fresh dataset of 10,000 movies:

1. **Set your TMDb API key** in `get_data.py`:
   ```python
   API_KEY = "your_tmdb_api_key"
   ```

2. **Run the script:**
   ```bash
   python get_data.py
   ```

3. It will generate:
   - `tmdb_10000_movies_full.csv`
   - intermediate backups in `movies_backup.csv`

### ✅ Collected Features Include:

- Titles, genres, keywords, plot summary, director, writer, main actors, rating
- Poster URL, release date, budget, revenue, popularity, certification, runtime, etc.

---

## 🎯 Part 2: Running the Recommendation App

```bash
python movie_recommendation_system.py
```

### 👇 Two Selection Modes:

- **Random 100 movies**: Get a random list to mark as watched
- **Manual selection**: Search and select from full dataset

After selection, click **"Recommend me movies!"** to see recommendations.

---

## 🧠 How the Recommendation Works

- SBERT model: `all-MiniLM-L6-v2`
- Feature aggregation: Multiple textual and numerical fields are concatenated
- Embeddings cached in `movie_embeddings_sbert.pkl`
- Cosine similarity used to compare user-liked movies to others

---

## 🖼️ GUI Features

- ✅ Poster image shown on selection
- ✅ Scrollable checklist of movies
- ✅ Clean details popup on clicking recommendations
- ✅ Handles missing fields gracefully (e.g., 0 revenue not shown)

---

## 📊 Example Fields in the Dataset

| Field              | Description                                 |
|-------------------|---------------------------------------------|
| `title`           | Movie title                                 |
| `genre`           | List of genres                              |
| `keywords`        | List of keywords                            |
| `plot_summary`    | Short description                           |
| `director`        | Main director(s)                            |
| `rating`          | Average user rating                         |
| `vote_count`      | Number of ratings                           |
| `budget`          | Budget (USD)                                |
| `revenue`         | Revenue (USD)                               |
| `release_date`    | YYYY-MM-DD format                           |
| `poster_url`      | Image link from TMDb                        |

---

## 🧾 License

This project is provided for educational and personal use only. API usage must comply with [TMDb's Terms of Use](https://www.themoviedb.org/documentation/api/terms-of-use).

---

## 🙌 Acknowledgements

- [TMDb (The Movie Database)](https://www.themoviedb.org/) for API access
- [Sentence Transformers](https://www.sbert.net/) for semantic embeddings
