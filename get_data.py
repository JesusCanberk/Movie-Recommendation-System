import requests
import time
import pandas as pd

API_KEY = "a5011cfb779ce0f439df74c05a2de186"
BASE_URL = "https://api.themoviedb.org/3"
MOVIES_TO_COLLECT = 10000  # or as much as you want

def get_certification(movie_id, api_key, country_code='US'):
    url = f"{BASE_URL}/movie/{movie_id}/release_dates"
    params = {"api_key": api_key}
    try:
        r = requests.get(url, params=params)
        data = r.json()
        results = data.get('results', [])
        for entry in results:
            if entry.get('iso_3166_1') == country_code:
                for rel in entry.get('release_dates', []):
                    cert = rel.get('certification')
                    if cert:
                        return cert
        return ""
    except:
        return ""

def get_movie_details(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": API_KEY,
        "append_to_response": "credits,images,keywords"
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return None
    data = response.json()
    # Certification
    certification = get_certification(movie_id, API_KEY)
    try:
        return {
            "movie_id": movie_id,
            "title": data.get("title"),
            "original_title": data.get("original_title"),
            "release_year": data.get("release_date", "")[:4],
            "original_language": data.get("original_language"),
            "spoken_languages": [lang.get("name") for lang in data.get("spoken_languages", [])],
            "country": [c.get("name") for c in data.get("production_countries", [])],
            "genre": [g.get("name") for g in data.get("genres", [])],
            "keywords": [k.get("name") for k in data.get("keywords", {}).get("keywords", [])],
            "tagline": data.get("tagline"),
            "plot_summary": data.get("overview"),
            "director": ", ".join([p['name'] for p in data['credits']['crew'] if p['job'] == 'Director']),
            "writer": ", ".join([p['name'] for p in data['credits']['crew'] if p['job'] == 'Writer']),
            "main_actors": ", ".join([a['name'] for a in data['credits']['cast'][:3]]),
            "character_names": ", ".join([a['character'] for a in data['credits']['cast'][:3]]),
            "rating": data.get("vote_average"),
            "poster_url": f"https://image.tmdb.org/t/p/original{data.get('poster_path')}" if data.get("poster_path") else "",
            "backdrop_url": f"https://image.tmdb.org/t/p/original{data.get('backdrop_path')}" if data.get("backdrop_path") else "",
            "description": data.get("overview"),
            "popularity": data.get("popularity"),
            "vote_average": data.get("vote_average"),
            "vote_count": data.get("vote_count"),
            "runtime": data.get("runtime"),
            "release_date": data.get("release_date"),
            "production_companies": [c.get("name") for c in data.get("production_companies", [])],
            "belongs_to_collection": data["belongs_to_collection"]["name"] if data.get("belongs_to_collection") else "",
            "budget": data.get("budget"),
            "revenue": data.get("revenue"),
            "certification": certification,
            "adult": data.get("adult"),
        }
    except Exception as e:
        print(f"Error with movie {movie_id}: {e}")
        return None

# 1. Collect movie IDs
movie_ids = []
page = 1
while len(movie_ids) < MOVIES_TO_COLLECT:
    print(f"Fetching discover page {page} ...")
    url = f"{BASE_URL}/discover/movie"
    params = {
        "api_key": API_KEY,
        "sort_by": "popularity.desc",
        "page": page
    }
    response = requests.get(url, params=params)
    data = response.json()
    results = data.get("results", [])
    if not results:
        break  # No more results
    for movie in results:
        movie_ids.append(movie["id"])
        if len(movie_ids) >= MOVIES_TO_COLLECT:
            break
    page += 1
    time.sleep(0.25)  # Respect API rate limits

print(f"Collected {len(movie_ids)} movie IDs.")

# 2. Gather detailed data for each movie ID
all_movies = []
for idx, movie_id in enumerate(movie_ids):
    print(f"Fetching details for movie {idx+1}/{len(movie_ids)}: {movie_id}")
    details = get_movie_details(movie_id)
    if details:
        all_movies.append(details)
    time.sleep(0.26)  # Stay below 4 req/sec
    if (idx + 1) % 100 == 0:
        print(f"Saved {idx+1} movies so far...")
        pd.DataFrame(all_movies).to_csv("movies_backup.csv", index=False)

# 3. Save all movies to CSV
df = pd.DataFrame(all_movies)
df.to_csv("tmdb_10000_movies_full.csv", index=False)
print("Done!")
