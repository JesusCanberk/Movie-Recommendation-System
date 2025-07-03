import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageTk
import requests
from io import BytesIO
import numpy as np
import os
import pickle

df = pd.read_csv("tmdb_10000_movies_full.csv")
current_year = pd.Timestamp.now().year
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
df = df[df["release_year"].notna() & (df["release_year"] <= current_year)]

def is_empty(val):
    s = str(val).strip()
    return (s == "") or (s == "[]") or (pd.isna(val))
df = df[~df["genre"].apply(is_empty)]
df = df[~df["keywords"].apply(is_empty)]
for col in ["title", "plot_summary", "description"]:
    df = df[df[col].notna()]
    df = df[df[col].astype(str).str.strip() != ""]
df = df.reset_index(drop=True)

for col in [
    "original_title", "original_language", "spoken_languages", "country", "genre", "keywords",
    "tagline", "plot_summary", "director", "writer", "main_actors", "character_names", "rating",
    "description", "popularity", "vote_average", "vote_count", "runtime", "release_date",
    "production_companies", "belongs_to_collection", "budget", "revenue", "certification", "adult"
]:
    if col in df.columns:
        df[col] = df[col].fillna("")

def combine_features(row):
    fields = [
        str(row.get('title', '')),
        str(row.get('original_title', '')),
        str(row.get('original_language', '')),
        str(row.get('spoken_languages', '')),
        str(row.get('country', '')),
        str(row.get('genre', '')),
        str(row.get('keywords', '')),
        str(row.get('tagline', '')),
        str(row.get('plot_summary', '')),
        str(row.get('director', '')),
        str(row.get('writer', '')),
        str(row.get('main_actors', '')),
        str(row.get('character_names', '')),
        str(row.get('description', '')),
        str(row.get('production_companies', '')),
        str(row.get('belongs_to_collection', '')),
        str(row.get('certification', '')),
        str(row.get('adult', '')),
        str(row.get('release_year', '')),
        str(row.get('runtime', '')),
        str(row.get('release_date', '')),
        str(row.get('popularity', '')),
        str(row.get('vote_average', '')),
        str(row.get('vote_count', '')),
        str(row.get('budget', '')),
        str(row.get('revenue', '')),
        str(row.get('rating', '')),
    ]
    return " ".join(fields)

df["combined"] = df.apply(combine_features, axis=1)

EMBEDDINGS_FILE = "movie_embeddings_sbert.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)
else:
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(df["combined"].tolist(), show_progress_bar=True, batch_size=64)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

embeddings = np.array(embeddings)

class MovieRecommenderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        self.title("Movie Recommender (SBERT Semantic Embedding)")
        self.geometry("1200x700")
        self.check_vars = []
        self.check_buttons = []
        self.sample_movies = pd.DataFrame()
        self.remaining_movies = pd.DataFrame()
        self.selected_movies = df.iloc[0:0].copy()
        self.poster_image = None
        self.manual_selected_titles = set()
        self.manual_mode = False
        self.recommended_movies = []
        self.rec_line_starts = []
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(fill="x")
        self.show_button = tk.Button(self.button_frame, text="", command=self.refresh_movies, font=("Arial", 11))
        self.show_button.pack(side="left", padx=5)
        self.recommend_btn = tk.Button(self.button_frame, text="Recommend me movies!", command=self.recommend_movies, font=("Arial", 11))
        self.recommend_btn.pack(side="right", padx=5)
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)
        self.left_frame = ttk.Frame(self.main_frame, width=650)
        self.left_frame.pack(side="left", fill="y")
        self.canvas = tk.Canvas(self.left_frame, width=600)
        self.scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.right_frame = tk.Frame(self.main_frame, width=400)
        self.right_frame.pack(side="left", fill="both", expand=True)
        self.poster_label = tk.Label(self.right_frame)
        self.poster_label.pack(pady=12)
        self.recommendations_text = tk.Text(self.right_frame, height=22, wrap="word", font=("Arial", 11))
        self.recommendations_text.pack(fill="both", expand=True, padx=10, pady=5)
        self.recommendations_text.tag_configure('title', font=('Arial', 12, 'bold'))
        self.recommendations_text.tag_configure('genre', font=('Arial', 10, 'italic'))
        self.recommendations_text.bind("<Button-1>", self.on_recommendation_click)
        self.after(300, self.ask_selection_mode)

    def ask_selection_mode(self):
        sel_win = tk.Toplevel(self)
        sel_win.title("How would you like to select watched movies?")
        sel_win.geometry("430x180")
        sel_win.grab_set()
        sel_win.focus_set()
        tk.Label(sel_win, text="How would you like to select your watched movies?", font=("Arial", 12, "bold")).pack(pady=(24,10), padx=10)
        def use_random():
            sel_win.destroy()
            self.deiconify()
            self.show_button.pack(side="left", padx=5)
            self.manual_mode = False
            self.refresh_movies(first_time=True)
        def use_manual():
            sel_win.destroy()
            self.deiconify()
            self.show_button.pack_forget()
            self.manual_mode = True
            self.manual_selected_titles.clear()
            self.show_manual_selection_panel()
        tk.Button(sel_win, text="Random 100 movies", font=("Arial", 11), width=22, command=use_random).pack(pady=7)
        tk.Button(sel_win, text="Manual selection from all movies", font=("Arial", 11), width=22, command=use_manual).pack()

    def refresh_movies(self, first_time=False):
        selected_titles = set()
        if hasattr(self, "check_vars") and hasattr(self, "check_buttons"):
            for var, chk in zip(self.check_vars, self.check_buttons):
                if var.get():
                    title = chk.cget('text').split(' (')[0]
                    selected_titles.add(title)
        num_selected = len(selected_titles)
        remaining_needed = max(0, 100 - num_selected)
        self.show_button.config(text=f"Show new {remaining_needed} movies")

        available = df[~df["title"].isin(selected_titles)]
        if remaining_needed > 0:
            new_sample = available.sample(n=remaining_needed, random_state=None)
            self.sample_movies = pd.concat([df[df['title'].isin(selected_titles)], new_sample]).reset_index(drop=True)
        else:
            self.sample_movies = df[df['title'].isin(selected_titles)].copy()
        self.remaining_movies = df[~df["title"].isin(self.sample_movies["title"])].reset_index(drop=True)

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.check_vars = []
        self.check_buttons = []

        def update_show_button(*_):
            checked = sum(var.get() for var in self.check_vars)
            remaining = max(0, 100 - checked)
            self.show_button.config(text=f"Show new {remaining} movies")

        for idx, row in self.sample_movies.iterrows():
            var = tk.BooleanVar(value=row['title'] in selected_titles)
            var.trace_add('write', update_show_button)
            chk = tk.Checkbutton(
                self.scrollable_frame,
                text=f"{row['title']} ({int(row['release_year'])}) | Genres: {', '.join(eval(row['genre'])) if row['genre'].startswith('[') else row['genre']}",
                variable=var,
                anchor="w", justify="left", width=70,
                command=lambda i=idx: self.show_poster(i)
            )
            chk.pack(anchor="w", padx=10, pady=2)
            self.check_vars.append(var)
            self.check_buttons.append(chk)
        update_show_button()
        self.recommendations_text.delete(1.0, tk.END)
        self.poster_label.config(image="", text="")

    def show_poster(self, idx):
        try:
            poster_url = self.sample_movies.iloc[idx]['poster_url']
            if pd.isna(poster_url) or poster_url.strip() == "":
                self.poster_label.config(image="", text="No poster available")
                return
            resp = requests.get(poster_url, timeout=5)
            img_data = resp.content
            img = Image.open(BytesIO(img_data)).resize((250, 375))
            self.poster_image = ImageTk.PhotoImage(img)
            self.poster_label.config(image=self.poster_image, text="")
            self.poster_label.pack_configure(anchor="center")
        except Exception:
            self.poster_label.config(image="", text="No poster available")

    def show_manual_selection_panel(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.check_vars = []
        self.check_buttons = []

        search_frame = tk.Frame(self.scrollable_frame)
        search_frame.pack(fill="x", pady=(3, 6))
        tk.Label(search_frame, text="Search:").pack(side="left", padx=(6, 3))
        search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=search_var, width=40)
        search_entry.pack(side="left")

        list_frame = tk.Frame(self.scrollable_frame)
        list_frame.pack(fill="both", expand=True)

        all_movies = df.reset_index(drop=True)
        selected_titles = set(self.manual_selected_titles)

        def update_list(*args):
            for widget in list_frame.winfo_children():
                widget.destroy()
            self.check_vars.clear()
            self.check_buttons.clear()
            filter_txt = search_var.get().lower()
            filtered = all_movies[all_movies['title'].str.lower().str.contains(filter_txt)].drop_duplicates(subset="title")
            for i, row in filtered.iterrows():
                var = tk.BooleanVar(value=row['title'] in selected_titles)
                def toggle(var=var, title=row['title'], row=row):
                    if var.get():
                        selected_titles.add(title)
                        poster_url = row['poster_url']
                        try:
                            if pd.isna(poster_url) or poster_url.strip() == "":
                                self.poster_label.config(image="", text="No poster available")
                                return
                            resp = requests.get(poster_url, timeout=5)
                            img_data = resp.content
                            img = Image.open(BytesIO(img_data)).resize((250, 375))
                            self.poster_image = ImageTk.PhotoImage(img)
                            self.poster_label.config(image=self.poster_image, text="")
                            self.poster_label.pack_configure(anchor="center")
                        except Exception:
                            self.poster_label.config(image="", text="No poster available")
                    else:
                        selected_titles.discard(title)
                    self.manual_selected_titles = set(selected_titles)
                chk = tk.Checkbutton(
                    list_frame,
                    text=f"{row['title']} ({int(row['release_year'])}) | Genres: {', '.join(eval(row['genre'])) if row['genre'].startswith('[') else row['genre']}",
                    variable=var, anchor="w", justify="left", width=70,
                    command=toggle
                )
                chk.pack(anchor="w", padx=10, pady=2)
                self.check_vars.append(var)
                self.check_buttons.append(chk)

        search_var.trace_add("write", update_list)
        update_list()

    def recommend_movies(self):
        if self.manual_mode:
            checked_titles = set()
            for var, btn in zip(self.check_vars, self.check_buttons):
                title = btn.cget('text').split(' (')[0]
                if var.get():
                    checked_titles.add(title)
            self.manual_selected_titles = set(list(self.manual_selected_titles) + list(checked_titles))
            if not self.manual_selected_titles:
                messagebox.showinfo("No selection", "Please select at least one movie!")
                return
            self.show_manual_recommendation()
            return

        liked_indices = [i for i, var in enumerate(self.check_vars) if var.get()]
        if not liked_indices:
            messagebox.showinfo("No selection", "Please select at least one movie!")
            return
        liked_df = self.sample_movies.iloc[liked_indices]
        liked_idx = df.index[df["title"].isin(liked_df["title"])].tolist()
        rem_idx = df.index[df["title"].isin(self.remaining_movies["title"])].tolist()
        liked_embs = embeddings[liked_idx]
        rem_embs = embeddings[rem_idx]
        mean_liked = liked_embs.mean(axis=0, keepdims=True)
        sims = cosine_similarity(mean_liked, rem_embs).flatten()
        top_idx = np.argsort(sims)[::-1]
        top_movies = self.remaining_movies.iloc[top_idx].copy()
        top_movies = top_movies.drop_duplicates(subset="title").head(10)
        self.show_recommendations(top_movies)

    def show_manual_recommendation(self):
        if not self.manual_selected_titles:
            messagebox.showinfo("No selection", "Please select at least one movie!")
            return
        liked_df = df[df['title'].isin(self.manual_selected_titles)]
        remaining_movies = df[~df['title'].isin(self.manual_selected_titles)]
        liked_idx = df.index[df["title"].isin(liked_df["title"])].tolist()
        rem_idx = df.index[df["title"].isin(remaining_movies["title"])].tolist()
        liked_embs = embeddings[liked_idx]
        rem_embs = embeddings[rem_idx]
        mean_liked = liked_embs.mean(axis=0, keepdims=True)
        sims = cosine_similarity(mean_liked, rem_embs).flatten()
        top_idx = np.argsort(sims)[::-1]
        top_movies = remaining_movies.iloc[top_idx].copy()
        top_movies = top_movies.drop_duplicates(subset="title").head(10)
        self.show_recommendations(top_movies)

    def show_recommendations(self, top_movies):
        self.recommendations_text.delete(1.0, tk.END)
        self.recommended_movies = []
        self.rec_line_starts = []
        for i, (_, row) in enumerate(top_movies.iterrows(), 1):
            line_start = int(self.recommendations_text.index(tk.END).split('.')[0])
            self.rec_line_starts.append(line_start)
            self.recommendations_text.insert(
                tk.END, f"{i}. {row['title']} ({int(row['release_year'])})\n", 'title')
            genres = ', '.join(eval(row['genre'])) if row['genre'].startswith('[') else row['genre']
            self.recommendations_text.insert(
                tk.END, f"    \u2022 Genres: {genres}\n", 'genre')
            keywords = ', '.join(eval(row['keywords'])) if row['keywords'].startswith('[') else row['keywords']
            self.recommendations_text.insert(
                tk.END, f"    \u2022 Keywords: {keywords}\n")
            summary = row['plot_summary'][:180]
            if len(row['plot_summary']) > 180:
                summary += '...'
            self.recommendations_text.insert(
                tk.END, f"    \u2022 Overview: {summary}\n")
            self.recommendations_text.insert(tk.END, f"{'-'*55}\n")
            self.recommended_movies.append(row)

    def on_recommendation_click(self, event):
        index = self.recommendations_text.index(f"@{event.x},{event.y}")
        line_clicked = int(str(index).split('.')[0])
        rec_idx = None
        for idx, start in enumerate(self.rec_line_starts):
            next_start = self.rec_line_starts[idx+1] if idx+1 < len(self.rec_line_starts) else 9999
            if start <= line_clicked < next_start:
                rec_idx = idx
                break
        if rec_idx is not None and 0 <= rec_idx < len(self.recommended_movies):
            movie = self.recommended_movies[rec_idx]
            self.show_movie_details_popup(movie)

    def show_movie_details_popup(self, row):
        win = tk.Toplevel(self)
        win.title(f"{row['title']} ({int(row['release_year'])})")
        win.geometry("510x760")
        poster_frame = tk.Frame(win)
        poster_frame.pack(pady=10)
        try:
            poster_url = row['poster_url']
            if pd.isna(poster_url) or poster_url.strip() == "":
                poster_lbl = tk.Label(poster_frame, text="No poster available")
            else:
                resp = requests.get(poster_url, timeout=5)
                img_data = resp.content
                img = Image.open(BytesIO(img_data)).resize((200, 300))
                poster_img = ImageTk.PhotoImage(img)
                poster_lbl = tk.Label(poster_frame, image=poster_img)
                poster_lbl.image = poster_img
            poster_lbl.pack()
        except Exception:
            tk.Label(poster_frame, text="No poster available").pack()

        text = tk.Text(win, wrap="word", height=32, width=62, font=("Arial", 11), bd=0, padx=8, pady=8)
        text.pack(padx=10, pady=(6,12), fill="both", expand=True)

        def safe_insert(label, val, tag=None):
            if val is None or pd.isna(val):
                return
            sval = str(val).strip().lower()
            if (isinstance(val, (int, float, np.integer, np.floating)) and float(val) == 0):
                return
            if sval in {"", "nan", "none", "0"}:
                return
            text.insert(tk.END, f"{label}: {val}\n", tag if tag else ())
        
        safe_insert("Title", f"{row['title']} ({int(row['release_year'])})", 'title')
        genres = ', '.join(eval(row['genre'])) if row['genre'].startswith('[') else row['genre']
        safe_insert("Genres", genres, 'genre')
        keywords = ', '.join(eval(row['keywords'])) if row['keywords'].startswith('[') else row['keywords']
        safe_insert("Keywords", keywords)
        safe_insert("Tagline", row.get('tagline'))
        safe_insert("Description", row.get('description'))
        safe_insert("Plot", row.get('plot_summary'))
        safe_insert("Director", row.get('director'))
        safe_insert("Writer", row.get('writer'))
        safe_insert("Main actors", row.get('main_actors'))
        safe_insert("Characters", row.get('character_names'))
        safe_insert("Production companies", row.get('production_companies'))
        safe_insert("Collection", row.get('belongs_to_collection'))
        safe_insert("Certification", row.get('certification'))
        safe_insert("Country", row.get('country'))
        safe_insert("Spoken languages", row.get('spoken_languages'))
        safe_insert("Original language", row.get('original_language'))
        safe_insert("Budget", row.get('budget'))
        safe_insert("Revenue", row.get('revenue'))
        safe_insert("Popularity", row.get('popularity'))
        safe_insert("Rating", row.get('vote_average'))
        safe_insert("Vote count", row.get('vote_count'))
        safe_insert("Runtime", row.get('runtime'))
        safe_insert("Release date", row.get('release_date'))
        safe_insert("Adult", row.get('adult'))
        text.config(state='disabled')

if __name__ == "__main__":
    app = MovieRecommenderApp()
    app.mainloop()
