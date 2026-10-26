"""Load the three real datasets into one common structure.

Every loader returns a dict:
    name      "movies" | "books" | "music"
    items     DataFrame (one row per item; the row number is the item id used everywhere)
    content   L2-normalised feature matrix (rows align with items) used for content similarity
    ratings   DataFrame[user, item, rating] or None (music has no ratings)
    like_min  rating that counts as "liked"
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


def _pct_rank(values):
    """0..1 percentile rank (ties averaged) used as a popularity prior."""
    return pd.Series(values).rank(pct=True, method="average").to_numpy(dtype=np.float32)


_SPECIAL = {"hip-hop": "Hip-Hop", "r-n-b": "R&B", "k-pop": "K-Pop", "j-pop": "J-Pop", "j-rock": "J-Rock",
            "j-dance": "J-Dance", "j-idol": "J-Idol", "edm": "EDM", "idm": "IDM", "mpb": "MPB", "trip-hop": "Trip-Hop",
            "drum-and-bass": "Drum and Bass", "synth-pop": "Synth-Pop", "power-pop": "Power-Pop",
            "alt-rock": "Alt-Rock", "non-fiction": "Non-Fiction", "pop-film": "Pop Film", "hard-rock": "Hard Rock",
            "heavy-metal": "Heavy Metal", "death-metal": "Death Metal", "black-metal": "Black Metal"}


def _pretty(tag):
    t = tag.replace("_", "-")
    if t in _SPECIAL:
        return _SPECIAL[t]
    return " ".join(w.capitalize() if w not in ("and", "of") else w for w in t.split("-"))


def _tfidf(dicts):
    vec = DictVectorizer(sparse=True)
    counts = vec.fit_transform(dicts)
    return normalize(TfidfTransformer(sublinear_tf=False).fit_transform(counts)).astype(np.float32).tocsr()


# --------------------------------------------------------------------------- movies
def _clean_movie_title(raw):
    m = re.search(r"\((\d{4})\)\s*$", raw)
    year = int(m.group(1)) if m else None
    t = raw[: m.start()].strip() if m else raw.strip()
    m2 = re.match(r"^(.*?), (The|A|An)(\s\(.*\))?$", t)  # "Matrix, The" -> "The Matrix"
    if m2:
        t = f"{m2.group(2)} {m2.group(1)}{m2.group(3) or ''}"
    return t, year


def load_movies(data_dir=DATA_DIR):
    d = os.path.join(data_dir, "movies")
    movies = pd.read_csv(os.path.join(d, "movies.csv"))
    links = pd.read_csv(os.path.join(d, "links.csv"))
    ratings = pd.read_csv(os.path.join(d, "ratings.csv"))
    tags = pd.read_csv(os.path.join(d, "tags.csv"))

    movies = movies.merge(links, on="movieId", how="left").reset_index(drop=True)
    parsed = movies["title"].map(_clean_movie_title)
    movies["title"] = [p[0] for p in parsed]
    movies["year"] = [p[1] for p in parsed]
    movies["genres"] = movies["genres"].map(lambda g: [] if g == "(no genres listed)" else g.split("|"))
    idx = {mid: i for i, mid in enumerate(movies["movieId"])}

    r = ratings.rename(columns={"userId": "user", "movieId": "item"})
    r["item"] = r["item"].map(idx)
    r = r.dropna(subset=["item"]).astype({"item": int})[["user", "item", "rating"]]
    stats = r.groupby("item")["rating"].agg(["mean", "count"])
    movies["rating"] = stats["mean"].reindex(range(len(movies))).to_numpy()
    movies["n_ratings"] = stats["count"].reindex(range(len(movies))).fillna(0).astype(int).to_numpy()

    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tags["item"] = tags["movieId"].map(idx)
    top_tags = (tags.dropna(subset=["item"]).groupby(["item", "tag"]).size().reset_index(name="n")
                .sort_values(["item", "n"], ascending=[True, False]).groupby("item").head(8))
    tag_map = top_tags.groupby("item")["tag"].apply(list).to_dict()
    movies["tags"] = [tag_map.get(i, []) for i in range(len(movies))]

    docs = []
    for g, t, y in zip(movies["genres"], movies["tags"], movies["year"]):
        f = {f"g:{x}": 2.0 for x in g}
        f.update({f"t:{x}": 1.0 for x in t})
        if pd.notna(y):
            f[f"d:{int(y) // 10}"] = 0.6
        docs.append(f)
    movies["pop"] = _pct_rank(movies["n_ratings"])
    movies["subtitle"] = movies["year"].map(lambda y: "" if pd.isna(y) else str(int(y)))
    return dict(name="movies", items=movies, content=_tfidf(docs), ratings=r, like_min=4.0, quality_m=20)


# --------------------------------------------------------------------------- books
_JUNK = re.compile(r"(?:to-?read|currently|favou?rite|owned|own\b|books?-i|my-|library|default|kindle|ebook|e-book|audio|"
                   r"wish|tbr|dnf|abandon|finished|^read|^\d|shelf|shelve|borrowed|bought|purchased|re-?read|"
                   r"book-club|paperback|hardcover|nook|netgalley|arc|giveaway|^-|series$)", re.I)
_SYN = {"sci-fi": "science-fiction", "scifi": "science-fiction", "sci-fi-fantasy": "science-fiction",
        "ya": "young-adult", "childrens-books": "childrens", "kids": "childrens", "chick-lit": "romance",
        "historical": "historical-fiction", "comics": "graphic-novels", "graphic-novel": "graphic-novels",
        "mysteries": "mystery", "mystery-thriller": "mystery", "nonfiction": "non-fiction"}
_GENRES = ["fantasy", "romance", "mystery", "thriller", "science-fiction", "historical-fiction", "young-adult",
           "horror", "classics", "non-fiction", "biography", "history", "contemporary", "humor", "dystopian",
           "crime", "adventure", "childrens", "poetry", "philosophy", "graphic-novels", "paranormal", "self-help",
           "memoir", "psychology", "science", "religion", "christian", "war", "politics"]


def _isbn_str(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if re.fullmatch(r"\d{9,13}[\dXx]?", s):
        return s.zfill(10) if len(s) < 10 else s
    return None


def load_books(data_dir=DATA_DIR):
    d = os.path.join(data_dir, "books")
    books = pd.read_csv(os.path.join(d, "books.csv"), dtype={"isbn": str})
    bt = pd.read_csv(os.path.join(d, "book_tags.csv"))
    tg = pd.read_csv(os.path.join(d, "tags.csv"))
    ratings = pd.read_csv(os.path.join(d, "ratings.csv"))

    books = books.sort_values("id").reset_index(drop=True)
    id2idx = dict(zip(books["id"], range(len(books))))      # ratings.book_id  -> row
    gid2idx = dict(zip(books["book_id"], range(len(books))))  # book_tags.goodreads_book_id -> row

    books["series"] = books["title"].str.extract(r"\(([^()]*#\d+[^()]*)\)\s*$")[0]
    books["title"] = books["title"].str.replace(r"\s*\([^()]*#\d+[^()]*\)\s*$", "", regex=True).str.strip()
    books["authors"] = books["authors"].fillna("Unknown")
    books["subtitle"] = books["authors"].map(lambda a: ", ".join(a.split(", ")[:2]))
    books["year"] = books["original_publication_year"]
    books["rating"] = books["average_rating"]
    books["n_ratings"] = books["ratings_count"].astype(int)
    books["isbn13"] = books["isbn13"].map(_isbn_str)
    books["isbn"] = books["isbn"].map(_isbn_str)
    books["image_ok"] = ~books["image_url"].fillna("nophoto").str.contains("nophoto")

    bt = bt.merge(tg, on="tag_id")
    bt["tag_name"] = bt["tag_name"].map(lambda t: _SYN.get(t, t))
    bt = bt[~bt["tag_name"].str.contains(_JUNK)]
    bt["item"] = bt["goodreads_book_id"].map(gid2idx)
    bt = bt.dropna(subset=["item"])
    bt = bt.groupby(["item", "tag_name"], as_index=False)["count"].sum()
    bt["w"] = bt["count"] / bt.groupby("item")["count"].transform("max")
    bt = bt.sort_values(["item", "count"], ascending=[True, False]).groupby("item").head(14)
    bt = bt[bt["w"] >= 0.08]
    tag_w = {i: dict(zip(g["tag_name"], g["w"])) for i, g in bt.groupby("item")}
    gset = set(_GENRES)
    genre_lists = {i: [_pretty(t) for t in g["tag_name"] if t in gset][:3] for i, g in bt.groupby("item")}
    books["genres"] = [genre_lists.get(i, []) for i in range(len(books))]
    books["tags"] = [[_pretty(t) for t in list(tag_w.get(i, {}))[:6]] for i in range(len(books))]

    docs = []
    for i in range(len(books)):
        f = {f"t:{t}": float(np.sqrt(w)) for t, w in tag_w.get(i, {}).items()}
        for a in books["authors"].iloc[i].split(", ")[:2]:
            f[f"a:{a}"] = 1.5
        docs.append(f)

    r = ratings.rename(columns={"user_id": "user", "book_id": "item"})
    r["item"] = r["item"].map(id2idx)
    r = r.dropna(subset=["item"]).astype({"item": int})[["user", "item", "rating"]]
    books["pop"] = _pct_rank(books["n_ratings"])
    return dict(name="books", items=books, content=_tfidf(docs), ratings=r, like_min=4.0, quality_m=20000)


# --------------------------------------------------------------------------- music
AUDIO = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
         "liveness", "valence", "tempo"]


def load_music(data_dir=DATA_DIR):
    d = os.path.join(data_dir, "music")
    path = next(os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".csv"))
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")]).dropna(subset=["track_name", "artists"])

    genres = df.groupby("track_id")["track_genre"].apply(lambda s: sorted(set(s))).rename("genres")
    df = df.sort_values("popularity", ascending=False).drop_duplicates("track_id").merge(genres, on="track_id")
    df["_k"] = (df["track_name"].str.lower().str.strip() + "|" + df["artists"].str.lower().str.strip())
    df = df.drop_duplicates("_k").drop(columns="_k").sort_values("popularity", ascending=False).reset_index(drop=True)

    df["title"] = df["track_name"].str.strip()
    df["subtitle"] = df["artists"].str.replace(";", ", ")
    df["year"] = np.nan
    df["rating"] = np.nan
    df["n_ratings"] = 0
    df["pop"] = (df["popularity"] / 100.0).astype(np.float32)
    df["tags"] = [[] for _ in range(len(df))]

    a = df[AUDIO].astype(float)
    a = ((a - a.mean()) / a.std().replace(0, 1)).clip(-3, 3).to_numpy(np.float32) * 0.5
    all_g = sorted({g for gs in df["genres"] for g in gs})
    gi = {g: i for i, g in enumerate(all_g)}
    G = np.zeros((len(df), len(all_g)), np.float32)
    for r, gs in enumerate(df["genres"]):
        for g in gs:
            G[r, gi[g]] = 1.2 / np.sqrt(len(gs))
    X = normalize(np.hstack([a, G])).astype(np.float32)
    df["genres"] = df["genres"].map(lambda gs: [_pretty(g) for g in gs])
    return dict(name="music", items=df, content=X, ratings=None, like_min=None, quality_m=0)


LOADERS = {"movies": load_movies, "books": load_books, "music": load_music}
