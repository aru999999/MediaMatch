"""Tests. Run from the backend folder:   python -m pytest -q      (or: python test_api.py)

They build tiny fixture datasets with exactly the same file names and columns as the real ones,
so they run in seconds and don't need the big downloads.
"""
import os
import tempfile

import numpy as np
import pandas as pd

from app import build_indexes, create_app
from covers import CoverService, NetworkError

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi"]
rng = np.random.default_rng(0)


def make_fixtures(root):
    # ---- movies
    n = 60
    movies = pd.DataFrame({"movieId": np.arange(1, n + 1) * 10,
                           "title": [f"Movie {i}, The ({1980 + i % 30})" if i == 5 else f"Movie {i} ({1980 + i % 30})" for i in range(n)],
                           "genres": ["|".join(rng.choice(GENRES, 2, replace=False)) for _ in range(n)]})
    movies.loc[3, "genres"] = "(no genres listed)"
    links = pd.DataFrame({"movieId": movies.movieId, "imdbId": 1000 + np.arange(n), "tmdbId": np.where(np.arange(n) == 7, np.nan, 500 + np.arange(n))})
    rows = [(u, mid, float(rng.integers(1, 6)), 0) for u in range(1, 41) for mid in rng.choice(movies.movieId, 25, replace=False)]
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    tags = pd.DataFrame({"userId": [1, 2, 3], "movieId": [10, 10, 20], "tag": ["Funny", "funny", "dark"], "timestamp": 0})
    d = os.path.join(root, "movies"); os.makedirs(d)
    for name, df in [("movies", movies), ("links", links), ("ratings", ratings), ("tags", tags)]:
        df.to_csv(os.path.join(d, name + ".csv"), index=False)

    # ---- books
    nb = 50
    books = pd.DataFrame({"id": np.arange(1, nb + 1), "book_id": 9000 + np.arange(nb), "best_book_id": 0, "work_id": 0, "books_count": 1,
                          "isbn": [f"{123456789 + i}" for i in range(nb)], "isbn13": [9.78e12 + i if i % 2 else np.nan for i in range(nb)],
                          "authors": [f"Author {i % 7}, Illustrator" for i in range(nb)], "original_publication_year": 1990.0 + np.arange(nb) % 20,
                          "original_title": "", "title": [f"Book {i} (Saga, #{i % 3 + 1})" if i % 4 == 0 else f"Book {i}" for i in range(nb)],
                          "language_code": "eng", "average_rating": 3.0 + (np.arange(nb) % 20) / 10, "ratings_count": 100 + np.arange(nb) * 7,
                          "work_ratings_count": 1, "work_text_reviews_count": 1, "ratings_1": 0, "ratings_2": 0, "ratings_3": 0, "ratings_4": 0, "ratings_5": 0,
                          "image_url": ["https://s.gr-assets.com/assets/nophoto/book/x.png" if i % 3 == 0 else f"https://img/{i}.jpg" for i in range(nb)],
                          "small_image_url": ""})
    tg = pd.DataFrame({"tag_id": range(6), "tag_name": ["fantasy", "romance", "to-read", "mystery", "sci-fi", "favorites"]})
    bt = pd.DataFrame([(9000 + i, t, int(rng.integers(5, 100))) for i in range(nb) for t in rng.choice(6, 3, replace=False)],
                      columns=["goodreads_book_id", "tag_id", "count"])
    rows = [(i, u, int(rng.integers(1, 6))) for u in range(1, 61) for i in rng.choice(np.arange(1, nb + 1), 20, replace=False)]
    d = os.path.join(root, "books"); os.makedirs(d)
    for name, df in [("books", books), ("tags", tg), ("book_tags", bt), ("ratings", pd.DataFrame(rows, columns=["book_id", "user_id", "rating"]))]:
        df.to_csv(os.path.join(d, name + ".csv"), index=False)

    # ---- music (dupes across genres, one duplicate song, one row with a missing title)
    cols = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    m = pd.DataFrame(rng.random((90, 9)), columns=cols); m["loudness"] = -rng.random(90) * 20; m["tempo"] = 60 + rng.random(90) * 100
    m.insert(0, "track_id", [f"t{i % 80}" for i in range(90)])
    m["artists"] = [f"Artist {(i % 80) % 9};Guest" for i in range(90)]
    m["album_name"] = "Album"; m["track_name"] = [f"Song {i % 80}" for i in range(90)]
    m.loc[85, ["track_id", "track_name"]] = ["t85", "Song 1"]; m.loc[85, "artists"] = m.loc[1, "artists"]   # same name+artist, new id
    m.loc[86, "track_name"] = np.nan
    m["popularity"] = rng.integers(0, 100, 90); m["duration_ms"] = 200000; m["explicit"] = False
    m["track_genre"] = [["pop", "rock", "hip-hop", "jazz"][i % 4] for i in range(90)]
    d = os.path.join(root, "music"); os.makedirs(d)
    m.to_csv(os.path.join(d, "spotify-tracks-dataset.csv"), index=True)


_tmp = tempfile.TemporaryDirectory()
make_fixtures(_tmp.name)
INDEXES = build_indexes(_tmp.name)
FAKE = {"calls": []}


def fake_fetch(url, timeout=6):
    FAKE["calls"].append(url)
    if "deezer" in url:
        return {"data": [{"album": {"cover_big": "https://cdn/x/images/cover/abc123/500x500.jpg"}}]}
    if "themoviedb" in url:
        return {"poster_path": "/p.jpg"}
    raise NetworkError("offline")


def make_client(key="k"):
    os.environ["TMDB_API_KEY"] = key
    cov = CoverService(cache_path=os.path.join(_tmp.name, f"c{len(FAKE['calls'])}{key}.json"), fetch=fake_fetch)
    cov._wait = lambda gap=0: None
    return create_app(indexes=INDEXES, covers=cov, results_path=os.path.join(_tmp.name, "none.csv")).test_client()


client = make_client()


# ----------------------------------------------------------------------- loaders
def test_movie_title_cleaning_and_missing_genres():
    m = INDEXES["movies"]
    assert "The Movie 5" in m.title and m.items.loc[3, "genres"] == []
    assert m.items["year"].notna().all()


def test_music_dedupes_and_drops_bad_rows():
    m = INDEXES["music"]
    assert m.N <= 80 and m.items["track_id"].is_unique
    key = (m.items["title"].str.lower() + "|" + m.items["subtitle"].str.lower())
    assert key.is_unique and m.items["title"].notna().all()


def test_books_series_stripped_and_junk_tags_dropped():
    b = INDEXES["books"]
    assert all("#" not in t for t in b.title)
    assert not any(t.lower() in ("to read", "favorites") for tags in b.items["tags"] for t in tags)


# ----------------------------------------------------------------------- API
def test_meta_and_health():
    assert client.get("/api/health").get_json() == {"status": "ok"}
    meta = client.get("/api/meta").get_json()
    assert set(meta["domains"]) == {"movies", "books", "music"} and meta["domains"]["music"]["has_ratings"] is False


def test_search_prefers_prefix_and_short_queries_return_nothing():
    r = client.get("/api/movies/search?q=movie 1").get_json()["results"]
    assert r and r[0]["title"].lower().startswith("movie 1")
    assert client.get("/api/movies/search?q=m").get_json()["results"] == []


def test_similar_all_domains_have_match_and_why():
    for d in ("movies", "books", "music"):
        r = client.get(f"/api/{d}/similar/3?n=8").get_json()
        res = r["results"]
        assert r["seed"]["id"] == 3 and 1 <= len(res) <= 8
        assert 3 not in [x["id"] for x in res]
        assert all(50 <= x["match"] <= 99 and x["why"] for x in res)
        assert [x["match"] for x in res] == sorted((x["match"] for x in res), reverse=True)


def test_music_artist_cap():
    res = client.get("/api/music/similar/0?n=30").get_json()["results"]
    counts = pd.Series([x["subtitle"] for x in res]).value_counts()
    assert counts.max() <= 2


def test_for_list_combines_seeds_and_excludes_them():
    r = client.post("/api/movies/for_list", json={"ids": [1, 2, 3], "n": 10}).get_json()["results"]
    assert r and not {1, 2, 3} & {x["id"] for x in r}
    assert client.post("/api/movies/for_list", json={"ids": []}).status_code == 400


def test_browse_genre_and_pagination():
    a = client.get("/api/music/browse?genre=Jazz&limit=5").get_json()
    b = client.get("/api/music/browse?genre=Jazz&limit=5&offset=5").get_json()
    assert a["total"] > 0 and all("Jazz" in x["genres"] for x in a["items"])
    assert not {x["id"] for x in a["items"]} & {x["id"] for x in b["items"]}


def test_home_rows_and_items_endpoint():
    rows = client.get("/api/books/home").get_json()["rows"]
    assert rows[0]["title"] == "Popular right now" and rows[0]["items"]
    ids = ",".join(str(i["id"]) for i in rows[0]["items"][:3])
    assert len(client.get(f"/api/books/items?ids={ids}").get_json()["items"]) == 3


def test_errors_are_json():
    assert client.get("/api/podcasts/home").status_code == 404
    assert client.get("/api/movies/item/99999").get_json()["error"]
    assert client.get("/api/movies/items?ids=a,b").status_code == 400


def test_book_covers_use_isbn_fallback_when_placeholder():
    b = INDEXES["books"]
    i = int(np.flatnonzero(~b.items["image_ok"].to_numpy())[0])
    urls = b.card(i).get("cover_urls", [])
    assert not any("nophoto" in u for u in urls) and all(u.startswith("https://covers.openlibrary.org/") for u in urls)


# ----------------------------------------------------------------------- covers
def test_movie_cover_via_tmdb_and_cache():
    c = make_client()
    before = len(FAKE["calls"])
    r1 = c.get("/api/movies/cover/2").get_json()
    r2 = c.get("/api/movies/cover/2").get_json()
    assert r1["url"] == "https://image.tmdb.org/t/p/w342/p.jpg" == r2["url"]
    assert len(FAKE["calls"]) - before == 1  # second call came from the cache


def test_movie_without_tmdb_id_or_key_gives_no_url():
    c = make_client()
    assert c.get("/api/movies/cover/7").get_json()["url"] is None
    c2 = make_client(key="")
    r = c2.get("/api/movies/cover/2").get_json()
    assert r["url"] is None and r["reason"] == "no_tmdb_key"
    make_client()  # restore the key


def test_music_cover_via_deezer_and_network_failure_is_not_cached():
    c = make_client()
    assert c.get("/api/music/cover/0").get_json()["url"].endswith("500x500.jpg")
    cov = CoverService(cache_path=os.path.join(_tmp.name, "bad.json"), fetch=lambda u, timeout=6: (_ for _ in ()).throw(NetworkError("x")))
    cov._wait = lambda gap=0: None
    app = create_app(indexes=INDEXES, covers=cov, results_path="none").test_client()
    assert app.get("/api/music/cover/1").get_json() == {"url": None, "retry": True}
    assert "music:1" not in cov.cache


def test_frontend_is_served():
    r = client.get("/")
    assert r.status_code == 200 and b"MediaMatch" in r.data
    assert client.get("/static/app.js").status_code == 200


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
