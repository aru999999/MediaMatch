"""Hybrid item-to-item recommender for one domain (movies, books or music).

For a seed item (or several seeds, e.g. a "My List") every other item gets a score that blends

    content similarity   TF-IDF over genres / tags / authors (movies, books), audio features + genre (music)
    collaborative signal item-item adjusted cosine on real user ratings, shrunk by the number of co-raters
    popularity prior     a small nudge toward well-known items

Items with few ratings lean on content only. The "match %" shown in the UI is the blended score
relative to the best candidate, so it measures closeness to the seed, not the chance you'll enjoy it.
"""
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

PREFERRED_GENRES = {
    "movies": ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Animation", "Romance", "Horror", "Adventure",
               "Crime", "Fantasy", "Documentary"],
    "books": ["Fantasy", "Romance", "Mystery", "Science Fiction", "Thriller", "Historical Fiction", "Young Adult",
              "Horror", "Classics", "Non-Fiction", "Biography", "Humor"],
    "music": ["Pop", "Rock", "Hip-Hop", "Jazz", "Classical", "Electronic", "Country", "R&B", "Metal", "Indie",
              "Latin", "Blues"],
}
AUDIO_PHRASES = {"energy": "similar energy", "valence": "similar mood", "danceability": "similar groove",
                 "acousticness": "similar sound texture"}
PER_CREATOR_CAP = {"movies": None, "books": 4, "music": 2}


class DomainIndex:
    def __init__(self, spec):
        self.name = spec["name"]
        self.items = spec["items"].reset_index(drop=True)
        self.X = spec["content"]
        self.N = len(self.items)
        self.like_min = spec["like_min"]
        it = self.items
        self.title = it["title"].tolist()
        self.subtitle = it["subtitle"].fillna("").tolist()
        self.genres = it["genres"].tolist()
        self.pop = it["pop"].to_numpy(np.float32)
        self.pop_weight = 0.3 if self.name == "music" else 0.06
        self.conf_floor = 0.4  # candidates with almost no ratings are down-weighted (see _build_cf)
        self._tl = it["title"].str.lower()
        self._key = (it["title"] + " " + it["subtitle"].fillna("")).str.lower()
        self._build_cf(spec["ratings"])
        self.refresh()
        self._build_quality(spec["quality_m"])
        self._build_genre_index()

    # ------------------------------------------------------------------ setup
    def _build_cf(self, r):
        self.has_cf = r is not None and len(r) > 0
        self.cnt = np.zeros(self.N)
        if not self.has_cf:
            return
        u = pd.factorize(r["user"])[0]
        i = r["item"].to_numpy()
        mean = r.groupby("user")["rating"].transform("mean").to_numpy()
        centered = (r["rating"].to_numpy() - mean).astype(np.float32)
        shape = (self.N, u.max() + 1)
        R = sparse.csr_matrix((centered, (i, u)), shape=shape)
        self.Rn = normalize(R).tocsr()
        self.B = sparse.csr_matrix((np.ones(len(i), np.float32), (i, u)), shape=shape)
        self.cnt = np.asarray(self.B.sum(axis=1)).ravel()

    def _confidence(self):
        """0.4..1 multiplier: a candidate with few ratings is a riskier suggestion than a well-rated one."""
        if not self.has_cf:
            return np.ones(self.N, np.float32)
        c = np.minimum(1.0, np.log1p(self.cnt) / np.log1p(15))
        return (self.conf_floor + (1 - self.conf_floor) * c).astype(np.float32)

    def refresh(self):
        """Recompute derived arrays after changing pop_weight / conf_floor (used by evaluate.py)."""
        self.conf = self._confidence()

    def _build_quality(self, m):
        it = self.items
        if self.name == "music":
            self.quality = self.pop.copy()
            return
        n = it["n_ratings"].to_numpy(float)
        avg = it["rating"].fillna(0).to_numpy(float)
        mu = np.average(avg[n > 0], weights=n[n > 0])
        self.quality = ((n * avg + m * mu) / (n + m)).astype(np.float32)

    def _build_genre_index(self):
        gi = {}
        for i, gs in enumerate(self.genres):
            for g in gs:
                gi.setdefault(g, []).append(i)
        self.genre_index = {g: np.array(v) for g, v in gi.items()}

    # ------------------------------------------------------------------ similarity
    def _content(self, s):
        if sparse.issparse(self.X):
            c = (self.X @ self.X[s].T).toarray().ravel()
        else:
            c = self.X @ self.X[s]
        c = np.clip(c, 0, None).astype(np.float32)
        c[s] = 0
        return c

    def _cf(self, s):
        if not self.has_cf or self.cnt[s] < 3:
            return np.zeros(self.N, np.float32)
        v = (self.Rn @ self.Rn[s].T).toarray().ravel()
        common = (self.B @ self.B[s].T).toarray().ravel()
        cf = np.clip(v * common / (common + 10.0), 0, None).astype(np.float32)
        cf[s] = 0
        return cf

    def _alpha(self, s, mode):
        if mode == "content" or not self.has_cf:
            return 0.0
        if mode == "cf":
            return 1.0
        n = self.cnt[s]
        return 0.5 if n >= 20 else 0.25 if n >= 5 else 0.0

    def score(self, seeds, mode="hybrid"):
        """Return (final score vector with seeds masked out, mean normalised collaborative component)."""
        acc = np.zeros(self.N, np.float32)
        cfacc = np.zeros(self.N, np.float32)
        for s in seeds:
            c = self._content(s)
            cn = c / c.max() if c.max() > 0 else c
            cf = self._cf(s)
            cfn = cf / cf.max() if cf.max() > 0 else cf
            a = self._alpha(s, mode)
            acc += (1 - a) * cn + a * cfn
            cfacc += cfn
        acc /= len(seeds)
        cfacc /= len(seeds)
        if mode == "cf":
            f = acc
        else:
            f = (acc + self.pop_weight * self.pop) * self.conf
            if self.name == "music":  # keep suggestions to tracks people actually know
                f = f - 0.35 * (self.pop < 0.2)
        f[list(seeds)] = -np.inf
        return f, cfacc

    def similar(self, seeds, n=24, mode="hybrid"):
        seeds = [int(s) for s in seeds if 0 <= int(s) < self.N]
        if not seeds:
            return []
        f, cfn = self.score(seeds, mode)
        order = np.argsort(-f)[: max(n * 6, 120)]
        cap = PER_CREATOR_CAP[self.name]
        seen_creator, picked = {}, []
        for i in order:
            if not np.isfinite(f[i]) or f[i] <= 0:
                break
            if cap:
                k = self.subtitle[i].lower()
                if seen_creator.get(k, 0) >= cap:
                    continue
                seen_creator[k] = seen_creator.get(k, 0) + 1
            picked.append(int(i))
            if len(picked) == n:
                break
        if not picked:
            return []
        top = float(f[picked[0]])
        out = []
        for i in picked:
            m = int(round(100 * (0.40 + 0.59 * float(f[i]) / top)))
            c = self.card(i)
            c["match"] = max(50, min(99, m))
            c["why"] = self._why(seeds, i, float(cfn[i]))
            out.append(c)
        return out

    def _why(self, seeds, i, cf_strength):
        multi = len(seeds) > 1
        s0 = seeds[0]
        ref = "your list" if multi else self.title[s0]
        seed_genres = {g for s in seeds for g in self.genres[s]}
        shared = [g for g in self.genres[i] if g in seed_genres][:2]
        parts = []
        if self.name == "music":
            if not multi and self.subtitle[i] == self.subtitle[s0]:
                parts.append("Same artist")
            if shared:
                parts.append("Genre: " + ", ".join(shared))
            if not multi:
                a, b = self.items.iloc[s0], self.items.iloc[i]
                close = [AUDIO_PHRASES[k] for k in AUDIO_PHRASES if abs(float(a[k]) - float(b[k])) < 0.1]
                parts += close[:2]
            return " · ".join(parts) or "Similar sound"
        if self.name == "books" and not multi and self.items["authors"].iloc[i] == self.items["authors"].iloc[s0]:
            parts.append("Same author")
        if not multi and cf_strength >= 0.5:
            who = "Viewers" if self.name == "movies" else "Readers"
            parts.append(f"{who} of {ref} also liked this")
        if multi and cf_strength >= 0.5:
            parts.append("Loved by people with similar taste")
        if shared:
            parts.append("Shares " + ", ".join(shared))
        return " · ".join(parts) or "Similar themes"

    # ------------------------------------------------------------------ item views
    def _covers(self, i):
        r = self.items.iloc[i]
        if self.name == "books":
            urls = [r["image_url"]] if r["image_ok"] else []
            for code in (r["isbn13"], r["isbn"]):
                if code:
                    urls.append(f"https://covers.openlibrary.org/b/isbn/{code}-M.jpg?default=false")
            return {"cover_urls": urls}
        if self.name == "movies" and pd.isna(r["tmdbId"]):
            return {}
        return {"cover_api": f"/api/{self.name}/cover/{i}"}

    def card(self, i):
        r = self.items.iloc[i]
        year = None if pd.isna(r["year"]) else int(r["year"])
        c = {"id": int(i), "domain": self.name, "title": self.title[i], "subtitle": self.subtitle[i],
             "year": year, "genres": self.genres[i][:3],
             "rating": None if pd.isna(r["rating"]) else round(float(r["rating"]), 1),
             "n_ratings": int(r["n_ratings"]), "popularity": round(float(self.pop[i]), 2)}
        c.update(self._covers(i))
        return c

    def detail(self, i):
        r = self.items.iloc[i]
        d = self.card(i)
        d["genres"] = self.genres[i]
        if self.name == "movies":
            d["tags"] = [t.title() for t in r["tags"][:6]]
        elif self.name == "books":
            d.update(series=None if pd.isna(r["series"]) else r["series"], authors=r["authors"],
                     tags=list(r["tags"][:6]))
        else:
            d.update(album=r["album_name"], explicit=bool(r["explicit"]), tempo=int(round(r["tempo"])),
                     duration=int(r["duration_ms"] // 1000),
                     audio={k: round(float(r[k]), 2) for k in ("danceability", "energy", "valence", "acousticness")})
        return d

    def cards(self, ids):
        return [self.card(int(i)) for i in ids if 0 <= int(i) < self.N]

    # ------------------------------------------------------------------ search / browse / home
    def search(self, q, limit=8):
        ql = q.lower().strip()
        if len(ql) < 2:
            return []
        idx = np.flatnonzero(self._key.str.contains(ql, regex=False).to_numpy())
        if not len(idx):
            return []
        t = self._tl.iloc[idx]
        rank = np.where(t.str.startswith(ql), 3, np.where(t.str.contains(" " + ql, regex=False), 2, 1))
        order = np.lexsort((-self.pop[idx], -rank))[:limit]
        return [{k: v for k, v in self.card(int(idx[o])).items() if k in ("id", "title", "subtitle", "year", "domain")}
                for o in order]

    def genre_list(self, k=12):
        pref = [g for g in PREFERRED_GENRES[self.name] if len(self.genre_index.get(g, [])) >= 30]
        rest = sorted(self.genre_index, key=lambda g: -len(self.genre_index[g]))
        return (pref + [g for g in rest if g not in pref])[:k]

    def browse(self, genre=None, sort="popular", offset=0, limit=24):
        idx = self.genre_index.get(genre, np.array([], dtype=int)) if genre else np.arange(self.N)
        key = self.quality[idx] if sort == "top" else self.pop[idx] + 1e-6 * self.items["n_ratings"].to_numpy()[idx]
        order = idx[np.argsort(-key, kind="stable")]
        return {"total": int(len(order)), "items": self.cards(order[offset: offset + limit])}

    def home(self):
        rows = [{"title": "Popular right now", "items": self.browse(sort="popular", limit=20)["items"]},
                {"title": "Top rated" if self.name != "music" else "Chart toppers",
                 "items": self.browse(sort="top", limit=20)["items"]}]
        for g in self.genre_list(12)[:4]:
            rows.append({"title": f"{g}", "genre": g, "items": self.browse(g, "popular", limit=20)["items"]})
        return rows
