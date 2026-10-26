"""
recsys.py - AI-based Movie / Book / Music recommendation system.

Implements the design from the project report:
  * PopularityRecommender   - most-rated items (baseline + cold-start fallback)
  * ContentBasedRecommender - TF-IDF item vectors + cosine similarity to a user profile
  * MatrixFactorization     - biased MF trained with SGD (collaborative filtering)
  * HybridRecommender       - weighted blend of the above, with cold-start handling
  * Evaluation helpers      - RMSE, MAE, Precision@K, Recall@K, NDCG@K, coverage

Everything is plain numpy / pandas / scikit-learn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

LIKE_THRESHOLD = 4  # a rating >= 4 counts as "liked" / relevant


# --------------------------------------------------------------------------- #
# 1. Data: synthetic catalogue + ratings, and a MovieLens loader
# --------------------------------------------------------------------------- #
DOMAIN_TAGS = {
    "movie": ["scifi", "romance", "thriller", "comedy", "horror", "drama", "action"],
    "book": ["fantasy", "mystery", "romance", "scifi", "history", "biography", "thriller"],
    "music": ["rock", "pop", "jazz", "classical", "hiphop", "electronic", "folk"],
}

# tag -> (title adjectives, title nouns, description keywords)
_VOCAB = {
    "scifi": ("Distant Silent Quantum Orbital Last Infinite", "Galaxy Signal Colony Horizon Paradox Station",
              "space alien robot future planet starship android time"),
    "romance": ("Tender Summer Golden Secret Endless Gentle", "Letters Promise Kiss Wedding Embrace Heart",
                "love couple wedding heartbreak passion kiss longing marriage"),
    "thriller": ("Dark Cold Hidden Final Silent Fatal", "Witness Deadline Conspiracy Chase Alibi Trap",
                 "chase conspiracy spy killer suspense hostage detective betrayal"),
    "comedy": ("Wild Crazy Awkward Lucky Big Hilarious", "Weekend Roommates Party Mixup Roadtrip Family",
               "funny prank friends slapstick wedding awkward silly misunderstanding"),
    "horror": ("Haunted Cursed Midnight Hollow Pale Screaming", "Manor Whisper Ritual Cellar Doll Forest",
               "ghost haunted demon curse blood nightmare monster possession"),
    "drama": ("Quiet Broken Long Fragile Open Honest", "Years Season Promise Distance Silence Return",
              "family struggle loss redemption grief hope sacrifice community"),
    "action": ("Iron Rapid Burning Savage Lone Steel", "Fury Strike Pursuit Vengeance Ambush Storm",
               "fight explosion chase soldier mission heist battle combat"),
    "fantasy": ("Ancient Crimson Silver Forgotten Emerald Wandering", "Kingdom Dragon Crown Sorcerer Realm Prophecy",
                "magic dragon wizard kingdom quest sword prophecy elf"),
    "mystery": ("Vanishing Curious Missing Hidden Crooked Silent", "Inheritance Letter Clue Mansion Verdict Case",
                "detective clue murder suspect puzzle alibi village inspector"),
    "history": ("Imperial Forgotten Great Lost Ancient Fallen", "Empire Dynasty Revolution Frontier Treaty Campaign",
                "empire war revolution dynasty battle century archive politics"),
    "biography": ("Unfinished Ordinary Restless Daring Complete Private", "Life Journey Memoir Portrait Letters Legacy",
                  "life memoir childhood career legacy diary struggle achievement"),
    "rock": ("Electric Loud Wild Rusty Burning Midnight", "Highway Riot Amplifier Anthem Thunder Garage",
             "guitar drums riff stage loud band anthem distortion"),
    "pop": ("Bright Neon Sweet Shiny Radio Golden", "Heartbeat Dance Chorus Spotlight Summer Sparkle",
            "catchy chorus dance hook radio synth upbeat singalong"),
    "jazz": ("Blue Smoky Late Velvet Swinging Cool", "Quartet Session Lounge Sax Midnight Standard",
             "saxophone improvisation swing trumpet club bebop trio smooth"),
    "classical": ("Grand Solemn Noble Serene Pastoral Royal", "Sonata Symphony Concerto Nocturne Overture Requiem",
                  "orchestra violin piano symphony composer concerto baroque strings"),
    "hiphop": ("Street Gold Raw Urban Real Boom", "Cipher Hustle Flow Block Crown Legacy",
               "rap beat lyrics flow bass sampler street rhyme"),
    "electronic": ("Digital Pulse Neon Binary Cosmic Deep", "Circuit Wave Drift Grid Echo Loop",
                   "synth beat techno bass club ambient loop sequencer"),
    "folk": ("Wandering Old Humble Dusty Green Quiet", "River Hills Porch Harvest Road Ballad",
             "acoustic guitar storytelling banjo country rural harmony ballad"),
}
ALL_TAGS = sorted(_VOCAB)
_ROMAN = ["", " II", " III", " IV", " V", " VI", " VII", " VIII"]


def generate_data(n_users=400, n_items=450, seed=0):
    """Create a synthetic catalogue (movies, books, music) and a sparse rating log.

    Users have hidden tastes over genres; ratings follow those tastes plus noise, and
    users are more likely to rate popular items and items they like (like real logs).
    Returns (items_df, ratings_df). items_df: item_id, title, domain, tags, text.
    """
    rng = np.random.default_rng(seed)
    domains = list(DOMAIN_TAGS)
    rows, used = [], {}
    for i in range(n_items):
        dom = domains[i % len(domains)]
        tags = list(rng.choice(DOMAIN_TAGS[dom], size=int(rng.integers(1, 4)), replace=False))
        adj, noun, _ = _VOCAB[tags[0]]
        a, n = rng.choice(adj.split()), rng.choice(noun.split())
        title = f"{a} {n}" if dom == "music" else f"The {a} {n}"
        c = used.get(title, 0)
        used[title] = c + 1
        title += _ROMAN[min(c, len(_ROMAN) - 1)]
        kws = [w for t in tags for w in _VOCAB[t][2].split()]
        text = " ".join(rng.choice(kws, size=8))
        rows.append((i, title, dom, tags, text))
    items = pd.DataFrame(rows, columns=["item_id", "title", "domain", "tags", "text"])

    # hidden taste model
    tag_idx = {t: j for j, t in enumerate(ALL_TAGS)}
    T = np.zeros((n_items, len(ALL_TAGS)))
    for i, tags in enumerate(items["tags"]):
        for t in tags:
            T[i, tag_idx[t]] = 1.0 / len(tags)
    taste = rng.dirichlet(np.full(len(ALL_TAGS), 0.25), size=n_users)
    dom_aff = rng.uniform(0.4, 1.0, size=(n_users, len(domains)))
    dom_of_item = np.array([domains.index(d) for d in items["domain"]])
    aff = (taste @ T.T) * dom_aff[:, dom_of_item]
    z = (aff - aff.mean()) / aff.std()
    bu, bi = rng.normal(0, 0.3, n_users), rng.normal(0, 0.4, n_items)
    latent = 3.3 + 0.9 * z + bu[:, None] + bi[None, :] + rng.normal(0, 0.5, (n_users, n_items))
    rating_mat = np.clip(np.rint(latent), 1, 5).astype(int)

    pop = 1.0 / (np.arange(1, n_items + 1) ** 0.6)
    rng.shuffle(pop)
    recs = []
    for u in range(n_users):
        k = min(n_items - 1, 12 + int(rng.poisson(28)))
        w = pop * np.exp(0.7 * z[u])
        chosen = rng.choice(n_items, size=k, replace=False, p=w / w.sum())
        recs += [(u, int(i), int(rating_mat[u, i])) for i in chosen]
    return items, pd.DataFrame(recs, columns=["user", "item", "rating"])


def load_movielens(folder, min_user=20, min_item=5, max_users=None, seed=0):
    """Load a MovieLens-style folder (ratings.csv + movies.csv) into the same format."""
    import os
    ratings = pd.read_csv(os.path.join(folder, "ratings.csv"))
    movies = pd.read_csv(os.path.join(folder, "movies.csv"))
    ratings = ratings.rename(columns={"userId": "user", "movieId": "item"})[["user", "item", "rating"]]
    ratings["rating"] = ratings["rating"].round().clip(1, 5).astype(int)
    ratings = ratings[ratings.groupby("item")["rating"].transform("size") >= min_item]
    ratings = ratings[ratings.groupby("user")["rating"].transform("size") >= min_user]
    if max_users:
        keep = np.random.default_rng(seed).choice(ratings["user"].unique(), size=max_users, replace=False) \
            if ratings["user"].nunique() > max_users else ratings["user"].unique()
        ratings = ratings[ratings["user"].isin(keep)]
    movies = movies[movies["movieId"].isin(ratings["item"].unique())].reset_index(drop=True)
    item_map = {m: i for i, m in enumerate(movies["movieId"])}
    user_map = {u: i for i, u in enumerate(sorted(ratings["user"].unique()))}
    ratings = ratings.assign(user=ratings["user"].map(user_map), item=ratings["item"].map(item_map))
    tags = movies["genres"].fillna("").map(
        lambda g: [x.lower().replace("-", "").replace(" ", "") for x in g.split("|") if x and "genres" not in x])
    items = pd.DataFrame({"item_id": range(len(movies)), "title": movies["title"], "domain": "movie",
                          "tags": tags, "text": ""})
    return items, ratings.reset_index(drop=True)


def split_ratings(ratings, n_items, cold_frac=0.08, test_frac=0.2, val_frac=0.1, seed=0):
    """Per-user random split into train / val / test, plus a 'cold item' set.

    Cold items are removed from train and val entirely (the model never sees a rating
    for them), and their ratings go to `cold_test` to measure new-item performance.
    """
    rng = np.random.default_rng(seed)
    cold_items = rng.choice(n_items, size=int(cold_frac * n_items), replace=False)
    is_cold = ratings["item"].isin(cold_items).to_numpy()
    cold_test, warm = ratings[is_cold], ratings[~is_cold]
    parts = {"train": [], "val": [], "test": []}
    for _, g in warm.groupby("user"):
        idx = rng.permutation(len(g))
        n_test, n_val = max(1, round(test_frac * len(g))), max(1, round(val_frac * len(g)))
        parts["test"].append(g.iloc[idx[:n_test]])
        parts["val"].append(g.iloc[idx[n_test:n_test + n_val]])
        parts["train"].append(g.iloc[idx[n_test + n_val:]])
    out = {k: pd.concat(v) for k, v in parts.items()}
    out["cold_test"], out["cold_items"] = cold_test, np.sort(cold_items)
    return out


# --------------------------------------------------------------------------- #
# 2. Models
# --------------------------------------------------------------------------- #
def _pct(x):
    """Rank-normalise to [0, 1], ignoring NaNs (keeps NaN where input is NaN)."""
    out = np.full(len(x), np.nan)
    ok = ~np.isnan(x)
    if ok.sum() > 1:
        out[ok] = (rankdata(x[ok]) - 1) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 1.0
    return out


class PopularityRecommender:
    """Most-rated items first (ties broken by damped mean rating)."""

    def __init__(self, damping=5.0):
        self.damping = damping

    def fit(self, ratings, n_users, n_items):
        self.mu = ratings["rating"].mean()
        g = ratings.groupby("item")["rating"].agg(["sum", "count"]).reindex(range(n_items)).fillna(0)
        self.count = g["count"].to_numpy()
        self.item_mean = (g["sum"].to_numpy() + self.damping * self.mu) / (self.count + self.damping)
        self.rank_scores = self.count + self.item_mean / 10.0  # mean (<=5) only breaks ties
        return self

    def score_user(self, u):
        return self.rank_scores

    def predict(self, users, items):
        return self.item_mean[items]


class ContentBasedRecommender:
    """TF-IDF item vectors; user profile = mean vector of items the user liked."""

    def fit(self, items, ratings, n_users, n_items):
        self.items = items
        texts = [(" ".join(t) + " ") * 3 + d + " " + s for t, d, s in zip(items["tags"], items["domain"], items["text"])]
        self.vec = TfidfVectorizer(sublinear_tf=True)
        self.X = normalize(self.vec.fit_transform(texts)).tocsr()
        liked = ratings[ratings["rating"] >= LIKE_THRESHOLD]
        self.liked = {u: g["item"].to_numpy() for u, g in liked.groupby("user")}
        self.profile = {}
        for u, idx in self.liked.items():
            p = np.asarray(self.X[idx].mean(axis=0)).ravel()
            nrm = np.linalg.norm(p)
            if nrm > 0:
                self.profile[u] = p / nrm
        return self

    def score_user(self, u):
        if u not in self.profile:
            return np.full(self.X.shape[0], np.nan)
        return np.asarray(self.X @ self.profile[u]).ravel()

    def score_query(self, tags):
        q = normalize(self.vec.transform([(" ".join(tags) + " ") * 3]))
        return np.asarray((self.X @ q.T).todense()).ravel()

    def explain(self, u, item, k=2):
        """Titles of the user's liked items most similar to `item`."""
        liked = self.liked.get(u)
        if liked is None or len(liked) == 0:
            return []
        sims = np.asarray((self.X[liked] @ self.X[item].T).todense()).ravel()
        order = np.argsort(-sims)[:k]
        return [self.items["title"].iloc[liked[j]] for j in order if sims[j] > 0]


class MatrixFactorization:
    """Biased matrix factorisation trained with SGD:  r_hat = mu + b_u + b_i + p_u . q_i"""

    def __init__(self, k=32, lr=0.01, reg=0.05, epochs=60, seed=0):
        self.k, self.lr, self.reg, self.epochs, self.seed = k, lr, reg, epochs, seed

    def fit(self, ratings, n_users, n_items):
        rng = np.random.default_rng(self.seed)
        u_all = ratings["user"].to_numpy()
        i_all = ratings["item"].to_numpy()
        r_all = ratings["rating"].to_numpy(dtype=float)
        self.mu = r_all.mean()
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        self.P = rng.normal(0, 0.1, (n_users, self.k))
        self.Q = rng.normal(0, 0.1, (n_items, self.k))
        self.item_count = np.bincount(i_all, minlength=n_items)
        a, L, mu, bu, bi, P, Q = self.lr, self.reg, self.mu, self.bu, self.bi, self.P, self.Q
        order = np.arange(len(r_all))
        for _ in range(self.epochs):
            rng.shuffle(order)
            for n in order:
                u, i = u_all[n], i_all[n]
                pu, qi = P[u].copy(), Q[i].copy()
                e = r_all[n] - (mu + bu[u] + bi[i] + pu @ qi)
                bu[u] += a * (e - L * bu[u])
                bi[i] += a * (e - L * bi[i])
                P[u] += a * (e * qi - L * pu)
                Q[i] += a * (e * pu - L * qi)
        return self

    def predict(self, users, items):
        out = self.mu + self.bu[users] + self.bi[items] + np.einsum("ij,ij->i", self.P[users], self.Q[items])
        return np.clip(out, 1, 5)

    def score_user(self, u):
        s = self.mu + self.bu[u] + self.bi + self.Q @ self.P[u]
        s = s.astype(float)
        s[self.item_count == 0] = np.nan  # no ratings => nothing was learned for this item
        return s


class HybridRecommender:
    """Blend of collaborative (MF) and content-based scores with cold-start fallbacks.

    score(u, i) = a_u * cf(u, i) + (1 - a_u) * content(u, i)    (both rank-normalised)
      * a_u shrinks to 0 for users with few ratings; items with no ratings use content only
      * users with no ratings at all get the popularity ranking
    """

    def __init__(self, items, alpha=0.6, min_ratings=5, **mf_kwargs):
        self.items, self.alpha, self.min_ratings = items, alpha, min_ratings
        self.pop, self.cb, self.mf = PopularityRecommender(), ContentBasedRecommender(), MatrixFactorization(**mf_kwargs)

    def fit(self, ratings, n_users, n_items):
        self.n_users, self.n_items = n_users, n_items
        self.pop.fit(ratings, n_users, n_items)
        self.cb.fit(self.items, ratings, n_users, n_items)
        self.mf.fit(ratings, n_users, n_items)
        self.n_ratings = np.bincount(ratings["user"].to_numpy(), minlength=n_users)
        self.seen = np.zeros((n_users, n_items), dtype=bool)
        self.seen[ratings["user"].to_numpy(), ratings["item"].to_numpy()] = True
        self._pop_pct = _pct(self.pop.rank_scores)
        return self

    def score_user(self, u):
        n_u = self.n_ratings[u] if u < len(self.n_ratings) else 0
        if n_u == 0:
            return self._pop_pct
        cf, cb = _pct(self.mf.score_user(u)), _pct(self.cb.score_user(u))
        a = self.alpha * min(1.0, n_u / self.min_ratings)
        out = np.where(np.isnan(cf), cb, np.where(np.isnan(cb), cf, a * cf + (1 - a) * cb))
        return np.where(np.isnan(out), 0.1 * self._pop_pct, out)

    def recommend(self, u, n=10, domain=None):
        s = self.score_user(u).copy()
        s[self.seen[u]] = -np.inf
        if domain:
            s[(self.items["domain"] != domain).to_numpy()] = -np.inf
        top = np.argsort(-s)[:n]
        rows = []
        for i in top:
            why = self.cb.explain(u, i)
            rows.append((self.items["title"].iloc[i], self.items["domain"].iloc[i],
                         ", ".join(self.items["tags"].iloc[i]), round(float(s[i]), 3),
                         ("because you liked " + " and ".join(f"'{w}'" for w in why)) if why else "popular with similar users"))
        return pd.DataFrame(rows, columns=["title", "domain", "tags", "score", "why"])

    def recommend_for_tags(self, tags, n=10, domain=None):
        """Cold-start: a brand-new user who just picked a few favourite genres."""
        s = 0.7 * _pct(self.cb.score_query(tags)) + 0.3 * self._pop_pct
        if domain:
            s[(self.items["domain"] != domain).to_numpy()] = -np.inf
        top = np.argsort(-s)[:n]
        return pd.DataFrame({"title": self.items["title"].iloc[top].to_numpy(),
                             "domain": self.items["domain"].iloc[top].to_numpy(),
                             "tags": [", ".join(t) for t in self.items["tags"].iloc[top]],
                             "score": np.round(s[top], 3)})


class RandomRecommender:
    """Reference point for the evaluation tables."""

    def score_user(self, u):
        return np.zeros(1)


# --------------------------------------------------------------------------- #
# 3. Evaluation
# --------------------------------------------------------------------------- #
def rating_metrics(model, test):
    """RMSE and MAE on held-out ratings (models that predict ratings only)."""
    pred = model.predict(test["user"].to_numpy(), test["item"].to_numpy())
    err = pred - test["rating"].to_numpy()
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def ranking_metrics(model, n_items, train, test, candidates=None, K=10, seed=0):
    """Precision@K, Recall@K, NDCG@K (relevant = held-out items rated >= 4) and coverage.

    Items the user already rated in `train` are never recommended. `candidates` limits
    which items may be ranked (default: all).
    """
    rng = np.random.default_rng(seed)
    cand = np.ones(n_items, dtype=bool) if candidates is None else candidates
    seen = {u: set(g["item"]) for u, g in train.groupby("user")}
    pos = {u: set(g.loc[g["rating"] >= LIKE_THRESHOLD, "item"]) for u, g in test.groupby("user")}
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    P = R = N = 0.0
    users, recommended = 0, set()
    for u, rel in pos.items():
        if not rel:
            continue
        s = np.asarray(model.score_user(u), dtype=float)
        if s.shape[0] != n_items:
            s = np.zeros(n_items)
        s = np.where(np.isnan(s), (np.nanmin(s) if np.isfinite(s).any() else 0.0) - 1.0, s)
        s = s + 1e-9 * rng.random(n_items)  # random tie-breaking
        allow = cand.copy()
        allow[list(seen.get(u, ()))] = False
        s = np.where(allow, s, -np.inf)
        top = np.argsort(-s)[:min(K, int(allow.sum()))]
        hits = np.array([i in rel for i in top], dtype=float)
        P += hits.sum() / K
        R += hits.sum() / len(rel)
        N += (hits * disc[:len(hits)]).sum() / disc[:min(len(rel), K)].sum()
        recommended.update(int(i) for i in top)
        users += 1
    if users == 0:
        return dict(P=np.nan, R=np.nan, NDCG=np.nan, Coverage=np.nan, users=0)
    return dict(P=P / users, R=R / users, NDCG=N / users, Coverage=len(recommended) / int(cand.sum()), users=users)


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
