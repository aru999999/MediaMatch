"""Offline evaluation (leave-one-out) for the movie and book recommenders.

For a sample of users we hide ONE liked item, then try to recover it from up to 10 of their other liked
items. The held-out ratings are removed from the model before it is built, so there is no leakage.

    python backend/evaluate.py                # ~1-3 minutes, writes backend/results.csv
    python backend/evaluate.py --users 400

Metrics (top-10): HitRate = was the hidden item in the top 10; NDCG = rank-aware version; MRR = 1/rank;
Coverage = share of the catalogue that ever gets recommended. Music has no ratings, so it isn't evaluated.
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd

from data_loader import LOADERS
from engine import DomainIndex

HERE = os.path.dirname(os.path.abspath(__file__))
K = 10


def evaluate(name, n_users=250, seed=0, n_seeds=10, tweak=None):
    rng = np.random.default_rng(seed)
    spec = LOADERS[name]()
    r = spec["ratings"]
    liked = r[r["rating"] >= spec["like_min"]].groupby("user")["item"].apply(np.array)
    eligible = liked[liked.map(len) >= 8].index.to_numpy()
    users = rng.choice(eligible, min(n_users, len(eligible)), replace=False)
    held = {u: int(rng.choice(liked[u])) for u in users}

    drop = pd.DataFrame({"user": list(held), "item": list(held.values())})
    train = r.merge(drop, how="left", indicator=True)
    train = train[train["_merge"] == "left_only"][["user", "item", "rating"]]
    D = DomainIndex(dict(spec, ratings=train))
    if tweak:
        tweak(D)
        D.refresh()

    seen = train[train["user"].isin(users)].groupby("user")["item"].apply(np.array)
    liked_tr = train[(train["user"].isin(users)) & (train["rating"] >= spec["like_min"])].groupby("user")["item"].apply(np.array)

    methods = {"Popularity": None, "Content only": "content", "Collaborative only": "cf", "Hybrid": "hybrid"}
    stats = {m: dict(hit=0, ndcg=0.0, mrr=0.0, recs=set()) for m in methods}
    for u in users:
        seeds = rng.choice(liked_tr[u], min(n_seeds, len(liked_tr[u])), replace=False).tolist()
        for m, mode in methods.items():
            f = D.pop.astype(np.float64).copy() if mode is None else D.score(seeds, mode)[0].astype(np.float64)
            f[seen[u]] = -np.inf
            f[seeds] = -np.inf
            rank = int((f > f[held[u]]).sum()) + 1
            s = stats[m]
            s["hit"] += rank <= K
            s["ndcg"] += 1 / np.log2(rank + 1) if rank <= K else 0
            s["mrr"] += 1 / rank
            s["recs"].update(np.argpartition(-f, K)[:K].tolist())
    n = len(users)
    return [dict(Domain=name.capitalize(), Model=m, **{"HitRate@10": s["hit"] / n, "NDCG@10": s["ndcg"] / n,
            "MRR": s["mrr"] / n, "Coverage@10": len(s["recs"]) / D.N}) for m, s in stats.items()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = []
    for name in ("movies", "books"):
        print(f"Evaluating {name} ...")
        rows += evaluate(name, a.users, a.seed)
    out = os.path.join(HERE, "results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("Saved", out)
