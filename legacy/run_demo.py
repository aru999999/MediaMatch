"""
run_demo.py - trains and evaluates the recommender, then prints sample recommendations.

    python run_demo.py                       # synthetic movies + books + music
    python run_demo.py --user 7 --topn 8     # recommendations for user 7
    python run_demo.py --movielens ml-latest-small   # use a real MovieLens folder
"""
import argparse
import time

import numpy as np
import pandas as pd

import recsys as rs

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=400)
    ap.add_argument("--items", type=int, default=450)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--user", type=int, default=None, help="user id to show recommendations for")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--k", type=int, default=10, help="K for Precision@K / Recall@K / NDCG@K")
    ap.add_argument("--factors", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--movielens", default=None, help="folder containing ratings.csv and movies.csv")
    ap.add_argument("--max-users", type=int, default=None, help="subsample users (MovieLens only)")
    args = ap.parse_args()

    t0 = time.time()
    if args.movielens:
        items, ratings = rs.load_movielens(args.movielens, max_users=args.max_users, seed=args.seed)
    else:
        items, ratings = rs.generate_data(args.users, args.items, args.seed)
    n_users, n_items = ratings["user"].max() + 1, len(items)
    sparsity = 1 - len(ratings) / (n_users * n_items)
    print(f"Data: {n_users} users, {n_items} items, {len(ratings)} ratings "
          f"({sparsity:.1%} of the rating matrix is empty)")
    if not args.movielens:
        print("      items per domain:", items["domain"].value_counts().to_dict())

    sp = rs.split_ratings(ratings, n_items, seed=args.seed)
    train, val, test = sp["train"], sp["val"], sp["test"]
    cold_items, cold_test = sp["cold_items"], sp["cold_test"]
    warm = np.ones(n_items, dtype=bool)
    warm[cold_items] = False
    print(f"Split: train {len(train)}, validation {len(val)}, test {len(test)}; "
          f"{len(cold_items)} cold items held out ({len(cold_test)} ratings)\n")

    mf_args = dict(k=args.factors, epochs=args.epochs, seed=args.seed)

    # ---- tune the hybrid weight alpha on the validation set --------------------
    print("Tuning hybrid weight alpha on the validation set ...")
    hyb = rs.HybridRecommender(items, **mf_args).fit(train, n_users, n_items)
    best_alpha, best_ndcg = 0.5, -1
    for a in np.linspace(0, 1, 11):
        hyb.alpha = float(a)
        m = rs.ranking_metrics(hyb, n_items, train, val, candidates=warm, K=args.k, seed=args.seed)
        print(f"   alpha={a:.1f}  NDCG@{args.k}={m['NDCG']:.4f}")
        if m["NDCG"] > best_ndcg:
            best_alpha, best_ndcg = float(a), m["NDCG"]
    print(f"   -> best alpha = {best_alpha:.1f}\n")

    # ---- refit on train + validation, evaluate on test -------------------------
    full = pd.concat([train, val])
    hyb = rs.HybridRecommender(items, alpha=best_alpha, **mf_args).fit(full, n_users, n_items)
    pop, cb, mf = hyb.pop, hyb.cb, hyb.mf
    K = args.k

    class GlobalMean:
        def predict(self, u, i):
            return np.full(len(u), full["rating"].mean())

    rows = []
    for name, model, rater in [("Popularity", pop, pop), ("Content-based", cb, None),
                               ("Matrix factorisation", mf, mf), ("Hybrid", hyb, None)]:
        rank = rs.ranking_metrics(model, n_items, full, test, candidates=warm, K=K, seed=args.seed)
        rmse, mae = rs.rating_metrics(rater, test) if rater is not None else (np.nan, np.nan)
        rows.append([name, rmse, mae, rank["P"], rank["R"], rank["NDCG"], rank["Coverage"]])
    table = pd.DataFrame(rows, columns=["Model", "RMSE", "MAE", f"Precision@{K}", f"Recall@{K}",
                                        f"NDCG@{K}", "Coverage"]).set_index("Model")
    g_rmse, g_mae = rs.rating_metrics(GlobalMean(), test)
    print(f"=== Test-set results (warm items, {rank['users']} users with a liked test item) ===")
    print(table.round(4).fillna("-").to_string())
    print(f"(reference: always predicting the global mean gives RMSE {g_rmse:.4f}, MAE {g_mae:.4f})\n")
    table.round(4).to_csv("results.csv")

    # ---- cold-start items: never seen in training ------------------------------
    cand = ~warm
    rows = []
    for name, model in [("Random", rs.RandomRecommender()), ("Popularity", pop), ("Content-based", cb),
                        ("Matrix factorisation", mf), ("Hybrid", hyb)]:
        m = rs.ranking_metrics(model, n_items, full, cold_test, candidates=cand, K=K, seed=args.seed)
        rows.append([name, m["P"], m["R"], m["NDCG"]])
    cold = pd.DataFrame(rows, columns=["Model", f"Precision@{K}", f"Recall@{K}", f"NDCG@{K}"]).set_index("Model")
    print(f"=== Cold-start items: ranking {len(cold_items)} items that have no training ratings ===")
    print(cold.round(4).to_string())
    print()

    # ---- sample recommendations -----------------------------------------------
    rng = np.random.default_rng(args.seed)
    counts = np.bincount(full["user"], minlength=n_users)
    busy = np.where(counts >= 30)[0]
    u = args.user if args.user is not None else int(rng.choice(busy) if len(busy) else counts.argmax())
    mine = full[full["user"] == u].sort_values("rating", ascending=False).head(5)
    print(f"=== User {u} ({counts[u]} ratings). Their top-rated items ===")
    for _, r in mine.iterrows():
        print(f"   {int(r['rating'])}*  {items['title'][r['item']]}  [{items['domain'][r['item']]}: "
              f"{', '.join(items['tags'][r['item']])}]")
    print(f"\nTop {args.topn} recommendations for user {u}:")
    print(hyb.recommend(u, args.topn).to_string(index=False))
    if not args.movielens:
        print(f"\nTop 3 per domain for user {u}:")
        for dom in ("movie", "book", "music"):
            print(f"  {dom}: " + "; ".join(hyb.recommend(u, 3, domain=dom)["title"]))

    tags = ["scifi", "thriller"]
    print(f"\nBrand-new user who picks the genres {tags} (cold start, no history):")
    print(hyb.recommend_for_tags(tags, args.topn).to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f}s. Metrics saved to results.csv")


if __name__ == "__main__":
    main()
