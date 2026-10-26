"""Quick checks.  Run:  python -m pytest -q   (or:  python test_recsys.py)"""
import numpy as np

import recsys as rs


def small():
    items, ratings = rs.generate_data(n_users=60, n_items=90, seed=1)
    return items, ratings, 60, 90


def test_report_worked_example():
    # Asha liked Inception [1,0,1] and Interstellar [1,0,0]; profile = [1,0,0.5]
    profile = np.mean([[1, 0, 1], [1, 0, 0]], axis=0)
    assert abs(rs.cosine(profile, [1, 0, 1]) - 0.949) < 0.001   # Tenet
    assert rs.cosine(profile, [0, 1, 0]) == 0.0                 # La La Land


def test_mf_beats_global_mean():
    items, ratings, nu, ni = small()
    sp = rs.split_ratings(ratings, ni, seed=1)
    mf = rs.MatrixFactorization(k=8, epochs=15).fit(sp["train"], nu, ni)
    rmse, _ = rs.rating_metrics(mf, sp["test"])
    baseline = np.sqrt(np.mean((sp["test"]["rating"] - sp["train"]["rating"].mean()) ** 2))
    assert rmse < baseline


def test_recommendations_are_unseen_and_sized():
    items, ratings, nu, ni = small()
    hyb = rs.HybridRecommender(items, k=8, epochs=10).fit(ratings, nu, ni)
    recs = hyb.recommend(3, n=7)
    assert len(recs) == 7
    seen_titles = set(items["title"][ratings[ratings["user"] == 3]["item"]])
    assert not (set(recs["title"]) & seen_titles)


def test_domain_filter():
    items, ratings, nu, ni = small()
    hyb = rs.HybridRecommender(items, k=8, epochs=10).fit(ratings, nu, ni)
    assert set(hyb.recommend(0, n=5, domain="music")["domain"]) == {"music"}


def test_new_user_falls_back_to_popularity():
    items, ratings, nu, ni = small()
    train = ratings[ratings["user"] != 5]          # user 5 has no history at all
    hyb = rs.HybridRecommender(items, k=8, epochs=10).fit(train, nu, ni)
    s = hyb.score_user(5)
    assert np.allclose(s, hyb._pop_pct)


def test_cold_item_still_gets_scored_by_content():
    items, ratings, nu, ni = small()
    sp = rs.split_ratings(ratings, ni, seed=1)
    full = sp["train"]
    hyb = rs.HybridRecommender(items, k=8, epochs=10).fit(full, nu, ni)
    cold = sp["cold_items"][0]
    assert hyb.mf.item_count[cold] == 0
    u = int(full["user"].iloc[0])
    assert np.isfinite(hyb.score_user(u)[cold])


def test_tag_cold_start_prefers_matching_genre():
    items, ratings, nu, ni = small()
    hyb = rs.HybridRecommender(items, k=8, epochs=10).fit(ratings, nu, ni)
    recs = hyb.recommend_for_tags(["jazz"], n=5)
    assert sum("jazz" in t for t in recs["tags"]) >= 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
