# AI-Based Movie / Book / Music Recommendation System

A working implementation of the design in the project report: a hybrid recommender that
blends content-based filtering and matrix factorisation, with a popularity fallback for
cold-start users.

## Run it

```bash
pip install -r requirements.txt
python run_demo.py                      # trains, tunes, evaluates, prints recommendations (~20 s)
python run_demo.py --user 7 --topn 5    # recommendations for a specific user
python test_recsys.py                   # 7 quick checks (includes the report's worked example)
```

Options: `--users`, `--items`, `--seed`, `--k` (K for the ranking metrics), `--factors`, `--epochs`.
Metrics are also written to `results.csv`.

## Files

| File | What it holds |
| --- | --- |
| `recsys.py` | Data generator + MovieLens loader, the four models, evaluation metrics |
| `run_demo.py` | Split, tune alpha, evaluate, cold-start experiments, sample recommendations |
| `test_recsys.py` | Unit tests |

## What is implemented (maps to report section 6)

- **Popularity**: most-rated items first. Baseline, and the fallback for users with no history.
- **Content-based**: TF-IDF vectors from genre tags, domain and description; user profile = mean
  vector of items rated >= 4; items ranked by cosine similarity. Also produces the
  "because you liked ..." explanations.
- **Matrix factorisation**: `r_hat = mu + b_u + b_i + p_u . q_i`, trained with SGD and L2 regularisation.
- **Hybrid**: rank-normalised blend `alpha * CF + (1 - alpha) * content`. `alpha` is tuned on a
  validation split and shrinks for users with few ratings. Items with no ratings use content only;
  users with no ratings get popularity. A brand-new user can also pick favourite genres
  (`recommend_for_tags`).

Metrics: RMSE, MAE (rating prediction), Precision@K, Recall@K, NDCG@K, and catalogue coverage
(ranking; "relevant" = a held-out item the user rated >= 4).

## Results on the bundled synthetic data

400 users, 450 items (150 each of movies, books, music), 16,008 ratings, 91% sparse. Test split:

| Model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 | Coverage |
| --- | --- | --- | --- | --- | --- | --- |
| Popularity | 1.031 | 0.878 | 0.056 | 0.128 | 0.110 | 0.05 |
| Content-based | - | - | 0.117 | 0.271 | 0.236 | 0.99 |
| Matrix factorisation | 0.886 | 0.708 | 0.052 | 0.123 | 0.100 | 0.70 |
| **Hybrid** | - | - | **0.121** | **0.286** | **0.254** | 0.95 |

Cold-start items (36 items with zero training ratings), NDCG@10: random 0.151, popularity 0.151,
matrix factorisation 0.151, content-based and hybrid 0.560.

How to read this honestly:

- The data is **synthetic**: ratings are generated from hidden genre tastes, which favours content
  features. The numbers show the system works and how the pieces behave, not how it would score on
  real users. Use real data for the headline numbers in your report (below).
- Matrix factorisation predicts ratings best (lowest RMSE) but ranks poorly here, because users
  mostly rate items they were already exposed to. This is the popularity-bias and sparsity
  limitation from the report. The validation search therefore picks a small alpha (0.1).
- Re-running with another `--seed` changes the numbers slightly; the ordering of the models is
  what to look at.

## Use real data (MovieLens)

Download a MovieLens dataset (for example `ml-latest-small` from grouplens.org), unzip it, and run:

```bash
python run_demo.py --movielens path/to/ml-latest-small
python run_demo.py --movielens path/to/ml-latest-small --max-users 300   # faster
```

It reads `ratings.csv` and `movies.csv` (genres become the content tags). MF training is pure
Python SGD, so a large dataset takes a few minutes; lower `--epochs` or use `--max-users`.
The loader was checked on a small file in the MovieLens format, not on the full download.
Books and music need their own catalogue and ratings in the same shape: an items table
(`item_id, title, domain, tags, text`) and a ratings table (`user, item, rating`).
