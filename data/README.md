# Datasets

The datasets are **not** committed to the repository (size and licences). Download them and place the files as below.

```
data/
├── movies/   movies.csv  ratings.csv  links.csv  tags.csv
├── books/    books.csv  ratings.csv  book_tags.csv  tags.csv
└── music/    spotify-tracks-dataset.csv      (any single CSV with the Spotify Tracks columns works)
```

| Tab | Dataset | Where to get it | Notes |
| --- | --- | --- | --- |
| Movies | **MovieLens `ml-latest-small`** (100,836 ratings, 9,742 movies, 610 users) | GroupLens: grouplens.org/datasets/movielens/ | Free for research and non-commercial use. Cite GroupLens and don't redistribute the files. |
| Books | **Goodbooks-10k** (10,000 books, ~1M ratings in this version) | github.com/zygmuntz/goodbooks-10k, also on Kaggle | Goodreads-derived data; see the repository for terms. |
| Music | **Spotify Tracks Dataset** (114,000 rows, 114 genres, audio features) | Kaggle: search "Spotify Tracks Dataset" | No user ratings; recommendations use audio features and genre. |

Check each dataset's licence before reusing it for anything beyond a personal or portfolio project.
