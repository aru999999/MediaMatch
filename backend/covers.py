"""Cover-art lookups (movie posters, album art) with an on-disk cache.

    movies  TMDB            needs a free API key in TMDB_API_KEY (env var or backend/.env)
    music   Deezer, then iTunes Search  (no key needed)
    books   not handled here: the frontend loads Goodreads / Open Library image URLs directly

Results are cached in backend/cache/covers.json. Failures caused by the network are *not* cached,
so a temporary outage doesn't permanently blank out a cover. "No cover exists" answers are cached.
"""
import json
import os
import threading
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "cache", "covers.json")
UA = {"User-Agent": "MediaMatch/1.0 (portfolio project)"}


def load_env(path=os.path.join(HERE, ".env")):
    """Tiny .env reader (KEY=VALUE per line) so no extra dependency is needed."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class NetworkError(Exception):
    pass


def _get_json(url, timeout=6):
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:  # timeouts, DNS, HTTP errors, bad JSON
        raise NetworkError(str(e))


class CoverService:
    def __init__(self, cache_path=CACHE_PATH, fetch=_get_json):
        load_env()
        self.cache_path = cache_path
        self.fetch = fetch
        self.lock = threading.Lock()
        self.throttle = threading.Lock()
        self._last_call = 0.0
        self.cache = {}
        try:
            with open(cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)
        except (OSError, ValueError):
            pass

    @property
    def tmdb_key(self):
        return os.environ.get("TMDB_API_KEY", "").strip()

    def status(self):
        return {"tmdb_configured": bool(self.tmdb_key)}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
        except OSError:
            pass

    def _wait(self, gap=0.25):
        with self.throttle:  # polite to free public APIs
            wait = self._last_call + gap - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    # ------------------------------------------------------------------ resolvers
    def _movie(self, tmdb_id):
        if not self.tmdb_key or tmdb_id is None:
            return None
        self._wait()
        data = self.fetch(f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={self.tmdb_key}")
        path = data.get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None

    def _music(self, title, artist):
        first = artist.split(",")[0].strip()
        clean = title.split(" - ")[0].split(" (")[0].strip()
        queries = [f'artist:"{first}" track:"{clean}"', f"{first} {clean}"]
        for q in queries:
            self._wait()
            data = self.fetch("https://api.deezer.com/search?limit=1&q=" + urllib.parse.quote(q))
            hit = (data.get("data") or [None])[0]
            cover = hit and (hit.get("album") or {}).get("cover_big")
            if cover and "d41d8cd98f00b204e9800998ecf8427e" not in cover:  # that hash is Deezer's "no cover" image
                return cover
        self._wait()
        data = self.fetch("https://itunes.apple.com/search?entity=song&limit=1&term=" +
                          urllib.parse.quote(f"{first} {clean}"))
        res = (data.get("results") or [None])[0]
        return res["artworkUrl100"].replace("100x100", "300x300") if res and res.get("artworkUrl100") else None

    # ------------------------------------------------------------------ public
    def get(self, domain, key, **kw):
        """Return {"url": str|None, "cached": bool}. Raises NetworkError if the lookup could not be completed."""
        ck = f"{domain}:{key}"
        if domain == "movies" and not self.tmdb_key:  # don't cache a miss caused by a missing key
            return {"url": None, "cached": False, "reason": "no_tmdb_key"}
        with self.lock:
            if ck in self.cache:
                return {"url": self.cache[ck], "cached": True}
        if domain == "movies":
            url = self._movie(kw.get("tmdb_id"))
        elif domain == "music":
            url = self._music(kw["title"], kw["artist"])
        else:
            url = None
        with self.lock:
            self.cache[ck] = url
            self._save()
        return {"url": url, "cached": False}
