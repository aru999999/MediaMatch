"""MediaMatch backend: Flask REST API + static frontend.

    python backend/app.py                 -> http://127.0.0.1:5000
    python backend/app.py --port 8000
"""
import argparse
import csv
import os
import time

from flask import Flask, abort, jsonify, request, send_from_directory

from covers import CoverService, NetworkError
from data_loader import DATA_DIR, LOADERS
from engine import DomainIndex

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(os.path.dirname(HERE), "frontend")
DOMAINS = ("movies", "books", "music")


def build_indexes(data_dir=DATA_DIR, verbose=False):
    out = {}
    for name in DOMAINS:
        t = time.time()
        out[name] = DomainIndex(LOADERS[name](data_dir))
        if verbose:
            print(f"  {name:<7} {out[name].N:>6} items ready ({time.time() - t:.1f}s)")
    return out


def create_app(indexes=None, covers=None, results_path=os.path.join(HERE, "results.csv")):
    app = Flask(__name__, static_folder=os.path.join(FRONTEND, "static"), static_url_path="/static")
    app.json.sort_keys = False
    idx = indexes or build_indexes(verbose=True)
    cov = covers or CoverService()

    def dom(name):
        if name not in idx:
            abort(404, description="unknown domain")
        return idx[name]

    def int_arg(name, default, lo, hi):
        try:
            return max(lo, min(hi, int(request.args.get(name, default))))
        except ValueError:
            abort(400, description=f"{name} must be an integer")

    @app.errorhandler(400)
    @app.errorhandler(404)
    def err(e):
        return jsonify({"error": e.description}), e.code

    @app.get("/")
    def index():
        return send_from_directory(FRONTEND, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/meta")
    def meta():
        return jsonify({
            "domains": {n: {"items": d.N, "genres": d.genre_list(12), "has_ratings": d.has_cf,
                            "n_ratings": int(d.cnt.sum()) if d.has_cf else 0} for n, d in idx.items()},
            "covers": cov.status(), "has_results": os.path.exists(results_path)})

    @app.get("/api/<domain>/home")
    def home(domain):
        return jsonify({"rows": dom(domain).home()})

    @app.get("/api/<domain>/search")
    def search(domain):
        return jsonify({"results": dom(domain).search(request.args.get("q", ""), int_arg("limit", 8, 1, 20))})

    @app.get("/api/<domain>/browse")
    def browse(domain):
        sort = request.args.get("sort", "popular")
        return jsonify(dom(domain).browse(request.args.get("genre") or None, "top" if sort == "top" else "popular",
                                          int_arg("offset", 0, 0, 100000), int_arg("limit", 24, 1, 60)))

    @app.get("/api/<domain>/items")
    def items(domain):
        try:
            ids = [int(x) for x in request.args.get("ids", "").split(",") if x][:60]
        except ValueError:
            abort(400, description="ids must be comma-separated integers")
        return jsonify({"items": dom(domain).cards(ids)})

    @app.get("/api/<domain>/item/<int:i>")
    def item(domain, i):
        d = dom(domain)
        if not 0 <= i < d.N:
            abort(404, description="unknown item")
        return jsonify(d.detail(i))

    @app.get("/api/<domain>/similar/<int:i>")
    def similar(domain, i):
        d = dom(domain)
        if not 0 <= i < d.N:
            abort(404, description="unknown item")
        return jsonify({"seed": d.detail(i), "results": d.similar([i], int_arg("n", 24, 1, 48))})

    @app.post("/api/<domain>/for_list")
    def for_list(domain):
        body = request.get_json(silent=True) or {}
        try:
            ids = [int(x) for x in body.get("ids", [])][:30]
        except (TypeError, ValueError):
            abort(400, description="ids must be integers")
        if not ids:
            abort(400, description="send {\"ids\": [..]}")
        return jsonify({"results": dom(domain).similar(ids, min(48, int(body.get("n", 20))))})

    @app.get("/api/<domain>/cover/<int:i>")
    def cover(domain, i):
        d = dom(domain)
        if not 0 <= i < d.N or domain == "books":
            abort(404, description="no cover lookup for this item")
        r = d.items.iloc[i]
        try:
            if domain == "movies":
                tm = None if r["tmdbId"] != r["tmdbId"] else int(r["tmdbId"])
                res = cov.get("movies", str(i), tmdb_id=tm)
            else:
                res = cov.get("music", str(i), title=r["title"], artist=r["subtitle"])
        except NetworkError:
            return jsonify({"url": None, "retry": True})
        resp = jsonify({"url": res["url"], **({"reason": res["reason"]} if "reason" in res else {})})
        resp.headers["Cache-Control"] = "public, max-age=86400" if res["url"] else "no-store"
        return resp

    @app.get("/api/results")
    def results():
        if not os.path.exists(results_path):
            return jsonify([])
        with open(results_path, newline="", encoding="utf-8") as f:
            return jsonify(list(csv.DictReader(f)))

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print("Loading datasets and building indexes ...")
    application = create_app()
    print(f"Ready -> http://{args.host}:{args.port}")
    application.run(host=args.host, port=args.port, debug=False, threaded=True)
