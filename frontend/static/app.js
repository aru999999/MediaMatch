/* MediaMatch frontend: vanilla JS, hash routing, no build step. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const DOMAINS = {
  movies: { label: "Movie", eyebrow: "Movies", title: "Find your next favourite movie",
    sub: "Type a film you love. We'll show what to watch next, with a match score.", ph: "Search a movie, e.g. The Matrix" },
  books: { label: "Book", eyebrow: "Books", title: "Find your next great read",
    sub: "Type a book you loved and discover what to read next.", ph: "Search a book, e.g. The Hunger Games" },
  music: { label: "Song", eyebrow: "Music", title: "Find your next favourite song",
    sub: "Type a track you can't stop playing and we'll find songs with the same vibe.", ph: "Search a song, e.g. Blinding Lights" },
};
const S = { domain: "movies", meta: null, items: new Map(), genre: null, reqId: 0 };
const key = (d, id) => `${d}:${id}`;

/* ---------- helpers ---------- */
async function api(path, body) {
  const opts = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("show"), 1800);
}
const hue = s => { let h = 0; for (const c of s) h = (h * 31 + c.charCodeAt(0)) % 360; return h; };
const initials = t => (t.replace(/^(the|a|an)\s+/i, "").match(/[\p{L}\p{N}]/gu) || ["?"]).slice(0, 2).join("").toUpperCase();
const fmtN = n => Number(n).toLocaleString();

/* ---------- My List (per domain, localStorage) ---------- */
const listKey = d => `mm_list_${d}`;
function getList(d) { try { return JSON.parse(localStorage.getItem(listKey(d)) || "[]"); } catch { return []; } }
function setList(d, v) { try { localStorage.setItem(listKey(d), JSON.stringify(v)); } catch { /* storage unavailable */ } }
function toggleList(d, id) {
  const l = getList(d), i = l.indexOf(id);
  i >= 0 ? l.splice(i, 1) : l.unshift(id);
  setList(d, l);
  toast(i >= 0 ? "Removed from My List" : "Added to My List");
  $$(`.card[data-domain="${d}"][data-id="${id}"] .heart, .d-heart[data-id="${id}"]`).forEach(h => {
    h.classList.toggle("on", i < 0);
    if (h.classList.contains("d-heart")) h.textContent = i < 0 ? "✓ In My List" : "+ My List";
  });
  return i < 0;
}

/* ---------- covers ---------- */
function tryUrls(img, urls, i = 0) {
  if (i >= urls.length) return;
  const probe = new Image();
  probe.referrerPolicy = "no-referrer";
  probe.onload = () => { if (probe.naturalWidth > 8) { img.src = probe.src; img.classList.add("ok"); } else tryUrls(img, urls, i + 1); };
  probe.onerror = () => tryUrls(img, urls, i + 1);
  probe.src = urls[i];
}
const coverQueue = []; let coverActive = 0;
function pumpCovers() {
  while (coverActive < 4 && coverQueue.length) {
    const p = coverQueue.shift(); coverActive++;
    fetch(p.dataset.api).then(r => r.json()).then(r => {
      if (r.url) tryUrls($("img", p), [r.url]);
      else if (r.retry && !p.dataset.retried) { p.dataset.retried = 1; setTimeout(() => { coverQueue.push(p); pumpCovers(); }, 4000); }
    }).catch(() => {}).finally(() => { coverActive--; pumpCovers(); });
  }
}
const io = "IntersectionObserver" in window ? new IntersectionObserver(es => es.forEach(e => {
  if (e.isIntersecting) { io.unobserve(e.target); coverQueue.push(e.target); pumpCovers(); }
}), { rootMargin: "300px" }) : null;

function hydrateCovers(root) {
  $$(".poster:not([data-done])", root).forEach(p => {
    p.dataset.done = 1;
    const it = S.items.get(key(p.dataset.domain, p.dataset.id));
    if (!it) return;
    if (it.cover_urls && it.cover_urls.length) tryUrls($("img", p), it.cover_urls);
    else if (it.cover_api) { p.dataset.api = it.cover_api; io ? io.observe(p) : (coverQueue.push(p), pumpCovers()); }
  });
}

/* ---------- card rendering ---------- */
function subline(it) {
  if (it.domain === "movies") return [it.year, it.genres[0]].filter(Boolean).join(" · ");
  return it.subtitle;
}
function posterHtml(it, withHeart = true) {
  S.items.set(key(it.domain, it.id), it);
  const on = getList(it.domain).includes(it.id);
  return `<div class="poster" style="--h:${hue(it.title)}" data-domain="${it.domain}" data-id="${it.id}">
    <img alt="" decoding="async" referrerpolicy="no-referrer">
    <div class="ph"><b>${esc(initials(it.title))}</b><span>${esc(it.title)}</span></div>
    ${it.rating && it.domain !== "music" ? `<span class="rate-pill">★ ${it.rating.toFixed(1)}</span>` : ""}
    ${withHeart ? `<button class="heart ${on ? "on" : ""}" aria-label="Toggle My List" data-id="${it.id}" data-domain="${it.domain}">
      <svg viewBox="0 0 24 24"><path d="M12 21s-8-5.3-8-11a4.7 4.7 0 0 1 8-3.3A4.7 4.7 0 0 1 20 10c0 5.7-8 11-8 11z"/></svg></button>` : ""}
  </div>`;
}
function cardHtml(it) {
  return `<article class="card" tabindex="0" role="button" data-id="${it.id}" data-domain="${it.domain}" aria-label="${esc(it.title)}">
    ${posterHtml(it)}
    <div class="meta"><div class="t">${esc(it.title)}</div><div class="s">${esc(subline(it))}</div>
    ${it.match != null ? `<div class="match"><span>${it.match}% match</span><div class="bar"><i style="width:${it.match}%"></i></div></div>` : ""}
    ${it.why ? `<div class="why">${esc(it.why)}</div>` : ""}</div></article>`;
}
const skeletonRow = (n = 9) => `<section class="row"><h2>&nbsp;</h2><div class="rail">${'<div class="skel"></div>'.repeat(n)}</div></section>`;
function rowHtml(title, items, sub = "") {
  if (!items.length) return "";
  return `<section class="row"><h2>${esc(title)} ${sub ? `<small>${esc(sub)}</small>` : ""}</h2>
    <div class="rail-wrap"><button class="arrow l" aria-label="Scroll left">‹</button>
    <div class="rail">${items.map(cardHtml).join("")}</div><button class="arrow r" aria-label="Scroll right">›</button></div></section>`;
}
function show(html) { const v = $("#view"); v.innerHTML = html; hydrateCovers(v); return v; }
function fail(e) { show(`<p class="empty">${esc(e.message === "Failed to fetch" ? "Can't reach the API. Is the Flask server running?" : e.message)}</p>`); }

/* ---------- views ---------- */
async function renderHome() {
  const d = S.domain, req = ++S.reqId;
  show(skeletonRow() + skeletonRow());
  try {
    const list = getList(d);
    const [home, mine, because] = await Promise.all([
      api(`/api/${d}/home`),
      list.length ? api(`/api/${d}/items?ids=${list.join(",")}`) : null,
      list.length ? api(`/api/${d}/for_list`, { ids: list.slice(0, 10), n: 20 }) : null,
    ]);
    if (req !== S.reqId) return;
    let html = "";
    if (mine) html += rowHtml("My List", mine.items.map(i => i), `${list.length} saved`);
    if (because) html += rowHtml("Picked from your list", because.results);
    html += home.rows.map(r => rowHtml(r.title, r.items)).join("");
    show(html);
  } catch (e) { if (req === S.reqId) fail(e); }
}

async function renderLike(id) {
  const d = S.domain, req = ++S.reqId;
  show(skeletonRow(12));
  try {
    const r = await api(`/api/${d}/similar/${id}?n=30`);
    if (req !== S.reqId) return;
    const seed = r.seed; S.items.set(key(d, seed.id), seed);
    $("#q").value = seed.title; $("#q-clear").hidden = false;
    show(`<div class="result-head">
        <div class="mini">${posterHtml(seed, false)}</div>
        <div><p class="eyebrow" style="margin:0">Because you picked</p><h2>${esc(seed.title)}</h2><p>${esc(subline(seed))}</p></div>
        <button class="btn" id="clear-search">Clear</button></div>
      <h2 class="section-title">Top picks for you <small>${r.results.length} matches</small></h2>
      ${r.results.length ? `<div class="grid">${r.results.map(cardHtml).join("")}</div>` : '<p class="empty">No close matches found.</p>'}`);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (e) { if (req === S.reqId) fail(e); }
}

async function renderGenre(genre, offset = 0) {
  const d = S.domain, req = ++S.reqId;
  if (!offset) show(skeletonRow(12));
  try {
    const r = await api(`/api/${d}/browse?genre=${encodeURIComponent(genre)}&offset=${offset}&limit=36`);
    if (req !== S.reqId) return;
    const cards = r.items.map(cardHtml).join("");
    if (!offset) {
      show(`<h2 class="section-title">${esc(genre)} <small>${fmtN(r.total)} titles</small></h2><div class="grid" id="genre-grid">${cards}</div>
        ${r.total > 36 ? '<button class="btn more" id="more">Load more</button>' : ""}`);
      $("#more")?.addEventListener("click", e => { e.target.dataset.off = 36; renderGenreMore(genre, e.target); });
    }
  } catch (e) { if (req === S.reqId) fail(e); }
}
async function renderGenreMore(genre, btn) {
  const off = +btn.dataset.off, d = S.domain;
  btn.disabled = true;
  try {
    const r = await api(`/api/${d}/browse?genre=${encodeURIComponent(genre)}&offset=${off}&limit=36`);
    const grid = $("#genre-grid"); grid.insertAdjacentHTML("beforeend", r.items.map(cardHtml).join("")); hydrateCovers(grid);
    btn.dataset.off = off + 36; btn.disabled = false;
    if (off + 36 >= r.total) btn.remove();
  } catch (e) { btn.disabled = false; toast(e.message); }
}

/* ---------- detail modal ---------- */
const modal = $("#modal");
function closeModal() { modal.hidden = true; document.body.style.overflow = ""; }
function openModalShell(html) { $("#modal-body").innerHTML = html; modal.hidden = false; document.body.style.overflow = "hidden"; modal.scrollTop = 0; }

async function openItem(d, id) {
  openModalShell('<div class="pad"><div class="skel" style="flex:none;width:200px"></div></div>');
  try {
    const [it, sim] = await Promise.all([api(`/api/${d}/item/${id}`), api(`/api/${d}/similar/${id}?n=12`)]);
    S.items.set(key(d, id), it);
    const inList = getList(d).includes(id);
    const stats = [];
    if (it.rating) stats.push(`<span><span class="star">★ ${it.rating.toFixed(1)}</span> ${d === "movies" ? `· ${fmtN(it.n_ratings)} ratings` : `· ${fmtN(it.n_ratings)} ratings on Goodreads`}</span>`);
    if (it.year) stats.push(`<span>${it.year}</span>`);
    if (d === "music") {
      stats.push(`<span>${Math.floor(it.duration / 60)}:${String(it.duration % 60).padStart(2, "0")}</span>`, `<span>${it.tempo} BPM</span>`);
      if (it.explicit) stats.push("<span>Explicit</span>");
      stats.push(`<span>Popularity ${Math.round(it.popularity * 100)}/100</span>`);
    }
    const tags = [...it.genres.map(g => `<span class="tag g">${esc(g)}</span>`), ...(it.tags || []).slice(0, 5).map(t => `<span class="tag">${esc(t)}</span>`)].join("");
    const MLAB = { danceability: "Danceability", energy: "Energy", valence: "Mood (positivity)", acousticness: "Acoustic" };
    const meters = d === "music" ? `<div class="meters">${Object.entries(it.audio).map(([k, v]) =>
      `<div class="meter"><label><span>${MLAB[k] || k}</span><span>${Math.round(v * 100)}</span></label><div class="bar"><i style="width:${v * 100}%"></i></div></div>`).join("")}</div>` : "";
    const sub = d === "music" ? `${it.subtitle} · ${it.album}` : d === "books" ? (it.series ? `${it.authors} · ${it.series}` : it.authors) : "";
    openModalShell(`<div class="d-top"><div>${posterHtml(it, false)}</div>
      <div class="d-info"><p class="eyebrow" style="margin:0">${DOMAINS[d].label}</p><h2 id="m-title">${esc(it.title)}</h2>
        <div class="d-sub">${esc(sub)}</div>
        <div class="d-stats">${stats.join("")}</div><div class="tags">${tags}</div>${meters}
        <div class="d-actions"><button class="btn primary" id="d-find">Show full match list</button>
          <button class="btn d-heart ${inList ? "on" : ""}" data-id="${id}">${inList ? "✓ In My List" : "+ My List"}</button></div></div></div>
      <div class="d-more"><h3 class="section-title">More like this <small>closest matches first</small></h3>
        <div class="grid">${sim.results.map(cardHtml).join("")}</div></div>`);
    hydrateCovers($("#modal-body"));
    $("#d-find").onclick = () => { closeModal(); location.hash = `#/${d}/like/${id}`; };
  } catch (e) { openModalShell(`<p class="empty pad">${esc(e.message)}</p>`); }
}

async function openResults() {
  openModalShell('<div class="pad"><p class="empty">Loading…</p></div>');
  try {
    const rows = await api("/api/results");
    if (!rows.length) { openModalShell('<p class="empty pad">No results yet. Run <code>python backend/evaluate.py</code>.</p>'); return; }
    const cols = Object.keys(rows[0]);
    const by = {}; rows.forEach(r => (by[r[cols[0]]] ||= []).push(r));
    openModalShell(`<div class="pad"><h2 style="margin-top:0">Offline evaluation</h2>
      <p class="d-sub">Leave-one-out test: hide one liked item per user, then see whether the model finds it from their other likes.
      Higher is better. Hit rate and NDCG are measured on the top 10.</p>
      ${Object.entries(by).map(([k, rs]) => `<h3 class="section-title">${esc(k)}</h3><div style="overflow-x:auto"><table class="results-table">
        <tr>${cols.slice(1).map(c => `<th>${esc(c)}</th>`).join("")}</tr>
        ${rs.map(r => `<tr>${cols.slice(1).map((c, i) => `<td>${i === 0 ? esc(r[c]) : (+r[c]).toFixed(3)}</td>`).join("")}</tr>`).join("")}</table></div>`).join("")}</div>`);
  } catch (e) { openModalShell(`<p class="empty pad">${esc(e.message)}</p>`); }
}

/* ---------- search + autocomplete ---------- */
const q = $("#q"), sug = $("#suggest");
let sugItems = [], sugSel = -1, sugTimer, sugReq = 0;
function closeSuggest() { sug.hidden = true; sugSel = -1; }
function paintSuggest() {
  sug.innerHTML = sugItems.length
    ? sugItems.map((s, i) => `<li role="option" data-i="${i}" aria-selected="${i === sugSel}"><span class="st">${esc(s.title)}</span><span class="ss">${esc([s.year, s.subtitle].filter(Boolean).join(" · "))}</span></li>`).join("")
    : '<li class="none">No matches. Try another spelling.</li>';
  sug.hidden = false;
}
function pick(s) { closeSuggest(); q.value = s.title; location.hash = `#/${S.domain}/like/${s.id}`; q.blur(); }
q.addEventListener("input", () => {
  $("#q-clear").hidden = !q.value;
  clearTimeout(sugTimer);
  if (q.value.trim().length < 2) { closeSuggest(); return; }
  sugTimer = setTimeout(async () => {
    const my = ++sugReq;
    try { const r = await api(`/api/${S.domain}/search?q=${encodeURIComponent(q.value.trim())}`); if (my === sugReq) { sugItems = r.results; sugSel = -1; paintSuggest(); } } catch { /* ignore */ }
  }, 160);
});
q.addEventListener("keydown", async e => {
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    if (!sugItems.length) return; e.preventDefault();
    sugSel = (sugSel + (e.key === "ArrowDown" ? 1 : -1) + sugItems.length) % sugItems.length; paintSuggest();
  } else if (e.key === "Enter") {
    e.preventDefault();
    if (sugSel >= 0) return pick(sugItems[sugSel]);
    if (!q.value.trim()) return;
    const r = await api(`/api/${S.domain}/search?q=${encodeURIComponent(q.value.trim())}`).catch(() => ({ results: [] }));
    r.results.length ? pick(r.results[0]) : (sugItems = [], paintSuggest());
  } else if (e.key === "Escape") closeSuggest();
});
sug.addEventListener("mousedown", e => { const li = e.target.closest("li[data-i]"); if (li) { e.preventDefault(); pick(sugItems[+li.dataset.i]); } });
document.addEventListener("click", e => { if (!e.target.closest("#searchbox")) closeSuggest(); });
$("#q-clear").addEventListener("click", () => { q.value = ""; $("#q-clear").hidden = true; closeSuggest(); if (/\/like\//.test(location.hash)) location.hash = `#/${S.domain}`; q.focus(); });

/* ---------- global click handling ---------- */
document.addEventListener("click", e => {
  const heart = e.target.closest(".heart");
  if (heart) { e.stopPropagation(); toggleList(heart.dataset.domain, +heart.dataset.id); return; }
  const dh = e.target.closest(".d-heart"); if (dh) { toggleList(S.domain, +dh.dataset.id); return; }
  const arrow = e.target.closest(".arrow");
  if (arrow) { const rail = $(".rail", arrow.parentElement); rail.scrollBy({ left: (arrow.classList.contains("r") ? 1 : -1) * rail.clientWidth * 0.85, behavior: "smooth" }); return; }
  if (e.target.closest("[data-close]")) { closeModal(); return; }
  if (e.target.id === "clear-search") { q.value = ""; $("#q-clear").hidden = true; location.hash = `#/${S.domain}`; return; }
  const card = e.target.closest(".card");
  if (card) openItem(card.dataset.domain, +card.dataset.id);
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !modal.hidden) closeModal();
  if ((e.key === "Enter" || e.key === " ") && e.target.classList?.contains("card")) { e.preventDefault(); openItem(e.target.dataset.domain, +e.target.dataset.id); }
  if (e.key === "/" && !/INPUT|TEXTAREA/.test(document.activeElement.tagName) && modal.hidden) { e.preventDefault(); q.focus(); }
});
$("#results-link").addEventListener("click", e => { e.preventDefault(); openResults(); });

/* ---------- theme ---------- */
$("#theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("mm_theme", next); } catch { /* ignore */ }
});

/* ---------- routing ---------- */
function setDomain(d) {
  const changed = S.domain !== d;
  S.domain = d; document.body.dataset.domain = d;
  $$(".tab").forEach(t => { const on = t.dataset.domain === d; t.classList.toggle("active", on); t.setAttribute("aria-selected", on); });
  const c = DOMAINS[d];
  $("#hero-eyebrow").textContent = c.eyebrow; $("#hero-title").textContent = c.title; $("#hero-sub").textContent = c.sub; q.placeholder = c.ph;
  if (changed) { q.value = ""; $("#q-clear").hidden = true; closeSuggest(); }
  document.title = `MediaMatch · ${c.eyebrow}`;
  if (S.meta) {
    $("#genre-chips").innerHTML = S.meta.domains[d].genres.map(g => `<a class="chip ${S.genre === g ? "on" : ""}" href="#/${d}/genre/${encodeURIComponent(g)}">${esc(g)}</a>`).join("");
    const a = d === "movies" ? (S.meta.covers.tmdb_configured ? "Movie posters and data from TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB."
        : "Tip: add a free TMDB API key to show real movie posters (see the README).")
      : d === "music" ? "Album art via Deezer and iTunes Search." : "Book covers via Goodreads and Open Library.";
    $("#attribution").textContent = a;
  }
}
function route() {
  const [, d0, kind, arg] = location.hash.split("/");
  const d = DOMAINS[d0] ? d0 : "movies";
  S.genre = kind === "genre" ? decodeURIComponent(arg || "") : null;
  document.body.classList.toggle("compact", kind === "like" && !!arg);
  setDomain(d);
  if (kind === "like" && arg) renderLike(+arg);
  else if (S.genre) renderGenre(S.genre);
  else renderHome();
  if (kind !== "like") window.scrollTo({ top: 0 });
}
window.addEventListener("hashchange", route);

(async function init() {
  try {
    S.meta = await api("/api/meta");
    $("#results-link").hidden = !S.meta.has_results;
    route();
  } catch (e) { fail(e); }
})();
