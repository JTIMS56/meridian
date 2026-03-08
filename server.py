"""
MERIDIAN // Flask Web Server (v2 - Profile-Aware)
===================================================
Each profile now has its own output files:
  output/meridian_nlp_chancay.json
  output/meridian_nlp_taiwan.json
  output/meridian_nlp_plan_pacific.json
  output/meridian_nlp_bri_africa.json

API: /api/nlp?profile=chancay  /api/feed?profile=taiwan  etc.

Run:    py -3.11 server.py
Access: http://localhost:5000

Cloud (Render): Automatically detected. Serves from output_demo/ folder.
To update cloud data:
  copy output\meridian_nlp_chancay.json output_demo\meridian_nlp_chancay.json
  git add output_demo\ && git commit -m "Update demo data" && git push
"""

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from pathlib import Path
import json, os, subprocess, sys, threading, time, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("meridian.server")

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Environment detection ─────────────────────────────────────────────────────
# Render sets RENDER=true automatically.
# We use output_demo/ on cloud so real output/ files aren't needed in git.

ON_RENDER  = os.environ.get("RENDER") == "true"
OUTPUT_DIR = Path("output_demo") if ON_RENDER else Path("output")
BASE_DIR   = Path(".")

if ON_RENDER:
    log.info("Running on Render — serving from output_demo/ (demo mode)")
else:
    log.info("Running locally — serving from output/")

KNOWN_PROFILES  = ["chancay", "taiwan", "plan_pacific", "bri_africa"]
DEFAULT_PROFILE = "chancay"

# ── Helpers ───────────────────────────────────────────────────────────────────

def profile_files(profile: str) -> dict:
    p = profile.lower().strip()
    return {
        "nlp":      OUTPUT_DIR / f"meridian_nlp_{p}.json",
        "feed":     OUTPUT_DIR / f"meridian_feed_{p}.json",
        "timeline": OUTPUT_DIR / f"meridian_timeline_{p}.json",
    }

def get_profile(req) -> str:
    p = req.args.get("profile", DEFAULT_PROFILE).lower().strip()
    return p if p in KNOWN_PROFILES else DEFAULT_PROFILE

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def migrate_legacy_files():
    """Copy old single-profile files to profile-aware names on first run."""
    import shutil
    legacy = {
        "meridian_nlp.json":      "meridian_nlp_chancay.json",
        "meridian_feed.json":     "meridian_feed_chancay.json",
        "meridian_timeline.json": "meridian_timeline_chancay.json",
    }
    for old, new in legacy.items():
        o, n = OUTPUT_DIR / old, OUTPUT_DIR / new
        if o.exists() and not n.exists():
            shutil.copy2(o, n)
            log.info(f"Migrated {old} → {new}")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return send_file("io-dashboard.html")

@app.route("/website")
def website():
    return send_file("index.html")

@app.route("/compare")
def compare():
    p = BASE_DIR / "compare.html"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return f.read()
    return "compare.html not found", 404

@app.route("/api/nlp")
def api_nlp():
    profile = get_profile(request)
    data    = load_json(profile_files(profile)["nlp"])
    if data is None:
        msg = (
            f"No demo data for '{profile}'. "
            f"Copy output/meridian_nlp_{profile}.json to output_demo/ and push to GitHub."
            if ON_RENDER else
            f"No NLP data for '{profile}'. Run: py -3.11 run_meridian.py --profile {profile}"
        )
        return jsonify({"error": msg}), 404
    data["_profile"] = profile
    data["_demo"]    = ON_RENDER
    return jsonify(data)

@app.route("/api/social")
def api_social():
    profile = get_profile(request)
    path    = OUTPUT_DIR / f"meridian_social_{profile}.json"
    data    = load_json(path)
    if data is None:
        return jsonify({"error": f"No social data for '{profile}'.",
                        "posts": [], "summary": {"total_posts": 0}}), 200
    return jsonify(data)

@app.route("/api/feed")
def api_feed():
    profile = get_profile(request)
    data    = load_json(profile_files(profile)["feed"])
    if data is None:
        return jsonify({"error": f"No feed for '{profile}'."}), 404
    return jsonify(data)

@app.route("/api/timeline")
def api_timeline():
    profile = get_profile(request)
    data    = load_json(profile_files(profile)["timeline"])
    if data is None:
        return jsonify({"data": [], "note": f"No timeline for '{profile}'."})
    return jsonify(data)

@app.route("/api/profiles")
def api_profiles():
    profiles = {}
    for name in KNOWN_PROFILES:
        nlp  = load_json(profile_files(name)["nlp"])
        feed = load_json(profile_files(name)["feed"])
        if nlp or feed:
            s      = (nlp or {}).get("sentiment", {}).get("overall_pct", {})
            topics = (nlp or {}).get("topics", {}).get("topics", [])
            profiles[name] = {
                "profile":        name,
                "has_data":       True,
                "articles":       (nlp or {}).get("article_count", 0),
                "languages":      len((nlp or {}).get("sentiment", {}).get("by_language", {})),
                "countries":      len((feed or {}).get("summary", {}).get("top_countries", [])),
                "neg_sentiment":  round(s.get("negative", 0), 1),
                "pos_sentiment":  round(s.get("positive", 0), 1),
                "neu_sentiment":  round(s.get("neutral",  0), 1),
                "top_topic":      topics[0].get("label", "") if topics else "",
                "top_topic_pct":  topics[0].get("prevalence_pct", 0) if topics else 0,
                "last_collected": (nlp or {}).get("generated", ""),
            }
        else:
            profiles[name] = {"profile": name, "has_data": False}
    return jsonify({"profiles": profiles, "default": DEFAULT_PROFILE})

@app.route("/api/status")
def api_status():
    status = {"server": "online", "mode": "demo" if ON_RENDER else "live", "profiles": {}}
    for name in KNOWN_PROFILES:
        feed = load_json(profile_files(name)["feed"])
        status["profiles"][name] = {
            "has_data":       profile_files(name)["nlp"].exists(),
            "articles":       (feed or {}).get("summary", {}).get("total_articles", 0),
            "last_collected": (feed or {}).get("meta", {}).get("collection_timestamp"),
        }
    return jsonify(status)

@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    if ON_RENDER:
        return jsonify({"status": "unavailable",
                        "message": "Run server.py to enable live refresh."}), 503
    profile = get_profile(request)
    def run():
        log.info(f"Background refresh: {profile}")
        env = {**os.environ, "PYTHONUTF8": "1", "MERIDIAN_PROFILE": profile}
        subprocess.run([sys.executable, "run_meridian.py"], env=env)
        log.info(f"Refresh complete: {profile}")
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started", "profile": profile})

# ── Auto-refresh (local only) ─────────────────────────────────────────────────

def schedule_auto_refresh(interval_minutes=15):
    def loop():
        while True:
            time.sleep(interval_minutes * 60)
            log.info(f"Auto-refresh: {DEFAULT_PROFILE}")
            try:
                env = {**os.environ, "PYTHONUTF8": "1", "MERIDIAN_PROFILE": DEFAULT_PROFILE}
                subprocess.run([sys.executable, "run_meridian.py"], env=env, capture_output=True)
            except Exception as e:
                log.error(f"Auto-refresh failed: {e}")
    threading.Thread(target=loop, daemon=True).start()
    log.info(f"Auto-refresh every {interval_minutes} min ({DEFAULT_PROFILE})")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import socket
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not ON_RENDER:
        migrate_legacy_files()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "localhost"

    mode = "DEMO (Render cloud)" if ON_RENDER else "LIVE (local)"
    print("""
╔═══════════════════════════════════════════════════════════════╗
║  MERIDIAN SERVER v2 // Profile-Aware                          ║
╚═══════════════════════════════════════════════════════════════╝""")
    print(f"  Mode:       {mode}")
    print(f"  Data dir:   {OUTPUT_DIR}/")
    print(f"  Dashboard:  http://localhost:5000")
    print(f"  Website:    http://localhost:5000/website")
    print(f"  Compare:    http://localhost:5000/compare")
    print(f"  Network:    http://{local_ip}:5000")
    print(f"  Profiles:   http://localhost:5000/api/profiles")
    print(f"\n  Profile API: /api/nlp?profile=chancay")
    print(f"               /api/nlp?profile=taiwan")
    print(f"               /api/nlp?profile=plan_pacific")
    print(f"               /api/nlp?profile=bri_africa")
    print(f"\n  Press Ctrl+C to stop.\n")

    if not ON_RENDER:
        schedule_auto_refresh(15)

    port = int(os.environ.get("PORT", 5000))  # Render injects PORT automatically
    app.run(host="0.0.0.0", port=port, debug=False)
