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
"""

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from pathlib import Path
import json, os, subprocess, sys, threading, time, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("meridian.server")

app    = Flask(__name__, static_folder=".")
CORS(app)

OUTPUT_DIR = Path("output")
BASE_DIR   = Path(".")

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
        return jsonify({"error": f"No NLP data for '{profile}'. Run: py -3.11 run_meridian.py --profile {profile}"}), 404
    data["_profile"] = profile
    return jsonify(data)

@app.route("/api/social")
def api_social():
    profile = get_profile(request)
    path    = OUTPUT_DIR / f"meridian_social_{profile}.json"
    data    = load_json(path)
    if data is None:
        return jsonify({"error": f"No social data for '{profile}'. Run: py -3.11 run_meridian.py --profile {profile}",
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
    status = {"server": "online", "profiles": {}}
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
    profile = get_profile(request)
    def run():
        log.info(f"Background refresh: {profile}")
        env = {**os.environ, "PYTHONUTF8": "1", "MERIDIAN_PROFILE": profile}
        subprocess.run([sys.executable, "run_meridian.py"], env=env)
        log.info(f"Refresh complete: {profile}")
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started", "profile": profile})

# ── Auto-refresh ──────────────────────────────────────────────────────────────

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
    migrate_legacy_files()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "localhost"

    print("""
╔═══════════════════════════════════════════════════════════════╗
║  MERIDIAN SERVER v2 // Profile-Aware                          ║
╚═══════════════════════════════════════════════════════════════╝""")
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

    schedule_auto_refresh(15)
    app.run(host="0.0.0.0", port=5000, debug=False)
