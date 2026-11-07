
import os
import shutil
import subprocess
import uuid
import base64

import cv2

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    jsonify,
    send_from_directory
)

from werkzeug.utils import secure_filename

from database import (
    add_run,
    get_runs,
    init_db,
    add_live_detection,
    get_live_detections
)
from detect import process_video

# Live webcam detection
from live_detection import detect_frame, reset_tracking


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024   # 25 MB upload limit

ALLOWED = {".mp4", ".avi", ".mov", ".mkv"}
DARKNESS = {"0.5": 0.5, "0.25": 0.25, "0.1": 0.1}
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)
init_db()


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def make_browser_playable(path):
    """OpenCV writes MPEG-4 video that browsers can't play; re-encode it to H.264."""
    cap = cv2.VideoCapture(path)
    code = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="ignore").lower()
    cap.release()
    if code in ("avc1", "h264", "x264"):
        return True
    exe = ffmpeg_exe()
    if not exe:
        return False
    tmp = path + ".tmp.mp4"
    try:
        subprocess.run(
            [exe, "-y", "-i", path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", tmp],
            check=True, capture_output=True,
        )
        os.replace(tmp, path)
        return True
    except Exception:
        return False


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("video")
        if not file or file.filename == "":
            flash("Please choose a video file.")
            return redirect("/")

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED:
            flash("Unsupported file type. Use mp4, avi, mov or mkv.")
            return redirect("/")

        uid = uuid.uuid4().hex[:10]
        in_path = os.path.join("uploads", f"{uid}{ext}")
        out_name = f"{uid}.mp4"
        out_path = os.path.join("results", out_name)
        file.save(in_path)

        simulate = "simulate" in request.form
        brightness = DARKNESS.get(request.form.get("brightness", "0.25"), 0.25)
        try:
            sensitivity = int(request.form.get("sensitivity", 1500))
        except ValueError:
            sensitivity = 1500

        try:
            stats = process_video(in_path, out_path, min_area=sensitivity,
                                  simulate_dark=simulate, brightness=brightness)
        except Exception as e:
            flash(f"Could not process this video: {e}")
            return redirect("/")
        finally:
            if os.path.exists(in_path):
                os.remove(in_path)          # don't keep user uploads

        playable = make_browser_playable(out_path)
        add_run(secure_filename(file.filename), out_name, simulate,
                stats["duration"], len(stats["events"]))
        return render_template("result.html", video=out_name, stats=stats, playable=playable)

    return render_template("index.html")


@app.route("/history")
def history():
    return render_template("history.html", runs=get_runs())
@app.route("/live-history")
def live_history():
    records = get_live_detections()
    return render_template("live_history.html", records=records)

@app.route("/results/<name>")
def results(name):
    return send_from_directory("results", name)



# ==========================================
# LIVE WEBCAM DETECTION ROUTES
# ==========================================

@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/api/live-reset", methods=["POST"])
def api_live_reset():
    reset_tracking()

    return jsonify({
        "success": True,
        "message": "Tracking reset successfully"
    })


@app.route("/api/live-detect", methods=["POST"])
def api_live_detect():
    data = request.get_json(silent=True) or {}

    image = data.get("image", "")

    if not image:
        return jsonify({
            "error": "No webcam image received"
        }), 400

    try:
        # Remove the data URL prefix if present
        if "," in image:
            image = image.split(",", 1)[1]

        # Decode the Base64 webcam frame
        image_data = base64.b64decode(image, validate=True)

        # Run human detection and movement tracking
        result = detect_frame(image_data)

        return jsonify(result)

    except Exception:
        app.logger.exception("Live detection failed")

        return jsonify({
            "error": "Could not process webcam frame"
        }), 500
@app.route("/api/live-log", methods=["POST"])
def api_live_log():
    data = request.get_json(silent=True) or {}

    try:
        person_count = int(data.get("count", 0))
        movement = bool(data.get("movement", False))

        if person_count < 0:
            return jsonify({"error": "Invalid person count"}), 400

        add_live_detection(person_count, movement)

        return jsonify({
            "success": True,
            "message": "Detection saved"
        })

    except (TypeError, ValueError):
        return jsonify({"error": "Invalid detection data"}), 400
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)