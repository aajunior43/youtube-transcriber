import os, tempfile, json, time, threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app)

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")
MAX_DURATION = int(os.environ.get("MAX_DURATION_MINUTES", "20")) * 60
model = None
transcribing = False
model_lock = threading.Lock()
transcribe_lock = threading.Lock()

def get_model():
    global model
    with model_lock:
        if model is None:
            model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        return model

def cleanup(old_files):
    for f in old_files:
        try: os.remove(f)
        except: pass

@app.route("/api/transcript", methods=["GET", "POST"])
def transcribe():
    global transcribing
    video_id = request.args.get("videoId") or (request.json or {}).get("videoId")
    if not video_id or len(video_id) != 11:
        return jsonify({"error": "Invalid video ID"}), 400

    # Check video duration first (lightweight)
    import subprocess as sp
    info = sp.run([
        "yt-dlp", "--print", "duration", "--no-playlist",
        "--quiet",
        f"https://www.youtube.com/watch?v={video_id}"
    ], capture_output=True, text=True, timeout=30)
    if info.returncode == 0 and info.stdout.strip():
        try:
            dur = int(info.stdout.strip())
            if dur > MAX_DURATION:
                return jsonify({"error": f"Video too long ({dur//60}min). Max: {MAX_DURATION//60}min."}), 413
        except ValueError:
            pass

    if not transcribe_lock.acquire(blocking=False):
        return jsonify({"error": "Server busy transcribing another video. Try again in a few minutes."}), 429

    try:
        transcribing = True
        import subprocess
        tmp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(tmp_dir, f"{video_id}.mp3")

        try:
            result = subprocess.run([
                "yt-dlp",
                "-f", "bestaudio/best",
                "--extract-audio",
                "--audio-format", "mp3",
                "--audio-quality", "0",
                "-o", audio_path,
                "--no-playlist",
                "--quiet",
                f"https://www.youtube.com/watch?v={video_id}"
            ], capture_output=True, text=True, timeout=180)

            if result.returncode != 0:
                return jsonify({"error": "Failed to download audio", "detail": result.stderr[:300]}), 500

            actual_path = audio_path
            if not os.path.exists(actual_path):
                for f in os.listdir(tmp_dir):
                    if f.endswith(".mp3"):
                        actual_path = os.path.join(tmp_dir, f)
                        break

            if not os.path.exists(actual_path) or os.path.getsize(actual_path) < 1000:
                return jsonify({"error": "Downloaded audio too small or missing"}), 500

            wh_model = get_model()
            segments, info = wh_model.transcribe(actual_path, beam_size=5)

            result_segments = []
            for seg in segments:
                result_segments.append({
                    "start": round(seg.start, 1),
                    "end": round(seg.end, 1),
                    "text": seg.text.strip()
                })

            return jsonify({
                "videoId": video_id,
                "segments": result_segments,
                "duration": round(info.duration, 1) if info.duration else 0,
                "language": info.language if info.language else "unknown"
            })

        except subprocess.TimeoutExpired:
            return jsonify({"error": "Download timed out"}), 504
        except Exception as e:
            return jsonify({"error": str(e)[:300]}), 500
        finally:
            cleanup([os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)] if os.path.isdir(tmp_dir) else [])
            try: os.rmdir(tmp_dir)
            except: pass
    finally:
        transcribing = False
        transcribe_lock.release()

@app.route("/api/transcript/status")
def status():
    return jsonify({"model": MODEL_SIZE, "transcribing": transcribing, "ready": model is not None})

@app.route("/health")
def health():
    return "ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8771, threaded=True)
