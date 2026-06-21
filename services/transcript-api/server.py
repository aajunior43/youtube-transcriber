import os, tempfile, json, time, threading, uuid, subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app)

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")
MAX_DURATION = int(os.environ.get("MAX_DURATION_MINUTES", "20")) * 60
model = None
model_lock = threading.Lock()

# Queue system
queue = []
completed = []
MAX_COMPLETED = 20
queue_lock = threading.Lock()
worker_active = False

def get_model():
    global model
    with model_lock:
        if model is None:
            model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        return model

def cleanup(files):
    for f in files:
        try: os.remove(f)
        except: pass

def worker():
    global worker_active
    while True:
        job = None
        with queue_lock:
            if queue:
                job = queue[0]
        if job is None:
            worker_active = False
            break
        job["status"] = "downloading"
        job["progress"] = 5
        tmp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(tmp_dir, f"{job['videoId']}.webm")
        try:
            job["progress"] = 10
            result = subprocess.run([
                "yt-dlp", "-f", "bestaudio/best",
                "--extract-audio", "--audio-format", "mp3",
                "--audio-quality", "0",
                "-o", audio_path, "--no-playlist", "--quiet",
                f"https://www.youtube.com/watch?v={job['videoId']}"
            ], capture_output=True, text=True, timeout=180)

            if result.returncode != 0:
                job["status"] = "error"
                job["error"] = "Falha ao baixar áudio"
                cleanup([os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)] if os.path.isdir(tmp_dir) else [])
                try: os.rmdir(tmp_dir)
                except: pass
                job["completedAt"] = time.time()
                with queue_lock:
                    if queue:
                        completed.insert(0, queue.pop(0))
                continue

            actual = audio_path
            if not os.path.exists(actual):
                for f in os.listdir(tmp_dir):
                    if f.endswith(".mp3"):
                        actual = os.path.join(tmp_dir, f)
                        break

            if not os.path.exists(actual) or os.path.getsize(actual) < 1000:
                job["status"] = "error"
                job["error"] = "Áudio baixado é muito pequeno"
                cleanup([os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)] if os.path.isdir(tmp_dir) else [])
                try: os.rmdir(tmp_dir)
                except: pass
                job["completedAt"] = time.time()
                with queue_lock:
                    if queue:
                        completed.insert(0, queue.pop(0))
                continue

            job["status"] = "transcribing"
            job["progress"] = 30
            wh_model = get_model()
            segments_gen, info = wh_model.transcribe(actual, beam_size=5)

            segments = []
            total = 0
            for seg in segments_gen:
                segments.append({"start": round(seg.start, 1), "end": round(seg.end, 1), "text": seg.text.strip()})
                total += 1
                if total % 10 == 0:
                    job["progress"] = min(95, 30 + int((seg.end / max(info.duration, 1)) * 60))

            job["segments"] = segments
            job["duration"] = round(info.duration, 1) if info.duration else 0
            job["language"] = info.language if info.language else "unknown"
            job["status"] = "done"
            job["progress"] = 100
            job["completedAt"] = time.time()

        except subprocess.TimeoutExpired:
            job["status"] = "error"
            job["error"] = "Download excedeu tempo limite"
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)[:200]
        finally:
            cleanup([os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)] if os.path.isdir(tmp_dir) else [])
            try: os.rmdir(tmp_dir)
            except: pass
            job["completedAt"] = time.time()
            with queue_lock:
                if queue:
                    completed.insert(0, queue.pop(0))

def enqueue(video_id, title=""):
    job = {
        "id": uuid.uuid4().hex[:8],
        "videoId": video_id,
        "title": title,
        "status": "queued",
        "progress": 0,
        "segments": [],
        "duration": 0,
        "language": "",
        "error": "",
        "createdAt": time.time()
    }
    with queue_lock:
        queue.append(job)
    global worker_active
    if not worker_active:
        worker_active = True
        threading.Thread(target=worker, daemon=True).start()
    return job["id"]

@app.route("/api/transcript", methods=["GET", "POST"])
def transcribe():
    video_id = request.args.get("videoId") or (request.json or {}).get("videoId")
    title = request.args.get("title") or (request.json or {}).get("title", "")
    if not video_id or len(video_id) != 11:
        return jsonify({"error": "Invalid video ID"}), 400

    # Check duration first
    info = subprocess.run([
        "yt-dlp", "--print", "duration", "--no-playlist", "--quiet",
        f"https://www.youtube.com/watch?v={video_id}"
    ], capture_output=True, text=True, timeout=30)
    if info.returncode == 0 and info.stdout.strip():
        try:
            dur = int(info.stdout.strip())
            if dur > MAX_DURATION:
                return jsonify({"error": f"Video muito longo ({dur//60}min). Máx: {MAX_DURATION//60}min."}), 413
        except ValueError:
            pass

    job_id = enqueue(video_id, title)
    return jsonify({"jobId": job_id, "position": len(queue)}), 202

@app.route("/api/transcript/status")
def get_status():
    job_id = request.args.get("jobId")
    with queue_lock:
        all_jobs = queue + completed
        if not job_id:
            return jsonify({
                "running": any(j["status"] in ("downloading","transcribing") for j in queue),
                "queueSize": len(queue),
                "completedSize": len(completed),
                "jobs": [{
                    "id": j["id"], "videoId": j["videoId"], "title": j["title"],
                    "status": j["status"], "progress": j["progress"]
                } for j in all_jobs]
            })
        for j in all_jobs:
            if j["id"] == job_id:
                return jsonify({
                    "id": j["id"], "videoId": j["videoId"], "title": j["title"],
                    "status": j["status"], "progress": j["progress"],
                    "segments": j.get("segments", []),
                    "duration": j.get("duration", 0),
                    "language": j.get("language", ""),
                    "error": j.get("error", "")
                })
    return jsonify({"error": "Job not found"}), 404

@app.route("/api/transcript/queue", methods=["GET"])
def get_queue():
    with queue_lock:
        return jsonify({
            "queue": [{
                "id": j["id"], "videoId": j["videoId"], "title": j["title"],
                "status": j["status"], "progress": j["progress"]
            } for j in queue]
        })

@app.route("/health")
def health():
    with queue_lock:
        busy = any(j["status"] in ("downloading","transcribing") for j in queue)
    return jsonify({"status": "ok", "queueSize": len(queue), "busy": busy, "model": MODEL_SIZE})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8771, threaded=True)
