import os, tempfile, time, threading, uuid, subprocess
import psycopg2
from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
from psycopg2.extras import Json, RealDictCursor
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")
MAX_DURATION = int(os.environ.get("MAX_DURATION_MINUTES", "20")) * 60
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
DB_HOST = os.environ.get("DB_HOST", "hub-postgres")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "hub_master")
DB_USER = os.environ.get("DB_USER", "hubmaster")
DB_PASS = os.environ.get("DB_PASS", "hubmaster_secret_2026")
ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".flac",
    ".webm", ".mp4", ".mov", ".mkv", ".avi", ".mpeg", ".mpg"
}
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
model = None
model_lock = threading.Lock()

# Queue system
queue = []
completed = []
MAX_COMPLETED = 20
queue_lock = threading.Lock()
worker_active = False

def get_db():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
    )

def init_db():
    with get_db() as conn, conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id TEXT PRIMARY KEY,
                media_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'unknown',
                duration DOUBLE PRECISION NOT NULL DEFAULT 0,
                full_text TEXT NOT NULL DEFAULT '',
                segments JSONB NOT NULL DEFAULT '[]'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_completed_at ON transcripts(completed_at DESC)")

def save_transcript(job):
    full_text = " ".join(segment["text"] for segment in job["segments"]).strip()
    with get_db() as conn, conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO transcripts (
                id, media_id, filename, language, duration, full_text, segments,
                created_at, completed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s))
            ON CONFLICT (id) DO UPDATE SET
                language = EXCLUDED.language,
                duration = EXCLUDED.duration,
                full_text = EXCLUDED.full_text,
                segments = EXCLUDED.segments,
                completed_at = EXCLUDED.completed_at
        """, (
            job["id"], job["videoId"], job["filename"], job["language"],
            job["duration"], full_text, Json(job["segments"]),
            job["createdAt"], job["completedAt"]
        ))

def db_row_to_job(row):
    return {
        "id": row["id"], "videoId": row["media_id"], "title": row["filename"],
        "filename": row["filename"], "status": "done", "progress": 100,
        "segments": row["segments"], "duration": row["duration"],
        "language": row["language"], "error": "", "source": "upload",
        "createdAt": row["created_at"].timestamp(),
        "completedAt": row["completed_at"].timestamp()
    }

def load_transcript(job_id):
    with get_db() as conn, conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("SELECT * FROM transcripts WHERE id = %s", (job_id,))
        row = cursor.fetchone()
    return db_row_to_job(row) if row else None

def list_transcripts(limit=50):
    with get_db() as conn, conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("SELECT * FROM transcripts ORDER BY completed_at DESC LIMIT %s", (limit,))
        rows = cursor.fetchall()
    return [db_row_to_job(row) for row in rows]

init_db()

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

def media_duration(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ], capture_output=True, text=True, timeout=30)
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("Arquivo de áudio ou vídeo inválido.")
    return float(result.stdout.strip())

def worker():
    global worker_active
    while True:
        with queue_lock:
            if queue:
                job = queue[0]
            else:
                worker_active = False
                return
        source_path = job["sourcePath"]
        try:
            job["status"] = "transcribing"
            job["progress"] = 15
            wh_model = get_model()
            segments_gen, info = wh_model.transcribe(source_path, beam_size=5)

            segments = []
            total = 0
            for seg in segments_gen:
                segments.append({"start": round(seg.start, 1), "end": round(seg.end, 1), "text": seg.text.strip()})
                total += 1
                if total % 10 == 0:
                    job["progress"] = min(95, 15 + int((seg.end / max(info.duration, 1)) * 80))

            job["segments"] = segments
            job["duration"] = round(info.duration, 1) if info.duration else 0
            job["language"] = info.language if info.language else "unknown"
            job["status"] = "saving"
            job["progress"] = 98
            job["completedAt"] = time.time()
            save_transcript(job)
            job["status"] = "done"
            job["progress"] = 100

        except subprocess.TimeoutExpired:
            job["status"] = "error"
            job["error"] = "Processamento excedeu o tempo limite"
        except Exception as e:
            print(f"transcript job {job['id']} failed: {e}", flush=True)
            job["status"] = "error"
            job["error"] = str(e)[:200]
        finally:
            cleanup([source_path])
            job["completedAt"] = time.time()
            with queue_lock:
                if queue:
                    completed.insert(0, queue.pop(0))
                    del completed[MAX_COMPLETED:]

def enqueue(media_id, title, source_path, filename):
    global worker_active
    job = {
        "id": uuid.uuid4().hex[:8],
        "videoId": media_id,
        "title": title,
        "status": "queued",
        "progress": 0,
        "segments": [],
        "duration": 0,
        "language": "",
        "error": "",
        "source": "upload",
        "sourcePath": source_path,
        "filename": filename,
        "createdAt": time.time()
    }
    with queue_lock:
        queue.append(job)
        start_worker = not worker_active
        if start_worker:
            worker_active = True
    if start_worker:
        threading.Thread(target=worker, daemon=True).start()
    return job["id"]

@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": f"Arquivo muito grande. Máximo: {MAX_UPLOAD_MB} MB."}), 413

@app.route("/api/transcript/upload", methods=["POST"])
def upload_media():
    media = request.files.get("file")
    if not media or not media.filename:
        return jsonify({"error": "Selecione um arquivo de áudio ou vídeo."}), 400

    filename = secure_filename(media.filename)
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Formato não suportado. Envie áudio ou vídeo comum."}), 415

    fd, source_path = tempfile.mkstemp(prefix="transcript-", suffix=extension)
    os.close(fd)
    try:
        media.save(source_path)
        size = os.path.getsize(source_path)
        if size < 1000:
            raise ValueError("O arquivo está vazio ou é muito pequeno.")
        duration = media_duration(source_path)
        if duration > MAX_DURATION:
            cleanup([source_path])
            return jsonify({
                "error": f"Arquivo muito longo ({int(duration)//60}min). Máx: {MAX_DURATION//60}min."
            }), 413
    except (ValueError, subprocess.TimeoutExpired) as error:
        cleanup([source_path])
        return jsonify({"error": str(error)}), 400
    except Exception:
        cleanup([source_path])
        raise

    media_id = "upload-" + uuid.uuid4().hex[:8]
    job_id = enqueue(media_id, filename, source_path, filename)
    return jsonify({"jobId": job_id, "position": len(queue), "filename": filename}), 202

@app.route("/api/transcript/status")
def get_status():
    job_id = request.args.get("jobId")
    with queue_lock:
        all_jobs = queue + completed
        current_queue = list(queue)
        current_completed = list(completed)
        for j in all_jobs:
            if j["id"] == job_id:
                return jsonify({
                    "id": j["id"], "videoId": j["videoId"], "title": j["title"],
                    "status": j["status"], "progress": j["progress"],
                    "segments": j.get("segments", []),
                    "duration": j.get("duration", 0),
                    "language": j.get("language", ""),
                    "error": j.get("error", ""),
                    "source": j.get("source", "youtube"),
                    "filename": j.get("filename", "")
                })
    if job_id:
        stored = load_transcript(job_id)
        return jsonify(stored) if stored else (jsonify({"error": "Job not found"}), 404)

    stored_jobs = list_transcripts()
    stored_ids = {job["id"] for job in stored_jobs}
    memory_jobs = current_queue + [job for job in current_completed if job["id"] not in stored_ids]
    jobs = memory_jobs + stored_jobs
    return jsonify({
        "running": any(j["status"] in ("transcribing", "saving") for j in current_queue),
        "queueSize": len(current_queue),
        "completedSize": len(stored_jobs),
        "jobs": [{
            "id": j["id"], "videoId": j["videoId"], "title": j["title"],
            "status": j["status"], "progress": j["progress"],
            "source": j.get("source", "upload"), "error": j.get("error", "")
        } for j in jobs]
    })

@app.route("/api/transcript/history")
def get_history():
    return jsonify({"transcripts": list_transcripts()})

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
        busy = any(j["status"] in ("transcribing", "saving") for j in queue)
    return jsonify({
        "status": "ok", "queueSize": len(queue), "busy": busy,
        "model": MODEL_SIZE, "maxUploadMb": MAX_UPLOAD_MB
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8771, threaded=True)
