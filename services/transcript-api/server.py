import os, tempfile, time, threading, uuid, subprocess
from io import BytesIO
from xml.sax.saxutils import escape
import psycopg2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from faster_whisper import WhisperModel
from psycopg2.extras import Json, RealDictCursor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_created_at ON transcripts(created_at DESC)")

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
        cursor.execute("SELECT * FROM transcripts ORDER BY created_at DESC LIMIT %s", (limit,))
        rows = cursor.fetchall()
    return [db_row_to_job(row) for row in rows]

def timestamp_label(seconds):
    total = int(seconds or 0)
    return f"{total // 60:02d}:{total % 60:02d}"

def markdown_document(job):
    lines = [
        f"# {job['title']}", "",
        f"- Idioma: {job['language']}",
        f"- Duração: {timestamp_label(job['duration'])}", "",
        "## Transcrição", ""
    ]
    lines.extend(
        f"**[{timestamp_label(segment['start'])}]** {segment['text']}"
        for segment in job["segments"]
    )
    return "\n\n".join(lines) + "\n"

def pdf_document(job):
    output = BytesIO()
    document = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=job["title"], author="Hub Master"
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TranscriptTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontName="Helvetica-Bold", fontSize=16, leading=20, spaceAfter=12
    )
    meta_style = ParagraphStyle(
        "TranscriptMeta", parent=styles["Normal"], fontSize=9,
        leading=12, textColor="#555555", spaceAfter=12
    )
    line_style = ParagraphStyle(
        "TranscriptLine", parent=styles["BodyText"], fontSize=10,
        leading=14, spaceAfter=7
    )
    story = [
        Paragraph(escape(job["title"]), title_style),
        Paragraph(
            f"Idioma: {escape(job['language'])} &nbsp;&nbsp; "
            f"Duração: {timestamp_label(job['duration'])}", meta_style
        ),
        Spacer(1, 3 * mm)
    ]
    for segment in job["segments"]:
        story.append(Paragraph(
            f"<b>[{timestamp_label(segment['start'])}]</b> {escape(segment['text'])}",
            line_style
        ))
    document.build(story)
    output.seek(0)
    return output

init_db()

def get_model():
    global model
    with model_lock:
        if model is None:
            model = WhisperModel(
                MODEL_SIZE, device="cpu", compute_type="int8",
                cpu_threads=2, num_workers=1
            )
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

def normalize_media(path):
    fd, wav_path = tempfile.mkstemp(prefix="transcript-audio-", suffix=".wav")
    os.close(fd)
    result = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path
    ], capture_output=True, text=True, timeout=300)
    if result.returncode != 0 or os.path.getsize(wav_path) < 1000:
        cleanup([wav_path])
        detail = (result.stderr or "").strip().splitlines()
        raise ValueError(detail[-1][:180] if detail else "Não foi possível converter a mídia.")
    return wav_path

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
        normalized_path = None
        try:
            job["status"] = "preparing"
            job["progress"] = 10
            normalized_path = normalize_media(source_path)
            job["status"] = "transcribing"
            job["progress"] = 15
            wh_model = get_model()
            segments_gen, info = wh_model.transcribe(
                normalized_path,
                beam_size=BEAM_SIZE,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500}
            )

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
            cleanup([source_path, normalized_path])
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
    jobs.sort(key=lambda job: job.get("createdAt", 0), reverse=True)
    return jsonify({
        "running": any(j["status"] in ("preparing", "transcribing", "saving") for j in current_queue),
        "queueSize": len(current_queue),
        "completedSize": len(stored_jobs),
        "jobs": [{
            "id": j["id"], "videoId": j["videoId"], "title": j["title"],
            "status": j["status"], "progress": j["progress"],
            "source": j.get("source", "upload"), "error": j.get("error", ""),
            "createdAt": j.get("createdAt", 0)
        } for j in jobs]
    })

@app.route("/api/transcript/history")
def get_history():
    return jsonify({"transcripts": list_transcripts()})

@app.route("/api/transcript/<job_id>", methods=["DELETE"])
def delete_transcript(job_id):
    with queue_lock:
        if any(job["id"] == job_id for job in queue):
            return jsonify({"error": "Aguarde o processamento terminar antes de excluir."}), 409
        before = len(completed)
        completed[:] = [job for job in completed if job["id"] != job_id]
        deleted_from_memory = len(completed) != before

    with get_db() as conn, conn.cursor() as cursor:
        cursor.execute("DELETE FROM transcripts WHERE id = %s", (job_id,))
        deleted_from_db = cursor.rowcount > 0

    if not deleted_from_memory and not deleted_from_db:
        return jsonify({"error": "Transcrição não encontrada."}), 404
    return jsonify({"deleted": True, "id": job_id})

@app.route("/api/transcript/<job_id>/download")
def download_transcript(job_id):
    file_format = request.args.get("format", "md").lower()
    if file_format not in ("md", "pdf"):
        return jsonify({"error": "Formato de download inválido."}), 400

    job = load_transcript(job_id)
    if not job:
        return jsonify({"error": "Transcrição não encontrada."}), 404

    base_name = secure_filename(os.path.splitext(job["filename"])[0]) or "transcricao"
    if file_format == "md":
        content = BytesIO(markdown_document(job).encode("utf-8"))
        return send_file(
            content, mimetype="text/markdown", as_attachment=True,
            download_name=f"{base_name}.md"
        )
    return send_file(
        pdf_document(job), mimetype="application/pdf", as_attachment=True,
        download_name=f"{base_name}.pdf"
    )

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
        busy = any(j["status"] in ("preparing", "transcribing", "saving") for j in queue)
    return jsonify({
        "status": "ok", "queueSize": len(queue), "busy": busy,
        "model": MODEL_SIZE, "beamSize": BEAM_SIZE, "maxUploadMb": MAX_UPLOAD_MB
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8771, threaded=True)
