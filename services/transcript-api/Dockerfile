FROM python:3.12-slim
WORKDIR /app

RUN apt-get update -qq && apt-get install -y -qq ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir flask flask-cors faster-whisper gunicorn psycopg2-binary reportlab

COPY services/transcript-api/server.py .

ENV WHISPER_MODEL=small
ENV WHISPER_BEAM_SIZE=5
ENV MAX_DURATION_MINUTES=20

EXPOSE 8771

CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "600", "--bind", "0.0.0.0:8771", "server:app"]
