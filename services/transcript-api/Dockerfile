FROM python:3.12-slim
WORKDIR /app

RUN apt-get update -qq && apt-get install -y -qq ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir yt-dlp flask flask-cors faster-whisper

COPY services/transcript-api/server.py .

ENV WHISPER_MODEL=base

EXPOSE 8771

CMD ["python3", "server.py"]
