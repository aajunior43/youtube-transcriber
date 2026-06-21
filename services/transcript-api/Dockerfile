FROM python:3.12-slim
WORKDIR /app

RUN apt-get update -qq && apt-get install -y -qq ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir yt-dlp flask flask-cors faster-whisper

COPY services/transcript-api/server.py .

ENV WHISPER_MODEL=tiny
ENV MAX_DURATION_MINUTES=20

EXPOSE 8771

CMD ["python3", "server.py"]
