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
);

CREATE INDEX IF NOT EXISTS idx_transcripts_completed_at
    ON transcripts(completed_at DESC);

CREATE INDEX IF NOT EXISTS idx_transcripts_created_at
    ON transcripts(created_at DESC);
