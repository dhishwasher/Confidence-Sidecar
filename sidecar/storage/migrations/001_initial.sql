CREATE TABLE IF NOT EXISTS traces (
    id                  TEXT    PRIMARY KEY,
    customer_id         TEXT    NOT NULL,
    created_at          REAL    NOT NULL,
    model               TEXT    NOT NULL,
    provider            TEXT    NOT NULL,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    tier                INTEGER NOT NULL DEFAULT 0,
    confidence          REAL,
    confidence_raw      REAL,
    confidence_tier     INTEGER,
    stop_reason         TEXT,
    request_hash        TEXT    NOT NULL,
    streaming           INTEGER NOT NULL DEFAULT 0,
    latency_ms          INTEGER,
    upstream_latency_ms INTEGER
);

CREATE TABLE IF NOT EXISTS signals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id         TEXT    NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
    signal_name      TEXT    NOT NULL,
    signal_value     REAL    NOT NULL,
    signal_metadata  TEXT,
    computed_at      REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id        TEXT    NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
    sample_index    INTEGER NOT NULL,
    response_text   TEXT    NOT NULL,
    cluster_id      INTEGER,
    logprob_entropy REAL,
    created_at      REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id   TEXT    NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
    label      TEXT    NOT NULL CHECK (label IN ('correct', 'incorrect', 'partial')),
    score      REAL    CHECK (score IS NULL OR (score >= 0.0 AND score <= 1.0)),
    metadata   TEXT,
    created_at REAL    NOT NULL,
    source     TEXT    NOT NULL DEFAULT 'human'
);

CREATE INDEX IF NOT EXISTS idx_traces_customer_time ON traces(customer_id, created_at);
CREATE INDEX IF NOT EXISTS idx_traces_confidence    ON traces(customer_id, confidence);
CREATE INDEX IF NOT EXISTS idx_signals_trace        ON signals(trace_id);
CREATE INDEX IF NOT EXISTS idx_feedback_trace       ON feedback(trace_id);
CREATE INDEX IF NOT EXISTS idx_samples_trace        ON samples(trace_id);
