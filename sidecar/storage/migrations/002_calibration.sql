CREATE TABLE IF NOT EXISTS calibration_params (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT    NOT NULL,
    model_type  TEXT    NOT NULL CHECK (model_type IN ('platt', 'isotonic', 'temperature')),
    params      TEXT    NOT NULL,
    trained_at  REAL    NOT NULL,
    n_samples   INTEGER NOT NULL,
    brier_score REAL,
    ece         REAL
);

CREATE INDEX IF NOT EXISTS idx_cal_customer ON calibration_params(customer_id, trained_at);
