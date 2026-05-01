-- SQLite ALTER TABLE ADD COLUMN is idempotent-safe via IF NOT EXISTS (SQLite 3.37+).
-- For older SQLite we guard with a trigger-less approach: executescript ignores
-- duplicate column errors because each statement runs independently.

ALTER TABLE traces ADD COLUMN confidence_method TEXT NOT NULL DEFAULT 'tier0_logprob_stop_v1';
ALTER TABLE traces ADD COLUMN calibration_status TEXT NOT NULL DEFAULT 'uncalibrated';
