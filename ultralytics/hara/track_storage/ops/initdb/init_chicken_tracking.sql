CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS tracking_points (
    COLUMN hid SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,

    video_id TEXT NOT NULL,
    pen_id TEXT,
    frame_index INTEGER NOT NULL,

    track_id INTEGER NOT NULL, -- track ID number. It is assigned by the tracker, and is not guaranteed to be consistent across videos or runs.
    chicken_id INTEGER, -- ID on X-any-labelling

    x_px DOUBLE PRECISION NOT NULL,
    y_px DOUBLE PRECISION NOT NULL,

     -- Pixel-coordinate geometry. SRID 0 means no real-world CRS.
    geom GEOMETRY(Point, 0)
        GENERATED ALWAYS AS (
            ST_SetSRID(ST_MakePoint(x_px, y_px), 0)
    ) STORED,

    confidence REAL,

    -- Whether this point comes from an actual detector result.
    -- TRUE means detected by the model; FALSE can mean interpolated, predicted, or manually filled.
    detected BOOLEAN NOT NULL DEFAULT TRUE,

    -- Store OBB/HBB details, detector metadata, angle, width, height, etc.
    bbox JSONB,

    -- Extra flexible metadata: model version, ReID score, occlusion flag, etc.
    extra JSONB,
    );

-- TimescaleDB hypertables are PostgreSQL tables automatically partitioned by time into chunks
SELECT create_hypertable(
               'tracking_points',
               'ts', -- The primary key includes ts because TimescaleDB unique constraints on hypertables must include the time partitioning column.
               if_not_exists => TRUE,
               chunk_time_interval => INTERVAL '1 day'
       );

CREATE INDEX IF NOT EXISTS idx_tracking_video_track_time
    ON tracking_points (video_id, track_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_tracking_video_chicken_time
    ON tracking_points (video_id, chicken_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_tracking_geom
    ON tracking_points
    USING GIST (geom);