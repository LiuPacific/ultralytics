import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, Json
from tqdm import tqdm


# =========================
# Config
# =========================

CSV_PATH = "tracking_output.csv"

FPS = 30

# Use timezone-aware timestamp for TIMESTAMPTZ.
# I recommend UTC for video-derived timestamps.
START_TIME = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")

# Metadata for this CSV/video
VIDEO_ID = "video_001"
PEN_ID = "pen_01"

# Optional metadata
DETECTOR_NAME = "yolo_obb"
TRACKER_NAME = "deepsort_obb"
MODEL_VERSION = "your_model_version"

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 15432,          # use 5432 if your Docker maps 5432:5432
    "dbname": "chicken_tracking",
    "user": "hara",
    "password": "123456",
}

BATCH_SIZE = 10000


# =========================
# Read and prepare dataframe
# =========================

def read_and_prepare_tracking_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [
        "frame_id",
        "track_id",
        "center_x",
        "center_y",
        "width",
        "height",
        "confidence",
        "detected",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Basic type conversion
    df["frame_id"] = df["frame_id"].astype(int)
    df["track_id"] = df["track_id"].astype(int)

    df["center_x"] = df["center_x"].astype(float)
    df["center_y"] = df["center_y"].astype(float)
    df["width"] = df["width"].astype(float)
    df["height"] = df["height"].astype(float)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # Convert detected column safely
    df["detected"] = (
        df["detected"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    # Map frame_id to frame_index
    # If your DB column is named frame_index, use frame_index.
    df["frame_index"] = df["frame_id"]

    # frame_id = 1 -> START_TIME
    # frame_id = 2 -> START_TIME + 1/30 second
    df["video_time_s"] = (df["frame_index"] - 1) / FPS
    df["ts"] = START_TIME + pd.to_timedelta(df["video_time_s"], unit="s")

    # Map CSV center coordinates to DB x/y
    df["x_px"] = df["center_x"]
    df["y_px"] = df["center_y"]

    # Add metadata columns
    df["video_id"] = VIDEO_ID
    df["pen_id"] = PEN_ID

    # If you do not know real chicken_id, keep it NULL.
    # track_id is tracker ID, not necessarily biological chicken ID.
    df["chicken_id"] = None

    df = df.sort_values(["frame_index", "track_id"]).reset_index(drop=True)

    return df


# =========================
# Convert dataframe rows
# =========================

def dataframe_to_insert_rows(df: pd.DataFrame):
    rows = []

    for r in df.itertuples(index=False):
        bbox = {
            "type": "HBB",
            "cx": float(r.center_x),
            "cy": float(r.center_y),
            "width": float(r.width),
            "height": float(r.height),
        }

        extra = {
            "source": "csv_import",
            "fps": FPS,
            "video_time_s": float(r.video_time_s),
            "detector": DETECTOR_NAME,
            "tracker": TRACKER_NAME,
            "model_version": MODEL_VERSION,
        }

        confidence = None if pd.isna(r.confidence) else float(r.confidence)

        rows.append(
            (
                r.ts.to_pydatetime(),      # ts
                str(r.video_id),           # video_id
                str(r.pen_id),             # pen_id
                int(r.frame_index),        # frame_index
                int(r.track_id),           # track_id
                None,                      # chicken_id
                float(r.x_px),             # x_px
                float(r.y_px),             # y_px
                confidence,                # confidence
                Json(bbox),                # bbox JSONB
                Json(extra),               # extra JSONB
                bool(r.detected),          # detected
            )
        )

    return rows


# =========================
# Insert into PostgreSQL
# =========================

def insert_tracking_points(df: pd.DataFrame):
    insert_sql = """
                 INSERT INTO tracking_points (
                     ts,
                     video_id,
                     pen_id,
                     frame_index,
                     track_id,
                     chicken_id,
                     x_px,
                     y_px,
                     confidence,
                     bbox,
                     extra,
                     detected
                 )
                 VALUES %s
                     ON CONFLICT DO NOTHING; \
                 """

    rows = dataframe_to_insert_rows(df)

    if not rows:
        print("No rows to insert.")
        return

    conn = psycopg2.connect(**DB_CONFIG)

    try:
        with conn:
            with conn.cursor() as cur:
                for start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Inserting"):
                    batch = rows[start:start + BATCH_SIZE]
                    execute_values(cur, insert_sql, batch, page_size=BATCH_SIZE)

        print(f"Finished inserting {len(rows)} rows into tracking_points.")

    finally:
        conn.close()


# =========================
# Optional verification
# =========================

def verify_insert():
    conn = psycopg2.connect(**DB_CONFIG)

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(DISTINCT video_id) AS num_videos,
                    COUNT(DISTINCT track_id) AS num_tracks,
                    MIN(frame_index) AS min_frame,
                    MAX(frame_index) AS max_frame,
                    MIN(ts) AS min_ts,
                    MAX(ts) AS max_ts
                FROM tracking_points
                WHERE video_id = %s;
                """,
                (VIDEO_ID,),
            )

            result = cur.fetchone()

            print("\nVerification:")
            print(f"Total rows: {result[0]}")
            print(f"Number of videos: {result[1]}")
            print(f"Number of tracks: {result[2]}")
            print(f"Frame range: {result[3]} to {result[4]}")
            print(f"Time range: {result[5]} to {result[6]}")

    finally:
        conn.close()


# =========================
# Main
# =========================

if __name__ == "__main__":
    df = read_and_prepare_tracking_csv(CSV_PATH)

    print("Preview dataframe:")
    print(df.head(10).to_string(index=False))

    print("\nData summary:")
    print(f"Rows: {len(df)}")
    print(f"Frames: {df['frame_index'].nunique()}")
    print(f"Tracks: {df['track_id'].nunique()}")
    print(f"Detected ratio: {df['detected'].mean():.4f}")
    print(f"Time range: {df['ts'].min()} to {df['ts'].max()}")

    insert_tracking_points(df)
    verify_insert()