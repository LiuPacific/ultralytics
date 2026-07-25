import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# =========================
# Config
# =========================

CSV_PATH = "tracking_output.csv"

FPS = 30

# frame_id = 1 means this start time
START_TIME = pd.Timestamp("2026-06-01 00:00:00")

# Optional image size.
# If you know your video size, set it here, for example:
# IMAGE_WIDTH = 3000
# IMAGE_HEIGHT = 2000
IMAGE_WIDTH = None
IMAGE_HEIGHT = None


# =========================
# Read data
# =========================

def read_tracking_csv(csv_path: str) -> pd.DataFrame:
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
        raise ValueError(f"Missing columns: {missing}")

    df["frame_id"] = df["frame_id"].astype(int)
    df["track_id"] = df["track_id"].astype(int)

    df["center_x"] = df["center_x"].astype(float)
    df["center_y"] = df["center_y"].astype(float)
    df["width"] = df["width"].astype(float)
    df["height"] = df["height"].astype(float)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # Convert True/False strings to bool safely
    df["detected"] = (
        df["detected"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    # frame_id = 1 -> START_TIME
    # frame_id = 2 -> START_TIME + 1/30 second
    df["video_time_s"] = (df["frame_id"] - 1) / FPS
    df["ts"] = START_TIME + pd.to_timedelta(df["video_time_s"], unit="s")

    df = df.sort_values(["frame_id", "track_id"]).reset_index(drop=True)

    return df


# =========================
# Summary
# =========================

def summarize_tracks(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("track_id")
        .agg(
            start_frame=("frame_id", "min"),
            end_frame=("frame_id", "max"),
            num_points=("frame_id", "count"),
            detected_points=("detected", "sum"),
            mean_confidence=("confidence", "mean"),
            start_time=("ts", "min"),
            end_time=("ts", "max"),
        )
        .reset_index()
    )

    summary["duration_s"] = (summary["end_frame"] - summary["start_frame"]) / FPS
    summary["detected_ratio"] = summary["detected_points"] / summary["num_points"]

    summary = summary.sort_values("num_points", ascending=False)

    return summary


# =========================
# Plot trajectories
# =========================

def plot_trajectories(
        df: pd.DataFrame,
        max_tracks: int = 30,
        only_longest_tracks: bool = True,
        title: str = "Chicken tracking trajectories",
):
    """
    Plot center_x, center_y trajectories.

    Because image coordinates usually have y going downward,
    the y-axis is inverted.
    """

    if only_longest_tracks:
        selected_tracks = (
            df.groupby("track_id")
            .size()
            .sort_values(ascending=False)
            .head(max_tracks)
            .index
        )
    else:
        selected_tracks = df["track_id"].drop_duplicates().head(max_tracks)

    plot_df = df[df["track_id"].isin(selected_tracks)]

    plt.figure(figsize=(10, 8))

    for track_id, g in plot_df.groupby("track_id"):
        g = g.sort_values("frame_id")

        plt.plot(
            g["center_x"],
            g["center_y"],
            linewidth=1,
            alpha=0.7,
            label=f"ID {track_id}",
        )

        # Detected points
        detected_g = g[g["detected"]]
        plt.scatter(
            detected_g["center_x"],
            detected_g["center_y"],
            s=8,
            alpha=0.7,
        )

        # Non-detected/interpolated points
        missed_g = g[~g["detected"]]
        if len(missed_g) > 0:
            plt.scatter(
                missed_g["center_x"],
                missed_g["center_y"],
                s=20,
                marker="x",
                alpha=0.9,
            )

    plt.gca().invert_yaxis()
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")
    plt.title(title)

    if len(selected_tracks) <= 20:
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# =========================
# Plot one frame
# =========================

def plot_frame(
        df: pd.DataFrame,
        frame_id: int,
        image_width: int | None = None,
        image_height: int | None = None,
        show_track_id: bool = True,
):
    frame_df = df[df["frame_id"] == frame_id].copy()

    if frame_df.empty:
        print(f"No data found for frame_id={frame_id}")
        return

    if image_width is None:
        image_width = int((frame_df["center_x"] + frame_df["width"] / 2).max() * 1.05)

    if image_height is None:
        image_height = int((frame_df["center_y"] + frame_df["height"] / 2).max() * 1.05)

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    for _, row in frame_df.iterrows():
        x1 = row["center_x"] - row["width"] / 2
        y1 = row["center_y"] - row["height"] / 2

        rect = Rectangle(
            (x1, y1),
            row["width"],
            row["height"],
            fill=False,
            linewidth=1.5,
            alpha=0.8,
        )
        ax.add_patch(rect)

        marker = "o" if row["detected"] else "x"

        ax.scatter(
            row["center_x"],
            row["center_y"],
            s=30,
            marker=marker,
        )

        if show_track_id:
            ax.text(
                row["center_x"],
                row["center_y"],
                str(int(row["track_id"])),
                fontsize=8,
            )

    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_title(f"Tracking preview, frame {frame_id}")

    plt.tight_layout()
    plt.show()


# =========================
# Plot one track over time
# =========================

def plot_single_track(df: pd.DataFrame, track_id: int):
    g = df[df["track_id"] == track_id].sort_values("frame_id")

    if g.empty:
        print(f"No data found for track_id={track_id}")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(g["video_time_s"], g["center_x"], label="center_x")
    plt.plot(g["video_time_s"], g["center_y"], label="center_y")

    missed = g[~g["detected"]]
    if len(missed) > 0:
        plt.scatter(
            missed["video_time_s"],
            missed["center_x"],
            marker="x",
            label="non-detected x",
        )
        plt.scatter(
            missed["video_time_s"],
            missed["center_y"],
            marker="x",
            label="non-detected y",
        )

    plt.xlabel("video time, seconds")
    plt.ylabel("pixel position")
    plt.title(f"Track {track_id} position over time")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# Main
# =========================

if __name__ == "__main__":
    df = read_tracking_csv(CSV_PATH)

    print("\nFirst rows:")
    print(df.head(10).to_string(index=False))

    print("\nBasic info:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of frames: {df['frame_id'].nunique()}")
    print(f"Number of tracks: {df['track_id'].nunique()}")
    print(f"Frame range: {df['frame_id'].min()} to {df['frame_id'].max()}")
    print(f"Time range: {df['ts'].min()} to {df['ts'].max()}")
    print(f"Detected ratio: {df['detected'].mean():.3f}")

    track_summary = summarize_tracks(df)

    print("\nTrack summary:")
    print(track_summary.head(20).to_string(index=False))

    # Save summary
    track_summary.to_csv("track_summary.csv", index=False)

    # Preview trajectories of longest 30 tracks
    plot_trajectories(df, max_tracks=30)

    # Preview one frame
    plot_frame(
        df,
        frame_id=1,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    )

    # Preview one track
    plot_single_track(df, track_id=1)