
# 1. Extract the same frame number (separate snapshots)

```shell
# First video
ffmpeg -ss 00:01:23 -i video1.mkv -frames:v 1 snapshot1.png

# Second video
ffmpeg -ss 00:01:23 -i video2.mkv -frames:v 1 snapshot2.png
```

- `-ss` → seek to time.
- `-frames:v 1` → capture just one frame.
- Output: two separate PNG images (snapshot1.png and snapshot2.png) you can compare.

# 2. Side-by-side comparison (single output image)

## by frame

```shell
ffmpeg -i video1.mkv -i video2.mkv \
-filter_complex "[0:v]select='eq(n,100)'[a]; \
                 [1:v]select='eq(n,100)'[b]; \
                 [a][b]hstack=inputs=2" \
-frames:v 1 comparison.png
```

- `select='eq(n,100)'` → pick frame #100 from each video.
- `[a][b]hstack=inputs=2` → put them side by side.
- Result: `comparison.png` shows both frames next to each other.


## by time

```shell
ffmpeg `
-ss 00:01:23 -i ori.mkv `
-ss 00:01:23 -i output.mkv `
-filter_complex "[0:v][1:v]hstack=inputs=2" `
-frames:v 1 comparison.png
```

- `-ss` for time-based snapshots.
- `select=eq(n,frame_number)` for frame index.
- `hstack`: Add `hstack` filter to make one combined image.
